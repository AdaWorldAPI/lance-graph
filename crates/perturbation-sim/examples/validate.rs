//! Verification harness: compute the aging modifiers from the data we have
//! (topology `DensityProxy` — the Gegenhypothese), sweep a fixed contingency set
//! across several time slices, extract the 4 field-tier factors per contingency,
//! and run the reliability/validity battery (Cronbach α, ICC(2,1), pairwise
//! Pearson/Spearman) + a time test-retest. Verifies the basin modeling + time +
//! the 4 factors hang together.
//!
//! Run (synthetic fallback, always works):
//!   cargo run --manifest-path crates/perturbation-sim/Cargo.toml --example validate
//! Run on the real Iberian core:
//!   cargo run … --example validate -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES
//!
//! Significance note: contingencies are weakly dependent → use the Jirak
//! n^(p/2−1) rate for p-values, not IID (this harness reports point estimates).

use perturbation_sim::{
    apply_aging, contingency_features, cronbach_alpha, dc_flows, edge_age_factors, icc_a1,
    spearman, symmetric_eigen, zscore, AgeModel, CascadeConfig, Edge, Grid,
};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn balanced(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Rng(seed);
    let raw: Vec<f64> = (0..n).map(|_| r.f()).collect();
    let m = raw.iter().sum::<f64>() / n as f64;
    raw.iter().map(|x| x - m).collect()
}

fn synthetic_lattice(rows: usize, cols: usize) -> Grid {
    let id = |r: usize, c: usize| r * cols + c;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                edges.push(Edge::new(id(r, c), id(r, c + 1), 1.0, 1.0));
            }
            if r + 1 < rows {
                edges.push(Edge::new(id(r, c), id(r + 1, c), 1.0, 1.0));
            }
        }
    }
    Grid::new(rows * cols, edges)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid0 = if args.len() >= 3 {
        let buses = std::fs::read_to_string(&args[1]).expect("buses.csv");
        let lines = std::fs::read_to_string(&args[2]).expect("lines.csv");
        let country = args.get(3).map(|s| s.as_str()).unwrap_or("ES");
        let imp = perturbation_sim::from_pypsa_csv(&buses, &lines, Some(country))
            .expect("import")
            .largest_component();
        println!(
            "grid: {country} PyPSA core — {} buses, {} lines",
            imp.grid.n,
            imp.grid.edges.len()
        );
        imp.grid
    } else {
        let g = synthetic_lattice(5, 5);
        println!(
            "grid: synthetic 5×5 lattice — {} buses, {} lines (pass CSVs for real data)",
            g.n,
            g.edges.len()
        );
        g
    };

    // Modifiers from the data we have: topology density proxy (Gegenhypothese).
    let alive = vec![true; grid0.edges.len()];
    let age = edge_age_factors(&grid0, &alive, &AgeModel::DensityProxy);
    let grid = apply_aging(&grid0, &age, 0.7);
    let n = grid.n;
    println!("aging: DensityProxy (sparse=older), oldest derate ×0.7\n");

    let k_slices = 3usize;
    let m_seeds = 12.min(grid.edges.len());

    // Fixed contingency set = top-loaded lines under slice-0 base (same subjects
    // every slice, so test-retest is comparable).
    let p0 = balanced(n, 0xA11CE);
    let base0 = {
        let eig = symmetric_eigen(&grid.laplacian_of(&alive), n);
        dc_flows(&grid, &alive, &eig.pseudo_apply(&p0, 1e-9))
    };
    let mut order: Vec<usize> = (0..base0.len()).collect();
    order.sort_by(|&a, &b| base0[b].abs().total_cmp(&base0[a].abs()));
    let seeds: Vec<usize> = order.into_iter().take(m_seeds).collect();

    let factor_names = [
        "d_lambda2",
        "dk_rotation",
        "d_conductance",
        "infight",
        "raumgewinn",
    ];
    let mut slice_rows: Vec<Vec<[f64; 5]>> = Vec::new();
    let mut severity: Vec<Vec<f64>> = Vec::new();

    for s in 0..k_slices {
        let p = balanced(n, 0xA11CE + s as u64);
        // Self-calibrate limits to this slice's loading so contingencies bite.
        let mut g = grid.clone();
        let eig = symmetric_eigen(&g.laplacian_of(&alive), n);
        let base = dc_flows(&g, &alive, &eig.pseudo_apply(&p, 1e-9));
        for (e, edge) in g.edges.iter_mut().enumerate() {
            edge.limit = (1.1 * base[e].abs()).max(1e-6);
        }
        let mut rows = Vec::new();
        let mut sev = Vec::new();
        for &seed in &seeds {
            let f = contingency_features(&g, &p, seed, CascadeConfig::default());
            rows.push(f.as_row());
            sev.push(f.infight + f.raumgewinn);
        }
        slice_rows.push(rows);
        severity.push(sev);
    }

    // ── Factor battery on slice 0 ────────────────────────────────────────────
    let cols: Vec<Vec<f64>> = (0..5)
        .map(|c| slice_rows[0].iter().map(|r| r[c]).collect())
        .collect();
    let z: Vec<Vec<f64>> = cols.iter().map(|c| zscore(c)).collect();

    println!(
        "== 4-factor reliability battery ({} contingencies, slice 0) ==",
        m_seeds
    );
    println!(
        "  Cronbach α (5 factors)  : {:.4}  (high→one criticality scale; low→distinct facets)",
        cronbach_alpha(&z)
    );
    println!("  ICC(2,1) across factors : {:.4}  (vs α: gap = systematic inter-factor bias = the Go duality)", icc_a1(&z));

    println!("\n== pairwise Spearman ρ among factors (convergent/discriminant) ==");
    print!("{:>13}", "");
    for nm in &factor_names {
        print!("{:>12}", nm);
    }
    println!();
    for i in 0..5 {
        print!("{:>13}", factor_names[i]);
        for j in 0..5 {
            print!("{:>12.3}", spearman(&cols[i], &cols[j]));
        }
        println!();
    }

    // ── Time test-retest (same subjects, different slices) ───────────────────
    println!(
        "\n== time test-retest (severity-ranking stability across {} slices) ==",
        k_slices
    );
    let mut sum = 0.0;
    let mut cnt = 0;
    for a in 0..k_slices {
        for b in (a + 1)..k_slices {
            let rho = spearman(&severity[a], &severity[b]);
            println!("  slice {a} vs {b}: Spearman ρ = {:.4}", rho);
            sum += rho;
            cnt += 1;
        }
    }
    if cnt > 0 {
        println!(
            "  mean test-retest ρ      : {:.4}  (high→the basin ranking is stable over time)",
            sum / cnt as f64
        );
    }

    println!("\n(point estimates; significance needs the Jirak n^(p/2−1) rate, not IID.)");
}
