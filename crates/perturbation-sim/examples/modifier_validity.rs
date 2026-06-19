//! Does the modifier `m = Weyl × (1/Fiedler) = Δλ₂ × (1/λ₂)` actually predict
//! anything? A direct criterion-validity probe — the question "is it confirmed
//! by the results?".
//!
//! For a stride sample of N-1 contingencies on the real ES core we compute, per
//! tripped line `e`: the numerator `Δλ₂(e)` = EXACT single-trip
//! algebraic-connectivity loss (Weyl); the denominator `λ₂_local` = the Fiedler
//! value of the LEAF BASIN containing `e` (the multi-scale "regional
//! connectivity", Mode 2); the modifier `m(e) = Δλ₂(e) / λ₂_local`. And two
//! OUTCOMES under self-calibrated stress (mean over R injection raters): cascade
//! size (line-count fraction tripped — the "infight" axis) and connectivity-loss
//! (`1 − λ₂'/λ₂` after the cascade — the "Raumgewinn" axis).
//!
//! Then Spearman(m, each outcome), compared against the bare numerator (Weyl Δλ₂)
//! and the global-λ₂ variant. This says, with numbers, WHETHER the modifier is
//! confirmed and AGAINST WHICH outcome.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example modifier_validity -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{
    contingency_features, dc_flows, simulate_outage, spearman, symmetric_eigen, weyl_over_fiedler,
    CascadeConfig, Edge, Grid,
};
use std::collections::HashMap;

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

fn synthetic(rows: usize, cols: usize) -> Grid {
    let id = |r: usize, c: usize| r * cols + c;
    let mut e = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                e.push(Edge::new(id(r, c), id(r, c + 1), 1.0, 1.0));
            }
            if r + 1 < rows {
                e.push(Edge::new(id(r, c), id(r + 1, c), 1.0, 1.0));
            }
        }
    }
    Grid::new(rows * cols, e)
}

fn induced(grid: &Grid, members: &[usize]) -> Grid {
    let mut remap = HashMap::new();
    for (i, &m) in members.iter().enumerate() {
        remap.insert(m, i);
    }
    let edges = grid
        .edges
        .iter()
        .filter_map(|e| match (remap.get(&e.from), remap.get(&e.to)) {
            (Some(&a), Some(&b)) => Some(Edge::new(a, b, e.susceptance, e.limit)),
            _ => None,
        })
        .collect();
    Grid::new(members.len(), edges)
}

fn bisect(grid: &Grid, members: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
    if members.len() < 6 {
        return None;
    }
    let sub = induced(grid, members);
    let c = perturbation_sim::cheeger_sweep(&sub, &vec![true; sub.edges.len()]);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, &mm) in members.iter().enumerate() {
        if c.partition[i] {
            a.push(mm);
        } else {
            b.push(mm);
        }
    }
    if a.is_empty() || b.is_empty() {
        None
    } else {
        Some((a, b))
    }
}

fn lambda2(g: &Grid, alive: &[bool]) -> f64 {
    symmetric_eigen(&g.laplacian_of(alive), g.n)
        .values
        .get(1)
        .copied()
        .unwrap_or(0.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid = if args.len() >= 3 {
        let buses = std::fs::read_to_string(&args[1]).expect("buses.csv");
        let lines = std::fs::read_to_string(&args[2]).expect("lines.csv");
        let cc = args.get(3).map(|s| s.as_str()).unwrap_or("ES");
        let imp = perturbation_sim::from_pypsa_csv(&buses, &lines, Some(cc))
            .expect("import")
            .largest_component();
        println!(
            "grid: {cc} PyPSA core — {} buses, {} lines",
            imp.grid.n,
            imp.grid.edges.len()
        );
        imp.grid
    } else {
        let g = synthetic(10, 10);
        println!("grid: synthetic 10×10 — {} buses", g.n);
        g
    };
    let n = grid.n;
    let m = grid.edges.len();
    let alive = vec![true; m];

    let base = symmetric_eigen(&grid.laplacian_of(&alive), n);
    let lam2_global = base.values.get(1).copied().unwrap_or(0.0);
    let v2 = base.eigenvector(1);

    // Stride sample across the first-order sensitivity ranking (variety).
    let mut sens: Vec<(usize, f64)> = (0..m)
        .map(|e| {
            let d = v2[grid.edges[e].from] - v2[grid.edges[e].to];
            (e, d * d * grid.edges[e].susceptance)
        })
        .collect();
    sens.sort_by(|x, y| y.1.total_cmp(&x.1));
    let k = 30.min(m);
    let step = (m / k).max(1);
    let cand: Vec<usize> = (0..k).map(|i| sens[(i * step).min(m - 1)].0).collect();

    // Leaf basins (depth 4) → each line's local λ₂ (the regional Fiedler).
    let mut basins: Vec<Vec<usize>> = vec![(0..n).collect()];
    for _ in 0..4 {
        let mut next = Vec::new();
        for b in &basins {
            match bisect(&grid, b) {
                Some((x, y)) => {
                    next.push(x);
                    next.push(y);
                }
                None => next.push(b.clone()),
            }
        }
        basins = next;
    }
    let mut basin_of = vec![usize::MAX; n];
    let mut basin_lam2 = Vec::with_capacity(basins.len());
    for (bi, b) in basins.iter().enumerate() {
        for &node in b {
            basin_of[node] = bi;
        }
        let sub = induced(&grid, b);
        let l2 = if sub.edges.is_empty() {
            lam2_global
        } else {
            lambda2(&sub, &vec![true; sub.edges.len()]).max(1e-12)
        };
        basin_lam2.push(l2);
    }
    let local_lam2 = |e: usize| -> f64 {
        let (a, b) = (grid.edges[e].from, grid.edges[e].to);
        // Use the smaller-λ₂ endpoint basin (the weaker regional mode the line sits on).
        let la = basin_lam2[basin_of[a]];
        let lb = basin_lam2[basin_of[b]];
        la.min(lb).max(1e-12)
    };

    // Per-candidate predictors.
    let mut weyl_exact = Vec::with_capacity(k); // EXACT single-trip Δλ₂ (Weyl)
    let mut m_local = Vec::with_capacity(k); // Δλ₂ / λ₂_local
    let mut m_global = Vec::with_capacity(k); // Δλ₂ / λ₂_global
    for &e in &cand {
        let mut after = alive.clone();
        after[e] = false;
        let dl = (lam2_global - lambda2(&grid, &after)).max(0.0);
        weyl_exact.push(dl);
        m_local.push(weyl_over_fiedler(dl, local_lam2(e)));
        m_global.push(weyl_over_fiedler(dl, lam2_global));
    }

    // Outcomes under self-calibrated stress: cascade size (mean over raters) +
    // connectivity-loss (Raumgewinn) from a representative injection.
    let r_raters = 3usize;
    let cfg = CascadeConfig {
        max_rounds: 10,
        ..CascadeConfig::default()
    };
    let mut cascade_sz = vec![0.0; k];
    for r in 0..r_raters {
        let mut rng = Rng(0xA11CE + r as u64 * 0x1000);
        let raw: Vec<f64> = (0..n).map(|_| rng.f()).collect();
        let mean = raw.iter().sum::<f64>() / n as f64;
        let p: Vec<f64> = raw.iter().map(|x| x - mean).collect();
        let flows = dc_flows(&grid, &alive, &base.pseudo_apply(&p, 1e-9));
        let mut g = grid.clone();
        for (e, edge) in g.edges.iter_mut().enumerate() {
            edge.limit = (1.1 * flows[e].abs()).max(1e-6);
        }
        for (i, &e) in cand.iter().enumerate() {
            cascade_sz[i] += simulate_outage(&g, &p, e, cfg).fraction_tripped / r_raters as f64;
        }
    }
    // Connectivity-loss (Raumgewinn) under one stressed injection.
    let mut rng = Rng(0xA11CE);
    let raw: Vec<f64> = (0..n).map(|_| rng.f()).collect();
    let mean = raw.iter().sum::<f64>() / n as f64;
    let p0: Vec<f64> = raw.iter().map(|x| x - mean).collect();
    let f0 = dc_flows(&grid, &alive, &base.pseudo_apply(&p0, 1e-9));
    let mut g0 = grid.clone();
    for (e, edge) in g0.edges.iter_mut().enumerate() {
        edge.limit = (1.1 * f0[e].abs()).max(1e-6);
    }
    let conn_loss: Vec<f64> = cand
        .iter()
        .map(|&e| contingency_features(&g0, &p0, e, cfg).raumgewinn)
        .collect();

    let sn = |x: &[f64], y: &[f64]| -> (f64, f64) {
        let rho = spearman(x, y);
        (rho, rho.abs() * (k as f64).sqrt())
    };
    let row = |name: &str, x: &[f64], y: &[f64]| {
        let (rho, jn) = sn(x, y);
        println!("  {name:<38} Spearman ρ = {rho:+.3}   (|ρ|√n = {jn:.2})");
    };

    println!("\n  N = {k} contingencies · {r_raters} raters · λ₂_global = {lam2_global:.3e}\n");
    println!("== vs CASCADE SIZE (line-count, the infight axis) ==");
    row("Weyl Δλ₂ (numerator alone)", &weyl_exact, &cascade_sz);
    row(
        "m = Δλ₂ × (1/λ₂_local)   [the modifier]",
        &m_local,
        &cascade_sz,
    );
    row("m = Δλ₂ × (1/λ₂_global)", &m_global, &cascade_sz);

    println!("\n== vs CONNECTIVITY-LOSS (Raumgewinn / fragmentation axis) ==");
    row("Weyl Δλ₂ (numerator alone)", &weyl_exact, &conn_loss);
    row(
        "m = Δλ₂ × (1/λ₂_local)   [the modifier]",
        &m_local,
        &conn_loss,
    );
    row("m = Δλ₂ × (1/λ₂_global)", &m_global, &conn_loss);

    println!(
        "\nVerdict: the modifier is CONFIRMED for whichever outcome shows a strong,\n  \
         positive ρ (|ρ|√n ≳ 2 at the Jirak rate). Expectation from §4.8: Δλ₂ predicts\n  \
         FRAGMENTATION (connectivity-loss), not cascade line-count — so the modifier\n  \
         should land on the Raumgewinn axis. The numbers above settle it. Synthetic\n  \
         injections + estimated limits; harden with real ESIOS/ENTSO-E load."
    );
}
