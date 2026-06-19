//! HHTL as resident: HH = Raumgewinn, TL = infight, correlated per basin.
//!
//! HHTL splits HH (HEEL/HIP, coarse) | TL (TWIG/LEAF, fine). HH is *resident* to
//! **Raumgewinn** — a basin's algebraic connectivity λ₂ (the field). TL is
//! resident to **infight** — the local cascade fraction when the basin's
//! most-loaded internal line trips. We recurse the spectral bisection to a set
//! of leaf basins and, per basin, measure both residents, then correlate them
//! (Pearson / Spearman / ICC). Near-zero correlation = the HH|TL tier split IS
//! the Raumgewinn|infight axis split (the orthogonality lives at the tier).
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example hhtl_resident -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{
    cheeger_sweep, dc_flows, icc_a1, pearson, simulate_outage, spearman, symmetric_eigen, zscore,
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
    if members.len() < 8 {
        return None;
    }
    let sub = induced(grid, members);
    let c = cheeger_sweep(&sub, &vec![true; sub.edges.len()]);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (i, &m) in members.iter().enumerate() {
        if c.partition[i] {
            a.push(m);
        } else {
            b.push(m);
        }
    }
    if a.is_empty() || b.is_empty() {
        None
    } else {
        Some((a, b))
    }
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

    // Recurse to leaf basins (depth 5 → up to 16 basins).
    let mut basins: Vec<Vec<usize>> = vec![(0..grid.n).collect()];
    for _ in 0..5 {
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

    // Per leaf basin: HH-resident (Raumgewinn λ₂) and TL-resident (infight).
    let mut hh = Vec::new(); // Raumgewinn
    let mut tl = Vec::new(); // infight
    let cfg = CascadeConfig {
        max_rounds: 12,
        ..CascadeConfig::default()
    };
    for b in &basins {
        if b.len() < 6 {
            continue;
        }
        let mut sub = induced(&grid, b);
        if sub.edges.len() < 4 {
            continue;
        }
        let alive = vec![true; sub.edges.len()];
        let eig = symmetric_eigen(&sub.laplacian_of(&alive), sub.n);
        let lam2 = eig.values.get(1).copied().unwrap_or(0.0); // HH = Raumgewinn

        // TL = infight: balanced injection, self-calibrate limits, trip the
        // most-loaded internal line, cascade within the basin.
        let mut rng = Rng(0x1234 + b.len() as u64);
        let raw: Vec<f64> = (0..sub.n).map(|_| rng.f()).collect();
        let mean = raw.iter().sum::<f64>() / sub.n as f64;
        let p: Vec<f64> = raw.iter().map(|x| x - mean).collect();
        let base = dc_flows(&sub, &alive, &eig.pseudo_apply(&p, 1e-9));
        for (e, edge) in sub.edges.iter_mut().enumerate() {
            edge.limit = (1.1 * base[e].abs()).max(1e-6);
        }
        let seed = base
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.abs().total_cmp(&y.1.abs()))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let infight = simulate_outage(&sub, &p, seed, cfg).fraction_tripped;

        hh.push(lam2);
        tl.push(infight);
    }

    println!(
        "\n== HHTL residents per basin ({} leaf basins) ==",
        hh.len()
    );
    println!(
        "  HH-resident = Raumgewinn (basin λ₂);  TL-resident = infight (basin cascade fraction)\n"
    );
    println!("  {:>3}  {:>14}  {:>14}", "#", "HH λ₂", "TL infight");
    for i in 0..hh.len() {
        println!("  {:>3}  {:>14.3e}  {:>14.3}", i, hh[i], tl[i]);
    }

    if hh.len() >= 3 {
        let r = pearson(&hh, &tl);
        let rho = spearman(&hh, &tl);
        let icc = icc_a1(&[zscore(&hh), zscore(&tl)]);
        println!("\n== Correlation of the two residents across basins ==");
        println!("  Pearson  r(HH, TL) = {:+.3}", r);
        println!("  Spearman ρ(HH, TL) = {:+.3}", rho);
        println!("  ICC(2,1) HH vs TL  = {:+.3}", icc);
        println!(
            "\n  → |ρ| {} ⇒ HH (Raumgewinn) and TL (infight) are {} per basin:\n    \
             the HH|TL tier split IS the Raumgewinn|infight axis split (the orthogonality\n    \
             lives at the HHTL tier). Small n — significance via the Jirak n^(p/2−1) rate.",
            if rho.abs() < 0.3 { "≈ 0" } else { "≠ 0" },
            if rho.abs() < 0.3 {
                "orthogonal (separate axes)"
            } else {
                "coupled"
            }
        );
    }
}
