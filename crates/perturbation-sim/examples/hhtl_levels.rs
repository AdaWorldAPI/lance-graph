//! The 4 models on 4 HHTL levels.
//!
//! Builds an HHTL tier tree (HEEL→HIP→TWIG→LEAF) by recursive spectral
//! (Cheeger/Fiedler) bisection of the grid into basins, then computes the four
//! field-tier theorems at EACH level:
//!
//! - Weyl → per-basin algebraic connectivity λ₂.
//! - Davis–Kahan → spectral gap λ₃−λ₂ (how well-defined the basin's Fiedler partition is — the DK denominator).
//! - Cheeger → the basin's normalized gap μ₂ and sweep conductance φ.
//! - Kron → contract each basin to a super-node; the super-graph (#super-nodes = #basins, #inter-basin = out-of-family ties).
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example hhtl_levels -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{cheeger_sweep, symmetric_eigen, Edge, Grid};
use std::collections::HashMap;

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

/// Induced sub-grid on `members` (reindexed 0..k), edges with both endpoints in.
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

/// Spectral bisection: split `members` by the normalized Fiedler sweep.
fn bisect(grid: &Grid, members: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
    if members.len() < 4 {
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

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
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
        let g = synthetic(8, 8);
        println!(
            "grid: synthetic 8×8 — {} buses, {} lines",
            g.n,
            g.edges.len()
        );
        g
    };

    // Build the 4-level HHTL tree by recursive bisection.
    let tiers = ["HEEL", "HIP ", "TWIG", "LEAF"];
    let mut levels: Vec<Vec<Vec<usize>>> = vec![vec![(0..grid.n).collect()]];
    for _ in 1..4 {
        let mut next = Vec::new();
        for basin in levels.last().unwrap() {
            match bisect(&grid, basin) {
                Some((a, b)) => {
                    next.push(a);
                    next.push(b);
                }
                None => next.push(basin.clone()),
            }
        }
        levels.push(next);
    }

    println!("\n  the 4 theorems × the 4 HHTL tiers (recursive spectral bisection)\n");
    println!(
        "{:<5} {:>7} {:>22} {:>14} {:>20} {:>18}",
        "tier",
        "basins",
        "Weyl λ₂ (min/med/max)",
        "DK gap λ₃−λ₂",
        "Cheeger μ₂ / φ (med)",
        "Kron super(N,ties)"
    );
    println!("  {}", "─".repeat(92));

    for (l, level) in levels.iter().enumerate() {
        // Per-basin spectra.
        let (mut lam2s, mut gaps, mut mu2s, mut phis) = (vec![], vec![], vec![], vec![]);
        for basin in level {
            if basin.len() < 2 {
                continue;
            }
            let sub = induced(&grid, basin);
            let eig = symmetric_eigen(&sub.laplacian_of(&vec![true; sub.edges.len()]), sub.n);
            lam2s.push(eig.values.get(1).copied().unwrap_or(0.0));
            gaps.push(
                eig.values.get(2).copied().unwrap_or(0.0)
                    - eig.values.get(1).copied().unwrap_or(0.0),
            );
            let c = cheeger_sweep(&sub, &vec![true; sub.edges.len()]);
            mu2s.push(c.mu2);
            phis.push(c.conductance);
        }
        // Kron super-graph: contract each basin to a node, count inter-basin ties.
        let mut basin_of = vec![usize::MAX; grid.n];
        for (bi, basin) in level.iter().enumerate() {
            for &m in basin {
                basin_of[m] = bi;
            }
        }
        let ties = grid
            .edges
            .iter()
            .filter(|e| basin_of[e.from] != basin_of[e.to])
            .count();

        let lmin = lam2s.iter().cloned().fold(f64::INFINITY, f64::min);
        let lmax = lam2s.iter().cloned().fold(0.0_f64, f64::max);
        println!(
            "{:<5} {:>7} {:>9.2e}/{:>5.2e}/{:>5.2e} {:>14.2e} {:>9.2e} / {:>6.2e} {:>10}({:>3},{:>4})",
            tiers[l],
            level.len(),
            if lmin.is_finite() { lmin } else { 0.0 },
            median(lam2s.clone()),
            lmax,
            median(gaps),
            median(mu2s),
            median(phis),
            "",
            level.len(),
            ties,
        );
    }

    println!(
        "\nReads (one Laplacian, four readings, per tier):\n  \
         Weyl λ₂ rises as basins get smaller/tighter (finer tiers = better-connected sub-basins);\n  \
         DK gap = how cleanly each basin wants to split again (large gap ⇒ stable partition);\n  \
         Cheeger μ₂/φ = the basin's own bottleneck; Kron ties = the out-of-family corridors\n  \
         between basins at that tier (the reinforcement targets). Cauchy interlacing binds the\n  \
         per-tier λ₂ to the parent — finer λ₂ ≥ coarser. Electrical embedding, not geography."
    );
}
