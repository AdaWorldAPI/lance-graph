//! Perturbation-agnostic resilience certificate on the real ES core — read the
//! field ONCE, never replay a perturbation.
//!
//! Stepping back from "predict the same trip's cascade" (circular, §4.9.1) to
//! "certify resilience": the spectrum and the self-inverse `L⁺` reference already
//! integrate the response over ALL perturbations. We read, globally and per
//! Cheeger compartment: λ₂ (worst-case margin) and Kf = n·Σ 1/λ_k (total
//! effective resistance). Then the perturbation-AGNOSTIC reinforcement: the one
//! corridor across the global Cheeger seam that maximizes the first-order λ₂ gain
//! `(v₂[a]−v₂[b])²` — raising the worst-case margin against the NEXT unknown
//! perturbation, not against the last one. No `simulate_outage` anywhere.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example resilience -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{cheeger_sweep, symmetric_eigen, Edge, Grid, Resilience};
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

fn cert(grid: &Grid) -> Resilience {
    let eig = symmetric_eigen(&grid.laplacian_of(&vec![true; grid.edges.len()]), grid.n);
    Resilience::from_eigenvalues(&eig.values, 1e-9)
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
    let alive = vec![true; grid.edges.len()];

    // Global certificate — one eigensolve, no perturbation replay.
    let base = symmetric_eigen(&grid.laplacian_of(&alive), n);
    let g_cert = Resilience::from_eigenvalues(&base.values, 1e-9);
    println!("\n== Global resilience certificate (read once from the L⁺ spectrum) ==");
    println!("  λ₂ (worst-case margin)        : {:.4e}", g_cert.lambda2);
    println!("  Kf = n·Σ 1/λ_k (total eff. R) : {:.4e}", g_cert.kirchhoff);
    println!(
        "  mean pairwise resistance       : {:.4e}",
        g_cert.mean_resistance()
    );
    println!("  non-trivial modes              : {}", g_cert.modes);

    // Per-compartment certificate (depth-3 Cheeger basins → up to 8 compartments).
    let mut basins: Vec<Vec<usize>> = vec![(0..n).collect()];
    for _ in 0..3 {
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
    let mut certs: Vec<(usize, Resilience)> = basins
        .iter()
        .enumerate()
        .filter(|(_, b)| b.len() >= 4 && induced(&grid, b).edges.len() >= 3)
        .map(|(i, b)| (i, cert(&induced(&grid, b))))
        .collect();
    // Weakest compartment first = smallest λ₂ margin.
    certs.sort_by(|a, b| a.1.lambda2.partial_cmp(&b.1.lambda2).unwrap());

    println!(
        "\n== Per-compartment certificate ({} compartments, weakest λ₂ first) ==",
        certs.len()
    );
    println!(
        "  {:>4} {:>6} {:>13} {:>13} {:>13}",
        "comp", "nodes", "λ₂ margin", "Kf", "mean R"
    );
    for (i, c) in &certs {
        println!(
            "  {:>4} {:>6} {:>13.4e} {:>13.4e} {:>13.4e}",
            i,
            c.n,
            c.lambda2,
            c.kirchhoff,
            c.mean_resistance()
        );
    }
    if let Some((wid, w)) = certs.first() {
        println!(
            "  → weakest compartment = {wid} (λ₂ = {:.3e}); that is where the next\n    \
             unknown perturbation has the least margin — the resilience target.",
            w.lambda2
        );
    }

    // Perturbation-agnostic reinforcement: the corridor across the GLOBAL Cheeger
    // seam maximizing first-order λ₂ gain (v₂[a]−v₂[b])². Raises the worst-case
    // margin against the next perturbation — computed from the field, no trip.
    let v2 = base.eigenvector(1);
    let part = cheeger_sweep(&grid, &alive).partition;
    let w_new = grid.edges.iter().map(|e| e.susceptance).sum::<f64>() / grid.edges.len() as f64;
    let mut best = (0usize, 0usize, 0.0f64);
    for a in 0..n {
        for b in (a + 1)..n {
            if part[a] != part[b] {
                let d = v2[a] - v2[b];
                let gain = w_new * d * d;
                if gain > best.2 {
                    best = (a, b, gain);
                }
            }
        }
    }
    let mut g2 = grid.clone();
    g2.edges
        .push(Edge::new(best.0, best.1, w_new, f64::INFINITY));
    let g2 = Grid::new(n, g2.edges);
    let new_cert = cert(&g2);
    println!("\n== Perturbation-agnostic reinforcement (max-margin corridor, no trip replayed) ==");
    println!(
        "  best seam corridor: bus {} — bus {}  (first-order λ₂ gain ≈ {:+.3e})",
        best.0, best.1, best.2
    );
    println!(
        "  λ₂: {:.4e} → {:.4e}  (+{:.0}%)    Kf: {:.4e} → {:.4e}  ({:+.1}%)",
        g_cert.lambda2,
        new_cert.lambda2,
        100.0 * (new_cert.lambda2 / g_cert.lambda2 - 1.0),
        g_cert.kirchhoff,
        new_cert.kirchhoff,
        100.0 * (new_cert.kirchhoff / g_cert.kirchhoff - 1.0),
    );
    println!(
        "\n  Read: λ₂ up / Kf down = more resilient to the NEXT (unknown) perturbation,\n  \
         read straight off the self-inverse L⁺ spectrum — never by replaying a trip.\n  \
         BRAESS CAVEAT (§4.4): raising the margin can still worsen one specific FLOW\n  \
         cascade, so co-design the new corridor's rating with the margin. The margin is\n  \
         the perturbation-agnostic certificate; the cascade is the per-perturbation check."
    );
}
