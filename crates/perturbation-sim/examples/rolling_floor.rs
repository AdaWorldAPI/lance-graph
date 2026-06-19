//! The 4-D rolling floor on the real ES core — L1..L4 HHTL eigenmode tiers as an
//! HDR popcount-stacking, early-exit, Belichtungsmesser cascade.
//!
//! Per tier (recursive spectral bisection = Bardioc Mode 1 Global → Mode 4):
//!   intensity = weyl_over_fiedler(Δλ₂ under the basin's worst single trip,
//!               basin λ₂)  =  Weyl × (1/Fiedler)  — the mode-instability modifier.
//! The four per-tier intensities are stacked coarse→fine; each tier's rolling
//! floor (mu + k·σ, preheated from the per-basin modifier sample) meters the
//! stacked value into a band; the cascade EARLY-EXITS at the first Alarm.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example rolling_floor -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{
    symmetric_eigen, weyl_over_fiedler, Edge, Eigen, Grid, RollingFloor, TierFloors,
};
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
    if members.len() < 4 {
        return None;
    }
    let sub = induced(grid, members);
    let c = perturbation_sim::cheeger_sweep(&sub, &vec![true; sub.edges.len()]);
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

/// One eigensolve per basin: returns `(λ₂, worst single-trip Δλ₂)`. The worst Δλ₂
/// uses the **first-order Fiedler sensitivity** `∂λ₂/∂wₑ = (v₂[a]−v₂[b])²` (exact
/// derivative), so removing line `e` drops λ₂ by ≈ `(v₂[a]−v₂[b])²·bₑ`. One
/// eigensolve ranks all lines — O(1) per line instead of an eigensolve per line.
fn lambda2_and_worst_weyl(g: &Grid) -> (f64, f64) {
    if g.edges.is_empty() {
        return (0.0, 0.0);
    }
    let eig: Eigen = symmetric_eigen(&g.laplacian_of(&vec![true; g.edges.len()]), g.n);
    let l2 = eig.values.get(1).copied().unwrap_or(0.0);
    let v2 = eig.eigenvector(1);
    let worst = g
        .edges
        .iter()
        .map(|e| {
            let d = v2[e.from] - v2[e.to];
            (d * d * e.susceptance).max(0.0)
        })
        .fold(0.0, f64::max);
    (l2, worst.min(l2)) // Δλ₂ cannot exceed λ₂ (it cannot push below 0)
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
        println!("grid: synthetic 8×8 — {} buses", g.n);
        g
    };

    // Build the 4-level HHTL tree (Mode 1 Global → finer modes) by bisection.
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

    // Per tier: the modifier sample (one weyl/fiedler per basin) + the tier median
    // as the stacked intensity. The sample preheats that tier's rolling floor.
    let tiers = ["HEEL(L1)", "HIP (L2)", "TWIG(L3)", "LEAF(L4)"];
    let mut tier_samples: Vec<Vec<f64>> = Vec::with_capacity(4);
    let mut tier_intensity = [0.0f64; 4];
    println!("\n== Per-tier mode-instability modifier  m = Weyl × (1/Fiedler)  =  Δλ₂ / λ₂ ==");
    println!(
        "  {:<9} {:>7} {:>13} {:>13} {:>13}",
        "tier", "basins", "median λ₂", "median Δλ₂", "median m"
    );
    let med = |mut v: Vec<f64>| -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    for (l, level) in levels.iter().enumerate() {
        // One eigensolve per basin → (λ₂, worst Δλ₂, modifier m).
        let (mut l2s, mut dls, mut ms) = (Vec::new(), Vec::new(), Vec::new());
        for basin in level {
            if basin.len() < 4 {
                continue;
            }
            let sub = induced(&grid, basin);
            if sub.edges.len() < 3 {
                continue;
            }
            let (l2, dl) = lambda2_and_worst_weyl(&sub);
            l2s.push(l2);
            dls.push(dl);
            ms.push(weyl_over_fiedler(dl, l2));
        }
        if ms.is_empty() {
            ms.push(0.0);
        }
        tier_intensity[l] = med(ms.clone());
        tier_samples.push(ms.clone());
        println!(
            "  {:<9} {:>7} {:>13.3e} {:>13.3e} {:>13.3e}",
            tiers[l],
            level.len(),
            med(l2s),
            med(dls),
            med(ms),
        );
    }

    // Preheat the 4-D rolling floor from the per-tier modifier samples, then run
    // the coarse→fine stacked early-exit pass.
    let k = 2.0; // σ-multiplier ≈ 97.7% one-sided (Jirak-honest: approximate)
    let mut floors = TierFloors::new(k);
    floors.preheat(&tier_samples);
    println!("\n== 4-D rolling floor (preheated, k = {k} σ; bands by quarters of mu+kσ) ==");
    let snapshot: Vec<(f64, f64, f64)> = floors
        .floors
        .iter()
        .map(|f: &RollingFloor| (f.mu(), f.sigma(), f.threshold()))
        .collect();
    for (l, (mu, sg, th)) in snapshot.iter().enumerate() {
        println!(
            "  {:<9} floor: mu={:>10.3e}  σ={:>10.3e}  threshold(mu+{k}σ)={:>10.3e}",
            tiers[l], mu, sg, th
        );
    }

    let r = floors.stack_early_exit(tier_intensity);
    println!("\n== Coarse→fine stacked early-exit (popcount-stacking analogue) ==");
    print!("  per-tier intensity stacked: ");
    let mut acc = 0.0;
    for (l, &x) in tier_intensity.iter().enumerate() {
        acc += x;
        print!("{}={acc:.2e} ", tiers[l].trim());
        if l == r.exit_tier {
            break;
        }
    }
    println!(
        "\n  EXIT at {} (stacked={:.3e}, band={:?}, early={})",
        tiers[r.exit_tier], r.stacked, r.band, r.early
    );
    print!("  z (σ above each tier floor): ");
    for (l, zv) in r.z.iter().enumerate() {
        print!("{}={zv:+.2} ", tiers[l].trim());
    }
    println!(
        "\n\n  Read: the modifier m = Weyl×(1/Fiedler) is the per-mode early-warning\n  \
         signal (big spectral shift in an already weakly-connected mode). The floor\n  \
         is metered like a camera's exposure (Belichtungsmesser): coarse tiers preheat\n  \
         the fine floors, the stack early-exits at the first Alarm — the multi-scale\n  \
         zoom of perturbation theory (Bardioc pillar 04) as a confidence-gated cascade.\n  \
         {} — significance via the Jirak n^(p/2−1) rate, not IID. CONJECTURE [H].",
        if r.early {
            "An early exit here would skip the finer eigenmode computation"
        } else {
            "Calm stack: no tier alarmed, ran to the leaf"
        }
    );
}
