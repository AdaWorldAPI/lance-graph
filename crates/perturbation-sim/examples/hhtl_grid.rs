//! The deterministic HHTL topology grid for the real ES core: each bus → its
//! (HEEL, HIP, TWIG) cascade key, by recursive Cheeger bisection. Topology IS the
//! key — value members hang off it (HHTL-OGAR). Verifies determinism + prints the
//! per-key basin sizes and λ₂.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example hhtl_grid -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{basin_lambda2, hhtl_keys, Edge, Grid};
use std::collections::BTreeMap;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let grid = if args.len() >= 3 {
        let buses = std::fs::read_to_string(&args[1]).expect("buses.csv");
        let lines = std::fs::read_to_string(&args[2]).expect("lines.csv");
        let cc = args.get(3).map(|s| s.as_str()).unwrap_or("ES");
        let imp = perturbation_sim::from_pypsa_csv(&buses, &lines, Some(cc))
            .expect("import")
            .largest_component();
        println!("grid: {cc} PyPSA core — {} buses", imp.grid.n);
        imp.grid
    } else {
        let g = synthetic(8, 8);
        println!("grid: synthetic 8×8 — {} buses", g.n);
        g
    };

    let keys = hhtl_keys(&grid);
    // Determinism: a pure function of the topology.
    assert_eq!(keys, hhtl_keys(&grid), "HHTL grid must be deterministic");

    let l2 = basin_lambda2(&grid, &keys);
    let mut sizes: BTreeMap<(u16, u16, u16), usize> = BTreeMap::new();
    for k in &keys {
        *sizes.entry((k.heel, k.hip, k.twig)).or_insert(0) += 1;
    }

    println!("\n== Deterministic HHTL grid (HEEL.HIP.TWIG → basin) ==");
    println!("  {:>10}  {:>6}  {:>12}", "key", "buses", "basin λ₂");
    let mut weakest = ((0u16, 0u16, 0u16), f64::INFINITY);
    for (k, n) in &sizes {
        let key = perturbation_sim::HhtlKey {
            heel: k.0,
            hip: k.1,
            twig: k.2,
        };
        let lam = l2.get(&key).copied().unwrap_or(0.0);
        println!("  {}.{}.{:>6}  {n:>6}  {lam:>12.3e}", k.0, k.1, k.2);
        if lam < weakest.1 {
            weakest = (*k, lam);
        }
    }
    println!(
        "\n  → {} keyed basins; weakest = {}.{}.{} (λ₂ = {:.3e}) — the deterministic\n  \
         topology address of the fail-first compartment. Value members (study factors,\n  \
         helix residues) hang off this key, orthogonal to it by the key/value split.\n  \
         Binary-Cheeger tiers here; OGAR widens each to a 16-ary/256-centroid tile.",
        sizes.len(),
        (weakest.0).0,
        (weakest.0).1,
        (weakest.0).2,
        weakest.1
    );
}
