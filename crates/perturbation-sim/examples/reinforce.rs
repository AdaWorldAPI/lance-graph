//! Reinforcement what-if: add the optimal third corridor across the Cheeger
//! seam (a third out-of-family edge between the two HIP basins) and measure the
//! gain — Δλ₂ and the change in the seam-line cascade.
//!
//! The optimal single new edge maximizes the first-order λ₂ gain
//! `∂λ₂/∂w = (v₂[a]−v₂[b])²`, so the best pair is the Fiedler extremes, one per
//! basin. Reports the exact Δλ₂ and the before/after cascade of tripping the
//! most-loaded seam line.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example reinforce -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

use perturbation_sim::{
    cheeger_sweep, dc_flows, simulate_outage, symmetric_eigen, CascadeConfig, Edge, Grid,
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

fn lambda2(grid: &Grid) -> f64 {
    let alive = vec![true; grid.edges.len()];
    symmetric_eigen(&grid.laplacian_of(&alive), grid.n)
        .values
        .get(1)
        .copied()
        .unwrap_or(0.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (grid, ids) = if args.len() >= 3 {
        let buses = std::fs::read_to_string(&args[1]).expect("buses.csv");
        let lines = std::fs::read_to_string(&args[2]).expect("lines.csv");
        let cc = args.get(3).map(|s| s.as_str()).unwrap_or("ES");
        let imp = perturbation_sim::from_pypsa_csv(&buses, &lines, Some(cc))
            .expect("import")
            .largest_component();
        println!("grid: {cc} PyPSA core — {} buses, {} lines", imp.grid.n, imp.grid.edges.len());
        (imp.grid, imp.bus_ids)
    } else {
        let g = synthetic(8, 8);
        let ids = (0..g.n).map(|i| i.to_string()).collect();
        println!("grid: synthetic 8×8 — {} buses, {} lines", g.n, g.edges.len());
        (g, ids)
    };
    let n = grid.n;
    let alive = vec![true; grid.edges.len()];

    // Fiedler vector + Cheeger basins.
    let eig = symmetric_eigen(&grid.laplacian_of(&alive), n);
    let lam2 = eig.values[1];
    let v2 = eig.eigenvector(1);
    let part = cheeger_sweep(&grid, &alive).partition;
    let w_new = grid.edges.iter().map(|e| e.susceptance).sum::<f64>() / grid.edges.len() as f64;

    // Greedy: rank candidate inter-basin edges by first-order λ₂ gain w·(v₂[a]−v₂[b])².
    let mut cands: Vec<(usize, usize, f64)> = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            if part[a] != part[b] {
                let d = v2[a] - v2[b];
                cands.push((a, b, w_new * d * d));
            }
        }
    }
    cands.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());

    println!("\n== Reinforcement: optimal 3rd corridor across the Cheeger seam ==");
    println!("  base λ₂ = {lam2:.3e}   (new-line susceptance b = {w_new:.3})");
    println!("  top-5 candidate ties (first-order λ₂ gain w·(v₂[a]−v₂[b])²):");
    for (a, b, g) in cands.iter().take(5) {
        println!("     {} — {}   Δλ₂(1st-order) ≈ {:+.3e}", ids[*a], ids[*b], g);
    }

    // Exact Δλ₂ for the best tie.
    let (a, b, _) = cands[0];
    let mut grid2 = grid.clone();
    grid2.edges.push(Edge::new(a, b, w_new, f64::INFINITY));
    let grid2 = Grid::new(n, grid2.edges);
    let lam2_new = lambda2(&grid2);
    println!(
        "\n  best tie {} — {} : λ₂ {:.3e} → {:.3e}  (exact Δλ₂ {:+.3e}, +{:.0}%)",
        ids[a], ids[b], lam2, lam2_new, lam2_new - lam2,
        100.0 * (lam2_new / lam2 - 1.0)
    );

    // Cascade before/after: trip the most-loaded seam line, with self-calibrated
    // limits, and see if the reinforcement contains the cascade.
    let mut rng = Rng(0xBEEF);
    let raw: Vec<f64> = (0..n).map(|_| rng.f()).collect();
    let mean = raw.iter().sum::<f64>() / n as f64;
    let p: Vec<f64> = raw.iter().map(|x| x - mean).collect();
    let base = dc_flows(&grid, &alive, &eig.pseudo_apply(&p, 1e-9));

    // Self-calibrated limits; pick the most-loaded line that crosses the seam.
    let mut g_lim = grid.clone();
    for (e, edge) in g_lim.edges.iter_mut().enumerate() {
        edge.limit = (1.1 * base[e].abs()).max(1e-6);
    }
    let seam: Vec<usize> = (0..grid.edges.len())
        .filter(|&e| part[grid.edges[e].from] != part[grid.edges[e].to])
        .collect();
    let seed = *seam
        .iter()
        .max_by(|&&x, &&y| base[x].abs().partial_cmp(&base[y].abs()).unwrap())
        .unwrap_or(&0);
    let cfg = CascadeConfig { max_rounds: 16, ..CascadeConfig::default() };
    let before = simulate_outage(&g_lim, &p, seed, cfg);

    let mut g_lim2 = g_lim.clone();
    g_lim2.edges.push(Edge::new(a, b, w_new, f64::INFINITY)); // reinforced tie, well-rated
    let g_lim2 = Grid::new(n, g_lim2.edges);
    let after = simulate_outage(&g_lim2, &p, seed, cfg);

    println!(
        "\n== Seam-line cascade, with vs without the 3rd corridor (seed {} — {}) ==",
        ids[grid.edges[seed].from], ids[grid.edges[seed].to]
    );
    println!(
        "  WITHOUT tie: {} lines tripped, connectivity-loss {:.1}%, islanded {} ({} comps)",
        before.shape.n_tripped(), 100.0 * before.spectral.connectivity_loss(),
        before.islanded, before.components_final
    );
    println!(
        "  WITH    tie: {} lines tripped, connectivity-loss {:.1}%, islanded {} ({} comps)",
        after.shape.n_tripped(), 100.0 * after.spectral.connectivity_loss(),
        after.islanded, after.components_final
    );
    println!(
        "\n  → the 3rd out-of-family corridor raises λ₂ by +{:.0}% and {} the seam cascade.",
        100.0 * (lam2_new / lam2 - 1.0),
        if after.shape.n_tripped() < before.shape.n_tripped() { "shrinks" } else { "does not shrink" }
    );
    println!("\n(Reinforcement = populating a 3rd out-of-family EdgeBlock slot; λ₂ gain bounded by Cauchy interlacing. Synthetic injections + estimated limits.)");
}
