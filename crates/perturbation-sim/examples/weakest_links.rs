//! Weakest-links + local-boundary ("flap") analysis.
//!
//! On a grid (real PyPSA core via args, else synthetic), reports:
//!  1. STRUCTURAL weakest links — per-line single-trip algebraic-connectivity
//!     loss (Weyl/Fiedler; limit-independent → pure topology).
//!  2. The CHEEGER local boundary — the min-conductance sweep cut: which buses
//!     sit on the small side and how many lines cross it (the seam that flaps).
//!  3. OPERATIONAL weakest links — per-line N-1 cascade size under
//!     self-calibrated limits (which seed trip cascades furthest).
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example weakest_links -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES

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

fn synthetic(rows: usize, cols: usize) -> (Grid, Vec<String>) {
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
    let ids = (0..rows * cols).map(|i| i.to_string()).collect();
    (Grid::new(rows * cols, edges), ids)
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
        println!(
            "grid: {cc} PyPSA core — {} buses, {} lines\n",
            imp.grid.n,
            imp.grid.edges.len()
        );
        (imp.grid, imp.bus_ids)
    } else {
        let (g, ids) = synthetic(6, 6);
        println!(
            "grid: synthetic 6×6 — {} buses, {} lines\n",
            g.n,
            g.edges.len()
        );
        (g, ids)
    };

    let n = grid.n;
    let m = grid.edges.len();
    let alive = vec![true; m];
    let lbl = |e: usize| format!("{}–{}", ids[grid.edges[e].from], ids[grid.edges[e].to]);

    // 1. STRUCTURAL weakest links via first-order Fiedler sensitivity.
    //    ∂λ₂/∂wₑ = (v₂[a]−v₂[b])²  (exact derivative), so removing line e drops
    //    λ₂ by ≈ (v₂[a]−v₂[b])²·bₑ to first order. One eigensolve ranks all m
    //    lines; the exact λ₂-loss is then recomputed only for the top few.
    let base_eig = symmetric_eigen(&grid.laplacian_of(&alive), n);
    let lam2 = base_eig.values.get(1).copied().unwrap_or(0.0);
    let v2 = base_eig.eigenvector(1);
    let mut struct_rank: Vec<(usize, f64)> = (0..m)
        .map(|e| {
            let (a, b) = (grid.edges[e].from, grid.edges[e].to);
            let d = v2[a] - v2[b];
            (e, d * d * grid.edges[e].susceptance) // first-order Δλ₂ proxy
        })
        .collect();
    struct_rank.sort_by(|x, y| y.1.total_cmp(&x.1));

    println!("== 1. Structural weakest links (first-order Fiedler sensitivity ∂λ₂/∂wₑ) ==");
    println!("   base λ₂ = {lam2:.3e}");
    let mut bridges = 0usize;
    for (e, sens) in struct_rank.iter().take(10) {
        // Exact recompute for the top lines only.
        let mut after = alive.clone();
        after[*e] = false;
        let lam2_after = symmetric_eigen(&grid.laplacian_of(&after), n)
            .values
            .get(1)
            .copied()
            .unwrap_or(0.0);
        let loss = if lam2 > 1e-12 {
            1.0 - lam2_after / lam2
        } else {
            0.0
        };
        let splits = lam2_after < 1e-9;
        if splits {
            bridges += 1;
        }
        println!(
            "  line {e:>4}  {:<16}  sens {:>10.3e}  exact λ₂-loss {:>6.2}%{}",
            lbl(*e),
            sens,
            100.0 * loss,
            if splits {
                "   ← BRIDGE (trip disconnects the core)"
            } else {
                ""
            }
        );
    }
    println!("  → {bridges}/10 top-sensitivity lines are bridges\n");

    // 2. CHEEGER local boundary — where the grid wants to separate (the flap).
    let c = cheeger_sweep(&grid, &alive);
    let small = c.partition.iter().filter(|&&b| b).count();
    let cut_lines: Vec<usize> = (0..m)
        .filter(|&e| c.partition[grid.edges[e].from] != c.partition[grid.edges[e].to])
        .collect();
    println!("== 2. Cheeger local boundary (the seam that flaps) ==");
    println!("  μ₂ (normalized gap)      : {:.5}", c.mu2);
    println!(
        "  conductance φ of the cut : {:.5}   (Cheeger {:.5} ≤ h ≤ {:.5})",
        c.conductance, c.lower, c.upper
    );
    println!(
        "  partition                : {small} | {} buses (small side | rest)",
        n - small
    );
    println!("  the boundary crosses {} lines:", cut_lines.len());
    for &e in cut_lines.iter().take(8) {
        println!("     line {e:>4}  {}", lbl(e));
    }
    if cut_lines.len() > 8 {
        println!("     … +{} more", cut_lines.len() - 8);
    }
    println!();

    // 3. OPERATIONAL weakest links: N-1 cascade size under self-calibrated limits.
    let mut rng = Rng(0xBEEF);
    let raw: Vec<f64> = (0..n).map(|_| rng.f()).collect();
    let mean = raw.iter().sum::<f64>() / n as f64;
    let p: Vec<f64> = raw.iter().map(|x| x - mean).collect();
    let mut g = grid.clone();
    let eig = symmetric_eigen(&g.laplacian_of(&alive), n);
    let base = dc_flows(&g, &alive, &eig.pseudo_apply(&p, 1e-9));
    for (e, edge) in g.edges.iter_mut().enumerate() {
        edge.limit = (1.1 * base[e].abs()).max(1e-6);
    }
    // Cascade only the top structural candidates (full N-1 is O(m·rounds)
    // eigensolves — intractable at m=348); bound rounds too.
    let cfg = CascadeConfig {
        max_rounds: 16,
        ..CascadeConfig::default()
    };
    let candidates: Vec<usize> = struct_rank.iter().take(25).map(|x| x.0).collect();
    let mut op_rank: Vec<(usize, usize, f64, bool)> = candidates
        .iter()
        .map(|&e| {
            let r = simulate_outage(&g, &p, e, cfg);
            (e, r.shape.n_tripped(), r.fraction_tripped, r.islanded)
        })
        .collect();
    op_rank.sort_by_key(|x| std::cmp::Reverse(x.1));

    println!("== 3. Operational weakest links (cascade size of the top-25 structural candidates, headroom ×1.1) ==");
    for (e, ntrip, frac, islanded) in op_rank.iter().take(10) {
        println!(
            "  seed {e:>4}  {:<16}  → {ntrip:>3} lines ({:>4.1}%){}",
            lbl(*e),
            100.0 * frac,
            if *islanded { "   ISLANDS the grid" } else { "" }
        );
    }
    let big = op_rank.iter().filter(|(_, nt, _, _)| *nt >= 3).count();
    println!(
        "  → {big}/{} candidate seed trips cascade to ≥3 lines under 10% headroom\n",
        candidates.len()
    );

    println!(
        "Reads: structural rank = WHERE the grid is topologically thin (bridges/cut);\n\
         the Cheeger boundary = the seam it separates along; operational rank = WHICH\n\
         seed trips snowball once loaded. Synthetic injections + estimated limits —\n\
         feed real ENTSO-E/ESIOS load for the operational ranking; significance via Jirak."
    );
}
