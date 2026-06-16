//! End-to-end demo: build a regional transmission lattice, stress it, trip the
//! most-loaded line, and print the perturbation shape of the resulting cascade.
//!
//! Run: `cargo run --manifest-path crates/perturbation-sim/Cargo.toml --example simulate`

use perturbation_sim::{simulate_outage, CascadeConfig, Edge, Grid};

/// A `rows × cols` lattice of buses with unit-susceptance lines, line limits
/// set to `headroom × |base flow|` (so the network sits just inside its limits
/// and a single outage can tip it). Returns the grid and the balanced injection
/// vector (generators across the top row, loads across the bottom row).
fn build_lattice(rows: usize, cols: usize, headroom: f64) -> (Grid, Vec<f64>) {
    let n = rows * cols;
    let id = |r: usize, c: usize| r * cols + c;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                edges.push(Edge::new(id(r, c), id(r, c + 1), 1.0, 0.0));
            }
            if r + 1 < rows {
                edges.push(Edge::new(id(r, c), id(r + 1, c), 1.0, 0.0));
            }
        }
    }
    // Balanced injections: +1/cols at each top bus, −1/cols at each bottom bus.
    let mut p = vec![0.0; n];
    let gen = 1.0 / cols as f64;
    for c in 0..cols {
        p[id(0, c)] += gen;
        p[id(rows - 1, c)] -= gen;
    }

    let mut grid = Grid::new(n, edges);

    // Solve base flows once, then set each limit from the base loading.
    let all = vec![true; grid.edges.len()];
    let eig = perturbation_sim::symmetric_eigen(&grid.laplacian_of(&all), n);
    let theta = eig.pseudo_apply(&p, 1e-9);
    let f = perturbation_sim::dc_flows(&grid, &all, &theta);
    for (e, edge) in grid.edges.iter_mut().enumerate() {
        edge.limit = (headroom * f[e].abs()).max(1e-3);
    }
    (grid, p)
}

fn bar(x: f64, max: f64, width: usize) -> String {
    if max <= 0.0 {
        return String::new();
    }
    let filled = ((x / max) * width as f64).round() as usize;
    "█".repeat(filled.min(width))
}

fn main() {
    let (rows, cols) = (4, 4);
    let (grid, p) = build_lattice(rows, cols, 1.15);

    // Seed the cascade by tripping the most-loaded line in the base case.
    let all = vec![true; grid.edges.len()];
    let eig = perturbation_sim::symmetric_eigen(&grid.laplacian_of(&all), grid.n);
    let theta = eig.pseudo_apply(&p, 1e-9);
    let base = perturbation_sim::dc_flows(&grid, &all, &theta);
    let seed = base
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let r = simulate_outage(&grid, &p, seed, CascadeConfig::default());

    println!(
        "=== perturbation-sim :: {rows}×{cols} lattice, {} lines ===\n",
        grid.edges.len()
    );
    let se = &grid.edges[seed];
    println!(
        "Seed trip: line {seed}  (bus {} — bus {}), base flow {:.4}\n",
        se.from, se.to, base[seed]
    );

    println!("-- Spectral perturbation of the seed trip (rank-1, ‖E‖₂ = 2·b_k) --");
    let s = &r.spectral;
    println!("  ‖E‖₂ (Weyl budget)        : {:.6}", s.e_norm);
    println!(
        "  max |Δλ| (realized)       : {:.6}",
        s.max_eigenvalue_shift
    );
    println!(
        "  Weyl |Δλ| ≤ ‖E‖₂          : {}",
        if s.weyl_satisfied {
            "HOLDS ✓"
        } else {
            "VIOLATED ✗"
        }
    );
    println!(
        "  Fiedler λ₂  before → after: {:.6} → {:.6}",
        s.fiedler_before, s.fiedler_after
    );
    println!(
        "  algebraic-connectivity loss: {:.2}%",
        100.0 * s.connectivity_loss()
    );
    println!(
        "  Fiedler rotation sinθ      : {:.6}  (Davis–Kahan bound {:.6})",
        s.fiedler_rotation_sin, s.davis_kahan_bound
    );

    println!("\n-- Cascade (edge propagation) --");
    println!("  rounds                    : {}", r.rounds);
    println!(
        "  lines tripped             : {} / {}  ({:.1}%)",
        r.shape.n_tripped(),
        grid.edges.len(),
        100.0 * r.fraction_tripped
    );
    println!(
        "  islanded                  : {}  (final components: {})",
        r.islanded, r.components_final
    );

    println!("\n-- Tripped lines, by round --");
    let mut order: Vec<(usize, i32)> = r
        .shape
        .trip_round
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, rd)| rd >= 0)
        .collect();
    order.sort_by_key(|&(_, rd)| rd);
    for (e, rd) in order {
        let edge = &grid.edges[e];
        println!(
            "  round {rd:>2}: line {e:>2}  bus {} — bus {}   (carried {:.4})",
            edge.from, edge.to, base[e]
        );
    }

    println!("\n-- Perturbation shape: per-bus |Δθ| (the red-edge epicentre) --");
    let max_node = r.shape.node_field.iter().cloned().fold(0.0_f64, f64::max);
    for bus in 0..grid.n {
        let v = r.shape.node_field[bus];
        println!(
            "  bus {bus:>2} (r{},c{}) {:.5}  {}",
            bus / cols,
            bus % cols,
            v,
            bar(v, max_node, 32)
        );
    }
    println!("\n  epicentre (top 3 buses): {:?}", r.shape.epicentre(3));

    println!(
        "\nNote: node_field is ready for predicted-vs-observed validity\n      \
         (ndarray::hpc::reliability — Pearson/Spearman/ICC); use the Jirak\n      \
         n^(p/2−1) weak-dependence rate for significance, not IID Berry–Esseen."
    );
}
