//! CAKES + CHAODA over the HHTL basins of a real grid core — the similarity
//! (attraction) / anomaly (repulsion) pair from the CLAM family, scoring grid
//! compartments the way CHAODA scores papillary muscles or terrain tiles.
//!
//! HHTL is the family basin ("where"); CAKES finds each basin's relatives ("who
//! looks similar"); CHAODA scores how far a basin sits from its family ("why am I
//! different") — the top-anomaly basin is the fail-first compartment.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example chaoda -- /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES
//! (no args → a synthetic 8×8 grid with one deliberately weakened block).

use perturbation_sim::{
    anomaly_ranking, cakes_neighbors, chaoda_scores, resilience_basin_features, Edge, Grid,
    HhtlKey, CHAODA_FLAG,
};

/// Synthetic 8×8 lattice with one corner block weakened (low-susceptance internal
/// edges) — a planted fail-first compartment for the no-data demo.
fn synthetic() -> Grid {
    let (rows, cols) = (8usize, 8usize);
    let id = |r: usize, c: usize| r * cols + c;
    let mut e = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            // The 3×3 top-left block is brittle (weak internal coupling).
            let brittle = r < 3 && c < 3;
            let b = if brittle { 0.05 } else { 1.0 };
            if c + 1 < cols {
                e.push(Edge::new(id(r, c), id(r, c + 1), b, 1.0));
            }
            if r + 1 < rows {
                e.push(Edge::new(id(r, c), id(r + 1, c), b, 1.0));
            }
        }
    }
    Grid::new(rows * cols, e)
}

/// Deterministic per-bus inertia proxy (real `H` is not in PyPSA topology): a small
/// fixed cycle, decoupled from wiring, so the buffer axis is an independent input —
/// the structure (basins as families, the outlier as fail-first) holds regardless.
fn proxy_inertia(n: usize) -> Vec<f64> {
    (0..n).map(|i| 2.0 + (i % 5) as f64).collect()
}

fn fmt_key(k: &HhtlKey) -> String {
    format!("{}.{}.{}", k.heel, k.hip, k.twig)
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
        let g = synthetic();
        println!(
            "grid: synthetic 8×8 with a planted brittle 3×3 block — {} buses",
            g.n
        );
        g
    };

    let h = proxy_inertia(grid.n);
    let (basins, rows) = resilience_basin_features(&grid, &h, 0.2);
    let scores = chaoda_scores(&rows, 2);

    println!("\n== HHTL family basins — CAKES (similar) + CHAODA (anomalous) ==");
    println!(
        "  {:>10}  {:>6}  {:>8}  {:>8}  {:>8}  flag",
        "basin", "size", "λ₂n", "inertia", "CHAODA"
    );
    for (i, k) in basins.iter().enumerate() {
        let flag = if scores[i] >= CHAODA_FLAG {
            "◀ ANOMALY"
        } else {
            ""
        };
        println!(
            "  {:>10}  {:>6.2}  {:>8.3}  {:>8.3}  {:>8.3}  {}",
            fmt_key(k),
            rows[i][1],
            rows[i][0],
            rows[i][2],
            scores[i],
            flag
        );
    }

    // CHAODA: the fail-first compartment is the top anomaly.
    let rank = anomaly_ranking(&rows, 2);
    if let Some(&(top, score)) = rank.first() {
        println!(
            "\n  CHAODA → fail-first compartment: basin {} (score {:.3}{})\n  \
             \"why am I different\" — the basin whose resilience profile deviates most\n  \
             from its family. {} basins; flag threshold {CHAODA_FLAG}.",
            fmt_key(&basins[top]),
            score,
            if score >= CHAODA_FLAG {
                ", FLAGGED"
            } else {
                ""
            },
            basins.len()
        );

        // CAKES: the top-anomaly basin's nearest relatives ("who looks similar").
        let nbrs = cakes_neighbors(&rows, top, 3);
        let rel: Vec<String> = nbrs
            .iter()
            .map(|(i, d)| format!("{} (d={:.3})", fmt_key(&basins[*i]), d))
            .collect();
        println!(
            "  CAKES → {}'s nearest relatives: [{}]\n  \
             attraction vs repulsion: the family it resembles, and how far it still sits from them.",
            fmt_key(&basins[top]),
            rel.join(", ")
        );
    }

    println!(
        "\n  CAKES pulls in the similar; CHAODA pushes out the unusual.\n  \
         Family basin (HHTL) + deviation-from-family (CHAODA) = the fail-first locator.\n  \
         (CHAODA-lite kNN scorer; ndarray::clam's ClamTree ensemble is the gated production path.)"
    );
}
