//! Run the cascade on a REAL grid harvested from open data.
//!
//! Expects the PyPSA-Eur / OSM prebuilt network (Zenodo 13358976, ODbL):
//!
//! ```sh
//! mkdir -p /tmp/pypsa && cd /tmp/pypsa
//! curl -L -o buses.csv 'https://zenodo.org/records/13358976/files/buses.csv?download=1'
//! curl -L -o lines.csv 'https://zenodo.org/records/13358976/files/lines.csv?download=1'
//! cargo run --manifest-path crates/perturbation-sim/Cargo.toml --example iberian -- \
//!     /tmp/pypsa/buses.csv /tmp/pypsa/lines.csv ES
//! ```
//!
//! Args: <buses.csv> <lines.csv> [country=ES] [headroom=1.10]. With no usable
//! args it prints these instructions and exits 0 (so `cargo run` never errors).
//! See `crates/perturbation-sim/HARVESTING.md` for other sources/countries.

use perturbation_sim::{from_pypsa_csv, simulate_outage, CascadeConfig};
use std::fs;

/// Deterministic 64-bit PRNG (Knuth golden constant), so the synthetic
/// injection pattern is reproducible run-to-run.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn usage() {
    eprintln!(
        "usage: iberian <buses.csv> <lines.csv> [country=ES] [headroom=1.10]\n\n\
         Download the PyPSA-Eur/OSM network (Zenodo 13358976, ODbL):\n  \
         curl -L -o buses.csv 'https://zenodo.org/records/13358976/files/buses.csv?download=1'\n  \
         curl -L -o lines.csv 'https://zenodo.org/records/13358976/files/lines.csv?download=1'\n\n\
         Then re-run with the two paths. See HARVESTING.md for other sources/countries."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        usage();
        return;
    }
    let country = args.get(3).map(|s| s.as_str()).unwrap_or("ES");
    let headroom: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.10);

    let buses = match fs::read_to_string(&args[1]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[1]);
            usage();
            return;
        }
    };
    let lines = match fs::read_to_string(&args[2]) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[2]);
            usage();
            return;
        }
    };

    let full = match from_pypsa_csv(&buses, &lines, Some(country)) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("import failed: {e}");
            return;
        }
    };
    println!(
        "=== perturbation-sim :: {country} grid from PyPSA-Eur/OSM (Zenodo 13358976, ODbL) ==="
    );
    println!(
        "parsed: {} buses, {} lines  (reactance estimated: {}, limit estimated: {}, cross-border lines dropped: {})",
        full.grid.n,
        full.grid.edges.len(),
        full.n_estimated_reactance,
        full.n_estimated_limit,
        full.n_dropped_lines
    );

    // A country extract is fragmented (cross-border ties dropped, OSM gaps), so
    // run the contingency study on the connected core.
    let mut imp = full.largest_component();
    let n = imp.grid.n;
    println!(
        "largest connected component: {n} buses, {} lines (the rest are disjoint islands)\n",
        imp.grid.edges.len()
    );

    // Synthetic balanced injection (no generation/load in the topology CSV —
    // those come from ENTSO-E/ESIOS; see HARVESTING.md). Deterministic.
    let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
    let raw: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
    let mean = raw.iter().sum::<f64>() / n as f64;
    let p: Vec<f64> = raw.iter().map(|r| r - mean).collect();

    // Self-calibrate limits to headroom × base loading so a contingency can tip
    // the network (the real s_nom values are kept in the Grid but a synthetic
    // injection won't match them; real injection + s_nom is the next step).
    let all = vec![true; imp.grid.edges.len()];
    let eig = perturbation_sim::symmetric_eigen(&imp.grid.laplacian_of(&all), n);
    let theta = eig.pseudo_apply(&p, 1e-9);
    let base = perturbation_sim::dc_flows(&imp.grid, &all, &theta);
    for (e, edge) in imp.grid.edges.iter_mut().enumerate() {
        edge.limit = (headroom * base[e].abs()).max(1e-6);
    }

    let seed = base
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let r = simulate_outage(&imp.grid, &p, seed, CascadeConfig::default());

    let se = &imp.grid.edges[seed];
    println!(
        "Seed trip: line {seed}  ({} — {})   base flow {:.4}\n",
        imp.bus_ids[se.from], imp.bus_ids[se.to], base[seed]
    );

    let s = &r.spectral;
    println!("-- Spectral perturbation (rank-1, ‖E‖₂ = 2·b_k) --");
    println!(
        "  ‖E‖₂={:.5}  max|Δλ|={:.5}  Weyl: {}",
        s.e_norm,
        s.max_eigenvalue_shift,
        if s.weyl_satisfied {
            "HOLDS ✓"
        } else {
            "VIOLATED ✗"
        }
    );
    println!(
        "  Fiedler λ₂ {:.6} → {:.6}  (connectivity loss {:.2}%)",
        s.fiedler_before,
        s.fiedler_after,
        100.0 * s.connectivity_loss()
    );

    println!("\n-- Cascade --");
    println!(
        "  rounds {}   tripped {}/{} ({:.1}%)   islanded {} (components {})",
        r.rounds,
        r.shape.n_tripped(),
        imp.grid.edges.len(),
        100.0 * r.fraction_tripped,
        r.islanded,
        r.components_final
    );

    println!("\n-- Perturbation-shape epicentre (top 10 buses by |Δθ|) --");
    for (bus, mag) in r.shape.epicentre(10) {
        println!(
            "  {:>14}  |Δθ|={:.5}   ({:.3}, {:.3})",
            imp.bus_ids[bus], mag, imp.lon[bus], imp.lat[bus]
        );
    }

    println!(
        "\nNext: feed real generation/load (ENTSO-E/ESIOS) as injections, use the\n      \
         imported s_nom limits, and correlate this node_field against the OBSERVED\n      \
         outage footprint via ndarray::hpc::reliability (ICC), Jirak-significance."
    );
}
