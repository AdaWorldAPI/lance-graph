//! Probe 3 — the per-bus inertia (H) ingest path: parse a `bus,H` table, align it
//! to the grid's bus order, disclose the proxy fallback, and feed the
//! inertia-buffer column. No external data is bundled; this uses an inline fixture
//! standing in for an ENTSO-E / ESIOS / TSO inertia export.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example inertia_ingest

use perturbation_sim::{
    inertia_buffer_column, inertia_for_buses, parse_bus_inertia, proxy_inertia,
};

// Stand-in for an operator inertia export (real H is never committed).
const INERTIA_CSV: &str = "\
bus_id,inertia_h,source
ES_NUC_1,6.5,nuclear
ES_HYD_1,4.0,hydro
ES_GAS_1,5.5,gas-synchronous
ES_WND_1,0.8,wind-synthetic
";

fn main() {
    // The grid's buses (would come from PypsaImport.bus_ids). ES_SOLAR_1 has no
    // measured H → it falls back to the proxy value, and that is disclosed.
    let bus_ids: Vec<String> = ["ES_NUC_1", "ES_HYD_1", "ES_GAS_1", "ES_WND_1", "ES_SOLAR_1"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let measured = parse_bus_inertia(INERTIA_CSV);
    let fallback = 1.0; // low-inertia stand-in for an undocumented (likely inverter) bus
    let (h, prov) = inertia_for_buses(&bus_ids, &measured, fallback);
    let col = inertia_buffer_column(&h, 0.2);

    println!("per-bus inertia ingest (measured H → grid order → buffer column)\n");
    println!(
        "  {:>12}  {:>8}  {:>10}  source",
        "bus", "H (s)", "buffer_n"
    );
    for (i, id) in bus_ids.iter().enumerate() {
        let src = if measured.contains_key(id) {
            "measured"
        } else {
            "proxy"
        };
        println!("  {:>12}  {:>8.2}  {:>10.3}  {src}", id, h[i], col[i]);
    }
    println!(
        "\n  provenance: {} measured, {} proxy (disclose the proxy fraction, like\n  \
         PypsaImport's n_estimated_* counters).",
        prov.measured, prov.proxy
    );

    // No-data path: a fully deterministic, topology-blind proxy field.
    let demo = proxy_inertia(bus_ids.len(), 2.0, 6.0, 0xDEFA17);
    println!(
        "\n  no-data fallback — proxy_inertia(n=5, base=2, span=6, seed=0xDEFA17):\n  \
         [{}]\n  decoupled from wiring on purpose (buffer ⊥ topology); deterministic, never\n  \
         bundled. Wire measured H when an operator export is available; the column math\n  \
         is identical either way.",
        demo.iter()
            .map(|h| format!("{h:.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}
