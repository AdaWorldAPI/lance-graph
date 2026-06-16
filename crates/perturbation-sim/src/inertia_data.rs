//! Probe 3 — per-bus inertia (`H`) ingest path.
//!
//! Real per-bus inertia constants `H` (seconds of stored rotational energy) are
//! **not** in the PyPSA/OSM topology (`buffer.rs` flags this). They come from
//! operator data: an ENTSO-E generation inventory, ESIOS dispatch, or a TSO inertia
//! estimate, mapped to buses. This module is the **ingest path** — parse a `bus,H`
//! table when available, align it to the grid's bus order ([`crate::PypsaImport::bus_ids`]),
//! and **disclose** how many buses fell back to a proxy (mirroring `ingest.rs`'s
//! `n_estimated_*` honesty). The result feeds [`crate::inertia_buffer_column`].
//!
//! **No external data is bundled** (network/data policy): the caller supplies the
//! file contents (the lib takes `&str`, like `from_pypsa_csv`), or uses the
//! deterministic [`proxy_inertia`] for a no-data demo. A proxy is decoupled from
//! wiring on purpose — the buffer axis is storage, orthogonal to topology, so a
//! topology-blind proxy is the honest stand-in until measured `H` is wired.

use std::collections::HashMap;

/// Provenance of an aligned per-bus inertia vector — how many buses carried a
/// measured `H` vs a proxy fallback. Disclose it the way `PypsaImport` discloses
/// estimated reactance/limit counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InertiaProvenance {
    /// Buses whose `H` came from the measured table.
    pub measured: usize,
    /// Buses that fell back to the proxy value.
    pub proxy: usize,
}

/// Parse a `bus,H` table (CSV, `,` or `;`): a header naming a bus column
/// (`bus_id` / `name` / `bus` / `bus0`) and an inertia column (`h` / `inertia` /
/// `inertia_h` / `inertia_s` / `h_s`). Returns `bus_id → H`. Rows with an
/// unparseable / non-positive `H` are skipped. Unknown columns are ignored, so an
/// ENTSO-E/ESIOS export with extra fields parses as-is.
pub fn parse_bus_inertia(text: &str) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        return out;
    };
    let delim = if header.matches(';').count() > header.matches(',').count() {
        ';'
    } else {
        ','
    };
    let cols: Vec<String> = header
        .split(delim)
        .map(|h| h.trim().to_ascii_lowercase())
        .collect();
    let find = |names: &[&str]| -> Option<usize> {
        names.iter().find_map(|w| cols.iter().position(|c| c == w))
    };
    let (Some(bi), Some(hi)) = (
        find(&["bus_id", "name", "bus", "bus0"]),
        find(&["h", "inertia", "inertia_h", "inertia_s", "h_s"]),
    ) else {
        return out;
    };
    for row in lines {
        let f: Vec<&str> = row.split(delim).collect();
        let (Some(b), Some(h)) = (f.get(bi), f.get(hi)) else {
            continue;
        };
        let bus = b.trim().to_string();
        if bus.is_empty() {
            continue;
        }
        if let Ok(val) = h.trim().parse::<f64>() {
            if val > 0.0 {
                out.insert(bus, val);
            }
        }
    }
    out
}

/// Align a measured `bus → H` map to the grid's bus order, filling missing buses
/// with `fallback`. Returns the per-bus `H` vector (length `bus_ids.len()`) plus its
/// [`InertiaProvenance`] so the caller can disclose the proxy fraction.
pub fn inertia_for_buses(
    bus_ids: &[String],
    measured: &HashMap<String, f64>,
    fallback: f64,
) -> (Vec<f64>, InertiaProvenance) {
    let mut h = Vec::with_capacity(bus_ids.len());
    let mut n_measured = 0usize;
    for id in bus_ids {
        match measured.get(id) {
            Some(&v) => {
                n_measured += 1;
                h.push(v);
            }
            None => h.push(fallback),
        }
    }
    (
        h,
        InertiaProvenance {
            measured: n_measured,
            proxy: bus_ids.len() - n_measured,
        },
    )
}

/// Deterministic per-bus inertia proxy in `[base, base+span]` (SplitMix64-seeded,
/// decoupled from wiring). Honest no-data stand-in: the buffer axis is storage, so a
/// topology-blind proxy preserves buffer ⊥ topology. Same `(n, base, span, seed)` ⇒
/// same vector.
pub fn proxy_inertia(n: usize, base: f64, span: f64, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            base + span * (z as f64 / u64::MAX as f64)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inertia_buffer_column;

    const INERTIA_CSV: &str = "\
bus_id,inertia_h,source
A,5.0,nuclear
B,2.5,wind
C,0.0,solar
D,abc,bad
E,8.0,hydro
";

    #[test]
    fn parses_bus_inertia_skipping_bad_rows() {
        let m = parse_bus_inertia(INERTIA_CSV);
        assert_eq!(m.len(), 3, "C (H=0) and D (unparseable) are skipped");
        assert_eq!(m["A"], 5.0);
        assert_eq!(m["B"], 2.5);
        assert_eq!(m["E"], 8.0);
        assert!(!m.contains_key("C") && !m.contains_key("D"));
    }

    #[test]
    fn aligns_to_grid_order_and_discloses_proxy_fill() {
        let m = parse_bus_inertia(INERTIA_CSV);
        // Grid has A,B,X (X has no measured H) → X uses the fallback.
        let bus_ids = vec!["A".to_string(), "B".to_string(), "X".to_string()];
        let (h, prov) = inertia_for_buses(&bus_ids, &m, 1.0);
        assert_eq!(h, vec![5.0, 2.5, 1.0]);
        assert_eq!(prov.measured, 2);
        assert_eq!(prov.proxy, 1);
    }

    #[test]
    fn proxy_is_deterministic_and_bounded() {
        let a = proxy_inertia(64, 2.0, 6.0, 0xC0FFEE);
        let b = proxy_inertia(64, 2.0, 6.0, 0xC0FFEE);
        assert_eq!(a, b, "same seed ⇒ same proxy");
        assert!(
            a.iter().all(|&h| (2.0..=8.0).contains(&h)),
            "in [base, base+span]"
        );
        // A different seed gives a different vector (decoupled, not constant).
        assert_ne!(a, proxy_inertia(64, 2.0, 6.0, 0xBEEF));
    }

    #[test]
    fn ingested_h_feeds_the_inertia_buffer_column() {
        let m = parse_bus_inertia(INERTIA_CSV);
        let bus_ids = vec!["A".to_string(), "B".to_string(), "E".to_string()];
        let (h, _) = inertia_for_buses(&bus_ids, &m, 1.0);
        let col = inertia_buffer_column(&h, 0.2);
        // E (H=8) is max → 1.0, B (H=2.5) is min → 0.0; all normalized.
        assert_eq!(col.len(), 3);
        assert_eq!(col[2], 1.0);
        assert_eq!(col[1], 0.0);
        assert!(col.iter().all(|&c| (0.0..=1.0).contains(&c)));
    }
}
