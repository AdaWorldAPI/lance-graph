//! The 8 batching-method PRESETS — every batching/ownership/delivery
//! method this probe measured, frozen as named, reproducible
//! configurations so any session can run serious lab sweeps without
//! re-deriving knob combinations.
//!
//! Findings (agnostic numbers): `FINDINGS.md`. Interpretation (one
//! session's reading): `COMMENTARY.md`. Both at the crate root.
//!
//! Every preset has the same signature — `(data, workers) → map` — and
//! every preset's output is asserted byte-identical to lane A by the
//! `all_presets_agree_with_lane_a` test, so a sweep never needs its own
//! correctness harness.

use crate::Stats;
use std::collections::BTreeMap;

/// One frozen batching-method configuration.
pub struct Preset {
    pub id: u8,
    pub name: &'static str,
    /// What the preset does, mechanism-level (agnostic).
    pub description: &'static str,
    /// Which measured lane/knobs it freezes (for cross-reference into
    /// `FINDINGS.md`'s tables).
    pub frozen_from: &'static str,
}

/// The preset catalogue. Ordered from least to most machinery.
pub const PRESETS: [Preset; 8] = [
    Preset {
        id: 0,
        name: "map-private-merge",
        description: "Per-worker BTreeMap accumulation, one commutative \
                      merge at the end. No actors, no batching, no witness.",
        frozen_from: "lane C",
    },
    Preset {
        id: 1,
        name: "grid-private-merge",
        description: "Per-worker flat Morton-tile SoA table (64K cells, \
                      open-addressed), one merge at the end. No actors, \
                      no witness. The fastest measured shape.",
        frozen_from: "lane F",
    },
    Preset {
        id: 2,
        name: "stream-single-owner",
        description: "Workers pre-reduce 64K-row morsels and stream \
                      dirty entries to ONE owner mailbox actor holding \
                      the canonical SoA; every applied batch witnessed.",
        frozen_from: "lane G, shards=1",
    },
    Preset {
        id: 3,
        name: "orchestrated-lazy-owners",
        description: "Router tier with LAZY per-tile owner activation \
                      (live mailboxes track occupancy, not address \
                      space) + ahead-firing batched delivery (batch_k \
                      = 64). Nominal granularity 65536.",
        frozen_from: "lane H, owners_nominal=65536",
    },
    Preset {
        id: 4,
        name: "batch-64k-registry",
        description: "The full batch pipeline at 64K grid WITH the \
                      standing per-cell mailbox registry: codebook CAM \
                      addressing, whole-table Arc double-cast to \
                      ownership + persistence sinks, flush cache. \
                      Registry residency is the measured cost.",
        frozen_from: "lane I (== lane J: grid=65536, lanes=1, registry=on)",
    },
    Preset {
        id: 5,
        name: "gridlake",
        description: "The measured sweet spot: 64x64 gridlake batch SoA \
                      (4096 cells, ~80 KB integer-exact), codebook CAM \
                      addressing, 1 sink lane pair, whole-table Arc \
                      double-cast, flush cache, NO standing registry.",
        frozen_from: "lane J: grid=4096, lanes=1, registry=off",
    },
    Preset {
        id: 6,
        name: "gridlake-8-lanes",
        description: "gridlake with 8 ownership+persistence sink lane \
                      pairs (row-range sliced) — free at light apply \
                      work; headroom for heavy per-batch apply.",
        frozen_from: "lane J: grid=4096, lanes=8, registry=off",
    },
    Preset {
        id: 7,
        name: "batch-64k-no-registry",
        description: "The batch pipeline at the full 64K grid without \
                      the registry — the grid-size control against \
                      preset 5 (isolates the cache-matching win).",
        frozen_from: "lane J: grid=65536, lanes=1, registry=off",
    },
];

/// Run preset `id` (0..=7). Panics on an unknown id — sweeps should
/// iterate [`PRESETS`].
pub fn run_preset(id: u8, data: &[u8], workers: usize) -> BTreeMap<String, Stats> {
    match id {
        0 => crate::lane_c_threads(data, workers),
        1 => crate::lane_f_morton(data, workers),
        2 => crate::lane_g::lane_g_kanban_soa(data, workers, 1),
        3 => crate::lane_h::lane_h_orchestrated(data, workers, 65536),
        4 => crate::lane_j::lane_j_grid_pipeline_with(data, workers, 65536, 1, true, 1 << 16),
        5 => crate::lane_j::lane_j_grid_pipeline_with(data, workers, 4096, 1, false, 1 << 16),
        6 => crate::lane_j::lane_j_grid_pipeline_with(data, workers, 4096, 8, false, 1 << 16),
        7 => crate::lane_j::lane_j_grid_pipeline_with(data, workers, 65536, 1, false, 1 << 16),
        other => panic!("unknown preset {other} (valid: 0..=7)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset must produce byte-identical aggregates to lane A —
    /// the single correctness harness a lab sweep inherits for free.
    /// (Small corpus; preset 4 spawns its full registry, so this test
    /// is the slowest in the crate by design.)
    #[test]
    fn all_presets_agree_with_lane_a() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("onebrc_probe_test_p_{}.txt", std::process::id()));
        let result = crate::gen::gen(&path, 30_000, 7).expect("gen");
        assert_eq!(result.rows, 30_000);

        let data = std::fs::read(&path).expect("read generated corpus");
        std::fs::remove_file(&path).ok();

        let a = crate::lane_a_scalar(&data);
        for preset in PRESETS.iter() {
            let out = run_preset(preset.id, &data, 3);
            assert_eq!(
                a, out,
                "preset {} ({}) must match lane A",
                preset.id, preset.name
            );
        }
        assert!(!a.is_empty());
    }
}
