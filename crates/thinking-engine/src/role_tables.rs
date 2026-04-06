//! Per-role BF16 distance tables with gate modulation.
//!
//! Builds one BF16 distance table per role from StackedN centroids:
//!
//! ```text
//! Role     Gate Modulation    Formula
//! ────     ───────────────    ───────
//! Q        NONE (extern)      cos(Q_i, Q_j) → BF16
//! K        NONE (attn proj)   cos(K_i, K_j) → BF16
//! V        NONE (attn proj)   cos(V_i, V_j) → BF16
//! Gate     NONE (topology)    cos(Gate_i, Gate_j) → BF16
//! Up       silu(gate) × Up    cos(silu(g)⊙Up_i, silu(g)⊙Up_j) → BF16
//! Down     NONE (funnel)      cos(Down_i, Down_j) → BF16
//! ```
//!
//! Only Up gets gate modulation (FFN activation).
//! The 33% correction lives HERE — raw Up vs silu(gate)×Up.

use crate::bf16_engine::BF16ThinkingEngine;
use bgz_tensor::stacked_n::{StackedN, ClamCodebook, bf16_to_f32, f32_to_bf16};
use ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd;

/// SiLU activation: x / (1 + exp(-x))
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply silu(gate) elementwise to a role vector.
/// Returns the gate-modulated activation: silu(gate[k]) × role[k]
pub fn gate_modulate(gate_f32: &[f32], role_f32: &[f32]) -> Vec<f32> {
    gate_f32.iter().zip(role_f32)
        .map(|(&g, &r)| silu(g) * r)
        .collect()
}

/// Build a BF16 distance table from raw StackedN centroids (no gate modulation).
/// Used for: Q, K, V, Gate, Down.
pub fn build_raw_table(codebook: &ClamCodebook) -> BF16ThinkingEngine {
    BF16ThinkingEngine::from_codebook(codebook)
}

/// Build a gate-modulated BF16 distance table.
/// Used for: Up (the 33% correction).
///
/// For each centroid pair (i, j):
///   1. Hydrate gate centroid i and Up centroid i to f32
///   2. Compute silu(gate_i) ⊙ up_i (elementwise)
///   3. Same for j
///   4. Cosine of the activated vectors → BF16
///
/// Requires gate and role codebooks with SAME centroid count and SPD.
pub fn build_gate_modulated_table(
    gate_codebook: &ClamCodebook,
    role_codebook: &ClamCodebook,
) -> BF16ThinkingEngine {
    let n = role_codebook.entries.len();
    assert_eq!(n, gate_codebook.entries.len(),
        "gate and role codebooks must have same centroid count");
    assert_eq!(
        gate_codebook.entries[0].stacked.samples_per_dim,
        role_codebook.entries[0].stacked.samples_per_dim,
        "gate and role codebooks must have same SPD"
    );

    let mut cosines = vec![0.0f64; n * n];

    for i in 0..n {
        cosines[i * n + i] = 1.0; // self = max
        let gate_i = gate_codebook.entries[i].stacked.hydrate_f32();
        let role_i = role_codebook.entries[i].stacked.hydrate_f32();
        let activated_i = gate_modulate(&gate_i, &role_i);

        for j in (i + 1)..n {
            let gate_j = gate_codebook.entries[j].stacked.hydrate_f32();
            let role_j = role_codebook.entries[j].stacked.hydrate_f32();
            let activated_j = gate_modulate(&gate_j, &role_j);

            let cos = cosine_f32_to_f64_simd(&activated_i, &activated_j);
            cosines[i * n + j] = cos;
            cosines[j * n + i] = cos;
        }
    }

    BF16ThinkingEngine::from_f64_cosines(&cosines, n)
}

/// Complete set of per-role BF16 distance tables for one layer.
pub struct LayerTables {
    /// Attention Q: raw cosine (extern, world asks).
    pub attn_q: BF16ThinkingEngine,
    /// Attention K: raw cosine (knowledge index).
    pub attn_k: BF16ThinkingEngine,
    /// Attention V: raw cosine (content).
    pub attn_v: BF16ThinkingEngine,
    /// FFN Gate: raw cosine (gate topology, NARS trust reference).
    pub ffn_gate: BF16ThinkingEngine,
    /// FFN Up: silu(gate) × Up (gate-modulated, the 33% correction).
    pub ffn_up: BF16ThinkingEngine,
    /// FFN Down: raw cosine (funnel, receives gated result).
    pub ffn_down: BF16ThinkingEngine,
}

impl LayerTables {
    /// Build from per-role ClamCodebooks.
    ///
    /// All codebooks must have the same centroid count.
    /// Gate codebook is used for Up modulation.
    pub fn build(
        q_codebook: &ClamCodebook,
        k_codebook: &ClamCodebook,
        v_codebook: &ClamCodebook,
        gate_codebook: &ClamCodebook,
        up_codebook: &ClamCodebook,
        down_codebook: &ClamCodebook,
    ) -> Self {
        Self {
            attn_q: build_raw_table(q_codebook),
            attn_k: build_raw_table(k_codebook),
            attn_v: build_raw_table(v_codebook),
            ffn_gate: build_raw_table(gate_codebook),
            ffn_up: build_gate_modulated_table(gate_codebook, up_codebook),
            ffn_down: build_raw_table(down_codebook),
        }
    }

    /// Measure gate modulation effect: how much does the Up table differ
    /// from raw Up cosine?
    pub fn gate_modulation_effect(
        gate_codebook: &ClamCodebook,
        up_codebook: &ClamCodebook,
    ) -> GateModulationStats {
        let raw = build_raw_table(up_codebook);
        let modulated = build_gate_modulated_table(gate_codebook, up_codebook);

        let n = raw.size;
        let mut cells_changed = 0usize;
        let mut total_delta = 0.0f64;
        let mut max_delta = 0.0f64;

        let raw_table = raw.distance_table_ref();
        let mod_table = modulated.distance_table_ref();

        for i in 0..n * n {
            let r = bf16_to_f32(raw_table[i]);
            let m = bf16_to_f32(mod_table[i]);
            let delta = (r - m).abs() as f64;
            if delta > 0.001 {
                cells_changed += 1;
                total_delta += delta;
                if delta > max_delta { max_delta = delta; }
            }
        }

        let total = n * n;
        GateModulationStats {
            cells_changed,
            cells_total: total,
            change_pct: cells_changed as f32 / total as f32 * 100.0,
            mean_delta: if cells_changed > 0 { total_delta / cells_changed as f64 } else { 0.0 },
            max_delta,
        }
    }
}

/// Statistics about the gate modulation effect on a distance table.
#[derive(Clone, Debug)]
pub struct GateModulationStats {
    pub cells_changed: usize,
    pub cells_total: usize,
    pub change_pct: f32,
    pub mean_delta: f64,
    pub max_delta: f64,
}

impl std::fmt::Display for GateModulationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gate modulation: {}/{} cells changed ({:.1}%), mean Δ={:.4}, max Δ={:.4}",
            self.cells_changed, self.cells_total, self.change_pct,
            self.mean_delta, self.max_delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_codebook(n: usize, spd: usize, seed: u64) -> ClamCodebook {
        let entries: Vec<bgz_tensor::stacked_n::CodebookEntry> = (0..n).map(|i| {
            let data: Vec<u16> = (0..17 * spd).map(|d| {
                let v = ((i as f64 * 0.1 + d as f64 * 0.03 + seed as f64 * 0.01)
                    .sin() * 0.5) as f32;
                f32_to_bf16(v)
            }).collect();
            bgz_tensor::stacked_n::CodebookEntry {
                stacked: StackedN { samples_per_dim: spd, data },
                population: 100,
                radius: 0.5,
            }
        }).collect();

        ClamCodebook {
            entries,
            assignments: vec![0u16; 1000],
            samples_per_dim: spd,
            metric: "cosine",
        }
    }

    #[test]
    fn silu_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7311).abs() < 0.01);
        assert!(silu(-5.0).abs() < 0.04); // near zero
        assert!((silu(5.0) - 4.97).abs() < 0.04); // near identity
    }

    #[test]
    fn gate_modulate_masks_zero() {
        let gate = vec![0.0f32; 10]; // all zero = all masked
        let role = vec![1.0f32; 10];
        let result = gate_modulate(&gate, &role);
        // silu(0) = 0 → all masked
        for &v in &result { assert!(v.abs() < 1e-6); }
    }

    #[test]
    fn gate_modulate_passes_positive() {
        let gate = vec![5.0f32; 10]; // large positive = pass through
        let role = vec![1.0f32; 10];
        let result = gate_modulate(&gate, &role);
        // silu(5) ≈ 4.97 → nearly passes through
        for &v in &result { assert!(v > 4.0); }
    }

    #[test]
    fn raw_table_from_codebook() {
        let codebook = make_test_codebook(16, 4, 1);
        let engine = build_raw_table(&codebook);
        assert_eq!(engine.size, 16);

        // Diagonal should be ~1.0
        let table = engine.distance_table_ref();
        for i in 0..16 {
            let self_cos = bf16_to_f32(table[i * 16 + i]);
            assert!((self_cos - 1.0).abs() < 0.01, "self-cos={}", self_cos);
        }
    }

    #[test]
    fn gate_modulated_differs_from_raw() {
        let gate_cb = make_test_codebook(16, 4, 1);
        let up_cb = make_test_codebook(16, 4, 2); // different seed

        let raw = build_raw_table(&up_cb);
        let modulated = build_gate_modulated_table(&gate_cb, &up_cb);

        // Tables should differ (gate changes the cosines)
        let raw_t = raw.distance_table_ref();
        let mod_t = modulated.distance_table_ref();

        let mut diffs = 0;
        for i in 0..16 * 16 {
            if raw_t[i] != mod_t[i] { diffs += 1; }
        }
        assert!(diffs > 0, "gate modulation should change at least some entries");
        eprintln!("Gate modulation changed {}/256 entries ({:.1}%)",
            diffs, diffs as f32 / 256.0 * 100.0);
    }

    #[test]
    fn gate_modulation_stats() {
        let gate_cb = make_test_codebook(16, 4, 1);
        let up_cb = make_test_codebook(16, 4, 2);

        let stats = LayerTables::gate_modulation_effect(&gate_cb, &up_cb);
        eprintln!("{}", stats);
        assert!(stats.cells_changed > 0);
        assert!(stats.change_pct > 0.0);
    }

    #[test]
    fn layer_tables_build() {
        let q = make_test_codebook(8, 4, 1);
        let k = make_test_codebook(8, 4, 2);
        let v = make_test_codebook(8, 4, 3);
        let gate = make_test_codebook(8, 4, 4);
        let up = make_test_codebook(8, 4, 5);
        let down = make_test_codebook(8, 4, 6);

        let tables = LayerTables::build(&q, &k, &v, &gate, &up, &down);
        assert_eq!(tables.attn_q.size, 8);
        assert_eq!(tables.ffn_up.size, 8);
        assert_eq!(tables.ffn_down.size, 8);
    }
}
