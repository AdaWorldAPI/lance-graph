//! Heterogeneous NeuronPrint: spatial + thinking style + spectral transform.
//!
//! Not every tensor role is the same data type:
//! - Q, K, V → spatial directions in attention space (StackedBF16 + codebook)
//! - Gate → thinking style control signal (u64 fingerprint, Hamming-searchable)
//! - Up×Down → paired bottleneck transform (spectral metadata)
//!
//! This module implements the complete heterogeneous encoding.

use crate::stacked_n::{StackedN, bf16_to_f32, f32_to_bf16};

// ═══════════════════════════════════════════════════════════════════════════
// SpatialRole: Q, K, V encoded as StackedBF16 + codebook index
// ═══════════════════════════════════════════════════════════════════════════

/// Spatial role: a direction in attention space.
/// Encodes Q, K, or V weight rows.
#[derive(Clone, Debug)]
pub struct SpatialRole {
    /// Full stacked encoding (1 KB at SPD=32).
    pub stacked: StackedN,
    /// Codebook index for fast search (12-bit = 4096 entries).
    pub codebook_index: u16,
}

impl SpatialRole {
    pub fn byte_size(&self) -> usize {
        self.stacked.byte_size() + 2
    }

    /// Cosine similarity between two spatial roles.
    pub fn cosine(&self, other: &SpatialRole) -> f64 {
        self.stacked.cosine(&other.stacked)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ThinkingStyleFingerprint: Gate as cognitive control signal
// ═══════════════════════════════════════════════════════════════════════════

/// 64-bit cognitive fingerprint derived from Gate activation patterns
/// and runtime observation markers.
///
/// NOT a compressed vector — a CLASSIFICATION of thinking dynamics.
/// Hamming distance IS the correct metric (uniform bits, POPCOUNT).
///
/// The 64 bits encode observed cognitive operations:
///
/// ```text
/// Bits  0-7:   activation pattern  (sign balance, sparsity, magnitude class)
/// Bits  8-15:  NARS-derived        (exploit/explore, confidence tier, expectation)
/// Bits 16-23:  flow/homeostasis    (flow state, allostatic load, DK position)
/// Bits 24-31:  transform character (compression ratio class, rank class, rotation type)
/// Bits 32-39:  variance profile    (CV class, entropy tier, stability)
/// Bits 40-47:  cross-role relation (Q/K alignment class, V coherence, Gate dominance)
/// Bits 48-55:  temporal dynamics   (trend direction, oscillation, convergence)
/// Bits 56-63:  reserved / model-specific
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ThinkingStyleFingerprint {
    pub bits: u64,
}

impl ThinkingStyleFingerprint {
    /// Hamming distance (POPCOUNT of XOR). O(1).
    #[inline]
    pub fn hamming(&self, other: &ThinkingStyleFingerprint) -> u32 {
        (self.bits ^ other.bits).count_ones()
    }

    /// Hamming similarity = 1.0 - hamming/64.
    #[inline]
    pub fn similarity(&self, other: &ThinkingStyleFingerprint) -> f64 {
        1.0 - self.hamming(other) as f64 / 64.0
    }

    /// Empty fingerprint.
    pub fn zero() -> Self {
        ThinkingStyleFingerprint { bits: 0 }
    }

    /// Observe thinking style from Gate weight statistics.
    ///
    /// This extracts cognitive markers from the Gate tensor's statistical
    /// profile — NOT from runtime execution, but from the weight structure
    /// that DETERMINES runtime behavior.
    ///
    /// Gate weights encode HOW the neuron decides to fire. The statistical
    /// signature reveals its decision-making character.
    pub fn from_gate_weights(gate_row: &[f32]) -> Self {
        let n = gate_row.len();
        if n == 0 { return Self::zero(); }

        let mut bits = 0u64;

        // ── Bits 0-7: activation pattern ────────────────────────────────

        // Sign balance: fraction of positive values
        let n_positive = gate_row.iter().filter(|&&v| v > 0.0).count();
        let sign_balance = n_positive as f64 / n as f64;
        // 0=all negative, 128=balanced, 255=all positive → 3 bits
        let sign_class = ((sign_balance * 7.0).round() as u64).min(7);
        bits |= sign_class;

        // Sparsity: fraction of near-zero values (|v| < 0.01 * max)
        let max_abs = gate_row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let threshold = max_abs * 0.01;
        let n_sparse = gate_row.iter().filter(|&&v| v.abs() < threshold).count();
        let sparsity = n_sparse as f64 / n as f64;
        let sparsity_class = ((sparsity * 7.0).round() as u64).min(7);
        bits |= sparsity_class << 3;

        // Magnitude class: log2 of max absolute value → 2 bits
        let mag_class = if max_abs < 0.01 { 0u64 }
            else if max_abs < 0.1 { 1 }
            else if max_abs < 1.0 { 2 }
            else { 3 };
        bits |= mag_class << 6;

        // ── Bits 8-15: distribution shape ───────────────────────────────

        // Mean / std ratio (coefficient of variation)
        let mean = gate_row.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let variance = gate_row.iter().map(|&v| {
            let d = v as f64 - mean;
            d * d
        }).sum::<f64>() / n as f64;
        let std = variance.sqrt();
        let cv = if mean.abs() > 1e-10 { (std / mean.abs()).min(7.0) } else { 3.5 };
        let cv_class = ((cv * 2.0).round() as u64).min(7);
        bits |= cv_class << 8;

        // Kurtosis indicator (heavy-tailed vs light-tailed)
        let m4 = gate_row.iter().map(|&v| {
            let d = v as f64 - mean;
            d * d * d * d
        }).sum::<f64>() / n as f64;
        let kurtosis = if variance > 1e-12 { m4 / (variance * variance) } else { 3.0 };
        let kurt_class = if kurtosis < 2.0 { 0u64 } // platykurtic (light tails)
            else if kurtosis < 3.5 { 1 }             // mesokurtic (normal)
            else if kurtosis < 6.0 { 2 }              // leptokurtic (heavy tails)
            else { 3 };                                // extreme
        bits |= kurt_class << 11;

        // Skewness direction
        let m3 = gate_row.iter().map(|&v| {
            let d = v as f64 - mean;
            d * d * d
        }).sum::<f64>() / n as f64;
        let skew = if std > 1e-12 { m3 / (std * std * std) } else { 0.0 };
        let skew_class = if skew < -0.5 { 0u64 }      // left-skewed
            else if skew < 0.5 { 1 }                   // symmetric
            else { 2 };                                 // right-skewed
        bits |= skew_class << 13;

        // ── Bits 16-23: spatial structure within Gate row ───────────────

        // First-half vs second-half energy ratio
        let half = n / 2;
        let energy_first: f64 = gate_row[..half].iter().map(|v| (v.abs() as f64).powi(2)).sum();
        let energy_second: f64 = gate_row[half..].iter().map(|v| (v.abs() as f64).powi(2)).sum();
        let energy_ratio = if energy_first + energy_second > 1e-12 {
            energy_first / (energy_first + energy_second)
        } else { 0.5 };
        let energy_class = ((energy_ratio * 7.0).round() as u64).min(7);
        bits |= energy_class << 16;

        // Alternating sign pattern (indicator of oscillatory behavior)
        let mut sign_changes = 0u32;
        for i in 1..n {
            if (gate_row[i] > 0.0) != (gate_row[i - 1] > 0.0) {
                sign_changes += 1;
            }
        }
        let oscillation = sign_changes as f64 / (n - 1) as f64;
        let osc_class = ((oscillation * 7.0).round() as u64).min(7);
        bits |= osc_class << 19;

        // Dominant frequency (simple: count zero-crossings)
        let freq_class = ((sign_changes as f64 / n as f64 * 15.0).round() as u64).min(3);
        bits |= freq_class << 22;

        // ── Bits 24-31: extremal structure ──────────────────────────────

        // Number of values > 2σ (outlier count)
        let two_sigma = mean + 2.0 * std;
        let n_outliers = gate_row.iter().filter(|&&v| (v as f64).abs() > two_sigma.abs()).count();
        let outlier_frac = n_outliers as f64 / n as f64;
        let outlier_class = ((outlier_frac * 15.0).round() as u64).min(7);
        bits |= outlier_class << 24;

        // Max positive position (where in the row is the strongest activation)
        let max_pos = gate_row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let pos_class = ((max_pos as f64 / n as f64) * 7.0).round() as u64;
        bits |= pos_class.min(7) << 27;

        // Effective rank estimate (how many dims carry signal)
        // Simple: count dims with |v| > 10% of max
        let rank_threshold = max_abs * 0.1;
        let effective_rank = gate_row.iter().filter(|&&v| v.abs() > rank_threshold).count();
        let rank_class = if effective_rank < n / 10 { 0u64 }
            else if effective_rank < n / 4 { 1 }
            else if effective_rank < n / 2 { 2 }
            else { 3 };
        bits |= rank_class << 30;

        // ── Bits 32-63: hash of fine structure (for collision avoidance) ─

        // XOR-fold the gate values into remaining bits
        let mut hash = 0u32;
        for chunk in gate_row.chunks(4) {
            for &v in chunk {
                hash ^= v.to_bits();
                hash = hash.rotate_left(7);
            }
        }
        bits |= (hash as u64) << 32;

        ThinkingStyleFingerprint { bits }
    }

    /// Observe from NARS truth values (runtime markers).
    pub fn with_nars(mut self, frequency: f32, confidence: f32) -> Self {
        // Encode NARS into bits 8-10 (overwrite distribution shape)
        let freq_class = ((frequency * 3.0).round() as u64).min(3);
        let conf_class = ((confidence * 3.0).round() as u64).min(3);
        let exploit = if frequency > 0.5 && confidence > 0.5 { 1u64 } else { 0 };
        self.bits = (self.bits & !0x700) | (freq_class << 8) | (conf_class << 9) | (exploit << 11);
        self
    }

    /// Extract human-readable profile summary.
    pub fn profile(&self) -> String {
        let sign_bal = (self.bits & 0x07) as f32 / 7.0;
        let sparsity = ((self.bits >> 3) & 0x07) as f32 / 7.0;
        let mag = (self.bits >> 6) & 0x03;
        let cv = ((self.bits >> 8) & 0x07) as f32 / 2.0;
        let kurt = (self.bits >> 11) & 0x03;
        let osc = ((self.bits >> 19) & 0x07) as f32 / 7.0;

        format!(
            "sign_bal={:.2} sparse={:.2} mag={} cv={:.1} kurt={} osc={:.2}",
            sign_bal, sparsity,
            ["tiny", "small", "medium", "large"][mag as usize],
            cv,
            ["platy", "normal", "lepto", "extreme"][kurt as usize],
            osc,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TransformSpectrum: Up×Down as paired bottleneck transform
// ═══════════════════════════════════════════════════════════════════════════

/// Spectral metadata for the Up×Down bottleneck transform.
///
/// Up and Down together define a learned bottleneck. The information
/// is in what the PAIR does — not in each vector separately.
///
/// PolarQuant is the natural encoding here: Up×Down IS a rotation + scaling.
#[derive(Clone, Copy, Debug)]
pub struct TransformSpectrum {
    /// Effective rank: how many dims carry signal (0-255).
    pub effective_rank: u8,
    /// Compression ratio: Down_dim / Up_dim as fixed-point (0.0-4.0).
    pub compression_ratio: u16,
    /// Energy concentration: fraction of energy in top 10% of dims.
    pub energy_concentration: u8,
    /// Correlation between Up and Down rows (sign agreement).
    pub up_down_correlation: i8,
}

impl TransformSpectrum {
    pub const BYTE_SIZE: usize = 5;

    /// Compute spectral metadata from Up and Down weight rows.
    pub fn from_up_down(up_row: &[f32], down_row: &[f32]) -> Self {
        let up_n = up_row.len();
        let down_n = down_row.len();

        // Effective rank of Up (dims with |v| > 10% of max)
        let up_max = up_row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let up_thresh = up_max * 0.1;
        let up_rank = up_row.iter().filter(|&&v| v.abs() > up_thresh).count();
        let effective_rank = (up_rank.min(255)) as u8;

        // Compression ratio
        let ratio = if up_n > 0 { down_n as f64 / up_n as f64 } else { 1.0 };
        let compression_ratio = ((ratio * 1000.0).round() as u16).min(4000);

        // Energy concentration: sort |values|, fraction in top 10%
        let mut up_sorted: Vec<f32> = up_row.iter().map(|v| v.abs()).collect();
        up_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top10_count = (up_n / 10).max(1);
        let top10_energy: f64 = up_sorted[..top10_count].iter().map(|&v| (v as f64).powi(2)).sum();
        let total_energy: f64 = up_sorted.iter().map(|&v| (v as f64).powi(2)).sum();
        let concentration = if total_energy > 1e-12 {
            (top10_energy / total_energy * 255.0).round() as u8
        } else { 0 };

        // Up-Down correlation: sign agreement on shared dims
        let shared = up_n.min(down_n);
        let agreements = (0..shared)
            .filter(|&i| (up_row[i] > 0.0) == (down_row[i] > 0.0))
            .count();
        let corr = if shared > 0 {
            ((agreements as f64 / shared as f64 - 0.5) * 254.0).round() as i8
        } else { 0 };

        TransformSpectrum {
            effective_rank,
            compression_ratio,
            energy_concentration: concentration,
            up_down_correlation: corr,
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "rank={} ratio={:.2} concentration={:.0}% corr={:.2}",
            self.effective_rank,
            self.compression_ratio as f64 / 1000.0,
            self.energy_concentration as f64 / 255.0 * 100.0,
            self.up_down_correlation as f64 / 127.0,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HeterogeneousNeuronPrint: the complete picture
// ═══════════════════════════════════════════════════════════════════════════

/// Complete heterogeneous neuron representation.
///
/// NOT 6 × same-type. Three different data types for three different functions:
/// - Spatial (Q, K, V): directions in attention space
/// - Style (Gate): cognitive control fingerprint
/// - Transform (Up×Down): bottleneck spectral metadata
///
/// ```text
/// Full resolution:  3 × StackedN(SPD=32) + u64 + 5 bytes = ~3 KB + 13 bytes
/// Compact:          3 × u16 codebook + u64 + 5 bytes = 19 bytes
/// The 19 bytes carry MORE cognitive information than 6 KB of uniform encoding
/// ```
#[derive(Clone, Debug)]
pub struct HeterogeneousNeuronPrint {
    /// Layer index in the model.
    pub layer: u16,
    /// Feature/row index within the layer.
    pub feature: u32,

    // ── Spatial roles: directions in attention space ─────────────────
    /// Query: what triggers this knowledge.
    pub q: SpatialRole,
    /// Key: what this neuron matches against.
    pub k: SpatialRole,
    /// Value: what this neuron retrieves.
    pub v: SpatialRole,

    // ── Thinking style: cognitive control ────────────────────────────
    /// Gate fingerprint: HOW this neuron decides to fire.
    /// Hamming-searchable via POPCOUNT.
    pub thinking_style: ThinkingStyleFingerprint,

    // ── Transform: bottleneck spectral metadata ─────────────────────
    /// Up×Down spectral characterization.
    pub transform: TransformSpectrum,
}

impl HeterogeneousNeuronPrint {
    /// Full byte size (spatial at full resolution + metadata).
    pub fn full_byte_size(&self) -> usize {
        self.q.byte_size() + self.k.byte_size() + self.v.byte_size()
            + 8 // thinking_style
            + TransformSpectrum::BYTE_SIZE
            + 6 // layer + feature
    }

    /// Compact byte size (codebook indices + metadata).
    pub fn compact_byte_size() -> usize {
        3 * 2  // Q, K, V codebook indices
        + 8    // thinking_style
        + TransformSpectrum::BYTE_SIZE
        + 6    // layer + feature
    }

    /// Spatial similarity (Q×K alignment proxy).
    pub fn attention_affinity(&self, other: &HeterogeneousNeuronPrint) -> f64 {
        self.q.cosine(&other.k)
    }

    /// Thinking style match (Hamming distance on Gate fingerprints).
    pub fn style_match(&self, other: &HeterogeneousNeuronPrint) -> f64 {
        self.thinking_style.similarity(&other.thinking_style)
    }

    /// Combined relevance score: spatial + style.
    ///
    /// Alpha controls the blend: 0.0 = pure spatial, 1.0 = pure style.
    pub fn relevance(&self, query: &HeterogeneousNeuronPrint, alpha: f64) -> f64 {
        let spatial = self.attention_affinity(query);
        let style = self.style_match(query);
        (1.0 - alpha) * spatial + alpha * style
    }
}

/// Build a HeterogeneousNeuronPrint from raw weight rows.
///
/// Takes the 6 role weight rows for one neuron (same row index across tensors).
/// Gate becomes ThinkingStyleFingerprint.
/// Up×Down becomes TransformSpectrum.
/// Q, K, V become SpatialRole (StackedN + placeholder codebook index).
pub fn build_neuron(
    layer: u16,
    feature: u32,
    q_row: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    gate_row: &[f32],
    up_row: &[f32],
    down_row: &[f32],
    spd: usize,
) -> HeterogeneousNeuronPrint {
    HeterogeneousNeuronPrint {
        layer,
        feature,
        q: SpatialRole {
            stacked: StackedN::from_f32(q_row, spd),
            codebook_index: 0, // assigned later by codebook
        },
        k: SpatialRole {
            stacked: StackedN::from_f32(k_row, spd),
            codebook_index: 0,
        },
        v: SpatialRole {
            stacked: StackedN::from_f32(v_row, spd),
            codebook_index: 0,
        },
        thinking_style: ThinkingStyleFingerprint::from_gate_weights(gate_row),
        transform: TransformSpectrum::from_up_down(up_row, down_row),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gate(pattern: &str) -> Vec<f32> {
        match pattern {
            "sparse" => {
                let mut v = vec![0.0f32; 1024];
                for i in (0..1024).step_by(100) { v[i] = 2.0; }
                v
            }
            "dense" => (0..1024).map(|i| (i as f32 * 0.01).sin()).collect(),
            "oscillating" => (0..1024).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(),
            "one_sided" => vec![1.0; 1024],
            _ => vec![0.0; 1024],
        }
    }

    #[test]
    fn thinking_style_distinct_patterns() {
        let sparse = ThinkingStyleFingerprint::from_gate_weights(&make_gate("sparse"));
        let dense = ThinkingStyleFingerprint::from_gate_weights(&make_gate("dense"));
        let oscillating = ThinkingStyleFingerprint::from_gate_weights(&make_gate("oscillating"));
        let one_sided = ThinkingStyleFingerprint::from_gate_weights(&make_gate("one_sided"));

        // Different patterns should have different fingerprints
        assert_ne!(sparse.bits, dense.bits, "sparse and dense should differ");
        assert_ne!(sparse.bits, oscillating.bits, "sparse and oscillating should differ");
        assert_ne!(dense.bits, one_sided.bits, "dense and one-sided should differ");

        // Hamming distance should reflect structural difference
        let d_sparse_dense = sparse.hamming(&dense);
        let d_sparse_osc = sparse.hamming(&oscillating);
        eprintln!("sparse:      {}", sparse.profile());
        eprintln!("dense:       {}", dense.profile());
        eprintln!("oscillating: {}", oscillating.profile());
        eprintln!("one_sided:   {}", one_sided.profile());
        eprintln!("hamming(sparse, dense) = {}", d_sparse_dense);
        eprintln!("hamming(sparse, osc)   = {}", d_sparse_osc);
    }

    #[test]
    fn thinking_style_self_zero() {
        let gate = make_gate("dense");
        let a = ThinkingStyleFingerprint::from_gate_weights(&gate);
        assert_eq!(a.hamming(&a), 0);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn transform_spectrum_basic() {
        let up = vec![1.0f32; 4096];
        let down = vec![0.5f32; 1024];
        let spec = TransformSpectrum::from_up_down(&up, &down);
        assert!(spec.compression_ratio > 0);
        assert!(spec.effective_rank > 0);
        eprintln!("spectrum: {}", spec.summary());
    }

    #[test]
    fn transform_spectrum_sparse_vs_dense() {
        let up_dense = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>();
        let up_sparse = {
            let mut v = vec![0.0f32; 4096];
            for i in (0..4096).step_by(100) { v[i] = 5.0; }
            v
        };
        let down = vec![0.1f32; 1024];

        let spec_dense = TransformSpectrum::from_up_down(&up_dense, &down);
        let spec_sparse = TransformSpectrum::from_up_down(&up_sparse, &down);

        assert!(spec_sparse.effective_rank < spec_dense.effective_rank,
            "sparse should have lower rank: {} vs {}",
            spec_sparse.effective_rank, spec_dense.effective_rank);
        eprintln!("dense:  {}", spec_dense.summary());
        eprintln!("sparse: {}", spec_sparse.summary());
    }

    #[test]
    fn heterogeneous_neuron_sizes() {
        assert_eq!(HeterogeneousNeuronPrint::compact_byte_size(), 25);
    }

    #[test]
    fn build_neuron_roundtrip() {
        let q = (0..512).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>();
        let k = (0..512).map(|i| (i as f32 * 0.02).cos()).collect::<Vec<_>>();
        let v = vec![0.1f32; 512];
        let gate = make_gate("dense");
        let up = vec![0.5f32; 2048];
        let down = vec![0.3f32; 512];

        let neuron = build_neuron(5, 42, &q, &k, &v, &gate, &up, &down, 32);
        assert_eq!(neuron.layer, 5);
        assert_eq!(neuron.feature, 42);
        assert!(neuron.thinking_style.bits != 0);
        assert!(neuron.transform.effective_rank > 0);
        eprintln!("Full size: {} bytes", neuron.full_byte_size());
        eprintln!("Compact size: {} bytes", HeterogeneousNeuronPrint::compact_byte_size());
        eprintln!("Style: {}", neuron.thinking_style.profile());
        eprintln!("Transform: {}", neuron.transform.summary());
    }
}
