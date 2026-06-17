//! Qualia Contract — 17D observation vector.
//!
//! A qualia vector is the output of a cognitive cycle: seventeen
//! `f32` observables computed from the convergence pattern. Each
//! dimension captures a distinct facet of the cycle's behaviour.
//!
//! The contract exposes the axis layout, labels, and conversion
//! utilities. Downstream consumers that need richer labelling
//! (e.g. a companion agent's felt-state overlay) supply their own
//! translation table; the contract stays neutral.
//!
//! # Relationship to Proprioception
//!
//! - [`QualiaVector`] (17D) is the direct observation — what the
//!   cycle *computed*.
//! - [`crate::proprioception::StateClassifier`] takes an 11D state
//!   vector — what the agent *recognises itself as*.
//! - [`qualia_to_state`] is the canonical projection from 17D → 11D.

// ═══════════════════════════════════════════════════════════════════════════
// 17D axis layout
// ═══════════════════════════════════════════════════════════════════════════

/// Dimensionality of the qualia vector.
pub const QUALIA_DIMS: usize = 17;

/// Canonical axis labels, index-aligned with [`QualiaVector`].
pub const AXIS_LABELS: [&str; QUALIA_DIMS] = [
    "arousal",      // 0
    "valence",      // 1
    "tension",      // 2
    "warmth",       // 3
    "clarity",      // 4
    "boundary",     // 5
    "depth",        // 6
    "velocity",     // 7
    "entropy",      // 8
    "coherence",    // 9
    "intimacy",     // 10
    "presence",     // 11
    "assertion",    // 12
    "receptivity",  // 13
    "groundedness", // 14
    "expansion",    // 15
    "integration",  // 16
];

/// A single 17D observation.
///
/// All values are expected to be in `[0.0, 1.0]`, though implementors
/// may temporarily carry out-of-range values during arithmetic.
pub type QualiaVector = [f32; QUALIA_DIMS];

/// Axis index by name. Returns `None` for unknown labels.
pub fn axis_index(label: &str) -> Option<usize> {
    AXIS_LABELS.iter().position(|&l| l == label)
}

/// Axis label by index. Returns `None` if out of range.
pub fn axis_label(index: usize) -> Option<&'static str> {
    AXIS_LABELS.get(index).copied()
}

// ═══════════════════════════════════════════════════════════════════════════
// Projection: 17D qualia → 11D proprioceptive state
// ═══════════════════════════════════════════════════════════════════════════

use crate::proprioception::{CORE_AXES, DRIVE_AXES, STATE_DIMS};

/// Project a 17D qualia vector into the 11D proprioceptive state space.
///
/// Qualia → State mapping (core 7):
///   `warmth  ← qualia[3]`
///   `clarity ← qualia[4]`
///   `depth   ← qualia[6]`
///   `safety  ← qualia[14]` (groundedness)
///   `vitality← 0.6·qualia[0] + 0.4·qualia[13]` (arousal + receptivity)
///   `insight ← 0.5·qualia[11] + 0.5·qualia[16]` (presence + integration)
///   `contact ← qualia[10]`
///
/// Drive 4:
///   `tension   ← 0.5·(1-qualia[1]) + 0.5·qualia[2]` (inv valence + tension)
///   `novelty   ← qualia[15]` (expansion)
///   `wonder    ← sqrt(qualia[9]·qualia[15])` (coherence × expansion)
///   `attunement ← qualia[10]·(1-qualia[2])` (intimacy × relaxation)
pub fn qualia_to_state(q: &QualiaVector) -> [f32; STATE_DIMS] {
    let arousal = q[0];
    let valence = q[1];
    let tension = q[2];
    let warmth = q[3];
    let clarity = q[4];
    let depth = q[6];
    let coherence = q[9];
    let intimacy = q[10];
    let presence = q[11];
    let receptivity = q[13];
    let groundedness = q[14];
    let expansion = q[15];
    let integration = q[16];

    let vitality = (0.6 * arousal + 0.4 * receptivity).clamp(0.0, 1.0);
    let insight = (0.5 * presence + 0.5 * integration).clamp(0.0, 1.0);

    let tension_axis = (0.5 * (1.0 - valence) + 0.5 * tension).clamp(0.0, 1.0);
    let wonder_axis = (coherence * expansion).sqrt().clamp(0.0, 1.0);
    let attune_axis = (intimacy * (1.0 - tension)).clamp(0.0, 1.0);

    let _ = (CORE_AXES, DRIVE_AXES); // structural tie to proprioception contract
    [
        warmth,       // 0 core: warmth
        clarity,      // 1 core: clarity
        depth,        // 2 core: depth
        groundedness, // 3 core: safety
        vitality,     // 4 core: vitality
        insight,      // 5 core: insight
        intimacy,     // 6 core: contact
        tension_axis, // 7 drive: tension
        expansion,    // 8 drive: novelty
        wonder_axis,  // 9 drive: wonder
        attune_axis,  // 10 drive: attunement
    ]
}

/// Zero qualia vector (neutral baseline).
pub const ZERO: QualiaVector = [0.0; QUALIA_DIMS];

/// All-half qualia vector (balanced midpoint).
pub const MIDPOINT: QualiaVector = [0.5; QUALIA_DIMS];

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// i4-16D packed qualia vector (QualiaI4_16D)
// ═══════════════════════════════════════════════════════════════════════════

/// Dimensionality of the i4-16D packed qualia vector.
/// Dim 16 ("integration") from the canonical 17D is dropped to fit 16 lanes.
pub const QUALIA_I4_DIMS: usize = 16;

/// Canonical labels for the i4-16D packed qualia vector, index-aligned with
/// [`QualiaI4_16D`]. Matches the first 16 entries of [`AXIS_LABELS`]; the
/// 17th ("integration") is omitted — recoverable on demand from valence +
/// coherence + cycle-delta if needed.
pub const QUALIA_I4_LABELS: [&str; QUALIA_I4_DIMS] = [
    "arousal",      // 0
    "valence",      // 1
    "tension",      // 2
    "warmth",       // 3
    "clarity",      // 4
    "boundary",     // 5
    "depth",        // 6
    "velocity",     // 7
    "entropy",      // 8
    "coherence",    // 9
    "intimacy",     // 10
    "presence",     // 11
    "assertion",    // 12
    "receptivity",  // 13
    "groundedness", // 14
    "expansion",    // 15
];

/// i4-16D signed packed qualia vector. 8 bytes / 16 dims / range −8..+7 per dim.
/// 9× compression vs `[f32; 18]` historical / `QualiaVector = [f32; 17]` canonical.
/// Per-dim semantics: see `QUALIA_I4_LABELS` (the canonical convergence-observable
/// vocab from `Qualia17D`/`QualiaVector`, with dim 16 "integration" dropped to fit
/// 16 lanes — recoverable on demand from valence + coherence + cycle-delta).
///
/// Lane width: 32 i4 lanes per AVX-512 register; one `QualiaI4_16D` is half a lane group.
/// Magnitude is computed on demand: `coherence × valence → i8` (1 SIMD multiply per row).
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct QualiaI4_16D(pub u64);

impl QualiaI4_16D {
    pub const ZERO: Self = Self(0);

    /// Read the signed i4 value at dim index 0..16.
    /// Sign-extends 4-bit → i8 via arithmetic shift.
    /// Out-of-range index returns 0 (defensive; bound check at call site preferred).
    #[inline]
    pub fn get(self, dim: usize) -> i8 {
        if dim >= QUALIA_I4_DIMS {
            return 0;
        }
        let raw = ((self.0 >> (dim * 4)) & 0xF) as i8;
        (raw << 4) >> 4 // sign-extend 4 → 8 bits
    }

    /// Set the signed i4 value at dim index. Clamps `value` to −8..+7.
    /// Out-of-range index is a no-op (defensive).
    #[inline]
    pub fn set(&mut self, dim: usize, value: i8) {
        if dim >= QUALIA_I4_DIMS {
            return;
        }
        let v = value.clamp(-8, 7);
        let nibble = (v as u8 & 0xF) as u64;
        let mask = 0xFu64 << (dim * 4);
        self.0 = (self.0 & !mask) | (nibble << (dim * 4));
    }

    /// Builder-shape variant.
    #[inline]
    pub fn with(self, dim: usize, value: i8) -> Self {
        let mut copy = self;
        copy.set(dim, value);
        copy
    }

    /// Convert a canonical 17D f32 qualia vector to packed i4-16D.
    /// Each f32 dim in `[0.0, 1.0]` maps to i4 `[0, +7]` via `round(v * 7.0)`.
    /// Each f32 dim in `[-1.0, 0.0)` maps to i4 `[-8, -1]` via `round(v * 8.0)`.
    /// Dim 16 ("integration") from the 17D is DROPPED (recoverable on demand).
    /// Out-of-range f32 values are clamped to [-1.0, 1.0] before quantization.
    #[inline]
    pub fn from_f32_17d(v: &QualiaVector) -> Self {
        let mut out = Self(0);
        for (dim, &f) in v.iter().take(QUALIA_I4_DIMS).enumerate() {
            let clamped = f.clamp(-1.0, 1.0);
            let i = if clamped >= 0.0 {
                (clamped * 7.0).round() as i8
            } else {
                (clamped * 8.0).round() as i8
            };
            out.set(dim, i);
        }
        out
    }

    /// Convert a packed i4-16D back to a 17D f32 qualia vector.
    /// Dim 16 ("integration") is zero-filled (the i4 lacks it; consumer should
    /// recompute from valence + coherence if needed).
    /// Reverse of `from_f32_17d`: positive i4 in [0, +7] → f32 in [0.0, 1.0];
    /// negative i4 in [-8, -1] → f32 in [-1.0, 0.0).
    #[inline]
    pub fn to_f32_17d(self) -> QualiaVector {
        let mut out = [0.0f32; QUALIA_DIMS];
        for (dim, slot) in out.iter_mut().enumerate().take(QUALIA_I4_DIMS) {
            let i = self.get(dim);
            *slot = if i >= 0 {
                i as f32 / 7.0
            } else {
                i as f32 / 8.0
            };
        }
        // dim 16 stays 0.0 (integration dropped)
        out
    }

    /// On-demand magnitude: `coherence × valence` as i8 (replaces the
    /// historical [f32; 18] dim 13 "Magnitude" derived field).
    /// Per plan L-4: coherence × valence (intensity × polarity).
    /// One i8 saturating multiply (SIMD-friendly).
    #[inline]
    pub fn magnitude(self) -> i8 {
        let coherence = self.get(9); // dim 9 in QUALIA_I4_LABELS
        let valence = self.get(1);
        coherence.saturating_mul(valence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_labels_have_exact_17() {
        assert_eq!(AXIS_LABELS.len(), QUALIA_DIMS);
        assert_eq!(QUALIA_DIMS, 17);
    }

    #[test]
    fn axis_index_roundtrip() {
        for (i, label) in AXIS_LABELS.iter().enumerate() {
            assert_eq!(axis_index(label), Some(i));
            assert_eq!(axis_label(i), Some(*label));
        }
        assert!(axis_index("nonexistent").is_none());
        assert!(axis_label(99).is_none());
    }

    #[test]
    fn projection_outputs_11_dims() {
        let q = MIDPOINT;
        let state = qualia_to_state(&q);
        assert_eq!(state.len(), STATE_DIMS);
        for v in &state {
            assert!(*v >= 0.0 && *v <= 1.0, "projected dim out of range: {}", v);
        }
    }

    #[test]
    fn projection_preserves_direct_axes() {
        let mut q = ZERO;
        q[3] = 0.8; // warmth
        q[4] = 0.9; // clarity
        q[14] = 0.7; // groundedness
        let state = qualia_to_state(&q);
        assert!((state[0] - 0.8).abs() < 1e-6, "warmth should pass through");
        assert!((state[1] - 0.9).abs() < 1e-6, "clarity should pass through");
        assert!((state[3] - 0.7).abs() < 1e-6, "safety ← groundedness");
    }

    #[test]
    fn projection_computes_wonder_from_coherence_and_expansion() {
        let mut q = ZERO;
        q[9] = 0.64; // coherence
        q[15] = 0.64; // expansion
        let state = qualia_to_state(&q);
        // wonder = sqrt(0.64 * 0.64) = 0.64
        assert!((state[9] - 0.64).abs() < 1e-5);
    }
    // ── QualiaI4_16D tests ──────────────────────────────────────────────────

    #[test]
    fn test_qualia_i4_size_8b() {
        use std::mem::size_of;
        assert_eq!(size_of::<QualiaI4_16D>(), 8);
    }

    #[test]
    fn test_qualia_i4_zero_default() {
        let q = QualiaI4_16D::ZERO;
        for dim in 0..QUALIA_I4_DIMS {
            assert_eq!(q.get(dim), 0, "dim {} should be 0", dim);
        }
    }

    #[test]
    fn test_qualia_i4_signed_roundtrip() {
        let test_values: &[i8] = &[-8, -7, -1, 0, 1, 7];
        let test_dims: &[usize] = &[0, 1, 5, 10, 14, 15];
        for &val in test_values {
            for &dim in test_dims {
                let q = QualiaI4_16D::ZERO.with(dim, val);
                assert_eq!(
                    q.get(dim),
                    val,
                    "roundtrip failed for val={} dim={}",
                    val,
                    dim
                );
            }
        }
    }

    #[test]
    fn test_qualia_i4_clamp() {
        let mut q = QualiaI4_16D::ZERO;
        q.set(3, 100);
        assert_eq!(q.get(3), 7, "positive overflow should clamp to +7");
        q.set(3, -100);
        assert_eq!(q.get(3), -8, "negative overflow should clamp to -8");
    }

    #[test]
    fn test_qualia_i4_isolation() {
        // Setting dim 5 to 7 must not affect adjacent dims 4 and 6
        let q = QualiaI4_16D::ZERO.with(5, 7);
        assert_eq!(q.get(4), 0, "dim 4 must remain 0 after setting dim 5");
        assert_eq!(q.get(5), 7, "dim 5 must be 7");
        assert_eq!(q.get(6), 0, "dim 6 must remain 0 after setting dim 5");
    }

    #[test]
    fn test_qualia_i4_from_f32_17d_roundtrip() {
        // Build a representative 17D vector with varied values
        let mut v: QualiaVector = [0.0; QUALIA_DIMS];
        v[0] = 0.8; // arousal
        v[1] = 0.5; // valence
        v[2] = 0.1; // tension
        v[3] = 1.0; // warmth
        v[4] = 0.0; // clarity
        v[5] = 0.3; // boundary
        v[6] = 0.6; // depth
        v[7] = 0.7; // velocity
        v[8] = 0.2; // entropy
        v[9] = 0.9; // coherence
        v[10] = 0.4; // intimacy
        v[11] = 0.55; // presence
        v[12] = 0.15; // assertion
        v[13] = 0.85; // receptivity
        v[14] = 0.45; // groundedness
        v[15] = 0.65; // expansion
        v[16] = 0.75; // integration — should be DROPPED

        let packed = QualiaI4_16D::from_f32_17d(&v);
        let restored = packed.to_f32_17d();

        // dim 16 must be zero in the round-trip output
        assert_eq!(
            restored[16], 0.0,
            "dim 16 (integration) must be zero after round-trip"
        );

        // All other dims must be within quantization error
        // Positive path: max error = 1/7 ≈ 0.143; negative path: max error = 1/8 = 0.125
        let epsilon = 1.0f32 / 7.0 + 1e-5;
        for dim in 0..QUALIA_I4_DIMS {
            let err = (restored[dim] - v[dim]).abs();
            assert!(
                err <= epsilon,
                "dim {} round-trip error {} exceeds epsilon {} (original={}, restored={})",
                dim,
                err,
                epsilon,
                v[dim],
                restored[dim]
            );
        }
    }

    #[test]
    fn test_qualia_i4_label_alignment() {
        // All 16 i4 labels must match the first 16 canonical AXIS_LABELS
        for i in 0..QUALIA_I4_DIMS {
            assert_eq!(
                QUALIA_I4_LABELS[i], AXIS_LABELS[i],
                "label mismatch at index {}: i4='{}' axis='{}'",
                i, QUALIA_I4_LABELS[i], AXIS_LABELS[i]
            );
        }
    }

    #[test]
    fn test_qualia_i4_magnitude() {
        // magnitude = coherence (dim 9) × valence (dim 1), saturating_mul
        // Known values: coherence=3, valence=2 → 6
        let q = QualiaI4_16D::ZERO.with(9, 3).with(1, 2);
        assert_eq!(q.magnitude(), 6);

        // Negative × positive: coherence=-4, valence=2 → -8
        let q2 = QualiaI4_16D::ZERO.with(9, -4).with(1, 2);
        assert_eq!(q2.magnitude(), -8);

        // Saturation: coherence=7, valence=7 → saturating_mul → 49, clamped to i8::MAX=127
        let q3 = QualiaI4_16D::ZERO.with(9, 7).with(1, 7);
        assert_eq!(q3.magnitude(), 7i8.saturating_mul(7));

        // Extremes: coherence=-8, valence=-8 → saturating_mul(-8,-8)=64 (fits in i8)
        let q4 = QualiaI4_16D::ZERO.with(9, -8).with(1, -8);
        assert_eq!(q4.magnitude(), (-8i8).saturating_mul(-8));

        // Zero magnitude when either is zero
        let q5 = QualiaI4_16D::ZERO;
        assert_eq!(q5.magnitude(), 0);
    }

    // ── i4 affective-cutover FIDELITY PROBE (2026-06-17) ──────────────────
    //
    // PURPOSE: measure whether the D-CSV-5b f32→i4 affective cutover preserved
    // rank-order fidelity of the 16 AFFECTIVE qualia dims. This is the probe
    // the with-engine BusDto round-trip surfaced the need for: the round-trip
    // is i4-lossy by design, so the right question is not "is it bit-exact?"
    // (it is not) but "does the quantization preserve the ORDERING of affective
    // intensities?" — which is what every downstream consumer (resonance,
    // magnitude, nearest-archetype classification) actually relies on.
    //
    // We compute Spearman rank-correlation ρ between a representative set of
    // affective f32 vectors (the 16 affective dims in their canonical ranges —
    // [0,1] for the unsigned axes and [-1,1] for the signed axes, NOT identity
    // indices) and their i4 round-trips, and assert ρ >= a calibration floor.
    //
    // THRESHOLD JUSTIFICATION (0.95): i4 gives 15 distinct levels across
    // [-1,1] (8 negative + 7 positive + zero). With the test vectors spread
    // across that range, ties on round-trip are the only source of rank
    // disagreement; ρ >= 0.95 is a defensible "ordering essentially preserved"
    // floor for a 4-bit quantizer over ~15 levels. It is a CALIBRATION floor,
    // NOT a statistical-significance claim: per I-NOISE-FLOOR-JIRAK, qualia
    // dims are weakly dependent by construction (shared codebook, overlapping
    // role slices), so classical IID Berry-Esseen does not apply here; any
    // "N σ above noise floor" framing would require Jirak-2016 weak-dependence
    // rates, which this probe deliberately does NOT assert. We only assert the
    // hand-set 0.95 floor and label it as such.
    fn spearman_rho(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        // Fractional (tie-averaged) ranks.
        let rank = |v: &[f32]| -> Vec<f32> {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
            let mut ranks = vec![0.0f32; n];
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n && v[idx[j]] == v[idx[i]] {
                    j += 1;
                }
                // average rank for the tie group [i, j)
                let avg = ((i + j - 1) as f32) / 2.0 + 1.0; // 1-based
                for &k in &idx[i..j] {
                    ranks[k] = avg;
                }
                i = j;
            }
            ranks
        };
        let ra = rank(a);
        let rb = rank(b);
        let mean = (n as f32 + 1.0) / 2.0;
        let (mut cov, mut va, mut vb) = (0.0f32, 0.0f32, 0.0f32);
        for k in 0..n {
            let da = ra[k] - mean;
            let db = rb[k] - mean;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        if va == 0.0 || vb == 0.0 {
            return 1.0;
        }
        cov / (va.sqrt() * vb.sqrt())
    }

    #[test]
    fn qualia_i4_affective_fidelity_probe_spearman() {
        // A representative spread of affective values across the i4 range.
        // 32 vectors deterministically constructed; each of the 16 dims sweeps
        // [-1, 1] at a different phase so the pooled (orig, restored) pairs
        // cover the whole quantizer.
        let mut originals: Vec<f32> = Vec::new();
        let mut restored: Vec<f32> = Vec::new();
        for s in 0..32u32 {
            let mut v: QualiaVector = [0.0; QUALIA_DIMS];
            for (dim, slot) in v.iter_mut().enumerate().take(QUALIA_I4_DIMS) {
                // phase-shifted ramp in [-1, 1]
                let t = ((s as f32) * 0.31 + (dim as f32) * 0.137).fract();
                *slot = 2.0 * t - 1.0;
            }
            let rt = QualiaI4_16D::from_f32_17d(&v).to_f32_17d();
            for dim in 0..QUALIA_I4_DIMS {
                originals.push(v[dim]);
                restored.push(rt[dim]);
            }
        }
        let rho = spearman_rho(&originals, &restored);
        // Calibration floor (NOT a significance claim — see comment above;
        // I-NOISE-FLOOR-JIRAK forbids classical-IID significance framing here).
        assert!(
            rho >= 0.95,
            "i4 affective cutover preserved rank fidelity below floor: \
             Spearman ρ = {rho:.4} (floor 0.95, n = {} pairs)",
            originals.len()
        );
    }
}
