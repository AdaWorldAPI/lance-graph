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
}
