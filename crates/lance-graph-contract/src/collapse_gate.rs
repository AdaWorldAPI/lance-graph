//! CollapseGate write protocol — MergeMode + GateDecision.
//!
//! CollapseGate enum (Flow/Block/Hold) lives in ndarray::hpc::bnn_cross_plane.
//! This module adds the write-back protocol types consumed by the 7-layer stack.
//!
//! Layer 3: CollapseGate decides SHOULD this delta land?
//! MergeMode decides HOW overlapping writes merge.
//! GateDecision = gate + merge mode (owned microcopy, 2 bytes).

/// Default α-saturation threshold for [`MergeMode::AlphaFrontToBack`].
/// Once accumulated α exceeds this, the front-to-back loop terminates
/// early — the Kerbl 2023 EWA-splatting "early ray termination" rule
/// (Sec. 4.1 of the 3D Gaussian-Splatting paper) ported from pixel
/// rasterization to BindSpace columns.
pub const ALPHA_SATURATION_THRESHOLD: f32 = 0.99;

/// How overlapping writers merge their deltas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MergeMode {
    /// XOR commit: `target ^= delta`. Self-inverse, reversible.
    /// For single-target updates where order doesn't matter.
    Xor = 0,
    /// Bundle: majority vote across all pending deltas.
    /// For multi-writer consensus (e.g., multiple agents posting to blackboard).
    Bundle = 1,
    /// Superposition: keep ALL deltas without resolution.
    /// For ambiguous cases where we want to preserve all variants.
    Superposition = 2,
    /// Pillar-7 front-to-back α-compositing (Kerbl 2023 EWA splatting,
    /// adapted from pixels to BindSpace columns).
    ///
    /// Hits — assumed sorted by confidence DESC — are composited
    /// front-to-back:
    ///
    /// ```text
    ///   color_acc += color_i * α_i * (1 - α_acc)
    ///   α_acc     += α_i * (1 - α_acc)
    ///   if α_acc > ALPHA_SATURATION_THRESHOLD { break }   // early ray termination
    /// ```
    ///
    /// Top-K hit aggregation is replaced by this volumetric merge:
    /// the strongest hit dominates, weaker hits fill in transparency,
    /// and saturation lets us skip the long tail without losing the
    /// dominant signal. Concentration-of-measure in d=10000 keeps the
    /// remainder mass bounded (per `I-NOISE-FLOOR-JIRAK`).
    ///
    /// The saturation threshold defaults to [`ALPHA_SATURATION_THRESHOLD`]
    /// (0.99). Per-dispatch overrides ride on
    /// `ShaderDispatch::alpha_saturation_override`, keeping this enum
    /// `Copy + Eq + Hash + #[repr(u8)]` so existing call sites don't
    /// regress.
    AlphaFrontToBack = 3,
}

/// A gate decision: what the CollapseGate decided + how to merge.
/// Copy type, 2 bytes. The microcopy returned by gate evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateDecision {
    /// Flow = apply delta. Block = reject. Hold = queue for next cycle.
    pub gate: u8, // 0=Flow, 1=Block, 2=Hold (matches ndarray CollapseGate ordinals)
    /// How to merge if Flow.
    pub merge: MergeMode,
}

impl GateDecision {
    pub const FLOW_XOR: Self = Self {
        gate: 0,
        merge: MergeMode::Xor,
    };
    pub const FLOW_BUNDLE: Self = Self {
        gate: 0,
        merge: MergeMode::Bundle,
    };
    pub const FLOW_SUPER: Self = Self {
        gate: 0,
        merge: MergeMode::Superposition,
    };
    pub const BLOCK: Self = Self {
        gate: 1,
        merge: MergeMode::Xor,
    };
    pub const HOLD: Self = Self {
        gate: 2,
        merge: MergeMode::Xor,
    };

    #[inline]
    pub fn is_flow(&self) -> bool {
        self.gate == 0
    }
    #[inline]
    pub fn is_block(&self) -> bool {
        self.gate == 1
    }
    #[inline]
    pub fn is_hold(&self) -> bool {
        self.gate == 2
    }
}
