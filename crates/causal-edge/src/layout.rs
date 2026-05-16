//! CausalEdge64 v2 layout constants — FINAL (Option F, locked 2026-05-16).
//!
//! Cite: cognitive-substrate-convergence-v1.md §6 (authoritative bit layout)
//!       + pr-ce64-mb-2-causaledge64-v2.md §2 (implementation contract).
//! OQ-LAYOUT-1: RESOLVED. G-slot dropped (L-3). Mantissa = 4b signed i4 (L-4).

// ── v1 fields preserved (shifts unchanged from v1) ─────────────────────────
pub const S_SHIFT:      u32 = 0;
pub const P_SHIFT:      u32 = 8;
pub const O_SHIFT:      u32 = 16;
pub const FREQ_SHIFT:   u32 = 24;
pub const CONF_SHIFT:   u32 = 32;
pub const CAUSAL_SHIFT: u32 = 40;
pub const DIR_SHIFT:    u32 = 43;

// ── v1→v2 EXPANDED field ────────────────────────────────────────────────────
/// Inference mantissa: 4-bit signed (−8..+7).
/// sign = chain direction (+ = forward-chain, − = backward-chain).
/// abs(val) = base NARS rule index:
///   0=Identity/neutral, 1=Deduction/Abduction, 2=Induction/Contraposition,
///   3=Exemplification/Analogy-negative, 4=Revision+/Revision-,
///   5=Synthesis/Decomposition, 6=PR-LL-1 Intervention/Counterfactual (L-9),
///   7=Extension/Intension-negative (future).
/// Encodes direction × NARS rule in one field.
/// See pr-ce64-mb-2-causaledge64-v2.md §"Signed Mantissa Rationale".
pub const INFER_SHIFT:  u32 = 46;

/// 4-bit unsigned mask for pack/unpack of the signed i4 mantissa field.
pub const BITS4_MASK:   u64 = 0xF;

/// Mask covering the mantissa field (bits 46-49) in the u64 word.
pub const INFER_MASK:   u64 = BITS4_MASK << INFER_SHIFT;

// ── v1 field SHIFTED ────────────────────────────────────────────────────────
/// Plasticity flags: bits 50-52 (shifted by +1 from v1's bits 49-51 due to
/// mantissa expansion from 3b unsigned to 4b signed i4 per L-4).
pub const PLAST_SHIFT:  u32 = 50;

// ── v1 field DEPRECATED ─────────────────────────────────────────────────────
/// Deprecated: temporal field shift from v1. Bits 52-63 reclaimed in v2.
/// Time is structural: chain-position in SpoWitnessChain + AriGraph Triplet.timestamp.
/// Per cognitive-substrate-convergence-v1.md L-2.
#[deprecated(
    since = "0.2.0",
    note = "bits 52-63 reclaimed for W/truth/spare + mantissa expansion; \
            time is structural (chain-position + AriGraph Triplet.timestamp); \
            see cognitive-substrate-convergence-v1.md L-2 and AriGraph chain-position migration."
)]
pub const V1_TEMPORAL_SHIFT: u32 = 52;

// ── v2 NEW fields (reclaimed from dropped temporal 12 bits) ─────────────────
/// W slot: 6-bit witness corpus root handle (bits 53-58), 0..=63.
/// 0 = no corpus anchor. Per cognitive-substrate-convergence-v1.md L-6.
pub const W_SHIFT:      u32 = 53;

/// Truth-band lens: 2-bit TrustTexture ordinal (bits 59-60).
/// 0 = Crystalline. Per cognitive-substrate-convergence-v1.md L-7.
pub const TRUTH_SHIFT:  u32 = 59;

/// Spare: 3-bit reserved for sprint-12+ (bits 61-63).
/// Candidates: Rubicon-commit marker, Markov-decay quantum, I-NOISE-FLOOR-JIRAK threshold.
pub const SPARE_SHIFT:  u32 = 61;

// ── Common masks ─────────────────────────────────────────────────────────────
pub const BYTE_MASK:    u64 = 0xFF;
pub const BITS3_MASK:   u64 = 0x7;
pub const BITS6_MASK:   u64 = 0x3F;
pub const BITS2_MASK:   u64 = 0x3;

pub const PLAST_MASK:   u64 = BITS3_MASK << PLAST_SHIFT;
pub const W_MASK:       u64 = BITS6_MASK << W_SHIFT;
pub const TRUTH_MASK:   u64 = BITS2_MASK << TRUTH_SHIFT;
pub const SPARE_MASK:   u64 = BITS3_MASK << SPARE_SHIFT;

// ── Compile-time layout coverage assertion ────────────────────────────────────
/// Const-assert: all 64 bits covered exactly once.
/// 8+8+8+8+8+3+3+4+3+6+2+3 = 64.
/// Fails at compile time if the bit layout has gaps or overlaps.
const _LAYOUT_COVERAGE: () = {
    let all: u64 = (BYTE_MASK  << S_SHIFT)       // bits  0-7
        | (BYTE_MASK  << P_SHIFT)                 // bits  8-15
        | (BYTE_MASK  << O_SHIFT)                 // bits 16-23
        | (BYTE_MASK  << FREQ_SHIFT)              // bits 24-31
        | (BYTE_MASK  << CONF_SHIFT)              // bits 32-39
        | (BITS3_MASK << CAUSAL_SHIFT)            // bits 40-42
        | (BITS3_MASK << DIR_SHIFT)               // bits 43-45
        | (BITS4_MASK << INFER_SHIFT)             // bits 46-49 (4b signed mantissa)
        | (BITS3_MASK << PLAST_SHIFT)             // bits 50-52 (shifted from v1)
        | (BITS6_MASK << W_SHIFT)                 // bits 53-58 (NEW)
        | (BITS2_MASK << TRUTH_SHIFT)             // bits 59-60 (NEW)
        | (BITS3_MASK << SPARE_SHIFT);            // bits 61-63 (NEW)
    assert!(
        all == u64::MAX,
        "CausalEdge64 v2 bit layout must cover all 64 bits exactly once"
    );
};

/// Two-bit truth-band lens — 4 levels of epistemic texture.
///
/// Lens projection table (per causaledge64-mailbox-rename-soa-v1.md §2):
/// ```text
///   0b00 = Crystalline | Mastered     | Quiet  | Proceed
///   0b01 = Solid       | Calibrated   | Mild   | Proceed
///   0b10 = Fuzzy       | Uncertain    | Active | Sandbox
///   0b11 = Murky       | Contradiction| Loud   | Compass (veto)
/// ```
///
/// NOTE: Local definition in causal-edge (zero-dep crate). The canonical
/// contract type is `lance_graph_contract::mul::TrustTexture`.
/// The 2-bit encoding is byte-compatible by construction.
/// Long-term: add `From<TrustTexture> for contract::TrustTexture` at the planner boundary.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
#[repr(u8)]
pub enum TrustTexture {
    /// Fully crystalline — mastered / quiet / proceed. Default.
    #[default]
    Crystalline = 0,
    /// Solid — calibrated / mild / proceed.
    Solid = 1,
    /// Fuzzy — uncertain / active / sandbox.
    Fuzzy = 2,
    /// Murky — contradiction / loud / compass (veto).
    Murky = 3,
}

impl TrustTexture {
    /// Construct from the raw 2-bit field value (bits masked automatically).
    #[inline]
    pub fn from_bits_2(v: u8) -> Self {
        match v & 0b11 {
            0 => Self::Crystalline,
            1 => Self::Solid,
            2 => Self::Fuzzy,
            _ => Self::Murky,
        }
    }

    /// Return the raw 2-bit value (0..=3).
    #[inline]
    pub fn to_bits_2(self) -> u8 {
        self as u8
    }
}
