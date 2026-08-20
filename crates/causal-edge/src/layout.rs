//! CausalEdge64 v2 layout constants — FINAL (Option F, locked 2026-05-16).
//!
//! Cite: cognitive-substrate-convergence-v1.md §6 (authoritative bit layout)
//!       + pr-ce64-mb-2-causaledge64-v2.md §2 (implementation contract).
//! OQ-LAYOUT-1: RESOLVED. G-slot dropped (L-3). Mantissa = 4b signed i4 (L-4).

// ── v1 fields preserved (shifts unchanged from v1) ─────────────────────────
pub const S_SHIFT: u32 = 0;
pub const P_SHIFT: u32 = 8;
pub const O_SHIFT: u32 = 16;
pub const FREQ_SHIFT: u32 = 24;
pub const CONF_SHIFT: u32 = 32;
pub const CAUSAL_SHIFT: u32 = 40;
pub const DIR_SHIFT: u32 = 43;

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
pub const INFER_SHIFT: u32 = 46;

/// 4-bit unsigned mask for pack/unpack of the signed i4 mantissa field.
pub const BITS4_MASK: u64 = 0xF;

/// Mask covering the mantissa field (bits 46-49) in the u64 word.
pub const INFER_MASK: u64 = BITS4_MASK << INFER_SHIFT;

// ── v1 field SHIFTED ────────────────────────────────────────────────────────
/// Plasticity flags: bits 50-52 (shifted by +1 from v1's bits 49-51 due to
/// mantissa expansion from 3b unsigned to 4b signed i4 per L-4).
pub const PLAST_SHIFT: u32 = 50;

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
pub const W_SHIFT: u32 = 53;

/// Truth-band lens: 2-bit TrustTexture ordinal (bits 59-60).
/// 0 = Crystalline. Per cognitive-substrate-convergence-v1.md L-7.
///
/// Same two bits also carry an ADDITIVE factual view, [`CausalTopology`]
/// (below) — see its doc comment for the wire/ordinal/behavioural/
/// provenance compatibility statement. `TrustTexture` remains the
/// canonical epistemic-trust reading; `CausalTopology` is a second,
/// orthogonal reading of the identical bits for producers that want to
/// record topology instead of (or alongside) trust texture. No bits move.
pub const TRUTH_SHIFT: u32 = 59;

/// Spare: 3-bit reserved for sprint-12+ (bits 61-63).
/// Candidates: Rubicon-commit marker, Markov-decay quantum, I-NOISE-FLOOR-JIRAK threshold.
///
/// Same three bits also carry an ADDITIVE quantized-projection view,
/// [`ReasoningBand`] (below). No auto-derivation: nothing writes this
/// field except an explicit `with_reasoning_band()` call.
///
/// **v1 provenance:** bits 61-63 were temporal bits 9-11, so a v1 edge with
/// `temporal >= 512` reads a NON-ZERO band. Apply a version gate on edges of
/// unknown provenance — the same rule `truth()` states for bits 59-60.
pub const SPARE_SHIFT: u32 = 61;

// ── Common masks ─────────────────────────────────────────────────────────────
pub const BYTE_MASK: u64 = 0xFF;
pub const BITS3_MASK: u64 = 0x7;
pub const BITS6_MASK: u64 = 0x3F;
pub const BITS2_MASK: u64 = 0x3;

pub const PLAST_MASK: u64 = BITS3_MASK << PLAST_SHIFT;
pub const W_MASK: u64 = BITS6_MASK << W_SHIFT;
pub const TRUTH_MASK: u64 = BITS2_MASK << TRUTH_SHIFT;
pub const SPARE_MASK: u64 = BITS3_MASK << SPARE_SHIFT;

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
        | (BITS3_MASK << SPARE_SHIFT); // bits 61-63 (NEW)
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
/// NOTE: Local definition in causal-edge (zero-dep crate).
///
/// **CORRECTION (measured):** an earlier version of this note claimed
/// `lance_graph_contract::mul::TrustTexture` is "the canonical contract type"
/// and "byte-compatible by construction". Both halves are false. That type's
/// variants are `Calibrated / Overconfident / Uncertain / Underconfident` —
/// a different ontology (felt-vs-demonstrated competence), with no semantic
/// mapping onto `Crystalline / Solid / Fuzzy / Murky`, and no `From` impl
/// exists in either direction. `docs/TYPE_DUPLICATION_MAP.md` rules the
/// opposite of the old note: **"Canonical: NONE — both are domain-correct
/// and should keep distinct names."** Do not build a cast on the old claim.
///
/// This enum is the LEGACY/COMPATIBILITY projection of bits 59-60; the
/// factual view over the same bits is [`CausalTopology`]. Not deprecated —
/// nothing is ready to move, and the wider rename is owned by the existing
/// TrustTexture-duplication debt item.
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

/// Two-bit CAUSAL TOPOLOGY lens — an ADDITIVE factual view over the same
/// two bits (59-60) that [`TrustTexture`] occupies. No bits move; this is
/// a second reading of the identical register, not a new field and not a
/// layout revision (no CE64 v3, no `ENVELOPE_LAYOUT_VERSION` bump).
///
/// `TrustTexture` reads those bits as an EPISTEMIC-TRUST texture (how much
/// to trust the edge: mastered/calibrated/uncertain/contradiction).
/// `CausalTopology` reads the identical bits as a FACTUAL TOPOLOGY
/// classification (how the edge's causal path is structured — direct vs.
/// mediated, known vs. unknown intermediates).
///
/// ```text
///   0b00 = Direct
///   0b01 = IndirectKnownIntermediates
///   0b10 = IndirectUnknownIntermediates
///   0b11 = Unknown
/// ```
///
/// ## Migration contract — the ordinal identity is deliberate, not coincidence
///
/// `Crystalline == Direct == 0`, `Solid == IndirectKnownIntermediates == 1`,
/// `Fuzzy == IndirectUnknownIntermediates == 2`, `Murky == Unknown == 3`.
///
/// Four compatibility axes, stated explicitly because they are NOT the same
/// claim and must not be conflated:
///
/// ```text
/// wire compatibility:            exact
/// ordinal compatibility:         exact
/// legacy behavioural projection: intentional
/// historical factual provenance: NOT guaranteed for old rows
/// ```
///
/// - **Wire compatibility is exact** — same two bits (59-60), same shift,
///   same mask, same byte layout. Nothing about the CE64 wire changes.
/// - **Ordinal compatibility is exact** — `TrustTexture as u8 ==
///   CausalTopology as u8` for every one of the four variants (verified by
///   test).
/// - **Legacy behavioural projection is intentional.** Reading an existing
///   row's bits 59-60 through [`CausalEdge64::topology`] reproduces exactly
///   the ordinal a `TrustTexture` reader would have gotten from the same
///   bits. That projection is BY DESIGN, so a consumer that has not moved to
///   `CausalTopology` yet is unaffected by rows a `CausalTopology`-aware
///   writer produces, and vice versa — this is a staged migration, not a
///   flag day.
/// - **Historical factual provenance is NOT guaranteed for old rows.** A row
///   written before this change had its bits 59-60 stamped with
///   TRUST-TEXTURE semantics (how confident the writer was in the edge), not
///   TOPOLOGY semantics (how the causal path is shaped). The two concepts
///   are correlated in practice but they are not the same fact, and nothing
///   in this change infers, repairs, or backfills the topology of a
///   pre-existing row. **Do not treat `old_edge.topology()` as ground truth
///   about that row's actual causal topology** — it is only the
///   same-ordinal projection through the new lens. Source-authoritative
///   topology begins only when a later producer explicitly writes
///   `CausalTopology` via [`CausalEdge64::with_topology`].
///
/// [`CausalEdge64::ZERO`] therefore reads `CausalTopology::Direct` under
/// this view exactly as it reads `TrustTexture::Crystalline` under the old
/// one — that is intentional for this staged migration (the all-zero
/// default), not a sentinel asserting "known to be direct."
///
/// [`CausalEdge64::ZERO`]: super::edge::CausalEdge64::ZERO
/// [`CausalEdge64::topology`]: super::edge::CausalEdge64::topology
/// [`CausalEdge64::with_topology`]: super::edge::CausalEdge64::with_topology
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
#[repr(u8)]
pub enum CausalTopology {
    /// Direct causal edge, no intermediates. Default.
    /// Ordinal-identical to `TrustTexture::Crystalline`.
    #[default]
    Direct = 0,
    /// Indirect, with known/named intermediate nodes on the causal path.
    /// Ordinal-identical to `TrustTexture::Solid`.
    IndirectKnownIntermediates = 1,
    /// Indirect, but the intermediate nodes are unknown/unnamed.
    /// Ordinal-identical to `TrustTexture::Fuzzy`.
    IndirectUnknownIntermediates = 2,
    /// Topology not established. Ordinal-identical to `TrustTexture::Murky`.
    Unknown = 3,
}

impl CausalTopology {
    /// Construct from the raw 2-bit field value (bits masked automatically).
    #[inline]
    pub fn from_bits_2(v: u8) -> Self {
        match v & 0b11 {
            0 => Self::Direct,
            1 => Self::IndirectKnownIntermediates,
            2 => Self::IndirectUnknownIntermediates,
            _ => Self::Unknown,
        }
    }

    /// Return the raw 2-bit value (0..=3).
    #[inline]
    pub fn to_bits_2(self) -> u8 {
        self as u8
    }
}

/// Three-bit quantized reasoning-level projection over the SPARE bits
/// (61-63) — the same three bits [`super::edge::CausalEdge64::spare`] /
/// [`super::edge::CausalEdge64::with_spare`] expose as a raw, uninterpreted
/// 3-bit scalar. ADDITIVE only: no bits move, no layout revision.
///
/// ```text
///   0b000 = Surface
///   0b001 = Association
///   0b010 = Relation
///   0b011 = Causal
///   0b100 = Counterfactual
///   0b101 = Perspective
///   0b110 = Meta
///   0b111 = Transcendent
/// ```
///
/// This is a QUANTIZED HOT-PATH PROJECTION of a potentially richer future
/// texture model — 8 ordinals is what fits in 3 bits on the hot register,
/// not a claim that reasoning has exactly 8 levels. Treat it as a coarse,
/// register-resident classifier only, never as that richer model itself.
///
/// ## Deliberately orthogonal to existing fields with colliding names
///
/// The words above collide with existing `CausalEdge64` vocabulary. Every
/// collision below is INTENTIONAL and the two meanings are ORTHOGONAL —
/// setting one never implies, derives, or requires the other:
///
/// - `Causal` = the cognition currently operating at the causal reasoning
///   level. This is **not** [`super::pearl::CausalMask`] (bits 40-42),
///   which says WHICH Pearl/SPO projection (S/P/O planes) is represented.
///   An edge may carry `ReasoningBand::Causal` under any `CausalMask`.
/// - `Counterfactual` = the reasoning CONTEXT is counterfactual. This is
///   **not** the signed inference mantissa's −6 slot (bits 46-49; see
///   `InferenceType::Counterfactual::to_mantissa() == -6`), which names one
///   specific NARS operation as counterfactual. An edge may carry
///   `ReasoningBand::Counterfactual` while its mantissa encodes any NARS
///   rule, and an edge with mantissa −6 need not carry
///   `ReasoningBand::Counterfactual`.
/// - `Perspective` = perspective/decentration reasoning (I/Thou/It,
///   Self/Other/World). This is **not**
///   [`super::edge::CausalEdge64::direction`] (bits 43-45), the
///   pathology-per-plane sign triad.
/// - `Meta` = reasoning ABOUT reasoning/evidence/revision (meta-cognition),
///   independent of every other field on the edge.
/// - `Transcendent` = mechanically, the highest ordinal this 3-bit
///   projection can express — the topmost cross-frame reasoning level in
///   this band. Nothing more: no mystical or philosophical behaviour is
///   implied or triggered by this value anywhere in this crate.
///
/// ## No auto-derivation
///
/// Nothing in this crate derives this field from `CausalMask`,
/// `InferenceType`, NARS frequency/confidence, MUL, ReasoningGap,
/// potholes, or `ThinkingStyle`. It is set ONLY by an explicit
/// `with_reasoning_band()` call, and reads whatever the SPARE bits already
/// hold otherwise (0 / `Surface` for `CausalEdge64::ZERO` and for every row
/// produced by this crate's own constructors, since `spare()` already
/// defaults to 0 there — but not guaranteed for a raw `u64` from elsewhere).
/// # Why NOT "TextureBand"
///
/// The name was settled by a workspace vocabulary audit, not by preference.
/// "Texture" is the most collided word in this stack's cognitive vocabulary:
/// FOUR distinct `TrustTexture` enums exist (this crate's, the contract's
/// `mul::TrustTexture`, the planner's 5-variant `mul::trust::TrustTexture`,
/// and AriGraph's 3-variant orchestrator one), that duplication is already
/// booked debt whose recorded remedy is a RENAME, and an operator ruling
/// holds "Texture = binding topology, not polarity" — i.e. Texture is
/// deliberately NOT an ordinal. Naming a 3-bit ordinal `TextureBand`, in
/// this file, directly beneath `TrustTexture` and over the adjacent bits,
/// would read as one 5-bit widening of it. It is an unrelated field.
///
/// # NOT `RungLevel`, despite four shared variant names
///
/// This band shares four variant NAMES with
/// `lance_graph_contract::cognitive_shader::RungLevel` — `Surface`,
/// `Counterfactual`, `Meta`, `Transcendent` — at DIFFERENT ordinals
/// (0 / 6 / 7 / 9 there vs 0 / 4 / 6 / 7 here). The two are unrelated enums:
/// never cast, compare, or map between them by ordinal.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
#[repr(u8)]
pub enum ReasoningBand {
    /// Surface-level reasoning. Default.
    #[default]
    Surface = 0,
    /// Association-level reasoning.
    Association = 1,
    /// Relation-level reasoning.
    Relation = 2,
    /// Causal-level reasoning. See the orthogonality note re: `CausalMask`.
    Causal = 3,
    /// Counterfactual reasoning context. See the orthogonality note re: the
    /// inference mantissa's −6 slot.
    Counterfactual = 4,
    /// Perspective/decentration reasoning. See the orthogonality note re:
    /// `direction`.
    Perspective = 5,
    /// Meta-cognitive reasoning (reasoning about reasoning/evidence/revision).
    Meta = 6,
    /// Highest ordinal in this band. Mechanical only — see note above.
    Transcendent = 7,
}

impl ReasoningBand {
    /// Construct from the raw 3-bit field value (bits masked automatically).
    #[inline]
    pub fn from_bits_3(v: u8) -> Self {
        match v & 0b111 {
            0 => Self::Surface,
            1 => Self::Association,
            2 => Self::Relation,
            3 => Self::Causal,
            4 => Self::Counterfactual,
            5 => Self::Perspective,
            6 => Self::Meta,
            _ => Self::Transcendent,
        }
    }

    /// Return the raw 3-bit value (0..=7).
    #[inline]
    pub fn to_bits_3(self) -> u8 {
        self as u8
    }
}
