//! Per-axis quorum projection — D-ATOM-3 of `atom-mailbox-substrate-v1`.
//!
//! # Concept
//!
//! A bipolar dichotomy does **not** yield its pole assignment for free. To
//! place a measurement between two poles you need a **quorum**: a structured
//! agreement among the `InnerCouncil` archetypes (and, optionally, the wider
//! `a2a_blackboard` `support[u16;4]` + `dissonance` field) that the signal
//! belongs to the positive vs negative half of the axis.
//!
//! The output of a successful quorum is an [`AxisProjection`]:
//!
//! ```text
//!   AxisProjection { position: i8, confidence: f32 }
//!                       ↑                 ↑
//!           I4 pole (−8 … +7)     quorum agreement ∈ [0, 1]
//!           = NARS frequency      = NARS confidence
//!           (normalised)
//! ```
//!
//! This maps directly onto **NARS truth per axis**: `(frequency ≈ position
//! normalised to [0,1], confidence ≈ quorum strength)`. The I4 integer is the
//! coarse pole; the f32 confidence is how strongly the quorum agrees.
//!
//! # Split quorums are Contradictions — NEVER averaged
//!
//! When [`InnerCouncil::deliberate`] fires `CouncilVerdict::split = true`,
//! the projection is **contested**: the majority pole is recorded, but
//! [`AxisProjection::is_contested`] returns `true`. The caller MUST hand the
//! contested projection off to the counterfactual path (D-ATOM-4) rather than
//! averaging away the disagreement. Averaging a split launders false
//! confidence — the cardinal OSINT sin.
//!
//! # Tiering non-decision (architectural note)
//!
//! `EPIPHANIES.md` E-LADDER-SERVES-MAILBOX §5 explicitly chose the
//! **counterfactual-fork** strategy (D-ATOM-4) OVER quorum-tiering. This
//! module therefore exposes *only* the projection surface (`AxisProjection`,
//! `quorum_project`, `is_contested`) and hands contested cases off to
//! D-ATOM-4. The quorum does **not** widen through tiers — that complexity
//! was intentionally rejected.
//!
//! # BLOCKED
//!
//! - `atoms` axis type (D-ATOM-1, parallel — referenced below as the intended
//!   `AxisId` or equivalent) — not yet defined; this module uses `u8` as a
//!   placeholder index and marks every call-site.
//! - The exact `a2a_blackboard::Blackboard` constructor / `next_round` reset
//!   policy when used *per-axis* vs *per-round* is unclear from the source;
//!   the wide-quorum path below is therefore fully `BLOCKED`.
//!
//! Zero-dep crate — no external dependencies beyond `crate::escalation`.

use crate::escalation::{CouncilVerdict, InnerCouncil};

// ═══════════════════════════════════════════════════════════════════════════
// AxisProjection — the quorum output
// ═══════════════════════════════════════════════════════════════════════════

/// The result of projecting a set of signals onto one bipolar axis via quorum.
///
/// Semantically this is **NARS truth per axis**:
///
/// | Field        | NARS role     | Range     | Semantics                                  |
/// |---|---|---|---|
/// | `position`   | frequency     | `−8 … +7` | I4 pole; negative = "−" half, positive = "+" half of the dichotomy; `0` = indeterminate |
/// | `confidence` | confidence    | `[0, 1]`  | Quorum agreement strength; low on a split |
///
/// A `position` of `0` with any confidence indicates the quorum could not
/// place the signal on either pole (degenerate input). This is distinct from a
/// *contested* projection (where `is_contested` returns `true`): `position ≠ 0`
/// but the council split.
///
/// # Relationship to I4-32D atoms (D-ATOM-1)
///
/// // BLOCKED: D-ATOM-1 has not yet defined the atom catalogue or the `I4x32`
/// // pack/unpack API. Once D-ATOM-1 lands, each atom slot in `I4x32` is filled
/// // by one `AxisProjection::position` value, with the accompanying
/// // `confidence` stored alongside for NARS downstream use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisProjection {
    /// I4 pole on this axis (`−8 … +7`). Positive = "+" pole; negative = "−"
    /// pole; `0` = indeterminate (quorum could not resolve).
    pub position: i8,
    /// Quorum agreement in `[0, 1]`. Carries NARS-confidence semantics:
    /// high = strong quorum agreement; low = weak or contested.
    pub confidence: f32,
    /// True when the `InnerCouncil` deliberation produced `split = true`,
    /// meaning the projection is contested and MUST be handed to the
    /// counterfactual path (D-ATOM-4), not averaged.
    contested: bool,
}

impl AxisProjection {
    /// Construct a settled (non-contested) projection.
    #[inline]
    pub fn settled(position: i8, confidence: f32) -> Self {
        Self {
            position: position.clamp(-8, 7),
            confidence: confidence.clamp(0.0, 1.0),
            contested: false,
        }
    }

    /// Construct a contested projection.  The majority `position` is recorded,
    /// but callers MUST check [`is_contested`] and route to D-ATOM-4.
    #[inline]
    pub fn contested(position: i8, confidence: f32) -> Self {
        Self {
            position: position.clamp(-8, 7),
            confidence: confidence.clamp(0.0, 1.0),
            contested: true,
        }
    }

    /// True when the underlying quorum was a split — the projection is
    /// **contested** and MUST NOT be averaged into a final atom value.
    ///
    /// A contested projection is a **Contradiction** (see
    /// `E-LADDER-SERVES-MAILBOX §3`): the majority pole is stored but the
    /// minority pole is the counterfactual mantissa owned by D-ATOM-4.
    /// Callers that ignore this flag and treat the projection as settled are
    /// laundering false confidence — this is the cardinal OSINT sin.
    #[inline]
    pub fn is_contested(&self) -> bool {
        self.contested
    }

    /// NARS frequency: `position` mapped to `[0, 1]` as
    /// `(position + 8) / 15.0`.  Positive pole → frequency > 0.5; negative
    /// pole → frequency < 0.5; indeterminate (0) → ≈ 0.533.
    ///
    /// This is a lossy normalisation; callers that need the raw I4 value
    /// should use `position` directly.
    #[inline]
    pub fn nars_frequency(&self) -> f32 {
        (self.position as f32 + 8.0) / 15.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Signal type — the raw per-axis observables fed into the quorum
// ═══════════════════════════════════════════════════════════════════════════

/// One contributing signal for a single axis quorum.
///
/// Signals are the raw per-axis observables — e.g. trust, humility, flow,
/// load from a [`crate::mul::MulAssessment`] or from a
/// `crate::a2a_blackboard::BlackboardEntry` — normalised to `[0, 1]`.
///
/// # BLOCKED
///
/// // BLOCKED: D-ATOM-1 — axis identity type (`AxisId` or equivalent in
/// // `contract::atoms`) is not yet defined. Until D-ATOM-1 lands, the
/// // *which axis* question is answered by the caller; this struct carries
/// // only the scalar signal payload.
#[derive(Debug, Clone, Copy)]
pub struct AxisSignal {
    /// Trust component (see [`InnerCouncil::from_signals`]) ∈ `[0, 1]`.
    pub trust: f32,
    /// Humility / DK component ∈ `[0, 1]`.
    pub humility: f32,
    /// Flow component ∈ `[0, 1]`.
    pub flow: f32,
    /// Allostatic load ∈ `[0, 1]`.
    pub load: f32,
    /// Optional raw polarity hint: positive means the signal tilts toward
    /// the "+" pole; negative toward the "−" pole. When `None` the polarity
    /// is inferred entirely from the council deliberation.
    pub polarity_hint: Option<f32>,
}

// ═══════════════════════════════════════════════════════════════════════════
// quorum_project — the core per-axis projection function
// ═══════════════════════════════════════════════════════════════════════════

/// Project a set of contributing `signals` onto one bipolar axis via the
/// `InnerCouncil` quorum, returning an [`AxisProjection`].
///
/// # Mechanism
///
/// 1. Each signal is fed into [`InnerCouncil::from_signals`] to obtain a
///    per-signal [`CouncilVerdict`].
/// 2. The verdicts are aggregated: the mean confidence across all signals
///    serves as the quorum confidence; the dominant polarity is derived from
///    the weighted polarity hints (or, if absent, from the Balanced/Flow vs
///    Guardian/Catalyst ratio).
/// 3. If **any** single verdict is `split = true`, the aggregate is marked
///    contested (see [`AxisProjection::is_contested`]). A split is
///    **not averaged** — the majority pole is committed and the minority
///    handed to D-ATOM-4.
/// 4. The resulting I4 `position` is clamped to `−8 … +7`; `confidence`
///    is the mean quorum agreement, ×1.2 amplified (clamped to 1.0) on a
///    split (mirroring `InnerCouncil::deliberate` split amplification —
///    disagreement IS the learning signal, per
///    `E-LADDER-SERVES-MAILBOX §3`).
///
/// # Empty signal set
///
/// If `signals` is empty the function returns an indeterminate projection
/// (`position = 0, confidence = 0.0, contested = false`).
///
/// # BLOCKED
///
/// // BLOCKED: D-ATOM-1 — the axis identity (`AxisId`) type is not yet
/// // defined. The caller currently selects the axis implicitly by passing
/// // the right signals; once D-ATOM-1 lands this should accept an
/// // `AxisId` parameter.
///
/// // BLOCKED: wide-quorum path — for multi-expert quorum the
/// // `a2a_blackboard::Blackboard::support[u16;4]` + `dissonance` fields
/// // are the right wide-quorum substrate, but the per-axis vs per-round
/// // reset policy for `Blackboard::next_round` is unclear. The wide-quorum
/// // variant is deferred until D-ATOM-3 implementation.
pub fn quorum_project(signals: &[AxisSignal], council: &InnerCouncil) -> AxisProjection {
    todo!(
        "D-ATOM-3 — implement: aggregate InnerCouncil verdicts from signals, \
         derive I4 position from polarity hints + majority, mark contested on any split; \
         BLOCKED: AxisId (D-ATOM-1), wide-quorum Blackboard reset policy"
    );
    // Silence unused-variable warnings in the scaffold.
    #[allow(unused_variables)]
    let _ = (signals, council);
}

// ═══════════════════════════════════════════════════════════════════════════
// Wide-quorum variant (a2a_blackboard path) — BLOCKED
// ═══════════════════════════════════════════════════════════════════════════

/// Project one axis using the wide `a2a_blackboard` quorum
/// (`BlackboardEntry::support[u16;4]` + `dissonance`).
///
/// The wide-quorum path is the Layer-1 A2A complement to the three-archetype
/// `InnerCouncil`: where `InnerCouncil` is a local (sync, 3-archetype) quorum,
/// `quorum_project_blackboard` aggregates across all posted
/// `BlackboardEntry` values on the current round, using `dissonance` as the
/// split signal.
///
/// # BLOCKED
///
/// // BLOCKED: `a2a_blackboard::Blackboard` constructor / `next_round` reset
/// // policy when sliced per-axis is unclear from the source. The `support`
/// // field carries top-K atom indices (`[u16; 4]`) but the mapping from
/// // support-slot semantics to the bipolar axis pole is not yet specified
/// // (D-ATOM-1 defines the atom catalogue this indexes into).
/// //
/// // BLOCKED: D-ATOM-1 — `AxisId` type needed to filter `BlackboardEntry`
/// // entries by axis.
pub fn quorum_project_blackboard(
    _bb: &crate::a2a_blackboard::Blackboard,
    // BLOCKED: AxisId parameter once D-ATOM-1 lands.
) -> AxisProjection {
    todo!(
        "D-ATOM-3 wide-quorum path — BLOCKED: per-axis Blackboard slice semantics \
         + AxisId (D-ATOM-1)"
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// ContestHandler — handoff contract to D-ATOM-4
// ═══════════════════════════════════════════════════════════════════════════

/// Outcome of inspecting a contested projection.
///
/// When [`AxisProjection::is_contested`] returns `true` the caller uses this
/// enum to decide the handoff strategy. This enum is the type-level seam
/// between D-ATOM-3 (quorum projection) and D-ATOM-4 (counterfactual
/// mantissa). **D-ATOM-3 never resolves the contest itself** — the tiering
/// non-decision (see module-level note) means resolution belongs to D-ATOM-4.
///
/// # Staging (mirrors `E-LADDER-SERVES-MAILBOX §5`)
///
/// - **v1 (now):** `DropMinority` — commit the majority pole, drop the
///   minority. No counterfactual record.
/// - **v2 (D-ATOM-4 deposit):** `DepositMantissa` — commit the majority
///   pole, deposit the minority as a `CausalEdge64` −6 (Counterfactual)
///   nibble.
/// - **v3 (D-ATOM-4 mailbox + revision):** `SpawnCounterfactual` — full
///   ghost-tier test mailbox + `awareness.revise` on minority-wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContestHandler {
    /// v1 — commit majority, discard minority silently.
    DropMinority,
    /// v2 — commit majority, deposit minority as `CausalEdge64 −6` nibble
    /// (D-ATOM-4 deposit phase). The 4-bit mantissa is the road-not-taken.
    DepositMantissa,
    /// v3 — spawn a ghost-tier counterfactual mailbox; if the minority pole
    /// later beats the committed pole's free energy, call `awareness.revise`
    /// (D-ATOM-4 full phase, gated on β headroom and Staunen threshold).
    SpawnCounterfactual,
}

/// Resolve a contested [`AxisProjection`] according to the given
/// [`ContestHandler`] strategy.
///
/// Returns `(committed_projection, minority_pole)` where `committed_projection`
/// has `contested = false` (the contest is resolved by policy, not averaged)
/// and `minority_pole` is the I4 value of the road-not-taken (negation of
/// the committed position for a pure split; the caller may use this to seed
/// D-ATOM-4's counterfactual record).
///
/// When `handler == ContestHandler::DropMinority` the minority pole is still
/// returned (for diagnostic purposes) but no counterfactual record is created —
/// that is D-ATOM-4's responsibility, not this module's.
///
/// # Panics
///
/// Does not panic; if `projection` is not contested this is a no-op and
/// returns the projection unchanged with `minority_pole = 0`.
pub fn resolve_contest(
    projection: AxisProjection,
    _handler: ContestHandler,
) -> (AxisProjection, i8) {
    todo!(
        "D-ATOM-3 — implement resolve_contest: strip the contested flag from \
         the majority projection, compute minority pole; for DepositMantissa + \
         SpawnCounterfactual the actual mantissa deposit / mailbox spawn is \
         delegated to D-ATOM-4 (this function returns the raw minority pole for \
         the caller to pass through); BLOCKED: D-ATOM-4 API"
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests (scaffold — bodies todo!())
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Settled projection carries the expected NARS truth fields.
    #[test]
    fn settled_projection_fields() {
        let p = AxisProjection::settled(3, 0.8);
        assert_eq!(p.position, 3);
        assert!((p.confidence - 0.8).abs() < 1e-6);
        assert!(!p.is_contested());
    }

    /// Contested projection sets the flag correctly.
    #[test]
    fn contested_projection_flag() {
        let p = AxisProjection::contested(-4, 0.6);
        assert!(p.is_contested());
        assert_eq!(p.position, -4);
    }

    /// I4 position is clamped to −8 … +7.
    #[test]
    fn position_clamps_to_i4_range() {
        assert_eq!(AxisProjection::settled(100, 1.0).position, 7);
        assert_eq!(AxisProjection::settled(-100, 1.0).position, -8);
    }

    /// nars_frequency maps the I4 range to [0, 1].
    #[test]
    fn nars_frequency_range() {
        let neg = AxisProjection::settled(-8, 1.0);
        let pos = AxisProjection::settled(7, 1.0);
        assert!((neg.nars_frequency() - 0.0).abs() < 1e-6);
        assert!((pos.nars_frequency() - 1.0).abs() < 1e-6);
        // Midpoint (position 0) should be > 0.5 (asymmetric I4 range).
        let mid = AxisProjection::settled(0, 1.0);
        assert!(mid.nars_frequency() > 0.5);
    }

    /// quorum_project with empty signals → indeterminate (todo! body not yet
    /// reached — this test documents the *intended* contract only).
    #[test]
    #[should_panic(expected = "D-ATOM-3")]
    fn quorum_project_is_scaffolded() {
        let _ = quorum_project(&[], &InnerCouncil);
    }

    /// resolve_contest is scaffolded — panics with D-ATOM-3 message.
    #[test]
    #[should_panic(expected = "D-ATOM-3")]
    fn resolve_contest_is_scaffolded() {
        let p = AxisProjection::contested(2, 0.7);
        let _ = resolve_contest(p, ContestHandler::DropMinority);
    }
}
