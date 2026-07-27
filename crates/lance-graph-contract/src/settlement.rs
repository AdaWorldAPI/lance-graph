//! Settlement as a FOUR-dimensional field, never a score.
//!
//! ## The discriminator is closure × competence
//!
//! Two independent questions, and collapsing them is the whole failure mode:
//!
//! - **Closure density** — how structurally complete the belief field is: how
//!   much of what could be derived has been.
//! - **Evidence competence** — how well-grounded that structure is, per the
//!   `deepnsm-v2` `1 - U` reading (confidence · contradiction · derived-share).
//!
//! | Closure | Competence | Cell |
//! |---|---|---|
//! | high | high | [`Crystal`](SettlementCell::Crystal) — settled and deserved |
//! | high | low | [`Glass`](SettlementCell::Glass) — **dense closure on thin evidence** |
//! | low | high | [`GroundedUnresolved`](SettlementCell::GroundedUnresolved) |
//! | low | low | [`Fog`](SettlementCell::Fog) |
//!
//! **Glass is the dangerous cell**, and it is exactly what a scalar hides: it
//! looks like Crystal from the closure side and like Fog from the evidence
//! side, so any single number averages it into something unremarkable.
//!
//! ## Entropy is a THIRD signal, not one of the two axes
//!
//! An earlier formulation put entropy on both axes — "crystal = low entropy
//! high closure, glass = low entropy low closure" — which silently deleted
//! competence and made the matrix a restatement of one variable. Entropy
//! describes **field concentration**: how narrow the surviving hypothesis
//! space is. Concentration says nothing about whether the concentration is
//! structurally closed or evidentially earned. Same for eigenvalue
//! concentration, which measures how much of the field is dominated by one
//! lineage. Both refine the cell; neither defines it:
//!
//! - Glass + low entropy + high eigenvalue → confidently calcified monoculture
//! - Glass + high entropy → many thinly-supported derived structures
//! - Crystal + low entropy + low concentration → legitimate settlement
//! - Crystal + high eigenvalue → perhaps right, but dominated by one lineage
//!
//! ## Scope alignment is a precondition, so it is a field
//!
//! Closure is a whole-arena property; competence is per-basin (often a single
//! subject). Subtracting one from the other across mismatched scopes produces
//! a confident number about nothing. [`SettlementScope`] is carried WITH the
//! signals and [`SettlementSignals::comparable_to`] refuses mismatched pairs —
//! the alignment requirement made structural rather than remembered.
//!
//! ## No derived scalar is provided, on purpose
//!
//! There is deliberately no `glass_gap()` here. A difference between closure
//! and competence is only meaningful once both are calibrated at the same
//! scope, and that calibration has not been done. Shipping the subtraction
//! first is how the four signals become one again.

/// What a [`SettlementSignals`] measurement covers.
///
/// Two readings are comparable only when every component matches. Version and
/// branch are included because a settlement reading is an epistemic
/// observation: "how settled, as of when, on which line of development".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SettlementScope {
    /// The arena / field the reading covers.
    pub arena_id: u32,
    /// The basin within it, or `None` for a whole-arena reading.
    ///
    /// A whole-arena closure and a single-basin competence are NOT comparable;
    /// this is the field that makes that checkable.
    pub basin_id: Option<u32>,
    /// The dataset version read as-of.
    pub version: u64,
    /// The line of development.
    pub branch_id: u32,
    /// How far back the evidence horizon extends, in versions. Two readings
    /// over different horizons see different evidence and are not comparable.
    pub witness_horizon: u32,
}

/// Which settlement cell a reading falls in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SettlementCell {
    /// High closure, high competence — settled, and the settlement is earned.
    Crystal,
    /// High closure, LOW competence — dense structure on thin evidence. Reads
    /// as settled from the outside and is the cell most worth interrupting.
    Glass,
    /// Low closure, high competence — well-grounded and honestly unfinished.
    GroundedUnresolved,
    /// Low closure, low competence — neither structured nor grounded.
    Fog,
}

impl SettlementCell {
    /// Does this cell present as settled, whether or not it deserves to?
    /// True for [`Crystal`](Self::Crystal) AND [`Glass`](Self::Glass) — that
    /// shared appearance is precisely why the second axis is needed.
    #[inline]
    #[must_use]
    pub const fn appears_settled(self) -> bool {
        matches!(self, Self::Crystal | Self::Glass)
    }

    /// Is the settlement evidentially earned?
    #[inline]
    #[must_use]
    pub const fn is_earned(self) -> bool {
        matches!(self, Self::Crystal | Self::GroundedUnresolved)
    }
}

/// The four preserved settlement signals for one scope.
///
/// All four are kept. There is no constructor that reduces them to a score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SettlementSignals {
    /// What this reading covers — carried so comparisons can be refused.
    pub scope: SettlementScope,
    /// Structural completeness, `0.0..=1.0`.
    pub closure_density: f32,
    /// Evidential grounding (`1 - U`), `0.0..=1.0`.
    pub evidence_competence: f32,
    /// Field concentration — how narrow the surviving space is. A refining
    /// signal, NOT one of the two classifying axes.
    pub field_entropy: f32,
    /// How much of the field one lineage dominates. Also refining.
    pub eigenvalue_concentration: f32,
}

/// Midpoint split for both classifying axes.
///
/// Hand-chosen, and said out loud per `I-NOISE-FLOOR-JIRAK`: this is NOT a
/// bound-derived threshold. It is the neutral split for an uncalibrated
/// `0..1` reading, and the falsification matrix below is what would expose it
/// if real data clusters away from the midpoint.
pub const SETTLEMENT_MIDPOINT: f32 = 0.5;

impl SettlementSignals {
    /// Which cell this reading falls in — closure × competence ONLY.
    ///
    /// Entropy and eigenvalue concentration are deliberately not consulted:
    /// they refine a cell, they do not choose it.
    #[must_use]
    pub fn cell(&self) -> SettlementCell {
        let closed = self.closure_density >= SETTLEMENT_MIDPOINT;
        let grounded = self.evidence_competence >= SETTLEMENT_MIDPOINT;
        match (closed, grounded) {
            (true, true) => SettlementCell::Crystal,
            (true, false) => SettlementCell::Glass,
            (false, true) => SettlementCell::GroundedUnresolved,
            (false, false) => SettlementCell::Fog,
        }
    }

    /// May these two readings be compared at all?
    ///
    /// Every scope component must match. This is the precondition that made
    /// `wisdom - competence` meaningless: whole-arena closure against
    /// per-basin competence is a confident number about nothing.
    #[inline]
    #[must_use]
    pub fn comparable_to(&self, other: &Self) -> bool {
        self.scope == other.scope
    }

    /// Is this a confidently-calcified monoculture — glass, narrow, and
    /// dominated by one lineage?
    ///
    /// The composite worth naming, because all three signals must agree before
    /// it means anything, and it is still a PREDICATE over preserved fields,
    /// never a score that replaces them.
    #[must_use]
    pub fn is_calcified_monoculture(&self, entropy_ceiling: f32, dominance_floor: f32) -> bool {
        self.cell() == SettlementCell::Glass
            && self.field_entropy <= entropy_ceiling
            && self.eigenvalue_concentration >= dominance_floor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> SettlementScope {
        SettlementScope {
            arena_id: 1,
            basin_id: None,
            version: 7,
            branch_id: 0,
            witness_horizon: 32,
        }
    }

    fn signals(closure: f32, competence: f32, entropy: f32, eigen: f32) -> SettlementSignals {
        SettlementSignals {
            scope: scope(),
            closure_density: closure,
            evidence_competence: competence,
            field_entropy: entropy,
            eigenvalue_concentration: eigen,
        }
    }

    /// The falsification matrix: all four cells are reachable, and each is
    /// reached by varying ONLY the two classifying axes.
    #[test]
    fn all_four_cells_are_reachable() {
        assert_eq!(signals(0.9, 0.9, 0.1, 0.1).cell(), SettlementCell::Crystal);
        assert_eq!(signals(0.9, 0.1, 0.1, 0.1).cell(), SettlementCell::Glass);
        assert_eq!(
            signals(0.1, 0.9, 0.1, 0.1).cell(),
            SettlementCell::GroundedUnresolved
        );
        assert_eq!(signals(0.1, 0.1, 0.1, 0.1).cell(), SettlementCell::Fog);
    }

    /// **Entropy must NOT move the cell.** This is the regression guard against
    /// the earlier formulation that put entropy on both axes and thereby
    /// deleted competence from the matrix.
    #[test]
    fn entropy_and_eigenvalue_never_change_the_cell() {
        for entropy in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for eigen in [0.0, 0.5, 1.0] {
                assert_eq!(
                    signals(0.9, 0.2, entropy, eigen).cell(),
                    SettlementCell::Glass,
                    "closure/competence decide; entropy={entropy} eigen={eigen} must not"
                );
            }
        }
    }

    /// Closure and competence are INDEPENDENTLY variable — the orthogonality
    /// receipt, two non-trivial witnesses.
    #[test]
    fn closure_and_competence_are_independently_variable() {
        // Witness 1: closure varies, competence fixed low → Fog ⇄ Glass.
        assert_ne!(
            signals(0.1, 0.2, 0.5, 0.5).cell(),
            signals(0.9, 0.2, 0.5, 0.5).cell()
        );
        // Witness 2: competence varies, closure fixed high → Glass ⇄ Crystal.
        assert_ne!(
            signals(0.9, 0.2, 0.5, 0.5).cell(),
            signals(0.9, 0.8, 0.5, 0.5).cell()
        );
    }

    /// Glass and Crystal are indistinguishable on appearance and separated only
    /// by competence — the reason a single settlement score cannot work.
    #[test]
    fn glass_and_crystal_both_appear_settled() {
        let glass = signals(0.9, 0.2, 0.2, 0.9);
        let crystal = signals(0.9, 0.9, 0.2, 0.2);
        assert!(glass.cell().appears_settled());
        assert!(crystal.cell().appears_settled());
        assert!(!glass.cell().is_earned());
        assert!(crystal.cell().is_earned());
    }

    /// Mismatched scope refuses comparison — whole-arena vs per-basin is the
    /// exact mix that made the earlier subtraction meaningless.
    #[test]
    fn mismatched_scope_is_not_comparable() {
        let whole = signals(0.9, 0.9, 0.2, 0.2);
        let mut per_basin = whole;
        per_basin.scope.basin_id = Some(4);
        assert!(!whole.comparable_to(&per_basin));
        assert!(whole.comparable_to(&whole.clone()));

        // Version and horizon are equally disqualifying.
        let mut later = whole;
        later.scope.version = 8;
        assert!(!whole.comparable_to(&later));
        let mut wider = whole;
        wider.scope.witness_horizon = 64;
        assert!(!whole.comparable_to(&wider));
    }

    /// The composite predicate discriminates in BOTH directions — it fires on a
    /// calcified monoculture and stays silent on non-trivial near-misses, one
    /// per conjunct.
    #[test]
    fn calcified_monoculture_fires_and_stays_silent() {
        let (ceil, floor) = (0.3, 0.7);

        assert!(signals(0.9, 0.2, 0.1, 0.9).is_calcified_monoculture(ceil, floor));

        // Earned settlement — Crystal, not Glass.
        assert!(!signals(0.9, 0.9, 0.1, 0.9).is_calcified_monoculture(ceil, floor));
        // Glass, but the field is still wide.
        assert!(!signals(0.9, 0.2, 0.8, 0.9).is_calcified_monoculture(ceil, floor));
        // Glass and narrow, but no single lineage dominates.
        assert!(!signals(0.9, 0.2, 0.1, 0.2).is_calcified_monoculture(ceil, floor));
    }

    /// The thresholds are live knobs, not decoration: tightening must silence,
    /// loosening must admit.
    #[test]
    fn monoculture_thresholds_are_not_inert() {
        let s = signals(0.9, 0.2, 0.4, 0.6);
        assert!(!s.is_calcified_monoculture(0.3, 0.7), "outside both bounds");
        assert!(s.is_calcified_monoculture(0.5, 0.5), "loosening admits it");
    }
}
