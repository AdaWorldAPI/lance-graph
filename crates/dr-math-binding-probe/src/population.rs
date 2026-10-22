//! D-DRMC-2: deliberately small, independent FINITE POPULATION audit oracle.
//!
//! NOT a physical runtime executor, graph engine, statistical test, proof of
//! causality, or independent authentication of supplied source data. Reads
//! one 64x64 QR focus to find concrete counterexamples before a law is admitted.
//! Hot production folds remain existing Quack/R2IL/mask-risc/native SIMD.
use crate::{BoundDr64, FrameStore, Truth};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PopulationReport {
    /// Pairs selected by Gamma within this concrete 64x64 focus.
    pub eligible: u32,
    /// Eligible positions with an observed TRUE/valid relationship.
    pub true_pairs: u32,
    /// Eligible positions observed FALSE (counterexamples).
    pub false_pairs: u32,
    /// Eligible positions without a known TRUE/FALSE observation.
    pub unknown_pairs: u32,
}

impl PopulationReport {
    pub const fn checked(self) -> u32 {
        self.true_pairs + self.false_pairs
    }
    /// Finite-focus only: no statement about unobserved people or any theorem
    /// over every possible population.
    pub fn universal_on_focus(self) -> Result<Self, ClaimRefusal> {
        // Refuted beats unknown (certain-answer semantics): one observed
        // counterexample refutes the claim whatever the unknown pairs hold,
        // so it is reported first and never hidden behind `Unknown`.
        if self.eligible == 0 {
            return Err(ClaimRefusal::Vacuous);
        }
        if self.false_pairs != 0 {
            return Err(ClaimRefusal::Counterexamples(self.false_pairs));
        }
        if self.unknown_pairs != 0 {
            return Err(ClaimRefusal::Unknown(self.unknown_pairs));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimRefusal {
    DifferentAxes,
    /// The selection is three-valued; Gamma selects or it does not.
    SelectionNotTwoValued,
    /// Gamma and R were not bound to the same issue of the same frame.
    DifferentFrame,
    /// The frame they share is no longer the store's current seal.
    StaleFrame,
    /// Gamma and R read different windows of the frame.
    DifferentFocus,
    Vacuous,
    Unknown(u32),
    Counterexamples(u32),
}

/// Reference checker over the selection Gamma and the relation R:
/// 1. Gamma: which DR pairs satisfy the candidate constraints (two-valued);
/// 2. R: which DR pairs satisfy the proposed relational law, with R's own
///    validity bits when R's method is three-valued. UNKNOWN is never FALSE.
///
/// It compares the bound axes, that both operands share one current frame
/// issue and one focus. It does not verify that the source exists or that its
/// relation facts are true.
pub fn observe_focus(
    store: &FrameStore,
    gamma: &BoundDr64<'_>,
    relation: &BoundDr64<'_>,
) -> Result<PopulationReport, ClaimRefusal> {
    if (gamma.method.row, gamma.method.column) != (relation.method.row, relation.method.column) {
        return Err(ClaimRefusal::DifferentAxes);
    }
    if gamma.method.truth != Truth::TwoValued {
        return Err(ClaimRefusal::SelectionNotTwoValued);
    }
    if !gamma.frame.same_issue(&relation.frame) {
        return Err(ClaimRefusal::DifferentFrame);
    }
    if store.check(&gamma.frame).is_err() {
        return Err(ClaimRefusal::StaleFrame);
    }
    if gamma.focus != relation.focus {
        return Err(ClaimRefusal::DifferentFocus);
    }
    let mut report = PopulationReport {
        eligible: 0,
        true_pairs: 0,
        false_pairs: 0,
        unknown_pairs: 0,
    };
    for i in 0..64 {
        let focus = gamma.rows[i];
        let known = relation.validity.map_or(u64::MAX, |v| v[i]);
        let positive = focus & relation.rows[i] & known;
        let negative = focus & !relation.rows[i] & known;
        let unknown = focus & !known;
        report.eligible += focus.count_ones();
        report.true_pairs += positive.count_ones();
        report.false_pairs += negative.count_ones();
        report.unknown_pairs += unknown.count_ones();
    }
    debug_assert_eq!(
        report.eligible,
        report.true_pairs + report.false_pairs + report.unknown_pairs
    );
    Ok(report)
}
