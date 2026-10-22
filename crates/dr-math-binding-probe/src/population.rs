//! D-DRMC-2: deliberately small, independent FINITE POPULATION audit oracle.
//!
//! NOT a physical runtime executor, graph engine, statistical test, proof of
//! causality, or independent authentication of supplied source data. Reads
//! one 64x64 QR focus to find concrete counterexamples before a law is admitted.
//! Hot production folds remain existing Quack/R2IL/mask-risc/native SIMD.
use crate::{BindingWitness, BoundDr64};

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
        if self.eligible == 0 {
            return Err(ClaimRefusal::Vacuous);
        }
        if self.unknown_pairs != 0 {
            return Err(ClaimRefusal::Unknown(self.unknown_pairs));
        }
        if self.false_pairs != 0 {
            return Err(ClaimRefusal::Counterexamples(self.false_pairs));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimRefusal {
    DifferentAxes,
    ChangedWitness,
    Vacuous,
    Unknown(u32),
    Counterexamples(u32),
}

/// Reference checker over THREE separate boolean planes:
/// 1. Gamma: which DR pairs satisfy candidate constraints;
/// 2. R: which DR pairs satisfy proposed relational law;
/// 3. validity: which R bits are known, None only for explicitly total
///    two-valued observations. UNKNOWN is never treated as FALSE.
///
/// It compares already bound axes and full supplied witnesses. It does not
/// verify that an underlying source exists or that its relation facts are true.
/// `validity` is a separate *physical* carrier and no SQL-NULL encoding in DR.
pub fn observe_focus(
    gamma: &BoundDr64<'_>,
    relation: &BoundDr64<'_>,
    validity: Option<&BoundDr64<'_>>,
    snapshot: BindingWitness,
) -> Result<PopulationReport, ClaimRefusal> {
    let a = (gamma.method.row, gamma.method.column);
    if a != (relation.method.row, relation.method.column)
        || validity.is_some_and(|v| a != (v.method.row, v.method.column))
    {
        return Err(ClaimRefusal::DifferentAxes);
    }
    if gamma.witness != snapshot || relation.witness != snapshot
        || validity.is_some_and(|v| v.witness != snapshot)
    {
        return Err(ClaimRefusal::ChangedWitness);
    }
    let mut report = PopulationReport {
        eligible: 0, true_pairs: 0, false_pairs: 0, unknown_pairs: 0,
    };
    for i in 0..64 {
        let focus = gamma.rows[i];
        let known = validity.map_or(u64::MAX, |v| v.rows[i]);
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
