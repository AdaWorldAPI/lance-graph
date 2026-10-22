//! Stage D (lab): pre-certified mathematical patterns with fallback.
//!
//! Egglog is invoked ONLY when a candidate is cold-certified/promoted.
//! The lookup path is a BTreeMap + exact semantic/target/binding guard + an
//! already-lowered R2IL scalar program. Every miss/refusal calls the existing
//! caller-supplied trusted baseline. This is NOT a production cache, data
//! attestation, benchmark framework, or background task scheduler.

use std::collections::BTreeMap;

use crate::artifact::{ReplayCapsule, ReplayRefusal};
use crate::{LowerRefusal, Problem, oracle};

/// Full source expression signature. Deliberately preserve premise order,
/// coefficient, and variable names; no hashes or accidentally merged laws.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SemanticKey {
    goal: String,
    target: String,
    premises: Vec<(String, i64, String)>,
}

impl From<&Problem> for SemanticKey {
    fn from(p: &Problem) -> Self {
        Self {
            goal: p.goal.to_owned(),
            target: p.target_var.to_owned(),
            premises: p
                .premises
                .iter()
                .map(|e| (e.lhs.to_owned(), e.coeff, e.rhs.to_owned()))
                .collect(),
        }
    }
}

/// Caller-supplied lab witness. A production owner must independently attest
/// these values from the sealed source and coordinates before any cache use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingWitness {
    pub source_generation: u64,
    pub scope: u64,
    pub coordinate_generation: u64,
}

/// Plan compatibility, independent of mathematical identity and bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetAbi {
    pub revision: u32,
    pub target: String,
}

/// All times are supplied from external measurements; none are inferred from
/// AST size or fabricated by the catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostEvidence {
    pub baseline_ns: u64,
    pub candidate_ns: u64,
    pub compile_and_certify_ns: u64,
    pub expected_reuse: u64,
}

impl CostEvidence {
    pub fn pays_for_itself(&self) -> bool {
        self.candidate_ns < self.baseline_ns
            && self.expected_reuse > 0
            && u128::from(self.baseline_ns - self.candidate_ns) * u128::from(self.expected_reuse)
                > u128::from(self.compile_and_certify_ns)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionRefusal {
    Uncertified(ReplayRefusal),
    NotAmortized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    CacheMiss,
    BindingChanged,
    TargetChanged,
    CompiledRefused,
    /// The input lies outside [`CheckedRange`]: the eager checked baseline
    /// may refuse there, and a certified hit would have answered instead.
    OutsideCheckedRange,
}

/// The inputs on which the eager checked baseline cannot overflow.
///
/// An exact-integer proof says the compiled program and the premises AGREE
/// as mathematics. It says nothing about the baseline's checked `i64`
/// intermediates: `a = -1*c; z = -1*a` compiles to `1*c`, which answers
/// `i64::MIN` where the baseline overflows computing `a`. Every premise is
/// linear in the target, so each intermediate is `k * x` for a known `k`,
/// and the set of `x` where every intermediate fits `i64` is one interval.
/// Inside it the certified hit and the baseline have the same outcome;
/// outside it the catalogue runs the baseline.
///
/// The interval covers the eager baseline that evaluates EVERY premise. A
/// caller whose baseline computes something else must not use the catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedRange {
    pub lo: i64,
    pub hi: i64,
}

impl CheckedRange {
    fn of(problem: &Problem) -> Result<Self, ReplayRefusal> {
        let coeff = oracle::coefficients(problem).map_err(|_| ReplayRefusal::InvalidPremises)?;
        let (mut lo, mut hi) = (i128::from(i64::MIN), i128::from(i64::MAX));
        for &k in coeff.values() {
            if k == 0 {
                continue;
            }
            // MIN <= k * x <= MAX, solved for x.
            let (a, b) = (
                div_ceil(i128::from(i64::MIN), k),
                div_floor(i128::from(i64::MAX), k),
            );
            let (a, b) = if k > 0 {
                (a, b)
            } else {
                (
                    div_ceil(i128::from(i64::MAX), k),
                    div_floor(i128::from(i64::MIN), k),
                )
            };
            lo = lo.max(a);
            hi = hi.min(b);
        }
        // 0 is always inside (k * 0 = 0), so the interval is never empty.
        Ok(Self {
            lo: i64::try_from(lo).map_err(|_| ReplayRefusal::InvalidPremises)?,
            hi: i64::try_from(hi).map_err(|_| ReplayRefusal::InvalidPremises)?,
        })
    }

    pub fn contains(&self, x: i64) -> bool {
        self.lo <= x && x <= self.hi
    }
}

fn div_floor(a: i128, b: i128) -> i128 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

fn div_ceil(a: i128, b: i128) -> i128 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) == (b < 0)) {
        q + 1
    } else {
        q
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Certified,
    Baseline(FallbackReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Execution {
    pub route: Route,
    pub result: Result<i64, LowerRefusal>,
}

struct Entry {
    capsule: ReplayCapsule,
    range: CheckedRange,
    binding: BindingWitness,
    abi: TargetAbi,
    _cost: CostEvidence,
}

/// In-memory lab catalogue; never loads an unverified disk blob.
#[derive(Default)]
pub struct Catalogue {
    entries: BTreeMap<SemanticKey, Entry>,
}

impl Catalogue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cold gate. A proof, replay, independent i128 oracle and supplied
    /// break-even evidence are required before inserting executable steps.
    /// This retains the existing Stage B exact-integer / E1-only restriction.
    pub fn promote(
        &mut self,
        capsule: ReplayCapsule,
        binding: BindingWitness,
        abi: TargetAbi,
        cost: CostEvidence,
    ) -> Result<(), PromotionRefusal> {
        capsule.recheck().map_err(PromotionRefusal::Uncertified)?;
        for x in [0_i64, 1, -1, 7] {
            capsule.replay(x).map_err(PromotionRefusal::Uncertified)?;
        }
        if !cost.pays_for_itself() {
            return Err(PromotionRefusal::NotAmortized);
        }
        let range = CheckedRange::of(capsule.problem()).map_err(PromotionRefusal::Uncertified)?;
        self.entries.insert(
            SemanticKey::from(capsule.problem()),
            Entry {
                capsule,
                range,
                binding,
                abi,
                _cost: cost,
            },
        );
        Ok(())
    }

    /// Hot-compatible lookup and execution: neither egglog nor the i128
    /// oracle runs here. Source/coordinate/target validation is fail-closed,
    /// but caller-supplied witnesses are not source attestation.
    ///
    /// Any miss or compiled execution refusal invokes baseline unconditionally.
    pub fn execute_or<F>(
        &self,
        problem: &Problem,
        binding: BindingWitness,
        abi: &TargetAbi,
        x: i64,
        baseline: F,
    ) -> Execution
    where
        F: FnOnce(i64) -> Result<i64, LowerRefusal>,
    {
        let entry = match self.entries.get(&SemanticKey::from(problem)) {
            Some(e) => e,
            None => {
                return Execution {
                    route: Route::Baseline(FallbackReason::CacheMiss),
                    result: baseline(x),
                };
            }
        };
        let refusal = if entry.binding != binding {
            Some(FallbackReason::BindingChanged)
        } else if &entry.abi != abi {
            Some(FallbackReason::TargetChanged)
        } else if !entry.range.contains(x) {
            Some(FallbackReason::OutsideCheckedRange)
        } else {
            None
        };
        if let Some(reason) = refusal {
            return Execution {
                route: Route::Baseline(reason),
                result: baseline(x),
            };
        }
        match entry.capsule.execute(x) {
            Ok(value) => Execution {
                route: Route::Certified,
                result: Ok(value),
            },
            Err(_) => Execution {
                route: Route::Baseline(FallbackReason::CompiledRefused),
                result: baseline(x),
            },
        }
    }
}
