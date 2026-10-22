//! Stage D (lab): pre-certified mathematical patterns with fallback.
//!
//! Egglog is invoked ONLY when a candidate is cold-certified/promoted.
//! The lookup path is a BTreeMap + exact semantic/target/binding guard + an
//! already-lowered R2IL scalar program. Every miss/refusal calls the existing
//! caller-supplied trusted baseline. This is NOT a production cache, data
//! attestation, benchmark framework, or background task scheduler.

use std::collections::BTreeMap;

use crate::artifact::{ReplayCapsule, ReplayRefusal};
use crate::{LowerRefusal, Problem};

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
        self.entries.insert(
            SemanticKey::from(capsule.problem()),
            Entry {
                capsule,
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
