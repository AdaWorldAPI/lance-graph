//! Deterministic ACT-R-inspired production arbitration.
//!
//! The caller evaluates production conditions through relalg/R2IL/Quack/Mask-RISC.
//! This crate never interprets expressions, reads storage, or executes effects.
//! It chooses an eligible rule and returns a witnessed *proposal*.
//! Not a complete ACT-R implementation (no buffer timing or learning yet).

/// Address of the observed cognitive state. These local types are an experimental
/// boundary; replace with canonical lance-graph contract types before integration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub thought_address: u128,
    pub witness_version: u64,
}

/// Rule eligibility is supplied by the sole external masking evaluator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub rule_id: u64,
    pub eligible: bool,
    /// Expected learned usefulness of the procedure, not epistemic truth.
    pub utility: f32,
    /// ACT-R-inspired accessibility, not evidence or permission.
    pub activation: f32,
}

/// An intent is not an authorized effect or committed state transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub snapshot: Snapshot,
    pub rule_id: u64,
    pub utility: f32,
    pub activation: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NoEligibleProduction,
    InvalidScore,
}

/// Deterministic arbitration: maximize utility, then activation, then prefer
/// smaller rule ID. NaN and infinity are refused rather than silently ordered.
/// No hidden RNG, external LLM, SQL or JSON enters this path.
pub fn arbitrate(snapshot: Snapshot, candidates: &[Candidate]) -> Result<Decision, Refusal> {
    let mut best: Option<Candidate> = None;
    for &candidate in candidates.iter().filter(|c| c.eligible) {
        if !candidate.utility.is_finite() || !candidate.activation.is_finite() {
            return Err(Refusal::InvalidScore);
        }
        let replace = match best {
            None => true,
            Some(previous) => {
                candidate.utility > previous.utility
                    || (candidate.utility == previous.utility
                        && (candidate.activation > previous.activation
                            || (candidate.activation == previous.activation
                                && candidate.rule_id < previous.rule_id)))
            }
        };
        if replace {
            best = Some(candidate);
        }
    }
    let selected = best.ok_or(Refusal::NoEligibleProduction)?;
    Ok(Decision {
        snapshot,
        rule_id: selected.rule_id,
        utility: selected.utility,
        activation: selected.activation,
    })
}

/// Only checks that a proposal still refers to the observed snapshot.
/// The caller must independently authorize and atomically commit any effect.
pub fn witness_matches(decision: &Decision, current: Snapshot) -> bool {
    decision.snapshot == current
}

#[cfg(test)]
mod tests {
    use super::*;
    const STATE: Snapshot = Snapshot { thought_address: 42, witness_version: 7 };
    fn candidate(rule_id: u64, eligible: bool, utility: f32, activation: f32) -> Candidate {
        Candidate { rule_id, eligible, utility, activation }
    }

    #[test]
    fn eligible_utility_beats_ineligible_high_score() {
        let result = arbitrate(STATE, &[
            candidate(1, false, 100.0, 100.0),
            candidate(2, true, 0.7, 0.3),
        ]).unwrap();
        assert_eq!(result.rule_id, 2);
    }

    #[test]
    fn tie_break_is_independent_of_candidate_order() {
        let a = candidate(3, true, 0.7, 0.5);
        let b = candidate(2, true, 0.7, 0.5);
        assert_eq!(arbitrate(STATE, &[a, b]).unwrap().rule_id, 2);
        assert_eq!(arbitrate(STATE, &[b, a]).unwrap().rule_id, 2);
    }

    #[test]
    fn activation_is_secondary_to_utility() {
        assert_eq!(arbitrate(STATE, &[
            candidate(1, true, 0.8, 0.1),
            candidate(2, true, 0.7, 0.9),
        ]).unwrap().rule_id, 1);
    }

    #[test]
    fn refuses_non_finite_eligible_scores() {
        assert_eq!(arbitrate(STATE, &[candidate(1, true, f32::NAN, 0.0)]),
            Err(Refusal::InvalidScore));
    }

    #[test]
    fn no_eligible_production_is_explicit() {
        assert_eq!(arbitrate(STATE, &[candidate(1, false, 1.0, 1.0)]),
            Err(Refusal::NoEligibleProduction));
    }

    #[test]
    fn stale_witness_cannot_pass_check() {
        let decision = arbitrate(STATE, &[candidate(1, true, 0.5, 0.5)]).unwrap();
        assert!(witness_matches(&decision, STATE));
        assert!(!witness_matches(&decision,
            Snapshot { thought_address: 42, witness_version: 8 }));
        assert!(!witness_matches(&decision,
            Snapshot { thought_address: 43, witness_version: 7 }));
    }
}
