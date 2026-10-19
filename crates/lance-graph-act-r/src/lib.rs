//! Deterministic ACT-R production arbitration.
//!
//! ACT-R is the cognitive architecture *Adaptive Control of Thought, Rational*.
//! This crate is not an actor or mailbox framework.
//!
//! The caller evaluates production conditions through relalg/R2IL/Quack/Mask-RISC.
//! This crate never interprets expressions, reads storage, or executes effects.
//! It chooses an eligible rule and returns a witnessed *proposal*.
//! Not a complete ACT-R implementation (no buffer timing or learning yet).

/// Address of the observed cognitive state. These local types are an experimental
/// boundary; replace with canonical lance-graph contract types before integration.
///
/// Two snapshots are the same state only when both fields agree: a thought
/// address that is reused under a new `witness_version` is a different state.
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
    /// The state the evaluator read when it computed `eligible`.
    /// [`arbitrate`] refuses the batch unless this equals its `snapshot`.
    pub evaluated_at: Snapshot,
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
    /// A candidate was evaluated against another state than the one being
    /// arbitrated. The whole batch is refused: dropping the candidate would
    /// hide a rule whose current eligibility is unknown.
    StaleCandidate {
        rule_id: u64,
        evaluated_at: Snapshot,
    },
}

/// Deterministic arbitration: maximize utility, then activation, then prefer
/// smaller rule ID. NaN and infinity are refused rather than silently ordered.
/// No hidden RNG, external LLM, SQL or JSON enters this path.
///
/// Every candidate, eligible or not, must have been evaluated at `snapshot`;
/// otherwise the batch is refused with [`Refusal::StaleCandidate`] before any
/// selection, so a decision is never stamped with a state its eligibility was
/// not computed against.
pub fn arbitrate(snapshot: Snapshot, candidates: &[Candidate]) -> Result<Decision, Refusal> {
    if let Some(stale) = candidates.iter().find(|c| c.evaluated_at != snapshot) {
        return Err(Refusal::StaleCandidate {
            rule_id: stale.rule_id,
            evaluated_at: stale.evaluated_at,
        });
    }
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
    const STATE: Snapshot = Snapshot {
        thought_address: 42,
        witness_version: 7,
    };
    const NEXT: Snapshot = Snapshot {
        thought_address: 42,
        witness_version: 8,
    };

    fn candidate(rule_id: u64, eligible: bool, utility: f32, activation: f32) -> Candidate {
        at(STATE, rule_id, eligible, utility, activation)
    }

    fn at(s: Snapshot, rule_id: u64, eligible: bool, utility: f32, activation: f32) -> Candidate {
        Candidate {
            rule_id,
            eligible,
            utility,
            activation,
            evaluated_at: s,
        }
    }

    #[test]
    fn eligible_utility_beats_ineligible_high_score() {
        let result = arbitrate(
            STATE,
            &[
                candidate(1, false, 100.0, 100.0),
                candidate(2, true, 0.7, 0.3),
            ],
        )
        .unwrap();
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
        assert_eq!(
            arbitrate(
                STATE,
                &[candidate(1, true, 0.8, 0.1), candidate(2, true, 0.7, 0.9)]
            )
            .unwrap()
            .rule_id,
            1
        );
    }

    #[test]
    fn refuses_non_finite_eligible_scores() {
        assert_eq!(
            arbitrate(STATE, &[candidate(1, true, f32::NAN, 0.0)]),
            Err(Refusal::InvalidScore)
        );
        assert_eq!(
            arbitrate(STATE, &[candidate(1, true, 0.0, f32::INFINITY)]),
            Err(Refusal::InvalidScore)
        );
    }

    #[test]
    fn no_eligible_production_is_explicit() {
        assert_eq!(
            arbitrate(STATE, &[candidate(1, false, 1.0, 1.0)]),
            Err(Refusal::NoEligibleProduction)
        );
    }

    /// The #1615 defect: eligibility computed at v7, arbitrated at v8.
    /// Before the fix this returned a decision stamped with v8.
    #[test]
    fn stale_candidate_eligibility_is_refused() {
        assert_eq!(
            arbitrate(NEXT, &[candidate(1, true, 0.5, 0.5)]),
            Err(Refusal::StaleCandidate {
                rule_id: 1,
                evaluated_at: STATE
            })
        );
    }

    /// Silence twin: the same candidate at its own state is accepted.
    #[test]
    fn fresh_candidate_is_accepted() {
        let d = arbitrate(NEXT, &[at(NEXT, 1, true, 0.5, 0.5)]).unwrap();
        assert_eq!((d.rule_id, d.snapshot), (1, NEXT));
    }

    /// One stale candidate refuses the whole batch, even when it is ineligible
    /// and would lose: its current eligibility is unknown.
    #[test]
    fn mixed_candidate_generations_are_refused() {
        let batch = [at(NEXT, 1, true, 0.9, 0.9), at(STATE, 2, false, 0.1, 0.1)];
        assert_eq!(
            arbitrate(NEXT, &batch),
            Err(Refusal::StaleCandidate {
                rule_id: 2,
                evaluated_at: STATE
            })
        );
    }

    /// A reused thought address under a new witness version is another state.
    #[test]
    fn recycled_thought_address_is_refused() {
        let recycled = Snapshot {
            thought_address: 42,
            witness_version: 1_000,
        };
        assert!(matches!(
            arbitrate(recycled, &[candidate(1, true, 0.5, 0.5)]),
            Err(Refusal::StaleCandidate { .. })
        ));
        let other_thought = Snapshot {
            thought_address: 43,
            witness_version: 7,
        };
        assert!(matches!(
            arbitrate(other_thought, &[candidate(1, true, 0.5, 0.5)]),
            Err(Refusal::StaleCandidate { .. })
        ));
    }

    #[test]
    fn stale_decision_is_rejected_by_witness_check() {
        let decision = arbitrate(STATE, &[candidate(1, true, 0.5, 0.5)]).unwrap();
        assert!(witness_matches(&decision, STATE));
        assert!(!witness_matches(&decision, NEXT));
        assert!(!witness_matches(
            &decision,
            Snapshot {
                thought_address: 43,
                witness_version: 7
            }
        ));
    }
}
