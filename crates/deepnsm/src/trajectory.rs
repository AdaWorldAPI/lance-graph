//! Trajectory — the Think carrier that speaks for itself.
//!
//! Holds a Markov-braided role-indexed bundle + references to thinking
//! tissue (episodic memory, triplet graph, global context). Methods
//! on Trajectory compute free energy, resolve ambiguity, and observe
//! outcomes — the object IS the inference engine.

use lance_graph_contract::grammar::role_keys::{
    Vsa10k, VSA_ZERO, VSA_WORDS, RoleKey, vsa_xor, vsa_similarity,
    SUBJECT_KEY, PREDICATE_KEY, OBJECT_KEY, MODIFIER_KEY,
    TEMPORAL_KEY, KAUSAL_KEY, MODAL_KEY, LOKAL_KEY, INSTRUMENT_KEY,
};
use lance_graph_contract::grammar::free_energy::{
    FreeEnergy, Hypothesis, Resolution,
};
use lance_graph_contract::grammar::thinking_styles::{
    GrammarStyleAwareness, GrammarStyleConfig, ParamKey, ParseOutcome,
};
use lance_graph_contract::grammar::ticket::FailureTicket;

/// A resolved Markov ±5 trajectory with tissue references.
///
/// This is the Think struct from The Click (CLAUDE.md § P-1):
/// trajectory = Subject (what), awareness = Modal (how confidently),
/// free_energy = Kausal (why this thought), resolution = Predicate
/// (what it concludes), global_context = Lokal (where in fact-space).
pub struct Trajectory {
    pub bundle: Vsa10k,
    pub global_context: Vsa10k,
}

impl Trajectory {
    pub fn new(bundle: Vsa10k, global_context: Vsa10k) -> Self {
        Self { bundle, global_context }
    }

    /// Unbind a single role from the trajectory bundle.
    /// Returns the content that was bound into that role's slice.
    pub fn role_bundle(&self, role: &RoleKey) -> Vsa10k {
        role.unbind(&self.bundle)
    }

    /// Mean recovery margin across a set of role keys, comparing
    /// the trajectory's unbound content against hypothesis fillers.
    ///
    /// This IS the likelihood term in the free-energy decomposition:
    /// "how well do the hypothesized role fillers match what the
    /// trajectory actually carries?"
    pub fn mean_recovery_margin(&self, hypothesis: &Hypothesis) -> f32 {
        if hypothesis.role_fillers.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f32;
        let mut count = 0u32;
        for (role_label, _filler_label) in &hypothesis.role_fillers {
            if let Some(key) = label_to_key(role_label) {
                let unbound = key.unbind(&self.bundle);
                // Use the filler's hash as the expected content fingerprint.
                let expected = crate::content_fp::content_fp(
                    filler_rank_from_label(_filler_label),
                );
                let m = key.recovery_margin(&unbound, &expected);
                total += m;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { total / count as f32 }
    }

    /// Recovery margin against the global context (ambient prior).
    /// Measures how well this trajectory's content aligns with the
    /// accumulated story-so-far.
    pub fn ambient_similarity(&self) -> f32 {
        vsa_similarity(&self.bundle, &self.global_context)
    }

    /// Compute free energy for a hypothesis against this trajectory.
    pub fn free_energy(
        &self,
        hypothesis: &Hypothesis,
        awareness: &GrammarStyleAwareness,
        prior: &GrammarStyleConfig,
    ) -> FreeEnergy {
        let local_likelihood = self.mean_recovery_margin(hypothesis);
        let ambient = self.ambient_similarity().max(0.0);
        let likelihood = 0.7 * local_likelihood + 0.3 * ambient;
        let kl = awareness.divergence_from(prior);
        FreeEnergy::compose(likelihood, kl)
    }

    /// Score and rank hypotheses, returning the resolution.
    pub fn resolve(
        &self,
        candidates: Vec<Hypothesis>,
        awareness: &GrammarStyleAwareness,
        prior: &GrammarStyleConfig,
        ticket_factory: impl FnOnce() -> FailureTicket,
    ) -> Resolution {
        let mut ranked: Vec<(Hypothesis, FreeEnergy)> = candidates
            .into_iter()
            .map(|h| {
                let fe = self.free_energy(&h, awareness, prior);
                (h, fe)
            })
            .collect();
        ranked.sort_by(|a, b| {
            a.1.total
                .partial_cmp(&b.1.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Resolution::from_ranked(&ranked, ticket_factory)
    }
}

/// Map a role label string to the corresponding static RoleKey.
fn label_to_key(label: &str) -> Option<&'static RoleKey> {
    match label {
        "SUBJECT" => Some(&*SUBJECT_KEY),
        "PREDICATE" => Some(&*PREDICATE_KEY),
        "OBJECT" => Some(&*OBJECT_KEY),
        "MODIFIER" => Some(&*MODIFIER_KEY),
        "TEMPORAL" => Some(&*TEMPORAL_KEY),
        "KAUSAL" => Some(&*KAUSAL_KEY),
        "MODAL" => Some(&*MODAL_KEY),
        "LOKAL" => Some(&*LOKAL_KEY),
        "INSTRUMENT" => Some(&*INSTRUMENT_KEY),
        _ => None,
    }
}

/// Derive a vocabulary rank from a filler label (for test/stub use).
/// Real pipeline will carry actual ranks, not labels.
fn filler_rank_from_label(label: &str) -> u16 {
    let mut h: u32 = 0;
    for b in label.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    (h % 4096) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content_fp::content_fp;
    use crate::markov_bundle::{MarkovBundler, encode_sentence};
    use crate::parser::SentenceStructure;
    use crate::spo::SpoTriple;
    use lance_graph_contract::grammar::context_chain::WeightingKernel;
    use lance_graph_contract::grammar::thinking_styles::{
        NarsPriorityChain, MorphologyPolicy, MorphologyTableId,
        TekamoloPolicy, MarkovPolicy, ReplayStrategy,
        SpoCausalPolicy, CoveragePolicy,
    };
    use lance_graph_contract::grammar::inference::NarsInference;
    use lance_graph_contract::grammar::tekamolo::TekamoloSlot;
    use lance_graph_contract::thinking::ThinkingStyle;

    fn test_prior() -> GrammarStyleConfig {
        GrammarStyleConfig {
            style: ThinkingStyle::Analytical,
            nars: NarsPriorityChain {
                primary: NarsInference::Deduction,
                fallback: NarsInference::Abduction,
            },
            morphology: MorphologyPolicy {
                tables: vec![MorphologyTableId::EnglishSvo],
                agglutinative_mode: false,
            },
            tekamolo: TekamoloPolicy {
                priority: vec![TekamoloSlot::Temporal, TekamoloSlot::Lokal],
                require_fillable: true,
            },
            markov: MarkovPolicy {
                radius: 5,
                kernel: WeightingKernel::MexicanHat,
                replay: ReplayStrategy::Forward,
            },
            spo_causal: SpoCausalPolicy {
                pearl_mask: 0x01,
                ambiguity_tolerance: 0.1,
            },
            coverage: CoveragePolicy {
                local_threshold: 0.90,
                escalate_below: 0.85,
            },
        }
    }

    fn mk_sentence(s: u16, p: u16, o: u16) -> SentenceStructure {
        SentenceStructure {
            triples: vec![SpoTriple::new(s, p, o)],
            modifiers: vec![],
            negations: vec![],
            temporals: vec![],
        }
    }

    #[test]
    fn trajectory_role_unbind_recovers_subject() {
        let sentence = mk_sentence(42, 100, 200);
        let bundle = encode_sentence(&sentence);
        let trajectory = Trajectory::new(bundle, VSA_ZERO);
        let recovered = trajectory.role_bundle(&SUBJECT_KEY);
        let expected = content_fp(42);
        let margin = SUBJECT_KEY.recovery_margin(&recovered, &expected);
        assert!(
            margin > 0.99,
            "SUBJECT should recover from trajectory, got {margin}"
        );
    }

    #[test]
    fn trajectory_free_energy_lower_for_correct_hypothesis() {
        let sentence = mk_sentence(42, 100, 200);
        let bundle = encode_sentence(&sentence);
        let trajectory = Trajectory::new(bundle, VSA_ZERO);

        let prior = test_prior();
        let awareness = GrammarStyleAwareness::bootstrap(prior.style);

        // Correct hypothesis: subject=42, predicate=100, object=200
        let correct = Hypothesis::new("correct")
            .fill("SUBJECT", "42")
            .fill("PREDICATE", "100")
            .fill("OBJECT", "200");

        // Wrong hypothesis: subject=999
        let wrong = Hypothesis::new("wrong")
            .fill("SUBJECT", "999")
            .fill("PREDICATE", "888")
            .fill("OBJECT", "777");

        let f_correct = trajectory.free_energy(&correct, &awareness, &prior);
        let f_wrong = trajectory.free_energy(&wrong, &awareness, &prior);

        assert!(
            f_correct.total < f_wrong.total,
            "correct hypothesis should have lower F ({}) than wrong ({})",
            f_correct.total, f_wrong.total
        );
    }

    #[test]
    fn trajectory_resolve_commits_best_hypothesis() {
        let sentence = mk_sentence(42, 100, 200);
        let bundle = encode_sentence(&sentence);
        let trajectory = Trajectory::new(bundle, VSA_ZERO);

        let prior = test_prior();
        let awareness = GrammarStyleAwareness::bootstrap(prior.style);

        let correct = Hypothesis::new("correct")
            .fill("SUBJECT", "42");
        let wrong = Hypothesis::new("wrong")
            .fill("SUBJECT", "999");

        let resolution = trajectory.resolve(
            vec![wrong, correct],
            &awareness,
            &prior,
            || panic!("should not create failure ticket"),
        );

        match resolution {
            Resolution::Commit { hypothesis, .. } => {
                assert_eq!(hypothesis.label, "correct");
            }
            Resolution::Epiphany { winner, .. } => {
                assert_eq!(winner.label, "correct");
            }
            Resolution::FailureTicket(_) => {
                panic!("should not escalate — correct hypothesis has recoverable content");
            }
        }
    }

    #[test]
    fn bundled_trajectory_through_markov_bundler() {
        let mut bundler = MarkovBundler::new(WeightingKernel::MexicanHat);
        for i in 0..11 {
            bundler.push(&mk_sentence(i + 10, i + 100, i + 200));
        }
        let bundle = bundler.build_bundle();
        let trajectory = Trajectory::new(bundle, VSA_ZERO);

        // The focal sentence (last pushed: s=20, p=110, o=210).
        // Its SUBJECT should still be recoverable despite braiding.
        let recovered = trajectory.role_bundle(&SUBJECT_KEY);
        let expected = content_fp(20);
        let margin = SUBJECT_KEY.recovery_margin(&recovered, &expected);
        // After 11-way braided superposition, recovery is approximate.
        // MexicanHat weights the focal heavily so margin should be > 0.5.
        assert!(
            margin > 0.5,
            "focal SUBJECT should be recoverable from braided bundle, got {margin}"
        );
    }
}
