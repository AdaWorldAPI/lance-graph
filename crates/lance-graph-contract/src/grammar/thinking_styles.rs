//! Grammar thinking styles — meta-inference policies.
//!
//! A grammar style is a **policy** over four axes:
//!
//! 1. **SPO 2³** — which Pearl causal-mask bits to commit to; ambiguity
//!    tolerance.
//! 2. **Morphology** — which case tables to consult, in what order;
//!    agglutinative suffix-peeling on/off.
//! 3. **TEKAMOLO** — slot priority; whether all slots must be fillable.
//! 4. **Markov bundling** — radius (default 5), kernel shape, replay
//!    direction.
//!
//! The static side ([`GrammarStyleConfig`]) is the YAML-loaded prior.
//! The dynamic side ([`GrammarStyleAwareness`]) is the NARS-revised
//! belief over which parameter values work on this style's content.
//! `effective_config(prior, awareness)` composes them at dispatch.
//!
//! ## Permanent / empirical split
//!
//! The signal-profile → NARS-inference dispatch rules are **axiomatic**
//! (permanent logical core). What drifts is the style's **prior over
//! signal-profile frequency** — how often each profile appears on the
//! style's content distribution. Priors revise via NARS on parse
//! outcomes; the dispatch table stays fixed.

use std::collections::HashMap;

use super::context_chain::WeightingKernel;
use super::inference::NarsInference;
use super::tekamolo::TekamoloSlot;
use crate::crystal::TruthValue;
use crate::thinking::ThinkingStyle;

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

/// Primary NARS inference to try, with a fallback when primary fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NarsPriorityChain {
    pub primary: NarsInference,
    pub fallback: NarsInference,
}

/// Which morphology tables to consult, in order; and whether to
/// run agglutinative right-to-left suffix peeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MorphologyTableId {
    EnglishSvo,
    FinnishCase,
    RussianCase,
    GermanCase,
    TurkishAgglutinative,
    JapaneseParticles,
}

#[derive(Debug, Clone)]
pub struct MorphologyPolicy {
    pub tables: Vec<MorphologyTableId>,
    pub agglutinative_mode: bool,
}

/// TEKAMOLO dispatch: which slot priority order + whether the parser
/// requires all attempted slots to be fillable (strict) or tolerates
/// gaps (permissive, e.g. exploratory style).
#[derive(Debug, Clone)]
pub struct TekamoloPolicy {
    pub priority: Vec<TekamoloSlot>,
    pub require_fillable: bool,
}

/// Replay direction across the Markov chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayStrategy {
    Forward,
    Backward,
    BothAndCompare,
}

#[derive(Debug, Clone)]
pub struct MarkovPolicy {
    pub radius: u8,
    pub kernel: WeightingKernel,
    pub replay: ReplayStrategy,
}

/// SPO 2³ causal-mask policy: which bits to commit and how much
/// mask-level ambiguity is tolerated before escalating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpoCausalPolicy {
    pub pearl_mask: u8,
    pub ambiguity_tolerance: f32,
}

/// Coverage policy — local-parse acceptance threshold + escalate-below
/// threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoveragePolicy {
    pub local_threshold: f32,
    pub escalate_below: f32,
}

// ---------------------------------------------------------------------------
// Static prior (from YAML) + runtime awareness (NARS-revised)
// ---------------------------------------------------------------------------

/// Static prior loaded from YAML. One per thinking style.
#[derive(Debug, Clone)]
pub struct GrammarStyleConfig {
    pub style: ThinkingStyle,
    pub nars: NarsPriorityChain,
    pub morphology: MorphologyPolicy,
    pub tekamolo: TekamoloPolicy,
    pub markov: MarkovPolicy,
    pub spo_causal: SpoCausalPolicy,
    pub coverage: CoveragePolicy,
}

/// Key identifying a single tunable parameter within the style config.
/// Used to index NARS truth in [`GrammarStyleAwareness`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParamKey {
    NarsPrimary(NarsInference),
    MorphologyTable(MorphologyTableId),
    TekamoloSlot(TekamoloSlot),
    MarkovKernel(WeightingKernel),
    SpoCausalMask(u8),
}

/// What happened to a parse dispatched under a given style.
///
/// The variants map to NARS truth deltas; [`revise_truth`] converts an
/// outcome into `(observed_frequency, observed_confidence)` so it can
/// fold into the running truth via the standard revision rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseOutcome {
    /// Local parse succeeded and no LLM consulted.
    LocalSuccess,
    /// Local parse succeeded and LLM later confirmed.
    LocalSuccessConfirmedByLLM,
    /// Escalated to LLM and LLM agreed with the local interpretation.
    EscalatedButLLMAgreed,
    /// Escalated and LLM disagreed.
    EscalatedAndLLMDisagreed,
    /// Local parse failed, LLM succeeded — hardest negative signal.
    LocalFailureLLMSucceeded,
}

impl ParseOutcome {
    /// Convert to NARS-revision observation `(f_obs, c_obs)`.
    /// `f_obs` ∈ {0, 1} polarity; `c_obs` weights the update.
    pub fn observation(self) -> (f32, f32) {
        match self {
            Self::LocalSuccess                => (1.0, 1.0),
            Self::LocalSuccessConfirmedByLLM  => (1.0, 2.0),
            Self::EscalatedButLLMAgreed       => (1.0, 0.5),
            Self::EscalatedAndLLMDisagreed    => (0.0, 1.0),
            Self::LocalFailureLLMSucceeded    => (0.0, 2.0),
        }
    }
}

/// Runtime NARS-revised awareness — the style's track record.
#[derive(Debug, Clone)]
pub struct GrammarStyleAwareness {
    pub style: ThinkingStyle,
    pub param_truths: HashMap<ParamKey, TruthValue>,
    pub recent_success: TruthValue,
    pub parse_count: u64,
}

impl GrammarStyleAwareness {
    /// Fresh awareness with neutral priors (f=0.5, c=0.01) and zero parses.
    pub fn bootstrap(style: ThinkingStyle) -> Self {
        Self {
            style,
            param_truths: HashMap::new(),
            recent_success: TruthValue::new(0.5, 0.01),
            parse_count: 0,
        }
    }

    /// Revise the truth value for `key` under the observed `outcome`,
    /// and fold the same observation into `recent_success`.
    pub fn revise(&mut self, key: ParamKey, outcome: ParseOutcome) {
        let (f_obs, c_obs) = outcome.observation();

        let current = self
            .param_truths
            .get(&key)
            .copied()
            .unwrap_or(TruthValue::new(0.5, 0.01));
        let revised = revise_truth(current, f_obs, c_obs);
        self.param_truths.insert(key, revised);

        self.recent_success = revise_truth(self.recent_success, f_obs, c_obs);
        self.parse_count += 1;
    }

    /// Best NARS inference given current awareness — either the YAML
    /// primary (if its truth is healthy) or the highest-ranked NARS
    /// parameter we've accumulated evidence for.
    pub fn top_nars_inference(&self, prior: &GrammarStyleConfig) -> NarsInference {
        let primary_key = ParamKey::NarsPrimary(prior.nars.primary);
        let primary_truth = self
            .param_truths
            .get(&primary_key)
            .copied()
            .unwrap_or(TruthValue::new(0.5, 0.01));

        // If the primary still looks healthy (f > 0.5 AND any confidence),
        // keep using it — awareness has not yet contradicted the prior.
        if primary_truth.frequency > 0.5 {
            return prior.nars.primary;
        }

        // Otherwise pick the NARS parameter with the highest expected
        // value (frequency × confidence). Ties resolved by iteration
        // order over a stable sort.
        let mut ranked: Vec<(NarsInference, f32)> = self
            .param_truths
            .iter()
            .filter_map(|(k, t)| match k {
                ParamKey::NarsPrimary(inf) => Some((*inf, t.frequency * t.confidence)),
                _ => None,
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.first().map(|(inf, _)| *inf).unwrap_or(prior.nars.fallback)
    }

    /// Derive a runtime config from the prior + accumulated awareness.
    /// Mutations are small (we don't rebuild tables); the effective
    /// primary NARS inference is swapped when awareness has pulled
    /// the prior under 0.5.
    pub fn effective_config(&self, prior: &GrammarStyleConfig) -> GrammarStyleConfig {
        let effective_primary = self.top_nars_inference(prior);
        GrammarStyleConfig {
            style: prior.style,
            nars: NarsPriorityChain {
                primary: effective_primary,
                fallback: prior.nars.fallback,
            },
            morphology: prior.morphology.clone(),
            tekamolo: prior.tekamolo.clone(),
            markov: prior.markov.clone(),
            spo_causal: prior.spo_causal,
            coverage: prior.coverage,
        }
    }
}

// ---------------------------------------------------------------------------
// NARS revision rule
// ---------------------------------------------------------------------------

/// Standard NARS revision:
/// - `f_new = (f_old · c_old + f_obs · c_obs) / (c_old + c_obs)`
/// - `c_new = (c_old + c_obs) / (c_old + c_obs + 1)`
///
/// Confidence stays in [0, 1); frequency stays in [0, 1]. Observed
/// confidence `c_obs` is the observation's weight (not a raw count).
pub fn revise_truth(current: TruthValue, f_obs: f32, c_obs: f32) -> TruthValue {
    let c_old = current.confidence.max(0.0);
    let c_obs = c_obs.max(0.0);
    let denom = c_old + c_obs;
    if denom <= 0.0 {
        return current;
    }
    let f_new = (current.frequency * c_old + f_obs * c_obs) / denom;
    let c_new = denom / (denom + 1.0);
    TruthValue::new(f_new.clamp(0.0, 1.0), c_new.clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base_prior() -> GrammarStyleConfig {
        GrammarStyleConfig {
            style: ThinkingStyle::Analytical,
            nars: NarsPriorityChain {
                primary: NarsInference::Deduction,
                fallback: NarsInference::Abduction,
            },
            morphology: MorphologyPolicy {
                tables: vec![
                    MorphologyTableId::EnglishSvo,
                    MorphologyTableId::FinnishCase,
                ],
                agglutinative_mode: false,
            },
            tekamolo: TekamoloPolicy {
                priority: vec![
                    TekamoloSlot::Temporal,
                    TekamoloSlot::Lokal,
                    TekamoloSlot::Kausal,
                    TekamoloSlot::Modal,
                ],
                require_fillable: true,
            },
            markov: MarkovPolicy {
                radius: 5,
                kernel: WeightingKernel::Uniform,
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

    #[test]
    fn bootstrap_is_neutral() {
        let a = GrammarStyleAwareness::bootstrap(ThinkingStyle::Analytical);
        assert_eq!(a.parse_count, 0);
        assert!(a.param_truths.is_empty());
        assert!((a.recent_success.frequency - 0.5).abs() < 1e-6);
        assert!(a.recent_success.confidence > 0.0);
    }

    #[test]
    fn revision_monotone_on_positive_outcomes() {
        let a = TruthValue::new(0.5, 0.01);
        let b = revise_truth(a, 1.0, 1.0);
        let c = revise_truth(b, 1.0, 1.0);
        let d = revise_truth(c, 1.0, 1.0);
        assert!(
            b.frequency < c.frequency && c.frequency < d.frequency,
            "frequency should monotonically rise on repeated positives: {}, {}, {}",
            b.frequency, c.frequency, d.frequency
        );
        assert!(b.confidence < c.confidence && c.confidence < d.confidence);
        assert!(d.confidence < 1.0, "confidence must stay below 1");
    }

    #[test]
    fn revision_frequency_falls_on_negatives_confidence_still_rises() {
        let a = TruthValue::new(0.5, 0.01);
        let b = revise_truth(a, 0.0, 1.0);
        let c = revise_truth(b, 0.0, 1.0);
        assert!(
            b.frequency > c.frequency,
            "f should fall: {} -> {}",
            b.frequency, c.frequency
        );
        assert!(
            b.confidence < c.confidence,
            "c should rise regardless of polarity: {} -> {}",
            b.confidence, c.confidence
        );
    }

    #[test]
    fn awareness_drifts_primary_nars_on_50_bad_outcomes() {
        let prior = base_prior();
        let mut a = GrammarStyleAwareness::bootstrap(prior.style);
        // 50 opposing outcomes on Deduction → primary truth falls below 0.5.
        for _ in 0..50 {
            a.revise(
                ParamKey::NarsPrimary(NarsInference::Deduction),
                ParseOutcome::LocalFailureLLMSucceeded,
            );
            a.revise(
                ParamKey::NarsPrimary(NarsInference::Abduction),
                ParseOutcome::LocalSuccessConfirmedByLLM,
            );
        }
        let eff = a.effective_config(&prior);
        assert_eq!(
            eff.nars.primary,
            NarsInference::Abduction,
            "effective primary must drift away from a saturating-fail Deduction to Abduction"
        );
    }

    #[test]
    fn awareness_keeps_primary_on_50_good_outcomes() {
        let prior = base_prior();
        let mut a = GrammarStyleAwareness::bootstrap(prior.style);
        for _ in 0..50 {
            a.revise(
                ParamKey::NarsPrimary(NarsInference::Deduction),
                ParseOutcome::LocalSuccess,
            );
        }
        let eff = a.effective_config(&prior);
        assert_eq!(eff.nars.primary, NarsInference::Deduction);
        // Recent success confidence should be well above the low-conf line.
        assert!(
            a.recent_success.frequency > 0.9,
            "recent_success.frequency should saturate on positives, got {}",
            a.recent_success.frequency
        );
    }

    #[test]
    fn recent_success_confidence_saturates_under_revision() {
        // After many revisions — regardless of polarity mix — confidence
        // must rise toward 1.0. Callers gate on high-confidence-low-
        // frequency as the "style has lost grip on this profile" signal.
        let prior = base_prior();
        let mut a = GrammarStyleAwareness::bootstrap(prior.style);
        let c0 = a.recent_success.confidence;
        for i in 0..50 {
            let outcome = if i % 2 == 0 {
                ParseOutcome::LocalSuccess
            } else {
                ParseOutcome::EscalatedAndLLMDisagreed
            };
            a.revise(ParamKey::NarsPrimary(NarsInference::Deduction), outcome);
        }
        // NARS revision with c_obs = 1.0 per step asymptotes at φ - 1 ≈ 0.618
        // (solving x = (x+1)/(x+2)). So `c > 0.6` is the saturation signal.
        assert!(
            a.recent_success.confidence > 0.6,
            "50 revisions at c_obs=1 should saturate confidence near 0.618, got {} (bootstrap was {})",
            a.recent_success.confidence, c0
        );
        assert_eq!(a.parse_count, 50);
    }

    #[test]
    fn param_truths_keyed_distinctly() {
        let mut a = GrammarStyleAwareness::bootstrap(ThinkingStyle::Exploratory);
        a.revise(
            ParamKey::NarsPrimary(NarsInference::Deduction),
            ParseOutcome::LocalSuccess,
        );
        a.revise(
            ParamKey::NarsPrimary(NarsInference::Abduction),
            ParseOutcome::LocalFailureLLMSucceeded,
        );
        a.revise(
            ParamKey::MorphologyTable(MorphologyTableId::FinnishCase),
            ParseOutcome::LocalSuccessConfirmedByLLM,
        );
        a.revise(
            ParamKey::MarkovKernel(WeightingKernel::MexicanHat),
            ParseOutcome::LocalSuccess,
        );
        // Four distinct keys, four distinct entries.
        assert_eq!(a.param_truths.len(), 4);
    }

    #[test]
    fn observation_polarity_matches_intent() {
        // Success outcomes emit f=1.0, failure outcomes emit f=0.0.
        assert_eq!(ParseOutcome::LocalSuccess.observation().0, 1.0);
        assert_eq!(
            ParseOutcome::LocalSuccessConfirmedByLLM.observation().0,
            1.0
        );
        assert_eq!(ParseOutcome::EscalatedButLLMAgreed.observation().0, 1.0);
        assert_eq!(
            ParseOutcome::EscalatedAndLLMDisagreed.observation().0,
            0.0
        );
        assert_eq!(
            ParseOutcome::LocalFailureLLMSucceeded.observation().0,
            0.0
        );
        // Strong negatives and confirmations carry double weight.
        assert_eq!(ParseOutcome::LocalSuccessConfirmedByLLM.observation().1, 2.0);
        assert_eq!(ParseOutcome::LocalFailureLLMSucceeded.observation().1, 2.0);
    }

    #[test]
    fn effective_config_clones_collections_without_mutating_prior() {
        let prior = base_prior();
        let prior_tables_len = prior.morphology.tables.len();
        let a = GrammarStyleAwareness::bootstrap(prior.style);
        let eff = a.effective_config(&prior);
        assert_eq!(eff.morphology.tables.len(), prior_tables_len);
        assert_eq!(prior.morphology.tables.len(), prior_tables_len);
    }
}
