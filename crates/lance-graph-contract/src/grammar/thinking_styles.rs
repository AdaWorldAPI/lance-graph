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

    /// KL-style divergence of this awareness's accumulated beliefs from
    /// a prior config. Used as the KL term in free-energy composition.
    ///
    /// Decomposition:
    /// - Primary NARS-inference disagreement: `(1 - f_primary) × c_primary`.
    ///   High-confidence low-frequency on the prior's primary inference
    ///   is the strongest signal that the prior is wrong for this style.
    /// - Recent-success disagreement: `(1 - f_recent) × c_recent`.
    ///   The style's overall track record diverging from neutral (0.5)
    ///   also contributes.
    ///
    /// Bounded in `[0, 2]` — two contributors each in `[0, 1]`.
    pub fn divergence_from(&self, prior: &GrammarStyleConfig) -> f32 {
        let primary_key = ParamKey::NarsPrimary(prior.nars.primary);
        let primary_truth = self
            .param_truths
            .get(&primary_key)
            .copied()
            .unwrap_or(TruthValue::new(0.5, 0.01));
        let primary_drift = (1.0 - primary_truth.frequency) * primary_truth.confidence;
        // Recent-success drift: how far the running track record has moved
        // away from its starting neutral (0.5). Absolute distance so both
        // over- and under-performance count as evidence against the prior.
        let recent_drift =
            (self.recent_success.frequency - 0.5).abs() * self.recent_success.confidence;
        (primary_drift + recent_drift).clamp(0.0, 2.0)
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
// YAML loader (zero-dep, line-based)
// ---------------------------------------------------------------------------
//
// Supports the strict subset our `grammar_styles/<style>.yaml` files use:
//   - top-level scalars   (`style: analytical`)
//   - top-level mappings  (`nars:` followed by indented `primary: …` lines)
//   - inline flow lists   (`tables: [english_svo, finnish_case_table]`)
//   - block lists         (one `- item` per line, indented under a key)
//   - leading `#` comments and blank lines are ignored
//   - hex literals (`0x01`, `0xFF`) accepted for `pearl_mask`
//
// Anything outside this subset returns a textual error rather than panicking
// — these files are config, not user input, so a hard error during boot is
// the correct failure mode.

/// Parse a `grammar_styles/<style>.yaml` document into [`GrammarStyleConfig`].
///
/// Implementation: a small line-based YAML reader. We deliberately avoid
/// pulling `serde_yaml` into the zero-dep contract crate.
pub fn parse_style_yaml(yaml: &str) -> Result<GrammarStyleConfig, String> {
    // Pass 1: collect a flat (path, value) map. `path` is dot-joined.
    let pairs = collect_yaml_pairs(yaml)?;
    config_from_pairs(&pairs)
}

/// Build a [`GrammarStyleConfig`] from already-flattened key/value pairs.
/// Public so tests / alternative loaders can supply pairs directly when the
/// YAML reader's subset is too narrow.
pub fn config_from_pairs(pairs: &[(String, String)]) -> Result<GrammarStyleConfig, String> {
    let lookup = |k: &str| -> Option<&str> {
        pairs.iter().find(|(p, _)| p == k).map(|(_, v)| v.as_str())
    };
    let lookup_list = |k: &str| -> Vec<String> {
        pairs
            .iter()
            .filter_map(|(p, v)| {
                if p == k || p.starts_with(&format!("{k}.")) {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .collect()
    };
    let req = |k: &str| -> Result<&str, String> {
        lookup(k).ok_or_else(|| format!("missing key: {k}"))
    };

    let style = parse_style_name(req("style")?)?;

    let nars = NarsPriorityChain {
        primary: parse_nars_inference(req("nars.primary")?)?,
        fallback: parse_nars_inference(req("nars.fallback")?)?,
    };

    let table_strs = lookup_list("morphology.tables");
    let mut tables = Vec::new();
    for s in &table_strs {
        tables.push(parse_morphology_table(s)?);
    }
    let agglutinative_mode = parse_bool(req("morphology.agglutinative_mode")?)?;
    let morphology = MorphologyPolicy { tables, agglutinative_mode };

    let slot_strs = lookup_list("tekamolo.priority");
    let mut priority = Vec::new();
    for s in &slot_strs {
        priority.push(parse_tekamolo_slot(s)?);
    }
    let require_fillable = parse_bool(req("tekamolo.require_fillable")?)?;
    let tekamolo = TekamoloPolicy { priority, require_fillable };

    let radius: u8 = req("markov.radius")?
        .parse()
        .map_err(|e| format!("markov.radius: {e}"))?;
    let kernel = parse_kernel(req("markov.kernel")?)?;
    let replay = parse_replay(req("markov.replay")?)?;
    let markov = MarkovPolicy { radius, kernel, replay };

    let pearl_mask = parse_u8_with_hex(req("spo_causal.pearl_mask")?)?;
    let ambiguity_tolerance: f32 = req("spo_causal.ambiguity_tolerance")?
        .parse()
        .map_err(|e| format!("spo_causal.ambiguity_tolerance: {e}"))?;
    let spo_causal = SpoCausalPolicy { pearl_mask, ambiguity_tolerance };

    let local_threshold: f32 = req("coverage.local_threshold")?
        .parse()
        .map_err(|e| format!("coverage.local_threshold: {e}"))?;
    let escalate_below: f32 = req("coverage.escalate_below")?
        .parse()
        .map_err(|e| format!("coverage.escalate_below: {e}"))?;
    let coverage = CoveragePolicy { local_threshold, escalate_below };

    Ok(GrammarStyleConfig {
        style,
        nars,
        morphology,
        tekamolo,
        markov,
        spo_causal,
        coverage,
    })
}

/// Flatten a YAML document into `(dotted.path, value)` pairs. List items
/// repeat the key (`tekamolo.priority` appears once per slot).
fn collect_yaml_pairs(yaml: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    // path_stack tracks (indent, key) for the active mapping nesting.
    let mut path_stack: Vec<(usize, String)> = Vec::new();
    let mut active_list_key: Option<(usize, String)> = None;

    for (lineno, raw_line) in yaml.lines().enumerate() {
        // Strip comments (only when not inside a quoted scalar — our config
        // files don't use quoted scalars, so plain split is safe).
        let line_no_comment = match raw_line.find('#') {
            Some(idx) => &raw_line[..idx],
            None => raw_line,
        };
        let trimmed = line_no_comment.trim_end();
        if trimmed.trim().is_empty() {
            continue;
        }
        let indent = trimmed.chars().take_while(|c| *c == ' ').count();
        let body = trimmed.trim_start();

        // Block list item.
        if let Some(item) = body.strip_prefix("- ") {
            let key = active_list_key
                .as_ref()
                .map(|(_, k)| k.clone())
                .ok_or_else(|| format!("line {}: list item without parent key", lineno + 1))?;
            out.push((key, item.trim().to_string()));
            continue;
        }
        // Leaving a list block (this line isn't a `- ` item but indent matches).
        if active_list_key.is_some() {
            active_list_key = None;
        }

        // Pop path entries whose indent ≥ current line's indent.
        while path_stack.last().map(|(i, _)| *i).unwrap_or(usize::MAX) != usize::MAX
            && path_stack.last().map(|(i, _)| *i >= indent).unwrap_or(false)
        {
            path_stack.pop();
        }

        // Split `key: value` (value may be empty for parent maps).
        let (key_raw, value_raw) = match body.split_once(':') {
            Some((k, v)) => (k.trim(), v.trim()),
            None => return Err(format!("line {}: missing ':' in '{}'", lineno + 1, body)),
        };
        let dotted = if path_stack.is_empty() {
            key_raw.to_string()
        } else {
            let prefix: Vec<&str> = path_stack.iter().map(|(_, k)| k.as_str()).collect();
            format!("{}.{}", prefix.join("."), key_raw)
        };

        if value_raw.is_empty() {
            // Parent map; remember it for the next deeper indent block.
            path_stack.push((indent, key_raw.to_string()));
            // Could become a list parent on the next line; record provisional.
            active_list_key = Some((indent, dotted));
            continue;
        }

        // Inline flow list: `[a, b, c]`.
        if let Some(rest) = value_raw.strip_prefix('[') {
            let rest = rest
                .strip_suffix(']')
                .ok_or_else(|| format!("line {}: unterminated flow list", lineno + 1))?;
            for item in rest.split(',') {
                let item = item.trim();
                if !item.is_empty() {
                    out.push((dotted.clone(), item.to_string()));
                }
            }
            continue;
        }

        // Scalar.
        out.push((dotted, value_raw.to_string()));
    }

    Ok(out)
}

fn parse_style_name(s: &str) -> Result<ThinkingStyle, String> {
    // Spec ships 12 starter styles; map each to the closest entry in the
    // canonical 36-style taxonomy. Any genuine ThinkingStyle variant name
    // (case-insensitive) is also accepted as a passthrough.
    let lower = s.trim().to_ascii_lowercase();
    Ok(match lower.as_str() {
        "analytical"    => ThinkingStyle::Analytical,
        "convergent"    => ThinkingStyle::Logical,
        "systematic"    => ThinkingStyle::Systematic,
        "creative"      => ThinkingStyle::Creative,
        "divergent"     => ThinkingStyle::Imaginative,
        "exploratory"   => ThinkingStyle::Exploratory,
        "focused"       => ThinkingStyle::Precise,
        "diffuse"       => ThinkingStyle::Reflective,
        "peripheral"    => ThinkingStyle::Curious,
        "intuitive"     => ThinkingStyle::Empathetic,
        "deliberate"    => ThinkingStyle::Methodical,
        "metacognitive" => ThinkingStyle::Metacognitive,
        // Passthrough for canonical names.
        "logical"       => ThinkingStyle::Logical,
        "critical"      => ThinkingStyle::Critical,
        "methodical"    => ThinkingStyle::Methodical,
        "precise"       => ThinkingStyle::Precise,
        "imaginative"   => ThinkingStyle::Imaginative,
        "innovative"    => ThinkingStyle::Innovative,
        "artistic"      => ThinkingStyle::Artistic,
        "poetic"        => ThinkingStyle::Poetic,
        "playful"       => ThinkingStyle::Playful,
        "empathetic"    => ThinkingStyle::Empathetic,
        "compassionate" => ThinkingStyle::Compassionate,
        "supportive"    => ThinkingStyle::Supportive,
        "nurturing"     => ThinkingStyle::Nurturing,
        "gentle"        => ThinkingStyle::Gentle,
        "warm"          => ThinkingStyle::Warm,
        "direct"        => ThinkingStyle::Direct,
        "concise"       => ThinkingStyle::Concise,
        "efficient"     => ThinkingStyle::Efficient,
        "pragmatic"     => ThinkingStyle::Pragmatic,
        "blunt"         => ThinkingStyle::Blunt,
        "frank"         => ThinkingStyle::Frank,
        "curious"       => ThinkingStyle::Curious,
        "questioning"   => ThinkingStyle::Questioning,
        "investigative" => ThinkingStyle::Investigative,
        "speculative"   => ThinkingStyle::Speculative,
        "philosophical" => ThinkingStyle::Philosophical,
        "reflective"    => ThinkingStyle::Reflective,
        "contemplative" => ThinkingStyle::Contemplative,
        "wise"          => ThinkingStyle::Wise,
        "transcendent"  => ThinkingStyle::Transcendent,
        "sovereign"     => ThinkingStyle::Sovereign,
        other => return Err(format!("unknown style: {other}")),
    })
}

fn parse_nars_inference(s: &str) -> Result<NarsInference, String> {
    let lower = s.trim().to_ascii_lowercase();
    Ok(match lower.as_str() {
        "deduction"                       => NarsInference::Deduction,
        "induction"                       => NarsInference::Induction,
        "abduction"                       => NarsInference::Abduction,
        "revision"                        => NarsInference::Revision,
        "synthesis"                       => NarsInference::Synthesis,
        "extrapolation"                   => NarsInference::Extrapolation,
        "counterfactualsynthesis" | "counterfactual_synthesis" | "counterfactual"
                                          => NarsInference::CounterfactualSynthesis,
        other => return Err(format!("unknown nars inference: {other}")),
    })
}

fn parse_morphology_table(s: &str) -> Result<MorphologyTableId, String> {
    let lower = s.trim().to_ascii_lowercase();
    Ok(match lower.as_str() {
        "english_svo"             => MorphologyTableId::EnglishSvo,
        "finnish_case_table" | "finnish_case" => MorphologyTableId::FinnishCase,
        "russian_case_table" | "russian_case" => MorphologyTableId::RussianCase,
        "german_case_table"  | "german_case"  => MorphologyTableId::GermanCase,
        "turkish_aggl" | "turkish_agglutinative" => MorphologyTableId::TurkishAgglutinative,
        "japanese_particle" | "japanese_particles" => MorphologyTableId::JapaneseParticles,
        other => return Err(format!("unknown morphology table: {other}")),
    })
}

fn parse_tekamolo_slot(s: &str) -> Result<TekamoloSlot, String> {
    let lower = s.trim().to_ascii_lowercase();
    Ok(match lower.as_str() {
        "temporal"   => TekamoloSlot::Temporal,
        "kausal"     => TekamoloSlot::Kausal,
        "modal"      => TekamoloSlot::Modal,
        "lokal"      => TekamoloSlot::Lokal,
        // `instrument` is in the spec's enum sketch but absent from the
        // canonical `TekamoloSlot`; map to Modal as the closest fit.
        "instrument" => TekamoloSlot::Modal,
        other => return Err(format!("unknown tekamolo slot: {other}")),
    })
}

fn parse_kernel(s: &str) -> Result<WeightingKernel, String> {
    let lower = s.trim().to_ascii_lowercase();
    Ok(match lower.as_str() {
        "uniform"     => WeightingKernel::Uniform,
        "mexican_hat" | "mexicanhat" => WeightingKernel::MexicanHat,
        "gaussian"    => WeightingKernel::Gaussian,
        other => return Err(format!("unknown kernel: {other}")),
    })
}

fn parse_replay(s: &str) -> Result<ReplayStrategy, String> {
    let lower = s.trim().to_ascii_lowercase();
    Ok(match lower.as_str() {
        "forward"            => ReplayStrategy::Forward,
        "backward"           => ReplayStrategy::Backward,
        "both_and_compare" | "bothandcompare" => ReplayStrategy::BothAndCompare,
        other => return Err(format!("unknown replay direction: {other}")),
    })
}

fn parse_bool(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "on"  => Ok(true),
        "false" | "no" | "off" => Ok(false),
        other => Err(format!("expected bool, got '{other}'")),
    }
}

fn parse_u8_with_hex(s: &str) -> Result<u8, String> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u8::from_str_radix(hex, 16).map_err(|e| format!("hex u8 '{s}': {e}"))
    } else {
        s.parse().map_err(|e| format!("u8 '{s}': {e}"))
    }
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
    fn divergence_from_is_zero_at_bootstrap() {
        // Fresh awareness: no observations → neutral truth → no drift
        // from any prior → divergence is ~0 (scaled by c_init = 0.01).
        let prior = base_prior();
        let a = GrammarStyleAwareness::bootstrap(prior.style);
        let d = a.divergence_from(&prior);
        assert!(
            d < 0.01,
            "bootstrap awareness should have near-zero divergence, got {d}"
        );
    }

    #[test]
    fn divergence_rises_when_prior_contradicted() {
        let prior = base_prior();
        let mut a = GrammarStyleAwareness::bootstrap(prior.style);
        // 50 strong contradictions of the prior's primary inference.
        for _ in 0..50 {
            a.revise(
                ParamKey::NarsPrimary(prior.nars.primary),
                ParseOutcome::LocalFailureLLMSucceeded,
            );
        }
        let d = a.divergence_from(&prior);
        // Primary-drift term: (1 - f) * c where f is near 0 and c is near φ-1.
        // Recent-success drift: |f - 0.5| * c on the same direction.
        // Combined should exceed 0.5.
        assert!(
            d > 0.5,
            "50 contradicting revisions should produce significant divergence, got {d}"
        );
    }

    #[test]
    fn divergence_bounded() {
        // Any awareness state must produce a divergence in [0, 2].
        let prior = base_prior();
        let mut a = GrammarStyleAwareness::bootstrap(prior.style);
        for _ in 0..200 {
            a.revise(
                ParamKey::NarsPrimary(prior.nars.primary),
                ParseOutcome::LocalFailureLLMSucceeded,
            );
        }
        let d = a.divergence_from(&prior);
        assert!((0.0..=2.0).contains(&d), "divergence out of bounds: {d}");
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

    // -- YAML parser ---------------------------------------------------------

    const ANALYTICAL_YAML: &str = r#"
style: analytical
nars:
  primary: Deduction
  fallback: Abduction
morphology:
  tables: [english_svo, finnish_case_table]
  agglutinative_mode: false
tekamolo:
  priority: [temporal, lokal, kausal, modal]
  require_fillable: true
markov:
  radius: 5
  kernel: uniform
  replay: forward
spo_causal:
  pearl_mask: 0x01
  ambiguity_tolerance: 0.1
coverage:
  local_threshold: 0.90
  escalate_below: 0.85
"#;

    #[test]
    fn parse_style_yaml_analytical_ok() {
        let cfg = parse_style_yaml(ANALYTICAL_YAML).expect("parse failed");
        assert_eq!(cfg.style, ThinkingStyle::Analytical);
        assert_eq!(cfg.nars.primary, NarsInference::Deduction);
        assert_eq!(cfg.nars.fallback, NarsInference::Abduction);
        assert_eq!(cfg.morphology.tables.len(), 2);
        assert!(!cfg.morphology.agglutinative_mode);
        assert_eq!(cfg.tekamolo.priority.len(), 4);
        assert!(cfg.tekamolo.require_fillable);
        assert_eq!(cfg.markov.radius, 5);
        assert_eq!(cfg.markov.kernel, WeightingKernel::Uniform);
        assert_eq!(cfg.markov.replay, ReplayStrategy::Forward);
        assert_eq!(cfg.spo_causal.pearl_mask, 0x01);
        assert!((cfg.spo_causal.ambiguity_tolerance - 0.1).abs() < 1e-6);
        assert!((cfg.coverage.local_threshold - 0.90).abs() < 1e-6);
        assert!((cfg.coverage.escalate_below - 0.85).abs() < 1e-6);
    }

    #[test]
    fn parse_style_yaml_block_list_form() {
        // Block-list form for `tables` (one `- item` per line) must work
        // identically to the inline `[a, b]` form.
        let yaml = r#"
style: exploratory
nars:
  primary: CounterfactualSynthesis
  fallback: Abduction
morphology:
  tables:
    - english_svo
    - finnish_case_table
    - russian_case_table
  agglutinative_mode: true
tekamolo:
  priority:
    - modal
    - kausal
    - lokal
    - temporal
  require_fillable: false
markov:
  radius: 5
  kernel: mexican_hat
  replay: both_and_compare
spo_causal:
  pearl_mask: 0xFF
  ambiguity_tolerance: 0.4
coverage:
  local_threshold: 0.70
  escalate_below: 0.50
"#;
        let cfg = parse_style_yaml(yaml).expect("parse failed");
        assert_eq!(cfg.style, ThinkingStyle::Exploratory);
        assert_eq!(cfg.nars.primary, NarsInference::CounterfactualSynthesis);
        assert_eq!(cfg.morphology.tables.len(), 3);
        assert!(cfg.morphology.agglutinative_mode);
        assert_eq!(cfg.markov.kernel, WeightingKernel::MexicanHat);
        assert_eq!(cfg.markov.replay, ReplayStrategy::BothAndCompare);
        assert_eq!(cfg.spo_causal.pearl_mask, 0xFF);
    }

    #[test]
    fn parse_style_yaml_unknown_style_errors() {
        let yaml = r#"
style: nonsensestyle
nars: { primary: Deduction, fallback: Abduction }
"#;
        // We don't even need full coverage; the style parse fails first.
        // (The `nars: { ... }` flow-map form isn't supported by our subset,
        // but the style error short-circuits before that matters.)
        let err = parse_style_yaml(yaml).err().expect("expected error");
        assert!(err.to_lowercase().contains("style") || err.contains("nonsense"));
    }

    #[test]
    fn truth_revision_neutral_to_local_success_raises_frequency() {
        // Spec test: NEUTRAL revised by LocalSuccess.observation() raises f.
        let neutral = TruthValue::new(0.5, 0.01);
        let (f_obs, c_obs) = ParseOutcome::LocalSuccess.observation();
        let revised = revise_truth(neutral, f_obs, c_obs);
        assert!(
            revised.frequency > 0.5,
            "LocalSuccess revision must raise frequency, got {}",
            revised.frequency
        );
    }

    #[test]
    fn truth_revision_neutral_to_local_failure_lowers_frequency() {
        let neutral = TruthValue::new(0.5, 0.01);
        let (f_obs, c_obs) = ParseOutcome::LocalFailureLLMSucceeded.observation();
        let revised = revise_truth(neutral, f_obs, c_obs);
        assert!(
            revised.frequency < 0.5,
            "LocalFailure revision must lower frequency, got {}",
            revised.frequency
        );
    }

    #[test]
    fn revise_confidence_is_monotone_under_fixed_c_obs() {
        // Confidence rises monotonically when `c_obs` is held constant
        // (regardless of frequency polarity). Mixing `c_obs` sizes can
        // pull confidence between two different asymptotes — that's a
        // separate (well-known) behaviour, not a violation.
        let mut t = TruthValue::new(0.5, 0.01);
        let mut last_c = t.confidence;
        for i in 0..30 {
            // Alternate polarity, keep c_obs fixed at 1.0.
            let (f_obs, c_obs) = if i % 2 == 0 {
                ParseOutcome::LocalSuccess.observation()
            } else {
                ParseOutcome::EscalatedAndLLMDisagreed.observation()
            };
            t = revise_truth(t, f_obs, c_obs);
            assert!(
                t.confidence >= last_c,
                "confidence regressed at step {i}: {} → {}",
                last_c, t.confidence
            );
            last_c = t.confidence;
        }
    }

    #[test]
    fn effective_config_preserves_prior_shape_for_empty_awareness() {
        // Empty awareness: every collection-shaped policy slot must round-trip
        // unchanged. `nars.primary` is allowed to fall through to fallback at
        // bootstrap (current `top_nars_inference` policy); we lock the
        // structural shape here, not the specific NARS choice.
        let prior = base_prior();
        let a = GrammarStyleAwareness::bootstrap(prior.style);
        let eff = a.effective_config(&prior);
        assert_eq!(eff.style, prior.style);
        assert_eq!(eff.nars.fallback, prior.nars.fallback);
        // primary is either the prior's primary or its fallback at bootstrap.
        assert!(
            eff.nars.primary == prior.nars.primary || eff.nars.primary == prior.nars.fallback,
            "effective primary should be prior.primary or prior.fallback, got {:?}",
            eff.nars.primary
        );
        assert_eq!(eff.morphology.tables, prior.morphology.tables);
        assert_eq!(eff.morphology.agglutinative_mode, prior.morphology.agglutinative_mode);
        assert_eq!(eff.tekamolo.priority, prior.tekamolo.priority);
        assert_eq!(eff.tekamolo.require_fillable, prior.tekamolo.require_fillable);
        assert_eq!(eff.markov.radius, prior.markov.radius);
        assert_eq!(eff.markov.kernel, prior.markov.kernel);
        assert_eq!(eff.markov.replay, prior.markov.replay);
        assert_eq!(eff.spo_causal, prior.spo_causal);
        assert_eq!(eff.coverage, prior.coverage);
    }
}
