//! # elixir-template — the compiled-cognition Elixir-shaped template
//!
//! Thinking styles, JITson, and the i4-32D thinking-style vectors already exist
//! in the workspace. The **Elixir template** is the missing piece: the
//! declarative, replayable macro a successful LLM run compiles down to, whose
//! steps bind to OGAR actions and run deterministically without the LLM.
//!
//! ```text
//! defmacro source_ranking_v1(input) do
//!   pipeline do
//!     step :extract_sources
//!     step :score_independence
//!     step :emit_ranked_sources
//!   end
//! end
//! ```
//!
//! This crate provides the *representation* ([`ElixirTemplate`] / [`Step`] /
//! [`OgarAction`]), a deterministic *parser* ([`ElixirTemplate::parse`]) for the
//! `pipeline do … end` shape, a *builder*, and the first vertical slice
//! ([`source_ranking_v1`]). OGAR's existing `ogar-from-elixir` is the eventual
//! richer front-end; this is the in-workspace canonical shape the
//! `template-runtime` executes. Zero-dep, real logic (no LLM here — parsing and
//! representation are deterministic).
#![forbid(unsafe_code)]

/// The OGAR action a template step binds to. Known variants cover the
/// source-ranking first slice; anything else is [`OgarAction::Custom`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OgarAction {
    ExtractSources,
    NormalizeClaims,
    ScorePrimaryProximity,
    ScoreIndependence,
    ScoreEvidenceDensity,
    PenalizeIncentiveRisk,
    EmitRankedSources,
    /// An action not in the known catalogue (snake_case atom preserved).
    Custom(String),
}

impl OgarAction {
    /// Map a `step :snake_case` atom to a known action, else `Custom`.
    pub fn from_atom(atom: &str) -> Self {
        match atom {
            "extract_sources" => OgarAction::ExtractSources,
            "normalize_claims" => OgarAction::NormalizeClaims,
            "score_primary_proximity" => OgarAction::ScorePrimaryProximity,
            "score_independence" => OgarAction::ScoreIndependence,
            "score_evidence_density" => OgarAction::ScoreEvidenceDensity,
            "penalize_incentive_risk" => OgarAction::PenalizeIncentiveRisk,
            "emit_ranked_sources" => OgarAction::EmitRankedSources,
            other => OgarAction::Custom(other.to_string()),
        }
    }

    /// The canonical OGAR action name the runtime registry dispatches on.
    pub fn ogar_name(&self) -> String {
        let pascal = match self {
            OgarAction::ExtractSources => "ExtractSources",
            OgarAction::NormalizeClaims => "NormalizeClaims",
            OgarAction::ScorePrimaryProximity => "ScorePrimaryProximity",
            OgarAction::ScoreIndependence => "ScoreIndependence",
            OgarAction::ScoreEvidenceDensity => "ScoreEvidenceDensity",
            OgarAction::PenalizeIncentiveRisk => "PenalizeIncentiveRisk",
            OgarAction::EmitRankedSources => "EmitRankedSources",
            OgarAction::Custom(s) => return format!("ogar.action.{}", to_pascal(s)),
        };
        format!("ogar.action.{pascal}")
    }
}

fn to_pascal(snake: &str) -> String {
    snake
        .split('_')
        .filter(|s| !s.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

/// One step of the pipeline: the source atom plus its bound OGAR action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub atom: String,
    pub action: OgarAction,
}

/// OSINT guardrail (§14 of the plan). Every OSINT template MUST carry one;
/// the brutal reviewers re-check it before promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OsintGuardrail {
    pub public_interest_reason: String,
    pub scope_boundary: String,
    pub source_provenance_required: bool,
    pub harm_minimization_checked: bool,
}

/// A compiled Elixir-shaped template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElixirTemplate {
    pub name: String,
    pub version: u32,
    pub steps: Vec<Step>,
    /// Present for OSINT-domain templates; `None` otherwise.
    pub osint: Option<OsintGuardrail>,
}

/// Parse failures for the `pipeline do … end` shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingDefmacro,
    MissingPipeline,
    NoSteps,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseError::MissingDefmacro => write!(f, "elixir-template: no `defmacro NAME(...)` found"),
            ParseError::MissingPipeline => write!(f, "elixir-template: no `pipeline do` block found"),
            ParseError::NoSteps => write!(f, "elixir-template: pipeline has no `step :atom` lines"),
        }
    }
}

impl std::error::Error for ParseError {}

impl ElixirTemplate {
    /// Deterministically parse the `defmacro NAME(...) do pipeline do step :a … end end`
    /// shape. A trailing `_vN` on the name is read as the version (default 1).
    pub fn parse(src: &str) -> Result<Self, ParseError> {
        let raw_name = extract_defmacro_name(src).ok_or(ParseError::MissingDefmacro)?;
        // Restrict everything to the FIRST macro's body, and within it the FIRST
        // `pipeline do … end` block. A `step` in a later macro, or outside the
        // pipeline, must NOT be folded into this template.
        let body = &src[src.find("defmacro").expect("defmacro found above")..];
        if !body.contains("pipeline") {
            return Err(ParseError::MissingPipeline);
        }
        let (name, version) = split_version(&raw_name);
        let steps: Vec<Step> = steps_in_first_pipeline(body)
            .into_iter()
            .map(|atom| Step { action: OgarAction::from_atom(&atom), atom })
            .collect();
        if steps.is_empty() {
            return Err(ParseError::NoSteps);
        }
        Ok(ElixirTemplate { name, version, steps, osint: None })
    }

    /// Start a template builder.
    pub fn builder(name: impl Into<String>) -> Builder {
        Builder { name: name.into(), version: 1, steps: Vec::new(), osint: None }
    }

    /// The canonical OGAR action names this template dispatches, in order.
    pub fn ogar_action_names(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.action.ogar_name()).collect()
    }

    /// Render back to the Elixir-shaped source (round-trips through [`parse`]).
    pub fn to_source(&self) -> String {
        let mut s = format!("defmacro {}_v{}(input) do\n  pipeline do\n", self.name, self.version);
        for step in &self.steps {
            s.push_str(&format!("    step :{}\n", step.atom));
        }
        s.push_str("  end\nend\n");
        s
    }
}

/// Fluent builder for [`ElixirTemplate`].
#[derive(Debug, Clone)]
pub struct Builder {
    name: String,
    version: u32,
    steps: Vec<Step>,
    osint: Option<OsintGuardrail>,
}

impl Builder {
    pub fn version(mut self, v: u32) -> Self {
        self.version = v;
        self
    }
    /// Add a step by its snake_case atom (the OGAR action is inferred).
    pub fn step(mut self, atom: impl Into<String>) -> Self {
        let atom = atom.into();
        self.steps.push(Step { action: OgarAction::from_atom(&atom), atom });
        self
    }
    pub fn osint(mut self, guard: OsintGuardrail) -> Self {
        self.osint = Some(guard);
        self
    }
    pub fn build(self) -> ElixirTemplate {
        ElixirTemplate { name: self.name, version: self.version, steps: self.steps, osint: self.osint }
    }
}

fn extract_defmacro_name(src: &str) -> Option<String> {
    let idx = src.find("defmacro")?;
    let after = &src[idx + "defmacro".len()..];
    let after = after.trim_start();
    let end = after.find(['(', ' ', '\n', '\t']).unwrap_or(after.len());
    let name = after[..end].trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

fn split_version(raw: &str) -> (String, u32) {
    if let Some(pos) = raw.rfind("_v") {
        let (base, ver) = raw.split_at(pos);
        if let Ok(n) = ver[2..].parse::<u32>() {
            return (base.to_string(), n);
        }
    }
    (raw.to_string(), 1)
}

/// Collect the `step :atom` lines inside the FIRST `pipeline do … end` block of
/// `body` (which begins at the first `defmacro`). Stops at that block's closing
/// `end`, so steps in a later macro or outside the pipeline are excluded.
fn steps_in_first_pipeline(body: &str) -> Vec<String> {
    let mut steps = Vec::new();
    let mut in_pipeline = false;
    for line in body.lines() {
        let t = line.trim();
        if !in_pipeline {
            if t.starts_with("pipeline") {
                in_pipeline = true;
            }
            continue;
        }
        if t == "end" {
            break; // closes the `pipeline do` block
        }
        if let Some(atom) = parse_step_line(line) {
            steps.push(atom);
        }
    }
    steps
}

fn parse_step_line(line: &str) -> Option<String> {
    let t = line.trim();
    let rest = t.strip_prefix("step")?.trim_start();
    let atom = rest.strip_prefix(':')?;
    let end = atom
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(atom.len());
    let atom = &atom[..end];
    if atom.is_empty() {
        None
    } else {
        Some(atom.to_string())
    }
}

/// The first vertical slice (§15): source ranking for public-narrative claims.
pub fn source_ranking_v1() -> ElixirTemplate {
    ElixirTemplate::builder("source_ranking")
        .version(1)
        .step("extract_sources")
        .step("normalize_claims")
        .step("score_primary_proximity")
        .step("score_independence")
        .step("score_evidence_density")
        .step("penalize_incentive_risk")
        .step("emit_ranked_sources")
        .osint(OsintGuardrail {
            public_interest_reason: "rank credibility of sources for a public narrative claim".into(),
            scope_boundary: "public claims / officials / institutions / media only".into(),
            source_provenance_required: true,
            harm_minimization_checked: true,
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_shape_with_version() {
        let src = "defmacro source_ranking_v1(input) do\n  pipeline do\n    step :extract_sources\n    step :emit_ranked_sources\n  end\nend\n";
        let t = ElixirTemplate::parse(src).unwrap();
        assert_eq!(t.name, "source_ranking");
        assert_eq!(t.version, 1);
        assert_eq!(t.steps.len(), 2);
        assert_eq!(t.steps[0].action, OgarAction::ExtractSources);
    }

    #[test]
    fn missing_pipeline_is_an_error() {
        let src = "defmacro foo(input) do\n  step :x\nend";
        assert_eq!(ElixirTemplate::parse(src), Err(ParseError::MissingPipeline));
    }

    #[test]
    fn no_steps_is_an_error() {
        let src = "defmacro foo(input) do\n  pipeline do\n  end\nend";
        assert_eq!(ElixirTemplate::parse(src), Err(ParseError::NoSteps));
    }

    #[test]
    fn parse_scopes_to_first_pipeline_only() {
        let src = "defmacro a_v1(input) do\n  pipeline do\n    step :extract_sources\n  end\nend\n\ndefmacro b_v1(input) do\n  pipeline do\n    step :score_independence\n  end\nend\n";
        let t = ElixirTemplate::parse(src).unwrap();
        assert_eq!(t.name, "a");
        assert_eq!(t.steps.len(), 1);
        assert_eq!(t.steps[0].action, OgarAction::ExtractSources);
    }

    #[test]
    fn parse_ignores_steps_outside_the_pipeline() {
        let src = "defmacro a_v1(input) do\n  step :score_independence\n  pipeline do\n    step :extract_sources\n  end\nend\n";
        let t = ElixirTemplate::parse(src).unwrap();
        assert_eq!(t.steps.len(), 1);
        assert_eq!(t.steps[0].action, OgarAction::ExtractSources);
    }

    #[test]
    fn unknown_atom_becomes_custom() {
        let a = OgarAction::from_atom("wibble_wobble");
        assert_eq!(a, OgarAction::Custom("wibble_wobble".into()));
        assert_eq!(a.ogar_name(), "ogar.action.WibbleWobble");
    }

    #[test]
    fn first_slice_has_seven_steps_and_guardrail() {
        let t = source_ranking_v1();
        assert_eq!(t.steps.len(), 7);
        assert!(t.osint.is_some());
        assert_eq!(t.ogar_action_names()[0], "ogar.action.ExtractSources");
    }

    #[test]
    fn source_round_trips_through_parse() {
        let t = source_ranking_v1();
        let reparsed = ElixirTemplate::parse(&t.to_source()).unwrap();
        assert_eq!(reparsed.name, t.name);
        assert_eq!(reparsed.version, t.version);
        assert_eq!(reparsed.steps, t.steps);
    }
}
