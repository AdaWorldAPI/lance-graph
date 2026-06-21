//! # cognitive-compiler — trace → template synthesis
//!
//! The **compiler** in `LLM = teacher / compiler / critic`. Given a recorded
//! [`ExecutionTrace`] of a successful LLM run, its OGAR schema, and the source
//! spans that backed every claim, it distills a deterministic, replayable
//! [`TemplateCandidate`] — an Elixir-shaped pipeline whose steps map to OGAR
//! actions.
//!
//! > Iron rule (§18): **no trace → no template.** [`TraceCompiler::synthesize`]
//! > cannot be called without an `ExecutionTrace`; the type system enforces it.
//!
//! This is a **scaffold**: every compute method returns
//! [`CompileError::NotImplemented`]. The DTOs are the canonical type surface;
//! their long-term home is `lance-graph-contract` / `ogar-cognitive` (see the
//! plan at `.claude/plans/cognitive-compilation-v1.md`).
#![forbid(unsafe_code)]

/// The five Rubicon phases (Heckhausen). A trace records which phase produced
/// it; the compiler only synthesizes from `Execution`-phase traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RubiconPhase {
    Deliberation,
    Commitment,
    Planning,
    Execution,
    Evaluation,
}

/// A byte-range provenance pointer into a source document. Every [`Claim`] must
/// carry at least one (§18: no source span → no claim).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSpan {
    pub source_id: String,
    pub start: usize,
    pub end: usize,
}

/// One tool/provider invocation captured during the LLM run.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub tool: String,
    pub input: String,
    pub output: String,
}

/// One recorded action within a trace (maps to a future OGAR action).
#[derive(Debug, Clone)]
pub struct TraceAction {
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Whether this action's effect is reproducible without the LLM.
    pub deterministic: bool,
}

/// A claim asserted during the run, with its supporting provenance.
#[derive(Debug, Clone)]
pub struct Claim {
    pub text: String,
    pub source_spans: Vec<SourceSpan>,
}

/// The raw material for compilation (§5 of the plan). No trace ⇒ no template.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    pub trace_id: String,
    pub task_id: String,
    pub input_hash: String,
    pub rubicon_phase: RubiconPhase,
    pub thinking_style: String,
    pub actions: Vec<TraceAction>,
    pub tool_calls: Vec<ToolCall>,
    pub claims: Vec<Claim>,
    pub final_output: String,
    pub confidence: f32,
    pub model: String,
    pub schema_version: String,
}

/// One step of a synthesized template; `ogar_action` is the canonical binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateStep {
    pub label: String,
    pub ogar_action: String,
}

/// The compiler's output: a candidate template plus its replay fixture pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateCandidate {
    pub template_id: String,
    pub source_trace_id: String,
    pub schema_version: String,
    pub thinking_style: String,
    pub steps: Vec<TemplateStep>,
    /// Steps the compiler judged essential vs incidental (§6 source plan).
    pub essential_steps: Vec<String>,
    /// Steps that must keep an LLM fallback (non-deterministic).
    pub fallback_steps: Vec<String>,
}

/// Everything synthesis needs: the trace, its OGAR schema id, and the source set.
#[derive(Debug, Clone)]
pub struct SynthesisInput {
    pub trace: ExecutionTrace,
    pub ogar_schema_id: String,
    pub source_ids: Vec<String>,
}

/// Errors from the compilation surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// The scaffold contract: synthesis logic is not yet implemented.
    NotImplemented,
    /// Trace was not produced in the `Execution` phase.
    NotExecutionPhase,
    /// A claim lacked a source span (§18 violation).
    UnsourcedClaim(String),
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CompileError::NotImplemented => write!(f, "cognitive-compiler: synthesis not implemented"),
            CompileError::NotExecutionPhase => write!(f, "cognitive-compiler: trace is not Execution-phase"),
            CompileError::UnsourcedClaim(c) => write!(f, "cognitive-compiler: claim without source span: {c}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// The compiler surface. Implementors turn a proven trace into a candidate
/// template; the trait signature makes "no trace → no template" unrepresentable.
pub trait TraceCompiler {
    /// Distill a [`TemplateCandidate`] from a proven execution trace.
    fn synthesize(&self, input: &SynthesisInput) -> Result<TemplateCandidate, CompileError>;
}

/// The default scaffold compiler. Validates the §18 preconditions it *can*
/// check structurally, then returns [`CompileError::NotImplemented`] for the
/// synthesis itself (no fabricated templates).
#[derive(Debug, Default, Clone, Copy)]
pub struct ScaffoldCompiler;

impl TraceCompiler for ScaffoldCompiler {
    fn synthesize(&self, input: &SynthesisInput) -> Result<TemplateCandidate, CompileError> {
        if input.trace.rubicon_phase != RubiconPhase::Execution {
            return Err(CompileError::NotExecutionPhase);
        }
        if let Some(c) = input.trace.claims.iter().find(|c| c.source_spans.is_empty()) {
            return Err(CompileError::UnsourcedClaim(c.text.clone()));
        }
        // Synthesis itself is the first probe (D-CC-COMPILER logic), not shipped.
        Err(CompileError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exec_trace() -> ExecutionTrace {
        ExecutionTrace {
            trace_id: "t1".into(),
            task_id: "task1".into(),
            input_hash: "h".into(),
            rubicon_phase: RubiconPhase::Execution,
            thinking_style: "skeptical".into(),
            actions: vec![],
            tool_calls: vec![],
            claims: vec![Claim {
                text: "x".into(),
                source_spans: vec![SourceSpan { source_id: "a".into(), start: 0, end: 1 }],
            }],
            final_output: "{}".into(),
            confidence: 0.9,
            model: "teacher".into(),
            schema_version: "v1".into(),
        }
    }

    #[test]
    fn scaffold_returns_not_implemented_for_valid_trace() {
        let input = SynthesisInput { trace: exec_trace(), ogar_schema_id: "s".into(), source_ids: vec![] };
        assert_eq!(ScaffoldCompiler.synthesize(&input), Err(CompileError::NotImplemented));
    }

    #[test]
    fn rejects_non_execution_phase() {
        let mut t = exec_trace();
        t.rubicon_phase = RubiconPhase::Deliberation;
        let input = SynthesisInput { trace: t, ogar_schema_id: "s".into(), source_ids: vec![] };
        assert_eq!(ScaffoldCompiler.synthesize(&input), Err(CompileError::NotExecutionPhase));
    }

    #[test]
    fn rejects_unsourced_claim() {
        let mut t = exec_trace();
        t.claims[0].source_spans.clear();
        let input = SynthesisInput { trace: t, ogar_schema_id: "s".into(), source_ids: vec![] };
        assert!(matches!(ScaffoldCompiler.synthesize(&input), Err(CompileError::UnsourcedClaim(_))));
    }
}
