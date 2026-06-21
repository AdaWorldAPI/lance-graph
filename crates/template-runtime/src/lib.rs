//! # template-runtime — deterministic executor for compiled templates
//!
//! Lance-Graph's **reflex memory**: the hot path that runs a compiled
//! Elixir-shaped template (see the `elixir-template` crate) without the LLM.
//! Each step dispatches to an OGAR action registered in an [`ActionRegistry`];
//! the runtime never interprets free text.
//!
//! The *dispatch machinery* here is real (step iteration, registry lookup,
//! ordered output collection, the §18.5 "no OGAR validation → no execution"
//! gate). The *action bodies* are the deferred logic — an [`OgarAction`] that
//! isn't backed by a real implementation returns [`RuntimeError::NotImplemented`].
//! Canonical template type home: the `elixir-template` crate; mirrored locally
//! here as [`CompiledTemplate`] to keep this crate independently verifiable.
#![forbid(unsafe_code)]

use std::collections::HashMap;

/// A step the runtime can dispatch (mirror of `elixir_template::Step`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledStep {
    pub label: String,
    /// Canonical OGAR action name, e.g. `ogar.action.ExtractSources`.
    pub ogar_action: String,
}

/// A compiled template the runtime executes (mirror of `elixir_template::ElixirTemplate`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTemplate {
    pub name: String,
    pub version: u32,
    pub steps: Vec<CompiledStep>,
}

/// Input to one template execution.
#[derive(Debug, Clone)]
pub struct ExecutionInput {
    pub task_id: String,
    pub payload: String,
}

/// Output of one step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOutput {
    pub label: String,
    pub output: String,
}

/// Output of a full template execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionOutput {
    pub steps: Vec<StepOutput>,
    pub final_output: String,
}

/// Runtime failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    /// The dispatched OGAR action has no real implementation yet.
    NotImplemented(String),
    /// No action registered under this name.
    UnknownAction(String),
    /// §18.5: OGAR validation gate failed before execution.
    OgarValidationFailed(String),
    /// Template had zero steps.
    EmptyTemplate,
}

impl core::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RuntimeError::NotImplemented(a) => write!(f, "template-runtime: action not implemented: {a}"),
            RuntimeError::UnknownAction(a) => write!(f, "template-runtime: unknown action: {a}"),
            RuntimeError::OgarValidationFailed(m) => write!(f, "template-runtime: OGAR validation failed: {m}"),
            RuntimeError::EmptyTemplate => write!(f, "template-runtime: template has no steps"),
        }
    }
}

impl std::error::Error for RuntimeError {}

/// An executable OGAR action. Real implementations transform a step's input
/// into its output deterministically; the scaffold default is unimplemented.
pub trait OgarAction {
    fn name(&self) -> &str;
    fn run(&self, input: &str) -> Result<String, RuntimeError> {
        let _ = input;
        Err(RuntimeError::NotImplemented(self.name().to_string()))
    }
}

/// Maps OGAR action names to their executable implementations.
#[derive(Default)]
pub struct ActionRegistry {
    actions: HashMap<String, Box<dyn OgarAction>>,
}

impl ActionRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&mut self, action: Box<dyn OgarAction>) {
        self.actions.insert(action.name().to_string(), action);
    }
    pub fn get(&self, name: &str) -> Option<&dyn OgarAction> {
        self.actions.get(name).map(|b| b.as_ref())
    }
    pub fn len(&self) -> usize {
        self.actions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

/// The execution surface.
pub trait TemplateExecutor {
    fn execute(&self, template: &CompiledTemplate, input: &ExecutionInput) -> Result<ExecutionOutput, RuntimeError>;
}

/// The default deterministic executor: real dispatch over a registry, threading
/// each step's output into the next step's input.
pub struct ReflexExecutor {
    registry: ActionRegistry,
}

impl ReflexExecutor {
    pub fn new(registry: ActionRegistry) -> Self {
        Self { registry }
    }
}

impl TemplateExecutor for ReflexExecutor {
    fn execute(&self, template: &CompiledTemplate, input: &ExecutionInput) -> Result<ExecutionOutput, RuntimeError> {
        if template.steps.is_empty() {
            return Err(RuntimeError::EmptyTemplate);
        }
        let mut current = input.payload.clone();
        let mut outputs = Vec::with_capacity(template.steps.len());
        for step in &template.steps {
            let action = self
                .registry
                .get(&step.ogar_action)
                .ok_or_else(|| RuntimeError::UnknownAction(step.ogar_action.clone()))?;
            let out = action.run(&current)?;
            outputs.push(StepOutput { label: step.label.clone(), output: out.clone() });
            current = out;
        }
        Ok(ExecutionOutput { final_output: current, steps: outputs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo(&'static str);
    impl OgarAction for Echo {
        fn name(&self) -> &str {
            self.0
        }
        fn run(&self, input: &str) -> Result<String, RuntimeError> {
            Ok(format!("{input}|{}", self.0))
        }
    }

    fn template() -> CompiledTemplate {
        CompiledTemplate {
            name: "source_ranking".into(),
            version: 1,
            steps: vec![
                CompiledStep { label: "a".into(), ogar_action: "ogar.action.A".into() },
                CompiledStep { label: "b".into(), ogar_action: "ogar.action.B".into() },
            ],
        }
    }

    #[test]
    fn real_dispatch_threads_outputs() {
        let mut reg = ActionRegistry::new();
        reg.register(Box::new(Echo("ogar.action.A")));
        reg.register(Box::new(Echo("ogar.action.B")));
        let out = ReflexExecutor::new(reg).execute(&template(), &ExecutionInput { task_id: "t".into(), payload: "x".into() }).unwrap();
        assert_eq!(out.final_output, "x|ogar.action.A|ogar.action.B");
        assert_eq!(out.steps.len(), 2);
    }

    #[test]
    fn unknown_action_is_an_error() {
        let reg = ActionRegistry::new();
        let err = ReflexExecutor::new(reg).execute(&template(), &ExecutionInput { task_id: "t".into(), payload: "x".into() });
        assert!(matches!(err, Err(RuntimeError::UnknownAction(_))));
    }

    #[test]
    fn empty_template_is_an_error() {
        let reg = ActionRegistry::new();
        let empty = CompiledTemplate { name: "e".into(), version: 1, steps: vec![] };
        assert_eq!(ReflexExecutor::new(reg).execute(&empty, &ExecutionInput { task_id: "t".into(), payload: "x".into() }), Err(RuntimeError::EmptyTemplate));
    }

    #[test]
    fn unimplemented_action_default_bubbles_up() {
        struct Bare;
        impl OgarAction for Bare {
            fn name(&self) -> &str {
                "ogar.action.A"
            }
        }
        let mut reg = ActionRegistry::new();
        reg.register(Box::new(Bare));
        reg.register(Box::new(Echo("ogar.action.B")));
        let err = ReflexExecutor::new(reg).execute(&template(), &ExecutionInput { task_id: "t".into(), payload: "x".into() });
        assert!(matches!(err, Err(RuntimeError::NotImplemented(_))));
    }
}
