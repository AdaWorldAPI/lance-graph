use serde::{Deserialize, Serialize};

/// State of a single function ("neuron") in the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronState {
    /// Called during execution, returns valid data
    Alive,
    /// Compiles, has code, but never called in current execution path
    Static,
    /// Contains todo!(), unimplemented!(), unreachable!(), or panic!()
    Dead,
    /// Called but returns NaN/Inf/None where it shouldn't
    Nan,
    /// Exists but returns hardcoded/default values
    Stub,
    /// Called by something, but output is never consumed
    WiredUnused,
}

impl NeuronState {
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Alive => "alive",
            Self::Static => "static",
            Self::Dead => "dead",
            Self::Nan => "nan",
            Self::Stub => "stub",
            Self::WiredUnused => "wired_unused",
        }
    }

    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Alive | Self::Static)
    }
}

/// Metadata about a single function detected by static analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMeta {
    /// Fully qualified function ID: "crate::module::Type::method"
    pub id: String,
    /// Source file path (relative to repo root)
    pub file: String,
    /// Line number in the source file
    pub line: usize,
    /// Module name (directory-level grouping)
    pub module: String,
    /// Repository name
    pub repo: String,
    /// Function signature (simplified)
    pub signature: String,
    /// Whether the function body contains todo!()
    pub has_todo: bool,
    /// Whether the function body contains unimplemented!()
    pub has_unimplemented: bool,
    /// Whether the function body contains panic!() directly
    pub has_panic: bool,
    /// Whether the function appears to be a stub
    pub is_stub: bool,
    /// Whether the function has NaN risk (returns f32/f64 with division)
    pub has_nan_risk: bool,
    /// Compile-time determined state
    pub state: NeuronState,
    /// Lines of code in the function body
    pub body_loc: usize,
}

/// Diagnosis of an entire module (directory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDiagnosis {
    pub name: String,
    pub repo: String,
    pub path: String,
    pub functions: Vec<FunctionMeta>,
    pub total: usize,
    pub alive_or_static: usize,
    pub dead: usize,
    pub stub: usize,
    pub nan_risk: usize,
    pub health_pct: f32,
}

/// Diagnosis of an entire repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoDiagnosis {
    pub name: String,
    pub modules: Vec<ModuleDiagnosis>,
    pub total_functions: usize,
    pub total_dead: usize,
    pub total_stub: usize,
    pub total_nan_risk: usize,
    pub health_pct: f32,
}

/// Full stack diagnosis across all repos.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackDiagnosis {
    pub repos: Vec<RepoDiagnosis>,
    pub total_functions: usize,
    pub total_files: usize,
    pub total_dead: usize,
    pub total_stub: usize,
    pub total_nan_risk: usize,
    pub health_pct: f32,
    pub scan_duration_ms: u64,
}
