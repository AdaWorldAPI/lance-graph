//! Orchestration bridge contract.
//!
//! This is THE key trait that replaces duplicated routing logic
//! across crewai-rust (StepRouter), n8n-rs (crew_router/ladybug_router),
//! and ladybug-rs (HybridEngine).
//!
//! # The Problem
//!
//! Before this contract:
//! - crewai-rust had StepRouter with StepDomain enum
//! - n8n-rs had crew_router.rs + ladybug_router.rs (HTTP proxies)
//! - ladybug-rs had HybridEngine with vector+cypher+temporal
//! - Each system re-invented step routing and thinking mode dispatch
//!
//! # The Solution
//!
//! One trait: `OrchestrationBridge`. Implemented ONCE in lance-graph.
//! Consumed by all three systems. In a single binary, this is a
//! direct function call. In multi-process, this is Arrow Flight.
//!
//! ```text
//! crewai-rust ──┐
//!               ├──► OrchestrationBridge (trait) ──► lance-graph (impl)
//! n8n-rs ───────┤
//!               │
//! ladybug-rs ───┘
//! ```

use crate::thinking::ThinkingStyle;
use crate::nars::InferenceType;
use crate::plan::ThinkingContext;

/// Step domain: which subsystem handles this step.
///
/// Replaces crewai-rust's StepDomain enum AND n8n-rs's routing logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepDomain {
    /// crewai-rust agent execution.
    Crew,
    /// ladybug-rs BindSpace / cognitive operations.
    Ladybug,
    /// n8n-rs workflow orchestration.
    N8n,
    /// lance-graph query execution.
    LanceGraph,
    /// Direct ndarray SIMD operation.
    Ndarray,
}

impl StepDomain {
    /// Parse step type prefix to domain.
    ///
    /// ```text
    /// "crew.agent.think" → Crew
    /// "lb.resonate"      → Ladybug
    /// "n8n.set"          → N8n
    /// "lg.cypher"        → LanceGraph
    /// "nd.hamming"       → Ndarray
    /// ```
    pub fn from_step_type(step_type: &str) -> Option<Self> {
        let prefix = step_type.split('.').next()?;
        match prefix {
            "crew" => Some(Self::Crew),
            "lb"   => Some(Self::Ladybug),
            "n8n"  => Some(Self::N8n),
            "lg"   => Some(Self::LanceGraph),
            "nd"   => Some(Self::Ndarray),
            _      => None,
        }
    }
}

/// Unified step — the unit of work crossing system boundaries.
///
/// This is the canonical type. crewai-rust's UnifiedStep and
/// n8n-contract's UnifiedStep should both be replaced by this.
#[derive(Debug, Clone)]
pub struct UnifiedStep {
    pub step_id: String,
    pub step_type: String,
    pub status: StepStatus,
    /// Thinking context (if resolved by planner).
    pub thinking: Option<ThinkingContext>,
    /// Agent decision trail.
    pub reasoning: Option<String>,
    /// NARS confidence (0.0–1.0).
    pub confidence: Option<f64>,
}

/// Step execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// Orchestration bridge — the single routing contract.
///
/// Replaces:
/// - crewai-rust StepRouter.dispatch()
/// - n8n-rs crew_router.rs / ladybug_router.rs
/// - ladybug-rs HybridEngine
///
/// In a single binary, these are direct function calls.
/// In multi-process, these become Arrow Flight RPCs.
pub trait OrchestrationBridge: Send + Sync {
    /// Route a step to the appropriate subsystem.
    fn route(&self, step: &mut UnifiedStep) -> Result<(), OrchestrationError>;

    /// Resolve thinking context for a step (before routing).
    fn resolve_thinking(
        &self,
        style: ThinkingStyle,
        inference_type: InferenceType,
    ) -> ThinkingContext;

    /// Check if a domain is available (feature-gated in single binary).
    fn domain_available(&self, domain: StepDomain) -> bool;
}

/// Orchestration error.
#[derive(Debug, Clone)]
pub enum OrchestrationError {
    /// Domain not available.
    DomainUnavailable(StepDomain),
    /// Step routing failed.
    RoutingFailed(String),
    /// Step execution failed.
    ExecutionFailed(String),
}

impl core::fmt::Display for OrchestrationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DomainUnavailable(d) => write!(f, "Domain unavailable: {d:?}"),
            Self::RoutingFailed(s) => write!(f, "Routing failed: {s}"),
            Self::ExecutionFailed(s) => write!(f, "Execution failed: {s}"),
        }
    }
}

/// Blackboard slot contract — for TypedSlot interop.
///
/// crewai-rust's Blackboard TypedSlots can store any `dyn Any`.
/// This trait defines the contract for slots that cross the bridge.
pub trait BridgeSlot: Send + Sync + core::fmt::Debug {
    /// Slot key.
    fn key(&self) -> &str;
    /// Step type that produced this slot.
    fn step_type(&self) -> &str;
    /// Whether this is a TypedSlot (zero-serde) or JSON slot.
    fn is_typed(&self) -> bool;
}
