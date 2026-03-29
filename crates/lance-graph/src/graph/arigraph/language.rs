//! Language backend abstraction — internal LLM + external API routing.
//!
//! Trait that both `InternalBackend` (OpenChat/GPT-2, in-process) and
//! `ExternalBackend` (xAI/Grok, HTTP) implement. The `MetaOrchestrator`
//! selects which backend to use based on DK position, temperature,
//! contradiction rate, and NARS topology quality tracking.
//!
//! # Path P18 — does not contradict P4-P16
//!
//! P18 is the **parsing layer**, orthogonal to data flow paths.
//! It enables P8 (entity resolution), P9 (causal reasoning),
//! P11 (scale cartography) by providing the language interface.

use super::triplet_graph::{Triplet, TripletGraph};
use std::collections::HashSet;

// ============================================================================
// LanguageBackend trait
// ============================================================================

/// Entity type classification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Event,
    Concept,
    Unknown,
}

/// Plan output from language backend.
#[derive(Clone, Debug)]
pub struct PlanResult {
    pub main_goal: String,
    pub steps: Vec<String>,
    pub requires_exploration: bool,
}

/// Abstract language backend — internal LLM or external API.
///
/// Both `InternalBackend` (OpenChat, in-process) and `ExternalBackend`
/// (xAI/Grok, HTTP) implement this same trait. The orchestrator picks
/// which one to call based on MUL assessment.
pub trait LanguageBackend: Send + Sync {
    /// Extract SPO triplets from a natural language observation.
    /// `context` provides relevant graph facts for grounding.
    fn extract_triplets(
        &self,
        observation: &str,
        context: &[String],
        timestamp: u64,
    ) -> Result<Vec<Triplet>, String>;

    /// Classify entities by type (Person, Org, Location, etc.).
    fn classify_entities(
        &self,
        entities: &[String],
    ) -> Result<Vec<(String, EntityType)>, String>;

    /// Generate a plan given the current blackboard state.
    fn plan(
        &self,
        blackboard: &ContextBlackboard,
        observation: &str,
    ) -> Result<PlanResult, String>;

    /// Identify outdated triplets that should be replaced by new observation.
    fn refine(
        &self,
        existing: &[String],
        observation: &str,
    ) -> Result<Vec<(String, String, String)>, String>;

    /// Free-form chat response grounded in blackboard context.
    fn chat(
        &self,
        user_message: &str,
        blackboard: &ContextBlackboard,
    ) -> Result<String, String>;

    /// Backend name for logging/metrics.
    fn name(&self) -> &str;

    /// True if this backend runs in-process (no network).
    fn is_local(&self) -> bool;
}

// ============================================================================
// ContextBlackboard
// ============================================================================

/// Shared context between internal LLM, external API, and orchestrator.
///
/// The blackboard IS the "unlimited context window" — it holds a BFS-expanded
/// view of the knowledge graph relevant to the current query, plus episodic
/// memory, attention edges, and orchestrator state.
///
/// Each turn: BFS → blackboard → LLM → new triplets → graph → NARS revision.
#[derive(Clone, Debug)]
pub struct ContextBlackboard {
    /// Current conversation turn.
    pub turn: u64,
    /// Active entity set (BFS frontier from last query).
    pub active_entities: HashSet<String>,
    /// Retrieved graph context (formatted triplets from BFS).
    pub graph_context: Vec<String>,
    /// Episodic memory hits (past observations).
    pub episodic_context: Vec<String>,
    /// Pending triplets to add (from LLM extraction).
    pub pending_triplets: Vec<Triplet>,
    /// Pending refinements (outdated triplets to replace).
    pub pending_refinements: Vec<(String, String, String)>,
    /// CausalEdge64 attention log (from last inference pass).
    pub attention_edges: Vec<u64>,
    /// Current thinking style name (from orchestrator).
    pub style: String,
    /// Current graph health bias (from sensorium).
    pub graph_bias: String,
    /// Maximum BFS depth for context retrieval.
    pub max_depth: usize,
    /// Maximum triplets to include in context.
    pub top_k: usize,
}

impl Default for ContextBlackboard {
    fn default() -> Self {
        Self {
            turn: 0,
            active_entities: HashSet::new(),
            graph_context: Vec::new(),
            episodic_context: Vec::new(),
            pending_triplets: Vec::new(),
            pending_refinements: Vec::new(),
            attention_edges: Vec::new(),
            style: "plan".into(),
            graph_bias: "balanced".into(),
            max_depth: 2,
            top_k: 20,
        }
    }
}

impl ContextBlackboard {
    /// Create a new blackboard for a conversation turn.
    pub fn new(turn: u64) -> Self {
        Self { turn, ..Default::default() }
    }

    /// Load context from a triplet graph via BFS from seed entities.
    pub fn load_from_graph(&mut self, graph: &TripletGraph, seed_entities: &[String]) {
        self.active_entities.clear();
        for e in seed_entities {
            self.active_entities.insert(e.clone());
        }

        let associated = graph.get_associated(&self.active_entities, self.max_depth);
        self.graph_context = associated.iter()
            .take(self.top_k)
            .map(|t| t.to_string_repr())
            .collect();
    }

    /// Format blackboard as context string for LLM prompt.
    pub fn as_context_string(&self) -> String {
        let mut ctx = String::new();

        if !self.graph_context.is_empty() {
            ctx.push_str("Known facts:\n");
            for fact in &self.graph_context {
                ctx.push_str("- ");
                ctx.push_str(fact);
                ctx.push('\n');
            }
        }

        if !self.episodic_context.is_empty() {
            ctx.push_str("\nPast observations:\n");
            for obs in &self.episodic_context {
                ctx.push_str("- ");
                ctx.push_str(obs);
                ctx.push('\n');
            }
        }

        ctx
    }

    /// Flush pending triplets (after graph commit).
    pub fn flush_pending(&mut self) {
        self.pending_triplets.clear();
        self.pending_refinements.clear();
    }

    /// Total context size (triplets + episodic).
    pub fn context_size(&self) -> usize {
        self.graph_context.len() + self.episodic_context.len()
    }
}

// ============================================================================
// Backend routing policy
// ============================================================================

/// Routing decision: which backend to use for this step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    /// Use internal LLM (fast, local, no network).
    Internal,
    /// Use external API (OSINT, web search, higher quality open-ended).
    External,
    /// Try internal first, fall back to external on low confidence.
    InternalWithFallback,
    /// Use both and merge results (highest quality, highest cost).
    Ensemble,
}

/// Decide which backend to use based on orchestrator state.
///
/// Rules:
/// - MountStupid → External only (don't trust internal yet)
/// - SlopeOfEnlightenment → Internal with fallback
/// - PlateauOfMastery → Internal only (fast, confident)
/// - High temperature (stagnation) → swap backend (break loops)
/// - High contradiction rate → External verify
pub fn select_backend(
    dk_position: &str,
    temperature: f32,
    contradiction_rate: f32,
    current_backend_is_internal: bool,
) -> BackendChoice {
    // Stagnation: swap to break loops
    if temperature > 0.7 {
        return if current_backend_is_internal {
            BackendChoice::External
        } else {
            BackendChoice::Internal
        };
    }

    // High contradictions: external verify
    if contradiction_rate > 0.3 {
        return BackendChoice::External;
    }

    // DK-based routing
    match dk_position {
        "mount_stupid" => BackendChoice::External,
        "valley_of_despair" => BackendChoice::InternalWithFallback,
        "slope_of_enlightenment" => BackendChoice::InternalWithFallback,
        "plateau_of_mastery" => BackendChoice::Internal,
        _ => BackendChoice::InternalWithFallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blackboard_default() {
        let bb = ContextBlackboard::default();
        assert_eq!(bb.turn, 0);
        assert_eq!(bb.max_depth, 2);
        assert_eq!(bb.top_k, 20);
        assert!(bb.active_entities.is_empty());
        assert!(bb.graph_context.is_empty());
    }

    #[test]
    fn test_blackboard_context_string() {
        let mut bb = ContextBlackboard::new(1);
        bb.graph_context = vec!["alice - knows - bob".into(), "bob - works_at - acme".into()];
        bb.episodic_context = vec!["saw alice at office".into()];
        let ctx = bb.as_context_string();
        assert!(ctx.contains("Known facts:"));
        assert!(ctx.contains("alice - knows - bob"));
        assert!(ctx.contains("Past observations:"));
        assert!(ctx.contains("saw alice at office"));
    }

    #[test]
    fn test_blackboard_flush() {
        let mut bb = ContextBlackboard::new(1);
        bb.pending_triplets.push(Triplet::new("a", "b", "c", 0));
        bb.pending_refinements.push(("x".into(), "y".into(), "z".into()));
        assert!(!bb.pending_triplets.is_empty());
        bb.flush_pending();
        assert!(bb.pending_triplets.is_empty());
        assert!(bb.pending_refinements.is_empty());
    }

    #[test]
    fn test_context_size() {
        let mut bb = ContextBlackboard::new(1);
        bb.graph_context = vec!["a".into(), "b".into()];
        bb.episodic_context = vec!["c".into()];
        assert_eq!(bb.context_size(), 3);
    }

    #[test]
    fn test_select_backend_mount_stupid() {
        let choice = select_backend("mount_stupid", 0.1, 0.1, true);
        assert_eq!(choice, BackendChoice::External);
    }

    #[test]
    fn test_select_backend_plateau() {
        let choice = select_backend("plateau_of_mastery", 0.1, 0.1, false);
        assert_eq!(choice, BackendChoice::Internal);
    }

    #[test]
    fn test_select_backend_stagnation_swaps() {
        // High temperature swaps from current backend
        let choice = select_backend("slope_of_enlightenment", 0.8, 0.1, true);
        assert_eq!(choice, BackendChoice::External);
        let choice = select_backend("slope_of_enlightenment", 0.8, 0.1, false);
        assert_eq!(choice, BackendChoice::Internal);
    }

    #[test]
    fn test_select_backend_high_contradictions() {
        let choice = select_backend("plateau_of_mastery", 0.1, 0.5, true);
        assert_eq!(choice, BackendChoice::External);
    }

    #[test]
    fn test_select_backend_default_fallback() {
        let choice = select_backend("unknown", 0.3, 0.1, true);
        assert_eq!(choice, BackendChoice::InternalWithFallback);
    }

    #[test]
    fn test_entity_type_variants() {
        assert_ne!(EntityType::Person, EntityType::Organization);
        assert_eq!(EntityType::Unknown, EntityType::Unknown);
    }
}
