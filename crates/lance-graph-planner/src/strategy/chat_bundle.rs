//! Strategy #17: Chat Bundle — BindSpace-backed conversation context.
//!
//! When the query is a chat completion (not Cypher/GQL/SPARQL), this strategy
//! activates and handles the hot path: bundle chat history into a Base17
//! fingerprint, route via HHTL palette, select thinking style.
//!
//! This is the bridge between OpenAI-compatible chat API and the lance-graph
//! planner pipeline. Cold path (graph queries) bypasses this entirely.
//!
//! ```text
//! POST /v1/chat/completions
//!   → ChatBundle.affinity() = 0.95 (chat detected)
//!   → ChatBundle.plan():
//!       1. Tokenize messages → Base17 fingerprints
//!       2. Bundle history (weighted, recency-decayed)
//!       3. Detect intent → ThinkingStyle
//!       4. Route via HHTL palette (Skip/Attend/Escalate)
//!       5. If Escalate → fall through to graph query strategies
//!       6. If Attend → direct response via palette distance lookup
//! ```

use crate::ir::{Arena, LogicalOp, LogicalPlan, Node};
use crate::traits::{PlanCapability, PlanContext, PlanInput, PlanStrategy};
use crate::PlanError;

/// Chat message fingerprint (17 × i16 = 34 bytes).
/// Same as ndarray::hpc::bgz17_bridge::Base17.
#[derive(Clone, Debug)]
pub struct Base17 {
    pub dims: [i16; 17],
}

impl Base17 {
    pub fn l1(&self, other: &Self) -> u32 {
        let mut d = 0u32;
        for i in 0..17 {
            d += (self.dims[i] as i32 - other.dims[i] as i32).unsigned_abs();
        }
        d
    }
}

/// Chat conversation bundle — accumulates fingerprints across messages.
#[derive(Clone, Debug)]
pub struct ChatBundle {
    /// Current bundle (weighted mean of all messages).
    pub bundle: Base17,
    /// Number of messages accumulated.
    pub message_count: u32,
    /// NARS confidence in the bundle (grows with messages).
    pub confidence: f32,
}

impl ChatBundle {
    pub fn new() -> Self {
        Self {
            bundle: Base17 { dims: [0; 17] },
            message_count: 0,
            confidence: 0.0,
        }
    }

    /// Add a message fingerprint to the bundle with recency weighting.
    /// Newer messages get weight 1.0, bundle gets weight decay.
    pub fn add(&mut self, message: &Base17, decay: f32) {
        let w_old = decay;
        let w_new = 1.0;
        let total = w_old + w_new;
        for d in 0..17 {
            let old = self.bundle.dims[d] as f32 * w_old;
            let new = message.dims[d] as f32 * w_new;
            self.bundle.dims[d] = ((old + new) / total).round() as i16;
        }
        self.message_count += 1;
        // Confidence grows with messages, asymptotes at 0.99
        self.confidence = (1.0 - 1.0 / (1.0 + self.message_count as f32)).min(0.99);
    }

    /// Detect topic shift: L1 distance between new message and bundle.
    /// Returns normalized distance (0.0 = same topic, 1.0 = completely different).
    pub fn topic_shift(&self, message: &Base17) -> f32 {
        let max_l1 = (17u32 * 65535) as f32;
        self.bundle.l1(message) as f32 / max_l1
    }
}

/// Route action from HHTL palette lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatRoute {
    /// Direct response — palette has the answer.
    Direct,
    /// Needs graph query (Escalate to cold path).
    GraphQuery,
    /// Needs deeper reasoning (Escalate to thinking orchestration).
    DeepThinking,
}

/// Chat bundle planner strategy.
#[derive(Debug)]
pub struct ChatBundleStrategy;

impl PlanStrategy for ChatBundleStrategy {
    fn name(&self) -> &str { "ChatBundle" }

    fn capability(&self) -> PlanCapability { PlanCapability::Extension }

    fn affinity(&self, context: &PlanContext) -> f32 {
        // High affinity for chat-like queries (not Cypher/GQL/SPARQL)
        let q = &context.query;
        if q.starts_with("MATCH ") || q.starts_with("SELECT ")
            || q.starts_with("g.") || q.contains("WHERE {") {
            return 0.0; // Graph query → skip chat bundle
        }
        // Natural language or JSON chat request → high affinity
        if q.contains("\"messages\"") || q.contains("\"role\"") {
            return 0.95;
        }
        // Plain text → moderate affinity (could be chat or free-form)
        0.5
    }

    fn plan(
        &self,
        input: PlanInput,
        arena: &mut Arena<LogicalOp>,
    ) -> Result<PlanInput, PlanError> {
        // For now: pass through. The actual bundling happens at the
        // HTTP handler level where we have the ChatBundle state.
        // This strategy's job is to signal to the planner that
        // chat-mode is active (via features).
        let mut output = input;
        output.context.features.has_resonance = true;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{PlanContext, QueryFeatures};

    #[test]
    fn test_chat_bundle_accumulation() {
        let mut bundle = ChatBundle::new();
        assert_eq!(bundle.message_count, 0);
        assert_eq!(bundle.confidence, 0.0);

        bundle.add(&Base17 { dims: [100; 17] }, 0.8);
        assert_eq!(bundle.message_count, 1);
        assert!(bundle.confidence > 0.0);

        bundle.add(&Base17 { dims: [100; 17] }, 0.8);
        assert_eq!(bundle.message_count, 2);
        assert!(bundle.confidence > 0.5);
    }

    #[test]
    fn test_topic_shift_detection() {
        let mut bundle = ChatBundle::new();
        bundle.add(&Base17 { dims: [100; 17] }, 0.8);

        let same_topic = Base17 { dims: [105; 17] };
        let different_topic = Base17 { dims: [10000; 17] };

        assert!(bundle.topic_shift(&same_topic) < 0.01);
        assert!(bundle.topic_shift(&different_topic) > 0.1);
    }

    #[test]
    fn test_affinity_chat_vs_cypher() {
        let strategy = ChatBundleStrategy;

        let chat_ctx = PlanContext {
            query: r#"{"messages":[{"role":"user","content":"hello"}]}"#.into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert!(strategy.affinity(&chat_ctx) > 0.9);

        let cypher_ctx = PlanContext {
            query: "MATCH (n:Person) RETURN n".into(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: None,
            nars_hint: None,
        };
        assert_eq!(strategy.affinity(&cypher_ctx), 0.0);
    }

    #[test]
    fn test_recency_decay() {
        let mut bundle = ChatBundle::new();
        // Add many messages about topic A
        for _ in 0..10 {
            bundle.add(&Base17 { dims: [100; 17] }, 0.9);
        }
        let before_b = bundle.bundle.dims[0];
        // Add one message about topic B
        bundle.add(&Base17 { dims: [500; 17] }, 0.9);
        // Bundle should shift toward B but not be at B
        assert!(bundle.bundle.dims[0] > before_b, "should shift toward new topic");
        assert!(bundle.bundle.dims[0] < 500, "should not jump to new topic entirely");
    }
}
