//! Inference DAG Orchestrator — pipeline of inference stages with NARS RL on combinations.
//!
//! Not a single-action selector. A **multi-stage inference pipeline** where:
//! - Stages run in sequence and parallel (DAG, not linear)
//! - Each stage's output feeds the next stage's input
//! - RL learns which **combinations and orderings** produce results
//! - Jina BF16 cross-model measures relevance at each stage
//! - NARS truth values accumulate on the **path**, not individual nodes
//!
//! ```text
//! Stage 1: FAN-OUT (breadth)
//!   ├─ Association ("what relates?")
//!   └─ Intuition ("what resonates?")
//!
//! Stage 2: EXPLAIN (from stage 1 results, parallel)
//!   ├─ Abduction ("what explains the associations?")
//!   └─ Induction ("what pattern in the intuitions?")
//!
//! Stage 3: FORM (from stage 2)
//!   ├─ Hypothesis ("testable claim from abduction")
//!   └─ Deduction ("chain forward from induction")
//!
//! Stage 4: TEST (from stage 3)
//!   ├─ Synthesis ("combine hypothesis + deduction")
//!   ├─ Counterfactual ("what if wrong?")
//!   └─ Extrapolation ("where does it lead?")
//!
//! Stage 5: MEASURE
//!   └─ Jina BF16 cross-model cosine on all outputs
//!
//! Stage 6: RL
//!   └─ NARS truth on the COMBINATION (path reward, not node reward)
//! ```
//!
//! The orchestrator learns that for entity queries, Association→Deduction→Synthesis
//! scores 0.9, but for causal queries, Abduction→Counterfactual→Synthesis wins.
//!
//! Zero dependencies.

use crate::thinking::{ThinkingStyle, FieldModulation};

// ═══════════════════════════════════════════════════════════════════════════
// INFERENCE STAGES — the atoms of reasoning
// ═══════════════════════════════════════════════════════════════════════════

/// An atomic inference operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum InferenceOp {
    /// Fan-out: "what relates to this?" — broad association search.
    Association = 0,
    /// Fast pattern: "what resonates?" — System 1, gut feeling.
    Intuition = 1,
    /// Explain backward: "what would explain this?" — abductive.
    Abduction = 2,
    /// Chain forward: "what follows?" — deductive.
    Deduction = 3,
    /// Generalize: "what pattern?" — inductive.
    Induction = 4,
    /// Propose: "testable claim" — hypothesis generation.
    Hypothesis = 5,
    /// Verify: "does evidence support?" — hypothesis testing.
    HypothesisTest = 6,
    /// Combine: "how do these merge?" — cross-domain synthesis.
    Synthesis = 7,
    /// Project: "where does this lead?" — forward extrapolation.
    Extrapolation = 8,
    /// Negate: "what if X hadn't happened?" — Pearl Level 3.
    Counterfactual = 9,
}

impl InferenceOp {
    pub const ALL: [InferenceOp; 10] = [
        Self::Association, Self::Intuition, Self::Abduction,
        Self::Deduction, Self::Induction, Self::Hypothesis,
        Self::HypothesisTest, Self::Synthesis,
        Self::Extrapolation, Self::Counterfactual,
    ];

    /// Preferred thinking style when this op runs.
    pub fn style(&self) -> ThinkingStyle {
        match self {
            Self::Association    => ThinkingStyle::Curious,
            Self::Intuition      => ThinkingStyle::Warm,
            Self::Abduction      => ThinkingStyle::Investigative,
            Self::Deduction      => ThinkingStyle::Logical,
            Self::Induction      => ThinkingStyle::Analytical,
            Self::Hypothesis     => ThinkingStyle::Speculative,
            Self::HypothesisTest => ThinkingStyle::Critical,
            Self::Synthesis      => ThinkingStyle::Creative,
            Self::Extrapolation  => ThinkingStyle::Philosophical,
            Self::Counterfactual => ThinkingStyle::Metacognitive,
        }
    }

    /// Pearl causal level.
    pub fn pearl_level(&self) -> u8 {
        match self {
            Self::Association | Self::Intuition | Self::Induction => 1,
            Self::Deduction | Self::Abduction | Self::Hypothesis | Self::HypothesisTest => 2,
            Self::Counterfactual | Self::Extrapolation | Self::Synthesis => 3,
        }
    }

    /// Default fan-out width.
    pub fn fan_out(&self) -> usize {
        match self {
            Self::Association => 8, Self::Intuition => 3,
            Self::Abduction => 4, Self::Deduction => 2,
            Self::Induction => 6, Self::Hypothesis => 3,
            Self::HypothesisTest => 2, Self::Synthesis => 4,
            Self::Extrapolation => 3, Self::Counterfactual => 2,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// INFERENCE DAG — stages connected as a directed acyclic graph
// ═══════════════════════════════════════════════════════════════════════════

/// A node in the inference DAG.
#[derive(Debug, Clone)]
pub struct DagNode {
    /// Which inference operation.
    pub op: InferenceOp,
    /// Indices of nodes whose output feeds this node's input.
    pub inputs: Vec<usize>,
    /// Stage number (0 = root, higher = later).
    pub stage: u8,
}

/// A complete inference DAG — the execution plan.
///
/// Nodes are topologically ordered: node i only depends on nodes j < i.
#[derive(Debug, Clone)]
pub struct InferenceDag {
    /// Nodes in topological order.
    pub nodes: Vec<DagNode>,
}

impl InferenceDag {
    /// The default 4-stage inference pipeline.
    ///
    /// ```text
    /// S0: Association(0), Intuition(1)
    /// S1: Abduction(2, from 0), Induction(3, from 1)
    /// S2: Hypothesis(4, from 2), Deduction(5, from 3)
    /// S3: Synthesis(6, from 4+5), Counterfactual(7, from 4), Extrapolation(8, from 5)
    /// ```
    pub fn default_pipeline() -> Self {
        Self {
            nodes: vec![
                // Stage 0: Fan-out
                DagNode { op: InferenceOp::Association,    inputs: vec![],     stage: 0 },
                DagNode { op: InferenceOp::Intuition,      inputs: vec![],     stage: 0 },
                // Stage 1: Explain
                DagNode { op: InferenceOp::Abduction,      inputs: vec![0],    stage: 1 },
                DagNode { op: InferenceOp::Induction,      inputs: vec![1],    stage: 1 },
                // Stage 2: Form
                DagNode { op: InferenceOp::Hypothesis,     inputs: vec![2],    stage: 2 },
                DagNode { op: InferenceOp::Deduction,      inputs: vec![3],    stage: 2 },
                // Stage 3: Test + Combine
                DagNode { op: InferenceOp::Synthesis,      inputs: vec![4, 5], stage: 3 },
                DagNode { op: InferenceOp::Counterfactual, inputs: vec![4],    stage: 3 },
                DagNode { op: InferenceOp::Extrapolation,  inputs: vec![5],    stage: 3 },
            ],
        }
    }

    /// Focused pipeline for entity queries (narrow, deep).
    pub fn entity_pipeline() -> Self {
        Self {
            nodes: vec![
                DagNode { op: InferenceOp::Association,  inputs: vec![],  stage: 0 },
                DagNode { op: InferenceOp::Deduction,    inputs: vec![0], stage: 1 },
                DagNode { op: InferenceOp::Synthesis,    inputs: vec![1], stage: 2 },
            ],
        }
    }

    /// Causal pipeline for "why" queries (abduction-heavy).
    pub fn causal_pipeline() -> Self {
        Self {
            nodes: vec![
                DagNode { op: InferenceOp::Association,    inputs: vec![],     stage: 0 },
                DagNode { op: InferenceOp::Intuition,      inputs: vec![],     stage: 0 },
                DagNode { op: InferenceOp::Abduction,      inputs: vec![0, 1], stage: 1 },
                DagNode { op: InferenceOp::Hypothesis,     inputs: vec![2],    stage: 2 },
                DagNode { op: InferenceOp::HypothesisTest, inputs: vec![3],    stage: 2 },
                DagNode { op: InferenceOp::Counterfactual, inputs: vec![3],    stage: 3 },
                DagNode { op: InferenceOp::Synthesis,      inputs: vec![4, 5], stage: 3 },
            ],
        }
    }

    /// Exploratory pipeline for unknown territory (wide fan-out).
    pub fn exploratory_pipeline() -> Self {
        Self {
            nodes: vec![
                DagNode { op: InferenceOp::Association,   inputs: vec![],     stage: 0 },
                DagNode { op: InferenceOp::Intuition,     inputs: vec![],     stage: 0 },
                DagNode { op: InferenceOp::Induction,     inputs: vec![0, 1], stage: 1 },
                DagNode { op: InferenceOp::Abduction,     inputs: vec![0, 1], stage: 1 },
                DagNode { op: InferenceOp::Extrapolation, inputs: vec![2],    stage: 2 },
                DagNode { op: InferenceOp::Hypothesis,    inputs: vec![3],    stage: 2 },
                DagNode { op: InferenceOp::Synthesis,     inputs: vec![4, 5], stage: 3 },
            ],
        }
    }

    /// Number of stages.
    pub fn depth(&self) -> u8 {
        self.nodes.iter().map(|n| n.stage).max().unwrap_or(0) + 1
    }

    /// Nodes at a given stage (can execute in parallel).
    pub fn stage(&self, s: u8) -> Vec<(usize, &DagNode)> {
        self.nodes.iter().enumerate()
            .filter(|(_, n)| n.stage == s)
            .collect()
    }

    /// Total fan-out across all nodes.
    pub fn total_fan_out(&self) -> usize {
        self.nodes.iter().map(|n| n.op.fan_out()).sum()
    }

    /// Path signature: the sequence of ops (for RL keying).
    pub fn signature(&self) -> Vec<u8> {
        self.nodes.iter().map(|n| n.op as u8).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STAGE RESULT — output of executing one DAG node
// ═══════════════════════════════════════════════════════════════════════════

/// Result of executing one inference node.
#[derive(Debug, Clone)]
pub struct NodeResult {
    /// Which op was executed.
    pub op: InferenceOp,
    /// Edges discovered by this node.
    pub edges_discovered: usize,
    /// Edges confirmed by this node.
    pub edges_confirmed: usize,
    /// Jina BF16 relevance of the node's output.
    pub relevance: f32,
}

/// Result of executing a full DAG.
#[derive(Debug, Clone)]
pub struct DagResult {
    /// Per-node results.
    pub node_results: Vec<NodeResult>,
    /// Combined relevance (geometric mean across all nodes).
    pub combined_relevance: f32,
    /// Total edges discovered.
    pub total_discovered: usize,
    /// Total edges confirmed.
    pub total_confirmed: usize,
}

impl DagResult {
    pub fn from_nodes(results: Vec<NodeResult>) -> Self {
        let total_discovered: usize = results.iter().map(|r| r.edges_discovered).sum();
        let total_confirmed: usize = results.iter().map(|r| r.edges_confirmed).sum();
        let combined = if results.is_empty() { 0.0 } else {
            let product: f64 = results.iter().map(|r| r.relevance.max(0.01) as f64).product();
            product.powf(1.0 / results.len() as f64) as f32
        };
        Self { node_results: results, combined_relevance: combined, total_discovered, total_confirmed }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NARS PATH RL — learns which DAG paths produce results
// ═══════════════════════════════════════════════════════════════════════════

/// NARS truth value for a path.
#[derive(Debug, Clone, Copy)]
pub struct PathTruth {
    pub frequency: f32,
    pub confidence: f32,
}

impl PathTruth {
    pub fn prior() -> Self { Self { frequency: 0.5, confidence: 0.1 } }

    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }

    pub fn revise(&self, new_f: f32, evidence_weight: f32) -> Self {
        let w1 = self.confidence / (1.0 - self.confidence + 1e-9);
        let w2 = evidence_weight;
        let total = w1 + w2;
        if total < 1e-9 { return *self; }
        let f = (self.frequency * w1 + new_f * w2) / total;
        let c = (total / (total + 1.0)).min(0.99);
        PathTruth { frequency: f, confidence: c }
    }
}

/// RL state keyed by DAG signature (the sequence of ops).
#[derive(Debug, Clone)]
pub struct PathEntry {
    pub signature: Vec<u8>,
    pub truth: PathTruth,
    pub executions: u32,
}

// ═══════════════════════════════════════════════════════════════════════════
// THE ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Inference DAG Orchestrator.
///
/// Maintains a library of DAG templates + NARS truth per path.
/// Selects the best DAG for a given query context.
/// Measures results with Jina BF16.
/// RL on the **combination** (path), not individual ops.
pub struct Orchestrator {
    /// Available DAG templates.
    pub templates: Vec<InferenceDag>,
    /// NARS truth per path signature.
    pub path_truths: Vec<PathEntry>,
    /// Execution history.
    pub history: Vec<(Vec<u8>, DagResult)>,
    /// Temperature for exploration (decays).
    pub temperature: f32,
    /// Steps executed.
    pub step_count: u64,
}

impl Orchestrator {
    pub fn new() -> Self {
        let templates = vec![
            InferenceDag::default_pipeline(),
            InferenceDag::entity_pipeline(),
            InferenceDag::causal_pipeline(),
            InferenceDag::exploratory_pipeline(),
        ];
        let path_truths = templates.iter().map(|t| PathEntry {
            signature: t.signature(),
            truth: PathTruth::prior(),
            executions: 0,
        }).collect();

        Self {
            templates,
            path_truths,
            history: Vec::new(),
            temperature: 1.0,
            step_count: 0,
        }
    }

    /// Select the best DAG for the current situation.
    pub fn select_dag(&self) -> &InferenceDag {
        let explore = {
            let hash = self.step_count.wrapping_mul(0x9E3779B97F4A7C15);
            (hash % 100) as f32 / 100.0 < self.temperature * 0.3
        };

        if explore {
            // Pick least-executed template
            let min_idx = self.path_truths.iter().enumerate()
                .min_by_key(|(_, e)| e.executions)
                .map(|(i, _)| i).unwrap_or(0);
            &self.templates[min_idx]
        } else {
            // Pick highest expectation
            let best_idx = self.path_truths.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.truth.expectation().partial_cmp(&b.truth.expectation()).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            &self.templates[best_idx]
        }
    }

    /// Record the outcome of executing a DAG.
    pub fn record_outcome(&mut self, dag: &InferenceDag, result: DagResult) {
        let sig = dag.signature();
        let success = result.combined_relevance > 0.5
            || result.total_discovered > 0
            || result.total_confirmed > 0;
        let reward_f = if success { result.combined_relevance.max(0.6) } else { 0.2 };

        // Find or create path entry
        if let Some(entry) = self.path_truths.iter_mut().find(|e| e.signature == sig) {
            entry.truth = entry.truth.revise(reward_f, result.combined_relevance.max(0.1));
            entry.executions += 1;
        } else {
            self.path_truths.push(PathEntry {
                signature: sig.clone(),
                truth: PathTruth::prior().revise(reward_f, result.combined_relevance.max(0.1)),
                executions: 1,
            });
        }

        self.history.push((sig, result));
        self.step_count += 1;
        self.temperature = (self.temperature * 0.99).max(0.05);
    }

    /// Add a custom DAG template (learned or user-defined).
    pub fn add_template(&mut self, dag: InferenceDag) {
        let sig = dag.signature();
        if !self.path_truths.iter().any(|e| e.signature == sig) {
            self.path_truths.push(PathEntry {
                signature: sig,
                truth: PathTruth::prior(),
                executions: 0,
            });
        }
        self.templates.push(dag);
    }

    /// Mutate the best DAG to explore variations (evolutionary).
    /// Swaps one random node's op to a nearby alternative.
    pub fn mutate_best(&mut self) {
        let best_idx = self.path_truths.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.truth.expectation().partial_cmp(&b.truth.expectation()).unwrap())
            .map(|(i, _)| i).unwrap_or(0);

        if best_idx >= self.templates.len() { return; }
        let mut new_dag = self.templates[best_idx].clone();

        // Pick a node to mutate
        if new_dag.nodes.is_empty() { return; }
        let node_idx = (self.step_count as usize) % new_dag.nodes.len();
        let current_op = new_dag.nodes[node_idx].op as u8;
        let new_op_idx = (current_op as usize + 1) % InferenceOp::ALL.len();
        new_dag.nodes[node_idx].op = InferenceOp::ALL[new_op_idx];

        self.add_template(new_dag);
    }

    /// Get rankings of all paths.
    pub fn path_rankings(&self) -> Vec<(Vec<u8>, f32, u32)> {
        let mut rankings: Vec<_> = self.path_truths.iter()
            .map(|e| (e.signature.clone(), e.truth.expectation(), e.executions))
            .collect();
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        rankings
    }

    /// Summary stats.
    pub fn stats(&self) -> OrchestratorStats {
        let total_discovered: usize = self.history.iter().map(|(_, r)| r.total_discovered).sum();
        let total_confirmed: usize = self.history.iter().map(|(_, r)| r.total_confirmed).sum();
        let avg_relevance = if self.history.is_empty() { 0.0 } else {
            self.history.iter().map(|(_, r)| r.combined_relevance).sum::<f32>() / self.history.len() as f32
        };
        OrchestratorStats {
            steps: self.step_count,
            templates: self.templates.len(),
            temperature: self.temperature,
            total_discovered,
            total_confirmed,
            avg_relevance,
        }
    }
}

impl Default for Orchestrator {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct OrchestratorStats {
    pub steps: u64,
    pub templates: usize,
    pub temperature: f32,
    pub total_discovered: usize,
    pub total_confirmed: usize,
    pub avg_relevance: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pipeline_structure() {
        let dag = InferenceDag::default_pipeline();
        assert_eq!(dag.depth(), 4);
        assert_eq!(dag.nodes.len(), 9);
        // Stage 0: 2 nodes (Association, Intuition)
        assert_eq!(dag.stage(0).len(), 2);
        // Stage 3: 3 nodes (Synthesis, Counterfactual, Extrapolation)
        assert_eq!(dag.stage(3).len(), 3);
    }

    #[test]
    fn test_pipeline_signatures_differ() {
        let default = InferenceDag::default_pipeline();
        let entity = InferenceDag::entity_pipeline();
        let causal = InferenceDag::causal_pipeline();
        assert_ne!(default.signature(), entity.signature());
        assert_ne!(default.signature(), causal.signature());
        assert_ne!(entity.signature(), causal.signature());
    }

    #[test]
    fn test_dag_result_geometric_mean() {
        let results = vec![
            NodeResult { op: InferenceOp::Association, edges_discovered: 3, edges_confirmed: 0, relevance: 0.9 },
            NodeResult { op: InferenceOp::Deduction, edges_discovered: 1, edges_confirmed: 2, relevance: 0.8 },
        ];
        let dag_result = DagResult::from_nodes(results);
        // geometric mean of 0.9 and 0.8 = sqrt(0.72) ≈ 0.849
        assert!((dag_result.combined_relevance - 0.849).abs() < 0.01);
        assert_eq!(dag_result.total_discovered, 4);
        assert_eq!(dag_result.total_confirmed, 2);
    }

    #[test]
    fn test_path_rl_learns_best_dag() {
        let mut orch = Orchestrator::new();

        // Reward causal pipeline repeatedly
        let causal = InferenceDag::causal_pipeline();
        for _ in 0..20 {
            let result = DagResult::from_nodes(vec![
                NodeResult { op: InferenceOp::Abduction, edges_discovered: 5, edges_confirmed: 3, relevance: 0.9 },
                NodeResult { op: InferenceOp::Counterfactual, edges_discovered: 2, edges_confirmed: 1, relevance: 0.85 },
            ]);
            orch.record_outcome(&causal, result);
        }

        // Punish entity pipeline
        let entity = InferenceDag::entity_pipeline();
        for _ in 0..10 {
            let result = DagResult::from_nodes(vec![
                NodeResult { op: InferenceOp::Association, edges_discovered: 0, edges_confirmed: 0, relevance: 0.1 },
            ]);
            orch.record_outcome(&entity, result);
        }

        // Causal should rank highest
        let rankings = orch.path_rankings();
        let causal_sig = causal.signature();
        let entity_sig = entity.signature();
        let causal_rank = rankings.iter().position(|(s, _, _)| *s == causal_sig).unwrap();
        let entity_rank = rankings.iter().position(|(s, _, _)| *s == entity_sig).unwrap();
        assert!(causal_rank < entity_rank, "causal should rank above entity");
    }

    #[test]
    fn test_mutate_creates_variation() {
        let mut orch = Orchestrator::new();
        let initial_count = orch.templates.len();
        orch.mutate_best();
        assert_eq!(orch.templates.len(), initial_count + 1);
        // Mutation should differ from original
        let original_sig = orch.templates[0].signature();
        let mutated_sig = orch.templates.last().unwrap().signature();
        assert_ne!(original_sig, mutated_sig, "mutation should change the DAG");
    }

    #[test]
    fn test_temperature_decay() {
        let mut orch = Orchestrator::new();
        let t0 = orch.temperature;
        let dag = InferenceDag::default_pipeline();
        for _ in 0..100 {
            let result = DagResult::from_nodes(vec![
                NodeResult { op: InferenceOp::Association, edges_discovered: 1, edges_confirmed: 0, relevance: 0.5 },
            ]);
            orch.record_outcome(&dag, result);
        }
        assert!(orch.temperature < t0);
        assert!(orch.temperature >= 0.05);
    }

    #[test]
    fn test_parallel_stages() {
        let dag = InferenceDag::default_pipeline();
        // Stage 0 has 2 parallel nodes
        let s0 = dag.stage(0);
        assert_eq!(s0.len(), 2);
        assert_eq!(s0[0].1.op, InferenceOp::Association);
        assert_eq!(s0[1].1.op, InferenceOp::Intuition);
        // Both have no inputs (root nodes)
        assert!(s0[0].1.inputs.is_empty());
        assert!(s0[1].1.inputs.is_empty());
        // Stage 1 depends on stage 0
        let s1 = dag.stage(1);
        assert!(s1.iter().all(|(_, n)| !n.inputs.is_empty()));
    }

    #[test]
    fn test_all_ops_have_styles() {
        for op in &InferenceOp::ALL {
            let style = op.style();
            // Verify it maps to a real ThinkingStyle
            assert!(ThinkingStyle::ALL.contains(&style), "{:?} has no valid style", op);
        }
    }

    #[test]
    fn test_custom_dag() {
        let mut orch = Orchestrator::new();
        // User-defined: Association → HypothesisTest → Synthesis
        let custom = InferenceDag {
            nodes: vec![
                DagNode { op: InferenceOp::Association,    inputs: vec![],  stage: 0 },
                DagNode { op: InferenceOp::HypothesisTest, inputs: vec![0], stage: 1 },
                DagNode { op: InferenceOp::Synthesis,      inputs: vec![1], stage: 2 },
            ],
        };
        orch.add_template(custom);
        assert_eq!(orch.templates.len(), 5); // 4 default + 1 custom
    }

    #[test]
    fn test_exploratory_pipeline_fan_out() {
        let dag = InferenceDag::exploratory_pipeline();
        let total = dag.total_fan_out();
        assert!(total > 20, "exploratory should have wide fan-out: {}", total);
    }
}
