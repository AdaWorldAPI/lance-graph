//! NARS Mass Exploration — expand a LiteralGraph by querying the web edge-by-edge.
//!
//! ```text
//! Seed graph (aiwar: 221 nodes, 326 edges)
//!   → Pick frontier edge (lowest confidence, highest curiosity)
//!   → Pearl 2³ decompose: SEE/DO/IMAGINE queries
//!   → reader-lm: fetch + clean HTML → markdown
//!   → DeepNSM/extractor: extract new triplets
//!   → NARS revision: confirm/deny existing edges, add new ones
//!   → New edges join frontier → repeat
//!
//! The graph grows exponentially at the frontier.
//! NARS controls the explosion via confidence thresholds.
//! Sensorium monitors: when deduction_yield drops, stop exploring.
//! ```
//!
//! Zero external LLM API. Cost: $0. Speed: limited by HTTP fetch.

use crate::literal_graph::{LiteralEdge, LiteralGraph, LiteralNode};
use crate::mul::{DkPosition, FlowState, MulAssessment, TrustTexture};
use crate::qualia::{QualiaVector, ZERO as QUALIA_ZERO};
use crate::sensorium::GraphSignals;

/// Pearl 2³ query decomposition for a single edge.
///
/// Given `(Palantir, developed, Gotham)`, produces:
/// - SEE:     "Palantir developed Gotham" (observational — did this happen?)
/// - DO:      "what does Palantir developing Gotham cause" (interventional)
/// - IMAGINE: "what if Palantir had not developed Gotham" (counterfactual)
pub fn pearl_queries(edge: &LiteralEdge) -> Vec<PearlQuery> {
    let see = format!("{} {} {}", edge.source, edge.label, edge.target);
    let do_q = format!(
        "what does {} {} {} cause",
        edge.source, edge.label, edge.target
    );
    let imagine = format!(
        "what if {} had not {} {}",
        edge.source, edge.label, edge.target
    );

    vec![
        PearlQuery {
            text: see,
            level: PearlLevel::See,
            edge_source: edge.source.clone(),
            edge_target: edge.target.clone(),
            edge_label: edge.label.clone(),
        },
        PearlQuery {
            text: do_q,
            level: PearlLevel::Do,
            edge_source: edge.source.clone(),
            edge_target: edge.target.clone(),
            edge_label: edge.label.clone(),
        },
        PearlQuery {
            text: imagine,
            level: PearlLevel::Imagine,
            edge_source: edge.source.clone(),
            edge_target: edge.target.clone(),
            edge_label: edge.label.clone(),
        },
    ]
}

/// A Pearl-decomposed query.
#[derive(Debug, Clone)]
pub struct PearlQuery {
    pub text: String,
    pub level: PearlLevel,
    pub edge_source: String,
    pub edge_target: String,
    pub edge_label: String,
}

/// Pearl's three levels of causal inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PearlLevel {
    /// Level 1: Association — P(Y|X). "Did this happen?"
    See,
    /// Level 2: Intervention — P(Y|do(X)). "What does this cause?"
    Do,
    /// Level 3: Counterfactual — P(Y|do(X'), X=x). "What if not?"
    Imagine,
}

/// NARS truth value (frequency, confidence).
#[derive(Debug, Clone, Copy)]
pub struct NarsTruth {
    /// Frequency: P(positive) in [0, 1].
    pub frequency: f32,
    /// Confidence: weight of evidence in [0, 1).
    pub confidence: f32,
}

impl NarsTruth {
    pub fn new(f: f32, c: f32) -> Self {
        Self {
            frequency: f.clamp(0.0, 1.0),
            confidence: c.clamp(0.0, 0.99),
        }
    }

    /// Weak prior (no evidence).
    pub fn prior() -> Self {
        Self::new(0.5, 0.1)
    }

    /// NARS revision: merge two truth values.
    pub fn revision(&self, other: &NarsTruth) -> NarsTruth {
        let w1 = self.confidence / (1.0 - self.confidence + 1e-9);
        let w2 = other.confidence / (1.0 - other.confidence + 1e-9);
        let total = w1 + w2;
        if total < 1e-9 {
            return *self;
        }
        let f = (self.frequency * w1 + other.frequency * w2) / total;
        let c = total / (total + 1.0);
        NarsTruth::new(f, c.min(0.99))
    }

    /// Expectation: E = c(f - 0.5) + 0.5.
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }

    // NOTE: the syllogism truth-functions (deduction/induction/abduction/
    // exemplification/comparison/analogy/resemblance) are NOT redefined here.
    // The canonical, tested implementation is `nars_engine::nars_infer`
    // (crate lance-graph-planner). This `NarsTruth` is the *proposer-side*
    // (offline, float) frequency/confidence carrier; it intentionally exposes
    // only revision (evidence fusion) + expectation. Wiring the named
    // OpenNARS vocabulary (`cognitive_codebook::NarsInference`) into executable
    // truth-functions happens in the engine — see EPIPHANIES `E-NARS-ONE-ENGINE`.
}

/// An edge in the exploration frontier.
#[derive(Debug, Clone)]
pub struct FrontierEdge {
    pub source: String,
    pub target: String,
    pub label: String,
    pub truth: NarsTruth,
    /// How many times this edge has been queried.
    pub query_count: u32,
    /// Was this edge in the seed graph or discovered during exploration?
    pub is_seed: bool,
}

impl FrontierEdge {
    /// Curiosity score: prefer low-confidence, un-queried edges.
    pub fn curiosity(&self) -> f32 {
        let novelty = 1.0 / (self.query_count as f32 + 1.0);
        let uncertainty = 1.0 - self.truth.confidence;
        novelty * uncertainty
    }

    /// MUL-weighted curiosity — the exploration-gateway ranking key
    /// (`D-SCI-4a`, `E-MUL-EXPLORATION-GATEWAY-1`). The scalar magnitude of
    /// [`curiosity_gestalt`](Self::curiosity_gestalt).
    ///
    /// The graph's own uncertainty ([`GraphSignals`]) and self-assessment
    /// ([`MulAssessment`]) weight the base [`curiosity`](Self::curiosity):
    /// surprise (Staunen) pulls exploration up; over-confidence (MountStupid +
    /// Overconfident) pulls it down (humility); unsafe ground dampens it (never
    /// to zero). When MUL is neutral and the graph is quiet, this reduces EXACTLY
    /// to [`curiosity`](Self::curiosity) — MUL earns its weight or is inert
    /// (`G-CM-1`).
    ///
    /// Non-circular: reads only contract types; the substrate supplies
    /// `GraphSignals`/`NarsTruth`, MUL/the planner consumes.
    pub fn curiosity_mul(&self, assessment: &MulAssessment, signals: &GraphSignals) -> f32 {
        self.curiosity_gestalt(assessment, signals).magnitude
    }

    /// The full exploration **texture gestalt** — qualia AS a felt awareness of
    /// this edge's worth, not a bare number (operator steer: "wire qualia as a
    /// texture gestalt awareness"). Returns the 17D [`QualiaVector`] felt-quality
    /// plus the scalar `magnitude` (the frontier ranking key) projected from it.
    pub fn curiosity_gestalt(
        &self,
        assessment: &MulAssessment,
        signals: &GraphSignals,
    ) -> TextureGestalt {
        let novelty = 1.0 / (self.query_count as f32 + 1.0);
        let uncertainty = 1.0 - self.truth.confidence;
        // Staunen = graph-wide surprise (entropy + contradiction). NOTE: this is
        // NOT the qualia `wonder = √(coherence·expansion)` (coherent-novelty) —
        // Staunen is surprise-wonder, carried by arousal/entropy/tension below.
        let staunen = (0.5 * signals.truth_entropy + 0.5 * signals.contradiction_rate).clamp(0.0, 1.0);

        // ── the qualia texture (the awareness) ──
        let mut texture: QualiaVector = QUALIA_ZERO;
        texture[0] = (0.5 * staunen + 0.5 * uncertainty).clamp(0.0, 1.0); // arousal (Staunen activation)
        texture[1] = valence_axis(assessment.dk_position); // valence: approach vs avoid
        texture[2] = signals.contradiction_rate.clamp(0.0, 1.0); // tension: contradiction
        texture[4] = if assessment.complexity_mapped { 0.8 } else { 0.4 }; // clarity
        texture[6] = uncertainty.clamp(0.0, 1.0); // depth of the unknown
        texture[7] = signals.revision_velocity.clamp(0.0, 1.0); // velocity
        texture[8] = (0.5 * uncertainty + 0.5 * signals.truth_entropy).clamp(0.0, 1.0); // entropy
        texture[9] = trust_coherence(assessment.trust.texture); // coherence: how solid
        texture[14] = groundedness(assessment.trust.texture, assessment.dk_position); // groundedness
        texture[15] = novelty.clamp(0.0, 1.0); // expansion: novelty drive

        // ── project the magnitude from base curiosity × MUL × texture ──
        let base = novelty * uncertainty; // == self.curiosity()
        let fw = assessment.free_will_modifier as f32;
        let dk = dk_factor(assessment.dk_position);
        let flow = flow_factor(assessment.homeostasis.flow_state);
        let trust = trust_factor(assessment.trust.texture);
        let staunen_boost = 1.0 + staunen; // surprise pulls exploration
        let ground_gate = 0.4 + 0.6 * texture[14]; // dampen on unsafe ground, floor 0.4 (never 0)
        let magnitude = (base * fw * dk * flow * trust * staunen_boost * ground_gate).max(0.0);

        TextureGestalt { texture, magnitude }
    }
}

/// The exploration **texture gestalt**: qualia as a felt awareness of an edge's
/// exploration-worth. The `texture` is the 17D felt-quality (the awareness); the
/// `magnitude` is its scalar projection — the frontier ranking key.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureGestalt {
    /// The 17D qualia felt-texture of exploring this edge.
    pub texture: QualiaVector,
    /// The scalar exploration magnitude (frontier ranking key).
    pub magnitude: f32,
}

/// Dunning-Kruger exploration weight: over-confident novices are discounted
/// hard (the 0.3 humility factor); the slope-of-enlightenment is boosted.
fn dk_factor(dk: DkPosition) -> f32 {
    match dk {
        DkPosition::MountStupid => 0.3,
        DkPosition::ValleyOfDespair => 0.9,
        DkPosition::SlopeOfEnlightenment => 1.15,
        DkPosition::Plateau => 1.0,
    }
}

/// Flow-state exploration weight: boredom (under-challenged) seeks novelty;
/// anxiety (overwhelmed) explores less.
fn flow_factor(flow: FlowState) -> f32 {
    match flow {
        FlowState::Flow => 1.0,
        FlowState::Boredom => 1.2,
        FlowState::Transition => 0.9,
        FlowState::Anxiety => 0.5,
    }
}

/// Trust-texture exploration weight: don't trust the urge to explore when
/// over-confident; be slightly more curious when uncertain.
fn trust_factor(t: TrustTexture) -> f32 {
    match t {
        TrustTexture::Calibrated => 1.0,
        TrustTexture::Overconfident => 0.5,
        TrustTexture::Uncertain => 1.1,
        TrustTexture::Underconfident => 1.0,
    }
}

/// Coherence qualia axis from trust texture (how solid/calibrated the ground).
fn trust_coherence(t: TrustTexture) -> f32 {
    match t {
        TrustTexture::Calibrated => 0.9,
        TrustTexture::Underconfident => 0.6,
        TrustTexture::Overconfident => 0.5,
        TrustTexture::Uncertain => 0.3,
    }
}

/// Groundedness qualia axis: how SAFE it feels to explore here. Over-confident
/// ground (MountStupid) is treacherous regardless of texture.
fn groundedness(t: TrustTexture, dk: DkPosition) -> f32 {
    if dk == DkPosition::MountStupid {
        return 0.25;
    }
    match t {
        TrustTexture::Calibrated => 1.0,
        TrustTexture::Underconfident => 0.7,
        TrustTexture::Uncertain => 0.4,
        TrustTexture::Overconfident => 0.3,
    }
}

/// Valence qualia axis: approach (high) vs avoid (low) an exploration target
/// by Dunning-Kruger position.
fn valence_axis(dk: DkPosition) -> f32 {
    match dk {
        DkPosition::MountStupid => 0.2, // avoid: over-confident trap
        DkPosition::ValleyOfDespair => 0.5,
        DkPosition::SlopeOfEnlightenment => 0.9, // approach: learning
        DkPosition::Plateau => 0.7,
    }
}

/// Result of processing one search result page.
#[derive(Debug, Clone)]
pub struct ExplorationResult {
    /// New triplets discovered.
    pub new_edges: Vec<FrontierEdge>,
    /// Existing edges confirmed (truth revised upward).
    pub confirmed: Vec<(String, String, String, NarsTruth)>,
    /// Existing edges denied (truth revised downward).
    pub denied: Vec<(String, String, String, NarsTruth)>,
}

/// The mass exploration engine.
///
/// Maintains a frontier of edges to explore. Each iteration:
/// 1. Pick highest-curiosity edge from frontier
/// 2. Generate Pearl queries
/// 3. Fetch + extract (via reader-lm + DeepNSM/extractor)
/// 4. NARS revision on existing edges
/// 5. Add new discovered edges to frontier
/// 6. Repeat until budget exhausted or sensorium says stop
pub struct MassExplorer {
    /// The knowledge graph being expanded.
    pub graph: LiteralGraph,
    /// Frontier edges sorted by curiosity.
    pub frontier: Vec<FrontierEdge>,
    /// Exploration budget (max queries).
    pub budget: usize,
    /// Queries executed so far.
    pub queries_executed: usize,
    /// Total new edges discovered.
    pub edges_discovered: usize,
    /// Total confirmations.
    pub confirmations: usize,
    /// Total denials.
    pub denials: usize,
}

impl MassExplorer {
    /// Create from a seed LiteralGraph.
    pub fn from_graph(graph: LiteralGraph, budget: usize) -> Self {
        // TD-EXPLORATION-1: edge iteration goes through `seed_frontier()` later;
        // the `from_graph` constructor leaves `frontier` empty by design.
        // The original placeholder loop and `edges` vec were dead code, removed.
        let frontier = Vec::new();

        Self {
            graph,
            frontier,
            budget,
            queries_executed: 0,
            edges_discovered: 0,
            confirmations: 0,
            denials: 0,
        }
    }

    /// Initialize frontier from all graph edges.
    pub fn seed_frontier(&mut self) {
        self.frontier.clear();
        // Pre-collected list (kept for future filtering by node_id allow-list).
        // TD-EXPLORATION-2: `node_ids` is a placeholder for filtering by
        // a node_id allow-list. Currently unused; marked _ to silence
        // the lint until the filter wiring lands.
        let _node_ids: Vec<String> = (0..self.graph.node_count())
            .filter_map(|_| None::<String>) // TODO: needs node_id access
            .collect();

        // Use the graph's internal iteration
        for node_id in self.graph.all_node_ids() {
            for edge in self.graph.edges_from(&node_id) {
                self.frontier.push(FrontierEdge {
                    source: edge.source.clone(),
                    target: edge.target.clone(),
                    label: edge.label.clone(),
                    truth: NarsTruth::new(0.9, 0.5), // seed edges start with moderate confidence
                    query_count: 0,
                    is_seed: true,
                });
            }
        }
    }

    /// Pick the next edge to explore (highest curiosity).
    pub fn next_frontier_edge(&mut self) -> Option<FrontierEdge> {
        if self.frontier.is_empty() || self.queries_executed >= self.budget {
            return None;
        }
        // Sort by curiosity descending
        self.frontier
            .sort_by(|a, b| b.curiosity().partial_cmp(&a.curiosity()).unwrap());
        Some(self.frontier[0].clone())
    }

    /// Pick the next edge, steered by MUL — the exploration gateway
    /// (`E-MUL-EXPLORATION-GATEWAY-1`). Ranks the frontier by
    /// [`FrontierEdge::curiosity_mul`] (base curiosity weighted by the graph's
    /// own uncertainty + self-assessment) instead of the MUL-blind
    /// [`curiosity`](FrontierEdge::curiosity). When MUL is neutral and the graph
    /// is quiet this returns the SAME edge as [`next_frontier_edge`](Self::next_frontier_edge)
    /// (`G-CM-1`).
    pub fn next_frontier_edge_mul(
        &mut self,
        assessment: &MulAssessment,
        signals: &GraphSignals,
    ) -> Option<FrontierEdge> {
        if self.frontier.is_empty() || self.queries_executed >= self.budget {
            return None;
        }
        self.frontier.sort_by(|a, b| {
            b.curiosity_mul(assessment, signals)
                .partial_cmp(&a.curiosity_mul(assessment, signals))
                .unwrap()
        });
        Some(self.frontier[0].clone())
    }

    /// Process exploration results: revise existing edges, add new ones.
    pub fn process_results(&mut self, query_edge: &FrontierEdge, result: ExplorationResult) {
        self.queries_executed += 1;

        // Mark the queried edge
        if let Some(fe) = self.frontier.iter_mut().find(|e| {
            e.source == query_edge.source
                && e.target == query_edge.target
                && e.label == query_edge.label
        }) {
            fe.query_count += 1;
        }

        // Process confirmations
        for (s, t, l, new_truth) in &result.confirmed {
            if let Some(fe) = self
                .frontier
                .iter_mut()
                .find(|e| &e.source == s && &e.target == t && &e.label == l)
            {
                fe.truth = fe.truth.revision(new_truth);
                self.confirmations += 1;
            }
        }

        // Process denials
        for (s, t, l, denial_truth) in &result.denied {
            if let Some(fe) = self
                .frontier
                .iter_mut()
                .find(|e| &e.source == s && &e.target == t && &e.label == l)
            {
                fe.truth = fe.truth.revision(denial_truth);
                self.denials += 1;
            }
        }

        // Add newly discovered edges
        for new_edge in result.new_edges {
            // Check if already in frontier
            let exists = self.frontier.iter().any(|e| {
                e.source == new_edge.source
                    && e.target == new_edge.target
                    && e.label == new_edge.label
            });
            if !exists {
                // Add to graph
                let source_id = new_edge.source.clone();
                let target_id = new_edge.target.clone();
                // Ensure nodes exist
                self.graph.add_node(LiteralNode {
                    id: source_id.clone(),
                    name: source_id.clone(),
                    label: "Discovered".into(),
                    props: vec![],
                    dn: 0,
                });
                self.graph.add_node(LiteralNode {
                    id: target_id.clone(),
                    name: target_id.clone(),
                    label: "Discovered".into(),
                    props: vec![],
                    dn: 0,
                });
                self.graph.add_edge(LiteralEdge {
                    source: source_id,
                    target: target_id,
                    label: new_edge.label.clone(),
                    weight: None,
                    reference: None,
                });
                self.frontier.push(new_edge);
                self.edges_discovered += 1;
            }
        }
    }

    /// Simulate processing text from a web fetch (for testing without network).
    /// Extracts triplets and returns ExplorationResult.
    pub fn extract_from_text(&self, text: &str, query_edge: &FrontierEdge) -> ExplorationResult {
        let mut new_edges = Vec::new();
        let mut confirmed = Vec::new();

        // Simple verb-based extraction (same as OSINT extractor)
        for sentence in text.split(['.', '!', '?']) {
            let sentence = sentence.trim();
            if sentence.split_whitespace().count() < 3 {
                continue;
            }

            let words: Vec<&str> = sentence.split_whitespace().collect();
            // Find verb
            let verb_idx = words.iter().position(|w| is_likely_verb(w));
            if let Some(vi) = verb_idx {
                if vi == 0 || vi >= words.len() - 1 {
                    continue;
                }
                let subj = words[..vi].join(" ").to_lowercase();
                let verb = words[vi].to_lowercase();
                let obj = words[vi + 1..].join(" ").to_lowercase();

                if subj.is_empty() || obj.is_empty() {
                    continue;
                }

                // Check if this confirms the query edge
                let subj_matches = subj.contains(&query_edge.source.to_lowercase())
                    || query_edge.source.to_lowercase().contains(&subj);
                let obj_matches = obj.contains(&query_edge.target.to_lowercase())
                    || query_edge.target.to_lowercase().contains(&obj);

                if subj_matches && obj_matches {
                    confirmed.push((
                        query_edge.source.clone(),
                        query_edge.target.clone(),
                        query_edge.label.clone(),
                        NarsTruth::new(0.9, 0.6),
                    ));
                } else {
                    // New edge discovered
                    new_edges.push(FrontierEdge {
                        source: clean_entity(&subj),
                        target: clean_entity(&obj),
                        label: verb,
                        truth: NarsTruth::new(0.7, 0.3), // lower confidence for discovered edges
                        query_count: 0,
                        is_seed: false,
                    });
                }
            }
        }

        ExplorationResult {
            new_edges,
            confirmed,
            denied: vec![],
        }
    }

    /// Monitoring snapshot.
    pub fn stats(&self) -> ExplorationStats {
        let frontier_size = self.frontier.len();
        let avg_curiosity = if frontier_size > 0 {
            self.frontier.iter().map(|e| e.curiosity()).sum::<f32>() / frontier_size as f32
        } else {
            0.0
        };
        let crystallized = self
            .frontier
            .iter()
            .filter(|e| e.truth.confidence > 0.8)
            .count();
        let hot = self
            .frontier
            .iter()
            .filter(|e| e.truth.confidence < 0.3)
            .count();

        ExplorationStats {
            graph_nodes: self.graph.node_count(),
            graph_edges: self.graph.edge_count(),
            frontier_size,
            queries_executed: self.queries_executed,
            edges_discovered: self.edges_discovered,
            confirmations: self.confirmations,
            denials: self.denials,
            avg_curiosity,
            crystallized_edges: crystallized,
            hot_edges: hot,
            budget_remaining: self.budget.saturating_sub(self.queries_executed),
        }
    }
}

/// Exploration statistics for cognitive monitoring.
#[derive(Debug, Clone)]
pub struct ExplorationStats {
    pub graph_nodes: usize,
    pub graph_edges: usize,
    pub frontier_size: usize,
    pub queries_executed: usize,
    pub edges_discovered: usize,
    pub confirmations: usize,
    pub denials: usize,
    pub avg_curiosity: f32,
    pub crystallized_edges: usize,
    pub hot_edges: usize,
    pub budget_remaining: usize,
}

impl core::fmt::Display for ExplorationStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "Graph: {} nodes, {} edges",
            self.graph_nodes, self.graph_edges
        )?;
        writeln!(
            f,
            "Frontier: {} edges (curiosity avg: {:.3})",
            self.frontier_size, self.avg_curiosity
        )?;
        writeln!(
            f,
            "Queries: {}/{} budget",
            self.queries_executed,
            self.queries_executed + self.budget_remaining
        )?;
        writeln!(
            f,
            "Discovered: {} new edges, {} confirmations, {} denials",
            self.edges_discovered, self.confirmations, self.denials
        )?;
        writeln!(
            f,
            "Crystallized: {}, Hot: {}",
            self.crystallized_edges, self.hot_edges
        )
    }
}

fn is_likely_verb(word: &str) -> bool {
    let w = word.to_lowercase();
    w.ends_with("ed")
        || w.ends_with("es")
        || w.ends_with("ing")
        || matches!(
            w.as_str(),
            "is" | "are"
                | "was"
                | "were"
                | "has"
                | "had"
                | "does"
                | "did"
                | "can"
                | "will"
                | "may"
                | "must"
                | "shall"
                | "developed"
                | "created"
                | "founded"
                | "owns"
                | "invested"
                | "contracts"
                | "employed"
                | "provides"
                | "used"
                | "built"
                | "acquired"
                | "funded"
                | "operates"
                | "manages"
                | "controls"
        )
}

fn clean_entity(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_alphanumeric() && c != ' ' && c != '-')
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal_graph::ingest_aiwar_json;

    #[test]
    fn test_pearl_queries() {
        let edge = LiteralEdge {
            source: "Palantir".into(),
            target: "Gotham".into(),
            label: "developed".into(),
            weight: None,
            reference: None,
        };
        let queries = pearl_queries(&edge);
        assert_eq!(queries.len(), 3);
        assert_eq!(queries[0].level, PearlLevel::See);
        assert!(queries[0].text.contains("Palantir"));
        assert!(queries[0].text.contains("Gotham"));
        assert_eq!(queries[1].level, PearlLevel::Do);
        assert_eq!(queries[2].level, PearlLevel::Imagine);
    }

    #[test]
    fn test_nars_revision() {
        let a = NarsTruth::new(0.8, 0.5);
        let b = NarsTruth::new(0.9, 0.6);
        let merged = a.revision(&b);
        assert!(
            merged.confidence > a.confidence,
            "confidence should increase"
        );
        assert!(merged.frequency > 0.8 && merged.frequency < 0.95);
    }

    #[test]
    fn test_curiosity_ordering() {
        let queried = FrontierEdge {
            source: "A".into(),
            target: "B".into(),
            label: "x".into(),
            truth: NarsTruth::new(0.9, 0.8),
            query_count: 5,
            is_seed: true,
        };
        let fresh = FrontierEdge {
            source: "C".into(),
            target: "D".into(),
            label: "y".into(),
            truth: NarsTruth::new(0.5, 0.2),
            query_count: 0,
            is_seed: false,
        };
        assert!(
            fresh.curiosity() > queried.curiosity(),
            "un-queried low-confidence edge should have higher curiosity"
        );
    }

    #[test]
    fn test_extract_from_text() {
        let graph = LiteralGraph::new();
        let explorer = MassExplorer {
            graph,
            frontier: vec![],
            budget: 100,
            queries_executed: 0,
            edges_discovered: 0,
            confirmations: 0,
            denials: 0,
        };

        let query_edge = FrontierEdge {
            source: "Palantir".into(),
            target: "Gotham".into(),
            label: "developed".into(),
            truth: NarsTruth::prior(),
            query_count: 0,
            is_seed: true,
        };

        let text = "Palantir developed Gotham for the CIA. The CIA funded the project through In-Q-Tel. NSA also adopted Gotham for signals intelligence.";
        let result = explorer.extract_from_text(text, &query_edge);

        eprintln!("Confirmed: {:?}", result.confirmed);
        eprintln!(
            "New edges: {:?}",
            result
                .new_edges
                .iter()
                .map(|e| format!("{} -{}- {}", e.source, e.label, e.target))
                .collect::<Vec<_>>()
        );

        assert!(
            !result.confirmed.is_empty() || !result.new_edges.is_empty(),
            "should extract at least one confirmation or new edge"
        );
    }

    #[test]
    fn test_mass_exploration_simulation() {
        // Simulate exploration without network using synthetic text responses
        let json = r#"{
            "Schema": [], "N_Civic": [], "N_Historical": [], "N_People": [],
            "N_Systems": [
                {"id": "Gotham", "name": "Palantir Gotham", "type": "DataManagement"},
                {"id": "Foundry", "name": "Palantir Foundry", "type": "DataManagement"}
            ],
            "N_Stakeholders": [
                {"id": "Palantir", "name": "Palantir", "type": "TechCompany"},
                {"id": "CIA", "name": "CIA", "type": "Institution"},
                {"id": "NSA", "name": "NSA", "type": "Institution"}
            ],
            "E_connection": [
                {"source": "CIA", "target": "NSA", "label": "part of"}
            ],
            "E_isDevelopedBy": [
                {"source": "Palantir", "target": "Gotham", "label": "developed"},
                {"source": "Palantir", "target": "Foundry", "label": "developed"}
            ],
            "E_isDeployedBy": [
                {"source": "CIA", "target": "Gotham", "label": "employed"}
            ],
            "E_place": [], "E_people": [], "E_hierarchical": []
        }"#;

        let graph = ingest_aiwar_json(json).unwrap();
        let initial_nodes = graph.node_count();
        let initial_edges = graph.edge_count();

        let mut explorer = MassExplorer {
            graph,
            frontier: vec![],
            budget: 10,
            queries_executed: 0,
            edges_discovered: 0,
            confirmations: 0,
            denials: 0,
        };
        explorer.seed_frontier();

        eprintln!("\n══════════════════════════════════════════════════════════");
        eprintln!("  Mass Exploration Simulation");
        eprintln!("══════════════════════════════════════════════════════════");
        eprintln!(
            "Seed: {} nodes, {} edges, {} frontier",
            initial_nodes,
            initial_edges,
            explorer.frontier.len()
        );

        // Simulate 5 exploration rounds with synthetic "web results"
        let synthetic_responses = [
            "Palantir developed Gotham originally for the CIA under project In-Q-Tel. The system was later adopted by FBI and DHS.",
            "CIA funded Palantir through In-Q-Tel venture capital arm. Peter Thiel invested $30M in Palantir.",
            "NSA uses Gotham for signals intelligence. GCHQ also adopted the platform.",
            "Palantir Foundry processes data for BP and Airbus. NHS contracted Foundry during COVID.",
            "Peter Thiel founded Palantir with Alex Karp. Thiel also invested in Facebook and SpaceX.",
        ];

        for (round, response) in synthetic_responses.iter().enumerate() {
            if let Some(edge) = explorer.next_frontier_edge() {
                eprintln!(
                    "\nRound {}: querying ({} -{}- {}), curiosity={:.3}",
                    round + 1,
                    edge.source,
                    edge.label,
                    edge.target,
                    edge.curiosity()
                );

                let result = explorer.extract_from_text(response, &edge);
                let new_count = result.new_edges.len();
                let conf_count = result.confirmed.len();

                explorer.process_results(&edge, result);

                eprintln!("  → {} new edges, {} confirmations", new_count, conf_count);
            }
        }

        let stats = explorer.stats();
        eprintln!("\n{}", stats);

        eprintln!("══════════════════════════════════════════════════════════");
        eprintln!(
            "  EXPANSION: {} → {} nodes, {} → {} edges",
            initial_nodes, stats.graph_nodes, initial_edges, stats.graph_edges
        );
        eprintln!("══════════════════════════════════════════════════════════\n");

        assert!(
            stats.graph_nodes > initial_nodes,
            "graph should have grown: {} → {}",
            initial_nodes,
            stats.graph_nodes
        );
        assert!(
            stats.edges_discovered > 0,
            "should have discovered new edges"
        );
    }

    // ── D-SCI-4a: curiosity_mul + qualia texture gestalt (G-CM-1..5) ──
    use crate::mul::{Homeostasis, MulAssessment, TrustQualia};

    fn edge(conf: f32, queries: u32) -> FrontierEdge {
        FrontierEdge {
            source: "a".into(),
            target: "b".into(),
            label: "rel".into(),
            truth: NarsTruth::new(0.9, conf),
            query_count: queries,
            is_seed: true,
        }
    }

    fn assess(
        dk: DkPosition,
        trust: TrustTexture,
        flow: FlowState,
        free_will: f64,
    ) -> MulAssessment {
        MulAssessment {
            trust: TrustQualia {
                value: 1.0,
                texture: trust,
            },
            dk_position: dk,
            homeostasis: Homeostasis {
                flow_state: flow,
                allostatic_load: 0.0,
            },
            complexity_mapped: true,
            free_will_modifier: free_will,
        }
    }

    fn neutral() -> MulAssessment {
        assess(
            DkPosition::Plateau,
            TrustTexture::Calibrated,
            FlowState::Flow,
            1.0,
        )
    }

    /// G-CM-1 — the anti-vacuity falsifier: with a neutral assessment and a quiet
    /// graph, `curiosity_mul` reduces EXACTLY to `curiosity()`.
    #[test]
    fn g_cm_1_neutral_identity() {
        let e = edge(0.3, 2);
        let m = e.curiosity_mul(&neutral(), &GraphSignals::default());
        assert!(
            (m - e.curiosity()).abs() < 1e-6,
            "neutral MUL must be inert: {m} vs {}",
            e.curiosity()
        );
    }

    /// G-CM-2 — Staunen boost: graph-wide surprise strictly raises curiosity.
    #[test]
    fn g_cm_2_staunen_boost() {
        let e = edge(0.3, 2);
        let surprised = GraphSignals {
            truth_entropy: 0.8,
            contradiction_rate: 0.4,
            ..GraphSignals::default()
        };
        let m = e.curiosity_mul(&neutral(), &surprised);
        assert!(
            m > e.curiosity(),
            "surprise must pull exploration up: {m} !> {}",
            e.curiosity()
        );
    }

    /// G-CM-3 — humility gate: MountStupid + Overconfident strictly lowers it.
    #[test]
    fn g_cm_3_humility_gate() {
        let e = edge(0.3, 2);
        let overconfident = assess(
            DkPosition::MountStupid,
            TrustTexture::Overconfident,
            FlowState::Flow,
            1.0,
        );
        let m = e.curiosity_mul(&overconfident, &GraphSignals::default());
        assert!(
            m < e.curiosity(),
            "over-confidence must dampen exploration: {m} !< {}",
            e.curiosity()
        );
    }

    /// G-CM-4 — ground gate dampens but NEVER hits zero (reversibility, not
    /// paralysis); softer ground ranks below firm ground.
    #[test]
    fn g_cm_4_ground_gate_never_zero() {
        let e = edge(0.3, 2);
        let worst = assess(
            DkPosition::MountStupid,
            TrustTexture::Overconfident,
            FlowState::Anxiety,
            0.5,
        );
        let m = e.curiosity_mul(&worst, &GraphSignals::default());
        assert!(m > 0.0, "ground gate must never zero a positive edge: {m}");

        let uncertain = e.curiosity_mul(
            &assess(
                DkPosition::Plateau,
                TrustTexture::Uncertain,
                FlowState::Flow,
                1.0,
            ),
            &GraphSignals::default(),
        );
        let calibrated = e.curiosity_mul(&neutral(), &GraphSignals::default());
        assert!(
            uncertain < calibrated,
            "softer ground must rank below firm ground: {uncertain} !< {calibrated}"
        );
    }

    /// G-CM-5 — texture fidelity: the emitted qualia axes track their sources.
    /// (Recorded finding: the qualia `wonder = √(coherence·expansion)` is
    /// coherent-NOVELTY wonder, NOT Staunen; Staunen — surprise-wonder — is
    /// carried by `arousal`/`entropy`/`tension`, which is what this asserts.)
    #[test]
    fn g_cm_5_texture_fidelity() {
        let e = edge(0.2, 0); // uncertainty 0.8, novelty 1.0
        let quiet = e.curiosity_gestalt(&neutral(), &GraphSignals::default());
        let loud = e.curiosity_gestalt(
            &neutral(),
            &GraphSignals {
                truth_entropy: 0.9,
                contradiction_rate: 0.6,
                revision_velocity: 0.5,
                ..GraphSignals::default()
            },
        );
        // expansion == novelty
        assert!((quiet.texture[15] - 1.0).abs() < 1e-6, "expansion=novelty");
        // tension == contradiction_rate
        assert!((loud.texture[2] - 0.6).abs() < 1e-6, "tension=contradiction");
        // entropy rises with truth_entropy
        assert!(loud.texture[8] > quiet.texture[8], "entropy tracks truth_entropy");
        // arousal (Staunen activation) rises with surprise
        assert!(loud.texture[0] > quiet.texture[0], "arousal tracks Staunen");
        // velocity tracks revision_velocity
        assert!((loud.texture[7] - 0.5).abs() < 1e-6, "velocity tracks revision");
        // groundedness: Uncertain ground < Calibrated ground
        let g_unc = e
            .curiosity_gestalt(
                &assess(
                    DkPosition::Plateau,
                    TrustTexture::Uncertain,
                    FlowState::Flow,
                    1.0,
                ),
                &GraphSignals::default(),
            )
            .texture[14];
        assert!(g_unc < quiet.texture[14], "groundedness tracks trust texture");
    }

    /// The MUL-steered pick matches the MUL-blind pick when MUL is neutral
    /// (the frontier ordering is unchanged until MUL has something to say).
    #[test]
    fn next_frontier_edge_mul_matches_blind_when_neutral() {
        use crate::literal_graph::LiteralGraph;
        let mut ex = MassExplorer::from_graph(LiteralGraph::new(), 100);
        ex.frontier = vec![edge(0.9, 5), edge(0.2, 0), edge(0.5, 1)];
        let blind = ex.next_frontier_edge().unwrap();
        let steered = ex
            .next_frontier_edge_mul(&neutral(), &GraphSignals::default())
            .unwrap();
        assert_eq!(
            (blind.source, blind.target, blind.query_count),
            (steered.source, steered.target, steered.query_count),
            "neutral MUL must not change the pick"
        );
    }
}
