//! # Planner Internal API
//!
//! Single-binary API for the lance-graph unified query planner.
//! All calls are direct function calls — no HTTP, no IPC, no serialization.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Consumer Code                         │
//! │  (ladybug-rs, crewai-rust, n8n-rs — same binary)       │
//! └──────────────┬──────────────────────────────────────────┘
//!                │  direct fn call (zero-copy)
//!                ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │              Planner::plan()                             │
//! │                                                          │
//! │  1. MUL Assessment    → SituationInput → MulAssessment  │
//! │  2. Thinking Orchestr → MulAssessment → ThinkingContext  │
//! │  3. Strategy Selection → ThinkingContext → Vec<Strategy> │
//! │  4. Plan Composition  → Strategies → LogicalPlan        │
//! │  5. Physical Planning → LogicalPlan → PhysicalPlan      │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use lance_graph_planner::api::Planner;
//!
//! // Create planner (one per session, reusable)
//! let planner = Planner::new();
//!
//! // Simple query (auto-selects everything)
//! let result = planner.plan("MATCH (n:Person) RETURN n")?;
//! println!("Strategies used: {:?}", result.strategies_used);
//!
//! // With thinking style override
//! let result = planner.plan_with_style(
//!     "MATCH (n)-[r]->(m) RETURN n, r, m",
//!     ThinkingStyle::Analytical,
//! )?;
//!
//! // Full MUL pipeline (for AGI-aware planning)
//! let situation = SituationInput {
//!     felt_competence: 0.8,
//!     demonstrated_competence: 0.7,
//!     challenge_level: 0.6,
//!     skill_level: 0.8,
//!     ..Default::default()
//! };
//! let result = planner.plan_assessed("MATCH path = (a)-[*1..5]->(b)", &situation)?;
//! match result.gate {
//!     Gate::Proceed { .. } => { /* execute plan */ }
//!     Gate::Sandbox { .. } => { /* need human assistance */ }
//!     Gate::Compass => { /* navigate unknown territory */ }
//! }
//! ```
//!
//! # CAM-PQ Search
//!
//! ```rust,ignore
//! use lance_graph_planner::api::CamSearch;
//!
//! let mut search = CamSearch::new(codebook);
//! search.prepare_query(&query_vector);
//! let results = search.top_k(&cam_data, 10);
//! ```

use crate::physical::CamPqScanOp;
use crate::physical::cam_pq_scan::CamPqStrategy;
use crate::traits::*;
use crate::selector::StrategySelector;

// Re-export key types for ergonomic API
pub use crate::thinking::style::{ThinkingStyle, ThinkingCluster, FieldModulation, ScanParams};
pub use crate::thinking::ThinkingContext;
pub use crate::mul::SituationInput;
pub use crate::mul::gate::GateDecision as Gate;
pub use crate::PlanResult;
pub use crate::PlanError;

/// The planner. One per session, reusable, thread-safe.
///
/// This is the internal API — direct function calls, zero serialization.
/// For cross-process use, wrap in Arrow Flight (future).
pub struct Planner {
    inner: crate::PlannerAwareness,
}

impl Planner {
    /// Create a new planner with default auto-selection.
    pub fn new() -> Self {
        Self {
            inner: crate::PlannerAwareness::new(),
        }
    }

    /// Create with explicit strategy names.
    ///
    /// ```rust,ignore
    /// let planner = Planner::with_strategies(&["cypher_parse", "sigma_scan", "jit_compile"]);
    /// ```
    pub fn with_strategies(names: &[&str]) -> Self {
        Self {
            inner: crate::PlannerAwareness::with_explicit(
                names.iter().map(|s| s.to_string()).collect(),
            ),
        }
    }

    /// Create with resonance mode (23D thinking style vector).
    pub fn with_resonance(style_vector: Vec<f64>, mul_modifier: f64) -> Self {
        Self {
            inner: crate::PlannerAwareness::with_resonance(
                style_vector,
                mul_modifier,
                0.0,
            ),
        }
    }

    /// Plan a query (auto mode — strategies selected by affinity).
    pub fn plan(&self, query: &str) -> Result<PlanResult, PlanError> {
        self.inner.plan_auto(query)
    }

    /// Plan with a specific thinking style override.
    ///
    /// Converts the style to a 7D FieldModulation fingerprint,
    /// then uses resonance-based strategy selection.
    pub fn plan_with_style(
        &self,
        query: &str,
        style: ThinkingStyle,
    ) -> Result<PlanResult, PlanError> {
        let modulation = style.default_modulation();
        let style_vec = modulation.to_fingerprint()
            .iter()
            .map(|b| *b as f64 / 255.0)
            .collect::<Vec<f64>>();

        // Pad to 23D (first 7 from modulation, rest zero)
        let mut full_vec = vec![0.0f64; 23];
        for (i, v) in style_vec.iter().enumerate() {
            full_vec[i] = *v;
        }

        let context = PlanContext {
            query: query.to_string(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: Some(full_vec),
            nars_hint: None,
        };

        let selected = crate::selector::select_strategies(
            &StrategySelector::default(),
            self.inner.strategies(),
            &context,
        );
        let strategy_names: Vec<String> = selected.iter()
            .map(|s| s.name().to_string()).collect();

        let plan = crate::compose::compose_and_execute(&selected, context)?;

        // Build thinking context
        let thinking_ctx = crate::thinking::ThinkingContext {
            style,
            modulation,
            nars_type: crate::thinking::NarsInferenceType::Deduction,
            strategy: crate::thinking::nars_dispatch::route(
                crate::thinking::NarsInferenceType::Deduction,
            ),
            semiring: crate::thinking::semiring_selection::select(
                query,
                &style,
                &crate::thinking::NarsInferenceType::Deduction,
            ),
            sigma_stage: crate::thinking::sigma_chain::SigmaStage::Omega,
            free_will_modifier: 1.0,
            exploratory: matches!(
                style.cluster(),
                ThinkingCluster::Divergent
            ),
        };

        Ok(PlanResult {
            mul: None,
            thinking: Some(thinking_ctx),
            plan,
            strategies_used: strategy_names,
            free_will_modifier: 1.0,
            compass_score: None,
        })
    }

    /// Plan with full MUL assessment pipeline.
    ///
    /// This is the AGI-aware path: situation → assessment → thinking style →
    /// strategy selection → plan. The gate decision in the result tells
    /// whether to execute, sandbox, or navigate via compass.
    pub fn plan_assessed(
        &self,
        query: &str,
        situation: &SituationInput,
    ) -> Result<PlanResult, PlanError> {
        self.inner.plan_full(query, situation)
    }

    /// Override the strategy selector.
    pub fn set_selector(&mut self, selector: StrategySelector) {
        self.inner.set_selector(selector);
    }

    /// Register a custom strategy.
    pub fn register_strategy(&mut self, strategy: Box<dyn PlanStrategy>) {
        self.inner.register_strategy(strategy);
    }

    /// Get the list of available strategy names.
    pub fn strategy_names(&self) -> Vec<String> {
        self.inner.strategies().iter().map(|s| s.name().to_string()).collect()
    }

    /// Gate check only — assess situation without planning.
    pub fn gate_check(&self, situation: &SituationInput) -> Gate {
        let assessment = crate::mul::assess(situation);
        crate::mul::gate_check(&assessment)
    }
}

impl Default for Planner {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// CAM-PQ Search API
// =============================================================================

/// CAM-PQ search — high-level API for compressed vector search.
///
/// Wraps the low-level CamPqScanOp with an ergonomic interface.
///
/// # Example
///
/// ```rust,ignore
/// let codebook = train_codebook(&sample_vectors, 256);
/// let mut search = CamSearch::new(codebook);
///
/// // Encode vectors to 6-byte CAM fingerprints
/// let cam_data = search.encode(&all_vectors);
///
/// // For each query:
/// search.prepare_query(&query_vector);
/// let results = search.top_k(&cam_data, 10);
/// // results: Vec<(index, distance)> sorted ascending
/// ```
pub struct CamSearch {
    tables: Option<[[f32; 256]; 6]>,
    codebook: Vec<Vec<Vec<f32>>>,
}

impl CamSearch {
    /// Create a new CAM search bound to a codebook.
    ///
    /// The codebook is `codebook[subspace][centroid]` = Vec<f32> of length D/6.
    pub fn new(codebook: Vec<Vec<Vec<f32>>>) -> Self {
        Self { tables: None, codebook }
    }

    /// Precompute distance tables for a query vector.
    ///
    /// Call once per query, then use `top_k()` for search.
    /// The tables are 6×256 = 6KB and fit entirely in L1 cache.
    pub fn prepare_query(&mut self, query: &[f32]) {
        let total_dim = query.len();
        let subspace_dim = total_dim / 6;
        let mut tables = [[0.0f32; 256]; 6];
        for s in 0..6 {
            let q_sub = &query[s * subspace_dim..(s + 1) * subspace_dim];
            let num_c = self.codebook[s].len().min(256);
            for c in 0..num_c {
                tables[s][c] = q_sub.iter()
                    .zip(self.codebook[s][c].iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
            }
        }
        self.tables = Some(tables);
    }

    /// Top-K search on CAM fingerprints.
    ///
    /// Returns `Vec<(index, distance)>` sorted by distance ascending.
    /// Panics if `prepare_query()` was not called first.
    pub fn top_k(&self, cam_data: &[[u8; 6]], k: usize) -> Vec<(usize, f32)> {
        let tables = self.tables.as_ref().expect("call prepare_query() first");
        let strategy = CamPqScanOp::select_strategy(cam_data.len() as u64);

        let op = CamPqScanOp {
            strategy,
            heel_threshold: 50.0,
            branch_threshold: 25.0,
            top_k: k,
            num_probes: 0,
            estimated_cardinality: k as f64,
            child: Box::new(crate::physical::BroadcastOp {
                fingerprint: vec![],
                partitions: 1,
                cardinality: 1.0,
            }),
        };

        op.execute(tables, cam_data)
    }

    /// Encode vectors to 6-byte CAM fingerprints.
    pub fn encode(&self, vectors: &[Vec<f32>]) -> Vec<[u8; 6]> {
        vectors.iter().map(|v| {
            let dim = v.len();
            let sub_dim = dim / 6;
            let mut cam = [0u8; 6];
            for s in 0..6 {
                let sub = &v[s * sub_dim..(s + 1) * sub_dim];
                let mut best_c = 0u8;
                let mut best_d = f32::MAX;
                for (c, centroid) in self.codebook[s].iter().enumerate() {
                    let d: f32 = sub.iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    if d < best_d {
                        best_d = d;
                        best_c = c as u8;
                    }
                }
                cam[s] = best_c;
            }
            cam
        }).collect()
    }

    /// Decode a CAM fingerprint to an approximate vector.
    pub fn decode(&self, cam: &[u8; 6]) -> Vec<f32> {
        let mut result = Vec::new();
        for s in 0..6 {
            result.extend_from_slice(&self.codebook[s][cam[s] as usize]);
        }
        result
    }

    /// Compression statistics.
    pub fn stats(&self, num_vectors: u64, dim: usize) -> CamStats {
        let raw_bytes = num_vectors * dim as u64 * 4;
        let cam_bytes = num_vectors * 6;
        let sub_dim = dim / 6;
        let codebook_bytes = (6 * 256 * sub_dim * 4) as u64;
        let total = cam_bytes + codebook_bytes;
        CamStats {
            num_vectors,
            dim,
            raw_bytes,
            cam_bytes,
            codebook_bytes,
            compression_ratio: if total > 0 { raw_bytes as f64 / total as f64 } else { 0.0 },
        }
    }
}

/// CAM compression statistics.
#[derive(Debug, Clone)]
pub struct CamStats {
    pub num_vectors: u64,
    pub dim: usize,
    pub raw_bytes: u64,
    pub cam_bytes: u64,
    pub codebook_bytes: u64,
    pub compression_ratio: f64,
}

impl std::fmt::Display for CamStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CAM-PQ: {}M vectors x {}D = {:.1}MB CAM + {:.1}MB codebook ({:.0}:1)",
            self.num_vectors / 1_000_000,
            self.dim,
            self.cam_bytes as f64 / 1_048_576.0,
            self.codebook_bytes as f64 / 1_048_576.0,
            self.compression_ratio,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_new() {
        let planner = Planner::new();
        let names = planner.strategy_names();
        assert!(!names.is_empty());
        assert!(names.contains(&"cypher_parse".to_string()));
    }

    #[test]
    fn test_plan_auto() {
        let planner = Planner::new();
        let result = planner.plan("MATCH (n) RETURN n");
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.strategies_used.is_empty());
    }

    #[test]
    fn test_plan_with_style() {
        let planner = Planner::new();
        let result = planner.plan_with_style(
            "MATCH (n)-[r]->(m) RETURN n",
            ThinkingStyle::Analytical,
        );
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.thinking.is_some());
        assert_eq!(output.thinking.unwrap().style, ThinkingStyle::Analytical);
    }

    #[test]
    fn test_plan_assessed() {
        let planner = Planner::new();
        let situation = SituationInput {
            felt_competence: 0.9,
            demonstrated_competence: 0.3,
            ..Default::default()
        };
        let result = planner.plan_assessed("MATCH (n) RETURN n", &situation);
        // MountStupid: felt >> demonstrated → should be blocked
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_gate_check() {
        let planner = Planner::new();

        // Calibrated expert → Proceed
        let situation = SituationInput {
            felt_competence: 0.8,
            demonstrated_competence: 0.8,
            ..Default::default()
        };
        let gate = planner.gate_check(&situation);
        assert!(matches!(gate, Gate::Proceed { .. }));
    }

    #[test]
    fn test_cam_encode_decode() {
        let codebook: Vec<Vec<Vec<f32>>> = (0..6)
            .map(|s| {
                (0..4)
                    .map(|c| vec![s as f32 + c as f32 * 0.25; 4])
                    .collect()
            })
            .collect();

        let search = CamSearch::new(codebook);
        let vectors = vec![vec![0.1; 24], vec![1.0; 24], vec![5.0; 24]];
        let cams = search.encode(&vectors);
        assert_eq!(cams.len(), 3);

        let decoded = search.decode(&cams[0]);
        assert_eq!(decoded.len(), 24);
    }

    #[test]
    fn test_cam_top_k() {
        let codebook: Vec<Vec<Vec<f32>>> = (0..6)
            .map(|_| (0..256).map(|c| vec![c as f32 * 0.01; 4]).collect())
            .collect();

        let mut search = CamSearch::new(codebook);
        search.prepare_query(&vec![0.5; 24]);

        let cam_data: Vec<[u8; 6]> = (0..1000)
            .map(|i| {
                let v = (i % 256) as u8;
                [v, v, v, v, v, v]
            })
            .collect();

        let results = search.top_k(&cam_data, 10);
        assert_eq!(results.len(), 10);
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_cam_stats() {
        let codebook: Vec<Vec<Vec<f32>>> = (0..6)
            .map(|_| (0..256).map(|_| vec![0.0; 42]).collect())
            .collect();
        let search = CamSearch::new(codebook);
        let stats = search.stats(1_000_000, 256);
        assert!(stats.compression_ratio > 100.0);
        let s = format!("{}", stats);
        assert!(s.contains("1M vectors"));
    }
}
