//! # Thinking Orchestration (Middle Layer)
//!
//! "How should I think about this?"
//!
//! Before the query planner fires, thinking orchestration decides:
//! 1. Which thinking style to use (36 styles in 6 clusters)
//! 2. Which NARS inference type maps to this query
//! 3. Which semiring algebra the planner should use
//! 4. Field modulation parameters (thresholds, fan-out, depth)
//!
//! The sigma chain (Ω→Δ→Φ→Θ→Λ) tracks the epistemic lifecycle
//! of each thinking atom through the planning process.

pub mod nars_dispatch;
pub mod semiring_selection;
pub mod sigma_chain;
pub mod style;

pub use nars_dispatch::{NarsInferenceType, QueryStrategy};
pub use semiring_selection::SemiringChoice;
pub use sigma_chain::{SigmaStage, ThinkingAtom};
pub use style::{FieldModulation, ThinkingCluster, ThinkingStyle};

use crate::mul::MulAssessment;
use crate::plan::PlannerConfig;

/// Complete thinking context produced by orchestration.
/// This is what the query planner receives as input alongside the query.
#[derive(Debug, Clone)]
pub struct ThinkingContext {
    /// Selected thinking style.
    pub style: ThinkingStyle,
    /// Field modulation parameters (control planner behavior).
    pub modulation: FieldModulation,
    /// NARS inference type for this query.
    pub nars_type: NarsInferenceType,
    /// Query strategy derived from NARS type.
    pub strategy: QueryStrategy,
    /// Semiring choice for graph traversal.
    pub semiring: SemiringChoice,
    /// Current sigma chain stage.
    pub sigma_stage: SigmaStage,
    /// Free will modifier (passed from MUL).
    pub free_will_modifier: f64,
    /// Whether we're in exploratory mode (from compass).
    pub exploratory: bool,
}

impl ThinkingContext {
    /// Force exploratory mode (called when compass says Exploratory).
    pub fn force_exploratory(&mut self) {
        self.exploratory = true;
        self.modulation.exploration = 1.0;
        self.modulation.noise_tolerance = 0.8;
        self.modulation.fan_out = 20;
        self.style = ThinkingStyle::Exploratory;
    }
}

/// Orchestrate thinking: select style, NARS type, semiring from query + MUL assessment.
pub fn orchestrate(query: &str, mul: &MulAssessment, _config: &PlannerConfig) -> ThinkingContext {
    // 1. Select thinking style from MUL state
    let style = style::select_from_mul(mul);
    let modulation = style.default_modulation();

    // 2. Detect NARS inference type from query shape
    let nars_type = nars_dispatch::detect_from_query(query);
    let strategy = nars_dispatch::route(nars_type);

    // 3. Select semiring from query + thinking style
    let semiring = semiring_selection::select(query, &style, &nars_type);

    // 4. Determine sigma chain stage
    let sigma_stage = SigmaStage::Omega; // Start at observation

    ThinkingContext {
        style,
        modulation,
        nars_type,
        strategy,
        semiring,
        sigma_stage,
        free_will_modifier: mul.free_will_modifier,
        exploratory: false,
    }
}
