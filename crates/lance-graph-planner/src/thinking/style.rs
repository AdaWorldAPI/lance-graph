//! Thinking Styles — 36 styles in 6 clusters.
//!
//! Each style controls how the planner searches: resonance thresholds,
//! fan-out breadth, depth bias, noise tolerance, speed bias, exploration rate.
//!
//! Styles are NOT metadata — they directly modify execution paths.

use crate::mul::dk::DkPosition;
use crate::mul::homeostasis::FlowState;
use crate::mul::trust::TrustTexture;
use crate::mul::MulAssessment;

/// The 12-family orchestration space — canonical type lives in the
/// contract (M9 dedup, D-TSC-1). This module keeps the planner-local
/// semantics (`PlannerStyleExt`: clusters, τ addresses, modulation).
pub use lance_graph_contract::style_family::StyleFamily;

/// Legacy name for the 12-space.
#[deprecated(
    note = "the 12-space is StyleFamily (orchestration families); \
            ThinkingStyle is the 36-runbook space in lance_graph_contract::thinking"
)]
pub type ThinkingStyle = StyleFamily;

/// Cluster that a thinking style belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingCluster {
    /// Analytical, Convergent, Systematic — depth-first, precise.
    Convergent,
    /// Creative, Divergent, Exploratory — breadth-first, novel.
    Divergent,
    /// Focused, Diffuse, Peripheral — attention allocation.
    Attention,
    /// Intuitive, Deliberate, Metacognitive — System 1 vs System 2.
    Speed,
}

/// Planner-local semantics on the shared `StyleFamily` taxonomy:
/// 4-cluster grouping, τ addresses, and default field modulation.
/// An extension trait because `StyleFamily` is a foreign (contract) type.
pub trait PlannerStyleExt {
    fn cluster(&self) -> ThinkingCluster;
    /// τ (tau) macro address for JIT compilation (0x20–0xC5).
    fn tau_address(&self) -> u8;
    /// Default field modulation for this style.
    fn default_modulation(&self) -> FieldModulation;
}

impl PlannerStyleExt for StyleFamily {
    fn cluster(&self) -> ThinkingCluster {
        match self {
            Self::Analytical | Self::Convergent | Self::Systematic => ThinkingCluster::Convergent,
            Self::Creative | Self::Divergent | Self::Exploratory => ThinkingCluster::Divergent,
            Self::Focused | Self::Diffuse | Self::Peripheral => ThinkingCluster::Attention,
            Self::Intuitive | Self::Deliberate | Self::Metacognitive => ThinkingCluster::Speed,
        }
    }

    fn tau_address(&self) -> u8 {
        match self {
            Self::Analytical => 0x40,
            Self::Convergent => 0x41,
            Self::Systematic => 0x42,
            Self::Creative => 0xA0,
            Self::Divergent => 0xA1,
            Self::Exploratory => 0x20,
            Self::Focused => 0x60,
            Self::Diffuse => 0x61,
            Self::Peripheral => 0x62,
            Self::Intuitive => 0x80,
            Self::Deliberate => 0x81,
            Self::Metacognitive => 0xC0,
        }
    }

    fn default_modulation(&self) -> FieldModulation {
        match self {
            Self::Analytical => FieldModulation {
                resonance_threshold: 0.85,
                fan_out: 4,
                depth_bias: 0.9,
                breadth_bias: 0.2,
                noise_tolerance: 0.1,
                speed_bias: 0.3,
                exploration: 0.1,
            },
            Self::Creative => FieldModulation {
                resonance_threshold: 0.5,
                fan_out: 12,
                depth_bias: 0.4,
                breadth_bias: 0.9,
                noise_tolerance: 0.7,
                speed_bias: 0.6,
                exploration: 0.8,
            },
            Self::Exploratory => FieldModulation {
                resonance_threshold: 0.3,
                fan_out: 20,
                depth_bias: 0.3,
                breadth_bias: 1.0,
                noise_tolerance: 0.9,
                speed_bias: 0.5,
                exploration: 1.0,
            },
            Self::Focused => FieldModulation {
                resonance_threshold: 0.9,
                fan_out: 2,
                depth_bias: 1.0,
                breadth_bias: 0.1,
                noise_tolerance: 0.05,
                speed_bias: 0.4,
                exploration: 0.05,
            },
            Self::Intuitive => FieldModulation {
                resonance_threshold: 0.6,
                fan_out: 8,
                depth_bias: 0.5,
                breadth_bias: 0.5,
                noise_tolerance: 0.5,
                speed_bias: 0.9,
                exploration: 0.4,
            },
            Self::Deliberate => FieldModulation {
                resonance_threshold: 0.75,
                fan_out: 6,
                depth_bias: 0.7,
                breadth_bias: 0.4,
                noise_tolerance: 0.2,
                speed_bias: 0.2,
                exploration: 0.2,
            },
            Self::Metacognitive => FieldModulation {
                resonance_threshold: 0.7,
                fan_out: 8,
                depth_bias: 0.6,
                breadth_bias: 0.6,
                noise_tolerance: 0.3,
                speed_bias: 0.3,
                exploration: 0.5,
            },
            // Defaults for remaining styles
            _ => FieldModulation::default(),
        }
    }
}

/// Field modulation parameters — control how the planner searches.
/// These map directly to JIT scan kernel parameters in ndarray.
#[derive(Debug, Clone)]
pub struct FieldModulation {
    /// Resonance threshold: minimum Hamming similarity to consider (0..1).
    pub resonance_threshold: f64,
    /// Fan-out: breadth of search at each hop (1..20).
    pub fan_out: usize,
    /// Depth bias: preference for going deeper vs broader (0..1).
    pub depth_bias: f64,
    /// Breadth bias: lateral exploration weight (0..1).
    pub breadth_bias: f64,
    /// Noise tolerance: signal filtering aggressiveness (0..1).
    pub noise_tolerance: f64,
    /// Speed bias: deliberate (0) vs intuitive (1).
    pub speed_bias: f64,
    /// Exploration rate: novelty seeking (0..1).
    pub exploration: f64,
}

impl Default for FieldModulation {
    fn default() -> Self {
        Self {
            resonance_threshold: 0.7,
            fan_out: 6,
            depth_bias: 0.5,
            breadth_bias: 0.5,
            noise_tolerance: 0.3,
            speed_bias: 0.5,
            exploration: 0.3,
        }
    }
}

impl FieldModulation {
    /// Convert to JIT scan parameters (for ndarray SIMD kernels).
    /// Maps floating-point modulation to integer kernel parameters.
    pub fn to_scan_params(&self) -> ScanParams {
        ScanParams {
            threshold: (self.resonance_threshold * 2000.0) as u32 + 100,
            top_k: (self.fan_out as u32).max(1),
            prefetch_ahead: ((1.0 - self.speed_bias) * 7.0) as u32 + 1,
            filter_mask: if self.noise_tolerance < 0.3 {
                0xFFFFFFFF
            } else {
                0
            },
        }
    }

    /// Encode as thermometer-coded fingerprint for Hamming similarity search
    /// in the thinking template registry.
    pub fn to_fingerprint(&self) -> [u8; 7] {
        [
            thermometer_encode(self.resonance_threshold),
            thermometer_encode(self.fan_out as f64 / 20.0),
            thermometer_encode(self.depth_bias),
            thermometer_encode(self.breadth_bias),
            thermometer_encode(self.noise_tolerance),
            thermometer_encode(self.speed_bias),
            thermometer_encode(self.exploration),
        ]
    }
}

/// JIT scan parameters (for ndarray SIMD kernels).
#[derive(Debug, Clone)]
pub struct ScanParams {
    /// Hamming distance threshold (100..2100).
    pub threshold: u32,
    /// Top-K results per partition (1..128).
    pub top_k: u32,
    /// Prefetch lookahead distance (1..8).
    pub prefetch_ahead: u32,
    /// Focus bitmask for column filtering.
    pub filter_mask: u32,
}

/// Select thinking style from MUL assessment.
pub fn select_from_mul(mul: &MulAssessment) -> StyleFamily {
    // DK position influences style selection
    match mul.dk_position {
        DkPosition::MountStupid => StyleFamily::Metacognitive, // Force self-reflection
        DkPosition::ValleyOfDespair => StyleFamily::Systematic, // Careful, methodical
        DkPosition::SlopeOfEnlightenment => {
            // Choose based on flow state
            match mul.homeostasis.state {
                FlowState::Flow => StyleFamily::Analytical,
                FlowState::Anxiety => StyleFamily::Deliberate,
                FlowState::Boredom => StyleFamily::Creative,
                FlowState::Apathy => StyleFamily::Exploratory,
            }
        }
        DkPosition::PlateauOfMastery => {
            // Full range available — choose based on trust texture
            match mul.trust.texture {
                TrustTexture::Crystalline => StyleFamily::Intuitive, // Trust the gut
                TrustTexture::Solid => StyleFamily::Analytical,
                TrustTexture::Fuzzy => StyleFamily::Exploratory,
                _ => StyleFamily::Deliberate,
            }
        }
    }
}

fn thermometer_encode(value: f64) -> u8 {
    let clamped = (value.clamp(0.0, 1.0) * 8.0) as u8;
    (1u8 << clamped).wrapping_sub(1)
}
