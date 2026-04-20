//! Wire types for the external REST + protobuf API.
//!
//! These are the EXTERNAL types — serde + prost, owned strings, no lifetimes.
//! Internal types in lance-graph-contract stay zero-dep and zero-copy.
//! The conversion layer lives here.
//!
//! ```text
//! External (wire)                    Internal (zero-copy)
//! ──────────────                     ─────────────────────
//! WireDispatch  ←→ ShaderDispatch    (serde JSON + protobuf)
//! WireResonance ←→ ShaderResonance
//! WireBus       ←→ ShaderBus
//! WireCrystal   ←→ ShaderCrystal
//! WireHit       ←→ ShaderHit
//! WireMeta      ←→ MetaSummary
//! WireHealth    ←  neural-debug       (diagnostic overlay)
//! ```

use serde::{Deserialize, Serialize};

use lance_graph_contract::cognitive_shader::{
    ColumnWindow, EmitMode, MetaFilter, MetaSummary, RungLevel, ShaderBus, ShaderCrystal,
    ShaderDispatch, ShaderHit, ShaderResonance, StyleSelector,
};

// ═══════════════════════════════════════════════════════════════════════════
// Request types (client → server)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireDispatch {
    #[serde(default)]
    pub row_start: u32,
    #[serde(default)]
    pub row_end: u32,
    #[serde(default = "default_layer_mask")]
    pub layer_mask: u8,
    #[serde(default = "default_radius")]
    pub radius: u16,
    #[serde(default)]
    pub style: WireStyleSelector,
    #[serde(default)]
    pub rung: u8,
    #[serde(default = "default_max_cycles")]
    pub max_cycles: u16,
    #[serde(default = "default_entropy_floor")]
    pub entropy_floor: f32,
    #[serde(default)]
    pub emit: String,
    #[serde(default)]
    pub meta_filter: Option<WireMetaFilter>,
}

fn default_layer_mask() -> u8 { 0xFF }
fn default_radius() -> u16 { u16::MAX }
fn default_max_cycles() -> u16 { 10 }
fn default_entropy_floor() -> f32 { 0.05 }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type", content = "value")]
pub enum WireStyleSelector {
    #[default]
    Auto,
    Ordinal(u8),
    Named(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMetaFilter {
    #[serde(default)]
    pub thinking_mask: u64,
    #[serde(default)]
    pub awareness_min: u8,
    #[serde(default)]
    pub nars_f_min: u8,
    #[serde(default)]
    pub nars_c_min: u8,
    #[serde(default = "default_free_e_max")]
    pub free_e_max: u8,
}

fn default_free_e_max() -> u8 { 63 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireIngest {
    pub codebook_indices: Vec<u16>,
    #[serde(default)]
    pub source_ordinal: u8,
    #[serde(default)]
    pub timestamp: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Codec research DTOs (for remote-controlled codec benchmarking)
//
// These extend the canonical shader-driver API with codec-experimentation
// operations. No new feature gate — they ride on the existing `serve`/`grpc`
// features. EmbedAnything-style: one DTO surface for all shader operations.
// ═══════════════════════════════════════════════════════════════════════════

/// List tensors in a safetensors file with routing classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTensorsRequest {
    pub model_path: String,
    /// Optional filter: "CamPq" | "Passthrough" | "Skip". None = all.
    #[serde(default)]
    pub route_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTensorEntry {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: String,
    pub route: String,
    pub n_elements: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTensorsResponse {
    pub total: usize,
    pub shown: usize,
    pub cam_pq: usize,
    pub passthrough: usize,
    pub skip: usize,
    pub tensors: Vec<WireTensorEntry>,
}

/// Calibrate CAM-PQ codebook on a single tensor and measure ICC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireCalibrateRequest {
    pub model_path: String,
    pub tensor_name: String,
    #[serde(default = "default_cal_subspaces")]
    pub num_subspaces: usize,
    #[serde(default = "default_cal_centroids")]
    pub num_centroids: usize,
    #[serde(default = "default_cal_iters")]
    pub kmeans_iterations: usize,
    #[serde(default)]
    pub max_rows: Option<usize>,
    #[serde(default = "default_icc_samples")]
    pub icc_samples: usize,
}

fn default_cal_subspaces() -> usize { 6 }
fn default_cal_centroids() -> usize { 256 }
fn default_cal_iters() -> usize { 20 }
fn default_icc_samples() -> usize { 512 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireCalibrateResponse {
    pub tensor_name: String,
    pub dims: Vec<u64>,
    pub n_rows: usize,
    pub row_dim: usize,
    pub adjusted_dim: usize,
    pub num_subspaces: usize,
    pub num_centroids: usize,
    pub calibration_rows: usize,
    pub icc_3_1: f32,
    pub mean_reconstruction_error: f32,
    pub relative_l2_error: f32,
    pub codebook_bytes: usize,
    pub fingerprints_bytes: usize,
    pub elapsed_ms: u64,
}

/// ICC vs calibration-row-count diagnostic probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireProbeRequest {
    pub model_path: String,
    pub tensor_name: String,
    #[serde(default = "default_probe_counts")]
    pub row_counts: Vec<usize>,
    #[serde(default = "default_icc_samples")]
    pub icc_samples: usize,
}

fn default_probe_counts() -> Vec<usize> { vec![128, 256, 512, 1024] }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireProbeEntry {
    pub n_train: usize,
    pub icc_train: f32,
    pub icc_all_rows: f32,
    pub relative_l2_error: f32,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireProbeResponse {
    pub tensor_name: String,
    pub n_rows: usize,
    pub row_dim: usize,
    pub adjusted_dim: usize,
    pub num_subspaces: usize,
    pub num_centroids: usize,
    pub entries: Vec<WireProbeEntry>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Response types (server → client)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireHit {
    pub row: u32,
    pub distance: u16,
    pub predicates: u8,
    pub resonance: f32,
    pub cycle_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireResonance {
    pub top_k: Vec<WireHit>,
    pub hit_count: u16,
    pub cycles_used: u16,
    pub entropy: f32,
    pub std_dev: f32,
    pub style_ord: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireBus {
    pub cycle_fingerprint_hex: String,
    pub emitted_edges: Vec<u64>,
    pub gate: String,
    pub resonance: WireResonance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMeta {
    pub confidence: f32,
    pub meta_confidence: f32,
    pub brier: f32,
    pub should_admit_ignorance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireCrystal {
    pub bus: WireBus,
    pub persisted_row: Option<u32>,
    pub meta: WireMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireQualia {
    pub row: u32,
    pub experienced: Vec<f32>,
    pub classification_distance: f32,
    pub style_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireHealth {
    pub row_count: u32,
    pub byte_footprint: usize,
    pub styles: Vec<WireStyleInfo>,
    pub neural_debug: Option<WireNeuralDiag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireStyleInfo {
    pub ordinal: u8,
    pub name: String,
    pub layer_mask: u8,
    pub density_target: f32,
    pub resonance_threshold: f32,
    pub fan_out: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireNeuralDiag {
    pub total_functions: usize,
    pub total_dead: usize,
    pub total_stub: usize,
    pub health_pct: f32,
}

// ═══════════════════════════════════════════════════════════════════════════
// Conversions: wire → internal
// ═══════════════════════════════════════════════════════════════════════════

impl WireDispatch {
    pub fn to_internal(&self) -> ShaderDispatch {
        let style = match &self.style {
            WireStyleSelector::Auto => StyleSelector::Auto,
            WireStyleSelector::Ordinal(n) => StyleSelector::Ordinal(*n),
            WireStyleSelector::Named(s) => {
                StyleSelector::Ordinal(named_to_ordinal(s))
            }
        };
        let rung = match self.rung {
            0 => RungLevel::Surface,
            1 => RungLevel::Shallow,
            2 => RungLevel::Contextual,
            3 => RungLevel::Analogical,
            4 => RungLevel::Abstract,
            5 => RungLevel::Structural,
            6 => RungLevel::Counterfactual,
            7 => RungLevel::Meta,
            8 => RungLevel::Recursive,
            _ => RungLevel::Transcendent,
        };
        let emit = match self.emit.as_str() {
            "bundle" => EmitMode::Bundle,
            "persist" => EmitMode::Persist,
            _ => EmitMode::Cycle,
        };
        let meta_prefilter = self.meta_filter.as_ref().map(|f| MetaFilter {
            thinking_mask: f.thinking_mask,
            awareness_min: f.awareness_min,
            nars_f_min: f.nars_f_min,
            nars_c_min: f.nars_c_min,
            free_e_max: f.free_e_max,
        }).unwrap_or(MetaFilter::ALL);

        ShaderDispatch {
            meta_prefilter,
            rows: ColumnWindow::new(self.row_start, self.row_end),
            layer_mask: self.layer_mask,
            radius: self.radius,
            style,
            rung,
            max_cycles: self.max_cycles,
            entropy_floor: self.entropy_floor,
            emit,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Conversions: internal → wire
// ═══════════════════════════════════════════════════════════════════════════

impl From<&ShaderHit> for WireHit {
    fn from(h: &ShaderHit) -> Self {
        Self {
            row: h.row,
            distance: h.distance,
            predicates: h.predicates,
            resonance: h.resonance,
            cycle_index: h.cycle_index,
        }
    }
}

impl From<&ShaderResonance> for WireResonance {
    fn from(r: &ShaderResonance) -> Self {
        Self {
            top_k: r.top_k.iter().filter(|h| h.resonance > 0.0).map(WireHit::from).collect(),
            hit_count: r.hit_count,
            cycles_used: r.cycles_used,
            entropy: r.entropy,
            std_dev: r.std_dev,
            style_ord: r.style_ord,
        }
    }
}

impl From<&ShaderBus> for WireBus {
    fn from(b: &ShaderBus) -> Self {
        let hex: String = b.cycle_fingerprint.iter()
            .map(|w| format!("{:016x}", w))
            .collect::<Vec<_>>()
            .join("");
        let gate_str = if b.gate.is_flow() { "flow" }
            else if b.gate.is_hold() { "hold" }
            else { "block" };
        Self {
            cycle_fingerprint_hex: hex,
            emitted_edges: b.emitted_edges[..b.emitted_edge_count as usize].to_vec(),
            gate: gate_str.to_string(),
            resonance: WireResonance::from(&b.resonance),
        }
    }
}

impl From<&MetaSummary> for WireMeta {
    fn from(m: &MetaSummary) -> Self {
        Self {
            confidence: m.confidence,
            meta_confidence: m.meta_confidence,
            brier: m.brier,
            should_admit_ignorance: m.should_admit_ignorance,
        }
    }
}

impl From<&ShaderCrystal> for WireCrystal {
    fn from(c: &ShaderCrystal) -> Self {
        Self {
            bus: WireBus::from(&c.bus),
            persisted_row: c.persisted_row,
            meta: WireMeta::from(&c.meta),
        }
    }
}

/// Resolve a named style to an ordinal without leaking memory.
/// The 12 known names are matched; unknown falls back to Deliberate (0).
fn named_to_ordinal(s: &str) -> u8 {
    match s.to_lowercase().as_str() {
        "deliberate" => 0,
        "analytical" => 1,
        "convergent" => 2,
        "systematic" => 3,
        "creative" => 4,
        "divergent" => 5,
        "exploratory" => 6,
        "focused" => 7,
        "diffuse" => 8,
        "peripheral" => 9,
        "intuitive" => 10,
        "metacognitive" => 11,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_dispatch_defaults() {
        let json = r#"{"row_start": 0, "row_end": 100}"#;
        let wd: WireDispatch = serde_json::from_str(json).unwrap();
        assert_eq!(wd.layer_mask, 0xFF);
        assert_eq!(wd.max_cycles, 10);
        let internal = wd.to_internal();
        assert_eq!(internal.rows.start, 0);
        assert_eq!(internal.rows.end, 100);
    }

    #[test]
    fn wire_dispatch_with_style() {
        let json = r#"{"row_start": 0, "row_end": 50, "style": {"type": "Named", "value": "analytical"}}"#;
        let wd: WireDispatch = serde_json::from_str(json).unwrap();
        let internal = wd.to_internal();
        matches!(internal.style, StyleSelector::Ordinal(1));
    }

    #[test]
    fn wire_crystal_serializes() {
        let crystal = ShaderCrystal {
            bus: ShaderBus::empty(),
            persisted_row: Some(42),
            meta: MetaSummary { confidence: 0.9, meta_confidence: 0.8, brier: 0.1, should_admit_ignorance: false },
        };
        let wire = WireCrystal::from(&crystal);
        let json = serde_json::to_string(&wire).unwrap();
        assert!(json.contains("\"confidence\":0.9"));
        assert!(json.contains("\"persisted_row\":42"));
    }
}
