//! **LAB-ONLY.** Wire types for the REST + protobuf transports used by the
//! shader-lab binary. Not part of the canonical consumer surface.
//!
//! The canonical consumer surface is `UnifiedStep` + `OrchestrationBridge`
//! from `lance-graph-contract`. Everything in this file — per-op DTOs
//! (`WireDispatch`, `WireCalibrateRequest`, `WireProbeRequest`,
//! `WireTensorsRequest`, `WirePlanRequest`, `WireRunbookRequest`) — is
//! test-convenience scaffolding that ultimately dispatches through the
//! same bridge. Consumers (including the research consumer) speak
//! `UnifiedStep`, not these per-op shortcuts.
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
///
/// D0.1 extension: `params` and `tensor_view` fields carry the codec-sweep
/// shape introduced in PR #225 (`CodecParams` via `WireCodecParams` mirror).
/// When `params` is None, the legacy `num_subspaces` / `num_centroids` /
/// `kmeans_iterations` / `max_rows` fields construct a default `CodecParams`
/// for backward compatibility. Either path lands in a single `CodecParams`
/// object after ingress — per Rule F, there is no second deserialise anywhere
/// in the pipeline after the handler consumes the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireCalibrateRequest {
    pub model_path: String,
    pub tensor_name: String,

    /// New (D0.1): the full codec-sweep parameter shape. When present, takes
    /// precedence over the legacy num_* fields. Per Rule E, this mirrors
    /// `lance_graph_contract::cam::CodecParams` one-for-one; the handler
    /// converts via `TryFrom<WireCodecParams> for CodecParams`.
    #[serde(default)]
    pub params: Option<WireCodecParams>,

    /// New (D0.1): inline tensor payload for test harnesses and synthetic
    /// injection. When `None`, the handler mmaps from `model_path` +
    /// `tensor_name` as before. When `Some`, the base64 bytes decode **once**
    /// at ingress (Rule F) into a 64-byte-aligned buffer consumable directly
    /// by `F32x16::from_slice` via `slice::array_windows::<64>()` (Rule A).
    #[serde(default)]
    pub tensor_view: Option<WireTensorView>,

    // Legacy fields (used only when params is None).
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

    // D0.1 additions — SIMD-observability fields populated when JIT lands
    // (D1.1). Default values (0 / "none") are emitted by the legacy path
    // so existing clients keep parsing.
    /// `CodecParams::kernel_signature()` for the kernel actually executed.
    #[serde(default)]
    pub kernel_hash: u64,
    /// Cranelift compile time in microseconds. 0 = legacy non-JIT path or cache hit.
    #[serde(default)]
    pub compile_time_us: u64,
    /// SIMD tier the kernel ran on: "amx" | "vnni" | "avx512" | "avx2" | "legacy".
    /// Never "scalar" on a SoA path — iron rule.
    #[serde(default = "default_backend")]
    pub backend: String,
}

fn default_backend() -> String { "legacy".to_string() }

// ═══════════════════════════════════════════════════════════════════════════
// D0.1 — CodecParams serde mirrors (Rule F: serialise at edges only)
//
// lance-graph-contract is zero-dep; the contract types (CodecParams et al.)
// don't carry serde derives. The `Wire*` mirrors below hold the JSON/YAML
// shape and convert to the contract types via TryFrom. After ingress the
// contract types own the lifetime — no serde between layers.
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireLaneWidth { F32x16, U8x64, F64x8, BF16x32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireDistance { AdcU8, AdcI8 }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum WireRotation {
    Identity,
    Hadamard { dim: u32 },
    Opq { matrix_blob_id: u64, dim: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireResidualSpec {
    pub depth: u8,
    pub centroids: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireCodecParams {
    pub subspaces: u32,
    pub centroids: u32,
    #[serde(default = "default_wire_residual")]
    pub residual: WireResidualSpec,
    #[serde(default = "default_wire_lane")]
    pub lane_width: WireLaneWidth,
    #[serde(default = "default_wire_rotation")]
    pub pre_rotation: WireRotation,
    #[serde(default = "default_wire_distance")]
    pub distance: WireDistance,
    #[serde(default = "default_calibration_rows")]
    pub calibration_rows: u32,
    #[serde(default)]
    pub measurement_rows: u32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_wire_residual() -> WireResidualSpec { WireResidualSpec { depth: 0, centroids: 256 } }
fn default_wire_lane() -> WireLaneWidth { WireLaneWidth::F32x16 }
fn default_wire_rotation() -> WireRotation { WireRotation::Identity }
fn default_wire_distance() -> WireDistance { WireDistance::AdcU8 }
fn default_calibration_rows() -> u32 { 2048 }
fn default_seed() -> u64 { 42 }

// ─────── TryFrom conversions — one decode at ingress (Rule F) ───────

use lance_graph_contract::cam::{
    CodecParams, CodecParamsBuilder, CodecParamsError, Distance as CamDistance,
    LaneWidth as CamLaneWidth, ResidualSpec as CamResidualSpec, Rotation as CamRotation,
};

impl From<WireLaneWidth> for CamLaneWidth {
    fn from(w: WireLaneWidth) -> Self {
        match w {
            WireLaneWidth::F32x16 => CamLaneWidth::F32x16,
            WireLaneWidth::U8x64 => CamLaneWidth::U8x64,
            WireLaneWidth::F64x8 => CamLaneWidth::F64x8,
            WireLaneWidth::BF16x32 => CamLaneWidth::BF16x32,
        }
    }
}

impl From<WireDistance> for CamDistance {
    fn from(d: WireDistance) -> Self {
        match d {
            WireDistance::AdcU8 => CamDistance::AdcU8,
            WireDistance::AdcI8 => CamDistance::AdcI8,
        }
    }
}

impl From<WireRotation> for CamRotation {
    fn from(r: WireRotation) -> Self {
        match r {
            WireRotation::Identity => CamRotation::Identity,
            WireRotation::Hadamard { dim } => CamRotation::Hadamard { dim },
            WireRotation::Opq { matrix_blob_id, dim } => CamRotation::Opq { matrix_blob_id, dim },
        }
    }
}

impl From<WireResidualSpec> for CamResidualSpec {
    fn from(r: WireResidualSpec) -> Self {
        CamResidualSpec { depth: r.depth, centroids: r.centroids }
    }
}

impl TryFrom<WireCodecParams> for CodecParams {
    type Error = CodecParamsError;
    fn try_from(w: WireCodecParams) -> Result<Self, Self::Error> {
        CodecParamsBuilder::new()
            .subspaces(w.subspaces)
            .centroids(w.centroids)
            .residual(w.residual.into())
            .lane_width(w.lane_width.into())
            .rotation(w.pre_rotation.into())
            .distance(w.distance.into())
            .calibration_rows(w.calibration_rows)
            .measurement_rows(w.measurement_rows)
            .seed(w.seed)
            .build()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// D0.1 — WireTensorView: the tensor payload object per Rule E
//
// Wire surface IS the SIMD surface. WireTensorView:
//   - names its lane_width explicitly (not inferred later)
//   - base64-decodes ONCE at ingress (Rule F)
//   - lands the bytes in a 64-byte-aligned buffer ready for F32x16::from_slice
//   - exposes methods (row(), row_count(), lanes_f32x16(), subspace())
//     that mirror the SoA + SIMD operations the JIT kernel will perform
//
// Consumers never reassemble a tensor from a Vec<f32>. They hold a
// WireTensorView, call .row(i), call .array_windows::<64>() on that,
// call F32x16::from_slice — zero adapter layer.
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireTensorView {
    /// [rows, cols] in elements (not bytes). Actual byte size inferred from lane_width.
    pub shape: [u32; 2],
    /// SIMD lane width the codec kernel is expected to consume.
    pub lane_width: WireLaneWidth,
    /// Base64-encoded raw bytes in row-major order. Decoded ONCE at
    /// `WireTensorView::decode` (Rule F). Kept as owned `String` on the
    /// wire; the decoded `AlignedBytes` is produced once at ingress.
    pub bytes_base64: String,
}

/// 64-byte-aligned owned buffer produced by `WireTensorView::decode`.
/// Used directly as the input to `slice::array_windows::<64>()` +
/// `F32x16::from_slice` — zero copy, zero re-align after this point.
#[derive(Debug)]
pub struct AlignedBytes {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

// SAFETY: `AlignedBytes` owns a heap buffer allocated via `alloc::alloc` with
// a 64-byte-aligned `Layout`. The raw pointer is never shared; Drop frees it
// exactly once with the matching layout. No interior mutability — all access
// goes through `&self` or `&mut self`. Send + Sync are safe.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    /// Allocate a zero-initialised 64-byte-aligned buffer of `len` bytes.
    pub fn alloc_zeroed(len: usize) -> Self {
        use std::alloc::{alloc_zeroed, Layout};
        let cap = (len + 63) & !63; // round up to 64
        let layout = Layout::from_size_align(cap.max(64), 64)
            .expect("AlignedBytes layout must be valid (len + 64 alignment)");
        // SAFETY: layout has non-zero size (we took max(64)) and 64-byte alignment;
        // `alloc_zeroed` is defined for this layout and returns a zero-initialised
        // buffer. Null return handled explicitly.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, len, cap }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: self.ptr was allocated for at least self.cap bytes; self.len <= self.cap;
        // the buffer is exclusively owned (no aliased &mut elsewhere given &self).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: same as as_slice but &mut self guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn is_aligned_64(&self) -> bool {
        (self.ptr as usize) % 64 == 0
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        use std::alloc::{dealloc, Layout};
        // SAFETY: matches the alloc_zeroed layout in alloc_zeroed above.
        let layout = Layout::from_size_align(self.cap.max(64), 64).expect("layout");
        unsafe { dealloc(self.ptr, layout) };
    }
}

/// Error returned by `WireTensorView::decode` when the wire bytes are
/// malformed. Base64 decode error, lane-size/shape mismatch, or short buffer.
#[derive(Debug)]
pub enum WireTensorViewError {
    Base64(base64::DecodeError),
    SizeMismatch { expected: usize, actual: usize },
    ZeroShape,
}

impl std::fmt::Display for WireTensorViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base64(e) => write!(f, "base64 decode: {}", e),
            Self::SizeMismatch { expected, actual } => {
                write!(f, "byte size mismatch: expected {} got {}", expected, actual)
            }
            Self::ZeroShape => write!(f, "tensor view shape contains zero dimension"),
        }
    }
}

impl std::error::Error for WireTensorViewError {}

impl WireTensorView {
    /// Bytes per element for the declared `lane_width`.
    pub fn element_bytes(&self) -> usize {
        match self.lane_width {
            WireLaneWidth::F32x16 => 4,
            WireLaneWidth::U8x64 => 1,
            WireLaneWidth::F64x8 => 8,
            WireLaneWidth::BF16x32 => 2,
        }
    }

    /// Expected total bytes: `rows × cols × element_bytes`.
    pub fn expected_bytes(&self) -> usize {
        self.shape[0] as usize * self.shape[1] as usize * self.element_bytes()
    }

    /// Row count.
    pub fn row_count(&self) -> u32 { self.shape[0] }

    /// Column count (elements per row).
    pub fn col_count(&self) -> u32 { self.shape[1] }

    /// Row stride in bytes.
    pub fn row_bytes(&self) -> usize {
        self.shape[1] as usize * self.element_bytes()
    }

    /// Decode the base64 payload ONCE into a 64-byte-aligned buffer (Rule F).
    /// The returned `AlignedBytes` is consumed directly by
    /// `slice::array_windows::<64>()` + `F32x16::from_slice` — no adapter.
    #[cfg(feature = "serve")]
    pub fn decode(&self) -> Result<AlignedBytes, WireTensorViewError> {
        use base64::{engine::general_purpose::STANDARD, Engine};
        if self.shape[0] == 0 || self.shape[1] == 0 {
            return Err(WireTensorViewError::ZeroShape);
        }
        let raw = STANDARD
            .decode(&self.bytes_base64)
            .map_err(WireTensorViewError::Base64)?;
        let expected = self.expected_bytes();
        if raw.len() != expected {
            return Err(WireTensorViewError::SizeMismatch { expected, actual: raw.len() });
        }
        let mut aligned = AlignedBytes::alloc_zeroed(expected);
        aligned.as_mut_slice().copy_from_slice(&raw);
        debug_assert!(aligned.is_aligned_64());
        Ok(aligned)
    }

    /// View of a single row's bytes inside a decoded buffer. Consumer calls
    /// `slice::array_windows::<64>()` on the returned slice (Rule A).
    pub fn row<'a>(&self, decoded: &'a AlignedBytes, idx: usize) -> Option<&'a [u8]> {
        let rb = self.row_bytes();
        let start = idx.checked_mul(rb)?;
        let end = start.checked_add(rb)?;
        decoded.as_slice().get(start..end)
    }

    /// View of one subspace slice within a row. The JIT decode kernel reads
    /// (subspace_count × subspace_bytes) contiguous bytes per row; this method
    /// returns subspace `k`. `sub_bytes` = ceil(col_bytes / subspaces).
    pub fn subspace<'a>(
        &self,
        decoded: &'a AlignedBytes,
        row_idx: usize,
        k: u32,
        sub_bytes: usize,
    ) -> Option<&'a [u8]> {
        let row = self.row(decoded, row_idx)?;
        let start = (k as usize).checked_mul(sub_bytes)?;
        let end = start.checked_add(sub_bytes)?;
        row.get(start..end)
    }
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
// Plan — delegate to lance-graph-planner (Layer 4 per INTEGRATION_PLAN_CS.md)
//
// Exposes the PlannerAwareness entry points (plan_auto, plan_full) through
// the same Wire DTO surface. Behind `with-planner` feature; when disabled
// the handler returns 503. Keeps the unified endpoint as the only REST
// surface even as planning-layer capabilities land.
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WirePlanRequest {
    pub query: String,
    /// "auto" (no MUL) or "full" (MUL + compass + thinking orchestration).
    #[serde(default = "default_plan_mode")]
    pub mode: String,
    /// Named strategies to use explicitly (overrides auto-selection).
    /// Empty = auto.
    #[serde(default)]
    pub strategies: Vec<String>,
    /// Optional situation input for plan_full (ignored when mode="auto").
    #[serde(default)]
    pub situation: Option<WireSituation>,
}

fn default_plan_mode() -> String { "auto".to_string() }

// ═══════════════════════════════════════════════════════════════════════════
// Generic OrchestrationBridge routing — UnifiedStep as JSON
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireUnifiedStep {
    pub step_id: String,
    pub step_type: String,
    #[serde(default)]
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireStepResult {
    pub step_id: String,
    pub step_type: String,
    pub status: String,
    pub reasoning: Option<String>,
    pub confidence: Option<f64>,
}

/// Mirror of lance_graph_contract::mul::SituationInput for JSON transport.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WireSituation {
    #[serde(default = "half")]  pub felt_competence: f64,
    #[serde(default = "half")]  pub demonstrated_competence: f64,
    #[serde(default = "half")]  pub source_reliability: f64,
    #[serde(default = "half")]  pub environment_stability: f64,
    #[serde(default = "half")]  pub calibration_accuracy: f64,
    #[serde(default = "half")]  pub complexity_ratio: f64,
}

fn half() -> f64 { 0.5 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WirePlanResponse {
    pub mode: String,
    pub strategies_used: Vec<String>,
    pub free_will_modifier: f64,
    pub compass_score: Option<f64>,
    /// Populated when plan_full was invoked and MUL returned a decision.
    pub mul_gate: Option<String>,
    /// Populated when thinking orchestration was run.
    pub thinking_style_name: Option<String>,
    pub nars_type: Option<String>,
    pub elapsed_ms: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Runbook — scheduled sequence of operations for REST/gRPC test injection
//
// One POST submits a list of steps; the server executes them in order and
// returns a matching list of results. Each step reuses an existing Wire*
// request type — no new operation surface. The only new concepts are
// (a) the step enum, (b) the label field per step (for result tracking).
//
// Use cases:
//   - Inject a codec test suite from a script / notebook / CI
//   - Replay a calibration protocol across many tensors
//   - Seed BindSpace with ingests then dispatch queries in one round-trip
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", content = "args")]
pub enum WireRunbookStep {
    Tensors(WireTensorsRequest),
    Calibrate(WireCalibrateRequest),
    Probe(WireProbeRequest),
    Dispatch(WireDispatch),
    Ingest(WireIngest),
    /// Planner delegation — plan_auto / plan_full via PlannerAwareness.
    /// Requires shader-driver compiled with `--features with-planner`.
    Plan(WirePlanRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRunbookRequest {
    /// Human label for the whole runbook (e.g. "qwen3-tts full-size ICC sweep").
    #[serde(default)]
    pub label: String,
    pub steps: Vec<WireRunbookStepLabeled>,
    /// If true, abort remaining steps on the first error. If false, continue
    /// and report each step's outcome individually.
    #[serde(default)]
    pub stop_on_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRunbookStepLabeled {
    /// Per-step label, surfaces in the result so callers can correlate.
    #[serde(default)]
    pub label: String,
    #[serde(flatten)]
    pub step: WireRunbookStep,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum WireRunbookStepResult {
    Tensors { label: String, response: WireTensorsResponse },
    Calibrate { label: String, response: WireCalibrateResponse },
    Probe { label: String, response: WireProbeResponse },
    Dispatch { label: String, response: WireCrystal },
    Ingest { label: String, ingested: u32, row_start: u32, row_end: u32, write_cursor: u32 },
    Plan { label: String, response: WirePlanResponse },
    Error { label: String, step: String, error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRunbookResponse {
    pub label: String,
    pub total_steps: usize,
    pub completed: usize,
    pub errors: usize,
    pub total_elapsed_ms: u64,
    pub results: Vec<WireRunbookStepResult>,
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
    fn wire_plan_request_defaults() {
        let json = r#"{"query": "MATCH (n) RETURN n"}"#;
        let p: WirePlanRequest = serde_json::from_str(json).unwrap();
        assert_eq!(p.query, "MATCH (n) RETURN n");
        assert_eq!(p.mode, "auto");
        assert!(p.strategies.is_empty());
        assert!(p.situation.is_none());
    }

    #[test]
    fn wire_plan_request_full_mode() {
        let json = r#"{"query": "MATCH (n)-[:KNOWS]->(m)", "mode": "full",
                       "strategies": ["cypher_parse", "dp_join"],
                       "situation": {"felt_competence": 0.8}}"#;
        let p: WirePlanRequest = serde_json::from_str(json).unwrap();
        assert_eq!(p.mode, "full");
        assert_eq!(p.strategies.len(), 2);
        assert_eq!(p.situation.as_ref().unwrap().felt_competence, 0.8);
        // Other situation fields default to 0.5
        assert_eq!(p.situation.as_ref().unwrap().source_reliability, 0.5);
    }

    #[test]
    fn wire_runbook_accepts_plan_step() {
        let json = r#"{
          "label": "plan then calibrate",
          "steps": [
            {"label": "parse", "op": "Plan",
             "args": {"query": "MATCH (n) RETURN n", "mode": "auto"}},
            {"label": "calibrate", "op": "Calibrate",
             "args": {"model_path": "/m.st", "tensor_name": "q_proj"}}
          ]
        }"#;
        let rb: WireRunbookRequest = serde_json::from_str(json).unwrap();
        assert_eq!(rb.steps.len(), 2);
        match &rb.steps[0].step {
            WireRunbookStep::Plan(p) => assert_eq!(p.mode, "auto"),
            _ => panic!("expected Plan step"),
        }
    }

    #[test]
    fn wire_runbook_parses_mixed_steps() {
        // Test injection payload: list tensors, calibrate one, then probe it.
        let json = r#"{
          "label": "qwen3-tts full-size ICC sweep",
          "stop_on_error": false,
          "steps": [
            {"label": "inventory", "op": "Tensors",
             "args": {"model_path": "/m.safetensors", "route_filter": "CamPq"}},
            {"label": "gate_proj full", "op": "Calibrate",
             "args": {"model_path": "/m.safetensors",
                      "tensor_name": "layers.5.mlp.gate_proj"}},
            {"label": "gate_proj probe", "op": "Probe",
             "args": {"model_path": "/m.safetensors",
                      "tensor_name": "layers.5.mlp.gate_proj",
                      "row_counts": [128, 256, 512, 1024]}}
          ]
        }"#;
        let rb: WireRunbookRequest = serde_json::from_str(json).unwrap();
        assert_eq!(rb.steps.len(), 3);
        assert_eq!(rb.label, "qwen3-tts full-size ICC sweep");
        assert!(!rb.stop_on_error);
        match &rb.steps[0].step {
            WireRunbookStep::Tensors(r) => assert_eq!(r.route_filter.as_deref(), Some("CamPq")),
            _ => panic!("expected Tensors step"),
        }
        match &rb.steps[1].step {
            WireRunbookStep::Calibrate(r) => {
                assert_eq!(r.num_subspaces, 6);
                assert_eq!(r.num_centroids, 256);
            }
            _ => panic!("expected Calibrate step"),
        }
        match &rb.steps[2].step {
            WireRunbookStep::Probe(r) => assert_eq!(r.row_counts, vec![128, 256, 512, 1024]),
            _ => panic!("expected Probe step"),
        }
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

    // ═════════════════════════════════════════════════════════════════════
    // D0.1 — WireCodecParams / WireTensorView tests (Rules A, E, F)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn wire_codec_params_round_trip_to_contract() {
        let wire = WireCodecParams {
            subspaces: 6,
            centroids: 1024,
            residual: WireResidualSpec { depth: 1, centroids: 256 },
            lane_width: WireLaneWidth::BF16x32,
            pre_rotation: WireRotation::Opq { matrix_blob_id: 0xDEADBEEF, dim: 4096 },
            distance: WireDistance::AdcU8,
            calibration_rows: 2048,
            measurement_rows: 512,
            seed: 42,
        };
        let params: CodecParams = wire.try_into().expect("OPQ + BF16x32 is precision-ladder valid");
        assert_eq!(params.subspaces, 6);
        assert_eq!(params.centroids, 1024);
        assert!(params.is_matmul_heavy(), "OPQ + wide codebook must be matmul-heavy");
    }

    #[test]
    fn wire_codec_params_rejects_opq_with_f32x16() {
        // Rule E precision ladder: OPQ requires BF16x32.
        let wire = WireCodecParams {
            subspaces: 6,
            centroids: 256,
            residual: WireResidualSpec { depth: 0, centroids: 256 },
            lane_width: WireLaneWidth::F32x16,
            pre_rotation: WireRotation::Opq { matrix_blob_id: 1, dim: 4096 },
            distance: WireDistance::AdcU8,
            calibration_rows: 2048,
            measurement_rows: 0,
            seed: 42,
        };
        let err = CodecParams::try_from(wire).unwrap_err();
        assert!(matches!(err, CodecParamsError::OpqRequiresBf16 { .. }));
    }

    #[test]
    fn wire_codec_params_rejects_calibration_equals_measurement() {
        // Overfit guard from PR #225 — the PR #219 pattern.
        let wire = WireCodecParams {
            subspaces: 6,
            centroids: 256,
            residual: WireResidualSpec { depth: 0, centroids: 256 },
            lane_width: WireLaneWidth::F32x16,
            pre_rotation: WireRotation::Identity,
            distance: WireDistance::AdcU8,
            calibration_rows: 128,
            measurement_rows: 128,
            seed: 42,
        };
        let err = CodecParams::try_from(wire).unwrap_err();
        assert!(matches!(err, CodecParamsError::CalibrationEqualsMeasurement { rows: 128 }));
    }

    #[test]
    fn wire_codec_params_deserializes_from_minimal_json() {
        let json = r#"{"subspaces":6,"centroids":256}"#;
        let wire: WireCodecParams = serde_json::from_str(json).unwrap();
        assert_eq!(wire.lane_width, WireLaneWidth::F32x16);       // default
        assert_eq!(wire.distance, WireDistance::AdcU8);           // default
        assert_eq!(wire.pre_rotation, WireRotation::Identity);    // default
        assert_eq!(wire.calibration_rows, 2048);                  // default
        assert_eq!(wire.seed, 42);                                // default
    }

    #[cfg(feature = "serve")]
    #[test]
    fn wire_tensor_view_decode_lands_in_64byte_aligned_buffer() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        // A 4-row × 16-col F32 tensor = 4 × 16 × 4 = 256 bytes, exactly
        // 4 × F32x16 lanes worth, cleanly aligned.
        let bytes: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        let view = WireTensorView {
            shape: [4, 16],
            lane_width: WireLaneWidth::F32x16,
            bytes_base64: STANDARD.encode(&bytes),
        };
        assert_eq!(view.expected_bytes(), 256);
        assert_eq!(view.row_bytes(), 64);
        let decoded = view.decode().expect("valid base64 + matching size");
        assert!(decoded.is_aligned_64(), "Rule A: decoded buffer MUST be 64-byte aligned");
        assert_eq!(decoded.len(), 256);

        // Rule A: slice::array_windows::<64>() must consume the row directly.
        let row0 = view.row(&decoded, 0).expect("row 0 exists");
        assert_eq!(row0.len(), 64);
        let mut windows = 0;
        for _w in row0.array_windows::<64>() {
            windows += 1;
        }
        assert_eq!(windows, 1, "exactly one 64-byte window per row at this size");
    }

    #[cfg(feature = "serve")]
    #[test]
    fn wire_tensor_view_rejects_size_mismatch() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let wrong_bytes = vec![0u8; 100]; // expected 256
        let view = WireTensorView {
            shape: [4, 16],
            lane_width: WireLaneWidth::F32x16,
            bytes_base64: STANDARD.encode(&wrong_bytes),
        };
        let err = view.decode().unwrap_err();
        assert!(matches!(err, WireTensorViewError::SizeMismatch { expected: 256, actual: 100 }));
    }

    #[cfg(feature = "serve")]
    #[test]
    fn wire_tensor_view_subspace_slicing() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        // 2 rows × 24 cols F32 = 2 × 24 × 4 = 192 bytes; 6 subspaces → 16 bytes each.
        let bytes: Vec<u8> = (0..192u32).map(|i| i as u8).collect();
        let view = WireTensorView {
            shape: [2, 24],
            lane_width: WireLaneWidth::F32x16,
            bytes_base64: STANDARD.encode(&bytes),
        };
        let decoded = view.decode().unwrap();
        let sub = view.subspace(&decoded, 0, 2, 16).expect("subspace 2 of row 0");
        assert_eq!(sub.len(), 16);
        // subspace 2 of row 0 starts at byte 32 (0 × 96 + 2 × 16) — value = 32.
        assert_eq!(sub[0], 32);
        assert_eq!(sub[15], 47);
    }

    #[test]
    fn wire_calibrate_request_accepts_new_params_field() {
        let json = r#"{
            "model_path": "m.safetensors",
            "tensor_name": "w",
            "params": { "subspaces": 6, "centroids": 1024, "calibration_rows": 2048, "measurement_rows": 512 }
        }"#;
        let req: WireCalibrateRequest = serde_json::from_str(json).unwrap();
        assert!(req.params.is_some());
        let p = req.params.unwrap();
        assert_eq!(p.centroids, 1024);
    }

    #[test]
    fn wire_calibrate_request_back_compat_legacy_fields() {
        // Legacy payload (no `params`) still parses; defaults preserved.
        let json = r#"{
            "model_path": "m.safetensors",
            "tensor_name": "w",
            "num_subspaces": 6,
            "num_centroids": 256
        }"#;
        let req: WireCalibrateRequest = serde_json::from_str(json).unwrap();
        assert!(req.params.is_none());
        assert_eq!(req.num_centroids, 256);
    }
}
