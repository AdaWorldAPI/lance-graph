//! Research endpoints — remote-controlled codec benchmarking.
//!
//! Feature-gated behind `research`. Enables interactive codec research
//! via REST API instead of recompiling for each test.
//!
//! # Endpoints
//!
//! ```text
//! POST /v1/research/tensors   — list tensors in a safetensors file with routes
//! POST /v1/research/calibrate — calibrate one tensor, return ICC + stats
//! POST /v1/research/probe     — ICC vs row-count diagnostic on one tensor
//! GET  /v1/research/stats     — aggregate stats from last calibration
//! ```
//!
//! # Example
//!
//! ```bash
//! # Start server with research enabled:
//! cargo run --manifest-path crates/lance-graph-planner/Cargo.toml \
//!   --features research --bin serve --release
//!
//! # List tensors in a model:
//! curl -X POST http://localhost:3000/v1/research/tensors \
//!   -H "Content-Type: application/json" \
//!   -d '{"model_path": "/home/user/models/qwen3-tts-0.6b/model.safetensors"}'
//!
//! # Calibrate a single tensor and measure ICC:
//! curl -X POST http://localhost:3000/v1/research/calibrate \
//!   -H "Content-Type: application/json" \
//!   -d '{
//!     "model_path": "/home/user/models/qwen3-tts-0.6b/model.safetensors",
//!     "tensor_name": "layers.5.mlp.gate_proj",
//!     "num_subspaces": 6,
//!     "num_centroids": 256,
//!     "kmeans_iterations": 20,
//!     "max_rows": null,
//!     "icc_samples": 512
//!   }'
//!
//! # Row-count probe (ICC degradation curve):
//! curl -X POST http://localhost:3000/v1/research/probe \
//!   -H "Content-Type: application/json" \
//!   -d '{
//!     "model_path": "/home/user/models/qwen3-tts-0.6b/model.safetensors",
//!     "tensor_name": "layers.5.mlp.gate_proj",
//!     "row_counts": [128, 256, 512, 1024]
//!   }'
//! ```

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

use lance_graph_contract::cam::{route_tensor, CodecRoute};
use ndarray::hpc::cam_pq::{self, CamCodebook, NUM_CENTROIDS, NUM_SUBSPACES};
use ndarray::hpc::gguf::read_tensor_f32;
use ndarray::hpc::safetensors::read_safetensors_header;

// ─── State ──────────────────────────────────────────────────────────────────

pub struct ResearchState {
    pub last_results: Vec<CalibrationResult>,
}

impl ResearchState {
    pub fn new() -> Self {
        Self { last_results: Vec::new() }
    }
}

type RState = Arc<Mutex<ResearchState>>;

pub fn router() -> Router {
    let state: RState = Arc::new(Mutex::new(ResearchState::new()));
    Router::new()
        .route("/tensors", post(list_tensors))
        .route("/calibrate", post(calibrate_tensor))
        .route("/probe", post(row_count_probe))
        .route("/stats", get(stats))
        .with_state(state)
}

// ─── DTOs ───────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TensorsRequest {
    model_path: String,
    #[serde(default)]
    route_filter: Option<String>,
}

#[derive(Serialize)]
struct TensorEntry {
    name: String,
    dims: Vec<u64>,
    dtype: String,
    route: String,
    n_elements: u64,
}

#[derive(Deserialize)]
struct CalibrateRequest {
    model_path: String,
    tensor_name: String,
    #[serde(default = "default_subspaces")]
    num_subspaces: usize,
    #[serde(default = "default_centroids")]
    num_centroids: usize,
    #[serde(default = "default_iterations")]
    kmeans_iterations: usize,
    #[serde(default)]
    max_rows: Option<usize>,
    #[serde(default = "default_icc_samples")]
    icc_samples: usize,
}

fn default_subspaces() -> usize { NUM_SUBSPACES }
fn default_centroids() -> usize { NUM_CENTROIDS }
fn default_iterations() -> usize { 20 }
fn default_icc_samples() -> usize { 512 }

#[derive(Serialize, Clone)]
pub struct CalibrationResult {
    tensor_name: String,
    dims: Vec<u64>,
    n_rows: usize,
    row_dim: usize,
    adjusted_dim: usize,
    num_subspaces: usize,
    num_centroids: usize,
    kmeans_iterations: usize,
    calibration_rows: usize,
    icc_3_1: f32,
    mean_reconstruction_error: f32,
    relative_l2_error: f32,
    codebook_bytes: usize,
    fingerprints_bytes: usize,
    elapsed_ms: u64,
}

#[derive(Deserialize)]
struct ProbeRequest {
    model_path: String,
    tensor_name: String,
    #[serde(default = "default_probe_row_counts")]
    row_counts: Vec<usize>,
    #[serde(default = "default_icc_samples")]
    icc_samples: usize,
}

fn default_probe_row_counts() -> Vec<usize> { vec![128, 256, 512, 1024] }

#[derive(Serialize)]
struct ProbeEntry {
    n_train: usize,
    icc_train: f32,
    icc_all_rows: f32,
    relative_l2_error: f32,
    elapsed_ms: u64,
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn list_tensors(
    Json(req): Json<TensorsRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let file = std::fs::File::open(&req.model_path).map_err(|e| err(e))?;
    let mut reader = std::io::BufReader::new(file);
    let gguf = read_safetensors_header(&mut reader).map_err(|e| err(e))?;

    let route_filter = req.route_filter.as_deref();
    let entries: Vec<TensorEntry> = gguf
        .tensors
        .iter()
        .filter_map(|t| {
            let route = route_tensor(&t.name, &t.dimensions);
            let route_str = route_str(route);
            if let Some(f) = route_filter {
                if route_str != f { return None; }
            }
            Some(TensorEntry {
                name: t.name.clone(),
                dims: t.dimensions.clone(),
                dtype: format!("{:?}", t.dtype),
                route: route_str.to_string(),
                n_elements: t.element_count(),
            })
        })
        .collect();

    let summary = json!({
        "total": gguf.tensors.len(),
        "shown": entries.len(),
        "cam_pq": entries.iter().filter(|e| e.route == "CamPq").count(),
        "passthrough": entries.iter().filter(|e| e.route == "Passthrough").count(),
        "skip": entries.iter().filter(|e| e.route == "Skip").count(),
    });

    Ok(Json(json!({ "summary": summary, "tensors": entries })))
}

async fn calibrate_tensor(
    State(state): State<RState>,
    Json(req): Json<CalibrateRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (rows, tensor_name, dims) = load_tensor_rows(&req.model_path, &req.tensor_name)?;
    let n_rows = rows.len();
    let row_dim = if rows.is_empty() { 0 } else { rows[0].len() };
    let adjusted_dim = (row_dim / req.num_subspaces) * req.num_subspaces;

    if adjusted_dim == 0 {
        return Err(err(format!("row_dim {row_dim} < num_subspaces {}", req.num_subspaces)));
    }

    let t0 = std::time::Instant::now();

    let calibration_rows: Vec<Vec<f32>> = match req.max_rows {
        Some(n) if n < n_rows => rows[..n].iter().map(|r| r[..adjusted_dim].to_vec()).collect(),
        _ => rows.iter().map(|r| r[..adjusted_dim].to_vec()).collect(),
    };
    let cal_n = calibration_rows.len();

    let codebook = cam_pq::train_geometric(&calibration_rows, adjusted_dim, req.kmeans_iterations);

    let sliced: Vec<Vec<f32>> = rows.iter().map(|r| r[..adjusted_dim].to_vec()).collect();
    let icc = measure_icc(&sliced, &codebook, req.icc_samples);
    let sample_n = sliced.len().min(512);
    let mean_err = codebook.mean_reconstruction_error(&sliced[..sample_n]);
    let rel_err = relative_l2_error(&codebook, &sliced[..sample_n]);

    let elapsed = t0.elapsed();

    let codebook_bytes = 24 + req.num_subspaces * req.num_centroids * (adjusted_dim / req.num_subspaces) * 4;
    let fp_bytes = 20 + n_rows * 6;

    let result = CalibrationResult {
        tensor_name: tensor_name.clone(),
        dims: dims.clone(),
        n_rows,
        row_dim,
        adjusted_dim,
        num_subspaces: req.num_subspaces,
        num_centroids: req.num_centroids,
        kmeans_iterations: req.kmeans_iterations,
        calibration_rows: cal_n,
        icc_3_1: icc,
        mean_reconstruction_error: mean_err,
        relative_l2_error: rel_err,
        codebook_bytes,
        fingerprints_bytes: fp_bytes,
        elapsed_ms: elapsed.as_millis() as u64,
    };

    state.lock().unwrap().last_results.push(result.clone());

    Ok(Json(json!(result)))
}

async fn row_count_probe(
    Json(req): Json<ProbeRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (rows, tensor_name, _dims) = load_tensor_rows(&req.model_path, &req.tensor_name)?;
    let n_rows = rows.len();
    let row_dim = if rows.is_empty() { 0 } else { rows[0].len() };
    let adjusted_dim = (row_dim / NUM_SUBSPACES) * NUM_SUBSPACES;

    if adjusted_dim == 0 {
        return Err(err(format!("row_dim {row_dim} < 6")));
    }

    let sliced: Vec<Vec<f32>> = rows.iter().map(|r| r[..adjusted_dim].to_vec()).collect();

    let mut all_counts = req.row_counts.clone();
    if !all_counts.contains(&n_rows) {
        all_counts.push(n_rows);
    }
    all_counts.sort();
    all_counts.dedup();

    let mut entries: Vec<ProbeEntry> = Vec::new();
    for &n in &all_counts {
        if n > n_rows { continue; }
        let t0 = std::time::Instant::now();
        let training = &sliced[..n];
        let cb = cam_pq::train_geometric(training, adjusted_dim, 20);
        let elapsed = t0.elapsed();

        entries.push(ProbeEntry {
            n_train: n,
            icc_train: measure_icc(training, &cb, req.icc_samples),
            icc_all_rows: measure_icc(&sliced, &cb, req.icc_samples),
            relative_l2_error: relative_l2_error(&cb, &sliced[..sliced.len().min(512)]),
            elapsed_ms: elapsed.as_millis() as u64,
        });
    }

    Ok(Json(json!({
        "tensor_name": tensor_name,
        "n_rows": n_rows,
        "row_dim": row_dim,
        "adjusted_dim": adjusted_dim,
        "num_subspaces": NUM_SUBSPACES,
        "num_centroids": NUM_CENTROIDS,
        "entries": entries,
    })))
}

async fn stats(
    State(state): State<RState>,
) -> Json<Value> {
    let s = state.lock().unwrap();
    if s.last_results.is_empty() {
        return Json(json!({ "message": "No calibration results yet. POST /v1/research/calibrate first." }));
    }
    let n = s.last_results.len();
    let min_icc = s.last_results.iter().map(|r| r.icc_3_1).fold(f32::INFINITY, f32::min);
    let max_icc = s.last_results.iter().map(|r| r.icc_3_1).fold(f32::NEG_INFINITY, f32::max);
    let mean_icc: f32 = s.last_results.iter().map(|r| r.icc_3_1).sum::<f32>() / n as f32;
    let good = s.last_results.iter().filter(|r| r.icc_3_1 >= 0.99).count();
    let ok = s.last_results.iter().filter(|r| r.icc_3_1 >= 0.9).count();

    Json(json!({
        "calibrated_tensors": n,
        "min_icc": min_icc,
        "max_icc": max_icc,
        "mean_icc": mean_icc,
        "icc_gte_099": good,
        "icc_gte_09": ok,
        "results": s.last_results,
    }))
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn err<E: std::fmt::Display>(e: E) -> (StatusCode, Json<Value>) {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() })))
}

fn route_str(r: CodecRoute) -> &'static str {
    match r {
        CodecRoute::CamPq => "CamPq",
        CodecRoute::Passthrough => "Passthrough",
        CodecRoute::Skip => "Skip",
    }
}

fn load_tensor_rows(
    model_path: &str,
    tensor_pattern: &str,
) -> Result<(Vec<Vec<f32>>, String, Vec<u64>), (StatusCode, Json<Value>)> {
    let file = std::fs::File::open(model_path).map_err(|e| err(e))?;
    let mut reader = std::io::BufReader::new(file);
    let gguf = read_safetensors_header(&mut reader).map_err(|e| err(e))?;

    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name.contains(tensor_pattern))
        .ok_or_else(|| err(format!("no tensor matching '{tensor_pattern}'")))?;

    if tensor.dimensions.len() != 2 {
        return Err(err(format!("expected 2D tensor, got {:?}", tensor.dimensions)));
    }

    let flat = read_tensor_f32(&mut reader, &gguf, tensor).map_err(|e| err(e))?;
    let row_dim = tensor.dimensions[1] as usize;
    let rows: Vec<Vec<f32>> = flat.chunks_exact(row_dim).map(|c| c.to_vec()).collect();

    Ok((rows, tensor.name.clone(), tensor.dimensions.clone()))
}

fn measure_icc(rows: &[Vec<f32>], cb: &CamCodebook, samples: usize) -> f32 {
    let n = rows.len();
    if n < 3 { return f32::NAN; }
    let samples = samples.min(n * (n - 1) / 2);
    let mut rng = SplitMix64(0x9E3779B97F4A7C15);
    let mut truth = Vec::with_capacity(samples);
    let mut pred = Vec::with_capacity(samples);
    let mut count = 0;
    while count < samples {
        let i = (rng.next() as usize) % n;
        let j = (rng.next() as usize) % n;
        if i == j { continue; }
        truth.push(cosine(&rows[i], &rows[j]));
        let di = cb.decode(&cb.encode(&rows[i]));
        let dj = cb.decode(&cb.encode(&rows[j]));
        pred.push(cosine(&di, &dj));
        count += 1;
    }
    icc_3_1(&truth, &pred)
}

fn relative_l2_error(cb: &CamCodebook, rows: &[Vec<f32>]) -> f32 {
    let mut sum_err = 0.0f64;
    let mut sum_norm = 0.0f64;
    for row in rows {
        let decoded = cb.decode(&cb.encode(row));
        for (a, b) in row.iter().zip(decoded.iter()) {
            sum_err += ((a - b) as f64).powi(2);
        }
        for &a in row { sum_norm += (a as f64).powi(2); }
    }
    if sum_norm > 0.0 { (sum_err / sum_norm).sqrt() as f32 } else { 0.0 }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y; na += x * x; nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d > 0.0 { (dot / d) as f32 } else { 0.0 }
}

fn icc_3_1(truth: &[f32], pred: &[f32]) -> f32 {
    let n = truth.len();
    if n < 2 { return f32::NAN; }
    let mut grand = 0.0f64;
    for i in 0..n { grand += (truth[i] + pred[i]) as f64; }
    grand /= (2 * n) as f64;
    let (mut ms_r, mut ms_w) = (0.0f64, 0.0f64);
    for i in 0..n {
        let rm = ((truth[i] + pred[i]) as f64) / 2.0;
        ms_r += 2.0 * (rm - grand).powi(2);
        ms_w += (truth[i] as f64 - rm).powi(2) + (pred[i] as f64 - rm).powi(2);
    }
    ms_r /= (n - 1) as f64;
    ms_w /= n as f64;
    ((ms_r - ms_w) / (ms_r + ms_w)) as f32
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}
