//! Codec research — backing logic for the `/v1/shader/{tensors,calibrate,probe}`
//! DTO operations on the unified shader-driver API.
//!
//! Reuses:
//! - `ndarray::hpc::cam_pq::{train_geometric, CamCodebook}` — production codec
//! - `ndarray::hpc::safetensors::read_safetensors_header` — tensor directory
//! - `ndarray::hpc::gguf::read_tensor_f32` — BF16/F16/F32 dequant
//! - `lance_graph_contract::cam::route_tensor` — CamPq / Passthrough / Skip
//!
//! Zero new feature gates — rides on the existing `serve` / `grpc`.

use std::fs::File;
use std::io::BufReader;

use lance_graph_contract::cam::{route_tensor, CodecRoute, NUM_CENTROIDS, NUM_SUBSPACES};
use ndarray::hpc::cam_pq::{self, CamCodebook};
use ndarray::hpc::gguf::read_tensor_f32;
use ndarray::hpc::safetensors::read_safetensors_header;

use crate::wire::{
    WireCalibrateRequest, WireCalibrateResponse, WireProbeEntry, WireProbeRequest,
    WireProbeResponse, WireTensorEntry, WireTensorsRequest, WireTensorsResponse,
};

// ─── Public entry points ────────────────────────────────────────────────────

pub fn list_tensors(req: &WireTensorsRequest) -> Result<WireTensorsResponse, String> {
    let file = File::open(&req.model_path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(file);
    let gguf = read_safetensors_header(&mut reader)?;

    let filter = req.route_filter.as_deref();
    let entries: Vec<WireTensorEntry> = gguf
        .tensors
        .iter()
        .filter_map(|t| {
            let route = route_tensor(&t.name, &t.dimensions);
            let route_str = route_str(route);
            if let Some(f) = filter {
                if route_str != f {
                    return None;
                }
            }
            Some(WireTensorEntry {
                name: t.name.clone(),
                dims: t.dimensions.clone(),
                dtype: format!("{:?}", t.dtype),
                route: route_str.to_string(),
                n_elements: t.element_count(),
            })
        })
        .collect();

    Ok(WireTensorsResponse {
        total: gguf.tensors.len(),
        shown: entries.len(),
        cam_pq: entries.iter().filter(|e| e.route == "CamPq").count(),
        passthrough: entries.iter().filter(|e| e.route == "Passthrough").count(),
        skip: entries.iter().filter(|e| e.route == "Skip").count(),
        tensors: entries,
    })
}

pub fn calibrate_tensor(req: &WireCalibrateRequest) -> Result<WireCalibrateResponse, String> {
    let (rows, tensor_name, dims) = load_tensor_rows(&req.model_path, &req.tensor_name)?;
    let n_rows = rows.len();
    let row_dim = if rows.is_empty() { 0 } else { rows[0].len() };
    let adjusted_dim = (row_dim / req.num_subspaces) * req.num_subspaces;
    if adjusted_dim == 0 {
        return Err(format!("row_dim {row_dim} < num_subspaces {}", req.num_subspaces));
    }

    let t0 = std::time::Instant::now();
    let calibration_rows: Vec<Vec<f32>> = match req.max_rows {
        Some(n) if n < n_rows => rows[..n].iter().map(|r| r[..adjusted_dim].to_vec()).collect(),
        _ => rows.iter().map(|r| r[..adjusted_dim].to_vec()).collect(),
    };
    let cal_n = calibration_rows.len();

    let codebook =
        cam_pq::train_geometric(&calibration_rows, adjusted_dim, req.kmeans_iterations);

    let sliced: Vec<Vec<f32>> = rows.iter().map(|r| r[..adjusted_dim].to_vec()).collect();
    let icc = measure_icc(&sliced, &codebook, req.icc_samples);
    let sample_n = sliced.len().min(512);
    let mean_err = codebook.mean_reconstruction_error(&sliced[..sample_n]);
    let rel_err = relative_l2_error(&codebook, &sliced[..sample_n]);

    let elapsed = t0.elapsed();
    let subspace_dim = adjusted_dim / req.num_subspaces;
    let codebook_bytes = 24 + req.num_subspaces * req.num_centroids * subspace_dim * 4;
    let fp_bytes = 20 + n_rows * 6;

    Ok(WireCalibrateResponse {
        tensor_name,
        dims,
        n_rows,
        row_dim,
        adjusted_dim,
        num_subspaces: req.num_subspaces,
        num_centroids: req.num_centroids,
        calibration_rows: cal_n,
        icc_3_1: icc,
        mean_reconstruction_error: mean_err,
        relative_l2_error: rel_err,
        codebook_bytes,
        fingerprints_bytes: fp_bytes,
        elapsed_ms: elapsed.as_millis() as u64,
    })
}

pub fn row_count_probe(req: &WireProbeRequest) -> Result<WireProbeResponse, String> {
    let (rows, tensor_name, _dims) = load_tensor_rows(&req.model_path, &req.tensor_name)?;
    let n_rows = rows.len();
    let row_dim = if rows.is_empty() { 0 } else { rows[0].len() };
    let adjusted_dim = (row_dim / NUM_SUBSPACES) * NUM_SUBSPACES;
    if adjusted_dim == 0 {
        return Err(format!("row_dim {row_dim} < 6"));
    }
    let sliced: Vec<Vec<f32>> = rows.iter().map(|r| r[..adjusted_dim].to_vec()).collect();

    let mut counts = req.row_counts.clone();
    if !counts.contains(&n_rows) {
        counts.push(n_rows);
    }
    counts.sort();
    counts.dedup();

    let mut entries = Vec::new();
    for &n in &counts {
        if n > n_rows || n == 0 {
            continue;
        }
        let t0 = std::time::Instant::now();
        let training = &sliced[..n];
        let cb = cam_pq::train_geometric(training, adjusted_dim, 20);
        let elapsed = t0.elapsed();
        entries.push(WireProbeEntry {
            n_train: n,
            icc_train: measure_icc(training, &cb, req.icc_samples),
            icc_all_rows: measure_icc(&sliced, &cb, req.icc_samples),
            relative_l2_error: relative_l2_error(&cb, &sliced[..sliced.len().min(512)]),
            elapsed_ms: elapsed.as_millis() as u64,
        });
    }

    Ok(WireProbeResponse {
        tensor_name,
        n_rows,
        row_dim,
        adjusted_dim,
        num_subspaces: NUM_SUBSPACES,
        num_centroids: NUM_CENTROIDS,
        entries,
    })
}

// ─── Internals ──────────────────────────────────────────────────────────────

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
) -> Result<(Vec<Vec<f32>>, String, Vec<u64>), String> {
    let file = File::open(model_path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(file);
    let gguf = read_safetensors_header(&mut reader)?;

    let tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name.contains(tensor_pattern))
        .ok_or_else(|| format!("no tensor matching '{tensor_pattern}'"))?;

    if tensor.dimensions.len() != 2 {
        return Err(format!("expected 2D tensor, got {:?}", tensor.dimensions));
    }

    let flat = read_tensor_f32(&mut reader, &gguf, tensor)?;
    let row_dim = tensor.dimensions[1] as usize;
    let rows: Vec<Vec<f32>> = flat.chunks_exact(row_dim).map(|c| c.to_vec()).collect();
    Ok((rows, tensor.name.clone(), tensor.dimensions.clone()))
}

fn measure_icc(rows: &[Vec<f32>], cb: &CamCodebook, samples: usize) -> f32 {
    let n = rows.len();
    if n < 3 {
        return f32::NAN;
    }
    let samples = samples.min(n * (n - 1) / 2);
    let mut rng = SplitMix64(0x9E3779B97F4A7C15);
    let mut truth = Vec::with_capacity(samples);
    let mut pred = Vec::with_capacity(samples);
    let mut count = 0;
    while count < samples {
        let i = (rng.next() as usize) % n;
        let j = (rng.next() as usize) % n;
        if i == j {
            continue;
        }
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
        for &a in row {
            sum_norm += (a as f64).powi(2);
        }
    }
    if sum_norm > 0.0 {
        (sum_err / sum_norm).sqrt() as f32
    } else {
        0.0
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d > 0.0 {
        (dot / d) as f32
    } else {
        0.0
    }
}

fn icc_3_1(truth: &[f32], pred: &[f32]) -> f32 {
    let n = truth.len();
    if n < 2 {
        return f32::NAN;
    }
    let mut grand = 0.0f64;
    for i in 0..n {
        grand += (truth[i] + pred[i]) as f64;
    }
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
