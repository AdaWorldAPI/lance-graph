//! CAM-PQ calibration CLI — D2 of the CAM-PQ production wiring plan.
//!
//! Reads a safetensors / GGUF model checkpoint, classifies every tensor via
//! `lance_graph_contract::cam::route_tensor`, trains a per-tensor CAM-PQ
//! codebook for argmax-regime tensors (attention Q/K/V/O, MLP gate/up/down),
//! encodes each row to a 6-byte fingerprint, and writes codebooks +
//! fingerprints + a manifest to disk.
//!
//! # Output layout
//!
//! ```text
//! <out_dir>/
//!   codebooks/<sanitized>.cbk        per-tensor CamCodebook (binary)
//!   fingerprints/<sanitized>.fp      per-tensor fingerprints (flat u8, 6 bytes × n_rows)
//!   passthrough/<sanitized>.f32      index-regime tensors stored as raw f32 LE
//!   manifest.json                    list of tensors with route, dims, paths, ICC, err
//! ```
//!
//! # Binary formats
//!
//! Codebook (`*.cbk`):
//! ```text
//! magic        [u8; 4]  b"CMPQ"
//! version      u32 LE   1
//! subspaces    u32 LE   6
//! centroids    u32 LE   256
//! subspace_dim u32 LE   original_dim / 6
//! total_dim    u32 LE   original row dim (subspaces × subspace_dim)
//! then 6 × (centroids × subspace_dim) f32 LE centroids
//! ```
//!
//! Fingerprints (`*.fp`):
//! ```text
//! magic    [u8; 4]  b"CMFP"
//! version  u32 LE   1
//! n_rows   u64 LE
//! row_dim  u32 LE   original row dim
//! then 6 × n_rows bytes of packed fingerprints (row-major).
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features calibrate --bin cam_pq_calibrate \
//!     --manifest-path crates/bgz-tensor/Cargo.toml \
//!     -- <model.safetensors> <out_dir> [--max-rows N] [--icc-samples K]
//! ```

use ndarray::hpc::cam_pq::{self, CamCodebook, CamFingerprint, NUM_CENTROIDS, NUM_SUBSPACES};
use ndarray::hpc::gguf::read_tensor_f32;
use ndarray::hpc::safetensors::read_safetensors_header;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use lance_graph_contract::cam::{route_tensor, CodecRoute};

const KMEANS_ITERATIONS: usize = 20;
const DEFAULT_ICC_SAMPLES: usize = 512;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: cam_pq_calibrate <model.safetensors|model.gguf> <out_dir> \
             [--max-rows N] [--icc-samples K]"
        );
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let mut max_rows: Option<usize> = None;
    let mut icc_samples: usize = DEFAULT_ICC_SAMPLES;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--max-rows" => {
                i += 1;
                max_rows = Some(args[i].parse().expect("--max-rows expects integer"));
            }
            "--icc-samples" => {
                i += 1;
                icc_samples = args[i].parse().expect("--icc-samples expects integer");
            }
            other => {
                eprintln!("Unknown flag: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("cam_pq_calibrate");
    eprintln!("  model:       {}", model_path.display());
    eprintln!("  out_dir:     {}", out_dir.display());
    eprintln!(
        "  max_rows:    {}",
        max_rows.map_or("all".to_string(), |n| n.to_string())
    );
    eprintln!("  icc_samples: {icc_samples}");

    fs::create_dir_all(out_dir.join("codebooks")).expect("mkdir codebooks");
    fs::create_dir_all(out_dir.join("fingerprints")).expect("mkdir fingerprints");
    fs::create_dir_all(out_dir.join("passthrough")).expect("mkdir passthrough");

    let file = File::open(&model_path).expect("open model");
    let mut reader = BufReader::new(file);

    // Dispatch on extension: .gguf vs .safetensors
    let is_safetensors = model_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|e| e.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false);

    let gguf = if is_safetensors {
        read_safetensors_header(&mut reader).expect("read safetensors header")
    } else {
        ndarray::hpc::gguf::read_gguf_header(&mut reader).expect("read gguf header")
    };

    eprintln!("  tensors:     {}", gguf.tensors.len());

    let mut manifest_entries: Vec<ManifestEntry> = Vec::new();
    let t_start = Instant::now();

    for (idx, tensor) in gguf.tensors.iter().enumerate() {
        let dims_u64: Vec<u64> = tensor.dimensions.clone();
        let route = route_tensor(&tensor.name, &dims_u64);

        let sanitized = sanitize_name(&tensor.name);

        eprint!(
            "[{:>4}/{}] {:>12?} {:<60} dims={:?}",
            idx + 1,
            gguf.tensors.len(),
            route,
            truncate(&tensor.name, 60),
            dims_u64
        );

        match route {
            CodecRoute::CamPq => {
                let t0 = Instant::now();
                let (row_dim, n_rows) = match row_layout(&dims_u64) {
                    Some(v) => v,
                    None => {
                        eprintln!("  [skip: not a 2D matrix]");
                        continue;
                    }
                };
                let row_dim_u = row_dim as usize;
                let n_rows_u = n_rows as usize;

                // Read the full tensor as f32.
                let flat = match read_tensor_f32(&mut reader, &gguf, tensor) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("  [read error: {e}]");
                        continue;
                    }
                };

                // Chunk into rows. Limit row count for calibration if requested.
                let rows_full: Vec<Vec<f32>> =
                    flat.chunks_exact(row_dim_u).map(|c| c.to_vec()).collect();
                assert_eq!(rows_full.len(), n_rows_u);

                let calibration_rows: &[Vec<f32>] = match max_rows {
                    Some(n) if n < n_rows_u => &rows_full[..n],
                    _ => &rows_full,
                };

                // CAM-PQ requires total_dim divisible by 6. If row_dim isn't,
                // train on the largest multiple of 6 ≤ row_dim.
                let adjusted_dim = (row_dim_u / NUM_SUBSPACES) * NUM_SUBSPACES;
                if adjusted_dim == 0 {
                    eprintln!("  [skip: row_dim {row_dim_u} < 6]");
                    continue;
                }

                let codebook =
                    cam_pq::train_geometric(calibration_rows, adjusted_dim, KMEANS_ITERATIONS);

                // Encode every row (including any beyond max_rows).
                let fingerprints: Vec<CamFingerprint> =
                    rows_full.iter().map(|r| codebook.encode(r)).collect();

                // Reconstruction error on a sample of the full population.
                // Slice each row to adjusted_dim — CAM-PQ only encodes the
                // first `adjusted_dim` floats; ndarray's
                // `mean_reconstruction_error` would panic on row_dim mismatch
                // when adjusted_dim < row_dim (non-6-multiple case).
                let sample_n = rows_full.len().min(1024);
                let recon_sample: Vec<Vec<f32>> = rows_full[..sample_n]
                    .iter()
                    .map(|r| r[..adjusted_dim].to_vec())
                    .collect();
                let mean_err = codebook.mean_reconstruction_error(&recon_sample);
                let rel_err = relative_l2_error(&codebook, &recon_sample);

                // Write codebook.
                let cbk_path = out_dir.join("codebooks").join(format!("{sanitized}.cbk"));
                write_codebook(&cbk_path, &codebook).expect("write codebook");
                let cbk_sha = sha256_file(&cbk_path).expect("sha256 codebook");

                // Write fingerprints.
                let fp_path = out_dir.join("fingerprints").join(format!("{sanitized}.fp"));
                write_fingerprints(&fp_path, row_dim as u32, &fingerprints)
                    .expect("write fingerprints");
                let fp_sha = sha256_file(&fp_path).expect("sha256 fingerprints");

                // ICC_3_1 on pairwise cosines between ground-truth rows and
                // their decoded counterparts. D5 gate fires on this number.
                let icc = measure_icc(&rows_full, &codebook, icc_samples);

                let elapsed = t0.elapsed();
                eprintln!(
                    "  codebook={} KB fp={} KB err={:.4} rel_err={:.4} icc={:.4} time={:.1}s",
                    fs::metadata(&cbk_path).map(|m| m.len() / 1024).unwrap_or(0),
                    fs::metadata(&fp_path).map(|m| m.len() / 1024).unwrap_or(0),
                    mean_err,
                    rel_err,
                    icc,
                    elapsed.as_secs_f32(),
                );

                manifest_entries.push(ManifestEntry {
                    name: tensor.name.clone(),
                    dtype: format!("{:?}", tensor.dtype),
                    dims: dims_u64.clone(),
                    route: "CamPq".into(),
                    codebook_file: Some(format!("codebooks/{sanitized}.cbk")),
                    codebook_sha256: Some(cbk_sha),
                    fingerprints_file: Some(format!("fingerprints/{sanitized}.fp")),
                    fingerprints_sha256: Some(fp_sha),
                    passthrough_file: None,
                    passthrough_sha256: None,
                    n_rows: Some(n_rows),
                    row_dim: Some(row_dim as u32),
                    mean_reconstruction_error: Some(mean_err),
                    relative_l2_error: Some(rel_err),
                    icc_3_1: Some(icc),
                });
            }
            CodecRoute::Passthrough => {
                let flat = match read_tensor_f32(&mut reader, &gguf, tensor) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("  [read error: {e}]");
                        continue;
                    }
                };
                let pt_path = out_dir.join("passthrough").join(format!("{sanitized}.f32"));
                write_f32_le(&pt_path, &flat).expect("write passthrough");
                let pt_sha = sha256_file(&pt_path).expect("sha256 passthrough");
                eprintln!(
                    "  [passthrough {:.1} MB]",
                    fs::metadata(&pt_path)
                        .map(|m| m.len() as f64 / 1e6)
                        .unwrap_or(0.0),
                );
                manifest_entries.push(ManifestEntry {
                    name: tensor.name.clone(),
                    dtype: format!("{:?}", tensor.dtype),
                    dims: dims_u64.clone(),
                    route: "Passthrough".into(),
                    codebook_file: None,
                    codebook_sha256: None,
                    fingerprints_file: None,
                    fingerprints_sha256: None,
                    passthrough_file: Some(format!("passthrough/{sanitized}.f32")),
                    passthrough_sha256: Some(pt_sha),
                    n_rows: None,
                    row_dim: None,
                    mean_reconstruction_error: None,
                    relative_l2_error: None,
                    icc_3_1: None,
                });
            }
            CodecRoute::Skip => {
                eprintln!("  [skip]");
                manifest_entries.push(ManifestEntry {
                    name: tensor.name.clone(),
                    dtype: format!("{:?}", tensor.dtype),
                    dims: dims_u64.clone(),
                    route: "Skip".into(),
                    codebook_file: None,
                    codebook_sha256: None,
                    fingerprints_file: None,
                    fingerprints_sha256: None,
                    passthrough_file: None,
                    passthrough_sha256: None,
                    n_rows: None,
                    row_dim: None,
                    mean_reconstruction_error: None,
                    relative_l2_error: None,
                    icc_3_1: None,
                });
            }
        }
    }

    // Write manifest.
    let manifest = Manifest {
        model: model_path.display().to_string(),
        kmeans_iterations: KMEANS_ITERATIONS,
        num_subspaces: NUM_SUBSPACES as u32,
        num_centroids: NUM_CENTROIDS as u32,
        max_rows_calibration: max_rows,
        icc_samples,
        entries: manifest_entries,
    };
    let manifest_path = out_dir.join("manifest.json");
    let file = File::create(&manifest_path).expect("create manifest");
    serde_json::to_writer_pretty(BufWriter::new(file), &manifest).expect("write manifest");

    let total = t_start.elapsed();
    eprintln!(
        "done in {:.1}s ({:.1} min)",
        total.as_secs_f32(),
        total.as_secs_f32() / 60.0
    );
    eprintln!("manifest: {}", manifest_path.display());

    // Summary.
    let campq = manifest
        .entries
        .iter()
        .filter(|e| e.route == "CamPq")
        .count();
    let pt = manifest
        .entries
        .iter()
        .filter(|e| e.route == "Passthrough")
        .count();
    let skip = manifest
        .entries
        .iter()
        .filter(|e| e.route == "Skip")
        .count();
    eprintln!("  CamPq       tensors: {campq}");
    eprintln!("  Passthrough tensors: {pt}");
    eprintln!("  Skip        tensors: {skip}");

    let min_icc = manifest
        .entries
        .iter()
        .filter_map(|e| e.icc_3_1)
        .fold(f32::INFINITY, f32::min);
    let max_err = manifest
        .entries
        .iter()
        .filter_map(|e| e.relative_l2_error)
        .fold(0.0f32, f32::max);
    if campq > 0 {
        eprintln!("  min ICC_3_1 across CamPq tensors: {min_icc:.4}");
        eprintln!("  max relative L2 error:             {max_err:.4}");
        if min_icc < 0.99 {
            eprintln!("WARN: at least one tensor has ICC < 0.99 — D7 fallback threshold applies.");
        }
    }
}

// ─── serialization ──────────────────────────────────────────────────────────

fn write_codebook(path: &Path, cb: &CamCodebook) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(b"CMPQ")?;
    w.write_all(&1u32.to_le_bytes())?; // version
    w.write_all(&(NUM_SUBSPACES as u32).to_le_bytes())?;
    w.write_all(&(NUM_CENTROIDS as u32).to_le_bytes())?;
    w.write_all(&(cb.subspace_dim as u32).to_le_bytes())?;
    w.write_all(&(cb.total_dim as u32).to_le_bytes())?;
    for s in 0..NUM_SUBSPACES {
        let cb_s = &cb.codebooks[s];
        // Pad to NUM_CENTROIDS if the subspace had fewer unique centroids
        // (kmeans may return fewer than NUM_CENTROIDS when n < k). Remaining
        // centroids are zero-filled — encoder will never select them because
        // squared_l2 against a zero vector dominates for any non-trivial row.
        for c in 0..NUM_CENTROIDS {
            let centroid = cb_s.centroids.get(c).map(|v| v.as_slice()).unwrap_or(&[]);
            for d in 0..cb.subspace_dim {
                let val = centroid.get(d).copied().unwrap_or(0.0);
                w.write_all(&val.to_le_bytes())?;
            }
        }
    }
    w.flush()?;
    Ok(())
}

fn write_fingerprints(path: &Path, row_dim: u32, fps: &[CamFingerprint]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(b"CMFP")?;
    w.write_all(&1u32.to_le_bytes())?; // version
    w.write_all(&(fps.len() as u64).to_le_bytes())?;
    w.write_all(&row_dim.to_le_bytes())?;
    for fp in fps {
        w.write_all(fp)?;
    }
    w.flush()?;
    Ok(())
}

fn write_f32_le(path: &Path, data: &[f32]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

// ─── measurement ────────────────────────────────────────────────────────────

/// ICC_3_1 computed on pairwise cosines between ground-truth rows and
/// decoded rows. Matches the protocol used by `codec_rnd_bench.rs`.
fn measure_icc(rows: &[Vec<f32>], cb: &CamCodebook, samples: usize) -> f32 {
    let n_rows = rows.len();
    if n_rows < 3 {
        return f32::NAN;
    }
    let samples = samples.min(n_rows * (n_rows - 1) / 2);
    let mut rng = SimpleRng::new(0x9E3779B97F4A7C15);
    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(samples);
    while pairs.len() < samples {
        let i = (rng.next() as usize) % n_rows;
        let j = (rng.next() as usize) % n_rows;
        if i != j {
            pairs.push((i.min(j), i.max(j)));
        }
    }
    let adjusted_dim = (rows[0].len() / NUM_SUBSPACES) * NUM_SUBSPACES;
    let mut truth: Vec<f32> = Vec::with_capacity(samples);
    let mut pred: Vec<f32> = Vec::with_capacity(samples);
    for (i, j) in pairs {
        let t = cosine(&rows[i][..adjusted_dim], &rows[j][..adjusted_dim]);
        let di = cb.decode(&cb.encode(&rows[i]));
        let dj = cb.decode(&cb.encode(&rows[j]));
        let p = cosine(&di, &dj);
        truth.push(t);
        pred.push(p);
    }
    icc_3_1(&truth, &pred)
}

fn relative_l2_error(cb: &CamCodebook, rows: &[Vec<f32>]) -> f32 {
    if rows.is_empty() {
        return f32::NAN;
    }
    let adjusted_dim = (rows[0].len() / NUM_SUBSPACES) * NUM_SUBSPACES;
    let mut sum_err = 0.0f64;
    let mut sum_norm = 0.0f64;
    for row in rows {
        let decoded = cb.decode(&cb.encode(row));
        let slice = &row[..adjusted_dim.min(row.len())];
        for (a, b) in slice.iter().zip(decoded.iter()) {
            let d = (a - b) as f64;
            sum_err += d * d;
        }
        for &a in slice {
            sum_norm += (a as f64) * (a as f64);
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
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..n {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom > 0.0 {
        (dot / denom) as f32
    } else {
        0.0
    }
}

fn icc_3_1(truth: &[f32], pred: &[f32]) -> f32 {
    let n = truth.len();
    if n < 2 {
        return f32::NAN;
    }
    let mut ms_r = 0.0f64;
    let mut ms_w = 0.0f64;
    let mut grand = 0.0f64;
    for i in 0..n {
        grand += (truth[i] + pred[i]) as f64;
    }
    grand /= (2 * n) as f64;
    for i in 0..n {
        let row_mean = ((truth[i] + pred[i]) as f64) / 2.0;
        ms_r += 2.0 * (row_mean - grand).powi(2);
        ms_w += (truth[i] as f64 - row_mean).powi(2) + (pred[i] as f64 - row_mean).powi(2);
    }
    ms_r /= (n - 1) as f64;
    ms_w /= n as f64;
    let icc = (ms_r - ms_w) / (ms_r + ms_w);
    icc as f32
}

// ─── helpers ────────────────────────────────────────────────────────────────

/// Determine (row_dim, n_rows) for a 2D tensor.
/// Convention: safetensors stores tensors in row-major with `[n_rows, row_dim]`
/// shape. CAM-PQ encodes one fingerprint per row, per this layout.
fn row_layout(dims: &[u64]) -> Option<(u64, u64)> {
    if dims.len() == 2 {
        Some((dims[1], dims[0]))
    } else {
        None
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn truncate(s: &str, n: usize) -> &str {
    if s.len() <= n {
        s
    } else {
        &s[s.len() - n..]
    }
}

/// SplitMix64 — deterministic seed → sample index pairs for ICC.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// ─── manifest ───────────────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct Manifest {
    model: String,
    kmeans_iterations: usize,
    num_subspaces: u32,
    num_centroids: u32,
    max_rows_calibration: Option<usize>,
    icc_samples: usize,
    entries: Vec<ManifestEntry>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ManifestEntry {
    name: String,
    dtype: String,
    dims: Vec<u64>,
    route: String,
    codebook_file: Option<String>,
    codebook_sha256: Option<String>,
    fingerprints_file: Option<String>,
    fingerprints_sha256: Option<String>,
    passthrough_file: Option<String>,
    passthrough_sha256: Option<String>,
    n_rows: Option<u64>,
    row_dim: Option<u32>,
    mean_reconstruction_error: Option<f32>,
    relative_l2_error: Option<f32>,
    icc_3_1: Option<f32>,
}
