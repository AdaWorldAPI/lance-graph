//! codec_rnd_bench — R&D psychometric benchmark for codec candidates.
//!
//! Runs all codec candidates against the same tensor population,
//! computes the full metric suite (10 metrics), outputs a markdown table.
//! Cronbach's α across codecs reveals factor structure.
//!
//! Usage:
//!   cargo run --release --example codec_rnd_bench \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use bgz_tensor::quality::{
    pearson, spearman, kendall_tau, mae, rmse, top_k_recall,
    cronbach_alpha, cohens_kappa, bias_variance, icc_3_1,
};
use bgz_tensor::projection::Base17;
use bgz_tensor::hhtl_cache::HhtlCache;
use bgz_tensor::hhtl_d::build_hip_families;
use highheelbgz::rehydrate::SpiralEncoding;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::GgmlType;
use ndarray::simd::bf16_to_f32_batch;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_SAMPLE: usize = 128;
const TOP_K: usize = 5;

// ═════════════════════════════════════════════════════════════════════
// Tensor loading
// ═════════════════════════════════════════════════════════════════════

fn load_rows(path: &str, tensor_substr: &str, max_rows: usize) -> (Vec<Vec<f32>>, String) {
    let file = File::open(path).expect("open");
    let mut reader = BufReader::new(file);
    let header = read_safetensors_header(&mut reader).expect("parse");
    let t = header.tensors.iter().find(|t| t.name.contains(tensor_substr))
        .expect(&format!("tensor '{}' not found", tensor_substr));
    let n: usize = t.dimensions.iter().map(|&d| d as usize).product();
    let n_rows = t.dimensions[0] as usize;
    let n_cols: usize = t.dimensions.iter().skip(1).map(|&d| d as usize).product();
    reader.seek(SeekFrom::Start(header.tensor_data_offset + t.offset)).unwrap();
    let mut raw = vec![0u8; n * 2];
    reader.read_exact(&mut raw).unwrap();
    let f32_data: Vec<f32> = {
        let u16s: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut out = vec![0.0f32; u16s.len()];
        bf16_to_f32_batch(&u16s, &mut out);
        out
    };
    let stride = n_rows.max(1) / max_rows.min(n_rows);
    let rows: Vec<Vec<f32>> = (0..max_rows.min(n_rows))
        .map(|i| {
            let ri = (i * stride).min(n_rows - 1);
            f32_data[ri * n_cols..(ri + 1) * n_cols].to_vec()
        }).collect();
    (rows, format!("{} [{}×{}]", t.name, n_rows, n_cols))
}

// ═════════════════════════════════════════════════════════════════════
// Ground truth pairwise cosine
// ═════════════════════════════════════════════════════════════════════

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64; let y = b[i] as f64;
        dot += x * y; na += x * x; nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-15 { 0.0 } else { dot / d }
}

fn pairwise_cosines(rows: &[Vec<f32>]) -> Vec<f64> {
    let n = rows.len();
    let mut scores = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            scores.push(cosine(&rows[i], &rows[j]));
        }
    }
    scores
}

// ═════════════════════════════════════════════════════════════════════
// Codec candidates — each produces a pairwise score vector
// ═════════════════════════════════════════════════════════════════════

trait CodecCandidate {
    fn name(&self) -> &str;
    fn bytes_per_row(&self) -> usize;
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64>;
}

/// Passthrough — raw cosine (baseline, exact).
struct Passthrough;
impl CodecCandidate for Passthrough {
    fn name(&self) -> &str { "Passthrough" }
    fn bytes_per_row(&self) -> usize { 0 } // not compressed
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> { pairwise_cosines(rows) }
}

/// Base17 signature — project to 17-dim, cosine there.
struct Base17Sig;
impl CodecCandidate for Base17Sig {
    fn name(&self) -> &str { "Base17" }
    fn bytes_per_row(&self) -> usize { 34 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        let b17s: Vec<Base17> = rows.iter().map(|r| Base17::from_f32(r)).collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(b17s[i].cosine(&b17s[j]));
            }
        }
        scores
    }
}

/// Direct i8 quantization of pairwise cosines.
struct DirectI8;
impl CodecCandidate for DirectI8 {
    fn name(&self) -> &str { "Direct-i8" }
    fn bytes_per_row(&self) -> usize { 1 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        let gt = pairwise_cosines(rows);
        gt.iter().map(|&c| {
            let q = (c * 127.0).round().clamp(-127.0, 127.0) as i8;
            q as f64 / 127.0
        }).collect()
    }
}

/// Spiral signature at K=8.
struct SpiralK8;
impl CodecCandidate for SpiralK8 {
    fn name(&self) -> &str { "Spiral-K8" }
    fn bytes_per_row(&self) -> usize { 278 }
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        let encs: Vec<SpiralEncoding> = rows.iter()
            .map(|r| SpiralEncoding::encode(r, 0, 3, 8))
            .collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                scores.push(encs[i].cosine(&encs[j]));
            }
        }
        scores
    }
}

/// RaBitQ — sign-quantized JL + Hamming + dot_correction.
struct RaBitQCodec { dim: usize }
impl CodecCandidate for RaBitQCodec {
    fn name(&self) -> &str { "RaBitQ" }
    fn bytes_per_row(&self) -> usize { (self.dim + 63) / 64 * 8 + 8 } // binary + norm + corr
    fn pairwise_scores(&self, rows: &[Vec<f32>]) -> Vec<f64> {
        use bgz17::rabitq_compat::{OrthogonalMatrix, RaBitQEncoding};
        use bgz17::palette::Palette;
        let rot = OrthogonalMatrix::hadamard(self.dim);
        let empty_palette = Palette::build(&[], 0, 0);
        let encs: Vec<RaBitQEncoding> = rows.iter()
            .map(|r| RaBitQEncoding::encode(r, &rot, &empty_palette))
            .collect();
        let n = rows.len();
        let mut scores = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                // RaBitQ distance → cosine estimate
                let d = encs[i].distance_rabitq(&encs[j]);
                // Convert L2² to cosine: cos = 1 - d/(2*norm_i*norm_j)
                let ni = encs[i].norm; let nj = encs[j].norm;
                let cos_est = if ni > 0.0 && nj > 0.0 {
                    1.0 - d as f64 / (2.0 * ni as f64 * nj as f64)
                } else { 0.0 };
                scores.push(cos_est.clamp(-1.0, 1.0));
            }
        }
        scores
    }
}

// ═════════════════════════════════════════════════════════════════════
// Bench runner
// ═════════════════════════════════════════════════════════════════════

struct BenchResult {
    codec_name: String,
    bytes_per_row: usize,
    pearson_r: f64,
    spearman_rho: f64,
    kendall_t: f64,
    mae_val: f64,
    rmse_val: f64,
    top_k_recall: f64,
    icc: f64,
    bias: f64,
    variance: f64,
}

fn run_bench(codecs: &[Box<dyn CodecCandidate>], rows: &[Vec<f32>], gt: &[f64]) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let mut all_scores: Vec<Vec<f64>> = Vec::new(); // for Cronbach's α

    for codec in codecs {
        let scores = codec.pairwise_scores(rows);
        let n = gt.len().min(scores.len());
        let gt_slice = &gt[..n];
        let sc_slice = &scores[..n];

        let errors: Vec<f64> = (0..n).map(|i| sc_slice[i] - gt_slice[i]).collect();
        let (b, v) = bias_variance(&errors);

        // Top-1 argmax per row
        let n_rows = rows.len();
        let mut gt_argmax = vec![0usize; n_rows];
        let mut sc_argmax = vec![0usize; n_rows];
        let mut pair_idx = 0usize;
        for i in 0..n_rows {
            let mut best_gt = f64::NEG_INFINITY;
            let mut best_sc = f64::NEG_INFINITY;
            for j in (i + 1)..n_rows {
                if gt[pair_idx] > best_gt { best_gt = gt[pair_idx]; gt_argmax[i] = j; }
                if scores[pair_idx] > best_sc { best_sc = scores[pair_idx]; sc_argmax[i] = j; }
                pair_idx += 1;
            }
        }

        results.push(BenchResult {
            codec_name: codec.name().to_string(),
            bytes_per_row: codec.bytes_per_row(),
            pearson_r: pearson(gt_slice, sc_slice),
            spearman_rho: spearman(gt_slice, sc_slice),
            kendall_t: kendall_tau(gt_slice, sc_slice),
            mae_val: mae(gt_slice, sc_slice),
            rmse_val: rmse(gt_slice, sc_slice),
            top_k_recall: top_k_recall(gt_slice, sc_slice, TOP_K),
            icc: icc_3_1(gt_slice, sc_slice),
            bias: b,
            variance: v,
        });
        all_scores.push(scores);
    }

    // Cronbach's α across all codecs (excluding passthrough which IS ground truth)
    if all_scores.len() >= 2 {
        let items: Vec<Vec<f64>> = all_scores[1..].to_vec(); // skip passthrough
        let alpha = cronbach_alpha(&items);
        eprintln!("  Cronbach's α (inter-codec, excl. passthrough): {:.4}", alpha);
    }

    results
}

fn print_table(population: &str, results: &[BenchResult]) {
    println!("\n### {}", population);
    println!();
    println!("| Codec | B/row | Pearson | Spearman | Kendall | MAE | RMSE | Top-{} | ICC | Bias | Var |",
        TOP_K);
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for r in results {
        println!("| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2e} | {:.2e} |",
            r.codec_name, r.bytes_per_row,
            r.pearson_r, r.spearman_rho, r.kendall_t,
            r.mae_val, r.rmse_val, r.top_k_recall,
            r.icc, r.bias, r.variance);
    }
}

// ═════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════

fn main() {
    let model_path = std::env::args().nth(1)
        .expect("usage: codec_rnd_bench <model.safetensors>");

    println!("# Codec R&D Bench — Psychometric Measurement");
    println!();
    println!("Model: `{}`", model_path);
    println!("Sample: {} rows per population, {} metrics per codec", N_SAMPLE, 10);

    // Populations from the model
    let populations: Vec<(&str, &str)> = vec![
        ("self_attn.k_proj.weight", "Attention k_proj (argmax regime)"),
        ("mlp.gate_proj.weight", "MLP gate_proj (argmax, SiLU-gated)"),
        ("text_embedding.weight", "Text embedding (index regime)"),
        ("code_predictor.model.codec_embedding.0.weight", "Audio codec emb (index)"),
        ("self_attn.q_proj.weight", "Attention q_proj (argmax, must match k)"),
    ];

    let t0 = Instant::now();

    for (tensor_substr, pop_name) in &populations {
        let (rows, tensor_name) = match load_rows(&model_path, tensor_substr, N_SAMPLE) {
            r => r,
        };
        let n_cols = if rows.is_empty() { 0 } else { rows[0].len() };
        println!("\n---");
        println!("**Population: {}** — `{}`", pop_name, tensor_name);

        let gt = pairwise_cosines(&rows);

        let codecs: Vec<Box<dyn CodecCandidate>> = vec![
            Box::new(Passthrough),
            Box::new(Base17Sig),
            Box::new(DirectI8),
            Box::new(SpiralK8),
            Box::new(RaBitQCodec { dim: n_cols }),
        ];

        let results = run_bench(&codecs, &rows, &gt);
        print_table(pop_name, &results);
    }

    println!("\n---");
    println!("Total wall: {:.1}s", t0.elapsed().as_secs_f32());
    println!();
    println!("## Interpretation");
    println!();
    println!("- **ICC** is the key metric: value-level agreement, not just ranking.");
    println!("- **Cronbach's α** printed to stderr per population — shows inter-codec factor structure.");
    println!("- **Bias/Var decomposition** reveals whether errors are systematic (correctable) or random (fundamental).");
    println!("- Compare across populations to test generalizability of each codec.");
}
