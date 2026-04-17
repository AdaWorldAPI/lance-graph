//! spiral_reconstruction_probe — P1 from docs/CODEC_INVARIANTS_AND_EXPERIMENTS.md
//!
//! `SpiralEncoding` is a SIGNATURE codec (17 Base17 dims × K anchor samples),
//! not a dense row reconstructor — it encodes a row as its spiral trajectory.
//! This probe measures whether the signature preserves the NEIGHBORHOOD
//! STRUCTURE of real Qwen3-TTS weight rows.
//!
//! Gates measured:
//!   G1: signature self-cosine ≈ 1.0 (identity check — should always pass)
//!   G2: nearest-neighbour preservation — for each row's raw-cosine nearest,
//!       does signature-cosine also rank it #1 (or top-5)?
//!   G3: rank correlation (Spearman-like) between raw-cos and signature-cos
//!       on a sample of pairs
//!
//! If G2/G3 pass → SpiralEncoding is a viable cascade-inference palette
//! substrate (Path B from #184, but with signature instead of Base17 index).
//! If they fail → spiral trajectory doesn't preserve inner-product structure
//! on Qwen3 weights, and we need a different signature primitive.
//!
//! Usage:
//!   cargo run --release --example spiral_reconstruction_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use highheelbgz::rehydrate::SpiralEncoding;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, GgufFile};
use ndarray::simd::bf16_to_f32_batch;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const TARGET_TENSOR: &str = "talker.model.layers.0.self_attn.k_proj.weight";
const N_SAMPLE_ROWS: usize = 256;   // how many rows to probe
const K_VALUES: &[usize] = &[4, 8, 16];  // anchor counts to sweep
const TOP_K: usize = 5;              // top-K agreement threshold
const SPIRAL_START: u32 = 0;
const SPIRAL_STRIDE: u32 = 3;        // k_proj role uses stride=3 per NeuronPrint design

fn load_tensor(path: &str, name_substr: &str) -> (Vec<f32>, [usize; 2]) {
    let file = File::open(path).expect("open model");
    let mut reader = BufReader::new(file);
    let header: GgufFile = read_safetensors_header(&mut reader).expect("parse header");

    let t = header.tensors.iter()
        .find(|t| t.name.contains(name_substr))
        .expect(&format!("tensor '{}' not found", name_substr));

    let n: usize = t.dimensions.iter().map(|&d| d as usize).product();
    let elem_size = match t.dtype { GgmlType::BF16 | GgmlType::F16 => 2, GgmlType::F32 => 4, _ => 2 };
    reader.seek(SeekFrom::Start(header.tensor_data_offset + t.offset)).unwrap();
    let mut raw = vec![0u8; n * elem_size];
    reader.read_exact(&mut raw).unwrap();

    let f32_data: Vec<f32> = match t.dtype {
        GgmlType::BF16 => {
            let u16s: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let mut out = vec![0.0f32; u16s.len()];
            bf16_to_f32_batch(&u16s, &mut out);
            out
        }
        GgmlType::F32 => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        _ => raw.chunks_exact(2)
            .map(|c| ndarray::hpc::gguf::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    };
    let shape = [t.dimensions[0] as usize, t.dimensions.iter().skip(1).map(|&d| d as usize).product()];
    println!("  Loaded '{}' shape={:?}", t.name, shape);
    (f32_data, shape)
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64; let y = b[i] as f64;
        dot += x * y; na += x * x; nb += y * y;
    }
    let d = (na * nb).sqrt();
    if d < 1e-15 { 0.0 } else { dot / d }
}

/// Measure nearest-neighbor preservation for one K value.
fn probe_k(rows: &[Vec<f32>], k: usize) {
    let n = rows.len();
    let t0 = Instant::now();

    // Encode all rows at this K.
    let encodings: Vec<SpiralEncoding> = rows.iter()
        .map(|r| SpiralEncoding::encode(r, SPIRAL_START, SPIRAL_STRIDE, k))
        .collect();
    let encode_ms = t0.elapsed().as_secs_f32() * 1000.0;

    // G1: self-cosine identity check
    let mut self_cos_min = 1.0f64;
    for enc in &encodings {
        let c = enc.cosine(enc);
        if c < self_cos_min { self_cos_min = c; }
    }

    // G2: NN preservation. For each row, compare:
    //   raw_nn = argmax_j raw_cos(row[i], row[j])  (j != i)
    //   sig_nn = argmax_j sig_cos(enc[i], enc[j])  (j != i)
    //
    // This is O(n^2) in pairwise scoring. Cap n at 256 to keep wall time bounded.
    let t1 = Instant::now();
    let mut top1_match = 0usize;
    let mut topk_match = 0usize;

    for i in 0..n {
        // Raw pairwise scores excluding self.
        let mut raw_scores: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, cosine_f32(&rows[i], &rows[j])))
            .collect();
        raw_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let raw_top1 = raw_scores[0].0;

        // Signature pairwise scores excluding self.
        let mut sig_scores: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, encodings[i].cosine(&encodings[j])))
            .collect();
        sig_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let sig_top1 = sig_scores[0].0;
        let sig_topk: std::collections::HashSet<usize> = sig_scores.iter().take(TOP_K).map(|(j, _)| *j).collect();

        if raw_top1 == sig_top1 { top1_match += 1; }
        if sig_topk.contains(&raw_top1) { topk_match += 1; }
    }
    let probe_ms = t1.elapsed().as_secs_f32() * 1000.0;

    // G3: rank correlation on a sample of pairs.
    // For each row i, rank all j by raw_cos and by sig_cos, measure Spearman-ish
    // via the fraction of inversions.
    let sample = n.min(64);
    let mut rank_agreement_sum = 0.0f64;
    for i in 0..sample {
        let raw_ranks: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, cosine_f32(&rows[i], &rows[j])))
            .collect();
        let sig_ranks: Vec<(usize, f64)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, encodings[i].cosine(&encodings[j])))
            .collect();

        // Spearman-esque: how often do the TWO rankings agree on a random pair?
        let mut agree = 0usize; let mut total = 0usize;
        for a in 0..raw_ranks.len() {
            for b in (a + 1)..raw_ranks.len() {
                let raw_order = raw_ranks[a].1 > raw_ranks[b].1;
                // Find sig_rank entries matching a and b
                let sa_val = sig_ranks.iter().find(|(j, _)| *j == raw_ranks[a].0).unwrap().1;
                let sb_val = sig_ranks.iter().find(|(j, _)| *j == raw_ranks[b].0).unwrap().1;
                let sig_order = sa_val > sb_val;
                if raw_order == sig_order { agree += 1; }
                total += 1;
            }
        }
        rank_agreement_sum += agree as f64 / total.max(1) as f64;
    }
    let rank_agreement = rank_agreement_sum / sample as f64;

    println!("  K={:<2}  encode={:>6.1}ms  probe={:>6.1}ms  self_cos_min={:.6}  top-1={:>3}/{}={:.2}%  top-{}={:>3}/{}={:.2}%  pairwise-rank-agree={:.4}",
        k, encode_ms, probe_ms,
        self_cos_min,
        top1_match, n, 100.0 * top1_match as f64 / n as f64,
        TOP_K, topk_match, n, 100.0 * topk_match as f64 / n as f64,
        rank_agreement);
}

fn main() {
    let model_path = std::env::args().nth(1)
        .expect("usage: spiral_reconstruction_probe <model.safetensors>");

    println!("═══ spiral_reconstruction_probe — SpiralEncoding on real Qwen3 ═══");
    println!("  Model:        {}", model_path);
    println!("  Tensor:       {}", TARGET_TENSOR);
    println!("  Sample rows:  {}", N_SAMPLE_ROWS);
    println!("  Spiral stride: {} (k_proj role)", SPIRAL_STRIDE);
    println!();

    let (flat, shape) = load_tensor(&model_path, TARGET_TENSOR);
    let (n_rows, n_cols) = (shape[0], shape[1]);
    let take = N_SAMPLE_ROWS.min(n_rows);

    // Stride-sample across the row set to avoid local-clustering artifacts.
    let stride = n_rows.max(1) / take;
    let rows: Vec<Vec<f32>> = (0..take)
        .map(|i| {
            let row_idx = (i * stride).min(n_rows - 1);
            flat[row_idx * n_cols..(row_idx + 1) * n_cols].to_vec()
        })
        .collect();

    println!("  Probing {} rows at K ∈ {:?}", rows.len(), K_VALUES);
    println!();

    for &k in K_VALUES {
        probe_k(&rows, k);
    }

    println!();
    println!("═══ DECISION GATES ═══");
    println!("  G1 self_cos_min ≈ 1.0  — identity (always expected)");
    println!("  G2 top-1 NN match ≥ 90% → SpiralEncoding signature preserves neighborhood");
    println!("  G3 pairwise-rank-agree ≥ 0.85 → rank correlation strong enough for cascade");
    println!();
    println!("  If G2+G3 pass: Path B from #184 is viable with SpiralEncoding as the palette substrate.");
    println!("  If either fails: spiral trajectory insufficient for Qwen3 weight inner-product structure.");
    println!("═══ DONE ═══");
}
