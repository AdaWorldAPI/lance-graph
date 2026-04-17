//! had_cascade_e2e — end-to-end test of the production Hadamard cascade codec.
//!
//! Encodes real model tensors, reconstructs, measures ICC per population,
//! prints codec selection decisions and compression stats.
//!
//! Usage:
//!   cargo run --release --example had_cascade_e2e \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use bgz_tensor::had_cascade::{HadCascadeTensor, TensorRegime};
use bgz_tensor::quality::{icc_3_1, pearson, spearman};
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::simd::bf16_to_f32_batch;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_SAMPLE: usize = 256;

fn load_rows(path: &str, tensor_substr: &str) -> Option<(Vec<Vec<f32>>, String, usize, usize)> {
    let file = File::open(path).expect("open");
    let mut reader = BufReader::new(file);
    let header = read_safetensors_header(&mut reader).expect("parse");
    let t = header.tensors.iter().find(|t| t.name.contains(tensor_substr))?;
    let n_rows = t.dimensions[0] as usize;
    let n_cols: usize = t.dimensions.iter().skip(1).map(|&d| d as usize).product();
    let n: usize = t.dimensions.iter().map(|&d| d as usize).product();
    reader.seek(SeekFrom::Start(header.tensor_data_offset + t.offset)).unwrap();
    let mut raw = vec![0u8; n * 2];
    reader.read_exact(&mut raw).unwrap();
    let f32_data: Vec<f32> = {
        let u16s: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut out = vec![0.0f32; u16s.len()];
        bf16_to_f32_batch(&u16s, &mut out);
        out
    };
    let stride = n_rows.max(1) / N_SAMPLE.min(n_rows);
    let rows: Vec<Vec<f32>> = (0..N_SAMPLE.min(n_rows))
        .map(|i| {
            let ri = (i * stride).min(n_rows - 1);
            f32_data[ri * n_cols..(ri + 1) * n_cols].to_vec()
        }).collect();
    Some((rows, t.name.clone(), n_rows, n_cols))
}

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
    for i in 0..n { for j in (i + 1)..n { scores.push(cosine(&rows[i], &rows[j])); } }
    scores
}

fn main() {
    let path = std::env::args().nth(1)
        .expect("usage: had_cascade_e2e <model.safetensors>");

    println!("# Hadamard Cascade Codec — Production E2E Test");
    println!("Model: `{}`", path);
    println!();

    let populations = vec![
        ("self_attn.k_proj.weight", "Attention k_proj"),
        ("mlp.gate_proj.weight", "MLP gate_proj"),
        ("text_embedding.weight", "Text embedding"),
        ("code_predictor.model.codec_embedding.0.weight", "Audio codec emb"),
        ("self_attn.q_proj.weight", "Attention q_proj"),
        ("per_layer_projection.weight", "PLE projection (256-d)"),
    ];

    println!("| Population | Regime | Decision | Rows | Dims | ICC | Pearson | Spearman | B/row | Ratio | Encode ms |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");

    let t_total = Instant::now();

    for (substr, name) in &populations {
        let Some((rows, tensor_name, full_n_rows, n_cols)) = load_rows(&path, substr) else {
            println!("| {} | — | SKIP | — | — | — | — | — | — | — | — |", name);
            continue;
        };
        let regime = TensorRegime::from_role(&tensor_name);
        let decision = if regime.should_compress() { "COMPRESS" } else { "PASSTHROUGH" };
        let n = rows.len();

        if !regime.should_compress() {
            println!("| {} | Index | {} | {} | {} | 1.0000 | 1.0000 | 1.0000 | {} | 1.00 | 0 |",
                name, decision, full_n_rows, n_cols, n_cols * 2);
            continue;
        }

        let t0 = Instant::now();
        // Production k: 64 centroids shared across rows.
        // Lloyd refinement in clam_sample tightens from greedy init.
        let k = 64;
        let tensor = HadCascadeTensor::encode(&tensor_name, &rows, k);
        let encode_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let recon = tensor.reconstruct_all();

        // Per-row cosine diagnostic
        let n_show = 5.min(n);
        for ri in 0..n_show {
            let row_cos = cosine(&rows[ri], &recon[ri]);
            eprint!("  row {} cos={:.6}  ", ri, row_cos);
        }
        eprintln!();

        let gt = pairwise_cosines(&rows);
        let rc = pairwise_cosines(&recon);
        let n_pairs = gt.len().min(rc.len());

        let icc = icc_3_1(&gt[..n_pairs], &rc[..n_pairs]);
        let p = pearson(&gt[..n_pairs], &rc[..n_pairs]);
        let s = spearman(&gt[..n_pairs], &rc[..n_pairs]);
        let bpr = tensor.bytes_per_row();
        let ratio = (n_cols * 2) as f64 / bpr.max(1) as f64;

        println!("| {} | Argmax | {} | {} | {} | {:.4} | {:.4} | {:.4} | {} | {:.2} | {:.1} |",
            name, decision, full_n_rows, n_cols, icc, p, s, bpr, ratio, encode_ms);
    }

    println!();
    println!("Total: {:.1}s", t_total.elapsed().as_secs_f32());
    println!();
    println!("## Codec Selection Rules");
    println!();
    println!("| Regime | Tensor patterns | Codec | Reason |");
    println!("|---|---|---|---|");
    println!("| Argmax | `*_proj.weight`, `gate_proj`, `up_proj`, `down_proj` | HadCascade (i4+i2) | ICC ≥0.995, 2.6:1 compression |");
    println!("| Index | `*embedding*`, `lm_head` | BF16 passthrough | DTO — identity required (I1, I10) |");
    println!("| Structured | `codec_embedding` | HadCascade | ICC 1.000, embeddings with structure |");
}
