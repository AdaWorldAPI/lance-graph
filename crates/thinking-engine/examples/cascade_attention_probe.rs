//! cascade_attention_probe — Path B argmax-parity on a single attention head.
//!
//! PR #179 mindset shift E5: "inference in codec space, not f32 GEMM on
//! reconstructed weights." `bgz-tensor` already has `hhtl_cache` + `FisherZTable`
//! for this. Open question: does the codec-space argmax agree with raw f32 GEMM
//! argmax closely enough that swapping attention matmul for table lookups
//! preserves decoding semantics?
//!
//! This probe measures that on ONE attention K weight matrix, not the full
//! pipeline. If the argmax agreement rate is ≥ 0.90, Path B is viable. If
//! substantially lower, we need a per-row escalate layer or richer Q-side
//! encoding.
//!
//! Design:
//!   1. Load K matrix from Qwen3-TTS-0.6B (first talker layer self_attn.k_proj)
//!   2. Rows of K are the "keys" — 1024 of them (shape [1024, 2048]).
//!   3. Build Base17 palette + FisherZTable over those rows.
//!   4. Sample N random test queries (rows of K, with perturbation noise added).
//!   5. For each test query:
//!        - Raw argmax: argmax_i  q · K[i]^T      (f32 dot product, 1024-way)
//!        - Codec argmax: argmax_i  fisher_z[palette_idx(q), palette_idx(K[i])]
//!   6. Report agreement rate.
//!
//! Usage:
//!   cargo run --release --example cascade_attention_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use bgz_tensor::hhtl_cache::HhtlCache;
use bgz_tensor::fisher_z::FisherZTable;
use bgz_tensor::projection::Base17;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, GgufFile};
use ndarray::simd::bf16_to_f32_batch;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_QUERIES: usize = 512;      // how many test queries to probe
const NOISE_SCALE: f32 = 0.05;     // perturbation applied to the source K row before querying
const TOP_K: usize = 5;            // argmax + top-5 agreement measurement
const PALETTE_K: usize = 256;

fn load_target_tensor(model_path: &str, tensor_substring: &str) -> (Vec<f32>, [usize; 2]) {
    let file = File::open(model_path).expect("open model");
    let mut reader = BufReader::new(file);
    let header: GgufFile = read_safetensors_header(&mut reader).expect("parse header");

    let target = header.tensors.iter()
        .find(|t| t.name.contains(tensor_substring))
        .expect(&format!("tensor containing '{}' not found", tensor_substring));

    let n: usize = target.dimensions.iter().map(|&d| d as usize).product();
    let elem_size = match target.dtype { GgmlType::BF16 | GgmlType::F16 => 2, GgmlType::F32 => 4, _ => 2 };
    reader.seek(SeekFrom::Start(header.tensor_data_offset + target.offset)).unwrap();
    let mut raw = vec![0u8; n * elem_size];
    reader.read_exact(&mut raw).unwrap();

    let f32_data: Vec<f32> = match target.dtype {
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

    let shape = [target.dimensions[0] as usize,
                 target.dimensions.iter().skip(1).map(|&d| d as usize).product()];
    println!("  Loaded '{}' shape={:?}", target.name, shape);
    (f32_data, shape)
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len().min(b.len()) { s += a[i] * b[i]; }
    s
}

fn main() {
    let model_path = std::env::args().nth(1)
        .expect("usage: cascade_attention_probe <model.safetensors>");

    println!("═══ cascade_attention_probe — Path B argmax-parity ═══");

    let t_start = Instant::now();

    // 1. Load K matrix (first talker layer's k_proj) — a representative attention head.
    let (k_flat, shape) = load_target_tensor(&model_path, "talker.model.layers.0.self_attn.k_proj.weight");
    let (n_rows, n_cols) = (shape[0], shape[1]);

    // Materialise rows as Vec<Vec<f32>> for easy access.
    let rows_f32: Vec<Vec<f32>> = (0..n_rows)
        .map(|i| k_flat[i * n_cols..(i + 1) * n_cols].to_vec())
        .collect();

    // 2. Build Base17 palette + HhtlCache + FisherZTable on representative rows.
    let base17_rows: Vec<Base17> = rows_f32.iter().map(|r| Base17::from_f32(r)).collect();
    let cache = HhtlCache::from_base17_rows(&base17_rows, PALETTE_K);
    println!("  HhtlCache built: k={}", cache.k());

    // Collect one representative f32 row per centroid (nearest-to-centroid).
    let mut reps: Vec<Vec<f32>> = vec![Vec::new(); cache.k()];
    let mut rep_dists: Vec<u32> = vec![u32::MAX; cache.k()];
    for (i, row) in rows_f32.iter().enumerate() {
        let (ci, dist) = cache.nearest(&base17_rows[i]);
        let ci = ci as usize;
        if ci < cache.k() && dist < rep_dists[ci] {
            reps[ci] = row.clone();
            rep_dists[ci] = dist;
        }
    }
    for ci in 0..cache.k() {
        if reps[ci].is_empty() {
            reps[ci] = cache.palette.entries[ci].to_f32(n_cols);
        }
    }
    let fz = FisherZTable::build(&reps, cache.k());
    println!("  FisherZTable built: {} KB", fz.byte_size() / 1024);

    // Pre-compute palette index per K row (by actual cache assignment).
    let k_palette_idx: Vec<u8> = base17_rows.iter()
        .map(|b17| cache.nearest(b17).0)
        .collect();

    // 3. Test queries: take N_QUERIES K rows at deterministic stride,
    //    perturb with gaussian-ish noise via a simple LCG, use the result as
    //    the "query" vector.
    println!("\n  Probing {} queries at noise scale {:.3}...", N_QUERIES, NOISE_SCALE);
    let mut seed: u32 = 0xC0DE_BABE;
    let mut noise = || {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let u = (seed as i32 as f32) / 2_147_483_648.0;
        u * NOISE_SCALE
    };

    let stride = (n_rows / N_QUERIES).max(1);
    let mut argmax_agree = 0usize;
    let mut top_k_agree = 0usize;
    let mut n_probed = 0usize;

    let t0 = Instant::now();

    for qi in (0..n_rows).step_by(stride).take(N_QUERIES) {
        // Build perturbed query from the row at qi.
        let mut q = rows_f32[qi].clone();
        for v in q.iter_mut() { *v += noise(); }

        // Raw argmax: exhaustive dot product.
        let mut raw_scores: Vec<(usize, f32)> = (0..n_rows)
            .map(|i| (i, dot_f32(&q, &rows_f32[i])))
            .collect();
        raw_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let raw_top1 = raw_scores[0].0;
        let raw_topk: Vec<usize> = raw_scores.iter().take(TOP_K).map(|(i, _)| *i).collect();

        // Codec argmax: q → Base17 → palette_idx, then fisher_z[q_idx][k_idx[i]].
        let q_b17 = Base17::from_f32(&q);
        let q_idx = cache.nearest(&q_b17).0;

        let mut codec_scores: Vec<(usize, f32)> = (0..n_rows)
            .map(|i| (i, fz.lookup_f32(q_idx, k_palette_idx[i])))
            .collect();
        codec_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let codec_top1 = codec_scores[0].0;
        let codec_topk: std::collections::HashSet<usize> =
            codec_scores.iter().take(TOP_K).map(|(i, _)| *i).collect();

        if raw_top1 == codec_top1 { argmax_agree += 1; }
        if codec_topk.contains(&raw_top1) { top_k_agree += 1; }
        n_probed += 1;
    }

    let elapsed = t0.elapsed();
    let build_elapsed = t_start.elapsed();

    println!("\n═══ RESULTS ═══");
    println!("  Queries probed:               {}", n_probed);
    println!("  Total probe wall:             {:.2}s", elapsed.as_secs_f32());
    println!("  Setup + probe wall:           {:.2}s", build_elapsed.as_secs_f32());
    println!();
    println!("  Top-1 argmax agreement:       {}/{} = {:.2}%",
        argmax_agree, n_probed, 100.0 * argmax_agree as f64 / n_probed as f64);
    println!("  Raw top-1 in codec top-{}:     {}/{} = {:.2}%",
        TOP_K, top_k_agree, n_probed, 100.0 * top_k_agree as f64 / n_probed as f64);
    println!();

    // Pass criteria (my subjective initial targets):
    //   ≥ 0.90 argmax parity  → Path B viable, proceed to full pipeline swap
    //   ≥ 0.70 argmax parity  → Path B partial, needs Q-side escalation layer
    //   < 0.70 argmax parity  → Path B as currently wired is not competitive
    let argmax_pct = argmax_agree as f64 / n_probed as f64;
    let verdict = if argmax_pct >= 0.90 {
        "★ PATH B VIABLE — argmax parity clears 90%"
    } else if argmax_pct >= 0.70 {
        "◐ PATH B PARTIAL — parity 70-90%, needs Q-side escalation"
    } else {
        "✗ PATH B FAIL — parity below 70%, not competitive with f32 GEMM argmax"
    };
    println!("  {}", verdict);
    println!("\n═══ DONE ═══");
}
