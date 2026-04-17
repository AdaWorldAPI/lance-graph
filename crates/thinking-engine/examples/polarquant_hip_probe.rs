//! PolarQuant HIP family probe (P7) — does gain-shape split improve basin clustering?
//!
//! Current HIP family assignment (`hhtl_d::build_hip_families`) partitions
//! 256 Base17 palette centroids into 16 families via farthest-pair binary
//! splits on Base17 L1 distance. This confounds direction and magnitude.
//!
//! Hypothesis: PolarQuant gain-shape split (unit-normalize rows, cluster
//! on directions only) gives families that better predict inner-product
//! neighborhoods — because attention scoring is cos-based (direction).
//!
//! Probe: load real Qwen3 k_proj, build palette, assign HIP families both
//! ways (Base17 L1 vs PolarQuant-normalized), measure NN-preservation per
//! family for each. Better families → higher within-family NN recall.
//!
//! Usage:
//!   cargo run --release --example polarquant_hip_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use bgz_tensor::hhtl_d::build_hip_families;
use bgz_tensor::hhtl_cache::HhtlCache;
use bgz_tensor::projection::Base17;
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::GgmlType;
use ndarray::simd::bf16_to_f32_batch;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const TARGET: &str = "talker.model.layers.0.self_attn.k_proj.weight";
const N_SAMPLE: usize = 256;
const PALETTE_K: usize = 256;

fn load_rows(path: &str) -> Vec<Vec<f32>> {
    let file = File::open(path).expect("open");
    let mut reader = BufReader::new(file);
    let header = read_safetensors_header(&mut reader).expect("parse");
    let t = header.tensors.iter().find(|t| t.name.contains(TARGET)).expect("tensor");
    let n: usize = t.dimensions.iter().map(|&d| d as usize).product();
    let n_rows = t.dimensions[0] as usize;
    let n_cols: usize = t.dimensions.iter().skip(1).map(|&d| d as usize).product();
    reader.seek(SeekFrom::Start(header.tensor_data_offset + t.offset)).unwrap();
    let mut raw = vec![0u8; n * 2];
    reader.read_exact(&mut raw).unwrap();
    let f32_data: Vec<f32> = match t.dtype {
        GgmlType::BF16 => {
            let u16s: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let mut out = vec![0.0f32; u16s.len()];
            bf16_to_f32_batch(&u16s, &mut out);
            out
        }
        _ => raw.chunks_exact(2).map(|c| {
            ndarray::hpc::gguf::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))
        }).collect(),
    };
    let stride = n_rows.max(1) / N_SAMPLE.min(n_rows);
    (0..N_SAMPLE.min(n_rows))
        .map(|i| {
            let ri = (i * stride).min(n_rows - 1);
            f32_data[ri * n_cols..(ri + 1) * n_cols].to_vec()
        })
        .collect()
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

fn unit_normalize(row: &[f32]) -> Vec<f32> {
    let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 { return row.to_vec(); }
    row.iter().map(|x| x / norm).collect()
}

/// Measure within-family NN recall: for each row, does its raw-cosine
/// nearest neighbor land in the SAME family?
fn within_family_nn_recall(rows: &[Vec<f32>], families: &[u8], n_families: usize) -> f64 {
    let n = rows.len();
    let mut same_family = 0usize;
    for i in 0..n {
        let mut best_j = 0usize;
        let mut best_cos = f64::NEG_INFINITY;
        for j in 0..n {
            if j == i { continue; }
            let c = cosine(&rows[i], &rows[j]);
            if c > best_cos { best_cos = c; best_j = j; }
        }
        if families[i] == families[best_j] { same_family += 1; }
    }
    same_family as f64 / n as f64
}

/// Build HIP families on PolarQuant-normalized palette.
fn build_hip_families_polarquant(palette: &[Base17], rows: &[Vec<f32>]) -> Vec<u8> {
    // Unit-normalize the rows, then re-project to Base17.
    let normalized: Vec<Vec<f32>> = rows.iter().map(|r| unit_normalize(r)).collect();
    let norm_b17: Vec<Base17> = normalized.iter().map(|r| Base17::from_f32(r)).collect();

    // Build a temporary palette from normalized projections.
    let norm_palette: Vec<Base17> = if norm_b17.len() >= PALETTE_K {
        // Use the same indices as the original palette (approximation: the
        // palette centroids are the same rows, just normalized).
        palette.iter().enumerate().map(|(i, _)| {
            if i < norm_b17.len() { norm_b17[i].clone() } else { Base17::zero() }
        }).collect()
    } else {
        norm_b17.clone()
    };

    build_hip_families(&norm_palette)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: polarquant_hip_probe <model.safetensors>");
    println!("═══ PolarQuant HIP Family Probe (P7) ═══");
    println!("  Model: {}", path);
    println!("  Target: {}", TARGET);

    let t0 = Instant::now();
    let rows = load_rows(&path);
    println!("  Loaded {} rows in {:.2}s", rows.len(), t0.elapsed().as_secs_f32());

    // Build Base17 palette + cache
    let base17_rows: Vec<Base17> = rows.iter().map(|r| Base17::from_f32(r)).collect();
    let cache = HhtlCache::from_base17_rows(&base17_rows, PALETTE_K);
    println!("  Palette: {} centroids", cache.k());

    // Method A: current HIP families (Base17 L1 distance)
    let hip_base17 = build_hip_families(&cache.palette.entries);

    // Method B: PolarQuant-normalized HIP families
    let hip_polar = build_hip_families_polarquant(&cache.palette.entries, &rows);

    // Assign each row to its nearest centroid → get family label per row
    let row_families_base17: Vec<u8> = rows.iter().enumerate().map(|(i, _)| {
        let (ci, _) = cache.nearest(&base17_rows[i]);
        hip_base17[ci as usize]
    }).collect();

    let row_families_polar: Vec<u8> = rows.iter().enumerate().map(|(i, _)| {
        let (ci, _) = cache.nearest(&base17_rows[i]);
        hip_polar[ci as usize]
    }).collect();

    // Measure within-family NN recall for both
    let recall_base17 = within_family_nn_recall(&rows, &row_families_base17, 16);
    let recall_polar = within_family_nn_recall(&rows, &row_families_polar, 16);

    // Family distribution analysis
    let mut dist_b17 = HashMap::new();
    let mut dist_pol = HashMap::new();
    for &f in &row_families_base17 { *dist_b17.entry(f).or_insert(0usize) += 1; }
    for &f in &row_families_polar { *dist_pol.entry(f).or_insert(0usize) += 1; }

    println!("\n═══ RESULTS ═══");
    println!("  Method                  │ Within-family NN recall │ Families used");
    println!("  ────────────────────────┼─────────────────────────┼──────────────");
    println!("  Base17 L1 (current)     │ {:>22.4}% │ {}/16",
        recall_base17 * 100.0, dist_b17.len());
    println!("  PolarQuant normalized   │ {:>22.4}% │ {}/16",
        recall_polar * 100.0, dist_pol.len());

    let improvement = recall_polar - recall_base17;
    println!("\n  Delta: {:+.4}%", improvement * 100.0);

    if improvement > 0.05 {
        println!("  ★ PolarQuant families are BETTER — adopt gain-shape split in build_hip_families");
    } else if improvement > 0.0 {
        println!("  ◐ PolarQuant marginal improvement — may not be worth the complexity");
    } else {
        println!("  ✗ PolarQuant does NOT improve — Base17 L1 families are sufficient");
    }

    println!("\n═══ DONE ═══");
}
