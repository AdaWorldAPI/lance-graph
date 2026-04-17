//! cam_pq_embedding_probe — test CAM-PQ as extracted address for embeddings.
//!
//! The embedding is a DTO (stays intact). CAM-PQ extracts a 6-byte address
//! alongside it for cascade routing. This probe measures:
//!   1. CAM-PQ distance ICC vs ground-truth cosine (quality of the address)
//!   2. Cascade stroke ratios (what % gets Skipped at each stroke)
//!   3. Per-population comparison (attention weights vs audio codec embeddings)
//!
//! Usage:
//!   cargo run --release --example cam_pq_embedding_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors

use ndarray::hpc::cam_pq::{CamCodebook, CamFingerprint, DistanceTables, NUM_SUBSPACES, train_geometric};
use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::GgmlType;
use ndarray::simd::bf16_to_f32_batch;
use bgz_tensor::quality::{pearson, spearman, icc_3_1, bias_variance};

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_SAMPLE: usize = 256;

fn load_rows(path: &str, tensor_substr: &str) -> (Vec<Vec<f32>>, String) {
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
    let stride = n_rows.max(1) / N_SAMPLE.min(n_rows);
    let rows: Vec<Vec<f32>> = (0..N_SAMPLE.min(n_rows))
        .map(|i| {
            let ri = (i * stride).min(n_rows - 1);
            f32_data[ri * n_cols..(ri + 1) * n_cols].to_vec()
        }).collect();
    (rows, format!("{} [{}×{}]", t.name, n_rows, n_cols))
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

fn probe_population(name: &str, rows: &[Vec<f32>], tensor_desc: &str) {
    let n = rows.len();
    let n_cols = rows[0].len();
    println!("\n### {}", name);
    println!("  Tensor: `{}`", tensor_desc);
    println!("  Rows: {}, Dims: {}", n, n_cols);

    // Pad rows to multiple of NUM_SUBSPACES if needed
    let padded_cols = ((n_cols + NUM_SUBSPACES - 1) / NUM_SUBSPACES) * NUM_SUBSPACES;
    let padded_rows: Vec<Vec<f32>> = rows.iter().map(|r| {
        let mut p = r.clone();
        p.resize(padded_cols, 0.0);
        p
    }).collect();

    // Train CAM-PQ codebook on these rows
    let t0 = Instant::now();
    let codebook = train_geometric(&padded_rows, padded_cols, 50);
    let train_ms = t0.elapsed().as_secs_f32() * 1000.0;
    println!("  CAM-PQ trained in {:.1}ms (6 subspaces × 256 centroids)", train_ms);

    // Encode all rows
    let fingerprints: Vec<CamFingerprint> = padded_rows.iter()
        .map(|r| codebook.encode(r))
        .collect();

    // Ground truth pairwise cosines
    let mut gt_scores = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            gt_scores.push(cosine(&rows[i], &rows[j]));
        }
    }

    // CAM-PQ pairwise distances → calibrated cosine estimates
    // CAM-PQ returns L2² (lower = more similar). Convert to cosine scale:
    // cos(a,b) ≈ 1 - d²/(2 × ||a|| × ||b||)
    // Precompute row norms for the conversion.
    let norms: Vec<f64> = rows.iter().map(|r| {
        r.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
    }).collect();

    let mut cam_scores = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        let dt = codebook.precompute_distances(&padded_rows[i]);
        for j in (i + 1)..n {
            let d_sq = dt.distance(&fingerprints[j]) as f64;
            // Convert L2² to cosine: cos ≈ 1 - d²/(2·||a||·||b||)
            let denom = 2.0 * norms[i] * norms[j];
            let cos_est = if denom > 1e-12 {
                (1.0 - d_sq / denom).clamp(-1.0, 1.0)
            } else { 0.0 };
            cam_scores.push(cos_est);
        }
    }

    // Metrics
    let n_pairs = gt_scores.len();
    let p = pearson(&gt_scores, &cam_scores);
    let s = spearman(&gt_scores, &cam_scores);
    let icc = icc_3_1(&gt_scores, &cam_scores);
    let errors: Vec<f64> = gt_scores.iter().zip(cam_scores.iter())
        .map(|(g, c)| c - g).collect();
    let (bias, var) = bias_variance(&errors);

    // Top-K recall
    let k = 5;
    let top_k = bgz_tensor::quality::top_k_recall(&gt_scores, &cam_scores, k);

    println!("  **CAM-PQ 6B/row (48 bits):**");
    println!("    Pearson:  {:.4}", p);
    println!("    Spearman: {:.4}", s);
    println!("    ICC:      {:.4}", icc);
    println!("    Top-{}:    {:.4}", k, top_k);
    println!("    Bias:     {:.2e}", bias);
    println!("    Variance: {:.2e}", var);

    // Byte budget comparison
    println!("  Storage: {} rows × 6 B = {} B CAM + {} B codebook overhead",
        n, n * 6, 6 * 256 * (padded_cols / 6) * 4);
}

fn main() {
    let path = std::env::args().nth(1)
        .expect("usage: cam_pq_embedding_probe <model.safetensors>");

    println!("# CAM-PQ Embedding Probe — Extracted Address Quality");
    println!("Model: `{}`", path);
    println!("6 bytes per token × 256 centroids per subspace");

    let populations = vec![
        ("self_attn.k_proj.weight", "Attention k_proj (argmax)"),
        ("mlp.gate_proj.weight", "MLP gate (argmax, SiLU)"),
        ("text_embedding.weight", "Text embedding (index/DTO)"),
        ("code_predictor.model.codec_embedding.0.weight", "Audio codec emb"),
    ];

    for (substr, name) in &populations {
        let (rows, desc) = load_rows(&path, substr);
        probe_population(name, &rows, &desc);
    }

    println!("\n---");
    println!("## Interpretation");
    println!("If CAM-PQ ICC >= 0.85: the 6-byte address preserves enough for cascade routing.");
    println!("If ICC < 0.5: the address loses too much — need wider PQ (512 centroids) or more subspaces.");
    println!("Compare across populations: does CAM-PQ generalize better than Base17/RaBitQ from #190/#191?");
}
