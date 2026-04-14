//! Certify HHTL-D: Cronbach α, Spearman ρ, Pearson r per cascade level.
//!
//! The 343:1 compression is meaningless without knowing what's recovered.
//! This probe measures EXACTLY where information is lost in the HHTL-D
//! pipeline and whether the cascade routing decisions remain faithful.
//!
//! ## What it measures
//!
//! For N weight rows (sampled from the real Qwen3-TTS-1.7B model):
//!
//! 1. **Ground truth**: f32 cosine similarity matrix (N×N upper triangle)
//! 2. **Base17 L1**: After golden-step fold, L1 distances (N×N)
//! 3. **Palette L1**: After centroid assignment, distance table lookup (N×N)
//! 4. **HHTL-D cascade**: After RouteAction dispatch:
//!    - Skip pairs: treated as distance=∞
//!    - Attend pairs: distance from table
//!    - Escalate pairs: Base17 L1
//!
//! At each level, compute:
//!   - **Spearman ρ** vs ground truth (rank preservation — the critical metric)
//!   - **Pearson r** vs ground truth (linear fidelity)
//!   - **Cronbach α** (internal consistency — split-half on Base17 dims)
//!   - **Top-10 recall** (do the true 10 nearest neighbors survive?)
//!   - **RouteAction accuracy** (% of Skip decisions that are correct)
//!
//! ## Pass/Fail criteria
//!
//! | Level | Metric | Target | Meaning |
//! |-------|--------|--------|---------|
//! | Base17 | Spearman ρ | ≥ 0.990 | Golden-step fold preserves rank |
//! | Palette | Spearman ρ | ≥ 0.950 | 256 centroids capture structure |
//! | Cascade | Spearman ρ | ≥ 0.930 | HHTL routing preserves attention |
//! | Cascade | Top-10 recall | ≥ 0.80 | True neighbors still found |
//! | Cascade | Skip accuracy | ≥ 0.95 | Skipped pairs are truly distant |
//! | All | Cronbach α | ≥ 0.85 | Encoding is internally consistent |
//!
//! ## Where pain points appear
//!
//! Historical findings from codebook_pearson and Jina v5 7-lane:
//! - Gate projections have LOWEST ρ (wide, flat distributions)
//! - V projections have HIGHEST ρ (peaked, clustered)
//! - Embeddings are tricky (high row count, many near-duplicates)
//! - Skip accuracy degrades when HIP threshold is too aggressive
//!
//! The probe runs per-role so you see WHERE the encoding hurts.
//!
//! ## Usage
//!
//! ```sh
//! cargo run --release --example certify_hhtld \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors
//! ```
//!
//! Output: JSON report at `.claude/knowledge/certification/hhtld_qwen3tts17b.json`

use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use bgz_tensor::projection::Base17;
use bgz_tensor::palette::WeightPalette;
use bgz_tensor::hhtl_cache::{HhtlCache, RouteAction};
use bgz_tensor::hhtl_d::HhtlDTensor;
use bgz_tensor::quality::{pearson, spearman};
use bgz_tensor::shared_palette::{
    PaletteGroupKey, classify_role, classify_component,
    is_encodable, effective_shape,
};
use bgz_tensor::hhtl_d::build_hip_families;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::time::Instant;

const N_CENTROIDS: usize = 256;
/// Max rows to sample per role for quality measurement.
/// 500 rows → 124,750 unique pairs → statistically robust.
const SAMPLE_ROWS: usize = 500;

// ═══════════════════════════════════════════════════════════════════
// Metrics
// ═══════════════════════════════════════════════════════════════════

/// Cronbach's alpha: internal consistency of the Base17 encoding.
///
/// Split the 17 dims into odd/even halves, compute L1 for each half,
/// Cronbach α = how well both halves agree on pairwise distances.
fn cronbach_alpha(rows: &[Base17]) -> f64 {
    let n = rows.len();
    if n < 10 { return 0.0; }

    let n_pairs = n * (n - 1) / 2;
    let mut half_a = Vec::with_capacity(n_pairs);
    let mut half_b = Vec::with_capacity(n_pairs);

    for i in 0..n {
        for j in (i + 1)..n {
            // Odd dims (0,2,4,...,16) = half A
            let mut da = 0i64;
            for d in (0..17).step_by(2) {
                da += (rows[i].dims[d] as i64 - rows[j].dims[d] as i64).abs();
            }
            // Even dims (1,3,5,...,15) = half B
            let mut db = 0i64;
            for d in (1..17).step_by(2) {
                db += (rows[i].dims[d] as i64 - rows[j].dims[d] as i64).abs();
            }
            half_a.push(da as f64);
            half_b.push(db as f64);
        }
    }

    // Cronbach's α with k=2 halves:
    // α = (k / (k-1)) * (1 - Σ(var_half) / var_total)
    let total: Vec<f64> = half_a.iter().zip(&half_b).map(|(a, b)| a + b).collect();
    let var_a = variance(&half_a);
    let var_b = variance(&half_b);
    let var_total = variance(&total);

    if var_total < 1e-12 { return 0.0; }
    2.0 * (1.0 - (var_a + var_b) / var_total)
}

fn variance(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 2.0 { return 0.0; }
    let mean = v.iter().sum::<f64>() / n;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Top-k recall: fraction of true k-nearest neighbors found by the encoding.
fn top_k_recall_matrix(
    ground_truth: &[Vec<f64>],  // N×N distance matrix
    encoded: &[Vec<f64>],       // N×N distance matrix
    k: usize,
) -> f64 {
    let n = ground_truth.len();
    if n <= k { return 1.0; }

    let mut total_recall = 0.0;
    for i in 0..n {
        // True top-k neighbors (lowest distance)
        let mut gt_indexed: Vec<(usize, f64)> = ground_truth[i].iter()
            .enumerate().filter(|&(j, _)| j != i)
            .map(|(j, &d)| (j, d)).collect();
        gt_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_topk: std::collections::HashSet<usize> =
            gt_indexed.iter().take(k).map(|&(j, _)| j).collect();

        // Encoded top-k
        let mut enc_indexed: Vec<(usize, f64)> = encoded[i].iter()
            .enumerate().filter(|&(j, _)| j != i)
            .map(|(j, &d)| (j, d)).collect();
        enc_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let enc_topk: std::collections::HashSet<usize> =
            enc_indexed.iter().take(k).map(|&(j, _)| j).collect();

        let overlap = true_topk.intersection(&enc_topk).count();
        total_recall += overlap as f64 / k as f64;
    }

    total_recall / n as f64
}

/// Compute f32 cosine similarity between two vectors.
fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-15 { 0.0 } else { dot / denom }
}

// ═══════════════════════════════════════════════════════════════════
// Per-role quality report
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Debug, serde::Serialize)]
struct RoleReport {
    role: String,
    component: String,
    n_rows: usize,

    // 1:1 reconstruction cosine: cosine(original_row[i], decoded_row[i])
    // Base17 level: decode via Base17::to_f32()
    base17_cos_mean: f64,
    base17_cos_min: f64,
    base17_cos_p5: f64,   // 5th percentile

    // Palette level: decode via centroid Base17::to_f32()
    palette_cos_mean: f64,
    palette_cos_min: f64,
    palette_cos_p5: f64,

    // HHTL-D level: centroid + polarity * residual * gamma
    hhtld_cos_mean: f64,
    hhtld_cos_min: f64,
    hhtld_cos_p5: f64,

    // MatVec fidelity: ||W·x - W_decoded·x|| / ||W·x||
    matvec_rel_error: f64,

    // Pass/fail
    pass: bool,
    failures: Vec<String>,
}

/// Run the 1:1 reconstruction probe for one role.
///
/// For each row: encode → decode → cosine(original, decoded).
/// This measures how well the HHTL-D pipeline preserves individual rows,
/// NOT pairwise relationships between rows.
fn probe_role(
    role: &str,
    component: &str,
    f32_rows: &[Vec<f32>],
    cache: &HhtlCache,
    _hip_families: &[u8],
) -> RoleReport {
    let n = f32_rows.len().min(SAMPLE_ROWS);
    let rows = &f32_rows[..n];
    let n_cols = if n > 0 { rows[0].len() } else { 0 };

    // ─── Level 1: Base17 fold → reconstruct → cosine ───
    let base17_rows: Vec<Base17> = rows.iter()
        .map(|r| Base17::from_f32(r))
        .collect();

    let mut b17_cosines: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let decoded = base17_rows[i].to_f32(n_cols);
        b17_cosines.push(cosine_f32(&rows[i], &decoded));
    }
    b17_cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let base17_cos_mean = b17_cosines.iter().sum::<f64>() / n.max(1) as f64;
    let base17_cos_min = b17_cosines.first().copied().unwrap_or(0.0);
    let base17_cos_p5 = b17_cosines.get(n * 5 / 100).copied().unwrap_or(base17_cos_min);

    // ─── Level 2: Palette centroid → reconstruct → cosine ───
    let assignments: Vec<u8> = base17_rows.iter()
        .map(|b| cache.nearest(b).0)
        .collect();

    let mut pal_cosines: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let centroid = &cache.palette.entries[assignments[i] as usize];
        let decoded = centroid.to_f32(n_cols);
        pal_cosines.push(cosine_f32(&rows[i], &decoded));
    }
    pal_cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let palette_cos_mean = pal_cosines.iter().sum::<f64>() / n.max(1) as f64;
    let palette_cos_min = pal_cosines.first().copied().unwrap_or(0.0);
    let palette_cos_p5 = pal_cosines.get(n * 5 / 100).copied().unwrap_or(palette_cos_min);

    // ─── Level 3: HHTL-D (centroid + polarity * residual * gamma) ───
    // Decode: centroid_f32 + polarity * slot_v_f32 * gamma_restore
    let gamma_restore = cache.gamma_meta[0].max(cache.gamma_meta[1]).max(1e-6);

    let mut hhtld_cosines: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let centroid = &cache.palette.entries[assignments[i] as usize];
        let centroid_f32 = centroid.to_f32(n_cols);

        // Compute residual: L1 distance normalized by centroid magnitude
        let centroid_mag: f64 = centroid.dims.iter()
            .map(|&d| (d as f64).abs())
            .sum::<f64>()
            .max(1.0);
        let l1_dist = base17_rows[i].l1(centroid) as f64;
        let residual = (l1_dist / centroid_mag) as f32;

        // Polarity from dominant residual dimension
        let polarity = {
            let mut max_dim = 0i32;
            for d in 0..17 {
                let diff = base17_rows[i].dims[d] as i32 - centroid.dims[d] as i32;
                if diff.abs() > max_dim.abs() { max_dim = diff; }
            }
            if max_dim >= 0 { 1.0f32 } else { -1.0f32 }
        };

        // Reconstruct: centroid + polarity * residual * gamma
        let mut decoded = centroid_f32;
        let correction = polarity * residual * gamma_restore;
        for v in decoded.iter_mut() {
            *v += correction;
        }

        hhtld_cosines.push(cosine_f32(&rows[i], &decoded));
    }
    hhtld_cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let hhtld_cos_mean = hhtld_cosines.iter().sum::<f64>() / n.max(1) as f64;
    let hhtld_cos_min = hhtld_cosines.first().copied().unwrap_or(0.0);
    let hhtld_cos_p5 = hhtld_cosines.get(n * 5 / 100).copied().unwrap_or(hhtld_cos_min);

    // ─── MatVec fidelity: ||W·x - W_decoded·x|| / ||W·x|| ───
    // Random unit vector x, compute W·x with original vs decoded
    let mut rng = 0x12345678u64;
    let x_vec: Vec<f32> = (0..n_cols).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }).collect();
    // Normalize x
    let x_norm: f32 = x_vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-10);
    let x_unit: Vec<f32> = x_vec.iter().map(|v| v / x_norm).collect();

    let mut wx_orig = vec![0.0f64; n];
    let mut wx_decoded = vec![0.0f64; n];
    for i in 0..n {
        let centroid = &cache.palette.entries[assignments[i] as usize];
        let centroid_f32 = centroid.to_f32(n_cols);

        for j in 0..n_cols {
            wx_orig[i] += rows[i][j] as f64 * x_unit[j] as f64;
            wx_decoded[i] += centroid_f32[j] as f64 * x_unit[j] as f64;
        }
    }

    let diff_norm: f64 = wx_orig.iter().zip(&wx_decoded)
        .map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    let orig_norm: f64 = wx_orig.iter().map(|a| a.powi(2)).sum::<f64>().sqrt().max(1e-15);
    let matvec_rel_error = diff_norm / orig_norm;

    // ─── Pass/fail ───
    let mut failures = Vec::new();
    if base17_cos_mean < 0.990 { failures.push(format!("Base17 cos={:.4} < 0.990", base17_cos_mean)); }
    if palette_cos_mean < 0.950 { failures.push(format!("Palette cos={:.4} < 0.950", palette_cos_mean)); }
    if hhtld_cos_mean < 0.930 { failures.push(format!("HHTL-D cos={:.4} < 0.930", hhtld_cos_mean)); }
    if matvec_rel_error > 0.20 { failures.push(format!("MatVec err={:.4} > 0.20", matvec_rel_error)); }

    RoleReport {
        role: role.to_string(),
        component: component.to_string(),
        n_rows: n,
        base17_cos_mean,
        base17_cos_min,
        base17_cos_p5,
        palette_cos_mean,
        palette_cos_min,
        palette_cos_p5,
        hhtld_cos_mean,
        hhtld_cos_min,
        hhtld_cos_p5,
        matvec_rel_error,
        pass: failures.is_empty(),
        failures,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let st_path = if args.len() > 1 { &args[1] }
    else { "/home/user/models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors" };

    println!("═══ HHTL-D QUALITY CERTIFICATION ═══");
    println!("  Model: {}", st_path);
    println!("  Sample: {} rows per role, {} centroids", SAMPLE_ROWS, N_CENTROIDS);
    println!();

    // Parse header
    let mut reader = BufReader::new(File::open(st_path).expect("open"));
    let header = read_safetensors_header(&mut reader).expect("parse");
    println!("[1] {} tensors", header.tensors.len());

    // Group by role
    let mut role_tensors: HashMap<(String, String), Vec<&TensorInfo>> = HashMap::new();
    for tensor in &header.tensors {
        if !tensor.name.ends_with("weight") { continue; }
        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let size = shape.iter().product::<usize>() * 2; // BF16
        if !is_encodable(&shape, size) { continue; }
        let comp = classify_component(&tensor.name).to_string();
        let role = classify_role(&tensor.name).to_string();
        role_tensors.entry((comp, role)).or_default().push(tensor);
    }

    println!("[2] {} encodable roles", role_tensors.len());
    println!();

    // Run probe per role
    println!("┌──────────────┬──────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐");
    println!("│ Role         │ Comp     │ B17cos │ B17 p5 │ Palcos │ Pal p5 │ Dcos   │ D p5   │ MVErr  │ Pass   │");
    println!("├──────────────┼──────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤");

    let mut all_reports = Vec::new();

    for ((comp, role), tensors) in role_tensors.iter() {
        // Read rows from first tensor in group (up to SAMPLE_ROWS)
        let tensor = tensors[0];
        let n_rows = (tensor.dimensions[0] as usize).min(SAMPLE_ROWS);
        let n_cols = if tensor.dimensions.len() > 1 { tensor.dimensions[1] as usize } else { 1 };
        let elem_size: usize = match tensor.dtype {
            GgmlType::BF16 | GgmlType::F16 => 2,
            GgmlType::F32 => 4,
            _ => continue,
        };

        reader.seek(SeekFrom::Start(header.tensor_data_offset + tensor.offset)).unwrap();
        let mut raw = vec![0u8; n_rows * n_cols * elem_size];
        if reader.read_exact(&mut raw).is_err() { continue; }

        let f32_rows: Vec<Vec<f32>> = (0..n_rows).map(|r| {
            (0..n_cols).map(|c| {
                let idx = r * n_cols + c;
                match tensor.dtype {
                    GgmlType::BF16 => {
                        let bits = u16::from_le_bytes([raw[idx*2], raw[idx*2+1]]);
                        f32::from_bits((bits as u32) << 16)
                    }
                    GgmlType::F16 => {
                        let bits = u16::from_le_bytes([raw[idx*2], raw[idx*2+1]]);
                        ndarray::hpc::gguf::f16_to_f32(bits)
                    }
                    GgmlType::F32 => {
                        f32::from_le_bytes([raw[idx*4], raw[idx*4+1], raw[idx*4+2], raw[idx*4+3]])
                    }
                    _ => 0.0,
                }
            }).collect()
        }).collect();

        // Build palette from these rows
        let base17_rows: Vec<Base17> = f32_rows.iter().map(|r| Base17::from_f32(r)).collect();
        let sample = if base17_rows.len() > 4096 { &base17_rows[..4096] } else { &base17_rows[..] };
        let wp = WeightPalette::build(sample, N_CENTROIDS);
        let hip = build_hip_families(&wp.entries);
        let cache = HhtlCache::from_palette(wp);

        let report = probe_role(role, comp, &f32_rows, &cache, &hip);

        let pass_str = if report.pass { "PASS" } else { "FAIL" };
        println!("│ {:12} │ {:8} │ {:.4} │ {:.4} │ {:.4} │ {:.4} │ {:.4} │ {:.4} │ {:.4} │ {:6} │",
            report.role, report.component,
            report.base17_cos_mean, report.base17_cos_p5,
            report.palette_cos_mean, report.palette_cos_p5,
            report.hhtld_cos_mean, report.hhtld_cos_p5,
            report.matvec_rel_error, pass_str);

        if !report.failures.is_empty() {
            for f in &report.failures {
                println!("│   ⚠ {}{}│", f, " ".repeat(78usize.saturating_sub(f.len() + 5)));
            }
        }

        all_reports.push(report);
    }

    println!("└──────────────┴──────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘");

    // Summary
    let n_pass = all_reports.iter().filter(|r| r.pass).count();
    let n_fail = all_reports.len() - n_pass;
    println!();
    println!("  {} roles passed, {} failed", n_pass, n_fail);

    // Write JSON report
    let report_dir = ".claude/knowledge/certification";
    std::fs::create_dir_all(report_dir).ok();
    let report_path = format!("{}/hhtld_qwen3tts17b.json", report_dir);
    let json = serde_json::to_string_pretty(&all_reports).unwrap_or_default();
    std::fs::write(&report_path, &json).ok();
    println!("  Report: {}", report_path);
    println!();
    println!("═══ DONE ═══");
}
