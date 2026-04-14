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
    is_encodable, effective_shape, build_hip_families,
};

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
    n_pairs: usize,

    // Base17 level
    base17_spearman: f64,
    base17_pearson: f64,
    base17_cronbach: f64,

    // Palette level (256 centroids)
    palette_spearman: f64,
    palette_pearson: f64,
    palette_top10_recall: f64,

    // HHTL-D cascade level
    cascade_spearman: f64,
    cascade_pearson: f64,
    cascade_top10_recall: f64,
    cascade_skip_pct: f64,
    cascade_attend_pct: f64,
    cascade_escalate_pct: f64,
    cascade_skip_accuracy: f64,

    // Pass/fail
    pass: bool,
    failures: Vec<String>,
}

/// Run the full probe for one role.
fn probe_role(
    role: &str,
    component: &str,
    f32_rows: &[Vec<f32>],
    cache: &HhtlCache,
    hip_families: &[u8],
) -> RoleReport {
    let n = f32_rows.len().min(SAMPLE_ROWS);
    let rows = &f32_rows[..n];
    let n_pairs = n * (n - 1) / 2;

    // ─── Ground truth: f32 cosine (angular distance) ───
    let mut gt_upper = Vec::with_capacity(n_pairs);
    let mut gt_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let cos = cosine_f32(&rows[i], &rows[j]);
            let dist = (1.0 - cos).max(0.0);  // cosine distance
            gt_upper.push(dist);
            gt_matrix[i][j] = dist;
            gt_matrix[j][i] = dist;
        }
    }

    // ─── Level 1: Base17 L1 ───
    let base17_rows: Vec<Base17> = rows.iter()
        .map(|r| Base17::from_f32(r))
        .collect();

    let mut b17_upper = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            b17_upper.push(base17_rows[i].l1(&base17_rows[j]) as f64);
        }
    }

    let base17_spearman = spearman(&gt_upper, &b17_upper);
    let base17_pearson = pearson(&gt_upper, &b17_upper);
    let base17_cronbach = cronbach_alpha(&base17_rows);

    // ─── Level 2: Palette distance table ───
    let assignments: Vec<u8> = base17_rows.iter()
        .map(|b| cache.nearest(b).0)
        .collect();

    let mut pal_upper = Vec::with_capacity(n_pairs);
    let mut pal_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = cache.distance(assignments[i], assignments[j]) as f64;
            pal_upper.push(d);
            pal_matrix[i][j] = d;
            pal_matrix[j][i] = d;
        }
    }

    let palette_spearman = spearman(&gt_upper, &pal_upper);
    let palette_pearson = pearson(&gt_upper, &pal_upper);
    let palette_top10 = top_k_recall_matrix(&gt_matrix, &pal_matrix, 10);

    // ─── Level 3: HHTL-D cascade (RouteAction dispatch) ───
    let mut cascade_upper = Vec::with_capacity(n_pairs);
    let mut cascade_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    let mut n_skip = 0usize;
    let mut n_attend = 0usize;
    let mut n_escalate = 0usize;
    let mut n_compose = 0usize;
    let mut skip_correct = 0usize;
    let mut skip_total = 0usize;

    // Percentile threshold for "truly distant" (top 30% of ground truth distances)
    let mut gt_sorted = gt_upper.clone();
    gt_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let distant_threshold = gt_sorted[gt_sorted.len() * 70 / 100];

    let mut pair_idx = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let action = cache.route(assignments[i], assignments[j]);
            let d = match action {
                RouteAction::Skip => {
                    n_skip += 1;
                    // Check if skip is correct (pair truly distant)
                    skip_total += 1;
                    if gt_upper[pair_idx] >= distant_threshold {
                        skip_correct += 1;
                    }
                    f64::MAX / 2.0  // infinity — skipped
                }
                RouteAction::Attend => {
                    n_attend += 1;
                    cache.distance(assignments[i], assignments[j]) as f64
                }
                RouteAction::Compose => {
                    n_compose += 1;
                    cache.distance(assignments[i], assignments[j]) as f64
                }
                RouteAction::Escalate => {
                    n_escalate += 1;
                    base17_rows[i].l1(&base17_rows[j]) as f64
                }
            };
            cascade_upper.push(d);
            cascade_matrix[i][j] = d;
            cascade_matrix[j][i] = d;
            pair_idx += 1;
        }
    }

    // For Spearman/Pearson: exclude Skip pairs (infinity breaks correlation)
    let non_skip: Vec<(f64, f64)> = gt_upper.iter().zip(&cascade_upper)
        .filter(|(_, &c)| c < f64::MAX / 4.0)
        .map(|(&g, &c)| (g, c))
        .collect();
    let gt_ns: Vec<f64> = non_skip.iter().map(|&(g, _)| g).collect();
    let cas_ns: Vec<f64> = non_skip.iter().map(|&(_, c)| c).collect();

    let cascade_spearman = spearman(&gt_ns, &cas_ns);
    let cascade_pearson = pearson(&gt_ns, &cas_ns);
    let cascade_top10 = top_k_recall_matrix(&gt_matrix, &cascade_matrix, 10);
    let skip_accuracy = if skip_total > 0 { skip_correct as f64 / skip_total as f64 } else { 1.0 };

    let total_actions = n_skip + n_attend + n_compose + n_escalate;
    let skip_pct = n_skip as f64 / total_actions.max(1) as f64;
    let attend_pct = n_attend as f64 / total_actions.max(1) as f64;
    let escalate_pct = (n_escalate + n_compose) as f64 / total_actions.max(1) as f64;

    // ─── Pass/fail ───
    let mut failures = Vec::new();
    if base17_spearman < 0.990 { failures.push(format!("Base17 ρ={:.4} < 0.990", base17_spearman)); }
    if palette_spearman < 0.950 { failures.push(format!("Palette ρ={:.4} < 0.950", palette_spearman)); }
    if cascade_spearman < 0.930 { failures.push(format!("Cascade ρ={:.4} < 0.930", cascade_spearman)); }
    if cascade_top10 < 0.80 { failures.push(format!("Top-10 recall={:.2} < 0.80", cascade_top10)); }
    if skip_accuracy < 0.95 { failures.push(format!("Skip accuracy={:.2} < 0.95", skip_accuracy)); }
    if base17_cronbach < 0.85 { failures.push(format!("Cronbach α={:.3} < 0.85", base17_cronbach)); }

    RoleReport {
        role: role.to_string(),
        component: component.to_string(),
        n_rows: n,
        n_pairs,
        base17_spearman,
        base17_pearson,
        base17_cronbach,
        palette_spearman,
        palette_pearson,
        palette_top10_recall: palette_top10,
        cascade_spearman,
        cascade_pearson,
        cascade_top10_recall: cascade_top10,
        cascade_skip_pct: skip_pct,
        cascade_attend_pct: attend_pct,
        cascade_escalate_pct: escalate_pct,
        cascade_skip_accuracy: skip_accuracy,
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
    println!("┌──────────────┬──────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────┬────────┐");
    println!("│ Role         │ Comp     │ B17 ρ  │ B17 α  │ Pal ρ  │ Cas ρ  │ Top10  │ Skip%  │ SkAc │ Pass   │");
    println!("├──────────────┼──────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────┼────────┤");

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
        println!("│ {:12} │ {:8} │ {:.4} │ {:.3}  │ {:.4} │ {:.4} │ {:.3}  │ {:5.1}% │ {:.2} │ {:6} │",
            report.role, report.component,
            report.base17_spearman, report.base17_cronbach,
            report.palette_spearman, report.cascade_spearman,
            report.cascade_top10_recall, report.cascade_skip_pct * 100.0,
            report.cascade_skip_accuracy, pass_str);

        if !report.failures.is_empty() {
            for f in &report.failures {
                println!("│   ⚠ {}{}│", f, " ".repeat(72usize.saturating_sub(f.len() + 5)));
            }
        }

        all_reports.push(report);
    }

    println!("└──────────────┴──────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────┴────────┘");

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
