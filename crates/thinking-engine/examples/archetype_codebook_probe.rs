//! Hierarchical archetype codebook: weight rows as vocabulary entries.
//!
//! Instead of compressing row DATA (lossy), index the row's FUNCTION
//! in a multi-resolution codebook. Same principle as audio RVQ:
//!
//! ```text
//! Audio RVQ (Qwen-TTS codec):     Weight RVQ (this probe):
//!   Level 0: 2048 entries (pitch)    Level 0: 256 entries (HEEL basin)
//!   Level 1: 2048 entries (formant)  Level 1: 512 entries (HIP family)
//!   Level 2: 2048 entries (texture)  Level 2: 1024 entries (TWIG)
//!   ...15 levels                     Level 3: 4096 entries (LEAF)
//!                                    Level 4: 17408 entries (Base17 space)
//!                                    Level 5: 65536 entries (exhaustive)
//! ```
//!
//! Each level's codebook captures the residual from the previous level.
//! The accumulation IS the row. No separate reconstruction needed.
//!
//! Storage per row: 6 indices (one per level) = 6-12 bytes.
//! Reconstruction: sum codebook entries at each level.
//!
//! ```sh
//! cargo run --release --example archetype_codebook_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors
//! ```

use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use bgz_tensor::quality::{pearson, spearman};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

/// Codebook level sizes — the matryoshka nesting.
/// Each level captures what the previous level missed.
const LEVELS: &[(usize, &str)] = &[
    (256,   "L0:HEEL"),    // basin routing
    (512,   "L1:HIP"),     // family
    (1024,  "L2:TWIG"),    // centroid
    (4096,  "L3:LEAF"),    // fine grain
    (17408, "L4:Base17"),  // Base17 address space (17^3 ≈ 4913, or 17×1024)
    (65536, "L5:FULL"),    // exhaustive
];

/// Maximum rows to sample per role for codebook building.
const MAX_SAMPLE: usize = 4096;

// ═══════════════════════════════════════════════════════════════════
// Codebook: CLAM furthest-point sampling at each level
// ═══════════════════════════════════════════════════════════════════

/// One level of the hierarchical codebook.
struct CodebookLevel {
    /// Centroid vectors (the archetype rows).
    centroids: Vec<Vec<f32>>,
    /// Assignment: which centroid each input row maps to.
    assignments: Vec<usize>,
    /// Residuals: input - centroid for each row.
    residuals: Vec<Vec<f32>>,
    /// Mean residual L2 norm.
    mean_residual_norm: f64,
    /// Number of unique centroids actually used.
    n_active: usize,
}

/// CLAM furthest-point sampling on f32 rows.
fn clam_sample(rows: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    let n = rows.len();
    let k = k.min(n);
    if k == 0 { return Vec::new(); }

    let cols = rows[0].len();
    let mut centroids = Vec::with_capacity(k);
    let mut used = vec![false; n];

    // First centroid: the row with largest L2 norm (most distinctive)
    let first = (0..n).max_by(|&a, &b| {
        let na: f64 = rows[a].iter().map(|x| (*x as f64).powi(2)).sum();
        let nb: f64 = rows[b].iter().map(|x| (*x as f64).powi(2)).sum();
        na.partial_cmp(&nb).unwrap()
    }).unwrap_or(0);

    centroids.push(rows[first].clone());
    used[first] = true;

    // Track min distance from each row to nearest centroid
    let mut min_dist = vec![f64::MAX; n];
    for i in 0..n {
        min_dist[i] = l2_dist(&rows[i], &centroids[0]);
    }

    // Furthest-point: pick the row maximally distant from all centroids
    for _ in 1..k {
        let next = (0..n)
            .filter(|&i| !used[i])
            .max_by(|&a, &b| min_dist[a].partial_cmp(&min_dist[b]).unwrap())
            .unwrap_or(0);

        centroids.push(rows[next].clone());
        used[next] = true;

        // Update min distances
        for i in 0..n {
            if !used[i] {
                let d = l2_dist(&rows[i], &centroids.last().unwrap());
                if d < min_dist[i] {
                    min_dist[i] = d;
                }
            }
        }
    }

    centroids
}

/// Build one codebook level: assign rows → centroids, compute residuals.
fn build_level(rows: &[Vec<f32>], k: usize) -> CodebookLevel {
    let n = rows.len();
    let cols = if n > 0 { rows[0].len() } else { 0 };
    let centroids = clam_sample(rows, k);
    let k_actual = centroids.len();

    let mut assignments = Vec::with_capacity(n);
    let mut residuals = Vec::with_capacity(n);
    let mut centroid_usage = vec![0usize; k_actual];
    let mut total_residual_norm = 0.0f64;

    for row in rows {
        // Find nearest centroid
        let (best_idx, _) = (0..k_actual)
            .map(|c| (c, l2_dist(row, &centroids[c])))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, 0.0));

        assignments.push(best_idx);
        centroid_usage[best_idx] += 1;

        // Compute residual
        let residual: Vec<f32> = row.iter().zip(&centroids[best_idx])
            .map(|(&a, &b)| a - b)
            .collect();
        let res_norm: f64 = residual.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        total_residual_norm += res_norm;
        residuals.push(residual);
    }

    let n_active = centroid_usage.iter().filter(|&&c| c > 0).count();
    let mean_residual_norm = if n > 0 { total_residual_norm / n as f64 } else { 0.0 };

    CodebookLevel {
        centroids,
        assignments,
        residuals,
        mean_residual_norm,
        n_active,
    }
}

/// Build the full hierarchy: each level operates on the RESIDUALS of the previous.
fn build_hierarchy(rows: &[Vec<f32>]) -> Vec<(usize, String, CodebookLevel)> {
    let mut levels = Vec::new();
    let mut current_input = rows.to_vec();

    for &(k, name) in LEVELS {
        if k > current_input.len() * 2 {
            // More centroids than 2× rows — skip this level
            continue;
        }

        let level = build_level(&current_input, k);

        // Next level operates on residuals
        current_input = level.residuals.clone();

        levels.push((k, name.to_string(), level));
    }

    levels
}

/// Reconstruct a row by accumulating centroids from each level.
fn reconstruct_from_hierarchy(
    row_idx: usize,
    levels: &[(usize, String, CodebookLevel)],
) -> Vec<f32> {
    if levels.is_empty() { return Vec::new(); }

    let cols = levels[0].2.centroids[0].len();
    let mut reconstructed = vec![0.0f32; cols];

    for (_, _, level) in levels {
        let centroid_idx = level.assignments[row_idx];
        let centroid = &level.centroids[centroid_idx];
        for j in 0..cols {
            reconstructed[j] += centroid[j];
        }
    }

    reconstructed
}

/// Storage per row: one index per level.
fn storage_per_row(levels: &[(usize, String, CodebookLevel)]) -> usize {
    levels.iter().map(|&(k, _, _)| {
        if k <= 256 { 1 }       // u8
        else if k <= 65536 { 2 } // u16
        else { 3 }              // u24
    }).sum()
}

/// Total codebook storage (centroids across all levels).
fn total_codebook_bytes(levels: &[(usize, String, CodebookLevel)]) -> usize {
    levels.iter().map(|(_, _, level)| {
        let cols = if level.centroids.is_empty() { 0 }
            else { level.centroids[0].len() };
        level.centroids.len() * cols * 2 // BF16 per element
    }).sum()
}

// ═══════════════════════════════════════════════════════════════════
// Quality metrics
// ═══════════════════════════════════════════════════════════════════

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

fn l2_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Measure quality of progressive reconstruction.
/// Returns (row_cosine, pairwise_rho) for reconstruction using levels 0..=depth.
fn measure_progressive(
    original: &[Vec<f32>],
    levels: &[(usize, String, CodebookLevel)],
    max_depth: usize,
) -> (f64, f64) {
    let n = original.len();
    if n == 0 { return (0.0, 0.0); }

    // Reconstruct using levels 0..=max_depth
    let cols = original[0].len();
    let reconstructed: Vec<Vec<f32>> = (0..n).map(|i| {
        let mut row = vec![0.0f32; cols];
        for d in 0..=max_depth.min(levels.len() - 1) {
            let centroid_idx = levels[d].2.assignments[i];
            let centroid = &levels[d].2.centroids[centroid_idx];
            for j in 0..cols {
                row[j] += centroid[j];
            }
        }
        row
    }).collect();

    // Average row cosine
    let avg_cos: f64 = (0..n).map(|i| cosine_f32(&original[i], &reconstructed[i])).sum::<f64>() / n as f64;

    // Pairwise Spearman on 200 random pairs
    let n_pairs = 200.min(n * (n - 1) / 2);
    let mut gt = Vec::with_capacity(n_pairs);
    let mut rc = Vec::with_capacity(n_pairs);

    let mut seed = 0x9E3779B97F4A7C15u64;
    for _ in 0..n_pairs {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 31;
        let a = (z as usize) % n;
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 31;
        let b = (z as usize) % n;
        if a == b { continue; }
        gt.push(cosine_f32(&original[a], &original[b]));
        rc.push(cosine_f32(&reconstructed[a], &reconstructed[b]));
    }

    let rho = spearman(&gt, &rc);
    (avg_cos, rho)
}

// ═══════════════════════════════════════════════════════════════════
// Tensor reading
// ═══════════════════════════════════════════════════════════════════

fn read_rows(
    reader: &mut BufReader<File>,
    tensor: &TensorInfo,
    data_offset: u64,
    max_rows: usize,
) -> Vec<Vec<f32>> {
    let n_rows = (tensor.dimensions[0] as usize).min(max_rows);
    let n_cols = if tensor.dimensions.len() > 1 { tensor.dimensions[1] as usize } else { 1 };
    let elem_size = match tensor.dtype {
        GgmlType::BF16 | GgmlType::F16 => 2,
        GgmlType::F32 => 4,
        _ => return Vec::new(),
    };

    reader.seek(SeekFrom::Start(data_offset + tensor.offset)).unwrap();
    let mut raw = vec![0u8; n_rows * n_cols * elem_size];
    if reader.read_exact(&mut raw).is_err() { return Vec::new(); }

    (0..n_rows).map(|r| {
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
    }).collect()
}

fn classify_role(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.contains("q_proj") { "q_proj" }
    else if n.contains("k_proj") { "k_proj" }
    else if n.contains("v_proj") { "v_proj" }
    else if n.contains("o_proj") { "o_proj" }
    else if n.contains("gate_proj") { "gate_proj" }
    else if n.contains("up_proj") { "up_proj" }
    else if n.contains("down_proj") { "down_proj" }
    else if n.contains("embed") { "embed" }
    else if n.contains("lm_head") { "lm_head" }
    else { "other" }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let st_path = if args.len() > 1 { &args[1] }
    else { "/home/user/models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors" };

    println!("═══ HIERARCHICAL ARCHETYPE CODEBOOK PROBE ═══");
    println!("  Model: {}", st_path);
    println!("  Levels: {:?}", LEVELS.iter().map(|(k, n)| format!("{}({})", n, k)).collect::<Vec<_>>());
    println!();

    let mut reader = BufReader::new(File::open(st_path).expect("open"));
    let header = read_safetensors_header(&mut reader).expect("parse");
    println!("[1] {} tensors", header.tensors.len());

    // Group tensors by role
    let target_roles = ["q_proj", "k_proj", "v_proj", "gate_proj", "down_proj", "embed", "lm_head"];
    let mut role_tensors: HashMap<String, Vec<&TensorInfo>> = HashMap::new();
    for tensor in &header.tensors {
        if !tensor.name.ends_with("weight") { continue; }
        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        if shape.len() < 2 || shape[0] < 256 || shape[1] < 256 { continue; }
        let role = classify_role(&tensor.name);
        if target_roles.contains(&role) {
            role_tensors.entry(role.to_string()).or_default().push(tensor);
        }
    }

    println!("[2] {} target roles", role_tensors.len());
    println!();

    // Header
    println!("┌──────────┬───────┬──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Role     │ Rows  │ Level → (active/k, residual_norm, row_cos, pairwise_ρ, bytes/row)           │");
    println!("├──────────┼───────┼──────────────────────────────────────────────────────────────────────────────┤");

    for (role, tensors) in role_tensors.iter() {
        let tensor = tensors[0]; // Sample from first tensor in role group
        let rows = read_rows(&mut reader, tensor, header.tensor_data_offset, MAX_SAMPLE);
        if rows.is_empty() { continue; }
        let n = rows.len();
        let cols = rows[0].len();

        let original_bytes_per_row = cols * 2; // BF16

        let t0 = Instant::now();
        let hierarchy = build_hierarchy(&rows);
        let build_time = t0.elapsed();

        print!("│ {:8} │ {:5} │ ", role, n);

        // Progressive quality at each depth
        let mut level_info = Vec::new();
        for (depth, (k, name, level)) in hierarchy.iter().enumerate() {
            let (row_cos, pairwise_rho) = measure_progressive(&rows, &hierarchy, depth);
            let spr = storage_per_row(&hierarchy[..=depth]);

            level_info.push(format!(
                "{}:{}/{} res={:.3} cos={:.3} ρ={:.3} {}B",
                name, level.n_active, k,
                level.mean_residual_norm,
                row_cos, pairwise_rho, spr
            ));
        }

        println!("{}", level_info.join("  │  "));

        // Summary line
        let total_cb = total_codebook_bytes(&hierarchy);
        let full_spr = storage_per_row(&hierarchy);
        let final_depth = hierarchy.len() - 1;
        let (final_cos, final_rho) = measure_progressive(&rows, &hierarchy, final_depth);

        println!("│          │       │ Σ codebook={:.0}KB  indices={}B/row  orig={}B/row  ratio={:.0}:1  final cos={:.4}  ρ={:.4}  ({:?})",
            total_cb as f64 / 1024.0,
            full_spr,
            original_bytes_per_row,
            original_bytes_per_row as f64 / full_spr as f64,
            final_cos, final_rho,
            build_time);

        // Find the level where row_cos first exceeds 0.999 (the TTS survival threshold)
        for (depth, (k, name, _)) in hierarchy.iter().enumerate() {
            let (cos, rho) = measure_progressive(&rows, &hierarchy, depth);
            if cos >= 0.999 {
                let spr = storage_per_row(&hierarchy[..=depth]);
                println!("│          │       │ ★ cos≥0.999 at {} (depth {}, {}B/row, {:.0}:1)",
                    name, depth, spr, original_bytes_per_row as f64 / spr as f64);
                break;
            }
        }

        println!("│──────────│───────│──────────────────────────────────────────────────────────────────────────────│");
    }

    println!("└──────────┴───────┴──────────────────────────────────────────────────────────────────────────────┘");

    // The key question: at what codebook total size does reconstruction
    // reach cos=0.999 (the threshold for surviving 33 transformer layers)?
    println!();
    println!("═══ KEY QUESTION ═══");
    println!("  cos=0.999 per row → (0.999)^33 = 0.967 output fidelity");
    println!("  The level where cos≥0.999 determines the minimum codebook size.");
    println!("  If L3:LEAF (4096 centroids) reaches 0.999, the codebook is ~32MB");
    println!("  shared across all 28 layers. Indices: 4-6 bytes/row.");
    println!("  Total: 32MB codebook + 28×2048×6 = 32.3 MB (12:1 compression).");
    println!();
    println!("  Compare: GGUF Q4 = ~900 MB (4:1). RVQ codebook could beat GGUF");
    println!("  if the codebook entries are good enough archetypes.");
    println!();
    println!("═══ DONE ═══");
}
