//! Budget rotation test: staggered offsets per dim vs raw stacking.
//!
//! Instead of SPD=4 keeping the first 4 octave samples:
//!   dim[0] = octaves 0, 1, 2, 3  (adjacent, see same local structure)
//!
//! Use staggered offsets with stride 8:
//!   dim[0] = octaves 20, 28, 36, 44  (each sees different "paragraph")
//!   dim[1] = octaves 21, 29, 37, 45
//!   ...
//!
//! Same byte budget (136 bytes = 4 samples × 17 dims × 2 bytes BF16).
//! Question: does structured spacing preserve more than raw stacking?
//!
//! cargo run --release --manifest-path crates/bgz-tensor/Cargo.toml --example budget_rotation_test

use std::f64::consts::GOLDEN_RATIO;

const BASE_DIM: usize = 17;

fn main() {
    println!("=== Budget Rotation vs Raw Stacking ===\n");

    let mut raw: Vec<Vec<f32>> = Vec::new();
    for path in &["/tmp/jina_batch1.json", "/tmp/jina_batch2.json"] {
        let texts: Vec<&str> = (0..20).map(|_| "x").collect();
        if let Ok(json) = std::fs::read_to_string(path) {
            if let Ok(embs) = bgz_tensor::jina::parse_jina_response(&json, &texts) {
                for e in embs { raw.push(e.vector); }
            }
        }
    }
    if raw.is_empty() { eprintln!("No data"); return; }
    let n = raw.len();
    let dim = raw[0].len();
    println!("{} vectors, dim={}\n", n, dim);

    let mut gt: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n { for j in (i+1)..n {
        gt.push((i, j, cosine_f32(&raw[i], &raw[j])));
    }}
    let gt_cos: Vec<f64> = gt.iter().map(|p| p.2).collect();

    // ═══ Compare at 4 samples per dim (136 bytes) ════════════════════════

    println!("--- 4 samples/dim (136 bytes, same as StackedBF16×4) ---\n");

    let configs: Vec<(&str, usize, usize, bool)> = vec![
        // (name, start_offset, stride_between_samples, use_phi_fractional)
        ("SPD=4 raw stacking (current)",        0,  1, false),
        ("offset=20 stride=8 (your proposal)",  20, 8, true),
        ("offset=20 stride=4",                  20, 4, true),
        ("offset=20 stride=12",                 20, 12, true),
        ("offset=10 stride=8",                  10, 8, true),
        ("offset=30 stride=8",                  30, 8, true),
        ("offset=20 stride=8 integer",          20, 8, false),
        ("offset=0 stride=8 phi",               0,  8, true),
        ("offset=50 stride=5",                  50, 5, true),
    ];

    println!("┌─────────────────────────────────────────┬──────────┬──────────┐");
    println!("│ Strategy (4 samples/dim = 136 bytes)    │  Pearson │ Spearman │");
    println!("├─────────────────────────────────────────┼──────────┼──────────┤");

    for (name, start, stride, use_phi) in &configs {
        let projected: Vec<[[f32; 4]; BASE_DIM]> = raw.iter()
            .map(|v| budget_project(v, 4, *start, *stride, *use_phi))
            .collect();
        let cos: Vec<f64> = gt.iter()
            .map(|&(i,j,_)| cosine_budget(&projected[i], &projected[j]))
            .collect();
        let p = bgz_tensor::quality::pearson(&gt_cos, &cos);
        let s = bgz_tensor::quality::spearman(&gt_cos, &cos);
        println!("│ {:<39} │ {:>8.4} │ {:>8.4} │", name, p, s);
    }
    println!("└─────────────────────────────────────────┴──────────┴──────────┘");

    // ═══ Compare at 8 samples per dim (272 bytes) ════════════════════════

    println!("\n--- 8 samples/dim (272 bytes) ---\n");

    let configs8: Vec<(&str, usize, usize, bool)> = vec![
        ("SPD=8 raw stacking",                  0,  1, false),
        ("offset=20 stride=4 (8 samples)",      20, 4, true),
        ("offset=20 stride=8 (8 samples)",      20, 8, true),
        ("offset=20 stride=3",                  20, 3, true),
    ];

    println!("┌─────────────────────────────────────────┬──────────┬──────────┐");
    println!("│ Strategy (8 samples/dim = 272 bytes)    │  Pearson │ Spearman │");
    println!("├─────────────────────────────────────────┼──────────┼──────────┤");

    for (name, start, stride, use_phi) in &configs8 {
        let projected: Vec<[[f32; 8]; BASE_DIM]> = raw.iter()
            .map(|v| budget_project_n(v, 8, *start, *stride, *use_phi))
            .collect();
        let cos: Vec<f64> = gt.iter()
            .map(|&(i,j,_)| cosine_budget_n(&projected[i], &projected[j]))
            .collect();
        let p = bgz_tensor::quality::pearson(&gt_cos, &cos);
        let s = bgz_tensor::quality::spearman(&gt_cos, &cos);
        println!("│ {:<39} │ {:>8.4} │ {:>8.4} │", name, p, s);
    }
    println!("└─────────────────────────────────────────┴──────────┴──────────┘");

    // ═══ Budget split: 1/3 + 1/4 + 1/8 + remainder ══════════════════════

    println!("\n--- Budget split: different vector regions get different rotations ---\n");

    let budget_projected: Vec<[[f32; 4]; BASE_DIM]> = raw.iter()
        .map(|v| budget_split_project(v))
        .collect();
    let budget_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_budget(&budget_projected[i], &budget_projected[j]))
        .collect();
    let bp = bgz_tensor::quality::pearson(&gt_cos, &budget_cos);
    let bs = bgz_tensor::quality::spearman(&gt_cos, &budget_cos);
    println!("Budget split (1/3 + 1/4 + 1/8 + rest): Pearson={:.4}, Spearman={:.4}", bp, bs);

    // ═══ Reference: full SPD=32 stacked ═══

    println!("\n--- Reference ---\n");
    let stacked32: Vec<bgz_tensor::StackedN> = raw.iter()
        .map(|v| bgz_tensor::StackedN::from_f32(v, 32)).collect();
    let s32_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| stacked32[i].cosine(&stacked32[j])).collect();
    println!("SPD=32 stacked (1088 bytes):  Pearson={:.4}, Spearman={:.4}",
        bgz_tensor::quality::pearson(&gt_cos, &s32_cos),
        bgz_tensor::quality::spearman(&gt_cos, &s32_cos));

    println!("\n=== DONE ===");
}

/// Project with staggered offsets: dim d gets samples at octaves
/// (start + d), (start + d + stride), (start + d + 2*stride), ...
fn budget_project(weights: &[f32], samples: usize, start: usize, stride: usize, use_phi: bool) -> [[f32; 4]; BASE_DIM] {
    let n = weights.len();
    let n_oct = (n + BASE_DIM - 1) / BASE_DIM;
    let mut result = [[0.0f32; 4]; BASE_DIM];

    for bi in 0..BASE_DIM {
        for s in 0..samples.min(4) {
            let octave = start + bi + s * stride;
            if octave >= n_oct { continue; }

            let pos = if use_phi {
                let phi_pos = frac((bi + octave) as f64 * GOLDEN_RATIO) * BASE_DIM as f64;
                phi_pos as usize
            } else {
                (bi * 11) % BASE_DIM
            };

            let dim = octave * BASE_DIM + pos;
            if dim < n {
                result[bi][s] = weights[dim];
            }
        }
    }
    result
}

/// Same for N samples (generic).
fn budget_project_n(weights: &[f32], samples: usize, start: usize, stride: usize, use_phi: bool) -> [[f32; 8]; BASE_DIM] {
    let n = weights.len();
    let n_oct = (n + BASE_DIM - 1) / BASE_DIM;
    let mut result = [[0.0f32; 8]; BASE_DIM];

    for bi in 0..BASE_DIM {
        for s in 0..samples.min(8) {
            let octave = start + bi + s * stride;
            if octave >= n_oct { continue; }
            let pos = if use_phi {
                (frac((bi + octave) as f64 * GOLDEN_RATIO) * BASE_DIM as f64) as usize
            } else {
                (bi * 11) % BASE_DIM
            };
            let dim = octave * BASE_DIM + pos;
            if dim < n { result[bi][s] = weights[dim]; }
        }
    }
    result
}

/// Budget split: 1/3 + 1/4 + 1/8 + rest, each with different offset.
fn budget_split_project(weights: &[f32]) -> [[f32; 4]; BASE_DIM] {
    let n = weights.len();
    let cuts = [0, n/3, n/3 + n/4, n/3 + n/4 + n/8, n];
    let offsets = [20, 37, 53, 71];
    let mut result = [[0.0f32; 4]; BASE_DIM];

    for (budget, (&off, window)) in offsets.iter().zip(cuts.windows(2)).enumerate() {
        let start_dim = window[0];
        let end_dim = window[1];
        let slice = &weights[start_dim..end_dim];
        let slice_octaves = (slice.len() + BASE_DIM - 1) / BASE_DIM;

        // Each budget fills one sample slot across all 17 dims
        if budget >= 4 { break; }
        for bi in 0..BASE_DIM {
            let octave = off % slice_octaves.max(1);
            let pos = (frac((bi + off) as f64 * GOLDEN_RATIO) * BASE_DIM as f64) as usize;
            let dim = octave * BASE_DIM + pos;
            if dim < slice.len() {
                result[bi][budget] = slice[dim];
            }
        }
    }
    result
}

fn cosine_budget(a: &[[f32; 4]; BASE_DIM], b: &[[f32; 4]; BASE_DIM]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for d in 0..BASE_DIM {
        for s in 0..4 {
            let x = a[d][s] as f64; let y = b[d][s] as f64;
            dot += x * y; na += x * x; nb += y * y;
        }
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

fn cosine_budget_n(a: &[[f32; 8]; BASE_DIM], b: &[[f32; 8]; BASE_DIM]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for d in 0..BASE_DIM {
        for s in 0..8 {
            let x = a[d][s] as f64; let y = b[d][s] as f64;
            dot += x * y; na += x * x; nb += y * y;
        }
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[inline]
fn frac(x: f64) -> f64 { x - x.floor() }

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}
