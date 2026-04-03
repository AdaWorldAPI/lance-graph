//! Test: golden-step projection with i32 precision and variable offset.
//!
//! Compare:
//!   A) Current: i16, stride 11 mod 17, start at 0 (the mush path)
//!   B) i32, stride 11 mod 17, start at 0 (more precision, same grid)
//!   C) i32, actual φ fractional stepping, start at 0
//!   D) i32, actual φ fractional stepping, start at offset 20
//!   E) i32, actual φ fractional stepping, start at offset 50
//!
//! Measures: pairwise cosine Pearson vs f32 ground truth.
//! Uses real Jina API embeddings (1024-dim, cosine range [-0.1, 0.99]).
//!
//! cargo run --release --manifest-path crates/bgz-tensor/Cargo.toml --example golden_offset_test

use std::f64::consts::GOLDEN_RATIO;

const BASE_DIM: usize = 17;
const FP_SCALE_I16: f64 = 256.0;
const FP_SCALE_I32: f64 = 65536.0; // 16-bit fractional precision in i32

fn main() {
    println!("=== Golden-Step Projection: i32 + Offset Test ===\n");

    // Load Jina embeddings
    let mut raw: Vec<Vec<f32>> = Vec::new();
    for path in &["/tmp/jina_batch1.json", "/tmp/jina_batch2.json"] {
        let texts: Vec<&str> = (0..20).map(|_| "x").collect();
        if let Ok(json) = std::fs::read_to_string(path) {
            if let Ok(embs) = bgz_tensor::jina::parse_jina_response(&json, &texts) {
                for e in embs { raw.push(e.vector); }
            }
        }
    }
    if raw.is_empty() { eprintln!("No Jina data"); return; }
    let n = raw.len();
    let dim = raw[0].len();
    println!("{} vectors, dim={}\n", n, dim);

    // Ground truth: f32 pairwise cosines
    let mut gt: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n { for j in (i+1)..n {
        gt.push((i, j, cosine_f32(&raw[i], &raw[j])));
    }}
    let gt_cos: Vec<f64> = gt.iter().map(|p| p.2).collect();

    // ═══ Strategy A: i16, integer stride 11 mod 17, offset 0 (current) ═══
    let a_projected: Vec<[i16; BASE_DIM]> = raw.iter()
        .map(|v| project_i16_integer_stride(v, 11, 0))
        .collect();
    let a_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_i16(&a_projected[i], &a_projected[j]))
        .collect();

    // ═══ Strategy B: i32, integer stride 11 mod 17, offset 0 ═══
    let b_projected: Vec<[i32; BASE_DIM]> = raw.iter()
        .map(|v| project_i32_integer_stride(v, 11, 0))
        .collect();
    let b_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_i32(&b_projected[i], &b_projected[j]))
        .collect();

    // ═══ Strategy C: i32, actual φ fractional, offset 0 ═══
    let c_projected: Vec<[i32; BASE_DIM]> = raw.iter()
        .map(|v| project_i32_phi_fractional(v, 0))
        .collect();
    let c_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_i32(&c_projected[i], &c_projected[j]))
        .collect();

    // ═══ Strategy D: i32, actual φ fractional, offset 20 ═══
    let d_projected: Vec<[i32; BASE_DIM]> = raw.iter()
        .map(|v| project_i32_phi_fractional(v, 20))
        .collect();
    let d_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_i32(&d_projected[i], &d_projected[j]))
        .collect();

    // ═══ Strategy E: i32, actual φ fractional, offset 50 ═══
    let e_projected: Vec<[i32; BASE_DIM]> = raw.iter()
        .map(|v| project_i32_phi_fractional(v, 50))
        .collect();
    let e_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_i32(&e_projected[i], &e_projected[j]))
        .collect();

    // ═══ Strategy F: i32, φ fractional, offset 20, skip 3 per step ═══
    let f_projected: Vec<[i32; BASE_DIM]> = raw.iter()
        .map(|v| project_i32_phi_skip(v, 20, 3))
        .collect();
    let f_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_i32(&f_projected[i], &f_projected[j]))
        .collect();

    // ═══ Strategy G: i32, φ fractional, offset 20, skip 7 ═══
    let g_projected: Vec<[i32; BASE_DIM]> = raw.iter()
        .map(|v| project_i32_phi_skip(v, 20, 7))
        .collect();
    let g_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_i32(&g_projected[i], &g_projected[j]))
        .collect();

    // ═══ Strategy H: i32, φ fractional, offset 100 ═══
    let h_projected: Vec<[i32; BASE_DIM]> = raw.iter()
        .map(|v| project_i32_phi_fractional(v, 100))
        .collect();
    let h_cos: Vec<f64> = gt.iter()
        .map(|&(i,j,_)| cosine_i32(&h_projected[i], &h_projected[j]))
        .collect();

    // ═══ Results ═══
    println!("┌────┬──────────────────────────────────────────┬──────────┬──────────┐");
    println!("│    │ Strategy                                 │  Pearson │ Spearman │");
    println!("├────┼──────────────────────────────────────────┼──────────┼──────────┤");

    let strategies: Vec<(&str, &[f64])> = vec![
        ("A: i16, stride 11, offset 0 (CURRENT)",  &a_cos),
        ("B: i32, stride 11, offset 0",            &b_cos),
        ("C: i32, φ-frac, offset 0",               &c_cos),
        ("D: i32, φ-frac, offset 20",              &d_cos),
        ("E: i32, φ-frac, offset 50",              &e_cos),
        ("F: i32, φ-frac, offset 20, skip 3",      &f_cos),
        ("G: i32, φ-frac, offset 20, skip 7",      &g_cos),
        ("H: i32, φ-frac, offset 100",             &h_cos),
    ];

    for (i, (name, cos)) in strategies.iter().enumerate() {
        let p = bgz_tensor::quality::pearson(&gt_cos, cos);
        let s = bgz_tensor::quality::spearman(&gt_cos, cos);
        let label = (b'A' + i as u8) as char;
        println!("│ {}  │ {:<40} │ {:>8.4} │ {:>8.4} │", label, name, p, s);
    }
    println!("└────┴──────────────────────────────────────────┴──────────┴──────────┘");

    // ═══ Uniformity analysis: how evenly do positions distribute? ═══
    println!("\n=== Position Distribution (first 17 octave positions) ===\n");

    println!("Integer stride 11 mod 17:");
    let int_pos: Vec<usize> = (0..BASE_DIM).map(|i| (i * 11) % BASE_DIM).collect();
    println!("  {:?}", int_pos);
    let int_gaps: Vec<usize> = (1..BASE_DIM).map(|i| {
        let mut sorted = int_pos[..=i].to_vec(); sorted.sort(); sorted[i] - sorted[i-1]
    }).collect();
    println!("  Sorted gaps: {:?}\n", int_gaps);

    println!("φ-fractional, offset 0:");
    let phi_pos_0: Vec<f64> = (0..BASE_DIM).map(|i| frac(i as f64 * GOLDEN_RATIO) * BASE_DIM as f64).collect();
    println!("  {:?}", phi_pos_0.iter().map(|p| format!("{:.2}", p)).collect::<Vec<_>>());

    println!("φ-fractional, offset 20:");
    let phi_pos_20: Vec<f64> = (0..BASE_DIM).map(|i| frac((i + 20) as f64 * GOLDEN_RATIO) * BASE_DIM as f64).collect();
    println!("  {:?}", phi_pos_20.iter().map(|p| format!("{:.2}", p)).collect::<Vec<_>>());

    println!("φ-fractional, offset 50:");
    let phi_pos_50: Vec<f64> = (0..BASE_DIM).map(|i| frac((i + 50) as f64 * GOLDEN_RATIO) * BASE_DIM as f64).collect();
    println!("  {:?}", phi_pos_50.iter().map(|p| format!("{:.2}", p)).collect::<Vec<_>>());

    // ═══ Resolution comparison: distinct cosine values ═══
    println!("\n=== Distinct Cosine Values (resolution) ===\n");
    for (name, cos) in &strategies {
        let mut sorted: Vec<i64> = cos.iter().map(|c| (c * 1e8) as i64).collect();
        sorted.sort();
        sorted.dedup();
        println!("  {}: {} distinct values out of {} pairs", name, sorted.len(), cos.len());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Projection strategies
// ═══════════════════════════════════════════════════════════════════════════

/// A: Current i16 path — integer stride, FP_SCALE=256
fn project_i16_integer_stride(weights: &[f32], stride: usize, offset: usize) -> [i16; BASE_DIM] {
    let n = weights.len();
    let n_oct = (n + BASE_DIM - 1) / BASE_DIM;
    let mut sum = [0.0f64; BASE_DIM];
    let mut count = [0u32; BASE_DIM];
    for oct in 0..n_oct {
        for bi in 0..BASE_DIM {
            let pos = ((bi + offset) * stride) % BASE_DIM;
            let dim = oct * BASE_DIM + pos;
            if dim < n {
                sum[bi] += weights[dim] as f64;
                count[bi] += 1;
            }
        }
    }
    let mut dims = [0i16; BASE_DIM];
    for d in 0..BASE_DIM {
        if count[d] > 0 {
            dims[d] = (sum[d] / count[d] as f64 * FP_SCALE_I16).round().clamp(-32768.0, 32767.0) as i16;
        }
    }
    dims
}

/// B: i32 with integer stride — more precision, same grid
fn project_i32_integer_stride(weights: &[f32], stride: usize, offset: usize) -> [i32; BASE_DIM] {
    let n = weights.len();
    let n_oct = (n + BASE_DIM - 1) / BASE_DIM;
    let mut sum = [0.0f64; BASE_DIM];
    let mut count = [0u32; BASE_DIM];
    for oct in 0..n_oct {
        for bi in 0..BASE_DIM {
            let pos = ((bi + offset) * stride) % BASE_DIM;
            let dim = oct * BASE_DIM + pos;
            if dim < n {
                sum[bi] += weights[dim] as f64;
                count[bi] += 1;
            }
        }
    }
    let mut dims = [0i32; BASE_DIM];
    for d in 0..BASE_DIM {
        if count[d] > 0 {
            dims[d] = (sum[d] / count[d] as f64 * FP_SCALE_I32).round()
                .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        }
    }
    dims
}

/// C/D/E/H: i32 with actual φ fractional stepping + variable offset
fn project_i32_phi_fractional(weights: &[f32], offset: usize) -> [i32; BASE_DIM] {
    let n = weights.len();
    let n_oct = (n + BASE_DIM - 1) / BASE_DIM;
    let mut sum = [0.0f64; BASE_DIM];
    let mut count = [0u32; BASE_DIM];
    for oct in 0..n_oct {
        for bi in 0..BASE_DIM {
            // Actual φ-fractional position: frac((bi + offset) × φ) × 17
            let phi_pos = frac((bi + offset) as f64 * GOLDEN_RATIO) * BASE_DIM as f64;
            let dim_idx = phi_pos as usize; // floor
            let actual_dim = oct * BASE_DIM + dim_idx;
            if actual_dim < n {
                sum[bi] += weights[actual_dim] as f64;
                count[bi] += 1;
            }
        }
    }
    let mut dims = [0i32; BASE_DIM];
    for d in 0..BASE_DIM {
        if count[d] > 0 {
            dims[d] = (sum[d] / count[d] as f64 * FP_SCALE_I32).round()
                .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        }
    }
    dims
}

/// F/G: i32 with φ fractional + offset + skip (sparse sampling)
fn project_i32_phi_skip(weights: &[f32], offset: usize, skip: usize) -> [i32; BASE_DIM] {
    let n = weights.len();
    let n_oct = (n + BASE_DIM - 1) / BASE_DIM;
    let step = skip.max(1);
    let mut sum = [0.0f64; BASE_DIM];
    let mut count = [0u32; BASE_DIM];
    for oct in (0..n_oct).step_by(step) {
        for bi in 0..BASE_DIM {
            let phi_pos = frac((bi + offset) as f64 * GOLDEN_RATIO) * BASE_DIM as f64;
            let dim_idx = phi_pos as usize;
            let actual_dim = oct * BASE_DIM + dim_idx;
            if actual_dim < n {
                sum[bi] += weights[actual_dim] as f64;
                count[bi] += 1;
            }
        }
    }
    let mut dims = [0i32; BASE_DIM];
    for d in 0..BASE_DIM {
        if count[d] > 0 {
            dims[d] = (sum[d] / count[d] as f64 * FP_SCALE_I32).round()
                .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        }
    }
    dims
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Fractional part: frac(x) = x - floor(x), always in [0, 1)
#[inline]
fn frac(x: f64) -> f64 { x - x.floor() }

fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..n { dot += a[i] as f64 * b[i] as f64; na += (a[i] as f64).powi(2); nb += (b[i] as f64).powi(2); }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

fn cosine_i16(a: &[i16; BASE_DIM], b: &[i16; BASE_DIM]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..BASE_DIM { let x = a[i] as f64; let y = b[i] as f64; dot += x*y; na += x*x; nb += y*y; }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}

fn cosine_i32(a: &[i32; BASE_DIM], b: &[i32; BASE_DIM]) -> f64 {
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    for i in 0..BASE_DIM { let x = a[i] as f64; let y = b[i] as f64; dot += x*y; na += x*x; nb += y*y; }
    let d = (na * nb).sqrt(); if d < 1e-12 { 0.0 } else { dot / d }
}
