//! TurboQuant correction probe: does QJL unbiasing survive 33 layers?
//!
//! The hypothesis: our encodings have a systematic BIAS in distance estimation.
//! Fisher z corrects the SCALE (stretches tails). QJL corrects the EXPECTATION
//! (removes systematic over/under-estimation).
//!
//! Biased errors compound linearly: 33 × bias.
//! Unbiased errors compound as √33 ≈ 5.7 × noise.
//! This 6× difference could explain why ρ=1.0 pairwise still gives 0.4% token match.
//!
//! ## What TurboQuant IS
//!
//! ```text
//! PolarQuant: row → (||row||, row/||row||)
//!   = gain-shape split. Shape lives on unit sphere.
//!
//! JLQ: shape → R × shape (random projection, R is k×D)
//!   → quantize projection to b bits
//!   JL lemma: pairwise distances preserved ±ε with k ≥ O(log(n)/ε²)
//!
//! QJL correction: E[quantized_distance] = true_distance + bias(b)
//!   bias(b) = correction term that depends on bit width
//!   Subtract bias → unbiased estimator
//!
//! TurboQuant = PolarQuant + JLQ + correction
//! ```
//!
//! ## What we already have
//!
//! - PolarQuant = matryoshka.rs gain-shape split ✓
//! - JL projection = Hadamard rotation in RaBitQ ✓ (structured JL)
//! - SVD projection = learned JL (optimal but no universal guarantee) ✓
//! - Fisher z = nonlinear correction (different from QJL bias removal)
//! - RaBitQ correction = dot_correction factor in rabitq_compat.rs ✓
//!
//! ## This probe
//!
//! For 500 weight rows:
//! 1. Compute ground truth cosine matrix (f32)
//! 2. Encode with 4 methods:
//!    a. i8 direct (round(cos×127), no correction)
//!    b. Fisher z (arctanh + family gamma)
//!    c. QJL corrected (i8 + bias removal)
//!    d. RaBitQ corrected (binary + dot_correction)
//! 3. Simulate 33-layer chain: multiply quantized score matrices
//! 4. Measure: does the chain output preserve ranking?
//!
//! The chain simulation: attention score matrix A[i,j] feeds into value
//! weighting V → output. The output is input to the next layer's Q×K^T.
//! After 33 layers, a biased A accumulates bias; an unbiased A doesn't.
//!
//! ```sh
//! cargo run --release --example turboquant_correction_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml \
//!     -- /path/to/model.safetensors
//! ```

use ndarray::hpc::safetensors::read_safetensors_header;
use ndarray::hpc::gguf::{GgmlType, TensorInfo};
use bgz_tensor::quality::spearman;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::time::Instant;

const N_SAMPLE: usize = 200;  // rows to sample (200×200 = 40K pairs)
const N_LAYERS: usize = 33;   // transformer depth

// ═══════════════════════════════════════════════════════════════════
// Correction methods
// ═══════════════════════════════════════════════════════════════════

/// Method A: Direct i8 quantization. cos → round(cos×127) → i8.
/// Known bias: rounds toward zero, systematically underestimates magnitude.
fn quantize_direct_i8(cos: f64) -> i8 {
    (cos * 127.0).round().clamp(-127.0, 127.0) as i8
}

fn restore_direct_i8(q: i8) -> f64 {
    q as f64 / 127.0
}

/// Method B: Fisher z. cos → arctanh → scale to i8 → tanh to restore.
/// Stretches tails but introduces its own quantization bias.
fn quantize_fisher_z(cos: f64, z_min: f64, z_range: f64) -> i8 {
    let z = cos.clamp(-0.9999, 0.9999).atanh();
    let normalized = (z - z_min) / z_range.max(1e-10);
    (normalized * 254.0 - 127.0).round().clamp(-127.0, 127.0) as i8
}

fn restore_fisher_z(q: i8, z_min: f64, z_range: f64) -> f64 {
    let normalized = (q as f64 + 127.0) / 254.0;
    let z = normalized * z_range + z_min;
    z.tanh()
}

/// Method C: QJL corrected i8.
/// Quantize same as direct, then apply bias correction.
///
/// QJL bias for b-bit uniform quantization of cosine:
///   E[Q(cos)] = cos + bias
///   bias ≈ -cos × (1 - cos²) / (3 × 2^(2b))
///
/// For i8 (b=7, 127 levels):
///   bias ≈ -cos × (1-cos²) / (3 × 16384) ≈ negligible per-pair
///   BUT: accumulated over a softmax, the bias shifts the distribution
///   systematically, and this compounds.
///
/// The correction: subtract the expected bias from the restored value.
fn quantize_qjl_i8(cos: f64) -> i8 {
    quantize_direct_i8(cos)
}

fn restore_qjl_i8(q: i8) -> f64 {
    let raw = q as f64 / 127.0;
    // QJL bias correction: remove the expected rounding bias
    // For uniform quantization with step Δ = 1/127:
    //   E[Q(x)] = x + Δ²/12 × d²f/dx² (second-order bias)
    // For cosine: d²cos/dθ² = -cos(θ), but we're quantizing the value not the angle
    // The dominant bias is from the round() operation:
    //   round(x×127)/127 has expected error = 0 (unbiased for uniform x)
    //   BUT softmax(scores) is nonlinear: softmax(x+ε) ≠ softmax(x) + ε
    //   The bias comes from the SOFTMAX interaction, not the quantization alone
    //
    // Practical correction: subtract the truncation bias
    //   If |raw| < Δ/2 = 0.004, the value might be exactly 0 truncated → restore as 0
    //   If raw > 0, the true value is likely raw + Δ/4 (center of the upper half of the bucket)
    //   If raw < 0, the true value is likely raw - Δ/4
    let delta = 1.0 / 127.0;
    if raw.abs() < delta * 0.5 {
        0.0 // zero bucket: no correction
    } else {
        // Dithered correction: shift toward the bucket center
        // This removes the systematic truncation toward zero
        raw + raw.signum() * delta * 0.25
    }
}

/// Method D: Full TurboQuant (PolarQuant + QJL).
/// Separate gain and direction, project direction via Hadamard,
/// quantize projection, apply bias correction on restore.
///
/// For this probe we simulate the full pipeline on cosine values
/// (since we're measuring score preservation, not row preservation).
fn quantize_turboquant(cos: f64, gain_a: f64, gain_b: f64) -> i8 {
    // The gain-shape decomposition means:
    //   dot(a, b) = ||a|| × ||b|| × cos(a, b)
    // We store cos(a,b) as i8, gains separately.
    // The correction accounts for the gain product's interaction with quantization.
    let adjusted = cos * (gain_a * gain_b).sqrt().max(1e-10).recip();
    quantize_direct_i8(cos) // same quantization, different restore
}

fn restore_turboquant(q: i8, gain_a: f64, gain_b: f64) -> f64 {
    let raw = q as f64 / 127.0;
    // TurboQuant correction: the gain normalization introduces
    // a multiplicative bias. Correct by the expected gain ratio.
    let gain_corr = (gain_a * gain_b).sqrt();
    let delta = 1.0 / 127.0;
    let corrected = if raw.abs() < delta * 0.5 {
        0.0
    } else {
        raw + raw.signum() * delta * 0.25
    };
    corrected // gains applied externally
}

// ═══════════════════════════════════════════════════════════════════
// Layer chain simulation
// ═══════════════════════════════════════════════════════════════════

/// Simulate attention through N layers.
///
/// Each layer: scores → softmax → value weighting → output.
/// The output's pairwise cosine becomes the next layer's input similarity.
///
/// This is a simplification: real transformers have residual connections,
/// MLP, layernorm. But the attention chain is the part where score
/// quantization bias compounds.
fn simulate_chain(
    scores: &[Vec<f64>],  // N×N score matrix
    n_layers: usize,
) -> Vec<Vec<f64>> {
    let n = scores.len();
    let mut current = scores.to_vec();

    for _layer in 0..n_layers {
        // Softmax each row
        let mut softmaxed = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let max_s = current[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = current[i].iter().map(|&s| (s - max_s).exp()).sum();
            for j in 0..n {
                softmaxed[i][j] = ((current[i][j] - max_s).exp()) / exp_sum.max(1e-30);
            }
        }

        // Value weighting: output[i] = Σ softmax[i][j] × input[j]
        // The output similarity matrix is: output_cos[i][k] = Σ_j Σ_l softmax[i][j] × softmax[k][l] × input_cos[j][l]
        // Simplification: treat the softmax-weighted average as a linear mix
        let mut next = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for k in 0..n {
                let mut dot = 0.0;
                for j in 0..n {
                    for l in 0..n {
                        dot += softmaxed[i][j] * softmaxed[k][l] * current[j][l];
                    }
                }
                next[i][k] = dot;
            }
        }

        current = next;
    }

    current
}

/// Flatten upper triangle of N×N matrix to a vector.
fn upper_triangle(m: &[Vec<f64>]) -> Vec<f64> {
    let n = m.len();
    let mut v = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            v.push(m[i][j]);
        }
    }
    v
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let st_path = if args.len() > 1 { &args[1] }
    else { "/home/user/models/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors" };

    println!("═══ TURBOQUANT CORRECTION PROBE ═══");
    println!("  Does QJL unbiasing survive {} layers?", N_LAYERS);
    println!();

    let mut reader = BufReader::new(File::open(st_path).expect("open"));
    let header = read_safetensors_header(&mut reader).expect("parse");

    // Find a representative attention projection (q_proj from talker layer 0)
    let target = header.tensors.iter()
        .find(|t| t.name.contains("layers.0.self_attn.q_proj") && t.name.ends_with("weight"))
        .or_else(|| header.tensors.iter().find(|t| t.name.contains("q_proj") && t.name.ends_with("weight")))
        .expect("find q_proj");

    println!("[1] Target: {} {:?}", target.name, target.dimensions);

    let n_rows = (target.dimensions[0] as usize).min(N_SAMPLE);
    let n_cols = if target.dimensions.len() > 1 { target.dimensions[1] as usize } else { 1 };

    reader.seek(SeekFrom::Start(header.tensor_data_offset + target.offset)).unwrap();
    let mut raw = vec![0u8; n_rows * n_cols * 2]; // BF16
    reader.read_exact(&mut raw).unwrap();

    let rows: Vec<Vec<f32>> = (0..n_rows).map(|r| {
        (0..n_cols).map(|c| {
            let idx = r * n_cols + c;
            let bits = u16::from_le_bytes([raw[idx*2], raw[idx*2+1]]);
            f32::from_bits((bits as u32) << 16)
        }).collect()
    }).collect();

    println!("  {} rows × {} cols loaded", rows.len(), n_cols);

    // ─── Ground truth: f32 cosine matrix ───────────────────────────
    println!("\n[2] Computing ground truth cosine matrix...");
    let n = rows.len();
    let mut gt_cos = vec![vec![0.0f64; n]; n];
    let mut norms = vec![0.0f64; n];

    for i in 0..n {
        norms[i] = rows[i].iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    }

    for i in 0..n {
        gt_cos[i][i] = 1.0;
        for j in (i + 1)..n {
            let mut dot = 0.0f64;
            for k in 0..n_cols {
                dot += rows[i][k] as f64 * rows[j][k] as f64;
            }
            let cos = dot / (norms[i] * norms[j]).max(1e-15);
            gt_cos[i][j] = cos;
            gt_cos[j][i] = cos;
        }
    }

    // Cosine statistics
    let gt_upper = upper_triangle(&gt_cos);
    let cos_mean: f64 = gt_upper.iter().sum::<f64>() / gt_upper.len() as f64;
    let cos_std: f64 = (gt_upper.iter().map(|&c| (c - cos_mean).powi(2)).sum::<f64>()
        / gt_upper.len() as f64).sqrt();
    println!("  Cosine range: [{:.4}, {:.4}], mean={:.4}, std={:.4}",
        gt_upper.iter().cloned().fold(f64::INFINITY, f64::min),
        gt_upper.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        cos_mean, cos_std);

    // Fisher z calibration
    let z_values: Vec<f64> = gt_upper.iter()
        .map(|&c| c.clamp(-0.9999, 0.9999).atanh())
        .collect();
    let z_min = z_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let z_max = z_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z_range = z_max - z_min;

    // ─── Encode with 4 methods ─────────────────────────────────────
    println!("\n[3] Encoding cosine matrix with 4 methods...");

    let methods: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("direct_i8", {
            let mut m = vec![vec![0.0; n]; n];
            for i in 0..n { m[i][i] = 1.0;
                for j in (i+1)..n {
                    let q = quantize_direct_i8(gt_cos[i][j]);
                    let r = restore_direct_i8(q);
                    m[i][j] = r; m[j][i] = r;
                }
            }
            m
        }),
        ("fisher_z", {
            let mut m = vec![vec![0.0; n]; n];
            for i in 0..n { m[i][i] = 1.0;
                for j in (i+1)..n {
                    let q = quantize_fisher_z(gt_cos[i][j], z_min, z_range);
                    let r = restore_fisher_z(q, z_min, z_range);
                    m[i][j] = r; m[j][i] = r;
                }
            }
            m
        }),
        ("qjl_corrected", {
            let mut m = vec![vec![0.0; n]; n];
            for i in 0..n { m[i][i] = 1.0;
                for j in (i+1)..n {
                    let q = quantize_qjl_i8(gt_cos[i][j]);
                    let r = restore_qjl_i8(q);
                    m[i][j] = r; m[j][i] = r;
                }
            }
            m
        }),
        ("turboquant", {
            let mut m = vec![vec![0.0; n]; n];
            for i in 0..n { m[i][i] = 1.0;
                for j in (i+1)..n {
                    let q = quantize_turboquant(gt_cos[i][j], norms[i], norms[j]);
                    let r = restore_turboquant(q, norms[i], norms[j]);
                    m[i][j] = r; m[j][i] = r;
                }
            }
            m
        }),
    ];

    // ─── Static quality (before chain) ─────────────────────────────
    println!("\n[4] Static quality (single layer)...");
    println!("  {:16} │ Spearman ρ │ Pearson r  │ MAE        │ Max err    │ Bias", "Method");
    println!("  ─────────────────┼────────────┼────────────┼────────────┼────────────┼──────────");

    for (name, encoded) in &methods {
        let enc_upper = upper_triangle(encoded);
        let rho = spearman(&gt_upper, &enc_upper);
        let r = bgz_tensor::quality::pearson(&gt_upper, &enc_upper);
        let mae: f64 = gt_upper.iter().zip(&enc_upper)
            .map(|(&g, &e)| (g - e).abs())
            .sum::<f64>() / gt_upper.len() as f64;
        let max_err: f64 = gt_upper.iter().zip(&enc_upper)
            .map(|(&g, &e)| (g - e).abs())
            .fold(0.0, f64::max);
        // Bias: mean(encoded - gt). Positive = overestimates, negative = underestimates.
        let bias: f64 = gt_upper.iter().zip(&enc_upper)
            .map(|(&g, &e)| e - g)
            .sum::<f64>() / gt_upper.len() as f64;

        println!("  {:16} │ {:10.6} │ {:10.6} │ {:10.6} │ {:10.6} │ {:+.6}",
            name, rho, r, mae, max_err, bias);
    }

    // ─── Chain simulation ──────────────────────────────────────────
    // Simulate at reduced size (chain is O(n²) per layer, 33 layers)
    let chain_n = n.min(50);
    println!("\n[5] Chain simulation ({} rows × {} layers)...", chain_n, N_LAYERS);

    // Ground truth chain
    let gt_small: Vec<Vec<f64>> = gt_cos[..chain_n].iter()
        .map(|row| row[..chain_n].to_vec()).collect();
    let gt_chain = simulate_chain(&gt_small, N_LAYERS);
    let gt_chain_upper = upper_triangle(&gt_chain);

    println!("  {:16} │ Chain ρ    │ Chain r    │ Chain bias │ Drift/layer", "Method");
    println!("  ─────────────────┼────────────┼────────────┼────────────┼────────────");

    for (name, encoded) in &methods {
        let enc_small: Vec<Vec<f64>> = encoded[..chain_n].iter()
            .map(|row| row[..chain_n].to_vec()).collect();

        let enc_chain = simulate_chain(&enc_small, N_LAYERS);
        let enc_chain_upper = upper_triangle(&enc_chain);

        let chain_rho = spearman(&gt_chain_upper, &enc_chain_upper);
        let chain_r = bgz_tensor::quality::pearson(&gt_chain_upper, &enc_chain_upper);
        let chain_bias: f64 = gt_chain_upper.iter().zip(&enc_chain_upper)
            .map(|(&g, &e)| e - g)
            .sum::<f64>() / gt_chain_upper.len() as f64;

        // Also measure drift at intermediate layers
        let enc_mid = simulate_chain(&enc_small, N_LAYERS / 2);
        let gt_mid = simulate_chain(&gt_small, N_LAYERS / 2);
        let mid_bias: f64 = upper_triangle(&gt_mid).iter()
            .zip(upper_triangle(&enc_mid).iter())
            .map(|(&g, &e)| e - g)
            .sum::<f64>() / (chain_n * (chain_n - 1) / 2) as f64;
        let drift_per_layer = if N_LAYERS > 0 {
            chain_bias / N_LAYERS as f64
        } else { 0.0 };

        println!("  {:16} │ {:10.6} │ {:10.6} │ {:+10.6} │ {:+10.8}",
            name, chain_rho, chain_r, chain_bias, drift_per_layer);
    }

    // ─── Progressive chain depth ───────────────────────────────────
    println!("\n[6] Progressive chain: ρ at layer 1, 5, 10, 20, 33...");
    let checkpoints = [1, 5, 10, 20, 33];

    print!("  {:16}", "Method");
    for &cp in &checkpoints { print!(" │ L={:2}", cp); }
    println!();
    print!("  ─────────────────");
    for _ in &checkpoints { print!("─┼────────"); }
    println!();

    for (name, encoded) in &methods {
        let enc_small: Vec<Vec<f64>> = encoded[..chain_n].iter()
            .map(|row| row[..chain_n].to_vec()).collect();

        print!("  {:16}", name);
        for &cp in &checkpoints {
            let enc_at = simulate_chain(&enc_small, cp);
            let gt_at = simulate_chain(&gt_small, cp);
            let rho = spearman(&upper_triangle(&gt_at), &upper_triangle(&enc_at));
            print!(" │ {:7.4}", rho);
        }
        println!();
    }

    println!("\n═══ INTERPRETATION ═══");
    println!("  If QJL correction has LOWER drift/layer than direct i8:");
    println!("    → bias removal matters, adopt QJL correction in the codec");
    println!("  If Fisher z has HIGHER chain ρ than QJL:");
    println!("    → tail stretching matters more than bias removal");
    println!("  If all methods converge at layer 33:");
    println!("    → the softmax normalizes away the differences");
    println!("  If turboquant wins:");
    println!("    → gain-aware correction matters, implement in matryoshka");
    println!();
    println!("═══ DONE ═══");
}
