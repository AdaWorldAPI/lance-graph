//! AMX BF16 probe — validates ndarray::hpc::bf16_tile_gemm on this host.
//!
//! The polyfill auto-dispatches:
//!   amx_available() → TDPBF16PS tile GEMM  (ndarray::hpc::amx_matmul)
//!   otherwise        → AVX-512 F32x16 + mul_add FMA fallback
//!
//! Probe prints which path was taken, computes a 16×16×K=64 GEMM, and
//! compares against a scalar BF16-truncated reference. Pass: max |err| < 1e-2.
//!
//! Usage:
//!   cargo run --release --example amx_bf16_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml

use ndarray::hpc::amx_matmul::amx_available;
use ndarray::hpc::bf16_tile_gemm::bf16_tile_gemm_16x16;
use ndarray::simd::{F32x16, f32_to_bf16_batch, bf16_to_f32_batch};

/// Scalar BF16-truncated reference using F32x16 + mul_add FMA
/// (canonical ndarray "array_window" idiom).
fn ref_gemm_f32x16(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut col = vec![0.0f32; k];
            for kk in 0..k { col[kk] = b[kk * n + j]; }
            let row = &a[i * k .. i * k + k];
            let mut acc = F32x16::splat(0.0);
            for (rc, cc) in row.chunks_exact(16).zip(col.chunks_exact(16)) {
                acc = F32x16::from_slice(rc).mul_add(F32x16::from_slice(cc), acc);
            }
            c[i * n + j] = acc.reduce_sum();
        }
    }
}

fn main() {
    println!("═══ bf16_tile_gemm polyfill probe ═══");
    let path = if amx_available() { "AMX (TDPBF16PS)" } else { "AVX-512 F32x16 fallback" };
    println!("  Dispatch path: {}", path);

    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 64;

    // Deterministic pseudo-random inputs in f32
    let mut a_f32 = vec![0.0f32; M * K];
    let mut b_f32 = vec![0.0f32; K * N];
    for i in 0..a_f32.len() {
        a_f32[i] = (((i as i32).wrapping_mul(1103515245).wrapping_add(12345) >> 8) as f32
                    / 2147483648.0).clamp(-1.0, 1.0);
    }
    for i in 0..b_f32.len() {
        b_f32[i] = (((i as i32).wrapping_mul(69069).wrapping_add(1) >> 8) as f32
                    / 2147483648.0).clamp(-1.0, 1.0);
    }

    // Quantize inputs to bf16 (the GEMM operands)
    let mut a_bf16 = vec![0u16; a_f32.len()];
    let mut b_bf16 = vec![0u16; b_f32.len()];
    f32_to_bf16_batch(&a_f32, &mut a_bf16);
    f32_to_bf16_batch(&b_f32, &mut b_bf16);

    // BF16-truncated reference (what the GEMM sees internally)
    let mut a_back = vec![0.0f32; a_f32.len()];
    let mut b_back = vec![0.0f32; b_f32.len()];
    bf16_to_f32_batch(&a_bf16, &mut a_back);
    bf16_to_f32_batch(&b_bf16, &mut b_back);
    let mut c_ref = vec![0.0f32; M * N];
    ref_gemm_f32x16(&a_back, &b_back, &mut c_ref, M, N, K);
    println!("  [ref bf16→f32] c_ref[0..4] = {:?}", &c_ref[..4]);

    // Run the polyfill
    let mut c = vec![0.0f32; M * N];
    bf16_tile_gemm_16x16(&a_bf16, &b_bf16, &mut c, K);
    println!("  [polyfill]     c[0..4]     = {:?}", &c[..4]);

    // Compare
    let mut max_abs = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    for i in 0..(M * N) {
        let e = (c[i] - c_ref[i]).abs();
        if e > max_abs { max_abs = e; }
        sum_sq_err += (e as f64) * (e as f64);
    }
    let rmse = (sum_sq_err / (M * N) as f64).sqrt();
    println!("\n  max |err| = {:.6}", max_abs);
    println!("  rmse      = {:.6}", rmse);

    let pass = max_abs < 1e-2;
    println!("\n  {} (threshold max |err| < 1e-2)",
        if pass { "★ PASS" } else { "✗ FAIL" });
    if !pass { std::process::exit(1); }
}
