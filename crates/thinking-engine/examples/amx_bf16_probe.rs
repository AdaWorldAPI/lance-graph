//! AMX BF16 probe — validates TDPBF16PS works on this hardware.
//!
//! Tests one 16×16 output block computed from K=32 BF16 elements per tile
//! step, accumulated over multiple K-blocks. Compares to a scalar reference
//! that uses the canonical ndarray pattern: chunks_exact(16) windowed
//! iteration + F32x16::mul_add (FMA).
//!
//! SIGILL protection: runtime `amx_available()` check — exits early if
//! the hypervisor hasn't granted tile state (CPUID + _xgetbv(0) +
//! prctl ARCH_REQ_XCOMP_PERM).
//!
//! Usage:
//!   cargo run --release --example amx_bf16_probe \
//!     --manifest-path crates/thinking-engine/Cargo.toml
//!
//! Pass criterion: max |amx - ref| < 1e-2 (BF16 is ~7-bit mantissa ≈ 1/128).

use std::arch::asm;
use ndarray::hpc::amx_matmul::{TileConfig, tile_loadconfig, tile_zero, tile_load, tile_store, tile_release};
use ndarray::hpc::amx_matmul::amx_available;
use ndarray::simd::{F32x16, f32_to_bf16_batch, bf16_to_f32_batch};

// ═════════════════════════════════════════════════════════════════════
// TDPBF16PS primitive (NOT in ndarray — adding locally here)
// ═════════════════════════════════════════════════════════════════════

/// TDPBF16PS tmm0, tmm1, tmm2 — encoded as raw bytes.
///
/// Encoding: VEX.128.F3.0F38.W0 5C /r with ModRM=11 000 001 (tmm0, tmm1, tmm2)
///   C4 E2 72 5C C1
/// vs TDPBUSD (C4 E2 73 5E C1): pp field flips F2→F3 (73→72), opcode 5E→5C.
///
/// Performs: tmm0[i,j] += Σ_{k=0..K/2} tmm1[i, 2k]*tmm2[k, 2j] + tmm1[i, 2k+1]*tmm2[k, 2j+1]
/// Tile shapes at K=32, M=16, N=16:
///   tmm0 (C): 16×16 f32   (16 rows × 64 bytes)
///   tmm1 (A): 16×32 bf16  (16 rows × 64 bytes, plain row-major)
///   tmm2 (B): 16×16 bf16 pairs (K/2=16 rows × 64 bytes, VNNI-packed)
///
/// SAFETY: tiles 0/1/2 must be configured and loaded; AMX enabled.
#[inline]
unsafe fn tile_dpbf16ps() {
    asm!(".byte 0xc4, 0xe2, 0x72, 0x5c, 0xc1", options(nostack, nomem));
}

// ═════════════════════════════════════════════════════════════════════
// VNNI packing helper — rearranges a row-major B[K,N] bf16 matrix into
// the K/2 × (N*2) VNNI layout required by TDPBF16PS tile 2.
// ═════════════════════════════════════════════════════════════════════

/// Pack B[K, N] bf16 row-major → [K/2, N*2] VNNI pairs.
/// Output[i, 2j]   = B[2i,   j]
/// Output[i, 2j+1] = B[2i+1, j]
fn vnni_pack_bf16(src: &[u16], k: usize, n: usize) -> Vec<u16> {
    debug_assert_eq!(src.len(), k * n);
    debug_assert_eq!(k % 2, 0, "K must be even for VNNI BF16");
    let mut out = vec![0u16; k * n];
    for i in 0..(k / 2) {
        for j in 0..n {
            out[i * n * 2 + 2 * j]     = src[(2 * i)     * n + j];
            out[i * n * 2 + 2 * j + 1] = src[(2 * i + 1) * n + j];
        }
    }
    out
}

// ═════════════════════════════════════════════════════════════════════
// AMX BF16 tile GEMM: C[16,16] += A[16, K] × B[K, 16] accumulated over K.
// K must be a multiple of 32. A in row-major bf16. B_vnni is pre-packed.
// ═════════════════════════════════════════════════════════════════════

/// C += A × B where A is [16, K] bf16 and B is pre-packed VNNI [K, 16] bf16.
/// Output C is [16, 16] f32.
///
/// SAFETY: AMX must be available; buffers must outlive the call.
unsafe fn amx_bf16_gemm_16x16xK(
    a_bf16:   &[u16],   // [16, K] row-major, stride K
    b_vnni:   &[u16],   // [K/2, 32] VNNI pairs (stride 32 = 16*2 bf16)
    c_f32:    &mut [f32], // [16, 16] row-major, stride 16
    k: usize,
) {
    assert_eq!(k % 32, 0, "K must be multiple of 32");
    assert_eq!(a_bf16.len(), 16 * k);
    assert_eq!(b_vnni.len(), k * 16);  // K/2 * 32 = k*16
    assert_eq!(c_f32.len(), 16 * 16);

    // Tile config: same shape as TDPBUSD max (K_bytes=64) by coincidence.
    let cfg = TileConfig::for_dpbusd(64);
    tile_loadconfig(&cfg);
    tile_zero(0);

    // Iterate K in 32-element blocks
    let k_blocks = k / 32;
    for kb in 0..k_blocks {
        // A block: 16 rows × 32 bf16 = 16 × 64 bytes, stride=K*2 bytes total row
        let a_ptr = a_bf16.as_ptr().add(kb * 32) as *const u8;
        let a_stride = k * 2;  // full A row stride in bytes

        // B VNNI block: 16 rows (K/2 per block = 16) × 64 bytes, stride=64
        let b_ptr = b_vnni.as_ptr().add(kb * 16 * 32) as *const u8;
        let b_stride = 64;

        tile_load(1, a_ptr, a_stride);
        tile_load(2, b_ptr, b_stride);
        tile_dpbf16ps();
    }

    // Store result: 16 rows × 16 f32 = 16 × 64 bytes, stride 64
    tile_store(0, c_f32.as_mut_ptr() as *mut u8, 64);
    tile_release();
}

// ═════════════════════════════════════════════════════════════════════
// Scalar reference: same computation, using chunks_exact (ndarray's
// "array_window" idiom) + F32x16::mul_add (FMA).
// ═════════════════════════════════════════════════════════════════════

/// Reference: C[i,j] = Σ_k A_f32[i, k] * B_f32[k, j], using F32x16 FMA.
fn ref_gemm_f32_f32x16(
    a_f32: &[f32], b_f32: &[f32], c_f32: &mut [f32],
    m: usize, n: usize, k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            // One output element: dot of a row (A[i, :]) with a column (B[:, j])
            // We gather the column into a temp buffer then chunks_exact(16) + mul_add.
            // (Allocating once per (i,j) is fine here — this is a REFERENCE.)
            let mut col = vec![0.0f32; k];
            for kk in 0..k { col[kk] = b_f32[kk * n + j]; }

            let row = &a_f32[i * k .. i * k + k];
            let mut acc = F32x16::splat(0.0);
            for (rc, cc) in row.chunks_exact(16).zip(col.chunks_exact(16)) {
                // FMA: acc = row_v * col_v + acc
                let rv = F32x16::from_slice(rc);
                let cv = F32x16::from_slice(cc);
                acc = rv.mul_add(cv, acc);
            }
            let mut s = acc.reduce_sum();
            // Tail (should be zero if k % 16 == 0)
            let tail_start = (k / 16) * 16;
            for kk in tail_start..k { s += row[kk] * col[kk]; }
            c_f32[i * n + j] = s;
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Probe
// ═════════════════════════════════════════════════════════════════════

fn main() {
    println!("═══ AMX BF16 TDPBF16PS PROBE ═══");

    if !amx_available() {
        println!("  AMX not available (CPUID or OS check failed). Exiting 0.");
        return;
    }
    println!("  AMX available: CPUID + XCR0 + prctl(ARCH_REQ_XCOMP_PERM) OK");

    // Build a small test case: M=16, N=16, K=64 (two 32-wide tile steps)
    const M: usize = 16;
    const N: usize = 16;
    const K: usize = 64;

    // Deterministic pseudo-random inputs in f32
    let mut a_f32 = vec![0.0f32; M * K];
    let mut b_f32 = vec![0.0f32; K * N];
    for i in 0..a_f32.len() { a_f32[i] = ((i as i32 * 1103515245 + 12345) as f32 / 2147483648.0).clamp(-1.0, 1.0); }
    for i in 0..b_f32.len() { b_f32[i] = (((i as i32 * 69069 + 1) as f32).sin() * 0.5).clamp(-1.0, 1.0); }

    // Reference computation (f32 × f32 → f32, no quantization)
    let mut c_ref = vec![0.0f32; M * N];
    ref_gemm_f32_f32x16(&a_f32, &b_f32, &mut c_ref, M, N, K);
    println!("  [ref f32×f32] computed {}×{} from K={}", M, N, K);
    println!("    c_ref[0..4] = {:?}", &c_ref[..4]);

    // Also a scalar BF16-truncated reference (matches what AMX sees internally)
    let a_bf16 = {
        let mut buf = vec![0u16; a_f32.len()];
        f32_to_bf16_batch(&a_f32, &mut buf); buf
    };
    let b_bf16 = {
        let mut buf = vec![0u16; b_f32.len()];
        f32_to_bf16_batch(&b_f32, &mut buf); buf
    };
    let a_bf16_as_f32 = {
        let mut buf = vec![0.0f32; a_bf16.len()];
        bf16_to_f32_batch(&a_bf16, &mut buf); buf
    };
    let b_bf16_as_f32 = {
        let mut buf = vec![0.0f32; b_bf16.len()];
        bf16_to_f32_batch(&b_bf16, &mut buf); buf
    };
    let mut c_ref_bf16 = vec![0.0f32; M * N];
    ref_gemm_f32_f32x16(&a_bf16_as_f32, &b_bf16_as_f32, &mut c_ref_bf16, M, N, K);
    println!("  [ref bf16→f32] c_ref_bf16[0..4] = {:?}", &c_ref_bf16[..4]);

    // AMX computation — VNNI-pack B, call tile GEMM
    let b_vnni = vnni_pack_bf16(&b_bf16, K, N);
    let mut c_amx = vec![0.0f32; M * N];
    // SAFETY: amx_available() checked above.
    unsafe { amx_bf16_gemm_16x16xK(&a_bf16, &b_vnni, &mut c_amx, K); }
    println!("  [amx bf16×bf16] c_amx[0..4]     = {:?}", &c_amx[..4]);

    // Compare AMX to BF16-truncated reference (apples to apples)
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    for i in 0..(M * N) {
        let e = (c_amx[i] - c_ref_bf16[i]).abs();
        if e > max_abs { max_abs = e; }
        let r = e / c_ref_bf16[i].abs().max(1e-6);
        if r > max_rel { max_rel = r; }
        sum_sq_err += (e as f64) * (e as f64);
    }
    let rmse = (sum_sq_err / (M * N) as f64).sqrt();

    println!("\n  Compare amx vs ref_bf16:");
    println!("    max |err| = {:.6}", max_abs);
    println!("    max rel   = {:.6}", max_rel);
    println!("    rmse      = {:.6}", rmse);

    // Pass: BF16 mantissa precision ~7 bits; for K=64 accumulation we expect
    // roughly O(sqrt(K)) growth → still ≪ 1e-2.
    let pass = max_abs < 1e-2;
    println!("\n  {} (threshold max |err| < 1e-2)",
        if pass { "★ PASS" } else { "✗ FAIL" });
    if !pass { std::process::exit(1); }
}
