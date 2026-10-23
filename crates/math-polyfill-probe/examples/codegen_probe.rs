//! D-MPF-2 LLVM witness: the ndarray primitives the native fold delegates to,
//! each behind one `#[inline(never)]` symbol so `--emit=asm` shows what they
//! compile to, after a self-check against a scalar reference. Modelled on
//! ndarray `scripts/codegen-witness.sh` + `examples/ternlog_codegen_probe.rs`.
//!
//! cargo rustc --release --example codegen_probe -- --emit=asm -C debuginfo=0

use std::hint::black_box;

#[inline(never)]
pub fn probe_gt_i32_to_mask(x: &[i32], t: i32, m: &mut [u64]) {
    ndarray::simd::gt_i32_to_mask(x, t, m);
}

#[inline(never)]
pub fn probe_masked_sum_i32(x: &[i32], m: &[u64]) -> i64 {
    ndarray::simd::masked_sum_i32(x, m)
}

/// The proposed `masked_sum_i32` body: branch-free, one select per row, so
/// its cost should not depend on how many bits are set.
#[inline(never)]
pub fn probe_masked_sum_branchless(x: &[i32], m: &[u64]) -> i64 {
    let mut acc = 0i64;
    for (w, chunk) in x.chunks(64).enumerate() {
        let word = m[w];
        let mut part = 0i64;
        for (j, &v) in chunk.iter().enumerate() {
            part += i64::from(v) & -(((word >> j) & 1) as i64);
        }
        acc += part;
    }
    acc
}

#[inline(never)]
pub fn probe_scalar_sum_gt(x: &[i32], t: i32) -> i64 {
    x.iter()
        .map(|&v| if v > t { i64::from(v) } else { 0 })
        .sum()
}

fn main() {
    let x: Vec<i32> = (0..4096i32).map(|i| (i * 13) % 97 - 48).collect();
    let mut m = vec![0u64; 64];
    for t in [-100, -1, 0, 7, 100] {
        probe_gt_i32_to_mask(black_box(&x), t, &mut m);
        for (r, &v) in x.iter().enumerate() {
            assert_eq!(
                m[r / 64] >> (r % 64) & 1 == 1,
                v > t,
                "mask bit {r} at t={t}"
            );
        }
        let want: i64 = x.iter().filter(|&&v| v > t).map(|&v| i64::from(v)).sum();
        assert_eq!(probe_masked_sum_i32(black_box(&x), &m), want);
        assert_eq!(probe_scalar_sum_gt(black_box(&x), t), want);
        assert_eq!(probe_masked_sum_branchless(black_box(&x), &m), want);
    }
    println!("codegen_probe self-check: ok");
}
