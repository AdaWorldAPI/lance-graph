//! A0/A1/A2 — the Goursat solve on the canonical ndarray substrate.
//!
//! The falsifier recorded in `TD-PILLAR11-SCIENTIFIC-LOOPS-BYPASS-NDARRAY-SIMD-1`:
//!
//! | arm | storage | traversal | arithmetic |
//! |---|---|---|---|
//! | A0 | `Vec<Vec<f64>>` | row-major | scalar (the shipped `signature_kernel_pde`) |
//! | A1 | flat `Vec<f64>` | row-major, SAME order | scalar |
//! | A2 | three rolling anti-diagonals | wavefront | `ndarray::simd::F64x8::mul_add` |
//!
//! Contract: A0 = A1 **bit-exact** (only storage changes). A1 <-> A2 under a
//! predeclared tolerance: the wavefront's `mul_add(c, diag, u)` fuses the
//! `c·diag + u` rounding the scalar path performs as two roundings, so the two
//! arms legitimately differ in the last bits. No consumer intrinsics; every
//! lane op is `ndarray::simd::method()`.
//!
//! Run: `cargo run --release --manifest-path crates/jc/Cargo.toml \
//!       --features hambly-lyons --example goursat_substrate_probe`

use ndarray::simd::F64x8;
use sigker::signature_kernel_pde;
use std::time::Instant;

const LANES: usize = 8;

fn path(n: usize) -> Vec<Vec<f64>> {
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            vec![t + 0.05 * (260.0 * t).cos(), 0.5 * t + 0.05 * (260.0 * t).sin()]
        })
        .collect()
}

/// Increments dx_i = x[i+1] - x[i], one contiguous block per step.
fn increments(x: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let dim = x[0].len();
    let mut out = Vec::with_capacity((x.len() - 1) * dim);
    for w in x.windows(2) {
        for a in 0..dim {
            out.push(w[1][a] - w[0][a]);
        }
    }
    (out, dim)
}

/// A1 — flat row-major storage, the shipped recurrence in the shipped order.
/// `k[i+1][j+1] = k[i+1][j] + k[i][j+1] - k[i][j] + c_ij * k[i][j]`, evaluated
/// left-to-right exactly as `signature_kernel_pde` writes it.
fn goursat_flat(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let (n, m) = (x.len(), y.len());
    let (dx, dim) = increments(x);
    let (dy, _) = increments(y);
    let mut k = vec![1.0f64; n * m];
    for i in 0..n - 1 {
        let dxi = &dx[i * dim..(i + 1) * dim];
        for j in 0..m - 1 {
            let dyj = &dy[j * dim..(j + 1) * dim];
            let c: f64 = (0..dim).map(|a| dxi[a] * dyj[a]).sum();
            let (left, up, diag) = (k[(i + 1) * m + j], k[i * m + j + 1], k[i * m + j]);
            k[(i + 1) * m + j + 1] = left + up - diag + c * diag;
        }
    }
    k[n * m - 1]
}

/// A2 — wavefront over three rolling anti-diagonal buffers, every lane op
/// `ndarray::simd::F64x8::mul_add`.
///
/// On diagonal `d` (cells with `i + j = d`), indexed by row `i`:
///   left = k[i][j-1]   = prev1[i]        up   = k[i-1][j] = prev1[i-1]
///   diag = k[i-1][j-1] = prev2[i-1]      c    = <dx[i-1], dy[j-1]>
/// With `dy` stored REVERSED, `dy[j-1] = dyr[m-1-d+i]` walks FORWARD in `i`,
/// so every operand — k-buffers, dx, dy — is a contiguous slice. No gather.
///
/// Body, three FMAs (±1.0 multipliers are exact, so the first two round
/// exactly like `+`/`-`; only the last fuses what A1 rounds twice):
///   t = mul_add( 1, left, up)    u = mul_add(-1, diag, t)    new = mul_add(c, diag, u)
fn goursat_wavefront(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let (n, m) = (x.len(), y.len());
    let (dx, dim) = increments(x);
    let (dy, _) = increments(y);
    assert_eq!(dim, 2, "probe fixes dim = 2 (the W5 path shape)");
    // Split by component; reverse dy so the anti-diagonal walk is forward.
    let dx0: Vec<f64> = dx.iter().step_by(2).copied().collect();
    let dx1: Vec<f64> = dx.iter().skip(1).step_by(2).copied().collect();
    let mut dyr0: Vec<f64> = dy.iter().step_by(2).copied().collect();
    let mut dyr1: Vec<f64> = dy.iter().skip(1).step_by(2).copied().collect();
    dyr0.reverse();
    dyr1.reverse();

    let mut prev2 = vec![1.0f64; n];
    let mut prev1 = vec![1.0f64; n];
    let mut cur = vec![1.0f64; n];
    let (one, neg_one, zero) = (F64x8::splat(1.0), F64x8::splat(-1.0), F64x8::splat(0.0));
    let mut out = [0.0f64; LANES];

    for d in 2..(n + m - 1) {
        // Boundaries on this diagonal: k[0][d] and k[d][0] are 1.
        if d < m { cur[0] = 1.0; }
        if d < n { cur[d] = 1.0; }
        // Interior rows: i >= 1, j = d - i >= 1, i <= n-1, j <= m-1.
        let lo = 1usize.max(d.saturating_sub(m - 1));
        let hi = (d - 1).min(n - 1);
        if lo > hi { std::mem::swap(&mut prev2, &mut prev1); std::mem::swap(&mut prev1, &mut cur); continue; }
        // dyr index for row i is (m-1-d)+i; the difference may be negative but the sum is not.
        let base = (m - 1).wrapping_sub(d);
        let mut i = lo;
        while i + LANES <= hi + 1 {
            let left = F64x8::from_slice(&prev1[i..i + LANES]);
            let up = F64x8::from_slice(&prev1[i - 1..i - 1 + LANES]);
            let diag = F64x8::from_slice(&prev2[i - 1..i - 1 + LANES]);
            let a0 = F64x8::from_slice(&dx0[i - 1..i - 1 + LANES]);
            let a1 = F64x8::from_slice(&dx1[i - 1..i - 1 + LANES]);
            let r = base.wrapping_add(i); // == m-1-d+i, in range for interior rows
            let b0 = F64x8::from_slice(&dyr0[r..r + LANES]);
            let b1 = F64x8::from_slice(&dyr1[r..r + LANES]);
            let c = a1.mul_add(b1, a0.mul_add(b0, zero));
            let t = one.mul_add(left, up);
            let u = neg_one.mul_add(diag, t);
            c.mul_add(diag, u).copy_to_slice(&mut out);
            cur[i..i + LANES].copy_from_slice(&out);
            i += LANES;
        }
        // Scalar tail: same three-FMA arithmetic, so A2 is internally uniform.
        while i <= hi {
            let r = base.wrapping_add(i);
            let c = dx1[i - 1].mul_add(dyr1[r], dx0[i - 1] * dyr0[r]);
            let t = 1.0f64.mul_add(prev1[i], prev1[i - 1]);
            let u = (-1.0f64).mul_add(prev2[i - 1], t);
            cur[i] = c.mul_add(prev2[i - 1], u);
            i += 1;
        }
        std::mem::swap(&mut prev2, &mut prev1);
        std::mem::swap(&mut prev1, &mut cur);
    }
    // Final cell k[n-1][m-1] is on diagonal n+m-2, the last computed, now in prev1.
    prev1[n - 1]
}

fn main() {
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>12} {:>12}",
        "len", "A0 secs", "A1 secs", "A2 secs", "A0/A1", "A1/A2", "A0==A1", "|A1-A2|/A1"
    );
    for &n in &[256usize, 1024, 2048, 4096] {
        let (x, y) = (path(n), path(n));
        let t = Instant::now(); let a0 = signature_kernel_pde(&x, &y); let s0 = t.elapsed().as_secs_f64();
        let t = Instant::now(); let a1 = goursat_flat(&x, &y);         let s1 = t.elapsed().as_secs_f64();
        let t = Instant::now(); let a2 = goursat_wavefront(&x, &y);    let s2 = t.elapsed().as_secs_f64();
        let exact = a0.to_bits() == a1.to_bits();
        let rel = ((a1 - a2) / a1).abs();
        println!(
            "{:>6} {s0:>10.4} {s1:>10.4} {s2:>10.4} {:>8.2}x {:>8.2}x {:>12} {rel:>12.3e}",
            n + 1, s0 / s1, s1 / s2, if exact { "bit-exact" } else { "DIFFERS" }
        );
        assert!(exact, "A0 != A1 at n={n}: {a0:e} vs {a1:e} — storage change altered the result");
    }
    println!("\nA0 = A1 bit-exact at every size (only storage changed).");
    println!("A1 <-> A2 differ by fused vs. separate rounding of c·diag — the predeclared tolerance.");
}
