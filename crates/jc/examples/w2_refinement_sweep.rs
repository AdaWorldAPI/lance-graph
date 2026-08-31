//! W2 pre-registration: the refinement sweep that ε(N) is pinned FROM.
//!
//! Run: `cargo run -p jc --features hambly-lyons --example w2_refinement_sweep`
//!
//! The amendment to the plan (post-merge codex P2) established that a raw
//! 3-point out-and-back is NOT tree-invariant under the first-order Goursat
//! scheme: for increment `u` with `a = ‖u‖²` the discrete recurrence gives
//! `K(x,x) = 1 + a²` against 1 for the constant base — a DISCRETIZATION
//! artifact, not a uniqueness failure. So the leg resamples and gates against
//! a resolution-dependent tolerance. This example measures the convergence
//! before any tolerance is written down.
//!
//! Statistic (both legs): the normalized kernel against the constant base,
//! `dev = |K(x,y)/√(K(x,x)K(y,y)) − 1|`. For the constant base `K(y,y) = 1`
//! and `K(x,y) = 1` (a constant path has zero increments, so every cell
//! coefficient vanishes), hence `dev = |1/√(K(x,x)) − 1|`.
//!
//! The two legs must behave DIFFERENTLY under refinement, and that difference
//! is the whole content of the theorem at depth-∞:
//!   forward  (out-and-back, tree-like)  → 0     as N → ∞   [artifact only]
//!   converse (triangle, non-tree)       → const as N → ∞   [real signature]

use sigker::signature_kernel_pde;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn rand_in(state: &mut u64, lo: f64, hi: f64) -> f64 {
    let u = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
    lo + u * (hi - lo)
}
fn random_point(state: &mut u64, dim: usize) -> Vec<f64> {
    (0..dim).map(|_| rand_in(state, -1.0, 1.0)).collect()
}

/// Resample a polyline so each segment carries `per_seg` sub-intervals.
pub fn resample(corners: &[Vec<f64>], per_seg: usize) -> Vec<Vec<f64>> {
    let dim = corners[0].len();
    let mut out = vec![corners[0].clone()];
    for w in corners.windows(2) {
        for s in 1..=per_seg {
            let t = s as f64 / per_seg as f64;
            out.push(
                (0..dim)
                    .map(|a| w[0][a] + t * (w[1][a] - w[0][a]))
                    .collect(),
            );
        }
    }
    out
}

/// `|1/√(K(x,x)) − 1|` — deviation of the normalized kernel from the
/// constant base. Zero iff the path's depth-∞ signature is the identity.
pub fn deviation_from_constant(path: &[Vec<f64>]) -> f64 {
    let kxx = signature_kernel_pde(path, path);
    (1.0 / kxx.sqrt() - 1.0).abs()
}

fn main() {
    const N_PAIRS: usize = 100;
    const DIM: usize = 3;

    println!(
        "{:>8} {:>7} {:>14} {:>14} {:>14} {:>10} {:>9}",
        "per_seg", "points", "fwd max", "conv min", "conv/fwd", "fwd ratio", "secs"
    );
    let mut prev_fwd = f64::NAN;
    let t_all = std::time::Instant::now();
    for &per_seg in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536] {
        let t0 = std::time::Instant::now();
        let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut fwd_max = 0.0f64;
        let mut conv_min = f64::INFINITY;
        let mut points = 0usize;
        for _ in 0..N_PAIRS {
            let p0 = random_point(&mut state, DIM);
            let p1 = random_point(&mut state, DIM);
            let p2 = random_point(&mut state, DIM);

            let oab = resample(&[p0.clone(), p1.clone(), p0.clone()], per_seg);
            points = oab.len();
            fwd_max = fwd_max.max(deviation_from_constant(&oab));

            let tri = resample(&[p0.clone(), p1, p2, p0], per_seg);
            conv_min = conv_min.min(deviation_from_constant(&tri));
        }
        let ratio = if prev_fwd.is_nan() {
            f64::NAN
        } else {
            prev_fwd / fwd_max
        };
        println!(
            "{per_seg:>8} {points:>7} {fwd_max:>14.6e} {conv_min:>14.6e} {:>14.3e} {ratio:>10.2} {:>9.1}",
            conv_min / fwd_max,
            t0.elapsed().as_secs_f64()
        );
        prev_fwd = fwd_max;
    }
    println!("\ntotal {:.1}s", t_all.elapsed().as_secs_f64());
    println!("\nfwd ratio = previous fwd max / this one; 2.0 would be first order in per_seg.");
}
