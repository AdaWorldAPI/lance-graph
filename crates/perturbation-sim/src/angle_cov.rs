//! Injection-uncertainty push-forward: `Σθ = L⁺ · Σp · L⁺`.
//!
//! The stochastic twin of [`crate::eigen::Eigen::pseudo_apply`]. For a fixed
//! topology the DC model `θ = L⁺ p` is linear, so a nodal injection covariance
//! `Σp` maps to the angle covariance `Σθ = L⁺ Σp (L⁺)ᵀ` **exactly**, not as a
//! first-order approximation. `L⁺` is symmetric, so this is the sandwich
//! `M·Σ·M` with `M = L⁺`, and it is computed by ndarray's certified
//! [`CovHighD::sandwich`] (Pillar-9) rather than by a second kernel here.
//!
//! # What this does NOT cover
//!
//! - **A line trip.** That changes `L`, so it is a rank-1 update of `L⁺`
//!   (Sherman–Morrison / LODF), not a sandwich of the old covariance. Push a
//!   post-contingency covariance by re-decomposing the post-trip Laplacian.
//! - **Line-flow covariance.** `f = B·A·θ` has a rectangular, non-symmetric
//!   Jacobian, and `sandwich` assumes a symmetric `M`. Per-line variance is a
//!   2-sparse quadratic form over `Σθ` and does not need a sandwich at all.
//!
//! # Precision and size
//!
//! `CovHighD` is `f32` and dense `O(N³)`; this crate is `f64`. Each input is
//! divided by its largest absolute entry before it is narrowed to `f32`, and
//! the result is multiplied back in `f64` (the sandwich is bilinear, so this
//! is exact up to rounding). A uniformly huge or tiny `Σp` is therefore as
//! accurate as one near `1`.
//!
//! The error bound is relative to the INPUT magnitude `max|Σp| · max|L⁺|²`,
//! not to the output: expect roughly `1e-6` of that. When most of `Σp`'s mass
//! lies in `L⁺`'s null space (e.g. a large variance on an isolated bus), the
//! output is small next to that bound and its relative error is large. A
//! `Σp` whose dynamic range exceeds `f32`'s — a nonzero entry that would
//! flush below `f32`'s smallest normal after the division — is refused rather
//! than silently zeroed. Both limits are `f32` limits of `CovHighD`; lifting
//! them is an `f64` sandwich in ndarray, not a second kernel here. `N` is a
//! compile-time constant in `CovHighD`, while a [`crate::graph::Grid`]'s bus
//! count is a runtime value: the caller picks `N` and the call panics when the
//! decomposition disagrees with it. A runtime-sized sandwich would be a change
//! to ndarray, not something to route around here.

use ndarray::hpc::pillar::cov_high_d::CovHighD;

use crate::eigen::Eigen;

/// Relative asymmetry tolerated in `Σp` before it is rejected.
///
/// `CovHighD::from_symmetric_fn` reads only the lower triangle, so an
/// asymmetric input would be silently replaced by its lower half. Refusing is
/// the honest alternative to that silent substitution.
pub const SYMMETRY_TOL: f64 = 1e-9;

/// Angle covariance `Σθ = L⁺ Σp L⁺` for a nodal injection covariance `Σp`.
///
/// `eig` is the decomposition of the Laplacian (see
/// [`crate::eigen::symmetric_eigen`]); `rel_tol` is the null-space cutoff
/// passed to [`Eigen::pseudo_inverse`], identical to the one `pseudo_apply`
/// uses. `sigma_p` and the result are row-major `N×N`.
///
/// # Panics
/// If `eig.n != N`, if `sigma_p.len() != N*N`, if any entry of `sigma_p` is
/// not finite, if `sigma_p` is not symmetric within [`SYMMETRY_TOL`]
/// (relative to its largest entry), or if a nonzero entry of `sigma_p` is
/// below `f32::MIN_POSITIVE` relative to its largest entry (its dynamic range
/// does not fit `f32`, so that entry would silently become zero).
pub fn angle_covariance<const N: usize>(eig: &Eigen, sigma_p: &[f64], rel_tol: f64) -> Vec<f64> {
    assert_eq!(
        eig.n, N,
        "decomposition has {} buses, CovHighD<N> has N = {N}",
        eig.n
    );
    assert_eq!(sigma_p.len(), N * N, "sigma_p must be N*N");
    assert!(
        sigma_p.iter().all(|v| v.is_finite()),
        "sigma_p has a non-finite entry"
    );
    let scale = sigma_p
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    if let Some(v) = sigma_p
        .iter()
        .find(|v| **v != 0.0 && (*v / scale).abs() < f64::from(f32::MIN_POSITIVE))
    {
        panic!(
            "sigma_p dynamic range exceeds f32: entry {v:e} next to max {scale:e} \
             would be zeroed by the f32 sandwich"
        );
    }
    for i in 0..N {
        for j in 0..i {
            let d = (sigma_p[i * N + j] - sigma_p[j * N + i]).abs();
            assert!(
                d <= SYMMETRY_TOL * scale,
                "sigma_p is not symmetric at ({i},{j}): off by {d}"
            );
        }
    }

    let l_plus = eig.pseudo_inverse(rel_tol);
    let l_scale = l_plus
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    // Both inputs land in [-1, 1] before narrowing, so no entry overflows to
    // infinity or flushes to zero merely because of its magnitude; the f32
    // sandwich then sums at most N² products bounded by 1.
    let m = CovHighD::<N>::from_symmetric_fn(|i, j| (l_plus[i * N + j] / l_scale) as f32);
    let s = CovHighD::<N>::from_symmetric_fn(|i, j| (sigma_p[i * N + j] / scale) as f32);
    let out = s.sandwich(&m);

    let mut dense = vec![0.0_f64; N * N];
    for i in 0..N {
        for j in 0..N {
            dense[i * N + j] = unscale(out.get(i, j), scale, l_scale);
        }
    }
    dense
}

/// One sandwich entry back to `f64` units: `o · scale · l_scale²`.
///
/// Applied factor by factor on the entry, never through a precomputed
/// `scale · l_scale²`: that product can overflow to infinity while the entry
/// itself does not, and an exact-zero entry would then become `0 · ∞ = NaN`.
fn unscale(o: f32, scale: f64, l_scale: f64) -> f64 {
    f64::from(o) * scale * l_scale * l_scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eigen::symmetric_eigen;
    use crate::graph::{Edge, Grid};

    const N: usize = 6;
    const TOL: f64 = 1e-9;

    /// A 6-bus ring with two chords and unequal susceptances.
    fn grid() -> Grid {
        Grid::new(
            N,
            vec![
                Edge::new(0, 1, 4.0, 10.0),
                Edge::new(1, 2, 2.5, 10.0),
                Edge::new(2, 3, 3.0, 10.0),
                Edge::new(3, 4, 1.5, 10.0),
                Edge::new(4, 5, 5.0, 10.0),
                Edge::new(5, 0, 2.0, 10.0),
                Edge::new(0, 3, 1.0, 10.0),
                Edge::new(1, 4, 0.7, 10.0),
            ],
        )
    }

    fn rel_err(a: &[f64], b: &[f64]) -> f64 {
        let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
            / scale
    }

    /// `Σp = A Aᵀ` for a fixed non-trivial `A` — SPD and genuinely coupled.
    fn sigma_p() -> Vec<f64> {
        let a: Vec<f64> = (0..N * N)
            .map(|k| ((k * 7 + 3) % 11) as f64 / 11.0 - 0.4)
            .collect();
        let mut s = vec![0.0; N * N];
        for i in 0..N {
            for j in 0..N {
                s[i * N + j] = (0..N).map(|k| a[i * N + k] * a[j * N + k]).sum();
            }
        }
        s
    }

    #[test]
    fn rank_one_injection_matches_pseudo_apply_outer_product() {
        // Σp = p pᵀ  ⇒  Σθ = (L⁺p)(L⁺p)ᵀ. The right-hand side comes from the
        // eigen-coefficient path `pseudo_apply`, which never forms L⁺ — so this
        // cross-checks the dense L⁺ AND the sandwich against an independent route.
        let eig = symmetric_eigen(&grid().laplacian(), N);
        let p = [1.2, -0.4, 0.9, -1.5, 0.3, -0.5]; // balanced: sums to 0
        let sp: Vec<f64> = (0..N * N).map(|k| p[k / N] * p[k % N]).collect();
        let theta = eig.pseudo_apply(&p, TOL);
        let want: Vec<f64> = (0..N * N).map(|k| theta[k / N] * theta[k % N]).collect();
        let got = angle_covariance::<N>(&eig, &sp, TOL);
        let e = rel_err(&got, &want);
        eprintln!("rank-1 push-forward rel err {e:e}");
        assert!(e < 1e-5, "rank-1 push-forward rel err {e:e}");
        assert!(
            want.iter().any(|v| v.abs() > 1e-3),
            "fixture is vacuous: θ is ~0"
        );
    }

    /// `L⁺ Σp L⁺` in plain `f64`, the reference every test compares against.
    fn triple_product(eig: &Eigen, sp: &[f64]) -> Vec<f64> {
        let l = eig.pseudo_inverse(TOL);
        let mut want = vec![0.0; N * N];
        for i in 0..N {
            for m in 0..N {
                let mut acc = 0.0;
                for j in 0..N {
                    for k in 0..N {
                        acc += l[i * N + j] * sp[j * N + k] * l[k * N + m];
                    }
                }
                want[i * N + m] = acc;
            }
        }
        want
    }

    #[test]
    fn matches_f64_dense_triple_product() {
        let eig = symmetric_eigen(&grid().laplacian(), N);
        let sp = sigma_p();
        let want = triple_product(&eig, &sp);
        let got = angle_covariance::<N>(&eig, &sp, TOL);
        let e = rel_err(&got, &want);
        eprintln!("sandwich vs f64 triple product rel err {e:e}");
        assert!(e < 1e-5, "sandwich vs f64 triple product rel err {e:e}");
        // Non-trivial: off-diagonal coupling must be present, else a diagonal-only
        // implementation would pass.
        assert!(
            want[1].abs() > 1e-3 * want[0].abs(),
            "fixture has no off-diagonal mass"
        );
    }

    #[test]
    fn matches_monte_carlo_of_the_deterministic_solver() {
        // Draw p ~ N(0, Σp) via p = A z, solve θ = L⁺p with pseudo_apply, and
        // compare the empirical covariance to the sandwich. This checks the
        // STATISTICAL claim (push-forward of a distribution), not just algebra.
        let eig = symmetric_eigen(&grid().laplacian(), N);
        let a: Vec<f64> = (0..N * N)
            .map(|k| ((k * 7 + 3) % 11) as f64 / 11.0 - 0.4)
            .collect();
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut unif = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let samples = 40_000;
        let mut emp = vec![0.0; N * N];
        for _ in 0..samples {
            let z: Vec<f64> = (0..N)
                .map(|_| {
                    let (u1, u2) = (unif().max(1e-300), unif());
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                })
                .collect();
            let p: Vec<f64> = (0..N)
                .map(|i| (0..N).map(|k| a[i * N + k] * z[k]).sum())
                .collect();
            let th = eig.pseudo_apply(&p, TOL);
            for i in 0..N {
                for j in 0..N {
                    emp[i * N + j] += th[i] * th[j];
                }
            }
        }
        emp.iter_mut().for_each(|v| *v /= samples as f64);
        let got = angle_covariance::<N>(&eig, &sigma_p(), TOL);
        let e = rel_err(&emp, &got);
        // Sampling error of a second moment at n = 40k is ~1/√n·√2 ≈ 0.7%.
        eprintln!("Monte-Carlo vs sandwich rel err {e:.4}");
        assert!(e < 0.03, "Monte-Carlo vs sandwich rel err {e:.4}");
    }

    /// FAILS IF: an input is narrowed to `f32` at its raw magnitude, so a
    /// `Σp` past `f32::MAX` becomes infinity (or one below the smallest
    /// normal flushes to zero) although the `f64` product is finite.
    #[test]
    fn magnitude_outside_f32_range_is_exact_up_to_rounding() {
        let eig = symmetric_eigen(&grid().laplacian(), N);
        for k in [1e45_f64, 1e-45] {
            let sp: Vec<f64> = sigma_p().iter().map(|v| v * k).collect();
            assert!(
                sp.iter().any(|v| (*v as f32).is_infinite())
                    || sp
                        .iter()
                        .all(|v| *v == 0.0 || (*v as f32).abs() < f32::MIN_POSITIVE),
                "fixture must leave f32's normal range at k = {k:e}"
            );
            let want = triple_product(&eig, &sp);
            let got = angle_covariance::<N>(&eig, &sp, TOL);
            assert!(got.iter().all(|v| v.is_finite()), "k = {k:e}: non-finite");
            let e = rel_err(&got, &want);
            assert!(e < 1e-5, "k = {k:e}: rel err {e:e}");
        }
    }

    /// FAILS IF: the result is rescaled through the combined factor
    /// `scale · l_scale²`, which overflows here and turns an exact zero into
    /// NaN.
    #[test]
    fn an_exact_zero_stays_zero_when_the_combined_factor_overflows() {
        let (scale, l_scale) = (1e300_f64, 1e10_f64);
        assert!(
            (scale * l_scale * l_scale).is_infinite(),
            "fixture must overflow"
        );
        assert_eq!(unscale(0.0, scale, l_scale), 0.0);
        // A nonzero entry whose true value is finite stays finite.
        let v = unscale(1e-20, scale, l_scale);
        let want = f64::from(1e-20_f32) * 1e300 * 1e20;
        assert!(
            v.is_finite() && (v - want).abs() <= 1e-12 * want.abs(),
            "{v:e} vs {want:e}"
        );
    }

    /// FAILS IF: an entry too small to survive the f32 narrowing is silently
    /// zeroed instead of refused (a huge variance elsewhere, e.g. on a bus in
    /// `L⁺`'s null space, sets the scale).
    #[test]
    #[should_panic(expected = "dynamic range exceeds f32")]
    fn refuses_a_dynamic_range_wider_than_f32() {
        let eig = symmetric_eigen(&grid().laplacian(), N);
        let mut sp = vec![0.0; N * N];
        sp[0] = 1e100;
        for k in 1..N {
            sp[k * N + k] = 1.0;
        }
        let _ = angle_covariance::<N>(&eig, &sp, TOL);
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn refuses_a_non_finite_injection_covariance() {
        let eig = symmetric_eigen(&grid().laplacian(), N);
        let mut sp = sigma_p();
        sp[0] = f64::INFINITY;
        let _ = angle_covariance::<N>(&eig, &sp, TOL);
    }

    #[test]
    #[should_panic(expected = "CovHighD<N> has N")]
    fn refuses_a_const_n_that_disagrees_with_the_grid() {
        let eig = symmetric_eigen(&grid().laplacian(), N);
        let _ = angle_covariance::<4>(&eig, &[0.0; 16], TOL);
    }

    #[test]
    #[should_panic(expected = "not symmetric")]
    fn refuses_an_asymmetric_injection_covariance() {
        let eig = symmetric_eigen(&grid().laplacian(), N);
        let mut sp = sigma_p();
        sp[N] += 0.5; // (1,0) no longer equals (0,1)
        let _ = angle_covariance::<N>(&eig, &sp, TOL);
    }
}
