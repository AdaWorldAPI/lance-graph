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
//! `CovHighD` is `f32` and dense `O(N³)`; this crate is `f64`. Values are
//! narrowed to `f32` once on the way in and widened on the way out, so expect
//! roughly `1e-6` relative error against an `f64` triple product. `N` is a
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
/// If `eig.n != N`, if `sigma_p.len() != N*N`, or if `sigma_p` is not
/// symmetric within [`SYMMETRY_TOL`] (relative to its largest entry).
pub fn angle_covariance<const N: usize>(eig: &Eigen, sigma_p: &[f64], rel_tol: f64) -> Vec<f64> {
    assert_eq!(
        eig.n, N,
        "decomposition has {} buses, CovHighD<N> has N = {N}",
        eig.n
    );
    assert_eq!(sigma_p.len(), N * N, "sigma_p must be N*N");
    let scale = sigma_p
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
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
    let m = CovHighD::<N>::from_symmetric_fn(|i, j| l_plus[i * N + j] as f32);
    let s = CovHighD::<N>::from_symmetric_fn(|i, j| sigma_p[i * N + j] as f32);
    let out = s.sandwich(&m);

    let mut dense = vec![0.0_f64; N * N];
    for i in 0..N {
        for j in 0..N {
            dense[i * N + j] = out.get(i, j) as f64;
        }
    }
    dense
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

    #[test]
    fn matches_f64_dense_triple_product() {
        let eig = symmetric_eigen(&grid().laplacian(), N);
        let sp = sigma_p();
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
