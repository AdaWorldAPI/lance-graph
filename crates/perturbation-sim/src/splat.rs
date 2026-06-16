//! Gaussian-splat magnitude side of the pyramid (**PROTOTYPE**).
//!
//! The sign side of the Morton pyramid is Walsh/XOR ([`crate::sketch`]); this is
//! the **magnitude** side — the anisotropic-Gaussian footprint (EWA), the
//! `vsa_bundle` algebra. Two pieces:
//!
//! - [`splat_neighborhood`] — fits an SPD covariance `Σ` to a bus's local
//!   *electrical* neighborhood (the [`crate::basin::spectral_embedding`]
//!   coordinates, weighted by effective-resistance closeness). `Σ`'s anisotropy
//!   is the direction the perturbation spreads — the candidate cut normal. This
//!   splats in electrical coordinates, **not** geography (the trap).
//! - [`ewa_coarsen`] vs [`box_coarsen`] — coarsening a fine field one pyramid
//!   level. EWA weights each fine cell by an anisotropic Gaussian footprint, so
//!   spatially-distant cells joined only by a Morton "Z-jump" seam are
//!   down-weighted — removing the seam aliasing a hard box-average introduces.
//!
//! **Honesty (PROTOTYPE):** `Σ` is fit in a 2-D spectral embedding (the two
//! lowest non-trivial eigenvectors); the EWA `Σ` push-forward up the pyramid
//! (`J·Σ·Jᵀ`) is the certified `jc::ewa_sandwich` / ndarray pillar-12 form, not
//! re-derived here. This module shows the *construction*, not a tuned screen.

use crate::basin::{effective_resistance, laplacian_pinv, spectral_embedding};
use crate::graph::Grid;

/// An anisotropic Gaussian footprint in 2-D electrical-embedding coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Splat {
    /// Bus position in the embedding.
    pub center: [f64; 2],
    /// Covariance `Σ` (row-major 2×2, symmetric PSD).
    pub sigma: [f64; 4],
    /// `λ_max/λ_min` of `Σ` — how elongated the neighborhood is (`∞` if flat).
    pub anisotropy: f64,
}

/// Fit a [`Splat`] to bus `bus`: the resistance-closeness-weighted covariance of
/// its neighbours' offsets in the 2-D electrical embedding.
pub fn splat_neighborhood(grid: &Grid, alive: &[bool], bus: usize, rel_tol: f64) -> Splat {
    let n = grid.n;
    let emb = spectral_embedding(grid, alive, 2);
    let l_plus = laplacian_pinv(grid, alive, rel_tol);
    let center = [emb[bus][0], emb[bus][1]];

    let mut s = [0.0_f64; 4];
    let mut wsum = 0.0_f64;
    for (j, ej) in emb.iter().enumerate() {
        if j == bus {
            continue;
        }
        let r = effective_resistance(&l_plus, n, bus, j);
        if r > 0.0 {
            let w = (-r).exp(); // closeness: near buses weigh more
            let (dx, dy) = (ej[0] - center[0], ej[1] - center[1]);
            s[0] += w * dx * dx;
            s[1] += w * dx * dy;
            s[2] += w * dy * dx;
            s[3] += w * dy * dy;
            wsum += w;
        }
    }
    if wsum > 0.0 {
        for v in s.iter_mut() {
            *v /= wsum;
        }
    }

    // Anisotropy = λ_max/λ_min of the symmetric 2×2.
    let (a, b, d) = (s[0], s[1], s[3]);
    let tr = a + d;
    let det = a * d - b * b;
    let disc = (tr * tr / 4.0 - det).max(0.0).sqrt();
    let (l1, l2) = (tr / 2.0 + disc, tr / 2.0 - disc);
    let anisotropy = if l2 > 1e-12 { l1 / l2 } else { f64::INFINITY };

    Splat {
        center,
        sigma: s,
        anisotropy,
    }
}

/// Interleave two 16-bit coordinates into a 32-bit Morton (Z-order) code.
pub fn morton2(x: u16, y: u16) -> u32 {
    fn part(mut v: u32) -> u32 {
        v &= 0x0000_FFFF;
        v = (v | (v << 8)) & 0x00FF_00FF;
        v = (v | (v << 4)) & 0x0F0F_0F0F;
        v = (v | (v << 2)) & 0x3333_3333;
        v = (v | (v << 1)) & 0x5555_5555;
        v
    }
    part(x as u32) | (part(y as u32) << 1)
}

/// Plain box-average coarsen of a group of fine cells `(x, y, value)` → one
/// coarse cell `(centroid_x, centroid_y, mean_value)`.
pub fn box_coarsen(cells: &[(f64, f64, f64)]) -> (f64, f64, f64) {
    let n = cells.len().max(1) as f64;
    let (mut sx, mut sy, mut sv) = (0.0, 0.0, 0.0);
    for &(x, y, v) in cells {
        sx += x;
        sy += y;
        sv += v;
    }
    (sx / n, sy / n, sv / n)
}

/// EWA coarsen: value is an isotropic-Gaussian (width `sigma`) weighted mean
/// toward the centroid, so spatially-distant (Morton-seam) cells are
/// down-weighted. As `sigma → ∞` this converges to [`box_coarsen`].
pub fn ewa_coarsen(cells: &[(f64, f64, f64)], sigma: f64) -> (f64, f64, f64) {
    let n = cells.len().max(1) as f64;
    let (cx, cy) = {
        let (mut sx, mut sy) = (0.0, 0.0);
        for &(x, y, _) in cells {
            sx += x;
            sy += y;
        }
        (sx / n, sy / n)
    };
    let (mut wsum, mut vsum) = (0.0, 0.0);
    for &(x, y, v) in cells {
        let d2 = (x - cx).powi(2) + (y - cy).powi(2);
        let w = (-d2 / (2.0 * sigma * sigma)).exp();
        wsum += w;
        vsum += w * v;
    }
    (cx, cy, if wsum > 0.0 { vsum / wsum } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, Grid};

    fn triangle_pair_bridge() -> Grid {
        Grid::new(
            6,
            vec![
                Edge::new(0, 1, 1.0, 1e6),
                Edge::new(1, 2, 1.0, 1e6),
                Edge::new(2, 0, 1.0, 1e6),
                Edge::new(3, 4, 1.0, 1e6),
                Edge::new(4, 5, 1.0, 1e6),
                Edge::new(5, 3, 1.0, 1e6),
                Edge::new(2, 3, 1.0, 1e6),
            ],
        )
    }

    #[test]
    fn splat_sigma_is_symmetric_psd() {
        let g = triangle_pair_bridge();
        let alive = vec![true; g.edges.len()];
        let sp = splat_neighborhood(&g, &alive, 2, 1e-12);
        let s = sp.sigma;
        assert!((s[1] - s[2]).abs() < 1e-12, "Σ symmetric");
        let tr = s[0] + s[3];
        let det = s[0] * s[3] - s[1] * s[2];
        assert!(tr >= -1e-12 && det >= -1e-12, "Σ PSD (tr {tr}, det {det})");
        assert!(sp.anisotropy >= 1.0 - 1e-9);
    }

    #[test]
    fn morton2_known_codes() {
        assert_eq!(morton2(0, 0), 0);
        assert_eq!(morton2(1, 0), 1);
        assert_eq!(morton2(0, 1), 2);
        assert_eq!(morton2(1, 1), 3);
        assert_eq!(morton2(2, 0), 4);
        assert_eq!(morton2(3, 3), 15);
    }

    #[test]
    fn ewa_wide_sigma_converges_to_box() {
        let cells = [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 2.0),
            (0.0, 1.0, 3.0),
            (1.0, 1.0, 4.0),
        ];
        let b = box_coarsen(&cells);
        let e = ewa_coarsen(&cells, 1e6);
        assert!(
            (b.2 - e.2).abs() < 1e-6,
            "wide EWA == box: {} vs {}",
            b.2,
            e.2
        );
    }

    #[test]
    fn ewa_suppresses_a_morton_seam_outlier() {
        // A tight near-cluster + one far cell joined only by a Z-jump seam. EWA
        // (tight σ) down-weights the outlier; box averages it in.
        let cells = [(0.0, 0.0, 1.0), (0.1, 0.1, 1.0), (10.0, 10.0, 100.0)];
        let b = box_coarsen(&cells).2; // ≈ 34
        let e = ewa_coarsen(&cells, 0.5).2; // ≈ 1
        assert!(
            e < b,
            "EWA {e} should suppress the seam outlier below box {b}"
        );
        assert!(
            (e - 1.0).abs() < 0.5,
            "EWA should track the near cluster (~1), got {e}"
        );
    }
}
