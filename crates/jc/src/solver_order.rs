//! Solver-order advantage + carrier fidelity — the D-SK gates as a battery.
//!
//! # Why this is a NEW slot and not another "Pillar 11"
//!
//! Two batteries already carry that number across two repos (lance-graph's
//! `hambly_lyons` = uniqueness; ndarray's `hpc::pillar::signature` = truncated
//! kernel stability). This certifies a third, disjoint thing — how the Goursat
//! SOLVER behaves and what a coarse CARRIER may claim about fidelity — so it
//! takes its own name. Ruling Q1 of
//! `pillar11-signature-certification-unification-v1`: a new jc pillar slot.
//!
//! # What it certifies (census M-1, M-4)
//!
//! **(a) The area coefficient's advantage, in the regime the theory names.**
//! The shipped `signature_kernel_pde` uses increment-only cell coefficients —
//! first order. The rough higher-order solver's contribution is that cell
//! coefficients carrying level-2 iterated integrals converge better. Measured
//! (D-SK-B): with `c_ij = ⟨a_i, b_j⟩ + 2·A_i·B_j` the error falls by more than
//! 10x against increment-only in the SUPER-PERIOD regime — windows wide enough
//! to accumulate real loop area. Sub-period windows accumulate almost none, so
//! the advantage is regime-bound and gating it on a fixed absolute window range
//! is mis-posed (that mistake was made and falsified during D-SK-B; the regime
//! is W relative to the path's own oscillation period).
//!
//! **(b) The silence half.** On an area-free path — every micro increment
//! parallel to one direction, so every window's Lévy area is exactly zero —
//! the augmented scheme reduces IDENTICALLY to the increment-only one. The
//! improvement ratio must be exactly 1: an augmentation that "improves"
//! something with no area content is measuring noise, and this is the test
//! that would catch it.
//!
//! **(c) Carrier fidelity is gated in the AREA domain, never on the kernel
//! scalar.** This is the D-SK method finding promoted from prose to executable
//! law. Near the discretization floor the kernel scalar is
//! cancellation-dominated, so a COARSER carrier can post a smaller
//! kernel-scalar error than a finer one — an ordering that is real, measured,
//! and meaningless. The battery therefore gates fidelity on area-domain RMS
//! (which IS monotone in carrier resolution) and separately DEMONSTRATES the
//! trap, so it lives as a red test rather than a warning comment.

use crate::PillarResult;

#[cfg(feature = "hambly-lyons")]
mod active {
    use super::*;
    use sigker::signature_kernel_pde;
    use std::time::Instant;

    type P2 = [f64; 2];

    const M: usize = 2048;
    /// Super-period windows for these fixtures (period ~ 2π·M/ω micro-steps).
    const WINDOWS: [usize; 3] = [32, 48, 64];
    /// Area-register bit depths for the carrier-fidelity leg.
    const BITS: [u32; 4] = [3, 4, 6, 8];

    fn cross(a: P2, b: P2) -> f64 {
        a[0] * b[1] - a[1] * b[0]
    }

    /// Lévy area of a piecewise-linear path relative to its start point.
    fn levy_area(path: &[P2]) -> f64 {
        let p0 = path[0];
        let mut a = 0.0;
        for w in path.windows(2) {
            let r = [w[0][0] - p0[0], w[0][1] - p0[1]];
            let d = [w[1][0] - w[0][0], w[1][1] - w[0][1]];
            a += 0.5 * cross(r, d);
        }
        a
    }

    /// Coarse carrier: chord endpoints + per-window Lévy area.
    fn windows_of(micro: &[P2], w: usize) -> (Vec<P2>, Vec<f64>) {
        let mut pts = vec![micro[0]];
        let mut areas = Vec::new();
        let mut s = 0usize;
        while s + 1 < micro.len() {
            let e = (s + w).min(micro.len() - 1);
            areas.push(levy_area(&micro[s..=e]));
            pts.push(micro[e]);
            s = e;
        }
        (pts, areas)
    }

    /// First-order Goursat recursion with level-2-augmented cell coefficients:
    /// `c_ij = ⟨a_i, b_j⟩ + 2·A_i·B_j`. With all areas zero this is EXACTLY
    /// the shipped increment-only scheme — the identity the silence half uses.
    fn kernel_area_augmented(x: &[P2], ax: &[f64], y: &[P2], ay: &[f64]) -> f64 {
        let (n, m) = (x.len(), y.len());
        let mut k = vec![vec![1.0f64; m]; n];
        for i in 0..n - 1 {
            let dx = [x[i + 1][0] - x[i][0], x[i + 1][1] - x[i][1]];
            for j in 0..m - 1 {
                let dy = [y[j + 1][0] - y[j][0], y[j + 1][1] - y[j][1]];
                let c = dx[0] * dy[0] + dx[1] * dy[1] + 2.0 * ax[i] * ay[j];
                k[i + 1][j + 1] = k[i + 1][j] + k[i][j + 1] - k[i][j] + c * k[i][j];
            }
        }
        k[n - 1][m - 1]
    }

    /// Drift + high-frequency rotation: real loop area in every window.
    fn oscillatory_path(m: usize, omega: f64, amp: f64, phase: f64) -> Vec<P2> {
        (0..=m)
            .map(|i| {
                let t = i as f64 / m as f64;
                [
                    t + amp * (omega * t + phase).cos(),
                    0.5 * t + amp * (omega * t + phase).sin(),
                ]
            })
            .collect()
    }

    /// Every micro increment parallel to one direction: speed wobbles, every
    /// window's Lévy area is exactly zero. The silence fixture.
    fn area_free_path(m: usize, omega: f64, amp: f64) -> Vec<P2> {
        let u = [0.8, 0.6];
        (0..=m)
            .map(|i| {
                let t = i as f64 / m as f64;
                let s = t + amp * (omega * t).sin() / omega;
                [s * u[0], s * u[1]]
            })
            .collect()
    }

    fn to_vecs(p: &[P2]) -> Vec<Vec<f64>> {
        p.iter().map(|x| x.to_vec()).collect()
    }
    fn rel_err(k: f64, k_ref: f64) -> f64 {
        (k - k_ref).abs() / k_ref.abs().max(1e-12)
    }

    /// Mid-tread quantizer over the symmetric range `±peak`, `bits` wide.
    fn quantize(areas: &[f64], bits: u32, peak: f64) -> Vec<f64> {
        let levels = (1i64 << (bits - 1)) - 1;
        let step = peak / levels as f64;
        areas
            .iter()
            .map(|a| {
                let q = (a / step).round().clamp(-levels as f64, levels as f64);
                q * step
            })
            .collect()
    }

    fn rms(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / n).sqrt()
    }

    /// The (a)+(b) legs: best super-period improvement on each oscillatory
    /// fixture, and the improvement ratio on the area-free fixture.
    fn advantage_leg() -> (f64, f64, f64) {
        let pairs = [
            (
                oscillatory_path(M, 260.0, 0.045, 0.3),
                oscillatory_path(M, 340.0, 0.038, 1.9),
            ),
            (
                oscillatory_path(M, 180.0, 0.060, 2.4),
                oscillatory_path(M, 460.0, 0.030, 0.9),
            ),
        ];
        let mut best = [0.0f64; 2];
        for (idx, (x, y)) in pairs.iter().enumerate() {
            let kr = signature_kernel_pde(&to_vecs(x), &to_vecs(y));
            for &w in &WINDOWS {
                let (xp, xa) = windows_of(x, w);
                let (yp, ya) = windows_of(y, w);
                let (zx, zy) = (vec![0.0; xa.len()], vec![0.0; ya.len()]);
                let ea = rel_err(kernel_area_augmented(&xp, &zx, &yp, &zy), kr);
                let eb = rel_err(kernel_area_augmented(&xp, &xa, &yp, &ya), kr);
                best[idx] = best[idx].max(ea / eb.max(1e-300));
            }
        }

        // Silence: area-free fixture, same machinery.
        let (ax, ay) = (
            area_free_path(M, 260.0, 0.045),
            area_free_path(M, 340.0, 0.038),
        );
        let kr = signature_kernel_pde(&to_vecs(&ax), &to_vecs(&ay));
        let mut silence = 0.0f64;
        for &w in &WINDOWS {
            let (xp, xa) = windows_of(&ax, w);
            let (yp, ya) = windows_of(&ay, w);
            let (zx, zy) = (vec![0.0; xa.len()], vec![0.0; ya.len()]);
            let ea = rel_err(kernel_area_augmented(&xp, &zx, &yp, &zy), kr);
            let eb = rel_err(kernel_area_augmented(&xp, &xa, &yp, &ya), kr);
            silence = silence.max((ea / eb.max(1e-300) - 1.0).abs());
        }
        (best[0], best[1], silence)
    }

    /// The (c) leg: per-bit-depth area-domain RMS and kernel-scalar error.
    fn carrier_leg() -> Vec<(u32, f64, f64)> {
        let x = oscillatory_path(M, 260.0, 0.045, 0.3);
        let y = oscillatory_path(M, 340.0, 0.038, 1.9);
        let kr = signature_kernel_pde(&to_vecs(&x), &to_vecs(&y));
        let w = 64usize;
        let (xp, xa) = windows_of(&x, w);
        let (yp, ya) = windows_of(&y, w);
        let peak = xa
            .iter()
            .chain(&ya)
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(1e-300);
        BITS.iter()
            .map(|&b| {
                let qx = quantize(&xa, b, peak);
                let qy = quantize(&ya, b, peak);
                let area_rms = rms(&qx, &xa).max(rms(&qy, &ya));
                let kerr = rel_err(kernel_area_augmented(&xp, &qx, &yp, &qy), kr);
                (b, area_rms, kerr)
            })
            .collect()
    }

    pub fn prove() -> PillarResult {
        let t0 = Instant::now();
        let (adv1, adv2, silence) = advantage_leg();
        let carrier = carrier_leg();

        // (c1) area-domain RMS must be STRICTLY monotone in bit depth.
        let area_monotone = carrier.windows(2).all(|p| p[1].1 < p[0].1);
        // (c2) the trap must be DEMONSTRABLE: somewhere in the same sweep a
        // coarser carrier posts a smaller kernel-scalar error than a finer one.
        let trap_fires = carrier.windows(2).any(|p| p[1].2 > p[0].2);

        let pass = adv1 > 10.0 && adv2 > 10.0 && silence < 1e-12 && area_monotone && trap_fires;

        let table = carrier
            .iter()
            .map(|(b, r, k)| format!("{b}b: areaRMS {r:.3e} kerr {k:.3e}"))
            .collect::<Vec<_>>()
            .join("; ");

        let detail = format!(
            "M={M}, super-period windows {WINDOWS:?}. \
             (a) Best area-coefficient improvement over increment-only: \
             fixture 1 = {adv1:.1}x, fixture 2 = {adv2:.1}x (pass if > 10x each). \
             (b) Silence: on the area-free fixture every window area is exactly \
             zero, so the augmented scheme must reduce IDENTICALLY to \
             increment-only — max |ratio − 1| = {silence:.2e} (pass if < 1e-12). \
             (c) Carrier fidelity [{table}]: area-domain RMS strictly monotone \
             in bit depth = {area_monotone}; kernel-scalar ordering violated \
             somewhere in the SAME sweep = {trap_fires} — that violation is the \
             point. It is why fidelity is gated in the area domain and never on \
             the kernel scalar, and it is asserted here so the trap is a red \
             test rather than a comment.",
        );

        PillarResult {
            name: "Solver-order advantage + carrier fidelity (rough-path signature kernels)",
            pass,
            measured: adv1.min(adv2),
            predicted: 10.0,
            detail,
            runtime_ms: t0.elapsed().as_millis() as u64,
        }
    }

    /// Printed tables for the pre-registration sweep example.
    pub fn probe_tables() {
        let (adv1, adv2, silence) = advantage_leg();
        println!("(a) best super-period improvement: fixture1 {adv1:.2}x  fixture2 {adv2:.2}x");
        println!("(b) area-free silence |ratio-1| max: {silence:.3e}");
        println!("(c) {:>5} {:>14} {:>14}", "bits", "area RMS", "kernel err");
        for (b, r, k) in carrier_leg() {
            println!("    {b:>5} {r:>14.6e} {k:>14.6e}");
        }
    }
}

#[cfg(feature = "hambly-lyons")]
pub fn prove() -> PillarResult {
    active::prove()
}

/// Printed tables for `examples/w3_battery_sweep.rs`.
#[cfg(feature = "hambly-lyons")]
pub fn probe_tables() {
    active::probe_tables()
}

#[cfg(not(feature = "hambly-lyons"))]
pub fn prove() -> PillarResult {
    PillarResult::deferred(
        "Solver-order advantage + carrier fidelity (rough-path signature kernels)",
        "build with --features hambly-lyons to activate (pulls in the sigker \
         workspace sibling). Default JC build stays zero-dep per the \
         standalone-crate constitution.",
    )
}

#[cfg(all(test, feature = "hambly-lyons"))]
mod tests {
    use super::*;

    #[test]
    fn pillar_passes() {
        let r = prove();
        assert!(r.pass, "solver-order battery must pass: {}", r.detail);
    }
}
