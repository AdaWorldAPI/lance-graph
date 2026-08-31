//! D-SK-A probe — how much of the Lévy-area coefficient gain survives
//! quantization when window-area magnitudes are HETEROGENEOUS?
//!
//! # Question (pre-registered)
//!
//! D-SK-B′ (`probe_dsk_b_levy_window.rs`, board entry
//! E-LEVY-AREA-COEFFICIENT-BEATS-REFINEMENT-1) measured that sign-only area
//! carriers — the 24×i4 register's orientation-bit analog — retain 99.3% of
//! the area-coefficient gain, but on a fixture whose |window areas| were
//! near-constant (std/mean 0.118), where sign × global-mean-magnitude is
//! nearly exact BY FIXTURE SHAPE. D-SK-A asks the question that caveat left
//! open: as the |area| spread grows, how do the quantized carriers degrade?
//!
//! - sign-only × global mean magnitude — the ORIENTATION-BIT analog
//!   (1 signed bit per window, magnitude from a global codebook scale);
//! - i4 — 16 signed levels, half-steps of the global scale, clamped ±4×scale
//!   — the register's NIBBLE analog;
//! - exact f64 areas — the ceiling (Scheme B of D-SK-B′).
//!
//! # Fixture families (heterogeneity knob β)
//!
//! - AM: amplitude-modulated rotation, amp(t) = a₀·(1 + β·sin(2π·3t)).
//!   Window area ∝ amp², so β sweeps the |area| spread from near-constant
//!   (β=0, the D-SK-B′ shape) to strongly heterogeneous (β=1, amp touches 0).
//! - CHIRP: frequency sweep ω(t) = ω₀·(1 + β·t) at constant amplitude —
//!   heterogeneity through loop DENSITY instead of loop size, a second,
//!   mechanically different route to spread.
//!
//! W = 64 throughout (super-period for every ω used; D-SK-B′'s regime
//! finding: the coefficient pays at/above the oscillation period).
//!
//! # Pre-registered gates (validity hard, outcome reported either way)
//!
//! - G0 (sanity, STOP on fail): shipped solver vs closed form, rel err < 2e-2.
//! - G1 (knob inertness): AM spread(β) is strictly increasing in β and
//!   spread(β_max) > 0.5 — the knob genuinely reaches the heterogeneous
//!   regime the caveat named. A knob that moves nothing is decoration.
//! - G2 (carrier fidelity ordering, RE-REGISTERED): at every β and both
//!   families, the area-domain RMS quantization error satisfies
//!   rms(i4) ≤ rms(sign) + ε — the nibble is never a worse carrier than
//!   the bit. The FIRST formulation gated the kernel-scalar errors
//!   (err_exact ≤ 1.2·err_i4 ≤ 1.44·err_sign) and was falsified on the
//!   first run: near the discretization floor the kernel error is
//!   CANCELLATION-DOMINATED — a coarser carrier can land closer to K_ref
//!   by luck (measured: AM β=0.25 i4 kernel-err 2.654e-2 vs sign 2.014e-2;
//!   CHIRP β=1.5 i4 1.282e-2 "beating" exact 2.359e-2). Kernel-scalar
//!   ordering is therefore NOT a faithful readout of carrier fidelity and
//!   is printed, never gated. That mis-posedness is itself a banked
//!   finding: retention percentages below carry ±cancellation jitter of
//!   the same order as the gap between carriers near the floor.
//! - G3 (baseline reproduction): at β = 0 (AM), sign-only retention > 90% —
//!   reproduces the D-SK-B′ figure on this probe's own fixture.
//! - Decision rule, pre-registered BEFORE the run: if sign-only retention
//!   stays > 90% at every measured spread, the verdict is "the orientation
//!   bit suffices across the spread range measured"; if it falls below 50%
//!   at some spread, the verdict is "a magnitude tier is required from that
//!   spread upward" and the crossing spread is the banked number; between
//!   50% and 90% the verdict is "partial — magnitude tier recommended".
//!   Same rule applied to i4 separately. Either direction is a finding.
//!
//! Run: `cargo run --manifest-path crates/sigker/Cargo.toml --example probe_dsk_a_quantized_area`

use sigker::{linear_path_kernel_closed_form, signature_kernel_pde};

type P2 = [f64; 2];

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

/// Coarse windows: chord endpoints + per-window Lévy area.
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

/// First-order Goursat recursion with level-2-augmented cell coefficients
/// (identical scheme to D-SK-B′): c_ij = ⟨a_i, b_j⟩ + 2·A_i·B_j.
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

/// AM fixture: drift + rotation with slowly amplitude-modulated envelope.
/// β=0 reproduces the near-constant-|area| D-SK-B′ shape; β=1 drives the
/// envelope through zero — strongly heterogeneous window areas.
fn am_path(m: usize, omega: f64, a0: f64, beta: f64, phase: f64) -> Vec<P2> {
    (0..=m)
        .map(|i| {
            let t = i as f64 / m as f64;
            let amp = a0 * (1.0 + beta * (2.0 * std::f64::consts::PI * 3.0 * t).sin());
            [
                t + amp * (omega * t + phase).cos(),
                0.5 * t + amp * (omega * t + phase).sin(),
            ]
        })
        .collect()
}

/// CHIRP fixture: constant amplitude, frequency swept ω₀·(1 + β·t) —
/// heterogeneity through loop density rather than loop size.
fn chirp_path(m: usize, omega0: f64, a0: f64, beta: f64, phase: f64) -> Vec<P2> {
    (0..=m)
        .map(|i| {
            let t = i as f64 / m as f64;
            // Instantaneous phase = ∫ω dt = ω₀(t + β·t²/2).
            let ph = omega0 * (t + beta * t * t / 2.0) + phase;
            [t + a0 * ph.cos(), 0.5 * t + a0 * ph.sin()]
        })
        .collect()
}

fn to_vecs(p: &[P2]) -> Vec<Vec<f64>> {
    p.iter().map(|x| x.to_vec()).collect()
}

fn rel_err(k: f64, k_ref: f64) -> f64 {
    (k - k_ref).abs() / k_ref.abs().max(1e-12)
}

/// std/mean of |window areas| across both paths — the heterogeneity measure.
fn spread(ax: &[f64], ay: &[f64]) -> f64 {
    let all: Vec<f64> = ax.iter().chain(ay).map(|a| a.abs()).collect();
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let var = all.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / all.len() as f64;
    var.sqrt() / mean.max(1e-300)
}

struct Row {
    beta: f64,
    sp: f64,
    err_a: f64,
    err_exact: f64,
    err_i4: f64,
    err_sign: f64,
    ret_i4: f64,
    ret_sign: f64,
    rms_i4: f64,
    rms_sign: f64,
}

/// Area-domain RMS quantization error of a carrier against the exact areas.
fn rms_q(exact: &[f64], q: &[f64]) -> f64 {
    let s: f64 = exact.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
    (s / exact.len() as f64).sqrt()
}

/// One measurement: fixture pair at heterogeneity β, W=64.
fn measure(x: &[P2], y: &[P2], beta: f64) -> Row {
    let w = 64usize;
    let k_ref = signature_kernel_pde(&to_vecs(x), &to_vecs(y));
    let (xp, xa) = windows_of(x, w);
    let (yp, ya) = windows_of(y, w);
    let sp = spread(&xa, &ya);
    let scale = {
        let s: f64 = xa.iter().chain(&ya).map(|a| a.abs()).sum();
        s / (xa.len() + ya.len()) as f64
    };
    let sign_q = |a: &[f64]| -> Vec<f64> { a.iter().map(|v| v.signum() * scale).collect() };
    let i4_q = |a: &[f64]| -> Vec<f64> {
        a.iter()
            .map(|v| ((v / scale).clamp(-4.0, 4.0) * 2.0).round() / 2.0 * scale)
            .collect()
    };
    let (zx, zy) = (vec![0.0; xa.len()], vec![0.0; ya.len()]);
    let err_a = rel_err(kernel_area_augmented(&xp, &zx, &yp, &zy), k_ref);
    let err_exact = rel_err(kernel_area_augmented(&xp, &xa, &yp, &ya), k_ref);
    let err_i4 = rel_err(
        kernel_area_augmented(&xp, &i4_q(&xa), &yp, &i4_q(&ya)),
        k_ref,
    );
    let err_sign = rel_err(
        kernel_area_augmented(&xp, &sign_q(&xa), &yp, &sign_q(&ya)),
        k_ref,
    );
    let recov = |e: f64| ((err_a - e) / (err_a - err_exact).max(1e-300)).clamp(-1.0, 1.0);
    let all_exact: Vec<f64> = xa.iter().chain(&ya).cloned().collect();
    let all_i4: Vec<f64> = i4_q(&xa).into_iter().chain(i4_q(&ya)).collect();
    let all_sign: Vec<f64> = sign_q(&xa).into_iter().chain(sign_q(&ya)).collect();
    Row {
        beta,
        sp,
        err_a,
        err_exact,
        err_i4,
        err_sign,
        ret_i4: recov(err_i4),
        ret_sign: recov(err_sign),
        rms_i4: rms_q(&all_exact, &all_i4),
        rms_sign: rms_q(&all_exact, &all_sign),
    }
}

fn main() {
    // ── G0: solver sanity ────────────────────────────────────────────────
    println!("== 0. G0 sanity: shipped PDE solver vs I0 closed form ==");
    let n0 = 256;
    let (u, v) = ([0.6, 0.3, -0.2], [0.5, -0.1, 0.4]);
    let lin = |dir: &[f64]| -> Vec<Vec<f64>> {
        (0..=n0)
            .map(|i| dir.iter().map(|d| d * i as f64 / n0 as f64).collect())
            .collect()
    };
    let g0 = rel_err(
        signature_kernel_pde(&lin(&u), &lin(&v)),
        linear_path_kernel_closed_form(&u, &v),
    );
    println!("   rel err {g0:.2e}");
    assert!(
        g0 < 2e-2,
        "G0 FAIL: solver untrustworthy as reference — STOP"
    );

    let m = 2048usize;
    let betas = [0.0, 0.25, 0.5, 0.75, 1.0];

    // ── AM family ────────────────────────────────────────────────────────
    println!("\n== 1. AM family (amplitude-modulated |areas|), W=64 ==");
    println!(
        "   beta | spread | err incr-only | err exact | err i4 | err sign | ret i4 | ret sign"
    );
    let am_rows: Vec<Row> = betas
        .iter()
        .map(|&b| {
            let x = am_path(m, 260.0, 0.045, b, 0.3);
            let y = am_path(m, 340.0, 0.038, b, 1.9);
            let r = measure(&x, &y, b);
            println!(
                "   {:.2} | {:.3} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:5.1}% | {:5.1}%",
                r.beta,
                r.sp,
                r.err_a,
                r.err_exact,
                r.err_i4,
                r.err_sign,
                100.0 * r.ret_i4,
                100.0 * r.ret_sign
            );
            r
        })
        .collect();

    // G1: the heterogeneity knob works.
    for wpair in am_rows.windows(2) {
        assert!(
            wpair[1].sp > wpair[0].sp,
            "G1 FAIL: spread not strictly increasing in beta ({:.3} !> {:.3})",
            wpair[1].sp,
            wpair[0].sp
        );
    }
    let sp_max = am_rows.last().unwrap().sp;
    assert!(
        sp_max > 0.5,
        "G1 FAIL: max spread {sp_max:.3} !> 0.5 — knob never left the homogeneous regime"
    );
    println!("   G1 PASS: spread strictly increasing, reaches {sp_max:.3}");

    // G3: baseline reproduces D-SK-B' at beta=0.
    assert!(
        am_rows[0].ret_sign > 0.90,
        "G3 FAIL: beta=0 sign retention {:.1}% !> 90%",
        100.0 * am_rows[0].ret_sign
    );
    println!(
        "   G3 PASS: beta=0 sign-only retention {:.1}% reproduces the D-SK-B' figure",
        100.0 * am_rows[0].ret_sign
    );

    // ── CHIRP family ─────────────────────────────────────────────────────
    println!("\n== 2. CHIRP family (frequency-swept loop density), W=64 ==");
    println!(
        "   beta | spread | err incr-only | err exact | err i4 | err sign | ret i4 | ret sign"
    );
    let chirp_rows: Vec<Row> = [0.0, 0.5, 1.0, 1.5, 2.0]
        .iter()
        .map(|&b| {
            let x = chirp_path(m, 220.0, 0.045, b, 0.3);
            let y = chirp_path(m, 300.0, 0.038, b, 1.9);
            let r = measure(&x, &y, b);
            println!(
                "   {:.2} | {:.3} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:5.1}% | {:5.1}%",
                r.beta,
                r.sp,
                r.err_a,
                r.err_exact,
                r.err_i4,
                r.err_sign,
                100.0 * r.ret_i4,
                100.0 * r.ret_sign
            );
            r
        })
        .collect();

    // G2 (re-registered): area-domain carrier fidelity ordering. The
    // kernel-scalar ordering gate was falsified by cancellation near the
    // discretization floor (see header) — kernel errors stay printed only.
    for r in am_rows.iter().chain(&chirp_rows) {
        assert!(
            r.rms_i4 <= r.rms_sign + 1e-12,
            "G2 FAIL at beta={:.2}: rms(i4) {:.3e} !<= rms(sign) {:.3e}",
            r.beta,
            r.rms_i4,
            r.rms_sign
        );
    }
    println!("   G2 PASS: rms(i4) <= rms(sign) at every beta, both families (area-domain)");

    // ── Decision rule (pre-registered) ───────────────────────────────────
    println!("\n== 3. decision rule ==");
    let verdict = |rows: &[Row], carrier: &str, get: &dyn Fn(&Row) -> f64| {
        let min_ret = rows.iter().map(get).fold(f64::INFINITY, f64::min);
        let below_50 = rows.iter().find(|r| get(r) < 0.50);
        match below_50 {
            Some(r) => println!(
                "   {carrier}: falls below 50% at spread {:.3} (beta {:.2}) → magnitude tier required from that spread upward",
                r.sp, r.beta
            ),
            None if min_ret > 0.90 => println!(
                "   {carrier}: retention stays > 90% (min {:.1}%) → suffices across the spread range measured",
                100.0 * min_ret
            ),
            None => println!(
                "   {carrier}: min retention {:.1}% (in 50..90%) → partial; magnitude tier recommended",
                100.0 * min_ret
            ),
        }
    };
    let all: Vec<Row> = am_rows.into_iter().chain(chirp_rows).collect();
    verdict(&all, "sign-only (orientation bit)", &|r: &Row| r.ret_sign);
    verdict(&all, "i4 (register nibble)", &|r: &Row| r.ret_i4);

    println!("\nPROBE GREEN — G0–G3 held; decision-rule verdicts above are the banked outcome.");
}
