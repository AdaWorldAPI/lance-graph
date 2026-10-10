//! D-PHT-1 — Phasor evaluation without transcendental calls: native
//! `sin_cos` against a phase LUT, CORDIC and a complex recurrence, measured on
//! two observables that need nothing but phasors:
//!
//! 1. **Wankel apex coordinates** `z_k(θ) = e·e^{i3θ} + R·e^{i(θ + 2πk/3)}`
//!    (the ideal apex triangle; not the rotor flanks, not sealing).
//! 2. **Coherent interference** `I(P) = |Σ_i A_i e^{iφ_i(P)}|²` over many
//!    sources and detectors, with an amplitude-bounded exact early exit:
//!    `max(0, |S| − R)² ≤ I ≤ (|S| + R)²`, `R = Σ_{unevaluated} A_i`.
//!
//! The phase is a `u32` in TURNS: one full rotation is `2^32`, so `3θ` is
//! `phase.wrapping_mul(3)` and wraparound is free.
//!
//! Nothing here is a production primitive. Every approximate arm is measured
//! against an `f64` `sin_cos` reference that shares no code with it. Timings
//! are printed, never asserted; invariants and decisions are asserted.
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-mask-risc --example phasor_trig_probe
//! ```

// A source index addresses several parallel tables at once (position,
// amplitude, phase offset, suffix bound); an iterator per table would hide
// that the loops walk ONE source list.
#![allow(clippy::needless_range_loop)]

use std::f64::consts::{GOLDEN_RATIO, TAU};
use std::hint::black_box;
use std::time::Instant;

// ───────────────────────── rng ─────────────────────────

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

const TWO32: f64 = 4_294_967_296.0;

fn turns_to_rad(p: u32) -> f64 {
    f64::from(p) / TWO32 * TAU
}

// ───────────────────────── the four phasor evaluators ─────────────────────────

/// A `(cos, sin)` table over `2^bits` phases, `f32`.
struct Lut {
    bits: u32,
    t: Vec<(f32, f32)>,
}

impl Lut {
    fn new(bits: u32) -> Self {
        let n = 1usize << bits;
        let t = (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                (a.cos() as f32, a.sin() as f32)
            })
            .collect();
        Lut { bits, t }
    }
    /// Nearest entry.
    #[inline]
    fn nearest(&self, p: u32) -> (f32, f32) {
        let sh = 32 - self.bits;
        let i = (p.wrapping_add(1 << (sh - 1)) >> sh) as usize;
        self.t[i & (self.t.len() - 1)]
    }
    /// Linear interpolation between the two neighbouring entries.
    #[inline]
    fn lerp(&self, p: u32) -> (f32, f32) {
        let sh = 32 - self.bits;
        let i = (p >> sh) as usize;
        let f = (p & ((1 << sh) - 1)) as f32 / (1u32 << sh) as f32;
        let m = self.t.len() - 1;
        let (a, b) = (self.t[i & m], self.t[(i + 1) & m]);
        (a.0 + (b.0 - a.0) * f, a.1 + (b.1 - a.1) * f)
    }
    fn bytes(&self) -> usize {
        self.t.len() * 8
    }
    /// A conservative bound on `|lut(p) − e^{ip}|` for `nearest`: half a step
    /// of arc plus the `f32` rounding of both components.
    fn nearest_bound(&self) -> f64 {
        std::f64::consts::PI / (1u64 << self.bits) as f64 + 2.0 * f64::from(f32::EPSILON)
    }
}

/// Circular CORDIC in rotation mode, `i64` fixed point Q30, angle in turns.
struct Cordic {
    iters: usize,
    /// `atan(2^-i)` in turns · 2^32.
    atan: Vec<i64>,
    /// `K = Π 1/√(1 + 2^-2i)` in Q30: the start vector absorbs the gain.
    k_q30: i64,
}

impl Cordic {
    fn new(iters: usize) -> Self {
        let atan = (0..iters)
            .map(|i| ((2f64.powi(-(i as i32))).atan() / TAU * TWO32).round() as i64)
            .collect();
        let k: f64 = (0..iters)
            .map(|i| 1.0 / (1.0 + 4f64.powi(-(i as i32))).sqrt())
            .product();
        Cordic {
            iters,
            atan,
            k_q30: (k * (1u64 << 30) as f64).round() as i64,
        }
    }
    /// `(cos, sin)` of a phase in turns, as `f64` from Q30.
    #[inline]
    fn eval(&self, p: u32) -> (f64, f64) {
        // Quadrant from the top two bits; the residual is in [0, π/2), inside
        // CORDIC's ~1.74 rad convergence range.
        let q = p >> 30;
        let mut z = i64::from(p & 0x3FFF_FFFF);
        let (mut x, mut y) = (self.k_q30, 0i64);
        for i in 0..self.iters {
            let (dx, dy) = (y >> i, x >> i);
            if z >= 0 {
                x -= dx;
                y += dy;
                z -= self.atan[i];
            } else {
                x += dx;
                y -= dy;
                z += self.atan[i];
            }
        }
        let (c, s) = match q {
            0 => (x, y),
            1 => (-y, x),
            2 => (-x, -y),
            _ => (y, -x),
        };
        let s30 = (1u64 << 30) as f64;
        (c as f64 / s30, s as f64 / s30)
    }
}

// ───────────────────────── Wankel ─────────────────────────

struct Wankel {
    r: f64,
    e: f64,
}

impl Wankel {
    /// Reference apex `k` at rotor angle `θ` (radians), `f64` `sin_cos`.
    fn apex_ref(&self, theta: f64, k: usize) -> (f64, f64) {
        let (s3, c3) = (3.0 * theta).sin_cos();
        let (s, c) = (theta + TAU * k as f64 / 3.0).sin_cos();
        (self.e * c3 + self.r * c, self.e * s3 + self.r * s)
    }
    /// The housing curve `R e^{iθ} + e e^{i3θ}`.
    fn housing(&self, theta: f64) -> (f64, f64) {
        self.apex_ref(theta, 0)
    }
}

/// Invariants of the ideal geometry, on the reference implementation.
fn wankel_invariants() {
    println!("== Wankel invariants (f64 reference) ==");
    for &(r, e) in &[(100.0, 14.0), (100.0, 1e-9), (100.0, 0.0), (1.0, 0.15)] {
        let w = Wankel { r, e };
        let mut rng = Rng(0x3A11);
        let (mut on_curve, mut side, mut centre, mut sym) = (0f64, 0f64, 0f64, 0f64);
        for _ in 0..100_000 {
            // Includes angles far outside [0, 2π) to exercise wrapping.
            let th = (rng.unit() - 0.5) * 2000.0;
            let a: Vec<(f64, f64)> = (0..3).map(|k| w.apex_ref(th, k)).collect();
            for (k, ak) in a.iter().enumerate() {
                // 1. Every apex is on the housing curve at θ + 2πk/3.
                let h = w.housing(th + TAU * k as f64 / 3.0);
                on_curve = on_curve.max(((ak.0 - h.0).powi(2) + (ak.1 - h.1).powi(2)).sqrt());
                // 2. The apex triangle keeps its side R√3.
                let b = a[(k + 1) % 3];
                let d = ((ak.0 - b.0).powi(2) + (ak.1 - b.1).powi(2)).sqrt();
                side = side.max((d - r * 3f64.sqrt()).abs());
            }
            // 3. The rotor centre (mean of the apexes) is at distance e.
            let c = (
                (a[0].0 + a[1].0 + a[2].0) / 3.0,
                (a[0].1 + a[1].1 + a[2].1) / 3.0,
            );
            centre = centre.max(((c.0 * c.0 + c.1 * c.1).sqrt() - e).abs());
            // 4. Threefold symmetry: advancing θ by 2π/3 moves apex 0 to apex 1.
            let n = w.apex_ref(th + TAU / 3.0, 0);
            sym = sym.max(((n.0 - a[1].0).powi(2) + (n.1 - a[1].1).powi(2)).sqrt());
        }
        let tol = 1e-9 * r;
        assert!(
            on_curve < tol && side < tol && centre < tol && sym < tol,
            "R {r} e {e}"
        );
        println!(
            "  R {r:>5} e {e:>7}: max off-curve {on_curve:.1e}  side error {side:.1e}  centre error {centre:.1e}  symmetry {sym:.1e}"
        );
    }
    // The 3:1 law in integer turns is exact: 3·p wraps exactly as 3θ mod 2π.
    let mut rng = Rng(0x31);
    for _ in 0..100_000 {
        let p = rng.next() as u32;
        let lhs = turns_to_rad(p.wrapping_mul(3));
        let rhs = (3.0 * turns_to_rad(p)).rem_euclid(TAU);
        assert!((lhs - rhs).abs() < 1e-9 || (lhs - rhs).abs() > TAU - 1e-9);
    }
    println!(
        "  all hold; the 3:1 law as `phase.wrapping_mul(3)` matches 3θ mod 2π on 100,000 phases"
    );
}

fn bench_ns<T>(n: usize, reps: usize, mut f: impl FnMut() -> T) -> f64 {
    black_box(f());
    let t = Instant::now();
    for _ in 0..reps {
        black_box(f());
    }
    t.elapsed().as_nanos() as f64 / (reps * n) as f64
}

/// Apex coordinates for `n` phases: speed and max error against the reference.
fn wankel_eval(n: usize) {
    println!(
        "\n== Wankel apex coordinates, {n} random phases (R 100, e 14; error in housing units) =="
    );
    let w = Wankel { r: 100.0, e: 14.0 };
    let mut rng = Rng(0xA9);
    let phases: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
    let reference: Vec<(f64, f64)> = phases
        .iter()
        .map(|p| w.apex_ref(turns_to_rad(*p), 0))
        .collect();
    let reps = 5;
    let mut out = vec![(0f64, 0f64); n];

    let report = |name: &str, ns: f64, out: &[(f64, f64)], trig: &str, table: usize| {
        let err = out
            .iter()
            .zip(&reference)
            .map(|(a, b)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt())
            .fold(0f64, f64::max);
        println!("  {name:<30} {ns:>6.2} ns/apex  max error {err:>9.2e}  trig calls {trig:<8}  table {table:>7} B");
    };

    let ns = bench_ns(n, reps, || {
        for (o, p) in out.iter_mut().zip(&phases) {
            let th = turns_to_rad(*p);
            let (s, c) = th.sin_cos();
            let (s3, c3) = (3.0 * th).sin_cos();
            *o = (w.r * c + w.e * c3, w.r * s + w.e * s3);
        }
    });
    report("S64  native f64 sin_cos", ns, &out, "2/apex", 0);

    let (r32, e32) = (w.r as f32, w.e as f32);
    let ns = bench_ns(n, reps, || {
        for (o, p) in out.iter_mut().zip(&phases) {
            let th = turns_to_rad(*p) as f32;
            let (s, c) = th.sin_cos();
            let (s3, c3) = (3.0 * th).sin_cos();
            *o = (f64::from(r32 * c + e32 * c3), f64::from(r32 * s + e32 * s3));
        }
    });
    report("S32  native f32 sin_cos", ns, &out, "2/apex", 0);

    for bits in [10u32, 12, 16] {
        let lut = Lut::new(bits);
        let ns = bench_ns(n, reps, || {
            for (o, p) in out.iter_mut().zip(&phases) {
                let (c, s) = lut.nearest(*p);
                let (c3, s3) = lut.nearest(p.wrapping_mul(3));
                *o = (f64::from(r32 * c + e32 * c3), f64::from(r32 * s + e32 * s3));
            }
        });
        report(&format!("L{bits}  LUT nearest"), ns, &out, "0", lut.bytes());
    }
    for bits in [8u32, 10, 12] {
        let lut = Lut::new(bits);
        let ns = bench_ns(n, reps, || {
            for (o, p) in out.iter_mut().zip(&phases) {
                let (c, s) = lut.lerp(*p);
                let (c3, s3) = lut.lerp(p.wrapping_mul(3));
                *o = (f64::from(r32 * c + e32 * c3), f64::from(r32 * s + e32 * s3));
            }
        });
        report(
            &format!("L{bits}i LUT linear interpolation"),
            ns,
            &out,
            "0",
            lut.bytes(),
        );
    }
    for iters in [16usize, 24, 30] {
        let cd = Cordic::new(iters);
        let ns = bench_ns(n, reps, || {
            for (o, p) in out.iter_mut().zip(&phases) {
                let (c, s) = cd.eval(*p);
                let (c3, s3) = cd.eval(p.wrapping_mul(3));
                *o = (w.r * c + w.e * c3, w.r * s + w.e * s3);
            }
        });
        report(&format!("C{iters}  CORDIC Q30"), ns, &out, "0", iters * 8);
    }

    // ndarray's vector math over arrays: θ and 3θ must be materialised as
    // arrays first, and the outputs read back. Its cos/sin are per-lane
    // scalar calls inside an F32x16 wrapper (hpc/vml.rs).
    {
        use ndarray::Array1;
        let th: Array1<f32> = phases.iter().map(|p| turns_to_rad(*p) as f32).collect();
        let th3: Array1<f32> = phases
            .iter()
            .map(|p| turns_to_rad(p.wrapping_mul(3)) as f32)
            .collect();
        let (mut c, mut s, mut c3, mut s3) = (
            Array1::<f32>::zeros(n),
            Array1::<f32>::zeros(n),
            Array1::<f32>::zeros(n),
            Array1::<f32>::zeros(n),
        );
        let ns = bench_ns(n, reps, || {
            ndarray::hpc::vml::vscos(th.view(), c.view_mut());
            ndarray::hpc::vml::vssin(th.view(), s.view_mut());
            ndarray::hpc::vml::vscos(th3.view(), c3.view_mut());
            ndarray::hpc::vml::vssin(th3.view(), s3.view_mut());
            for (i, o) in out.iter_mut().enumerate() {
                *o = (
                    f64::from(r32 * c[i] + e32 * c3[i]),
                    f64::from(r32 * s[i] + e32 * s3[i]),
                );
            }
        });
        report("V    ndarray vml vscos/vssin", ns, &out, "4/apex", 0);
        println!(
            "       (V also materialises θ, 3θ and four output arrays: {} B)",
            6 * n * 4
        );
    }
}

/// A trajectory of `steps` equal increments: the complex recurrence against
/// the reference, with and without renormalisation.
fn recurrence(steps: usize) {
    println!("\n== Wankel trajectory, {steps} steps of Δθ = 2π/4096·φ (recurrence; error at the last step and max) ==");
    let w = Wankel { r: 100.0, e: 14.0 };
    // An irrational step, so the trajectory never revisits a phase exactly.
    let dth = TAU / 4096.0 * GOLDEN_RATIO;
    let reference = |n: usize| w.apex_ref(n as f64 * dth, 0);

    // f64 and f32 recurrences, renormalising every `every` steps (0 = never).
    for (label, every) in [("never", 0usize), ("every 1024", 1024), ("every 64", 64)] {
        let (w1, w3) = (
            (dth.cos(), dth.sin()),
            ((3.0 * dth).cos(), (3.0 * dth).sin()),
        );
        let (mut u, mut v) = ((1.0f64, 0.0f64), (1.0f64, 0.0f64));
        let (mut u32_, mut v32) = ((1.0f32, 0.0f32), (1.0f32, 0.0f32));
        let (w1f, w3f) = ((w1.0 as f32, w1.1 as f32), (w3.0 as f32, w3.1 as f32));
        let (mut e64, mut e32) = (0f64, 0f64);
        let mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let mulf = |a: (f32, f32), b: (f32, f32)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let t = Instant::now();
        for n in 1..=steps {
            u = mul(u, w1);
            v = mul(v, w3);
            u32_ = mulf(u32_, w1f);
            v32 = mulf(v32, w3f);
            if every != 0 && n % every == 0 {
                let m = (u.0 * u.0 + u.1 * u.1).sqrt();
                u = (u.0 / m, u.1 / m);
                let m = (v.0 * v.0 + v.1 * v.1).sqrt();
                v = (v.0 / m, v.1 / m);
                let m = (u32_.0 * u32_.0 + u32_.1 * u32_.1).sqrt();
                u32_ = (u32_.0 / m, u32_.1 / m);
                let m = (v32.0 * v32.0 + v32.1 * v32.1).sqrt();
                v32 = (v32.0 / m, v32.1 / m);
            }
            if n % 997 == 0 || n == steps {
                let r = reference(n);
                let z = (w.r * u.0 + w.e * v.0, w.r * u.1 + w.e * v.1);
                e64 = e64.max(((z.0 - r.0).powi(2) + (z.1 - r.1).powi(2)).sqrt());
                let zf = (
                    w.r * f64::from(u32_.0) + w.e * f64::from(v32.0),
                    w.r * f64::from(u32_.1) + w.e * f64::from(v32.1),
                );
                e32 = e32.max(((zf.0 - r.0).powi(2) + (zf.1 - r.1).powi(2)).sqrt());
            }
        }
        let ns = t.elapsed().as_nanos() as f64 / steps as f64;
        println!(
            "  renormalise {label:<10}: max error f64 {e64:>9.2e}  f32 {e32:>9.2e}  ({ns:.2} ns/step for both, incl. the sampled reference)"
        );
    }
    // The same trajectory as an integer phase: an exact u32 step accumulates
    // no drift at all; only the LUT's own error remains.
    let step = (dth / TAU * TWO32).round() as u32;
    let lut = Lut::new(12);
    let (mut p, mut emax) = (0u32, 0f64);
    let step_err = (f64::from(step) / TWO32 * TAU - dth).abs();
    for n in 1..=steps {
        p = p.wrapping_add(step);
        if n % 997 == 0 || n == steps {
            // Compare against the trajectory the integer step actually takes.
            let th = turns_to_rad(p);
            let (c, s) = lut.lerp(p);
            let (c3, s3) = lut.lerp(p.wrapping_mul(3));
            let z = (
                w.r * f64::from(c) + w.e * f64::from(c3),
                w.r * f64::from(s) + w.e * f64::from(s3),
            );
            let r = w.apex_ref(th, 0);
            emax = emax.max(((z.0 - r.0).powi(2) + (z.1 - r.1).powi(2)).sqrt());
        }
    }
    println!(
        "  integer phase + L12i     : max error {emax:.2e} against its own trajectory; the u32 step differs from Δθ by {step_err:.1e} rad per step (a frequency choice, not drift)"
    );
}

// ───────────────────────── interference ─────────────────────────

struct Field {
    /// Source positions, amplitudes (sorted descending), phase offsets in turns.
    sx: Vec<f64>,
    sy: Vec<f64>,
    a: Vec<f64>,
    p0: Vec<f64>,
    /// 1 / λ.
    inv_lambda: f64,
}

fn field(n: usize, heavy_tail: bool, seed: u64) -> Field {
    let mut r = Rng(seed);
    let mut a: Vec<f64> = (0..n)
        .map(|i| {
            if heavy_tail {
                ((i + 1) as f64).powf(-1.5)
            } else {
                0.5 + 0.5 * r.unit()
            }
        })
        .collect();
    a.sort_by(|x, y| y.partial_cmp(x).unwrap());
    Field {
        sx: (0..n).map(|_| -64.0 + 384.0 * r.unit()).collect(),
        sy: (0..n).map(|_| -64.0 + 384.0 * r.unit()).collect(),
        a,
        p0: (0..n).map(|_| r.unit()).collect(),
        inv_lambda: 1.0 / 7.3,
    }
}

/// Phase of source `i` at detector `(x, y)`, in turns (`f64`, unwrapped).
#[inline]
fn turns(f: &Field, i: usize, x: f64, y: f64) -> f64 {
    let (dx, dy) = (x - f.sx[i], y - f.sy[i]);
    (dx * dx + dy * dy).sqrt() * f.inv_lambda + f.p0[i]
}

/// Turns to a `u32` phase, modulo one turn. Through `i64`, not `u64`: a
/// negative turn count (a phase DIFFERENCE) saturates to 0 under `as u64`,
/// which the first run of this probe did, giving a 99 % error on two waves.
#[inline]
fn to_u32(t: f64) -> u32 {
    (t * TWO32) as i64 as u32
}

/// The bound tests for `I ≥ t`, given the partial sum `(re, im)` and an upper
/// bound `r` on what the unevaluated terms (plus any evaluation error) can
/// still add. `None` while undecided.
///
/// With nothing left (`r == 0`) the decision is `re² + im² ≥ t` computed
/// exactly as the full fold computes it, never through `√` then a square.
/// Before that, the `√` rounding is absorbed by a relative margin.
#[inline]
fn bound_decide(re: f64, im: f64, r: f64, t: f64) -> Option<bool> {
    let i2 = re * re + im * im;
    if r == 0.0 {
        return Some(i2 >= t);
    }
    let m = i2.sqrt();
    const MARGIN: f64 = 1e-12;
    if (m + r) * (m + r) < t * (1.0 - MARGIN) {
        Some(false)
    } else if (m - r).max(0.0).powi(2) >= t * (1.0 + MARGIN) {
        Some(true)
    } else {
        None
    }
}

/// Full fold at one detector with the given phasor evaluator.
#[inline]
fn fold(f: &Field, x: f64, y: f64, ev: &impl Fn(f64) -> (f64, f64)) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for i in 0..f.a.len() {
        let (c, s) = ev(turns(f, i, x, y));
        re += f.a[i] * c;
        im += f.a[i] * s;
    }
    re * re + im * im
}

fn interference(n_src: usize, side: usize, heavy_tail: bool) {
    let f = field(n_src, heavy_tail, if heavy_tail { 0x7A } else { 0x7B });
    let det: Vec<(f64, f64)> = (0..side * side)
        .map(|i| ((i % side) as f64, (i / side) as f64))
        .collect();
    let law = if heavy_tail {
        "A_i = i^-1.5"
    } else {
        "A_i uniform in [0.5, 1]"
    };
    println!(
        "\n== interference: {n_src} coherent sources, {} detectors, {law}, λ 7.3 ==",
        det.len()
    );

    let reference = |t: f64| {
        let (s, c) = (TAU * t).sin_cos();
        (c, s)
    };
    let t0 = Instant::now();
    let truth: Vec<f64> = det
        .iter()
        .map(|&(x, y)| fold(&f, x, y, &reference))
        .collect();
    let ns_ref = t0.elapsed().as_nanos() as f64 / (det.len() * n_src) as f64;
    let peak = truth.iter().cloned().fold(0f64, f64::max);
    println!("  S64  native f64 sin_cos      {ns_ref:>6.2} ns/term   (reference)");

    let err = |v: &[f64]| {
        v.iter()
            .zip(&truth)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max)
            / peak
    };
    let time_fold = |name: &str, ev: &dyn Fn(f64) -> (f64, f64)| {
        let t = Instant::now();
        let v: Vec<f64> = det.iter().map(|&(x, y)| fold(&f, x, y, &ev)).collect();
        let ns = t.elapsed().as_nanos() as f64 / (det.len() * n_src) as f64;
        println!(
            "  {name:<28} {ns:>6.2} ns/term   max |ΔI| / peak I {:.2e}",
            err(&v)
        );
    };
    time_fold("S32  native f32 sin_cos", &|t: f64| {
        let (s, c) = ((TAU * t) as f32).sin_cos();
        (f64::from(c), f64::from(s))
    });
    let l12 = Lut::new(12);
    time_fold("L12  LUT nearest", &|t: f64| {
        let (c, s) = l12.nearest(to_u32(t));
        (f64::from(c), f64::from(s))
    });
    let l10 = Lut::new(10);
    time_fold("L10i LUT linear interpolation", &|t: f64| {
        let (c, s) = l10.lerp(to_u32(t));
        (f64::from(c), f64::from(s))
    });
    let cd = Cordic::new(24);
    time_fold("C24  CORDIC Q30", &|t: f64| cd.eval(to_u32(t)));
    // The path length alone: the square root every arm shares.
    let t = Instant::now();
    let mut acc = 0.0;
    for &(x, y) in &det {
        for i in 0..n_src {
            acc += turns(&f, i, x, y);
        }
    }
    black_box(acc);
    let ns = t.elapsed().as_nanos() as f64 / (det.len() * n_src) as f64;
    println!("  geometry only (√ per term)   {ns:>6.2} ns/term   (the floor every arm pays)");

    // Early exit for I ≥ T, sources in descending amplitude.
    let suffix: Vec<f64> = {
        let mut s = vec![0.0; n_src + 1];
        for i in (0..n_src).rev() {
            s[i] = s[i + 1] + f.a[i];
        }
        s
    };
    let mut sorted = truth.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let eps = l12.nearest_bound();
    for q in [0.5, 0.9, 0.99] {
        let t_thr = sorted[((sorted.len() - 1) as f64 * q) as usize];
        // Exact path: f64 reference phasors.
        let mut used = Vec::with_capacity(det.len());
        // LUT path: per-term error ε inflates the bound; undecided at the end
        // falls back to the reference for that detector.
        let mut used_lut = Vec::with_capacity(det.len());
        let mut fallback = 0usize;
        for (k, &(x, y)) in det.iter().enumerate() {
            let want = truth[k] >= t_thr;
            // f64.
            let (mut re, mut im) = (0.0f64, 0.0f64);
            let mut decided = None;
            for i in 0..=n_src {
                if let Some(d) = bound_decide(re, im, suffix[i], t_thr) {
                    decided = Some((d, i));
                    break;
                }
                if i == n_src {
                    break;
                }
                let (c, s) = reference(turns(&f, i, x, y));
                re += f.a[i] * c;
                im += f.a[i] * s;
            }
            let (d, i) = decided.expect("with nothing left the bounds are the value itself");
            assert_eq!(d, want, "exact early exit disagrees at detector {k}");
            used.push(i);
            // LUT with conservative error.
            let (mut re, mut im, mut aerr) = (0.0f64, 0.0f64, 0.0f64);
            let mut decided = None;
            for i in 0..=n_src {
                // The LUT sum is NOT the reference value even with nothing
                // left: its own error `aerr` keeps the bound open. The floor
                // keeps `bound_decide` off its exact `r == 0` branch.
                let r = (suffix[i] + aerr).max(f64::MIN_POSITIVE);
                if let Some(d) = bound_decide(re, im, r, t_thr) {
                    decided = Some((d, i));
                    break;
                }
                if i == n_src {
                    break;
                }
                let (c, s) = l12.nearest(to_u32(turns(&f, i, x, y)));
                re += f.a[i] * f64::from(c);
                im += f.a[i] * f64::from(s);
                aerr += f.a[i] * eps;
            }
            match decided {
                Some((d, i)) => {
                    assert_eq!(
                        d, want,
                        "LUT early exit with error bound decided wrongly at detector {k}"
                    );
                    used_lut.push(i);
                }
                None => {
                    fallback += 1;
                    used_lut.push(n_src);
                }
            }
        }
        let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
        let p95 = |v: &mut Vec<usize>| {
            v.sort_unstable();
            v[(v.len() - 1) * 95 / 100]
        };
        println!(
            "  early exit, T at the {:>2.0} % quantile: f64 terms mean {:>6.1} p95 {:>4}  |  L12 + error bound terms mean {:>6.1} p95 {:>4}, fallbacks {} of {}",
            q * 100.0,
            mean(&used),
            p95(&mut used),
            mean(&used_lut),
            p95(&mut used_lut),
            fallback,
            det.len()
        );
    }
}

/// Two waves: the closed form needs one cosine of the phase DIFFERENCE,
/// `I = A1² + A2² + 2·A1·A2·cos(Δφ)`, instead of two complex phasors.
fn two_waves(side: usize) {
    println!("\n== two waves: two phasors against one cos(Δφ) ==");
    let f = field(2, false, 0x22);
    let det: Vec<(f64, f64)> = (0..side * side)
        .map(|i| ((i % side) as f64, (i / side) as f64))
        .collect();
    let (a1, a2) = (f.a[0], f.a[1]);
    let lut = Lut::new(12);
    let n = det.len();
    let mut out = vec![0f64; n];
    let ns_full = bench_ns(n, 5, || {
        for (o, &(x, y)) in out.iter_mut().zip(&det) {
            let (s1, c1) = (TAU * turns(&f, 0, x, y)).sin_cos();
            let (s2, c2) = (TAU * turns(&f, 1, x, y)).sin_cos();
            let (re, im) = (a1 * c1 + a2 * c2, a1 * s1 + a2 * s2);
            *o = re * re + im * im;
        }
    });
    let truth = out.clone();
    let ns_diff = bench_ns(n, 5, || {
        for (o, &(x, y)) in out.iter_mut().zip(&det) {
            let d = turns(&f, 0, x, y) - turns(&f, 1, x, y);
            *o = a1 * a1 + a2 * a2 + 2.0 * a1 * a2 * (TAU * d).cos();
        }
    });
    let e_diff = out
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    let ns_lut = bench_ns(n, 5, || {
        for (o, &(x, y)) in out.iter_mut().zip(&det) {
            let d = turns(&f, 0, x, y) - turns(&f, 1, x, y);
            let (c, _) = lut.lerp(to_u32(d));
            *o = a1 * a1 + a2 * a2 + 2.0 * a1 * a2 * f64::from(c);
        }
    });
    let e_lut = out
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    let peak = (a1 + a2) * (a1 + a2);
    // The closed form is an identity; the lookup only adds the table's error.
    assert!(
        e_diff / peak < 1e-12 && e_lut / peak < 1e-5,
        "two waves: {e_diff} / {e_lut}"
    );
    println!("  two sin_cos + |Σ|²          {ns_full:>6.2} ns/detector  (reference)");
    println!(
        "  one cos(Δφ), closed form     {ns_diff:>6.2} ns/detector  max error {:.1e} of peak",
        e_diff / peak
    );
    println!(
        "  one L12i lookup of Δφ        {ns_lut:>6.2} ns/detector  max error {:.1e} of peak",
        e_lut / peak
    );
}

fn main() {
    println!(
        "D-PHT-1 phasor trig probe  avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    wankel_invariants();
    wankel_eval(65_536);
    wankel_eval(1_000_000);
    recurrence(1_000_000);
    two_waves(256);
    interference(1000, 256, false);
    interference(1000, 256, true);
    println!("\nall invariants and decisions hold");
}
