//! PROBE-HELIX-CONTINUOUS-FIELD — does the helix ANALYTIC place-coder
//! reconstruct a bounded monotone continuous field (the canonical case:
//! geo-elevation) at 3-6 bytes, well enough to own the "continuous-field
//! exit" the honourable-mention currently assigns to the *materialized*
//! bgz-tensor ladder?
//!
//! Cross-ref: `.claude/board/EPIPHANIES.md` `E-BGZ-TENSOR-LANE-REVIEW-1`,
//! `E-PALETTE-RESIDUAL-LADDER-1` (le-contract §3 — "256 levels over
//! ~1500 m ⇒ ~6 m terraces" is the reference figure this probe reproduces
//! or beats), `E-FISHERZ-CANONICAL-COSINE-REPLACEMENT-1` (helix = the
//! analytic 2z rung). This is WI-3 (axis c) of TD-BGZ-TENSOR-PRE-LANE-REVIEW.
//!
//! ## Scope — HELIX ONLY
//!
//! No `bgz-tensor` dependency (the BBB boundary forbids it from this crate;
//! the ladder's materialized number is characterized elsewhere). This probe
//! measures ONLY the analytic side's absolute reconstruction quality: does a
//! plain scalar value survive `value -> bucket -> bucket_center` round-trip
//! well enough for a continuous field, independent of what any other codec
//! in the workspace achieves.
//!
//! ## API grounding — the honest quantizer path
//!
//! The crate's only value<->bucket round-trip for an arbitrary bounded
//! scalar is [`RollingFloor`] (`src/quantize.rs`):
//! [`RollingFloor::quantize`] (`&self`, value -> `u8` bucket, `quantize.rs:99`)
//! and [`RollingFloor::bucket_center`] (`&self`, bucket -> representative
//! `f64`, `quantize.rs:248`). This is the palette256 rung: **1 byte**,
//! [`PALETTE_SIZE`] = 256 buckets.
//!
//! [`ResidueEncoder`] / [`ResidueEdge`] / [`Signed360`] (`src/residue.rs`)
//! are NOT a generic scalar quantizer — `ResidueEncoder::encode` takes
//! `(place: u64, n: usize)` and runs the hemisphere-lift + Fisher-Z + Euler
//! hand-off pipeline that is specific to "the n-th point on the φ-spiral
//! template at HHTL place `place`" (see `sprite_replay.rs`'s doc comment,
//! which independently confirms: "there is no free 2-DOF direction codec in
//! this crate" and the encoder is one-way with no shipped `decode`). There
//! is no clean way to feed an arbitrary bounded scalar (e.g. "1500.0 metres
//! of elevation") into that pipeline without inventing a `place`/`n`
//! convention the crate does not define — doing so would be exactly the
//! "invented round-trip API" `sprite_replay.rs` explicitly warns against.
//! **Verdict: the floor is palette256-only (1 byte) for a plain continuous
//! scalar field.** `residue.rs` has no clean multi-byte scalar path.
//!
//! To still give the orchestrator a multi-byte data point, this module adds
//! an honest EXTENSION built purely from the already-shipped [`RollingFloor`]
//! primitive: a **2-byte stacked floor** (byte 0 = coarse bucket over the
//! full range; byte 1 = a second [`RollingFloor`] scoped to byte 0's
//! sub-range, i.e. simple recursive refinement — no new production code,
//! no residue.rs involvement). This is clearly NOT part of any shipped
//! multi-byte helix codec; it is reported separately and labelled as such.
//!
//! ## Method
//!
//! Three synthetic bounded monotone fields, 1024 samples each, spanning a
//! canyon-relief range (~0..1500 m — the honourable-mention's own reference
//! range):
//!
//! - **(a) linear ramp** — `elevation(t) = 1500 * t`.
//! - **(b) nonlinear ramp** — `elevation(t) = 1500 * t.powf(2.2)` (an
//!   erosion/gamma-style curve — still smooth, monotone, bounded).
//! - **(c) terraced** — 256 piecewise-constant steps of `1500 / 256 m` each
//!   (`elevation(t) = (floor(t * 256) + 0.5) * 1500 / 256`).
//!
//! All three are pure deterministic functions of the sample index (no
//! randomness needed to make them reproducible — no `rand` crate, no
//! seeded generator required for the field definitions themselves).
//!
//! For each field: build a [`RollingFloor`] auto-seeded from the sampled
//! data's own `min`/`max` (the real seeding path available in this crate —
//! there is no separate "auto-seed" constructor, so seeding a `uniform`
//! floor from the observed range IS the calibrated-use path), reconstruct
//! every sample through `quantize` -> `bucket_center`, and report RMSE, max
//! absolute error, and the effective terrace size (`range / PALETTE_SIZE`)
//! at 1 byte, then again (stacked-floor extension) at 2 bytes.
//!
//! This module makes NO pass/kill assertion — the WI-3 bands (PASS-for-
//! analytic if RMSE is within ~0.5% of range at <= 6 bytes) are adjudicated
//! by the orchestrator against the printed table. Assertions here are
//! structural sanity only (finiteness, sample counts) plus a determinism
//! check (build the whole pipeline twice, compare byte-for-byte).

use crate::quantize::RollingFloor;

/// One sampled field: `t` in `[0, 1)` mapped to a bounded elevation value.
struct Field {
    name: &'static str,
    samples: Vec<f64>,
}

const N_SAMPLES: usize = 1024;
const RANGE_M: f64 = 1500.0;

fn linear_ramp() -> Field {
    let samples = (0..N_SAMPLES)
        .map(|i| {
            let t = i as f64 / (N_SAMPLES - 1) as f64;
            RANGE_M * t
        })
        .collect();
    Field {
        name: "linear_ramp",
        samples,
    }
}

fn nonlinear_gamma_ramp() -> Field {
    const GAMMA: f64 = 2.2;
    let samples = (0..N_SAMPLES)
        .map(|i| {
            let t = i as f64 / (N_SAMPLES - 1) as f64;
            RANGE_M * t.powf(GAMMA)
        })
        .collect();
    Field {
        name: "nonlinear_gamma_ramp",
        samples,
    }
}

fn terraced_256() -> Field {
    const LEVELS: f64 = 256.0;
    let samples = (0..N_SAMPLES)
        .map(|i| {
            let t = i as f64 / (N_SAMPLES - 1) as f64;
            let level = (t * LEVELS).floor().min(LEVELS - 1.0);
            (level + 0.5) * (RANGE_M / LEVELS)
        })
        .collect();
    Field {
        name: "terraced_256",
        samples,
    }
}

/// Reconstruction stats for one (field, resolution) pair.
#[derive(Debug, Clone, Copy)]
struct Stats {
    rmse: f64,
    max_abs_err: f64,
    terrace_size: f64,
    count: usize,
}

fn min_max(samples: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in samples {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// 1-byte (palette256) reconstruction: `value -> quantize -> bucket_center`.
/// Returns (stats, the raw bucket bytes — for the determinism check).
fn reconstruct_1byte(samples: &[f64]) -> (Stats, Vec<u8>) {
    let (lo, hi) = min_max(samples);
    let floor = RollingFloor::uniform(lo, hi);
    let mut sq_err = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut bytes = Vec::with_capacity(samples.len());
    for &v in samples {
        let b = floor.quantize(v);
        let recon = floor.bucket_center(b);
        let err = (recon - v).abs();
        sq_err += err * err;
        if err > max_abs {
            max_abs = err;
        }
        bytes.push(b);
    }
    let n = samples.len();
    let rmse = (sq_err / n as f64).sqrt();
    let terrace_size = (hi - lo) / crate::constants::PALETTE_SIZE as f64;
    (
        Stats {
            rmse,
            max_abs_err: max_abs,
            terrace_size,
            count: n,
        },
        bytes,
    )
}

/// Bucket `b`'s `[sub_lo, sub_hi)` span under a `RollingFloor::uniform(lo, hi)`
/// — the inverse of `quantize`'s linear bucketing, derived from the same
/// `t = (value - lo) / (hi - lo)`, `idx = floor(t * 256)` formula
/// (`quantize.rs:99-108`).
fn bucket_span(lo: f64, hi: f64, b: u8) -> (f64, f64) {
    let span = hi - lo;
    let n = crate::constants::PALETTE_SIZE as f64;
    let sub_lo = lo + (b as f64 / n) * span;
    let sub_hi = lo + ((b as f64 + 1.0) / n) * span;
    (sub_lo, sub_hi)
}

/// **Extension, not a shipped helix codec.** 2-byte stacked-`RollingFloor`
/// refinement: byte 0 is the coarse palette256 bucket over the full range;
/// byte 1 is a second `RollingFloor` scoped to byte 0's sub-range. Built
/// entirely from the public [`RollingFloor`] API — no residue.rs, no new
/// production code. See module doc for why `residue.rs` itself has no clean
/// scalar multi-byte path.
fn reconstruct_2byte_stacked(samples: &[f64]) -> (Stats, Vec<(u8, u8)>) {
    let (lo, hi) = min_max(samples);
    let floor0 = RollingFloor::uniform(lo, hi);
    let mut sq_err = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut bytes = Vec::with_capacity(samples.len());
    for &v in samples {
        let b0 = floor0.quantize(v);
        let (sub_lo, sub_hi) = bucket_span(lo, hi, b0);
        // Degenerate sub-range guard (mirrors RollingFloor::quantize's own
        // hi <= lo guard) — shouldn't trigger for PALETTE_SIZE=256 over a
        // 1500 m span, but keep the reconstruction total either way.
        let (floor1, sub_lo, sub_hi) = if sub_hi > sub_lo {
            (RollingFloor::uniform(sub_lo, sub_hi), sub_lo, sub_hi)
        } else {
            (
                RollingFloor::uniform(sub_lo, sub_lo + 1.0),
                sub_lo,
                sub_lo + 1.0,
            )
        };
        let _ = (sub_lo, sub_hi);
        let b1 = floor1.quantize(v);
        let recon = floor1.bucket_center(b1);
        let err = (recon - v).abs();
        sq_err += err * err;
        if err > max_abs {
            max_abs = err;
        }
        bytes.push((b0, b1));
    }
    let n = samples.len();
    let rmse = (sq_err / n as f64).sqrt();
    let terrace_size =
        (hi - lo) / (crate::constants::PALETTE_SIZE as f64 * crate::constants::PALETTE_SIZE as f64);
    (
        Stats {
            rmse,
            max_abs_err: max_abs,
            terrace_size,
            count: n,
        },
        bytes,
    )
}

fn fields() -> Vec<Field> {
    vec![linear_ramp(), nonlinear_gamma_ramp(), terraced_256()]
}

#[test]
fn continuous_field_reconstruction_report() {
    eprintln!(
        "\nPROBE-HELIX-CONTINUOUS-FIELD — range {:.1} m, N={} samples per field\n",
        RANGE_M, N_SAMPLES
    );
    eprintln!(
        "{:<24} {:>10} {:>10} {:>12} | {:>10} {:>10} {:>12}",
        "field", "RMSE(1B)", "max(1B)", "terrace(1B)", "RMSE(2B)", "max(2B)", "terrace(2B)"
    );

    for f in fields() {
        let (s1, _) = reconstruct_1byte(&f.samples);
        let (s2, _) = reconstruct_2byte_stacked(&f.samples);

        assert!(s1.rmse.is_finite() && s1.max_abs_err.is_finite());
        assert!(s2.rmse.is_finite() && s2.max_abs_err.is_finite());
        assert_eq!(s1.count, N_SAMPLES);
        assert_eq!(s2.count, N_SAMPLES);

        eprintln!(
            "{:<24} {:>10.4} {:>10.4} {:>12.4} | {:>10.6} {:>10.6} {:>12.6}",
            f.name,
            s1.rmse,
            s1.max_abs_err,
            s1.terrace_size,
            s2.rmse,
            s2.max_abs_err,
            s2.terrace_size
        );
    }

    // Reference figure reproduction check: the honourable-mention's own
    // "256 levels over ~1500 m ⇒ ~6 m terraces" — palette256's terrace size
    // over this exact range should land at 1500/256 ≈ 5.859 m, i.e. the
    // reference figure this probe is meant to reproduce/beat.
    let ref_terrace = RANGE_M / crate::constants::PALETTE_SIZE as f64;
    eprintln!(
        "\nreference terrace (honourable-mention figure): {:.4} m (1500/256)",
        ref_terrace
    );
    assert!((ref_terrace - 5.859_375).abs() < 1e-6);
}

#[test]
fn continuous_field_reconstruction_is_deterministic() {
    // Build the whole pipeline twice; bucket bytes (1B and 2B) must be
    // byte-identical both times — the analytic lane has no hidden state
    // that would make two builds diverge.
    for f_name_idx in 0..3 {
        let f_a = fields().into_iter().nth(f_name_idx).unwrap();
        let f_b = fields().into_iter().nth(f_name_idx).unwrap();
        assert_eq!(
            f_a.samples, f_b.samples,
            "field samples must be deterministic"
        );

        let (_, bytes_a1) = reconstruct_1byte(&f_a.samples);
        let (_, bytes_b1) = reconstruct_1byte(&f_b.samples);
        assert_eq!(
            bytes_a1, bytes_b1,
            "{}: 1-byte reconstruction must be deterministic",
            f_a.name
        );

        let (_, bytes_a2) = reconstruct_2byte_stacked(&f_a.samples);
        let (_, bytes_b2) = reconstruct_2byte_stacked(&f_b.samples);
        assert_eq!(
            bytes_a2, bytes_b2,
            "{}: 2-byte reconstruction must be deterministic",
            f_a.name
        );
    }
}
