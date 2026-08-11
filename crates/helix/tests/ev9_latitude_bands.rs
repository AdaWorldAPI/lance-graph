//! EV-9 (v2) — commits the orphaned K-12/K-13 measurements from
//! `.claude/plans/weather-substrate-evaluation-v1.md` §1a/§3 as falsifiable
//! tests.
//!
//! K-12/K-13 were measured in scratch tests that were deliberately DELETED —
//! graded `[H]` in the plan's KNOWN ledger precisely because no committed
//! reproducer existed. This file is that reproducer, built to the **v2**
//! spec (§3 EV-9) that survived the 2026-08-11 audit (§8): the v1 spec's two
//! bare-point assertions (the `0.99-1.02x` equal-budget ratio, the bare
//! `0.352°` mean) were struck as near-tautologies — both follow from
//! arithmetic alone (the two arms' step sizes; `step/4` for a uniform
//! quantizer's mean absolute error) — and are NOT asserted here in that bare
//! form. What IS asserted: (i) the two byte-encoding arms are genuinely
//! different code paths, not one aliased twice; (ii) the equal-budget ratio
//! against a tolerance DERIVED from the two arms' analytic step sizes; (iii)
//! the pole/equator error SPREAD (the actual K-12 finding, not two point
//! values that could drift together); (iv) the palette-`circular()` vs
//! nearest-`(n,sign)` comparison (K-13 vs K-7) on the SAME sample; (v) the
//! sampling scheme (`N`, latitude stratification, azimuth sweep) pinned
//! explicitly in code rather than left to the mechanism's reputation.
//!
//! Decode convention (matches `bearing_encode_paths.rs`'s `decode` — the
//! crate's own house convention for measuring `Signed360` angular error):
//! `y = polar>=128 ? (polar-128)/127 : -((127-polar)/127)`,
//! `r = sqrt(max(0, 1-y^2))`, `az = azimuth/65536 * TAU`,
//! `v = (r*sin(az), r*cos(az), y)`.
//!
//! NOTE: `helix` is excluded from the root workspace and appears in no CI
//! workflow (K-8), so these run only when invoked by hand in this crate.
use helix::placement::{HemispherePoint, Sign};
use helix::residue::{ResidueEncoder, Signed360};
use helix::PALETTE_SIZE;
use std::f64::consts::TAU;
use std::ops::RangeInclusive;

// ── (v) sampling scheme, pinned explicitly ──────────────────────────────
//
// N, the latitude stratification, and the azimuth-sweep width are all named
// constants (not magic numbers buried in a loop), and
// `azimuth_actually_sweeps_within_each_latitude_band` below asserts the
// sweep is real rather than trusting golden-angle equidistribution by
// reputation.

/// Total residue count, matching K-12/K-13's own provenance ("measured this
/// session"/"N=65536" throughout `bearing_encode_paths.rs` and
/// `signed360_claims.rs`).
const N: usize = 65_536;

/// Equator band, matching K-12's own band definition ("equator (0-5°)").
const EQUATOR_BAND_DEG: RangeInclusive<f64> = 0.0..=5.0;

/// Pole band, matching K-12's own band definition ("pole (85-90°)").
const POLE_BAND_DEG: RangeInclusive<f64> = 85.0..=90.0;

/// How many azimuth targets the comparative test (iv) sweeps at elevation 0
/// (pure-azimuth, the axis K-13 measures) — evenly spaced over the full
/// circle, every 5°.
const AZ_SAMPLES: usize = 72;

/// A fixed, arbitrary PLACE anchor. `place` feeds only `rim` (the metric
/// carrier); it never reaches `polar`/`azimuth` (`residue.rs:182-204`), so
/// any fixed value is representative.
const PLACE: u64 = 0x1234;

// ── shared helpers (match the crate's existing test-file conventions) ──

/// `y` treated as `sin(latitude)`, matching `placement.rs`'s own doc
/// ("the pole (`y = 1`)"; "the equator (`y -> 0`)") — `y` IS the height
/// coordinate on the unit sphere, so `lat = asin(y)`.
fn latitude_deg(y: f64) -> f64 {
    y.clamp(-1.0, 1.0).asin().to_degrees()
}

/// Decode a `Signed360` to a unit-sphere Cartesian point. Identical to
/// `bearing_encode_paths.rs::decode` (the house convention for measuring
/// angular error against this codec).
fn decode(s: &Signed360) -> (f64, f64, f64) {
    let y = if s.polar >= 128 {
        (s.polar as f64 - 128.0) / 127.0
    } else {
        -((127.0 - s.polar as f64) / 127.0)
    };
    let r = (1.0 - y * y).max(0.0).sqrt();
    let a = s.azimuth as f64 / 65536.0 * TAU;
    (r * a.sin(), r * a.cos(), y)
}

/// Ground truth for residue `n`, `sign` — `signed_lift` per the brief:
/// `(r*sin(raw_azimuth), r*cos(raw_azimuth), y)`, raw azimuth `n*phi`
/// unbounded (matches `bearing_encode_paths.rs`'s `nearest` closure).
fn true_vec(n: usize, total: usize, sign: Sign) -> (f64, f64, f64) {
    let p = HemispherePoint::signed_lift(n, total, sign);
    (p.r * p.azimuth.sin(), p.r * p.azimuth.cos(), p.y)
}

/// Angle in degrees between two (unit-ish) vectors. Identical to
/// `bearing_encode_paths.rs::ang`.
fn ang(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    (a.0 * b.0 + a.1 * b.1 + a.2 * b.2)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// The naive 8-bit-over-`[-1,1]` linear arm — never implemented in the
/// crate; built inline here purely as the comparison reference (i)/(ii)
/// need. `y=-1 -> 0`, `y=0 -> 128` (rounds up), `y=1 -> 255`.
fn naive_linear8_polar(y: f64) -> u8 {
    let y = y.clamp(-1.0, 1.0);
    (((y + 1.0) / 2.0) * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Inverse of `naive_linear8_polar`, for the equal-budget error
/// measurement in (ii).
fn naive_linear8_decode_y(polar: u8) -> f64 {
    (polar as f64 / 255.0) * 2.0 - 1.0
}

/// The house arm's own decode (`polar` -> `y`), matching `residue.rs`'s
/// `encode_signed` encode direction exactly (inverse of `128 + round(y*127)`
/// / `127 - round(|y|*127)`).
fn house_decode_y(polar: u8) -> f64 {
    if polar >= 128 {
        (polar as f64 - 128.0) / 127.0
    } else {
        -((127.0 - polar as f64) / 127.0)
    }
}

// ═════════════════════════════════════════════════════════════════════
// (i) ANTI-VACUITY — the two byte-encoding arms are genuinely different
//     code paths, not one aliased twice.
// ═════════════════════════════════════════════════════════════════════

/// The house 7-bit-magnitude+sign arm (the REAL `encode_signed` polar byte)
/// vs the naive 8-bit-over-`[-1,1]` linear arm must produce DIFFERENT bytes
/// on a real, non-trivial fraction of samples.
///
/// Analytic derivation (full algebra in
/// `equal_budget_ratio_matches_the_analytic_step_sizes`'s doc comment):
/// the two arms' pre-rounding difference is `0.5*(1-y)`, ranging over
/// `(0, 0.5]` for `y in (0,1)` and concentrated near the equator (`y`
/// small) — so disagreement is EXPECTED on a real, but not dominant,
/// fraction, not "almost always" or "almost never". Hand-computed check at
/// `N=8`: 1/8 (12.5%) differ, at the smallest-`y` (equator-most) sample.
/// Bar set to 5% — comfortably below that spot-check and below the
/// continuous-approximation estimate (~16.7%, via
/// `E[0.5*(1-Y)] = 0.5*(1 - 2/3)` for the disk-equal-area `Y` density),
/// while remaining a real, non-zero claim.
///
/// Falsifier: an implementation where the "naive" arm is accidentally
/// computed with the SAME formula as the house arm — two names for one
/// code path — produces a 0% differing fraction and fails this test
/// outright. That is exactly the `[H]`-not-`[G]` gap EV-9 exists to close:
/// a measurement that never actually built a second, independent arm.
#[test]
fn house_and_naive_polar_arms_differ_on_a_stated_minimum_fraction() {
    let enc = ResidueEncoder::new(N);
    let mut differing = 0usize;
    for n in 0..N {
        let y = HemispherePoint::lift(n, N).y;
        let house = enc.encode_signed(PLACE, n, Sign::Pos).polar;
        let naive = naive_linear8_polar(y);
        if house != naive {
            differing += 1;
        }
    }
    let frac = differing as f64 / N as f64;
    assert!(
        frac >= 0.05,
        "house vs naive polar bytes must differ on >=5% of {N} samples \
         (proves two distinct code paths exist); measured {:.4} ({differing}/{N})",
        frac
    );
}

// ═════════════════════════════════════════════════════════════════════
// (ii) EQUAL-BUDGET RATIO — tolerance from the analytic step sizes, not
//      an eyeballed band.
// ═════════════════════════════════════════════════════════════════════

/// Analytic quantization step of the house 7-bit-magnitude+sign arm:
/// `|y|` quantized into 127 levels over `[0,1]`
/// (`mag = round(|y|*127).clamp(0,127)`, `residue.rs:191`) — resolution
/// `1/127` per unit `y`.
const STEP_HOUSE: f64 = 1.0 / 127.0;

/// Analytic quantization step of the naive 8-bit-over-`[-1,1]` arm: `y`
/// quantized into 256 levels (255 steps) over the FULL `[-1,1]` range —
/// resolution `2/255` per unit `y`.
const STEP_NAIVE: f64 = 2.0 / 255.0;

/// For a round-to-nearest uniform quantizer with step `d`, applied to an
/// input whose density is smooth relative to `d` (N=65536 samples spread
/// over ~127-256 levels — hundreds of samples per level, the regime where
/// the dithering approximation holds), the mean ABSOLUTE quantization error
/// converges to `d/4` (the mean of a value uniformly distributed over
/// `[-d/2, d/2]`), independent of the input distribution's shape. So the
/// two arms' mean-error RATIO converges to their step-size ratio.
const ANALYTIC_RATIO: f64 = STEP_HOUSE / STEP_NAIVE; // = 255/254 ~= 1.003937

/// Tolerance band around `ANALYTIC_RATIO`. `N=65536` over ~127-256
/// levels gives a well-converged mean (hundreds of samples per level), so
/// this stays tight relative to the per-latitude-band `0.99-1.02x` range
/// K-12 measured (those bands hold only ~500 samples each — far noisier
/// than the full-`N` mean measured here) while leaving real headroom
/// above pure float noise.
const RATIO_TOLERANCE: f64 = 0.03;

/// The equal-budget ratio, WITH the analytic tolerance derived from
/// `STEP_HOUSE`/`STEP_NAIVE` — the v1's bare `"0.99-1.02x"` was struck
/// as a near-tautology (§8 EV-9: "assertion implied by its own subject",
/// both arms have ~equal step size BY CONSTRUCTION). The bar here is the
/// ANALYTIC ratio derived from the two step sizes, pinned BEFORE the run,
/// not an empirically-eyeballed band.
///
/// Disable-run (orchestrator — the v2 spec's "ASYMMETRIC disable-run"
/// requirement): edit `crates/helix/src/residue.rs`'s `encode_signed`
/// magnitude-scale literal (`127.0` at `residue.rs:191`, in
/// `(p.y.abs() * 127.0).round().clamp(0.0, 127.0) as u8`) to something
/// else, e.g. `63.0` — this changes ONLY the house arm's real
/// behavior; `STEP_HOUSE`/`ANALYTIC_RATIO` are fixed constants computed
/// once in THIS file from the crate's documented design, never read back
/// from the running code, so the measured ratio drifts away from the
/// unmoved analytic bound and this assertion goes red. This is
/// deliberately asymmetric: a SYMMETRIC wiring bug that broke both arms'
/// effective precision by the same factor would leave the measured ratio
/// at ~1.000 and this test alone would not catch it — that is exactly why
/// (i)'s anti-vacuity check (byte-identity, not ratio-based) exists as an
/// independent tripwire.
#[test]
fn equal_budget_ratio_matches_the_analytic_step_sizes() {
    let enc = ResidueEncoder::new(N);
    let (mut sum_house, mut sum_naive) = (0.0f64, 0.0f64);
    for n in 0..N {
        let y_true = HemispherePoint::lift(n, N).y;
        let house_polar = enc.encode_signed(PLACE, n, Sign::Pos).polar;
        let naive_polar = naive_linear8_polar(y_true);
        sum_house += (y_true - house_decode_y(house_polar)).abs();
        sum_naive += (y_true - naive_linear8_decode_y(naive_polar)).abs();
    }
    let mean_house = sum_house / N as f64;
    let mean_naive = sum_naive / N as f64;
    let measured_ratio = mean_house / mean_naive;
    assert!(
        (measured_ratio - ANALYTIC_RATIO).abs() <= RATIO_TOLERANCE,
        "measured ratio {measured_ratio:.6} must sit within {RATIO_TOLERANCE} \
         of the analytic ratio {ANALYTIC_RATIO:.6} \
         (= STEP_HOUSE/STEP_NAIVE = {STEP_HOUSE:.6}/{STEP_NAIVE:.6}); \
         mean_house={mean_house:.6}, mean_naive={mean_naive:.6}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// (iii) THE SPREAD — pole/equator error RATIO, not two point values.
// ═════════════════════════════════════════════════════════════════════

/// Pole-band mean angular error / equator-band mean angular error, per
/// K-12 ("equator (0-5°) 0.112° mean ... pole (85-90°) 3.332° ... ~30x
/// spread"). A v1 that pinned the two means as independent point
/// assertions (`equator_mean ~= 0.112`, `pole_mean ~= 3.332`) could drift
/// TOGETHER under a shared systematic bug (e.g. a constant offset baked
/// into the decode formula) and stay green — the RATIO is the actual
/// finding under test, per the v2 spec's own framing ("independent point
/// assertions can drift together"). Bar loosened to 20x (K-12 measured
/// ~29.75x, i.e. 3.332/0.112) to leave headroom for run-to-run float
/// noise while staying far above any value a symmetric bug could produce.
///
/// Falsifier: swapping which band is the numerator/denominator (see the
/// disable-run below), or a decode bug that makes error latitude-
/// INDEPENDENT (e.g. a constant error regardless of `y`), collapses this
/// ratio toward 1.0 and fails the `>=20x` bar.
///
/// Disable-run (orchestrator): in this test, swap the ratio's numerator
/// and denominator (`eq_mean / pole_mean` instead of `pole_mean /
/// eq_mean`) — the ratio drops well below 1.0 (nowhere near `>=20`),
/// proving the `>=20x` bar is genuinely direction-sensitive rather than
/// something any two positive means would satisfy regardless of which
/// band is which.
#[test]
fn pole_band_error_dominates_equator_band_error_by_the_spread() {
    let enc = ResidueEncoder::new(N);
    let (mut eq_sum, mut eq_n) = (0.0f64, 0usize);
    let (mut pole_sum, mut pole_n) = (0.0f64, 0usize);
    for n in 0..N {
        let p = HemispherePoint::lift(n, N);
        let lat = latitude_deg(p.y);
        let in_equator = EQUATOR_BAND_DEG.contains(&lat);
        let in_pole = POLE_BAND_DEG.contains(&lat);
        if !(in_equator || in_pole) {
            continue;
        }
        let s = enc.encode_signed(PLACE, n, Sign::Pos);
        let truth = true_vec(n, N, Sign::Pos);
        let err = ang(truth, decode(&s));
        if in_equator {
            eq_sum += err;
            eq_n += 1;
        } else {
            pole_sum += err;
            pole_n += 1;
        }
    }
    assert!(
        eq_n > 0 && pole_n > 0,
        "the sampling scheme (N={N}) must populate both bands: eq_n={eq_n}, pole_n={pole_n}"
    );
    let eq_mean = eq_sum / eq_n as f64;
    let pole_mean = pole_sum / pole_n as f64;
    let spread = pole_mean / eq_mean;
    assert!(
        spread >= 20.0,
        "pole/equator mean-error spread must be >=20x (K-12 measured ~29.75x); \
         got {spread:.2}x ({pole_n} pole samples mean {pole_mean:.4} deg, \
         {eq_n} equator samples mean {eq_mean:.4} deg)"
    );
}

// ═════════════════════════════════════════════════════════════════════
// (iv) COMPARATIVE — palette-`circular()` beats nearest-`(n,sign)` on the
//      SAME sample.
// ═════════════════════════════════════════════════════════════════════

/// The nearest-`(n,sign)` lattice search, matching
/// `bearing_encode_paths.rs`'s "Path A" exactly (the same O(N) search over
/// the golden-spiral lattice, picking the `(n, sign)` whose `true_vec` is
/// angularly closest to `target`, then decoding via the real
/// `encode_signed`).
fn nearest_encode(enc: &ResidueEncoder, target: (f64, f64, f64)) -> Signed360 {
    let mut best = (f64::MAX, 0usize, Sign::Pos);
    for n in 0..N {
        for sg in [Sign::Pos, Sign::Neg] {
            let v = true_vec(n, N, sg);
            let e = ang(target, v);
            if e < best.0 {
                best = (e, n, sg);
            }
        }
    }
    enc.encode_signed(PLACE, best.1, best.2)
}

/// The palette-`circular()` azimuth arm: quantize the true azimuth into a
/// `PALETTE_SIZE` (256, i.e. 8-bit — the domain `DistanceLut::circular()`
/// operates over) circular index, decode at the bucket's own angle — a
/// PURE 1-D azimuth quantization, elevation held at 0 to isolate the axis
/// K-13 measures.
fn palette_circular_encode(az_true: f64) -> (f64, f64, f64) {
    let step = TAU / PALETTE_SIZE as f64;
    let raw_idx = (az_true.rem_euclid(TAU) / step).round() as i64;
    let idx = raw_idx.rem_euclid(PALETTE_SIZE as i64);
    let az_decoded = idx as f64 * step;
    (az_decoded.sin(), az_decoded.cos(), 0.0)
}

/// Comparative K-13: on the SAME azimuth targets, the u8-palette
/// `circular()` arm's mean angular error must be LOWER than the
/// nearest-`(n,sign)` lattice arm's — the mechanism K-7/K-13 both name:
/// the golden-spiral lattice couples latitude and azimuth through ONE
/// index, so near the horizon (elevation 0, the axis this test sweeps)
/// its azimuth resolution is starved by the sparse near-equator lattice
/// density, while the palette arm writes azimuth as an independent field.
/// (Recorded, K-7/K-13: nearest-n ~0.972° mean, palette-circular ~0.352°
/// mean.)
///
/// Falsifier: a palette arm that silently ignores the azimuth input (a
/// constant decode) would fail this in the OBVIOUS direction — its error
/// would be a large, target-independent average, losing badly to
/// nearest-n rather than winning; the strict `<` catches that as readily
/// as a genuine regression in the palette encode.
///
/// Disable-run (orchestrator): in `palette_circular_encode`, zero the
/// `az_true` input before quantizing (`let raw_idx = (0.0_f64.rem_euclid(TAU)
/// / step).round() as i64;`) — the palette arm now ignores azimuth
/// entirely, its mean error rises to roughly the average angular distance
/// from a fixed point over a swept circle, and the `<` assertion goes
/// red.
#[test]
fn palette_circular_azimuth_beats_nearest_n_lattice_search() {
    let enc = ResidueEncoder::new(N);
    let (mut sum_palette, mut sum_nearest) = (0.0f64, 0.0f64);
    for i in 0..AZ_SAMPLES {
        let az_true = (i as f64 / AZ_SAMPLES as f64) * TAU;
        let target = (az_true.sin(), az_true.cos(), 0.0);

        let palette_decoded = palette_circular_encode(az_true);
        sum_palette += ang(target, palette_decoded);

        let nearest = nearest_encode(&enc, target);
        sum_nearest += ang(target, decode(&nearest));
    }
    let mean_palette = sum_palette / AZ_SAMPLES as f64;
    let mean_nearest = sum_nearest / AZ_SAMPLES as f64;
    assert!(
        mean_palette < mean_nearest,
        "palette-circular azimuth mean {mean_palette:.4} deg must be LOWER \
         than nearest-(n,sign) mean {mean_nearest:.4} deg \
         (K-7 recorded ~0.972 deg, K-13 recorded ~0.352 deg)"
    );
}

// ═════════════════════════════════════════════════════════════════════
// (v) SAMPLING SCHEME — pinned explicitly, not trusted by reputation.
// ═════════════════════════════════════════════════════════════════════

/// Azimuth is genuinely SWEPT (not degenerate/constant) within each
/// latitude band `pole_band_error_dominates_equator_band_error_by_the_spread`
/// draws from. Golden-angle stepping makes near-full coverage likely (K-3:
/// 256/256 coarse buckets over the full `N=65536` sweep), but "likely" is
/// not "pinned in code" — this asserts it directly for the ~500-sample
/// bands (iii) actually uses, rather than trusting the mechanism by
/// assumption.
///
/// Bucketing matches `signed360_claims.rs::azimuth_spans_the_full_circle_
/// not_merely_varies` exactly (`(a >> 8) as usize` on the REAL encoded
/// `u16` azimuth field — no float-division bucketing, no wrap-then-scale
/// edge cases) — reusing the crate's own K-3 coverage technique rather
/// than reinventing it.
///
/// Bar (200/256 coarse buckets touched per band): a coupon-collector
/// estimate for ~500 i.i.d. uniform draws into 256 bins gives ~220 covered
/// buckets; golden-angle sequences are LOWER discrepancy than i.i.d.
/// (better equidistributed), so coverage should exceed that estimate. 200
/// leaves margin below it while still requiring broad coverage, not a
/// handful of buckets.
///
/// Falsifier: an off-by-something bug that fixes `n` (and hence azimuth)
/// near-constant within a band collapses `covered` toward 1 and fails the
/// `>=200` bar.
///
/// Disable-run (orchestrator): replace `s.azimuth` below with a constant
/// (e.g. `0u16`) before bucketing — `covered` drops to 1 for both bands
/// and the assertion goes red.
#[test]
fn azimuth_actually_sweeps_within_each_latitude_band() {
    let enc = ResidueEncoder::new(N);
    let mut eq_buckets = [false; 256];
    let mut pole_buckets = [false; 256];
    let (mut eq_n, mut pole_n) = (0usize, 0usize);
    for n in 0..N {
        let p = HemispherePoint::lift(n, N);
        let lat = latitude_deg(p.y);
        let in_equator = EQUATOR_BAND_DEG.contains(&lat);
        let in_pole = POLE_BAND_DEG.contains(&lat);
        if !(in_equator || in_pole) {
            continue;
        }
        let s = enc.encode_signed(PLACE, n, Sign::Pos);
        let bucket = (s.azimuth >> 8) as usize;
        if in_equator {
            eq_buckets[bucket] = true;
            eq_n += 1;
        } else {
            pole_buckets[bucket] = true;
            pole_n += 1;
        }
    }
    assert!(
        eq_n > 0 && pole_n > 0,
        "the sampling scheme (N={N}) must populate both bands: eq_n={eq_n}, pole_n={pole_n}"
    );
    let eq_covered = eq_buckets.iter().filter(|b| **b).count();
    let pole_covered = pole_buckets.iter().filter(|b| **b).count();
    assert!(
        eq_covered >= 200,
        "equator band ({eq_n} samples) must sweep azimuth broadly: \
         only {eq_covered}/256 coarse buckets hit"
    );
    assert!(
        pole_covered >= 200,
        "pole band ({pole_n} samples) must sweep azimuth broadly: \
         only {pole_covered}/256 coarse buckets hit"
    );
}
