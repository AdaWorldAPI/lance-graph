//! Drift statistics — **re-encode convergence** and **correction-delta
//! summaries**.
//!
//! NOT a pillar. A calibration toolkit lifted out of the thinking-engine lab
//! crate under D-TEH-3 (`thinking-engine-harvest-closure-v1` §1d), per the
//! ruling that scientifically calibrated math lives in `jc`
//! (`E-JC-IS-THE-HOME-OF-ALL-CALIBRATED-MATH-1`). The lab crate keeps the
//! codec-specific GLUE (which round trip: BF16, γ+φ, the full chain) and calls
//! this module for the statistic; it carries no private copy.
//!
//! # Re-encode drift
//!
//! A codec is *re-encode safe* if iterating `decode(encode(x))` stabilises:
//! the error against the ORIGINAL value stops changing after a bounded number
//! of round trips instead of accumulating. The test is the same one an ICC
//! colour profile has to pass — `encode(decode(encode(x))) == encode(x)`.
//! [`reencode_drift`] runs the iteration for one value against any round-trip
//! closure and reports where it converged; [`reencode_batch`] aggregates a
//! sweep.
//!
//! The convergence criterion is preserved from the lifted source: the run is
//! converged at iteration `i > 0` when `|e_i − e_{i−1}| < `
//! [`CONVERGENCE_EPS`], where `e_i` is the absolute error against the original
//! `f64` value (NOT against the previous iterate — a codec that collapses to a
//! wrong fixed point still counts as converged, and its `max_error` says how
//! wrong). The remaining history is filled with the converged error so
//! `error_history.len() == max_iterations` always holds.
//!
//! # Correction-delta summary
//!
//! [`delta_summary`] is the descriptive battery for a set of correction deltas
//! (the lifted use: `cos(activated) − cos(raw)` per centroid pair): mean,
//! mean absolute, max absolute, population standard deviation, and the
//! fraction of deltas above two caller-supplied thresholds ("material" and
//! "large"). The thresholds are parameters, not constants, so a caller cannot
//! inherit a cut-off that was tuned for a different table.

use std::fmt;

/// Convergence tolerance on consecutive absolute errors, in the units of the
/// original `f64` value. Preserved from the lifted source.
pub const CONVERGENCE_EPS: f64 = 1e-15;

/// Result of one re-encode drift run.
#[derive(Clone, Debug, PartialEq)]
pub struct ReencodeDrift {
    /// Iteration at which the error stopped changing; `max_iterations` if it
    /// never did.
    pub converged_at: usize,
    /// Largest absolute error against the original value over the run.
    pub max_error: f64,
    /// Absolute error at the last iteration.
    pub final_error: f64,
    /// Absolute error per iteration; always `max_iterations` long (the tail
    /// after convergence is filled with the converged error).
    pub error_history: Vec<f64>,
    /// `converged_at < max_iterations`.
    pub safe: bool,
    /// Caller-supplied codec label, for reports.
    pub codec: String,
}

impl fmt::Display for ReencodeDrift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} after {} iterations (max_err={:.2e}, final_err={:.2e})",
            self.codec,
            if self.safe { "SAFE" } else { "UNSAFE" },
            self.converged_at,
            self.max_error,
            self.final_error
        )
    }
}

/// Iterate `round_trip` on `value` up to `max_iterations` times and report
/// where the error against `value` stopped changing.
///
/// `round_trip` is one `decode(encode(x))` pass in the codec's own working
/// precision (`f32`, as every lifted codec operates); the error is measured in
/// `f64` against the ORIGINAL `value`, per the module doc.
///
/// `max_iterations == 0` yields an empty history, `converged_at == 0` and
/// `safe == false` — nothing was run, so nothing was proven.
///
/// ```
/// use jc::drift::reencode_drift;
/// // The identity codec is trivially safe: the error is 0 from the first
/// // pass and stops changing at the second.
/// let r = reencode_drift(0.123_456_789, 16, "identity", |x| x);
/// assert!(r.safe);
/// assert_eq!(r.converged_at, 1);
/// assert!(r.max_error < 1e-7); // only the f64 → f32 cast of the input
/// ```
pub fn reencode_drift(
    value: f64,
    max_iterations: usize,
    codec: impl Into<String>,
    mut round_trip: impl FnMut(f32) -> f32,
) -> ReencodeDrift {
    let mut current = value as f32;
    let mut errors: Vec<f64> = Vec::with_capacity(max_iterations);
    let mut converged_at = max_iterations;

    for i in 0..max_iterations {
        let decoded = round_trip(current);
        let error = (f64::from(decoded) - value).abs();
        errors.push(error);
        if i > 0 && (errors[i] - errors[i - 1]).abs() < CONVERGENCE_EPS {
            converged_at = i;
            errors.resize(max_iterations, error);
            break;
        }
        current = decoded;
    }

    let max_error = errors.iter().copied().fold(0.0_f64, f64::max);
    let final_error = errors.last().copied().unwrap_or(0.0);

    ReencodeDrift {
        converged_at,
        max_error,
        final_error,
        error_history: errors,
        safe: converged_at < max_iterations,
        codec: codec.into(),
    }
}

/// Aggregate of a sweep of [`reencode_drift`] runs.
#[derive(Clone, Debug, PartialEq)]
pub struct DriftBatch {
    /// Every run converged.
    pub all_safe: bool,
    /// The run with the largest `max_error`; `None` for an empty sweep.
    pub worst: Option<ReencodeDrift>,
    /// Largest `converged_at` over the sweep.
    pub max_converged_at: usize,
    /// Number of safe runs.
    pub safe_count: usize,
    /// Number of runs.
    pub total: usize,
}

/// Run `drift` on every value and aggregate.
pub fn reencode_batch(values: &[f64], mut drift: impl FnMut(f64) -> ReencodeDrift) -> DriftBatch {
    let mut all_safe = true;
    let mut worst: Option<ReencodeDrift> = None;
    let mut max_converged_at = 0;
    let mut safe_count = 0;
    for &v in values {
        let r = drift(v);
        if r.safe {
            safe_count += 1;
        } else {
            all_safe = false;
        }
        max_converged_at = max_converged_at.max(r.converged_at);
        if worst.as_ref().is_none_or(|w| r.max_error > w.max_error) {
            worst = Some(r);
        }
    }
    DriftBatch {
        all_safe,
        worst,
        max_converged_at,
        safe_count,
        total: values.len(),
    }
}

/// Descriptive summary of a set of correction deltas.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaSummary {
    /// Number of deltas.
    pub count: usize,
    /// Arithmetic mean (signed).
    pub mean: f64,
    /// Mean absolute delta.
    pub mean_abs: f64,
    /// Largest absolute delta.
    pub max_abs: f64,
    /// Population standard deviation (divisor `n`).
    pub std_dev: f64,
    /// Fraction of deltas with `|δ| > material_threshold`.
    pub material_fraction: f64,
    /// Fraction of deltas with `|δ| > large_threshold`.
    pub large_fraction: f64,
    /// The "material" cut-off the fractions were computed against.
    pub material_threshold: f64,
    /// The "large" cut-off the fractions were computed against.
    pub large_threshold: f64,
}

impl DeltaSummary {
    /// The summary of no deltas at all: every statistic is zero and `count`
    /// is `0`. For callers that must render something for an empty sample;
    /// [`delta_summary`] itself returns `None` there so the two cases are
    /// distinguishable at the call site.
    pub fn empty(material_threshold: f64, large_threshold: f64) -> Self {
        Self {
            count: 0,
            mean: 0.0,
            mean_abs: 0.0,
            max_abs: 0.0,
            std_dev: 0.0,
            material_fraction: 0.0,
            large_fraction: 0.0,
            material_threshold,
            large_threshold,
        }
    }
}

/// Summarise `deltas` against two magnitude cut-offs.
///
/// Returns `None` for an empty slice or any non-finite delta (the no-`NaN`
/// contract shared with [`crate::reliability`]).
///
/// ```
/// use jc::drift::delta_summary;
/// let s = delta_summary(&[0.02, -0.005, 0.15, 0.0], 0.01, 0.1).unwrap();
/// assert_eq!(s.count, 4);
/// assert!((s.material_fraction - 0.5).abs() < 1e-12); // 0.02 and 0.15
/// assert!((s.large_fraction - 0.25).abs() < 1e-12); // 0.15 only
/// ```
pub fn delta_summary(
    deltas: &[f64],
    material_threshold: f64,
    large_threshold: f64,
) -> Option<DeltaSummary> {
    if deltas.is_empty() || !deltas.iter().all(|d| d.is_finite()) {
        return None;
    }
    let n = deltas.len() as f64;
    let mean = deltas.iter().sum::<f64>() / n;
    let mean_abs = deltas.iter().map(|d| d.abs()).sum::<f64>() / n;
    let max_abs = deltas.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
    let variance = deltas.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n;
    let material = deltas
        .iter()
        .filter(|d| d.abs() > material_threshold)
        .count();
    let large = deltas.iter().filter(|d| d.abs() > large_threshold).count();
    Some(DeltaSummary {
        count: deltas.len(),
        mean,
        mean_abs,
        max_abs,
        std_dev: variance.sqrt(),
        material_fraction: material as f64 / n,
        large_fraction: large as f64 / n,
        material_threshold,
        large_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── re-encode drift ─────────────────────────────────────────────────

    /// Disable: skip the `|e_i − e_{i−1}| < EPS` check. The identity codec
    /// then reports `converged_at == max_iterations` and `safe == false`.
    #[test]
    fn identity_round_trip_converges_at_one_and_fills_the_history() {
        let r = reencode_drift(0.25, 32, "identity", |x| x);
        assert!(r.safe);
        assert_eq!(r.converged_at, 1);
        assert_eq!(r.error_history.len(), 32, "tail must be filled");
        assert_eq!(r.max_error, 0.0, "0.25 is exact in f32");
        assert_eq!(r.final_error, 0.0);
    }

    /// The other side: a codec that multiplies by (1 + 1e-3) on every pass
    /// never stabilises. Anti-vacuity: the error grows STRICTLY at every
    /// step, so "never converged" is the run's real shape, not a tolerance
    /// accident.
    #[test]
    fn a_multiplicative_drift_never_converges_and_is_unsafe() {
        let r = reencode_drift(0.5, 64, "drift", |x| x * 1.001);
        assert!(!r.safe);
        assert_eq!(r.converged_at, 64);
        assert!(
            r.error_history.windows(2).all(|w| w[1] > w[0]),
            "error must grow monotonically under a multiplicative drift"
        );
        assert_eq!(r.final_error, r.max_error);
    }

    /// Two-sided on `max_iterations`: a damped codec (x → ½ + ½(x − ½))
    /// converges after MORE than one pass but well within 64; the identical
    /// run with the budget set to exactly its convergence point is unsafe.
    /// Disable: fill `converged_at` with 0 instead of `i` → the first
    /// assertion on `> 1` goes red.
    #[test]
    fn a_damped_codec_converges_late_and_the_budget_binds() {
        let damped = |x: f32| 0.5 + (x - 0.5) * 0.5;
        let r = reencode_drift(0.0, 64, "damped", damped);
        assert!(r.safe, "{r}");
        assert!(
            r.converged_at > 1,
            "converged too early: {}",
            r.converged_at
        );
        assert!(r.converged_at < 64);
        // The fixed point is 0.5; the error against the original 0.0 tends
        // to 0.5 from below, so the run's max is its final value.
        assert!(approx(r.final_error, 0.5, 1e-6), "{r}");
        assert_eq!(r.max_error, r.final_error);

        let tight = reencode_drift(0.0, r.converged_at, "damped", damped);
        assert!(!tight.safe, "the budget must bind: {tight}");
        assert_eq!(tight.converged_at, r.converged_at);
    }

    #[test]
    fn a_zero_budget_proves_nothing() {
        let r = reencode_drift(0.3, 0, "identity", |x| x);
        assert!(!r.safe);
        assert!(r.error_history.is_empty());
        assert_eq!(r.max_error, 0.0);
    }

    /// The batch aggregate reports the worst run, the count, and a `false`
    /// `all_safe` as soon as one value drifts. Disable: `all_safe = true`
    /// unconditionally → red.
    #[test]
    fn batch_aggregates_the_worst_run_and_the_safe_count() {
        let values = [0.1, 0.2, 0.3];
        let b = reencode_batch(&values, |v| {
            // Only the middle value drifts.
            if approx(v, 0.2, 1e-12) {
                reencode_drift(v, 8, "drift", |x| x * 1.01)
            } else {
                reencode_drift(v, 8, "identity", |x| x)
            }
        });
        assert!(!b.all_safe);
        assert_eq!(b.total, 3);
        assert_eq!(b.safe_count, 2);
        assert_eq!(b.max_converged_at, 8);
        let worst = b.worst.expect("non-empty sweep has a worst run");
        assert_eq!(worst.codec, "drift");

        let empty = reencode_batch(&[], |v| reencode_drift(v, 8, "identity", |x| x));
        assert!(empty.all_safe);
        assert!(empty.worst.is_none());
        assert_eq!(empty.total, 0);
    }

    // ── delta summary ───────────────────────────────────────────────────

    /// Hand-computed: mean 0.04125, mean|δ| 0.04375, max|δ| 0.15,
    /// population σ = √(0.0161189/4) ≈ 0.0634801.
    #[test]
    fn delta_summary_matches_a_hand_computed_fixture() {
        let s = delta_summary(&[0.02, -0.005, 0.15, 0.0], 0.01, 0.1).unwrap();
        assert_eq!(s.count, 4);
        assert!(approx(s.mean, 0.04125, 1e-12));
        assert!(approx(s.mean_abs, 0.04375, 1e-12));
        assert!(approx(s.max_abs, 0.15, 1e-12));
        assert!(approx(s.std_dev, 0.063_480_1, 1e-6), "σ was {}", s.std_dev);
        assert!(approx(s.material_fraction, 0.5, 1e-12));
        assert!(approx(s.large_fraction, 0.25, 1e-12));
        assert_eq!(s.material_threshold, 0.01);
        assert_eq!(s.large_threshold, 0.1);
    }

    /// The cut-offs are live parameters, not decoration: raising the material
    /// threshold above every delta silences the fraction, lowering it to zero
    /// admits every non-zero delta. Disable: hardcode 0.01 / 0.1 inside →
    /// both halves go red.
    #[test]
    fn thresholds_are_load_bearing_in_both_directions() {
        let deltas = [0.02, -0.005, 0.15, 0.0];
        let strict = delta_summary(&deltas, 0.2, 0.5).unwrap();
        assert_eq!(strict.material_fraction, 0.0);
        assert_eq!(strict.large_fraction, 0.0);
        let loose = delta_summary(&deltas, 0.0, 0.0).unwrap();
        assert!(
            approx(loose.material_fraction, 0.75, 1e-12),
            "three non-zero deltas"
        );
        assert!(approx(loose.large_fraction, 0.75, 1e-12));
    }

    #[test]
    fn empty_or_non_finite_deltas_return_none() {
        assert_eq!(delta_summary(&[], 0.01, 0.1), None);
        assert_eq!(delta_summary(&[0.1, f64::NAN], 0.01, 0.1), None);
        assert_eq!(delta_summary(&[f64::INFINITY], 0.01, 0.1), None);
        let e = DeltaSummary::empty(0.01, 0.1);
        assert_eq!(e.count, 0);
        assert_eq!(e.material_threshold, 0.01);
    }
}
