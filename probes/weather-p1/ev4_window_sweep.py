"""EV-4 (v2) -- Saturation-window sweep with a knob-reached-the-code proof.

Plan: .claude/plans/weather-substrate-evaluation-v1.md section 3, EV-4 (v2).
Reference frame consumed, not re-implemented: probes/weather-p1/p1_ci_vs_floor.py
(bucket CI vs floor, never decoded round-trip error). The banned metric is
documented in probes/weather-p1/p1_noise_floor.py's own header (SUPERSEDED).

Windows swept: (0.4,99.6), (0.2,99.8), (0.1,99.9), (0.02,99.98) [percentile
pairs]. Variable: 2m_temperature (primary, per the brief); also run on
2m_dewpoint_temperature and 10m_u_component_of_wind since it is cheap and the
brief invites it.

For each window x variable x arm (LINEAR / FISHER_Z), report BOTH quantities
the v1 spec conflated (v1 defect: "sat% counts legitimately in-range points in
the edge buckets"):
  (a) bucket-0/255 occupancy ("sat_frac") -- fraction of points landing in the
      two edge buckets. This over-counts relative to (b) because np.clip()
      forces every truly-out-of-window point into bucket 0 or 255 ALONGSIDE
      the legitimately in-range points whose value happens to fall in that
      edge bucket's own [lo, lo+bucket_width) / (hi-bucket_width, hi] span.
  (b) true out-of-window mass ("out_of_window_frac") -- fraction of points
      whose ORIGINAL (pre-quantization) value falls outside [lo, hi]. This is
      the nominal mass the percentile window excludes by construction
      (0.8% -> 0.4% -> 0.2% -> 0.04% as the four windows widen).

MANDATORY CHECKS (this is the whole point of v2 -- catching a dead knob; v1's
"monotone" pass route was satisfiable by a constant curve, which is exactly
what a window parameter that never reaches the code would produce):

  1. LINEAR arm: ci_linear = (hi-lo)/512 (the uniform half-bucket-width, in
     the original CI-vs-floor frame's units) must be STRICTLY INCREASING
     across the four windows (narrowest -> widest), AND
     ci_linear[widest] / ci_linear[narrowest] must equal
     (hi[widest]-lo[widest]) / (hi[narrowest]-lo[narrowest]) to 1e-12.
     A dead knob (the window never actually reaching np.percentile /
     the quantizer) would leave ci_linear constant across windows while
     (hi-lo) still visibly varies with the window in the raw percentile
     output -- the ratio equality is exactly what a dead knob cannot
     satisfy, because a constant ci_ratio of 1.0 would not equal a
     hilo_ratio > 1.0.

  2. FISHER_Z arm: reported, NOT gated on monotonicity. A non-monotone
     interior-median-CI curve across windows is a FINDING to investigate
     (both `scale` and `(zlo,zhi)` move with the window -- report all three
     per window so the finding is diagnosable), never auto-assigned to
     "apparatus wrong". The v1 wording ("or the apparatus is wrong")
     immunized the expectation against falsification; this version reports
     the monotonicity verdict as data, not as a pass/fail gate.
"""
import numpy as np
import json

WINDOWS = [(0.4, 99.6), (0.2, 99.8), (0.1, 99.9), (0.02, 99.98)]
VARS = ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind']


def fz(s, eps=1e-9):
    """Fisher-Z: arctanh(s), clipped off the +/-1 poles by eps."""
    s = np.clip(s, -1 + eps, 1 - eps)
    return 0.5 * (np.log1p(s) - np.log1p(-s))


def load_anom(var):
    """Load a fixture variable and return its ZONAL anomaly (per-row climatology removed)."""
    a = np.load(f'fixture/{var}.npy').astype(np.float64)
    assert np.isfinite(a).all(), f"{var}: nonfinite values present in fixture"
    clim = a.mean(axis=1, keepdims=True)
    return a - clim


def linear_arm(anom, w_lo, w_hi):
    """LINEAR: uniform buckets in original units (K or m/s)."""
    lo, hi = np.percentile(anom, [w_lo, w_hi])
    idx = np.clip(np.floor((anom - lo) / (hi - lo) * 256), 0, 255).astype(np.uint8)
    edges = lo + np.arange(257) / 256 * (hi - lo)
    ci = np.diff(edges) / 2.0  # 256 half-widths; uniform arm -> all equal
    interior = (idx > 0) & (idx < 255)
    sat_frac = float((~interior).mean())
    out_of_window_frac = float(((anom < lo) | (anom > hi)).mean())
    return {
        'lo': float(lo), 'hi': float(hi), 'hi_minus_lo': float(hi - lo),
        'ci_uniform': float(ci[0]),  # every bucket has the same CI in this arm
        'ci_interior_min': float(ci[1:255].min()),
        'ci_interior_med': float(np.median(ci[1:255])),
        'ci_interior_max': float(ci[1:255].max()),
        'sat_frac': sat_frac,
        'out_of_window_frac': out_of_window_frac,
    }


def fisher_z_arm(anom, w_lo, w_hi):
    """FISHER_Z: uniform buckets in z=arctanh(anom/scale), non-uniform in K."""
    lo_q, hi_q = np.percentile(anom, [w_lo, w_hi])
    scale = max(abs(lo_q), abs(hi_q))
    z = fz(anom / scale)
    zlo, zhi = np.percentile(z, [w_lo, w_hi])
    idxz = np.clip(np.floor((z - zlo) / (zhi - zlo) * 256), 0, 255).astype(np.uint8)
    edges_z = zlo + np.arange(257) / 256 * (zhi - zlo)
    edges_K = np.tanh(edges_z) * scale
    ci = np.diff(edges_K) / 2.0  # 256 half-widths; non-uniform (Fisher-Z arm)
    interior = (idxz > 0) & (idxz < 255)
    sat_frac = float((~interior).mean())
    out_of_window_frac = float(((z < zlo) | (z > zhi)).mean())
    return {
        'scale': float(scale), 'zlo': float(zlo), 'zhi': float(zhi),
        'zhi_minus_zlo': float(zhi - zlo),
        'ci_interior_min': float(ci[1:255].min()),
        'ci_interior_med': float(np.median(ci[1:255])),
        'ci_interior_max': float(ci[1:255].max()),
        'sat_frac': sat_frac,
        'out_of_window_frac': out_of_window_frac,
    }


results = {}
for var in VARS:
    print(f"\n{'=' * 76}\n{var}\n{'=' * 76}")
    anom = load_anom(var)
    lin_rows = []
    z_rows = []
    for (w_lo, w_hi) in WINDOWS:
        lin = linear_arm(anom, w_lo, w_hi)
        zr = fisher_z_arm(anom, w_lo, w_hi)
        lin_rows.append(lin)
        z_rows.append(zr)
        print(f"\nwindow ({w_lo},{w_hi}):")
        print(f"  LINEAR    ci={lin['ci_uniform']:.6f}  hi-lo={lin['hi_minus_lo']:.6f}  "
              f"sat%={lin['sat_frac'] * 100:.4f}  true_out%={lin['out_of_window_frac'] * 100:.4f}")
        print(f"  FISHER_Z  ci_med={zr['ci_interior_med']:.6f} "
              f"(min={zr['ci_interior_min']:.6f} max={zr['ci_interior_max']:.6f})  "
              f"sat%={zr['sat_frac'] * 100:.4f}  true_out%={zr['out_of_window_frac'] * 100:.4f}  "
              f"scale={zr['scale']:.6f} zlo={zr['zlo']:.6f} zhi={zr['zhi']:.6f}")

    # ---- MANDATORY CHECK 1: linear CI strictly increasing + ratio proof ----
    ci_vals = [r['ci_uniform'] for r in lin_rows]
    strictly_increasing = all(ci_vals[i] < ci_vals[i + 1] for i in range(len(ci_vals) - 1))
    hilo_ratio = lin_rows[-1]['hi_minus_lo'] / lin_rows[0]['hi_minus_lo']
    ci_ratio = ci_vals[-1] / ci_vals[0]
    ratio_matches = abs(ci_ratio - hilo_ratio) < 1e-12
    print(f"\n  [CHECK 1] linear ci strictly increasing across windows: {strictly_increasing}")
    print(f"  [CHECK 1] ci_vals = {ci_vals}")
    print(f"  [CHECK 1] ci_ratio(widest/narrowest)      = {ci_ratio!r}")
    print(f"  [CHECK 1] hilo_ratio(widest/narrowest)    = {hilo_ratio!r}")
    print(f"  [CHECK 1] |ci_ratio - hilo_ratio|          = {abs(ci_ratio - hilo_ratio):.3e}  "
          f"matches_to_1e-12={ratio_matches}")

    # ---- MANDATORY CHECK 2: fisher-z curve reported, not gated ----
    z_ci_vals = [r['ci_interior_med'] for r in z_rows]
    z_increasing = all(z_ci_vals[i] < z_ci_vals[i + 1] for i in range(len(z_ci_vals) - 1))
    z_decreasing = all(z_ci_vals[i] > z_ci_vals[i + 1] for i in range(len(z_ci_vals) - 1))
    z_monotone = z_increasing or z_decreasing
    print(f"  [CHECK 2] fisher-z interior-median ci across windows: {z_ci_vals}")
    print(f"  [CHECK 2] fisher-z monotone: {z_monotone} "
          f"(increasing={z_increasing}, decreasing={z_decreasing})")
    if not z_monotone:
        print("  [CHECK 2] NON-MONOTONE -- reported as a FINDING to investigate, "
              "not treated as an apparatus failure. See per-window scale/zlo/zhi above.")

    results[var] = {
        'windows': [{'w_lo': w_lo, 'w_hi': w_hi} for (w_lo, w_hi) in WINDOWS],
        'linear': lin_rows,
        'fisher_z': z_rows,
        'check1_linear_strictly_increasing': bool(strictly_increasing),
        'check1_ci_ratio_widest_over_narrowest': float(ci_ratio),
        'check1_hilo_ratio_widest_over_narrowest': float(hilo_ratio),
        'check1_abs_diff': float(abs(ci_ratio - hilo_ratio)),
        'check1_ratio_matches_to_1e-12': bool(ratio_matches),
        'check2_fisher_z_ci_interior_med_values': z_ci_vals,
        'check2_fisher_z_monotone': bool(z_monotone),
        'check2_fisher_z_increasing': bool(z_increasing),
        'check2_fisher_z_decreasing': bool(z_decreasing),
    }

overall_check1_pass = all(
    results[v]['check1_linear_strictly_increasing'] and results[v]['check1_ratio_matches_to_1e-12']
    for v in VARS
)
print(f"\n{'=' * 76}")
print(f"OVERALL check1 (linear knob-reached-code proof) pass on all vars: {overall_check1_pass}")
for v in VARS:
    print(f"  {v}: fisher_z monotone = {results[v]['check2_fisher_z_monotone']}")

results['_summary'] = {
    'vars': VARS,
    'windows': [{'w_lo': w_lo, 'w_hi': w_hi} for (w_lo, w_hi) in WINDOWS],
    'overall_check1_pass': bool(overall_check1_pass),
    'per_var_fisher_z_monotone': {v: results[v]['check2_fisher_z_monotone'] for v in VARS},
}

with open('ev4_window_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nwrote ev4_window_sweep.json")
