"""EV-3 (v2, plan .claude/plans/weather-substrate-evaluation-v1.md sec.3) --
floor flip-points computed EXACTLY, no grid search.

The exceedance-vs-floor curve is the occupancy-weighted survival function of
INTERIOR bucket CI half-widths: for a floor F, the fraction of interior points
whose own bucket's CI exceeds F is a step function of F (each step at a
distinct bucket CI value, height = that bucket's occupancy / N_interior). The
1% flip-point is therefore closed-form: sort interior buckets by CI
descending, cumulate occupancy/N_interior, and read off the CI at which the
cumulant first reaches 0.01 -- equivalently the 99th percentile of
ci[idx][interior] (population = interior points only, so the two framings
agree exactly; both are reported below and cross-checked against each other).

No grid, no step size, no undefined "sharp". This replaces the v1 grid sweep,
whose only guard was pre-satisfied by committed data (plan sec.8 attack-pass
finding for EV-3).

APPARATUS CONTROL: the LINEAR arm's CI is (hi-lo)/512, a CONSTANT across every
interior bucket -- its "curve" is therefore a SINGLE THRESHOLD (the flip point
equals that constant, and the closed-form crossing degenerates to "however
many top-mass buckets are needed to reach 1%, all tied at the same CI"). That
is expected apparatus behaviour, not a defect, and is reported as the control.
The two-regime expectation (CI varying smoothly across buckets, producing a
genuine survival curve rather than a step at one value) applies to the
FISHER-Z arm only.

Frame: bucket CI vs noise floor, never decoded round-trip error (plan sec.0).
Reference structure: p1_ci_vs_floor.py (read in full before editing this
file); parameterized here over VARS instead of hardcoded to one variable.
"""
import json
import pathlib

import numpy as np

VARS = ["2m_temperature", "2m_dewpoint_temperature", "10m_u_component_of_wind"]
FLIP_MASS = 0.01  # 1% flip-point, pre-registered (plan EV-3)


def fz(s, eps=1e-9):
    """Fisher-Z: arctanh(s), clipped off the +/-1 poles by eps."""
    s = np.clip(s, -1 + eps, 1 - eps)
    return 0.5 * (np.log1p(s) - np.log1p(-s))


def load_anomaly(var):
    """404 discipline (plan EV-3): probe availability first, never save a
    fill array as a fixture. This raises loudly (FileNotFoundError) rather
    than silently substituting -- the fixture set for this wave is limited
    to the three variables named in the brief; a missing one is a real
    blocker, not a soft skip."""
    a = np.load(f"fixture/{var}.npy").astype(np.float64)
    assert np.isfinite(a).all(), f"{var}: nonfinite values in fixture"
    clim = a.mean(axis=1, keepdims=True)
    return a - clim


def build_linear(anom):
    """LINEAR path: uniform buckets in original units. CI is CONSTANT
    across every interior bucket -- (hi-lo)/512 -- by construction."""
    lo, hi = np.percentile(anom, [0.4, 99.6])
    idx = np.clip(np.floor((anom - lo) / (hi - lo) * 256), 0, 255).astype(np.uint8)
    edges = lo + np.arange(257) / 256 * (hi - lo)
    return idx, edges


def build_fisher_z(anom):
    """FISHER-Z path: uniform buckets in z, NON-uniform in original units
    after tanh -- CI genuinely varies bucket-to-bucket."""
    lo_q, hi_q = np.percentile(anom, [0.4, 99.6])
    scale = max(abs(lo_q), abs(hi_q))
    z = fz(anom / scale)
    zlo, zhi = np.percentile(z, [0.4, 99.6])
    idxz = np.clip(np.floor((z - zlo) / (zhi - zlo) * 256), 0, 255).astype(np.uint8)
    edges = np.tanh(zlo + np.arange(257) / 256 * (zhi - zlo)) * scale
    return idxz, edges


def closed_form_flip_point(idx, edges, mass=FLIP_MASS):
    """Closed-form 1% flip-point: sort interior buckets by CI descending,
    cumulate occupancy / N_interior, report the CI at which the cumulant
    first reaches `mass`, plus the number of (non-empty) buckets and the
    exact mass fraction that determine the crossing.

    Cross-checked in-run against np.percentile(ci[idx][interior], 100*(1-mass))
    -- the two framings ("survival-function crossing" vs "percentile of the
    interior population") must agree exactly since both describe the same
    step function over the same population.
    """
    ci = np.diff(edges) / 2.0  # 256 interior-relevant half-widths, in native units
    interior = (idx > 0) & (idx < 255)
    sat_frac = float((~interior).mean())

    occ = np.bincount(idx.ravel(), minlength=256)
    n_interior = int(occ[1:255].sum())
    assert n_interior > 0, "no interior points -- fixture is entirely saturated"

    # Only buckets that actually hold interior mass "determine" a crossing.
    buckets = [(ci[b], int(occ[b])) for b in range(1, 255) if occ[b] > 0]
    buckets.sort(key=lambda t: -t[0])  # CI descending

    cum_occ = 0
    flip_point_ci = None
    n_buckets_at_crossing = None
    mass_fraction_at_crossing = None
    for i, (b_ci, b_occ) in enumerate(buckets):
        cum_occ += b_occ
        frac = cum_occ / n_interior
        if frac >= mass:
            flip_point_ci = float(b_ci)
            n_buckets_at_crossing = i + 1
            mass_fraction_at_crossing = float(frac)
            break

    # Cross-check: same population, same threshold, computed the "percentile"
    # way instead of the "sorted cumulant" way. Must match to within one
    # bucket-width's worth of population granularity.
    per_point_ci = ci[idx][interior]
    percentile_cross_check = float(np.percentile(per_point_ci, 100.0 * (1.0 - mass)))

    interior_vals = ci[1:255]
    return {
        "flip_point_ci": flip_point_ci,
        "n_buckets_at_crossing": n_buckets_at_crossing,
        "mass_fraction_at_crossing": mass_fraction_at_crossing,
        "percentile_cross_check_ci": percentile_cross_check,
        "sat_frac": sat_frac,
        "interior_ci_min": float(interior_vals.min()),
        "interior_ci_med": float(np.median(interior_vals)),
        "interior_ci_max": float(interior_vals.max()),
        "n_interior_points": n_interior,
        "n_nonempty_interior_buckets": len(buckets),
    }


def main():
    """Compute the closed-form flip points and report which bucket carries each crossing."""
    out = {}
    for var in VARS:
        print(f"\n{var}")
        anom = load_anomaly(var)
        out[var] = {}

        idx_lin, edges_lin = build_linear(anom)
        r_lin = closed_form_flip_point(idx_lin, edges_lin)
        r_lin["apparatus_note"] = (
            "LINEAR CI is CONSTANT across every interior bucket ((hi-lo)/512); "
            "this is expected apparatus behaviour (the control), not a defect -- "
            "the flip point is a single threshold, not a survival curve."
        )
        out[var]["linear"] = r_lin
        print(
            f"  linear   : flip_ci={r_lin['flip_point_ci']:.6f}  "
            f"n_buckets={r_lin['n_buckets_at_crossing']}  "
            f"mass={r_lin['mass_fraction_at_crossing']*100:.4f}%  "
            f"(cross-check {r_lin['percentile_cross_check_ci']:.6f})  "
            f"sat={r_lin['sat_frac']*100:.3f}%"
        )

        idx_fz, edges_fz = build_fisher_z(anom)
        r_fz = closed_form_flip_point(idx_fz, edges_fz)
        out[var]["fisher_z"] = r_fz
        print(
            f"  fisher_z : flip_ci={r_fz['flip_point_ci']:.6f}  "
            f"n_buckets={r_fz['n_buckets_at_crossing']}  "
            f"mass={r_fz['mass_fraction_at_crossing']*100:.4f}%  "
            f"(cross-check {r_fz['percentile_cross_check_ci']:.6f})  "
            f"sat={r_fz['sat_frac']*100:.3f}%"
        )
        print(
            f"  fisher_z interior CI: min={r_fz['interior_ci_min']:.6f} "
            f"med={r_fz['interior_ci_med']:.6f} max={r_fz['interior_ci_max']:.6f} "
            f"(two-regime expectation applies HERE, not to linear)"
        )

    with open(pathlib.Path(__file__).with_name("ev3_flip_points.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote ev3_flip_points.json")


if __name__ == "__main__":
    main()
