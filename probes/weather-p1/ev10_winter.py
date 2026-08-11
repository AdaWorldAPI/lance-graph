"""EV-10 (v2, plan .claude/plans/weather-substrate-evaluation-v1.md sec.3) --
second season, ONE factor varied, with a mandatory stale-fixture guard.

Run A ONLY: SAME variable (2m_temperature). The factor varied is
season/timestep -- summer (t547476, the pinned 2021-06-15 12:00 UTC
timestep) vs winter (t543852, 2021-01-15 12:00Z). The v1 spec varied
season + timestep + variable simultaneously, which made any failure
unattributable (plan sec.8 attack-pass finding for EV-10); this run holds
the variable fixed so a divergence can only be read as a seasonal effect.

MANDATORY STALE-FIXTURE GUARD: the v1 fetcher hardcoded `t` and skipped
re-fetch if the target file already existed, so a "winter" run could
silently re-measure summer bytes under a winter label. This script reads
fixture/manifest.json and asserts the time_index of every array it loads
against the timestep it declares, and additionally asserts the summer and
winter arrays are NOT byte-identical (a genuine stale-fixture symptom no
time_index check alone would catch, since a manifest entry can be correct
while the file underneath it is not). Either failure produces a loud
NO-VERDICT result -- never a silent pass on stale data.

Compared quantity: the EV-3 closed-form 1% flip-point (sort interior
buckets by CI descending, cumulate occupancy/n_interior, report the CI at
which the cumulant first reaches 1% -- equivalently the 99th percentile of
ci[idx][interior]), computed independently here (not imported from
ev3_flip_points.py, which is owned by another agent) using the identical
definition so the cross-season comparison is apples-to-apples. Both the
linear and fisher_z arms are compared, per the v2 spec's correction of
v1's "same qualitative ordering" (undefined at floors >= 0.5 K, where both
arms read identically 0.0 per K-9).

Pre-registered pass bar: flip-points must agree across seasons within a
factor of 2 (max/min <= 2.0). This is reported, not gated on -- the script
reports the actual ratio for both arms and lets the plan's decision
register (D-1) consume it; it does not silently redefine "agree" if the
ratio comes out above 2.
"""
import json

import numpy as np

VAR = "2m_temperature"
# season -> (fixture subdir, declared time_index)
SEASONS = {
    "summer": ("t547476", 547476),
    "winter": ("t543852", 543852),
}
FLIP_MASS = 0.01  # 1% flip-point, pre-registered (plan EV-3, inherited here)


def fz(s, eps=1e-9):
    """Fisher-Z: arctanh(s), clipped off the +/-1 poles by eps."""
    s = np.clip(s, -1 + eps, 1 - eps)
    return 0.5 * (np.log1p(s) - np.log1p(-s))


def build_linear(anom):
    """LINEAR path: uniform buckets in original units (K). CI is CONSTANT
    across every interior bucket -- (hi-lo)/512 -- by construction."""
    lo, hi = np.percentile(anom, [0.4, 99.6])
    idx = np.clip(np.floor((anom - lo) / (hi - lo) * 256), 0, 255).astype(np.uint8)
    edges = lo + np.arange(257) / 256 * (hi - lo)
    return idx, edges


def build_fisher_z(anom):
    """FISHER-Z path: uniform buckets in z, NON-uniform in K after tanh --
    CI genuinely varies bucket-to-bucket."""
    lo_q, hi_q = np.percentile(anom, [0.4, 99.6])
    scale = max(abs(lo_q), abs(hi_q))
    z = fz(anom / scale)
    zlo, zhi = np.percentile(z, [0.4, 99.6])
    idxz = np.clip(np.floor((z - zlo) / (zhi - zlo) * 256), 0, 255).astype(np.uint8)
    edges = np.tanh(zlo + np.arange(257) / 256 * (zhi - zlo)) * scale
    return idxz, edges


def closed_form_flip_point(idx, edges, mass=FLIP_MASS):
    """The EV-3 closed-form 1% flip-point (independently reimplemented here
    to the same definition; ev3_flip_points.py is owned by another agent
    and is neither imported nor edited by this script).

    Sort interior buckets (1..254) by CI descending, cumulate occupancy /
    n_interior, report the CI at which the cumulant first reaches `mass`,
    plus the number of (non-empty) buckets and the exact mass fraction
    that determine the crossing. No grid, no step size.

    Cross-checked in-run against np.percentile(ci[idx][interior],
    100*(1-mass)) -- the two framings ("survival-function crossing" vs
    "percentile of the interior population") describe the same step
    function over the same population and must agree exactly.
    """
    ci = np.diff(edges) / 2.0  # 256 half-widths, native units
    interior = (idx > 0) & (idx < 255)
    sat_frac = float((~interior).mean())

    occ = np.bincount(idx.ravel(), minlength=256)
    n_interior = int(occ[1:255].sum())
    assert n_interior > 0, "no interior points -- fixture is entirely saturated"

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
    result = {
        "variable": VAR,
        "run": "A (season/timestep only, variable held fixed at 2m_temperature)",
    }

    # ---- MANDATORY STALE-FIXTURE GUARD ----
    manifest = json.load(open("fixture/manifest.json"))
    guard = {"checks": []}
    no_verdict_reason = None
    arrays = {}
    declared_time_index = {}

    for season, (tdir, declared_t) in SEASONS.items():
        declared_time_index[season] = declared_t
        key = f"{tdir}/{VAR}"
        entry = manifest.get(key)
        if entry is None:
            no_verdict_reason = f"manifest missing key {key!r}"
            break
        if entry["time_index"] != declared_t:
            no_verdict_reason = (
                f"manifest time_index mismatch for {key}: declared {declared_t}, "
                f"manifest says {entry['time_index']}"
            )
            break
        arr = np.load(f"fixture/{tdir}/{VAR}.npy").astype(np.float64)
        if list(arr.shape) != entry["shape"]:
            no_verdict_reason = (
                f"shape mismatch for {key}: array {list(arr.shape)} vs "
                f"manifest {entry['shape']}"
            )
            break
        if not np.isfinite(arr).all():
            no_verdict_reason = f"{key}: nonfinite values in fixture"
            break
        arrays[season] = arr
        guard["checks"].append(
            {
                "season": season,
                "key": key,
                "declared_time_index": declared_t,
                "manifest_time_index": entry["time_index"],
                "shape": entry["shape"],
                "time_index_matches": True,
            }
        )
        print(f"guard: {key} time_index={entry['time_index']} shape={entry['shape']} OK")

    if no_verdict_reason is None:
        max_abs_diff = float(np.max(np.abs(arrays["summer"] - arrays["winter"])))
        guard["max_abs_diff_K"] = max_abs_diff
        print(f"guard: max_abs_diff(summer, winter) = {max_abs_diff:.6f} K")
        if max_abs_diff == 0.0:
            no_verdict_reason = (
                "summer and winter arrays are byte-identical (max_abs_diff == 0.0) "
                "-- stale fixture (winter re-measured summer bytes); refusing to "
                "report a verdict"
            )

    result["time_index"] = declared_time_index
    result["stale_fixture_guard"] = guard

    if no_verdict_reason is not None:
        result["NO_VERDICT"] = True
        result["no_verdict_reason"] = no_verdict_reason
        print(f"\nNO-VERDICT: {no_verdict_reason}")
        json.dump(result, open("ev10_winter.json", "w"), indent=2)
        raise SystemExit(1)

    # ---- per-season flip-points (both arms) ----
    flip = {}
    for season, arr in arrays.items():
        print(f"\n{season} ({SEASONS[season][0]}, time_index={SEASONS[season][1]})")
        clim = arr.mean(axis=1, keepdims=True)
        anom = arr - clim
        flip[season] = {}

        idx_lin, edges_lin = build_linear(anom)
        r_lin = closed_form_flip_point(idx_lin, edges_lin)
        r_lin["apparatus_note"] = (
            "LINEAR CI is CONSTANT across every interior bucket ((hi-lo)/512); "
            "expected apparatus behaviour (the control), not a defect -- the "
            "flip point is a single threshold, not a survival curve."
        )
        flip[season]["linear"] = r_lin
        print(
            f"  linear   : flip_ci={r_lin['flip_point_ci']:.6f}  "
            f"n_buckets={r_lin['n_buckets_at_crossing']}  "
            f"mass={r_lin['mass_fraction_at_crossing']*100:.4f}%  "
            f"(cross-check {r_lin['percentile_cross_check_ci']:.6f})  "
            f"sat={r_lin['sat_frac']*100:.3f}%"
        )

        idx_fz, edges_fz = build_fisher_z(anom)
        r_fz = closed_form_flip_point(idx_fz, edges_fz)
        flip[season]["fisher_z"] = r_fz
        print(
            f"  fisher_z : flip_ci={r_fz['flip_point_ci']:.6f}  "
            f"n_buckets={r_fz['n_buckets_at_crossing']}  "
            f"mass={r_fz['mass_fraction_at_crossing']*100:.4f}%  "
            f"(cross-check {r_fz['percentile_cross_check_ci']:.6f})  "
            f"sat={r_fz['sat_frac']*100:.3f}%"
        )

    result["flip_points"] = flip

    # ---- cross-season comparison, pre-registered factor-of-2 agreement ----
    comparisons = {}
    print(f"\n{'arm':<10s} {'summer_K':>12s} {'winter_K':>12s} {'ratio':>8s} {'<=2x':>6s}")
    for arm_name in ["linear", "fisher_z"]:
        fp_s = flip["summer"][arm_name]["flip_point_ci"]
        fp_w = flip["winter"][arm_name]["flip_point_ci"]
        lo_fp, hi_fp = (fp_s, fp_w) if fp_s <= fp_w else (fp_w, fp_s)
        ratio = (hi_fp / lo_fp) if lo_fp > 0 else float("inf")
        within_factor_2 = bool(ratio <= 2.0)
        comparisons[arm_name] = {
            "summer_flip_point_K": fp_s,
            "winter_flip_point_K": fp_w,
            "ratio_max_over_min": ratio,
            "within_factor_of_2": within_factor_2,
        }
        print(f"{arm_name:<10s} {fp_s:12.6f} {fp_w:12.6f} {ratio:8.4f} {str(within_factor_2):>6s}")

    result["season_comparison"] = comparisons
    result["NO_VERDICT"] = False

    json.dump(result, open("ev10_winter.json", "w"), indent=2)
    print("\nwrote ev10_winter.json")


if __name__ == "__main__":
    main()
