"""EXPLORATORY — CT-F14, the correctly-scoped next step named in §5.10: a
SINGLE, properly powered, displacement-filtered-ONLY pre-registered sample.
Follow-up to comet_tail_f10_f11.py (e24ecf3d). NOT an EV; bars mine,
unaudited.

WHY THIS IS THE RIGHT TEST NOW (not a third exploratory rerun). Two prior
independent 10-storm samples gave a combined 14/20 unfiltered (one-sided
p~=0.058, noise floor) but a combined 6/7 among storms with >=250 km/6h
displacement (one-sided p~=0.0625) -- suggestive but n=7 is too small to
call, and that pooling was NOT itself pre-registered before either sample
ran. The regime-contamination explanation was checked directly against
sample 1's own data and did NOT survive (dropped sample 1 to exactly
chance). The apparatus explanation (motion-bearing noise on slow-moving
storms) is the one with surviving support. CT-F14 is the single test that
moves that number from "suggestive, n=7" to a real verdict: ONE
pre-registered displacement-filtered-only sample, sized for n>=20-30
qualifying storms.

SAMPLE INDEPENDENCE (load-bearing, third time in this chain): dates are
MECHANICALLY generated (fixed start + fixed stride, chosen before this file
was written and never adjusted after seeing output) and land in 1996-2014 --
zero overlap with sample 1 (2015-2021, hand-picked) or sample 2
(1980-1995, stride=411d from 1980-02-10). Expected qualifying rate is
estimated from the TWO PRIOR samples' combined empirical rate (7 qualifying
storms out of 25 total candidates tried = 0.28/candidate) to size N_CANDIDATES
for the sample -- an estimate, not a guarantee; actual attrition is reported
however it lands, never resampled to hit a target.

PRE-REGISTERED, single test:

CT-F14  Among valid (CT-E2 trackable) storms in this fresh sample with
        displacement >= 250 km/6h, sign-consistency (fraction with error<0,
        the same direction as storms 1-2 and both prior samples' filtered
        subsets) >= 0.70. Minimum n=20 for a real verdict (below that:
        NO-VERDICT, stated in advance -- consistent with the n=6 floor used
        for smaller strata in the prior probe, scaled up for a properly
        powered single test).

REPORTED ALONGSIDE (not additional bars, just always-computed descriptive
stats + the decision rule pre-committed for how this updates the arc's
overall verdict):

  - exact one-sided binomial p-value (H0: p=0.5) for CT-F14's own subset
  - THE COMBINED three-independent-sample figure: CT-F14's qualifying
    storms + the two prior samples' displacement>=250 subsets (n=4+3=7),
    with its own one-sided binomial p-value
  - median wn1_frac / R2_profile_wn1 among the qualifying subset, for
    comparison against N3/N4's already-established generalization
  - PRE-COMMITTED interpretation of the combined 3-sample p-value (decided
    now, before running, so the read cannot be tuned to the result):
      p < 0.05          -> established at this n, displacement-filtered
                            regime; ready for the CT-F10..F15 chain to close
                            and the finding to enter the audit-gate queue
                            as an [H]-graded, scope-limited claim
      0.05 <= p < 0.10   -> still suggestive; needs a further doubling of n
                            before promotion, not yet audit-gate-ready
      p >= 0.10          -> NOT established; the apparatus explanation is
                            itself now in question and the offset's
                            directional claim should be retired to [S]
                            pending a fundamentally different design (e.g.
                            modeling motion-bearing uncertainty explicitly
                            rather than a hard displacement cutoff)

NOT tested here: this remains a MSLP-only, NH-only, one-season-mix test; no
claim about SH, no claim about tropical cyclones, no claim beyond the
extratropical-low steering-flow framing §2 laid out from the start.
"""
import datetime
import json
import pathlib
import urllib.request
from math import comb

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
R_E = 6371.0
R_DISK = 1200.0
RING = 100.0
EPOCH = datetime.datetime(1959, 1, 1)

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=600).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


def t_index(dt):
    """WB2 time index for a datetime: 6-hourly steps since 1959-01-01."""
    return int(round((dt - EPOCH).total_seconds() / 3600 / 6))


assert t_index(datetime.datetime(2021, 6, 15, 12)) == 91246, \
    "t-index formula does not reproduce the pinned arc anchor T0=91246"
print("t-index anchor guard: OK", flush=True)

_MSLP_SHAPE = meta["mean_sea_level_pressure/.zarray"]["shape"]
_MAX_T = _MSLP_SHAPE[0] - 1
print(f"store bounds guard: max valid t={_MAX_T}", flush=True)

print("fetching static fields (latitude) ...", flush=True)
lat = fetch("latitude", "0").astype(np.float64).ravel()
NY = lat.size
NX = 1440
phi = np.deg2rad(lat)
lon_deg = np.arange(NX) * 0.25

# ---- verbatim helpers (matches comet_tail_f10_f11.py / f5_n10.py) ---------


def geom_ll(latc, lonc):
    """dx, dy, r (km) and azimuth theta (rad, CCW from east) relative to a CONTINUOUS (lat, lon) centre."""
    phic = np.deg2rad(latc)
    dlon = np.deg2rad((lon_deg[None, :] - lonc + 180) % 360 - 180)
    dphi = phi[:, None] - phic
    dx = R_E * np.cos(phic) * dlon * np.ones((NY, 1))
    dy = R_E * dphi * np.ones((1, NX))
    return dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def find_center(field, near=None, radius_km=600.0, lat_lo=25.0, lat_hi=75.0):
    """Deepest zonal-anomaly low; returns None when the (optionally `near`-limited) mask admits no finite candidate."""
    fa = field - field.mean(axis=1, keepdims=True)
    mask = (lat[:, None] > lat_lo) & (lat[:, None] < lat_hi)
    if near is not None:
        _, _, r, _ = geom_ll(*near)
        mask = mask & (r < radius_km)
    masked = np.where(mask, fa, np.inf)
    ci, cj = np.unravel_index(np.argmin(masked), field.shape)
    # An empty mask makes `masked` all-inf and argmin returns index 0, i.e. the
    # function would report grid cell (0,0) as a storm centre. A `near`-limited
    # search CAN be fully masked, so this must be checked on every path
    # (coderabbit on PR #926, 2026-08-11).
    if not np.isfinite(masked[ci, cj]):
        return None
    return int(ci), int(cj)


def decompose_ll(field, latc, lonc):
    """Ring-mean profile + per-ring wavenumber-1 fit about a continuous centre."""
    _, _, r, th = geom_ll(latc, lonc)
    disk = r <= R_DISK
    vals, rr, tt = field[disk], r[disk], th[disk]
    rings = np.clip((rr / RING).astype(int), 0, int(R_DISK / RING) - 1)
    nb = int(R_DISK / RING)
    prof = np.zeros(nb)
    a1 = np.zeros(nb)
    b1 = np.zeros(nb)
    for b in range(nb):
        m = rings == b
        if not m.any():
            continue
        v, t = vals[m], tt[m]
        prof[b] = v.mean()
        a1[b] = 2 * ((v - prof[b]) * np.cos(t)).mean()
        b1[b] = 2 * ((v - prof[b]) * np.sin(t)).mean()
    resid0 = vals - prof[rings]
    wn1 = a1[rings] * np.cos(tt) + b1[rings] * np.sin(tt)
    amp = np.hypot(a1, b1)
    w = amp * np.arange(nb)
    ph = np.arctan2(np.sum(b1 * w), np.sum(a1 * w))
    return {"low_pole_rad": float((ph + np.pi) % (2 * np.pi)),
            "wn1_frac": float(wn1.var() / max(resid0.var(), 1e-12)),
            "R2_profile_wn1": float(1.0 - (resid0 - wn1).var() / vals.var())}


def subgrid_min(field, ci, cj):
    # Longitude WRAPS: a centre at cj == 0 or NX-1 would otherwise slice a 3x2
    # neighbourhood, `A` would have 9 rows against 6 values and lstsq would
    # raise. Centres come from a global scan, so the seam at 0 deg is reachable
    # (coderabbit on PR #926, 2026-08-11). Rows are clamped, not wrapped — the
    # poles are not periodic.
    """Sub-grid minimum by 2-D quadratic fit on the 3x3 neighbourhood; longitude wraps, latitude clamps."""
    ri = np.clip(np.array([ci - 1, ci, ci + 1]), 0, field.shape[0] - 1)
    z = np.take(field[ri, :], [cj - 1, cj, cj + 1], axis=1, mode="wrap").ravel()
    gy, gx = np.meshgrid([-1., 0., 1.], [-1., 0., 1.], indexing="ij")
    A = np.column_stack([np.ones(9), gx.ravel(), gy.ravel(),
                         gx.ravel() ** 2, gy.ravel() ** 2,
                         (gx * gy).ravel()])
    c = np.linalg.lstsq(A, z, rcond=None)[0]
    _, b, cc, d, e, g = c
    H = np.array([[2 * d, g], [g, 2 * e]])
    try:
        dj, di = np.linalg.solve(H, [-b, -cc])
    except np.linalg.LinAlgError:
        dj = di = 0.0
    di, dj = float(np.clip(di, -1, 1)), float(np.clip(dj, -1, 1))
    lat_step = lat[ci + 1] - lat[ci] if ci + 1 < NY else lat[ci] - lat[ci - 1]
    return lat[ci] + di * lat_step, lon_deg[cj] + dj * 0.25


def wrap_deg(d):
    """Wrap degrees into (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def err_deg(low_pole_rad, motion_rad):
    """Signed alignment error, in degrees, of a low-pole bearing against the left-of-motion prediction."""
    return float(wrap_deg(np.rad2deg(
        low_pole_rad - (motion_rad + np.pi / 2))))


def binom_sf_ge(k, n, p=0.5):
    """Exact one-sided binomial tail P(X >= k) for n trials at probability p."""
    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
              for i in range(k, n + 1))


# ---- fresh, mechanically-generated candidate dates -------------------------
START = datetime.datetime(1996, 1, 15, 12)
STRIDE_DAYS = 61            # arbitrary, fixed before writing this loop,
                            # never adjusted after seeing output
N_CANDIDATES = 85           # sized from the two prior samples' combined
                            # empirical rate (7/25 candidates ~= 0.28) to
                            # target n>=20-30 qualifying storms
DATES = [START + datetime.timedelta(days=STRIDE_DAYS * i)
         for i in range(N_CANDIDATES)]
print(f"\nCT-F14: {N_CANDIDATES} candidate dates, stride={STRIDE_DAYS}d, "
      f"{DATES[0].date()} .. {DATES[-1].date()} "
      "(1996-2014, zero overlap with prior samples)", flush=True)

rows = []
for idx, dt in enumerate(DATES):
    t0 = t_index(dt)
    if t0 < 0 or t0 + 1 > _MAX_T:
        rows.append({"date": dt.isoformat(), "status": "OUT-OF-STORE-BOUNDS"})
        continue
    try:
        p0 = fetch("mean_sea_level_pressure", f"{t0}.0.0")[0].astype(np.float64)
        p1 = fetch("mean_sea_level_pressure", f"{t0+1}.0.0")[0].astype(np.float64)
    except Exception as e:
        rows.append({"date": dt.isoformat(), "status": f"FETCH-ERROR: {e}"})
        print(f"[{idx+1}/{N_CANDIDATES}] {dt.date()}: fetch error {e}",
              flush=True)
        continue

    ci0, cj0 = find_center(p0)
    la_a, lo_a = subgrid_min(p0, ci0, cj0)
    ci1, cj1 = find_center(p1, near=(la_a, lo_a))
    dx, dy, _, _ = geom_ll(la_a, lo_a)
    disp = (float(dx[ci1, cj1]), float(dy[ci1, cj1]))
    dist = float(np.hypot(*disp))
    mth = float(np.arctan2(disp[1], disp[0]))
    trackable = dist >= 100.0

    row = {"date": dt.isoformat(), "t0": t0,
          "center_lat": la_a, "center_lon": lo_a,
          "displacement_km": dist, "CT_E2_trackable": bool(trackable)}
    if not trackable:
        row["status"] = "NOT-TRACKABLE"
        rows.append(row)
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[{idx+1}/{N_CANDIDATES}] {dt.date()}: "
                  f"disp={dist:.0f}km not trackable", flush=True)
        continue

    d = decompose_ll(p0, la_a, lo_a)
    err = err_deg(d["low_pole_rad"], mth)
    row.update({"status": "OK", "wn1_frac": d["wn1_frac"],
               "R2_profile_wn1": d["R2_profile_wn1"], "error_deg": err,
               "sign_negative": bool(err < 0)})
    rows.append(row)
    tag = "  ***DISP>=250***" if dist >= 250 else ""
    print(f"[{idx+1}/{N_CANDIDATES}] {dt.date()}: disp={dist:.0f}km "
          f"err={err:+.1f}deg wn1={d['wn1_frac']:.2f}{tag}", flush=True)

valid = [r for r in rows if r.get("status") == "OK"]
excluded = [r for r in rows if r.get("status") != "OK"]
qualifying = [r for r in valid if r["displacement_km"] >= 250.0]

print(f"\n{len(valid)}/{len(rows)} valid (trackable); "
      f"{len(qualifying)} qualify (disp>=250km); "
      f"{len(excluded)} excluded", flush=True)
exclusion_counts = {}
for r in excluded:
    exclusion_counts[r["status"]] = exclusion_counts.get(r["status"], 0) + 1
print("exclusion breakdown:", exclusion_counts, flush=True)

out = {"store": B, "sample_start": START.isoformat(),
      "stride_days": STRIDE_DAYS, "n_candidates": N_CANDIDATES,
      "rows": rows, "n_valid": len(valid), "n_qualifying": len(qualifying),
      "n_excluded": len(excluded), "exclusion_breakdown": exclusion_counts}

print("\n=== CT-F14 verdict ===", flush=True)
MIN_N = 20
if len(qualifying) < MIN_N:
    print(f"n={len(qualifying)} < MIN_N={MIN_N} -> NO-VERDICT "
          "(pre-registered minimum, stated before results)")
    out["CT_F14"] = {"n": len(qualifying), "verdict": "NO-VERDICT-INSUFFICIENT-N"}
else:
    errs = np.array([r["error_deg"] for r in qualifying])
    neg = int((errs < 0).sum())
    frac = neg / len(qualifying)
    p_own = binom_sf_ge(neg, len(qualifying))
    passed = frac >= 0.70
    med_wn1 = float(np.median([r["wn1_frac"] for r in qualifying]))
    med_r2 = float(np.median([r["R2_profile_wn1"] for r in qualifying]))
    print(f"CT-F14: n={len(qualifying)}, neg={neg}/{len(qualifying)} = "
          f"{frac:.3f}, one-sided p={p_own:.4f} -> "
          f"{'PASS' if passed else 'FAIL'} (bar >= 0.70)")
    print(f"  median wn1_frac={med_wn1:.3f}, median R2={med_r2:.3f} "
          "(compare against N3/N4's 0.723/0.900)")
    out["CT_F14"] = {"n": len(qualifying), "n_negative": neg,
                     "sign_neg_frac": frac, "one_sided_p": p_own,
                     "pass": bool(passed), "median_wn1_frac": med_wn1,
                     "median_R2_profile_wn1": med_r2}

# combined with the two prior independent samples' disp>=250 subsets
PRIOR_S1 = {"n": 4, "neg": 3}   # comet_tail_f5_n10.json CT_N disp>=250
PRIOR_S2 = {"n": 3, "neg": 3}   # comet_tail_f10_f11.json CT_F10 disp>=250
n_qual = len(qualifying)
neg_qual = int(sum(1 for r in qualifying if r["error_deg"] < 0)) if n_qual else 0
n_combined = PRIOR_S1["n"] + PRIOR_S2["n"] + n_qual
neg_combined = PRIOR_S1["neg"] + PRIOR_S2["neg"] + neg_qual
frac_combined = neg_combined / n_combined if n_combined else float("nan")
p_combined = binom_sf_ge(neg_combined, n_combined) if n_combined else float("nan")

if p_combined < 0.05:
    interp = "ESTABLISHED at this n, displacement-filtered regime -> ready for audit-gate queue as [H]-graded, scope-limited claim"
elif p_combined < 0.10:
    interp = "SUGGESTIVE, needs further n before promotion, not audit-gate-ready"
else:
    interp = "NOT ESTABLISHED -> apparatus explanation itself in question; retire directional claim to [S] pending a fundamentally different design"

# THE POOLING RULE HAD A GAP, and the artifact must not hide it.
# The pre-registered rule above reads ONLY p_combined, so when CT-F14 -- the
# largest and most carefully powered component -- fails its OWN n>=20 floor,
# this block still emitted "ESTABLISHED ... ready for audit-gate queue"
# unconditionally. A result that failed its own gate then carried a promotion
# recommendation in the MACHINE-READABLE artifact, contradicting the report's
# own section 5.11 conclusion ("graded down to still suggestive"). The prose was
# corrected; this file was not. (coderabbit on PR #926, 2026-08-11.)
#
# The rule's mechanical output is KEPT -- deleting it would hide what the
# pre-registration actually said -- but renamed to
# `interpretation_preregistered_rule`, and is no longer the field a consumer
# reads as the verdict. `applied_verdict` is, and it is gated.
subset_below_min_n = n_qual < MIN_N
if subset_below_min_n:
    applied = (f"NOT PROMOTED -- the largest component (CT-F14, n={n_qual}) "
               f"failed its own pre-registered n>={MIN_N} floor. Pooling cannot "
               "rescue a component that did not qualify, and the "
               "pre-registration had no contingency for this case. Directional "
               "claim remains SUGGESTIVE (report section 5.11).")
else:
    applied = interp

print(f"\n=== COMBINED across THREE independent samples "
      f"(sample1 n=4 + sample2 n=3 + CT-F14 n={n_qual}) ===")
print(f"n={n_combined}, neg={neg_combined}/{n_combined} = {frac_combined:.3f}, "
      f"one-sided p={p_combined:.4f}")
print(f"PRE-REGISTERED RULE would say: {interp}")
if subset_below_min_n:
    print(f"GATED -> {applied}")

out["CT_F14_combined_3sample"] = {
    "n": n_combined, "n_negative": neg_combined,
    "sign_neg_frac": frac_combined, "one_sided_p": p_combined,
    "applied_verdict": applied,
    "interpretation_preregistered_rule": interp,
    "subset_below_min_n": subset_below_min_n,
    "largest_component_min_n": MIN_N,
    "components": {"sample1_disp250": PRIOR_S1, "sample2_disp250": PRIOR_S2,
                  "CT_F14": {"n": n_qual, "neg": neg_qual}}}

with open(pathlib.Path(__file__).with_name("comet_tail_f14.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote comet_tail_f14.json")
