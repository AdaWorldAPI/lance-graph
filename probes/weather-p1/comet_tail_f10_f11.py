"""EXPLORATORY — CT-F10 (displacement-filtered rerun) + CT-F11 (regime-
filtered rerun) + CT-F13 (raw replication check), all pre-registered.
Follow-up to comet_tail_f5_n10.py (ffa2e35d). NOT an EV; bars mine, unaudited.

CT-N (previous probe) found sign-consistency of the left-of-motion offset
FAILED at n=10 (6/10=0.60) despite structure claims (wn1 dominance, R2)
replicating cleanly. Two post-hoc leads were named but explicitly NOT used
to override that FAIL: (a) restricting to storms with >=250 km/6h
displacement raised consistency to 3/4=0.75; (b) dropping the single most
extreme low-displacement/near-polar outlier raised it to 6/9=0.667. The
correct way to chase a post-hoc lead is a FRESH pre-registered sample with
the filter applied a priori (at least at the reporting stage) -- not a
re-scoring of the sample that already failed. This probe is that fresh
sample.

SAMPLE INDEPENDENCE (load-bearing): dates are MECHANICALLY generated (fixed
start + fixed stride, no hand-picking) and land entirely in 1980-1997,
non-overlapping with the original 10-storm sample (2015-2021). A human did
not choose dates likely to produce a particular sign-consistency number.

PRE-REGISTERED (computed on ONE blind draw of candidate dates; all storms
found are reported, none discarded except by the stated CT-E2 trackability
gate that was already standard before any filter is applied):

CT-F13  RAW REPLICATION (no filter). Does the ~0.60 sign-consistency from
        the first n=10 sample reproduce on an independent second blind
        sample? No bar (this IS the replication test) -- report sign
        fraction and median|error| for direct comparison against the first
        sample's 0.60 / 40.2 deg [19.5,63.2].

CT-F10  DISPLACEMENT FILTER. Among valid (trackable) storms, restrict to
        displacement >= 250 km/6h (the threshold that separated storms 1-2's
        own regime in the post-hoc stratification). Bar: sign-consistency
        >= 0.70. Minimum n=6 for a real verdict (below that: NO-VERDICT,
        stated in advance -- a subset of 3-4 cannot meaningfully test a 0.70
        rate).

CT-F11  REGIME FILTER. Among valid storms, restrict to wn1_frac >= 0.40
        (CT-E1's own dominance bar -- a storm without a dominant wn-1 mode
        does not have a well-defined dipole bearing for the signed test to
        apply to). Bar: sign-consistency >= 0.70, same n=6 minimum.

CT-F12  INTERSECTION (both filters). No bar -- reported for completeness,
        likely too small an n for a real verdict, stated as such.

NOT tested here: this is n=~10-15 candidates yielding filtered subsets of
n=4-10 -- still far short of a properly powered sample for any of these
sub-questions. A PASS on CT-F10/F11 here is evidence the leads are worth a
dedicated large-n follow-up, not proof of a general rule. A FAIL closes the
lead, same as CT-N1 closed the unfiltered claim.
"""
import datetime
import json
import pathlib
import urllib.request

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
print("t-index anchor guard: OK")

_MSLP_SHAPE = meta["mean_sea_level_pressure/.zarray"]["shape"]
_MAX_T = _MSLP_SHAPE[0] - 1
print(f"store bounds guard: max valid t={_MAX_T}")

print("fetching static fields (latitude) ...", flush=True)
lat = fetch("latitude", "0").astype(np.float64).ravel()
NY = lat.size
NX = 1440
phi = np.deg2rad(lat)
lon_deg = np.arange(NX) * 0.25

# ---- verbatim helpers (matches comet_tail_f5_n10.py) -----------------------


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
            "R2_profile_wn1": float(1.0 - np.mean((resid0 - wn1) ** 2) / vals.var())}


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
    """Wrap degrees into [-180, 180).

    The half-open end is LOW, not high: at exactly +180 this returns -180.
    (Docstring said "(-180, 180]" until 2026-08-11 — CodeRabbit, PR #926 —
    which was wrong about the code, not a bug in it.) The boundary is not
    cosmetic: `stratum_verdict` scores offset > 0 as low-pole-left-of-motion,
    so a value landing exactly on the boundary counts NEGATIVE. That is the
    physically right call — +/-180 means the dipole points exactly OPPOSITE
    the motion, which is not left-of-motion under either spelling — so the
    convention is kept, now stated. Audited across every committed result
    JSON: 283 angle-like values, 0 boundary hits, closest 0.91 deg.
    """
    return (d + 180.0) % 360.0 - 180.0


def err_deg(low_pole_rad, motion_rad):
    """Signed alignment error, in degrees, of a low-pole bearing against the left-of-motion prediction."""
    return float(wrap_deg(np.rad2deg(
        low_pole_rad - (motion_rad + np.pi / 2))))


# ---- fresh, mechanically-generated candidate dates -------------------------
START = datetime.datetime(1980, 2, 10, 12)
STRIDE_DAYS = 411          # non-round, decorrelates from the annual cycle;
                           # drifts ~1.5 months/sample -> natural season mix
N_CANDIDATES = 15
DATES = [START + datetime.timedelta(days=STRIDE_DAYS * i)
         for i in range(N_CANDIDATES)]
print(f"\ncandidate dates ({N_CANDIDATES}, stride={STRIDE_DAYS}d, "
      f"mechanically generated, non-overlapping with the 2015-2021 sample):")
for d in DATES:
    print(f"   {d.date()}")

rows = []
for dt in DATES:
    t0 = t_index(dt)
    if t0 < 0 or t0 + 1 > _MAX_T:
        rows.append({"date": dt.isoformat(), "status": "OUT-OF-STORE-BOUNDS"})
        print(f"\n{dt.date()}: out of store bounds -> excluded")
        continue
    p0 = fetch("mean_sea_level_pressure", f"{t0}.0.0")[0].astype(np.float64)
    p1 = fetch("mean_sea_level_pressure", f"{t0+1}.0.0")[0].astype(np.float64)

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
        print(f"\n{dt.date()}: center ({la_a:.2f}N,{lo_a:.2f}E) "
              f"disp={dist:.0f}km -> not trackable, excluded")
        continue

    d = decompose_ll(p0, la_a, lo_a)
    err = err_deg(d["low_pole_rad"], mth)
    row.update({"status": "OK", "wn1_frac": d["wn1_frac"],
               "R2_profile_wn1": d["R2_profile_wn1"], "error_deg": err,
               "sign_negative": bool(err < 0)})
    rows.append(row)
    print(f"\n{dt.date()}: center ({la_a:.2f}N,{lo_a:.2f}E), "
          f"disp={dist:.0f}km, wn1_frac={d['wn1_frac']:.3f}, "
          f"R2={d['R2_profile_wn1']:.3f}, error={err:+.1f} deg "
          f"({'neg' if err < 0 else 'pos'})")

valid = [r for r in rows if r.get("status") == "OK"]
excluded = [r for r in rows if r.get("status") != "OK"]
print(f"\n{len(valid)}/{len(rows)} storms valid; {len(excluded)} excluded: "
      + ", ".join(f"{r['date'][:10]}={r['status']}" for r in excluded))


def stratum_verdict(name, subset, min_n=6, bar=0.70):
    """Report a stratum. Below `min_n` this emits an explicit NO-VERDICT and
    NEVER a `pass` field — an earlier version let CT-F12 (n=3) carry
    `pass: true`, which a consumer could lift out of context as a valid
    passing result (coderabbit on PR #926, 2026-08-11)."""
    if len(subset) < min_n:
        print(f"\n{name}: n={len(subset)} < min_n={min_n} -> NO-VERDICT "
              "(pre-registered minimum, stated before results)")
        return {"n": len(subset), "verdict": "NO-VERDICT-INSUFFICIENT-N"}
    errs = np.array([r["error_deg"] for r in subset])
    neg = float((errs < 0).mean())
    med_abs = float(np.median(np.abs(errs)))
    passed = neg >= bar
    print(f"\n{name}: n={len(subset)}, sign_neg_frac={neg:.2f} "
          f"({int((errs<0).sum())}/{len(subset)}), median|error|={med_abs:.1f} "
          f"deg -> {'PASS' if passed else 'FAIL'} (bar >= {bar})")
    return {"n": len(subset), "sign_neg_frac": neg,
           "median_abs_error_deg": med_abs, "pass": bool(passed)}


out = {"store": B, "sample_start": START.isoformat(),
      "stride_days": STRIDE_DAYS, "n_candidates": N_CANDIDATES,
      "rows": rows, "n_valid": len(valid), "n_excluded": len(excluded)}

print("\n=== CT-F13 raw replication (no filter) ===")
out["CT_F13"] = stratum_verdict(
    "CT-F13 (all valid, unfiltered)", valid, min_n=1, bar=0.70)
if valid:
    print("   (compare against the first sample: 0.60 sign_neg_frac, "
         "median|error|=40.2 deg)")

print("\n=== CT-F10 displacement filter (>=250 km/6h) ===")
disp_subset = [r for r in valid if r["displacement_km"] >= 250.0]
out["CT_F10"] = stratum_verdict("CT-F10 (disp>=250km)", disp_subset)

print("\n=== CT-F11 regime filter (wn1_frac >= 0.40) ===")
regime_subset = [r for r in valid if r["wn1_frac"] >= 0.40]
out["CT_F11"] = stratum_verdict("CT-F11 (wn1_frac>=0.40)", regime_subset)

print("\n=== CT-F12 intersection (both filters, exploratory, no bar) ===")
both_subset = [r for r in valid
              if r["displacement_km"] >= 250.0 and r["wn1_frac"] >= 0.40]
# min_n stays at the real evaluable floor: CT-F12 is REPORTED for
# completeness, but it must not be able to emit a pass/fail verdict on n=3.
out["CT_F12"] = stratum_verdict("CT-F12 (both filters)", both_subset)

with open(pathlib.Path(__file__).with_name("comet_tail_f10_f11.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote comet_tail_f10_f11.json")
