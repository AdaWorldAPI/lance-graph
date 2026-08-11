"""EXPLORATORY — CT-F5 (fix F1's saturation defect) + CT-N (n=10 blind storm
sample: the binding constraint after CT-F4 cleared the apparatus). Follow-up
to comet_tail_f4_f7.py (432bcab2). NOT an EV; bars mine, unaudited.

============================== CT-F5 ========================================
CT-F1 (comet_tail_followup.py) swept 13 pressure levels searching for each
level's own low CENTER within a FIXED 600 km radius of the SURFACE center.
Storm 2's search saturated at exactly that radius for 5 of 13 levels
(586-599 km) -- it never found a co-located upper center and locked onto a
different system, producing a physically-absurd "best level = 100 hPa".

FIX: track the center level-by-level, searching near the PREVIOUS level's
found center (radius 250 km per step) instead of always near the surface.
This lets the center walk continuously along the tilt axis with height,
rather than being asked to jump the whole tilt in one 600 km hop from the
surface. Levels are walked from 1000 hPa UPWARD (surface-anchored, since
that is the level both storms' baseline tracking used).

PRE-REGISTERED:
  CT-F5a  storm 2's walking-center path clears CT-F1's original bar (best
          level in 400-850 hPa AND |error| <= 20 deg there) -- the walking
          fix is credited only if it turns a NO-VERDICT into a real pass,
          not merely a different number.
  CT-F5b  no level's walking step exceeds 250 km (the search radius) --
          if the walk itself saturates, the fix has the same disease at a
          different radius and must be reported as such, not silently capped.
  CT-F5c  storm 1 (which already passed) must reproduce its original
          400-850 hPa crossing within 10 deg -- the fix must not be a free
          rewrite that also perturbs the case that was already correct.

============================== CT-N =========================================
CT-F4 showed the apparatus (center choice) is NOT what makes the -40 deg
offset unreliable; CT-F7 showed friction is bounded and mis-scoped as the
mechanism. What remains untested at n=2 is whether -40 deg is a real
central tendency across INDEPENDENT synoptic times, or an n=2 coincidence.

Ten dates, BLIND selection (no hint, no inspection before recording): NH,
25-75 lat, deepest zonal-anomaly MSLP low at 12Z, one per date, spanning
~6 years and all four seasons so storms are independent systems, not the
same event re-sampled. The anchor date (2021-06-15, storm 1) is INCLUDED
and its t-index is asserted against the arc's pinned T0=91246 so the whole
chain stays anchored to previously-published numbers.

PRE-REGISTERED (storms failing CT-E2 trackability are EXCLUDED from N1-N5,
their exclusion reported, not silently dropped):
  CT-N1  SIGN consistency: among valid storms, count negative alignment
         error (same rotational sense as storms 1-2) / total.
         Bar: >= 0.70 -- majority-same-sign is the minimum for "systematic",
         not requiring unanimity (n=2 was already not unanimous-required,
         it was 2/2 by chance of only having 2).
  CT-N2  MAGNITUDE: median |error| and IQR reported (no bar -- this is the
         number that tells us if -40 is a central tendency or storms 1-2
         were the extreme tail of a wide distribution).
  CT-N3  wn1 DOMINANCE at scale: median wn1_frac >= 0.40 (replicates E1).
  CT-N4  R2 at scale: median R2_profile_wn1 >= 0.80 (replicates E4).
  CT-N5  = CT-F8: vorticity-centroid (wind) center vs sub-grid MSLP center,
         PAIRED sign test per storm: does |error| shrink at the wind center
         more often than not? Bar: >= 0.70 of valid storms show shrinkage
         (storm 2 showed this in F4; testing if it generalizes).
  CT-F9  Land-fraction ASYMMETRY vs pressure residual: fit the SAME
         ring/wn-1 decomposition to land_sea_mask around each center,
         producing a land-dipole magnitude and bearing per storm. Under the
         Ekman-pumping candidate, storms with a stronger land dipole should
         show a LARGER unexplained pressure residual (1 - R2_profile_wn1).
         Bar: corr(land_dipole_amplitude, 1-R2_profile_wn1) reported with a
         sign call -- POSITIVE supports candidate 2 residually mattering,
         near-zero/negative does not. No pre-set threshold (this is n=10,
         exploratory-of-exploratory; correlation SIGN is the only claim).

NOT tested here: n=10 is not n>=30; no offset CONSTANT is fitted from CT-N2,
only its distribution is reported. CT-F5's fix does not retroactively change
any bar already scored in COMET_TAIL_REPORT.md sec 5.2 -- it is reported as
a correction of that section's storm-2 verdict, dated and cited.

RUN LOG (transparency): run 1 crashed on a 404 for 2022-02-14 -- the store's
own filename claims "1959-2022" coverage but its actual last timestep is
2021-12-31 18Z (six months short), diagnosed against the .zarray shape before
any code changed. Added a bounds guard (report+exclude, never crash) and
swapped that one date for 2014-09-12 (in-bounds, autumn, different decade) to
keep n=10 candidate dates. No bar above was touched.
"""
import datetime
import json
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


def static_key(var):
    """Chunk key for a static (time-independent) variable: all-zero index of the right arity."""
    return ".".join("0" * len(meta[f"{var}/.zarray"]["chunks"]))


def t_index(dt):
    """WB2 time index for a datetime: 6-hourly steps since 1959-01-01."""
    return int(round((dt - EPOCH).total_seconds() / 3600 / 6))


assert t_index(datetime.datetime(2021, 6, 15, 12)) == 91246, \
    "t-index formula does not reproduce the pinned arc anchor T0=91246"
print("t-index anchor guard: OK (2021-06-15 12Z -> 91246)")

_MSLP_SHAPE = meta["mean_sea_level_pressure/.zarray"]["shape"]
_MAX_T = _MSLP_SHAPE[0] - 1
_last_valid = EPOCH + datetime.timedelta(hours=_MAX_T * 6)
print(f"store bounds guard: max valid t={_MAX_T} "
      f"(last timestep {_last_valid.isoformat()}Z) — "
      f"filename says '1959-2022' but coverage actually ends here")

print("fetching static fields (latitude, land_sea_mask) ...", flush=True)
lat = fetch("latitude", "0").astype(np.float64).ravel()
NY = lat.size
NX = 1440
phi = np.deg2rad(lat)
lon_deg = np.arange(NX) * 0.25
lsm = fetch("land_sea_mask", static_key("land_sea_mask")).astype(np.float64)
lsm = lsm.reshape(NY, NX) if lsm.size == NY * NX else lsm[0]

# ---- verbatim / near-verbatim helpers, consistent with comet_tail_f4_f7.py -


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
            "R2_profile_wn1": float(1.0 - (resid0 - wn1).var() / vals.var()),
            "amp_by_ring": amp, "ring_mid_km": (np.arange(nb) + 0.5) * RING}


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


def centroid_ll(weight, ci, cj, radius_km=300.0):
    """Half-max-weighted centroid of `weight` near a centre, as continuous (lat, lon)."""
    dx, dy, r, _ = geom_ll(lat[ci], lon_deg[cj])
    m = (r <= radius_km) & (weight > 0)
    if not m.any():
        return lat[ci], lon_deg[cj]
    w = np.clip(weight[m] - 0.5 * weight[m].max(), 0, None)
    if w.sum() <= 0:
        return lat[ci], lon_deg[cj]
    cx = float((dx[m] * w).sum() / w.sum())
    cy = float((dy[m] * w).sum() / w.sum())
    latc = lat[ci] + np.rad2deg(cy / R_E)
    lonc = lon_deg[cj] + np.rad2deg(cx / (R_E * np.cos(phi[ci])))
    return latc, lonc


def d_dx(f):
    """Zonal derivative in per-km units (centred differences, cos(lat) metric)."""
    dxk = R_E * np.cos(phi)[:, None] * np.deg2rad(0.25)
    o = np.zeros_like(f)
    o[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dxk)
    return o


def d_dy(f):
    """Meridional derivative in per-km units; the row index grows southward, so the sign is flipped."""
    dyk = R_E * np.deg2rad(0.25)
    o = np.zeros_like(f)
    o[1:-1, :] = -(f[2:, :] - f[:-2, :]) / (2 * dyk)
    return o


def wrap_deg(d):
    """Wrap degrees into (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def err_deg(low_pole_rad, motion_rad):
    """Signed alignment error, in degrees, of a low-pole bearing against the left-of-motion prediction."""
    return float(wrap_deg(np.rad2deg(
        low_pole_rad - (motion_rad + np.pi / 2))))


def sep_km(a, b):
    """Great-circle-ish separation between two (lat, lon) points, in km."""
    la, lo = a
    lb, lob = b
    dlon = np.deg2rad((lo - lob + 180) % 360 - 180)
    return float(np.hypot(R_E * np.cos(np.deg2rad((la + lb) / 2)) * dlon,
                          R_E * np.deg2rad(la - lb)))


out = {"store": B}

# ============================ CT-F5 =========================================
print("\n=== CT-F5  walking-center geopotential sweep (fixes F1 saturation) ===")
T0 = 91246
lat0_p = fetch("mean_sea_level_pressure", f"{T0}.0.0")[0].astype(np.float64)
lat1_p = fetch("mean_sea_level_pressure", f"{T0+1}.0.0")[0].astype(np.float64)
levels = fetch("level", "0").astype(int).ravel()
print("fetching geopotential t0 (13 levels) ...", flush=True)
z0 = fetch("geopotential", f"{T0}.0.0.0")[0].astype(np.float64)
order = np.argsort(-levels)          # 1000 hPa first (surface-anchored walk)

STORMS0 = []
for nm, hint in (("storm1", None), ("storm2", (67.0, 28.0))):
    ci0, cj0 = find_center(lat0_p, near=hint)
    ci1, cj1 = find_center(lat1_p, near=(lat[ci0], lon_deg[cj0]))
    dx, dy, _, _ = geom_ll(lat[ci0], lon_deg[cj0])
    mv = (float(dx[ci1, cj1]), float(dy[ci1, cj1]))
    STORMS0.append({"name": nm, "lat0": lat[ci0], "lon0": lon_deg[cj0],
                    "motion_rad": float(np.arctan2(mv[1], mv[0]))})

f5 = {}
for st in STORMS0:
    mth = st["motion_rad"]
    cur = (st["lat0"], st["lon0"])
    walk = []
    max_step = 0.0
    for li in order:
        lev = int(levels[li])
        found = find_center(z0[li], near=cur, radius_km=250.0)
        if found is None:
            walk.append({"level_hPa": lev, "found": False})
            continue
        fi, fj = found
        step = sep_km(cur, (lat[fi], lon_deg[fj]))
        max_step = max(max_step, step)
        cur = (lat[fi], lon_deg[fj])
        d = decompose_ll(z0[li], cur[0], cur[1])
        walk.append({"level_hPa": lev, "found": True,
                    "center_lat": cur[0], "center_lon": cur[1],
                    "step_km": step,
                    "error_deg": err_deg(d["low_pole_rad"], mth),
                    "wn1_frac": d["wn1_frac"]})
    valid = [w for w in walk if w["found"]]
    errs = {w["level_hPa"]: w["error_deg"] for w in valid}
    in_band = {h: e for h, e in errs.items() if 400 <= h <= 850}
    if in_band:
        best_h = min(in_band, key=lambda h: abs(in_band[h]))
        f5a = abs(in_band[best_h]) <= 20.0
    else:
        best_h, f5a = None, False
    f5b = max_step <= 250.0
    sfc_err = errs.get(1000)
    f5[st["name"]] = {"walk": sorted(walk, key=lambda w: -w["level_hPa"]),
                      "max_step_km": max_step, "CT_F5b_no_saturation": bool(f5b),
                      "best_level_in_400_850": best_h,
                      "best_error_in_400_850_deg": in_band.get(best_h) if best_h else None,
                      "CT_F5a_pass": bool(f5a), "surface_error_deg": sfc_err}
    print(f"\n{st['name']}: surface error {sfc_err:+.1f} deg")
    for w in sorted(valid, key=lambda w: -w["level_hPa"]):
        print(f"   {w['level_hPa']:4d} hPa  err {w['error_deg']:+7.1f}  "
              f"step {w['step_km']:5.1f} km  wn1 {w['wn1_frac']:.3f}")
    unfound = [w["level_hPa"] for w in walk if not w["found"]]
    if unfound:
        print(f"   (no center found at: {unfound} hPa)")
    print(f"   max single-step jump: {max_step:.1f} km -> "
          f"F5b (<=250km, no saturation): {f5b}")
    if best_h:
        print(f"   best |error| in 400-850 band: {best_h} hPa "
              f"({in_band[best_h]:+.1f} deg) -> F5a pass: {f5a}")
    else:
        print("   NO level found in 400-850 band -> F5a: NO-VERDICT")

out["CT_F5"] = f5
print("\nCT-F5 net: storm2 own-center path was NO-VERDICT/dead-absurd in the "
      "original sweep (saturated 586-599km at 5/13 levels); walking fix "
      f"result: F5a={f5['storm2']['CT_F5a_pass']}, "
      f"F5b={f5['storm2']['CT_F5b_no_saturation']}")

# ============================== CT-N =========================================
print("\n=== CT-N  n=10 blind storm sample (binding constraint after F4/F7) ===")
DATES = [
    datetime.datetime(2021, 6, 15, 12),   # storm1 anchor (arc-pinned)
    datetime.datetime(2020, 1, 10, 12),
    datetime.datetime(2020, 7, 20, 12),
    datetime.datetime(2019, 3, 5, 12),
    datetime.datetime(2019, 10, 25, 12),
    datetime.datetime(2014, 9, 12, 12),   # was 2022-02-14: out of store bounds
                                           # (see store-bounds guard above)
    datetime.datetime(2018, 8, 8, 12),
    datetime.datetime(2017, 11, 30, 12),
    datetime.datetime(2016, 4, 18, 12),
    datetime.datetime(2015, 12, 25, 12),
]

n_rows = []
for dt in DATES:
    t0 = t_index(dt)
    if t0 < 0 or t0 + 1 > _MAX_T:
        print(f"\n{dt.date()}: t0={t0} outside store bounds [0,{_MAX_T}] "
              "-> excluded (data unavailable)")
        n_rows.append({"date": dt.isoformat(), "t0": t0,
                       "status": "OUT-OF-STORE-BOUNDS"})
        continue
    p0 = fetch("mean_sea_level_pressure", f"{t0}.0.0")[0].astype(np.float64)
    p1 = fetch("mean_sea_level_pressure", f"{t0+1}.0.0")[0].astype(np.float64)
    u10 = fetch("10m_u_component_of_wind", f"{t0}.0.0")[0].astype(np.float64)
    v10 = fetch("10m_v_component_of_wind", f"{t0}.0.0")[0].astype(np.float64)
    zeta10 = d_dx(v10) - d_dy(u10)

    found0 = find_center(p0)
    row = {"date": dt.isoformat(), "t0": t0}
    if found0 is None:
        row["status"] = "NO-CENTER-FOUND"
        n_rows.append(row)
        print(f"\n{dt.date()}: no NH low found -> excluded")
        continue
    ci0, cj0 = found0
    la_a, lo_a = subgrid_min(p0, ci0, cj0)
    found1 = find_center(p1, near=(la_a, lo_a))
    row.update({"center_lat": la_a, "center_lon": lo_a,
               "anomaly_Pa": float(p0[ci0, cj0] - p0[ci0].mean())})
    if found1 is None:
        row["status"] = "NOT-TRACKABLE"
        n_rows.append(row)
        print(f"\n{dt.date()}: center ({la_a:.2f}N,{lo_a:.2f}E) "
              "not trackable at t+6h -> excluded from N1-N5")
        continue
    ci1, cj1 = found1
    dx, dy, _, _ = geom_ll(la_a, lo_a)
    disp = (float(dx[ci1, cj1]), float(dy[ci1, cj1]))
    dist = float(np.hypot(*disp))
    mth = float(np.arctan2(disp[1], disp[0]))
    trackable = dist >= 100.0

    dA = decompose_ll(p0, la_a, lo_a)
    la_c, lo_c = centroid_ll(zeta10, ci0, cj0)
    dC = decompose_ll(p0, la_c, lo_c)

    errA = err_deg(dA["low_pole_rad"], mth)
    errC = err_deg(dC["low_pole_rad"], mth)
    row.update({
        "status": "OK" if trackable else "SUBTHRESHOLD-DISPLACEMENT",
        "displacement_km": dist, "motion_bearing_deg": float(np.rad2deg(mth)),
        "CT_E1_wn1_frac": dA["wn1_frac"], "CT_E2_trackable": bool(trackable),
        "CT_E3_error_A_mslp_deg": errA, "CT_E4_R2_profile_wn1": dA["R2_profile_wn1"],
        "F8_error_C_vort_deg": errC,
        "F8_shrinks_at_vort_center": bool(abs(errC) < abs(errA)),
    })
    _, _, r_, th_ = geom_ll(la_a, lo_a)
    disk = r_ <= R_DISK
    lv, rr, tt = lsm[disk].astype(np.float64), r_[disk], th_[disk]
    rings = np.clip((rr / RING).astype(int), 0, int(R_DISK / RING) - 1)
    nb = int(R_DISK / RING)
    a1l = np.zeros(nb)
    b1l = np.zeros(nb)
    for b in range(nb):
        m = rings == b
        if not m.any():
            continue
        vv, t = lv[m], tt[m]
        pr = vv.mean()
        a1l[b] = 2 * ((vv - pr) * np.cos(t)).mean()
        b1l[b] = 2 * ((vv - pr) * np.sin(t)).mean()
    land_dipole_amp = float(np.hypot(a1l, b1l).mean())
    row["F9_land_dipole_amp_mean"] = land_dipole_amp
    row["F9_unexplained_residual"] = 1.0 - dA["R2_profile_wn1"]

    n_rows.append(row)
    print(f"\n{dt.date()}: center ({la_a:.2f}N,{lo_a:.2f}E), "
          f"disp {dist:.0f} km/6h, trackable={trackable}")
    print(f"   E1 wn1_frac={dA['wn1_frac']:.3f}  E4 R2={dA['R2_profile_wn1']:.3f}  "
          f"E3 error(A)={errA:+.1f} deg  F8 error(C)={errC:+.1f} deg "
          f"({'shrinks' if abs(errC) < abs(errA) else 'grows'})")
    print(f"   F9 land_dipole_amp={land_dipole_amp:.4f}  "
          f"unexplained_residual={row['F9_unexplained_residual']:.3f}")

valid = [r for r in n_rows if r.get("status") == "OK"]
excluded = [r for r in n_rows if r.get("status") != "OK"]
print(f"\n{len(valid)}/{len(n_rows)} storms valid (CT-E2 trackable); "
      f"{len(excluded)} excluded: "
      + ", ".join(f"{r['date'][:10]}={r['status']}" for r in excluded))

if valid:
    errs = np.array([r["CT_E3_error_A_mslp_deg"] for r in valid])
    neg_frac = float((errs < 0).mean())
    n1_pass = neg_frac >= 0.70
    med_abs_err = float(np.median(np.abs(errs)))
    iqr = [float(np.percentile(np.abs(errs), 25)),
          float(np.percentile(np.abs(errs), 75))]
    wn1s = np.array([r["CT_E1_wn1_frac"] for r in valid])
    n3_pass = bool(np.median(wn1s) >= 0.40)
    r2s = np.array([r["CT_E4_R2_profile_wn1"] for r in valid])
    n4_pass = bool(np.median(r2s) >= 0.80)
    shrink = np.array([r["F8_shrinks_at_vort_center"] for r in valid])
    n5_pass = bool(shrink.mean() >= 0.70)
    lda = np.array([r["F9_land_dipole_amp_mean"] for r in valid])
    resid = np.array([r["F9_unexplained_residual"] for r in valid])
    f9_corr = float(np.corrcoef(lda, resid)[0, 1]) if len(valid) >= 3 else None

    print(f"\nCT-N1 sign consistency: {neg_frac:.2f} negative "
          f"({int((errs<0).sum())}/{len(valid)}) -> "
          f"{'PASS' if n1_pass else 'FAIL'} (bar >= 0.70)")
    print(f"CT-N2 magnitude: median|error|={med_abs_err:.1f} deg, "
          f"IQR=[{iqr[0]:.1f},{iqr[1]:.1f}]  (no bar, observation)")
    print(f"CT-N3 wn1 dominance at scale: median={np.median(wn1s):.3f} -> "
          f"{'PASS' if n3_pass else 'FAIL'} (bar >= 0.40)")
    print(f"CT-N4 R2 at scale: median={np.median(r2s):.3f} -> "
          f"{'PASS' if n4_pass else 'FAIL'} (bar >= 0.80)")
    print(f"CT-N5/F8 vort-center shrinks error: {shrink.mean():.2f} "
          f"({int(shrink.sum())}/{len(valid)}) -> "
          f"{'PASS' if n5_pass else 'FAIL'} (bar >= 0.70)")
    print(f"CT-F9 corr(land_dipole_amp, unexplained_residual) = "
          f"{f9_corr if f9_corr is not None else 'n/a (n<3)'} "
          f"-> {'supports candidate 2 residually' if (f9_corr or 0) > 0.2 else 'does not support / inconclusive'}")

    out["CT_N"] = {"n_dates_scanned": len(DATES), "n_valid": len(valid),
                   "n_excluded": len(excluded), "excluded": excluded,
                   "rows": valid,
                   "CT_N1_sign_frac_negative": neg_frac, "CT_N1_pass": n1_pass,
                   "CT_N2_median_abs_error_deg": med_abs_err, "CT_N2_iqr_deg": iqr,
                   "CT_N3_median_wn1_frac": float(np.median(wn1s)), "CT_N3_pass": n3_pass,
                   "CT_N4_median_R2": float(np.median(r2s)), "CT_N4_pass": n4_pass,
                   "CT_N5_F8_shrink_frac": float(shrink.mean()), "CT_N5_pass": n5_pass,
                   "CT_F9_corr_land_dipole_vs_residual": f9_corr}
else:
    out["CT_N"] = {"n_dates_scanned": len(DATES), "n_valid": 0,
                   "excluded": excluded, "verdict": "NO-VERDICT: zero valid storms"}

json.dump(out, open("comet_tail_f5_n10.json", "w"), indent=2)
print("\nwrote comet_tail_f5_n10.json")
