"""EXPLORATORY follow-up to comet_tail_probe.py (db57aac0) — CT-F1/F2/F3 as
pre-registered in COMET_TAIL_REPORT.md sec.5. NOT an EV; bars mine, unaudited.

THE QUESTION. The comet-tail probe found the wn-1 low pole left-of-motion on
2/2 storms, but with a COMMON offset of -42 deg / -40 deg from the naive
geostrophic prediction. Same magnitude, same sense, two storms => a systematic
mechanism, not noise. Three ranked candidates were named; this probe tests all
three, APPARATUS FIRST (a systematic number is a claim about the measurement
apparatus until proven otherwise -- the arc's standing rule).

geom/find_center/decompose are COPIED VERBATIM from comet_tail_probe.py so
every number below is comparable to the committed baseline. Baseline to beat:
storm1 error -41.97 deg, storm2 error -40.21 deg.

PRE-REGISTERED, in run order:

CT-F3  APPARATUS (gates the other two). Recompute the low-pole bearing from 6
       center choices per storm: the baseline MSLP minimum; a Laplacian-
       centroid center (grad^2 p is proportional to geostrophic vorticity, so
       its centroid is a circulation-center proxy); and +/-100 km jitters
       along- and across-track. A 100 km miscentering of a monopole INJECTS a
       wn-1 by construction, so this measures how much of the -40 deg is a
       choice I made with ~100 km of arbitrariness.
         spread <= 10 deg  -> ROBUST (offset is not center choice)
         10 < spread <= 20 -> SURVIVES-WITH-UNCERTAINTY
         spread  > 20 deg  -> APPARATUS-DOMINATED: CT-E3's verdict must be
                              re-graded and F1/F2 are moot.
       Directional sub-test: if the Laplacian-centroid center moves the error
       TOWARD zero, that is positive evidence the offset was center bias.

CT-F1  STEERING LEVEL (candidate 1: baroclinic tilt). The store ships all 13
       pressure levels in ONE chunk, so the yes/no becomes a sweep: decompose
       geopotential at every level about that level's own low center, and
       score each against the SAME surface-measured motion bearing.
         PASS  the |error|-minimising level lies in 400-850 hPa (the textbook
               extratropical steering layer) AND |error| there <= 20 deg.
         DEAD  the error is FLAT across levels (max-min < 15 deg) -- level
               structure then explains nothing -- or the minimising level sits
               at 50-100 hPa (physically absurd => apparatus).
       Reported twice: about each level's own center, and about the SURFACE
       center, to separate "the field changes with height" from "the center
       moves with height".

CT-F2  FRICTION (candidate 2: Ekman turning). Measure the actual 10m cross-
       isobar inflow angle: alpha = bearing(v10m) - bearing(v_geostrophic),
       positive = CCW = turned toward the low (the friction sign in NH).
       Sampled on rings 300-1000 km (excluding the core, where gradient-wind
       curvature is largest, and the outer edge), |v10m| > 3 m/s.
         Expect median alpha in [5, 40] deg over open ocean (textbook 10-30).
         The bar that matters: if median alpha << 40 deg, friction CANNOT own
         the -40 deg offset alone and is at most a contributor. If median
         alpha >= 40 deg, friction alone remains sufficient and this probe
         does NOT separate it from candidate 1.
       Land fraction from land_sea_mask is reported so "over ocean" is a
       measurement, not an assumption.

NOT tested here (stated, not hidden): n is still 2 storms at one synoptic
time. No offset constant may be baked into any predictor on this evidence;
the report's n>=10 rule stands regardless of how these three come out.

RUN LOG (transparency — the bars above are VERBATIM as first run, unedited):
  run 1  CT-F3 and CT-F1 completed; CT-F2 crashed on a shape bug in grad_p
         (dxk was (NY,1) and was being column-sliced). Fixed. Also ADDED a
         DIAGNOSTIC field `center_search_saturated` to CT-F1 after run 1
         exposed that find_center can pin at exactly the 600 km search radius
         — i.e. it found no co-located upper center and locked onto a
         different system. That is a diagnostic, not a bar: no pass/fail
         criterion above was added, removed, or loosened.
"""
import json
import pathlib
import urllib.request

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
T0, T1 = 91246, 91247
R_E = 6371.0
R_DISK = 1200.0
RING = 100.0

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=600).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


print("fetching MSLP t0,t1 ...", flush=True)
lat = fetch("latitude", "0").astype(np.float64).ravel()
p0 = fetch("mean_sea_level_pressure", f"{T0}.0.0")[0].astype(np.float64)
p1 = fetch("mean_sea_level_pressure", f"{T1}.0.0")[0].astype(np.float64)
NY, NX = p0.shape
phi = np.deg2rad(lat)
lon_deg = np.arange(NX) * 0.25

# ---- verbatim from comet_tail_probe.py -------------------------------------


def geom(ci, cj):
    """dx,dy,r (km) and azimuth theta (rad, CCW from east) rel. to center."""
    dlon = np.deg2rad((lon_deg[None, :] - lon_deg[cj] + 180) % 360 - 180)
    dphi = phi[:, None] - phi[ci]
    dx = R_E * np.cos(phi[ci]) * dlon * np.ones((NY, 1))
    dy = R_E * dphi * np.ones((1, NX))
    return dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def find_center(field, near=None, radius_km=600.0):
    """Deepest zonal-anomaly low, globally NH or within radius of `near`."""
    fa = field - field.mean(axis=1, keepdims=True)
    mask = lat[:, None] > 15
    if near is not None:
        _, _, r, _ = geom(*near)
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


def decompose(field, ci, cj):
    """Ring-mean + per-ring wn-1 fit; returns metrics dict."""
    dx, dy, r, th = geom(ci, cj)
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
        c, s = np.cos(t), np.sin(t)
        a1[b] = 2 * ((v - prof[b]) * c).mean()
        b1[b] = 2 * ((v - prof[b]) * s).mean()

    resid0 = vals - prof[rings]
    wn1 = a1[rings] * np.cos(tt) + b1[rings] * np.sin(tt)
    resid1 = resid0 - wn1

    var_t = vals.var()
    e1 = 1.0 - resid0.var() / var_t
    e2 = 1.0 - resid1.var() / var_t
    wn1_frac = wn1.var() / resid0.var()

    amp = np.hypot(a1, b1)
    w = amp * np.arange(nb)
    ph = np.arctan2(np.sum(b1 * w), np.sum(a1 * w))
    low_pole = (ph + np.pi) % (2 * np.pi)
    r_mid = (np.arange(nb) + 0.5) * RING
    a_corr = float(np.corrcoef(amp[1:], r_mid[1:])[0, 1])
    return {"R2_profile": float(e1), "R2_profile_wn1": float(e2),
            "wn1_frac_of_resid": float(wn1_frac),
            "low_pole_bearing_rad": float(low_pole),
            "amp_vs_r_corr": a_corr}

# ---- helpers new to this probe ---------------------------------------------


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
    pred = (motion_rad + np.pi / 2) % (2 * np.pi)
    return float(wrap_deg(np.rad2deg(low_pole_rad - pred)))


def offset_center(ci, cj, dx_km, dy_km):
    """Grid point nearest the (dx,dy) km offset from (ci,cj)."""
    dx, dy, _, _ = geom(ci, cj)
    d2 = (dx - dx_km) ** 2 + (dy - dy_km) ** 2
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    return int(i), int(j)


def lap_centroid(field, ci, cj, radius_km=300.0):
    """Centroid of grad^2 p (proportional to geostrophic vorticity) near the
    center — a circulation-center proxy independent of the depth minimum."""
    lap = np.zeros_like(field)
    lap[1:-1, :] += field[2:, :] + field[:-2, :] - 2 * field[1:-1, :]
    lap[:, 1:-1] += field[:, 2:] + field[:, :-2] - 2 * field[:, 1:-1]
    dx, dy, r, _ = geom(ci, cj)
    m = (r <= radius_km) & (lap > 0)
    if not m.any():
        return ci, cj
    w = lap[m] - 0.5 * lap[m].max()
    w = np.where(w > 0, w, 0.0)
    if w.sum() <= 0:
        return ci, cj
    return offset_center(ci, cj,
                         float((dx[m] * w).sum() / w.sum()),
                         float((dy[m] * w).sum() / w.sum()))


def grad_p(field):
    """d p/dx, d p/dy in Pa/km on the sphere (centered differences)."""
    dy_km = R_E * np.deg2rad(0.25)
    gy = np.zeros_like(field)
    gy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dy_km)
    gy = -gy                       # index increases southward => flip sign
    gx = np.zeros_like(field)
    dxk = R_E * np.cos(phi)[:, None] * np.deg2rad(0.25)   # (NY,1), broadcasts
    gx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dxk)
    return gx, gy


def static_key(var):
    """Chunk key for a variable, all-zero index of the right arity."""
    return ".".join("0" * len(meta[f"{var}/.zarray"]["chunks"]))


# ---- storms: reproduce the baseline track ----------------------------------
STORMS = []
for name, hint in (("storm1", None), ("storm2", (int(round((90 - 67.0) / 0.25)),
                                                 int(round(28.0 / 0.25))))):
    ci0, cj0 = find_center(p0, near=hint)
    ci1, cj1 = find_center(p1, near=(ci0, cj0))
    dx, dy, _, _ = geom(ci0, cj0)
    mv = np.array([dx[ci1, cj1], dy[ci1, cj1]])
    STORMS.append({"name": name, "c0": (ci0, cj0),
                   "motion_rad": float(np.arctan2(mv[1], mv[0])),
                   "disp_km": float(np.hypot(*mv))})

# STORMS carries the centre, motion bearing and displacement each per-storm
# CT-F1/F2/F3 record is computed from; writing `{}` dropped that provenance
# (coderabbit on PR #926, 2026-08-11).
out = {
    "store": B, "t0": T0, "t1": T1, "R_disk_km": R_DISK,
    "storms": {
        st["name"]: {
            "center_t0": {"lat": float(lat[st["c0"][0]]),
                          "lon": float(lon_deg[st["c0"][1]])},
            "motion_bearing_deg": float(np.rad2deg(st["motion_rad"])),
            "displacement_km": st["disp_km"],
        }
        for st in STORMS
    },
}

# ============================ CT-F3 : APPARATUS =============================
print("\n=== CT-F3  APPARATUS (center sensitivity) — runs first, gates F1/F2 ===")
f3_all = {}
for st in STORMS:
    ci0, cj0 = st["c0"]
    mth = st["motion_rad"]
    al = np.array([np.cos(mth), np.sin(mth)]) * 100.0     # along-track 100 km
    ac = np.array([-np.sin(mth), np.cos(mth)]) * 100.0    # across-track 100 km
    variants = {
        "mslp_min_baseline": (ci0, cj0),
        "laplacian_centroid": lap_centroid(p0, ci0, cj0),
        "jitter_along_+100km": offset_center(ci0, cj0, *al),
        "jitter_along_-100km": offset_center(ci0, cj0, *(-al)),
        "jitter_across_+100km": offset_center(ci0, cj0, *ac),
        "jitter_across_-100km": offset_center(ci0, cj0, *(-ac)),
    }
    rows = {}
    for k, (i, j) in variants.items():
        d = decompose(p0, i, j)
        rows[k] = {"lat": float(lat[i]), "lon": float(lon_deg[j]),
                   "error_deg": err_deg(d["low_pole_bearing_rad"], mth),
                   "wn1_frac": d["wn1_frac_of_resid"],
                   "R2_profile_wn1": d["R2_profile_wn1"]}
    errs = np.array([r["error_deg"] for r in rows.values()])
    # circular spread about the baseline
    rel = wrap_deg(errs - rows["mslp_min_baseline"]["error_deg"])
    spread = float(rel.max() - rel.min())
    verdict = ("ROBUST" if spread <= 10 else
               "SURVIVES-WITH-UNCERTAINTY" if spread <= 20 else
               "APPARATUS-DOMINATED")
    base_e = rows["mslp_min_baseline"]["error_deg"]
    lap_e = rows["laplacian_centroid"]["error_deg"]
    toward_zero = abs(lap_e) < abs(base_e)
    f3_all[st["name"]] = {"variants": rows, "spread_deg": spread,
                          "verdict": verdict,
                          "lap_center_moves_error_toward_zero": bool(toward_zero),
                          "baseline_error_deg": base_e,
                          "lap_error_deg": lap_e}
    print(f"\n{st['name']}: baseline error {base_e:+.1f} deg")
    for k, r in rows.items():
        print(f"   {k:22s} ({r['lat']:.2f}N,{r['lon']:.2f}E)  "
              f"error {r['error_deg']:+7.1f}  wn1_frac {r['wn1_frac']:.3f}")
    print(f"   spread across 6 centers = {spread:.1f} deg -> {verdict}")
    print(f"   Laplacian-centroid moves error toward zero: {toward_zero}")

f3_worst = max(v["spread_deg"] for v in f3_all.values())
f3_gate = f3_worst <= 20.0
out["CT_F3"] = {"per_storm": f3_all, "worst_spread_deg": f3_worst,
                "gate_passed": bool(f3_gate)}
print(f"\nCT-F3 GATE: worst spread {f3_worst:.1f} deg -> "
      f"{'PASS (F1/F2 interpretable)' if f3_gate else 'FAIL (F1/F2 moot; CT-E3 must be re-graded)'}")

# ========================= CT-F1 : STEERING LEVEL ===========================
print("\n=== CT-F1  STEERING LEVEL (geopotential sweep, 13 levels) ===")
levels = fetch("level", "0").astype(int).ravel()
print(f"fetching geopotential t0 (13 levels, one chunk) ...", flush=True)
z0 = fetch("geopotential", f"{T0}.0.0.0")[0].astype(np.float64)   # (13,ny,nx)

f1_all = {}
for st in STORMS:
    ci0, cj0 = st["c0"]
    mth = st["motion_rad"]
    per_level = []
    for li, lev in enumerate(levels):
        fld = z0[li]
        zi, zj = find_center(fld, near=(ci0, cj0))
        d_own = decompose(fld, zi, zj)
        d_sfc = decompose(fld, ci0, cj0)
        _, _, r_, _ = geom(ci0, cj0)
        per_level.append({
            "level_hPa": int(lev),
            "center_lat": float(lat[zi]), "center_lon": float(lon_deg[zj]),
            "center_offset_km": float(r_[zi, zj]),
            # DIAGNOSTIC (added after run 1, not a bar): the finder pinned at
            # the 600 km search radius => no co-located upper center found.
            "center_search_saturated": bool(r_[zi, zj] > 580.0),
            "error_own_center_deg": err_deg(d_own["low_pole_bearing_rad"], mth),
            "error_sfc_center_deg": err_deg(d_sfc["low_pole_bearing_rad"], mth),
            "wn1_frac": d_own["wn1_frac_of_resid"],
            "R2_profile_wn1": d_own["R2_profile_wn1"],
        })
    errs = np.array([p["error_own_center_deg"] for p in per_level])
    best = int(np.argmin(np.abs(errs)))
    flat = float(errs.max() - errs.min())
    best_lev = int(levels[best])
    passed = (400 <= best_lev <= 850) and abs(errs[best]) <= 20.0
    dead_flat = flat < 15.0
    dead_absurd = best_lev <= 100
    f1_all[st["name"]] = {
        "per_level": per_level, "best_level_hPa": best_lev,
        "best_error_deg": float(errs[best]),
        "sfc_error_deg": float(err_deg(
            decompose(p0, ci0, cj0)["low_pole_bearing_rad"], mth)),
        "error_spread_across_levels_deg": flat,
        "CT_F1_pass": bool(passed),
        "CT_F1_dead_flat": bool(dead_flat),
        "CT_F1_dead_absurd_level": bool(dead_absurd),
    }
    print(f"\n{st['name']}: (surface MSLP error "
          f"{f1_all[st['name']]['sfc_error_deg']:+.1f} deg)")
    for p in per_level:
        print(f"   {p['level_hPa']:4d} hPa  err(own ctr) {p['error_own_center_deg']:+7.1f}  "
              f"err(sfc ctr) {p['error_sfc_center_deg']:+7.1f}  "
              f"ctr offset {p['center_offset_km']:5.0f} km  "
              f"wn1 {p['wn1_frac']:.3f}")
    print(f"   |error| minimised at {best_lev} hPa ({errs[best]:+.1f} deg); "
          f"spread across levels {flat:.1f} deg")
    print(f"   CT-F1 pass(400-850 hPa & |err|<=20): {passed} | "
          f"dead-flat(<15 deg spread): {dead_flat} | "
          f"dead-absurd(<=100 hPa): {dead_absurd}")
out["CT_F1"] = f1_all

# ============================ CT-F2 : FRICTION ==============================
print("\n=== CT-F2  FRICTION (10m cross-isobar inflow angle) ===")
u10 = fetch("10m_u_component_of_wind", f"{T0}.0.0")[0].astype(np.float64)
v10 = fetch("10m_v_component_of_wind", f"{T0}.0.0")[0].astype(np.float64)
lsm = fetch("land_sea_mask", static_key("land_sea_mask")).astype(np.float64)
lsm = lsm.reshape(NY, NX) if lsm.size == NY * NX else lsm[0]
gx, gy = grad_p(p0)

f2_all = {}
for st in STORMS:
    ci0, cj0 = st["c0"]
    _, _, r, _ = geom(ci0, cj0)
    band = (r >= 300.0) & (r <= 1000.0)
    spd = np.hypot(u10, v10)
    m = band & (spd > 3.0)
    # geostrophic direction: v_g ∝ (-dp/dy, dp/dx)
    bg = np.arctan2(gx[m], -gy[m])
    ba = np.arctan2(v10[m], u10[m])
    alpha = wrap_deg(np.rad2deg(ba - bg))     # + = CCW = turned toward the low
    land = float(lsm[m].mean())
    ocean = alpha[lsm[m] < 0.5]
    med = float(np.median(alpha))
    f2_all[st["name"]] = {
        "n_points": int(m.sum()), "land_fraction": land,
        "median_inflow_deg": med,
        "q25_inflow_deg": float(np.percentile(alpha, 25)),
        "q75_inflow_deg": float(np.percentile(alpha, 75)),
        "median_inflow_ocean_only_deg":
            float(np.median(ocean)) if ocean.size else None,
        "n_ocean_points": int(ocean.size),
        "friction_alone_could_own_40deg": bool(abs(med) >= 40.0),
    }
    o = f2_all[st['name']]['median_inflow_ocean_only_deg']
    print(f"\n{st['name']}: n={int(m.sum())} pts in 300-1000 km, "
          f"land fraction {land:.2f}")
    print(f"   inflow angle median {med:+.1f} deg "
          f"(IQR {f2_all[st['name']]['q25_inflow_deg']:+.1f} .. "
          f"{f2_all[st['name']]['q75_inflow_deg']:+.1f})")
    print(f"   ocean-only median: "
          f"{('%+.1f deg' % o) if o is not None else 'n/a'} "
          f"(n={f2_all[st['name']]['n_ocean_points']})")
    print(f"   friction alone could own the -40 deg offset: "
          f"{f2_all[st['name']]['friction_alone_could_own_40deg']}")
out["CT_F2"] = f2_all

with open(pathlib.Path(__file__).with_name("comet_tail_followup.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote comet_tail_followup.json")
