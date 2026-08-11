"""EXPLORATORY — CT-F4 (sub-grid center, the blocking item) + CT-F7 (friction
over LAND, operator-requested replication). Follow-up to comet_tail_followup.py
(f6310b0e). NOT an EV; bars mine, unaudited.

WHY F4 IS SHAPED THIS WAY. CT-F3 failed its gate: a +/-100 km center jitter
moved the alignment error by up to 29.4 deg. But +/-100 km was an amplitude I
CHOSE, not one I measured — so "the apparatus dominates" was demonstrated at an
arbitrary scale. Re-running the same jitter at a smaller amplitude would be
goalpost-moving. The non-circular question is:

    how far apart do INDEPENDENT center definitions actually land?

That disagreement IS the center uncertainty. So F4's primary test needs no
jitter at all: compute the low-pole error from four center definitions built
from three different physical fields, and look at the spread of the ANSWER.
The jitter sweep is retained only as a supporting sensitivity CURVE (a knob
sweep, reported at four amplitudes, not a single chosen point).

Four center definitions (deliberately not variations of one idea):
  A  sub-grid MSLP minimum  — 2D quadratic fit on the 3x3, removes grid snap
                              (0.25 deg = 15.6 km zonal at 56N, 27.8 km merid.)
  B  grad^2 p centroid      — pressure curvature, i.e. geostrophic vorticity
  C  10m relative-vorticity centroid — WIND field, independent of pressure
  D  850 hPa geopotential minimum, sub-grid — a different field AND a
                              different altitude

PRE-REGISTERED:

CT-F4a PRIMARY. Spread of the alignment error across definitions A-D <= 10 deg
       => center choice does not dominate, and an offset becomes measurable.
       > 10 deg => the offset remains unmeasurable and no constant may be fitted;
       CT-E3's magnitude re-grade stands permanently, not provisionally.
CT-F4b SUPPORTING. Sensitivity curve: spread of the error over 4-direction
       jitters at amplitudes 25 / 50 / 100 / 200 km. Expect MONOTONE increase;
       non-monotone => the apparatus is worse than F3 already showed.
CT-F4c ANTI-VACUITY GUARD (checked BEFORE F4a is read). The four definitions
       must actually DISAGREE: max pairwise separation >= one grid diagonal
       (~32 km at 56N). If all four collapse to the same point, F4a passes
       trivially and proves nothing about method sensitivity -> NO-VERDICT.
       (This arc has shipped vacuous falsifiers repeatedly; this is the guard.)

CT-F7  FRICTION OVER LAND (operator: "Bodenreibung bei einem anderen Sturm auf
       dem Land"). Textbook: cross-isobar inflow ~10-30 deg over ocean,
       ~25-45 deg over land (roughness). CT-F2 measured +14.7 / +13.0 deg over
       ocean. If the apparatus is real it must resolve the land contrast.
       Storm selection is BLIND to the answer: deepest NH zonal-anomaly low
       whose 300-1000 km ring is >= 70% land, no inspection of its inflow.
  F7a  ACROSS-STORM: median alpha(land storm, land pts) >= median alpha(CT-F2
       ocean storm) + 8 deg, and lands in [20, 50] deg.
  F7b  WITHIN-STORM PAIRED (the stronger half — controls for depth, latitude,
       curvature, which differ between storms): inside the SAME disk,
       median alpha(land pts) > median alpha(ocean pts). Needs >= 500 points
       of each class, else NO-VERDICT on this half.
  F7c  OROGRAPHY GUARD. MSLP over high terrain is an extrapolated fiction and
       its gradient is unreliable, which would corrupt alpha through the
       geostrophic reference rather than through friction. Points with surface
       elevation > 1000 m are excluded; medians are reported BOTH ways so the
       guard's effect is visible rather than assumed. (Storms 1-2 were ocean,
       so CT-F2 is unaffected either way.)
  F7d  CONSEQUENCE, stated in advance so it cannot be spun afterwards: if the
       land median reaches ~40 deg, then over land friction ALONE could own an
       offset of that size — which would NOT overturn the storm-1/2 verdict
       (both ~99% and ocean-only) but WOULD mean the friction bound is
       surface-type dependent and cannot be applied globally.

Same store / timestep as the whole arc. geom_ll is a strict generalisation of
comet_tail_probe.py's geom to a continuous center; the identity
geom_ll(lat[ci], lon[cj]) == geom(ci, cj) is ASSERTED numerically below so the
refactor cannot silently change the baseline.
"""
import json
import urllib.request

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
T0, T1 = 91246, 91247
R_E = 6371.0
R_DISK = 1200.0
RING = 100.0
G0 = 9.80665

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=600).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


def static_key(var):
    return ".".join("0" * len(meta[f"{var}/.zarray"]["chunks"]))


print("fetching MSLP, winds, masks ...", flush=True)
lat = fetch("latitude", "0").astype(np.float64).ravel()
p0 = fetch("mean_sea_level_pressure", f"{T0}.0.0")[0].astype(np.float64)
p1 = fetch("mean_sea_level_pressure", f"{T1}.0.0")[0].astype(np.float64)
NY, NX = p0.shape
phi = np.deg2rad(lat)
lon_deg = np.arange(NX) * 0.25
u10 = fetch("10m_u_component_of_wind", f"{T0}.0.0")[0].astype(np.float64)
v10 = fetch("10m_v_component_of_wind", f"{T0}.0.0")[0].astype(np.float64)
lsm = fetch("land_sea_mask", static_key("land_sea_mask")).astype(np.float64)
lsm = lsm.reshape(NY, NX) if lsm.size == NY * NX else lsm[0]
zs = fetch("geopotential_at_surface",
           static_key("geopotential_at_surface")).astype(np.float64)
zs = zs.reshape(NY, NX) if zs.size == NY * NX else zs[0]
elev_m = zs / G0

# --------------------------------------------------------------------------


def geom_ll(latc, lonc):
    """Continuous-center generalisation of comet_tail_probe.py's geom()."""
    phic = np.deg2rad(latc)
    dlon = np.deg2rad((lon_deg[None, :] - lonc + 180) % 360 - 180)
    dphi = phi[:, None] - phic
    dx = R_E * np.cos(phic) * dlon * np.ones((NY, 1))
    dy = R_E * dphi * np.ones((1, NX))
    return dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def geom(ci, cj):
    return geom_ll(lat[ci], lon_deg[cj])


def find_center(field, near=None, radius_km=600.0):
    fa = field - field.mean(axis=1, keepdims=True)
    mask = lat[:, None] > 15
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
    """Verbatim decompose(), continuous center."""
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
            "wn1_frac": float(wn1.var() / resid0.var()),
            "R2_profile_wn1": float(1.0 - (resid0 - wn1).var() / vals.var())}


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def err_deg(low_pole_rad, motion_rad):
    return float(wrap_deg(np.rad2deg(
        low_pole_rad - (motion_rad + np.pi / 2))))


def subgrid_min(field, ci, cj):
    """2D quadratic LS fit on the 3x3; stationary point -> (lat, lon)."""
    # Longitude WRAPS: a centre at cj == 0 or NX-1 would otherwise slice a 3x2
    # neighbourhood, `A` would have 9 rows against 6 values and lstsq would
    # raise. Centres come from a global scan, so the seam at 0 deg is reachable
    # (coderabbit on PR #926, 2026-08-11). Rows are clamped, not wrapped — the
    # poles are not periodic.
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
    return (lat[ci] + di * (lat[ci + 1] - lat[ci]),
            lon_deg[cj] + dj * 0.25)


def centroid_ll(weight, ci, cj, radius_km=300.0):
    """Half-max-weighted centroid of `weight` near (ci,cj) -> (lat, lon)."""
    dx, dy, r, _ = geom(ci, cj)
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
    dxk = R_E * np.cos(phi)[:, None] * np.deg2rad(0.25)
    o = np.zeros_like(f)
    o[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dxk)
    return o


def d_dy(f):
    dyk = R_E * np.deg2rad(0.25)
    o = np.zeros_like(f)
    o[1:-1, :] = -(f[2:, :] - f[:-2, :]) / (2 * dyk)   # index grows southward
    return o


def sep_km(a, b):
    la, lo = a
    lb, lob = b
    dlon = np.deg2rad((lo - lob + 180) % 360 - 180)
    return float(np.hypot(R_E * np.cos(np.deg2rad((la + lb) / 2)) * dlon,
                          R_E * np.deg2rad(la - lb)))


# --- refactor guard: geom_ll must reproduce geom exactly -------------------
_a = geom_ll(lat[200], lon_deg[500])[2]
_b = geom(200, 500)[2]
assert np.abs(_a - _b).max() == 0.0, "geom_ll is NOT identical to geom"
print(f"geom_ll identity guard: max|diff| = {np.abs(_a - _b).max():.1e} OK")

lap_p = np.zeros_like(p0)
lap_p[1:-1, :] += p0[2:, :] + p0[:-2, :] - 2 * p0[1:-1, :]
lap_p[:, 1:-1] += p0[:, 2:] + p0[:, :-2] - 2 * p0[:, 1:-1]
zeta10 = d_dx(v10) - d_dy(u10)
gx, gy = d_dx(p0), d_dy(p0)

print("fetching geopotential t0 (13 levels) ...", flush=True)
levels = fetch("level", "0").astype(int).ravel()
z0 = fetch("geopotential", f"{T0}.0.0.0")[0].astype(np.float64)
z850 = z0[int(np.where(levels == 850)[0][0])]

out = {"store": B, "t0": T0, "t1": T1, "R_disk_km": R_DISK}

# ===================== CT-F4 : SUB-GRID CENTER =============================
print("\n=== CT-F4  independent center definitions (the blocking item) ===")
STORMS = []
for nm, hint in (("storm1", None), ("storm2", (67.0, 28.0))):
    ci0, cj0 = find_center(p0, near=hint)
    ci1, cj1 = find_center(p1, near=(lat[ci0], lon_deg[cj0]))
    dx, dy, _, _ = geom(ci0, cj0)
    mv = (float(dx[ci1, cj1]), float(dy[ci1, cj1]))
    STORMS.append({"name": nm, "ij": (ci0, cj0),
                   "motion_rad": float(np.arctan2(mv[1], mv[0]))})

f4 = {}
for st in STORMS:
    ci0, cj0 = st["ij"]
    mth = st["motion_rad"]
    zi, zj = find_center(z850, near=(lat[ci0], lon_deg[cj0]), radius_km=400.0)
    defs = {
        "A_mslp_min_subgrid": subgrid_min(p0, ci0, cj0),
        "B_lap_p_centroid": centroid_ll(lap_p, ci0, cj0),
        "C_vort10m_centroid": centroid_ll(zeta10, ci0, cj0),
        "D_z850_min_subgrid": subgrid_min(z850, zi, zj),
    }
    rows = {}
    for k, (la, lo) in defs.items():
        d = decompose_ll(p0, la, lo)
        rows[k] = {"lat": la, "lon": lo,
                   "error_deg": err_deg(d["low_pole_rad"], mth),
                   "wn1_frac": d["wn1_frac"]}
    keys = list(defs)
    pair = {f"{a}|{b}": sep_km(defs[a], defs[b])
            for i, a in enumerate(keys) for b in keys[i + 1:]}
    max_sep = max(pair.values())
    grid_diag = sep_km((lat[ci0], lon_deg[cj0]),
                       (lat[ci0 + 1], lon_deg[cj0] + 0.25))
    errs = np.array([r["error_deg"] for r in rows.values()])
    rel = wrap_deg(errs - errs[0])
    spread = float(rel.max() - rel.min())
    degenerate = max_sep < grid_diag

    curve = {}
    for amp in (25.0, 50.0, 100.0, 200.0):
        es = []
        for ang in (0, 90, 180, 270):
            a = np.deg2rad(ang) + mth
            la = lat[ci0] + np.rad2deg(amp * np.sin(a) / R_E)
            lo = lon_deg[cj0] + np.rad2deg(
                amp * np.cos(a) / (R_E * np.cos(phi[ci0])))
            es.append(err_deg(decompose_ll(p0, la, lo)["low_pole_rad"], mth))
        rl = wrap_deg(np.array(es) - errs[0])
        curve[int(amp)] = float(rl.max() - rl.min())
    amps = sorted(curve)
    monotone = all(curve[amps[i]] <= curve[amps[i + 1]] + 1e-9
                   for i in range(len(amps) - 1))

    verdict = ("NO-VERDICT (F4c degenerate: definitions coincide)" if degenerate
               else "PASS — center choice does not dominate" if spread <= 10
               else "FAIL — offset remains unmeasurable")
    f4[st["name"]] = {"definitions": rows, "pairwise_km": pair,
                      "max_separation_km": max_sep,
                      "grid_diagonal_km": grid_diag,
                      "F4c_degenerate": bool(degenerate),
                      "F4a_spread_deg": spread,
                      "F4a_pass": bool((not degenerate) and spread <= 10),
                      "F4b_curve_deg": curve, "F4b_monotone": bool(monotone),
                      "verdict": verdict}
    print(f"\n{st['name']}:")
    for k, r in rows.items():
        print(f"   {k:22s} ({r['lat']:7.3f}N,{r['lon']:8.3f}E)  "
              f"error {r['error_deg']:+7.1f}  wn1 {r['wn1_frac']:.3f}")
    print(f"   max pairwise separation {max_sep:6.1f} km "
          f"(grid diagonal {grid_diag:.1f} km) -> "
          f"F4c degenerate: {degenerate}")
    print(f"   F4a spread across A-D = {spread:.1f} deg (bar <= 10) -> {verdict}")
    print(f"   F4b sensitivity curve (km -> deg): "
          + ", ".join(f"{a}:{curve[a]:.1f}" for a in amps)
          + f"   monotone: {monotone}")
out["CT_F4"] = f4

# =============== CT-F7 : FRICTION OVER LAND (blind selection) ==============
print("\n=== CT-F7  friction over LAND (storm chosen blind to its inflow) ===")


def inflow(mask):
    bg = np.arctan2(gx[mask], -gy[mask])
    ba = np.arctan2(v10[mask], u10[mask])
    return wrap_deg(np.rad2deg(ba - bg))


fa = p0 - p0.mean(axis=1, keepdims=True)
cand = np.where((lat[:, None] > 25) & (lat[:, None] < 70), fa, np.inf)
picked = None
tried = []
work = cand.copy()
for _ in range(40):
    ci, cj = np.unravel_index(np.argmin(work), work.shape)
    if not np.isfinite(work[ci, cj]):
        break
    _, _, r, _ = geom(ci, cj)
    band = (r >= 300.0) & (r <= 1000.0)
    lf = float(lsm[band].mean())
    tried.append({"lat": float(lat[ci]), "lon": float(lon_deg[cj]),
                  "anomaly_Pa": float(fa[ci, cj]), "land_fraction": lf})
    if lf >= 0.70 and picked is None:
        picked = (ci, cj, lf)
        break
    work[r < 800.0] = np.inf

if picked is None:
    print("   no NH low with >=70% land in its 300-1000 km ring — NO-VERDICT")
    out["CT_F7"] = {"verdict": "NO-VERDICT (no qualifying land storm)",
                    "candidates_scanned": tried}
else:
    ci, cj, lf = picked
    _, _, r, _ = geom(ci, cj)
    spd = np.hypot(u10, v10)
    base = (r >= 300.0) & (r <= 1000.0) & (spd > 3.0)
    lo_oro = base & (elev_m <= 1000.0)
    land_a = inflow(base & (lsm >= 0.5))
    ocean_a = inflow(base & (lsm < 0.5))
    land_g = inflow(lo_oro & (lsm >= 0.5))
    ocean_g = inflow(lo_oro & (lsm < 0.5))
    med = lambda a: float(np.median(a)) if a.size else None       # noqa: E731
    OCEAN_REF = 14.7          # CT-F2 storm1, 99% ocean
    ml, mo = med(land_g), med(ocean_g)
    f7a = bool(ml is not None and ml >= OCEAN_REF + 8.0 and 20.0 <= ml <= 50.0)
    paired_ok = land_g.size >= 500 and ocean_g.size >= 500
    f7b = bool(paired_ok and ml is not None and mo is not None and ml > mo)
    out["CT_F7"] = {
        "center": {"lat": float(lat[ci]), "lon": float(lon_deg[cj])},
        "land_fraction_ring": lf, "candidates_scanned": tried,
        "n_land_unguarded": int(land_a.size), "n_ocean_unguarded": int(ocean_a.size),
        "n_land_oro_guarded": int(land_g.size),
        "n_ocean_oro_guarded": int(ocean_g.size),
        "median_land_unguarded_deg": med(land_a),
        "median_land_oro_guarded_deg": ml,
        "median_ocean_oro_guarded_deg": mo,
        "iqr_land_deg": [float(np.percentile(land_g, 25)),
                         float(np.percentile(land_g, 75))] if land_g.size else None,
        "ocean_reference_deg": OCEAN_REF,
        "F7a_across_storm_pass": f7a,
        "F7b_within_storm_paired_pass": f7b,
        "F7b_evaluable": bool(paired_ok),
        # The key names 40 deg and the docstring pre-registers 40 deg, so the
        # test uses 40 deg. It read >= 35.0, which would have reported true for
        # a 36 deg median under a key claiming 40 (coderabbit on PR #926).
        "F7d_friction_alone_could_own_40deg_over_land":
            bool(ml is not None and ml >= 40.0),
    }
    print(f"   storm chosen: ({lat[ci]:.2f}N, {lon_deg[cj]:.2f}E), "
          f"ring land fraction {lf:.2f}, anomaly {fa[ci, cj]:.0f} Pa")
    print(f"   land  n={land_g.size:5d} (oro-guarded, was {land_a.size}) "
          f"median {ml if ml is None else round(ml, 1)} deg")
    print(f"   ocean n={ocean_g.size:5d} median "
          f"{mo if mo is None else round(mo, 1)} deg")
    print(f"   F7a across-storm (>= {OCEAN_REF}+8 and in [20,50]): {f7a}")
    print(f"   F7b within-storm paired (land > ocean, n>=500 each): {f7b} "
          f"(evaluable: {paired_ok})")
    print(f"   F7d friction alone could own 40 deg OVER LAND: "
          f"{out['CT_F7']['F7d_friction_alone_could_own_40deg_over_land']}")

json.dump(out, open("comet_tail_f4_f7.json", "w"), indent=2)
print("\nwrote comet_tail_f4_f7.json")
