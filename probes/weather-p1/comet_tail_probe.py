"""EXPLORATORY — 'die fehlende Achsensymmetrie mit Flugzeug-Formel x Traegheit
berechnen, wie beim Kometen dessen Schweif nach hinten drueckt' (operator,
2026-08-11). NOT an EV; bars mine, unaudited.

The physics is textbook, not invented: a translating vortex = a vortex in a
steering flow (the airplane's relative wind). Geostrophy makes it SIGNED and
falsifiable: if the storm moves WITH the geostrophic steering flow, the
background pressure gradient is PERPENDICULAR to the motion with the LOW pole
to the LEFT of motion (NH). A linear background gradient has zero ring-mean,
so it survives ring-profile removal ENTIRELY as a wavenumber-1 residual with
amplitude growing ~linearly in r. The comet tail IS wavenumber-1.

PRE-REGISTERED (per storm; two storms = replication, n=2 smallness stated):
  CT-E1  wavenumber-1 carries >= 0.40 of the azimuthal-residual variance in
         the disk (R=1200 km) — the tail is the DOMINANT asymmetry mode.
  CT-E2  motion trackable: displacement t -> t+6h >= 100 km within a 600 km
         search radius, else NO-VERDICT (direction undefined below that).
  CT-E3  THE SIGNED TEST: bearing(wn-1 low pole) = bearing(motion) + 90 deg
         (CCW, x=east/y=north) within +/-45 deg. Null probability 0.25 per
         storm; 2/2 hits = 0.0625 under null — reported as exactly that.
  CT-E4  ring-profile + per-ring wn-1 explains >= 0.80 of in-disk variance
         (single-storm baseline was 0.639 profile-only).
  CT-E5  bonus observation (no bar): wn-1 amplitude a1(r) approx linear in r
         (the linear-background signature); report corr(a1, r).

Same store as the arc (WB2 ERA5 6h). t=91246 -> 91247 (2021-06-15 12Z->18Z).
Storm 1: the arc's storm (55.75N 334.5E). Storm 2: the #3 center from
go_territory_probe (67.0N 28.0E) as replication.
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

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=180).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


lat = fetch("latitude", "0").astype(np.float64).ravel()
p0 = fetch("mean_sea_level_pressure", f"{T0}.0.0")[0].astype(np.float64)
p1 = fetch("mean_sea_level_pressure", f"{T1}.0.0")[0].astype(np.float64)
phi = np.deg2rad(lat)
lon_deg = np.arange(p0.shape[1]) * 0.25


def geom(ci, cj):
    """dx,dy,r (km) and azimuth theta (rad, CCW from east) rel. to center."""
    dlon = np.deg2rad((lon_deg[None, :] - lon_deg[cj] + 180) % 360 - 180)
    dphi = phi[:, None] - phi[ci]
    dx = R_E * np.cos(phi[ci]) * dlon * np.ones((p0.shape[0], 1))
    dy = R_E * dphi * np.ones((1, p0.shape[1]))
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
        # least-squares wn-1 on the ring (cos/sin nearly orthogonal on a ring)
        a1[b] = 2 * ((v - prof[b]) * c).mean()
        b1[b] = 2 * ((v - prof[b]) * s).mean()

    resid0 = vals - prof[rings]                      # after profile
    wn1 = a1[rings] * np.cos(tt) + b1[rings] * np.sin(tt)
    resid1 = resid0 - wn1                            # after profile + wn-1

    # CONSTRAINED dipole — the model the report's storage claim actually
    # describes: ONE amplitude slope + ONE bearing, i.e. a1(r) = s*r*cos(t0),
    # b1(r) = s*r*sin(t0), which is the linear-background signature from §2.
    # The per-ring fit above has 2*nb = 24 free parameters and is NOT a
    # 2-value representation; conflating the two overstated the compression
    # (codex P1 on PR #926, 2026-08-11). Both are now reported.
    X = np.column_stack([rr * np.cos(tt), rr * np.sin(tt)])
    coef, *_ = np.linalg.lstsq(X, resid0, rcond=None)
    wn1_con = X @ coef

    var_t = vals.var()
    e1 = 1.0 - resid0.var() / var_t                  # profile-only R2
    e2 = 1.0 - resid1.var() / var_t                  # profile + per-ring wn-1
    e2c = 1.0 - (resid0 - wn1_con).var() / var_t     # profile + 2-param dipole
    wn1_frac = wn1.var() / resid0.var()

    # amplitude-weighted dipole phase: bearing of the LOW pole
    amp = np.hypot(a1, b1)
    w = amp * np.arange(nb)                          # outer rings weightier
    ph = np.arctan2(np.sum(b1 * w), np.sum(a1 * w))  # HIGH pole bearing
    low_pole = (ph + np.pi) % (2 * np.pi)
    r_mid = (np.arange(nb) + 0.5) * RING
    a_corr = float(np.corrcoef(amp[1:], r_mid[1:])[0, 1])
    return {"R2_profile": float(e1), "R2_profile_wn1": float(e2),
            "R2_profile_wn1_constrained_2param": float(e2c),
            "n_params_profile": int(nb),
            "n_params_profile_wn1_perring": int(nb + 2 * nb),
            "n_params_profile_wn1_constrained": int(nb + 2),
            "constrained_slope_pa_per_km": float(np.hypot(*coef)),
            "constrained_bearing_deg": float(np.rad2deg(np.arctan2(coef[1], coef[0]))),
            "wn1_frac_of_resid": float(wn1_frac),
            "low_pole_bearing_rad": float(low_pole),
            "amp_vs_r_corr": a_corr}


def track(name, c0_hint=None):
    ci0, cj0 = find_center(p0, near=c0_hint)
    ci1, cj1 = find_center(p1, near=(ci0, cj0))
    dx, dy, _, _ = geom(ci0, cj0)
    disp = np.array([dx[ci1, cj1], dy[ci1, cj1]])
    dist = float(np.hypot(*disp))
    motion_bearing = float(np.arctan2(disp[1], disp[0]))
    d = decompose(p0, ci0, cj0)

    predicted = (motion_bearing + np.pi / 2) % (2 * np.pi)   # left of motion
    diff = (d["low_pole_bearing_rad"] - predicted + np.pi) % (2 * np.pi) - np.pi
    res = {
        "center_t0": {"lat": float(lat[ci0]), "lon": float(lon_deg[cj0])},
        "center_t1": {"lat": float(lat[ci1]), "lon": float(lon_deg[cj1])},
        "displacement_km": dist,
        "motion_bearing_deg": float(np.rad2deg(motion_bearing)),
        "low_pole_bearing_deg": float(np.rad2deg(d["low_pole_bearing_rad"])),
        "predicted_low_pole_deg": float(np.rad2deg(predicted)),
        "alignment_error_deg": float(np.rad2deg(diff)),
        **d,
        "CT_E1_wn1_frac": d["wn1_frac_of_resid"],
        "CT_E2_trackable": dist >= 100.0,
        "CT_E3_left_of_motion_within_45": bool(abs(np.rad2deg(diff)) <= 45),
        "CT_E4_R2_profile_wn1": d["R2_profile_wn1"],
    }
    print(f"\n{name}: center ({lat[ci0]:.2f}N, {lon_deg[cj0]:.2f}E) "
          f"-> ({lat[ci1]:.2f}N, {lon_deg[cj1]:.2f}E), {dist:.0f} km/6h "
          f"(bearing {np.rad2deg(motion_bearing):.0f} deg CCW-from-east)")
    print(f"  CT-E1 wn-1 share of residual: {d['wn1_frac_of_resid']:.3f} "
          f"(bar >= 0.40)")
    print(f"  CT-E2 trackable: {res['CT_E2_trackable']} ({dist:.0f} km)")
    print(f"  CT-E3 low pole {res['low_pole_bearing_deg']:.0f} deg vs "
          f"predicted (motion+90) {res['predicted_low_pole_deg']:.0f} deg "
          f"-> error {res['alignment_error_deg']:+.0f} deg "
          f"({'HIT' if res['CT_E3_left_of_motion_within_45'] else 'MISS'}, "
          f"bar +/-45)")
    print(f"  CT-E4 R2 profile-only {d['R2_profile']:.3f} -> "
          f"profile+wn1 {d['R2_profile_wn1']:.3f} (bar >= 0.80)")
    print(f"        [param count: {d['n_params_profile_wn1_perring']} per-ring "
          f"vs {d['n_params_profile_wn1_constrained']} constrained]")
    print(f"        CONSTRAINED 2-param dipole (the storage claim's actual "
          f"model): R2 {d['R2_profile_wn1_constrained_2param']:.3f}")
    print(f"  CT-E5 corr(a1(r), r) = {d['amp_vs_r_corr']:.3f} "
          f"(linear-background signature; observation, no bar)")
    return res


print("== comet-tail probe: wavenumber-1 asymmetry vs storm motion ==")
s1 = track("STORM 1 (the arc's storm)")
# storm 2 hint: 67.0N 28.0E from go_territory_probe
hint = (int(round((90 - 67.0) / 0.25)), int(round(28.0 / 0.25)))
s2 = track("STORM 2 (replication)", c0_hint=hint)

hits = sum(s["CT_E3_left_of_motion_within_45"] for s in (s1, s2))
print(f"\nCT-E3 joint: {hits}/2 within +/-45 deg of left-of-motion "
      f"(null prob per storm 0.25; 2/2 -> p=0.0625 — n=2, stated, "
      f"not overclaimed)")

json.dump({"store": B, "t0": T0, "t1": T1, "R_disk_km": R_DISK,
           "storm1": s1, "storm2": s2, "CT_E3_joint_hits": int(hits)},
          open("comet_tail_probe.json", "w"), indent=2)
print("wrote comet_tail_probe.json")
