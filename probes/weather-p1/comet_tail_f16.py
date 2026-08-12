"""EXPLORATORY — CT-F16: score the dipole against the STEERING-LEVEL flow
instead of the 6h surface displacement. NOT an EV; bars mine, unaudited.

WHY THIS TEST, AND WHAT IT IS NOT.

CT-F14 -- the properly-powered, pre-registered, displacement-filtered sample --
scored the dipole's low-pole bearing against the **6h surface-centre
displacement** and returned 13/19 = 0.684, p=0.0835, NO-VERDICT. The report
names ONE moderator that this chain actually MEASURED and never wired: the
steering level. The height ladder (report SS5.2/5.8) runs monotonically 92-102
deg with a zero-crossing at 400-650 hPa on both storms it was measured on. The
physics SS2 lays out is a vortex embedded in a STEERING flow -- and 10m surface
displacement is a noisy proxy for that flow, not the flow itself.

So CT-F16 swaps ONLY the motion reference. Same storms, same centres, same
decomposition, same `err_deg` convention -- one variable changed.

**THIS IS A RE-SCORING, AND IT IS NOT A VERDICT.** The arc's own rule (SS5.10)
is explicit: *"the correct way to chase a post-hoc lead is a FRESH
pre-registered sample with the filter applied a priori -- not a re-scoring of
the sample that already failed."* That rule was written about re-FILTERING
(choosing which storms count after seeing them), and this is a change of
PREDICTOR on a fixed storm set -- a different operation, and the one the report
pre-named. It is still the same 19 storms whose surface-displacement scoring
already came back NO-VERDICT, so:

  * CT-F16 is the MECHANISTIC test -- does the physically-motivated predictor
    improve on IDENTICAL data? That question is answerable here and is worth
    answering before spending a fresh sample.
  * A VERDICT on the directional claim needs a fresh, pre-registered,
    steering-scored sample. That is **CT-F17**, named here, NOT run.

Reusing CT-F14's stored centres also removes centre-finding as a variable:
nothing about the storm selection or the disk geometry can move between arms.

STEERING FLOW, defined before the run: the disk-mean (u, v) over the SAME
1200 km disk, averaged over the **500/600/700 hPa** layer -- the textbook
extratropical steering layer, and the band containing the ladder's measured
400-650 hPa zero-crossing. Bearing = `arctan2(v_mean, u_mean)`, the identical
CCW-from-east convention `mth = arctan2(disp[1], disp[0])` uses in CT-F14, so
`err_deg` is untouched.

PRE-REGISTERED (all four stated before any fetch):

CT-F16a  PRIMARY. Sign consistency (fraction with error < 0, the same
         left-of-motion direction as every prior probe) scored against the
         steering flow, on CT-F14's 19 qualifying storms, >= 0.70.
         CT-F14's surface-displacement figure on these SAME storms is 0.684.

CT-F16b  THE PAIRED TEST, and the real content. If the steering level is the
         moderator, swapping to it should not merely nudge the sign count --
         it should TIGHTEN the residual. Bar: the standard deviation of the
         signed offset must DROP by >= 10 % versus the surface-displacement
         scoring on the same storms. **This can fail in both directions**: the
         spread can widen (steering flow is the wrong reference) or move < 10 %
         (it is not the moderator at this magnitude).

CT-F16c  ANTI-VACUITY / CAN-IT-STAY-SILENT. Two deliberately WRONG references
         are scored through the identical pipeline:
           (i)  the steering bearings DETERMINISTICALLY permuted across storms
                (storm i gets storm (i+7) mod 19's steering flow), and
           (ii) the true steering flow rotated by +90 deg.
         Neither may reach 0.70. If a scrambled or rotated reference scores as
         well as the true one, this probe measures nothing and F16a/F16b are
         void regardless of what they say. Reported FIRST in the verdicts.

CT-F16d  DESCRIPTIVE, not a bar. Sign consistency scored at EACH level
         (400/500/600/700/850 hPa separately). The ladder predicts an optimum
         inside 400-650 hPa; a flat sweep, or an optimum at 850/1000, would
         say the level structure is not what is driving any improvement.

NOT tested here: SH, tropical cyclones, or any claim beyond the extratropical
steering-flow framing. A fresh-sample verdict is CT-F17 and is not run.
"""
import json
import pathlib
import urllib.request
from math import comb

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
R_E, R_DISK, RING = 6371.0, 1200.0, 100.0
STEER_LEVELS = (500, 600, 700)      # pre-registered steering layer
SWEEP_LEVELS = (400, 500, 600, 700, 850)

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=900).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


def wrap_deg(d):
    """Wrap degrees into [-180, 180) — the identical convention CT-F14 uses."""
    return (d + 180.0) % 360.0 - 180.0


def err_deg(low_pole_rad, motion_rad):
    """Signed alignment error vs the left-of-motion prediction. VERBATIM from
    comet_tail_f14.py:205 — the paired comparison is only valid if the scoring
    function is byte-identical and ONLY the motion reference changes."""
    return float(wrap_deg(np.rad2deg(low_pole_rad - (motion_rad + np.pi / 2))))


def binom_sf_ge(k, n, p=0.5):
    """Exact one-sided binomial tail P(X >= k) for n trials at probability p."""
    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
               for i in range(k, n + 1))


lat = fetch("latitude", "0").astype(np.float64).ravel()
levels = fetch("level", "0").astype(int).ravel()
NY, NX = lat.size, 1440
phi = np.deg2rad(lat)
lon_deg = np.arange(NX) * 0.25
LEV_IDX = {int(v): i for i, v in enumerate(levels)}
print(f"levels: {list(levels)}")
print(f"steering layer (pre-registered): {STEER_LEVELS} hPa", flush=True)


def geom_ll(latc, lonc):
    """dx, dy, r (km), azimuth (rad CCW from east) about a continuous centre."""
    phic = np.deg2rad(latc)
    dlon = np.deg2rad((lon_deg[None, :] - lonc + 180) % 360 - 180)
    dx = R_E * np.cos(phic) * dlon * np.ones((NY, 1))
    dy = R_E * (phi[:, None] - phic) * np.ones((1, NX))
    return dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def low_pole_bearing(field, latc, lonc):
    """Ring-profile + per-ring wn-1 fit; returns the amplitude-weighted low-pole
    bearing. VERBATIM shape from comet_tail_f14.py's decompose_ll."""
    _, _, r, th = geom_ll(latc, lonc)
    disk = r <= R_DISK
    vals, rr, tt = field[disk], r[disk], th[disk]
    nb = int(R_DISK / RING)
    rings = np.clip((rr / RING).astype(int), 0, nb - 1)
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
    amp = np.hypot(a1, b1)
    w = amp * np.arange(nb)
    ph = np.arctan2(np.sum(b1 * w), np.sum(a1 * w))
    return float((ph + np.pi) % (2 * np.pi))


def disk_mean_uv(u3, v3, latc, lonc, lev_list):
    """Disk-mean (u, v) over the 1200 km disk, averaged across `lev_list`."""
    _, _, r, _ = geom_ll(latc, lonc)
    disk = r <= R_DISK
    us, vs = [], []
    for lev in lev_list:
        li = LEV_IDX[lev]
        us.append(u3[li][disk].mean())
        vs.append(v3[li][disk].mean())
    return float(np.mean(us)), float(np.mean(vs))


# ---- CT-F14's OWN qualifying storms (paired: nothing about selection moves) --
src = json.loads(
    pathlib.Path(__file__).with_name("comet_tail_f14.json").read_text())
storms = [r for r in src["rows"]
          if r.get("status") == "OK" and r["displacement_km"] >= 250.0]
assert len(storms) == 19, f"expected CT-F14's 19 qualifying storms, got {len(storms)}"
print(f"paired sample: {len(storms)} storms (CT-F14's qualifying subset)\n",
      flush=True)

rows = []
for i, s in enumerate(storms):
    t0, la, lo = s["t0"], s["center_lat"], s["center_lon"]
    p = fetch("mean_sea_level_pressure", f"{t0}.0.0")[0].astype(np.float64)
    u3 = fetch("u_component_of_wind", f"{t0}.0.0.0")[0].astype(np.float64)
    v3 = fetch("v_component_of_wind", f"{t0}.0.0.0")[0].astype(np.float64)

    lp = low_pole_bearing(p, la, lo)
    um, vm = disk_mean_uv(u3, v3, la, lo, STEER_LEVELS)
    steer_rad = float(np.arctan2(vm, um))
    e_steer = err_deg(lp, steer_rad)

    per_level = {}
    for lev in SWEEP_LEVELS:
        ul, vl = disk_mean_uv(u3, v3, la, lo, (lev,))
        per_level[str(lev)] = err_deg(lp, float(np.arctan2(vl, ul)))

    rows.append({
        "date": s["date"], "t0": t0, "center_lat": la, "center_lon": lo,
        "displacement_km": s["displacement_km"],
        "err_surface_deg": s["error_deg"],          # CT-F14's own number
        "err_steering_deg": e_steer,
        "steer_speed_ms": float(np.hypot(um, vm)),
        "steer_bearing_rad": steer_rad,
        "low_pole_rad": lp,
        "err_by_level_deg": per_level})
    print(f"[{i+1}/{len(storms)}] {s['date'][:10]}  "
          f"surface {s['error_deg']:+7.1f}  steering {e_steer:+7.1f}  "
          f"|steer| {np.hypot(um, vm):5.1f} m/s", flush=True)

out = {"store": B, "source": "comet_tail_f14.json qualifying subset",
       "steer_levels_hPa": list(STEER_LEVELS), "n": len(rows), "rows": rows,
       "scope": ("PAIRED re-scoring of CT-F14's own storms with ONLY the motion "
                 "reference changed. MECHANISTIC test, NOT a verdict — a "
                 "fresh-sample verdict is CT-F17, not run.")}

e_surf = np.array([r["err_surface_deg"] for r in rows])
e_steer = np.array([r["err_steering_deg"] for r in rows])
n = len(rows)


def frac_neg(e):
    """Fraction with error < 0 — the left-of-motion direction."""
    return float((e < 0).mean())


# ---- CT-F16c FIRST: if the wrong references also pass, nothing else counts ---
perm = np.array([rows[(i + 7) % n]["steer_bearing_rad"] for i in range(n)])
e_perm = np.array([err_deg(rows[i]["low_pole_rad"], perm[i]) for i in range(n)])
e_rot = np.array([err_deg(r["low_pole_rad"], r["steer_bearing_rad"] + np.pi / 2)
                  for r in rows])
f_perm, f_rot = frac_neg(e_perm), frac_neg(e_rot)
c_ok = (f_perm < 0.70) and (f_rot < 0.70)

print("\n=== CT-F16c ANTI-VACUITY (reported first — gates everything else) ===")
print(f"  permuted steering bearings : {f_perm:.3f}  (must be < 0.70)")
print(f"  steering rotated +90 deg   : {f_rot:.3f}  (must be < 0.70)")
print(f"  -> {'PASS — the test discriminates' if c_ok else 'FAIL — VOID: a wrong reference scores as well'}")

f_surf, f_steer = frac_neg(e_surf), frac_neg(e_steer)
p_steer = binom_sf_ge(int((e_steer < 0).sum()), n)
sd_surf, sd_steer = float(e_surf.std(ddof=1)), float(e_steer.std(ddof=1))
drop = (sd_surf - sd_steer) / sd_surf

print("\n=== CT-F16a PRIMARY — sign consistency vs the steering flow ===")
print(f"  surface displacement (CT-F14) : {f_surf:.3f} ({int((e_surf<0).sum())}/{n})")
print(f"  steering flow 500/600/700 hPa : {f_steer:.3f} ({int((e_steer<0).sum())}/{n}), "
      f"one-sided p={p_steer:.4f}")
print(f"  -> {'PASS' if f_steer >= 0.70 else 'FAIL'} (bar >= 0.70)")

print("\n=== CT-F16b PAIRED — does the residual TIGHTEN? ===")
print(f"  sd(signed offset) surface  : {sd_surf:7.2f} deg")
print(f"  sd(signed offset) steering : {sd_steer:7.2f} deg   ({100*drop:+.1f} %)")
print(f"  -> {'PASS' if drop >= 0.10 else 'FAIL'} (bar: >= 10 % reduction)")

print("\n=== CT-F16d DESCRIPTIVE — sign consistency by level ===")
lev_frac = {}
for lev in SWEEP_LEVELS:
    e = np.array([r["err_by_level_deg"][str(lev)] for r in rows])
    lev_frac[str(lev)] = {"sign_neg_frac": frac_neg(e),
                          "sd_deg": float(e.std(ddof=1))}
    print(f"  {lev:4d} hPa : frac {frac_neg(e):.3f}   sd {e.std(ddof=1):6.1f} deg")
best = max(lev_frac, key=lambda k: lev_frac[k]["sign_neg_frac"])
print(f"  best level: {best} hPa "
      f"({'inside' if 400 <= int(best) <= 650 else 'OUTSIDE'} the ladder's "
      "400-650 hPa zero-crossing band)")

out["verdicts"] = {
    "CT_F16c_antivacuity": {"permuted_frac": f_perm, "rotated90_frac": f_rot,
                            "discriminates": bool(c_ok)},
    "CT_F16a": {"surface_frac": f_surf, "steering_frac": f_steer,
                "one_sided_p": p_steer, "n": n,
                "pass": bool(f_steer >= 0.70)},
    "CT_F16b": {"sd_surface_deg": sd_surf, "sd_steering_deg": sd_steer,
                "sd_reduction_frac": drop, "pass": bool(drop >= 0.10)},
    "CT_F16d_by_level": lev_frac, "best_level_hPa": int(best)}

with open(pathlib.Path(__file__).with_name("comet_tail_f16.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote comet_tail_f16.json")
