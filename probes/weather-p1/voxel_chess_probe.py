"""EXPLORATORY — 'volumetrisches Voxel-Schach': rotation physics as BITBOARD ops,
raw vs u8-palette substrate. (Operator frame, 2026-08-11.)

The claim under test, stated as chess machinery: threshold masks over the voxel
grid are BITBOARDS, and the rotation physics of pressure systems is decidable by
POPCOUNTS over mask intersections — Stockfish's popcount(attacks & targets),
with weather masks. If the u8 palette substrate carries the physics, the same
popcounts on quantized fields give the same verdicts.

PRE-REGISTERED, before the run (NOT independently audited — measurements, not
an EV; promotion to an EV requires the adversarial audit gate per plan section 8):
  E1  NH lows rotate counterclockwise:  pop(L & Zpos & NH)/pop(L & NH) > 0.5
  E2  SH lows rotate clockwise:         pop(L & Zpos & SH)/pop(L & SH) < 0.5
      -- E1 vs E2 is the built-in two-sided control: the SAME statistic must
         INVERT across the equator. A wiring bug that ignores hemisphere
         cannot produce the inversion.
  E3  Highs mirror lows in both hemispheres (anticyclonic).
  E4  The u8-palette arm reproduces every E1-E3 verdict, fractions within
      0.05 of raw.
  E5  Geostrophic move-generation: corr(u_g, u) and corr(v_g, v) > 0.5 in
      each hemisphere band (|lat| in 20..70), raw AND palette.
  E6  Rankine ('Gluecksrad'): around the deepest NH low, azimuthal-mean
      tangential wind is positive (cyclonic), rises from the core (torque /
      solid-body zone), peaks, then decays outward (momentum zone).

Data: WeatherBench2 ERA5 6-hourly 0.25 deg, t=91246 (2021-06-15 12Z).
Fields: mean_sea_level_pressure, 10m u, 10m v. cos(lat) zonal spacing per the
EV-1 audit lesson; |lat| < 15 deg masked (f -> 0, geostrophy undefined).
"""
import json
import urllib.request

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
T_IDX = 91246
OMEGA, R_EARTH, RHO0 = 7.2921e-5, 6.371e6, 1.225

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=180).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


lat = fetch("latitude", "0").astype(np.float64).ravel()
p = fetch("mean_sea_level_pressure", f"{T_IDX}.0.0")[0].astype(np.float64)
u = fetch("10m_u_component_of_wind", f"{T_IDX}.0.0")[0].astype(np.float64)
v = fetch("10m_v_component_of_wind", f"{T_IDX}.0.0")[0].astype(np.float64)
print(f"fetched: lat[{lat[0]:.1f}..{lat[-1]:.1f}], p/u/v {p.shape}")

phi = np.deg2rad(lat)
y = R_EARTH * phi                              # meridional coordinate (m)
dlam = 2 * np.pi / p.shape[1]
dx = R_EARTH * np.cos(phi) * dlam              # zonal spacing per row (m)
f_cor = 2 * OMEGA * np.sin(phi)


def ddx(a):
    """Periodic central zonal derivative with per-row cos(lat) spacing."""
    return (np.roll(a, -1, 1) - np.roll(a, 1, 1)) / (2 * dx[:, None])


def ddy(a):
    return np.gradient(a, y, axis=0)


def physics(pf, uf, vf):
    zeta = ddx(vf) - ddy(uf)
    p_anom = pf - pf.mean(axis=1, keepdims=True)
    ug = -(1 / (RHO0 * f_cor[:, None])) * ddy(pf)
    vg = (1 / (RHO0 * f_cor[:, None])) * ddx(pf)
    return zeta, p_anom, ug, vg


def quant_u8(a, q=(0.4, 99.6)):
    """The shipped palette scheme: linear 256 buckets over the percentile
    window (mirrors helix quantize.rs), decoded at bucket centers."""
    lo, hi = np.percentile(a, q)
    idx = np.clip(np.floor((a - lo) / (hi - lo) * 256), 0, 255)
    return lo + (idx + 0.5) / 256 * (hi - lo)


def popfrac(mask_num, mask_den):
    d = int(mask_den.sum())
    return (int((mask_num & mask_den).sum()) / d if d else float("nan")), d


def board_eval(zeta, p_anom, tag):
    """The chess move: popcounts over bitboard intersections."""
    band = np.abs(lat)[:, None] >= 15.0
    nh = (lat[:, None] > 0) & band & np.ones_like(p_anom, bool)
    sh = (lat[:, None] < 0) & band & np.ones_like(p_anom, bool)
    sig = p_anom[band[:, 0], :].std()
    lows, highs, zpos = p_anom < -sig, p_anom > sig, zeta > 0
    r = {}
    r["E1_nh_lows_ccw"], r["n_nh_lows"] = popfrac(zpos, lows & nh)
    r["E2_sh_lows_ccw"], r["n_sh_lows"] = popfrac(zpos, lows & sh)
    r["E3_nh_highs_ccw"], r["n_nh_highs"] = popfrac(zpos, highs & nh)
    r["E3_sh_highs_ccw"], r["n_sh_highs"] = popfrac(zpos, highs & sh)
    print(f"  [{tag}] lows:  NH ccw-frac={r['E1_nh_lows_ccw']:.3f} "
          f"(n={r['n_nh_lows']})   SH ccw-frac={r['E2_sh_lows_ccw']:.3f} "
          f"(n={r['n_sh_lows']})")
    print(f"  [{tag}] highs: NH ccw-frac={r['E3_nh_highs_ccw']:.3f}          "
          f"SH ccw-frac={r['E3_sh_highs_ccw']:.3f}")
    return r


def geo_corr(ug, vg, tag):
    r = {}
    for name, lo_b, hi_b in [("nh", 20, 70), ("sh", -70, -20)]:
        m = (lat[:, None] >= lo_b) & (lat[:, None] <= hi_b) \
            & np.ones_like(u, bool)
        cu = np.corrcoef(ug[m], u[m])[0, 1]
        cv = np.corrcoef(vg[m], v[m])[0, 1]
        r[f"{name}_corr_u"], r[f"{name}_corr_v"] = float(cu), float(cv)
        print(f"  [{tag}] geostrophic {name.upper()}: corr(u_g,u)={cu:.3f}  "
              f"corr(v_g,v)={cv:.3f}")
    return r


out = {"store": B, "time_index": T_IDX, "preregistered":
       "E1>0.5, E2<0.5 (inversion = control), E3 mirrored, E4 |raw-u8|<=0.05, "
       "E5 corr>0.5 all bands, E6 rise-peak-decay"}

print("\n== RAW arm ==")
zeta, p_anom, ug, vg = physics(p, u, v)
out["raw"] = {**board_eval(zeta, p_anom, "raw"), **geo_corr(ug, vg, "raw")}

print("\n== u8-PALETTE arm (p, u, v each quantized to 256 buckets) ==")
p8, u8, v8 = quant_u8(p), quant_u8(u), quant_u8(v)
zeta8, p_anom8, ug8, vg8 = physics(p8, u8, v8)
out["palette_u8"] = {**board_eval(zeta8, p_anom8, "u8"),
                     **geo_corr(ug8, vg8, "u8")}

# E4: substrate fidelity of the popcount verdicts
keys = ["E1_nh_lows_ccw", "E2_sh_lows_ccw", "E3_nh_highs_ccw",
        "E3_sh_highs_ccw"]
devs = {k: abs(out["raw"][k] - out["palette_u8"][k]) for k in keys}
out["E4_max_popfrac_dev"] = max(devs.values())
print(f"\n  E4 max |raw - u8| popcount-fraction deviation: "
      f"{out['E4_max_popfrac_dev']:.4f}")

# E6: Rankine profile around the deepest NH low
print("\n== E6 Rankine ('Gluecksrad') around the deepest NH low ==")
nh_rows = lat > 15
pa_nh = np.where(nh_rows[:, None], p_anom, np.inf)
ci, cj = np.unravel_index(np.argmin(pa_nh), pa_nh.shape)
print(f"  center: lat={lat[ci]:.2f} lon={cj * 0.25:.2f}  "
      f"p'={p_anom[ci, cj]:.0f} Pa")
lon = np.arange(p.shape[1]) * 0.25
dlon = np.deg2rad((lon[None, :] - lon[cj] + 180) % 360 - 180)
dphi = phi[:, None] - phi[ci]
dx_m = R_EARTH * np.cos(phi[ci]) * dlon * np.ones_like(p)
dy_m = R_EARTH * dphi * np.ones_like(p)
r_km = np.hypot(dx_m, dy_m) / 1e3
alpha = np.arctan2(dy_m, dx_m)
v_t = -u * np.sin(alpha) + v * np.cos(alpha)     # >0 = counterclockwise
prof = []
for r0 in range(0, 1500, 150):
    ring = (r_km >= r0) & (r_km < r0 + 150)
    prof.append({"r_mid_km": r0 + 75, "vt_mean": float(v_t[ring].mean()),
                 "n": int(ring.sum())})
    print(f"  r={r0 + 75:>5} km  v_t={prof[-1]['vt_mean']:+7.2f} m/s  "
          f"(n={prof[-1]['n']})")
vts = [q["vt_mean"] for q in prof]
pk = int(np.argmax(vts))
out["E6_rankine"] = {"center_lat": float(lat[ci]), "center_lon": float(lon[cj]),
                     "profile": prof, "peak_ring": pk,
                     "cyclonic_at_peak": vts[pk] > 0,
                     "rises_then_decays": 0 < pk < len(vts) - 1
                     and vts[-1] < vts[pk]}
print(f"  peak at ring {pk} ({prof[pk]['r_mid_km']} km): torque zone inside, "
      f"momentum zone outside -> rises_then_decays="
      f"{out['E6_rankine']['rises_then_decays']}")

json.dump(out, open("voxel_chess_probe.json", "w"), indent=2)
print("\nwrote voxel_chess_probe.json")
