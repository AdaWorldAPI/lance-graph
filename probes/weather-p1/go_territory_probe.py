"""EXPLORATORY — 'overlapping golden spirals': eine Spirale pro Hoch/Tief,
Flaechen-vs-Nahkampf-Logik wie Go. (Operator, 2026-08-11.) NOT an EV; bars
mine, unaudited — measurements, not findings.

Two falsifiable halves:

A (FLAECHEN / territory): the pressure field as a SUPERPOSITION of radial
  (sunflower-samplable) profiles around MANY centers — matching pursuit with
  radial atoms. Directly tests the successor hypothesis from
  sunflower_cyclone_probe's E3 fail: the single-storm 'azimuthal residual'
  (36%) is largely the overlap of NEIGHBORING systems.
B (NAHKAMPF / contested zones): Go-style influence tessellation
  (influence_i = |A_i| / r_i^2 — scale-free, deliberately NO length knob).
  Contested cells (second/best influence > threshold) should carry the
  FRONTS: elevated |grad T2m|. Secured territory should be calm.

PRE-REGISTERED:
  A-E1  greedy K=10 multi-center radial superposition explains >= 0.55 of
        p_anom variance over the NH 25..75N band (fresh domain, fresh bar —
        NOT comparable to the single-disk 0.639).
  A-E2  control: K=10 RANDOM-center atoms explain <= HALF of A-E1's value.
  A-E3  observation only, NO bar: marginal-gain decay ratio reported and
        compared to 1/phi = 0.618 as a labelled CURIOSITY.
  B-E1  can-fire: mean |grad T| in contested cells (ratio>0.8) >= 1.4x mean
        in secured cells (ratio<0.3). Inertness: also reported at 0.7/0.9.
  B-E2  silence: secured-territory mean |grad T| <= band mean.
  B-E3  control: random-position centers, same rule -> ratio in [0.8, 1.2].

Same store/timestep as the whole arc (WB2, t=91246). Planar per-center km
approx with cos(lat_center) — consistent with the sibling probes, stated.
Hoehenprofil (geopotential stacking) deliberately NOT here — next probe.
"""
import json
import os
import urllib.request

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
T_IDX = 91246
R_E = 6371.0
RINGS_KM, R_MAX_KM = 100.0, 2000.0
MIN_SEP_KM = 400.0

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=180).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


lat = fetch("latitude", "0").astype(np.float64).ravel()
p = fetch("mean_sea_level_pressure", f"{T_IDX}.0.0")[0].astype(np.float64)
t2 = fetch("2m_temperature", f"{T_IDX}.0.0")[0].astype(np.float64)
phi = np.deg2rad(lat)
lon_deg = np.arange(p.shape[1]) * 0.25

r0, r1 = 60, 261                      # 75N .. 25N band (descending lat)
band = slice(r0, r1)
pa = p - p.mean(axis=1, keepdims=True)
pa_band = pa[band].copy()
print(f"band: lat {lat[r0]:.1f}..{lat[r1-1]:.1f}, shape {pa_band.shape}, "
      f"var {pa_band.var():.0f} Pa^2")


def dist_km(ci, cj):
    """km distance from band gridpoints to center (band-row ci, col cj)."""
    dlon = np.deg2rad((lon_deg[None, :] - lon_deg[cj] + 180) % 360 - 180)
    dphi = phi[band][:, None] - phi[r0 + ci]
    dx = R_E * np.cos(phi[r0 + ci]) * dlon
    dy = R_E * dphi
    return np.hypot(dx * np.ones_like(dy), dy * np.ones_like(dlon))


def fit_atom(res, ci, cj):
    """Ring-mean radial profile of `res` around (ci, cj), subtractable field."""
    r = dist_km(ci, cj)
    rings = (r / RINGS_KM).astype(int)
    atom = np.zeros_like(res)
    for b in range(int(R_MAX_KM / RINGS_KM)):
        m = rings == b
        if m.any():
            atom[m] = res[m].mean()
    atom[r > R_MAX_KM] = 0.0
    return atom, r


def matching_pursuit(field, k_max, centers=None):
    """Greedy K radial atoms; centers picked from residual argmax (or given)."""
    res = field.copy()
    # Fixed denominator = the CENTERED field's variance; the numerator is the
    # residual's MEAN SQUARE, not its variance. `res.var()` re-centres after
    # every atom, so a non-zero residual mean was excluded from the error and
    # explained variance came out inflated (coderabbit on PR #926).
    v0 = float(np.mean((field - field.mean()) ** 2))
    used, explained = [], []
    for k in range(k_max):
        if centers is not None:
            ci, cj = centers[k]
        else:
            cand = np.abs(res).copy()
            for (ui, uj) in used:
                cand[dist_km(ui, uj) < MIN_SEP_KM] = 0
            ci, cj = np.unravel_index(np.argmax(cand), res.shape)
        atom, _ = fit_atom(res, ci, cj)
        res -= atom
        used.append((int(ci), int(cj)))
        explained.append(1.0 - float(np.mean(res ** 2)) / v0)
    return used, explained


# ---- Part A -----------------------------------------------------------
K = 10
centers, expl = matching_pursuit(pa_band, K)
print("\nA: greedy multi-center radial superposition (variance explained):")
for k, e in enumerate(expl):
    ci, cj = centers[k]
    print(f"  K={k+1:2}  R2={e:.3f}   center lat={lat[r0+ci]:6.2f} "
          f"lon={lon_deg[cj]:6.2f}  p'={pa_band[ci, cj]:.0f} Pa "
          f"({'H' if pa_band[ci, cj] > 0 else 'T'})")
a_e1 = expl[-1]

rng = np.random.default_rng(7)
rand_centers = [(int(rng.integers(0, pa_band.shape[0])),
                 int(rng.integers(0, pa_band.shape[1]))) for _ in range(K)]
_, expl_rand = matching_pursuit(pa_band, K, centers=rand_centers)
a_e2 = expl_rand[-1]
print(f"  A-E1 K=10 matched:  R2={a_e1:.3f}  (bar >= 0.55)")
print(f"  A-E2 K=10 random:   R2={a_e2:.3f}  (bar <= {a_e1/2:.3f})")

gains = np.diff(np.concatenate([[0.0], expl]))
ratios = gains[1:] / np.maximum(gains[:-1], 1e-12)
print(f"  A-E3 marginal-gain decay ratios: "
      f"{[f'{x:.2f}' for x in ratios]}  mean={ratios.mean():.3f} "
      f"(curiosity vs 1/phi=0.618 — NO bar)")

# ---- Part B -----------------------------------------------------------
det = []
cand = np.abs(pa_band).copy()
while len(det) < 20:
    ci, cj = np.unravel_index(np.argmax(cand), cand.shape)
    if cand[ci, cj] < 400:
        break
    det.append((int(ci), int(cj), float(pa_band[ci, cj])))
    cand[dist_km(ci, cj) < 500.0] = 0
print(f"\nB: {len(det)} centers (|p'|>=400 Pa), "
      f"{sum(1 for d in det if d[2] > 0)} highs / "
      f"{sum(1 for d in det if d[2] < 0)} lows")


def contested_secured(center_list, thresh):
    inf = np.zeros((len(center_list),) + pa_band.shape)
    for i, (ci, cj, amp) in enumerate(center_list):
        r = np.maximum(dist_km(ci, cj), 50.0)
        inf[i] = np.abs(amp) / r**2
    s = np.sort(inf, axis=0)
    ratio = s[-2] / s[-1]
    return ratio > thresh, ratio < 0.3


dxm = R_E * 1e3 * np.cos(phi[band]) * np.deg2rad(0.25)
dym = R_E * 1e3 * np.deg2rad(0.25)
t2b = t2[band]
gTx = (np.roll(t2b, -1, 1) - np.roll(t2b, 1, 1)) / (2 * dxm[:, None])
gTy = np.gradient(t2b, axis=0) / -dym          # lat descends
gradT = np.hypot(gTx, gTy) * 1e5               # K per 100 km
band_mean = gradT.mean()

out_b = {}
for th in (0.7, 0.8, 0.9):
    contested, secured = contested_secured(det, th)
    ratio = gradT[contested].mean() / gradT[secured].mean()
    out_b[str(th)] = {
        "contested_frac": float(contested.mean()),
        "gradT_contested": float(gradT[contested].mean()),
        "gradT_secured": float(gradT[secured].mean()),
        "ratio": float(ratio)}
    print(f"  thresh {th}: contested {contested.mean()*100:4.1f}% of band, "
          f"|gradT| contested/secured = {gradT[contested].mean():.3f}/"
          f"{gradT[secured].mean():.3f} = {ratio:.2f}x")
b_e1 = out_b["0.8"]["ratio"]
_, secured08 = contested_secured(det, 0.8)
b_e2 = bool(gradT[secured08].mean() <= band_mean)

rand_det = [(int(rng.integers(0, pa_band.shape[0])),
             int(rng.integers(0, pa_band.shape[1])), amp)
            for (_, _, amp) in det]
c_r, s_r = contested_secured(rand_det, 0.8)
b_e3 = float(gradT[c_r].mean() / gradT[s_r].mean())
print(f"  B-E1 ratio@0.8 = {b_e1:.2f}x (bar >= 1.4)   "
      f"B-E2 secured {gradT[secured08].mean():.3f} <= band {band_mean:.3f}: "
      f"{b_e2}   B-E3 random-centers ratio = {b_e3:.2f} (bar 0.8..1.2)")

json.dump({
    "store": B, "time_index": T_IDX, "band_lat": [25, 75],
    "A": {"explained_by_k": expl, "centers": [
        {"lat": float(lat[r0 + ci]), "lon": float(lon_deg[cj])}
        for ci, cj in centers],
        "E1": a_e1, "E2_random": a_e2,
        "E3_decay_ratios": ratios.tolist(), "E3_mean": float(ratios.mean())},
    "B": {"n_centers": len(det), "by_thresh": out_b,
          "E2_secured_le_bandmean": b_e2, "E3_random_ratio": b_e3,
          "gradT_band_mean_K_per_100km": float(band_mean)},
}, open("go_territory_probe.json", "w"), indent=2)
print("\nwrote go_territory_probe.json")
