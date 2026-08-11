"""EXPLORATORY — 'die goldene Spirale encodiert das ganze Tiefdruckgebiet'
(operator, 2026-08-11, sunflower-ripples frame). NOT an EV; bars are mine,
unaudited — measurements, not findings, per this session's 0/11 lesson.

The claim, made falsifiable: a cyclone is near-axisymmetric about its center,
so sampling it along the EQUAL-AREA golden spiral (r = sqrt(u)*R, theta =
n*golden_angle — literally helix::HemispherePoint::lift's lattice) turns the
2-D pressure structure into a low-entropy 1-D 'ripple' signal: the radial
profile carries the mass, consecutive spiral samples are near in radius, so
deltas are small. If true, bgz17-palette + highheelbgz spiral-ADDRESSING
(start, stride, len — values recomputed on demand) has something real to
encode. If false, the spiral is decoration.

PRE-REGISTERED:
  E1  Axisymmetry: the ring-mean radial profile explains >= 70% of pressure
      variance in the storm disk (R=1200 km).
  E2  Equal budget N=256: golden-spiral sampling reconstructs the disk with
      RMSE <= uniform-grid sampling AND <= random sampling (nearest-neighbor
      reconstruction, same N, same reconstruction rule for every arm).
  E3  Ripples: Shannon entropy of u8 first-differences ALONG THE SPIRAL is
      lower than along a raster scan of the SAME samples — the spiral
      ordering is what makes the signal low-entropy, not the quantization.
  E4  CONTROL (can-the-metric-fail): recentering the disk 750 km off-storm
      drops the axisymmetry index by >= 0.15 absolute. If it does not, the
      index does not measure centeredness and E1 is decoration.
  E5  u8 palette quantization of the spiral samples adds reconstruction
      error of at most one bucket width on top of E2's sampling error.

Data: WB2 ERA5 6h msl, t=91246 (2021-06-15 12Z) — the SAME timestep and the
SAME storm (55.75N, 334.5E) as voxel_chess_probe.py E6 and ndarray's
examples/geostrophic_stencil.rs. cos(lat) spacing throughout.
"""
import json
import urllib.request

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
T_IDX = 91246
R_EARTH_KM = 6371.0
R_DISK_KM = 1200.0
GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))

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
phi = np.deg2rad(lat)

# Storm center: deepest NH low in zonal-anomaly space (same rule as E6).
p_anom = p - p.mean(axis=1, keepdims=True)
nh = lat > 15
ci, cj = np.unravel_index(
    np.argmin(np.where(nh[:, None], p_anom, np.inf)), p.shape)
print(f"storm center: lat={lat[ci]:.2f} lon={cj * 0.25:.2f} "
      f"p'={p_anom[ci, cj]:.0f} Pa")


def km_grid(ci_, cj_):
    """(dx_km, dy_km, r_km) of every gridpoint relative to center (ci_, cj_)."""
    lon = np.arange(p.shape[1]) * 0.25
    dlon = np.deg2rad((lon[None, :] - lon[cj_] + 180) % 360 - 180)
    dphi = phi[:, None] - phi[ci_]
    dx = R_EARTH_KM * np.cos(phi[ci_]) * dlon * np.ones_like(p)
    dy = R_EARTH_KM * dphi * np.ones_like(p)
    return dx, dy, np.hypot(dx, dy)


def bilinear(field, ci_, cj_, dx_km, dy_km):
    """Sample field at km-offsets from center (ci_, cj_), bilinear, lon-wrapped."""
    dlat = np.rad2deg(dy_km / R_EARTH_KM)
    dlon = np.rad2deg(dx_km / (R_EARTH_KM * np.cos(phi[ci_])))
    ri = ci_ - dlat / 0.25          # lat rows DESCEND 90..-90
    rj = (cj_ + dlon / 0.25) % p.shape[1]
    i0 = np.clip(np.floor(ri).astype(int), 0, p.shape[0] - 2)
    j0 = np.floor(rj).astype(int) % p.shape[1]
    fi, fj = ri - i0, rj - j0
    j1 = (j0 + 1) % p.shape[1]
    return (field[i0, j0] * (1 - fi) * (1 - fj)
            + field[i0 + 1, j0] * fi * (1 - fj)
            + field[i0, j1] * (1 - fi) * fj
            + field[i0 + 1, j1] * fi * fj)


def axisym_index(ci_, cj_):
    """1 - var(residual after removing the ring-mean profile)/var, in-disk."""
    _, _, r = km_grid(ci_, cj_)
    disk = r <= R_DISK_KM
    vals, rr = p[disk], r[disk]
    rings = np.clip((rr / 50.0).astype(int), 0, 23)
    prof = np.array([vals[rings == b].mean() if (rings == b).any() else np.nan
                     for b in range(24)])
    resid = vals - prof[rings]
    return 1.0 - resid.var() / vals.var(), disk


def spiral_pts(n):
    """n sunflower/golden-angle points on the disk (equal-area: r = sqrt((k+0.5)/n)*R)."""
    k = np.arange(n)
    r = np.sqrt((k + 0.5) / n) * R_DISK_KM
    th = k * GOLDEN_ANGLE
    return r * np.cos(th), r * np.sin(th)


def grid_pts(n):
    """EXACTLY n points on a uniform grid clipped to the disk.

    E2 is an EQUAL-BUDGET comparison, so this must return n, not "about n".
    An earlier version returned every in-disk lattice point, which handed the
    grid arm 80 samples against the spiral's 64 (and 293 vs 256, 1085 vs 1024)
    — nearest-neighbour reconstruction improves with samples, so the arm being
    compared was systematically advantaged and the verdict was not a controlled
    comparison (codex P2 + coderabbit on PR #926, 2026-08-11).

    The lattice is grown until it holds at least n in-disk points, then the n
    CLOSEST to the disk centre are kept — a deterministic, spatially even
    subset with no RNG and no dependence on iteration order.
    """
    side = int(np.ceil(np.sqrt(n * 4 / np.pi)))
    while True:
        g = (np.arange(side) + 0.5) / side * 2 * R_DISK_KM - R_DISK_KM
        gx, gy = np.meshgrid(g, g)
        rr = np.hypot(gx, gy)
        m = rr <= R_DISK_KM
        if m.sum() >= n:
            break
        side += 1
    gx, gy, rr = gx[m], gy[m], rr[m]
    keep = np.argsort(rr, kind="stable")[:n]
    return gx[keep], gy[keep]


def rand_pts(n, seed=7):
    """n uniform random points on the disk, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(0, 1, n)) * R_DISK_KM
    th = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(th), r * np.sin(th)


def recon_rmse(sx, sy, svals, dxg, dyg, disk):
    """Nearest-sample reconstruction of the in-disk field (one rule, all arms)."""
    tx, ty = dxg[disk], dyg[disk]
    truth = p[disk]
    # brute-force nearest over <=1024 samples x ~7e3 targets — fine
    d2 = (tx[:, None] - sx[None, :]) ** 2 + (ty[:, None] - sy[None, :]) ** 2
    rec = svals[np.argmin(d2, axis=1)]
    return float(np.sqrt(((rec - truth) ** 2).mean()))


def delta_entropy(q_u8):
    """Shannon entropy (bits/sample) of the FIRST DIFFERENCES of a u8 sequence.

    Measures how predictable the sequence is in the given traversal order — the
    quantity E3 compares between spiral order and raster order."""
    d = np.diff(q_u8.astype(np.int16))
    _, counts = np.unique(d, return_counts=True)
    pr = counts / counts.sum()
    return float(-(pr * np.log2(pr)).sum())


out = {"store": B, "time_index": T_IDX, "R_disk_km": R_DISK_KM,
       "center": {"lat": float(lat[ci]), "lon": float(cj * 0.25),
                  "p_anom_Pa": float(p_anom[ci, cj])}}

# E1 + E4 --------------------------------------------------------------
ax_storm, disk = axisym_index(ci, cj)
off_rows = int(round((750.0 / R_EARTH_KM) * 180 / np.pi / 0.25))
ax_off, _ = axisym_index(ci + off_rows, cj)   # 750 km south of the storm
out["E1_axisym_storm"] = ax_storm
out["E4_axisym_offset750km"] = ax_off
print(f"\nE1 axisymmetry (storm-centered): {ax_storm:.3f}  (bar >= 0.70)")
print(f"E4 axisymmetry (750 km off):     {ax_off:.3f}  "
      f"(bar: drop >= 0.15 -> {'FIRES' if ax_storm - ax_off >= 0.15 else 'fails'})")

# E2 -------------------------------------------------------------------
dxg, dyg, _ = km_grid(ci, cj)
out["E2_budgets"] = {}
for n in (64, 256, 1024):
    row = {}
    for name, (sx, sy) in [("spiral", spiral_pts(n)), ("grid", grid_pts(n)),
                           ("random", rand_pts(n))]:
        vals = bilinear(p, ci, cj, sx, sy)
        row[name] = {"n": len(sx),
                     "rmse_Pa": recon_rmse(sx, sy, vals, dxg, dyg, disk)}
    out["E2_budgets"][str(n)] = row
    print(f"E2 N~{n:4}: spiral={row['spiral']['rmse_Pa']:7.1f} Pa "
          f"(n={row['spiral']['n']})  grid={row['grid']['rmse_Pa']:7.1f} "
          f"(n={row['grid']['n']})  random={row['random']['rmse_Pa']:7.1f}")

# E3 + E5 --------------------------------------------------------------
n = 256
sx, sy = spiral_pts(n)
vals = bilinear(p, ci, cj, sx, sy)
lo, hi = vals.min(), vals.max()
q = np.clip(np.floor((vals - lo) / (hi - lo) * 256), 0, 255).astype(np.uint8)
h_spiral = delta_entropy(q)
# raster order of the SAME samples: sort by (y-band, x)
order = np.lexsort((sx, np.floor(sy / 150.0)))
h_raster = delta_entropy(q[order])
h_raw = delta_entropy(np.concatenate([[q[0]], q]))  # abs values, reference
out["E3_delta_entropy_bits"] = {"spiral_order": h_spiral,
                                "raster_order": h_raster}
print(f"\nE3 delta-entropy (u8, bits/sample): spiral-order={h_spiral:.2f}  "
      f"raster-order={h_raster:.2f}  (bar: spiral < raster)")

deq = lo + (q.astype(np.float64) + 0.5) / 256 * (hi - lo)
rmse_q = recon_rmse(sx, sy, deq, dxg, dyg, disk)
bucket = (hi - lo) / 256
out["E5_u8"] = {"rmse_Pa": rmse_q, "bucket_Pa": float(bucket),
                "added_vs_f64_Pa": rmse_q - out["E2_budgets"]["256"]["spiral"]["rmse_Pa"]}
print(f"E5 u8-palette spiral recon: {rmse_q:.1f} Pa "
      f"(f64 sampling: {out['E2_budgets']['256']['spiral']['rmse_Pa']:.1f}; "
      f"bucket={bucket:.1f} Pa; added={out['E5_u8']['added_vs_f64_Pa']:.1f}, "
      f"bar <= {bucket:.1f})")

json.dump(out, open("sunflower_cyclone_probe.json", "w"), indent=2)
print("\nwrote sunflower_cyclone_probe.json")
