"""W2s-a -- golden two-lattice pairing on REAL lat/lon geometry.

Per weather-w-probes-v1.md SS2 (Sonnet, zero fetch). Tests whether the
collision-node construction's assumed property -- two sunflower lattices
pair generically, no ties, even pair distances -- survives the real cos-lat
metric, not just an idealized disk. The #921 lesson: disk properties do NOT
automatically transfer to a projected lat/lon geometry.

INDEX FLOOR: N = F(17)^2 = 2,550,409 per lattice for the headline run (the
first draft's N=2048 was six orders of magnitude sub-floor per SS0's rule).

TIE DEFINITION: per-source near-tie (d1/d2 ratio for each H-point's own two
nearest T-candidates), NOT a global duplicate-distance count -- the first
draft's global count is blind to the actual pairing-ambiguity question and,
at million-point sizes, prone to unrelated-pair float collisions regardless
of mechanism. See SS0's G1/G4 correction note.
"""
import json
import pathlib
from math import gcd

import numpy as np
from scipy.spatial import cKDTree

SEED = 20260812
R_E = 6371.0
F17_SQ = 1597 * 1597  # = 2_550_409, the headline N


def geom_ll(lat_c, lon_c, lat_h, lon_h):
    """Project a lat/lon offset from a center to local km via the flat cos-lat
    metric this arc uses throughout (dx scaled by cos(lat_c), dy not)."""
    dlat = np.deg2rad(lat_h - lat_c)
    dlon = np.deg2rad(lon_h - lon_c)
    dx = R_E * np.cos(np.deg2rad(lat_c)) * dlon
    dy = R_E * dlat
    return dx, dy


def vogel_lattice(n, radius_km, center_lat, center_lon):
    """N-point Vogel spiral (r = c*sqrt(k), golden angle) covering a disk of
    the given radius (km), centered at (center_lat, center_lon). Returns
    (x_km, y_km, lat, lon) arrays, all length n. c chosen so max radius
    equals the requested radius_km."""
    phi = (1 + 5 ** 0.5) / 2
    golden_frac = 2 - phi
    k = np.arange(n)
    r = radius_km * np.sqrt((k + 0.5) / n)
    theta = k * 2 * np.pi * golden_frac
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    lat = center_lat + np.rad2deg(y / R_E)
    lon = center_lon + np.rad2deg(x / (R_E * np.cos(np.deg2rad(center_lat))))
    return x, y, lat, lon


def grid_lattice(n, radius_km, center_lat, center_lon):
    """Axis-aligned square grid control of matching point density, clipped
    to the same disk -- identical pairing procedure applies to it below."""
    side = int(np.ceil(np.sqrt(n * 4 / np.pi)))
    g = (np.arange(side) + 0.5) / side * 2 * radius_km - radius_km
    gx, gy = np.meshgrid(g, g)
    gx, gy = gx.ravel(), gy.ravel()
    rr = np.hypot(gx, gy)
    mask = rr <= radius_km
    x, y = gx[mask], gy[mask]
    lat = center_lat + np.rad2deg(y / R_E)
    lon = center_lon + np.rad2deg(x / (R_E * np.cos(np.deg2rad(center_lat))))
    return x, y, lat, lon


def project_to_center(x_src, y_src, lat_src_c, lon_src_c, lat_dst_c, lon_dst_c):
    """Re-project a lattice's local (x,y) km coords, built around its own
    center, into the OTHER center's local km frame -- needed because the
    cos-lat metric is center-dependent (dx scale differs at each center)."""
    lat = lat_src_c + np.rad2deg(y_src / R_E)
    lon = lon_src_c + np.rad2deg(x_src / (R_E * np.cos(np.deg2rad(lat_src_c))))
    dx, dy = geom_ll(lat_dst_c, lon_dst_c, lat, lon)
    return dx, dy


def near_tie_count(src_x, src_y, dst_x, dst_y, band_km):
    """G1/G4: for each source point within band_km of the dest center,
    find its 1st/2nd nearest dest-lattice neighbours and count near-ties
    (d1/d2 > 1-1e-6). Returns (n_in_band, n_near_ties, cv_of_d1)."""
    tree = cKDTree(np.column_stack([dst_x, dst_y]))
    r = np.hypot(src_x, src_y)
    in_band = r <= band_km
    if in_band.sum() == 0:
        return 0, 0, float("nan")
    pts = np.column_stack([src_x[in_band], src_y[in_band]])
    d, _ = tree.query(pts, k=2)
    d1, d2 = d[:, 0], d[:, 1]
    ratio = np.where(d2 > 0, d1 / d2, 0.0)
    near_ties = int((ratio > 1 - 1e-6).sum())
    cv = float(d1.std() / d1.mean()) if d1.mean() > 0 else float("nan")
    return int(in_band.sum()), near_ties, cv


def chi2_midpoint_uniform(src_x, src_y, dst_x, dst_y, band_km, bins=10):
    """G3 (descriptive): chi-square of pair-midpoint radial density against
    uniform-in-area expectation across the corridor band."""
    tree = cKDTree(np.column_stack([dst_x, dst_y]))
    r = np.hypot(src_x, src_y)
    in_band = r <= band_km
    if in_band.sum() < bins * 5:
        return None
    pts = np.column_stack([src_x[in_band], src_y[in_band]])
    d, idx = tree.query(pts, k=1)
    mid_x = (pts[:, 0] + dst_x[idx]) / 2
    mid_y = (pts[:, 1] + dst_y[idx]) / 2
    mid_r = np.hypot(mid_x, mid_y)
    edges = np.linspace(0, band_km, bins + 1)
    obs, _ = np.histogram(mid_r, bins=edges)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    exp = obs.sum() * area / area.sum()
    chi2 = float(((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    return chi2


def run_pair(n_h, n_t, band_km=900.0, radius_km=1500.0):
    """One full G1/G2/G3 run at a given N, both golden and grid, both
    directions (H->T and T->H symmetrized by just running H->T since the
    construction is deliberately symmetric in this test)."""
    lat_h, lon_h = 55.0, 340.0
    lat_t, lon_t = 55.0, 340.0 + np.rad2deg(1400.0 / (R_E * np.cos(np.deg2rad(55.0))))

    gx_h, gy_h, _, _ = vogel_lattice(n_h, radius_km, lat_h, lon_h)
    gx_t, gy_t, _, _ = vogel_lattice(n_t, radius_km, lat_t, lon_t)
    qx_h, qy_h, _, _ = grid_lattice(n_h, radius_km, lat_h, lon_h)
    qx_t, qy_t, _, _ = grid_lattice(n_t, radius_km, lat_t, lon_t)

    # project H's points into T's local frame for the H->T pairing
    gdx, gdy = project_to_center(gx_h, gy_h, lat_h, lon_h, lat_t, lon_t)
    qdx, qdy = project_to_center(qx_h, qy_h, lat_h, lon_h, lat_t, lon_t)

    n_band_g, ties_g, cv_g = near_tie_count(gdx, gdy, gx_t, gy_t, band_km)
    n_band_q, ties_q, cv_q = near_tie_count(qdx, qdy, qx_t, qy_t, band_km)
    chi2_g = chi2_midpoint_uniform(gdx, gdy, gx_t, gy_t, band_km)
    chi2_q = chi2_midpoint_uniform(qdx, qdy, qx_t, qy_t, band_km)

    return {
        "n_pairs_golden": n_band_g, "n_pairs_grid": n_band_q,
        "ties_golden": ties_g, "ties_grid": ties_q,
        "cv_golden": cv_g, "cv_grid": cv_q,
        "chi2_golden": chi2_g, "chi2_grid": chi2_q,
    }


def run():
    """Headline run at N=F(17)^2, then the G4 index-floor sweep at
    N=F(n)^2 for n in {8,10,12,14,17,19}. Checkpoints each stage."""
    out_dir = pathlib.Path(__file__).parent
    partial = out_dir / "sunflower_pairing_probe.partial.jsonl"

    with open(partial, "w") as pf:
        headline = run_pair(F17_SQ, F17_SQ)
        pf.write(json.dumps({"stage": "headline", "N": F17_SQ, **headline}) + "\n")
        pf.flush()

        fibs = {8: 21, 10: 55, 12: 144, 14: 377, 17: 1597, 19: 4181}
        sweep = []
        for n_idx, fn in fibs.items():
            N = fn * fn
            r = run_pair(N, N)
            row = {"n": n_idx, "N": N, "near_ties_golden": r["ties_golden"],
                   "near_ties_grid": r["ties_grid"],
                   "cv_golden": r["cv_golden"], "cv_grid": r["cv_grid"]}
            sweep.append(row)
            pf.write(json.dumps({"stage": "sweep", **row}) + "\n")
            pf.flush()

    verdict_g1 = "VOID" if headline["ties_grid"] == 0 else (
        "PASS" if headline["ties_golden"] == 0 else "FAIL")
    verdict_g2 = "PASS" if headline["cv_golden"] < headline["cv_grid"] else "FAIL"
    verdict_g4 = "PASS" if all(r["near_ties_golden"] == 0 for r in sweep) else "FAIL"

    out = {
        "N": F17_SQ, **headline,
        "sweep": sweep,
        "verdicts": {"G1": verdict_g1, "G2": verdict_g2, "G4": verdict_g4},
    }
    with open(out_dir / "sunflower_pairing_probe.json", "w") as fh:
        json.dump(out, fh, indent=2)
    partial.unlink()  # clean up after a successful completion, per repo convention
    return out


if __name__ == "__main__":
    r = run()
    print(f"N={r['N']}  pairs(golden)={r['n_pairs_golden']} pairs(grid)={r['n_pairs_grid']}")
    print(f"ties: golden={r['ties_golden']}  grid={r['ties_grid']}")
    print(f"CV:   golden={r['cv_golden']:.4f}  grid={r['cv_grid']:.4f}")
    print(f"chi2: golden={r['chi2_golden']}  grid={r['chi2_grid']}")
    print("verdicts:", r["verdicts"])
    print("\nG4 sweep:")
    for row in r["sweep"]:
        print(f"  n={row['n']:2d} N={row['N']:9d} "
              f"ties(g/q)={row['near_ties_golden']}/{row['near_ties_grid']} "
              f"CV(g/q)={row['cv_golden']:.4f}/{row['cv_grid']:.4f}")
