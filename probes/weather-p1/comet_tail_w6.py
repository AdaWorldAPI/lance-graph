"""W6 -- the dipole deconvolution: is D = c_geo*neighbor + c_bow*bow-wave?

Per weather-w-probes-v1.md §3 (the report §10.2 vector-sum model). This is a
MECHANISTIC test on CT-F14's 19 stored storms, NOT a verdict -- a fresh-
sample verdict is CT-F17, gated on this result plus an independent
adversarial spec audit.

Every geometry/statistics primitive below is copied VERBATIM from the
existing arc scripts, named at each site, so this probe carries no new
scoring convention: `wrap_deg`/`err_deg`/`geom_ll`/`disk_mean_uv` from
comet_tail_f16.py; `spine()` -- the CONSTRAINED 2-parameter dipole fit
(ring-profile means + a single global lstsq on [r*cos(theta), r*sin(theta)])
-- from l4_rail_probe.py (the brief names this file as comet_tail_f16.py;
verified against the tree it actually lives in l4_rail_probe.py, ported from
there instead); `circular()` from the report §10.1 statistics standard.

UNITS. c_geo and c_bow are dimensionless least-squares coefficients that
absorb P_geo's [Pa/km] and P_bow's [Pa] units into themselves -- their SIGN
is what B2 tests, not their magnitude, and no unit conversion is performed
or needed.
"""
import datetime
import json
import pathlib
import urllib.request

import numcodecs
import numpy as np

SEED = 20260812
B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
R_E, R_DISK, RING = 6371.0, 1200.0, 100.0
ANNULUS_LO, ANNULUS_HI = 600.0, 2500.0
RHO_AIR = 1.2  # kg/m^3, sea-level air density -- the bow-wave dynamic-pressure constant

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]

EPOCH = datetime.datetime(1959, 1, 1)


def t_index(dt):
    """WB2 time index: 6-hourly steps since 1959-01-01. Anchor guard, run once
    at import time so a broken store fails loudly before any storm fetch."""
    return int(round((dt - EPOCH).total_seconds() / 3600 / 6))


assert t_index(datetime.datetime(2021, 6, 15, 12)) == 91246


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=900).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


def wrap_deg(d):
    """Wrap degrees into [-180, 180) -- verbatim from comet_tail_f16.py."""
    return (d + 180.0) % 360.0 - 180.0


def err_deg(low_pole_rad, motion_rad):
    """Signed alignment error vs the left-of-motion prediction. Verbatim from
    comet_tail_f16.py's err_deg -- kept only for the algebraic-inversion step
    below, not used to re-score anything here."""
    return float(wrap_deg(np.rad2deg(low_pole_rad - (motion_rad + np.pi / 2))))


lat = fetch("latitude", "0").astype(np.float64).ravel()
levels = fetch("level", "0").astype(int).ravel()
NY, NX = lat.size, 1440
phi = np.deg2rad(lat)
lon_deg = np.arange(NX) * 0.25
LEV_IDX = {int(v): i for i, v in enumerate(levels)}
T_MAX = meta["mean_sea_level_pressure/.zarray"]["shape"][0] - 1


def geom_ll(latc, lonc):
    """dx, dy, r (km), azimuth (rad CCW from east) about a continuous centre.
    Verbatim from comet_tail_f16.py -- the SAME basis spine()'s coef and
    disk_mean_uv's (u,v) both live in, so D/P_geo/P_bow are directly
    combinable without a basis-change step."""
    phic = np.deg2rad(latc)
    dlon = np.deg2rad((lon_deg[None, :] - lonc + 180) % 360 - 180)
    dx = R_E * np.cos(phic) * dlon * np.ones((NY, 1))
    dy = R_E * (phi[:, None] - phic) * np.ones((1, NX))
    return dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def spine(la, lo, p0):
    """The f64 constrained spine: ring-profile means + the CONSTRAINED
    2-parameter dipole (a single global lstsq on [r*cos(theta), r*sin(theta)]
    against the ring-demeaned residual). Ported verbatim from
    l4_rail_probe.py:109-122 (the brief names comet_tail_f16.py; the actual
    function lives in l4_rail_probe.py -- verified against the tree, ported
    from the real location rather than guessed). Returns D = coef = (a1, b1),
    the dipole vector this whole probe deconvolves."""
    _, _, r, th = geom_ll(la, lo)
    disk = r <= R_DISK
    v, rr, tt = p0[disk], r[disk], th[disk]
    nb = int(R_DISK / RING)
    rings = np.clip((rr / RING).astype(int), 0, nb - 1)
    prof = np.array([v[rings == b].mean() if (rings == b).any() else 0.0
                     for b in range(nb)])
    resid = v - prof[rings]
    X = np.column_stack([rr * np.cos(tt), rr * np.sin(tt)])
    coef, *_ = np.linalg.lstsq(X, resid, rcond=None)
    return coef


def disk_mean_uv(u3, v3, latc, lonc, lev_list):
    """Disk-mean (u, v) over the 1200 km disk, averaged across `lev_list`.
    Verbatim from comet_tail_f16.py."""
    _, _, r, _ = geom_ll(latc, lonc)
    disk = r <= R_DISK
    us, vs = [], []
    for lev in lev_list:
        li = LEV_IDX[lev]
        us.append(u3[li][disk].mean())
        vs.append(v3[li][disk].mean())
    return float(np.mean(us)), float(np.mean(vs))


def neighbor_predictor(p0, latc, lonc):
    """Strongest POSITIVE zonal-anomaly cell in the 600-2500 km annulus,
    lat 20-80N, about the storm centre -- the background-high the report's
    §10.2 vector-sum model treats as a far-field neighbour. Returns
    (A_H [Pa], d_H [km], theta_H [rad]); None if the annulus admits no
    candidate (should not occur inside the storm's climatological band, but
    checked rather than assumed -- see the NO-VERDICT path in the main loop)."""
    fa = p0 - p0.mean(axis=1, keepdims=True)
    _, _, r, th = geom_ll(latc, lonc)
    lat_ok = (lat >= 20) & (lat <= 80)
    mask = (r >= ANNULUS_LO) & (r <= ANNULUS_HI) & lat_ok[:, None]
    cand = np.where(mask, fa, -np.inf)
    if not np.isfinite(cand).any() or np.nanmax(cand) <= 0:
        return None
    i, j = np.unravel_index(np.argmax(cand), cand.shape)
    return float(fa[i, j]), float(r[i, j]), float(th[i, j])


def circular(errs_deg, n):
    """Resultant length R_bar, mean direction mu (deg), Rayleigh p
    (Zar/Mardia small-n correction) -- report §10.1's binding statistics
    standard, verbatim."""
    th = np.deg2rad(np.asarray(errs_deg))
    c, s = np.cos(th).mean(), np.sin(th).mean()
    r = float(np.hypot(c, s))
    mu = float(np.rad2deg(np.arctan2(s, c)))
    z = n * r * r
    p = float(np.exp(-z) * (1 + (2 * z - z * z) / (4 * n)
                            - (24 * z - 132 * z**2 + 76 * z**3 - 9 * z**4)
                            / (288 * n * n)))
    return r, mu, max(min(p, 1.0), 0.0)


def r2_vec(D, Dhat):
    """Vector R^2: 1 - sum|D_i-Dhat_i|^2 / sum|D_i-mean(D)|^2, summed over
    BOTH x/y components and all storms -- the identifiability metric B0/B1
    are scored on."""
    D, Dhat = np.asarray(D), np.asarray(Dhat)
    sse = float(np.sum((D - Dhat) ** 2))
    sst = float(np.sum((D - D.mean(axis=0)) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def fit_joint(D, Pg, Pb):
    """Solve D = c_geo*Pg + c_bow*Pb by lstsq over the 2N-stacked scalar
    equations (38 for 19 storms); returns (c_geo, c_bow, Dhat, R2)."""
    D, Pg, Pb = np.asarray(D), np.asarray(Pg), np.asarray(Pb)
    y = D.ravel()
    X = np.column_stack([Pg.ravel(), Pb.ravel()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    Dhat = (coef[0] * Pg + coef[1] * Pb)
    return float(coef[0]), float(coef[1]), Dhat, r2_vec(D, Dhat)


def fit_single(D, P):
    """Solve D = c*P by lstsq; returns (c, Dhat, R2) for the single-predictor
    comparison B1 needs."""
    D, P = np.asarray(D), np.asarray(P)
    y = D.ravel()
    X = P.ravel()[:, None]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    Dhat = coef[0] * P
    return float(coef[0]), Dhat, r2_vec(D, Dhat)


def bearing_deg(vec2):
    """arctan2(y,x) in degrees, the same east-CCW convention geom_ll/
    disk_mean_uv use throughout this probe."""
    return float(np.rad2deg(np.arctan2(vec2[1], vec2[0])))


# ---- CT-F14's 19 qualifying storms, read directly from comet_tail_f16.json
# (the brief's own input spec) -- reuses ALL of CT-F16's already-stored
# fields, so nothing about centre-finding, displacement filtering, or the
# steering-level scoring can move between this probe and the prior ones.
src = json.loads(
    pathlib.Path(__file__).with_name("comet_tail_f16.json").read_text())
storms = src["rows"]
assert len(storms) == 19, f"expected CT-F14/F16's 19 qualifying storms, got {len(storms)}"

out_dir = pathlib.Path(__file__).parent
partial_path = out_dir.with_name("weather-p1") / "comet_tail_w6.partial.jsonl"
tag_path = out_dir / "exec-runs" / "comet_tail_w6.txt"
tag_path.parent.mkdir(exist_ok=True)


def load_completed():
    """Resume-skip: read the partial checkpoint file if it exists, return
    {t0: row} for every already-fetched storm so a stranded run resumes
    instead of re-fetching."""
    done = {}
    if partial_path.exists():
        for line in partial_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["t0"]] = row
    return done


def run():
    """Per-storm fetch + spine/neighbor/bow computation (checkpointed), then
    the global 38-equation joint fit and all four pre-registered bars,
    B0 (controls, reported FIRST) through B4."""
    with open(tag_path, "a") as tf:
        tf.write(f"START seed={SEED} n_storms={len(storms)}\n")

    done = load_completed()
    with open(partial_path, "a") as pf, open(tag_path, "a") as tf:
        for i, s in enumerate(storms):
            t0 = s["t0"]
            if t0 in done:
                continue
            if t0 > T_MAX:
                tf.write(f"SKIP t0={t0} beyond store coverage (T_MAX={T_MAX})\n")
                continue
            la, lo = s["center_lat"], s["center_lon"]
            p0 = fetch("mean_sea_level_pressure", f"{t0}.0.0")[0].astype(np.float64)
            u3 = fetch("u_component_of_wind", f"{t0}.0.0.0")[0].astype(np.float64)
            v3 = fetch("v_component_of_wind", f"{t0}.0.0.0")[0].astype(np.float64)

            D = spine(la, lo, p0)

            # Step 2: motion bearing recovered by ALGEBRA (exact inversion of
            # err_deg = wrap(lp - (mth+pi/2))), no tracking.
            mth_deg = wrap_deg(np.rad2deg(s["low_pole_rad"]) - 90.0
                                - s["err_surface_deg"])
            mth_rad = np.deg2rad(mth_deg)
            v_storm_ms = s["displacement_km"] * 1000.0 / (6 * 3600.0)
            v_storm = v_storm_ms * np.array([np.cos(mth_rad), np.sin(mth_rad)])

            # Step 3: v_rel = v_storm - v_env850; bow predictor.
            u850, v850 = disk_mean_uv(u3, v3, la, lo, (850,))
            v_env850 = np.array([u850, v850])
            v_rel = v_storm - v_env850
            speed_rel = float(np.hypot(*v_rel))
            bear_rel = float(np.arctan2(v_rel[1], v_rel[0]))
            P_bow = (0.5 * RHO_AIR * speed_rel ** 2) * np.array(
                [np.cos(bear_rel + np.pi), np.sin(bear_rel + np.pi)])

            # Step 4: neighbor predictor.
            nb = neighbor_predictor(p0, la, lo)
            if nb is None:
                tf.write(f"NO-VERDICT t0={t0}: no positive annulus anomaly\n")
                continue
            A_H, d_H, theta_H = nb
            P_geo = (A_H / d_H) * np.array(
                [np.cos(theta_H + np.pi), np.sin(theta_H + np.pi)])

            row = {"t0": t0, "date": s["date"], "D": D.tolist(),
                   "P_geo": P_geo.tolist(), "P_bow": P_bow.tolist(),
                   "v_storm_ms": v_storm_ms, "v_rel_ms": speed_rel,
                   "A_H_Pa": A_H, "d_H_km": d_H, "theta_H_rad": theta_H}
            pf.write(json.dumps(row) + "\n")
            pf.flush()
            done[t0] = row
            if (i + 1) % 5 == 0 or (i + 1) == len(storms):
                tf.write(f"progress {i + 1}/{len(storms)} t0={t0}\n")
                tf.flush()

    rows = [done[s["t0"]] for s in storms if s["t0"] in done]
    n = len(rows)
    D = np.array([r["D"] for r in rows])
    Pg = np.array([r["P_geo"] for r in rows])
    Pb = np.array([r["P_bow"] for r in rows])
    vstorm = np.array([r["v_storm_ms"] for r in rows])

    rng = np.random.default_rng(SEED)

    # B0 -- controls FIRST, reported before any real-model number.
    Pb_perm = Pb[np.array([(i + 7) % n for i in range(n)])]
    _, _, _, r2_perm = fit_joint(D, Pg, Pb_perm)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])  # +90deg rotation matrix
    Pb_rot = Pb @ rot.T
    _, _, _, r2_rot = fit_joint(D, Pg, Pb_rot)

    c_geo_single, Dhat_geo, r2_geo = fit_single(D, Pg)
    c_bow_single, Dhat_bow, r2_bow = fit_single(D, Pb)
    best_single = max(r2_geo, r2_bow)

    b0_pass = (r2_perm <= r2_geo + 0.03) and (r2_rot <= r2_geo + 0.03)

    c_geo, c_bow, Dhat_joint, r2_joint = fit_joint(D, Pg, Pb)

    b1_pass = r2_joint >= best_single + 0.10
    b2_pass = (c_bow > 0) and (c_geo > 0)

    resid_bear = np.array([wrap_deg(bearing_deg(D[i]) - bearing_deg(Dhat_joint[i]))
                            for i in range(n)])
    rbar_all, mu_all, p_all = circular(resid_bear, n)
    stranded = vstorm < 8.0
    moving = ~stranded
    rbar_s, mu_s, p_s = (circular(resid_bear[stranded], int(stranded.sum()))
                         if stranded.sum() >= 2 else (None, None, None))
    rbar_m, mu_m, p_m = (circular(resid_bear[moving], int(moving.sum()))
                         if moving.sum() >= 2 else (None, None, None))

    # per_storm carries the RAW predictors too, not only derived bearings --
    # a magnitude/units audit (are P_geo/P_bow physically sane, is there a
    # scale mismatch driving a near-zero coefficient) needs the actual
    # vectors committed, not just the fit's summary numbers. (Fixed after
    # the first run shipped only bearings; re-run rather than leaving the
    # audit gap, since a rerun costs the same ~3 min the brief itself
    # estimates.)
    per_storm = [{"t0": rows[i]["t0"], "bearing_D_deg": bearing_deg(D[i]),
                  "bearing_Dhat_deg": bearing_deg(Dhat_joint[i]),
                  "resid_deg": float(resid_bear[i]),
                  "v_storm_ms": float(vstorm[i]),
                  "D": D[i].tolist(), "P_geo": Pg[i].tolist(),
                  "P_bow": Pb[i].tolist(), "v_rel_ms": rows[i]["v_rel_ms"],
                  "A_H_Pa": rows[i]["A_H_Pa"], "d_H_km": rows[i]["d_H_km"],
                  "theta_H_rad": rows[i]["theta_H_rad"]} for i in range(n)]

    out = {
        "n": n, "seed": SEED,
        "B0_controls": {"r2_single_geo": r2_geo, "r2_permuted": r2_perm,
                        "r2_rotated90": r2_rot, "bar": "<= r2_single_geo + 0.03",
                        "verdict": "PASS" if b0_pass else "VOID"},
        "single_models": {"geo": {"c": c_geo_single, "R2": r2_geo},
                           "bow": {"c": c_bow_single, "R2": r2_bow}},
        "joint_model": {"c_geo": c_geo, "c_bow": c_bow, "R2": r2_joint},
        "B1_identifiability": {"joint_R2": r2_joint, "best_single_R2": best_single,
                               "margin": r2_joint - best_single,
                               "bar": ">= best_single_R2 + 0.10",
                               "verdict": ("PASS" if b1_pass else "FAIL")
                               if b0_pass else "VOID (B0 failed)"},
        "B2_sign": {"c_geo": c_geo, "c_bow": c_bow,
                   "verdict": ("PASS" if b2_pass else "FAIL")
                   if b0_pass else "VOID (B0 failed)"},
        "B3_residual_resultant": {
            "overall": {"R_bar": rbar_all, "mu_deg": mu_all, "rayleigh_p": p_all},
            "stranded_lt8ms": {"n": int(stranded.sum()), "R_bar": rbar_s,
                               "mu_deg": mu_s, "rayleigh_p": p_s},
            "moving_ge8ms": {"n": int(moving.sum()), "R_bar": rbar_m,
                             "mu_deg": mu_m, "rayleigh_p": p_m}},
        "B4_per_storm": per_storm,
    }
    with open(out_dir / "comet_tail_w6.json", "w") as fh:
        json.dump(out, fh, indent=2)
    with open(tag_path, "a") as tf:
        tf.write(f"DONE B0={out['B0_controls']['verdict']} "
                 f"B1={out['B1_identifiability']['verdict']} "
                 f"B2={out['B2_sign']['verdict']}\n")
    if partial_path.exists():
        partial_path.unlink()
    return out


if __name__ == "__main__":
    res = run()
    print(f"n={res['n']}")
    print("B0 controls:", res["B0_controls"])
    print("single models:", res["single_models"])
    print("joint model:", res["joint_model"])
    print("B1:", res["B1_identifiability"])
    print("B2:", res["B2_sign"])
    print("B3 overall:", res["B3_residual_resultant"]["overall"])
    print("B3 stranded (<8 m/s):", res["B3_residual_resultant"]["stranded_lt8ms"])
    print("B3 moving (>=8 m/s):", res["B3_residual_resultant"]["moving_ge8ms"])
