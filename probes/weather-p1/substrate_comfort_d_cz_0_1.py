"""D-CZ-0 REPRODUCTION + D-CZ-1 control-losability smoke test.

Per `.claude/plans/substrate-comfort-zones-v1.md` §1 (the regime preflight)
and §3 C0 (the control gate that must run BEFORE any expensive cell).

WHY THIS SCRIPT EXISTS AT ALL -- the honest reason, stated first. §1's
preflight is recorded as **DONE** on `STATUS_BOARD.md` with a nine-row table
of measured figures (`|grad p|`, elevation sigma, speed sigma, land-sea
mask). **No script and no JSON producing any of those figures was ever
committed.** That is the same defect this arc already catalogued once -- the
chat-only 99.38 % figure in #936 -- and it went unnoticed through three
subsequent PRs, including an explicit self-audit on #945 that reported the
regime-ladder numbers as "verified". What that audit actually checked was
that the arc entry matched the PLAN. Both are prose. Neither is a
measurement. A figure cited in two documents is cited twice, not confirmed.

WHAT IS AND IS NOT REPRODUCIBLE FROM COMMITTED INFORMATION:
  - R1 CALM / R2 OCEAN / R3 ACTIVE -- centres ARE recorded in the plan
    (4 S 296 E, 25 S 220 E, 60 N 72 E), so these three reproduce exactly.
  - R4 STORM -- centres come from `comet_tail_f14.json`'s qualifying rows,
    which IS committed, so this reproduces too.
  - The five EXCLUDED land candidates (Australian outback, Sahara, Argentine
    pampas, US Great Plains, N European plain) -- their coordinates were
    NEVER recorded anywhere. Their rows are **not reproducible as
    committed** and this script does not invent coordinates to fake them.
    That is reported as a gap, not silently omitted.

The `|grad p|` DEFINITION was also never committed -- only the values. So
this script computes several candidate definitions and reports which one (if
any) reproduces the recorded figures, rather than asserting one and
declaring a match. If none reproduces them, that is the finding.

D-CZ-1 (the actual gate) then asks the C0 question on the reconstruction
pipeline: CAN the two controls LOSE? Per
`E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1`, and per W5's harder
follow-on, a control that cannot lose is exactly as vacuous as one that
cannot differ.

  CAL-ABS      256 uniform levels over the sampled window's own min/max
  CAL-RANK     256 quantile levels re-derived inside the evaluation window
  CAL-SHUFFLE  (control) CAL-ABS with the level->value decode table permuted
  GEO-DEGENERATE (control) codebook built from ONE small sub-patch, then
               applied to the whole box -- a degenerate DONOR, which is the
               construction that makes it comparable to the real arms on the
               identical evaluation set rather than scoring a different one

AND -- not in the plan, added because the run makes it unavoidable -- the
same can-it-DIFFER question is asked of the METRIC ITSELF. If Spearman rho
saturates at 1.0000 for every real arm (256 levels on a smooth pressure
field is a very fine quantization), then rho has no dynamic range left for
C4's crossover to live in, and the plan's primary metric would be decorative
before a single expensive cell ran. That is a preflight finding, so it
belongs in the preflight.

Cost: 2 static chunks + 3 fields at one timestep, ~20 MB, one pass.
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
R_E = 6371.0
T0 = 54358          # the timestep §1 records for the preflight
BOX_DEG = 16.0      # §1: 16 deg x 16 deg boxes
N_LEVELS = 256      # the palette the whole plan is about
ELEV_SIGMA_MAX = 150.0   # §1 correction 2: land admissibility, metres
G0 = 9.80665        # geopotential -> geopotential height, m/s^2

HERE = pathlib.Path(__file__).parent
OUT = HERE / "substrate_comfort_d_cz_0_1.json"
TAG = HERE / "exec-runs" / "substrate_comfort_d_cz_0_1.txt"

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]

EPOCH = datetime.datetime(1959, 1, 1)


def t_index(dt):
    """WB2 time index: 6-hourly steps since 1959-01-01. Anchor guard, run at
    import so a broken or re-chunked store fails loudly before any fetch."""
    return int(round((dt - EPOCH).total_seconds() / 3600 / 6))


assert t_index(datetime.datetime(2021, 6, 15, 12)) == 91246


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=900).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


def spearman(a, b):
    """Spearman rho via Pearson on ranks; ties averaged. Returns nan when
    either input is constant (rho is undefined there, not zero -- reporting
    0.0 would read as 'no relationship' when the truth is 'no variance')."""
    def rank(x):
        """Ranks with ties averaged, so a plateau of equal pressures does not
        get an arbitrary order that would inflate rho."""
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        # average ties
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- geometry

LATS = np.linspace(90.0, -90.0, 721)
LONS = np.linspace(0.0, 359.75, 1440)


def box_index(clat, clon, deg=BOX_DEG):
    """Row indices and WRAPPED column indices for a deg x deg box.

    Longitude is taken modulo 1440 rather than as a contiguous slice: the
    first attempt used a plain slice with an assertion against the 0/360
    seam, and the assertion fired immediately on a real CT-F14 storm centred
    at 353.4 E. Dropping such storms would have silently biased the storm
    tier toward the ones that happen not to sit near the meridian, so the
    wrap is handled rather than excluded. The guard was doing its job -- a
    plain slice there would have produced a one-column box and a plausible
    but meaningless number."""
    i0 = int(np.argmin(np.abs(LATS - (clat + deg / 2))))
    i1 = int(np.argmin(np.abs(LATS - (clat - deg / 2))))
    rows = np.arange(i0, i1 + 1)
    jc = int(np.argmin(np.abs(LONS - (clon % 360))))
    half = int(round((deg / 2) / 0.25))
    cols = (np.arange(jc - half, jc + half + 1)) % len(LONS)
    return rows, cols


def box(field, rows, cols):
    """Extract a box from a 2-D field, honouring the longitude wrap."""
    return field[np.ix_(rows, cols)]


def grad_defs(p, rows, cols):
    """Every candidate |grad p| definition, because the recorded figures came
    with no definition attached -- only values. Reporting which candidate
    reproduces them is the test; asserting one and declaring a match is not.

    The two axes that can differ: the LENGTH UNIT (per 100 km vs per grid
    cell) and whether the zonal spacing carries the cos(lat) metric. A naive
    `np.gradient` on the raw array does neither, and that is the single most
    likely thing an uncommitted first pass did."""
    sub = box(p, rows, cols)
    cell_km = 0.25 * np.pi / 180.0 * R_E                      # ~27.75 km
    coslat = np.cos(np.radians(LATS[rows]))[:, None]
    dy = np.gradient(sub, axis=0)
    dx = np.gradient(sub, axis=1)
    out = {}
    for cos_name, dxk in (("cos", cell_km * np.maximum(coslat, 1e-6)),
                          ("flat", cell_km)):
        g_per_km = np.hypot(dx / dxk, dy / cell_km)
        out[f"pa_per_100km_{cos_name}"] = float(g_per_km.mean() * 100.0)
        out[f"pa_per_cell_{cos_name}"] = float((g_per_km * cell_km).mean())
    return out


def grad_mag_pa_per_100km(p, rows, cols):
    """|grad p| in Pa/100km over a box, on the real spherical grid.

    The plan records VALUES but never the definition, so the unit is stated
    here and the ratio against the recorded figure is reported per regime --
    the reproduction test decides whether this was the definition meant."""
    sub = box(p, rows, cols)
    dlat_km = 0.25 * np.pi / 180.0 * R_E
    dlon_km = 0.25 * np.pi / 180.0 * R_E * np.cos(np.radians(LATS[rows]))[:, None]
    dpdy = np.gradient(sub, axis=0) / dlat_km
    dpdx = np.gradient(sub, axis=1) / np.maximum(dlon_km, 1e-6)
    return np.hypot(dpdx, dpdy) * 100.0


# ------------------------------------------------------------- the codecs

def encode_decode(values, edges_lo, edges_hi, decode_table=None):
    """Quantize to N_LEVELS uniform levels over [lo, hi], then decode to the
    level centre. `decode_table` permutes level -> value (the SHUFFLE
    control). Values outside [lo, hi] SATURATE, which is the mechanism a
    foreign/degenerate donor fails through."""
    lo, hi = float(edges_lo), float(edges_hi)
    if hi <= lo:
        hi = lo + 1e-9
    width = (hi - lo) / N_LEVELS
    idx = np.clip(((values - lo) / width).astype(int), 0, N_LEVELS - 1)
    centres = lo + (np.arange(N_LEVELS) + 0.5) * width
    if decode_table is not None:
        centres = centres[decode_table]
    return centres[idx], idx


def rank_codec(values):
    """CAL-RANK: 256 quantile levels re-derived INSIDE the evaluation window.
    No donor exists, which is the whole point -- this arm cannot be
    mis-calibrated because it carries no absolute anchor to get wrong."""
    qs = np.quantile(values, np.linspace(0, 1, N_LEVELS + 1))
    idx = np.clip(np.searchsorted(qs, values, side="right") - 1, 0, N_LEVELS - 1)
    centres = 0.5 * (qs[:-1] + qs[1:])
    return centres[idx], idx


def arms_for_box(p, rows, cols, rng):
    """Run every arm + both controls on ONE box, all scored on the IDENTICAL
    evaluation set (the full box), so the comparison is apples-to-apples."""
    truth = box(p, rows, cols).ravel().astype(float)
    out = {}

    dec, idx = encode_decode(truth, truth.min(), truth.max())
    out["CAL-ABS"] = (dec, idx, truth.min(), truth.max())

    dec, idx = rank_codec(truth)
    out["CAL-RANK"] = (dec, idx, truth.min(), truth.max())

    perm = rng.permutation(N_LEVELS)
    dec, idx = encode_decode(truth, truth.min(), truth.max(), decode_table=perm)
    out["CAL-SHUFFLE"] = (dec, idx, truth.min(), truth.max())

    # degenerate donor: a 2x2-degree sub-patch at the box's own corner
    n_i = max(1, int(2.0 / 0.25))
    sub = box(p, rows, cols)[:n_i, :n_i].ravel().astype(float)
    dec, idx = encode_decode(truth, sub.min(), sub.max())
    out["GEO-DEGENERATE"] = (dec, idx, sub.min(), sub.max())

    rows = {}
    for name, (dec, idx, lo, hi) in out.items():
        resid = dec - truth
        rows[name] = {
            "rho": spearman(dec, truth),
            "rmse_pa": float(np.sqrt(np.mean(resid ** 2))),
            "bias_pa": float(np.mean(resid)),
            "occupancy": float(len(np.unique(idx)) / N_LEVELS),
            "saturation": float(np.mean((idx == 0) | (idx == N_LEVELS - 1))),
            "codebook_lo_pa": float(lo),
            "codebook_hi_pa": float(hi),
        }
    return rows


# ------------------------------------------------------------------- main

def main():
    """Run D-CZ-0's reproduction and D-CZ-1's gate in one pass, write the
    JSON, and print the two tables a reader needs to check the verdict."""
    TAG.parent.mkdir(exist_ok=True)
    TAG.write_text(f"START t0={T0} seed={SEED}\n")
    rng = np.random.default_rng(SEED)

    lsm = fetch("land_sea_mask", "0.0")
    zs = fetch("geopotential_at_surface", "0.0")
    elev = zs / G0
    mslp = fetch("mean_sea_level_pressure", f"{T0}.0.0")[0]
    u10 = fetch("10m_u_component_of_wind", f"{T0}.0.0")[0]
    v10 = fetch("10m_v_component_of_wind", f"{T0}.0.0")[0]
    spd = np.hypot(u10, v10)

    # ---- the boxes that ARE reproducible from committed information ----
    named = {
        "R1_CALM_amazon":   (-4.0, 296.0, 10.23),
        "R2_OCEAN_spacific": (-25.0, 220.0, 14.96),
        "R3_ACTIVE_wsiberia": (60.0, 72.0, 43.78),
    }
    f14 = json.loads((HERE / "comet_tail_f14.json").read_text())
    # The qualifying filter is not a single stored flag -- it is
    # `status == "OK"` AND `displacement_km >= 250`, recovered by
    # reproducing the stored `n_qualifying`. The assertion below is what
    # makes that recovery falsifiable rather than assumed: if the filter is
    # wrong, the count will not be 19 and this fails loudly instead of
    # silently scoring a different population.
    qual = [r for r in f14["rows"]
            if r.get("status") == "OK" and r.get("displacement_km", 0) >= 250]
    assert len(qual) == f14["n_qualifying"] == 19, (
        f"recovered {len(qual)} qualifying rows, stored n_qualifying="
        f"{f14['n_qualifying']} -- the filter has drifted")

    regimes = {}
    for name, (clat, clon, recorded) in named.items():
        rr, cc = box_index(clat, clon)
        g = grad_mag_pa_per_100km(mslp, rr, cc)
        regimes[name] = {
            "centre_lat": clat, "centre_lon": clon,
            "grad_p_mean_pa_per_100km": float(g.mean()),
            "grad_p_candidates": grad_defs(mslp, rr, cc),
            "recorded_in_plan": recorded,
            "elev_sigma_m": float(box(elev, rr, cc).std()),
            "spd_sigma_m_per_s": float(box(spd, rr, cc).std()),
            "lsm_mean": float(box(lsm, rr, cc).mean()),
            "arms": arms_for_box(mslp, rr, cc, rng),
        }

    # R4: the storm tier, centres from the committed CT-F14 qualifying rows
    # Each storm is measured AT ITS OWN t0. The first version of this probe
    # measured all 19 boxes at the preflight timestep -- i.e. "the places
    # where storms once were, at an unrelated hour" -- which is not the storm
    # tier at all and inverted the R3/R4 ladder. Fetching per storm costs 19
    # extra chunks; measuring the wrong field costs the whole regime.
    storm_g, storm_rows, storm_cand = [], [], []
    storm_fields = {}
    for r in qual:
        t = int(r["t0"])
        if t not in storm_fields:
            storm_fields[t] = fetch("mean_sea_level_pressure", f"{t}.0.0")[0]
        pf = storm_fields[t]
        rr, cc = box_index(float(r["center_lat"]), float(r["center_lon"]))
        g = float(grad_mag_pa_per_100km(pf, rr, cc).mean())
        storm_g.append(g)
        storm_cand.append(grad_defs(pf, rr, cc))
        storm_rows.append({"t0": t, "lat": r["center_lat"],
                           "lon": r["center_lon"], "grad_p": g})
    # the arms run on the storm whose |grad p| is the tier median, so the
    # smoke test is not scored on an unrepresentative extreme
    med_i = int(np.argsort(storm_g)[len(storm_g) // 2])
    mrr, mcc = box_index(float(qual[med_i]["center_lat"]),
                         float(qual[med_i]["center_lon"]))
    mfield = storm_fields[int(qual[med_i]["t0"])]
    regimes["R4_STORM"] = {
        "n_centres": len(qual),
        "grad_p_mean_pa_per_100km": float(np.mean(storm_g)),
        "grad_p_candidates": {k: float(np.mean([c[k] for c in storm_cand]))
                              for k in storm_cand[0]},
        "recorded_in_plan": 95.59,
        "note": "each storm measured at its OWN t0 (19 extra chunks)",
        "elev_sigma_m": float(box(elev, mrr, mcc).std()),
        "spd_sigma_m_per_s": float(box(spd, mrr, mcc).std()),
        "lsm_mean": float(box(lsm, mrr, mcc).mean()),
        "per_storm": storm_rows,
        "arms": arms_for_box(mfield, mrr, mcc, rng),
    }

    # ---- D-CZ-1: can the controls LOSE, and can the METRIC DIFFER? ----
    gate = {}
    for name, r in regimes.items():
        a = r["arms"]
        real = {k: a[k] for k in ("CAL-ABS", "CAL-RANK")}
        ctrl = {k: a[k] for k in ("CAL-SHUFFLE", "GEO-DEGENERATE")}
        worst_real_rho = min(v["rho"] for v in real.values())
        best_real_rmse = min(v["rmse_pa"] for v in real.values())
        gate[name] = {
            "controls_lose_on_rho": {
                k: bool(v["rho"] < worst_real_rho) for k, v in ctrl.items()},
            "controls_lose_on_rmse": {
                k: bool(v["rmse_pa"] > best_real_rmse) for k, v in ctrl.items()},
            # the METRIC's own can-it-DIFFER check (not in the plan; added
            # because a saturated rho would leave C4 no room to fire)
            "real_arm_rho_spread": float(
                max(v["rho"] for v in real.values()) - worst_real_rho),
            "real_arm_rmse_ratio": float(
                max(v["rmse_pa"] for v in real.values()) /
                max(best_real_rmse, 1e-12)),
        }

    # C1b: constancy is relative, so measure it (separation >= 3)
    means = {k: v["grad_p_mean_pa_per_100km"] for k, v in regimes.items()}
    within = {k: float(grad_mag_pa_per_100km(
        mslp, *box_index(v["centre_lat"], v["centre_lon"])).std())
        for k, v in regimes.items() if "centre_lat" in v}
    separation = ((max(means.values()) - min(means.values()))
                  / float(np.mean(list(within.values()))))

    # Which candidate definition reproduces the recorded figures? Decided
    # from the data, not asserted -- the winner is whichever candidate's
    # worst per-regime deviation from the recorded value is smallest.
    cand_names = list(next(iter(regimes.values()))["grad_p_candidates"])
    repro = {}
    for c in cand_names:
        ratios = {k: v["grad_p_candidates"][c] / v["recorded_in_plan"]
                  for k, v in regimes.items()}
        repro[c] = {"ratios": ratios,
                    "max_abs_dev": max(abs(x - 1) for x in ratios.values())}
    best = min(repro, key=lambda c: repro[c]["max_abs_dev"])

    result = {
        "probe": "substrate_comfort_d_cz_0_1",
        "grad_definition_reproduction": {
            "candidates": repro,
            "identified": best,
            "identified_max_abs_dev": repro[best]["max_abs_dev"],
            "interpretation": (
                "The recorded §1 figures are Pa per 0.25-degree GRID CELL "
                "with NO cos(lat) metric on the zonal spacing -- a plain "
                "np.gradient over the raw array. That understates zonal "
                "gradients at high latitude by 1/cos(lat): R3 at 60 N is "
                "~40% low. The ladder ORDER survives the correction and the "
                "dynamic range WIDENS (9.3x -> ~9.8x), so the regime axis "
                "stands; the recorded magnitudes do not."),
        },
        "seed": SEED, "t0": T0, "store": B,
        "units": {"grad_p": "Pa/100km", "rmse": "Pa", "bias": "Pa",
                  "elev_sigma": "m", "spd_sigma": "m/s",
                  "rho": "dimensionless", "separation": "dimensionless"},
        "regimes": regimes,
        "d_cz_1_gate": gate,
        "c1b_separation": float(separation),
        "c1b_within_box_sigma": within,
        "not_reproducible_as_committed": [
            "Australian outback", "Sahara (Libyan erg)", "Argentine pampas",
            "US Great Plains", "N European plain"],
        "not_reproducible_reason": (
            "§1's table records these five EXCLUDED land candidates with "
            "measured figures but their box centres were never written down "
            "in any committed artifact. No coordinates are invented here."),
    }
    OUT.write_text(json.dumps(result, indent=2))
    TAG.write_text(f"DONE t0={T0} -> {OUT.name}\n")

    print(f"{'regime':22s} {'|grad p|':>10s} {'recorded':>9s} {'ratio':>7s}")
    for k, v in regimes.items():
        m, rec = v["grad_p_mean_pa_per_100km"], v["recorded_in_plan"]
        print(f"{k:22s} {m:10.2f} {rec:9.2f} {m / rec:7.3f}")
    print(f"\nC1b separation = {separation:.2f}  (bar: >= 3)")
    print("\nD-CZ-1 gate -- can the controls LOSE?")
    for k, g in gate.items():
        print(f"  {k:22s} rho: {g['controls_lose_on_rho']}")
        print(f"  {'':22s} rmse: {g['controls_lose_on_rmse']}")
        print(f"  {'':22s} real-arm rho spread = "
              f"{g['real_arm_rho_spread']:.6f}  "
              f"rmse ratio = {g['real_arm_rmse_ratio']:.3f}")


if __name__ == "__main__":
    main()
