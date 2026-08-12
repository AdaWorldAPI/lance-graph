"""D-CZ-2..7 — the cross-swap transfer matrix and the gates that precede it.

Runs the bars pre-registered in `.claude/plans/substrate-comfort-zones-v1.md`
§3, in the order the plan itself requires: **C1c first**, because a null there
VOIDS the cross-swap interpretation rather than weakening it.

  C1   regime ladder holds on >= 3 independent timesteps
  C1b  separation = (between-box range) / (mean within-box sigma) >= 3
  C1c  the regimes must differ in CORRELATION STRUCTURE, not merely |grad p|
  C2   the degenerate row -- both halves (dynamic L == 0 exactly, AND
       CAL-ABS proven able to be non-degenerate through the same code path)
  C3   transfer loss shrinks with turbulence
  C4   the crossover, on the AMENDED metric (RMSE in Pa, not rho -- §6.4)
  C6   the full transfer matrix, every cell raw

C5 (the geometry axis) is NOT run here, and the reason is structural rather
than budgetary -- see `c5_structural_blocker` in the JSON. It is reported,
not skipped silently.

EQUAL BUDGET is enforced by construction: every regime is subsampled to the
same cell count with a seeded RNG before any codebook is built. The W2s-a
lesson (an unequal budget silently advantages the arm with more samples) is
not re-learned here.

R4's timesteps are the 19 storms' OWN t0 values -- measuring a storm box at
an unrelated hour is the defect this probe's predecessor already caught and
fixed. That also means R4 satisfies C1's ">= 3 independent timesteps" with
19, while R1..R3 are measured at 3 explicit timesteps each.
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
BOX_DEG = 16.0
N_LEVELS = 256
G0 = 9.80665

# C1 needs >= 3 independent timesteps. t0 is the preflight's; the other two are
# offset by ~6 months and ~1 year so the ladder is tested across seasons, not
# across three adjacent hours (which would be one condition sampled thrice).
TIMESTEPS = [54358, 54358 + 4 * 182, 54358 + 4 * 365]

HERE = pathlib.Path(__file__).parent
OUT = HERE / "substrate_comfort_d_cz_2_7.json"
TAG = HERE / "exec-runs" / "substrate_comfort_d_cz_2_7.txt"

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]
EPOCH = datetime.datetime(1959, 1, 1)


def t_index(dt):
    """WB2 time index: 6-hourly steps since 1959-01-01. Anchor guard."""
    return int(round((dt - EPOCH).total_seconds() / 3600 / 6))


assert t_index(datetime.datetime(2021, 6, 15, 12)) == 91246


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=900).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


LATS = np.linspace(90.0, -90.0, 721)
LONS = np.linspace(0.0, 359.75, 1440)


def box_index(clat, clon, deg=BOX_DEG):
    """Row indices and WRAPPED column indices for a deg x deg box."""
    i0 = int(np.argmin(np.abs(LATS - (clat + deg / 2))))
    i1 = int(np.argmin(np.abs(LATS - (clat - deg / 2))))
    jc = int(np.argmin(np.abs(LONS - (clon % 360))))
    half = int(round((deg / 2) / 0.25))
    return np.arange(i0, i1 + 1), (np.arange(jc - half, jc + half + 1)) % len(LONS)


def box(field, rows, cols):
    """Extract a box from a 2-D field, honouring the longitude wrap."""
    return field[np.ix_(rows, cols)]


def grad_flat(p, rows, cols):
    """|grad p| in Pa per grid cell, NO cos(lat) — the definition identified
    from the data in D-CZ-0 as the one the §1 preflight figures used. Kept so
    the ladder here is comparable to the recorded table, not to a different
    quantity."""
    sub = box(p, rows, cols)
    return np.hypot(np.gradient(sub, axis=1), np.gradient(sub, axis=0))


def spearman(a, b):
    """Spearman rho via Pearson on ranks, ties averaged; nan on constant
    input (rho is undefined there, not zero)."""
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
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


# ------------------------------------------------------- C1c instruments

def decay_length_cells(field2d):
    """Autocorrelation decay length in grid cells: the lag at which the
    field's spatial autocorrelation first drops below 1/e, averaged over the
    two axes. Measured on the ANOMALY (box mean removed) so the decay is of
    structure, not of the offset."""
    a = field2d - field2d.mean()
    out = []
    for axis in (0, 1):
        n = a.shape[axis]
        cors = []
        for lag in range(1, min(n // 2, 40)):
            x = np.take(a, range(0, n - lag), axis=axis)
            y = np.take(a, range(lag, n), axis=axis)
            xv, yv = x.ravel(), y.ravel()
            if xv.std() == 0 or yv.std() == 0:
                cors.append(0.0)
            else:
                cors.append(float(np.corrcoef(xv, yv)[0, 1]))
        thr = 1.0 / np.e
        hit = next((i + 1 for i, c in enumerate(cors) if c < thr), len(cors) + 1)
        out.append(float(hit))
    return float(np.mean(out))


def gini(x):
    """Gini coefficient of a non-negative sample — the concentration of the
    |grad p| distribution. 0 = every cell equally gradient-y, 1 = all the
    gradient in one cell."""
    v = np.sort(np.asarray(x, float).ravel())
    v = v[np.isfinite(v)]
    if v.size == 0 or v.sum() == 0:
        return float("nan")
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def tail_ratio(x):
    """p99 / p50 of |grad p| — how far the extreme sits above the typical."""
    v = np.asarray(x, float).ravel()
    med = float(np.percentile(v, 50))
    return float(np.percentile(v, 99) / med) if med > 0 else float("nan")


# ------------------------------------------------------------- the codecs

def encode_decode(values, lo, hi, decode_table=None):
    """Quantize to N_LEVELS uniform levels over [lo, hi], decode to the level
    centre. Values outside SATURATE — the mechanism a wrong donor fails by."""
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1e-9
    width = (hi - lo) / N_LEVELS
    idx = np.clip(((values - lo) / width).astype(int), 0, N_LEVELS - 1)
    centres = lo + (np.arange(N_LEVELS) + 0.5) * width
    if decode_table is not None:
        centres = centres[decode_table]
    return centres[idx], idx


def rank_codec(values):
    """CAL-RANK: 256 quantile levels re-derived INSIDE the window. No donor
    exists — that is the property under test, not a defect."""
    qs = np.quantile(values, np.linspace(0, 1, N_LEVELS + 1))
    idx = np.clip(np.searchsorted(qs, values, side="right") - 1, 0, N_LEVELS - 1)
    return 0.5 * (qs[:-1] + qs[1:])[idx], idx


def fisherz_codec(values):
    """CAL-FISHERZ: uniform levels in arctanh(2*rank-1) — resolution moved
    into the tails. Also window-local, so also donor-free."""
    n = len(values)
    r = (np.argsort(np.argsort(values)) + 0.5) / n
    s = np.clip(2 * r - 1, -0.999999, 0.999999)
    z = np.arctanh(s)
    dec_z, idx = encode_decode(z, z.min(), z.max())
    # map the decoded z back to a value by inverting through the sorted sample
    back = np.clip((np.tanh(dec_z) + 1) / 2, 0, 1)
    srt = np.sort(values)
    pos = np.clip((back * n).astype(int), 0, n - 1)
    return srt[pos], idx


def cell_metrics(dec, idx, truth):
    """One matrix cell's numbers. Every quantity the plan's C6 requires."""
    resid = dec - truth
    return {
        "rho": spearman(dec, truth),
        "rmse_pa": float(np.sqrt(np.mean(resid ** 2))),
        "bias_pa": float(np.mean(resid)),
        "occupancy": float(len(np.unique(idx)) / N_LEVELS),
        "saturation": float(np.mean((idx == 0) | (idx == N_LEVELS - 1))),
    }


# ------------------------------------------------------------------- main

def main():
    """Run C1c first (it can void everything), then C1/C1b, then the matrix."""
    TAG.parent.mkdir(exist_ok=True)
    TAG.write_text("START d-cz-2..7\n")
    rng = np.random.default_rng(SEED)

    named = {
        "R1_CALM": (-4.0, 296.0),
        "R2_OCEAN": (-25.0, 220.0),
        "R3_ACTIVE": (60.0, 72.0),
    }
    f14 = json.loads((HERE / "comet_tail_f14.json").read_text())
    qual = [r for r in f14["rows"]
            if r.get("status") == "OK" and r.get("displacement_km", 0) >= 250]
    assert len(qual) == f14["n_qualifying"] == 19

    # ---- fields ----
    fields = {t: fetch("mean_sea_level_pressure", f"{t}.0.0")[0] for t in TIMESTEPS}
    storm_fields = {}
    for r in qual:
        t = int(r["t0"])
        if t not in storm_fields:
            storm_fields[t] = fetch("mean_sea_level_pressure", f"{t}.0.0")[0]

    # ---- C1 / C1b : ladder + separation across timesteps ----
    ladder, within = {}, {}
    for name, (clat, clon) in named.items():
        rr, cc = box_index(clat, clon)
        ladder[name] = {str(t): float(grad_flat(fields[t], rr, cc).mean())
                        for t in TIMESTEPS}
        within[name] = {str(t): float(grad_flat(fields[t], rr, cc).std())
                        for t in TIMESTEPS}
    storm_g = []
    for r in qual:
        rr, cc = box_index(float(r["center_lat"]), float(r["center_lon"]))
        storm_g.append(float(grad_flat(storm_fields[int(r["t0"])], rr, cc).mean()))
    ladder["R4_STORM"] = {"per_storm_mean": float(np.mean(storm_g)),
                          "per_storm_min": float(np.min(storm_g)),
                          "n_independent_timesteps": len(set(int(r["t0"]) for r in qual))}

    c1 = {}
    for t in TIMESTEPS:
        vals = [ladder["R1_CALM"][str(t)], ladder["R2_OCEAN"][str(t)],
                ladder["R3_ACTIVE"][str(t)], float(np.mean(storm_g))]
        c1[str(t)] = {"values": vals, "ordered": bool(all(
            vals[i] < vals[i + 1] for i in range(3)))}
    c1_holds = all(v["ordered"] for v in c1.values())

    c1b = {}
    for t in TIMESTEPS:
        means = [ladder[k][str(t)] for k in named] + [float(np.mean(storm_g))]
        sig = float(np.mean([within[k][str(t)] for k in named]))
        c1b[str(t)] = float((max(means) - min(means)) / sig)

    # ---- C1c : do the regimes differ in STRUCTURE, not just in |grad p|? ----
    c1c = {}
    for name, (clat, clon) in named.items():
        rr, cc = box_index(clat, clon)
        f = box(fields[TIMESTEPS[0]], rr, cc)
        g = grad_flat(fields[TIMESTEPS[0]], rr, cc)
        c1c[name] = {"decay_length_cells": decay_length_cells(f),
                     "gini_gradp": gini(g), "tail_ratio_gradp": tail_ratio(g)}
    dl, gi, tr = [], [], []
    for r in qual:
        rr, cc = box_index(float(r["center_lat"]), float(r["center_lon"]))
        f = box(storm_fields[int(r["t0"])], rr, cc)
        g = grad_flat(storm_fields[int(r["t0"])], rr, cc)
        dl.append(decay_length_cells(f)); gi.append(gini(g)); tr.append(tail_ratio(g))
    c1c["R4_STORM"] = {"decay_length_cells": float(np.median(dl)),
                       "gini_gradp": float(np.median(gi)),
                       "tail_ratio_gradp": float(np.median(tr)),
                       "n": len(dl)}
    # the bar: R1 and R4 must be DISTINGUISHABLE on these
    r1, r4 = c1c["R1_CALM"], c1c["R4_STORM"]
    c1c_verdict = {
        "decay_ratio_r4_over_r1": r4["decay_length_cells"] / r1["decay_length_cells"],
        "gini_ratio_r4_over_r1": r4["gini_gradp"] / r1["gini_gradp"],
        "tail_ratio_r4_over_r1": r4["tail_ratio_gradp"] / r1["tail_ratio_gradp"],
    }
    c1c_verdict["distinguishable"] = bool(
        max(abs(v - 1.0) for k, v in c1c_verdict.items() if k.endswith("r1")) >= 0.20)

    # ---- equal-budget samples, one per regime -- AND a genuine spatial
    # sub-patch per regime for GEO-DEGENERATE, kept SEPARATE from the
    # equal-budget flat sample. Slicing the flat (post-subsample, post-
    # shuffle) array is NOT a spatial sub-patch -- rng.choice returns no
    # spatial order, so its first N elements are an ordinary random
    # subsample, not a narrow corner. The degenerate donor must be built
    # from the box's own 2-D layout, exactly as D-CZ-1 does it.
    boxes2d = {}
    for name, (clat, clon) in named.items():
        rr, cc = box_index(clat, clon)
        boxes2d[name] = box(fields[TIMESTEPS[0]], rr, cc).astype(float)
    storm_boxes2d = [
        box(storm_fields[int(r["t0"])],
            *box_index(float(r["center_lat"]), float(r["center_lon"]))).astype(float)
        for r in qual]
    # R4's "box" for the degenerate patch is the storm whose |grad p| is the
    # tier median, matching D-CZ-1's own choice for the same purpose.
    boxes2d["R4_STORM"] = storm_boxes2d[int(np.argsort(storm_g)[len(storm_g) // 2])]

    samples = {k: v.ravel() for k, v in boxes2d.items()}
    samples["R4_STORM"] = np.concatenate(
        [b.ravel() for b in storm_boxes2d]).astype(float)
    budget = min(len(v) for v in samples.values())
    for k in list(samples):
        if len(samples[k]) > budget:
            samples[k] = samples[k][rng.choice(len(samples[k]), budget, replace=False)]
    assert len({len(v) for v in samples.values()}) == 1, "equal budget violated"

    def degenerate_patch_range(name):
        """A real spatial corner -- n_i x n_i cells, n_i = max(1, side//8) --
        matching D-CZ-1's `p[si,sj][:n_i,:n_i]` construction exactly."""
        b2 = boxes2d[name]
        n_i = max(1, b2.shape[0] // 8)
        patch = b2[:n_i, :n_i]
        return float(patch.min()), float(patch.max())

    REG = list(samples)

    # ---- C2 / C3 / C4 / C6 : the transfer matrix ----
    matrix = {}
    for T in REG:
        truth = samples[T]
        matrix[T] = {}
        for D in REG:                      # CAL-ABS: a REAL donor matrix
            dec, idx = encode_decode(truth, samples[D].min(), samples[D].max())
            matrix[T][f"CAL-ABS|{D}"] = cell_metrics(dec, idx, truth)
        for D in REG:                      # dynamic arms: donor is IGNORED by
            dec, idx = rank_codec(truth)   # construction — that is the test
            matrix[T][f"CAL-RANK|{D}"] = cell_metrics(dec, idx, truth)
            dec, idx = fisherz_codec(truth)
            matrix[T][f"CAL-FISHERZ|{D}"] = cell_metrics(dec, idx, truth)
        perm = rng.permutation(N_LEVELS)   # controls, on the diagonal
        dec, idx = encode_decode(truth, truth.min(), truth.max(), decode_table=perm)
        matrix[T]["CAL-SHUFFLE|self"] = cell_metrics(dec, idx, truth)
        plo, phi = degenerate_patch_range(T)
        dec, idx = encode_decode(truth, plo, phi)
        matrix[T]["GEO-DEGENERATE|self"] = cell_metrics(dec, idx, truth)

    def L(arm, D, T):
        return matrix[T][f"{arm}|{T}"]["rho"] - matrix[T][f"{arm}|{D}"]["rho"]

    def undefined_donors(T):
        """Donors under which CAL-ABS's decode of T is fully saturated to one
        level -- rho is undefined there, not merely small. A first-class
        outcome, not a missing measurement: it means the donor's range is
        so narrow relative to the target that structure is destroyed
        entirely, the extreme end of what L is built to detect."""
        return [D for D in REG if np.isnan(matrix[T][f"CAL-ABS|{D}"]["rho"])]

    c2 = {"dynamic_rows_flat": {}, "cal_abs_can_differ": {},
          "cal_abs_undefined_donors": {T: undefined_donors(T) for T in REG}}
    for arm in ("CAL-RANK", "CAL-FISHERZ"):
        # dynamic arms have no donor at all, so L is 0 vs every D by
        # construction -- never nan; a plain max is correct here.
        c2["dynamic_rows_flat"][arm] = {
            T: max(abs(L(arm, D, T)) for D in REG) for T in REG}
    for T in REG:
        defined = [abs(L("CAL-ABS", D, T)) for D in REG if D not in undefined_donors(T)]
        c2["cal_abs_can_differ"][T] = max(defined) if defined else float("nan")
    c2["half_i_dynamic_L_is_exactly_zero"] = bool(all(
        v == 0.0 for a in c2["dynamic_rows_flat"].values() for v in a.values()))
    c2["half_ii_cal_abs_nonzero_somewhere"] = bool(
        np.nanmax(list(c2["cal_abs_can_differ"].values())) > 0)

    def l_bar(T):
        """Mean off-diagonal transfer loss over DEFINED donors, with the
        undefined (fully-saturated) donors counted separately rather than
        silently dropped or silently nan-poisoning the mean."""
        vals = [L("CAL-ABS", D, T) for D in REG if D != T
                and not np.isnan(matrix[T][f"CAL-ABS|{D}"]["rho"])]
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    c3 = {}
    for T in REG:
        lb, n_defined = l_bar(T)
        c3[T] = {"L_bar": lb, "n_defined_offdiag": n_defined,
                 "n_undefined_offdiag": len(REG) - 1 - n_defined,
                 "occupancy_mean_offdiag": float(np.mean(
                     [matrix[T][f"CAL-ABS|{D}"]["occupancy"] for D in REG if D != T])),
                 "saturation_mean_offdiag": float(np.mean(
                     [matrix[T][f"CAL-ABS|{D}"]["saturation"] for D in REG if D != T]))}
    # C3 as pre-registered compares L_bar(R4) to L_bar(R1). Both must be
    # DEFINED for the comparison to mean what the bar says; report which
    # case obtains rather than silently computing a NaN-laundered verdict.
    c3_r1_r4_comparable = bool(np.isfinite(c3["R1_CALM"]["L_bar"])
                               and np.isfinite(c3["R4_STORM"]["L_bar"]))
    c3_holds = (bool(c3["R4_STORM"]["L_bar"] < c3["R1_CALM"]["L_bar"])
               if c3_r1_r4_comparable else None)

    c4 = {}
    for T in REG:
        d_rmse = (matrix[T][f"CAL-RANK|{T}"]["rmse_pa"]
                  - matrix[T][f"CAL-ABS|{T}"]["rmse_pa"])
        weak = float(np.mean([matrix[T][f"CAL-ABS|{D}"]["rmse_pa"]
                              for D in REG if D != T])) - matrix[T][f"CAL-RANK|{T}"]["rmse_pa"]
        c4[T] = {"delta_rmse_rank_minus_absdiag": d_rmse,
                 "rho_floor_check_min": min(matrix[T][f"CAL-{a}|{T}"]["rho"]
                                            for a in ("ABS", "RANK", "FISHERZ")),
                 "weak_form_absforeign_minus_rank_rmse": weak}
    c4_flip = bool(c4["R1_CALM"]["delta_rmse_rank_minus_absdiag"] > 0
                   and c4["R2_OCEAN"]["delta_rmse_rank_minus_absdiag"] > 0
                   and c4["R4_STORM"]["delta_rmse_rank_minus_absdiag"] < 0)

    c0 = {}
    for T in REG:
        real = min(matrix[T][f"CAL-{a}|{T}"]["rho"] for a in ("ABS", "RANK", "FISHERZ"))
        c0[T] = {c: bool(matrix[T][f"{c}|self"]["rho"] < real)
                 for c in ("CAL-SHUFFLE", "GEO-DEGENERATE")}

    result = {
        "probe": "substrate_comfort_d_cz_2_7", "seed": SEED,
        "timesteps": TIMESTEPS, "budget_cells_per_regime": int(budget),
        "units": {"grad_p": "Pa/cell (flat, no cos-lat — the D-CZ-0 identified "
                            "definition)", "rmse": "Pa", "bias": "Pa",
                  "decay_length": "grid cells", "L": "dimensionless (rho difference)"},
        "C0_controls": c0,
        "C1_ladder": {"per_timestep": c1, "holds_all": bool(c1_holds),
                      "raw": ladder},
        "C1b_separation": {"per_timestep": c1b, "bar": 3.0,
                           "holds_all": bool(all(v >= 3.0 for v in c1b.values()))},
        "C1c_structure": {"per_regime": c1c, "verdict": c1c_verdict},
        "C2_degenerate_row": c2,
        "C3_transfer_loss": {"per_regime": c3,
                             "r1_r4_both_defined": c3_r1_r4_comparable,
                             "holds_r4_lt_r1": c3_holds},
        "C4_crossover": {"per_regime": c4, "sign_flip": bool(c4_flip)},
        "C6_matrix": matrix,
        "c5_structural_blocker": (
            "C5 (GEO-GOLDEN-LO vs GEO-GOLDEN-HI) is NOT RUN, and the reason is "
            "structural rather than budgetary. The golden index floor requires "
            "N >= F(17)^2 = 2,550,409 lattice points before the emergent "
            "parastichy stride behaves like phi at all. A 16-degree box at 0.25 "
            "degree resolution contains 65 x 65 = 4,225 cells -- three orders of "
            "magnitude below the floor. GEO-GOLDEN-HI therefore CANNOT be "
            "constructed at box scale on this grid, so the comparison the bar "
            "asks for has no admissible high arm. Reported rather than faked "
            "with an interpolated sub-grid lattice, which would sample the "
            "interpolator and not the field."),
    }
    OUT.write_text(json.dumps(result, indent=2))
    TAG.write_text("DONE d-cz-2..7\n")

    print(f"budget/regime = {budget} cells\n")
    print("C1c STRUCTURE (runs first — a null here VOIDS the reading):")
    for k, v in c1c.items():
        print(f"  {k:10s} decay={v['decay_length_cells']:5.2f} cells  "
              f"gini={v['gini_gradp']:.4f}  tail={v['tail_ratio_gradp']:.3f}")
    print(f"  R4/R1 ratios: {c1c_verdict}")
    print(f"\nC1 ladder holds all timesteps: {c1_holds}")
    print(f"C1b separation per timestep: "
          f"{ {k: round(v,2) for k,v in c1b.items()} }")
    print(f"\nC2 half(i) dynamic L==0 exactly: {c2['half_i_dynamic_L_is_exactly_zero']}")
    print(f"C2 half(ii) CAL-ABS non-degenerate: {c2['half_ii_cal_abs_nonzero_somewhere']}"
          f"  max|L| per regime: { {k: round(v,4) for k,v in c2['cal_abs_can_differ'].items()} }")
    print("\nC3 transfer loss L_bar (lower = travels better; "
          "undefined = donor saturated the target to one level):")
    for k, v in c3.items():
        lb = f"{v['L_bar']:+.5f}" if np.isfinite(v['L_bar']) else "UNDEFINED"
        print(f"  {k:10s} L_bar={lb:>10s}  defined={v['n_defined_offdiag']}/"
              f"{v['n_defined_offdiag']+v['n_undefined_offdiag']}  "
              f"occ={v['occupancy_mean_offdiag']:.3f}  sat={v['saturation_mean_offdiag']:.3f}")
    print(f"  C3 R1/R4 both defined: {c3_r1_r4_comparable}  "
          f"holds (R4 < R1): {c3_holds}")
    print("\nC4 crossover on RMSE (>0 = absolute wins, <0 = dynamic wins):")
    for k, v in c4.items():
        print(f"  {k:10s} delta={v['delta_rmse_rank_minus_absdiag']:+9.3f} Pa"
              f"   rho_floor={v['rho_floor_check_min']:.6f}")
    print(f"  C4 sign flip: {c4_flip}")


if __name__ == "__main__":
    main()
