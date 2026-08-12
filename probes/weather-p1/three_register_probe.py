"""EXPLORATORY — the THREE-REGISTER picture measured on real ERA5 MSLP.
Operator direction (2026-08-11): "du lebst noch in der Vorstellung dass alles
absolut ist — die Relativitaetstheorie widerlegt sogar das" + "in der Statistik
ist es gold wert alles auf Palette256 normalized zu haben". NOT an EV.

THE CORRECTION THIS PROBE TESTS. I had encoded pressure against a FIXED 1000
hPa reference and called that "a stable convention". It is an ABSOLUTE anchor,
and there is none: 994 hPa is unremarkable over Iceland and a record in the
subtropics. What is meaningful is a value's POSITION IN ITS OWN DISTRIBUTION.
Once every field is rank-normalised onto the SAME palette256, the byte itself
becomes the comparison — one LUT for pressure, temperature, wind, vorticity,
and cross-variable distance becomes defined at all.

THREE REGISTERS over one value, all 1 byte, all from the same global field:

  A  AFFINE (Pa)        uniform bucket over [lo,hi]. Steps equal in VALUE.
                        This is what the geostrophic_stencil example encodes.
  B  PERCENTILE RANK    bucket = floor(rank*256). Steps equal in PROBABILITY
                        MASS. This is what RollingFloor's own occupancy/
                        cumulative-walk calibration (`roll()`, quantize.rs:167-
                        195, the 0.004/0.996 quantile targets) converges to.
  C  FISHER-Z OF RANK   s = 2*rank-1 in [-1,1], z = arctanh(s), uniform bucket
                        over z. Steps equal in VARIANCE-STABILISED units —
                        rim-stretched, i.e. resolution moved INTO the tails.
                        (helix fisher_z.rs: "stretching rim-near differences
                        before quantisation".)

PRE-REGISTERED (all measured in Pa so the three are comparable at all):

 R1  A's per-band error is FLAT across percentile bands (uniform by
     construction) — the control that the band decomposition is meaningful.
 R2  B beats A in the BULK (40-60th pct) — equal-probability buckets are
     denser in Pa exactly where the mass is.
 R3  B is WORSE than A in the STORM TAIL (bottom 1%) — the tail holds little
     probability mass, so rank spends few buckets there. This is the cost of
     register B and it must be measured, not assumed.
 R4  C beats B in the STORM TAIL — that is the whole point of the rim stretch.
     THE decisive comparison for a storm substrate, since a storm IS a tail
     event.
 R5  CROSS-VARIABLE (the "statistical gold"): after rank-normalisation, the
     SAME u8 denotes the same rarity in MSLP, 2m-temperature and 10m wind
     speed. Falsifier: for each register value b, the empirical fraction of
     each field at or below b must agree across the three fields to within
     1/256 (one bucket). For the ABSOLUTE register this comparison is not
     merely worse, it is UNDEFINED (the fields share no unit) — reported as
     such rather than as a number.

Same store / timestep as the arc: WB2 ERA5, t=91246 (2021-06-15 12Z).
Reference distribution = the GLOBAL field at that timestep (the population a
rolling floor would have observed). A true multi-year climatology would be the
honest reference for operational extremity; single-timestep is stated as the
scope limit, not hidden.
"""
import json
import pathlib
import urllib.request

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
T0 = 91246
PALETTE = 256

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=600).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


# ---- the three registers ---------------------------------------------------


def reg_a_affine(vals, lo, hi):
    """Register A: uniform bucket in VALUE space. Returns (code, recon)."""
    t = (vals - lo) / (hi - lo)
    b = np.clip(np.floor(t * PALETTE), 0, PALETTE - 1).astype(np.uint8)
    recon = lo + ((b.astype(np.float64) + 0.5) / PALETTE) * (hi - lo)
    return b, recon


def _ranks(vals, ref_sorted):
    """Empirical rank of each value against the sorted reference population.

    MIDPOINT convention: the mean of the left- and right-insertion ranks.
    Until 2026-08-11 this used `side="left"` alone (CodeRabbit, PR #926), which
    is ASYMMETRIC against the midpoint ranks `reg_c_fisher_rank` uses to build
    its own z-reference — the population's smallest value got rank 0 while the
    reference curve placed it at 0.5/n. On a field with ties (ERA5 MSLP is
    f32-quantised, so ties are common) the two conventions differ by up to a
    tie-run's width, and the disagreement is worst in the lower tail — which is
    exactly the storm tail R4 is decided on. Both halves now use midpoints.
    """
    lo = np.searchsorted(ref_sorted, vals, side="left")
    hi = np.searchsorted(ref_sorted, vals, side="right")
    return (lo + hi) / (2.0 * len(ref_sorted))


def reg_b_rank(vals, ref_sorted):
    """Register B: uniform bucket in PROBABILITY-MASS space."""
    r = _ranks(vals, ref_sorted)
    b = np.clip(np.floor(r * PALETTE), 0, PALETTE - 1).astype(np.uint8)
    # reconstruct: the value at the bucket's mid-quantile (inverse CDF)
    q = (b.astype(np.float64) + 0.5) / PALETTE
    recon = np.quantile(ref_sorted, q)
    return b, recon


def reg_c_fisher_rank(vals, ref_sorted):
    """Register C: uniform bucket in FISHER-Z-of-rank space (rim-stretched)."""
    eps = 1e-9
    r = _ranks(vals, ref_sorted)
    s = np.clip(2.0 * r - 1.0, -1.0 + eps, 1.0 - eps)
    z = np.arctanh(s)
    # bucket over the z-range the reference population itself spans
    r_ref = (np.arange(len(ref_sorted)) + 0.5) / len(ref_sorted)
    z_ref = np.arctanh(np.clip(2.0 * r_ref - 1.0, -1.0 + eps, 1.0 - eps))
    zlo, zhi = z_ref.min(), z_ref.max()
    t = (z - zlo) / (zhi - zlo)
    b = np.clip(np.floor(t * PALETTE), 0, PALETTE - 1).astype(np.uint8)
    # reconstruct: z-centre -> rank -> inverse CDF
    zc = zlo + ((b.astype(np.float64) + 0.5) / PALETTE) * (zhi - zlo)
    rc = np.clip((np.tanh(zc) + 1.0) / 2.0, 0.0, 1.0)
    recon = np.quantile(ref_sorted, rc)
    return b, recon


# ---- data ------------------------------------------------------------------

print("fetching global MSLP, 2m temperature, 10m wind ...", flush=True)
p = fetch("mean_sea_level_pressure", f"{T0}.0.0")[0].astype(np.float64)
t2m = fetch("2m_temperature", f"{T0}.0.0")[0].astype(np.float64)
u10 = fetch("10m_u_component_of_wind", f"{T0}.0.0")[0].astype(np.float64)
v10 = fetch("10m_v_component_of_wind", f"{T0}.0.0")[0].astype(np.float64)
wind = np.hypot(u10, v10)

pf = p.ravel()
ref = np.sort(pf)
lo, hi = pf.min(), pf.max()
print(f"\nglobal MSLP at t={T0}: n={pf.size}, "
      f"range [{lo/100:.1f}, {hi/100:.1f}] hPa, "
      f"median {np.median(pf)/100:.1f} hPa")
print(f"  a UNIFORM byte over this range = {(hi-lo)/PALETTE:.1f} Pa/level")

bands = {
    "storm tail (bottom 1%)": (0.0, 0.01),
    "lower shoulder (1-10%)": (0.01, 0.10),
    "bulk (40-60%)": (0.40, 0.60),
    "high tail (top 1%)": (0.99, 1.0),
}

codes = {}
recons = {}
for name, fn in (("A_affine", lambda v: reg_a_affine(v, lo, hi)),
                 ("B_rank", lambda v: reg_b_rank(v, ref)),
                 ("C_fisher_rank", lambda v: reg_c_fisher_rank(v, ref))):
    c, rc = fn(pf)
    codes[name], recons[name] = c, rc

print("\n=== reconstruction error in Pa, by percentile band (1 byte each) ===")
print(f"{'band':<26} {'n':>8} {'A affine':>12} {'B rank':>12} {'C fisher':>12}")
out_bands = {}
for name, (qa, qb) in bands.items():
    valo, vahi = np.quantile(pf, qa), np.quantile(pf, qb)
    m = (pf >= valo) & (pf <= vahi)
    row = {}
    for reg in codes:
        row[reg] = float(np.sqrt(np.mean((recons[reg][m] - pf[m]) ** 2)))
    out_bands[name] = {"n": int(m.sum()), **row}
    print(f"{name:<26} {int(m.sum()):>8} {row['A_affine']:>12.2f} "
          f"{row['B_rank']:>12.2f} {row['C_fisher_rank']:>12.2f}")

overall = {reg: float(np.sqrt(np.mean((recons[reg] - pf) ** 2))) for reg in codes}
print(f"{'OVERALL':<26} {pf.size:>8} {overall['A_affine']:>12.2f} "
      f"{overall['B_rank']:>12.2f} {overall['C_fisher_rank']:>12.2f}")

st = out_bands["storm tail (bottom 1%)"]
bulk = out_bands["bulk (40-60%)"]
a_flat = max(v[r] for v in out_bands.values() for r in ["A_affine"]) / \
    min(v["A_affine"] for v in out_bands.values())
print("\n=== pre-registered verdicts ===")
print(f"R1 A flat across bands (max/min ratio {a_flat:.2f}, bar <= 1.5): "
      f"{'PASS' if a_flat <= 1.5 else 'FAIL'}")
print(f"R2 B beats A in the bulk ({bulk['B_rank']:.2f} < {bulk['A_affine']:.2f}): "
      f"{'PASS' if bulk['B_rank'] < bulk['A_affine'] else 'FAIL'}")
print(f"R3 B worse than A in the storm tail "
      f"({st['B_rank']:.2f} > {st['A_affine']:.2f}): "
      f"{'PASS' if st['B_rank'] > st['A_affine'] else 'FAIL'}")
print(f"R4 C beats B in the storm tail "
      f"({st['C_fisher_rank']:.2f} < {st['B_rank']:.2f}): "
      f"{'PASS' if st['C_fisher_rank'] < st['B_rank'] else 'FAIL'}"
      f"   [{st['B_rank']/max(st['C_fisher_rank'],1e-9):.1f}x tighter]")

# ---- R5: cross-variable comparability --------------------------------------
print("\n=== R5 cross-variable: does one u8 mean the same rarity everywhere? ===")
fields = {"mslp": pf, "t2m": t2m.ravel(), "wind10m": wind.ravel()}
rank_codes = {}
for fname, fv in fields.items():
    fref = np.sort(fv)
    rr = _ranks(fv, fref)
    rank_codes[fname] = np.clip(np.floor(rr * PALETTE), 0, PALETTE - 1).astype(np.uint8)

# EVERY byte decides the verdict, not a sample of five. The first version of
# this bar probed [8, 64, 128, 192, 248] only (CodeRabbit, PR #926): a
# 5-of-256 sample cannot support a claim quantified over all 256, and the
# worst byte is precisely the one a sparse probe is most likely to miss.
# Five representative rows are still PRINTED, but the PASS/FAIL is computed
# over the full sweep.
all_spreads = []
r5_rows = []
for b in range(PALETTE):
    fr = {f: float((rank_codes[f] <= b).mean()) for f in fields}
    spread = max(fr.values()) - min(fr.values())
    all_spreads.append(spread)
    r5_rows.append({"byte": b, **fr, "spread": spread})
max_spread = max(all_spreads)
argworst = int(np.argmax(all_spreads))
print(f"  {'byte':>6} " + " ".join(f"{f:>12}" for f in fields) + "   spread")
for b in [8, 64, 128, 192, 248, argworst]:
    r = r5_rows[b]
    tag = "  <- WORST of all 256" if b == argworst else ""
    print(f"  {b:>6} " + " ".join(f"{r[f]:>12.4f}" for f in fields)
          + f"   {r['spread']:.5f}{tag}")
one_bucket = 1.0 / PALETTE
print(f"\nR5 max spread over ALL {PALETTE} bytes {max_spread:.5f} "
      f"(worst at byte {argworst}) vs one bucket {one_bucket:.5f}: "
      f"{'PASS' if max_spread <= one_bucket else 'FAIL'}")
print("  ABSOLUTE register: the same comparison is UNDEFINED — Pa, K and m/s")
print("  share no unit, so byte 128 of each denotes no common quantity at all.")

# Provenance: pin what the store actually served, so a future re-run can tell
# "the numbers moved" from "the store moved" (CodeRabbit, PR #926).
tattrs = json.loads(op.open(B + "/time/.zattrs", timeout=90).read())
pz = meta["mean_sea_level_pressure/.zarray"]
prov = {"time_units": tattrs.get("units"),
        "time_calendar": tattrs.get("calendar"),
        "grid_shape": list(pz["shape"][1:]),
        "chunk_shape": list(pz["chunks"]),
        "dtype": pz["dtype"],
        "compressor": (pz.get("compressor") or {}).get("id"),
        "n_points_per_field": int(pf.size)}
print("\nprovenance: " + ", ".join(f"{k}={v}" for k, v in prov.items()))

json.dump({"store": B, "t0": T0, "palette": PALETTE, "provenance": prov,
           "global_range_pa": [float(lo), float(hi)],
           "uniform_step_pa": float((hi - lo) / PALETTE),
           "bands": out_bands, "overall": overall,
           "R5_cross_variable": r5_rows,
           "R5_max_spread": float(max_spread),
           "R5_one_bucket": one_bucket},
          open(pathlib.Path(__file__).with_name("three_register_probe.json"), "w"), indent=2)
print("\nwrote three_register_probe.json")
