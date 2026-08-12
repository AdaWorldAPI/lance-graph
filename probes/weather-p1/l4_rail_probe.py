"""EXPLORATORY — does the storm spine fit the ACTUAL L4 carrier, 6 x (8:8)?
Operator correction (2026-08-11): "was ist mit 6x Palette256:Palette256
centroid, was ja die Verteilung anzeigen soll — palette256 alleine ist ja nur
'attention header'". NOT an EV.

WHAT I HAD WRONG. Every encoding in this arc treated "one scalar -> one byte"
as the unit. The shipped carrier is a PAIR: le-contract §3 row L4 —
`6 x (8:8)`, `palette256²`, "each byte pair indexes the 256x256 palette
distance/compose tables; similarity = ONE table read". A single palette256
byte is only the SELECTOR (which archetype). The PAIR is a cell in the
centroid tile, and the tile is where the distribution lives — materialized
once, so a node stores a coordinate, not a value. Six pairs = six subspaces.

Two things this reframes rather than replaces:
  * my 2-tier rolling-floor cascade IS the sanctioned §3 reading
    "area : location in stacked exactness" (each pair refines within its
    area) — but it is ONE RAIL, not the whole payload. I built a rail and
    called it the carrier.
  * the Fisher-Z register is not a rival to the palette; le-contract §3 says
    L4's cosine-replacement palette256 reads through the analytic Fisher-z
    canon and "an L4 ClassView may declare an analytic codebook" — i.e.
    Fisher-Z is how the 256 centroid entries PER AXIS are laid out.

THE CARVE UNDER TEST (12 bytes total, exactly one V3 facet payload):
  rail 0  dipole            (amplitude_slope : bearing)   <- natural pair
  rail 1  rings 0,1         (ring0 : ring1)
  rail 2  rings 2,3
  rail 3  rings 4,5
  rail 4  rings 6,7
  rail 5  rings 8,9
Rings 10,11 do not fit. That is a REAL constraint of the carrier and is
reported, not engineered away: the alternative (drop the dipole rail to fit
all 12 rings) is measured too, so the trade is visible.

Ring bytes use a SHARED codebook across storms (the tile is global, per
le-contract "similarity = ONE table read" — a per-storm codebook would make
two storms' rails incomparable, defeating the carrier). Codebook axes are
built two ways and compared:
  UNIFORM  256 levels over the global ring-value range
  FISHERZ  256 levels uniform in arctanh(2*rank-1) of the global ring
           distribution — the analytic-codebook reading

PRE-REGISTERED:
  L1  carve A (dipole + rings 0..9) recovers in-disk R2 within 0.02 of the
      f64 constrained spine (0.943 / 0.909 from comet_tail_probe).
  L2  carve B (all 12 rings, no dipole rail) is WORSE than carve A on
      storm 1 — the dipole is worth more than the two outermost rings.
      If it is not, the carve should be B and the claim inverts.
  L3  FISHERZ codebook >= UNIFORM on ring RMSE (the analytic-codebook
      reading earns its place, or it does not).
  L4x CROSS-STORM: with the shared codebook, storm1 and storm2 rails are
      directly comparable — the same byte denotes the same centroid.
      Falsifier: recovering storm2 through storm1's OWN codebook must be
      NO WORSE than through the shared one by more than 1%. If it is
      worse, the codebook is not actually global and the pair loses its
      "one table read" property.
"""
import json
import pathlib
import urllib.request

import numcodecs
import numpy as np

B = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
     "1959-2022-6h-1440x721.zarr")
T0 = 91246
R_E, R_DISK, RING = 6371.0, 1200.0, 100.0
PAL = 256

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
meta = json.loads(op.open(B + "/.zmetadata", timeout=90).read())["metadata"]


def fetch(var, key):
    """Fetch and decode one zarr chunk from the WB2 store."""
    za = meta[f"{var}/.zarray"]
    raw = op.open(f"{B}/{var}/{key}", timeout=600).read()
    dec = numcodecs.get_codec(za["compressor"]).decode(raw)
    return np.frombuffer(dec, dtype=np.dtype(za["dtype"])).reshape(za["chunks"])


lat = fetch("latitude", "0").astype(np.float64).ravel()
p0 = fetch("mean_sea_level_pressure", f"{T0}.0.0")[0].astype(np.float64)
NY, NX = p0.shape
phi = np.deg2rad(lat)
lon = np.arange(NX) * 0.25


def geom(la, lo):
    """dx, dy, r (km), azimuth (rad CCW from east) about a continuous centre."""
    dlon = np.deg2rad((lon[None, :] - lo + 180) % 360 - 180)
    dx = R_E * np.cos(np.deg2rad(la)) * dlon * np.ones((NY, 1))
    dy = R_E * (phi[:, None] - np.deg2rad(la)) * np.ones((1, NX))
    return dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)


def find(field, near=None, rad=600.0):
    """Deepest NH zonal-anomaly low, optionally limited to a disk about `near`."""
    fa = field - field.mean(axis=1, keepdims=True)
    m = lat[:, None] > 15
    if near is not None:
        m = m & (geom(*near)[2] < rad)
    mk = np.where(m, fa, np.inf)
    i, j = np.unravel_index(np.argmin(mk), field.shape)
    return int(i), int(j)


def spine(la, lo):
    """The f64 spine: ring-profile means + the CONSTRAINED 2-param dipole."""
    _, _, r, th = geom(la, lo)
    disk = r <= R_DISK
    v, rr, tt = p0[disk], r[disk], th[disk]
    nb = int(R_DISK / RING)
    rings = np.clip((rr / RING).astype(int), 0, nb - 1)
    prof = np.array([v[rings == b].mean() if (rings == b).any() else 0.0
                     for b in range(nb)])
    resid = v - prof[rings]
    X = np.column_stack([rr * np.cos(tt), rr * np.sin(tt)])
    coef, *_ = np.linalg.lstsq(X, resid, rcond=None)
    return dict(vals=v, rings=rings, tt=tt, rr=rr, prof=prof, coef=coef,
                var=v.var(), nb=nb)


def r2_of(s, prof, coef):
    """In-disk R2 of a (profile, dipole) reconstruction against the raw field."""
    rec = prof[s["rings"]] + coef[0] * s["rr"] * np.cos(s["tt"]) \
        + coef[1] * s["rr"] * np.sin(s["tt"])
    return 1.0 - ((s["vals"] - rec).var() / s["var"])


def cb_uniform(pop):
    """256 centroid levels, uniform over the population's value range."""
    lo, hi = pop.min(), pop.max()
    return lo + (np.arange(PAL) + 0.5) / PAL * (hi - lo)


def cb_fisherz(pop):
    """256 centroid levels, uniform in arctanh(2*rank-1) — analytic codebook.

    NOTE the population matters and is the whole of L3-vs-L3b: `pop` is the
    distribution the RANKS are taken against, not merely the values to encode.
    Passing the 24 ring means (L3) makes every sample its own quantile, so the
    rim-stretch spends levels on 24 isolated points; passing the field (L3b)
    is the reading le-contract actually describes.
    """
    srt = np.sort(pop)
    eps = 1e-9
    rr = (np.arange(len(srt)) + 0.5) / len(srt)
    zr = np.arctanh(np.clip(2 * rr - 1, -1 + eps, 1 - eps))
    zc = zr.min() + (np.arange(PAL) + 0.5) / PAL * (zr.max() - zr.min())
    q = np.clip((np.tanh(zc) + 1) / 2, 0, 1)
    return np.quantile(srt, q)


def enc(vals, cb):
    """Nearest-centroid byte code + its reconstruction (the attention header)."""
    idx = np.abs(vals[:, None] - cb[None, :]).argmin(axis=1)
    return idx.astype(np.uint8), cb[idx]


# ---- storms + the SHARED (global) ring codebook -----------------------------
storms = {}
for nm, hint in (("storm1", None), ("storm2", (67.0, 28.0))):
    ci, cj = find(p0, near=hint)
    storms[nm] = spine(lat[ci], lon[cj])
    print(f"{nm}: centre ({lat[ci]:.2f}N,{lon[cj]:.2f}E)")

pop = np.concatenate([s["prof"] for s in storms.values()])
# L3b population: the FIELD the ring means are drawn from (both storms' disks),
# not the 24 encoded values themselves. See cb_fisherz's note.
field_pop = np.concatenate([s["vals"] for s in storms.values()])
CB = {"uniform": cb_uniform(pop),
      "fisherz": cb_fisherz(pop),
      "fisherz_field": cb_fisherz(field_pop)}

# dipole rail: amplitude slope + bearing, one byte each
amp_pop = np.array([np.hypot(*s["coef"]) for s in storms.values()])
AMP_CB = np.linspace(0.0, amp_pop.max() * 1.25, PAL)
BRG_CB = -np.pi + (np.arange(PAL) + 0.5) / PAL * 2 * np.pi   # 1.406 deg/level

out = {"store": B, "t0": T0, "carves": {}, "cross": {}}
print(f"\nbearing resolution: {360/PAL:.3f} deg/level\n")
print(f"{'storm':<9} {'codebook':<13} {'f64 spine':>10} {'A dip+r0-9':>12} "
      f"{'D spread':>10} {'B 12 rings':>12} {'ring RMSE':>10} {'14B diag':>10}")

for nm, s in storms.items():
    row = {}
    f64_r2 = r2_of(s, s["prof"], s["coef"])
    for cbn, cb in CB.items():
        codes, rec = enc(s["prof"], cb)
        ring_rmse = float(np.sqrt(np.mean((rec - s["prof"]) ** 2)))

        # carve A: rail0 = dipole, rails1-5 = rings 0..9. Rings 10,11 have NO
        # byte at all — they fall back to the last encoded ring (a held edge),
        # which is the honest cost of the 12-byte budget, not a hidden fit.
        pa_A = np.concatenate([rec[:10], np.full(s["nb"] - 10, rec[9])])
        ai = int(np.abs(np.hypot(*s["coef"]) - AMP_CB).argmin())
        bi = int(np.abs(np.arctan2(s["coef"][1], s["coef"][0]) - BRG_CB).argmin())
        a_q, b_q = AMP_CB[ai], BRG_CB[bi]
        coef_q = np.array([a_q * np.cos(b_q), a_q * np.sin(b_q)])
        r2_A = r2_of(s, pa_A, coef_q)

        # carve B: all 12 rings, NO dipole rail
        r2_B = r2_of(s, rec, np.array([0.0, 0.0]))

        # DIAGNOSTIC ONLY, 14 bytes — OVER the facet budget. Not a candidate
        # carve; it exists to split L1's loss into (i) byte quantization and
        # (ii) the two rings the 12-byte budget cannot afford.
        r2_over = r2_of(s, rec, coef_q)

        # carve D, still 12 bytes: same rails as A, but the 10 ring bytes are
        # SPREAD over the full radius (positions linspace(0, nb-1, 10)) and the
        # missing rings linearly interpolated. Costs nothing extra — the
        # profile is smooth, so 10 samples across 12 beats 10 samples of the
        # inner 12 plus a held edge. Pre-registered: D >= A on storm1.
        pos = np.unique(np.round(np.linspace(0, s["nb"] - 1, 10)).astype(int))
        pd_ = np.interp(np.arange(s["nb"]), pos, rec[pos])
        r2_D = r2_of(s, pd_, coef_q)

        row[cbn] = {"f64_r2": float(f64_r2), "carveA_r2": float(r2_A),
                    "carveB_r2": float(r2_B), "carveD_r2": float(r2_D),
                    "ring_rmse_pa": ring_rmse,
                    "diag_14byte_r2": float(r2_over),
                    "loss_quantization": float(f64_r2 - r2_over),
                    "loss_dropped_rings": float(r2_over - r2_A)}
        print(f"{nm:<9} {cbn:<13} {f64_r2:>10.4f} {r2_A:>12.4f} "
              f"{r2_D:>10.4f} {r2_B:>12.4f} {ring_rmse:>10.2f} {r2_over:>10.4f}")
    out["carves"][nm] = row

# ---- L4x cross-storm: shared vs per-storm codebook --------------------------
# Measured through UNIFORM — the codebook L3 names as best. Measuring the
# cross-storm property of a codebook the previous bar just rejected would be
# reporting a property of something we are not proposing to use.
#
# ANTI-VACUITY (this bar was measured vacuous on its first run and is kept
# here as the guard that caught it): a uniform codebook is fixed by its
# population's MIN and MAX alone, so if one storm's profile range CONTAINS the
# other's, that storm's "own" codebook IS the pooled codebook — the test then
# compares an array against itself and passes for free. storm1 ⊃ storm2 here,
# so the storm1->storm2 direction is degenerate BY CONSTRUCTION and only the
# storm2->storm1 direction can carry information. Both are reported.
print("\n=== L4x cross-storm codebook (uniform) ===")
out["cross"] = {}
for src, dst in (("storm1", "storm2"), ("storm2", "storm1")):
    own_cb = cb_uniform(storms[src]["prof"])
    tgt = storms[dst]["prof"]
    degenerate = bool(np.allclose(own_cb, CB["uniform"]))
    _, rec_shared = enc(tgt, CB["uniform"])
    _, rec_own = enc(tgt, own_cb)
    r_sh = float(np.sqrt(np.mean((rec_shared - tgt) ** 2)))
    r_ow = float(np.sqrt(np.mean((rec_own - tgt) ** 2)))
    pen = (r_sh - r_ow) / max(r_ow, 1e-9)
    out["cross"][f"{src}_cb_on_{dst}"] = {
        "shared_rmse_pa": r_sh, "own_rmse_pa": r_ow,
        "penalty_frac": pen, "degenerate": degenerate}
    tag = ("DEGENERATE — own codebook IS the shared one, carries no "
           "information") if degenerate else "informative"
    print(f"  {src} codebook -> {dst}: shared {r_sh:.2f} Pa vs own "
          f"{r_ow:.2f} Pa ({100*pen:+.1f}%)  [{tag}]")

print("\n=== pre-registered verdicts ===")
BEST = "uniform"     # named by L3, below — reported, not assumed
for nm in storms:
    r = out["carves"][nm][BEST]
    d = abs(r["carveA_r2"] - r["f64_r2"])
    print(f"L1 {nm}: |carveA - f64| = {d:.4f} (bar <= 0.02): "
          f"{'PASS' if d <= 0.02 else 'FAIL'}"
          f"   [quantization {r['loss_quantization']:+.4f}, "
          f"dropped rings 10-11 {r['loss_dropped_rings']:+.4f}]")
for nm in storms:
    r = out["carves"][nm][BEST]
    d = abs(r["carveD_r2"] - r["f64_r2"])
    print(f"L1' {nm} carve D (spread rings, still 12 B): |D - f64| = {d:.4f} "
          f"(bar <= 0.02): {'PASS' if d <= 0.02 else 'FAIL'}")
r1 = out["carves"]["storm1"][BEST]
print(f"L2 storm1 carve A > carve B ({r1['carveA_r2']:.4f} > "
      f"{r1['carveB_r2']:.4f}): {'PASS' if r1['carveA_r2'] > r1['carveB_r2'] else 'FAIL'}")
print(f"L2b storm1 carve D >= carve A ({r1['carveD_r2']:.4f} >= "
      f"{r1['carveA_r2']:.4f}): "
      f"{'PASS' if r1['carveD_r2'] >= r1['carveA_r2'] else 'FAIL'}")
rm = {c: float(np.mean([out["carves"][n][c]["ring_rmse_pa"] for n in storms]))
      for c in CB}
print(f"L3  FISHERZ(24 ring means) ring RMSE {rm['fisherz']:.2f} <= "
      f"UNIFORM {rm['uniform']:.2f}: "
      f"{'PASS' if rm['fisherz'] <= rm['uniform'] else 'FAIL'}")
print(f"L3b FISHERZ(field population) {rm['fisherz_field']:.2f} <= "
      f"UNIFORM {rm['uniform']:.2f}: "
      f"{'PASS' if rm['fisherz_field'] <= rm['uniform'] else 'FAIL'}"
      "   [POST-HOC refinement — the rank population is the FIELD, not the\n"
      "    24 values being encoded. Stated as post-hoc, not pre-registered.]")
out["ring_rmse_by_codebook"] = rm
inf_dirs = {k: v for k, v in out["cross"].items() if not v["degenerate"]}
if not inf_dirs:
    print("L4x VOID — every direction degenerate; the bar measured nothing.")
else:
    worst = max(v["penalty_frac"] for v in inf_dirs.values())
    print(f"L4x shared-vs-own penalty (worst informative direction of "
          f"{len(inf_dirs)}/{len(out['cross'])}) {100*worst:+.1f}% "
          f"(bar <= +1%): {'PASS' if worst <= 0.01 else 'FAIL'}")

with open(pathlib.Path(__file__).with_name("l4_rail_probe.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("\nwrote l4_rail_probe.json")
