# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Palette deskew: does a calibrated per-code correction delta[q] improve
softmax fidelity against the TRUE pre-quantization values?

Representatives tested per Palette256 bin (256 equal-probability bins of a
calibration sample): midpoint, conditional mean, conditional median, and
log-mean-exp  lme_b[q] = (1/b) log E[exp(b t) | q], the value whose
exponential is the bin's expected softmax weight. Each is a 256-entry table,
so any of them costs the same at run time. Reference: softmax over the true
held-out values (f64 with max subtraction and math.fsum), which is ~1e-16
accurate, far below the quantization effects measured here."""
import json, math, sys
import numpy as np
import mpmath as mp
EG = 0.5772156649015329  # std::f64::consts::EULER_GAMMA

SRC = {"bounded_skew": lambda rng, n: 2 * rng.beta(2.0, 5.0, n) - 1,   # cosine-like, skewed
       "heavy_tail": lambda rng, n: rng.lognormal(0.0, 0.8, n) - 1.0}
source = None

def calibrate(seed, betas):
    t = np.sort(source(np.random.default_rng(seed), 400_000))
    edges = np.quantile(t, np.linspace(0, 1, 257)); edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, 255)
    lo = np.quantile(t, (np.arange(256)) / 256); hi = np.quantile(t, (np.arange(256) + 1) / 256)
    reps = {"midpoint": (lo + hi) / 2, "cond_mean": np.array([t[idx == k].mean() for k in range(256)]),
            "cond_median": np.array([np.median(t[idx == k]) for k in range(256)])}
    for b in betas:
        reps[f"lme_b{b:g}"] = np.array([math.log(np.mean(np.exp(b * (t[idx == k] - t[idx == k].max())))) / b + t[idx == k].max() for k in range(256)])
    return edges, reps

def softmax(x, b):
    m = max(x); w = [math.exp(b * (v - m)) for v in x]; s = math.fsum(w); return [v / s for v in w]

def evaluate(edges, reps, b, seed, n, rows, enc_edges=None):
    rng = np.random.default_rng(seed); out = {k: dict(tv=[], ymax=[]) for k in reps}
    for _ in range(rows):
        t = source(rng, n); V = rng.normal(0, 1, n)
        q = np.clip(np.searchsorted(enc_edges if enc_edges is not None else edges, t, side="right") - 1, 0, 255)
        pt = softmax(list(t), b); yt = math.fsum(p * v for p, v in zip(pt, V))
        for k, r in reps.items():
            if k.startswith("lme") and k != f"lme_b{b:g}": continue
            pq = softmax(list(r[q]), b)
            out[k]["tv"].append(0.5 * sum(abs(a - c) for a, c in zip(pq, pt)))
            out[k]["ymax"].append(abs(math.fsum(p * v for p, v in zip(pq, V)) - yt))
            c = max(t)  # common shift; the bias ratio is shift-invariant
            out[k].setdefault("sbias", []).append(math.fsum(math.exp(b * (v - c)) for v in r[q]) / math.fsum(math.exp(b * (v - c)) for v in t) - 1)
    return {k: dict(mean_tv=float(np.mean(v["tv"])), mean_y_err=float(np.mean(v["ymax"])), mean_S_bias=float(np.mean(v["sbias"])), mean_abs_S_bias=float(np.mean(np.abs(v["sbias"])))) for k, v in out.items() if v["tv"]}

R = {}
betas = [1.0, 4.0]
for sname, fn in SRC.items():
    source = fn
    edges, reps = calibrate(1, betas)
    for b in betas:
        for n, rows in [(16, 400), (256, 100), (4096, 20)]:
            R[f"{sname}_b{b:g}_n{n}"] = evaluate(edges, reps, b, 2, n, rows)
source = SRC["bounded_skew"]; edges, reps = calibrate(1, betas)
# Codebook version mismatch: tables from calibration v1 applied to codes of v2.
edges2, reps2 = calibrate(3, betas)
R["version_mismatch_b4_n256"] = {
    "v2_codes_v2_tables": evaluate(edges2, {"lme_b4": reps2["lme_b4"]}, 4.0, 2, 256, 100)["lme_b4"],
    "v2_codes_v1_tables": evaluate(edges, {"lme_b4": reps["lme_b4"]}, 4.0, 2, 256, 100, enc_edges=edges2)["lme_b4"],
}
# Euler-gamma: a common additive offset cancels; a code-dependent one does not.
mp.mp.dps = 40
x = [mp.mpf(float(v)) for v in reps["cond_mean"][[100, 110, 120, 130]]]
def mpsm(xs, b=mp.mpf(4)):
    m = max(xs); w = [mp.e ** (b * (v - m)) for v in xs]; s = mp.fsum(w); return [v / s for v in w]
base = mpsm(x)
R["euler_gamma_common_offset_max_abs_change"] = float(max(abs(a - c) for a, c in zip(base, mpsm([v + EG for v in x]))))
R["euler_gamma_code_dependent_offset_max_abs_change"] = float(max(abs(a - c) for a, c in zip(base, mpsm([v + EG * i / 3 for i, v in enumerate(x)]))))
json.dump(R, open(sys.argv[1], "w"), indent=1)

