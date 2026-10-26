# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 5 driver: generates cases, runs the Rust arms (ndarray FMA),
and scores them against an independent mpmath reference (40 digits) computed
from the same f64 codebook representatives. The reference never sees an F24
table, a difference table or an f64 shortcut."""
import json, subprocess, sys
from fractions import Fraction
import numpy as np, mpmath as mp
mp.mp.dps = 40
BIN = sys.argv[1]; OUT = sys.argv[2]
rng = np.random.default_rng(5)
hx = lambda v: format(np.float64(v).view(np.uint64), "x")
fromhx = lambda s: float(np.uint64(int(s, 16)).view(np.float64))

def codebooks():
    i = np.arange(256)
    nq = np.sort(rng.normal(0, 1, 200000)); ln = np.sort(np.exp(rng.normal(0, 0.8, 200000)))
    qs = (i + 0.5) / 256
    return {"affine_s0.0084": -1.0 + 0.0084 * i, "affine_s0.039": -5.0 + 0.039 * i,
            "nonuniform_normal": np.quantile(nq, qs), "nonuniform_lognormal": np.quantile(ln, qs)}

def rows(n):
    yield "uniform", rng.integers(0, 256, n)
    s = np.zeros(n, int); s[n // 2] = 255; yield "spike_vs_floor", s
    yield "near_max", rng.integers(240, 256, n)
    yield "low_row", rng.integers(0, 32, n)  # row max far below the codebook max
    yield "skewed", np.minimum(255, rng.geometric(0.02, n) - 1)[::-1].copy()

cases = []
for kname, x in codebooks().items():
    for beta in ([1.0, 8.0, 32.0] if kname == "affine_s0.039" else [1.0, 8.0]):
        for n in [2, 3, 16, 256, 4096]:
            for dname, q in rows(n):
                cases.append((kname, beta, dname, x, q))
        for t in [511, 512, 513, 4096]:
            cases.append((kname, beta, f"ties{t}", x, np.full(t, 200)))
D = 5  # cols 0-3 integer (A applies); col 4 real-valued (A excluded)
DI = 4
def vrow(n):
    vi = rng.integers(-127, 128, (n, D)); vi[:, 3] = np.where(np.arange(n) % 2 == 0, 127, -127)  # cancellation column
    vf = vi.astype(float); vf[:, 4] = rng.normal(0, 50, n) * (1 + 2.0**-30); vi[:, 4] = 0
    return vi, vf
lines, meta = [], {}
for cid, (kname, beta, dname, x, q) in enumerate(cases):
    vi, vf = vrow(len(q)); meta[cid] = (kname, beta, dname, x, q, vi, vf)
    lines += [f"{cid} {hx(beta)} {len(q)} {D}", " ".join(hx(v) for v in x), " ".join(map(str, q)),
              " ".join(hx(float(v)) for v in vf.ravel()), " ".join(map(str, vi.ravel()))]
raw = subprocess.run([BIN], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
arms = {}
for l in raw.splitlines():
    t = l.split(); arms.setdefault(int(t[1]), {})[t[0]] = t[2:]

def code_p(p):  # TFLM-style int8 probability, scale 1/256, zp -128, ties away (p >= 0)
    return max(-128, min(127, int(mp.floor(p * 256 + mp.mpf("0.5"))) - 128))
def code_y(y):  # integer output code for y, ties away from zero
    r = int(mp.floor(abs(y) + mp.mpf("0.5"))) * (1 if y >= 0 else -1)
    return max(-128, min(127, r))

res = []
for cid, (kname, beta, dname, x, q, vi, vf) in meta.items():
    b = mp.mpf(beta); xs = [mp.mpf(float(x[k])) for k in q]; xm = max(xs)
    w = [mp.e ** (b * (xi - xm)) for xi in xs]; S = mp.fsum(w)
    pref = [wi / S for wi in w]
    yref = [mp.fsum(w[i] * mp.mpf(float(vf[i, k])) for i in range(len(q))) / S for k in range(D)]
    pc_ref = [code_p(p) for p in pref]; yc_ref = [code_y(y) for y in yref]
    a = arms[cid]; r = dict(codebook=kname, beta=beta, dist=dname, n=len(q))
    # A: exact rationals from the integers the Rust arm returned.
    Sa = int(a["A"][0]); Na = [int(v) for v in a["A"][1:]]
    pt = None
    # A's weights: recompute W_i/S exactly only through the returned S and per-element
    # codes would need W; derive p codes from B's p (exact W as f64 / S) instead:
    def score_p(tag, ps):
        err = max(abs(mp.mpf(ps[i]) - pref[i]) for i in range(len(q)))
        rel = max(abs(mp.mpf(ps[i]) - pref[i]) / pref[i] for i in range(len(q)) if pref[i] > 0)
        mism = sum(1 for i in range(len(q)) if code_p(mp.mpf(ps[i])) != pc_ref[i])
        s1 = abs(float(np.sum(np.array(ps, dtype=np.float64))) - 1.0)
        r[f"{tag}_p_abs"] = float(err); r[f"{tag}_p_rel"] = float(rel); r[f"{tag}_p_code_mism"] = mism; r[f"{tag}_sum1_dev"] = s1
    def score_y(tag, ys):
        r[f"{tag}_y_abs"] = float(max(abs(mp.mpf(ys[k]) - yref[k]) for k in range(D)))
        r[f"{tag}_y_code_mism"] = sum(1 for k in range(D) if code_y(mp.mpf(ys[k])) != yc_ref[k])
        r[f"{tag}_y_mism_margin"] = [float(abs(abs(yref[k]) - mp.floor(abs(yref[k])) - mp.mpf("0.5"))) for k in range(D) if code_y(mp.mpf(ys[k])) != yc_ref[k]]
    ya = [Fraction(Na[k], Sa) for k in range(DI)]
    r["A_y_abs"] = float(max(abs(mp.mpf(ya[k].numerator) / ya[k].denominator - yref[k]) for k in range(DI)))
    r["A_y_code_mism"] = sum(1 for k in range(DI) if code_y(mp.mpf(ya[k].numerator) / ya[k].denominator) != yc_ref[k])
    r["A_y_mism_margin"] = [float(abs(abs(yref[k]) - mp.floor(abs(yref[k])) - mp.mpf("0.5"))) for k in range(DI) if code_y(mp.mpf(ya[k].numerator) / ya[k].denominator) != yc_ref[k]]
    r["A_S_bits"] = Sa.bit_length(); r["A_N_bits"] = max(abs(v).bit_length() for v in Na)
    # A's materialized p = W_i / S exactly; W_i = PB_i * S is exact (W < 2^53, S exact sum < 2^53 here)
    pb = [fromhx(s) for s in a["PB"]]
    W = [round(pi * Sa) for pi in pb]
    assert sum(W) == Sa, (cid, sum(W), Sa)
    pa = [mp.mpf(Wi) / Sa for Wi in W]
    r["A_p_abs"] = float(max(abs(pa[i] - pref[i]) for i in range(len(q))))
    r["A_p_code_mism"] = sum(1 for i in range(len(q)) if code_p(pa[i]) != pc_ref[i])
    r["A_zero_weights"] = sum(1 for Wi in W if Wi == 0)
    for tag, ptag in [("B", "PB"), ("U", "PU"), ("C", "PC")]:
        score_p(tag, [fromhx(s) for s in a[ptag]])
    for tag in ["B", "M", "U", "C"]:
        score_y(tag, [fromhx(s) for s in a[tag]])
    r["B_vs_M_y_bits_differ"] = sum(1 for k in range(D) if a["B"][k] != a["M"][k])
    r["B_vs_M_y_bits_differ_realcol"] = int(a["B"][4] != a["M"][4])
    r["B_realcol_abs"] = float(abs(mp.mpf(fromhx(a["B"][4])) - yref[4])); r["M_realcol_abs"] = float(abs(mp.mpf(fromhx(a["M"][4])) - yref[4]))
    r["ref_p_ties_at_half"] = sum(1 for p in pref if (p * 256 - mp.floor(p * 256)) == mp.mpf("0.5"))
    res.append(r)
json.dump(res, open(OUT, "w"), indent=0)
