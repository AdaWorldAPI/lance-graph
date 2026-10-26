# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6: certified F24 error bound + conditional escalation.

Same 261 cases as driver.py (identical generator and seed). Per case:
  1. Table generation: W = round_nearest(2^24 * exp(beta*(x_a - x_m)))
     evaluated in mpmath INTERVAL arithmetic; an entry is certified when the
     interval cannot straddle a .5 point, which proves |2^24 w - W| <= 1/2.
     The same entries from an ordinary f64 `math.exp` generator are counted
     where they disagree.
  2. Arm A in exact integers: S = sum W, N = sum W*V, y_hat = N/S (Fraction).
  3. Bound (integer units, eps_i = 1/2):  |y - y_hat| <= sum eps_i|V_i - y_hat| / (S - E),
     E = n/2;  |p_i - p_hat_i| <= (eps_i + p_hat_i*E) / (S - E).
  4. Output certified iff the code at both interval ends agrees (codes are
     monotone; ties round away from zero, the TFLM output rule).
  5. Otherwise escalate the row: one distinct representative -> exact
     rational; else mpmath interval evaluation at 64..512 bits.
Reference for scoring: mpmath at 40 digits (as driver.py), never a table."""
import hashlib, json, math, sys, time
from fractions import Fraction
import contextlib
import numpy as np, mpmath as mp
mp.mp.dps = 40

@contextlib.contextmanager
def ivprec(bits):
    # mp.workprec does NOT reach the interval context; mp.iv has its own prec.
    old = mp.iv.prec; mp.iv.prec = bits
    try: yield
    finally: mp.iv.prec = old

def frac(x):
    """Exact Fraction of an mpf or of a degenerate ivmpf endpoint (no rounding)."""
    if isinstance(x, mp.ctx_iv.ivmpf):
        lo, hi = x._mpi_; assert lo == hi; sign, man, exp, _ = lo
        v = Fraction(man) * (Fraction(2) ** exp) if man else Fraction(0)
        return -v if sign else v
    # NOT mpf.man_exp: it returns the magnitude only, so negative values lost
    # their sign (measured: 565 spurious y "misses" on the first run).
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * (Fraction(2) ** int(exp)) if man else Fraction(0)
    return -v if sign else v
OUT = sys.argv[1]
rng = np.random.default_rng(5)
F = 24

# --- identical generator to driver.py ---------------------------------------
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
    yield "low_row", rng.integers(0, 32, n)
    yield "skewed", np.minimum(255, rng.geometric(0.02, n) - 1)[::-1].copy()
CB = codebooks()
cases = []
for kname, x in CB.items():
    for beta in ([1.0, 8.0, 32.0] if kname == "affine_s0.039" else [1.0, 8.0]):
        for n in [2, 3, 16, 256, 4096]:
            for dname, q in rows(n):
                cases.append((kname, beta, dname, x, q))
        for t in [511, 512, 513, 4096]:
            cases.append((kname, beta, f"ties{t}", x, np.full(t, 200)))
D = 5
def vrow(n):
    vi = rng.integers(-127, 128, (n, D)); vi[:, 3] = np.where(np.arange(n) % 2 == 0, 127, -127)
    vf = vi.astype(float); vf[:, 4] = rng.normal(0, 50, n) * (1 + 2.0**-30); vi[:, 4] = 0
    return vi, vf
meta = [(c, vrow(len(c[4]))) for c in cases]

# --- prepared generation: identity, refusal, certified table -----------------
class Refused(Exception): pass
def generation(x, beta):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (256,) or not np.all(np.isfinite(x)): raise Refused("non-finite or wrong-size codebook")
    if not (math.isfinite(beta) and beta > 0): raise Refused("beta must be finite and positive")
    h = hashlib.sha256(x.tobytes() + np.float64(beta).tobytes() + bytes([F]) + b"rhe").hexdigest()[:16]
    return dict(x=x, beta=beta, digest=h, cache={})
def table_row(g, m):
    """Certified W[a | m] for all a with x_a <= x_m. Returns (W, f64_disagreements, uncertified)."""
    if m in g["cache"]: return g["cache"][m]
    W, dis, unc = {}, 0, 0
    with ivprec(96):
        for a in range(256):
            if g["x"][a] > g["x"][m]: continue
            iv = mp.iv.exp(mp.iv.mpf(g["beta"]) * (mp.iv.mpf(float(g["x"][a])) - mp.iv.mpf(float(g["x"][m])))) * (2 ** F)
            lo, hi = iv.a, iv.b
            wl, wh = math.floor(frac(lo) + Fraction(1, 2)), math.floor(frac(hi) + Fraction(1, 2))
            if wl != wh: unc += 1  # would need a tie rule; not hit (counted)
            W[a] = wl
            f64w = round(math.exp(g["beta"] * (float(g["x"][a]) - float(g["x"][m]))) * (1 << F))
            dis += f64w != wl
    g["cache"][m] = (W, dis, unc); return g["cache"][m]

def code_y(y):  # nearest integer, ties away from zero, clamp to int8
    r = math.floor(abs(y) + Fraction(1, 2)) * (1 if y >= 0 else -1)
    return max(-128, min(127, r))
def code_p(p):  # TFLM output: scale 1/256, zp -128, ties away (p >= 0)
    return max(-128, min(127, math.floor(p * 256 + Fraction(1, 2)) - 128))

def escalate(g, q, V, cols):
    """Resolve y (all cols) and p (all elements) for one row. Returns (y_codes, p_codes, bits, how)."""
    groups = {}
    for i, a in enumerate(q): groups.setdefault(int(a), []).append(i)
    n = len(q)
    if len(set(float(g["x"][a]) for a in groups)) == 1:  # one representative: exact rationals
        yc = [code_y(sum(Fraction(V[i][k]) for i in range(n)) / n) for k in cols]
        return yc, [code_p(Fraction(1, n))] * n, 0, "exact_rational"
    m = max(groups, key=lambda a: g["x"][a])
    # Exact-rational y (Lindemann-Weierstrass): with distinct rational exponents
    # r_j, sum_j c_j e^{r_j} = 0 forces every c_j = 0. So y_k equals a rational t
    # iff every DISTINCT-representative group has V-mean t; otherwise y_k is
    # irrational, never sits on a midpoint, and interval refinement terminates.
    # Without this check an exactly-rational y on a .5 point never certifies.
    byx = {}
    for i, a in enumerate(q): byx.setdefault(float(g["x"][a]), []).append(i)
    exact_y = {}
    for k in cols:
        means = {sum(Fraction(V[i][k]) for i in ix) / len(ix) for ix in byx.values()}
        if len(means) == 1: exact_y[k] = code_y(means.pop())
    # p_i is never rational with >= 2 distinct representatives (same argument).
    for bits in (64, 128, 256, 512):
        with ivprec(bits):
            e = {a: mp.iv.exp(mp.iv.mpf(g["beta"]) * (mp.iv.mpf(float(g["x"][a])) - mp.iv.mpf(float(g["x"][m])))) for a in groups}
            S = sum((e[a] * len(ix) for a, ix in groups.items()), mp.iv.mpf(0))
            ok, yc = True, []
            for k in cols:
                if k in exact_y: yc.append(exact_y[k]); continue
                N = sum((e[a] * mp.iv.mpf(sum(Fraction(V[i][k]) for i in ix).numerator) / sum(Fraction(V[i][k]) for i in ix).denominator
                         for a, ix in groups.items()), mp.iv.mpf(0))
                yv = N / S
                c0, c1 = code_y(frac(yv.a)), code_y(frac(yv.b))
                ok &= c0 == c1; yc.append(c0)
            pc = {}
            for a in groups:
                pv = e[a] / S
                c0, c1 = code_p(frac(pv.a)), code_p(frac(pv.b))
                ok &= c0 == c1; pc[a] = c0
            if ok:
                return yc, [pc[int(a)] for a in q], bits, "interval"
    return None, None, 512, "UNRESOLVED"

def ref_codes(x, beta, q, V, cols):
    b = mp.mpf(beta); xs = [mp.mpf(float(x[a])) for a in q]; xm = max(xs)
    w = [mp.e ** (b * (xi - xm)) for xi in xs]; S = mp.fsum(w)
    yc = [code_y(frac(mp.fsum(w[i] * mp.mpf(float(V[i][k])) for i in range(len(q))) / S)) for k in cols]
    return yc, [code_p(frac(wi / S)) for wi in w]

res, gens = [], {}
for (kname, beta, dname, x, q), (vi, vf) in meta:
    key = (kname, beta)
    if key not in gens: gens[key] = generation(x, beta)
    g = gens[key]
    m = int(max(set(int(a) for a in q), key=lambda a: g["x"][a]))
    W, dis, unc = table_row(g, m)
    n = len(q); Vf = [[float(v) for v in row] for row in vf]
    Wi = [W[int(a)] for a in q]
    S = sum(Wi); E = Fraction(n, 2)
    r = dict(codebook=kname, beta=beta, dist=dname, n=n, f64_table_disagreements=dis, uncertified_entries=unc)
    r["S_gt_E"] = S > E
    cols = list(range(D))
    t0 = time.perf_counter()
    yA, ycert, yhat_codes = [], [], []
    for k in cols:
        Nk = sum(Fraction(Wi[i]) * Fraction(Vf[i][k]) for i in range(n))
        yh = Nk / S
        B = sum(abs(Fraction(Vf[i][k]) - yh) for i in range(n)) / 2 / (S - E) if S > E else None
        yhat_codes.append(code_y(yh))
        ycert.append(B is not None and code_y(yh - B) == code_y(yh + B))
        r.setdefault("y_bound_max", 0.0); r["y_bound_max"] = max(r["y_bound_max"], float(B) if B is not None else float("inf"))
    pcert, phat_codes = [], []
    for i in range(n):
        ph = Fraction(Wi[i], S)
        Bp = (Fraction(1, 2) + ph * E) / (S - E) if S > E else None
        phat_codes.append(code_p(ph))
        pcert.append(Bp is not None and code_p(max(Fraction(0), ph - Bp)) == code_p(min(Fraction(1), ph + Bp)))
    t_base = time.perf_counter() - t0
    yref, pref = ref_codes(g["x"], beta, q, Vf, cols)
    r["y_outputs"] = len(cols); r["p_outputs"] = n
    r["y_certified"] = sum(ycert); r["p_certified"] = sum(pcert)
    r["y_misses"] = sum(1 for k in range(len(cols)) if ycert[k] and yhat_codes[k] != yref[k])
    r["p_misses"] = sum(1 for i in range(n) if pcert[i] and phat_codes[i] != pref[i])
    r["y_A_wrong"] = sum(1 for k in range(len(cols)) if yhat_codes[k] != yref[k])
    r["p_A_wrong"] = sum(1 for i in range(n) if phat_codes[i] != pref[i])
    r["escalated"] = not (all(ycert) and all(pcert))
    r["t_base_s"] = t_base
    if r["escalated"]:
        t1 = time.perf_counter()
        yc, pc, bits, how = escalate(g, q, Vf, cols)
        r["t_escalate_s"] = time.perf_counter() - t1; r["escalation_bits"] = bits; r["escalation_how"] = how
        if yc is not None:
            final_y = [yhat_codes[k] if ycert[k] else yc[k] for k in range(len(cols))]
            final_p = [phat_codes[i] if pcert[i] else pc[i] for i in range(n)]
            r["final_y_wrong"] = sum(1 for k in range(len(cols)) if final_y[k] != yref[k])
            r["final_p_wrong"] = sum(1 for i in range(n) if final_p[i] != pref[i])
            r["false_alarm_outputs"] = sum(1 for k in range(len(cols)) if not ycert[k] and yhat_codes[k] == yref[k]) + \
                                       sum(1 for i in range(n) if not pcert[i] and phat_codes[i] == pref[i])
    else:
        r["final_y_wrong"] = r["y_A_wrong"]; r["final_p_wrong"] = r["p_A_wrong"]
    res.append(r)

# --- negative controls ----------------------------------------------------------
neg = {}
for bad, why in [(np.where(np.arange(256) == 7, np.nan, CB["affine_s0.0084"]), "nan_rep"),
                 (np.where(np.arange(256) == 7, np.inf, CB["affine_s0.0084"]), "inf_rep")]:
    try: generation(bad, 1.0); neg[why] = "ACCEPTED (defect)"
    except Refused as e: neg[why] = f"refused: {e}"
for b in (0.0, -1.0, float("nan")):
    try: generation(CB["affine_s0.0084"], b); neg[f"beta_{b}"] = "ACCEPTED (defect)"
    except Refused as e: neg[f"beta_{b}"] = f"refused: {e}"
# wrong codebook: a row declared under K2 evaluated with K1's generation.
g1, g2 = gens[("affine_s0.0084", 8.0)], gens[("nonuniform_normal", 8.0)]
neg["digests_distinct"] = g1["digest"] != g2["digest"]
qrow = rng.integers(0, 256, 256); Vrow = [[float(v)] for v in rng.integers(-127, 128, 256)]
m1 = int(max(set(int(a) for a in qrow), key=lambda a: g1["x"][a]))
W1 = table_row(g1, m1)[0]; Wr = [W1[int(a)] for a in qrow]; S1 = sum(Wr)
N1 = sum(Fraction(Wr[i]) * Fraction(Vrow[i][0]) for i in range(256)); yh = N1 / S1
B1 = sum(abs(Fraction(Vrow[i][0]) - yh) for i in range(256)) / 2 / (S1 - 128)
yref2, _ = ref_codes(g2["x"], 8.0, qrow, Vrow, [0])
neg["wrong_codebook_without_identity_check"] = dict(bound_certifies=code_y(yh - B1) == code_y(yh + B1),
                                                    code_correct=code_y(yh) == yref2[0], A_value=float(yh))
neg["wrong_codebook_with_identity_check"] = "refused (digest mismatch)" if g1["digest"] != g2["digest"] else "ACCEPTED (defect)"
# mask: masked elements are excluded before the max and the sums.
qm = np.array([200, 10, 255, 30]); keep = np.array([True, True, False, True])
ycodes_masked, _ = ref_codes(g1["x"], 8.0, qm[keep], [[1.0], [2.0], [3.0]], [0])
neg["mask_reference_code"] = ycodes_masked[0]
# exactly-rational y on a .5 midpoint with TWO distinct representatives: both
# groups have V-mean 5/2, so y = 5/2 exactly; the declared rule (ties away from
# zero) gives code 3. The interval path alone can never certify this.
gq = gens[("affine_s0.0084", 8.0)]; qt = np.array([200, 200, 10, 10]); Vt = [[2.0], [3.0], [1.0], [4.0]]
ye, pe, bits_t, how_t = escalate(gq, qt, Vt, [0])
neg["rational_midpoint_y"] = dict(code=ye[0] if ye else None, expected=3, how=how_t, bits=bits_t,
                                  p_codes=pe)
neg["rational_midpoint_ref_code"] = ref_codes(gq["x"], 8.0, qt, Vt, [0])[0][0]
json.dump(dict(cases=res, negative_controls=neg), open(OUT, "w"), indent=0, default=str)
