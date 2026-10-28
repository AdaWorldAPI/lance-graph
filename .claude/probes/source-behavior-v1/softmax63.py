# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3 part C: TFLM int8 SOFTMAX source oracle sweep and source-compatible recipe.

Usage: softmax63.py SOFTMAX_ORACLE_BIN TRACE63_BIN OUT.json

Arms per row (int8 input codes, output scale 1/256 zp -128, the TFLM int8 contract):
  oracle   real TFLM Register_SOFTMAX (../fisherz-softmax-v1/softmax_oracle.cc, reused)
  recipe   source-compatible prepared recipe:
             prepared (once per calibration, with TFLM's own primitives via trace63 `softexp`):
               A[d] = exp_on_negative_values(rescale(d)) raw Q0.31,  B[d] = Rescale<12>(A[d]) raw,
               admitted[d] = d >= diff_min, for d = -255..0
             per row (lab Python, integer only): max, S = sum B[d_i] (refuse if > INT32_MAX: source UB),
               GetReciprocal(S,12) -> (shifted_scale, nbits), refuse if exponent nbits+23 not in [0,31]
               (source TFLITE_CHECK abort), out_i = clamp(RDBPOT(SRDHM(shifted, A[d_i]), exp) - 128),
               non-admitted -> -128.  No exp/tanh evaluated at run time.
  f24      Amendment 6.1 exact-first / exact-zero F24 arm with certified escalation (math contract:
           p = softmax(beta*scale*(q - zp)) correctly rounded, ties away)
  ref      independent mpmath reference (40 digits) for the math contract
GetReciprocal is re-implemented in Python for the recipe and checked against TFLM's GetReciprocal.
"""
import json, math, random, subprocess, sys, time
from fractions import Fraction
import numpy as np, mpmath as mp

ORACLE, TRACE, OUT = sys.argv[1:4]
mp.mp.dps = 40
I32MIN, I32MAX = -(1 << 31), (1 << 31) - 1
def i32(x): x &= 0xFFFFFFFF; return x - (1 << 32) if x >> 31 else x
def trunc_div(a, b): q = abs(a) // b; return q if a >= 0 else -q
def srdhm(a, b):                                   # fixedpoint.h:328-338
    if a == b == I32MIN: return I32MAX
    ab = a * b; nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    return i32(trunc_div(ab + nudge, 1 << 31))
def rdbpot(x, e):                                  # fixedpoint.h:356-367
    mask = (1 << e) - 1; rem = x & mask; thr = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> e) + (1 if rem > thr else 0)
def satshl(x, e):                                  # fixedpoint.h:385-406
    thr = (1 << (31 - e)) - 1
    return I32MAX if x > thr else I32MIN if x < -thr else i32(x << e)
def rhs(a, b): s = a + b; sg = 1 if s >= 0 else -1; return trunc_div(s + sg, 2)   # :235-241
def one_over_one_plus_x(a):                        # fixedpoint.h:832-853
    hd = rhs(a, I32MAX)
    x = i32(1515870810 + srdhm(hd, -1010580540))
    for _ in range(3):
        hdx = srdhm(hd, x); om = i32((1 << 29) - hdx)
        x = i32(x + satshl(srdhm(x, om), 2))
    return satshl(x, 1)
def get_reciprocal(s):                             # common.h:968-982, x_integer_digits = 12
    hpo = 32 - (s & 0xFFFFFFFF).bit_length()
    ssm1 = i32(((s << hpo) & 0xFFFFFFFF) - (1 << 31))
    return one_over_one_plus_x(ssm1), 12 - hpo

def run(bin_, text): return subprocess.run([bin_], input=text, capture_output=True, text=True)
def f32bits(v): return format(np.float32(v).view(np.uint32), "08x")

# --- calibrations and rows -----------------------------------------------------------------
rng = random.Random(63)
CAL = [(s, z, b) for s in (0.05, 0.1, 1 / 16, 0.0234375, 0.5, 1 / 256, 0.0137) for z, b in
       ((rng.randint(-128, 127), 1.0), (rng.randint(-128, 127), 2.0))]
NS = [2, 3, 16, 64, 256, 511, 512, 513, 1024, 4096]
def rows_for(n):
    yield "uniform", [rng.randint(-128, 127) for _ in range(n)]
    yield "near_max", [rng.randint(112, 127) for _ in range(n)]
    yield "ties", [37] * n
    yield "spike", [-128] * (n - 1) + [127]
    yield "low_range", [rng.randint(-10, 10) for _ in range(n)]
    yield "two_max", [100, 100] + [rng.randint(-128, 60) for _ in range(n - 2)]
cases = [(cal, dn, q) for cal in CAL for n in NS for dn, q in rows_for(n)]

# --- prepared per-calibration weights (source primitives) ----------------------------------
prep = {}
for s, z, b in CAL:
    out = run(TRACE, f"softexp {f32bits(s)} {f32bits(b)}\n").stdout.splitlines()
    m, sh, dmin = map(int, out[0].split()[1:])
    E = {int(t[1]): (int(t[2]) == 1, int(t[4]), int(t[5])) for t in (l.split() for l in out[1:])}
    prep[(s, b)] = dict(mult=m, shift=sh, diff_min=dmin, E=E)

# --- oracle, batched, falling back to one process per case on an abort ---------------------
def oline(cal, q):
    s, z, b = cal; return f"{len(q)} {f32bits(s)} {z} {f32bits(b)} " + " ".join(map(str, q))
t0 = time.perf_counter()
ORA = [None] * len(cases)
CH = 30
for st in range(0, len(cases), CH):
    chunk = list(range(st, min(st + CH, len(cases))))
    r = run(ORACLE, "\n".join(oline(cases[i][0], cases[i][2]) for i in chunk) + "\n")
    lines = r.stdout.splitlines()
    if r.returncode == 0 and len(lines) == len(chunk):
        for i, l in zip(chunk, lines): ORA[i] = l.split()
    else:
        for i in chunk:
            r1 = run(ORACLE, oline(cases[i][0], cases[i][2]) + "\n")
            ORA[i] = r1.stdout.split() if r1.returncode == 0 and r1.stdout else ["ABORT", str(r1.returncode)]
t_oracle = time.perf_counter() - t0

# --- recipe ---------------------------------------------------------------------------------
def recipe(cal, q, disable=()):
    s, z, b = cal; P = prep[(s, b)]; E = P["E"]
    m = max(q); S = 0
    for v in q:
        adm, a, acc = E[v - m]
        if adm or "diff_min" in disable:
            S += acc if "rescale12" not in disable else (a >> 12)
    if S > I32MAX: return "REFUSE q12_accumulator_overflow"
    if "reciprocal" in disable:
        # DISABLE: exact correctly rounded reciprocal in place of the source Newton-Raphson
        hpo = 32 - S.bit_length(); nb = 12 - hpo
        sh = min(I32MAX, rh_away(Fraction(1 << 62, S << hpo)))
    else:
        sh, nb = get_reciprocal(S)
    ex = nb + 23
    if not 0 <= ex <= 31: return f"REFUSE source_abort_exponent_{ex}"
    out = []
    for v in q:
        adm, a, _ = E[v - m]
        out.append(max(-128, min(127, rdbpot(srdhm(sh, a), ex) - 128)) if adm or "diff_min" in disable else -128)
    return out

# --- F24 exact-first / exact-zero arm and independent reference ----------------------------
def rh_away(x): a = abs(x); r = math.floor(a + Fraction(1, 2)); return r if x >= 0 else -r
def code_p(p): return max(-128, min(127, rh_away(256 * p) - 128))
def frac(x):
    sg, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * (Fraction(2) ** int(exp)) if man else Fraction(0)
    return -v if sg else v
W24 = {}
def w24(s, b, d):  # certified round(2^24 e^{beta*s*d}) and exact-zero flag; d <= 0
    key = (s, b, d)
    if key not in W24:
        if d == 0: W24[key] = (1 << 24, 0)
        else:
            mp.iv.prec = 96
            k = Fraction(float(np.float32(b))) * Fraction(float(np.float32(s))) * d
            iv = mp.iv.exp(mp.iv.mpf(k.numerator) / k.denominator) * (1 << 24)
            lo, hi = math.floor(frac(iv.a) + Fraction(1, 2)), math.floor(frac(iv.b) + Fraction(1, 2))
            assert lo == hi; W24[key] = (lo, 1)
    return W24[key]
def f24(cal, q):
    s, z, b = cal; n = len(q); m = max(q)
    if len(set(q)) == 1: return [code_p(Fraction(1, n))] * n, "shortcut"
    W = [w24(s, b, v - m) for v in q]; S = sum(w for w, _ in W); E = Fraction(sum(e for _, e in W), 2)
    if S <= E: cert = [False] * n; ph = [Fraction(0)] * n
    else:
        ph = [Fraction(w, S) for w, _ in W]
        cert = []
        for (w, e), p in zip(W, ph):
            Bp = (Fraction(e, 2) + p * E) / (S - E)
            cert.append(code_p(max(Fraction(0), p - Bp)) == code_p(min(Fraction(1), p + Bp)))
    out = [code_p(p) if c else None for p, c in zip(ph, cert)]
    if all(cert): return out, "bound"
    kq = Fraction(float(np.float32(b))) * Fraction(float(np.float32(s)))
    for prec in (64, 128, 256, 512):
        mp.iv.prec = prec
        e = {v: mp.iv.exp(mp.iv.mpf((kq * (v - m)).numerator) / (kq * (v - m)).denominator) for v in set(q)}
        Sv = sum((e[v] for v in q), mp.iv.mpf(0)); done = True
        for i, v in enumerate(q):
            if out[i] is None:
                pv = e[v] / Sv; lo, hi = code_p(frac(pv.a)), code_p(frac(pv.b))
                if lo == hi: out[i] = lo
                else: done = False
        if done: return out, f"escalated@{prec}"
    return out, "UNRESOLVED"
def ref(cal, q):
    s, z, b = cal; k = mp.mpf(float(np.float32(b))) * mp.mpf(float(np.float32(s))); m = max(q)
    w = [mp.e ** (k * (v - m)) for v in q]; S = mp.fsum(w)
    return [code_p(frac(x / S)) for x in w]

# --- sweep -----------------------------------------------------------------------------------
res = dict(calibrations=[dict(scale=s, zp=z, beta=b, **{k: v for k, v in prep[(s, b)].items() if k != "E"}) for s, z, b in CAL],
           rows=len(cases), elements=sum(len(q) for _, _, q in cases))
cnt = dict(source_ok_rows=0, source_abort_rows=0, recipe_refused_rows=0, refusal_matches_abort=0,
           refusal_mismatch=0, recipe_vs_oracle_cells=0, recipe_vs_oracle_diff=0,
           oracle_vs_f24_diff=0, oracle_vs_ref_diff=0, f24_vs_ref_diff=0, f24_unresolved=0, cells_compared=0,
           disable_rescale12_diff=0, disable_reciprocal_diff=0, disable_diff_min_diff=0)
f24cls, abort_rows, diff_examples, by_len = {}, [], [], {}
t_recipe = t_f24 = 0.0
for (cal, dn, q), o in zip(cases, ORA):
    key = f"n{len(q)}"; bl = by_len.setdefault(key, dict(rows=0, abort=0, diff_cells_vs_ref=0, cells=0))
    bl["rows"] += 1
    t1 = time.perf_counter(); r = recipe(cal, q); t_recipe += time.perf_counter() - t1
    src_ok = o[0] == "1" and o[1] == "1"
    if not src_ok:
        cnt["source_abort_rows"] += 1; bl["abort"] += 1
        abort_rows.append(dict(cal=cal, dist=dn, n=len(q), oracle=o[:2], recipe=r if isinstance(r, str) else "ACCEPTED"))
        if isinstance(r, str): cnt["refusal_matches_abort"] += 1; cnt["recipe_refused_rows"] += 1
        else: cnt["refusal_mismatch"] += 1
        continue
    cnt["source_ok_rows"] += 1
    oc = [int(v) for v in o[2:]]
    if isinstance(r, str): cnt["recipe_refused_rows"] += 1; cnt["refusal_mismatch"] += 1; continue
    cnt["recipe_vs_oracle_cells"] += len(q); cnt["recipe_vs_oracle_diff"] += sum(a != b for a, b in zip(r, oc))
    for dis in ("rescale12", "reciprocal", "diff_min"):
        rd = recipe(cal, q, (dis,))
        cnt[f"disable_{dis}_diff"] += len(q) if isinstance(rd, str) else sum(a != b for a, b in zip(rd, oc))
    t1 = time.perf_counter(); fc, how = f24(cal, q); t_f24 += time.perf_counter() - t1
    f24cls[how.split("@")[0]] = f24cls.get(how.split("@")[0], 0) + 1
    rc = ref(cal, q)
    cnt["cells_compared"] += len(q); bl["cells"] += len(q)
    cnt["f24_unresolved"] += sum(v is None for v in fc)
    d_or = [i for i in range(len(q)) if oc[i] != rc[i]]
    cnt["oracle_vs_ref_diff"] += len(d_or); bl["diff_cells_vs_ref"] += len(d_or)
    cnt["oracle_vs_f24_diff"] += sum(oc[i] != fc[i] for i in range(len(q)))
    cnt["f24_vs_ref_diff"] += sum(fc[i] != rc[i] for i in range(len(q)))
    if d_or and len(diff_examples) < 25:
        i = d_or[0]; s, z, b = cal
        p = mp.e ** (mp.mpf(float(np.float32(b))) * mp.mpf(float(np.float32(s))) * (q[i] - max(q)))
        diff_examples.append(dict(scale=s, zp=z, beta=b, dist=dn, n=len(q), index=i, d=q[i] - max(q),
                                  oracle=oc[i], ref=rc[i], f24=fc[i], cells_differing_in_row=len(d_or)))
# reciprocal reimplementation vs TFLM GetReciprocal (independent check of the lab recipe's one non-table step)
Ss = sorted({rng.randint(1 << 19, (1 << 28) - 1) for _ in range(20000)} | {1 << 19, (1 << 28) - 1, (1 << 20), 3 << 19})
rr = run(TRACE, "".join(f"recip {s}\n" for s in Ss)).stdout.splitlines()
rec_bad = sum(1 for s, l in zip(Ss, rr) if tuple(map(int, l.split()[2:4])) != get_reciprocal(s))
res.update(counts=cnt, f24_row_classes=f24cls, by_length=by_len, aborts=abort_rows, oracle_vs_ref_examples=diff_examples,
           reciprocal_check=dict(values=len(Ss), mismatches=rec_bad),
           timing_python_s=dict(oracle_all=t_oracle, recipe_all=t_recipe, f24_all=t_f24))
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("aborts", "oracle_vs_ref_examples", "calibrations")}, indent=1, default=str))
