# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3: stage attribution of the softmax cells where TFLM differs from the certified math
contract. For each such row (from softmax63.json), the output code is recomputed with ONE source stage
at a time replaced by exact arithmetic. Usage: softdiff63.py TRACE63 OUT.json"""
import json, sys, math
from fractions import Fraction
src = open(__file__.replace("softdiff63.py", "softmax63.py")).read()
TRACE, RESULT_PATH = sys.argv[1:3]   # not OUT: exec of softmax63 rebinds OUT
sys.argv = [sys.argv[0], "x", TRACE, "/dev/null"]
exec(src[: src.index("# --- oracle, batched")])
exec(src[src.index("# --- recipe ----"): src.index("# --- sweep ---")])
D = json.load(open(__file__.replace("softdiff63.py", "softmax63.json")))["oracle_vs_ref_examples"]
rng2 = None
out = []
for e in D:
    cal = next(c for c in CAL if c[0] == e["scale"] and c[1] == e["zp"] and c[2] == e["beta"])
    # rebuild the row deterministically: the sweep generated rows with rng=Random(63); regenerate the case list
    q = next(qq for (c, dn, qq) in cases if c == cal and dn == e["dist"] and len(qq) == e["n"])
    s, z, b = cal; P = prep[(s, b)]; E = P["E"]; m = max(q); i = e["index"]
    kq = Fraction(float(np.float32(b))) * Fraction(float(np.float32(s)))
    rescale_real = lambda d: Fraction(P["mult"], 1 << 31) * d * (1 << P["shift"]) / (1 << 26)  # source multiplier, exact
    ex = lambda v: frac(mp.e ** (mp.mpf(v.numerator) / v.denominator))
    def code_from(w):  # exact normalization of given weights
        return code_p(w[i] / sum(w))
    w_math = [ex(kq * (v - m)) for v in q]
    w_srcmult = [ex(rescale_real(v - m)) for v in q]                    # source multiplier, exact exp
    w_srcexp = [Fraction(E[v - m][1], 1 << 31) if E[v - m][0] else Fraction(0) for v in q]  # source exp values
    w_srcacc = [Fraction(E[v - m][2], 1 << 19) if E[v - m][0] else Fraction(0) for v in q]   # Q12 rescaled
    S = sum(E[v - m][2] for v in q if E[v - m][0]); sh, nb = get_reciprocal(S)
    # source reciprocal, exact final product and rounding vs source SRDHM+RDBPOT
    a_i = E[q[i] - m][1]
    p_src_recip_exact_tail = Fraction(sh, 1 << 31) * Fraction(a_i, 1 << 31) / (1 << nb)
    out.append(dict(example=e, mult=P["mult"], shift=P["shift"], diff_min=P["diff_min"], S_q12_raw=S, shifted_scale=sh, nbits=nb,
                    code_math=code_from(w_math), code_source_multiplier_exact_rest=code_from(w_srcmult),
                    code_source_exp_exact_sum=code_from(w_srcexp), code_source_exp_q12_sum_exact_div=Fraction(w_srcexp[i]) and code_p(w_srcexp[i] / sum(w_srcacc)),
                    code_source_recip_exact_tail=code_p(p_src_recip_exact_tail), code_oracle=e["oracle"],
                    p_math=float(w_math[i] / sum(w_math)), p_source_effective=(e["oracle"] + 128) / 256))
json.dump(out, open(RESULT_PATH, "w"), indent=1, default=str)
for o in out: print({k: v for k, v in o.items() if k != "example"}, o["example"]["dist"], o["example"]["n"])
