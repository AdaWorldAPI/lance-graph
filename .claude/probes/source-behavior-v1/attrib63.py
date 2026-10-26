# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3 parts A and B.

Usage: attrib63.py TRACE63_BIN ARTIFACTS_BIN OUT.json

A. LOGISTIC stage attribution at intermediate codes +-1 (x = +-1/128), from the harness's
   kernel-verified trace (trace63 `ltrace`), against independently computed ideals.
B. Four-arm attribution over the 569,344 TANH -> LOGISTIC cells of Amendment 6.2, using the
   committed-recipe tables (artifacts.bin: Amendment 6.2 SourceParity / CertifiedMath TANH
   tables, sha256 db55e763...) and the source / certified LOGISTIC tables at (1/128, 0).
   Arms keep the intermediate int8 code in every case: SS, SC, CS, CC (TANH, LOGISTIC).
"""
import hashlib, json, math, subprocess, sys
from fractions import Fraction
from pathlib import Path
import mpmath as mp

TRACE, ART, OUT = sys.argv[1:4]
HERE = Path(__file__).resolve().parent
mp.mp.dps = 60
def rh_away(x): a = abs(x); r = math.floor(a + Fraction(1, 2)); return r if x >= 0 else -r
def clamp8(c): return max(-128, min(127, c))
def frac(x):
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * (Fraction(2) ** int(exp)) if man else Fraction(0)
    return -v if sign else v
def run(text): return subprocess.run([TRACE], input=text, capture_output=True, text=True, check=True).stdout.splitlines()

# ---------------- A. LOGISTIC trace ---------------------------------------------------------
rows = [l.split() for l in run("ltrace 3c000000 0\n")]
assert len(rows) == 256 and all(r[0] == "G" for r in rows), "trace did not reproduce the kernel on all codes"
G = {int(r[1]): [int(v) for v in r[1:]] for r in rows}
L2S = [G[c][12] for c in range(-128, 128)]                    # verified kernel bytes
A = {}
Q31 = 1 << 31
for code in (-1, 1):
    _, inp, radius, in_q4, abs_raw, am_raw, core_raw, rem, e_raw, rpos_raw, raw, q23, byte, kbyte = G[code]
    x = Fraction(code, 128)                                   # exact real input of the intermediate code
    x_fp = Fraction(in_q4, 1 << 27)                           # Q4.27 input the kernel uses
    ax = abs(x_fp)
    e_true = frac(mp.exp(-mp.mpf(ax.numerator) / ax.denominator))
    # Taylor polynomial the core evaluates (fixedpoint.h:711-732): degree 4 about -1/8, exact arithmetic
    t = -ax + Fraction(1, 8)
    c = frac(mp.exp(mp.mpf(-1) / 8))
    poly = c * (1 + t + t**2 / 2 + t**3 / 6 + t**4 / 24)
    e_k = Fraction(e_raw, Q31)
    r_exact_from_k = 1 / (1 + e_k)                            # exact division on the kernel's own exp
    sig_true = 1 / (1 + e_true)                               # sigma(|x|)
    one = Fraction(Q31 - 1, Q31)                              # F0::One() raw = 2^31 - 1
    if code > 0:
        out_true, out_from_k_exact, out_k = sig_true, r_exact_from_k, Fraction(rpos_raw, Q31)
    else:
        out_true, out_from_k_exact, out_k = 1 - sig_true, one - r_exact_from_k, Fraction(raw, Q31)
    code_of = lambda p: clamp8(rh_away(256 * p) - 128)
    boundary = Fraction(2 * (q23 if code > 0 else q23 + 1) - 1, 2 * 256)   # the half the codes straddle
    stage = ("exp polynomial core (Taylor degree 4 about -1/8)" if code_of(out_from_k_exact) != code_of(out_true)
             else "one_over_one_plus_x (Newton-Raphson)" if code_of(out_k) != code_of(out_from_k_exact) else "other")
    A[str(code)] = dict(
        x=str(x), in_q4=in_q4, in_q4_ideal=str(x * (1 << 27)), input_rescale_exact=x_fp == x,
        quarter_remainder_raw=rem, barrel_multipliers_applied=rem != 0,
        exp_core_raw=core_raw, exp_raw=e_raw, exp_true_raw=float(e_true * Q31), exp_poly_exact_raw=float(poly * Q31),
        exp_err_raw=float(e_raw - e_true * Q31), poly_truncation_err_raw=float((poly - e_true) * Q31),
        exp_fixed_point_rounding_err_raw=float(e_raw - poly * Q31),
        div_input_e_equals_tanh_62_value=e_raw == 2130771442,
        rpos_raw=rpos_raw, rpos_from_exact_division_raw=float(r_exact_from_k * Q31), div_err_raw=float(rpos_raw - r_exact_from_k * Q31),
        sigma_true_raw=float(sig_true * Q31), selected_raw=raw, selected_true_raw=float(out_true * Q31),
        one_raw=Q31 - 1, rounding_boundary_raw=float(boundary * Q31), q23=q23, byte=byte, kernel_byte=kbyte,
        math_byte=code_of(out_true), earliest_divergence=stage)

# ---------------- B. four-arm attribution ----------------------------------------------------
blob = Path(ART).read_bytes()
assert hashlib.sha256(blob).hexdigest().startswith("db55e763"), "artifacts.bin is not the Amendment 6.2 artifact"
n = int.from_bytes(blob[10:14], "little"); p = 14
T_S, T_C = [], []
for _ in range(n):
    p += 64
    T_S.append([b - 256 if b > 127 else b for b in blob[p:p + 256]]); p += 256
    T_C.append([b - 256 if b > 127 else b for b in blob[p:p + 256]]); p += 256
    p += 64; k = int.from_bytes(blob[p:p + 2], "little"); p += 2 + 2 * k
assert p == len(blob) and n == 2224

def cert_sigma_code(v):  # Q2(sigma(v)) = clamp(round_away(256 sigma) - 128), interval-certified
    if v == 0: return 0
    for prec in (64, 128, 256, 512):
        mp.iv.prec = prec
        s = 1 / (1 + mp.iv.exp(-(mp.iv.mpf(v.numerator) / v.denominator)))
        lo, hi = clamp8(rh_away(frac((256 * s).a)) - 128), clamp8(rh_away(frac((256 * s).b)) - 128)
        if lo == hi: return lo
    raise RuntimeError("undecided")
L2C = [cert_sigma_code(Fraction(c, 128)) for c in range(-128, 128)]
l2_diff = [c - 128 for c in range(256) if L2S[c] != L2C[c]]

cnt = dict(cells=0, SS_ne_CC=0, SS_ne_SC=0, SS_ne_CS=0, CS_ne_CC=0, SC_ne_CC=0,
           tanh_differs=0, logistic_differs_on_source_path=0, logistic_differs_on_certified_path=0)
cls = dict(logistic_only=0, tanh_only=0, both_required=0, each_alone_sufficient=0, one_dominant=0)
cancel = dict(tanh_error_absorbed_by_cert_logistic=0, logistic_error_absent_on_cert_tanh_path=0, both_errors_cancel_in_SS=0)
examples = []
for k in range(n):
    for i in range(256):
        s1, c1 = T_S[k][i], T_C[k][i]
        SS, SC, CS, CC = L2S[s1 + 128], L2C[s1 + 128], L2S[c1 + 128], L2C[c1 + 128]
        cnt["cells"] += 1
        cnt["SS_ne_CC"] += SS != CC; cnt["SS_ne_SC"] += SS != SC; cnt["SS_ne_CS"] += SS != CS
        cnt["CS_ne_CC"] += CS != CC; cnt["SC_ne_CC"] += SC != CC
        td = s1 != c1; ls = L2S[s1 + 128] != L2C[s1 + 128]; lc = L2S[c1 + 128] != L2C[c1 + 128]
        cnt["tanh_differs"] += td; cnt["logistic_differs_on_source_path"] += ls; cnt["logistic_differs_on_certified_path"] += lc
        if td and SC == CC: cancel["tanh_error_absorbed_by_cert_logistic"] += 1
        if td and ls and not lc: cancel["logistic_error_absent_on_cert_tanh_path"] += 1
        if (td or ls or lc) and SS == CC: cancel["both_errors_cancel_in_SS"] += 1
        if SS != CC:
            fix_t, fix_l = CS == CC, SC == CC      # does a single correction reach CC?
            if not td: cls["logistic_only"] += 1
            elif not ls and not lc: cls["tanh_only"] += 1
            # both_required: correcting EITHER stage alone already gives CC, so the deviation needs both errors.
            # each_alone_sufficient: neither single correction reaches CC.
            elif fix_t and fix_l: cls["both_required"] += 1
            elif not fix_t and not fix_l: cls["each_alone_sufficient"] += 1
            else: cls["one_dominant"] += 1
            if td and len(examples) < 17:
                examples.append(dict(calibration=k, input_code=i - 128, tanh_source=s1, tanh_certified=c1,
                                     SS=SS, SC=SC, CS=CS, CC=CC))
res = dict(A_logistic_trace=A, logistic_source_vs_certified_codes=l2_diff,
           B=dict(counts=cnt, classes_of_SS_ne_CC=cls, cancellations=cancel, tanh_cell_examples=examples))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1)[:6000])
