# /// script
# requires-python = ">=3.10"
# ///
"""D-TFLP Amendment 6.4: attribute every FC cell where TFLM (arm A) differs from the exact
mathematical arm (D) to a stage.

Usage: fcattr64.py RUN_DIR OUT.json     (RUN_DIR holds fc_cases.txt and tflm_fc.txt)

Per cell, with acc = bias + sum (a - in_zp) * w computed exactly:
  D   = round_half_away(acc * real),           real = in_scale * w_scale[c] / out_scale (exact, from f32 bits)
  Q   = round_half_away(acc * mult * 2^shift / 2^31)   (multiplier quantized as TFLM's FCQ, exact rounding)
  S1  = TFLM single-rounding formula on (mult, shift): (acc*mult + 2^(t-1)) >> t, t = 31 - shift
  A   = TFLM bytes (double rounding: SRDHM then RoundingDivideByPOT), recovered before offset as A - out_zp
Stage of A != D (before clamp, cells where the clamp did not bind):
  "multiplier_quantization"  Q != D and A == Q
  "integer_rounding"         Q == D and A != Q   (further split: exact negative half tie; double rounding = S1 == Q != A)
  "both"                     Q != D and A != Q
Cells where the clamp binds on either side are counted separately.
"""
import json, sys, struct
from fractions import Fraction
from collections import Counter
RUN, OUT = sys.argv[1:3]
def f32(h): return Fraction(struct.unpack("<f", struct.pack("<I", int(h, 16)))[0])
def rha(x):  # round half away from zero, exact
    n, d = x.numerator, x.denominator
    q, r = divmod(abs(n), d)
    if 2 * r >= d: q += 1
    return q if n >= 0 else -q
lines = open(f"{RUN}/fc_cases.txt").read().split("\n")
tf = {}
for l in open(f"{RUN}/tflm_fc.txt"):
    t = l.split()
    if not t: continue
    if t[0] == "FCOUT": tf.setdefault(t[1], {})["out"] = list(map(int, t[4:]))
    elif t[0] == "FCQ":
        i = t.index("|"); tf.setdefault(t[1], {})["m"] = list(map(int, t[2:i])); tf[t[1]]["s"] = list(map(int, t[i + 1:]))
cnt = Counter(); examples = {}
i = 0
while i < len(lines):
    h = lines[i].split()
    if not h or h[0] != "FC": i += 1; continue
    cid, pc, nb, ni, no = h[1], int(h[2]), int(h[3]), int(h[4]), int(h[5])
    s_in, izp, s_out, ozp, lo, hi = f32(h[6]), int(h[7]), f32(h[8]), int(h[9]), int(h[10]), int(h[11])
    IN = list(map(int, lines[i + 1].split()[1:])); WS = [f32(x) for x in lines[i + 2].split()[1:]]
    W = list(map(int, lines[i + 3].split()[1:])); B = list(map(int, lines[i + 4].split()[1:]))
    i += 5
    suite = "extended" if cid.startswith("X") else "ties" if cid.startswith("T") else "conforming"
    T = tf[cid]; M = T["m"] if len(T["m"]) == no else T["m"] * no; Sh = T["s"] if len(T["s"]) == no else T["s"] * no
    for b in range(nb):
        for c in range(no):
            acc = B[c] + sum((IN[b * ni + k] - izp) * W[c * ni + k] for k in range(ni))
            real = s_in * WS[c] / s_out
            D = rha(acc * real); Q = rha(acc * M[c] * Fraction(2) ** Sh[c] / 2 ** 31)
            t = 31 - Sh[c]; S1 = (acc * M[c] + (1 << (t - 1))) >> t
            a_byte = T["out"][b * no + c]
            clampD = max(lo, min(hi, D + ozp))
            if a_byte == clampD: continue
            key = (suite,)
            if not (lo < a_byte < hi) or not (lo < D + ozp < hi):
                kind = "clamp_bound_cell"
            else:
                A = a_byte - ozp
                if Q != D and A == Q: kind = "multiplier_quantization"
                elif Q == D and A != Q:
                    tie = (acc * M[c] * Fraction(2) ** Sh[c] / 2 ** 31).denominator == 2
                    neg = acc < 0
                    kind = ("integer_rounding:negative_half_tie" if tie and neg else
                            "integer_rounding:positive_half_tie" if tie else
                            "integer_rounding:double_rounding" if S1 == Q else "integer_rounding:other")
                else: kind = "both"
            cnt[(suite, kind)] += 1
            examples.setdefault(f"{suite}|{kind}", dict(case=cid, batch=b, channel=c, acc=acc, mult=M[c], shift=Sh[c],
                                                       real=str(real), exact=str(acc * real), D=D, Q=Q, S1=S1, A=a_byte - ozp, out_zp=ozp))
res = dict(counts={f"{s}|{k}": v for (s, k), v in sorted(cnt.items())}, examples=examples)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["counts"], indent=1))
