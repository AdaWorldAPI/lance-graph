# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3 addendum: numerical-error propagation, TANH -> LOGISTIC.

Usage: prop63.py TRACE63 ARTIFACTS_BIN OUT.json

1. Internal vs observable error, LOGISTIC at (1/128, 0): for every non-saturated code, the kernel's
   exp value vs the true exp (Q0.31 units), against whether the output byte differs from the
   certified byte. Same for the pre-rounding raw value.
2. Fan-in: per calibration, how many input codes land on each intermediate TANH code, and how many
   land on the two intermediate codes (-1, +1) where LOGISTIC's source byte differs. One wrong
   LOGISTIC entry is seen by every input that maps onto it.
3. The TANH discrepancy cells: visible after stage 1, then hidden / preserved / amplified / cancelled
   by each LOGISTIC table.
"""
import hashlib, json, subprocess, sys, statistics
from fractions import Fraction
import mpmath as mp
TRACE, ART, RESULT_PATH = sys.argv[1:4]
mp.mp.dps = 50
rows = [l.split() for l in subprocess.run([TRACE], input="ltrace 3c000000 0\n", capture_output=True, text=True, check=True).stdout.splitlines()]
assert len(rows) == 256
Q31 = 1 << 31
internal = dict(codes_nonsaturated=0, exp_error_nonzero=0, exp_err_min=None, exp_err_max=None,
                selected_error_nonzero=0, selected_err_max_abs=0, output_byte_differs_from_certified=0)
errs = []; serrs = []
for r in rows:
    code, inp, radius, in_q4, abs_raw, am_raw, core_raw, rem, e_raw, rpos_raw, raw, q23, byte, kb = map(int, r[1:])
    if abs(inp) >= radius: continue
    internal["codes_nonsaturated"] += 1
    x = mp.mpf(in_q4) / (1 << 27)
    e_true = mp.exp(-abs(x)) * Q31
    err = float(e_raw - e_true); errs.append(err)
    sig = 1 / (1 + mp.exp(-x)) * Q31
    serr = float(raw - sig); serrs.append(abs(serr))
    cert = max(-128, min(127, int(mp.floor(256 * sig / Q31 + mp.mpf(1) / 2)) - 128))
    internal["exp_error_nonzero"] += abs(err) > 0.5
    internal["selected_error_nonzero"] += abs(serr) > 0.5
    internal["output_byte_differs_from_certified"] += kb != cert
internal.update(exp_err_min=min(errs), exp_err_max=max(errs), selected_err_max_abs=max(serrs))
blob = open(ART, "rb").read()
assert hashlib.sha256(blob).hexdigest().startswith("db55e763")
n = int.from_bytes(blob[10:14], "little"); p = 14
T_S, T_C = [], []
for _ in range(n):
    p += 64
    T_S.append([b - 256 if b > 127 else b for b in blob[p:p + 256]]); p += 256
    T_C.append([b - 256 if b > 127 else b for b in blob[p:p + 256]]); p += 256
    p += 64; k = int.from_bytes(blob[p:p + 2], "little"); p += 2 + 2 * k
assert p == len(blob)
fan_pm1_S = [sum(v in (-1, 1) for v in t) for t in T_S]
fan_pm1_C = [sum(v in (-1, 1) for v in t) for t in T_C]
distinct = [len(set(t)) for t in T_S]
maxfan = [max(t.count(v) for v in set(t)) for t in T_S]
stage1 = sum(a != b for ts, tc in zip(T_S, T_C) for a, b in zip(ts, tc))
res = dict(
    internal_vs_observable_logistic_1_128=internal,
    fan_in=dict(calibrations=n,
                inputs_on_pm1_source_tanh=dict(total=sum(fan_pm1_S), min=min(fan_pm1_S), median=statistics.median(fan_pm1_S), max=max(fan_pm1_S),
                                               calibrations_with_zero=sum(f == 0 for f in fan_pm1_S)),
                inputs_on_pm1_certified_tanh=dict(total=sum(fan_pm1_C), min=min(fan_pm1_C), max=max(fan_pm1_C)),
                distinct_intermediate_codes=dict(min=min(distinct), median=statistics.median(distinct), max=max(distinct)),
                largest_single_intermediate_fan_in=dict(min=min(maxfan), median=statistics.median(maxfan), max=max(maxfan))),
    stage1_visible_tanh_cells=stage1)
json.dump(res, open(RESULT_PATH, "w"), indent=1)
print(json.dumps(res, indent=1))
