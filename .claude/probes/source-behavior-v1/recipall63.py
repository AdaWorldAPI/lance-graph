# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3 addendum: driver for recipall63.cc (exhaustive superset check of the reciprocal stage).

Usage: recipall63.py RECIPALL TRACE63 OUT.json [JOBS]
Prepared weights per calibration come from TFLM's own primitives (trace63 `softexp`), for the 14
softmax63.py calibrations (regenerated with its seed).
"""
import json, subprocess, sys, time, concurrent.futures as cf
from pathlib import Path
HERE = Path(__file__).resolve().parent
RA, TRACE, RESULT_PATH, *rest = sys.argv[1:]
JOBS = int(rest[0]) if rest else 4
DELTA = rest[1] if len(rest) > 1 else None   # negative control: perturbed reciprocal
ONLY = int(rest[2]) if len(rest) > 2 else None   # restrict to one calibration index
src = (HERE / "softmax63.py").read_text()
g = {"__name__": "softmax63_corpus"}; argv = sys.argv; sys.argv = ["x", "x", "x", "/dev/null"]
exec(src[: src.index("# --- prepared per-calibration weights")], g); sys.argv = argv
CAL, f32bits = g["CAL"], g["f32bits"]
tables = []
for i, (s, z, b) in enumerate(CAL):
    out = subprocess.run([TRACE], input=f"softexp {f32bits(s)} {f32bits(b)}\n", capture_output=True, text=True, check=True).stdout.splitlines()
    tables.append((i, out[1:]))
def job(item):
    i, lines = item
    r = subprocess.run([RA] + ([DELTA] if DELTA else []), input=f"C {i}\n" + "\n".join(lines) + "\n", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()
t0 = time.perf_counter()
sel = [t for t in tables if ONLY is None or t[0] == ONLY]
lines = [l for part in cf.ThreadPoolExecutor(JOBS).map(job, sel) for l in part]
per, cand = {}, []
for l in lines:
    t = l.split()
    if t[0] == "A": per[int(t[1])] = dict(cal=CAL[int(t[1])], s_lo=int(t[2]), s_hi=int(t[3]), s_where_reciprocal_differs=int(t[4]), candidate_pairs=int(t[5]))
    elif t[0] == "X": cand.append(dict(cal=int(t[1]), S=int(t[2]), d=int(t[3]), source_byte=int(t[4]), exact_byte=int(t[5])))
res = dict(arm="perturbed reciprocal +" + DELTA if DELTA else "correctly rounded reciprocal", domain="every S in [B[0], 2^28) x every distinct admitted weight; a superset of all reachable (S, d)",
           calibrations=len(per), s_values_checked=sum(v["s_hi"] - v["s_lo"] + 1 for v in per.values()),
           s_where_reciprocal_differs=sum(v["s_where_reciprocal_differs"] for v in per.values()),
           candidate_pairs=sum(v["candidate_pairs"] for v in per.values()),
           candidate_samples=cand, per_calibration=per, wall_s=round(time.perf_counter() - t0, 1))
json.dump(res, open(RESULT_PATH, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_calibration"}, indent=1))
