# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3 addendum: driver for recipwit63.cc (reciprocal observability gate).

Usage: recipwit63.py RECIPWIT TRACE63 ORACLE OUT.json M_MAX [JOBS]

Prepared weights per calibration come from TFLM's own primitives (trace63 `softexp`); the 14
calibrations are softmax63.py's, regenerated with its seed. Any witness the search reports is
re-run through the real TFLM SOFTMAX kernel (the oracle) before it is counted.
"""
import json, subprocess, sys, concurrent.futures as cf
from pathlib import Path
HERE = Path(__file__).resolve().parent
RW, TRACE, ORACLE, RESULT_PATH, M_MAX, *rest = sys.argv[1:]
JOBS = int(rest[0]) if rest else 4
DELTAS = [1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
src = (HERE / "softmax63.py").read_text()
g = {"__name__": "softmax63_corpus"}; argv = sys.argv; sys.argv = ["x", "x", "x", "/dev/null"]
exec(src[: src.index("# --- prepared per-calibration weights")], g); sys.argv = argv
CAL, f32bits = g["CAL"], g["f32bits"]
def oline(cal, q):  # identical to softmax63.oline
    s, z, b = cal; return f"{len(q)} {g['f32bits'](s)} {z} {g['f32bits'](b)} " + " ".join(map(str, q))
tables = []
for i, (s, z, b) in enumerate(CAL):
    out = subprocess.run([TRACE], input=f"softexp {f32bits(s)} {f32bits(b)}\n", capture_output=True, text=True, check=True).stdout.splitlines()
    tables.append((i, out[1:]))
def job(chunk):
    text = "".join(f"C {i}\n" + "\n".join(lines) + "\n" for i, lines in chunk)
    r = subprocess.run([RW, M_MAX, *map(str, DELTAS)], input=text, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.splitlines()
chunks = [tables[j::JOBS] for j in range(JOBS)]
lines = [l for part in cf.ThreadPoolExecutor(JOBS).map(job, chunks) for l in part]
per, wit, neg = {}, [], {}
for l in lines:
    t = l.split()
    if t[0] == "S": per[int(t[1])] = dict(rows=int(t[2]), witnesses=int(t[3]), nearest_margin_raw=int(t[4]))
    elif t[0] == "W": wit.append(dict(cal=int(t[1]), k=int(t[2]), m=int(t[3]), d=int(t[4]), elem=["max", "d"][int(t[5])], source_byte=int(t[6]), exact_byte=int(t[7])))
    elif t[0] == "N": neg[int(t[2])] = neg.get(int(t[2]), 0) + int(t[3])
confirmed = []
for w in wit:
    cal = CAL[w["cal"]]; q = [127] * w["k"] + [127 + w["d"]] * w["m"]
    o = subprocess.run([ORACLE], input=oline(cal, q) + "\n", capture_output=True, text=True).stdout.split()
    if len(o) >= 2 and o[0] == "1" and o[1] == "1":
        kb = int(o[2]) if w["elem"] == "max" else int(o[-1])
        confirmed.append({**w, "kernel_byte": kb, "kernel_equals_source_arm": kb == w["source_byte"]})
    else:
        confirmed.append({**w, "kernel": "abort/not-run"})
res = dict(family="rows [max]*k + [max+d]*m, k=1..511, d=-255..-1 (admitted), m=0..M_MAX, real sum < 512",
           m_max=int(M_MAX), calibrations=len(CAL),
           rows_searched=sum(v["rows"] for v in per.values()),
           witnesses=sum(v["witnesses"] for v in per.values()),
           nearest_margin_raw=min(v["nearest_margin_raw"] for v in per.values()),
           per_calibration=per, witness_samples=confirmed,
           negative_control_outputs_changed_by_delta=dict(sorted(neg.items())))
json.dump(res, open(RESULT_PATH, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_calibration"}, indent=1))
