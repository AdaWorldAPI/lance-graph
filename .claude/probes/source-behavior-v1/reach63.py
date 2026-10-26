# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3 addendum: turn recipall63 candidates into real int8 rows and confirm them on the kernel.

Usage: reach63.py REACH TRACE63 ORACLE RECIPALL_JSON OUT.json

For each candidate (calibration, S, d) the reach63 solver builds a row (multiplicities over admitted
differences; row maximum 127, an element at difference e has code 127 + e). The row is run through the
real TFLM SOFTMAX kernel. A candidate is an AUTHENTIC WITNESS iff the kernel completes and its byte at
the d element equals the source-reciprocal byte while the exact-reciprocal byte differs.
"""
import json, subprocess, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
REACH, TRACE, ORACLE, RA_JSON, RESULT_PATH = sys.argv[1:6]
src = (HERE / "softmax63.py").read_text()
g = {"__name__": "softmax63_corpus"}; argv = sys.argv; sys.argv = ["x", "x", "x", "/dev/null"]
exec(src[: src.index("# --- prepared per-calibration weights")], g); sys.argv = argv
CAL, f32bits = g["CAL"], g["f32bits"]
def oline(cal, q):  # identical to softmax63.oline
    s, z, b = cal; return f"{len(q)} {f32bits(s)} {z} {f32bits(b)} " + " ".join(map(str, q))
ra = json.load(open(RA_JSON))
by_cal = defaultdict(list)
for c in ra["candidate_samples"]: by_cal[c["cal"]].append(c)
out = []
for ci, cands in sorted(by_cal.items()):
    s, z, b = CAL[ci]
    table = subprocess.run([TRACE], input=f"softexp {f32bits(s)} {f32bits(b)}\n", capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    text = f"C {ci}\n" + "\n".join(table) + f"\nQ {len(cands)}\n" + "".join(f"{c['S']} {c['d']}\n" for c in cands)
    lines = subprocess.run([REACH], input=text, capture_output=True, text=True, check=True).stdout.splitlines()
    rows, cur = [], None
    for l in lines:
        t = l.split()
        if t[0] == "R": cur = dict(S=int(t[1]), d=int(t[2]), ok=t[5] == "1", mult={}); rows.append(cur)
        elif t[0] == "M": cur["mult"][int(t[1])] = int(t[2])
    for c, r in zip(cands, rows):
        rec = {**c, "reachable": r["ok"]}
        if r["ok"]:
            q = [v for e, n in sorted(r["mult"].items(), reverse=True) for v in [127 + e] * n]
            pos = q.index(127 + c["d"])
            o = subprocess.run([ORACLE], input=oline(CAL[ci], q) + "\n", capture_output=True, text=True).stdout.split()
            rec.update(row_length=len(q), row_distinct_codes=len(r["mult"]), row_code_counts={str(127 + e): n for e, n in sorted(r["mult"].items())})
            if len(o) >= 2 and o[0] == "1" and o[1] == "1":
                kb = int(o[2 + pos]); rec.update(kernel_byte=kb, authentic_witness=kb == c["source_byte"] != c["exact_byte"])
            else:
                rec.update(kernel="abort", authentic_witness=False)
        out.append(rec)
res = dict(candidates_checked=len(out), reachable=sum(r["reachable"] for r in out),
           authentic_witnesses=sum(r.get("authentic_witness", False) for r in out),
           rows=out)
json.dump(res, open(RESULT_PATH, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
print(json.dumps(out[:3], indent=1))
