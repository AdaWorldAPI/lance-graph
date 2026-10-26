# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""D-TFLP Amendment 6.4 follow-up: driver for recipprune64.cc (boundary-pruned reciprocal search).

Usage: recipprune64.py PRUNE TRACE63 REFERENCE_JSON OUT.json [DELTA [SLACK]]
REFERENCE_JSON is the exhaustive recipall63 result for the same arm (recipall63.json for the exact
reciprocal, recipall63_control_delta256.json for DELTA=256). The pruned candidate set must equal it.
The reference must list every candidate (candidate_samples length == candidate_pairs), which is checked.
"""
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PRUNE, TRACE, REF, RESULT_PATH, *rest = sys.argv[1:]
args = rest[:2]
src = (HERE / "softmax63.py").read_text()
g = {"__name__": "softmax63_corpus"}; argv = sys.argv; sys.argv = ["x", "x", "x", "/dev/null"]
exec(src[: src.index("# --- prepared per-calibration weights")], g); sys.argv = argv
CAL, f32bits = g["CAL"], g["f32bits"]
ref = json.load(open(REF))
assert len(ref["candidate_samples"]) == ref["candidate_pairs"], "reference does not list every candidate"
cals = sorted(int(k) for k in ref["per_calibration"])
text = ""
for i in cals:
    s, z, b = CAL[i]
    lines = subprocess.run([TRACE], input=f"softexp {f32bits(s)} {f32bits(b)}\n", capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    text += f"C {i}\n" + "\n".join(lines) + "\n"
t0 = time.perf_counter()
r = subprocess.run([PRUNE] + args, input=text, capture_output=True, text=True)
assert r.returncode == 0, (r.returncode, r.stderr)
wall = time.perf_counter() - t0
E, P, got = {}, {}, set()
for l in r.stdout.splitlines():
    t = l.split()
    if t[0] == "E": E[int(t[1])] = int(t[2])
    elif t[0] == "P": P[int(t[1])] = dict(intervals=int(t[2]), s_evaluated=int(t[3]), pairs_evaluated=int(t[4]))
    elif t[0] == "X": got.add((int(t[1]), int(t[2]), int(t[3]), int(t[4]), int(t[5])))
want = {(c["cal"], c["S"], c["d"], c["source_byte"], c["exact_byte"]) for c in ref["candidate_samples"]}
n_w = {i: len({x for x in want if x[0] == i}) for i in cals}
exhaustive_pairs = sum(ref["per_calibration"][str(i)]["s_hi"] - ref["per_calibration"][str(i)]["s_lo"] + 1 for i in cals)
res = dict(
    arm=("source + " + args[0]) if args else "correctly rounded reciprocal",
    window=("override " + args[1]) if len(args) > 1 else "E_h + |delta| per band",
    reciprocal_error_bound_per_band=E,
    calibrations=cals,
    pruned_candidates=len(got), exhaustive_candidates=len(want),
    identical_sets=got == want, missed_by_pruning=len(want - got), extra_in_pruning=len(got - want),
    missed_examples=sorted(want - got)[:10], extra_examples=sorted(got - want)[:10],
    s_values_evaluated=sum(p["s_evaluated"] for p in P.values()),
    distinct_pairs_evaluated=sum(p["pairs_evaluated"] for p in P.values()),
    exhaustive_s_values=exhaustive_pairs,
    per_calibration={i: {**P[i], "exhaustive_candidates": n_w[i]} for i in cals},
    wall_s=round(wall, 1))
json.dump(res, open(RESULT_PATH, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_calibration"}, indent=1))
