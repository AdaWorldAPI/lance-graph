# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""D-TFLP Amendment 6.5: held-out softmax calibrations and their exhaustive superset reference.

The 8 calibrations below are not among the 14 of softmax63.py (scales, zero points and betas chosen
outside that set; betas 0.5, 1.5 and 3.0 never appear there). This script runs the unchanged exhaustive
recipall63 binary on each, giving the reference candidate set the 6.5 gate is checked against.
Usage: heldout65.py RECIPALL TRACE63 OUT.json [JOBS]
"""
import json, subprocess, sys, time, concurrent.futures as cf
import numpy as np
HELDOUT = [(0.03, 10, 1.0), (0.2, -50, 1.0), (0.0078125, 0, 1.0), (0.125, 3, 2.0),
           (0.07, -100, 0.5), (0.25, 60, 1.5), (0.0019, 5, 1.0), (0.04, 0, 3.0)]
def f32bits(v): return format(np.float32(v).view(np.uint32), "08x")
if __name__ == "__main__":
    RA, TRACE, OUT, *rest = sys.argv[1:]
    JOBS = int(rest[0]) if rest else 3
    def job(i):
        s, z, b = HELDOUT[i]
        t = subprocess.run([TRACE], input=f"softexp {f32bits(s)} {f32bits(b)}\n", capture_output=True, text=True, check=True).stdout.splitlines()[1:]
        r = subprocess.run([RA], input=f"C {i}\n" + "\n".join(t) + "\n", capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    t0 = time.perf_counter()
    lines = [l for part in cf.ThreadPoolExecutor(JOBS).map(job, range(len(HELDOUT))) for l in part]
    per, cand = {}, []
    for l in lines:
        t = l.split()
        if t[0] == "A": per[int(t[1])] = dict(cal=HELDOUT[int(t[1])], s_lo=int(t[2]), s_hi=int(t[3]), s_where_reciprocal_differs=int(t[4]), candidate_pairs=int(t[5]))
        elif t[0] == "X": cand.append(dict(cal=int(t[1]), S=int(t[2]), d=int(t[3]), source_byte=int(t[4]), exact_byte=int(t[5])))
    res = dict(arm="correctly rounded reciprocal", calibrations=HELDOUT, candidate_pairs=len(cand), candidate_samples=cand,
               per_calibration=per, wall_s=round(time.perf_counter() - t0, 1))
    open(OUT, "w").write(json.dumps(res, separators=(",", ":")) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "candidate_samples"}, indent=1))
