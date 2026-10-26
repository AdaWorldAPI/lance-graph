# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""D-TFLP Amendment 6.4 follow-up: exact reachability of every recipall63 candidate, then kernel confirmation.

Usage: reach64.py REACH64 TRACE63 ORACLE RECIPALL_JSON OUT.json

Three levels are kept apart in the output:
  superset candidate  (S, weight) from recipall63: the output byte changes under the exact reciprocal;
  reachable pair      reach64 proves some admitted row has sum S and an element at difference d,
                      or proves no row does (R < 0, or R below the smallest representable value
                      in its residue class mod B[0]);
  witness row         the constructed row run through the real TFLM SOFTMAX kernel: the kernel byte
                      at the d element equals the source-reciprocal byte and differs from the exact one.
"""
import json, subprocess, sys
from collections import Counter, defaultdict
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
prev = {(r["cal"], r["S"], r["d"]): r for r in json.load(open(HERE / "reach63.json"))["rows"]}
by_cal = defaultdict(list)
for c in ra["candidate_samples"]: by_cal[c["cal"]].append(c)
out = []
for ci, cands in sorted(by_cal.items()):
    table = subprocess.run([TRACE], input=f"softexp {f32bits(CAL[ci][0])} {f32bits(CAL[ci][2])}\n", capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    adm = {int(t.split()[1]): int(t.split()[2]) for t in table}
    Bt = {int(t.split()[1]): int(t.split()[5]) if int(t.split()[2]) else 0 for t in table}
    text = f"C {ci}\n" + "\n".join(table) + f"\nQ {len(cands)}\n" + "".join(f"{c['S']} {c['d']}\n" for c in cands)
    r = subprocess.run([REACH], input=text, capture_output=True, text=True)
    assert r.returncode == 0, (ci, r.returncode, r.stderr)
    rows, cur = [], None
    for l in r.stdout.splitlines():
        t = l.split()
        if t[0] == "R": cur = dict(S=int(t[1]), d=int(t[2]), status=t[3], mult={}); rows.append(cur)
        elif t[0] == "M": cur["mult"][int(t[1])] = int(t[2])
    assert len(rows) == len(cands)
    for c, row in zip(cands, rows):
        assert (row["S"], row["d"]) == (c["S"], c["d"])
        p = prev.get((ci, c["S"], c["d"]), {})
        rec = {**c, "pair_status": row["status"], "reach63_found_row": p.get("reachable", False)}
        if row["status"] == "reachable":
            # the row must exist as written: non-negative counts, admitted differences, sum exactly S
            assert all(n > 0 and adm.get(e) for e, n in row["mult"].items()), (ci, c)
            assert sum(n * Bt[e] for e, n in row["mult"].items()) == c["S"], (ci, c)
            q = [v for e, n in sorted(row["mult"].items(), reverse=True) for v in [127 + e] * n]
            pos = q.index(127 + c["d"])
            o = subprocess.run([ORACLE], input=oline(CAL[ci], q) + "\n", capture_output=True, text=True).stdout.split()
            rec.update(row_length=len(q), row_distinct_codes=len(row["mult"]),
                       row_code_counts={str(127 + e): n for e, n in sorted(row["mult"].items())})
            if len(o) >= 2 and o[0] == "1" and o[1] == "1":
                kb = int(o[2 + pos]); rec.update(kernel_byte=kb, witness_row=kb == c["source_byte"] != c["exact_byte"])
            else:
                rec.update(kernel="abort", witness_row=False)
        out.append(rec)
st = Counter(r["pair_status"] for r in out)
res = dict(
    superset_candidates=len(out),
    reachable_pairs=st["reachable"],
    proved_unreachable_pairs=st["unreachable_negative"] + st["unreachable_gap"],
    proved_unreachable_by_reason={k: v for k, v in st.items() if k != "reachable"},
    unknown_pairs=len(out) - sum(st.values()),
    kernel_confirmed_witness_rows=sum(r.get("witness_row", False) for r in out),
    reach63_rows_reconfirmed=sum(r["reach63_found_row"] and r["pair_status"] == "reachable" for r in out),
    reach63_unknown_now_reachable=sum((not r["reach63_found_row"]) and r["pair_status"] == "reachable" for r in out),
    reach63_unknown_now_unreachable=sum((not r["reach63_found_row"]) and r["pair_status"] != "reachable" for r in out),
    calibrations_with_witness=sorted({r["cal"] for r in out if r.get("witness_row")}),
    rows=out)
json.dump(res, open(RESULT_PATH, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
