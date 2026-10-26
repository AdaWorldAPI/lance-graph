# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""D-TFLP Amendment 6.5 P1: run the softmax reciprocal admission gate and validate every decision independently.

Usage: softcert65.py SOFTCERT TRACE63 ORACLE OUT.json [--mutations]

Per calibration (the 14 of softmax63.py and the 8 held-out ones of heldout65.py):
  1. prepare the source weights (trace63 `softexp`);
  2. run `softcert65 gate` with the committed source-global certificate (softcert65_global.txt);
  3. validate the decision WITHOUT the gate's code:
     - table:      the weight table is recomputed from the claimed calibration identity; its digest must match;
     - candidates: the gate's candidate set must equal the exhaustive superset reference
                   (recipall63.json for the 14, recipall65_heldout.json for the held-out ones);
     - witnesses:  every reachable claim needs a row with positive counts of admitted codes that contains
                   the row maximum and whose recomputed sum is S; the independent Python recipe
                   (softmax63.py) must give the claimed source byte and, with the exact reciprocal, the
                   claimed alternative byte; the TFLM kernel must give the source byte;
     - unreachable: a negative remainder is recomputed; a gap claim is re-checked by a plain unbounded-coin
                   DP on Python integers when R <= 4,000,000 (otherwise UNVALIDATED);
     - equivalence: for a PROVED_OUTPUT_EQUIVALENT decision, 300 random admitted rows must give identical
                   bytes from the kernel, the source recipe and the exact-reciprocal recipe.
With --mutations, every certifier mutation and three input mutations are run on the 14 known calibrations
and each run is classified as typed refusal, detected by validation, or no observable effect.
"""
import hashlib, json, random, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
HERE = Path(__file__).resolve().parent
CERT_BIN, TRACE, ORACLE, OUT, *flags = sys.argv[1:]
MUTATE = "--mutations" in flags
TFLM = Path("/home/user/adaworldapi/tflite-micro/tensorflow/lite/micro/kernels/internal")
REV = "5f64e4ad"
DIGEST = hashlib.sha256(b"".join((TFLM / f).read_bytes() for f in ("common.h", "fixedpoint.h", "reference/softmax.h"))).hexdigest()
CERT = HERE / "softcert65_global.txt"

src = (HERE / "softmax63.py").read_text()
g = {"__name__": "softcert65_helpers"}; argv = sys.argv; sys.argv = ["x", "x", "x", "/dev/null"]
exec(src[: src.index("# --- calibrations and rows")], g); sys.argv = argv
_d = {"random": random}; exec(src[src.index("rng = random.Random(63)"): src.index("NS = [")], _d); CAL = _d["CAL"]
import heldout65
HELD = heldout65.HELDOUT
f32bits = g["f32bits"]
def rh_away(x): a = abs(x); r = (a + Fraction(1, 2)).__floor__(); return r if x >= 0 else -r
g["rh_away"] = rh_away
g["prep"] = prep = {}
exec(src[src.index("# --- recipe ----"): src.index("# --- F24 exact-first")], g)
recipe = g["recipe"]

def table(cal):
    s, z, b = cal
    t0 = time.perf_counter()
    out = subprocess.run([TRACE], input=f"softexp {f32bits(s)} {f32bits(b)}\n", capture_output=True, text=True, check=True).stdout
    dt = time.perf_counter() - t0
    lines = out.splitlines()
    m, sh, dmin = map(int, lines[0].split()[1:])
    E = {int(t[1]): (int(t[2]) == 1, int(t[4]), int(t[5])) for t in (l.split() for l in lines[1:])}
    prep[(s, b)] = dict(mult=m, shift=sh, diff_min=dmin, E=E)
    return out, dt

def gate(cal, tab, mutation=None, rev=REV, dig=DIGEST):
    s, z, b = cal
    t0 = time.perf_counter()
    r = subprocess.run([CERT_BIN, "gate", str(CERT), rev, dig] + ([mutation] if mutation else []),
                       input=f"K {f32bits(s)} {z} {f32bits(b)}\n" + tab, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    res = dict(rc=r.returncode, wall_s=wall, timings={}, cands=[], decision=None, reason=None)
    cur = None
    for l in r.stdout.splitlines():
        t = l.split()
        if t[0] == "T": res["timings"][t[1]] = float(t[2])
        elif t[0] == "I": res["info"] = dict(zip(t[1::2], map(int, t[2::2])))
        elif t[0] == "C" and len(t) >= 9:
            cur = dict(S=int(t[1]), d=int(t[2]), w=int(t[3]), sh_src=int(t[4]), sh_exact=int(t[5]), bs=int(t[6]), bx=int(t[7]), status=t[8], mult={})
            res["cands"].append(cur)
        elif t[0] == "M" and cur is not None: cur["mult"][int(t[1])] = int(t[2])
        elif t[0] == "D": res["decision"], res["reason"] = t[1], t[2]
    return res

def gap_unreachable(R, coins):
    if R > 4_000_000: return None
    mask = (1 << (R + 1)) - 1; reach = 1
    while True:
        prev = reach
        for c in coins:
            x, step = reach, c
            while step <= R:
                x |= (x << step) & mask; step *= 2
            reach = x
        if reach == prev: break
    return not (reach >> R) & 1

def oracle_rows(cal, rows):
    s, z, b = cal
    text = "".join(f"{len(q)} {f32bits(s)} {z} {f32bits(b)} " + " ".join(map(str, q)) + "\n" for q in rows)
    r = subprocess.run([ORACLE], input=text, capture_output=True, text=True)
    out = []
    for l in r.stdout.splitlines():
        t = l.split(); out.append(list(map(int, t[2:])) if t[:2] == ["1", "1"] else None)
    return out

def validate(cal, tab, res, ref_set, claimed_cal=None):
    s, z, b = claimed_cal or cal
    v = dict(table_ok=None, candidates_equal=None, witness_ok=0, witness_bad=[], unreach_ok=0, unreach_bad=[], unreach_unvalidated=0, equivalence_rows=None)
    t2, _ = table((s, z, b))
    v["table_ok"] = hashlib.sha256(t2.encode()).hexdigest() == hashlib.sha256(tab.encode()).hexdigest()
    E = prep[(s, b)]["E"]
    v["candidates_equal"] = {(c["S"], c["d"], c["bs"], c["bx"]) for c in res["cands"]} == ref_set
    B = {d: E[d][2] if E[d][0] else 0 for d in E}
    coins = sorted({B[d] for d in B if E[d][0] and B[d] > 0})
    wrows, wcands = [], []
    for c in res["cands"]:
        if c["status"] == "reachable":
            m = c["mult"]
            ok = bool(m) and all(n > 0 and E[d][0] for d, n in m.items()) and m.get(0, 0) >= 1 and m.get(c["d"], 0) >= 1 \
                and sum(n * B[d] for d, n in m.items()) == c["S"]
            if ok:
                q = [127 + d for d, n in sorted(m.items(), reverse=True) for _ in range(n)]
                pos = q.index(127 + c["d"])
                a, x = recipe((s, z, b), q), recipe((s, z, b), q, ("reciprocal",))
                ok = isinstance(a, list) and isinstance(x, list) and a[pos] == c["bs"] and x[pos] == c["bx"] and a[pos] != x[pos]
                if ok: wrows.append(q); wcands.append((c, pos))
            if not ok: v["witness_bad"].append(dict(S=c["S"], d=c["d"]))
        else:
            R = c["S"] - B[0] - (B[c["d"]] if c["d"] != 0 else 0)
            if c["status"] == "unreachable_negative":
                (v.__setitem__("unreach_ok", v["unreach_ok"] + 1) if R < 0 else v["unreach_bad"].append(c["S"]))
            else:
                u = gap_unreachable(R, coins) if R >= 0 else False
                if u is None: v["unreach_unvalidated"] += 1
                elif u: v["unreach_ok"] += 1
                else: v["unreach_bad"].append(c["S"])
    if wrows:
        ko = oracle_rows((s, z, b), wrows)
        for (c, pos), kb in zip(wcands, ko):
            if kb is not None and kb[pos] == c["bs"]: v["witness_ok"] += 1
            else: v["witness_bad"].append(dict(S=c["S"], d=c["d"], kernel=None if kb is None else kb[pos]))
    if res["decision"] == "PROVED_OUTPUT_EQUIVALENT":
        rng = random.Random(65)
        adm = [d for d in E if E[d][0]]
        rows = []
        for _ in range(300):
            n = rng.choice([2, 3, 8, 64, 256, 511])
            q = [127] + [127 + rng.choice(adm) for _ in range(n - 1)]
            rows.append(q)
        ko = oracle_rows((s, z, b), rows); same = 0; checked = 0
        for q, kb in zip(rows, ko):
            a, x = recipe((s, z, b), q), recipe((s, z, b), q, ("reciprocal",))
            if kb is None or not isinstance(a, list): continue
            checked += 1; same += kb == a == x
        v["equivalence_rows"] = dict(checked=checked, identical=same)
    return v

def ref_sets(path):
    r = json.load(open(HERE / path)); out = {}
    for c in r["candidate_samples"]: out.setdefault(c["cal"], set()).add((c["S"], c["d"], c["source_byte"], c["exact_byte"]))
    for k in r["per_calibration"]: out.setdefault(int(k), set())
    return out

def verdict(res, v, expect=None):
    if res["decision"] == "REFUSE_UNKNOWN": return "typed_refusal:" + res["reason"]
    problems = []
    if not v["table_ok"]: problems.append("table_digest")
    if not v["candidates_equal"]: problems.append("candidate_set")
    if v["witness_bad"]: problems.append("witness")
    if v["unreach_bad"]: problems.append("unreachable_claim")
    if v["equivalence_rows"] and v["equivalence_rows"]["identical"] != v["equivalence_rows"]["checked"]: problems.append("equivalence_rows")
    if expect and res["decision"] != expect: problems.append("decision")
    return "detected:" + "+".join(problems) if problems else "valid"

t_all = time.perf_counter()
cert_text = CERT.read_text()
assert f"source_revision {REV}" in cert_text and f"source_digest {DIGEST}" in cert_text, "committed certificate does not match the pinned source"
REF_K = ref_sets("recipall63.json")
REF_H = ref_sets("recipall65_heldout.json") if (HERE / "recipall65_heldout.json").exists() else None
runs = []
for group, cals, refs in (("known", CAL, REF_K), ("heldout", HELD, REF_H)):
    if refs is None:
        print("held-out reference missing: held-out group NOT RUN"); continue
    for i, cal in enumerate(cals):
        tab, t_tab = table(cal)
        res = gate(cal, tab)
        v = validate(cal, tab, res, refs[i])
        expect = ("PROVED_OUTPUT_EQUIVALENT" if i == 9 else "REQUIRES_SOURCE_RECIPROCAL") if group == "known" else None
        runs.append(dict(group=group, index=i, cal=cal, decision=res["decision"], reason=res["reason"],
                         candidates=len(res["cands"]), reachable=sum(c["status"] == "reachable" for c in res["cands"]),
                         unreachable=sum(c["status"] != "reachable" for c in res["cands"]), info=res.get("info"),
                         timings=dict(table_s=t_tab, gate_wall_s=res["wall_s"], **res["timings"]),
                         validation=v, verdict=verdict(res, v, expect),
                         witnesses=[dict(S=c["S"], d=c["d"], w=c["w"], sh_src=c["sh_src"], sh_exact=c["sh_exact"], bs=c["bs"], bx=c["bx"], mult=c["mult"])
                                    for c in res["cands"] if c["status"] == "reachable"],
                         unreachable_pairs=[dict(S=c["S"], d=c["d"], status=c["status"]) for c in res["cands"] if c["status"] != "reachable"]))
        print(group, i, res["decision"], len(res["cands"]), runs[-1]["reachable"], runs[-1]["verdict"], f"{res['wall_s']:.2f}s", flush=True)

mut = []
if MUTATE:
    MUTS = ["e_minus1", "e_minus2", "one_sided_window", "exclude_lower_endpoint", "wrong_band", "wrong_weight_threshold", "no_mandatory_max",
            "no_reachability", "allow_negative_multiplicity", "witness_sum_off_by_max"]
    for m in MUTS + ["input:wrong_revision", "input:wrong_digest", "input:wrong_calibration"]:
        per = []
        for i, cal in enumerate(CAL):
            tab, _ = table(cal)
            claimed = None
            if m == "input:wrong_revision": res = gate(cal, tab, rev="8f6596d")
            elif m == "input:wrong_digest": res = gate(cal, tab, dig="0" * 64)
            elif m == "input:wrong_calibration":
                other = CAL[(i + 1) % len(CAL)]; tab, _ = table(other); res = gate(cal, tab); claimed = cal
            else: res = gate(cal, tab, m)
            if res["decision"] is None: per.append("crash"); continue
            v = validate(cal, tab, res, REF_K[i], claimed) if res["decision"] != "REFUSE_UNKNOWN" else None
            per.append(verdict(res, v, "PROVED_OUTPUT_EQUIVALENT" if i == 9 else "REQUIRES_SOURCE_RECIPROCAL"))
        eff = sum(p != "valid" for p in per)
        mut.append(dict(mutation=m, per_calibration=per, calibrations_affected=eff, effective=eff > 0))
        print("MUT", m, eff, "/ 14", sorted(set(per)), flush=True)

json.dump(dict(certificate=cert_text.splitlines(), source_digest=DIGEST, runs=runs, mutations=mut,
               wall_s=round(time.perf_counter() - t_all, 1)), open(OUT, "w"), indent=1)
