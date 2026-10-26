# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""D-TFLP Amendment 6.5 P2: durable reciprocal-witness fixtures, regenerable without the exhaustive search.

Usage: fixtures65.py SOFTCERT_RUN.json TRACE63 ORACLE OUTDIR

Input is a softcert65.py result (the gate's decisions, already validated). Nothing here repeats the
3.75e9-value superset: the gate (boundary-pruned search + exact reachability) is what produces the
witnesses, and this script re-runs every witness row through the TFLM kernel to record the kernel byte.

Writes to OUTDIR:
  softmax_tables65.json     prepared source weights per calibration (admitted, A = exp, B = Rescale<12>),
                            straight from trace63 `softexp`, with each table's SHA-256;
  witness65_full.jsonl      every kernel-confirmed witness (the complete validation tier);
  witness65_reduced.jsonl   the reduced regression suite (selection rules in the "why" field);
  equivalence65.jsonl       PROVED_OUTPUT_EQUIVALENT rows (calibration 9) with full kernel bytes;
  unreachable65.jsonl       representative arithmetic candidates proved unreachable;
  manifest65.json           SHA-256 of every file above, the source/kernel identity and the commands.
"""
import hashlib, json, random, subprocess, sys
from pathlib import Path
import numpy as np
RUN, TRACE, ORACLE, OUTDIR = sys.argv[1:5]
out = Path(OUTDIR); out.mkdir(parents=True, exist_ok=True)
run = json.load(open(RUN))
KERNEL = "TFLM 5f64e4ad tensorflow/lite/micro/kernels/internal/reference/softmax.h, int8, g++ 13.3.0 -Os/-O2 baseline library, x86_64"
def f32bits(v): return format(np.float32(v).view(np.uint32), "08x")
def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"))

runs = [r for r in run["runs"] if r["group"] == "known"]
tables, tdig = {}, {}
for r in runs:
    s, z, b = r["cal"]
    t = subprocess.run([TRACE], input=f"softexp {f32bits(s)} {f32bits(b)}\n", capture_output=True, text=True, check=True).stdout.splitlines()
    rows = [l.split() for l in t[1:]]
    tab = dict(scale_bits=f32bits(s), beta_bits=f32bits(b), P=list(map(int, t[0].split()[1:])),
               d=[int(x[1]) for x in rows], admitted=[int(x[2]) for x in rows], A=[int(x[4]) for x in rows], B=[int(x[5]) for x in rows])
    key = f"{tab['scale_bits']}:{tab['beta_bits']}"
    tables[key] = tab; tdig[key] = hashlib.sha256(canon(tab).encode()).hexdigest()

def kernel(cal, rows):
    s, z, b = cal
    text = "".join(f"{len(q)} {f32bits(s)} {z} {f32bits(b)} " + " ".join(map(str, q)) + "\n" for q in rows)
    res = subprocess.run([ORACLE], input=text, capture_output=True, text=True).stdout.splitlines()
    assert len(res) == len(rows)
    return [list(map(int, l.split()[2:])) for l in res]

full = []
for r in runs:
    s, z, b = r["cal"]; key = f"{f32bits(s)}:{f32bits(b)}"
    rows = [[127 + int(d) for d, n in sorted(w["mult"].items(), key=lambda t: -int(t[0])) for _ in range(n)] for w in r["witnesses"]]
    kb = kernel(r["cal"], rows)
    for w, q, k in zip(r["witnesses"], rows, kb):
        pos = q.index(127 + w["d"])
        assert k[pos] == w["bs"] != w["bx"], "kernel does not confirm the witness"
        S = w["S"]; h = 32 - S.bit_length()
        full.append(dict(cal_index=r["index"], scale_bits=f32bits(s), zero_point=z, beta_bits=f32bits(b), table=key,
                         table_sha256=tdig[key], S=S, band_h=h, exponent=35 - h, d=w["d"], weight=w["w"],
                         row={str(127 + int(d)): n for d, n in w["mult"].items()}, row_length=len(q), distinct_codes=len(w["mult"]),
                         sh_source=w["sh_src"], sh_exact=w["sh_exact"], byte_source=w["bs"], byte_exact=w["bx"], kernel_byte=k[pos],
                         kernel=KERNEL))
for i, x in enumerate(full): x["id"] = f"W{i:03d}"

def threshold_margin(x):
    # distance of the nearer reciprocal arm to the byte-transition threshold T_m between the two bytes
    ex, w = x["exponent"], x["weight"]; m = min(x["byte_source"], x["byte_exact"]) + 128
    beta = (m << ex) + (1 << (ex - 1)); T = -(-(beta * (1 << 31) - (1 << 30)) // w)
    return min(abs(x["sh_source"] - T), abs(x["sh_exact"] - T)), T

for x in full:
    x["threshold_margin"], x["threshold"] = threshold_margin(x)
red = {}
def pick(x, why): red.setdefault(x["id"], dict(x)).setdefault("why", []).append(why)
for ci in sorted({x["cal_index"] for x in full}):
    pick(min((x for x in full if x["cal_index"] == ci), key=lambda x: (x["row_length"], x["id"])), f"shortest witness of divergent calibration {ci}")
for h in sorted({x["band_h"] for x in full}):
    pick(min((x for x in full if x["band_h"] == h), key=lambda x: (x["row_length"], x["id"])), f"exponent band h={h}")
for x in sorted(full, key=lambda x: (x["threshold_margin"], x["id"]))[:3]: pick(x, f"threshold endpoint (margin {x['threshold_margin']})")
pick(min(full, key=lambda x: (x["row_length"], x["id"])), "minimum row length")
pick(max(full, key=lambda x: (x["row_length"], x["id"])), "maximum row length")
pick(min(full, key=lambda x: (x["distinct_codes"], x["id"])), "fewest distinct codes")
pick(max(full, key=lambda x: (x["distinct_codes"], x["id"])), "most distinct codes")
pick(max(full, key=lambda x: (max(x["byte_source"], x["byte_exact"]), x["id"])), "nearest +127 saturation")
pick(min(full, key=lambda x: (min(x["byte_source"], x["byte_exact"]), x["id"])), "nearest -128 saturation")
reduced = sorted(red.values(), key=lambda x: x["id"])

# calibration 9: PROVED_OUTPUT_EQUIVALENT; rows with their full kernel bytes
eqv = []
r9 = next(r for r in runs if r["decision"] == "PROVED_OUTPUT_EQUIVALENT")
s, z, b = r9["cal"]; key = f"{f32bits(s)}:{f32bits(b)}"; tab = tables[key]
adm = [d for d, a in zip(tab["d"], tab["admitted"]) if a]
rng = random.Random(6509)
rows = [[127] + [127 + rng.choice(adm) for _ in range(n - 1)] for n in (2, 3, 16, 64, 256, 511)] + [[127] * 511]
for i, (q, k) in enumerate(zip(rows, kernel(r9["cal"], rows))):
    eqv.append(dict(id=f"E{i}", cal_index=r9["index"], scale_bits=f32bits(s), zero_point=z, beta_bits=f32bits(b), table=key, table_sha256=tdig[key],
                    row=q, kernel_bytes=k, kernel=KERNEL))

unr = []
for r in runs:
    for u in r["unreachable_pairs"]:
        if u["status"] not in {x["status"] for x in unr if x["cal_index"] == r["index"]}:
            s, z, b = r["cal"]; key = f"{f32bits(s)}:{f32bits(b)}"
            unr.append(dict(cal_index=r["index"], table=key, table_sha256=tdig[key], S=u["S"], d=u["d"], status=u["status"]))

files = {"softmax_tables65.json": json.dumps(dict(tables=tables, sha256=tdig), indent=1) + "\n",
         "witness65_full.jsonl": "".join(canon(x) + "\n" for x in full),
         "witness65_reduced.jsonl": "".join(canon(x) + "\n" for x in reduced),
         "equivalence65.jsonl": "".join(canon(x) + "\n" for x in eqv),
         "unreachable65.jsonl": "".join(canon(x) + "\n" for x in unr)}
for n, t in files.items(): (out / n).write_text(t)
man = dict(source="TFLM 5f64e4ad (tensorflow/ tree identical to 8f6596d)", kernel=KERNEL,
           certificate="softcert65_global.txt", generator=["softcert65.py SOFTCERT TRACE63 ORACLE run.json", "fixtures65.py run.json TRACE63 ORACLE OUTDIR"],
           counts=dict(full=len(full), reduced=len(reduced), equivalence_rows=len(eqv), unreachable=len(unr), tables=len(tables)),
           sha256={n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()})
(out / "manifest65.json").write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps(man, indent=1))
