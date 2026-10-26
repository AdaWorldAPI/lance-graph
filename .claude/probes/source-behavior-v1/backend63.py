# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3 addendum: compiler/backend differential of the pinned TFLM reference kernels.

Usage: backend63.py TFLM_ROOT BUILD_DIR OUT.json VARIANT [VARIANT ...]

Each VARIANT names a TFLM microlite build at TFLM_ROOT/gen_<VARIANT>/... (baseline: the original
TFLM_ROOT/gen/linux_x86_64_default_gcc). For each variant the three unchanged oracle sources
(tanh-sweep-v1/tanh_int8_sweep.cc, source-behavior-v1/trace63.cc, fisherz-softmax-v1/softmax_oracle.cc)
are linked against that variant's library, then run on one fixed corpus:
  TANH      2,224 calibrations of Amendment 6.1 (sweep.json), 256 codes each;
  LOGISTIC  the same 2,224 input calibrations (output 1/256, zp -128), via trace63 `ltrace`, which also
            aborts unless its stage trace equals the kernel byte;
  SOFTMAX   the 840 rows of softmax63.py (same seed) plus OVERFLOW rows: ties of length
            4096..16385 whose int32 sum of Q12 weights wraps (fixedpoint.h:91-95 Add is undefined on
            overflow).
Output bytes are compared row by row against the first variant listed.
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT, BUILD, RESULT_PATH, *VARIANTS = sys.argv[1:]
BUILD = Path(BUILD); BUILD.mkdir(parents=True, exist_ok=True)

# corpus: softmax63's calibrations and rows, regenerated with its seed (sections before the prepare step)
src = (HERE / "softmax63.py").read_text()
g = {"__name__": "softmax63_corpus"}
sys_argv = sys.argv; sys.argv = ["x", "x", "x", "/dev/null"]
exec(src[: src.index("# --- prepared per-calibration weights")], g)
sys.argv = sys_argv
CAL, cases = g["CAL"], g["cases"]
def oline(cal, q):  # identical to softmax63.oline
    s, z, b = cal; return f"{len(q)} {g['f32bits'](s)} {z} {g['f32bits'](b)} " + " ".join(map(str, q))
OVERFLOW_N = [4095, 4096, 4097, 6000, 8191, 8192, 8193, 8194, 8300, 8703, 8704, 12288, 16384, 16385]
over = [((0.05, 0, 1.0), f"ties_{n}", [37] * n) for n in OVERFLOW_N]
sweep = json.load(open(HERE.parent / "tanh-sweep-v1/sweep.json"))["cases"]
TCAL = [(c["scale_bits"], c["zp"]) for c in sweep]

FLAGS = ("-std=c++17 -Os -fno-rtti -fno-exceptions -fno-threadsafe-statics -DTFLM_REMOVE_VIRTUAL_DELETE "
         "-DTF_LITE_DISABLE_X86_NEON -DKERNELS_OPTIMIZED_FOR_SPEED -DTF_LITE_USE_CTIME -Wall "
         "-Wno-unused-parameter -Wno-unused-result").split()
def variant(v):
    if v == "baseline": return "g++", f"{ROOT}/gen/linux_x86_64_default_gcc", []
    tc = "clang" if v.startswith("clang") else "gcc"
    cx = "clang++" if tc == "clang" else "g++"
    extra = ["-fsanitize=undefined", "-fno-sanitize-recover=undefined"] if "ubsan" in v else []
    return cx, f"{ROOT}/gen_{v}/linux_x86_64_default_{tc}", extra
SRCS = {"tanh": HERE.parent / "tanh-sweep-v1/tanh_int8_sweep.cc", "trace": HERE / "trace63.cc",
        "softmax": HERE.parent / "fisherz-softmax-v1/softmax_oracle.cc"}
def build(v):
    cx, gen, extra = variant(v); out = {}
    inc = [f"-I{ROOT}", f"-I{ROOT}/tensorflow/lite/micro/tools/make/downloads",
           f"-I{ROOT}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include",
           f"-I{ROOT}/tensorflow/lite/micro/tools/make/downloads/kissfft", f"-I{gen}/genfiles/"]
    for k, s in SRCS.items():
        b = BUILD / f"{k}_{v}"
        if not b.exists():
            subprocess.run([cx, *FLAGS, *extra, *inc, str(s), f"{gen}/lib/libtensorflow-microlite.a",
                            "-Wl,--gc-sections", "-lm", "-o", str(b)], check=True)
        out[k] = str(b)
    return out
def run(b, text): return subprocess.run([b], input=text, capture_output=True, text=True)
def sha(x): return hashlib.sha256(x.encode()).hexdigest()[:16]

def measure(v):
    B = build(v); res = dict(variant=v, compiler=None)
    cx = variant(v)[0]
    res["compiler"] = subprocess.run([cx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    r = run(B["tanh"], "".join(f"{b} {z}\n" for b, z in TCAL))
    res["tanh_rc"] = r.returncode; res["tanh"] = r.stdout.splitlines()
    lg = []
    for i in range(0, len(TCAL), 100):
        r = run(B["trace"], "".join(f"ltrace {b} {z}\n" for b, z in TCAL[i:i + 100]))
        if r.returncode: lg.append(f"ABORT {r.returncode} {r.stderr.strip()[:200]}"); continue
        rows = [l.split() for l in r.stdout.splitlines()]
        for j in range(0, len(rows), 256): lg.append(" ".join(t[14] for t in rows[j:j + 256]))
    res["logistic"] = lg
    sm = []
    for cal, dn, q in [(c, d, q) for c, d, q in cases] + over:
        r = run(B["softmax"], oline(cal, q) + "\n")
        if r.returncode == 0 and r.stdout: sm.append(r.stdout.strip())
        else: sm.append(f"ABORT rc={r.returncode} {(r.stderr.strip().splitlines() or [''])[0][:160]}")
    res["softmax"] = sm
    return res

R = [measure(v) for v in VARIANTS]
base = R[0]
summary = []
for x in R:
    row = dict(variant=x["variant"], compiler=x["compiler"], tanh_rc=x["tanh_rc"],
               tanh_lines=len(x["tanh"]), tanh_diff_lines=sum(a != b for a, b in zip(x["tanh"], base["tanh"])),
               logistic_tables=len(x["logistic"]),
               logistic_diff_tables=sum(a != b for a, b in zip(x["logistic"], base["logistic"])),
               logistic_aborts=sum(l.startswith("ABORT") for l in x["logistic"]),
               softmax_rows=len(x["softmax"]),
               softmax_diff_rows_corpus=sum(a != b for a, b in zip(x["softmax"][:len(cases)], base["softmax"][:len(cases)])),
               softmax_corpus_aborts=sum(l.startswith("ABORT") for l in x["softmax"][:len(cases)]),
               softmax_admitted_rows_differing=sum(a != b for a, b in zip(x["softmax"][:len(cases)], base["softmax"][:len(cases)])
                                                   if not a.startswith("ABORT") and not b.startswith("ABORT")),
               softmax_diff_rows_detail=sorted({(cases[i][1], len(cases[i][2]), base["softmax"][i].split(" ")[0][:5], x["softmax"][i][:90])
                                                for i in range(len(cases)) if x["softmax"][i] != base["softmax"][i]}),
               overflow_rows={OVERFLOW_N[i]: (l if l.startswith("ABORT") else "OK " + sha(l) + " first=" + " ".join(l.split()[:4]))
                              for i, l in enumerate(x["softmax"][len(cases):])},
               digest_tanh=sha("\n".join(x["tanh"])), digest_logistic=sha("\n".join(x["logistic"])),
               digest_softmax_corpus=sha("\n".join(x["softmax"][:len(cases)])))
    summary.append(row)
json.dump(dict(corpus=dict(tanh_calibrations=len(TCAL), softmax_rows=len(cases), overflow_lengths=OVERFLOW_N),
               results=summary), open(RESULT_PATH, "w"), indent=1)
print(json.dumps(summary, indent=1))
