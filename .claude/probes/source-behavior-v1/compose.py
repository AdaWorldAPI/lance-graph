# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.2: source-behavior-preserving fusion of TFLM int8 TANH -> LOGISTIC.

Usage: compose.py TANH_SWEEP_BIN SOURCE_TRACE_BIN OUT.json [N_CAL]

Chain (a real TFLM chain: TANH's int8 output tensor, scale 1/128 zp 0, is LOGISTIC's input):
  K1 = TANH     (in: calibration under test   -> out: 1/128, 0)
  K2 = LOGISTIC (in: 1/128, 0                 -> out: 1/256, -128; TFLM requires zp -128)

Arms, over every input code of every calibration:
  sequence  two separate TFLM kernel invocations with an int8 tensor between them ("chain" mode)
  fused     one prepared table: fused[q] = L2[L1[q]], from INDEPENDENTLY prepared stage tables
            (L1 from tanh_int8_sweep, L2 from one LOGISTIC call over all 256 codes)
  math_iq   certified math WITH the declared intermediate quantization: Q2(sigma(D1(Q1(tanh(D0 q)))))
  math_cont certified math WITHOUT the intermediate quantization:   Q2(sigma(tanh(D0 q)))
Controls: missing TANH stage; wrong intermediate calibration (1/64 instead of 1/128).
"""
import json, math, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np, mpmath as mp

SWEEP, TRACE, OUT = sys.argv[1:4]
HERE = Path(__file__).resolve().parent
cal = [(c["scale_bits"], c["zp"], c["kind"]) for c in json.load(open(HERE.parent / "tanh-sweep-v1/sweep.json"))["cases"]]
NCAL = int(sys.argv[4]) if len(sys.argv) > 4 else len(cal)
cal = cal[:NCAL]

def f32(bits): return float(np.uint32(int(bits, 16)).view(np.float32))
def rh_away(x): a = abs(x); r = math.floor(a + Fraction(1, 2)); return r if x >= 0 else -r
def clamp8(c): return max(-128, min(127, c))
def frac(x):
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * (Fraction(2) ** int(exp)) if man else Fraction(0)
    return -v if sign else v

def iv_tanh(x):  e = mp.iv.exp(2 * x); return (e - 1) / (e + 1)
def iv_sigma(x): return 1 / (1 + mp.iv.exp(-x))
def q1(iv):  # TANH output contract: clamp(round_away(128 t))
    return clamp8(rh_away(frac((128 * iv).a))), clamp8(rh_away(frac((128 * iv).b)))
def q2(iv):  # LOGISTIC output contract: clamp(round_away(256 s) - 128)
    return clamp8(rh_away(frac((256 * iv).a)) - 128), clamp8(rh_away(frac((256 * iv).b)) - 128)
def certified(fn):
    for prec in (64, 128, 256, 512):
        mp.iv.prec = prec
        lo, hi = fn()
        if lo == hi: return lo
    raise RuntimeError("undecided at 512 bits")
def ivq(v): return mp.iv.mpf(v.numerator) / v.denominator

def run(bin_, text):
    return subprocess.run([bin_], input=text, capture_output=True, text=True, check=True).stdout.splitlines()

t0 = time.perf_counter()
L1 = {}
for (b, z, _), line in zip(cal, run(SWEEP, "".join(f"{b} {z}\n" for b, z, _ in cal))):
    t = line.split(); L1[(b, z)] = [int(v) for v in t[7:]]
lg = run(TRACE, "logistic 3c000000 0\n")[0].split(); assert lg[3] == "1" and lg[4] == "1"
L2 = [int(v) for v in lg[8:]]                      # LOGISTIC table over input codes -128..127 at (1/128, 0)
lg64 = run(TRACE, "logistic 3c800000 0\n")[0].split()
L2_wrong = [int(v) for v in lg64[8:]]              # same op, intermediate declared as 1/64
t_prep_tables = time.perf_counter() - t0

t0 = time.perf_counter()
seq = {}
for (b, z, _), line in zip(cal, run(TRACE, "".join(f"chain {b} {z}\n" for b, z, _ in cal))):
    t = line.split(); assert t[3] == "1"
    seq[(b, z)] = ([int(v) for v in t[4:260]], [int(v) for v in t[260:516]])
t_seq = time.perf_counter() - t0

# LOGISTIC alone at the chain's intermediate calibration: source vs certified
L2_math = [certified(lambda c=c: q2(iv_sigma(ivq(Fraction(c, 128))))) for c in range(-128, 128)]
logistic_diffs = [(c - 128, L2[c], L2_math[c]) for c in range(256) if L2[c] != L2_math[c]]

res = dict(chain=dict(K1="TANH int8 (in: calibration; out: scale 1/128 zp 0)",
                      K2="LOGISTIC int8 (in: scale 1/128 zp 0; out: scale 1/256 zp -128)",
                      connection="K1's int8 output tensor is K2's input tensor; calibrations equal by construction"),
           calibrations=len(cal), cells=256 * len(cal),
           logistic_at_intermediate_source_vs_math=dict(diff_cells=len(logistic_diffs), cells=logistic_diffs))
cnt = dict(fused_vs_sequence=0, L1_vs_sequence_stage1=0, math_iq_vs_sequence=0, math_cont_vs_sequence=0,
           math_cont_vs_math_iq=0, missing_stage_vs_sequence=0, wrong_intermediate_calibration_vs_sequence=0)
examples = dict(math_cont_vs_sequence=[], math_iq_vs_sequence=[], missing_stage_vs_sequence=[])
t_fuse = 0.0; t_math = 0.0
for (b, z, kind) in cal:
    s = Fraction(f32(b)); l1 = L1[(b, z)]; s1, s2 = seq[(b, z)]
    t1 = time.perf_counter()
    fused = [L2[l1[i] + 128] for i in range(256)]                # composition of prepared tables
    t_fuse += time.perf_counter() - t1
    cnt["fused_vs_sequence"] += sum(fused[i] != s2[i] for i in range(256))
    cnt["L1_vs_sequence_stage1"] += sum(l1[i] != s1[i] for i in range(256))
    missing = [L2[i] for i in range(256)]                        # TANH skipped: LOGISTIC on raw codes
    wrongc = [L2_wrong[l1[i] + 128] for i in range(256)]
    t1 = time.perf_counter()
    m_iq, m_cont = [], []
    for i, c in enumerate(range(-128, 128)):
        x = s * (c - z)
        tcode = 0 if x == 0 else certified(lambda: q1(iv_tanh(ivq(x))))
        m_iq.append(L2_math[tcode + 128])
        m_cont.append(certified(lambda: q2(iv_sigma(iv_tanh(ivq(x))))) if x != 0 else 0)
    t_math += time.perf_counter() - t1
    for name, arr in (("math_iq_vs_sequence", m_iq), ("math_cont_vs_sequence", m_cont),
                      ("missing_stage_vs_sequence", missing), ("wrong_intermediate_calibration_vs_sequence", wrongc)):
        d = [i for i in range(256) if arr[i] != s2[i]]
        cnt[name] += len(d)
        if name in examples and d and len(examples[name]) < 12:
            i = d[0]; c = i - 128; x = s * (c - z)
            examples[name].append(dict(scale_bits=b, zp=z, kind=kind, code=c, x=float(x),
                                       tanh_x=float(mp.tanh(mp.mpf(x.numerator) / x.denominator)),
                                       source_intermediate=s1[i], source_final=s2[i], arm_final=arr[i]))
    cnt["math_cont_vs_math_iq"] += sum(m_cont[i] != m_iq[i] for i in range(256))
res["counts"] = cnt; res["examples"] = examples
res["timing_python_s"] = dict(stage_tables_from_source=t_prep_tables, sequence_two_kernel_calls_all=t_seq,
                              fused_table_composition_all=t_fuse, certified_math_arms_all=t_math)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1)[:4000])
for k, v in examples.items(): print(k, v[:2])
