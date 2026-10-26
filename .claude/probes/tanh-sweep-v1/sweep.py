# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""TANH int8 calibration sweep (Amendment 6.1, P0).

Usage: sweep.py TFLM_SWEEP_BIN VSTANH_DUMP_BIN OUT.json [N_RANDOM]

Per input calibration (f32 scale, zero point), over all 256 int8 input codes, output
contract scale 1/128, zero point 0 (the TFLite int8 TANH output contract):

  tflm      the real TFLM kernel (tanh_int8_sweep.cc, linked against libtensorflow-microlite)
  ref_real  correctly rounded tanh of the EXACT real input scale*(c - zp)   (TFLite spec input)
  ref_f32   correctly rounded tanh of Codec8's decoded f32 (c - zp) as f32 * scale  (plan D7)
  lut       certified prepared code->code table (interval arithmetic) on the f32 decode
  libm      numpy float32 tanh (libm tanhf) then f32 round-half-away
  vstanh    ndarray::hpc::vml::vstanh then f32 round-half-away
Output codes round half away from zero (TfliteAct); tanh of a nonzero rational is
transcendental, so no tie can occur except tanh(0) = 0 (proved, not assumed: checked).
"""
import json, subprocess, sys, math
import numpy as np, mpmath as mp
from fractions import Fraction

TFLM, VST, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NR = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
rng = np.random.default_rng(61)
mp.mp.dps = 60

f32 = np.float32
FIXTURE7 = [(1/128, 0), (0.05, -3), (0.1, 20), (0.0137, -128), (0.5, 127), (0.0001, 0), (0.0234375, 60)]
cal = [(f32(s), z, "fixture7") for s, z in FIXTURE7]
cal += [(f32(2.0 ** -k), 0, "pow2_zp0") for k in range(0, 17)]
cal += [(f32(10 ** rng.uniform(-5, 0.7)), int(rng.integers(-128, 128)), "random") for _ in range(NR)]
cal += [(f32(10 ** rng.uniform(-5, 0.7)), 0, "symmetric_zp0") for _ in range(200)]

def bits(x): return format(np.float32(x).view(np.uint32), "08x")
inp = "".join(f"{bits(s)} {z}\n" for s, z, _ in cal)
raw = subprocess.run([TFLM], input=inp, capture_output=True, text=True, check=True).stdout.splitlines()
assert len(raw) == len(cal)

def rh_away(x):  # round half away from zero on an exact Fraction / mpf-free path
    a = abs(x); r = math.floor(a + Fraction(1, 2)); return r if x >= 0 else -r
def code_of(t128):  # t128 = 128*tanh as Fraction-convertible exact value
    return max(-128, min(127, rh_away(t128)))

def frac(x):
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * (Fraction(2) ** int(exp)) if man else Fraction(0)
    return -v if sign else v

def ref_code(v):  # v: exact Fraction; mpmath 60 digits; returns (code, distance of 128 tanh to nearest .5)
    if v == 0: return 0, Fraction(1, 2)
    t = 128 * mp.tanh(mp.mpf(v.numerator) / v.denominator)
    tf = frac(t); d = abs(abs(tf) - (math.floor(abs(tf)) + Fraction(1, 2)))
    return code_of(tf), d

def lut_code(v):  # certified: interval tanh = (e^{2v}-1)/(e^{2v}+1), narrow until the code is decided
    if v == 0: return 0, 0
    for prec in (64, 128, 256, 512):
        mp.iv.prec = prec
        x = mp.iv.mpf(v.numerator) / v.denominator
        e = mp.iv.exp(2 * x); t = 128 * (e - 1) / (e + 1)
        lo, hi = frac(t.a), frac(t.b)
        if code_of(lo) == code_of(hi): return code_of(lo), prec
    raise RuntimeError(f"LUT cell not decided at 512 bits for v={v}")

cases, inv = [], dict(tflm_vs_ref_real=0, tflm_vs_ref_f32=0, lut_vs_ref_f32=0, ref_real_vs_ref_f32=0,
                      libm_vs_ref_f32=0, vstanh_vs_ref_f32=0, prepare_fail=0, invoke_fail=0, cells=0)
mism = []
for (s, z, kind), line in zip(cal, raw):
    t = line.split(); assert t[0] == bits(s) and int(t[1]) == z
    prep, invk, mult, shift, radius = (int(t[2]), int(t[3]), int(t[4]), int(t[5]), int(t[6]))
    tflm = [int(v) for v in t[7:]]
    codes = list(range(-128, 128))
    xr = [Fraction(float(s)) * (c - z) for c in codes]                       # exact real input
    x32 = [f32(f32(c - z) * s) for c in codes]                               # Codec8 decode (f32 product)
    xf = [Fraction(float(v)) for v in x32]
    rr = [ref_code(v) for v in xr]; rf = [ref_code(v) for v in xf]
    lut = [lut_code(v) for v in xf]
    # native arms: f32 tanh then round half away in f32 (exact: |t*128| <= 128, t*128 exact)
    lm = np.tanh(np.array(x32, dtype=np.float32)).astype(np.float32)
    vs = subprocess.run([VST], input=" ".join(bits(v) for v in x32), capture_output=True, text=True, check=True).stdout.split()
    vsv = np.array([np.uint32(int(h, 16)) for h in vs], dtype=np.uint32).view(np.float32)
    def rf32(arr): return [max(-128, min(127, int(math.copysign(math.floor(abs(float(a) * 128) + 0.5), float(a))))) for a in arr]
    lmc, vsc = rf32(lm), rf32(vsv)
    inv["cells"] += 256; inv["prepare_fail"] += prep == 0; inv["invoke_fail"] += invk == 0
    rec = dict(scale=float(s), scale_bits=bits(s), zp=z, kind=kind, prepare_ok=bool(prep), invoke_ok=bool(invk),
               input_multiplier=mult, input_left_shift=shift, input_range_radius=radius, diffs={})
    for name, arr in (("tflm_vs_ref_real", tflm), ("tflm_vs_ref_f32", tflm), ("lut_vs_ref_f32", [c for c, _ in lut]),
                      ("libm_vs_ref_f32", lmc), ("vstanh_vs_ref_f32", vsc), ("ref_real_vs_ref_f32", [c for c, _ in rr])):
        ref = rr if name == "tflm_vs_ref_real" else rf
        d = [i for i in range(256) if arr[i] != ref[i][0]]
        inv[name] += len(d)
        if d:
            rec["diffs"][name] = len(d)
            for i in d[:20]:
                c = codes[i]
                mism.append(dict(arm=name, scale_bits=bits(s), zp=z, code=c, got=arr[i], ref=ref[i][0],
                                 x_real=float(xr[i]), x_f32=float(x32[i]),
                                 dist_128tanh_to_half=float(ref[i][1]),
                                 beyond_radius=bool(abs(c - z) > radius) if prep else None))
    rec["lut_max_bits"] = max(b for _, b in lut)
    # endpoint symmetry: the code space is -128..127, so saturation is asymmetric by construction
    rec["tflm_min"], rec["tflm_max"] = min(tflm), max(tflm)
    rec["ref_min"], rec["ref_max"] = min(c for c, _ in rf), max(c for c, _ in rf)
    cases.append(rec)

# Sym + NaNAtMin semantics on zp = 0 rows: code -128 means NaN, its output is the NaN code (-128).
# TFLM has no NaN code and evaluates -128 as a value. This is a semantic difference, never a numeric one.
sym = [c for c in cases if c["zp"] == 0]
sem = dict(rows=len(sym), note="TFLM evaluates input code -128 as a value; Sym+NaNAtMin maps it to the NaN code. "
           "Not counted as a mismatch: different contracts.")
# U8 storage (code ^ 0x80) is a permutation of the same table: checked on every row by construction
by_kind = {}
for c in cases:
    k = by_kind.setdefault(c["kind"], dict(rows=0, rows_with_tflm_vs_ref_real_diff=0, cells_tflm_vs_ref_real=0))
    k["rows"] += 1; d = c["diffs"].get("tflm_vs_ref_real", 0)
    k["rows_with_tflm_vs_ref_real_diff"] += d > 0; k["cells_tflm_vs_ref_real"] += d
json.dump(dict(calibrations=len(cal), inventory=inv, by_kind=by_kind, sym_nan_semantics=sem,
               mismatch_samples=mism[:400], cases=cases), open(OUT, "w"), indent=0)
print(json.dumps(dict(calibrations=len(cal), inventory=inv, by_kind=by_kind), indent=1))
