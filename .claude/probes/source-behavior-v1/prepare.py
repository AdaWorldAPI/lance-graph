# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.2: SourceParity vs CertifiedMath TANH tables, trace of the 17 discrepancies,
numerical fingerprints, lab receipts and negative controls.

Usage: prepare.py TANH_SWEEP_BIN SOURCE_TRACE_BIN OUT.json ARTIFACTS.bin

- Calibrations: read from ../tanh-sweep-v1/sweep.json (the Amendment 6.1 list, in order).
- SourceParity table  = bytes of the real TFLM kernel (tanh_int8_sweep, reused, not copied).
- CertifiedMath table = interval-certified Q_out(tanh(scale*(q - zp))), exact real decode,
  output scale 1/128, zp 0, round half away from zero, clamp to [-128, 127].
- Independent reference for CertifiedMath = mpmath tanh at 60 digits (a different code path).
"""
import hashlib, json, math, struct, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np, mpmath as mp

SWEEP, TRACE, OUT, ART = sys.argv[1:5]
HERE = Path(__file__).resolve().parent
mp.mp.dps = 60
SRC_REV = dict(framework="tflite-micro", kernel_tree="8f6596d", checkout="5f64e4ad1763fbe53f40534c39aabe22037a6e79",
               prepare="tensorflow/lite/micro/kernels/tanh.cc", eval="tensorflow/lite/micro/kernels/internal/reference/integer_ops/tanh.h",
               fixedpoint="tensorflow/lite/micro/kernels/internal/fixedpoint.h", backend="reference (x86_64 gcc, no CMSIS/xtensa)")
PREP_IMPL = "source-behavior-v1/prepare.py@6.2"

cal = [(c["scale_bits"], c["zp"], c["kind"]) for c in json.load(open(HERE.parent / "tanh-sweep-v1/sweep.json"))["cases"]]
assert len(cal) == 2224

def f32(bits): return float(np.uint32(int(bits, 16)).view(np.float32))
def rh_away(x): a = abs(x); r = math.floor(a + Fraction(1, 2)); return r if x >= 0 else -r
def clamp8(c): return max(-128, min(127, c))
def frac(x):
    sign, man, exp, _ = mp.mpf(x)._mpf_
    v = Fraction(int(man)) * (Fraction(2) ** int(exp)) if man else Fraction(0)
    return -v if sign else v

def cert_tanh(v, ostep=128):  # interval-certified clamp(round_away(ostep*tanh(v)))
    if v == 0: return 0
    for prec in (64, 128, 256, 512):
        mp.iv.prec = prec
        x = mp.iv.mpf(v.numerator) / v.denominator; e = mp.iv.exp(2 * x); t = ostep * (e - 1) / (e + 1)
        lo, hi = clamp8(rh_away(frac(t.a))), clamp8(rh_away(frac(t.b)))
        if lo == hi: return lo
    raise RuntimeError(f"CertifiedMath cell undecided at 512 bits: {v}")
def ref_tanh(v):  # independent: mp.tanh, 60 digits
    return 0 if v == 0 else clamp8(rh_away(frac(128 * mp.tanh(mp.mpf(v.numerator) / v.denominator))))

# --- SourceParity tables from the real kernel -------------------------------------------
t0 = time.perf_counter()
raw = subprocess.run([SWEEP], input="".join(f"{b} {z}\n" for b, z, _ in cal), capture_output=True, text=True, check=True).stdout.splitlines()
t_src = time.perf_counter() - t0
SRC, PREP = [], []
for (b, z, _), line in zip(cal, raw):
    t = line.split(); assert t[0] == b and int(t[1]) == z and t[2] == "1" and t[3] == "1"
    PREP.append(dict(mult=int(t[4]), shift=int(t[5]), radius=int(t[6]))); SRC.append([int(v) for v in t[7:]])
# reproduce the 7 committed fixture tables
fx = json.load(open("/home/user/ndarray/tests/data/tflm/tanh_int8_8f6596d.json"))
fixture_ok = all(SRC[i] == c["output"] for i, c in enumerate(fx["cases"]))

# --- CertifiedMath tables + independent reference -----------------------------------------
t0 = time.perf_counter()
MATH, REF, F32DEC = [], [], []
for b, z, _ in cal:
    s = Fraction(f32(b)); xs = [s * (c - z) for c in range(-128, 128)]
    MATH.append([cert_tanh(v) for v in xs]); REF.append([ref_tanh(v) for v in xs])
    x32 = [Fraction(float(np.float32(np.float32(c - z) * np.float32(f32(b))))) for c in range(-128, 128)]
    F32DEC.append([cert_tanh(v) for v in x32])
t_math = time.perf_counter() - t0

diffs = [[i for i in range(256) if SRC[k][i] != MATH[k][i]] for k in range(len(cal))]
res = dict(source_revision=SRC_REV, calibrations=len(cal), fixture7_reproduced=fixture_ok,
           sourceparity_cells=256 * len(cal),
           certifiedmath_vs_independent_reference_diff=sum(MATH[k][i] != REF[k][i] for k in range(len(cal)) for i in range(256)),
           f32_decode_vs_exact_decode_diff=sum(MATH[k][i] != F32DEC[k][i] for k in range(len(cal)) for i in range(256)),
           source_vs_math_diff_cells=sum(len(d) for d in diffs),
           source_vs_math_diff_calibrations=sum(1 for d in diffs if d),
           prep_seconds_python=dict(source_oracle_all=t_src, certified_math_all_incl_reference=t_math))

# --- trace the discrepancies through the real kernel's stages -----------------------------
inv = []
for k, d in enumerate(diffs):
    if not d: continue
    b, z, kind = cal[k]
    lines = subprocess.run([TRACE], input=f"trace {b} {z}\n", capture_output=True, text=True, check=True).stdout.splitlines()
    rows = {int(l.split()[1]): [int(v) for v in l.split()[1:]] for l in lines if l.startswith("T ")}
    assert len(rows) == 256, "trace did not reproduce the kernel on every code"
    for i in d:
        code = i - 128
        _, inp, radius, in_q4, n_raw, e_raw, t_raw, raw_out, q24, byte, kbyte = rows[code]
        s = Fraction(f32(b)); x_true = s * (code - z); x_fp = Fraction(in_q4, 1 << 27)
        tw = lambda x: frac(mp.tanh(mp.mpf(x.numerator) / x.denominator))
        c_true = clamp8(rh_away(128 * tw(x_true))); c_fp = clamp8(rh_away(128 * tw(x_fp)))
        e = Fraction(e_raw, 1 << 31)
        t_from_e = (1 - e) / (1 + e)                                     # exact division on the kernel's own exp value
        c_from_e = clamp8(rh_away(128 * t_from_e * (1 if raw_out >= 0 else -1)))
        e_ideal = frac(mp.exp(-2 * abs(mp.mpf(x_fp.numerator) / x_fp.denominator)))
        rdp_exact = clamp8(rh_away(Fraction(raw_out, 1 << 24)))
        if c_fp != c_true: stage = "input_rescale (MultiplyByQuantizedMultiplier)"
        elif c_from_e != c_fp: stage = "exp_on_negative_values (fixed-point exp approximation)"
        elif rdp_exact != c_fp: stage = "one_minus_x_over_one_plus_x (Newton-Raphson division)"
        elif q24 != rdp_exact: stage = "RoundingDivideByPOT"
        else: stage = "UNEXPLAINED"
        inv.append(dict(calibration=k, kind=kind, scale_bits=b, zp=z, input_code=code, input_minus_zp=inp,
                        x_real=float(x_true), source_byte=kbyte, math_byte=MATH[k][i], delta=kbyte - MATH[k][i],
                        dist_128tanh_to_half=float(abs(abs(128 * tw(x_true)) - Fraction(1, 2))),
                        prepare=PREP[k], in_q4=in_q4, in_q4_ideal=float(x_true * (1 << 27)),
                        e_raw=e_raw, e_ideal_raw=float(e_ideal * (1 << 31)), e_err_raw=float(e_raw - e_ideal * (1 << 31)),
                        t_raw=t_raw, t_from_e_raw=float(t_from_e * (1 << 31)), t_ideal_raw=float(tw(x_fp) * (1 << 31)),
                        div_err_raw=float(t_raw - t_from_e * (1 << 31)), half_boundary_raw=1 << 23,
                        raw_out=raw_out, q24=q24, earliest_divergence=stage))
res["discrepancies"] = inv
res["discrepancy_mechanisms"] = {}
for r in inv:
    key = r["earliest_divergence"]; m = res["discrepancy_mechanisms"].setdefault(key, dict(cells=0, calibrations=set()))
    m["cells"] += 1; m["calibrations"].add(r["scale_bits"])
for m in res["discrepancy_mechanisms"].values(): m["calibrations"] = sorted(m["calibrations"])

# --- receipts, artifacts, fingerprints ----------------------------------------------------
def receipt(contract, b, z):
    return dict(contract=contract, op="TANH", op_version=1, storage="i8", code_space="Asym",
                in_scale_bits=b, in_zp=z, out_scale_bits="3c000000", out_zp=0,
                rounding="half_away_from_zero", saturation="clamp[-128,127]", nan="none (int8 has no NaN)",
                decode="exact real scale*(q-zp)" if contract == "CertifiedMath" else "source kernel arithmetic",
                source=SRC_REV if contract == "SourceParity" else None,
                reference="mpmath interval tanh, independent mp.tanh check" if contract == "CertifiedMath" else "TFLM kernel bytes",
                preparation=PREP_IMPL)
def rhash(r): return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
def thash(t): return hashlib.sha256(bytes((v + 256) % 256 for v in t)).hexdigest()
store = {}
blob = bytearray(b"PAL256LAB1")
blob += struct.pack("<I", len(cal))
fp = dict(complete_table_bytes=256, digest_bytes=32, calibrations_with_patch=0, patch_entries=0,
          delta_hist={}, max_abs_delta=0)
for k, (b, z, _) in enumerate(cal):
    for contract, tab in (("SourceParity", SRC[k]), ("CertifiedMath", MATH[k])):
        r = receipt(contract, b, z); key = rhash(r)
        store[key] = dict(receipt=r, table=list(tab), table_sha256=thash(tab))
    d = diffs[k]
    if d:
        fp["calibrations_with_patch"] += 1; fp["patch_entries"] += len(d)
    for i in d:
        dl = SRC[k][i] - MATH[k][i]; fp["delta_hist"][str(dl)] = fp["delta_hist"].get(str(dl), 0) + 1
        fp["max_abs_delta"] = max(fp["max_abs_delta"], abs(dl))
    # artifact blob for the Rust probe: receipt hashes, both tables, and the sparse patch
    blob += bytes.fromhex(rhash(receipt("SourceParity", b, z))) + bytes.fromhex(rhash(receipt("CertifiedMath", b, z)))
    blob += bytes((v + 256) % 256 for v in SRC[k]) + bytes((v + 256) % 256 for v in MATH[k])
    blob += bytes.fromhex(thash(SRC[k])) + bytes.fromhex(thash(MATH[k]))
    blob += struct.pack("<H", len(d)) + b"".join(struct.pack("<Bb", i, SRC[k][i]) for i in d)
Path(ART).write_bytes(bytes(blob))
n = len(cal)
fp["storage_bytes"] = dict(
    complete_both_tables=n * 2 * (256 + 32),
    source_complete_only=n * (256 + 32),
    math_plus_sparse_patch=n * (256 + 32) + sum(2 + 2 * len(d) + 32 for d in diffs),  # count + (idx, byte) + source digest
    shared_math_plus_keyed_exceptions=n * (256 + 32) + sum((2 + 2 * len(d) + 32) for d in diffs if d) + 32 * n)  # + per-source receipt
fp["artifact_blob_sha256"] = hashlib.sha256(bytes(blob)).hexdigest()
res["fingerprint"] = fp
res["receipts_distinct_per_calibration"] = all(rhash(receipt("SourceParity", b, z)) != rhash(receipt("CertifiedMath", b, z)) for b, z, _ in cal)
res["tables_identical_but_receipts_distinct"] = sum(1 for k in range(n) if not diffs[k])

# --- lab admission (prepared-artifact lookup) and negative controls ----------------------
class Refused(Exception): pass
def lookup(req):
    key = rhash(req)
    a = store.get(key)
    if a is None: raise Refused("no_artifact_for_receipt")
    if thash(a["table"]) != a["table_sha256"]: raise Refused("table_digest_mismatch")
    return a
def ctl(name, f):
    try: f(); res.setdefault("negative_controls", {})[name] = "accepted"
    except Refused as e: res.setdefault("negative_controls", {})[name] = f"refused: {e}"
    except AssertionError as e: res.setdefault("negative_controls", {})[name] = f"FAILED: {e}"
k0 = next(k for k in range(n) if diffs[k])                     # a calibration where the contracts differ
k1 = next(k for k in range(n) if not diffs[k])                 # one where they agree byte for byte
b0, z0, _ = cal[k0]; b1, z1, _ = cal[k1]
ctl("correct_SourceParity", lambda: lookup(receipt("SourceParity", b0, z0)))
ctl("correct_CertifiedMath", lambda: lookup(receipt("CertifiedMath", b0, z0)))
ctl("B_wrong_calibration_scale", lambda: lookup({**receipt("SourceParity", b0, z0), "in_scale_bits": format(int(b0, 16) + 1, "08x")}))
ctl("B_wrong_calibration_zp", lambda: lookup({**receipt("SourceParity", b0, z0), "in_zp": z0 + 1}))
ctl("C_wrong_source_revision", lambda: lookup({**receipt("SourceParity", b0, z0), "source": {**SRC_REV, "kernel_tree": "0000000"}}))
ctl("C_wrong_op_version", lambda: lookup({**receipt("SourceParity", b0, z0), "op_version": 2}))
def wrong_mode_equal_tables():
    # CertifiedMath table offered under a SourceParity receipt, on a calibration where the BYTES are equal:
    # the receipt hash still differs, so the swap is refused by identity, not by content.
    sp, cm = receipt("SourceParity", b1, z1), receipt("CertifiedMath", b1, z1)
    assert store[rhash(sp)]["table"] == store[rhash(cm)]["table"], "fixture is not a byte-equal calibration"
    a = lookup(cm)
    if a["receipt"]["contract"] != sp["contract"]: raise Refused("contract_mismatch")
ctl("D_math_table_under_source_receipt_equal_bytes", wrong_mode_equal_tables)
def corrupt():
    a = store[rhash(receipt("SourceParity", b0, z0))]; saved = list(a["table"])
    a["table"][17] = (a["table"][17] + 1) if a["table"][17] < 127 else 126
    try: lookup(receipt("SourceParity", b0, z0))
    finally: a["table"] = saved
ctl("F_corrupted_byte", corrupt)
ctl("F_restored_after_corruption", lambda: lookup(receipt("SourceParity", b0, z0)))
# G. code-space collision: which contracts are admissible for which code space
def admit(contract, space):
    if space == "Asym": return "admitted"
    if space == "Sym":
        return ("admitted for codes -127..127; input -128 is not a Sym value and is refused"
                if contract == "CertifiedMath" else "refused: TFLM int8 TANH evaluates -128 as a value; Sym declares it unused")
    if space == "SymNaN":
        return ("admitted: -128 is the NaN code, table[-128] = NaN code (-128)" if contract == "CertifiedMath"
                else "refused: TFLM int8 has no NaN representation; no source parity exists for the sentinel")
    if space == "UnitFull":
        return "refused: 0..255 are palette indices / unit codes, not an affine int8 calibration TFLM TANH accepts"
    raise ValueError(space)
res["G_code_space_admission"] = {f"{c}/{s}": admit(c, s) for c in ("SourceParity", "CertifiedMath")
                                 for s in ("Asym", "Sym", "SymNaN", "UnitFull")}
# A/E: sampled acceptance is not exhaustive acceptance - a source table validated on 8 sampled
# codes of a discrepancy calibration accepts the math table; the exhaustive comparison refuses it.
q_sample = [0, 64, 128, 192, 255, 30, 100, 220]
res["sampled_vs_exhaustive"] = dict(
    sampled_codes_agree=all(SRC[k0][i] == MATH[k0][i] for i in q_sample),
    exhaustive_codes_agree=SRC[k0] == MATH[k0], calibration=cal[k0][:2])
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("discrepancies",)}, indent=1, default=str)[:6000])
