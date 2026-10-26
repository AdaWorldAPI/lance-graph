# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Fisher-z / Palette256 softmax probe.

Reference (independent of every LUT): mpmath at 50 digits on the REAL decoded
representatives z(q) = z_min + (q+127)/254 * z_range, with z_min, z_range taken
as the exact values of their f32 storage. The LUT arms are built from f64
exp(k*d) over code differences; the reference never sees a difference table.
"""
import json, math, subprocess, sys
import numpy as np
import mpmath as mp
mp.mp.dps = 50
f32 = np.float32
ORACLE = sys.argv[1]
rng = np.random.default_rng(20261026)

def atanh_clamp32(x):
    c = np.clip(f32(x), f32(-0.9999), f32(0.9999))
    return f32(0.5) * np.log((f32(1) + c) / (f32(1) - c))

class Gamma:
    def __init__(self, zmin, zr):
        self.zmin, self.zr = f32(zmin), f32(zr)
    def z_real(self, q):  # exact real decode of the stored f32 parameters
        return mp.mpf(float(self.zmin)) + (mp.mpf(int(q)) + 127) / 254 * mp.mpf(float(self.zr))
    def z32(self, q):     # FamilyGamma::decode before tanh, op by op in f32
        return (f32(q) + f32(127)) / f32(254) * self.zr + self.zmin
    def encode32(self, cos):  # FamilyGamma::encode: `as i8` truncates toward zero
        v = (atanh_clamp32(cos) - self.zmin) / self.zr * f32(254) - f32(127)
        return int(np.trunc(np.clip(v, f32(-128), f32(127))))

GAMMAS = {
    "G1(zmin=-0.5,zr=1)": Gamma(-0.5, 1.0),
    "G2(cos -0.3..0.95)": Gamma(float(atanh_clamp32(-0.3)), float(atanh_clamp32(0.95) - atanh_clamp32(-0.3))),
    "G3(full clamp range)": Gamma(float(atanh_clamp32(-1.0)), float(atanh_clamp32(1.0) - atanh_clamp32(-1.0))),
}
BETAS = [0.5, 1.0, 2.0, 8.0, 32.0]
R = {}

# ---- 1. Binary, exhaustive over all 256x256 code pairs (path A) -------------
codes = np.arange(-128, 128)
bin_rows = []
for gname, g in GAMMAS.items():
    z = [g.z_real(q) for q in codes]
    zf = np.array([g.z32(q) for q in codes], dtype=f32)
    for beta in BETAS:
        b = mp.mpf(beta); kf = beta * float(g.zr) / 508.0
        max_id, max_cf, nonfinite, max_f32 = mp.mpf(0), 0.0, 0, 0.0
        for i in range(256):
            for j in range(256):
                d = z[i] - z[j]
                p1 = 1 / (1 + mp.e ** (-b * d))           # softmax of two, computed
                chain = mp.atanh(2 * p1 - 1)               # path A, exact reals
                closed = b * d / 2
                max_id = max(max_id, abs(chain - closed))
                cf = kf * (int(codes[i]) - int(codes[j]))  # closed form from codes only
                max_cf = max(max_cf, abs(cf - float(chain)))
                # f32 chain as an implementation would run it
                x1, x2 = f32(beta) * zf[i], f32(beta) * zf[j]
                m = max(x1, x2); e1, e2 = np.exp(x1 - m), np.exp(x2 - m)
                p = e1 / (e1 + e2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    a = np.arctanh(f32(2) * p - f32(1))
                if not np.isfinite(a):
                    nonfinite += 1
                else:
                    max_f32 = max(max_f32, abs(float(a) - float(chain)))
        bin_rows.append(dict(gamma=gname, beta=beta,
            identity_max_err=float(max_id), closed_form_f64_max_err=max_cf,
            f32_chain_nonfinite=nonfinite, f32_chain_max_abs_err=max_f32))
R["binary"] = bin_rows

# ---- 2. N-way (path A): reference vs f32 chain vs LUT+integer vs TFLM -------
def dists(n):
    out = {"uniform": rng.integers(-128, 128, n)}
    out["all_equal"] = np.full(n, 17)
    s = np.full(n, -128); s[n // 2] = 127; out["spike_vs_floor"] = s
    c = np.where(np.arange(n) % 2 == 0, 120, -100); out["two_clusters"] = c
    out["near_max_ties"] = rng.integers(120, 128, n)
    out["normal"] = np.clip(np.round(rng.normal(0, 30, n)), -128, 127).astype(int)
    return out

def lut(k, F):
    return [int(round(math.exp(k * d) * (1 << F))) for d in range(-255, 1)]  # index d+255

def tflm(cases):
    lines = []
    for (codes_, scale, zp, beta) in cases:
        sb = np.array([scale], dtype=f32).view(np.uint32)[0]
        bb = np.array([beta], dtype=f32).view(np.uint32)[0]
        lines.append(f"{len(codes_)} {sb:x} {zp} {bb:x} " + " ".join(str(int(v)) for v in codes_))
    res = []
    for l in lines:  # one process per case: an abort must not hide the others
        r = subprocess.run([ORACLE], input=l + "\n", capture_output=True, text=True)
        res.append(list(map(int, r.stdout.split())) if r.returncode == 0 else None)
    return res

nway = []
for n in [2, 3, 16, 256, 4096]:
    for dname, q in dists(n).items():
        for gname in ["G2(cos -0.3..0.95)", "G3(full clamp range)"]:
            g = GAMMAS[gname]
            for beta in [1.0, 8.0]:
                k = beta * float(g.zr) / 254.0
                zr_ = [g.z_real(v) for v in q]
                m = max(zr_); b = mp.mpf(beta)
                w = [mp.e ** (b * (zi - m)) for zi in zr_]; S = mp.fsum(w)
                pref = [float(wi / S) for wi in w]
                # f32 chain (decode, max-subtract, exp, sequential f32 sum, divide)
                x = np.array([f32(beta) * g.z32(v) for v in q], dtype=f32)
                e = np.exp(x - x.max()); s32 = f32(0)
                for t in e: s32 = f32(s32 + t)
                p32 = e / s32
                qmax = int(q.max()); d = q - qmax
                row = dict(n=n, dist=dname, gamma=gname, beta=beta,
                           f32_chain_max_abs_err=float(np.max(np.abs(p32 - pref))))
                V = rng.normal(0, 1, (n, 8))
                yref = np.array([float(mp.fsum(w[i] * V[i, c] for i in range(n)) / S) for c in range(8)])
                for F, acc in [(16, "u32"), (24, "u64")]:
                    L = lut(k, F)
                    wi = [L[int(di) + 255] for di in d]
                    Sint = sum(wi)
                    hist = np.bincount(q + 128, minlength=256)
                    Shist = sum(int(hist[c]) * L[c - 128 - qmax + 255] for c in range(256) if hist[c] and c - 128 <= qmax)
                    assert Shist == Sint
                    plut = np.array(wi, dtype=float) / Sint
                    y = (np.array(wi, dtype=float) @ V) / Sint
                    row[f"lut_F{F}_max_abs_err"] = float(np.max(np.abs(plut - pref)))
                    row[f"lut_F{F}_fused_y_max_abs_err"] = float(np.max(np.abs(y - yref)))
                    row[f"lut_F{F}_S_bits"] = Sint.bit_length()
                    row[f"lut_F{F}_zeroed_terms"] = sum(1 for v in wi if v == 0)
                    row[f"_codes_F{F}"] = np.clip(np.round(plut * 256) - 128, -128, 127)
                row["S_max_bound_bits_F16"] = (n * (1 << 16)).bit_length()
                # TFLM int8 softmax: input scale = z_range/254 as f32; zp is unused by the kernel
                sc = float(f32(float(g.zr) / 254.0))
                (o0, o1) = tflm([(q, sc, 0, beta), (q, sc, 37, beta)])
                row["tflm_aborted"] = o0 is None
                row["real_sum_of_exps"] = float(np.sum(np.exp(beta * sc * d.astype(float))))
                if o0 is None:
                    row["tflm_ties_at_max"] = int(np.sum(q == qmax))
                    nway.append(row); continue
                assert o0[0] == 1 and o0[1] == 1
                ot = np.array(o0[2:]); row["tflm_zp_invariant"] = o0 == o1
                ptf = (ot + 128) / 256.0
                row["tflm_max_abs_err"] = float(np.max(np.abs(ptf - pref)))
                rq = np.clip(np.round(np.array(pref) * 256) - 128, -128, 127)
                row["tflm_codes_differing_from_rounded_ref"] = int(np.sum(rq != ot))
                for F in (16, 24):
                    row[f"lut_F{F}_codes_differing_from_tflm"] = int(np.sum(row[f"_codes_F{F}"] != ot))
                nway.append(row)
for r in nway:
    for F in (16, 24): r.pop(f"_codes_F{F}", None)
R["nway"] = nway

# ---- 3. Counterexamples -----------------------------------------------------
cx = {}
# C1 mixed calibrations in one vector: one k cannot serve both.
gA, gB = GAMMAS["G1(zmin=-0.5,zr=1)"], GAMMAS["G3(full clamp range)"]
qa, qb = np.array([100, 10, -50]), np.array([60, 0])
zs = [gA.z_real(v) for v in qa] + [gB.z_real(v) for v in qb]
m = max(zs); w = [mp.e ** (zi - m) for zi in zs]; S = mp.fsum(w); pref = [float(x / S) for x in w]
allq = np.concatenate([qa, qb]); kA = float(gA.zr) / 254.0
wf = np.exp(kA * (allq - allq.max())); pcode = wf / wf.sum()
cx["C1_mixed_gamma_code_formula_max_abs_err"] = float(np.max(np.abs(pcode - pref)))
cx["C1_argmax_ref_vs_code"] = [int(np.argmax(pref)), int(np.argmax(allq))]
# C2 path B (softmax over cos = tanh z): weight is not a function of q - q_max.
g = GAMMAS["G2(cos -0.3..0.95)"]
def wB(qi, qm): return float(mp.e ** (mp.tanh(g.z_real(qi)) - mp.tanh(g.z_real(qm))))
cx["C2_pathB_same_diff_minus10"] = [wB(117, 127), wB(-110, -100)]
# pair table over (q_i, q_max) reproduces it on representatives (64 KiB per beta, gamma)
cx["C2_pathB_pair_table_entries"] = 256 * 256
# C3 encode truncation: encode(decode(q)) == q ?
for gname, gg in GAMMAS.items():
    bad = [int(q) for q in codes if gg.encode32(float(np.tanh(gg.z32(q)))) != q]
    cx[f"C3_roundtrip_failures_{gname}"] = dict(count=len(bad), examples=bad[:8])
# C4 ranking collisions: distinct cosines -> same code
cs = np.sort(rng.uniform(-0.3, 0.95, 20000)).astype(f32)
enc = np.array([g.encode32(c) for c in cs])
cx["C4_order_inversions_after_encode"] = int(np.sum(np.diff(enc) < 0))
cx["C4_adjacent_distinct_inputs_merged"] = int(np.sum((np.diff(enc) == 0) & (np.diff(cs) > 0)))
# C5 code -128: reachable only below z_min - z_range/254; decode is the affine continuation
cx["C5_encode_of_cos_below_family"] = g.encode32(-0.5)
cx["C5_decode_minus128_cos"] = float(np.tanh(g.z32(-128)))
# C6 FisherZTable::lookup_i8 out-of-range returns code 0, which decodes to:
cx["C6_decode_code0_cos"] = {k: float(np.tanh(v.z32(0))) for k, v in GAMMAS.items()}
R["counterexamples"] = cx
json.dump(R, open(sys.argv[2], "w"), indent=1, default=str)
