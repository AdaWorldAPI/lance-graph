# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.1: exact-first certification, tighter F24 bounds, consumed masks.

The case generator, reference, code rules, table builder and baseline resolver
are LOADED FROM cert.py (its text up to the main loop), so the 261 cases are the
Amendment 6 cases byte for byte. cert.py and cert.json are not modified.

Arms (each is: table reference m, error model, exact-first shortcut on/off):
  base     rowmax  eps = 1/2 every entry           no shortcut   == Amendment 6
  zero     rowmax  eps = 0 where x_a == x_m        no shortcut
  pe       rowmax  eps_i certified per entry (G=8) no shortcut
  first    rowmax  eps = 1/2                       exact-uniform shortcut first
  first_pe rowmax  per-entry eps                   shortcut first
  unary    unary   eps = 1/2 (table referenced to the CODEBOOK max, one row)
  unary_pe unary   per-entry eps

Usage: cert2.py OUT.json [FIXTURE.txt]   (fixture = canonical input for the Rust lab)
"""
import json, math, sys, time, hashlib
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
_src = (HERE / "cert.py").read_text()
_marker = "res, gens = [], {}\n"
assert _src.count(_marker) == 1, "cert.py changed: marker moved"
_argv = sys.argv
sys.argv = [_argv[0], "/dev/null"]          # cert.py reads OUT from argv[1]; never written here
exec(compile(_src[: _src.index(_marker)], "cert.py", "exec"))
sys.argv = _argv
OUT = sys.argv[1]
FIX = sys.argv[2] if len(sys.argv) > 2 else None
G = 8                                        # per-entry eps grid: eps_i = e_i / 2^G, e_i <= 2^(G-1)

# --- certified table with per-entry error ------------------------------------------
def table_row2(g, m):
    """W[a|m] and e[a] (eps_a = e[a]/2^G, certified upper bound of |2^24 w - W|)."""
    key = ("pe", m)
    if key in g["cache"]: return g["cache"][key]
    W, e, unc = {}, {}, 0
    with ivprec(96):
        for a in range(256):
            if g["x"][a] > g["x"][m]: continue
            if g["x"][a] == g["x"][m]:               # exp(0): exact, error exactly zero
                W[a], e[a] = 1 << F, 0; continue
            iv = mp.iv.exp(mp.iv.mpf(g["beta"]) * (mp.iv.mpf(float(g["x"][a])) - mp.iv.mpf(float(g["x"][m])))) * (2 ** F)
            lo, hi = frac(iv.a), frac(iv.b)
            wl, wh = math.floor(lo + Fraction(1, 2)), math.floor(hi + Fraction(1, 2))
            unc += wl != wh
            dev = max(abs(lo - wl), abs(hi - wl))
            assert dev <= Fraction(1, 2), "entry not certified to 1/2"
            W[a], e[a] = wl, math.ceil(dev * (1 << G))
    g["cache"][key] = (W, e, unc); return g["cache"][key]

def ucode_max(g):  # codebook maximum (unary reference)
    return int(max(range(256), key=lambda a: g["x"][a]))

# --- the bound, with an explicit per-entry eps --------------------------------------
def bound_row(Wi, eps, Vf, cols):
    """Arm A + ledger-G bound. Returns per-output (code_hat, certified) for y and p."""
    n = len(Wi); S = sum(Wi); E = sum(eps)
    if S == 0:   # every weight rounded to zero (unary reference far above the row max)
        return [(None, False)] * len(cols), [(None, False)] * n
    ys, ps = [], []
    for k in cols:
        N = sum(Fraction(Wi[i]) * Fraction(Vf[i][k]) for i in range(n)); yh = N / S
        if S > E:
            B = sum(eps[i] * abs(Fraction(Vf[i][k]) - yh) for i in range(n)) / (S - E)
            ys.append((code_y(yh), code_y(yh - B) == code_y(yh + B)))
        else: ys.append((code_y(yh), False))
    for i in range(n):
        ph = Fraction(Wi[i], S)
        if S > E:
            Bp = (eps[i] + ph * E) / (S - E)
            ps.append((code_p(ph), code_p(max(Fraction(0), ph - Bp)) == code_p(min(Fraction(1), ph + Bp))))
        else: ps.append((code_p(ph), False))
    return ys, ps

def exact_uniform(Vf, cols, n):
    return [code_y(sum(Fraction(Vf[i][k]) for i in range(n)) / n) for k in cols], code_p(Fraction(1, n))

def escalate2(g, q, Vf, cols, disable=()):
    """Per-output resolver: exact_rational (one representative) / exact_group_mean (y) /
    interval@bits / UNRESOLVED. Same arithmetic as cert.py's escalate()."""
    n = len(q)
    byx = {}
    for i, a in enumerate(q): byx.setdefault(float(g["x"][a]), []).append(i)
    if len(byx) == 1:
        yc, pc = exact_uniform(Vf, cols, n)
        return [(c, "exact_rational") for c in yc], [(pc, "exact_rational")] * n
    groups = {}
    for i, a in enumerate(q): groups.setdefault(int(a), []).append(i)
    exact_y = {}
    for k in cols:
        means = {sum(Fraction(Vf[i][k]) for i in ix) / len(ix) for ix in byx.values()}
        if len(means) == 1 and "group_mean" not in disable: exact_y[k] = code_y(means.pop())
    m = max(groups, key=lambda a: g["x"][a])
    yres = {k: (exact_y[k], "exact_group_mean") for k in exact_y}
    pres = {}
    for bits in (64, 128, 256, 512):
        with ivprec(bits):
            e = {a: mp.iv.exp(mp.iv.mpf(g["beta"]) * (mp.iv.mpf(float(g["x"][a])) - mp.iv.mpf(float(g["x"][m])))) for a in groups}
            S = sum((e[a] * len(ix) for a, ix in groups.items()), mp.iv.mpf(0))
            for k in cols:
                if k in yres: continue
                N = sum((e[a] * mp.iv.mpf(sum(Fraction(Vf[i][k]) for i in ix).numerator) / sum(Fraction(Vf[i][k]) for i in ix).denominator
                         for a, ix in groups.items()), mp.iv.mpf(0))
                yv = N / S; c0, c1 = code_y(frac(yv.a)), code_y(frac(yv.b))
                if c0 == c1: yres[k] = (c0, f"interval@{bits}")
            for a in groups:
                if a in pres: continue
                pv = e[a] / S; c0, c1 = code_p(frac(pv.a)), code_p(frac(pv.b))
                if c0 == c1: pres[a] = (c0, f"interval@{bits}")
        if len(yres) == len(cols) and len(pres) == len(groups): break
    ys = [yres.get(k, (None, "UNRESOLVED")) for k in cols]
    ps = [pres.get(int(a), (None, "UNRESOLVED")) for a in q]
    return ys, ps

ARMS = {"base": ("rowmax", "half", False), "zero": ("rowmax", "zero", False), "pe": ("rowmax", "pe", False),
        "first": ("rowmax", "half", True), "first_pe": ("rowmax", "pe", True),
        "unary": ("unary", "half", False), "unary_pe": ("unary", "pe", False)}

class Refused(Exception): pass

def run(g, q, Vf, cols, arm, mask=None, disable=()):
    """Run one arm on one row. mask: bool per element (True = active). Masked elements are
    excluded from the max, the sums and y; their p is exactly 0 by the declared contract."""
    ref, err, first = ARMS[arm]
    n_all = len(q)
    act = [i for i in range(n_all) if mask is None or mask[i] or "mask_ignored" in disable]
    if not act: raise Refused("empty_after_mask")
    if g.get("symnan") and any(int(q[i]) == 0x80 for i in act): raise Refused("symnan_code_in_active_row")
    qa = [int(q[i]) for i in act]; Va = [Vf[i] for i in act]; n = len(act)
    out_y, out_p = None, None
    if first and "shortcut" not in disable and len({float(g["x"][a]) for a in qa}) == 1:
        yc, pc = exact_uniform(Va, cols, n)
        out_y = [(c, "shortcut") for c in yc]; out_p = [(pc, "shortcut")] * n
        row_class = "shortcut"
    else:
        m = (max(set(qa), key=lambda a: g["x"][a]) if "mask_max" not in disable else
             max(set(int(a) for a in q), key=lambda a: g["x"][a])) if ref == "rowmax" else ucode_max(g)
        W, e, _ = table_row2(g, m)
        if any(a not in W for a in qa): raise Refused("table_row_does_not_cover_code")
        Wi = [W[a] for a in qa]
        if err == "half": eps = [Fraction(1, 2)] * n
        elif err == "zero": eps = [Fraction(0) if e[a] == 0 else Fraction(1, 2) for a in qa]
        else: eps = [Fraction(e[a], 1 << G) for a in qa]
        ys, ps = bound_row(Wi, eps, Va, cols)
        if all(c for _, c in ys) and all(c for _, c in ps):
            out_y = [(c, "bound") for c, _ in ys]; out_p = [(c, "bound") for c, _ in ps]; row_class = "bound"
        else:
            ey, ep = escalate2(g, qa, Va, cols, disable)
            out_y = [(c, "bound") if ok else ey[k] for k, (c, ok) in enumerate(ys)]
            out_p = [(c, "bound") if ok else ep[i] for i, (c, ok) in enumerate(ps)]
            row_class = "UNRESOLVED" if any(h == "UNRESOLVED" for _, h in out_y + out_p) else "escalated"
    p_full = [(code_p(Fraction(0)), "mask")] * n_all
    for j, i in enumerate(act): p_full[i] = out_p[j]
    return row_class, out_y, p_full

def ref_masked(g, q, Vf, cols, mask=None):
    act = [i for i in range(len(q)) if mask is None or mask[i]]
    yc, pc = ref_codes(g["x"], g["beta"], [q[i] for i in act], [Vf[i] for i in act], cols)
    pf = [code_p(Fraction(0))] * len(q)
    for j, i in enumerate(act): pf[i] = pc[j]
    return yc, pf

# --- main sweep -------------------------------------------------------------------
cols = list(range(D))
base_json = json.load(open(HERE / "cert.json"))["cases"]
summary = {a: dict(rows={}, outputs={}, misses=0, final_wrong=0, t_s=0.0) for a in ARMS}
decision_diff = {a: [] for a in ARMS}
per_case = []
fix_lines = []
gens = {}
for cid, ((kname, beta, dname, x, q), (vi, vf)) in enumerate(meta):
    key = (kname, beta)
    if key not in gens: gens[key] = generation(x, beta)
    g = gens[key]
    Vf = [[float(v) for v in row] for row in vf]
    yref, pref = ref_masked(g, q, Vf, cols)
    rec = dict(case=cid, codebook=kname, beta=beta, dist=dname, n=len(q))
    outs = {}
    for arm in ARMS:
        t0 = time.perf_counter()
        rc, oy, op = run(g, q, Vf, cols, arm); outs[arm] = (rc, oy, op)
        summary[arm]["t_s"] += time.perf_counter() - t0
        s = summary[arm]
        s["rows"][rc] = s["rows"].get(rc, 0) + 1
        for (c, h), r in list(zip(oy, yref)) + list(zip(op, pref)):
            s["outputs"][h] = s["outputs"].get(h, 0) + 1
            if c != r: s["final_wrong"] += 1
            if h in ("bound", "shortcut") and c != r: s["misses"] += 1
        rec[arm] = dict(row=rc, y=[h for _, h in oy], y_codes=[c for c, _ in oy],
                        p_hist={h: sum(1 for _, hh in op if hh == h) for h in {hh for _, hh in op}})
    b = base_json[cid]
    yc_b = sum(1 for h in rec["base"]["y"] if h == "bound")
    pc_b = rec["base"]["p_hist"].get("bound", 0)
    same = (yc_b == b["y_certified"] and pc_b == b["p_certified"]
            and (rec["base"]["row"] != "bound") == b["escalated"])
    rec["base_matches_cert_json"] = same
    if not same: decision_diff["base"].append(cid)
    per_case.append(rec)
    if FIX:
        m = int(max(set(int(a) for a in q), key=lambda a: g["x"][a]))
        W, e, _ = table_row2(g, m); mu = ucode_max(g); Wu, eu, _ = table_row2(g, mu)
        hx = lambda v: format(np.float64(v).view(np.uint64), "x")
        fix_lines.append(f"case {cid} {hx(beta)} {len(q)} {D} {m} {mu}")
        fix_lines.append(" ".join(hx(v) for v in g["x"]))
        fix_lines.append(" ".join(map(str, (int(a) for a in q))))
        fix_lines.append(" ".join(hx(v) for row in Vf for v in row))
        fix_lines.append(" ".join(f"{W.get(a, -1)}:{e.get(a, -1)}" for a in range(256)))
        fix_lines.append(" ".join(f"{Wu.get(a, -1)}:{eu.get(a, -1)}" for a in range(256)))
        fix_lines.append(" ".join(map(str, yref)) + " | " + " ".join(map(str, pref)))
        for arm in ARMS:
            _, oy, op = outs[arm]
            fix_lines.append(arm + " " + rec[arm]["row"] + " " +
                             "".join("B" if h in ("bound",) else "S" if h == "shortcut" else "x" for _, h in oy) + " " +
                             "".join("B" if h == "bound" else "S" if h == "shortcut" else "x" for _, h in op) + " " +
                             " ".join(str(c) for c, _ in oy) + " " + " ".join(str(c) for c, _ in op))

# arm-vs-base decision differences (per output, final code equality)
for arm in ARMS:
    if arm == "base": continue
    for rec in per_case:
        if rec[arm]["y_codes"] != rec["base"]["y_codes"]: decision_diff[arm].append(rec["case"])

# --- negative controls (consumed by the candidate pipeline) -----------------------
neg = {}
g8 = gens[("affine_s0.0084", 8.0)]; g32 = gens[("affine_s0.039", 32.0)]
def ctl(name, g, q, V, mask=None, arms=("first", "base", "pe"), disable=()):
    q = np.asarray(q)
    if mask is not None and not any(mask): yr, pr, r = [None], None, dict(ref="undefined: empty row")
    else:
        yr, pr = ref_masked(g, q, V, [0], mask); r = dict(ref_y=yr[0], ref_p_head=pr[:6])
    for a in arms:
        try:
            rc, oy, op = run(g, q, V, [0], a, mask, disable)
            r[a] = dict(row=rc, y=oy[0], p_head=op[:6], y_ok=oy[0][0] == yr[0], p_ok=[c for c, _ in op] == pr)
        except Refused as ex: r[a] = f"refused: {ex}"
    neg[name] = r
# A. shortcut disabled reproduces the base arm on every case (checked above: decision_diff + below)
neg["A_first_with_shortcut_disabled_equals_base"] = all(
    run(gens[(c[0], c[1])], c[4], [[float(v) for v in row] for row in vv[1]], cols, "first", None, ("shortcut",))[1] ==
    run(gens[(c[0], c[1])], c[4], [[float(v) for v in row] for row in vv[1]], cols, "base")[1]
    for c, vv in meta[:: 7])          # every 7th case (38 cases): cost bound, stated in the amendment
# B. equal-group-mean disabled (rational midpoint, two representatives) -> must stay UNRESOLVED (from cert_disabled run)
ctl("B_rational_midpoint_pos", g8, [200, 200, 10, 10], [[2.0], [3.0], [1.0], [4.0]], arms=("base", "first"))
ctl("B_rational_midpoint_pos_DISABLE_group_mean", g8, [200, 200, 10, 10], [[2.0], [3.0], [1.0], [4.0]], arms=("base",), disable=("group_mean",))
ctl("B_rational_midpoint_neg", g8, [200, 200, 10, 10], [[-2.0], [-3.0], [-1.0], [-4.0]], arms=("base", "first"))
# D. masks consumed by the candidate pipeline
qm = [200, 10, 255, 30]; Vm = [[1.0], [2.0], [3.0], [4.0]]; mk = [True, True, False, True]
ctl("D_mask_excludes_max", g8, qm, Vm, mk, arms=("first", "base", "pe"))
ctl("D_mask_disable_max_selection", g8, qm, Vm, mk, arms=("base",), disable=("mask_max",))
ctl("D_mask_ignored_DISABLE", g8, qm, Vm, mk, arms=("base",), disable=("mask_ignored",))
ctl("D_mask_max_from_masked_steep_DISABLE", g32, [0, 3, 255, 1], Vm, mk, arms=("base",), disable=("mask_max",))
ctl("D_mask_max_from_masked_steep", g32, [0, 3, 255, 1], Vm, mk, arms=("base",))
ctl("D_mask_all_masked", g8, qm, Vm, [False] * 4, arms=("base",))
ctl("D_mask_uniform_after_mask", g8, [200, 7, 200, 200], [[1.0], [99.0], [2.0], [4.0]], [True, False, True, True], arms=("first", "base"))
# E. saturation and exceptional values
ctl("E_large_gap_weights_to_zero", g32, [255] + [0] * 255, [[float(v)] for v in rng.integers(-127, 128, 256)], arms=("base", "pe", "first"))
ctl("E_all_p_near_zero", g32, [255] + [128] * 4095, [[1.0]] * 4096, arms=("base", "pe"))
ctl("E_y_saturates_high", g8, [100, 120, 140], [[300.0], [300.0], [300.0]], arms=("base", "first"))
ctl("E_y_saturates_low", g8, [100, 120, 140], [[-300.0], [-300.0], [-300.0]], arms=("base", "first"))
ctl("E_y_exact_127_5", g8, [50, 50], [[127.0], [128.0]], arms=("base", "first"))
ctl("E_uniform_midpoint_pos", g8, [77, 77], [[2.0], [3.0]], arms=("base", "zero", "first"))
ctl("E_uniform_midpoint_neg", g8, [77, 77], [[-2.0], [-3.0]], arms=("base", "zero", "first"))
for b, why in ((0.0, "beta_zero"), (-1.0, "beta_negative"), (float("inf"), "beta_inf"), (float("nan"), "beta_nan")):
    try: generation(CB["affine_s0.0084"], b); neg["E_" + why] = "ACCEPTED (defect)"
    except Exception as ex: neg["E_" + why] = f"refused: {ex}"
gs = dict(generation(CB["affine_s0.0084"], 8.0)); gs["symnan"] = True; gs["cache"] = {}
ctl("E_symnan_code_0x80_refused", gs, [128, 10, 20], [[1.0], [2.0], [3.0]], arms=("base",))
ctl("E_asymmetric_code_0x80_ordinary", g8, [128, 10, 20], [[1.0], [2.0], [3.0]], arms=("base",))
# C. wrong-codebook identity: an actual prepared-artifact store keyed by digest.
store = {g["digest"]: g for g in gens.values()}
def lookup(declared_digest, beta, x):
    art = store.get(declared_digest)
    if art is None: raise Refused("no_artifact_for_digest")
    if art["beta"] != beta or not np.array_equal(art["x"], np.asarray(x, dtype=np.float64)):
        raise Refused("artifact_identity_mismatch")
    return art
gA, gB = gens[("affine_s0.0084", 8.0)], gens[("nonuniform_normal", 8.0)]
for name, (dig, beta, x) in {"C_correct_codebook": (gA["digest"], 8.0, CB["affine_s0.0084"]),
                             "C_wrong_codebook_declared_A_data_B": (gA["digest"], 8.0, CB["nonuniform_normal"]),
                             "C_wrong_beta": (gA["digest"], 1.0, CB["affine_s0.0084"]),
                             "C_unknown_digest": ("0" * 16, 8.0, CB["affine_s0.0084"])}.items():
    try: lookup(dig, beta, x); neg[name] = "accepted"
    except Refused as ex: neg[name] = f"refused: {ex}"

# economics: tables and metadata actually built
econ = {}
for key, g in gens.items():
    rows_pe = [k for k in g["cache"] if isinstance(k, tuple) and k[0] == "pe"]
    econ[f"{key[0]}|{key[1]}"] = dict(table_rows_cached=len(rows_pe))
json.dump(dict(arms={a: list(v) for a, v in ARMS.items()}, G=G, summary=summary,
               base_reproduces_cert_json=not decision_diff["base"], decision_diff=decision_diff,
               negative_controls=neg, economics=econ, cases=per_case),
          open(OUT, "w"), indent=0, default=str)
if FIX:
    Path(FIX).write_text("\n".join(fix_lines) + "\n")
    print("fixture sha256", hashlib.sha256(Path(FIX).read_bytes()).hexdigest())
print(json.dumps({a: {k: v for k, v in s.items()} for a, s in summary.items()}, indent=1, default=str))
print("base reproduces cert.json:", not decision_diff["base"], decision_diff["base"][:5])
