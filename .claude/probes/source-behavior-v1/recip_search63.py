# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3: search for rows where the source reciprocal (not an exact one) decides an output
byte, then confirm with the real TFLM SOFTMAX kernel. Usage: recip_search63.py ORACLE TRACE63 OUT.json
Reuses softmax63.py's calibrations, prepared weights and recipe (loaded from its text)."""
import json, sys, subprocess
src = open(__file__.replace("recip_search63.py", "softmax63.py")).read()
ORACLE, TRACE, RESULT_PATH = sys.argv[1:4]   # not OUT: exec(head) rebinds OUT to /dev/null
sys.argv = [sys.argv[0], ORACLE, TRACE, "/dev/null"]
head = src[: src.index("# --- oracle, batched")]
exec(head)                       # calibrations, prepare (via trace63), recipe helpers
exec(src[src.index("# --- recipe ----"): src.index("# --- F24 exact-first")])
def rh_away(x): a = abs(x); r = math.floor(a + Fraction(1, 2)); return r if x >= 0 else -r
found = []
# rows [max]*k + [max+d]: S = k*B[0] + B[d], two distinct outputs; computed in closed form, the full
# recipe() is re-run on each hit below
def two_level(cal, k, d, exact):
    E = prep[(cal[0], cal[2])]["E"]; adm0, a0, b0 = E[0]; admd, ad, bd = E[d]
    S = k * b0 + (bd if admd else 0)
    if S > I32MAX: return None
    if exact:
        hpo = 32 - S.bit_length(); nb = 12 - hpo; sh = min(I32MAX, rh_away(Fraction(1 << 62, S << hpo)))
    else:
        sh, nb = get_reciprocal(S)
    ex = nb + 23
    if not 0 <= ex <= 31: return None
    f = lambda a: max(-128, min(127, rdbpot(srdhm(sh, a), ex) - 128))
    return (f(a0), f(ad) if admd else -128)
for cal in CAL:
    hit = None
    for k in range(1, 512):
        for d in range(-255, 0):
            a, b = two_level(cal, k, d, False), two_level(cal, k, d, True)
            if a is not None and a != b: hit = (cal, [100] * k + [100 + d]); break
        if hit: break
    if hit: found.append(hit)
# non-vacuity of the search: how many searched rows have a source reciprocal that differs from the
# correctly rounded one (if 0, "no hit" would say nothing about the reciprocal)
searched = recip_differs = 0
for cal in CAL:
    E = prep[(cal[0], cal[2])]["E"]
    for k in range(1, 512):
        for d in range(-255, 0):
            admd, ad, bd = E[d]; S = k * E[0][2] + (bd if admd else 0)
            if S > I32MAX: continue
            hpo = 32 - S.bit_length()
            searched += 1; recip_differs += get_reciprocal(S)[0] != min(I32MAX, rh_away(Fraction(1 << 62, S << hpo)))
rows = []
for cal, q in found:
    r = run(ORACLE, oline(cal, q) + "\n")
    o = [int(v) for v in r.stdout.split()[2:]]
    rows.append(dict(cal=cal, n=len(q), k_max=q.count(100), d=min(q) - 100,
                     recipe=recipe(cal, q)[-2:], exact_reciprocal=recipe(cal, q, ("reciprocal",))[-2:], oracle=o[-2:],
                     recipe_equals_oracle=recipe(cal, q) == o, exact_reciprocal_equals_oracle=recipe(cal, q, ("reciprocal",)) == o))
json.dump(dict(searched="per calibration: rows [max]*k + [max+d], k=1..511, d=-255..-1; first hit only", hits=rows,
               rows_searched=searched, rows_where_source_reciprocal_differs=recip_differs), open(RESULT_PATH, "w"), indent=1, default=str)
print(json.dumps(rows, indent=1, default=str))
