# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Amendment 6.3 addendum: independent cross-check of reach63 witnesses with the softmax63 Python recipe.
Usage (from this directory): reachcheck63.py ORACLE TRACE63"""
import json, sys
src = open("softmax63.py").read()
sys.argv = ["x", sys.argv[1], sys.argv[2], "/dev/null"]
exec(src[: src.index("# --- oracle, batched")])
def rh_away(x): a = abs(x); r = math.floor(a + Fraction(1, 2)); return r if x >= 0 else -r
exec(src[src.index("# --- recipe ----"): src.index("# --- F24 exact-first")])
w = [x for x in json.load(open("reach63.json"))["rows"] if x.get("authentic_witness")]
agree = 0
for x in w:
    q = [int(c) for c, n in sorted(x["row_code_counts"].items(), key=lambda t: -int(t[0])) for _ in range(n)]
    pos = q.index(127 + x["d"]); cal = CAL[x["cal"]]
    a, b = recipe(cal, q), recipe(cal, q, ("reciprocal",))
    agree += a[pos] == x["kernel_byte"] and b[pos] == x["exact_byte"] and a[pos] != b[pos]
print("witnesses", len(w), "python recipe reproduces kernel byte AND exact arm byte:", agree)
