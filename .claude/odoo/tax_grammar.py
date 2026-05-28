"""Tax-method grammar coding — the high-signal lens.

For each tax-related method, code along the right axes:
  - Transitivity (T/I) from body: raise-without-return = I, else T
  - TEKAMOLO slot from body markers:
      TE  temporal      (date, period, lock_date, fiscalyear)
      KA  causal        (raise + validation, regulation implementation)
      MO  modal         (if/elif conditional branching)
      LO  locative      (country, jurisdiction, fiscal_position)
      QU  quantities    (money/percentage compute)
  - Mengenmaß: which type of quantity flows through?
      money | percent | rate | count | date | categorical | none
  - Regulatory anchor: which paragraph implements/cites?
      UStG | HGB | EStG | AO | GoBD | SKR04 | DATEV | ELSTER | none
  - Chain position: who calls this, who does this call (from delegation)

Output: tight table per method, plus chain visualization for the tax
compute pipeline.
"""
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

BUNDLES_DIR = Path("/tmp/odoo-extract/harvest-full/bundles")
SOURCE_ROOT = Path("/home/user/odoo/addons")

# Markers per TEKAMOLO slot (body text contains)
TEKAMOLO_MARKERS = {
    "TE": ["date", "period", "lock_date", "fiscalyear", "due_date", "maturity", "_at"],
    "KA": ["raise ", "ValidationError", "UserError", "AccessError"],
    "MO": ["if ", "elif "],
    "LO": ["country", "jurisdiction", "fiscal_position", "tax_country", "_de", "_fr", "_us"],
    "QU": ["amount", "balance", "debit", "credit", "rate", "percentage", "discount", "quantity", "subtotal", "total"],
}

MENGENMASS_MARKERS = {
    "money":    ["amount", "balance", "debit", "credit", "subtotal", "total", "price", "fee"],
    "percent":  ["percentage", "percent", "discount_perc"],
    "rate":     ["rate", "tax_rate", "currency_rate", "exchange_rate"],
    "count":    ["quantity", "qty", "count", "number"],
    "date":     ["date", "_at", "period", "fiscalyear", "due"],
}

REGULATORY_MARKERS = {
    "UStG":   ["UStG", "Umsatzsteuer", "VAT", "vat_", "tax_country"],
    "HGB":    ["HGB", "Festschreibung", "Geschäftsjahr", "fiscalyear_lock"],
    "EStG":   ["EStG", "Einkommensteuer"],
    "AO":     ["AO ", "Abgabenordnung", "tax_lock"],
    "GoBD":   ["GoBD", "Unveraenderbarkeit", "restrictive_audit", "audit_trail"],
    "SKR04":  ["SKR04", "skr04", "skr_04"],
    "SKR03":  ["SKR03", "skr03", "skr_03"],
    "DATEV":  ["DATEV", "datev"],
    "ELSTER": ["ELSTER", "elster"],
    "Peppol": ["Peppol", "peppol", "UBL", "ubl_"],
}


def code_method(body_source, function_name):
    body = body_source or ""
    # Transitivity
    has_raise = "raise " in body
    has_return = re.search(r"\breturn\b", body) is not None
    if function_name.startswith(("_check_", "_validate_")) or (has_raise and not has_return):
        transitivity = "I"
    else:
        transitivity = "T"

    # TEKAMOLO slot — pick the dominant
    tek_scores = {}
    for slot, markers in TEKAMOLO_MARKERS.items():
        tek_scores[slot] = sum(body.count(m) for m in markers)
    tek = max(tek_scores, key=tek_scores.get) if tek_scores else "—"
    if tek_scores[tek] == 0: tek = "—"

    # Mengenmaß
    men_scores = {}
    for kind, markers in MENGENMASS_MARKERS.items():
        men_scores[kind] = sum(body.count(m) for m in markers)
    men = max(men_scores, key=men_scores.get) if men_scores else "none"
    if men_scores[men] == 0: men = "none"

    # Regulatory anchor (first that matches)
    regs = []
    blob = body + " " + function_name
    for reg, markers in REGULATORY_MARKERS.items():
        if any(m in blob for m in markers):
            regs.append(reg)
    reg = "+".join(regs) if regs else "—"

    return transitivity, tek, men, reg


# Collect tax-related methods.
TAX_RE = re.compile(r"tax|vat|umsatzsteuer|ustva|withholding", re.IGNORECASE)

tax_methods = []  # list of dicts
for nd in sorted(BUNDLES_DIR.glob("*.ndjson")):
    family = nd.stem
    family_is_tax = TAX_RE.search(family) is not None
    with nd.open() as f:
        for line in f:
            if not line.strip(): continue
            b = json.loads(line)
            fn = b["function_name"]
            fn_is_tax = TAX_RE.search(fn) is not None
            if not (family_is_tax or fn_is_tax):
                # Also include @api.depends/constrains methods that mention tax in deco args
                deco_blob = " ".join(b.get("all_decorators", []))
                if not TAX_RE.search(deco_blob):
                    continue
            tr, tek, men, reg = code_method(b.get("body_source", ""), fn)
            tax_methods.append({
                "family": family,
                "fn": fn,
                "file": b["file"],
                "decorator": b.get("all_decorators", ["(none)"])[0] if b.get("all_decorators") else "(none)",
                "T": tr,
                "tek": tek,
                "men": men,
                "reg": reg,
                "body_lines": b.get("body_lines", 0),
            })

print(f"Tax-related methods: {len(tax_methods)}")

# Group by (T, tek, men, reg) — these are the grammar-coded synergy clusters.
from collections import Counter
grammar_clusters = Counter()
for m in tax_methods:
    key = (m["T"], m["tek"], m["men"], m["reg"])
    grammar_clusters[key] += 1

print(f"Grammar-coded clusters: {len(grammar_clusters)}")
print()
print("Top grammar-coded patterns (n >= 3):")
for key, n in grammar_clusters.most_common():
    if n < 3: break
    print(f"  {n:>3}  T={key[0]}  tek={key[1]:<3}  men={key[2]:<10}  reg={key[3]}")

# Also extract invoke-chains for the tax subset.
# Re-parse each tax method, collect self.X(...) invocations.
class InvokeV(ast.NodeVisitor):
    def __init__(self): self.invokes = []
    def visit_Call(self, n):
        if (isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "self"):
            self.invokes.append(n.func.attr)
        self.generic_visit(n)

cache = {}
def parse_cached(rel):
    if rel in cache: return cache[rel]
    try: cache[rel] = ast.parse((SOURCE_ROOT / rel).read_text(encoding="utf-8", errors="replace"))
    except Exception: cache[rel] = None
    return cache[rel]

invoke_edges = []
for m in tax_methods:
    mod = parse_cached(m["file"])
    if mod is None: continue
    # find the function
    fn_node = None
    for n in ast.walk(mod):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == m["fn"]:
            fn_node = n
            break
    if fn_node is None or not fn_node.body: continue
    v = InvokeV()
    for s in fn_node.body: v.visit(s)
    for callee in v.invokes:
        invoke_edges.append((f"{m['family']}.{m['fn']}", callee))

# Restrict to invoke-edges where callee is also in tax_methods set.
tax_fn_set = {f"{m['family']}.{m['fn']}".split(".")[-1] for m in tax_methods}
tax_invoke_edges = [(s, c) for s, c in invoke_edges if c in tax_fn_set]
print()
print(f"Total invoke edges from tax methods: {len(invoke_edges)}")
print(f"Tax->tax invoke edges (within-subgraph): {len(tax_invoke_edges)}")

# Build a small chain summary: for each method, who calls it and what it calls within tax subset
caller_of = defaultdict(list)
callee_of = defaultdict(list)
for s, c in tax_invoke_edges:
    callee_of[s].append(c)
    caller_of[c].append(s)

# Sample the chain heads (methods called BY others) and tails (methods that call none)
chain_heads = [m for m in tax_methods if not callee_of[f"{m['family']}.{m['fn']}"]]  # leaf computers
chain_tails = [m for m in tax_methods if not any(m['fn'] in p for p in caller_of)]   # not called by anyone

# Emit the artifact.
out = Path("/tmp/work/tax-grammar-coded.md")
with out.open("w") as f:
    f.write(f"# Odoo tax: grammar-coded methods\n\n")
    f.write(f"**Scope:** {len(tax_methods)} tax-related methods "
            f"(family or fn or decorator-arg matches `tax|vat|umsatzsteuer|ustva|withholding`).\n\n")
    f.write(f"**Coding axes** (per E-BUSINESS-LOGIC-IS-GRAMMAR-1):\n\n")
    f.write(f"- **T** — transitivity: T (transitive, returns/mutates) | I (intransitive, raises without return)\n")
    f.write(f"- **tek** — TEKAMOLO slot: TE (temporal) | KA (causal/regulatory) | MO (modal) | LO (locative) | QU (quantities)\n")
    f.write(f"- **men** — mengenmaß: money | percent | rate | count | date | none\n")
    f.write(f"- **reg** — regulatory anchor: UStG | HGB | EStG | AO | GoBD | SKR04 | SKR03 | DATEV | ELSTER | Peppol\n\n")
    f.write(f"## Top grammar-coded clusters (n ≥ 3)\n\n")
    f.write(f"| n | T | tek | men | reg |\n")
    f.write(f"| ---: | :---: | :---: | --- | --- |\n")
    for key, n in grammar_clusters.most_common():
        if n < 3: break
        f.write(f"| {n} | {key[0]} | {key[1]} | {key[2]} | {key[3]} |\n")
    f.write(f"\n## Top 50 methods by family\n\n")
    f.write(f"| family | fn | T | tek | men | reg | LOC |\n")
    f.write(f"| --- | --- | :---: | :---: | --- | --- | ---: |\n")
    by_family = defaultdict(list)
    for m in tax_methods:
        by_family[m["family"]].append(m)
    rows = 0
    for fam in sorted(by_family.keys()):
        for m in by_family[fam][:5]:  # cap 5 per family
            f.write(f"| {fam} | `{m['fn']}` | {m['T']} | {m['tek']} | {m['men']} | {m['reg']} | {m['body_lines']} |\n")
            rows += 1
            if rows >= 50: break
        if rows >= 50: break
    f.write(f"\n## Within-subgraph invoke chains (tax → tax)\n\n")
    f.write(f"{len(tax_invoke_edges)} edges among {len({s for s, _ in tax_invoke_edges} | {c for _, c in tax_invoke_edges})} "
            f"distinct nodes.\n\n")
    # Show top callees (methods that anchor chains)
    callee_counts = Counter(c for _, c in tax_invoke_edges)
    f.write(f"**Top chain anchors** (most-invoked tax methods within subgraph):\n\n")
    for c, n in callee_counts.most_common(15):
        f.write(f"- `{c}` ({n} callers)\n")
    f.write(f"\n**Sample chain (one walk from a high-anchor node):**\n\n")
    # Walk from the most-called anchor backward to find a chain
    if callee_counts:
        top = callee_counts.most_common(1)[0][0]
        f.write(f"```\n")
        f.write(f"  {top}  <-- called by:\n")
        for s in caller_of[top][:8]:
            f.write(f"    {s}\n")
        f.write(f"```\n")

print(f"\nWrote {out}")
