"""Deterministic ontology emit — v2, fixes emitted_by undercount.

v1 used parquet writes_fields (self.X only). The dominant Odoo compute
pattern is `for record in self: record.<field> = ...` — writes to the
LOOP VARIABLE, not self. v2 re-parses body_source from the bundles and
captures attribute-writes to ANY name (self OR loop var) so emitted_by
is complete.

Same triple schema + truth values as v1.
"""
import ast
import json
import re
import textwrap
from collections import Counter
from pathlib import Path

BUNDLES = Path("/tmp/odoo-extract/harvest-full/bundles")
OUT = Path("/tmp/work/odoo-ontology.spo.ndjson")

DEPENDS_RE = re.compile(r"@api\.depends(?:_context)?\(([^)]*)\)")
ARG_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")


class Writes(ast.NodeVisitor):
    """Capture attribute writes to self OR any single-Name loop/local var:
    `self.X = ...`, `record.X = ...`, `line.X = ...` — the computed fields.
    Also capture raises + self.<rel> for-loop traversals + reads.
    """
    def __init__(self):
        self.writes = set()
        self.reads = set()
        self.raises = set()
        self.traverses = set()

    def _attr_target(self, node):
        # X.field where X is a bare Name → field
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.attr
        return None

    def visit_Assign(self, n):
        for t in n.targets:
            a = self._attr_target(t)
            if a:
                self.writes.add(a)
        self.generic_visit(n)

    def visit_AugAssign(self, n):
        a = self._attr_target(n.target)
        if a:
            self.writes.add(a)
        self.generic_visit(n)

    def visit_Attribute(self, n):
        # reads: self.field (only self, to avoid noise from loop vars)
        if isinstance(n.value, ast.Name) and n.value.id == "self" and n.attr != "env":
            self.reads.add(n.attr)
        self.generic_visit(n)

    def visit_For(self, n):
        # for x in self.<rel>:
        it = n.iter
        if isinstance(it, ast.Attribute) and isinstance(it.value, ast.Name) and it.value.id == "self":
            self.traverses.add(it.attr)
        self.generic_visit(n)

    def visit_Raise(self, n):
        e = n.exc
        if isinstance(e, ast.Call):
            if isinstance(e.func, ast.Name):
                self.raises.add(e.func.id)
            elif isinstance(e.func, ast.Attribute):
                self.raises.add(e.func.attr)
        elif isinstance(e, ast.Name):
            self.raises.add(e.id)
        self.generic_visit(n)


def parse_body(src):
    if not src.strip():
        return None
    lines = src.splitlines()
    if len(lines) == 1:
        wrapped = f"def _f(self):\n    {src.strip()}\n"
    else:
        first = lines[0]
        if first.strip() and (len(first) - len(first.lstrip())) == 0:
            first = " " * 8 + first
        norm = "\n".join([first] + lines[1:])
        wrapped = "def _f(self):\n" + textwrap.indent(textwrap.dedent(norm), "    ")
    try:
        return ast.parse(wrapped).body[0].body
    except SyntaxError:
        return None


triples = []
seen = set()

def emit(s, p, o, f, c):
    if (s, p, o) in seen:
        return
    seen.add((s, p, o))
    triples.append({"s": s, "p": p, "o": o, "f": f, "c": c})


families = set()
fields_declared = set()
funcs_declared = set()
# field names known to exist (from writes) — used to keep depends_on edges
# pointing at real fields where possible (the dep's first segment).
known_fields_per_family = {}

# First pass: collect all written fields per family (so depends targets resolve)
records = []
for nd in sorted(BUNDLES.glob("*.ndjson")):
    family = nd.stem
    with nd.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            b = json.loads(line)
            body = parse_body(b.get("body_source", ""))
            w = Writes()
            if body:
                for s in body:
                    w.visit(s)
            deps = []
            for deco in (b.get("all_decorators") or []):
                m = DEPENDS_RE.search(deco)
                if m:
                    for a in ARG_RE.finditer(m.group(1)):
                        deps.append(a.group(1) or a.group(2))
            rec = {
                "family": family,
                "fn": b["function_name"],
                "writes": sorted(w.writes),
                "reads": sorted(w.reads),
                "raises": sorted(w.raises),
                "traverses": sorted(w.traverses),
                "deps": deps,
            }
            records.append(rec)
            known_fields_per_family.setdefault(family, set()).update(w.writes)

for rec in records:
    family = rec["family"]
    fn = rec["fn"]
    fam_iri = f"odoo:{family}"
    fn_iri = f"odoo:{family}.{fn}"
    families.add(family)

    emit(fam_iri, "rdf:type", "ogit:ObjectType", 1.0, 1.0)
    if fn_iri not in funcs_declared:
        emit(fn_iri, "rdf:type", "ogit:Function", 1.0, 1.0)
        emit(fam_iri, "has_function", fn_iri, 1.0, 0.95)
        funcs_declared.add(fn_iri)

    written = rec["writes"]
    for wf in written:
        field_iri = f"odoo:{family}.{wf}"
        if field_iri not in fields_declared:
            emit(field_iri, "rdf:type", "ogit:Property", 1.0, 1.0)
            fields_declared.add(field_iri)
        emit(field_iri, "emitted_by", fn_iri, 0.95, 0.90)

    targets = [f"odoo:{family}.{wf}" for wf in written] or [fn_iri]
    for dep in rec["deps"]:
        dep_iri = f"odoo:{family}.{dep}"
        for tgt in targets:
            emit(tgt, "depends_on", dep_iri, 0.95, 0.90)

    for rf in rec["reads"]:
        emit(fn_iri, "reads_field", f"odoo:{family}.{rf}", 0.85, 0.75)
    for exc in rec["raises"]:
        emit(fn_iri, "raises", f"exc:{exc}", 0.95, 0.90)
    for tr in rec["traverses"]:
        emit(fn_iri, "traverses_relation", f"odoo:{family}.{tr}", 0.85, 0.75)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    for tr in triples:
        f.write(json.dumps(tr, separators=(",", ":")) + "\n")

pred_hist = Counter(tr["p"] for tr in triples)
print(f"Methods read:        {len(records)}")
print(f"Object Types:        {len(families)}")
print(f"Properties declared: {len(fields_declared)}")
print(f"Functions declared:  {len(funcs_declared)}")
print(f"Total unique triples:{len(triples)}")
print()
print("Predicate histogram:")
for p, c in pred_hist.most_common():
    print(f"  {c:>6}  {p}")
print()

def show(field):
    print(f"--- {field} ---")
    rows = [tr for tr in triples if tr["s"] == field and tr["p"] in ("depends_on", "emitted_by")]
    for tr in rows[:12]:
        print(f"    {tr['p']:12} -> {tr['o']}")
    if not rows:
        print("    (no edges)")

show("odoo:account_move.amount_total")
show("odoo:sale_order.amount_total")
print(f"\nWrote {OUT} ({OUT.stat().st_size} bytes)")
