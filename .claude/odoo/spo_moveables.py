"""Second pass: SPO extraction + moveable-parts classification.

For each method body, extract assignment-level SPO triples:
  subject   = LHS recordset name (record / line / move / order / ...)
  predicate = the verb (assign / write_attr / raise / call)
  object    = the RHS expression as a STRUCTURAL TEMPLATE

Field names + literals are replaced by positional placeholders {0}, {1}, ...
The remaining shape IS the method-emit template. Methods sharing one template
differ ONLY in their parameter tuples — those parameters are the moveable parts.

Output:
  - Per opening: the top method-emit templates
  - For each template: the parameter slot list + the (family, fn) instances
  - Decomposition: N methods → K templates → each fits with M parameters
"""
import ast
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

BUNDLES_DIR = Path("/tmp/odoo-extract/harvest-full/bundles")
OUT_MD = Path("/tmp/work/spo-moveables.md")

# Reuse the opening classifier from openings_hops.py — same priority order.
import sys
sys.path.insert(0, str(Path("/tmp/work")))
from openings_hops import classify, parse_body, RECORDSET_NAMES  # noqa


# -------------------------------------------------------------------------
# Field-name normalisation: replace attribute names with positional slots
# -------------------------------------------------------------------------

class FieldAbstractor(ast.NodeTransformer):
    """Replace ATTRIBUTE names with positional placeholders.
    `record.X.Y` → `record.{0}.{1}`. String/number literals → `{LIT}`.
    Keeps structure (call shape, if/for/etc.) intact.
    """
    def __init__(self):
        self.slot = 0
        self.slot_map = {}  # original name → slot number (so same name → same slot)
        self.literals = []  # ordered list of literal strings

    def _slot_for(self, name):
        if name not in self.slot_map:
            self.slot_map[name] = self.slot
            self.slot += 1
        return self.slot_map[name]

    def visit_Attribute(self, node):
        # Only abstract leaf names; recurse into .value first.
        self.generic_visit(node)
        # Don't abstract `self` / recordset roots — they ARE the structure.
        if isinstance(node.value, ast.Name) and node.value.id in RECORDSET_NAMES:
            new_attr = f"_{self._slot_for(node.attr)}"
        else:
            new_attr = f"_{self._slot_for(node.attr)}"
        return ast.copy_location(
            ast.Attribute(value=node.value, attr=new_attr, ctx=node.ctx),
            node
        )

    def visit_Constant(self, node):
        # Replace string/int/float literals with a literal slot.
        if isinstance(node.value, (str, int, float)) and not isinstance(node.value, bool):
            self.literals.append(repr(node.value))
            return ast.copy_location(ast.Name(id=f"_LIT{len(self.literals)-1}", ctx=ast.Load()), node)
        return node


def abstract_body(body):
    """Return (template_source, slot_names_in_order, literals_in_order)."""
    abst = FieldAbstractor()
    new = abst.visit(ast.Module(body=body, type_ignores=[]))
    try:
        src = ast.unparse(new)
    except Exception:
        return None, [], []
    # Reverse the slot map: slot_idx → original name
    slots = [None] * abst.slot
    for name, idx in abst.slot_map.items():
        slots[idx] = name
    return src, slots, abst.literals


# -------------------------------------------------------------------------
# Subject / Predicate / Object — extract from each top-level statement
# -------------------------------------------------------------------------

def spo_of_stmt(stmt):
    """Yield (subject, predicate, object_repr) tuples from a statement.

    subject   = recordset variable name on the LHS (or '<implicit>' for raises)
    predicate = 'assign' | 'raise' | 'call' | 'return'
    object    = a compact textual repr of the RHS shape
    """
    triples = []

    # Direct assignment: X.Y = expr
    if isinstance(stmt, ast.Assign):
        for tgt in stmt.targets:
            if isinstance(tgt, ast.Attribute):
                root = tgt.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                subj = root.id if isinstance(root, ast.Name) else "<expr>"
                try:
                    obj_repr = ast.unparse(stmt.value)
                except Exception:
                    obj_repr = "<unparseable>"
                triples.append((subj, "assign", obj_repr[:120]))

    elif isinstance(stmt, ast.Return):
        if stmt.value is not None:
            try:
                triples.append(("<return>", "return", ast.unparse(stmt.value)[:120]))
            except Exception:
                pass

    elif isinstance(stmt, ast.Raise):
        if stmt.exc is not None:
            try:
                triples.append(("<raise>", "raise", ast.unparse(stmt.exc)[:120]))
            except Exception:
                pass

    elif isinstance(stmt, ast.For):
        for s in stmt.body:
            triples.extend(spo_of_stmt(s))

    elif isinstance(stmt, ast.If):
        for s in stmt.body:
            triples.extend(spo_of_stmt(s))
        for s in stmt.orelse:
            triples.extend(spo_of_stmt(s))

    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        try:
            triples.append(("<call>", "call", ast.unparse(stmt.value)[:120]))
        except Exception:
            pass

    return triples


# -------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------

# opening → Counter of template source
opening_templates = defaultdict(Counter)
# opening → template → list of (family, fn, slots, literals)
opening_template_instances = defaultdict(lambda: defaultdict(list))
# opening → method count
opening_count = Counter()

for nd in sorted(BUNDLES_DIR.glob("*.ndjson")):
    family = nd.stem
    with nd.open() as f:
        for line in f:
            if not line.strip(): continue
            b = json.loads(line)
            body = parse_body(b.get("body_source", ""))
            if body is None: continue
            opening = classify(body, b["function_name"])
            opening_count[opening] += 1
            template_src, slots, literals = abstract_body(body)
            if template_src is None: continue
            opening_templates[opening][template_src] += 1
            if len(opening_template_instances[opening][template_src]) < 10:
                opening_template_instances[opening][template_src].append({
                    "family": family,
                    "fn": b["function_name"],
                    "slots": slots,
                    "literals": literals,
                })

# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------

total = sum(opening_count.values())
total_templates = sum(len(c) for c in opening_templates.values())
total_multi_inst = sum(sum(1 for v in c.values() if v >= 2) for c in opening_templates.values())
methods_in_multi = sum(sum(v for v in c.values() if v >= 2) for c in opening_templates.values())

with OUT_MD.open("w") as f:
    f.write("# SPO templates × moveable parts — second pass\n\n")
    f.write(f"From {total} methods, field-name-abstracted templates yield "
            f"**{total_templates} unique templates**. **{total_multi_inst} templates "
            f"have ≥ 2 instances** (covering {methods_in_multi} methods); the rest "
            f"are singletons.\n\n")
    f.write("**Moveable parts** = the slot list `[_0, _1, _2, …]` that varies between "
            "instances of the same template. **Method-emit template** = the body source "
            "with slots in place of field names + literals.\n\n")
    f.write("## Decomposition per opening\n\n")
    f.write("| opening | methods | unique templates | ≥2-instance templates | methods in those | "
            "singleton templates |\n")
    f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
    for opening, n in sorted(opening_count.items(), key=lambda kv: -kv[1]):
        templates = opening_templates[opening]
        n_templ = len(templates)
        n_multi = sum(1 for v in templates.values() if v >= 2)
        n_methods_multi = sum(v for v in templates.values() if v >= 2)
        n_single = sum(1 for v in templates.values() if v == 1)
        f.write(f"| `{opening}` | {n} | {n_templ} | {n_multi} | {n_methods_multi} | {n_single} |\n")
    f.write("\n## Top templates per opening (≥2 instances only)\n\n")
    for opening, n in sorted(opening_count.items(), key=lambda kv: -kv[1]):
        templates = opening_templates[opening]
        multi = [(t, c) for t, c in templates.most_common() if c >= 2]
        if not multi: continue
        f.write(f"### `{opening}` ({n} methods, {len(multi)} reusable templates)\n\n")
        for i, (templ, cnt) in enumerate(multi[:5], 1):
            f.write(f"**Template #{i}** — {cnt} methods\n\n")
            f.write("```python\n")
            f.write(templ[:800])
            if len(templ) > 800: f.write("\n... (truncated)")
            f.write("\n```\n\n")
            instances = opening_template_instances[opening][templ][:4]
            slot_count = len(instances[0]["slots"]) if instances else 0
            f.write(f"**Moveable slots** ({slot_count}): per-instance parameter tuples →\n\n")
            f.write("| family | fn | slot values |\n| --- | --- | --- |\n")
            for inst in instances:
                slot_vals = ", ".join(f"`{s}`" for s in inst["slots"])
                f.write(f"| {inst['family']} | `{inst['fn']}` | {slot_vals} |\n")
            f.write("\n")

# Print headline.
print(f"Methods: {total}")
print(f"Unique templates (post field-abstraction): {total_templates}")
print(f"Multi-instance templates (≥2): {total_multi_inst}")
print(f"Methods inside multi-instance templates: {methods_in_multi}")
print(f"Decomposition: {total} methods → {total_templates} templates ({100*(total-total_templates)/total:.1f}% compression at template level)")
print()
print("Top reductions per opening:")
for opening, n in sorted(opening_count.items(), key=lambda kv: -kv[1])[:10]:
    templates = opening_templates[opening]
    multi = sum(v for v in templates.values() if v >= 2)
    n_multi_t = sum(1 for v in templates.values() if v >= 2)
    print(f"  {opening:40s}  {n:>5} methods → {len(templates):>4} templates ({n_multi_t:>3} reusable × {multi:>4} methods)")

print(f"\nWrote {OUT_MD}")
