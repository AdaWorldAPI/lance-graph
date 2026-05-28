"""Classify methods into openings + extract hop chains + emit Elixir shape.

Pipeline per method:
  1. parse body to AST
  2. priority-classify into one of ~15 openings (first match wins)
  3. extract dotted-attribute chains rooted at self / record-var (the "hops")
  4. per opening, correlate (1st hop, 2nd hop, 3rd hop, 4th hop) frequencies
  5. emit the top chain signatures as Elixir |> pipelines

No similarity search anywhere. Priority classifier → chain extraction →
correlation. The chains ARE the Elixir.
"""
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import textwrap

BUNDLES_DIR = Path("/tmp/odoo-extract/harvest-full/bundles")
OUT_MD = Path("/tmp/work/openings-hops.md")

# -------------------------------------------------------------------------
# Openings catalogue (priority order, first match wins)
# Each opening is (name, predicate) — the predicate inspects the body AST
# -------------------------------------------------------------------------

def _calls_super(stmt):
    """Does this stmt CALL super().something(...)?"""
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        c = stmt.value
        if isinstance(c.func, ast.Attribute):
            v = c.func.value
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super":
                return True
    if isinstance(stmt, (ast.Return, ast.Assign)) and isinstance(getattr(stmt, "value", None), ast.Call):
        c = stmt.value
        if isinstance(c.func, ast.Attribute):
            v = c.func.value
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super":
                return True
    return False


def _is_for_self(stmt):
    """`for X in self...`"""
    if not isinstance(stmt, ast.For): return False
    it = stmt.iter
    if isinstance(it, ast.Name) and it.id == "self": return True
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute):
        v = it.func.value
        if isinstance(v, ast.Name) and v.id == "self": return True
    return False


def _is_for_self_filtered(stmt):
    if not isinstance(stmt, ast.For): return False
    it = stmt.iter
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute):
        v = it.func.value
        if isinstance(v, ast.Name) and v.id == "self" and it.func.attr == "filtered":
            return True
    return False


def _body_raises(body):
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Raise): return True
    return False


def _body_assigns_self_attr_after_state_check(body):
    """state-transition with guard: check self.state, then assign self.state."""
    saw_state_check = False
    saw_state_assign = False
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" and n.attr == "state":
            if isinstance(n.ctx, ast.Store): saw_state_assign = True
            else: saw_state_check = True
    return saw_state_check and saw_state_assign


def _body_uses_sudo(body):
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "sudo":
            return True
    return False


def _body_uses_with_context(body):
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "with_context":
            return True
    return False


def _body_onchange_clear_pattern(body):
    """if not self.X: self.Y = False / None ..."""
    for n in body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not):
            if isinstance(n.test.operand, ast.Attribute) and isinstance(n.test.operand.value, ast.Name) and n.test.operand.value.id == "self":
                # body has self.Y = False/None
                for s in n.body:
                    if isinstance(s, ast.Assign):
                        for t in s.targets:
                            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                                return True
    return False


def _for_body_has_aggregate(for_body):
    """`X.Y = sum(...)` or `X.Y = ...mapped(...)`"""
    for n in ast.walk(ast.Module(body=for_body, type_ignores=[])):
        if isinstance(n, ast.Assign) and isinstance(n.value, (ast.Call,)):
            c = n.value
            if isinstance(c.func, ast.Name) and c.func.id in ("sum", "max", "min", "len", "any", "all"):
                return True
            if isinstance(c.func, ast.Attribute) and c.func.attr == "mapped":
                return True
    return False


def classify(body, fn_name):
    """Priority classifier. First match wins. Order is load-bearing."""
    if not body:
        return "empty"
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return "pass_override"
    if len(body) == 1 and _calls_super(body[0]):
        return "super_delegation_pure"
    if _calls_super(body[0]) and len(body) > 1:
        return "super_extend"
    if _body_assigns_self_attr_after_state_check(body):
        return "state_transition_with_guard"
    if _body_onchange_clear_pattern(body):
        return "onchange_clear_dependent_cascade"

    for stmt in body:
        if _is_for_self_filtered(stmt):
            if _body_raises(stmt.body):
                return "iter_filtered_raise_on_violation"
            return "iter_filtered_mutate"
        if _is_for_self(stmt):
            if _body_raises(stmt.body):
                return "iter_records_raise_on_violation"
            if _for_body_has_aggregate(stmt.body):
                return "iter_records_aggregate_relation"
            return "iter_records_compute_from_related"

    if _body_uses_sudo(body):
        return "sudo_escalation_lookup"
    if _body_uses_with_context(body):
        return "with_context_query_shift"

    if fn_name.startswith("_check_") or _body_raises(body):
        return "validator_other"
    if fn_name.startswith("_compute_"):
        return "compute_scalar_other"
    if fn_name.startswith("_onchange_"):
        return "onchange_other"
    return "other"


# -------------------------------------------------------------------------
# Hop extraction: dotted-attribute chains rooted at self or recordset-loop-var
# -------------------------------------------------------------------------

RECORDSET_NAMES = {"self", "record", "rec", "line", "move", "order", "invoice",
                   "payment", "partner", "product", "employee", "task", "lead",
                   "picking", "production", "event", "channel", "user", "company",
                   "journal", "tax", "account"}


def chain_from_attribute(node):
    """Walk an Attribute node down to its root Name; return (root_id, [hops...]) or None."""
    hops = []
    while isinstance(node, ast.Attribute):
        hops.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, list(reversed(hops))
    return None


def extract_hop_chains(body):
    """Walk all assignments + returns; collect chains rooted at recordset names."""
    chains = []
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, ast.Attribute):
            res = chain_from_attribute(n)
            if res:
                root, hops = res
                if root in RECORDSET_NAMES and len(hops) >= 1:
                    chains.append(tuple(hops))
    return chains


# -------------------------------------------------------------------------
# Normalize body & run
# -------------------------------------------------------------------------

def parse_body(body_source):
    if not body_source.strip():
        return None
    lines = body_source.splitlines()
    if len(lines) == 1:
        wrapped = f"def _f(self):\n    {body_source.strip()}\n"
    else:
        first = lines[0]
        if first.strip() and (len(first) - len(first.lstrip())) == 0:
            first = " " * 8 + first
        norm = "\n".join([first] + lines[1:])
        dedented = textwrap.dedent(norm)
        wrapped = "def _f(self):\n" + textwrap.indent(dedented, "    ")
    try:
        return ast.parse(wrapped).body[0].body
    except SyntaxError:
        return None


# opening -> (chain1_counts, chain2_counts_per_chain1, ...)
opening_method_count = Counter()
opening_chain_full = defaultdict(Counter)        # opening -> Counter of full chain tuples
opening_hop1 = defaultdict(Counter)              # opening -> Counter of 1st hop
opening_hop_chains_by_depth = defaultdict(lambda: defaultdict(Counter))  # opening -> depth -> Counter

method_chains_sample = defaultdict(list)         # opening -> [(family, fn, chain), ...]

for nd in sorted(BUNDLES_DIR.glob("*.ndjson")):
    family = nd.stem
    with nd.open() as f:
        for line in f:
            if not line.strip(): continue
            b = json.loads(line)
            body = parse_body(b.get("body_source", ""))
            if body is None: continue
            opening = classify(body, b["function_name"])
            opening_method_count[opening] += 1
            chains = extract_hop_chains(body)
            # Keep only the LONGEST chain per assignment context, dedup short prefixes.
            chain_strs = sorted({tuple(c) for c in chains}, key=lambda c: -len(c))
            seen = []
            for c in chain_strs:
                if not any(other[:len(c)] == c and len(other) > len(c) for other in seen):
                    seen.append(c)
            for c in seen[:6]:  # cap per-method
                opening_chain_full[opening][c] += 1
                if len(c) >= 1: opening_hop1[opening][c[0]] += 1
                opening_hop_chains_by_depth[opening][len(c)][c] += 1
                if len(method_chains_sample[opening]) < 6:
                    method_chains_sample[opening].append((family, b["function_name"], c))


# -------------------------------------------------------------------------
# Emit Elixir-shape for top chains per opening
# -------------------------------------------------------------------------

def to_elixir(opening, chain, fn_name):
    """Render a (opening, chain) as the Elixir pipeline shape."""
    pipe = " |> ".join(f":{h}" for h in chain)
    return f"def {fn_name}(record) do\n  record |> {pipe}\nend"


with OUT_MD.open("w") as f:
    f.write("# Odoo openings × hop chains — the Elixir shape\n\n")
    f.write(f"**From** {sum(opening_method_count.values())} method bodies (priority-classified, "
            "first-match wins). **Hop chains** are dotted-attribute paths rooted at recordset "
            "names (`self`, `record`, `line`, `move`, …).\n\n")
    f.write("## Opening distribution\n\n")
    f.write("| opening | methods | most common 1st hop | depth-1 | depth-2 | depth-3 | depth-≥4 |\n")
    f.write("| --- | ---: | --- | ---: | ---: | ---: | ---: |\n")
    for opening, n in sorted(opening_method_count.items(), key=lambda kv: -kv[1]):
        h1 = opening_hop1[opening].most_common(1)
        h1str = f"`{h1[0][0]}` ({h1[0][1]})" if h1 else "—"
        d = opening_hop_chains_by_depth[opening]
        d1 = sum(d[1].values()); d2 = sum(d[2].values()); d3 = sum(d[3].values())
        d4plus = sum(sum(d[k].values()) for k in d if k >= 4)
        f.write(f"| `{opening}` | {n} | {h1str} | {d1} | {d2} | {d3} | {d4plus} |\n")
    f.write("\n## Per-opening: top hop chains (correlated 1st → 2nd → 3rd → 4th)\n\n")
    for opening, n in sorted(opening_method_count.items(), key=lambda kv: -kv[1]):
        if opening in ("empty",): continue
        f.write(f"### `{opening}` ({n} methods)\n\n")
        f.write("**Top hop chains by frequency:**\n\n")
        f.write("| n | chain |\n| ---: | --- |\n")
        for chain, count in opening_chain_full[opening].most_common(10):
            chain_str = ".".join(chain)
            f.write(f"| {count} | `{chain_str}` |\n")
        f.write("\n**Sample Elixir emission for top chain:**\n\n")
        if opening_chain_full[opening]:
            top_chain = opening_chain_full[opening].most_common(1)[0][0]
            sample_fn = method_chains_sample[opening][0] if method_chains_sample[opening] else None
            fn_name = sample_fn[1] if sample_fn else "method"
            f.write("```elixir\n")
            f.write(to_elixir(opening, top_chain, fn_name))
            f.write("\n```\n\n")

print(f"Total methods classified: {sum(opening_method_count.values())}")
print(f"Openings: {len(opening_method_count)}")
print()
print("Opening distribution:")
for opening, n in sorted(opening_method_count.items(), key=lambda kv: -kv[1]):
    print(f"  {n:>4}  {opening}")
print()
print(f"Wrote {OUT_MD}")
