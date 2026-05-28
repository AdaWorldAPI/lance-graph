"""Cluster ruff-py-dto bundles by SoC synergy.

Two methods are the SAME concern iff they share:
  - name root (after stripping _compute_ / _check_ / _onchange_ / _get_ /
    _inverse_ / _search_ prefixes)
  - primary @api.* decorator family
  - delegation signature (reads + writes + invokes + raises, sorted)

Family-of-incidence is collected per cluster — the cluster carries the
list of (family, method_name) pairs it absorbed.

Output: one Markdown table, sorted by cluster size desc, with the
high-signal concerns at the top.
"""
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

BUNDLES_DIR = Path("/tmp/odoo-extract/harvest-full/bundles")
SOURCE_ROOT = Path("/home/user/odoo/addons")

PREFIX_RE = re.compile(
    r"^_+(compute|check|inverse|onchange|search|get|set|find|resolve|guess|"
    r"suggest|infer|match|detect|recommend|propose|validate|score|rank|"
    r"create|update|delete|prepare|build|generate|fetch|load|read|write)_"
)

class DelegationVisitor(ast.NodeVisitor):
    def __init__(self):
        self.invokes, self.reads, self.writes = set(), set(), set()
        self.raises, self.reads_env, self.traverses = set(), False, set()

    def _self_attr(self, n):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "self"):
            return n.attr
        return None

    def visit_Call(self, n):
        a = self._self_attr(n.func)
        if a: self.invokes.add(a)
        if isinstance(n.func, ast.Attribute):
            inner = n.func.value
            if (isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "self" and inner.attr == "env"):
                self.reads_env = True
        self.generic_visit(n)

    def visit_Subscript(self, n):
        if self._self_attr(n.value) == "env":
            self.reads_env = True
        self.generic_visit(n)

    def visit_Attribute(self, n):
        a = self._self_attr(n)
        if a and a != "env":
            self.reads.add(a)
        self.generic_visit(n)

    def visit_Assign(self, n):
        for t in n.targets:
            a = self._self_attr(t)
            if a: self.writes.add(a); self.reads.discard(a)
        self.generic_visit(n)

    def visit_For(self, n):
        a = self._self_attr(n.iter)
        if a: self.traverses.add(a); self.reads.discard(a)
        self.generic_visit(n)

    def visit_Raise(self, n):
        e = n.exc
        if isinstance(e, ast.Call):
            if isinstance(e.func, ast.Name): self.raises.add(e.func.id)
            elif isinstance(e.func, ast.Attribute): self.raises.add(e.func.attr)
        elif isinstance(e, ast.Name): self.raises.add(e.id)
        self.generic_visit(n)


def parse_cached(rel, cache):
    if rel in cache: return cache[rel]
    full = SOURCE_ROOT / rel
    try:
        cache[rel] = ast.parse(full.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        cache[rel] = None
    return cache[rel]


def find_fn(mod, name, line_start):
    cands = []
    for n in ast.walk(mod):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            rs = n.lineno
            if n.decorator_list: rs = min(d.lineno for d in n.decorator_list)
            re_ = getattr(n, "end_lineno", n.lineno) or n.lineno
            if rs <= line_start <= re_: cands.append(n)
    if not cands:
        for n in ast.walk(mod):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name and abs(n.lineno - line_start) <= 5:
                cands.append(n)
    return min(cands, key=lambda n: abs(n.lineno - line_start)) if cands else None


def name_root(method_name):
    m = PREFIX_RE.match(method_name)
    return method_name[m.end():] if m else method_name.lstrip("_")


def primary_decorator(decorators):
    for d in decorators:
        for tag in ("@api.depends_context", "@api.constrains", "@api.depends",
                    "@api.onchange", "@api.model_create_multi",
                    "@api.ondelete", "@api.model"):
            if tag in d:
                return tag
    return decorators[0].split("(")[0] if decorators else "(none)"


cache = {}
clusters = defaultdict(list)  # (root, decorator, sig) -> list of (family, method, file)

bundle_count = 0
analyzed = 0
for nd in sorted(BUNDLES_DIR.glob("*.ndjson")):
    family = nd.stem
    with nd.open() as f:
        for line in f:
            if not line.strip(): continue
            b = json.loads(line)
            bundle_count += 1
            mod = parse_cached(b["file"], cache)
            if mod is None: continue
            fn = find_fn(mod, b["function_name"], b.get("line_start", 0))
            if fn is None or not fn.body: continue
            v = DelegationVisitor()
            for s in fn.body: v.visit(s)
            sig = (
                tuple(sorted(v.reads)),
                tuple(sorted(v.writes)),
                tuple(sorted(v.invokes)),
                tuple(sorted(v.raises)),
                v.reads_env,
                tuple(sorted(v.traverses)),
            )
            root = name_root(b["function_name"])
            deco = primary_decorator(b.get("all_decorators", []))
            key = (root, deco, sig)
            clusters[key].append((family, b["function_name"], b["file"]))
            analyzed += 1

print(f"Bundles: {bundle_count}, analyzed: {analyzed}, clusters: {len(clusters)}")
print(f"Median cluster size: {sorted(len(v) for v in clusters.values())[len(clusters)//2]}")
print(f"Top-1 cluster size: {max(len(v) for v in clusters.values())}")

# Sort clusters by size desc, then by root.
sorted_clusters = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0][0]))

out = Path("/tmp/work/high-signal-concerns.md")
with out.open("w") as f:
    f.write(f"# Odoo high-signal concerns (SoC-deduplicated)\n\n")
    f.write(f"From {bundle_count} ruff-py-dto bundles, "
            f"{analyzed} delegation-resolved, "
            f"{len(clusters)} unique SoC-concerns after synergy dedup.\n\n")
    f.write("SoC key = `(name_root, primary_decorator, "
            "(reads, writes, invokes, raises, reads_env, traverses))`. "
            "Two methods are the same concern iff this tuple matches.\n\n")
    f.write(f"| # | n | root | decorator | reads | writes | invokes | raises | env | families |\n")
    f.write(f"| ---: | ---: | --- | --- | --- | --- | --- | --- | :---: | --- |\n")
    rank = 0
    for (root, deco, sig), members in sorted_clusters:
        if len(members) < 2:
            break  # cut off the singleton tail (no synergy)
        rank += 1
        reads, writes, invokes, raises, reads_env, traverses = sig
        def short(t, max_=4):
            t = list(t)
            if len(t) <= max_: return ", ".join(t) if t else "—"
            return ", ".join(t[:max_]) + f", … (+{len(t)-max_})"
        fams = sorted({m[0] for m in members})
        f.write(
            f"| {rank} | {len(members)} | `{root}` | `{deco}` | "
            f"{short(reads)} | {short(writes)} | {short(invokes)} | "
            f"{short(raises)} | {'✓' if reads_env else '—'} | "
            f"{short(fams, 5)} |\n"
        )
    f.write("\n\nSingleton concerns (one-of, no synergy): "
            f"{sum(1 for v in clusters.values() if len(v) == 1)}\n")
    # Print decorator histogram
    deco_hist = defaultdict(int)
    for (root, deco, _), members in sorted_clusters:
        if len(members) >= 2: deco_hist[deco] += 1
    f.write("\n## Concern count by primary decorator (synergistic only)\n\n")
    for k, n in sorted(deco_hist.items(), key=lambda x: -x[1]):
        f.write(f"- `{k}` — {n} concerns\n")

print(f"Wrote {out}")
