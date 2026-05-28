"""Decompose 3555 bundles into atomic SoC primitives.

Atoms (each = one minimal SoC fragment):
  read(F)       — method reads self.<F>
  write(F)      — method writes self.<F>
  invoke(M)     — method calls self.<M>(...)
  raises(E)     — method raises <E>(...)
  env           — method touches self.env (singleton atom)
  traverse(R)   — `for x in self.<R>` (relation traversal)

A method's full SoC = the multiset of atoms it carries.

Goal: at the tuple level the cluster cardinality is 1 (singleton)
because the FULL tuple is unique. At the ATOM level the same method
shares atoms with many others. We show:
  - the atom catalogue (top-N frequencies)
  - singleton collapse: |unique atoms across all methods| <<
    sum_of_per_method_atom_counts
  - per-singleton atom multiset (sorted by atom freq desc)
"""
import ast
import json
import re
from collections import defaultdict, Counter
from pathlib import Path

BUNDLES_DIR = Path("/tmp/odoo-extract/harvest-full/bundles")
SOURCE_ROOT = Path("/home/user/odoo/addons")

PREFIX_RE = re.compile(
    r"^_+(compute|check|inverse|onchange|search|get|set|find|resolve|guess|"
    r"suggest|infer|match|detect|recommend|propose|validate|score|rank|"
    r"create|update|delete|prepare|build|generate|fetch|load|read|write)_"
)

class V(ast.NodeVisitor):
    def __init__(self):
        self.atoms = []  # list of atom strings
    def _sa(self, n):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "self"):
            return n.attr
        return None
    def visit_Call(self, n):
        a = self._sa(n.func)
        if a: self.atoms.append(f"invoke:{a}")
        if isinstance(n.func, ast.Attribute):
            inner = n.func.value
            if (isinstance(inner, ast.Attribute) and isinstance(inner.value, ast.Name)
                    and inner.value.id == "self" and inner.attr == "env"):
                self.atoms.append("env")
        self.generic_visit(n)
    def visit_Subscript(self, n):
        if self._sa(n.value) == "env":
            self.atoms.append("env")
        self.generic_visit(n)
    def visit_Attribute(self, n):
        a = self._sa(n)
        if a and a != "env":
            self.atoms.append(f"read:{a}")
        self.generic_visit(n)
    def visit_Assign(self, n):
        for t in n.targets:
            a = self._sa(t)
            if a: self.atoms.append(f"write:{a}")
        self.generic_visit(n)
    def visit_For(self, n):
        a = self._sa(n.iter)
        if a: self.atoms.append(f"traverse:{a}")
        self.generic_visit(n)
    def visit_Raise(self, n):
        e = n.exc
        if isinstance(e, ast.Call):
            if isinstance(e.func, ast.Name): self.atoms.append(f"raise:{e.func.id}")
            elif isinstance(e.func, ast.Attribute): self.atoms.append(f"raise:{e.func.attr}")
        elif isinstance(e, ast.Name): self.atoms.append(f"raise:{e.id}")
        self.generic_visit(n)


def parse_cached(rel, cache):
    if rel in cache: return cache[rel]
    try:
        cache[rel] = ast.parse((SOURCE_ROOT / rel).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        cache[rel] = None
    return cache[rel]


def find_fn(mod, name, ls):
    cands = []
    for n in ast.walk(mod):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            rs = n.lineno
            if n.decorator_list: rs = min(d.lineno for d in n.decorator_list)
            re_ = getattr(n, "end_lineno", n.lineno) or n.lineno
            if rs <= ls <= re_: cands.append(n)
    if not cands:
        for n in ast.walk(mod):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name and abs(n.lineno - ls) <= 5:
                cands.append(n)
    return min(cands, key=lambda n: abs(n.lineno - ls)) if cands else None


def name_root(n):
    m = PREFIX_RE.match(n)
    return n[m.end():] if m else n.lstrip("_")


cache = {}
methods = []      # list of dicts: family, fn, root, decorator, atoms
atom_freq = Counter()
total_atom_emissions = 0

for nd in sorted(BUNDLES_DIR.glob("*.ndjson")):
    family = nd.stem
    with nd.open() as f:
        for line in f:
            if not line.strip(): continue
            b = json.loads(line)
            mod = parse_cached(b["file"], cache)
            if mod is None: continue
            fn = find_fn(mod, b["function_name"], b.get("line_start", 0))
            if fn is None or not fn.body: continue
            v = V()
            for s in fn.body: v.visit(s)
            atoms = sorted(set(v.atoms))  # dedup atoms per method (multiset → set)
            atom_freq.update(atoms)
            total_atom_emissions += len(atoms)
            deco = "(none)"
            for d in b.get("all_decorators", []):
                for tag in ("@api.depends_context", "@api.constrains", "@api.depends",
                             "@api.onchange", "@api.model_create_multi",
                             "@api.ondelete", "@api.model"):
                    if tag in d:
                        deco = tag
                        break
                if deco != "(none)": break
            methods.append({
                "family": family,
                "fn": b["function_name"],
                "root": name_root(b["function_name"]),
                "decorator": deco,
                "atoms": atoms,
            })

print(f"Methods: {len(methods)}")
print(f"Unique atoms: {len(atom_freq)}")
print(f"Total atom emissions across all methods: {total_atom_emissions}")
print(f"Avg atoms per method: {total_atom_emissions / max(1, len(methods)):.2f}")
print(f"Median atoms per method: {sorted(len(m['atoms']) for m in methods)[len(methods)//2]}")

# Atom-coverage threshold analysis: how many atoms cover N% of all emissions
sorted_atoms = atom_freq.most_common()
cum = 0
n50 = n80 = n95 = n99 = None
for i, (atom, c) in enumerate(sorted_atoms, 1):
    cum += c
    pct = cum / total_atom_emissions
    if n50 is None and pct >= 0.5: n50 = i
    if n80 is None and pct >= 0.8: n80 = i
    if n95 is None and pct >= 0.95: n95 = i
    if n99 is None and pct >= 0.99: n99 = i
print(f"Atom-coverage tiers: 50%={n50}, 80%={n80}, 95%={n95}, 99%={n99}")

# How tightly do "singletons" (methods unique at SoC-tuple level) overlap at atom level?
# Build atom multiset signature per method, count uniqueness at atom-set level.
method_atomsig = [tuple(m["atoms"]) for m in methods]
atomsig_freq = Counter(method_atomsig)
n_unique_at_atomset = sum(1 for s, c in atomsig_freq.items() if c == 1)
print(f"Unique-at-atomset methods: {n_unique_at_atomset}/{len(methods)}")
print(f"Methods with >=2 same-atomset partners: {sum(1 for s, c in atomsig_freq.items() if c >= 2)} atomsets, "
      f"covering {sum(c for s, c in atomsig_freq.items() if c >= 2)} methods")

# Now: pairwise atom overlap among singletons.
# For each method that's unique at SoC-tuple level, count how many OTHER methods
# share at least K atoms with it. Use K=3 as a soft synergy threshold.
# (Sampling 200 random singletons to keep this O(200*3555))
import random
random.seed(42)
sig_to_methods = defaultdict(list)
for m in methods:
    sig_to_methods[tuple(m["atoms"])].append(m)
singleton_methods = [m for m in methods if len(sig_to_methods[tuple(m["atoms"])]) == 1]
sample = random.sample(singleton_methods, min(200, len(singleton_methods)))

K = 3
overlap_counts = []
for m in sample:
    s = set(m["atoms"])
    if len(s) < K: continue
    n_overlap = 0
    for other in methods:
        if other is m: continue
        if len(s & set(other["atoms"])) >= K:
            n_overlap += 1
    overlap_counts.append(n_overlap)

overlap_counts.sort()
print(f"\nPairwise atom overlap (K>={K}) on {len(overlap_counts)} sampled singletons:")
if overlap_counts:
    print(f"  median partners: {overlap_counts[len(overlap_counts)//2]}")
    print(f"  p25: {overlap_counts[len(overlap_counts)//4]}")
    print(f"  p75: {overlap_counts[3*len(overlap_counts)//4]}")
    print(f"  max: {overlap_counts[-1]}")
    print(f"  zero-partner singletons: {sum(1 for c in overlap_counts if c == 0)}")

# Emit the atom catalogue.
out = Path("/tmp/work/atom-catalogue.md")
with out.open("w") as f:
    f.write(f"# Odoo SoC atom catalogue (singleton deconstruction)\n\n")
    f.write(f"**Headline:** {len(methods)} methods carry {total_atom_emissions} atom "
            f"emissions (avg {total_atom_emissions/len(methods):.1f} per method). "
            f"The **{len(atom_freq)} unique atoms** form the reusable pattern "
            f"catalogue — every method, including the 2655 SoC-tuple singletons, "
            f"decomposes into draws from this catalogue.\n\n")
    f.write(f"**Coverage tiers** (fraction of all atom emissions covered by top-N atoms):\n\n")
    f.write(f"- top {n50} atoms cover 50%\n")
    f.write(f"- top {n80} atoms cover 80%\n")
    f.write(f"- top {n95} atoms cover 95%\n")
    f.write(f"- top {n99} atoms cover 99%\n\n")
    f.write(f"**Singleton collapse at atom level (sample of 200 singletons, K>=3 shared atoms = synergy):**\n\n")
    if overlap_counts:
        f.write(f"- median partners per singleton: {overlap_counts[len(overlap_counts)//2]}\n")
        f.write(f"- p25: {overlap_counts[len(overlap_counts)//4]}, p75: {overlap_counts[3*len(overlap_counts)//4]}, max: {overlap_counts[-1]}\n")
        f.write(f"- zero-partner singletons: {sum(1 for c in overlap_counts if c == 0)}/{len(overlap_counts)}\n\n")
    f.write(f"## Top {n95} atoms (95% coverage)\n\n")
    f.write("| rank | atom | freq | %total |\n")
    f.write("| ---: | --- | ---: | ---: |\n")
    for i, (a, c) in enumerate(sorted_atoms[:n95], 1):
        f.write(f"| {i} | `{a}` | {c} | {100*c/total_atom_emissions:.2f}% |\n")
print(f"\nWrote {out}")
