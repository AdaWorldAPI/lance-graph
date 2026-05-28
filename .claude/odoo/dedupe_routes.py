"""Use codegen round-trip to find duplicate / near-duplicate routes.

For each extracted method body:
  1. parse → AST
  2. ast.unparse → canonical normalized source (whitespace / quote style / etc.
     erased; ONLY the AST shape remains)
  3. hash the canonical form

Methods sharing a hash = EXACT duplicate routes (factor-out candidates).
Near-duplicates would need AST edit-distance (next pass) — this run is the
exact-duplicate baseline; that alone tells us how much factoring is possible.

This is the dto-check / codegen composition: harvest produced the bundles,
codegen normalizes them, duplicate detection runs over the normalized form.
"""
import ast
import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path

BUNDLES_DIR = Path("/tmp/odoo-extract/harvest-full/bundles")

PREFIX_RE = re.compile(
    r"^_+(compute|check|inverse|onchange|search|get|set|find|resolve|guess|"
    r"suggest|infer|match|detect|recommend|propose|validate|score|rank|"
    r"create|update|delete|prepare|build|generate|fetch|load|read|write)_"
)


def normalize_body(body_source: str, function_name: str) -> str | None:
    """Parse body, ast.unparse to canonical form. Return None on parse fail."""
    if not body_source.strip():
        return None
    # Re-indent: bundles drop line-1 leading whitespace, re-add 8 columns
    # then dedent uniformly. Same trick as extract_delegation.py.
    import textwrap
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
        tree = ast.parse(wrapped)
    except SyntaxError:
        return None
    # The function body is what we normalize; throw away signature/name.
    fn = tree.body[0]
    if not fn.body:
        return None
    body_ast = ast.Module(body=fn.body, type_ignores=[])
    try:
        return ast.unparse(body_ast)
    except Exception:
        return None


buckets = defaultdict(list)  # hash → list of (family, fn, file, line_start)
total = 0
parsed = 0
for nd in sorted(BUNDLES_DIR.glob("*.ndjson")):
    family = nd.stem
    with nd.open() as f:
        for line in f:
            if not line.strip(): continue
            b = json.loads(line)
            total += 1
            canonical = normalize_body(b.get("body_source", ""), b["function_name"])
            if canonical is None: continue
            parsed += 1
            h = hashlib.sha256(canonical.encode()).hexdigest()[:16]
            buckets[h].append({
                "family": family,
                "fn": b["function_name"],
                "file": b["file"],
                "line_start": b.get("line_start", 0),
                "canonical": canonical,
                "body_lines": b.get("body_lines", 0),
            })

print(f"Total bundles: {total}")
print(f"Parsed (canonicalised): {parsed}")
print(f"Unique canonical-body hashes: {len(buckets)}")

dupe_clusters = {h: members for h, members in buckets.items() if len(members) >= 2}
print(f"Duplicate clusters (n >= 2): {len(dupe_clusters)}")
print(f"Methods inside a duplicate cluster: {sum(len(m) for m in dupe_clusters.values())}")
print(f"Singletons (unique body): {sum(1 for m in buckets.values() if len(m) == 1)}")

# Sort clusters by size desc.
sorted_clusters = sorted(dupe_clusters.items(), key=lambda kv: -len(kv[1]))

# Histogram of cluster sizes.
size_hist = defaultdict(int)
for members in dupe_clusters.values():
    size_hist[len(members)] += 1
print()
print("Cluster size distribution:")
for size in sorted(size_hist.keys()):
    print(f"  size={size:>3}  clusters={size_hist[size]:>4}  methods={size_hist[size]*size:>5}")

# Emit the top clusters as a markdown table.
out = Path("/tmp/work/duplicate-routes.md")
with out.open("w") as f:
    f.write("# Duplicate-route detection via codegen round-trip\n\n")
    f.write(f"From **{total}** ruff-py-dto bundles, **{parsed}** canonicalised via "
            f"`ast.unparse` (the `ruff_python_codegen::Generator.unparse_suite` "
            f"equivalent in Python).\n\n")
    f.write(f"- **{len(buckets)}** unique canonical bodies\n")
    f.write(f"- **{len(dupe_clusters)}** duplicate clusters (n ≥ 2)\n")
    f.write(f"- **{sum(len(m) for m in dupe_clusters.values())}** methods inside a duplicate cluster\n")
    f.write(f"- **{sum(1 for m in buckets.values() if len(m) == 1)}** singletons\n\n")
    f.write(f"Decomposition ceiling: if every duplicate cluster collapses to "
            f"one shared implementation, the implementation count drops from "
            f"{parsed} to {len(buckets)} — a "
            f"**{100*(parsed-len(buckets))/parsed:.1f}%** reduction.\n\n")
    f.write("## Cluster size distribution\n\n")
    f.write("| cluster size | num clusters | methods covered |\n")
    f.write("| ---: | ---: | ---: |\n")
    for size in sorted(size_hist.keys()):
        f.write(f"| {size} | {size_hist[size]} | {size_hist[size]*size} |\n")
    f.write("\n## Top 30 duplicate clusters (by member count)\n\n")
    f.write("| n | function names (deduped) | families | canonical body (first 200 chars) |\n")
    f.write("| ---: | --- | --- | --- |\n")
    for i, (h, members) in enumerate(sorted_clusters[:30], 1):
        unique_fns = sorted({m["fn"] for m in members})
        families = sorted({m["family"] for m in members})
        canon = members[0]["canonical"][:200].replace("\n", " ⏎ ").replace("|", "¦")
        fn_str = ", ".join(unique_fns[:5]) + (f", … (+{len(unique_fns)-5})" if len(unique_fns) > 5 else "")
        fam_str = ", ".join(families[:5]) + (f", … (+{len(families)-5})" if len(families) > 5 else "")
        f.write(f"| {len(members)} | `{fn_str}` | {fam_str} | `{canon}` |\n")
    # Emit 5 worked examples in full for the top clusters
    f.write("\n## Worked examples (top 5 clusters, full canonical body)\n\n")
    for i, (h, members) in enumerate(sorted_clusters[:5], 1):
        f.write(f"### Cluster #{i} — {len(members)} methods\n\n")
        f.write(f"**Function-name variants:** {sorted({m['fn'] for m in members})}\n\n")
        f.write(f"**Family-of-occurrence:** {sorted({m['family'] for m in members})[:10]}\n\n")
        f.write("**Canonical body (post-`ast.unparse`):**\n\n```python\n")
        f.write(members[0]["canonical"])
        f.write("\n```\n\n")
        f.write(f"**One representative site:** `{members[0]['file']}:L{members[0]['line_start']}`\n\n")

print(f"\nWrote {out}")
