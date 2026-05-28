#!/usr/bin/env python3
"""Second-pass AST analyzer: extract DelegationTuple edges from bundle methods.

For each NDJSON bundle, locate the corresponding `FunctionDef` in the
*original* Python source (via `file` + `line_start` + `function_name` from
the bundle) and walk its body to emit `ogit:invokes` / `reads_field` /
`writes_field` / `traverses_relation` / `reads_env` / `raises` edges.

Why this beats wrapping `body_source`: the bundle's `body_source` strips
line 1's leading whitespace, so re-wrapping into a synthetic `def __m__:`
fails on every method whose first statement is control-flow
(`for record in self:` etc.). Reading the original file and parsing the
whole module avoids all that — Python's `ast` library handles indentation
natively.

Per `E-BUSINESS-LOGIC-IS-GRAMMAR-1` DelegationTuple semantics. POC second
pass; the proper home is `ruff-py-dto`'s emit-path layer (D-RPYDTO-2a
step 4) where the visitor runs on the Rust-side AST.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

OGIT = "https://ogit.adaworldapi.com"
PREFIXES = f"""@prefix ogit: <{OGIT}/> .
@prefix odoo: <{OGIT}/odoo/> .
@prefix verb: <{OGIT}/grammar/verb/> .
"""


class DelegationVisitor(ast.NodeVisitor):
    """Walk a method body, collecting `self.X` and `self.env` usage."""

    def __init__(self):
        self.invokes: set[str] = set()
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.traverses: set[str] = set()
        self.reads_env: bool = False
        self.raises: set[str] = set()

    def _self_attr(self, node: ast.AST) -> str | None:
        """If node is `self.<name>`, return `<name>`; else None."""
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr
        return None

    def visit_Call(self, node: ast.Call):
        attr = self._self_attr(node.func)
        if attr:
            self.invokes.add(attr)
        if isinstance(node.func, ast.Attribute):
            inner = node.func.value
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
                and inner.attr == "env"
            ):
                self.reads_env = True
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        attr = self._self_attr(node.value)
        if attr == "env":
            self.reads_env = True
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        attr = self._self_attr(node)
        if attr and attr != "env":
            self.reads.add(attr)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            attr = self._self_attr(target)
            if attr:
                self.writes.add(attr)
                self.reads.discard(attr)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        attr = self._self_attr(node.target)
        if attr:
            self.writes.add(attr)
            self.reads.add(attr)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        attr = self._self_attr(node.iter)
        if attr:
            self.traverses.add(attr)
            self.reads.discard(attr)
        if isinstance(node.iter, ast.Call):
            attr2 = self._self_attr(node.iter.func)
            if attr2:
                self.invokes.add(attr2)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise):
        exc = node.exc
        if isinstance(exc, ast.Call):
            if isinstance(exc.func, ast.Name):
                self.raises.add(exc.func.id)
            elif isinstance(exc.func, ast.Attribute):
                self.raises.add(exc.func.attr)
        elif isinstance(exc, ast.Name):
            self.raises.add(exc.id)
        self.generic_visit(node)


def parse_file_cached(source_root: Path, rel_path: str, cache: dict) -> ast.Module | None:
    if rel_path in cache:
        return cache[rel_path]
    full = source_root / rel_path
    try:
        src = full.read_text(encoding="utf-8", errors="replace")
        mod = ast.parse(src, filename=str(full))
        cache[rel_path] = mod
        return mod
    except Exception:
        cache[rel_path] = None
        return None


def find_function_def(
    module: ast.Module, function_name: str, line_start: int
):
    """Find the FunctionDef whose decorator-or-def range contains `line_start`."""
    candidates = []
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != function_name:
                continue
            range_start = node.lineno
            if node.decorator_list:
                range_start = min(d.lineno for d in node.decorator_list)
            range_end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if range_start <= line_start <= range_end:
                candidates.append(node)
    if not candidates:
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name and abs(node.lineno - line_start) <= 5:
                    candidates.append(node)
    if not candidates:
        return None
    return min(candidates, key=lambda n: abs(n.lineno - line_start))


def analyze_method_from_file(
    source_root: Path, bundle: dict, cache: dict
):
    """Returns (delegation_dict, status).
    Status in: ok | file_parse_err | no_match | empty_body
    """
    rel = bundle.get("file", "")
    if not rel:
        return {}, "no_match"
    mod = parse_file_cached(source_root, rel, cache)
    if mod is None:
        return {}, "file_parse_err"
    fn = find_function_def(mod, bundle["function_name"], bundle.get("line_start", 0))
    if fn is None:
        return {}, "no_match"
    if not fn.body:
        return {}, "empty_body"
    visitor = DelegationVisitor()
    for stmt in fn.body:
        visitor.visit(stmt)
    out = {}
    if visitor.invokes:
        out["invokes"] = sorted(visitor.invokes)
    if visitor.reads:
        out["reads"] = sorted(visitor.reads)
    if visitor.writes:
        out["writes"] = sorted(visitor.writes)
    if visitor.traverses:
        out["traverses"] = sorted(visitor.traverses)
    if visitor.reads_env:
        out["reads_env"] = True
    if visitor.raises:
        out["raises"] = sorted(visitor.raises)
    return out, "ok"


def emit_edges_ttl(family: str, bundle: dict, deleg: dict) -> str:
    fn = bundle["function_name"]
    subject = f"odoo:{family}.{fn}"
    lines = []
    if not deleg:
        return ""
    if "invokes" in deleg:
        for m in deleg["invokes"]:
            lines.append(f"{subject} ogit:invokes odoo:{family}.{m} .")
    if "reads" in deleg:
        for f in deleg["reads"]:
            lines.append(f'{subject} ogit:reads_field "{f}" .')
    if "writes" in deleg:
        for f in deleg["writes"]:
            lines.append(f'{subject} ogit:writes_field "{f}" .')
    if "traverses" in deleg:
        for r in deleg["traverses"]:
            lines.append(f'{subject} ogit:traverses_relation "{r}" .')
    if deleg.get("reads_env"):
        lines.append(f"{subject} ogit:reads_env true .")
    if "raises" in deleg:
        for e in deleg["raises"]:
            lines.append(f'{subject} ogit:raises "{e}" .')
    return "\n".join(lines) + ("\n" if lines else "")


def main():
    if len(sys.argv) < 4:
        print(
            "usage: extract_delegation.py <bundles_dir> <source_root> <out_dir>",
            file=sys.stderr,
        )
        print(
            "  source_root: same path passed as --root to ruff-py-dto harvest "
            "(e.g. /home/user/odoo/addons)",
            file=sys.stderr,
        )
        sys.exit(1)
    bundles_dir = Path(sys.argv[1])
    source_root = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "delegation.ttl"
    cache = {}
    stats = {
        "methods_analyzed": 0,
        "methods_with_invokes": 0,
        "methods_with_reads": 0,
        "methods_with_writes": 0,
        "methods_with_traverses": 0,
        "methods_with_reads_env": 0,
        "methods_with_raises": 0,
        "total_invoke_edges": 0,
        "total_read_edges": 0,
        "total_write_edges": 0,
        "total_traverse_edges": 0,
        "total_raise_edges": 0,
        "file_parse_err": 0,
        "no_match": 0,
        "empty_body": 0,
        "ok": 0,
    }
    with out_path.open("w") as out:
        out.write("# DelegationTuple edges from ruff-py-dto bundles\n")
        out.write("# Per E-BUSINESS-LOGIC-IS-GRAMMAR-1 / DelegationTuple slot.\n")
        out.write("# POC: reads original .py files via bundle file+line metadata.\n\n")
        out.write(PREFIXES)
        out.write("\n")
        for nd in sorted(bundles_dir.glob("*.ndjson")):
            family = nd.stem
            with nd.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    bundle = json.loads(line)
                    deleg, status = analyze_method_from_file(source_root, bundle, cache)
                    stats["methods_analyzed"] += 1
                    stats[status] += 1
                    if status != "ok":
                        continue
                    if "invokes" in deleg:
                        stats["methods_with_invokes"] += 1
                        stats["total_invoke_edges"] += len(deleg["invokes"])
                    if "reads" in deleg:
                        stats["methods_with_reads"] += 1
                        stats["total_read_edges"] += len(deleg["reads"])
                    if "writes" in deleg:
                        stats["methods_with_writes"] += 1
                        stats["total_write_edges"] += len(deleg["writes"])
                    if "traverses" in deleg:
                        stats["methods_with_traverses"] += 1
                        stats["total_traverse_edges"] += len(deleg["traverses"])
                    if deleg.get("reads_env"):
                        stats["methods_with_reads_env"] += 1
                    if "raises" in deleg:
                        stats["methods_with_raises"] += 1
                        stats["total_raise_edges"] += len(deleg["raises"])
                    out.write(emit_edges_ttl(family, bundle, deleg))
    stats_path = out_dir / "delegation-stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Wrote {out_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
