"""SoA-shaped DTO extraction: 3555 methods -> one columnar Arrow/parquet file.

Replaces the AoS NDJSON output with a single columnar layout where each
column is one DTO field across all methods. Reading column N = one
sequential scan; cross-column queries = parallel sweeps; cache- and
SIMD-friendly by construction.

Schema = the 24-column union of:
  - raw extraction (function_name, family, file, line range, decorators, body)
  - delegation (reads/writes/invokes/raises/env/traverses)
  - grammar coding (T, tek, men, reg)
  - cost model placeholder (cost_estimate_ns -- filled in by future
    micro-benchmarks; column exists so downstream codegen can fill it)

Output: ONE parquet file at `/tmp/work/methods.parquet`. Codegen reads
columns selectively (Arrow zero-copy) instead of re-parsing 3555 JSON
lines. The codegen output (codegen-output-soa-layout.rs) is the
1-to-1 row image of this table.
"""
import ast
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BUNDLES_DIR = Path("/tmp/odoo-extract/harvest-full/bundles")
SOURCE_ROOT = Path("/home/user/odoo/addons")
OUT_PARQUET = Path("/tmp/work/methods.parquet")
OUT_SCHEMA_RS = Path("/tmp/work/methods-schema.rs")

PREFIX_RE = re.compile(
    r"^_+(compute|check|inverse|onchange|search|get|set|find|resolve|guess|"
    r"suggest|infer|match|detect|recommend|propose|validate|score|rank|"
    r"create|update|delete|prepare|build|generate|fetch|load|read|write)_"
)

TEKAMOLO_MARKERS = {
    "TE": ["date", "period", "lock_date", "fiscalyear", "due_date", "maturity"],
    "KA": ["raise ", "ValidationError", "UserError", "AccessError"],
    "MO": ["if ", "elif "],
    "LO": ["country", "jurisdiction", "fiscal_position", "tax_country"],
    "QU": ["amount", "balance", "debit", "credit", "rate", "percentage",
           "discount", "quantity", "subtotal", "total"],
}
MENGENMASS_MARKERS = {
    "money":   ["amount", "balance", "debit", "credit", "subtotal", "total", "price", "fee"],
    "percent": ["percentage", "percent", "discount_perc"],
    "rate":    ["rate", "tax_rate", "currency_rate", "exchange_rate"],
    "count":   ["quantity", "qty", "count", "number"],
    "date":    ["date", "_at", "period", "fiscalyear", "due"],
}
REG_MARKERS = {
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
DECO_TAGS = [
    "@api.depends_context", "@api.constrains", "@api.depends",
    "@api.onchange", "@api.model_create_multi", "@api.ondelete", "@api.model",
]


class V(ast.NodeVisitor):
    def __init__(self):
        self.invokes, self.reads, self.writes = set(), set(), set()
        self.raises, self.reads_env, self.traverses = set(), False, set()

    def _sa(self, n):
        if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id == "self"):
            return n.attr
        return None

    def visit_Call(self, n):
        a = self._sa(n.func)
        if a: self.invokes.add(a)
        if isinstance(n.func, ast.Attribute):
            inner = n.func.value
            if (isinstance(inner, ast.Attribute)
                    and isinstance(inner.value, ast.Name)
                    and inner.value.id == "self" and inner.attr == "env"):
                self.reads_env = True
        self.generic_visit(n)

    def visit_Subscript(self, n):
        if self._sa(n.value) == "env": self.reads_env = True
        self.generic_visit(n)

    def visit_Attribute(self, n):
        a = self._sa(n)
        if a and a != "env": self.reads.add(a)
        self.generic_visit(n)

    def visit_Assign(self, n):
        for t in n.targets:
            a = self._sa(t)
            if a: self.writes.add(a); self.reads.discard(a)
        self.generic_visit(n)

    def visit_For(self, n):
        a = self._sa(n.iter)
        if a: self.traverses.add(a); self.reads.discard(a)
        self.generic_visit(n)

    def visit_Raise(self, n):
        e = n.exc
        if isinstance(e, ast.Call):
            if isinstance(e.func, ast.Name): self.raises.add(e.func.id)
            elif isinstance(e.func, ast.Attribute): self.raises.add(e.func.attr)
        elif isinstance(e, ast.Name): self.raises.add(e.id)
        self.generic_visit(n)


cache = {}
def parse_cached(rel):
    if rel in cache: return cache[rel]
    try: cache[rel] = ast.parse((SOURCE_ROOT / rel).read_text(encoding="utf-8", errors="replace"))
    except Exception: cache[rel] = None
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
    return min(cands, key=lambda x: abs(x.lineno - ls)) if cands else None


def name_root(n):
    m = PREFIX_RE.match(n)
    return n[m.end():] if m else n.lstrip("_")


def primary_decorator(decos):
    for d in decos:
        for tag in DECO_TAGS:
            if tag in d: return tag
    return decos[0].split("(")[0] if decos else "(none)"


def code_grammar(body, fn_name):
    has_raise = "raise " in body
    has_return = re.search(r"\breturn\b", body) is not None
    if fn_name.startswith(("_check_", "_validate_")) or (has_raise and not has_return):
        T = "I"
    else:
        T = "T"
    tek_scores = {k: sum(body.count(m) for m in v) for k, v in TEKAMOLO_MARKERS.items()}
    tek = max(tek_scores, key=tek_scores.get) if tek_scores else None
    if tek_scores.get(tek, 0) == 0: tek = None
    men_scores = {k: sum(body.count(m) for m in v) for k, v in MENGENMASS_MARKERS.items()}
    men = max(men_scores, key=men_scores.get) if men_scores else "none"
    if men_scores.get(men, 0) == 0: men = "none"
    regs = [r for r, ms in REG_MARKERS.items() if any(m in (body + " " + fn_name) for m in ms)]
    return T, tek, men, regs


# ── SoA columns ──────────────────────────────────────────────────────
cols = {
    "function_name":     [],   # utf8
    "name_root":         [],   # utf8
    "family":            [],   # utf8 (dict-encoded effectively via parquet)
    "file":              [],   # utf8
    "line_start":        [],   # u32
    "line_end":          [],   # u32
    "body_lines":        [],   # u32
    "signature":         [],   # utf8
    "match_id":          [],   # utf8
    "primary_decorator": [],   # utf8 (enum-shaped, 9 values)
    "decorators_all":    [],   # list<utf8>
    "transitivity":      [],   # utf8: T | I
    "tekamolo":          [],   # utf8: TE | KA | MO | LO | QU | (none)
    "mengenmass":        [],   # utf8: money | percent | rate | count | date | none
    "regulatory_anchor": [],   # list<utf8>: UStG/HGB/EStG/AO/GoBD/SKR04/SKR03/DATEV/ELSTER/Peppol
    "axis":              [],   # utf8: Deterministic | Heuristic | Hybrid (heuristic via decorators)
    "reads_fields":      [],   # list<utf8>
    "writes_fields":     [],   # list<utf8>
    "invokes":           [],   # list<utf8>
    "raises":            [],   # list<utf8>
    "traverses":         [],   # list<utf8>
    "reads_env":         [],   # bool
    "atom_count":        [],   # u32 — sum of |reads| + |writes| + |invokes| + |raises| + |env?| + |traverses|
    "cost_estimate_ns":  [],   # u32 — placeholder; filled by future micro-benchmarks
}

def axis_for(decos, fn_name):
    blob = " ".join(decos)
    has_constrains = "@api.constrains" in blob
    has_depends    = "@api.depends" in blob
    has_onchange   = "@api.onchange" in blob
    HEUR = re.compile(r"^_?(?:get_fiscal_position|find|match|resolve|guess|suggest|classify|propose|score|rank|detect|infer|recommend)\b")
    name_h = bool(HEUR.match(fn_name))
    if has_onchange and (has_depends or has_constrains): return "Hybrid"
    if has_onchange or name_h: return "Heuristic"
    return "Deterministic"

for nd in sorted(BUNDLES_DIR.glob("*.ndjson")):
    family = nd.stem
    with nd.open() as f:
        for line in f:
            if not line.strip(): continue
            b = json.loads(line)
            mod = parse_cached(b["file"])
            fn_node = find_fn(mod, b["function_name"], b.get("line_start", 0)) if mod else None
            v = V()
            if fn_node and fn_node.body:
                for s in fn_node.body: v.visit(s)
            body = b.get("body_source", "")
            T, tek, men, regs = code_grammar(body, b["function_name"])
            cols["function_name"].append(b["function_name"])
            cols["name_root"].append(name_root(b["function_name"]))
            cols["family"].append(family)
            cols["file"].append(b["file"])
            cols["line_start"].append(b.get("line_start", 0))
            cols["line_end"].append(b.get("line_end", 0))
            cols["body_lines"].append(b.get("body_lines", 0))
            cols["signature"].append(b.get("signature", ""))
            cols["match_id"].append(b.get("match_id", ""))
            cols["primary_decorator"].append(primary_decorator(b.get("all_decorators", [])))
            cols["decorators_all"].append(b.get("all_decorators", []))
            cols["transitivity"].append(T)
            cols["tekamolo"].append(tek or "(none)")
            cols["mengenmass"].append(men)
            cols["regulatory_anchor"].append(regs)
            cols["axis"].append(axis_for(b.get("all_decorators", []), b["function_name"]))
            cols["reads_fields"].append(sorted(v.reads))
            cols["writes_fields"].append(sorted(v.writes))
            cols["invokes"].append(sorted(v.invokes))
            cols["raises"].append(sorted(v.raises))
            cols["traverses"].append(sorted(v.traverses))
            cols["reads_env"].append(v.reads_env)
            cols["atom_count"].append(
                len(v.reads) + len(v.writes) + len(v.invokes) + len(v.raises)
                + (1 if v.reads_env else 0) + len(v.traverses)
            )
            cols["cost_estimate_ns"].append(0)  # placeholder

n = len(cols["function_name"])
print(f"Rows: {n}")
print(f"Columns: {len(cols)}")
print(f"Schema:")
for k, v in cols.items():
    sample = v[0] if v else None
    if isinstance(sample, list):
        kind = f"list[{type(sample[0]).__name__ if sample else 'utf8'}]"
    else:
        kind = type(sample).__name__
    print(f"  {k:20s} {kind}")

# Convert to Arrow table — parquet writes are columnar by construction.
table = pa.table({
    "function_name":     pa.array(cols["function_name"], pa.string()),
    "name_root":         pa.array(cols["name_root"], pa.string()),
    "family":            pa.array(cols["family"], pa.string()),
    "file":              pa.array(cols["file"], pa.string()),
    "line_start":        pa.array(cols["line_start"], pa.uint32()),
    "line_end":          pa.array(cols["line_end"], pa.uint32()),
    "body_lines":        pa.array(cols["body_lines"], pa.uint32()),
    "signature":         pa.array(cols["signature"], pa.string()),
    "match_id":          pa.array(cols["match_id"], pa.string()),
    "primary_decorator": pa.array(cols["primary_decorator"], pa.string()),
    "decorators_all":    pa.array(cols["decorators_all"], pa.list_(pa.string())),
    "transitivity":      pa.array(cols["transitivity"], pa.string()),
    "tekamolo":          pa.array(cols["tekamolo"], pa.string()),
    "mengenmass":        pa.array(cols["mengenmass"], pa.string()),
    "regulatory_anchor": pa.array(cols["regulatory_anchor"], pa.list_(pa.string())),
    "axis":              pa.array(cols["axis"], pa.string()),
    "reads_fields":      pa.array(cols["reads_fields"], pa.list_(pa.string())),
    "writes_fields":     pa.array(cols["writes_fields"], pa.list_(pa.string())),
    "invokes":           pa.array(cols["invokes"], pa.list_(pa.string())),
    "raises":            pa.array(cols["raises"], pa.list_(pa.string())),
    "traverses":         pa.array(cols["traverses"], pa.list_(pa.string())),
    "reads_env":         pa.array(cols["reads_env"], pa.bool_()),
    "atom_count":        pa.array(cols["atom_count"], pa.uint32()),
    "cost_estimate_ns":  pa.array(cols["cost_estimate_ns"], pa.uint32()),
})
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(table, OUT_PARQUET, compression="zstd")
sz = OUT_PARQUET.stat().st_size
print(f"\nWrote {OUT_PARQUET} ({sz/1024:.1f} KB) — {n} rows × 24 columns, zstd-compressed")

# Emit the Rust schema mirror (for codegen direct consumption via arrow-rs).
OUT_SCHEMA_RS.write_text(f'''//! SoA-shaped DTO extraction schema — mirrors /tmp/work/methods.parquet.
//!
//! Auto-emitted by extract_soa.py. The columns below are the 1-to-1
//! Arrow schema; the codegen reads them via arrow-rs RecordBatch with
//! zero-copy column access.
//!
//! Each column is a contiguous Arrow buffer; cross-column sweeps run
//! through arrow::compute kernels (SIMD-accelerated). Reading
//! "all `family` values" = one ChunkedArray scan; filtering "where
//! transitivity = 'I' AND tekamolo = 'KA'" = two parallel column scans
//! + one boolean AND. The shape downstream IS the shape upstream.

use arrow::datatypes::{{DataType, Field, Schema}};
use std::sync::Arc;

pub fn methods_schema() -> Schema {{
    Schema::new(vec![
        Field::new("function_name",     DataType::Utf8,                              false),
        Field::new("name_root",         DataType::Utf8,                              false),
        Field::new("family",            DataType::Utf8,                              false),  // dict-encoded in parquet
        Field::new("file",              DataType::Utf8,                              false),
        Field::new("line_start",        DataType::UInt32,                            false),
        Field::new("line_end",          DataType::UInt32,                            false),
        Field::new("body_lines",        DataType::UInt32,                            false),
        Field::new("signature",         DataType::Utf8,                              false),
        Field::new("match_id",          DataType::Utf8,                              false),
        Field::new("primary_decorator", DataType::Utf8,                              false),
        Field::new("decorators_all",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("transitivity",      DataType::Utf8,                              false),  // T | I
        Field::new("tekamolo",          DataType::Utf8,                              false),  // TE|KA|MO|LO|QU|(none)
        Field::new("mengenmass",        DataType::Utf8,                              false),  // money|percent|rate|count|date|none
        Field::new("regulatory_anchor",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("axis",              DataType::Utf8,                              false),  // Deterministic|Heuristic|Hybrid
        Field::new("reads_fields",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("writes_fields",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("invokes",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("raises",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("traverses",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),     false),
        Field::new("reads_env",         DataType::Boolean,                           false),
        Field::new("atom_count",        DataType::UInt32,                            false),
        Field::new("cost_estimate_ns",  DataType::UInt32,                            false),
    ])
}}
''')
print(f"Wrote {OUT_SCHEMA_RS}")
