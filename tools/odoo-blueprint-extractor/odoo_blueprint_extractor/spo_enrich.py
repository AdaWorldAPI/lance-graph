"""SPO corpus enrichment — FK target/inverse_name (P1) + deep reads_field (P0).

Stdlib-only Python 3. Reads the Odoo ORM source (the same addons tree the
ORM extractor parses) to build a relation-target map, then enriches an
existing SPO ndjson corpus with two additive predicate families:

  * **P1 — `target` / `inverse_name`** (UPSTREAM_WISHLIST P1, ruff#18 shape):
    for every relational field (Many2one / One2many / Many2many / Reference)
    that is a `ogit:Property` node in the corpus and whose comodel resolves
    from source, emit a sibling triple keyed by the field IRI::

        (odoo:account_move.line_ids, target,       "account.move.line")
        (odoo:account_move.line_ids, inverse_name, "move_id")

    The object is the *raw* Odoo dotted model name / inverse string, matching
    the ratified cross-language shape in
    `AdaWorldAPI/ruff#18` — `(WorkPackage.owner, class_name, "User")`.

  * **P0 — deep `reads_field`** (UPSTREAM_WISHLIST P0, narrowed ask): for each
    existing `depends_on` triple whose object is a dotted relation-traversal
    path (`odoo:<model>.<rel>.<...>.<leaf>`, ≥ 2 path segments), resolve each
    relational hop via the target map and emit a deep read on the field that
    the dependent field is *emitted by*::

        @api.depends('line_ids.amount_residual') on amount_total
          (emitted by _compute_amount)
        ⇒ (odoo:account_move._compute_amount, reads_field,
             odoo:account_move_line.amount_residual)

    This is *in addition* to the existing shallow relation read
    `(_compute_amount, reads_field, odoo:account_move.line_ids)`. The deep
    read makes the cross-model recompute-ordering edge visible to
    `od_ontology::RecomputeDag` (which today sees only the relation read,
    leaving the line→move dependency structurally invisible).

# Why a separate enrichment pass (not the ORM extractor)

The ORM extractor (`parsers/`, `emitters/`) emits typed Rust `OdooEntity`
consts into `lance-graph-ontology`. The SPO ndjson corpus is a *separate*
artifact (the body-write / `@api.depends` graph). The corpus's original
generator (`emit_ontology2.py` over a `methods.parquet`, per the
`odoo_ontology.rs` module doc) is **not present in the tree** — only its
output is. This pass is the additive, idempotent enrichment step that runs
over the shipped corpus + the Odoo source, both of which ARE present.

Determinism: the same source + corpus always produce the same triples
(sorted, de-duplicated against existing keys). Re-running over an
already-enriched corpus is a no-op for the new predicates (they are
de-duplicated by `(s, p, o)`).

# Self-loop / dedup semantics

  * A deep read equal to its own emitter `(method, reads_field, <method>)`
    is never emitted (matches `RecomputeDag`'s self-loop drop).
  * `(s, p, o)` triples already present in the corpus are not re-emitted.
"""

import argparse
import ast
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# Relational ORM field kinds that carry a comodel + (sometimes) an inverse name.
RELATIONAL_KINDS = {"Many2one", "One2many", "Many2many", "Reference"}

# NARS truth values mirror the corpus conventions in odoo_ontology.rs:
#   authoritative (decorator / declared comodel) edges → (0.95, 0.90)
#   body-inferred edges                                → (0.85, 0.75)
TARGET_TRUTH = (0.95, 0.90)
INVERSE_TRUTH = (0.95, 0.90)
DEEP_READ_TRUTH = (0.85, 0.75)


def model_to_underscore(dotted: str) -> str:
    """`account.move.line` → `account_move_line` (corpus IRI convention)."""
    return dotted.replace(".", "_")


# ---------------------------------------------------------------------------
# Relation map — (model_underscore, field) → (comodel_dotted, inverse_or_None)
# ---------------------------------------------------------------------------


def _const_str(node: Optional[ast.expr]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_str_list(node: Optional[ast.expr]) -> List[str]:
    """`'x'` → `['x']`; `['a', 'b']` / `('a', 'b')` → `['a', 'b']`; else `[]`.

    Mirrors the package class parser's `_extract_list_or_string`, extended to
    accept tuples as well as lists (both are common `_inherit` forms).
    """
    s = _const_str(node)
    if s is not None:
        return [s]
    if isinstance(node, (ast.List, ast.Tuple)):
        out: List[str] = []
        for elt in node.elts:
            v = _const_str(elt)
            if v is not None:
                out.append(v)
        return out
    return []


def _scan_file(path: str, relmap: Dict[Tuple[str, str], Tuple[str, Optional[str]]]) -> None:
    """Parse one .py file; record every relational field on every named model.

    A model is keyed by its `_name` when present, ELSE by its `_inherit`
    target(s) — the common Odoo extension form `_inherit = "some.model"`
    (or `_inherit = ["a", "b"]`) with no `_name`. Relational fields capture
    comodel (kw `comodel_name` or positional arg 0) and inverse name
    (One2many's positional arg 1 / kw `inverse_name`; Many2one has no inverse
    here), and are mapped onto EVERY resolved model name.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError, ValueError):
        # Vendored Odoo occasionally carries py2 remnants / encoding quirks;
        # a file we cannot parse contributes nothing and is skipped.
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        name_model: Optional[str] = None
        inherit_models: List[str] = []
        local_fields: Dict[str, Tuple[str, Optional[str]]] = {}

        for stmt in node.body:
            if not (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                continue
            target_name = stmt.targets[0].id

            # _name = 'account.move.line'
            if target_name == "_name":
                s = _const_str(stmt.value)
                if s is not None:
                    name_model = s
                continue

            # _inherit = 'sale.order'  OR  _inherit = ['a', 'b']
            # The extension form (no _name) reopens an existing model to add
            # relational fields; those fields belong to the inherited model(s).
            if target_name == "_inherit":
                inherit_models = _const_str_list(stmt.value)
                continue

            # field = fields.X(...)
            if not (
                isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and isinstance(stmt.value.func.value, ast.Name)
                and stmt.value.func.value.id == "fields"
            ):
                continue
            field_type = stmt.value.func.attr
            if field_type not in RELATIONAL_KINDS:
                continue

            call = stmt.value
            comodel: Optional[str] = None
            inverse: Optional[str] = None

            # comodel_name kw, else positional arg 0
            for kw in call.keywords:
                if kw.arg == "comodel_name":
                    comodel = _const_str(kw.value)
                elif kw.arg == "inverse_name":
                    inverse = _const_str(kw.value)
            if comodel is None and call.args:
                comodel = _const_str(call.args[0])
            # One2many inverse is positional arg 1 when not given as kw
            if (
                field_type == "One2many"
                and inverse is None
                and len(call.args) >= 2
            ):
                inverse = _const_str(call.args[1])

            if comodel is not None:
                local_fields[target_name] = (comodel, inverse)

        # Resolve the model name(s) this class contributes fields to: `_name`
        # if present, else the `_inherit` target(s). A class with neither
        # contributes nothing.
        if name_model is not None:
            model_names = [name_model]
        else:
            model_names = inherit_models

        for model_name in model_names:
            mu = model_to_underscore(model_name)
            for field_name, (comodel, inverse) in local_fields.items():
                # Last write wins across _inherit reopenings of the same model;
                # comodel for a given (model, field) is stable in practice.
                relmap[(mu, field_name)] = (comodel, inverse)


def build_relation_map(addons_root: str) -> Dict[Tuple[str, str], Tuple[str, Optional[str]]]:
    """Scan every .py under `addons_root`; return (model_us, field) → (comodel, inverse)."""
    relmap: Dict[Tuple[str, str], Tuple[str, Optional[str]]] = {}
    pattern = os.path.join(addons_root, "**", "*.py")
    for path in glob.iglob(pattern, recursive=True):
        _scan_file(path, relmap)
    return relmap


# ---------------------------------------------------------------------------
# Corpus IO
# ---------------------------------------------------------------------------


def load_corpus(path: str) -> List[dict]:
    triples: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                triples.append(json.loads(line))
    return triples


def triple_line(s: str, p: str, o: str, f: float, c: float) -> str:
    """One ndjson line matching the corpus byte shape (no spaces, key order s/p/o/f/c)."""
    return json.dumps({"s": s, "p": p, "o": o, "f": f, "c": c}, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def resolve_path(
    start_model_us: str,
    segments: List[str],
    relmap: Dict[Tuple[str, str], Tuple[str, Optional[str]]],
) -> Optional[Tuple[str, str]]:
    """Walk relational hops `seg[0..n-1]`; leaf = `seg[n-1]`.

    Returns `(final_model_underscore, leaf)` or `None` if any non-leaf hop
    is not a known relational field (target unknown → skip the lift).
    """
    cur = start_model_us
    for hop in segments[:-1]:
        entry = relmap.get((cur, hop))
        if entry is None:
            return None
        cur = model_to_underscore(entry[0])
    return (cur, segments[-1])


def enrich(
    triples: List[dict],
    relmap: Dict[Tuple[str, str], Tuple[str, Optional[str]]],
) -> Tuple[List[str], dict]:
    """Compute the additive enrichment lines + a stats dict.

    Returns `(new_lines_sorted, stats)`. `new_lines_sorted` are ndjson lines
    NOT already present in `triples` (de-duplicated by `(s, p, o)`).
    """
    existing_spo: Set[Tuple[str, str, str]] = {
        (t["s"], t["p"], t["o"]) for t in triples
    }

    # ObjectType (model) IRIs declared in the corpus — the additive boundary:
    # P1 target/inverse_name is only ever emitted for a field whose MODEL the
    # corpus already declares. Never invents a relation on an unknown model.
    object_type_models: Set[str] = {
        t["s"][len("odoo:") :]
        for t in triples
        if t["p"] == "rdf:type"
        and t["o"] == "ogit:ObjectType"
        and t["s"].startswith("odoo:")
    }

    # Candidate relational-field IRIs to consider for target/inverse_name.
    # Two sources, both scoped to corpus-declared models:
    #   (a) declared Property nodes (e.g. `invoice_line_ids`, `partner_id`)
    #   (b) the first relation segment of every dotted depends_on / reads_field
    #       object (e.g. `line_ids` in `account_move.line_ids.amount_residual`)
    #       — these relations are *used* by the corpus but often not declared
    #       as standalone Property nodes, yet `RelationMap::from_corpus` needs
    #       their target to resolve the very paths that reference them.
    candidate_field_iris: Set[str] = set()
    for t in triples:
        if t["p"] == "rdf:type" and t["o"] == "ogit:Property" and t["s"].startswith(
            "odoo:"
        ):
            body = t["s"][len("odoo:") :]
            if "." in body and body.split(".", 1)[0] in object_type_models:
                candidate_field_iris.add(t["s"])
        if t["p"] in ("depends_on", "reads_field") and t["o"].startswith("odoo:"):
            parts = t["o"][len("odoo:") :].split(".")
            # model.rel.<...> — the first segment after the model is a relation
            if len(parts) >= 3 and parts[0] in object_type_models:
                candidate_field_iris.add(f"odoo:{parts[0]}.{parts[1]}")

    # field IRI → ALL emitting method IRIs (for the deep-read lift target).
    # A field can be emitted by more than one method (e.g. `stock_move.quantity`
    # is emitted by BOTH `_compute_quantity` AND `_onchange_product_uom_qty`).
    # The deep `reads_field` must be lifted onto EVERY emitter, or the
    # recompute-ordering edge is lost for all but one of them (typically the
    # `_compute_*`). Index is a de-duplicated, sorted list per field for
    # determinism.
    field_emitters: Dict[str, List[str]] = {}
    for t in triples:
        if t["p"] == "emitted_by":
            field_emitters.setdefault(t["s"], [])
            if t["o"] not in field_emitters[t["s"]]:
                field_emitters[t["s"]].append(t["o"])
    for methods in field_emitters.values():
        methods.sort()

    new_lines: List[str] = []
    stats = {
        "target": 0,
        "inverse_name": 0,
        "deep_reads_field": 0,
        "deep_skip_unknown_hop": 0,
        "deep_skip_self_loop": 0,
        "deep_skip_no_emitter": 0,
    }

    # ── P1: target / inverse_name ─────────────────────────────────────────
    for iri in sorted(candidate_field_iris):
        # iri = "odoo:<model>.<field>"  (split on FIRST dot after the namespace)
        if not iri.startswith("odoo:"):
            continue
        body = iri[len("odoo:") :]
        if "." not in body:
            continue
        model_us, field_name = body.split(".", 1)
        # A field name itself never contains a dot in the corpus Property nodes,
        # but split(maxsplit=1) keeps the field intact even if it did.
        entry = relmap.get((model_us, field_name))
        if entry is None:
            continue
        comodel, inverse = entry
        spo = (iri, "target", comodel)
        if spo not in existing_spo:
            new_lines.append(triple_line(iri, "target", comodel, *TARGET_TRUTH))
            existing_spo.add(spo)
            stats["target"] += 1
        if inverse:
            spo_inv = (iri, "inverse_name", inverse)
            if spo_inv not in existing_spo:
                new_lines.append(
                    triple_line(iri, "inverse_name", inverse, *INVERSE_TRUTH)
                )
                existing_spo.add(spo_inv)
                stats["inverse_name"] += 1

    # ── P0: deep reads_field ──────────────────────────────────────────────
    # For each depends_on triple (field, depends_on, odoo:<m>.<seg...>), if the
    # dependent FIELD is emitted by a method, lift the resolved deep read onto
    # that method.
    for t in triples:
        if t["p"] != "depends_on":
            continue
        dep_field = t["s"]
        methods = field_emitters.get(dep_field)
        if not methods:
            stats["deep_skip_no_emitter"] += 1
            continue
        obj = t["o"]
        if not obj.startswith("odoo:"):
            continue
        obj_body = obj[len("odoo:") :]
        parts = obj_body.split(".")
        # parts[0] = model, parts[1:] = traversal path.  Need ≥ 2 path
        # segments (rel + leaf) for a deep lift; a single segment is the
        # same-model field read already covered by depends_on.
        path = parts[1:]
        if len(path) < 2:
            continue
        resolved = resolve_path(parts[0], path, relmap)
        if resolved is None:
            stats["deep_skip_unknown_hop"] += 1
            continue
        final_model_us, leaf = resolved
        deep_obj = f"odoo:{final_model_us}.{leaf}"
        # Lift the deep read onto EVERY method that emits the dependent field
        # (sorted for determinism). A read equal to its own emitter is dropped
        # per-emitter (the self-loop guard), so a multi-emitter field can emit
        # the deep read for some emitters and skip it for the one that IS the
        # leaf's owning method.
        for method in methods:
            if deep_obj == method:
                stats["deep_skip_self_loop"] += 1
                continue
            spo = (method, "reads_field", deep_obj)
            if spo in existing_spo:
                continue
            new_lines.append(
                triple_line(method, "reads_field", deep_obj, *DEEP_READ_TRUTH)
            )
            existing_spo.add(spo)
            stats["deep_reads_field"] += 1

    new_lines.sort()
    return new_lines, stats


def run(corpus_path: str, addons_root: str, out_path: str) -> dict:
    """Enrich `corpus_path` using `addons_root`, write to `out_path`. Returns stats."""
    triples = load_corpus(corpus_path)
    relmap = build_relation_map(addons_root)
    new_lines, stats = enrich(triples, relmap)
    stats["corpus_triples_in"] = len(triples)
    stats["relmap_entries"] = len(relmap)
    stats["new_triples"] = len(new_lines)

    # Preserve the original corpus lines verbatim, then append the sorted new
    # triples. Additive: never rewrites or reorders the existing corpus.
    with open(corpus_path, encoding="utf-8") as fh:
        original = [ln.rstrip("\n") for ln in fh if ln.strip()]
    all_lines = original + new_lines
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(all_lines))
        fh.write("\n")
    stats["corpus_triples_out"] = len(all_lines)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m odoo_blueprint_extractor.spo_enrich",
        description=(
            "Enrich an SPO ndjson corpus with FK target/inverse_name (P1) and "
            "deep reads_field (P0) triples, read from the Odoo ORM source."
        ),
    )
    p.add_argument(
        "--corpus",
        required=True,
        metavar="NDJSON",
        help="Path to the SPO corpus ndjson to enrich (read).",
    )
    p.add_argument(
        "--addons",
        default="/home/user/odoo/addons",
        metavar="DIR",
        help="Odoo addons root (default: /home/user/odoo/addons).",
    )
    p.add_argument(
        "--out",
        metavar="NDJSON",
        help="Output path (default: in place over --corpus).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out_path = args.out or args.corpus
    if not os.path.isdir(args.addons):
        sys.exit(f"ERROR: addons root not found: {args.addons}")
    if not os.path.isfile(args.corpus):
        sys.exit(f"ERROR: corpus not found: {args.corpus}")
    stats = run(args.corpus, args.addons, out_path)
    print(
        "# spo_enrich: "
        f"in={stats['corpus_triples_in']} relmap={stats['relmap_entries']} "
        f"target={stats['target']} inverse_name={stats['inverse_name']} "
        f"deep_reads_field={stats['deep_reads_field']} "
        f"(skips: unknown_hop={stats['deep_skip_unknown_hop']} "
        f"self_loop={stats['deep_skip_self_loop']}) "
        f"out={stats['corpus_triples_out']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
