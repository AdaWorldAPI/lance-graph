"""SPO corpus enrichment — FK target/inverse_name (P1) + deep reads_field (P0)
+ inherits_from (P1b, ruff#19) + validation_kind (P2, ruff#21).

Stdlib-only Python 3. Reads the Odoo ORM source (the same addons tree the
ORM extractor parses) to build a relation-target map, then enriches an
existing SPO ndjson corpus with four additive predicate families:

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

  * **P1b — `inherits_from`** (UPSTREAM_WISHLIST P1, ruff#19 ratified the
    cross-language shape `(class, inherits_from, <base>)` for C++ + Rails):
    for every model class with a `_name` AND a `_inherit` declaration whose
    base is itself a corpus-declared `ogit:ObjectType`, emit::

        (odoo:account_move, inherits_from, odoo:mail_thread)
        (odoo:account_move, inherits_from, odoo:mail_activity_mixin)

    Self-inherits (`_inherit` equal to `_name`, the Odoo "extend-in-place"
    idiom) are dropped at scan time. `_inherits` (delegation, dict form) is
    flattened the same way — bases lift via their dict keys.
    `_inherit`-only classes (no `_name`) do NOT emit inherits_from: those
    extend an existing model in place, and the inheritance edge already
    exists at the original declaration site.

  * **P2 — `validation_kind`** (UPSTREAM_WISHLIST P2, ruff#21 ratified):
    for every `@api.constrains`-decorated method on a corpus-declared model,
    classify the body by AST pattern + emit one triple per detected kind.
    Subject is the *method* IRI (analogous to ruff#21 keying on the
    *attribute* IRI). Recognised kinds + conservative detectors::

        presence   — `not <Name|Attribute|Subscript>` / `<expr> is None`
        uniqueness — `<rs>.search_count(...)` call
        range      — `< | > | <= | >=` (LHS NOT a `search_count` call)
        format     — `re.match` / `re.fullmatch` / `re.search` call
        lookup     — `<rs>.search(...)` / `<rs>.browse(...)` call

    A method that matches no pattern emits nothing — the existing `raises`
    triple still records it can raise. Constraints on `_inherit`-only
    extension classes bind to `_inherit[0]` (mirrors `_scan_file`'s field
    binding decision per #525 codex review).

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
# inherits_from is a class-level declaration (authoritative).
INHERITS_FROM_TRUTH = (0.95, 0.90)
# validation_kind is an AST-pattern classification of a method body
# (heuristic, not authoritative).
VALIDATION_KIND_TRUTH = (0.85, 0.75)

# Recognised `validation_kind` strings (matches ruff#21's Rails set where
# semantics overlap; Odoo-specific patterns like `lookup` extend it).
_VALIDATION_KINDS = ("presence", "uniqueness", "range", "format", "lookup")


def _is_uniqueness_call(node: ast.expr) -> bool:
    """True iff `node` is a `<recordset>.search_count(...)` call — the LHS
    of the canonical Odoo uniqueness pattern. Used to suppress `range` for
    that specific shape."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "search_count"
    )


def _is_api_constrains(fn: ast.FunctionDef) -> bool:
    """True iff `fn` carries an `@api.constrains(...)` decorator."""
    for dec in fn.decorator_list:
        call = dec if isinstance(dec, ast.Call) else None
        if call is None:
            continue
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "api"
            and call.func.attr == "constrains"
        ):
            return True
    return False


def _classify_constrains_body(fn: ast.FunctionDef) -> Set[str]:
    """Walk an `@api.constrains` method body; return the set of detected
    `validation_kind` values. Conservative: a body that triggers none of the
    detectors returns the empty set (the existing `raises` triple still
    records the bare fact that it raises).

    Detector ordering follows the codex P2 fixes on #526:
      - `range` skipped when LHS is a `search_count(...)` (uniqueness shape).
      - `presence` triggers ONLY on `not <Name|Attribute|Subscript>` — NOT on
        `not <Call>` (that's the negated-call wrap for format/lookup).
    """
    kinds: Set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re"
                and node.func.attr in ("match", "fullmatch", "search")
            ):
                kinds.add("format")
            if node.func.attr == "search_count":
                kinds.add("uniqueness")
            if node.func.attr in ("search", "browse"):
                kinds.add("lookup")
        if isinstance(node, ast.Compare):
            if not _is_uniqueness_call(node.left):
                for op in node.ops:
                    if isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)):
                        kinds.add("range")
                        break
            for cmp_op, right in zip(node.ops, node.comparators):
                if isinstance(cmp_op, ast.Is) and isinstance(right, ast.Constant) and right.value is None:
                    kinds.add("presence")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            if isinstance(node.operand, (ast.Name, ast.Attribute, ast.Subscript)):
                kinds.add("presence")
    return kinds


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


def _scan_file(
    path: str,
    relmap: Dict[Tuple[str, str], Tuple[str, Optional[str]]],
    inherits: Optional[Dict[str, Set[str]]] = None,
    constrains: Optional[Dict[Tuple[str, str], Set[str]]] = None,
) -> None:
    """Parse one .py file; record every relational field on every named model.
    Optionally also record `inherits_from` edges and `@api.constrains`
    validation_kind facts.

    A model is keyed by its `_name` when present, ELSE by its `_inherit`
    target(s) — the common Odoo extension form `_inherit = "some.model"`
    (or `_inherit = ["a", "b"]`) with no `_name`. Relational fields capture
    comodel (kw `comodel_name` or positional arg 0) and inverse name
    (One2many's positional arg 1 / kw `inverse_name`; Many2one has no inverse
    here), and are mapped onto EVERY resolved model name.

    When `inherits` is provided AND `_name` is set, lift each `_inherit`
    base (string or list) and each `_inherits` dict key into `inherits`,
    dropping self-inherits (`_inherit == _name`). `_inherit`-only classes
    do NOT emit inherits_from — those extend in place; the inheritance edge
    already exists at the original declaration site.

    When `constrains` is provided, classify every `@api.constrains`
    method body and attach the resulting `validation_kind` set to each
    `model_name` the class binds to (same set the field binding uses —
    `[_name]` if present, else `inherit[0]`).
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
        inherits_models: List[str] = []  # _inherits dict keys (delegation)
        local_fields: Dict[str, Tuple[str, Optional[str]]] = {}
        local_constrains: List[Tuple[str, ast.FunctionDef]] = []

        for stmt in node.body:
            # Method definitions — gather @api.constrains methods for P2.
            if constrains is not None and isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                if _is_api_constrains(stmt):
                    local_constrains.append((stmt.name, stmt))
                continue

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

            # _inherits = {'foo.bar': 'foo_id'} — delegation; dict keys lift.
            if target_name == "_inherits" and isinstance(stmt.value, ast.Dict):
                for k in stmt.value.keys:
                    s = _const_str(k)
                    if s is not None:
                        inherits_models.append(s)
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
        # if present, else the single in-place `_inherit` target. A class with
        # neither contributes nothing.
        #
        # No-`_name` extension reopens ONE existing model in place. Odoo binds
        # such a class to `_inherit[0]`; a multi-element `_inherit` adds the rest
        # as mixins, NOT as additional homes for these local fields. Assigning
        # the whole list here would attach the fields to every secondary mixin
        # and let build_relation_map() emit bogus `target`/`reads_field` triples
        # for them. `parsers/classes.py` collapses the no-name case to
        # `inherit[0]` for the same reason; mirror it.
        if name_model is not None:
            model_names = [name_model]
        else:
            model_names = inherit_models[:1]

        for model_name in model_names:
            mu = model_to_underscore(model_name)
            for field_name, (comodel, inverse) in local_fields.items():
                # Last write wins across _inherit reopenings of the same model;
                # comodel for a given (model, field) is stable in practice.
                relmap[(mu, field_name)] = (comodel, inverse)

        # ── P1b: inherits_from edges ─────────────────────────────────────
        # Only when the class declares a NEW model (_name is set). An
        # _inherit-only class extends an existing model in place — the
        # inheritance edge already exists at the original declaration.
        if inherits is not None and name_model is not None:
            mu = model_to_underscore(name_model)
            bases = {
                model_to_underscore(b)
                for b in (*inherit_models, *inherits_models)
                if model_to_underscore(b) != mu  # drop "extend-in-place"
            }
            if bases:
                inherits.setdefault(mu, set()).update(bases)

        # ── P2: validation_kind per @api.constrains method ───────────────
        # Constraints bind to the SAME model_names the fields do (per #525
        # codex P2 decision: inherit[0] only for no-_name classes).
        if constrains is not None and local_constrains and model_names:
            for method_name, fn_node in local_constrains:
                kinds = _classify_constrains_body(fn_node)
                if not kinds:
                    continue
                for model_name in model_names:
                    mu = model_to_underscore(model_name)
                    constrains.setdefault((mu, method_name), set()).update(kinds)


def build_all_facts(
    addons_root: str,
) -> Tuple[
    Dict[Tuple[str, str], Tuple[str, Optional[str]]],
    Dict[str, Set[str]],
    Dict[Tuple[str, str], Set[str]],
]:
    """Single-pass scan that populates (relmap, inherits, constrains).

    Returns:
        relmap     — (model_us, field) → (comodel_dotted, inverse_or_None)
        inherits   — model_us → set(base_us)  from `_inherit` + `_inherits`
        constrains — (model_us, method) → set(kind) from `@api.constrains`
    """
    relmap: Dict[Tuple[str, str], Tuple[str, Optional[str]]] = {}
    inherits: Dict[str, Set[str]] = {}
    constrains: Dict[Tuple[str, str], Set[str]] = {}
    pattern = os.path.join(addons_root, "**", "*.py")
    for path in glob.iglob(pattern, recursive=True):
        _scan_file(path, relmap, inherits, constrains)
    return relmap, inherits, constrains


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
    inherits: Optional[Dict[str, Set[str]]] = None,
    constrains: Optional[Dict[Tuple[str, str], Set[str]]] = None,
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

    # Method IRIs declared via has_function — scope for P2 emissions.
    declared_methods: Set[str] = {
        t["o"] for t in triples if t["p"] == "has_function" and t["o"].startswith("odoo:")
    }

    new_lines: List[str] = []
    stats = {
        "target": 0,
        "inverse_name": 0,
        "deep_reads_field": 0,
        "deep_skip_unknown_hop": 0,
        "deep_skip_self_loop": 0,
        "deep_skip_no_emitter": 0,
        "inherits_from": 0,
        "inherits_skip_unknown_base": 0,
        "validation_kind": 0,
        "validation_skip_unknown_method": 0,
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

    # ── P1b: inherits_from ────────────────────────────────────────────────
    if inherits:
        for child_us in sorted(inherits):
            if child_us not in object_type_models:
                continue
            child_iri = f"odoo:{child_us}"
            for base_us in sorted(inherits[child_us]):
                if base_us not in object_type_models:
                    stats["inherits_skip_unknown_base"] += 1
                    continue
                base_iri = f"odoo:{base_us}"
                spo = (child_iri, "inherits_from", base_iri)
                if spo in existing_spo:
                    continue
                new_lines.append(
                    triple_line(child_iri, "inherits_from", base_iri, *INHERITS_FROM_TRUTH)
                )
                existing_spo.add(spo)
                stats["inherits_from"] += 1

    # ── P2: validation_kind ───────────────────────────────────────────────
    if constrains:
        for (model_us, method_name), kinds in sorted(constrains.items()):
            method_iri = f"odoo:{model_us}.{method_name}"
            if method_iri not in declared_methods:
                stats["validation_skip_unknown_method"] += 1
                continue
            for kind in sorted(kinds):
                if kind not in _VALIDATION_KINDS:
                    continue
                spo = (method_iri, "validation_kind", kind)
                if spo in existing_spo:
                    continue
                new_lines.append(
                    triple_line(method_iri, "validation_kind", kind, *VALIDATION_KIND_TRUTH)
                )
                existing_spo.add(spo)
                stats["validation_kind"] += 1

    new_lines.sort()
    return new_lines, stats


def run(corpus_path: str, addons_root: str, out_path: str) -> dict:
    """Enrich `corpus_path` using `addons_root`, write to `out_path`. Returns stats."""
    triples = load_corpus(corpus_path)
    relmap, inherits, constrains = build_all_facts(addons_root)
    new_lines, stats = enrich(triples, relmap, inherits, constrains)
    stats["corpus_triples_in"] = len(triples)
    stats["relmap_entries"] = len(relmap)
    stats["inherits_entries"] = sum(len(v) for v in inherits.values())
    stats["constrains_entries"] = sum(len(v) for v in constrains.values())
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
        f"inherits_from={stats['inherits_from']} "
        f"validation_kind={stats['validation_kind']} "
        f"out={stats['corpus_triples_out']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
