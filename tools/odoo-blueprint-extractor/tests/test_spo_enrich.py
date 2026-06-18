"""Unit tests for spo_enrich — FK target/inverse_name (P1) + deep reads_field (P0).

Run with:
    python tests/test_spo_enrich.py
or:
    python -m unittest tests.test_spo_enrich

Core logic (path resolution, P1/P0 emission, dedup, self-loop drop) is tested
against synthetic in-memory triples + a synthetic relation map, so the suite is
hermetic (no dependency on the Odoo source tree being present).
"""

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from odoo_blueprint_extractor.spo_enrich import (  # noqa: E402
    build_relation_map,
    enrich,
    model_to_underscore,
    resolve_path,
)


def t(s, p, o, f=1.0, c=1.0):
    return {"s": s, "p": p, "o": o, "f": f, "c": c}


# A tiny relation map: (model_us, field) -> (comodel_dotted, inverse_or_None).
RELMAP = {
    ("account_move", "line_ids"): ("account.move.line", "move_id"),
    ("account_move", "partner_id"): ("res.partner", None),
    ("account_move_line", "move_id"): ("account.move", None),
}


class TestModelNormalization(unittest.TestCase):
    def test_dotted_to_underscore(self):
        self.assertEqual(model_to_underscore("account.move.line"), "account_move_line")
        self.assertEqual(model_to_underscore("res.partner"), "res_partner")
        self.assertEqual(model_to_underscore("uom"), "uom")


class TestResolvePath(unittest.TestCase):
    def test_two_segment_relation_leaf(self):
        # line_ids.balance starting at account_move → (account_move_line, balance)
        self.assertEqual(
            resolve_path("account_move", ["line_ids", "balance"], RELMAP),
            ("account_move_line", "balance"),
        )

    def test_unknown_hop_returns_none(self):
        # `mystery` is not a known relation on account_move.
        self.assertIsNone(
            resolve_path("account_move", ["mystery", "leaf"], RELMAP)
        )

    def test_single_segment_is_leaf_on_start_model(self):
        # A single segment has no relational hop; leaf on the start model.
        self.assertEqual(
            resolve_path("account_move", ["amount_total"], RELMAP),
            ("account_move", "amount_total"),
        )


class TestP1TargetInverse(unittest.TestCase):
    def test_target_and_inverse_emitted_for_declared_relation(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.line_ids", "rdf:type", "ogit:Property"),
        ]
        lines, stats = enrich(triples, RELMAP)
        joined = "\n".join(lines)
        self.assertIn(
            '{"s":"odoo:account_move.line_ids","p":"target","o":"account.move.line"',
            joined,
        )
        self.assertIn(
            '{"s":"odoo:account_move.line_ids","p":"inverse_name","o":"move_id"',
            joined,
        )
        self.assertEqual(stats["target"], 1)
        self.assertEqual(stats["inverse_name"], 1)

    def test_many2one_emits_target_but_no_inverse(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.partner_id", "rdf:type", "ogit:Property"),
        ]
        lines, stats = enrich(triples, RELMAP)
        self.assertEqual(stats["target"], 1)
        self.assertEqual(stats["inverse_name"], 0)
        self.assertTrue(any('"o":"res.partner"' in ln for ln in lines))

    def test_relation_referenced_only_in_dotted_path_gets_target(self):
        # account_move.line_ids is NOT declared as a Property here, but it is
        # the first relation segment of a depends_on path on a known model.
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.amount_total", "rdf:type", "ogit:Property"),
            t(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.line_ids.balance",
            ),
        ]
        lines, stats = enrich(triples, RELMAP)
        self.assertEqual(stats["target"], 1)
        self.assertTrue(
            any(
                '"s":"odoo:account_move.line_ids","p":"target"' in ln
                for ln in lines
            )
        )

    def test_unknown_model_never_gets_target(self):
        # `mystery_model` is not a corpus ObjectType → no target invented.
        triples = [
            t("odoo:mystery_model.rel", "rdf:type", "ogit:Property"),
        ]
        lines, stats = enrich(triples, {("mystery_model", "rel"): ("x.y", None)})
        self.assertEqual(stats["target"], 0)
        self.assertEqual(lines, [])


class TestP0DeepReadsField(unittest.TestCase):
    def test_deep_read_lifted_onto_emitter(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.amount_total", "rdf:type", "ogit:Property"),
            # amount_total is emitted_by _compute_amount
            t(
                "odoo:account_move.amount_total",
                "emitted_by",
                "odoo:account_move._compute_amount",
            ),
            # @api.depends('line_ids.balance')
            t(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.line_ids.balance",
            ),
        ]
        lines, stats = enrich(triples, RELMAP)
        self.assertEqual(stats["deep_reads_field"], 1)
        self.assertTrue(
            any(
                '"s":"odoo:account_move._compute_amount","p":"reads_field",'
                '"o":"odoo:account_move_line.balance"' in ln
                for ln in lines
            )
        )

    def test_single_segment_depends_is_not_deep_lifted(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.amount_total", "rdf:type", "ogit:Property"),
            t(
                "odoo:account_move.amount_total",
                "emitted_by",
                "odoo:account_move._compute_amount",
            ),
            # same-model field dep — no relation hop, no deep lift.
            t(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.state",
            ),
        ]
        _, stats = enrich(triples, RELMAP)
        self.assertEqual(stats["deep_reads_field"], 0)

    def test_unknown_hop_is_skipped_and_counted(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.amount_total", "emitted_by", "odoo:account_move._c"),
            t(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.mystery.leaf",
            ),
        ]
        _, stats = enrich(triples, RELMAP)
        self.assertEqual(stats["deep_reads_field"], 0)
        self.assertEqual(stats["deep_skip_unknown_hop"], 1)

    def test_self_loop_deep_read_is_dropped(self):
        # A deep read whose object equals the emitting method is never emitted.
        relmap = {("m", "rel"): ("m", None)}
        triples = [
            t("odoo:m", "rdf:type", "ogit:ObjectType"),
            # field `f` emitted_by method whose IRI equals the resolved deep obj
            t("odoo:m.f", "emitted_by", "odoo:m._compute_x"),
            t("odoo:m.f", "depends_on", "odoo:m.rel._compute_x"),
        ]
        _, stats = enrich(triples, relmap)
        self.assertEqual(stats["deep_skip_self_loop"], 1)
        self.assertEqual(stats["deep_reads_field"], 0)


class TestIdempotenceAndDedup(unittest.TestCase):
    def test_existing_spo_not_re_emitted(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.line_ids", "rdf:type", "ogit:Property"),
            # target already present → must not duplicate
            t("odoo:account_move.line_ids", "target", "account.move.line"),
        ]
        lines, stats = enrich(triples, RELMAP)
        self.assertEqual(stats["target"], 0)
        self.assertFalse(any('"p":"target"' in ln for ln in lines))

    def test_output_lines_are_valid_json_and_sorted(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.line_ids", "rdf:type", "ogit:Property"),
            t("odoo:account_move.partner_id", "rdf:type", "ogit:Property"),
        ]
        lines, _ = enrich(triples, RELMAP)
        for ln in lines:
            obj = json.loads(ln)  # raises if malformed
            self.assertEqual(set(obj.keys()), {"s", "p", "o", "f", "c"})
        self.assertEqual(lines, sorted(lines), "new triples must be sorted")


class TestMultiEmitterDeepReads(unittest.TestCase):
    """Fix 1 — a field with multiple emitters lifts the deep read onto EACH.

    Mirrors the in-repo `stock_move.quantity` case: emitted by both a
    `_compute_*` and an `_onchange_*`; the deep cross-model read must land on
    both, not just the last `emitted_by` seen.
    """

    def test_deep_read_lifted_onto_all_emitters(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.amount_total", "rdf:type", "ogit:Property"),
            # amount_total is emitted by TWO methods
            t(
                "odoo:account_move.amount_total",
                "emitted_by",
                "odoo:account_move._compute_amount",
            ),
            t(
                "odoo:account_move.amount_total",
                "emitted_by",
                "odoo:account_move._onchange_lines",
            ),
            t(
                "odoo:account_move.amount_total",
                "depends_on",
                "odoo:account_move.line_ids.balance",
            ),
        ]
        lines, stats = enrich(triples, RELMAP)
        self.assertEqual(stats["deep_reads_field"], 2)
        for method in (
            "odoo:account_move._compute_amount",
            "odoo:account_move._onchange_lines",
        ):
            self.assertTrue(
                any(
                    f'"s":"{method}","p":"reads_field",'
                    '"o":"odoo:account_move_line.balance"' in ln
                    for ln in lines
                ),
                f"deep read must be lifted onto {method}",
            )

    def test_self_loop_dropped_per_emitter_others_kept(self):
        # Two emitters; the resolved deep object equals ONE of them (self-loop)
        # and differs from the other. The self-loop is dropped, the other kept.
        relmap = {("m", "rel"): ("m", None)}
        triples = [
            t("odoo:m", "rdf:type", "ogit:ObjectType"),
            t("odoo:m.f", "emitted_by", "odoo:m._compute_x"),
            t("odoo:m.f", "emitted_by", "odoo:m._other"),
            # resolves to odoo:m._compute_x (== first emitter → self-loop)
            t("odoo:m.f", "depends_on", "odoo:m.rel._compute_x"),
        ]
        _, stats = enrich(triples, relmap)
        self.assertEqual(stats["deep_skip_self_loop"], 1)
        self.assertEqual(stats["deep_reads_field"], 1)


class TestInheritOnlyRelationMap(unittest.TestCase):
    """Fix 2 — `_inherit`-only classes (no `_name`) contribute relational fields.

    The common Odoo extension form reopens an existing model via
    `_inherit = "some.model"` (string) or `_inherit = ["a", "b"]` (list) WITHOUT
    a `_name`; relational fields on such classes must map onto the single
    in-place extension target, or their target/inverse_name (and any deep hop
    through them) is lost. Odoo binds the no-`_name` case to `_inherit[0]`; any
    further entries are mixins, not additional homes for the local fields, so
    they must NOT receive the relational fields (mirrors `parsers/classes.py`).
    """

    def _scan_source(self, src: str):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "m.py"), "w", encoding="utf-8") as fh:
                fh.write(src)
            return build_relation_map(d)

    def test_inherit_string_binds_field_to_inherited_model(self):
        src = (
            "from odoo import fields, models\n"
            "class AccountMove(models.Model):\n"
            "    _inherit = 'account.move'\n"
            "    authorized_transaction_ids = fields.Many2many(\n"
            "        comodel_name='payment.transaction')\n"
        )
        relmap = self._scan_source(src)
        self.assertEqual(
            relmap.get(("account_move", "authorized_transaction_ids")),
            ("payment.transaction", None),
        )

    def test_inherit_list_binds_field_to_first_model_only(self):
        # No-`_name` multi-element `_inherit` extends `_inherit[0]` in place;
        # the rest are mixins and must NOT inherit the local relational fields
        # (otherwise build_relation_map() emits bogus target/reads_field triples
        # for secondary parents). Matches parsers/classes.py inherit[0] collapse.
        src = (
            "from odoo import fields, models\n"
            "class Mixin(models.AbstractModel):\n"
            "    _inherit = ['sale.order', 'purchase.order']\n"
            "    partner_id = fields.Many2one('res.partner')\n"
        )
        relmap = self._scan_source(src)
        self.assertEqual(
            relmap.get(("sale_order", "partner_id")), ("res.partner", None)
        )
        self.assertIsNone(relmap.get(("purchase_order", "partner_id")))

    def test_inherit_tuple_form_is_accepted(self):
        src = (
            "from odoo import fields, models\n"
            "class Ext(models.Model):\n"
            "    _inherit = ('stock.move',)\n"
            "    line_ids = fields.One2many('stock.move.line', 'move_id')\n"
        )
        relmap = self._scan_source(src)
        self.assertEqual(
            relmap.get(("stock_move", "line_ids")),
            ("stock.move.line", "move_id"),
        )

    def test_name_takes_precedence_over_inherit(self):
        # When both _name and _inherit are present, fields belong to _name only
        # (matches the package class parser: _name wins).
        src = (
            "from odoo import fields, models\n"
            "class SaleOrder(models.Model):\n"
            "    _name = 'sale.order'\n"
            "    _inherit = ['mail.thread']\n"
            "    partner_id = fields.Many2one('res.partner')\n"
        )
        relmap = self._scan_source(src)
        self.assertEqual(
            relmap.get(("sale_order", "partner_id")), ("res.partner", None)
        )
        self.assertIsNone(relmap.get(("mail_thread", "partner_id")))


# ---------------------------------------------------------------------------
# PR #526 — inherits_from (P1b/ruff#19) + validation_kind (P2/ruff#21)
# ---------------------------------------------------------------------------


def _scan_src(src):
    from odoo_blueprint_extractor.spo_enrich import _scan_file
    relmap = {}
    inherits = {}
    constrains = {}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        _scan_file(path, relmap, inherits, constrains)
    finally:
        os.unlink(path)
    return relmap, inherits, constrains


def _classify(src):
    import ast as _ast
    from odoo_blueprint_extractor.spo_enrich import _classify_constrains_body
    tree = _ast.parse(src)
    fn = next(n for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef))
    return _classify_constrains_body(fn)


class TestP1bInheritsFrom(unittest.TestCase):
    def test_emit_for_corpus_declared_base(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:mail_thread", "rdf:type", "ogit:ObjectType"),
        ]
        lines, stats = enrich(triples, RELMAP, inherits={"account_move": {"mail_thread"}})
        self.assertIn(
            '{"s":"odoo:account_move","p":"inherits_from","o":"odoo:mail_thread"',
            "\n".join(lines),
        )
        self.assertEqual(stats["inherits_from"], 1)

    def test_skip_unknown_base(self):
        triples = [t("odoo:account_move", "rdf:type", "ogit:ObjectType")]
        lines, stats = enrich(triples, RELMAP, inherits={"account_move": {"mail_thread"}})
        self.assertEqual(stats["inherits_from"], 0)
        self.assertEqual(stats["inherits_skip_unknown_base"], 1)
        self.assertEqual(lines, [])

    def test_dedup_against_existing(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:mail_thread", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move", "inherits_from", "odoo:mail_thread"),
        ]
        _, stats = enrich(triples, RELMAP, inherits={"account_move": {"mail_thread"}})
        self.assertEqual(stats["inherits_from"], 0)

    def test_inherit_string_lifts(self):
        _, inh, _ = _scan_src(
            "class M:\n    _name = 'account.move'\n    _inherit = 'mail.thread'\n"
        )
        self.assertEqual(inh, {"account_move": {"mail_thread"}})

    def test_inherit_list_lifts_all(self):
        _, inh, _ = _scan_src(
            "class M:\n    _name = 'account.move'\n"
            "    _inherit = ['mail.thread', 'mail.activity.mixin']\n"
        )
        self.assertEqual(inh, {"account_move": {"mail_thread", "mail_activity_mixin"}})

    def test_self_inherit_dropped(self):
        _, inh, _ = _scan_src(
            "class M:\n    _name = 'account.move'\n    _inherit = 'account.move'\n"
        )
        self.assertEqual(inh, {})

    def test_inherits_delegation_dict_keys_lift(self):
        _, inh, _ = _scan_src(
            "class M:\n    _name = 'account.move'\n"
            "    _inherits = {'mail.thread': 'thread_id'}\n"
        )
        self.assertEqual(inh, {"account_move": {"mail_thread"}})

    def test_inherit_only_class_no_new_edge(self):
        _, inh, _ = _scan_src(
            "class Ext:\n    _inherit = 'account.move'\n    foo = fields.Char()\n"
        )
        self.assertEqual(inh, {})


class TestP2ValidationKind(unittest.TestCase):
    def _triples(self, method_iri="odoo:account_move._check_x"):
        return [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move", "has_function", method_iri),
        ]

    def test_emit_for_declared_method(self):
        lines, stats = enrich(
            self._triples(),
            RELMAP,
            constrains={("account_move", "_check_x"): {"lookup"}},
        )
        self.assertEqual(stats["validation_kind"], 1)
        self.assertTrue(any('"validation_kind","o":"lookup"' in ln for ln in lines))

    def test_emit_multiple_kinds_per_method(self):
        _, stats = enrich(
            self._triples(),
            RELMAP,
            constrains={("account_move", "_check_x"): {"presence", "range"}},
        )
        self.assertEqual(stats["validation_kind"], 2)

    def test_skip_method_not_in_corpus(self):
        triples = [t("odoo:account_move", "rdf:type", "ogit:ObjectType")]
        lines, stats = enrich(
            triples, RELMAP, constrains={("account_move", "_check_x"): {"presence"}}
        )
        self.assertEqual(stats["validation_kind"], 0)
        self.assertEqual(stats["validation_skip_unknown_method"], 1)
        self.assertEqual(lines, [])

    def test_drop_non_canonical_kinds(self):
        _, stats = enrich(
            self._triples(),
            RELMAP,
            constrains={("account_move", "_check_x"): {"future_kind"}},
        )
        self.assertEqual(stats["validation_kind"], 0)


class TestAstClassifier(unittest.TestCase):
    def test_format_via_re_match(self):
        kinds = _classify(
            "import re\ndef _check_x(self):\n"
            "    if not re.match(r'\\d+', self.code):\n"
            "        raise ValidationError('bad')\n"
        )
        self.assertIn("format", kinds)
        # codex P2 #3: `not <Call>` is NOT presence.
        self.assertNotIn("presence", kinds)

    def test_uniqueness_via_search_count_gt(self):
        kinds = _classify(
            "def _check_x(self):\n"
            "    if self.env['m'].search_count([('a','=',1)]) > 1:\n"
            "        raise ValidationError('dup')\n"
        )
        self.assertIn("uniqueness", kinds)
        # codex P2 #2: `search_count(...) > N` is NOT also range.
        self.assertNotIn("range", kinds)

    def test_range_for_field_vs_numeric_bound(self):
        kinds = _classify(
            "def _check_x(self):\n"
            "    if self.amount < 0:\n"
            "        raise ValidationError('neg')\n"
        )
        self.assertIn("range", kinds)

    def test_lookup_via_search(self):
        kinds = _classify(
            "def _check_x(self):\n"
            "    if not self.env['r'].search([]):\n"
            "        raise ValidationError('miss')\n"
        )
        self.assertIn("lookup", kinds)
        self.assertNotIn("presence", kinds)

    def test_presence_via_attribute_truthiness(self):
        kinds = _classify(
            "def _check_x(self):\n"
            "    if not self.partner:\n"
            "        raise ValidationError('miss')\n"
        )
        self.assertIn("presence", kinds)

    def test_presence_via_is_none(self):
        kinds = _classify(
            "def _check_x(self):\n"
            "    if self.partner is None:\n"
            "        raise ValidationError('miss')\n"
        )
        self.assertIn("presence", kinds)


class TestConstrainsScan(unittest.TestCase):
    """Binding follows #525's `model_names` decision:
    `_name` if set, else `_inherit[0]` only — no broadcast to mixins."""

    def test_bind_to_name_model(self):
        _, _, c = _scan_src(
            "class M:\n    _name = 'account.move'\n"
            "    @api.constrains('amount')\n"
            "    def _check_amount(self):\n"
            "        if self.amount < 0:\n"
            "            raise ValidationError('neg')\n"
        )
        self.assertEqual(c, {("account_move", "_check_amount"): {"range"}})

    def test_inherit_only_binds_to_inherit_zero(self):
        # `_inherit = ['a', 'b']` with no `_name` binds to `a` only.
        _, _, c = _scan_src(
            "class Ext:\n"
            "    _inherit = ['account.move', 'mail.thread']\n"
            "    @api.constrains('amount')\n"
            "    def _check_amount(self):\n"
            "        if self.amount < 0:\n"
            "            raise ValidationError('neg')\n"
        )
        self.assertEqual(c, {("account_move", "_check_amount"): {"range"}})

    def test_name_plus_inherit_binds_to_name_only(self):
        _, _, c = _scan_src(
            "class M:\n    _name = 'account.move'\n    _inherit = 'mail.thread'\n"
            "    @api.constrains('subject')\n"
            "    def _check_subject(self):\n"
            "        if not self.subject:\n"
            "            raise ValidationError('miss')\n"
        )
        self.assertEqual(c, {("account_move", "_check_subject"): {"presence"}})


# ---------------------------------------------------------------------------
# PR #529 — selection_value (P3)
# ---------------------------------------------------------------------------


def _scan_sel(src):
    """Run `_scan_file` over a synthetic module; return the RESOLVED selections
    map (`None` = open domain, else the closed value list) — mirroring what
    `build_all_facts` returns."""
    from odoo_blueprint_extractor.spo_enrich import _scan_file, _resolve_selections
    relmap, inherits, constrains, sel_acc = {}, {}, {}, {}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        _scan_file(path, relmap, inherits, constrains, sel_acc)
    finally:
        os.unlink(path)
    return _resolve_selections(sel_acc)


def _classify_sel(src):
    """Resolve a single synthetic `fields.Selection(...)` →
    `(base_dynamic, base_keys, add_keys)`."""
    import ast as _ast
    from odoo_blueprint_extractor.spo_enrich import _extract_selection
    tree = _ast.parse(src)
    call = next(n for n in _ast.walk(tree) if isinstance(n, _ast.Call))
    return _extract_selection(call)


class TestSelectionExtraction(unittest.TestCase):
    def test_static_list_positional_is_base(self):
        self.assertEqual(
            _classify_sel(
                "fields.Selection([('draft','Draft'),('posted','Posted'),('cancel','Cancel')])"
            ),
            (False, ["draft", "posted", "cancel"], []),
        )

    def test_selection_kwarg_is_base(self):
        self.assertEqual(
            _classify_sel("fields.Selection(selection=[('a','A'),('b','B')], string='X')"),
            (False, ["a", "b"], []),
        )

    def test_dynamic_method_name_is_base_dynamic(self):
        # selection='_compute_states' = provided-but-unresolvable → POISON signal.
        self.assertEqual(
            _classify_sel("fields.Selection(selection='_compute_states')"), (True, [], [])
        )

    def test_bare_name_constant_is_base_dynamic(self):
        self.assertEqual(_classify_sel("fields.Selection(STATE_VALUES)"), (True, [], []))

    def test_no_selection_arg_is_not_dynamic(self):
        # No selection arg at all is NOT a provided-but-dynamic base.
        self.assertEqual(_classify_sel("fields.Selection(string='X')"), (False, [], []))

    def test_dedup_within_base(self):
        _, base, _ = _classify_sel("fields.Selection([('a','A'),('b','B'),('a','A2')])")
        self.assertEqual(base, ["a", "b"])

    def test_non_tuple_entries_skipped(self):
        _, base, _ = _classify_sel("fields.Selection([('a','A'), 'bogus', ('b','B')])")
        self.assertEqual(base, ["a", "b"])

    # codex #530 — selection_add is EXTEND, kept separate from base
    def test_selection_add_alone_is_extend(self):
        self.assertEqual(
            _classify_sel(
                "fields.Selection(selection_add=[('reviewed','Reviewed'),('void','Void')])"
            ),
            (False, [], ["reviewed", "void"]),
        )

    def test_base_and_add_kept_separate(self):
        self.assertEqual(
            _classify_sel(
                "fields.Selection(selection=[('draft','Draft'),('posted','Posted')], "
                "selection_add=[('reviewed','Reviewed')])"
            ),
            (False, ["draft", "posted"], ["reviewed"]),
        )

    # codex #532 #1 — dynamic base + add must flag dynamic (caller poisons)
    def test_dynamic_base_with_add_flags_dynamic(self):
        dyn, base, add = _classify_sel(
            "fields.Selection(STATE_CONST, selection_add=[('reviewed','Reviewed')])"
        )
        self.assertTrue(dyn, "provided-but-dynamic base must flag poison")
        self.assertEqual(base, [])
        self.assertEqual(add, ["reviewed"])  # parsed, but caller drops under poison


class TestSelectionScan(unittest.TestCase):
    """`_scan_file` binds Selection domains: static REPLACE, add EXTEND,
    dynamic POISON (sentinel `None`)."""

    def test_static_base_bound_to_name_model(self):
        sel = _scan_sel(
            "class M:\n"
            "    _name = 'account.move'\n"
            "    state = fields.Selection([('draft','Draft'),('posted','Posted')])\n"
        )
        self.assertEqual(sel, {("account_move", "state"): ["draft", "posted"]})

    def test_static_base_on_inherit_only_binds_inherit_zero(self):
        sel = _scan_sel(
            "class Ext:\n"
            "    _inherit = ['account.move', 'mail.thread']\n"
            "    kind = fields.Selection([('x','X'),('y','Y')])\n"
        )
        self.assertEqual(sel, {("account_move", "kind"): ["x", "y"]})

    def test_dynamic_base_poisons_field_to_none(self):
        # codex #532 #1: a dynamic base records the field as OPEN (None),
        # so enrich emits no closed `ASSERT IN` for it.
        sel = _scan_sel(
            "class M:\n"
            "    _name = 'account.move'\n"
            "    s = fields.Selection(selection='_compute_s')\n"
        )
        self.assertEqual(sel, {("account_move", "s"): None})

    def test_selection_add_merges_across_classes(self):
        # codex #530: base in the _name class + a selection_add extension in a
        # separate _inherit class must ACCUMULATE, not overwrite.
        sel = _scan_sel(
            "class Base:\n"
            "    _name = 'account.move'\n"
            "    state = fields.Selection([('draft','Draft'),('posted','Posted')])\n"
            "\n"
            "class Ext:\n"
            "    _inherit = 'account.move'\n"
            "    state = fields.Selection(selection_add=[('reviewed','Reviewed')])\n"
        )
        self.assertEqual(
            sel, {("account_move", "state"): ["draft", "posted", "reviewed"]}
        )

    def test_dynamic_base_plus_add_stays_poisoned(self):
        # codex #532 #1: a dynamic base in one class POISONS the domain even if
        # another class statically adds — never close an open domain.
        sel = _scan_sel(
            "class Base:\n"
            "    _name = 'account.move'\n"
            "    state = fields.Selection(STATE_CONST)\n"
            "\n"
            "class Ext:\n"
            "    _inherit = 'account.move'\n"
            "    state = fields.Selection(selection_add=[('reviewed','Reviewed')])\n"
        )
        self.assertEqual(sel, {("account_move", "state"): None})

    def test_full_redeclaration_replaces_not_unions(self):
        # codex #532 #2: a full static `selection=` redeclaration REPLACES the
        # domain; stale keys from the earlier declaration must not survive.
        sel = _scan_sel(
            "class Base:\n"
            "    _name = 'account.move'\n"
            "    state = fields.Selection([('draft','Draft'),('posted','Posted')])\n"
            "\n"
            "class Ext:\n"
            "    _inherit = 'account.move'\n"
            "    state = fields.Selection([('draft','Draft'),('open','Open')])\n"
        )
        keys = sel[("account_move", "state")]
        self.assertIn("open", keys)
        self.assertNotIn(
            "posted", keys, "stale key from the replaced declaration must be dropped"
        )

    def test_selection_add_before_base_is_order_independent(self):
        # codex #536 follow-up: the `selection_add` EXTENSION class is scanned
        # BEFORE the base `selection=` class (glob.iglob gives no order). The
        # base's REPLACE must NOT clobber the already-accumulated extension —
        # the resolved domain includes the added key regardless of order.
        sel = _scan_sel(
            "class Ext:\n"
            "    _inherit = 'account.move'\n"
            "    state = fields.Selection(selection_add=[('reviewed','Reviewed')])\n"
            "\n"
            "class Base:\n"
            "    _name = 'account.move'\n"
            "    state = fields.Selection([('draft','Draft'),('posted','Posted')])\n"
        )
        self.assertEqual(
            sel, {("account_move", "state"): ["draft", "posted", "reviewed"]}
        )


class TestP3SelectionValueEmission(unittest.TestCase):
    def _triples(self, field_iri="odoo:account_move.state"):
        return [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
        ]

    def test_emits_one_triple_per_value(self):
        lines, stats = enrich(
            self._triples(),
            RELMAP,
            selections={("account_move", "state"): ["draft", "posted", "cancel"]},
        )
        self.assertEqual(stats["selection_value"], 3)
        joined = "\n".join(lines)
        for v in ("draft", "posted", "cancel"):
            self.assertIn(
                f'{{"s":"odoo:account_move.state","p":"selection_value","o":"{v}"',
                joined,
            )

    def test_skip_unknown_model(self):
        triples = [t("odoo:account_move", "rdf:type", "ogit:ObjectType")]
        lines, stats = enrich(
            triples, RELMAP, selections={("mystery_model", "s"): ["a"]}
        )
        self.assertEqual(stats["selection_value"], 0)
        self.assertEqual(stats["selection_skip_unknown_model"], 1)
        self.assertEqual(lines, [])

    def test_dedup_against_existing(self):
        triples = [
            t("odoo:account_move", "rdf:type", "ogit:ObjectType"),
            t("odoo:account_move.state", "selection_value", "draft"),
        ]
        _, stats = enrich(
            triples, RELMAP, selections={("account_move", "state"): ["draft", "posted"]}
        )
        # 'draft' already present → only 'posted' is new.
        self.assertEqual(stats["selection_value"], 1)

    def test_open_domain_none_emits_nothing(self):
        # codex #532 #1: a poisoned (None) domain → no selection_value triples.
        triples = [t("odoo:account_move", "rdf:type", "ogit:ObjectType")]
        lines, stats = enrich(
            triples, RELMAP, selections={("account_move", "state"): None}
        )
        self.assertEqual(stats["selection_value"], 0)
        self.assertEqual(stats["selection_skip_open_domain"], 1)
        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
