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
    a `_name`; relational fields on such classes must map onto the inherited
    model(s), or their target/inverse_name (and any deep hop through them) is
    lost.
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

    def test_inherit_list_binds_field_to_each_model(self):
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
        self.assertEqual(
            relmap.get(("purchase_order", "partner_id")), ("res.partner", None)
        )

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
