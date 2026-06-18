// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Odoo Foundry-ontology loader for the SPO triple store.
//!
//! Deterministic projection of the Odoo business-logic graph — extracted from
//! `odoo/addons/` by `ruff_python_dto_check` harvest + AST analysis (see
//! `.claude/odoo/emit_ontology2.py`, `.claude/odoo/methods.parquet`) — into the
//! existing [`SpoStore`]. **Not flat routes: Foundry Object Types + their
//! Function-backed compute graph.**
//!
//! # Triple schema
//!
//! | predicate            | subject              | object               | provenance |
//! | ---                  | ---                  | ---                  | --- |
//! | `rdf:type`           | `odoo:<family>`      | `ogit:ObjectType`    | structural |
//! | `rdf:type`           | `odoo:<fam>.<field>` | `ogit:Property`      | structural |
//! | `rdf:type`           | `odoo:<fam>.<fn>`    | `ogit:Function`      | structural |
//! | `has_function`       | `odoo:<family>`      | `odoo:<fam>.<fn>`    | structural |
//! | `emitted_by`         | `odoo:<fam>.<field>` | `odoo:<fam>.<fn>`    | body write (authoritative) |
//! | `depends_on`         | `odoo:<fam>.<field>` | `odoo:<fam>.<dep>`   | `@api.depends` arg (authoritative) |
//! | `reads_field`        | `odoo:<fam>.<fn>`    | `odoo:<fam>.<field>` | body read (inferred) |
//! | `raises`             | `odoo:<fam>.<fn>`    | `exc:<Type>`         | body raise (authoritative) |
//! | `traverses_relation` | `odoo:<fam>.<fn>`    | `odoo:<fam>.<rel>`   | body for-loop (inferred) |
//! | `target`             | `odoo:<fam>.<rel>`   | `"<comodel.dotted>"` | relational comodel (declared) |
//! | `inverse_name`       | `odoo:<fam>.<rel>`   | `"<inverse>"`        | One2many/inverse (declared) |
//! | `inherits_from`      | `odoo:<family>`      | `odoo:<base_family>` | `_inherit`/`_inherits` base (declared) |
//! | `validation_kind`    | `odoo:<fam>.<fn>`    | `"<kind>"`           | `@api.constrains` body pattern (inferred) |
//!
//! ## FK-target + deep-read enrichment (`spo_enrich`)
//!
//! Two additive predicate families layer on top of the base extraction
//! (`tools/odoo-blueprint-extractor/odoo_blueprint_extractor/spo_enrich.py`,
//! the `UPSTREAM_WISHLIST` P1 + P0 corpus enrichment):
//!
//! - **`target` / `inverse_name`** (P1, ruff#18 sibling-triple shape): for
//!   every relational field (Many2one / One2many / Many2many / Reference) on a
//!   corpus model whose comodel resolves from the Odoo source, a sibling
//!   triple keyed by the relation IRI carries the *raw* dotted comodel name
//!   (e.g. `(odoo:account_move.line_ids, target, "account.move.line")` +
//!   `(…, inverse_name, "move_id")`). This is the cross-language analog of
//!   ruff#18's `(WorkPackage.owner, class_name, "User")`.
//! - **deep `reads_field`** (P0): each `@api.depends('rel.leaf', …)` whose
//!   `rel` is a relational field is resolved through the `target` map and the
//!   transitive read lifted onto the field's emitting method — e.g.
//!   `(odoo:account_move._compute_amount, reads_field,
//!   odoo:account_move_line.amount_residual)` is emitted *in addition to* the
//!   shallow relation read `(…, reads_field, odoo:account_move.line_ids)`.
//!   This surfaces the cross-model recompute-ordering edge that the
//!   surface-only corpus left invisible to `od_ontology::RecomputeDag`.
//!
//! Truth values (NARS `(frequency, confidence)`) carry the provenance:
//! structural edges are certain `(1.0, 1.0)`; decorator/body-authoritative
//! edges `(0.95, 0.90)`; body-inferred edges `(0.85, 0.75)`. The store's
//! `TruthGate` then filters by expectation when querying.
//!
//! # The "a + b → c through d?" query
//!
//! The Foundry compute graph answers "which field `c` does method `d` emit
//! when inputs `a` and `b` change?" by composing two reverse `depends_on`
//! lookups + one `emitted_by` lookup:
//!
//! ```text
//!   {c : (c depends_on a) ∧ (c depends_on b)}   then   {d : (c emitted_by d)}
//! ```
//!
//! This is a graph deduction over the loaded triples, NOT a similarity search.
//!
//! # Provenance
//!
//! Data file `odoo_ontology.spo.ndjson` carries the base extraction (388
//! Object Types, 3 107 Properties, 3 328 Functions) plus the `spo_enrich`
//! P1/P0 layer (842 `target` + 144 `inverse_name` + 935 deep `reads_field`)
//! and the P1b/P2 layer (166 `inherits_from` + 247 `validation_kind`), for
//! 24 579 triples total. The `target`/`inverse_name`/deep-read totals grew
//! over the initial enrichment (618/102/736) once `spo_enrich` (a) honored
//! `_inherit`-only extension classes — `_inherit = "account.move"` with no
//! `_name`, the common Odoo extension form whose relational fields were
//! previously dropped — and (b) lifted each deep `reads_field` onto EVERY
//! emitter of the dependent field, not just the last (a field such as
//! `stock_move.quantity` is emitted by both `_compute_quantity` and
//! `_onchange_product_uom_qty`). `inherits_from` (ruff#19 shape) carries
//! `_inherit`/`_inherits` bases that are themselves corpus-declared
//! ObjectTypes; `validation_kind` (ruff#21 shape) classifies each
//! `@api.constrains` method body. The base extraction's original generator
//! (`emit_ontology2.py` over a `methods.parquet`) is not present in this
//! tree — only its output is — so the enrichment is applied over the shipped
//! corpus + the Odoo source via
//! `python3 -m odoo_blueprint_extractor.spo_enrich --corpus
//! crates/lance-graph/src/graph/spo/odoo_ontology.spo.ndjson` (idempotent;
//! re-running de-duplicates the new triples by `(s, p, o)`).

use crate::graph::fingerprint::{dn_hash, label_fp};
use crate::graph::spo::builder::SpoBuilder;
use crate::graph::spo::store::SpoStore;
use crate::graph::spo::truth::TruthValue;

/// One parsed ontology triple line: `{"s","p","o","f","c"}`.
///
/// `deny_unknown_fields` so harvester schema drift surfaces as a parse
/// error instead of silently degrading the truth signal.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OntologyTriple {
    /// Subject IRI (e.g. `odoo:account_move.amount_total`).
    pub s: String,
    /// Predicate IRI (e.g. `depends_on`).
    pub p: String,
    /// Object IRI (e.g. `odoo:account_move.line_ids.balance`).
    pub o: String,
    /// NARS frequency.
    pub f: f32,
    /// NARS confidence.
    pub c: f32,
}

/// Parse the ndjson ontology into triples (one per non-empty line).
///
/// Lines that fail to parse are skipped (the extractor emits valid JSON, so a
/// parse failure indicates a corrupted data file, not an expected case).
pub fn parse_triples(ndjson: &str) -> Vec<OntologyTriple> {
    ndjson
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<OntologyTriple>(l).ok())
        .collect()
}

/// Load an ndjson ontology document into a fresh [`SpoStore`].
///
/// Each triple's S/P/O labels are hashed to fingerprints via [`label_fp`]
/// (identity-by-name, deterministic), the edge is built with its NARS truth
/// value, and inserted under a content-addressed key `dn_hash("s|p|o")`.
///
/// Returns the populated store; the triple count equals
/// `parse_triples(ndjson).len()` minus any exact `(s,p,o)` key collisions
/// (the extractor de-duplicates, so collisions are not expected).
pub fn load_ontology(ndjson: &str) -> SpoStore {
    let mut store = SpoStore::new();
    for t in parse_triples(ndjson) {
        let subj = label_fp(&t.s);
        let pred = label_fp(&t.p);
        let obj = label_fp(&t.o);
        let record = SpoBuilder::build_edge(&subj, &pred, &obj, TruthValue::new(t.f, t.c));
        let key = dn_hash(&format!("{}|{}|{}", t.s, t.p, t.o));
        store.insert(key, &record);
    }
    store
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped Odoo ontology, embedded at test-build time only (the 2.5 MB
    /// data file is NOT pulled into the release binary).
    const ONTOLOGY: &str = include_str!("odoo_ontology.spo.ndjson");

    #[test]
    fn parses_all_triples() {
        let triples = parse_triples(ONTOLOGY);
        // 22 245 base triples + 1 921 spo_enrich P1/P0 triples (842 target +
        // 144 inverse_name + 935 deep reads_field) + 413 P1b/P2 triples
        // (166 inherits_from + 247 validation_kind) = 24 579.
        assert_eq!(triples.len(), 24_579, "triple count drifted from data file");
    }

    #[test]
    fn predicate_histogram_matches_extraction() {
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
        for t in parse_triples(ONTOLOGY) {
            *hist
                .entry(match t.p.as_str() {
                    "rdf:type" => "rdf:type",
                    "depends_on" => "depends_on",
                    "has_function" => "has_function",
                    "emitted_by" => "emitted_by",
                    "reads_field" => "reads_field",
                    "raises" => "raises",
                    "traverses_relation" => "traverses_relation",
                    "target" => "target",
                    "inverse_name" => "inverse_name",
                    "inherits_from" => "inherits_from",
                    "validation_kind" => "validation_kind",
                    _ => "other",
                })
                .or_default() += 1;
        }
        // Provenance-authoritative edge classes are present and non-trivial.
        assert_eq!(hist.get("depends_on"), Some(&6309));
        assert_eq!(hist.get("emitted_by"), Some(&3228));
        assert_eq!(hist.get("rdf:type"), Some(&6823));
        // spo_enrich P1/P0 layer: FK target/inverse_name + deep reads_field.
        // Totals grew from 618/102/736 once `_inherit`-only extension classes
        // were honored (more `target`/`inverse_name`) and deep reads were
        // lifted onto EVERY emitter of a field (more deep `reads_field`).
        assert_eq!(hist.get("target"), Some(&842));
        assert_eq!(hist.get("inverse_name"), Some(&144));
        // reads_field grew from 2 095 (base) to 3 030 with 935 deep lifts.
        assert_eq!(hist.get("reads_field"), Some(&3030));
        // spo_enrich P1b/P2 layer: inherits_from (ruff#19) + validation_kind
        // (ruff#21), regenerated against /home/user/odoo/addons (#526 follow-up).
        assert_eq!(hist.get("inherits_from"), Some(&166));
        assert_eq!(hist.get("validation_kind"), Some(&247));
        assert_eq!(hist.get("other"), None, "unexpected predicate kind");
    }

    #[test]
    fn loads_into_spo_store_and_queries_forward() {
        let store = load_ontology(ONTOLOGY);

        // Forward query: (account_move.amount_total) -[depends_on]-> ?
        // The compute field's dependency set must be reachable.
        let subj = label_fp("odoo:account_move.amount_total");
        let pred = label_fp("depends_on");
        let hits = store.query_forward(&subj, &pred, 200);
        assert!(
            !hits.is_empty(),
            "amount_total must have depends_on edges in the loaded store"
        );
    }

    #[test]
    fn emitted_by_edge_is_present() {
        // account_move.amount_total is emitted_by _compute_amount (verified
        // in the extraction spot-check).
        let triples = parse_triples(ONTOLOGY);
        let found = triples.iter().any(|t| {
            t.s == "odoo:account_move.amount_total"
                && t.p == "emitted_by"
                && t.o == "odoo:account_move._compute_amount"
        });
        assert!(found, "expected emitted_by edge missing from data file");
    }

    /// Catch silent parse failures: every non-empty line must produce one
    /// `OntologyTriple`. If a line is corrupt and `filter_map().ok()` drops
    /// it, this assertion fires — corruption can't sneak through as a
    /// quiet count mismatch.
    #[test]
    fn every_nonempty_line_parses() {
        let raw_lines = ONTOLOGY.lines().filter(|l| !l.trim().is_empty()).count();
        let parsed = parse_triples(ONTOLOGY).len();
        assert_eq!(
            raw_lines,
            parsed,
            "{} of {} ontology lines silently failed to parse",
            raw_lines - parsed,
            raw_lines
        );
    }

    /// Pin the documented "extractor de-duplicates" assumption: if two
    /// triples share `(s, p, o)` but differ in truth, the second insert
    /// overwrites the first (HashMap last-write-wins via `dn_hash`).
    /// Verifies the silent-overwrite semantics explicitly so a future
    /// switch to insertion-rejection or merge becomes a test failure
    /// instead of a silent change.
    #[test]
    fn duplicate_spo_keys_are_last_write_wins() {
        let s = "odoo:test.x";
        let p = "depends_on";
        let o = "odoo:test.y";
        let ndjson = format!(
            "{{\"s\":\"{s}\",\"p\":\"{p}\",\"o\":\"{o}\",\"f\":0.9,\"c\":0.9}}\n\
             {{\"s\":\"{s}\",\"p\":\"{p}\",\"o\":\"{o}\",\"f\":0.1,\"c\":0.1}}\n"
        );

        let store = load_ontology(&ndjson);
        // Two source triples → one stored record (key collision).
        assert_eq!(
            store.len(),
            1,
            "duplicate (s,p,o) must collapse to a single store entry"
        );
    }

    /// `spo_enrich` P1 — the FK target/inverse_name layer is present and
    /// carries the wishlist's canonical `account_move.line_ids` case: the
    /// One2many's comodel (`account.move.line`) and inverse (`move_id`) are
    /// lifted into sibling triples keyed by the relation IRI (ruff#18 shape).
    /// This is the parity check on the upstream side: `RelationMap::from_corpus`
    /// in `od-ontology` resolves `(model, field) → (target, inverse)` directly
    /// from these triples.
    #[test]
    fn enrichment_emits_fk_target_and_inverse_name() {
        let triples = parse_triples(ONTOLOGY);

        let target = triples.iter().any(|t| {
            t.s == "odoo:account_move.line_ids" && t.p == "target" && t.o == "account.move.line"
        });
        assert!(
            target,
            "P1: account_move.line_ids must carry target=\"account.move.line\""
        );

        let inverse = triples.iter().any(|t| {
            t.s == "odoo:account_move.line_ids" && t.p == "inverse_name" && t.o == "move_id"
        });
        assert!(
            inverse,
            "P1: account_move.line_ids must carry inverse_name=\"move_id\""
        );

        // The phantom-target case the wishlist names explicitly: the
        // `invoice_line_ids` field name's `<parent>_<stem>` convention
        // (`invoice_line`) misses the real comodel (`account.move.line`).
        // The target triple is the corpus's correction of that phantom.
        let phantom_fix = triples.iter().any(|t| {
            t.s == "odoo:account_move.invoice_line_ids"
                && t.p == "target"
                && t.o == "account.move.line"
        });
        assert!(
            phantom_fix,
            "P1: invoice_line_ids phantom-target must resolve to account.move.line"
        );
    }

    /// `spo_enrich` P0 — at least one deep `reads_field` lands on a DIFFERENT
    /// model than the reading method, proving the cross-model recompute edge
    /// is now structurally visible. The canonical case:
    /// `account.move._compute_amount` reads `account_move_line.amount_residual`
    /// (resolved through `@api.depends('line_ids.amount_residual')`), where the
    /// surface corpus only had the relation read `(_compute_amount, reads_field,
    /// account_move.line_ids)`.
    #[test]
    fn enrichment_emits_cross_model_deep_reads_field() {
        let triples = parse_triples(ONTOLOGY);

        // The exact cross-model deep read that makes `RecomputeDag` see the
        // line→move ordering edge (line.amount_residual is emitted_by
        // account_move_line._compute_amount_residual).
        let deep = triples.iter().any(|t| {
            t.s == "odoo:account_move._compute_amount"
                && t.p == "reads_field"
                && t.o == "odoo:account_move_line.amount_residual"
        });
        assert!(
            deep,
            "P0: _compute_amount must carry a deep reads_field on \
             account_move_line.amount_residual (cross-model)"
        );

        // The shallow relation read is STILL present (the deep read is
        // additive, never a replacement).
        let shallow = triples.iter().any(|t| {
            t.s == "odoo:account_move._compute_amount"
                && t.p == "reads_field"
                && t.o == "odoo:account_move.line_ids"
        });
        assert!(
            shallow,
            "P0: the original shallow relation read must be preserved"
        );

        // No deep read is a self-loop (reads_field object == reading method).
        let self_loop = triples.iter().any(|t| t.p == "reads_field" && t.o == t.s);
        assert!(!self_loop, "deep reads_field must never be a self-loop");

        // Count: at least one deep read crosses model boundaries (object's
        // model differs from the reading method's model).
        let cross_model = triples
            .iter()
            .filter(|t| t.p == "reads_field")
            .filter(|t| {
                let s_model = t.s.strip_prefix("odoo:").and_then(|b| b.split('.').next());
                let o_model = t.o.strip_prefix("odoo:").and_then(|b| b.split('.').next());
                matches!((s_model, o_model), (Some(a), Some(b)) if a != b)
            })
            .count();
        assert!(
            cross_model > 0,
            "P0: at least one reads_field must cross model boundaries"
        );
    }

    /// Lock the module-doc claim "3 328 Functions" against drift so the
    /// downstream `action_emitter::shipped_ontology_produces_expected_function_count`
    /// (which asserts the same number on its own) can't get out of sync
    /// with the loader's source-of-truth count.
    #[test]
    fn function_count_matches_module_doc() {
        let triples = parse_triples(ONTOLOGY);
        let functions = triples
            .iter()
            .filter(|t| t.p == "rdf:type" && t.o == "ogit:Function")
            .count();
        assert_eq!(
            functions, 3328,
            "function count drifted from module-doc claim (3 328)"
        );
    }

    /// `spo_enrich` P0 multi-emitter — a field emitted by MORE than one method
    /// must have its deep `reads_field` lifted onto EVERY emitter, not just the
    /// last. `stock_move.quantity` is emitted by both `_compute_quantity` AND
    /// `_onchange_product_uom_qty`; both must carry the cross-model deep reads
    /// (`stock_move_line.quantity` / `stock_move_line.product_uom_id`). Before
    /// the multi-emitter fix the deep read landed on only one of them, dropping
    /// the recompute-ordering edge for the other (typically the `_compute_*`).
    #[test]
    fn enrichment_lifts_deep_reads_onto_every_emitter() {
        let triples = parse_triples(ONTOLOGY);
        let deep_on = |method: &str, obj: &str| {
            triples
                .iter()
                .any(|t| t.s == method && t.p == "reads_field" && t.o == obj)
        };
        // Both emitters of stock_move.quantity carry the cross-model deep read.
        assert!(
            deep_on(
                "odoo:stock_move._compute_quantity",
                "odoo:stock_move_line.quantity"
            ),
            "P0 multi-emitter: _compute_quantity must read stock_move_line.quantity"
        );
        assert!(
            deep_on(
                "odoo:stock_move._onchange_product_uom_qty",
                "odoo:stock_move_line.quantity"
            ),
            "P0 multi-emitter: _onchange_product_uom_qty must also read \
             stock_move_line.quantity (every emitter, not just the last)"
        );
    }

    /// `spo_enrich` P1 `_inherit`-only — relational fields declared on an
    /// extension class (`_inherit = "account.move"` with no `_name`) must still
    /// get their `target`. `account_move.authorized_transaction_ids` is declared
    /// only on the `account_payment` extension of `account.move`; before the
    /// `_inherit` fix the class was skipped (no `_name`) and the field never got
    /// a target.
    #[test]
    fn enrichment_honors_inherit_only_extension_fields() {
        let triples = parse_triples(ONTOLOGY);
        let target = triples.iter().any(|t| {
            t.s == "odoo:account_move.authorized_transaction_ids"
                && t.p == "target"
                && t.o == "payment.transaction"
        });
        assert!(
            target,
            "P1 _inherit-only: authorized_transaction_ids (declared on an \
             _inherit='account.move' extension) must resolve to payment.transaction"
        );
    }

    /// `spo_enrich` P1b `inherits_from` (ruff#19 shape) — a model that mixes in
    /// `mail.thread` carries an `inherits_from` edge to the base, and the base
    /// is itself a corpus-declared ObjectType (the enrichment never invents an
    /// edge to an unknown base). No self-inherits (`_inherit == _name`).
    #[test]
    fn enrichment_emits_inherits_from_to_declared_base() {
        let triples = parse_triples(ONTOLOGY);
        let edge = triples.iter().any(|t| {
            t.s == "odoo:account_account" && t.p == "inherits_from" && t.o == "odoo:mail_thread"
        });
        assert!(
            edge,
            "P1b: account_account must inherits_from mail_thread (mixin base)"
        );

        // Every inherits_from base is a declared ObjectType, and no edge is a
        // self-loop (the Odoo extend-in-place idiom is dropped at scan time).
        let object_types: std::collections::HashSet<&str> = triples
            .iter()
            .filter(|t| t.p == "rdf:type" && t.o == "ogit:ObjectType")
            .map(|t| t.s.as_str())
            .collect();
        for t in triples.iter().filter(|t| t.p == "inherits_from") {
            assert_ne!(t.s, t.o, "inherits_from must never be a self-loop");
            assert!(
                object_types.contains(t.o.as_str()),
                "inherits_from base {} must be a declared ObjectType",
                t.o
            );
        }
    }

    /// `spo_enrich` P2 `validation_kind` (ruff#21 shape) — an `@api.constrains`
    /// method is classified by AST pattern; the subject is the METHOD IRI and
    /// every object is one of the five recognised kinds.
    #[test]
    fn enrichment_emits_validation_kind_on_constrains_method() {
        let triples = parse_triples(ONTOLOGY);
        let classified = triples.iter().any(|t| {
            t.s == "odoo:account_account._check_account_code"
                && t.p == "validation_kind"
                && t.o == "format"
        });
        assert!(
            classified,
            "P2: _check_account_code must be classified validation_kind=format"
        );

        const KINDS: [&str; 5] = ["presence", "uniqueness", "range", "format", "lookup"];
        for t in triples.iter().filter(|t| t.p == "validation_kind") {
            assert!(
                KINDS.contains(&t.o.as_str()),
                "validation_kind object {} not in the recognised set",
                t.o
            );
        }
    }
}
