// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Structural Core extension — the typed home for `inherits_from` and
//! `selection_value`.
//!
//! # Why this module exists (Core-first correction, 2026-06-18)
//!
//! Two structural facts about an Odoo model — its `_inherit`/`_inherits`
//! **mixin chain** and a `fields.Selection` field's **allowed value set** —
//! are *Core* properties (identity / composition / value-domain), not
//! behavioural harvest. Per the Core-first transcode doctrine they belong in
//! the deliberate typed Core ([`super::OdooEntity`] / [`super::OdooField`]),
//! the single source of truth, **not** re-inferred onto the flat SPO ndjson
//! by a separate AST pass.
//!
//! They could not be added as *fields* on the existing structs: `OdooField`
//! has **3 554** literal sites and `OdooEntity` **404** across `l1..l15.rs`,
//! none using a constructor — adding a field would break every one. For a
//! literal sea that size the doctrine-correct "extend the Core deliberately"
//! is a **typed side-table**, which is what this module is. It is still Core
//! (`lance-graph-ontology::odoo_blueprint`), still authoritative, still
//! `OdooConfidence::Curated`-grade; it simply lives beside the mega-structs
//! instead of inside them.
//!
//! # Direction of truth: Core → SPO, never the reverse
//!
//! [`project_inherits_from`] and [`project_selection_value`] emit SPO triples
//! **from** this typed Core. The `spo_enrich.py` AST harvest is the
//! **Extracted leg** (per `odoo-extraction-strategies-v1.md`) — a *breadth*
//! feeder for the ~322 ObjectTypes the curated Core has not yet reached. On
//! convergence in the `SpoStore` the curated Core (this module,
//! `0.95/0.90`) **wins** over the harvest's extracted confidence for any
//! model it covers. The harvest never becomes the home for a structural
//! fact; it fills in where the Core is silent.
//!
//! Behavioural predicates (`reads_field` deep lifts, `emitted_by`,
//! transitive `depends_on`) are genuine harvest and stay in `spo_enrich.py`
//! — they describe a method *body*, not the model's structure.

/// One model's `_inherit` / `_inherits` mixin chain — the composition gap
/// the `OdooEntity` mega-struct does not carry (`OdooEntityKind` records the
/// ORM *base class* `Model`/`Transient`/`Abstract`, not the mixin list).
///
/// `bases` are dotted Odoo model names (`"mail.activity.mixin"`); the SPO
/// projection underscores them to match the corpus IRI convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OdooInherits {
    /// Owning model, dotted (`"account.move"`).
    pub model: &'static str,
    /// Mixin bases this model `_inherit`s, in declaration order.
    pub bases: &'static [&'static str],
}

/// One `fields.Selection` field's statically-known value domain — the gap
/// `OdooFieldKind::Selection` flags but `OdooField` does not store (only the
/// `state` field's domain is reachable today, via `OdooStateMachine`).
///
/// `values` are the stored *keys* (the first element of each `('key',
/// 'Label')` tuple), in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OdooFieldSelection {
    /// Owning model, dotted (`"account.move"`).
    pub model: &'static str,
    /// Selection field name (`"state"`, `"move_type"`).
    pub field: &'static str,
    /// Allowed value keys, in source order.
    pub values: &'static [&'static str],
}

// ─── Curated data — canonical `account.move` (grounded, not invented) ──────
//
// `INHERITS` is grounded in the #527-regenerated corpus
// (`grep '"s":"odoo:account_move","p":"inherits_from"'`): the corpus captured
// `mail_activity_mixin` + `sequence_mixin` (it drops bases that are not
// themselves corpus ObjectTypes — `mail.thread` / `portal.mixin` fell outside
// that boundary). The curated Core records the corpus-confirmed pair; a future
// L-doc curation pass may widen it to the full real chain (the Core is allowed
// to exceed the harvest's ObjectType-boundary, since it is authoritative).
//
// `FIELD_SELECTIONS` uses the canonical, stable Odoo `account.move` value sets
// (`state`, `move_type`) — standard across Odoo versions, verifiable against
// `addons/account/models/account_move.py`.

/// Curated inherit-chains. APPEND-ONLY as L-doc curation reaches more models.
pub const INHERITS: &[OdooInherits] = &[
    OdooInherits {
        model: "account.move",
        bases: &["mail.activity.mixin", "sequence.mixin"],
    },
    OdooInherits {
        model: "account.move.line",
        bases: &["analytic.mixin"],
    },
];

/// Curated Selection value domains. APPEND-ONLY.
pub const FIELD_SELECTIONS: &[OdooFieldSelection] = &[
    OdooFieldSelection {
        model: "account.move",
        field: "state",
        values: &["draft", "posted", "cancel"],
    },
    OdooFieldSelection {
        model: "account.move",
        field: "move_type",
        values: &[
            "entry",
            "out_invoice",
            "out_refund",
            "in_invoice",
            "in_refund",
            "out_receipt",
            "in_receipt",
        ],
    },
];

// ─── Core → SPO projection ─────────────────────────────────────────────────

/// `account.move` → `account_move` (corpus IRI local-part convention).
fn underscore(dotted: &str) -> String {
    dotted.replace('.', "_")
}

/// One projected SPO triple: `(subject, predicate, object)`. Truth values are
/// fixed at the curated grade `(0.95, 0.90)` by the projection (declared,
/// authoritative), so callers serialising to ndjson append `,"f":0.95,"c":0.9`.
pub type SpoTriple = (String, &'static str, String);

/// Project the curated inherit-chains into `inherits_from` SPO triples,
/// `(odoo:<model>, inherits_from, odoo:<base>)`. Both endpoints underscored.
///
/// This is the authoritative source for `inherits_from` on every model in
/// [`INHERITS`]; the harvest only supplies models absent here.
#[must_use]
pub fn project_inherits_from(table: &[OdooInherits]) -> Vec<SpoTriple> {
    let mut out = Vec::new();
    for row in table {
        let child = format!("odoo:{}", underscore(row.model));
        for base in row.bases {
            out.push((
                child.clone(),
                "inherits_from",
                format!("odoo:{}", underscore(base)),
            ));
        }
    }
    out
}

/// Project the curated Selection domains into `selection_value` SPO triples,
/// `(odoo:<model>.<field>, selection_value, "<key>")`. One per value, source
/// order preserved.
#[must_use]
pub fn project_selection_value(table: &[OdooFieldSelection]) -> Vec<SpoTriple> {
    let mut out = Vec::new();
    for row in table {
        let subj = format!("odoo:{}.{}", underscore(row.model), row.field);
        for v in row.values {
            out.push((subj.clone(), "selection_value", (*v).to_string()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inherits_projection_underscores_both_endpoints() {
        let triples = project_inherits_from(INHERITS);
        assert!(triples.contains(&(
            "odoo:account_move".to_string(),
            "inherits_from",
            "odoo:mail_activity_mixin".to_string(),
        )));
        assert!(triples.contains(&(
            "odoo:account_move".to_string(),
            "inherits_from",
            "odoo:sequence_mixin".to_string(),
        )));
        // account.move.line → account_move_line, analytic.mixin → analytic_mixin
        assert!(triples.contains(&(
            "odoo:account_move_line".to_string(),
            "inherits_from",
            "odoo:analytic_mixin".to_string(),
        )));
    }

    #[test]
    fn inherits_projection_matches_527_corpus() {
        // The two account.move bases the projection emits are exactly the pair
        // the #527 corpus regen captured — the typed Core is consistent with
        // (not contradicting) the harvest, and is the authoritative home.
        let triples = project_inherits_from(INHERITS);
        let am_bases: Vec<&str> = triples
            .iter()
            .filter(|(s, _, _)| s == "odoo:account_move")
            .map(|(_, _, o)| o.as_str())
            .collect();
        assert_eq!(
            am_bases,
            vec!["odoo:mail_activity_mixin", "odoo:sequence_mixin"]
        );
    }

    #[test]
    fn selection_projection_one_triple_per_value_in_order() {
        let triples = project_selection_value(FIELD_SELECTIONS);
        let state_vals: Vec<&str> = triples
            .iter()
            .filter(|(s, _, _)| s == "odoo:account_move.state")
            .map(|(_, _, o)| o.as_str())
            .collect();
        assert_eq!(state_vals, vec!["draft", "posted", "cancel"]);

        let move_type_vals: Vec<&str> = triples
            .iter()
            .filter(|(s, _, _)| s == "odoo:account_move.move_type")
            .map(|(_, _, o)| o.as_str())
            .collect();
        assert_eq!(move_type_vals[0], "entry");
        assert_eq!(move_type_vals.len(), 7);
    }

    #[test]
    fn selection_subject_is_field_iri_not_model() {
        let triples = project_selection_value(FIELD_SELECTIONS);
        // selection_value keys on the FIELD, never the model.
        assert!(triples.iter().all(|(s, _, _)| s.contains('.')));
    }

    #[test]
    fn registries_are_nonempty_and_curated() {
        assert!(!INHERITS.is_empty());
        assert!(!FIELD_SELECTIONS.is_empty());
    }
}
