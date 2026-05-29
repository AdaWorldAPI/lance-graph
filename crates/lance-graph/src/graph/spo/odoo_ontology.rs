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
//! Data file `odoo_ontology.spo.ndjson` is regenerable:
//! `python3 .claude/odoo/emit_ontology2.py` over `.claude/odoo/methods.parquet`.
//! 22 245 triples, 388 Object Types, 3 107 Properties, 3 328 Functions.

use crate::graph::fingerprint::{dn_hash, label_fp};
use crate::graph::spo::builder::SpoBuilder;
use crate::graph::spo::store::SpoStore;
use crate::graph::spo::truth::TruthValue;

/// One parsed ontology triple line: `{"s","p","o","f","c"}`.
#[derive(Debug, Clone, serde::Deserialize)]
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
        // 22 245 triples per the emit_ontology2.py run (2026-05-28).
        assert_eq!(triples.len(), 22_245, "triple count drifted from data file");
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
                    _ => "other",
                })
                .or_default() += 1;
        }
        // Provenance-authoritative edge classes are present and non-trivial.
        assert_eq!(hist.get("depends_on"), Some(&6309));
        assert_eq!(hist.get("emitted_by"), Some(&3228));
        assert_eq!(hist.get("rdf:type"), Some(&6823));
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
}
