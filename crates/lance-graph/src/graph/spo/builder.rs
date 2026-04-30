// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Builder for SPO edge records.
//!
//! An SPO record packs Subject, Predicate, Object fingerprints together
//! with a truth value into a structure that can be stored in an SpoStore
//! and queried via ANN search.
//!
//! The builder has two operational modes:
//!
//! 1. **Stateless** (legacy / happy path) — call associated functions
//!    such as [`SpoBuilder::build_edge`] to produce a single
//!    [`SpoRecord`] for direct insertion into an [`SpoStore`].
//! 2. **Stateful + schema-validated** (TD-INT-8) — `SpoBuilder::new()`
//!    returns an instance that accumulates `(predicate_name, key, record)`
//!    triples via [`SpoBuilder::stage`]. When attached to a
//!    [`Schema`](lance_graph_contract::property::Schema) via
//!    [`SpoBuilder::with_schema`], the staged set is validated against
//!    the schema's Required predicates on
//!    [`SpoBuilder::commit_validated`]. Missing predicates produce a
//!    [`FailureTicket`](lance_graph_contract::grammar::FailureTicket)
//!    instead of a silent insertion. The predicate fingerprint is a
//!    hash that loses the original label, so each `stage` call also
//!    carries the predicate name explicitly so validation can compare
//!    against the schema.

use lance_graph_contract::grammar::FailureTicket;
use lance_graph_contract::property::Schema;

use crate::graph::fingerprint::{Fingerprint, FINGERPRINT_WORDS};
use crate::graph::sparse::{pack_axes, Bitmap, BITMAP_WORDS};

use super::store::SpoStore;
use super::truth::TruthValue;

/// An SPO record representing a single edge in the graph.
///
/// Contains the packed search vector (for ANN queries) and the individual
/// components (for result interpretation).
#[derive(Debug, Clone)]
pub struct SpoRecord {
    /// Subject fingerprint.
    pub subject: Fingerprint,
    /// Predicate fingerprint.
    pub predicate: Fingerprint,
    /// Object fingerprint.
    pub object: Fingerprint,
    /// Packed bitmap: S|P|O for ANN similarity search.
    pub packed: Bitmap,
    /// Truth value of this edge.
    pub truth: TruthValue,
}

/// A staged triple held by an `SpoBuilder` between `stage` and
/// `commit_validated`. Carries the predicate name (since the fingerprint
/// loses the label after hashing) and the destination key in the store.
#[derive(Debug, Clone)]
struct StagedTriple {
    predicate_name: &'static str,
    key: u64,
    record: SpoRecord,
}

/// Builder for constructing SPO edge records.
///
/// Has two modes — see the module-level docs.
#[derive(Default)]
pub struct SpoBuilder {
    schema: Option<Schema>,
    staged: Vec<StagedTriple>,
}

impl SpoBuilder {
    /// Create a fresh builder with no schema and an empty staging area.
    pub fn new() -> Self {
        Self {
            schema: None,
            staged: Vec::new(),
        }
    }

    /// Attach a [`Schema`] to validate Required predicates against on
    /// [`Self::commit_validated`]. Without this, validation is a no-op.
    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Stage a triple for later validated commit. The `predicate_name`
    /// is the original predicate label (must match a name in the
    /// attached schema for validation to recognize it). `key` is where
    /// the record will be inserted in the store when the batch passes
    /// validation.
    pub fn stage(
        &mut self,
        predicate_name: &'static str,
        key: u64,
        record: SpoRecord,
    ) -> &mut Self {
        self.staged.push(StagedTriple {
            predicate_name,
            key,
            record,
        });
        self
    }

    /// Return the list of Required predicates from the attached schema
    /// that are NOT present in the currently staged triples. Empty when
    /// (a) no schema is attached, or (b) all Required predicates are
    /// staged.
    pub fn validate(&self) -> Vec<&'static str> {
        let Some(schema) = self.schema.as_ref() else {
            return Vec::new();
        };
        let present: Vec<&str> = self.staged.iter().map(|t| t.predicate_name).collect();
        schema.validate(&present)
    }

    /// Validate against the attached schema and, if valid, insert all
    /// staged triples into `store`. Returns the number of inserted
    /// records on success.
    ///
    /// On failure (one or more Required predicates missing) returns a
    /// [`FailureTicket`] carrying the missing predicate names; nothing
    /// is inserted and the staged set is preserved for retry.
    pub fn commit_validated(&mut self, store: &mut SpoStore) -> Result<usize, FailureTicket> {
        let missing = self.validate();
        if !missing.is_empty() {
            return Err(FailureTicket::missing_required(missing));
        }
        let n = self.staged.len();
        for staged in self.staged.drain(..) {
            store.insert(staged.key, &staged.record);
        }
        Ok(n)
    }

    /// Number of currently-staged triples (mostly for tests / inspection).
    pub fn staged_len(&self) -> usize {
        self.staged.len()
    }

    /// Build an edge record from S, P, O fingerprints and a truth value.
    ///
    /// The packed bitmap is the OR of all three fingerprints, used as
    /// the search vector for ANN queries in Lance.
    pub fn build_edge(
        subject: &Fingerprint,
        predicate: &Fingerprint,
        object: &Fingerprint,
        truth: TruthValue,
    ) -> SpoRecord {
        // Ensure sizes match (compile-time guarantee via type aliases,
        // but assert at runtime for safety during development).
        debug_assert_eq!(FINGERPRINT_WORDS, BITMAP_WORDS);

        let packed = pack_axes(subject, predicate, object);

        SpoRecord {
            subject: *subject,
            predicate: *predicate,
            object: *object,
            packed,
            truth,
        }
    }

    /// Build a forward query vector: S|P (looking for O).
    ///
    /// For SxP2O queries: given Subject and Predicate, find Object.
    pub fn build_forward_query(subject: &Fingerprint, predicate: &Fingerprint) -> Bitmap {
        let zero = [0u64; BITMAP_WORDS];
        pack_axes(subject, predicate, &zero)
    }

    /// Build a reverse query vector: P|O (looking for S).
    ///
    /// For PxO2S queries: given Predicate and Object, find Subject.
    pub fn build_reverse_query(predicate: &Fingerprint, object: &Fingerprint) -> Bitmap {
        let zero = [0u64; BITMAP_WORDS];
        pack_axes(&zero, predicate, object)
    }

    /// Build a relation query vector: S|O (looking for P).
    ///
    /// For SxO2P queries: given Subject and Object, find Predicate.
    pub fn build_relation_query(subject: &Fingerprint, object: &Fingerprint) -> Bitmap {
        let zero = [0u64; BITMAP_WORDS];
        pack_axes(subject, &zero, object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::fingerprint::label_fp;

    #[test]
    fn test_build_edge() {
        let s = label_fp("Jan");
        let p = label_fp("KNOWS");
        let o = label_fp("Ada");
        let record = SpoBuilder::build_edge(&s, &p, &o, TruthValue::new(0.9, 0.8));

        assert_eq!(record.subject, s);
        assert_eq!(record.predicate, p);
        assert_eq!(record.object, o);
        assert_eq!(record.truth.frequency, 0.9);
        // Packed should be S|P|O
        for i in 0..BITMAP_WORDS {
            assert_eq!(record.packed[i], s[i] | p[i] | o[i]);
        }
    }

    #[test]
    fn test_forward_query_vector() {
        let s = label_fp("Jan");
        let p = label_fp("KNOWS");
        let query = SpoBuilder::build_forward_query(&s, &p);
        // Should contain bits from both S and P
        for i in 0..BITMAP_WORDS {
            assert_eq!(query[i], s[i] | p[i]);
        }
    }

    // ── TD-INT-8: Schema-validated commit path ────────────────────────────

    use crate::graph::fingerprint::dn_hash;
    use crate::graph::spo::store::SpoStore;
    use lance_graph_contract::property::Schema;

    fn record_for(predicate_name: &str) -> SpoRecord {
        let s = label_fp("Customer:42");
        let p = label_fp(predicate_name);
        let o = label_fp(&format!("value:{}", predicate_name));
        SpoBuilder::build_edge(&s, &p, &o, TruthValue::new(0.9, 0.8))
    }

    #[test]
    fn validated_commit_without_schema_inserts_unchanged() {
        // Without a schema, validate() returns empty and commit_validated
        // behaves like a plain insert.
        let mut store = SpoStore::new();
        let mut builder = SpoBuilder::new();
        builder.stage("customer_name", dn_hash("k1"), record_for("customer_name"));

        assert!(builder.validate().is_empty());
        let n = builder
            .commit_validated(&mut store)
            .expect("no schema → no validation failure");
        assert_eq!(n, 1);
        assert_eq!(store.len(), 1);
        assert_eq!(builder.staged_len(), 0);
    }

    #[test]
    fn validated_commit_with_complete_schema_succeeds() {
        let schema = Schema::builder("Customer")
            .required("customer_name")
            .required("tax_id")
            .optional("address")
            .build();

        let mut store = SpoStore::new();
        let mut builder = SpoBuilder::new().with_schema(schema);

        builder
            .stage(
                "customer_name",
                dn_hash("c:name"),
                record_for("customer_name"),
            )
            .stage("tax_id", dn_hash("c:tax"), record_for("tax_id"))
            .stage("address", dn_hash("c:addr"), record_for("address"));

        assert!(builder.validate().is_empty());
        let n = builder
            .commit_validated(&mut store)
            .expect("all Required predicates present");
        assert_eq!(n, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn validated_commit_missing_required_returns_failure_ticket() {
        let schema = Schema::builder("Customer")
            .required("customer_name")
            .required("tax_id") // ← intentionally not staged
            .optional("address")
            .build();

        let mut store = SpoStore::new();
        let mut builder = SpoBuilder::new().with_schema(schema);

        builder
            .stage(
                "customer_name",
                dn_hash("c:name"),
                record_for("customer_name"),
            )
            .stage("address", dn_hash("c:addr"), record_for("address"));

        let missing = builder.validate();
        assert_eq!(missing, vec!["tax_id"]);

        let err = builder
            .commit_validated(&mut store)
            .expect_err("tax_id missing → FailureTicket");

        // Store untouched, staging preserved for retry.
        assert_eq!(store.len(), 0);
        assert_eq!(builder.staged_len(), 2);

        // Ticket carries the missing predicate names.
        let m: Vec<&'static str> = err.missing_predicates().collect();
        assert_eq!(m, vec!["tax_id"]);
    }

    #[test]
    fn validated_commit_retry_after_filling_missing() {
        let schema = Schema::builder("Customer")
            .required("customer_name")
            .required("tax_id")
            .build();

        let mut store = SpoStore::new();
        let mut builder = SpoBuilder::new().with_schema(schema);

        builder.stage(
            "customer_name",
            dn_hash("c:name"),
            record_for("customer_name"),
        );
        assert!(builder.commit_validated(&mut store).is_err());

        // Caller addresses the failure by adding the missing predicate.
        builder.stage("tax_id", dn_hash("c:tax"), record_for("tax_id"));
        let n = builder.commit_validated(&mut store).expect("now valid");
        assert_eq!(n, 2);
        assert_eq!(store.len(), 2);
    }
}
