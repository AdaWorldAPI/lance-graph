// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! AriGraph → SPO promotion bridge (warm L1 → cold L2 cache pair).
//!
//! AriGraph and the SPO store are an L1/L2 cache pair (per
//! `.claude/DECISION_SPO_ARIGRAPH.md`):
//!
//! - **L1 (AriGraph `triplet_graph::Triplet`)**: warm, string-keyed,
//!   episodic working memory with cheap lexical recall.
//! - **L2 (`spo::SpoStore`)**: cold, fingerprint-keyed, columnar
//!   persistent store tuned for batch ANN scans and Hamming-min
//!   semiring traversal.
//!
//! This module provides the one-way promotion writer
//! [`promote_to_spo`]. It is intentionally additive: the canonical
//! AriGraph triplet type and the canonical SPO store are not
//! modified. The bridge consumes both as upstream types.
//!
//! Closes ledger row SPO-1 (entropy ledger row 70 + 245). Per the
//! v5 plan D-ONTO-V5-2: warm string-keyed `Triplet`s with
//! sufficient confidence are promoted into cold fingerprint-keyed
//! `SpoRecord`s gated by a [`PromoteGate`].

use crate::graph::fingerprint::{dn_hash, label_fp};
use crate::graph::spo::builder::SpoBuilder;
use crate::graph::spo::store::SpoStore;
use crate::graph::spo::truth::TruthGate;

use super::triplet_graph::Triplet;

/// Handle for a single promoted triplet.
///
/// The `key` is the u64 address `dn_hash` produces from the
/// canonical `"subject -[relation]-> object"` DN path; it is the
/// key under which `SpoStore::insert` placed the record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpoHandle {
    /// The store key (FNV-1a hash of the canonical DN path).
    pub key: u64,
}

/// Errors that can occur during promotion.
#[derive(Debug, Clone, PartialEq)]
pub enum PromoteError {
    /// The triplet is soft-deleted (truth confidence == 0).
    Deleted,
    /// The truth value did not pass the gate.
    BelowGate,
}

/// Convenience [`Result`] alias for promotion operations.
pub type PromoteResult<T> = std::result::Result<T, PromoteError>;

/// Gate controlling which warm triplets are promoted to cold store.
///
/// The gate is the architectural seam that prevents the hot path
/// (cognitive cycles writing AriGraph) from over-flooding the cold
/// path (SPO store under columnar contract). A triplet must
/// (1) not be soft-deleted and (2) pass `truth` to be promoted.
#[derive(Debug, Clone, Copy)]
pub struct PromoteGate {
    /// Minimum NARS truth-expectation required for promotion.
    pub truth: TruthGate,
}

impl PromoteGate {
    /// An open gate: every non-deleted triplet promotes. Useful in
    /// tests and as the default when consumers want round-trip
    /// behavior without truth filtering.
    pub const OPEN: PromoteGate = PromoteGate {
        truth: TruthGate::OPEN,
    };

    /// A normal gate: expectation >= 0.6.
    pub const NORMAL: PromoteGate = PromoteGate {
        truth: TruthGate::NORMAL,
    };
}

impl Default for PromoteGate {
    fn default() -> Self {
        Self::OPEN
    }
}

/// Canonical DN path for a triplet — the string under which the
/// promoted record is keyed in the cold store.
///
/// Exposed so callers and tests can derive the same key without
/// guessing the format.
pub fn canonical_dn(triplet: &Triplet) -> String {
    format!("{} -[{}]-> {}", triplet.subject, triplet.relation, triplet.object)
}

/// Promote a single warm AriGraph triplet into the cold SPO store.
///
/// Maps the triplet's string subject/relation/object through
/// [`label_fp`] into fingerprints, packs them via [`SpoBuilder::build_edge`]
/// into an [`SpoRecord`] (preserving the NARS [`TruthValue`](super::super::spo::truth::TruthValue)),
/// keys it via [`dn_hash`] of the canonical DN path, and
/// inserts into `spo`.
///
/// Returns the [`SpoHandle`] for the inserted record on success;
/// returns [`PromoteError::Deleted`] for soft-deleted triplets and
/// [`PromoteError::BelowGate`] when the gate rejects the truth value.
pub fn promote_to_spo(
    triplet: &Triplet,
    gate: PromoteGate,
    spo: &mut SpoStore,
) -> PromoteResult<SpoHandle> {
    if triplet.is_deleted() {
        return Err(PromoteError::Deleted);
    }
    if !gate.truth.passes(&triplet.truth) {
        return Err(PromoteError::BelowGate);
    }

    let subject_fp = label_fp(&triplet.subject);
    let predicate_fp = label_fp(&triplet.relation);
    let object_fp = label_fp(&triplet.object);

    let record = SpoBuilder::build_edge(&subject_fp, &predicate_fp, &object_fp, triplet.truth);
    let key = dn_hash(&canonical_dn(triplet));

    spo.insert(key, &record);
    Ok(SpoHandle { key })
}

/// Promote every non-deleted, gate-passing triplet from a
/// `TripletGraph` into `spo`. Returns the count successfully promoted.
///
/// Convenience wrapper around [`promote_to_spo`] for batch
/// operations; not strictly required by D-ONTO-V5-2 but the natural
/// next step for ingestion-path consumers (D-CASCADE-V1-9).
pub fn promote_graph_to_spo(
    graph: &super::triplet_graph::TripletGraph,
    gate: PromoteGate,
    spo: &mut SpoStore,
) -> usize {
    let mut promoted = 0usize;
    for triplet in &graph.triplets {
        if promote_to_spo(triplet, gate, spo).is_ok() {
            promoted += 1;
        }
    }
    promoted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::fingerprint::label_fp;
    use crate::graph::spo::truth::TruthValue;

    #[test]
    fn promote_round_trips_one_triplet() {
        let triplet = Triplet::new("alice", "bob", "knows", 1);
        let mut spo = SpoStore::new();

        let handle = promote_to_spo(&triplet, PromoteGate::OPEN, &mut spo).expect("promote ok");
        assert_eq!(spo.len(), 1);

        // Round-trip read: the fingerprint-keyed record matches the
        // string-keyed source.
        let alice_fp = label_fp("alice");
        let knows_fp = label_fp("knows");
        let bob_fp = label_fp("bob");

        let hits = spo.query_forward(&alice_fp, &knows_fp, 200);
        assert!(!hits.is_empty(), "forward query should find promoted edge");
        let hit = &hits[0];
        assert_eq!(hit.record.subject, alice_fp);
        assert_eq!(hit.record.predicate, knows_fp);
        assert_eq!(hit.record.object, bob_fp);
        assert_eq!(hit.key, handle.key);
    }

    #[test]
    fn deleted_triplet_is_rejected() {
        let triplet = Triplet::with_truth("a", "b", "r", TruthValue::unknown(), 1);
        let mut spo = SpoStore::new();
        let err = promote_to_spo(&triplet, PromoteGate::OPEN, &mut spo).unwrap_err();
        assert_eq!(err, PromoteError::Deleted);
        assert_eq!(spo.len(), 0);
    }

    #[test]
    fn gate_filters_low_truth() {
        // Expectation 0.5 fails NORMAL gate (0.6).
        let triplet =
            Triplet::with_truth("a", "b", "r", TruthValue::new(0.5, 0.5), 1);
        let mut spo = SpoStore::new();
        let err = promote_to_spo(&triplet, PromoteGate::NORMAL, &mut spo).unwrap_err();
        assert_eq!(err, PromoteError::BelowGate);
        assert_eq!(spo.len(), 0);
    }
}
