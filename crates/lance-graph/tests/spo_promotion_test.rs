// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration tests for `arigraph::spo_bridge::promote_to_spo`
//! (closes ledger row SPO-1; D-ONTO-V5-2).
//!
//! Verifies the warm L1 (`Triplet`) → cold L2 (`SpoStore`) round-trip:
//! a hand-built AriGraph triplet is promoted to fingerprint-keyed
//! storage and read back via the SPO query API. The architectural
//! contract under test is: the L1/L2 cache pair preserves
//! subject/relation/object identity through the
//! `label_fp` projection, the NARS truth value, and the `dn_hash`
//! key derivation.

use lance_graph::graph::arigraph::spo_bridge::{
    canonical_dn, promote_graph_to_spo, promote_to_spo, PromoteError, PromoteGate,
};
use lance_graph::graph::arigraph::triplet_graph::{Triplet, TripletGraph};
use lance_graph::graph::fingerprint::{dn_hash, label_fp};
use lance_graph::graph::spo::store::SpoStore;
use lance_graph::graph::spo::truth::TruthValue;

/// Step 1: build one AriGraph triplet by hand. Step 2: call
/// `promote_to_spo`. Step 3: read back from SPO and assert the
/// triplet round-trips through the fingerprint projection.
#[test]
fn spo_promotion_round_trips_a_single_triplet() {
    // 1. Hand-built warm triplet (string-keyed L1 representation).
    let triplet = Triplet::with_truth("alice", "bob", "knows", TruthValue::new(0.9, 0.8), 42);

    // 2. Cold store starts empty; promote the warm entry.
    let mut spo = SpoStore::new();
    assert!(spo.is_empty());

    let handle = promote_to_spo(&triplet, PromoteGate::OPEN, &mut spo)
        .expect("OPEN gate accepts non-deleted triplet");
    assert_eq!(spo.len(), 1);

    // The returned key is the FNV-1a of the canonical DN path.
    assert_eq!(handle.key, dn_hash(&canonical_dn(&triplet)));

    // 3. Read back via fingerprint-keyed forward query: the cold
    //    record carries the same subject/relation/object identity
    //    after passing through `label_fp`, and the same NARS truth.
    let alice_fp = label_fp("alice");
    let knows_fp = label_fp("knows");
    let bob_fp = label_fp("bob");

    let hits = spo.query_forward(&alice_fp, &knows_fp, 200);
    assert_eq!(hits.len(), 1, "exactly one promoted record");
    let hit = &hits[0];
    assert_eq!(hit.key, handle.key, "round-trip key matches");
    assert_eq!(hit.record.subject, alice_fp);
    assert_eq!(hit.record.predicate, knows_fp);
    assert_eq!(hit.record.object, bob_fp);
    assert_eq!(hit.record.truth.frequency, 0.9);
    assert_eq!(hit.record.truth.confidence, 0.8);
}

/// Soft-deleted triplets must NOT cross the gate.
#[test]
fn spo_promotion_skips_soft_deleted() {
    let triplet = Triplet::with_truth("x", "y", "r", TruthValue::unknown(), 0);
    assert!(triplet.is_deleted());

    let mut spo = SpoStore::new();
    let err = promote_to_spo(&triplet, PromoteGate::OPEN, &mut spo).unwrap_err();
    assert_eq!(err, PromoteError::Deleted);
    assert!(spo.is_empty());
}

/// The PromoteGate's truth-floor blocks low-confidence triplets.
#[test]
fn spo_promotion_respects_normal_truth_gate() {
    // expectation = 0.5 * (0.5 - 0.5) + 0.5 = 0.5 < 0.6
    let weak = Triplet::with_truth("a", "b", "r", TruthValue::new(0.5, 0.5), 0);
    let mut spo = SpoStore::new();
    assert_eq!(
        promote_to_spo(&weak, PromoteGate::NORMAL, &mut spo).unwrap_err(),
        PromoteError::BelowGate
    );
    assert!(spo.is_empty());
}

/// Batch: graph-level convenience wrapper promotes every eligible
/// triplet and reports an accurate count.
#[test]
fn spo_promotion_batch_promotes_eligible_triplets_only() {
    let mut graph = TripletGraph::new();
    graph.add_triplets(&[
        Triplet::new("alice", "bob", "knows", 1),
        Triplet::new("bob", "carol", "knows", 2),
        // soft-deleted via unknown truth
        Triplet::with_truth("ghost", "void", "knows", TruthValue::unknown(), 3),
    ]);

    let mut spo = SpoStore::new();
    let count = promote_graph_to_spo(&graph, PromoteGate::OPEN, &mut spo);
    assert_eq!(count, 2, "two eligible triplets promoted");
    assert_eq!(spo.len(), 2);
}
