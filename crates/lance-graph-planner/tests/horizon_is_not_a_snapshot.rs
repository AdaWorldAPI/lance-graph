//! D-NFR-1 amendment: a pinned temporal horizon is not a physical snapshot.
//!
//! `QueryReference::at(v, rung)` says which rows a reader may treat as
//! contemporary (`row_version <= v`). It names no dataset, owner or table:
//! `at` sets `server_id` to 0 and the struct has no source field. Which bytes
//! "version v" means is decided where a reader opens a dataset and calls
//! `checkout_version(v)` (e.g. `lance-graph/src/graph/versioned.rs`), a
//! separate step this type does not record.

use lance_graph_planner::temporal::{classify, QueryReference, TemporalStatus};

/// Contemporary-row mask of one source under one reference.
fn admitted(rows: &[(u64, u64)], q: &QueryReference) -> u64 {
    rows.iter()
        .enumerate()
        .fold(0u64, |m, (i, &(row_v, knowable))| {
            m | (u64::from(classify(row_v, knowable, q) == TemporalStatus::Contemporary) << i)
        })
}

/// FAILS IF two sources under the same pinned reference give the same answer
/// (then the reference alone would suffice), or if the two references differ.
#[test]
fn one_pinned_reference_two_sources_two_answers() {
    // (row_version, knowable_from) per row. Expected masks by hand: source A
    // admits rows 0,1,3 (versions <= 5); source B admits rows 0,2.
    let src_a = [(1, 0), (5, 0), (6, 0), (2, 0)];
    let src_b = [(3, 0), (7, 0), (4, 0), (9, 0)];
    let qa = QueryReference::at(5, 0);
    let qb = QueryReference::at(5, 0);
    assert_eq!(qa, qb, "premise: one reference value for both reads");
    assert_eq!(admitted(&src_a, &qa), 0b1011);
    assert_eq!(admitted(&src_b, &qb), 0b0101);
    assert_ne!(admitted(&src_a, &qa), admitted(&src_b, &qb));
}

/// The reference carries no source identity: every field is horizon policy.
/// FAILS IF a source/owner field is added (then this test should be replaced
/// by one that checks it).
#[test]
fn the_reference_has_no_source_identity() {
    let QueryReference {
        server_id,
        ref_version,
        hlc_tick,
        mode: _,
        rung,
    } = QueryReference::at(5, 2);
    assert_eq!((server_id, ref_version, hlc_tick, rung), (0, 5, None, 2));
}
