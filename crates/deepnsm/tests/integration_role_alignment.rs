//! Integration test asserting markov_bundle and role_keys agree on slice
//! boundaries. THE TEST THAT SHOULD HAVE BLOCKED THE ORIGINAL MERGE.
//!
//! Background: PR #279 review CRITICAL #1 found that `markov_bundle.rs`
//! defined its own equal-partition slice layout (`16384 / 5 = 3277` per
//! SPO role) while `lance_graph_contract::grammar::role_keys` defined a
//! domain-specific layout (SUBJECT = 2000 dims, MODIFIER = 1500, …). Two
//! incompatible coordinate systems on the same VSA carrier → silent
//! corruption. This test now asserts a single source of truth.

use deepnsm::markov_bundle::GrammaticalRole;
use lance_graph_contract::grammar::role_keys::{
    CONTEXT_SLICE, INSTRUMENT_SLICE, KAUSAL_SLICE, LOKAL_SLICE, MODAL_SLICE,
    MODIFIER_SLICE, OBJECT_SLICE, PREDICATE_SLICE, RoleKeySlice, SUBJECT_SLICE,
    TEMPORAL_SLICE,
};

#[test]
fn subject_role_aligns_with_role_keys_subject_slice() {
    let role_slice = GrammaticalRole::Subject.slice();
    assert_eq!(role_slice.start, SUBJECT_SLICE.start);
    assert_eq!(role_slice.stop, SUBJECT_SLICE.stop);
}

#[test]
fn predicate_role_aligns() {
    assert_eq!(GrammaticalRole::Predicate.slice().start, PREDICATE_SLICE.start);
    assert_eq!(GrammaticalRole::Predicate.slice().stop, PREDICATE_SLICE.stop);
}

#[test]
fn object_role_aligns() {
    assert_eq!(GrammaticalRole::Object.slice().start, OBJECT_SLICE.start);
    assert_eq!(GrammaticalRole::Object.slice().stop, OBJECT_SLICE.stop);
}

#[test]
fn modifier_role_aligns() {
    assert_eq!(GrammaticalRole::Modifier.slice().start, MODIFIER_SLICE.start);
    assert_eq!(GrammaticalRole::Modifier.slice().stop, MODIFIER_SLICE.stop);
}

#[test]
fn context_role_aligns() {
    assert_eq!(GrammaticalRole::Context.slice().start, CONTEXT_SLICE.start);
    assert_eq!(GrammaticalRole::Context.slice().stop, CONTEXT_SLICE.stop);
}

#[test]
fn tekamolo_roles_align() {
    // The TEKAMOLO sub-slices (Temporal/Kausal/Modal/Lokal/Instrument) live in
    // the [9000..9750) post-context band per role_keys.rs — NOT inside the
    // Context band as the original (broken) markov_bundle layout claimed.
    assert_eq!(GrammaticalRole::Temporal.slice().start,   TEMPORAL_SLICE.start);
    assert_eq!(GrammaticalRole::Temporal.slice().stop,    TEMPORAL_SLICE.stop);
    assert_eq!(GrammaticalRole::Kausal.slice().start,     KAUSAL_SLICE.start);
    assert_eq!(GrammaticalRole::Kausal.slice().stop,      KAUSAL_SLICE.stop);
    assert_eq!(GrammaticalRole::Modal.slice().start,      MODAL_SLICE.start);
    assert_eq!(GrammaticalRole::Modal.slice().stop,       MODAL_SLICE.stop);
    assert_eq!(GrammaticalRole::Lokal.slice().start,      LOKAL_SLICE.start);
    assert_eq!(GrammaticalRole::Lokal.slice().stop,       LOKAL_SLICE.stop);
    assert_eq!(GrammaticalRole::Instrument.slice().start, INSTRUMENT_SLICE.start);
    assert_eq!(GrammaticalRole::Instrument.slice().stop,  INSTRUMENT_SLICE.stop);
}

#[test]
fn no_overlap_between_major_roles() {
    let mut spans: Vec<(&str, RoleKeySlice)> = vec![
        ("subject", SUBJECT_SLICE),
        ("predicate", PREDICATE_SLICE),
        ("object", OBJECT_SLICE),
        ("modifier", MODIFIER_SLICE),
        ("context", CONTEXT_SLICE),
    ];
    spans.sort_by_key(|(_, s)| s.start);
    for win in spans.windows(2) {
        assert!(
            win[0].1.stop <= win[1].1.start,
            "{} ends at {} but {} starts at {}",
            win[0].0, win[0].1.stop, win[1].0, win[1].1.start
        );
    }
}

#[test]
fn no_overlap_across_all_ten_roles() {
    // Stronger check: every GrammaticalRole variant's slice is disjoint from
    // every other variant's slice (sorted-pairwise via the role_keys layout).
    let mut spans: Vec<(&str, RoleKeySlice)> = vec![
        ("subject",    GrammaticalRole::Subject.slice()),
        ("predicate",  GrammaticalRole::Predicate.slice()),
        ("object",     GrammaticalRole::Object.slice()),
        ("modifier",   GrammaticalRole::Modifier.slice()),
        ("context",    GrammaticalRole::Context.slice()),
        ("temporal",   GrammaticalRole::Temporal.slice()),
        ("kausal",     GrammaticalRole::Kausal.slice()),
        ("modal",      GrammaticalRole::Modal.slice()),
        ("lokal",      GrammaticalRole::Lokal.slice()),
        ("instrument", GrammaticalRole::Instrument.slice()),
    ];
    spans.sort_by_key(|(_, s)| s.start);
    for win in spans.windows(2) {
        assert!(
            win[0].1.stop <= win[1].1.start,
            "{} [{}..{}) overlaps {} [{}..{})",
            win[0].0, win[0].1.start, win[0].1.stop,
            win[1].0, win[1].1.start, win[1].1.stop,
        );
    }
}
