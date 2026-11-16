//! D-XLOG-KNOWLEDGE-G1c: positive witnesses for the 15 refused rules whose
//! naive/Datafrog reference agreement in the G1b census (`census-656def9.tsv`,
//! draft PR #2109) held over empty answers only.
//!
//! The census seeds only binary `u32` base relations, and every one of the
//! four programs below reads a unary or ternary base, so on every seed those
//! bases were empty and both evaluators answered the empty set. Here each
//! program gets a minimal hand-written fixture. Each head's expected rows are
//! derived by hand in the comments, and both evaluators must return exactly
//! that set (set equality, not a count). The cube is never run; the census
//! classification of these rules is re-checked unchanged.
//!
//! Each program also has a negative control: one fixture row removed, and the
//! predicted smaller answer must follow, so the fixture rows are load-bearing
//! and neither evaluator is answering a constant.

use std::collections::BTreeMap;

use xlog_coverage_probe::{compare, reconcile, witness, Compare, Rung, DEFAULT_SEEDS};

type Rows = Vec<Vec<i128>>;

fn rows(v: &[&[i128]]) -> Rows {
    let mut r: Rows = v.iter().map(|r| r.to_vec()).collect();
    r.sort();
    r.dedup();
    r
}

fn fixture(v: &[(&str, &[&[i128]])]) -> BTreeMap<String, Rows> {
    v.iter().map(|(n, r)| (n.to_string(), rows(r))).collect()
}

/// Both evaluators answer exactly `want` for every head in it, and `want` is
/// non-empty (anti-vacuity). Returns nothing; panics with the head on failure.
fn assert_witness(src: &str, base: &BTreeMap<String, Rows>, want: &[(&str, Rows)]) {
    let got = witness(src, base).expect("program compiles");
    for (head, expected) in want {
        assert!(
            !expected.is_empty(),
            "{head}: an empty witness proves nothing"
        );
        let (naive, dfrog) = got
            .get(*head)
            .unwrap_or_else(|| panic!("no answer for {head}"));
        let naive = naive
            .as_ref()
            .unwrap_or_else(|e| panic!("{head}: naive: {e}"));
        let dfrog = dfrog
            .as_ref()
            .unwrap_or_else(|e| panic!("{head}: datafrog: {e}"));
        assert_eq!(naive, expected, "{head}: naive oracle");
        assert_eq!(dfrog, expected, "{head}: datafrog");
    }
}

/// The census classification is unchanged: every rule refused, with the
/// census's reason, in rule order.
fn assert_census(src: &str, reasons: &[(&str, &str)]) {
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_eq!(rec.program_rung, None, "{rec:?}");
    let got: Vec<(&str, Rung, &str)> = rec
        .rules
        .iter()
        .map(|r| (r.head.as_str(), r.rung, r.reason.as_str()))
        .collect();
    let want: Vec<(&str, Rung, &str)> = reasons
        .iter()
        .map(|(h, why)| (*h, Rung::Refused, *why))
        .collect();
    assert_eq!(got, want);
}

// ---- 1. xlog@656def9 crates/xlog-integration/tests/test_external_consumer_surface_preservation.rs, literal 1

const P1: &str = r#"
        pred committed_input(u32).
        pred tensor_support(u32).

        tensor_support(X) :- committed_input(X).

        ?- tensor_support(X).
    "#;

#[test]
fn p1_tensor_support_witness() {
    // committed_input = {1, 2}; tensor_support copies it; the query reads it.
    let base = fixture(&[("committed_input", &[&[1], &[2]])]);
    let want = rows(&[&[1], &[2]]);
    assert_witness(
        P1,
        &base,
        &[("tensor_support", want.clone()), ("__xlog_query_0", want)],
    );
    // Negative control: drop committed_input(2).
    let base = fixture(&[("committed_input", &[&[1]])]);
    let want = rows(&[&[1]]);
    assert_witness(
        P1,
        &base,
        &[("tensor_support", want.clone()), ("__xlog_query_0", want)],
    );
}

#[test]
fn p1_census_classification_is_unchanged() {
    assert_census(
        P1,
        &[
            (
                "tensor_support",
                "`committed_input` is not a binary relation in the cube",
            ),
            ("__xlog_query_0", "reads refused relation `tensor_support`"),
        ],
    );
}

// ---- 2. xlog@656def9 crates/xlog-integration/tests/test_recursive_setop_profile.rs, literal 0

const P2: &str = r#"
pred external_consumer_commit(i64, i64, i64).
pred external_consumer_body_0(i64, i64, i64).
pred external_consumer_body_1(i64, i64, i64).
pred external_consumer_body_len_2(i64).
pred usable(i64, i64, i64).
pred support_2(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).

usable(P, A0, A1) :- external_consumer_commit(P, A0, A1).

support_2(Head, V0, V1, RId, Body0Pred, V0, V2, Body1Pred, V2, V1) :-
    external_consumer_body_len_2(RId),
    external_consumer_body_0(RId, Head, Body0Pred),
    usable(Body0Pred, V0, V2),
    external_consumer_body_1(RId, Head, Body1Pred),
    usable(Body1Pred, V2, V1).

usable(Head, A0, A1) :- support_2(Head, A0, A1, _, _, _, _, _, _, _).

?- support_2(H, A0, A1, R, Body0Pred, Body0Arg0, Body0Arg1, Body1Pred, Body1Arg0, Body1Arg1).
?- usable(P, A0, A1).
"#;

/// Two committed facts of predicate 11 and one of 10. Rule 5 joins 10 and 11
/// into head 20; rule 6 joins 20 (derived by rule 5) and 11 into head 30, so
/// its row exists only after a second round of the recursion.
fn p2_base(with_rule_6: bool) -> BTreeMap<String, Rows> {
    let mut len: Vec<&[i128]> = vec![&[5]];
    let mut b0: Vec<&[i128]> = vec![&[5, 20, 10]];
    let mut b1: Vec<&[i128]> = vec![&[5, 20, 11]];
    if with_rule_6 {
        len.push(&[6]);
        b0.push(&[6, 30, 20]);
        b1.push(&[6, 30, 11]);
    }
    fixture(&[
        (
            "external_consumer_commit",
            &[&[10, 1, 2], &[11, 2, 3], &[11, 3, 4]],
        ),
        ("external_consumer_body_len_2", &len),
        ("external_consumer_body_0", &b0),
        ("external_consumer_body_1", &b1),
    ])
}

#[test]
fn p2_recursive_support_witness() {
    // Round 1: usable = commit = {(10,1,2), (11,2,3), (11,3,4)}.
    //   Rule 5: usable(10,V0,V2) = (1,2); usable(11,2,V1) gives V1 = 3
    //   -> support_2(20,1,3, 5, 10,1,2, 11,2,3); usable(20,1,3).
    // Round 2: rule 6: usable(20,V0,V2) = (1,3); usable(11,3,V1) gives V1 = 4
    //   -> support_2(30,1,4, 6, 20,1,3, 11,3,4); usable(30,1,4).
    let support = rows(&[
        &[20, 1, 3, 5, 10, 1, 2, 11, 2, 3],
        &[30, 1, 4, 6, 20, 1, 3, 11, 3, 4],
    ]);
    let usable = rows(&[
        &[10, 1, 2],
        &[11, 2, 3],
        &[11, 3, 4],
        &[20, 1, 3],
        &[30, 1, 4],
    ]);
    assert_witness(
        P2,
        &p2_base(true),
        &[
            ("support_2", support.clone()),
            ("usable", usable.clone()),
            ("__xlog_query_0", support),
            ("__xlog_query_1", usable),
        ],
    );
}

#[test]
fn p2_negative_control_without_the_second_round() {
    // Without rule 6 the round-2 rows must disappear, and only they.
    let support = rows(&[&[20, 1, 3, 5, 10, 1, 2, 11, 2, 3]]);
    let usable = rows(&[&[10, 1, 2], &[11, 2, 3], &[11, 3, 4], &[20, 1, 3]]);
    assert_witness(
        P2,
        &p2_base(false),
        &[
            ("support_2", support.clone()),
            ("usable", usable.clone()),
            ("__xlog_query_0", support),
            ("__xlog_query_1", usable),
        ],
    );
}

#[test]
fn p2_census_classification_is_unchanged() {
    let rec = reconcile(P2, &DEFAULT_SEEDS);
    let mut got: Vec<(String, Rung, String)> = rec
        .rules
        .iter()
        .map(|r| (r.head.clone(), r.rung, r.reason.clone()))
        .collect();
    got.sort();
    let mut want: Vec<(String, Rung, String)> = [
        ("support_2", "join key outside its input"),
        ("usable", "join key outside its input"),
        ("usable", "join key outside its input"),
        ("__xlog_query_0", "reads refused relation `support_2`"),
        ("__xlog_query_1", "reads refused relation `usable`"),
    ]
    .iter()
    .map(|(h, w)| (h.to_string(), Rung::Refused, w.to_string()))
    .collect();
    want.sort();
    assert_eq!(got, want);
}

// ---- 3. xlog@656def9 crates/xlog-integration/tests/test_sort_labels.rs, literal 0

const P3: &str = r#"
        pred external_consumer_commit(i64, i64, i64).
        pred external_consumer_rule(i64).
        pred support_1(i64, i64, i64, i64, i64, i64, i64).
        pred usable(i64, i64, i64).

        support_1(Head, A0, A1, RId, Body0Pred, Body0Arg0, Body0Arg1) :-
            external_consumer_rule(RId),
            external_consumer_commit(Head, A0, A1),
            external_consumer_commit(Body0Pred, Body0Arg0, Body0Arg1).
        usable(Head, A0, A1) :- support_1(Head, A0, A1, _, _, _, _).

        ?- support_1(Head, A0, A1, RId, Body0Pred, Body0Arg0, Body0Arg1).
        ?- usable(P, A0, A1).
    "#;

#[test]
fn p3_support_cross_product_witness() {
    // rule = {7}; commit = {(1,2,3), (4,5,6)}: support_1 is commit x {7} x
    // commit (4 rows); usable projects its first three columns.
    let base = fixture(&[
        ("external_consumer_rule", &[&[7]]),
        ("external_consumer_commit", &[&[1, 2, 3], &[4, 5, 6]]),
    ]);
    let support = rows(&[
        &[1, 2, 3, 7, 1, 2, 3],
        &[1, 2, 3, 7, 4, 5, 6],
        &[4, 5, 6, 7, 1, 2, 3],
        &[4, 5, 6, 7, 4, 5, 6],
    ]);
    let usable = rows(&[&[1, 2, 3], &[4, 5, 6]]);
    assert_witness(
        P3,
        &base,
        &[
            ("support_1", support.clone()),
            ("usable", usable.clone()),
            ("__xlog_query_0", support),
            ("__xlog_query_1", usable),
        ],
    );
    // Negative control: a second rule id doubles support_1, not usable.
    let base = fixture(&[
        ("external_consumer_rule", &[&[7], &[8]]),
        ("external_consumer_commit", &[&[1, 2, 3], &[4, 5, 6]]),
    ]);
    let got = witness(P3, &base).unwrap();
    let (n, d) = &got["support_1"];
    assert_eq!(n.as_ref().unwrap().len(), 8);
    assert_eq!(n, d);
    assert_witness(P3, &base, &[("usable", rows(&[&[1, 2, 3], &[4, 5, 6]]))]);
}

#[test]
fn p3_census_classification_is_unchanged() {
    assert_census(
        P3,
        &[
            ("support_1", "Project over Project"),
            ("__xlog_query_0", "reads refused relation `support_1`"),
            ("usable", "reads refused relation `support_1`"),
            ("__xlog_query_1", "reads refused relation `usable`"),
        ],
    );
}

// ---- 4. xlog@656def9 crates/xlog-integration/tests/test_widened_frontier_replay.rs, literal 0

const P4: &str = r#"
    pred frontier_pred(u32).
    pred widened_pred(u32).
    pred frontier_edge(u32, u32).
    pred blocked_pred(u32).
    pred promoted(u32).
    pred replay_reachable(u32).
    pred rollback_hit(u32).

    promoted(P) :- frontier_pred(P), widened_pred(P).
    replay_reachable(P) :- promoted(P).
    replay_reachable(Q) :- replay_reachable(P), frontier_edge(P, Q), frontier_pred(Q).
    rollback_hit(P) :- replay_reachable(P), blocked_pred(P).
"#;

fn p4_base(blocked: &[&[i128]]) -> BTreeMap<String, Rows> {
    // A chain 1 -> 2 -> 3 -> 4; 4 is not a frontier predicate, so the replay
    // stops at 3 even though an edge leaves it.
    fixture(&[
        ("frontier_pred", &[&[1], &[2], &[3]]),
        ("widened_pred", &[&[1]]),
        ("frontier_edge", &[&[1, 2], &[2, 3], &[3, 4]]),
        ("blocked_pred", blocked),
    ])
}

#[test]
fn p4_replay_witness() {
    // promoted = frontier ∩ widened = {1}; replay walks 1 -> 2 -> 3 and not
    // to 4; rollback_hit = replay ∩ blocked = {3}.
    assert_witness(
        P4,
        &p4_base(&[&[3]]),
        &[
            ("promoted", rows(&[&[1]])),
            ("replay_reachable", rows(&[&[1], &[2], &[3]])),
            ("rollback_hit", rows(&[&[3]])),
        ],
    );
    // Negative control: block 4 instead, which the replay never reaches.
    let got = witness(P4, &p4_base(&[&[4]])).unwrap();
    let (n, d) = &got["rollback_hit"];
    assert_eq!(n, &Ok(Vec::new()));
    assert_eq!(d, &Ok(Vec::new()));
}

#[test]
fn p4_census_classification_is_unchanged() {
    assert_census(
        P4,
        &[
            (
                "promoted",
                "`frontier_pred` is not a binary relation in the cube",
            ),
            ("replay_reachable", "reads refused relation `promoted`"),
            ("replay_reachable", "reads refused relation `promoted`"),
            ("rollback_hit", "reads refused relation `replay_reachable`"),
        ],
    );
}

// ---- the comparison itself can fail

#[test]
fn a_witness_with_one_extra_row_is_a_disagreement() {
    let got = witness(P4, &p4_base(&[&[3]])).unwrap();
    let (n, _) = &got["replay_reachable"];
    let mut wrong = n.clone().unwrap();
    wrong.push(vec![4]);
    assert!(matches!(
        compare(n.as_ref().unwrap(), &wrong),
        Compare::Disagree { .. }
    ));
}
