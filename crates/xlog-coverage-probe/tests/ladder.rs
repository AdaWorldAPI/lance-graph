//! The evidence ladder, end to end through `reconcile`.

use xlog_coverage_probe::{compare, reconcile, Compare, ProgramRecord, Rung, DEFAULT_SEEDS};
use xlog_cube::CubeExecutor;
use xlog_logic::Compiler;

fn rows(v: &[&[i128]]) -> Vec<Vec<i128>> {
    v.iter().map(|r| r.to_vec()).collect()
}

#[test]
fn equal_sets_agree_whatever_the_order_and_duplicates() {
    let cube = rows(&[&[3, 4], &[1, 2], &[1, 2]]);
    let oracle = rows(&[&[1, 2], &[3, 4]]);
    assert_eq!(compare(&cube, &oracle), Compare::Agree);
    assert_eq!(compare(&[], &[]), Compare::Agree);
}

#[test]
fn an_extra_row_on_each_side_is_named() {
    let cube = rows(&[&[1, 2], &[5, 6]]);
    let oracle = rows(&[&[1, 2], &[7, 8]]);
    assert_eq!(
        compare(&cube, &oracle),
        Compare::Disagree {
            cube_only: rows(&[&[5, 6]]),
            oracle_only: rows(&[&[7, 8]]),
        }
    );
    // One side empty is a disagreement, not agreement.
    assert!(matches!(
        compare(&rows(&[&[1, 2]]), &[]),
        Compare::Disagree { .. }
    ));
}

fn assert_all_agree(rec: &ProgramRecord, expected_rules: usize, host_loop: bool) {
    assert_eq!(rec.program_rung, None, "{rec:?}");
    assert_eq!(rec.rules.len(), expected_rules, "{rec:?}");
    for r in &rec.rules {
        assert_eq!(r.rung, Rung::OracleAgrees, "{r:?}");
        assert_eq!(r.host_loop, host_loop, "{r:?}");
        assert!(r.reason.is_empty(), "{r:?}");
    }
}

/// Program text verbatim from xlog-cube/tests/plans.rs (`PROGRAMS[0]`).
const CHAIN: &str = "q(X,Z) :- r(X,Y), s(Y,Z).";

#[test]
fn a_chain_join_agrees_with_the_reference_evaluator() {
    let rec = reconcile(CHAIN, &DEFAULT_SEEDS);
    assert_all_agree(&rec, 1, false);
}

/// Anti-vacuity for the ground-fact path: the same cube, driven here with the
/// program's own facts, produces rows, so the agreement above is not two empty
/// answers agreeing.
#[test]
fn a_chain_join_over_stated_facts_agrees_and_is_not_empty() {
    let src = "r(1, 2). r(2, 3). s(2, 4). s(3, 5). q(X,Z) :- r(X,Y), s(Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_all_agree(&rec, 1, false);

    let mut c = Compiler::new();
    let plan = c.compile(src).expect("compile");
    let mut ex = CubeExecutor::new();
    for (name, id) in c.rel_ids() {
        ex.register_relation(*id, name);
    }
    ex.put_relation("r", &[(1, 2), (2, 3)]);
    ex.put_relation("s", &[(2, 4), (3, 5)]);
    let report = ex.execute_plan(&plan).expect("plan");
    assert!(report.refused.is_empty(), "{report:?}");
    assert_eq!(ex.relation("q"), Some(vec![vec![1, 4], vec![2, 5]]));
}

/// The transitive closure of plans.rs (`tc(X,Y) :- r(X,Y).` first) is not
/// here: its one-atom base rule is not eligible for the reference evaluator
/// (see `a_one_atom_rule_has_no_oracle`). This closure starts from a
/// two-atom rule, so every rule of the SCC has an oracle.
#[test]
fn a_recursive_closure_agrees_and_is_a_host_loop() {
    let src = "tc(X,Z) :- r(X,Y), r(Y,Z). tc(X,Z) :- tc(X,Y), r(Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_all_agree(&rec, 2, true);
}

/// XLOG's reference evaluator refuses a rule with fewer than two positive
/// body atoms (`InsufficientPositiveAtoms`), so plans.rs's own closure has no
/// answer from it. The independent naive oracle does answer, and agrees. The
/// cross-check column keeps the reference evaluator's silence visible.
#[test]
fn a_one_atom_rule_is_judged_by_the_naive_oracle_alone() {
    let src = "path(X,Y) :- edge(X,Y). path(X,Z) :- path(X,Y), edge(Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_eq!(rec.program_rung, None, "{rec:?}");
    assert_eq!(rec.rules.len(), 2, "{rec:?}");
    for r in &rec.rules {
        assert_eq!(r.rung, Rung::OracleAgrees, "{r:?}");
        assert!(r.host_loop, "{r:?}");
        assert_eq!(r.oracle, "naive", "{r:?}");
        assert_eq!(r.xlog_ref, "unavailable", "{r:?}");
    }
}

/// Where both oracles answer, both agree, and the naive one decides.
#[test]
fn both_oracles_agree_on_the_chain_join() {
    let rec = reconcile(CHAIN, &DEFAULT_SEEDS);
    assert_eq!(rec.rules.len(), 1, "{rec:?}");
    assert_eq!(rec.rules[0].oracle, "naive", "{rec:?}");
    assert_eq!(rec.rules[0].xlog_ref, "agree", "{rec:?}");
}

#[test]
fn a_literal_that_does_not_compile_is_observed() {
    let rec = reconcile("this is not xlog", &DEFAULT_SEEDS);
    assert_eq!(rec.program_rung, Some(Rung::Observed), "{rec:?}");
    assert!(rec.rules.is_empty());
    assert!(!rec.program_reason.is_empty());
    assert!(!rec.program_reason.contains('\n'), "first line only");
}

#[test]
fn a_ternary_base_relation_reaches_no_agreement() {
    let rec = reconcile("q(X,Z) :- t(X,Y,Z), r(Z,W).", &DEFAULT_SEEDS);
    assert!(
        rec.rules.iter().all(|r| r.rung != Rung::OracleAgrees),
        "{rec:?}"
    );
    assert!(
        rec.rules.iter().any(|r| !r.reason.is_empty()),
        "a rule with a reason is expected: {rec:?}"
    );
}

#[test]
fn without_seeds_no_rule_is_verified() {
    let rec = reconcile(CHAIN, &[]);
    assert_eq!(rec.rules.len(), 1, "{rec:?}");
    assert_eq!(rec.rules[0].rung, Rung::LoweredUnverified, "{rec:?}");
    assert_eq!(rec.rules[0].reason, "no seeds");
}

/// Facts stated for a derived predicate are part of its answer (Datalog,
/// and XLOG's own `test_same_head_union_batching.rs` literal 2 expects them).
/// The oracle starts the predicate from its facts; the cube is handed them as
/// any host would, and drops them: it answers only the derived row. Pinned
/// as a disagreement naming exactly the dropped fact, so a cube that keeps
/// the facts turns this red and the pin is re-taken.
#[test]
fn a_derived_predicate_with_facts_is_judged_and_the_cube_drops_the_facts() {
    // r and s carry facts too, so XLOG types every column alike and the
    // program compiles (without them `q(1, 2)` types U32 against U64).
    let src = "r(1, 2). s(2, 3). q(1, 2). q(X,Z) :- r(X,Y), s(Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_eq!(rec.program_rung, None, "{rec:?}");
    assert_eq!(rec.rules.len(), 1, "{rec:?}");
    let q = &rec.rules[0];
    assert_eq!(q.rung, Rung::OracleDisagrees, "{q:?}");
    assert_eq!(q.oracle, "naive", "{q:?}");
    assert!(
        q.reason.ends_with("cube_only=[] oracle_only=[[1, 2]]"),
        "{q:?}"
    );
}

#[test]
fn the_ladder_is_ordered_lowest_to_highest() {
    let ladder = [
        Rung::Observed,
        Rung::Compiled,
        Rung::Refused,
        Rung::RefusedOnData,
        Rung::LoweredUnverified,
        Rung::OracleDisagrees,
        Rung::OracleAgrees,
    ];
    assert!(ladder.windows(2).all(|w| w[0] < w[1]));
    let labels: Vec<&str> = ladder.iter().map(|r| r.label()).collect();
    assert_eq!(
        labels,
        [
            "observed",
            "compiled",
            "refused",
            "refused-on-data",
            "lowered-unverified",
            "oracle-disagrees",
            "oracle-agrees"
        ]
    );
}

/// Agreement over answers that are empty on every seed is not evidence: the
/// rule stays lowered-unverified and the reason says so. The facts make the
/// join empty on every seed (no `s` row starts where an `r` row ends).
#[test]
fn an_agreement_over_only_empty_answers_is_not_oracle_agrees() {
    let src = "r(1, 2). s(3, 4). q(X,Z) :- r(X,Y), s(Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_eq!(rec.rules.len(), 1, "{rec:?}");
    let r = &rec.rules[0];
    assert_eq!(r.rung, Rung::LoweredUnverified, "{r:?}");
    assert_eq!(r.nonempty_seeds, 0, "{r:?}");
    assert!(r.reason.contains("empty answers"), "{r:?}");
}

/// A recursive SCC with a ternary member (shape from
/// test_cross_mode_determinism.rs literal 1). Over empty bases the seed delta
/// is empty, the loop ends before a ternary row is stored, and admission
/// reports the rules as ran; over non-empty bases the cube refuses them. The
/// census must not count them as admitted.
#[test]
fn a_rule_refused_only_over_data_is_refused_on_data() {
    let src = "tri(X,Y,Z) :- e(X,Y), e(Y,Z), e(Z,X). echo(X,Y,Z) :- tri(X,Y,Z). tri(X,Y,Z) :- echo(X,Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert!(!rec.rules.is_empty(), "{rec:?}");
    for r in &rec.rules {
        assert_eq!(r.rung, Rung::RefusedOnData, "{r:?}");
        assert!(r.reason.contains("non-empty bases"), "{r:?}");
    }
}

/// A ternary base relation makes only the rules that read it unjudgeable;
/// a rule of the same program over binary bases gets a verdict (shape of
/// real_world_tests.rs `role_perm`, where `has_role` never reads it).
#[test]
fn a_rule_that_does_not_read_a_ternary_base_is_judged() {
    let src = "t(1, 2, 3). e(1, 2). e(2, 3). p(X,Z) :- e(X,Y), e(Y,Z). q(X,Z) :- t(X,Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    let p = rec.rules.iter().find(|r| r.head == "p").expect("p");
    assert_eq!(p.rung, Rung::OracleAgrees, "{rec:?}");
    assert!(p.nonempty_seeds > 0, "{p:?}");
    let q = rec.rules.iter().find(|r| r.head == "q").expect("q");
    assert_ne!(q.rung, Rung::OracleAgrees, "{q:?}");
}

/// A query `?- p(X,Y).` compiles to the generated rule
/// `__xlog_query_0(X,Y) :- p(X,Y)`; the oracle builds the same rule from the
/// AST query, so the generated head is judged like any other.
#[test]
fn a_generated_query_head_is_judged() {
    let src = "e(1, 2). e(2, 3). p(X,Z) :- e(X,Y), e(Y,Z). ?- p(X, Y).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    let q = rec
        .rules
        .iter()
        .find(|r| r.head == "__xlog_query_0")
        .expect("generated head");
    assert_eq!(q.rung, Rung::OracleAgrees, "{rec:?}");
    assert!(q.nonempty_seeds > 0, "{q:?}");
}

/// The dense seed over 0..6 gives this 7-variable path 6^7 = 279,936 rows,
/// over the naive oracle's bound. That seed is skipped for this head, not
/// counted as a verdict, and the small dense seed (0..4, 4^7 = 16,384 rows)
/// still gives a non-empty answer to compare.
#[test]
fn a_seed_over_the_oracle_bound_is_skipped_not_fatal() {
    let src = "k(A,B,C,D,E,F,G) :- e(A,B), e(B,C), e(C,D), e(D,E), e(E,F), e(F,G).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_eq!(rec.rules.len(), 1, "{rec:?}");
    let k = &rec.rules[0];
    assert_eq!(k.rung, Rung::OracleAgrees, "{k:?}");
    assert!(k.nonempty_seeds > 0, "{k:?}");
    assert!(k.reason.is_empty(), "{k:?}");
}

/// A base relation the naive oracle cannot read (string facts) blocks only
/// the rules that read it; an unrelated rule of the same program is judged.
#[test]
fn an_unreadable_base_blocks_only_its_readers() {
    let src = "w(\"a\", \"b\"). e(1, 2). e(2, 3). p(X,Z) :- e(X,Y), e(Y,Z). v(X,Y) :- w(X,Y).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_eq!(rec.program_rung, None, "{rec:?}");
    let p = rec.rules.iter().find(|r| r.head == "p").expect("p");
    assert_eq!(p.rung, Rung::OracleAgrees, "{rec:?}");
    let v = rec.rules.iter().find(|r| r.head == "v").expect("v");
    assert_ne!(v.rung, Rung::OracleAgrees, "{v:?}");
}
