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

/// The reference evaluator refuses a rule with fewer than two positive body
/// atoms (`InsufficientPositiveAtoms`), and one refused rule leaves its whole
/// SCC without an oracle. plans.rs's own closure is therefore
/// lowered-unverified, not agreeing.
#[test]
fn a_one_atom_rule_has_no_oracle() {
    let src = "path(X,Y) :- edge(X,Y). path(X,Z) :- path(X,Y), edge(Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_eq!(rec.program_rung, None, "{rec:?}");
    assert_eq!(rec.rules.len(), 2, "{rec:?}");
    for r in &rec.rules {
        assert_eq!(r.rung, Rung::LoweredUnverified, "{r:?}");
        assert!(r.host_loop, "{r:?}");
        assert!(r.reason.contains("InsufficientPositiveAtoms"), "{r:?}");
    }
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

/// The cube ignores facts stated for a derived predicate, so the oracle
/// cannot be seeded with them: no verdict, and the reason says why.
#[test]
fn a_derived_predicate_with_facts_has_no_oracle() {
    // r and s carry facts too, so XLOG types every column alike and the
    // program compiles (without them `q(1, 2)` types U32 against U64).
    let src = "r(1, 2). s(2, 3). q(1, 2). q(X,Z) :- r(X,Y), s(Y,Z).";
    let rec = reconcile(src, &DEFAULT_SEEDS);
    assert_eq!(rec.program_rung, None, "{rec:?}");
    assert!(!rec.rules.is_empty(), "{rec:?}");
    for r in &rec.rules {
        assert_ne!(r.rung, Rung::OracleAgrees, "{r:?}");
        assert!(r.reason.contains("ground facts"), "{r:?}");
    }
}

#[test]
fn the_ladder_is_ordered_lowest_to_highest() {
    let ladder = [
        Rung::Observed,
        Rung::Compiled,
        Rung::Refused,
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
            "lowered-unverified",
            "oracle-disagrees",
            "oracle-agrees"
        ]
    );
}
