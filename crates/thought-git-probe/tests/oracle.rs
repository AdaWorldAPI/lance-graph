//! The oracle's four rules, each pinned on a fixture whose answer differs
//! under the rule's disable (named in each test).

use thought_git_probe::fixtures::{
    self, Outcome, cell, contradiction_thought, divergent_pair, givens, main_version,
    rubicon_outcomes, unremarkable_stack, unwritten_cells,
};
use thought_git_probe::oracle::{Contradiction, next_version, squash};

/// Rule 1. Disable: land every outcome → the Plan, Prune and Veto cells appear.
#[test]
fn only_commit_lands() {
    let g = givens();
    let v = next_version(&g, &main_version(), &rubicon_outcomes());
    assert_eq!(v.read(&g, cell(4, 1)), Some(41), "Commit lands");
    for c in [cell(5, 1), cell(6, 1), cell(7, 1)] {
        assert_eq!(v.read(&g, c), None, "{c:?} must not land");
    }
    assert_eq!(v.cells.len(), 1);
}

/// Rule 2, agreement half. Disable: store agreeing writes → `cells` holds (0, 0).
#[test]
fn agreement_with_a_given_is_not_stored() {
    let g = givens();
    let v = next_version(&g, &main_version(), &unremarkable_stack());
    assert!(
        !v.cells.contains_key(&cell(0, 0)),
        "agreement stored as a value"
    );
    assert_eq!(v.read(&g, cell(0, 0)), Some(100));
    assert!(
        v.contradictions.is_empty(),
        "agreement is not a contradiction"
    );
}

/// Rule 2, refusal half. Disable: let the offer overwrite the given → the read
/// returns 999 and no record exists.
#[test]
fn a_different_value_for_a_given_is_refused_and_recorded() {
    let g = givens();
    let t = contradiction_thought();
    let v = next_version(&g, &main_version(), std::slice::from_ref(&t));
    assert_eq!(v.read(&g, cell(1, 0)), Some(101), "given overwritten");
    assert_eq!(
        v.contradictions,
        vec![Contradiction {
            cell: cell(1, 0),
            given: 101,
            offered: 999,
            thought: 30,
        }]
    );
    // The thought's other write still lands: a refusal is per cell.
    assert_eq!(v.read(&g, cell(4, 2)), Some(42));
}

/// Rule 3. Disable: last writer wins → (5, 2) reads 8 and no divergence exists.
#[test]
fn a_divergence_in_one_version_stores_nothing_and_is_recorded() {
    let g = givens();
    let v = next_version(&g, &main_version(), &divergent_pair());
    assert_eq!(v.read(&g, cell(5, 2)), None);
    assert_eq!(v.divergences.len(), 1);
    assert_eq!(v.divergences[0].offers, vec![(7, 31), (8, 32)]);
    // The non-conflicting write of the same thought lands.
    assert_eq!(v.read(&g, cell(6, 2)), Some(62));
}

/// The same two writes in consecutive versions are not a divergence: the
/// later version replaces a value above layer 0. Pins that rule 3 is about
/// one version, not about history.
#[test]
fn consecutive_versions_replace_without_divergence() {
    let g = givens();
    let [a, b] = divergent_pair();
    let v1 = next_version(&g, &main_version(), std::slice::from_ref(&a));
    let v2 = next_version(&g, &v1, std::slice::from_ref(&b));
    assert_eq!(v2.read(&g, cell(5, 2)), Some(8));
    assert!(v2.divergences.is_empty());
}

/// Rule 4. Disable: read a missing cell as 0 → every read below is Some(0).
#[test]
fn an_unwritten_lane_reads_absent_not_zero() {
    let g = givens();
    let mut landing: Vec<_> = rubicon_outcomes().into_iter().collect();
    landing.extend(unremarkable_stack());
    landing.push(contradiction_thought());
    let v = next_version(&g, &main_version(), &landing);
    let unwritten = unwritten_cells();
    let written = fixtures::all_writes();
    assert!(
        unwritten.len() >= 10,
        "fixture must leave a real lane unwritten"
    );
    for c in unwritten {
        assert!(!written.contains_key(&c), "fixture writes {c:?}");
        assert_eq!(v.read(&g, c), None, "{c:?} read as a value");
    }
    // Anti-vacuity: the same version does hold values elsewhere.
    assert!(v.cells.len() >= 5);
}

/// The squash rule the plan relies on: squashing an unremarkable stack and
/// landing it equals landing the stack as it is.
#[test]
fn squashing_an_unremarkable_stack_changes_nothing() {
    let g = givens();
    let stack = unremarkable_stack();
    let separate = next_version(&g, &main_version(), &stack);
    let squashed = next_version(&g, &main_version(), &[squash(99, &stack)]);
    assert_eq!(separate.cells, squashed.cells);
    assert_eq!(separate.contradictions, squashed.contradictions);
    assert!(separate.cells.len() >= 5, "stack must write something");
}

/// ...and why an anomaly must stop the squash: squashing a divergent pair
/// hides the divergence and stores one side. Silence twin of the test above.
#[test]
fn squashing_a_divergent_pair_hides_the_divergence() {
    let g = givens();
    let pair = divergent_pair();
    let separate = next_version(&g, &main_version(), &pair);
    let squashed = next_version(&g, &main_version(), &[squash(99, &pair)]);
    assert_eq!(separate.divergences.len(), 1);
    assert!(squashed.divergences.is_empty());
    assert_eq!(squashed.read(&g, cell(5, 2)), Some(8));
    assert_ne!(separate.cells, squashed.cells);
}

/// Fixture sanity: one thought per outcome, all four present.
#[test]
fn every_outcome_has_a_fixture() {
    let o: Vec<Outcome> = rubicon_outcomes().iter().map(|t| t.outcome).collect();
    assert_eq!(
        o,
        vec![
            Outcome::Commit,
            Outcome::Plan,
            Outcome::Prune,
            Outcome::Veto
        ]
    );
}
