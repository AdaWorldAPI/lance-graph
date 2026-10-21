//! G2: genuine nondeterminism is refused, never encoded as a guarded choice.

use std::collections::BTreeSet;

use gkat_probe::oracle::{deterministic, explore};
use gkat_probe::rubicon;
use gkat_probe::{Action, Atoms, BExp, Exp, Refusal};

#[test]
fn unguarded_choice_is_refused() {
    assert_eq!(
        rubicon::mutant_unguarded_choice().lower().unwrap_err(),
        Refusal::UnguardedChoice
    );
}

#[test]
fn goto_and_goback_are_refused() {
    assert_eq!(
        rubicon::mutant_goto().lower().unwrap_err(),
        Refusal::UnstructuredJump("GoTo")
    );
    assert_eq!(
        rubicon::mutant_goback().lower().unwrap_err(),
        Refusal::UnstructuredJump("GoBack")
    );
}

/// The union "do p or do q, whatever the atom" violates the determinacy
/// property, so no GKAT expression denotes it. Encoding it as `if c then p
/// else q` under an invented test `c` changes the language: it adds a test
/// the flow never declared.
#[test]
fn a_union_of_two_actions_is_not_deterministic() {
    let atoms = Atoms::new(1, &BExp::True).unwrap();
    let p = explore(&Exp::Do(Action("p")), &atoms, 1).accepted;
    let q = explore(&Exp::Do(Action("q")), &atoms, 1).accepted;
    let union: BTreeSet<_> = p.union(&q).cloned().collect();
    assert!(deterministic(&p) && deterministic(&q));
    assert!(!deterministic(&union));
}

/// Every lowered Rubicon fixture is deterministic (can-stay-silent half of the
/// check above).
#[test]
fn admitted_flows_are_deterministic() {
    for flow in [rubicon::reference(), rubicon::optimized()] {
        let (e, atoms) = flow.lower().unwrap();
        assert!(deterministic(&explore(&e, &atoms, 2).accepted));
    }
}
