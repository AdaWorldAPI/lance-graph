//! G1: guarded factoring (paper §2.3, Example 2.5).
//!
//! `if b then {p; q} else {q}  ≡  (if b then {p} else {skip}); q`

use gkat_probe::oracle::same_up_to;
use gkat_probe::{Action, Atoms, BExp, Exp, equivalent};

const P: Action = Action("p");
const Q: Action = Action("q");
const R: Action = Action("r");

fn b() -> BExp {
    BExp::Test(0)
}

fn atoms() -> Atoms {
    Atoms::new(1, &BExp::True).unwrap()
}

fn lhs() -> Exp {
    Exp::if_(b(), Exp::Do(P).seq(Exp::Do(Q)), Exp::Do(Q))
}

fn rhs(trailing: Action, guard: BExp) -> Exp {
    Exp::if_(guard, Exp::Do(P), Exp::Skip).seq(Exp::Do(trailing))
}

#[test]
fn factoring_is_equivalent_and_the_oracle_agrees() {
    let v = equivalent(&lhs(), &rhs(Q, b()), &atoms()).unwrap();
    assert!(v.holds(), "{v:?}");
    for n in 0..=4 {
        assert!(same_up_to(&lhs(), &rhs(Q, b()), &atoms(), n), "bound {n}");
    }
}

#[test]
fn a_changed_action_label_is_rejected() {
    let v = equivalent(&lhs(), &rhs(R, b()), &atoms()).unwrap();
    assert!(!v.holds());
    assert!(!same_up_to(&lhs(), &rhs(R, b()), &atoms(), 2));
}

#[test]
fn a_reversed_guard_is_rejected() {
    let v = equivalent(&lhs(), &rhs(Q, b().negated()), &atoms()).unwrap();
    assert!(!v.holds());
    assert!(!same_up_to(&lhs(), &rhs(Q, b().negated()), &atoms(), 2));
}

/// The factoring needs the same trailing action in both branches; with
/// different trailing actions the two sides are not equivalent.
#[test]
fn factoring_different_tails_is_rejected() {
    let lhs = Exp::if_(b(), Exp::Do(P).seq(Exp::Do(Q)), Exp::Do(R));
    let v = equivalent(&lhs, &rhs(Q, b()), &atoms()).unwrap();
    assert!(!v.holds());
}
