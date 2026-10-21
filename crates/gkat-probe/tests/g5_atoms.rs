//! G5: the atom set is explicitly bounded and declared.

use gkat_probe::automaton::thompson;
use gkat_probe::flow::{Kind, Signature};
use gkat_probe::{Action, Atoms, BExp, Exp, MAX_TESTS, Refusal, equivalent};

#[test]
fn atom_counts_are_exact() {
    for k in 0..=MAX_TESTS as usize {
        assert_eq!(Atoms::new(k, &BExp::True).unwrap().len(), 1 << k);
    }
    // A K3 observable occupies two tests and has three values.
    let sig = |n: usize| Signature {
        observables: (0..n).map(|_| ("x", Kind::K3)).collect(),
        actions: vec![],
    };
    for n in 0..=4 {
        assert_eq!(sig(n).atoms().unwrap().len(), 3usize.pow(n as u32));
    }
}

#[test]
fn more_tests_than_the_bound_are_refused() {
    let k = MAX_TESTS as usize + 1;
    assert_eq!(
        Atoms::new(k, &BExp::True).unwrap_err(),
        Refusal::AtomBound {
            tests: k,
            max: MAX_TESTS
        }
    );
    // A DR64 register is not 64 GKAT tests: declaring 64 Boolean
    // observables is refused, not enumerated.
    let dr64 = Signature {
        observables: (0..64).map(|_| ("lane", Kind::Bool)).collect(),
        actions: vec![],
    };
    assert_eq!(
        dr64.atoms().unwrap_err(),
        Refusal::AtomBound {
            tests: 64,
            max: MAX_TESTS
        }
    );
    // Five K3 observables need ten tests.
    let k3 = Signature {
        observables: (0..5).map(|_| ("x", Kind::K3)).collect(),
        actions: vec![],
    };
    assert_eq!(
        k3.atoms().unwrap_err(),
        Refusal::AtomBound {
            tests: 10,
            max: MAX_TESTS
        }
    );
}

#[test]
fn an_undeclared_test_is_refused() {
    let atoms = Atoms::new(2, &BExp::True).unwrap();
    let e = Exp::Assert(BExp::Test(5));
    assert_eq!(
        equivalent(&e, &e, &atoms).unwrap_err(),
        Refusal::UndeclaredTest {
            test: 5,
            declared: 2
        }
    );
}

/// A chain of `k` independent guards: automaton states grow with the
/// expression (one state per action), the atom set grows as `2^k`. The
/// counts are what G5 measures; no timing is reported.
#[test]
fn states_grow_with_the_expression_atoms_with_the_tests() {
    let mut rows = Vec::new();
    for k in 1..=MAX_TESTS {
        let atoms = Atoms::new(k as usize, &BExp::True).unwrap();
        let chain = (0..k)
            .map(|t| Exp::if_(BExp::Test(t), Exp::Do(Action("p")), Exp::Do(Action("q"))))
            .reduce(Exp::seq)
            .unwrap();
        let states = thompson(&chain, &atoms).unwrap().states.len();
        let pairs = match equivalent(&chain, &chain.clone(), &atoms).unwrap() {
            gkat_probe::Equivalence::Equivalent { pairs } => pairs,
            other => panic!("{other:?}"),
        };
        rows.push((k, atoms.len(), states, pairs));
        assert_eq!(states, 2 * k as usize);
    }
    eprintln!("tests atoms states merged-pairs: {rows:?}");
}
