//! The Rubicon fixture: equivalence and safety are separate checks, and each
//! disable run fails the check it names.

use gkat_probe::flow::flows_equivalent;
use gkat_probe::oracle::{same_up_to, unsafe_within};
use gkat_probe::rubicon::{self, REQUEST_COMMIT};
use gkat_probe::{Refusal, commit_safe};

/// Oracle bound: two actions covers Deliberate or one Suspend, followed by
/// the decision. With 36 atoms that is 36^3 runs per flow.
const BOUND: usize = 2;

fn safe(flow: &gkat_probe::flow::Flow) -> bool {
    let (e, atoms) = flow.lower().unwrap();
    let auth = rubicon::authorized().unwrap();
    let automaton = commit_safe(&e, &atoms, REQUEST_COMMIT, &auth)
        .unwrap()
        .holds();
    let oracle_unsafe = unsafe_within(&e, &atoms, BOUND, REQUEST_COMMIT, &auth);
    // The oracle only sees runs up to the bound: an oracle violation must be
    // a real one, so it can never coexist with an automaton "safe".
    assert!(
        !(automaton && oracle_unsafe),
        "oracle found a violation the automaton missed"
    );
    automaton
}

fn equivalent(a: &gkat_probe::flow::Flow, b: &gkat_probe::flow::Flow) -> bool {
    let verdict = flows_equivalent(a, b).unwrap().holds();
    let (ea, atoms) = a.lower().unwrap();
    let (eb, _) = b.lower().unwrap();
    let oracle_same = same_up_to(&ea, &eb, &atoms, BOUND);
    // Equivalent ⇒ equal up to every bound; a bounded difference ⇒ different.
    if verdict {
        assert!(
            oracle_same,
            "automaton equivalent, oracle found a difference"
        );
    }
    verdict
}

#[test]
fn the_signature_declares_36_atoms() {
    assert_eq!(rubicon::signature().atoms().unwrap().len(), 36);
}

#[test]
fn reference_and_optimized_are_equivalent_and_both_safe() {
    assert!(equivalent(&rubicon::reference(), &rubicon::optimized()));
    assert!(safe(&rubicon::reference()));
    assert!(safe(&rubicon::optimized()));
}

#[test]
fn disable_1_reversed_polarity_fails_equivalence() {
    assert!(!equivalent(
        &rubicon::reference(),
        &rubicon::mutant_reversed_polarity()
    ));
}

#[test]
fn disable_2_commit_before_guard_fails_equivalence_and_safety() {
    let m = rubicon::mutant_commit_before_guard();
    assert!(!equivalent(&rubicon::reference(), &m));
    assert!(!safe(&m));
}

#[test]
fn disable_2b_stale_authorization_fails_safety() {
    assert!(!safe(&rubicon::mutant_stale_authorization()));
}

#[test]
fn disable_3_unguarded_choice_is_refused() {
    let m = rubicon::mutant_unguarded_choice();
    assert_eq!(m.lower().unwrap_err(), Refusal::UnguardedChoice);
    assert_eq!(
        flows_equivalent(&rubicon::reference(), &m).unwrap_err(),
        Refusal::UnguardedChoice
    );
}

/// Commit authorization must never inherit CHECK semantics.
#[test]
fn check_projection_for_commit_is_unsafe() {
    assert!(!safe(&rubicon::reference_with_check_auth()));
}

/// Equivalence does not establish safety: a flow equivalent to an unsafe
/// reference is unsafe.
#[test]
fn equivalent_to_an_unsafe_reference_is_still_unsafe() {
    let r = rubicon::reference_with_check_auth();
    let o = rubicon::optimized_with_check_auth();
    assert!(equivalent(&r, &o));
    assert!(!safe(&o));
}

/// The safety counterexample is a concrete atom on which an `U` revision or
/// authorization reaches `RequestCommit`.
#[test]
fn the_check_counterexample_has_an_unknown() {
    let (e, atoms) = rubicon::reference_with_check_auth().lower().unwrap();
    let auth = rubicon::authorized().unwrap();
    match commit_safe(&e, &atoms, REQUEST_COMMIT, &auth).unwrap() {
        gkat_probe::Safety::Unsafe { atom, .. } => {
            // rev is tests 0 (T) and 1 (F); mul is tests 2 and 3. UNKNOWN is
            // neither bit.
            let unknown = |t: u8| !atom.holds(t) && !atom.holds(t + 1);
            assert!(unknown(0) || unknown(2), "{atom:?}");
        }
        gkat_probe::Safety::Safe => panic!("expected unsafe"),
    }
}

#[test]
fn flows_under_different_signatures_are_refused() {
    let mut other = rubicon::optimized();
    other.signature.actions.pop();
    assert_eq!(
        flows_equivalent(&rubicon::reference(), &other).unwrap_err(),
        Refusal::DeclarationMismatch("signature")
    );
}

#[test]
fn undeclared_projections_are_refused() {
    use gkat_probe::flow::Cond;
    let sig = rubicon::signature();
    assert!(matches!(
        sig.guard(&Cond::Is(rubicon::REV)),
        Err(Refusal::UndeclaredProjection {
            projection: "Is",
            ..
        })
    ));
    assert!(matches!(
        sig.guard(&Cond::Where(rubicon::INPUT_READY)),
        Err(Refusal::UndeclaredProjection {
            projection: "Where",
            ..
        })
    ));
    assert!(matches!(
        sig.guard(&Cond::Is("nope")),
        Err(Refusal::UnknownObservable("nope"))
    ));
}

/// A K3 observable is encoded like Quack's `Filter::sql_where` verdict: two
/// tests, `T` (the rows where it is TRUE) and `F` (where it is FALSE), never
/// both. UNKNOWN is neither, so presence is `T ∨ F` and there is no separate
/// UNKNOWN bit. WHERE keeps `T`; CHECK keeps `¬F`.
#[test]
fn a_k3_observable_is_quacks_t_f_pair() {
    use gkat_probe::Atom;
    use gkat_probe::flow::Cond;
    let sig = rubicon::signature();
    let atoms = sig.atoms().unwrap();
    // rev occupies tests 0 (T) and 1 (F).
    assert!(atoms.iter().all(|a| !(a.holds(0) && a.holds(1))));
    let r#where = sig.guard(&Cond::Where(rubicon::REV)).unwrap();
    let check = sig.guard(&Cond::Check(rubicon::REV)).unwrap();
    let (t, f, u) = (Atom(0b01), Atom(0b10), Atom(0b00));
    assert_eq!(
        (r#where.eval(t), r#where.eval(f), r#where.eval(u)),
        (true, false, false)
    );
    assert_eq!(
        (check.eval(t), check.eval(f), check.eval(u)),
        (true, false, true)
    );
}
