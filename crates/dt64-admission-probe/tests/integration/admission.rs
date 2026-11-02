//! The five outcomes on the D-SAG-6 fixture's functions. Each class has a
//! case that fires and a non-trivial case where it must stay silent.

use dt64_admission_probe::evidence::Relation;
use dt64_admission_probe::fixture::{self, *};
use dt64_admission_probe::{
    Admitted, DoGrant, ExecRefusal, ExecutionPolicy, Request, UnknownCause,
};
use lance_graph_dt64::{MethodRef, Refusal, SigRev};

fn policy() -> ExecutionPolicy {
    ExecutionPolicy {
        trusted_external: vec!["core::slice::".into(), "core::iter::".into()],
    }
}

fn decide<'s>(
    store: &'s lance_graph_dt64::BindStore,
    id: lance_graph_dt64::MethodId,
    request: Request<'_>,
) -> Result<Admitted<'s>, ExecRefusal> {
    let r = method_ref(store, id);
    policy().decide(store, &r, &fixture::span_joined(), request)
}

// ---- binding is not authorization -------------------------------------

#[test]
fn every_fixture_method_admits_but_only_two_execute_pure() {
    let s = store();
    let all = [
        COUNT_BELOW,
        COUNT_WITH_TALLY,
        COUNT_DIV,
        COUNT_GENERIC,
        COUNT_DYN,
        MASK_BELOW,
        BELOW_TEST,
        UNATTACHED,
    ];
    let mut pure = Vec::new();
    for id in all {
        let r = method_ref(&s, id);
        assert!(s.admit(&r).is_ok(), "{id:?} must authenticate");
        if matches!(decide(&s, id, Request::Pure), Ok(Admitted::Pure { .. })) {
            pure.push(id);
        }
    }
    assert_eq!(pure, vec![COUNT_BELOW, BELOW_TEST]);
}

#[test]
fn an_admit_refusal_passes_through_even_with_perfect_evidence() {
    let s = store();
    let mut r = method_ref(&s, COUNT_BELOW);
    r.sig_rev = SigRev(2);
    let got = policy().decide(&s, &r, &fixture::span_joined(), Request::Pure);
    assert!(matches!(
        got,
        Err(ExecRefusal::Binding(
            Refusal::SignatureRevisionMismatch { .. }
        ))
    ));
    let mut r = method_ref(&s, COUNT_BELOW);
    r.guard = None;
    let got = policy().decide(&s, &r, &fixture::span_joined(), Request::Pure);
    assert!(matches!(
        got,
        Err(ExecRefusal::Binding(Refusal::GuardDropped { .. }))
    ));
}

// ---- 1. proven pure ---------------------------------------------------

#[test]
fn count_below_is_admitted_pure_and_names_its_assumptions() {
    let s = store();
    match decide(&s, COUNT_BELOW, Request::Pure) {
        Ok(Admitted::Pure { binding, assumed }) => {
            assert_eq!(binding.id, COUNT_BELOW);
            assert_eq!(assumed.len(), 3, "{assumed:?}");
            assert!(assumed.iter().all(|c| c.starts_with("core::")));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn count_below_without_trust_is_unknown_not_pure() {
    let s = store();
    let r = method_ref(&s, COUNT_BELOW);
    let got = ExecutionPolicy::default().decide(&s, &r, &fixture::span_joined(), Request::Pure);
    match got {
        Err(ExecRefusal::Unknown { causes }) => {
            assert_eq!(causes.len(), 3);
            assert!(causes
                .iter()
                .all(|c| matches!(c, UnknownCause::UnauditedCallee(_))));
        }
        other => panic!("{other:?}"),
    }
}

// ---- 2. witnessed effect, under an explicit DO grant ------------------

fn grant_for(r: &MethodRef, effects: &[(Relation, &str)]) -> DoGrant {
    DoGrant {
        method: r.clone(),
        effects: effects
            .iter()
            .map(|(rel, s)| (*rel, s.to_string()))
            .collect(),
    }
}

const TALLY_CLOSURE: &str = "sag_fixture::count_with_tally::{closure@L42}::call_mut";
const DIV_CLOSURE: &str = "sag_fixture::count_div::{closure@L51}::call_mut";

#[test]
fn count_with_tally_executes_only_under_a_covering_grant() {
    let s = store();
    let r = method_ref(&s, COUNT_WITH_TALLY);
    let g = grant_for(&r, &[(Relation::WritesCallerMemory, TALLY_CLOSURE)]);
    match decide(&s, COUNT_WITH_TALLY, Request::Do(&g)) {
        Ok(Admitted::UnderDoGrant { binding, effects }) => {
            assert_eq!(binding.id, COUNT_WITH_TALLY);
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].relation, Relation::WritesCallerMemory);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn count_div_executes_only_under_a_grant_naming_the_panic() {
    let s = store();
    let r = method_ref(&s, COUNT_DIV);
    let g = grant_for(&r, &[(Relation::MayPanic, DIV_CLOSURE)]);
    assert!(matches!(
        decide(&s, COUNT_DIV, Request::Do(&g)),
        Ok(Admitted::UnderDoGrant { .. })
    ));
    // The same subject under another relation does not cover it.
    let wrong = grant_for(&r, &[(Relation::WritesCallerMemory, DIV_CLOSURE)]);
    assert!(matches!(
        decide(&s, COUNT_DIV, Request::Do(&wrong)),
        Err(ExecRefusal::GrantDoesNotCover { uncovered }) if uncovered.len() == 1
    ));
}

// ---- 5. the DO gate refuses -------------------------------------------

#[test]
fn witnessed_effects_are_refused_for_pure_execution() {
    let s = store();
    for (id, rel) in [
        (COUNT_WITH_TALLY, Relation::WritesCallerMemory),
        (COUNT_DIV, Relation::MayPanic),
    ] {
        match decide(&s, id, Request::Pure) {
            Err(ExecRefusal::RequiresDoGate { effects }) => {
                assert_eq!(effects.len(), 1);
                assert_eq!(effects[0].relation, rel);
            }
            other => panic!("{id:?}: {other:?}"),
        }
    }
}

#[test]
fn an_empty_grant_or_one_for_another_method_does_not_open_the_gate() {
    let s = store();
    let r = method_ref(&s, COUNT_WITH_TALLY);
    let empty = grant_for(&r, &[]);
    assert!(matches!(
        decide(&s, COUNT_WITH_TALLY, Request::Do(&empty)),
        Err(ExecRefusal::GrantDoesNotCover { uncovered }) if uncovered.len() == 1
    ));
    let other = method_ref(&s, COUNT_DIV);
    let misdirected = grant_for(&other, &[(Relation::WritesCallerMemory, TALLY_CLOSURE)]);
    assert!(matches!(
        decide(&s, COUNT_WITH_TALLY, Request::Do(&misdirected)),
        Err(ExecRefusal::GrantForOtherMethod { .. })
    ));
}

// ---- 3. POSSIBLE ------------------------------------------------------

#[test]
fn generic_and_dyn_dispatch_are_possible_never_pure() {
    let s = store();
    for (id, object) in [(COUNT_GENERIC, "trait:Clause"), (COUNT_DYN, "dynamic")] {
        match decide(&s, id, Request::Pure) {
            Err(ExecRefusal::Possible { rows }) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].relation, Relation::MayDispatch);
                assert_eq!(rows[0].object, object);
            }
            other => panic!("{id:?}: {other:?}"),
        }
    }
}

#[test]
fn a_grant_naming_the_dispatch_row_does_not_lift_possible() {
    let s = store();
    let r = method_ref(&s, COUNT_DYN);
    let g = grant_for(
        &r,
        &[(
            Relation::MayDispatch,
            "sag_fixture::count_dyn::{closure@L36}::call_mut",
        )],
    );
    assert!(matches!(
        decide(&s, COUNT_DYN, Request::Do(&g)),
        Err(ExecRefusal::Possible { .. })
    ));
}

#[test]
fn a_local_buffer_write_is_possible_not_an_authorizable_effect() {
    let s = store();
    match decide(&s, MASK_BELOW, Request::Pure) {
        Err(ExecRefusal::Possible { rows }) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].relation, Relation::WritesThroughLocalRef);
        }
        other => panic!("{other:?}"),
    }
}

// ---- 4. UNKNOWN -------------------------------------------------------

#[test]
fn below_test_is_unknown_under_the_path_join_and_pure_under_the_span_join() {
    let s = store();
    let r = method_ref(&s, BELOW_TEST);
    let path = policy().decide(&s, &r, &fixture::path_joined(), Request::Pure);
    match path {
        Err(ExecRefusal::Unknown { causes }) => {
            assert!(causes.iter().any(
                |c| matches!(c, UnknownCause::Unjoined(row) if row.relation == Relation::Unjoined)
            ));
        }
        other => panic!("{other:?}"),
    }
    let span = policy().decide(&s, &r, &fixture::span_joined(), Request::Pure);
    assert!(matches!(span, Ok(Admitted::Pure { .. })), "{span:?}");
}

#[test]
fn a_method_named_like_a_pure_one_but_unattached_is_unknown() {
    let s = store();
    // UNATTACHED is bound under the name "count_below"; the name is never
    // identity, so count_below's evidence must not attach to it.
    let r = method_ref(&s, UNATTACHED);
    assert_eq!(s.admit(&r).unwrap().name, "count_below");
    match policy().decide(&s, &r, &fixture::span_joined(), Request::Pure) {
        Err(ExecRefusal::Unknown { causes }) => {
            assert!(matches!(causes[..], [UnknownCause::NotAttached { .. }]))
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn evidence_for_another_signature_revision_does_not_attach() {
    let s = store();
    let r = method_ref(&s, COUNT_BELOW);
    let mut ev = fixture::span_joined();
    for a in &mut ev.attachments {
        a.sig_rev = SigRev(0);
    }
    assert!(matches!(
        policy().decide(&s, &r, &ev, Request::Pure),
        Err(ExecRefusal::Unknown { .. })
    ));
}

#[test]
fn a_grant_does_not_lift_unknown() {
    let s = store();
    let r = method_ref(&s, BELOW_TEST);
    let g = grant_for(&r, &[(Relation::Unjoined, "sag_fixture::Impl::test")]);
    assert!(matches!(
        policy().decide(&s, &r, &fixture::path_joined(), Request::Do(&g)),
        Err(ExecRefusal::Unknown { .. })
    ));
}
