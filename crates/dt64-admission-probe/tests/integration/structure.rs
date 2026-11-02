//! Structural guards on synthetic rows (labelled synthetic: these subjects
//! are not in the D-SAG-6 fixture). They pin what the fixture alone cannot:
//! that effects are found through local callees, that UNKNOWN dominates
//! POSSIBLE dominates a witnessed effect, and that absence of rows is not
//! purity.

use dt64_admission_probe::evidence::{parse_rows, Evidence, Relation, Row, HEADER};
use dt64_admission_probe::fixture::{method_ref, store, CLASS, COUNT_BELOW};
use dt64_admission_probe::{Attachment, EvidenceClass, EvidenceSet, ExecutionPolicy, UnknownCause};
use lance_graph_dt64::SigRev;

fn evidence(body: &str, subject: &str) -> EvidenceSet {
    EvidenceSet {
        rows: parse_rows(&format!("{HEADER}\n{body}")).unwrap(),
        attachments: vec![Attachment {
            class: CLASS,
            method: COUNT_BELOW,
            sig_rev: SigRev(1),
            subject: subject.to_string(),
        }],
    }
}

fn classify(ev: &EvidenceSet) -> EvidenceClass {
    let s = store();
    let r = method_ref(&s, COUNT_BELOW);
    ExecutionPolicy::default().classify(&r, ev)
}

const DEF_F: &str = "DEFINES\tx::f\tFunction\trust-analyzer\tWITNESSED\n";
const DEF_G: &str = "DEFINES\tx::g\tFunction\trust-analyzer\tWITNESSED\n";
const F_CALLS_G: &str = "CALLS\tx::f\tx::g\tcharon\tWITNESSED\n";

#[test]
fn an_effect_in_a_local_callee_is_found() {
    let ev = evidence(
        &format!("{DEF_F}{DEF_G}{F_CALLS_G}MAY_PANIC\tx::g\t1 assert\tcharon\tWITNESSED\n"),
        "x::f",
    );
    match classify(&ev) {
        EvidenceClass::WitnessedEffect { effects } => {
            assert_eq!(effects.len(), 1);
            assert_eq!(effects[0].subject, "x::g");
        }
        other => panic!("{other:?}"),
    }
    // Silence: a pure local callee keeps the caller pure, with no assumption.
    let pure = evidence(&format!("{DEF_F}{DEF_G}{F_CALLS_G}"), "x::f");
    assert_eq!(
        classify(&pure),
        EvidenceClass::ProvenPure { assumed: vec![] }
    );
}

#[test]
fn a_recursive_cycle_terminates() {
    let ev = evidence(
        &format!("{DEF_F}{DEF_G}{F_CALLS_G}CALLS\tx::g\tx::f\tcharon\tWITNESSED\n"),
        "x::f",
    );
    assert_eq!(classify(&ev), EvidenceClass::ProvenPure { assumed: vec![] });
}

#[test]
fn unknown_dominates_possible_dominates_effect() {
    let effect = "MAY_PANIC\tx::f\t1 assert\tcharon\tWITNESSED\n";
    let possible = "MAY_DISPATCH\tx::f\tdynamic\tcharon\tPOSSIBLE\n";
    let unknown = "UNJOINED\tx::f::{closure@L3}::call_mut\tno symbol\tjoin\tUNKNOWN\n";
    assert!(matches!(
        classify(&evidence(
            &format!("{DEF_F}{effect}{possible}{unknown}"),
            "x::f"
        )),
        EvidenceClass::Unknown { .. }
    ));
    assert!(matches!(
        classify(&evidence(&format!("{DEF_F}{effect}{possible}"), "x::f")),
        EvidenceClass::Possible { .. }
    ));
    assert!(matches!(
        classify(&evidence(&format!("{DEF_F}{effect}"), "x::f")),
        EvidenceClass::WitnessedEffect { .. }
    ));
}

#[test]
fn absence_of_rows_is_not_purity() {
    // An attachment to a subject the frontend never defined.
    let ev = evidence(DEF_G, "x::f");
    match classify(&ev) {
        EvidenceClass::Unknown { causes } => {
            assert_eq!(
                causes,
                vec![UnknownCause::NoDefinition {
                    subject: "x::f".into()
                }]
            )
        }
        other => panic!("{other:?}"),
    }
    // Silence: the defined neighbour is pure.
    assert!(matches!(
        classify(&evidence(DEF_G, "x::g")),
        EvidenceClass::ProvenPure { .. }
    ));
}

#[test]
fn an_unnamed_trait_impl_call_is_unknown() {
    let ev = evidence(
        &format!("{DEF_F}CALLS\tx::f\tknown impl\tcharon\tWITNESSED\n"),
        "x::f",
    );
    assert!(matches!(
        classify(&ev),
        EvidenceClass::Unknown { causes } if matches!(causes[..], [UnknownCause::UnnamedCallee(_)])
    ));
}

#[test]
fn a_nested_subject_is_in_scope_but_a_name_sharing_a_prefix_is_not() {
    let closure_effect =
        "WRITES_CALLER_MEMORY\tx::f::{closure@L9}::call_mut\t1\tcharon\tWITNESSED\n";
    assert!(matches!(
        classify(&evidence(&format!("{DEF_F}{closure_effect}"), "x::f")),
        EvidenceClass::WitnessedEffect { .. }
    ));
    let sibling = "WRITES_CALLER_MEMORY\tx::f_other\t1\tcharon\tWITNESSED\n";
    assert!(matches!(
        classify(&evidence(&format!("{DEF_F}{sibling}"), "x::f")),
        EvidenceClass::ProvenPure { .. }
    ));
}

#[test]
fn a_hand_built_mislabelled_row_is_never_pure() {
    // `parse_rows` rejects this row; a caller that builds rows directly can
    // still hand one over.
    let mut ev = evidence(DEF_F, "x::f");
    ev.rows.push(Row {
        relation: Relation::MayDispatch,
        subject: "x::f".into(),
        object: "dynamic".into(),
        witness: "charon".into(),
        evidence: Evidence::Witnessed,
    });
    assert!(matches!(
        classify(&ev),
        EvidenceClass::Unknown { causes } if matches!(causes[..], [UnknownCause::Mislabelled(_)])
    ));
}

/// A call from a closure shim into the closure body names a subject that has
/// no `DEFINES` row (the span join generates closure subjects) but lies
/// inside the analyzed scope, whose rows are already read through its
/// prefix. It is not an unaudited external callee.
#[test]
fn a_call_into_an_in_scope_closure_is_not_an_unaudited_callee() {
    let shim =
        "CALLS\tx::f::{closure@L9}::call_once\tx::f::{closure@L9}::call_mut\tcharon\tWITNESSED\n";
    let ev = evidence(&format!("{DEF_F}{shim}"), "x::f");
    assert_eq!(classify(&ev), EvidenceClass::ProvenPure { assumed: vec![] });
}

/// The silent twin: a name that only shares a prefix with the scope, without
/// the `::` boundary, or a closure of another function, is still unaudited.
#[test]
fn a_call_into_another_functions_closure_stays_unaudited() {
    for callee in [
        "x::fg::{closure@L1}::call_mut",
        "x::h::{closure@L2}::call_mut",
    ] {
        let call = format!("CALLS\tx::f\t{callee}\tcharon\tWITNESSED\n");
        let ev = evidence(&format!("{DEF_F}{call}"), "x::f");
        match classify(&ev) {
            EvidenceClass::Unknown { causes } => assert!(
                causes
                    .iter()
                    .all(|c| matches!(c, UnknownCause::UnauditedCallee(_))),
                "{callee}: {causes:?}"
            ),
            other => panic!("{callee}: {other:?}"),
        }
    }
}
