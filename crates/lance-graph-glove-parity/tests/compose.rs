//! D-CMH-G1: one method reused inside another.
//!
//! A predicate method (`MethodBody::Select`) declares a selection and folds
//! nothing. Another method names it with `Cond::Method`; preparation resolves
//! the reference once and joins the predicate's conditions into the caller's
//! conjunction before binding. The caller then lowers to exactly the program a
//! hand-written conjunction lowers to: there is no call at run time, no second
//! evaluator, and no Rust per method.
//!
//! Keys and labels are FIXTURES.
mod common;

use common::*;
use lance_graph_glove_parity::basis::canonical_fields;
use lance_graph_glove_parity::method::{
    CallError, Cond, EffectClass, Fold, Frontend, Law, MethodBody, MethodKey, MethodRef,
    MethodSpec, MethodTable, PrepareError, ReadingId, SchemaSeal,
};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_sap::binder::CatsBinder;

const IS_BILLABLE: MethodKey = MethodKey([0x11; 16]);
const BILLABLE_HOURS: MethodKey = MethodKey([0x12; 16]);
const BILLABLE_COUNT: MethodKey = MethodKey([0x13; 16]);
const DIRECT_HOURS: MethodKey = MethodKey([0x14; 16]);

fn spec(key: MethodKey, body: MethodBody<'static>) -> MethodSpec<'static> {
    MethodSpec {
        key,
        concept: BILLABLE_WORK_ENTRY,
        schema: seal(),
        reading: READING,
        effect: EffectClass::Pure,
        body,
    }
}

const fn values(
    filter: &'static [Cond<'static>],
    folds: &'static [Fold<'static>],
) -> MethodBody<'static> {
    MethodBody::Query {
        filter,
        folds,
        law: Law::Values,
    }
}

const USES_IS_BILLABLE: &[Cond<'static>] = &[Cond::Method(MethodRef::Key(IS_BILLABLE))];

/// The three methods of the Lego example, plus the hand-written twin of the
/// sum, keyed `DIRECT_HOURS`. `predicate` is `is_billable`'s condition.
fn lego(predicate: &'static [Cond<'static>]) -> Vec<(&'static str, MethodSpec<'static>)> {
    vec![
        (
            "is_billable",
            spec(IS_BILLABLE, MethodBody::Select { filter: predicate }),
        ),
        (
            "billable_hours",
            spec(
                BILLABLE_HOURS,
                values(USES_IS_BILLABLE, &[Fold::Sum("duration")]),
            ),
        ),
        (
            "billable_count",
            spec(BILLABLE_COUNT, values(USES_IS_BILLABLE, &[Fold::Count])),
        ),
        (
            "direct_hours",
            spec(
                DIRECT_HOURS,
                values(
                    &[Cond::Eq {
                        field: "billable",
                        literal: "true",
                    }],
                    &[Fold::Sum("duration")],
                ),
            ),
        ),
    ]
}

const BILLABLE: &[Cond<'static>] = &[Cond::Eq {
    field: "billable",
    literal: "true",
}];
const NOT_BILLED: &[Cond<'static>] = &[Cond::Eq {
    field: "billable",
    literal: "false",
}];

/// FAILS IF: a method using a predicate by reference does not lower to the
/// program of the hand-written conjunction, or either consumer disagrees with
/// the row-loop oracle on either frontend.
#[test]
fn two_methods_reuse_one_predicate_method() {
    for n in SIZES {
        let batch = batch(n);
        let cats = CatsBinder::new(&batch);
        let (basis, claims) = (canonical_fields(), with_duration());
        let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &lego(BILLABLE));
        let get = |k| t.get(k).unwrap().as_ref().unwrap();
        assert_eq!(get(BILLABLE_HOURS).programs(), get(DIRECT_HOURS).programs());
        assert_eq!(get(BILLABLE_HOURS).required, get(DIRECT_HOURS).required);
        let (count, sum) = count_sum(&batch, |c| c == 1);
        for f in [Frontend::Quack, Frontend::R2il] {
            assert_eq!(
                call(&t, &batch, BILLABLE_HOURS, f).unwrap(),
                scalars(&[sum]),
                "n={n}"
            );
            assert_eq!(
                call(&t, &batch, BILLABLE_COUNT, f).unwrap(),
                scalars(&[count]),
                "n={n}"
            );
        }
    }
}

/// FAILS IF: the consumers copied the predicate instead of naming it: editing
/// only `is_billable` must move both of them.
#[test]
fn editing_the_predicate_moves_every_consumer() {
    let batch = batch(4097);
    let cats = CatsBinder::new(&batch);
    let (basis, claims) = (canonical_fields(), with_duration());
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &lego(NOT_BILLED));
    let (count, sum) = count_sum(&batch, |c| c == 0);
    let (bcount, bsum) = count_sum(&batch, |c| c == 1);
    assert!(
        count != bcount && sum != bsum,
        "non-vacuous: the predicates select differently"
    );
    for f in [Frontend::Quack, Frontend::R2il] {
        assert_eq!(
            call(&t, &batch, BILLABLE_HOURS, f).unwrap(),
            scalars(&[sum])
        );
        assert_eq!(
            call(&t, &batch, BILLABLE_COUNT, f).unwrap(),
            scalars(&[count])
        );
    }
}

/// A predicate method selects rows and folds nothing: a pure call has no
/// value to return and is refused by name, never run as a count.
#[test]
fn a_predicate_method_is_not_a_terminal() {
    let batch = batch(65);
    let cats = CatsBinder::new(&batch);
    let (basis, claims) = (canonical_fields(), with_duration());
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &lego(BILLABLE));
    let m = t.get(IS_BILLABLE).unwrap().as_ref().unwrap();
    assert!(m.programs().is_empty());
    assert!(matches!(
        call(&t, &batch, IS_BILLABLE, Frontend::Quack),
        Err(CallError::NotTerminal)
    ));
}

/// A reference may chain: `is_billable_and_known` names `is_billable`, and a
/// fold names `is_billable_and_known`. Labels resolve exactly like keys.
#[test]
fn references_chain_and_resolve_by_label() {
    const KNOWN: MethodKey = MethodKey([0x21; 16]);
    const COUNT: MethodKey = MethodKey([0x22; 16]);
    let specs = vec![
        (
            "is_billable",
            spec(IS_BILLABLE, MethodBody::Select { filter: BILLABLE }),
        ),
        (
            "is_billable_and_known",
            spec(
                KNOWN,
                MethodBody::Select {
                    filter: &[
                        Cond::Method(MethodRef::Label("is_billable")),
                        Cond::Ne {
                            field: "billable",
                            literal: "false",
                        },
                    ],
                },
            ),
        ),
        (
            "count_known",
            spec(
                COUNT,
                values(&[Cond::Method(MethodRef::Key(KNOWN))], &[Fold::Count]),
            ),
        ),
    ];
    let batch = batch(4097);
    let cats = CatsBinder::new(&batch);
    let (basis, claims) = (canonical_fields(), with_duration());
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &specs);
    let (count, _) = count_sum(&batch, |c| c == 1);
    for f in [Frontend::Quack, Frontend::R2il] {
        assert_eq!(call(&t, &batch, COUNT, f).unwrap(), scalars(&[count]));
    }
}

fn refusal(specs: &[(&str, MethodSpec<'_>)], key: MethodKey) -> PrepareError {
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let (basis, claims) = (canonical_fields(), with_duration());
    let t = MethodTable::prepare(&glove(&basis, &cats, &claims), specs);
    t.get(key).unwrap().clone().unwrap_err()
}

/// FAILS IF: a broken reference binds anyway. Each corruption is a typed
/// refusal of the CALLER at preparation.
#[test]
fn a_broken_reference_is_refused() {
    const A: MethodKey = MethodKey([0xA1; 16]);
    const B: MethodKey = MethodKey([0xA2; 16]);
    const MISSING: MethodKey = MethodKey([0xEE; 16]);
    let count_of = |filter: &'static [Cond<'static>]| values(filter, &[Fold::Count]);
    // Unknown key, unknown label.
    let s = [(
        "a",
        spec(A, count_of(&[Cond::Method(MethodRef::Key(MISSING))])),
    )];
    assert_eq!(
        refusal(&s, A),
        PrepareError::UnknownMethodRef("ee".repeat(16))
    );
    let s = [(
        "a",
        spec(A, count_of(&[Cond::Method(MethodRef::Label("nope"))])),
    )];
    assert_eq!(
        refusal(&s, A),
        PrepareError::UnknownMethodRef("nope".into())
    );
    // A reference to a fold, or to an action, is not a predicate.
    let s = [
        ("a", spec(A, count_of(&[Cond::Method(MethodRef::Key(B))]))),
        ("b", spec(B, values(BILLABLE, &[Fold::Count]))),
    ];
    assert_eq!(refusal(&s, A), PrepareError::MethodRefNotPredicate(B));
    let s = [
        ("a", spec(A, count_of(&[Cond::Method(MethodRef::Key(B))]))),
        (
            "b",
            MethodSpec {
                effect: EffectClass::ExternalMutate,
                ..spec(B, MethodBody::Action { action: "submit" })
            },
        ),
    ];
    assert_eq!(refusal(&s, A), PrepareError::MethodRefNotPredicate(B));
    // A cycle: self, and through a second predicate.
    let s = [(
        "a",
        spec(
            A,
            MethodBody::Select {
                filter: &[Cond::Method(MethodRef::Key(A))],
            },
        ),
    )];
    assert_eq!(refusal(&s, A), PrepareError::MethodCycle(A));
    let s = [
        (
            "a",
            spec(
                A,
                MethodBody::Select {
                    filter: &[Cond::Method(MethodRef::Key(B))],
                },
            ),
        ),
        (
            "b",
            spec(
                B,
                MethodBody::Select {
                    filter: &[Cond::Method(MethodRef::Label("a"))],
                },
            ),
        ),
        (
            "c",
            spec(
                MethodKey([0xA3; 16]),
                count_of(&[Cond::Method(MethodRef::Key(A))]),
            ),
        ),
    ];
    assert_eq!(refusal(&s, A), PrepareError::MethodCycle(A));
    assert_eq!(
        refusal(&s, MethodKey([0xA3; 16])),
        PrepareError::MethodCycle(A)
    );
    // A predicate written for another basis or reading is not inlined into
    // this one, even though the caller is fine.
    for other in [
        MethodSpec {
            schema: SchemaSeal(seal().0 ^ 1),
            ..spec(B, MethodBody::Select { filter: BILLABLE })
        },
        MethodSpec {
            reading: ReadingId(READING.0 + 1),
            ..spec(B, MethodBody::Select { filter: BILLABLE })
        },
    ] {
        let s = [
            ("a", spec(A, count_of(&[Cond::Method(MethodRef::Key(B))]))),
            ("b", other),
        ];
        assert_eq!(refusal(&s, A), PrepareError::MethodRefContract(B));
    }
    // An ambiguous label names nothing; an ambiguous key is refused.
    let s = [
        (
            "a",
            spec(A, count_of(&[Cond::Method(MethodRef::Label("p"))])),
        ),
        ("p", spec(B, MethodBody::Select { filter: BILLABLE })),
        (
            "p",
            spec(
                MethodKey([0xA4; 16]),
                MethodBody::Select { filter: BILLABLE },
            ),
        ),
    ];
    assert_eq!(refusal(&s, A), PrepareError::UnknownMethodRef("p".into()));
    // The predicate's own refusal is the caller's: an unknown field.
    let s = [
        ("a", spec(A, count_of(&[Cond::Method(MethodRef::Key(B))]))),
        (
            "b",
            spec(
                B,
                MethodBody::Select {
                    filter: &[Cond::Eq {
                        field: "colour",
                        literal: "red",
                    }],
                },
            ),
        ),
    ];
    assert_eq!(refusal(&s, A), PrepareError::UnknownField("colour".into()));
    // Silence twin: a predicate that only reads `billable` binds against
    // SAP's real claims, where `duration` is unproven.
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let basis = canonical_fields();
    let t = MethodTable::prepare(
        &glove(&basis, &cats, lance_graph_glove_parity::basis::SAP_FIELDS),
        &lego(BILLABLE),
    );
    assert!(t.get(BILLABLE_COUNT).unwrap().is_ok());
    assert_eq!(
        t.get(BILLABLE_HOURS).unwrap(),
        &Err(PrepareError::UnmappedRequired("duration".into()))
    );
}
