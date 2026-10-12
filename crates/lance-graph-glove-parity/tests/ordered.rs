//! Ordered conditions on `I32` fields and the `MIN`/`MAX` folds, on both
//! frontends, against a row loop over the indicator text (the SAP CATS glove
//! of `common`).
//!
//! An ordered literal is compared in the lane's own units: a method states a
//! number, never text, and no codebook resolves it.
mod common;

use common::*;
use lance_graph_glove_parity::basis::canonical_fields;
use lance_graph_glove_parity::decl::parse_decls;
use lance_graph_glove_parity::gremlin::{compose, GremlinRefusal, Pred, Step};
use lance_graph_glove_parity::method::{
    Answer, Cond, EffectClass, Fold, FoldValue, Frontend, Law, MethodBody, MethodKey, MethodSpec,
    MethodTable, Order, PrepareError,
};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_sap::binder::CatsBinder;

const BILLABLE: Cond = Cond::Eq {
    field: "billable",
    literal: "true",
};
const FOLDS: &[Fold] = &[
    Fold::Count,
    Fold::Sum("duration"),
    Fold::Min("duration"),
    Fold::Max("duration"),
];
const ORDERS: [Order; 4] = [Order::Lt, Order::Le, Order::Gt, Order::Ge];

fn holds(op: Order, a: i64, t: i64) -> bool {
    match op {
        Order::Lt => a < t,
        Order::Le => a <= t,
        Order::Gt => a > t,
        Order::Ge => a >= t,
    }
}

fn spec<'a>(key: u8, body: MethodBody<'a>) -> MethodSpec<'a> {
    MethodSpec {
        key: MethodKey([key; 16]),
        concept: BILLABLE_WORK_ENTRY,
        schema: seal(),
        reading: READING,
        effect: EffectClass::Pure,
        body,
    }
}

/// `(count, sum, min, max)` of billable rows whose hours satisfy `keep`.
fn oracle(batch: &lance_graph_sap::bind::CatsBatch, keep: impl Fn(i64) -> bool) -> Answer {
    let hours: Vec<i64> = rows(batch)
        .into_iter()
        .filter(|&(c, h)| c == 1 && keep(h))
        .map(|(_, h)| h)
        .collect();
    Answer::Values(vec![
        FoldValue::Scalar(hours.len() as i64),
        FoldValue::Scalar(hours.iter().sum()),
        FoldValue::Extremum(hours.iter().copied().min()),
        FoldValue::Extremum(hours.iter().copied().max()),
    ])
}

/// FAILS IF: an ordered condition or a `MIN`/`MAX` fold disagrees with the
/// row loop on either frontend, for each comparison and for thresholds below,
/// at, between and above the lane's values (the last selects nothing, so
/// `MIN`/`MAX` answer `None`).
#[test]
fn ordered_conditions_and_extrema_match_the_row_loop() {
    let (basis, claims) = (canonical_fields(), with_duration());
    for n in SIZES {
        let batch = batch(n);
        let cats = CatsBinder::new(&batch);
        let mut vals: Vec<i64> = rows(&batch).into_iter().map(|(_, h)| h).collect();
        vals.sort_unstable();
        vals.dedup();
        let (lo, hi) = (vals[0], vals[vals.len() - 1]);
        let thresholds = [lo - 1, lo, (lo + hi) / 2, hi, hi + 1];
        let filters: Vec<[Cond<'static>; 2]> = ORDERS
            .iter()
            .flat_map(|&op| {
                thresholds.iter().map(move |&t| {
                    [
                        BILLABLE,
                        Cond::Order {
                            field: "duration",
                            op,
                            value: i32::try_from(t).unwrap(),
                        },
                    ]
                })
            })
            .collect();
        let specs: Vec<(String, MethodSpec<'_>)> = filters
            .iter()
            .enumerate()
            .map(|(i, f)| {
                (
                    format!("m{i}"),
                    spec(
                        i as u8,
                        MethodBody::Query {
                            filter: f,
                            folds: FOLDS,
                            law: Law::Values,
                        },
                    ),
                )
            })
            .collect();
        let labelled: Vec<(&str, MethodSpec<'_>)> =
            specs.iter().map(|(l, s)| (l.as_str(), *s)).collect();
        let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &labelled);
        let mut empty = false;
        for (i, &op) in ORDERS.iter().enumerate() {
            for (j, &th) in thresholds.iter().enumerate() {
                let key = MethodKey([(i * thresholds.len() + j) as u8; 16]);
                let want = oracle(&batch, |h| holds(op, h, th));
                empty |= matches!(&want, Answer::Values(v) if v[2] == FoldValue::Extremum(None));
                for f in [Frontend::Quack, Frontend::R2il] {
                    assert_eq!(
                        call(&t, &batch, key, f).unwrap(),
                        want,
                        "{op:?} {th}, n={n}, {f:?}"
                    );
                }
            }
        }
        assert!(empty || n < 4, "some threshold selects nothing (n={n})");
    }
}

/// FAILS IF: an ordered condition on a coded field, or `MIN`/`MAX` of one,
/// prepares. Order is defined on the `I32` lane, not on a codebook's ids.
#[test]
fn order_on_a_coded_field_is_refused() {
    let (basis, claims) = (canonical_fields(), with_duration());
    let batch = batch(4);
    let cats = CatsBinder::new(&batch);
    let refused = |body| {
        let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &[("m", spec(1, body))]);
        t.get(MethodKey([1; 16])).unwrap().clone().unwrap_err()
    };
    let by_code: &[Cond] = &[Cond::Order {
        field: "billable",
        op: Order::Gt,
        value: 0,
    }];
    assert_eq!(
        refused(MethodBody::Query {
            filter: by_code,
            folds: &[Fold::Count],
            law: Law::Values,
        }),
        PrepareError::KindMismatch("billable".into())
    );
    assert_eq!(
        refused(MethodBody::Query {
            filter: &[],
            folds: &[Fold::Min("billable")],
            law: Law::Values,
        }),
        PrepareError::KindMismatch("billable".into())
    );
}

/// FAILS IF: `has(f, gt(n))…values(f).max()` stops composing to the same
/// body as the hand-written one, or the text form stops loading to it.
#[test]
fn ordered_steps_compose_and_load() {
    let v = Step::V(BILLABLE_WORK_ENTRY);
    let steps = [
        v,
        Step::Has {
            field: "billable",
            pred: Pred::Eq("true"),
        },
        Step::Has {
            field: "duration",
            pred: Pred::Gt(100),
        },
        Step::Values("duration"),
        Step::Max,
    ];
    let c = compose(BILLABLE_WORK_ENTRY, &steps).unwrap();
    assert_eq!(
        c.filter(),
        &[
            BILLABLE,
            Cond::Order {
                field: "duration",
                op: Order::Gt,
                value: 100
            }
        ]
    );
    assert_eq!(c.folds(), &[Fold::Max("duration")]);
    for (pred, op) in [
        (Pred::Lt(5), Order::Lt),
        (Pred::Lte(5), Order::Le),
        (Pred::Gte(5), Order::Ge),
    ] {
        let c = compose(
            BILLABLE_WORK_ENTRY,
            &[
                v,
                Step::Has {
                    field: "duration",
                    pred,
                },
                Step::Count,
            ],
        )
        .unwrap();
        assert_eq!(
            c.filter(),
            &[Cond::Order {
                field: "duration",
                op,
                value: 5
            }]
        );
    }
    let refuse = |steps: &[Step<'static>]| compose(BILLABLE_WORK_ENTRY, steps).unwrap_err();
    assert_eq!(refuse(&[v, Step::Min]), GremlinRefusal::ExtremumOfVertices);
    assert_eq!(refuse(&[v, Step::Max]), GremlinRefusal::ExtremumOfVertices);
    assert_eq!(
        refuse(&[
            v,
            Step::Out("r"),
            Step::Has {
                field: "duration",
                pred: Pred::Gt(1),
            },
            Step::Count,
        ]),
        GremlinRefusal::OrderThroughRelation("r".into())
    );

    // Text: the same method, loaded at run time, prepares to the same programs.
    let line = format!(
        "billable_work_entry::m key={} schema={:016x} reading=1 := \
         V(billable_work_entry).has(billable, \"true\").has(duration, gt(100)).values(duration).max()\n",
        "e3".repeat(16),
        seal().0
    );
    let decls = parse_decls(&line).unwrap();
    let (basis, claims) = (canonical_fields(), with_duration());
    let batch = batch(65);
    let cats = CatsBinder::new(&batch);
    let compiled = [("c", spec(0xE4, c.body()))];
    let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &compiled, &decls);
    let get = |l: &str| {
        t.get(t.resolve(l).unwrap())
            .unwrap()
            .as_ref()
            .unwrap()
            .programs()
            .to_vec()
    };
    assert_eq!(get("m"), get("c"));
}
