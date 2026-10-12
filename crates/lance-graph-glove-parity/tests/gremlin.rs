//! D-CMH-G1: Gremlin steps as typed building blocks of a class method.
//!
//! A traversal is a slice of typed [`Step`]s over ONE concept's population.
//! [`compose`] turns it into a [`MethodBody`] (conditions and folds in the
//! concept's canonical fields), which then goes through the same
//! `MethodTable::prepare` and Quack lowering as every other method. No step
//! runs at execution time: the steps are gone once the body exists.
//!
//! The anchor is the concept's table: each live row is one vertex, one
//! traverser of bulk 1. That is what makes `count()` and `dedup().count()`
//! the same number HERE, and why `dedup()` after `values(f)` (distinct values,
//! a different multiset) is refused rather than collapsed. A hop needs a
//! relation carrier the glove does not declare, so every `out`/`in`/`both` is
//! refused by name; the bag/set behaviour across hops is the #1311 witness
//! (`crates/lance-graph-quack/tests/gremlin_parity.rs`).
mod common;

use common::*;
use lance_graph_glove_parity::basis::canonical_fields;
use lance_graph_glove_parity::gremlin::{compose, GremlinRefusal, Pred, Step};
use lance_graph_glove_parity::method::{
    Cond, EffectClass, Fold, Frontend, Law, MethodBody, MethodKey, MethodRef, MethodSpec,
    MethodTable,
};
use lance_graph_ogar::ogar_vocab::class_ids::{BILLABLE_WORK_ENTRY, DOCUMENT};
use lance_graph_quack::bind::table;
use lance_graph_quack::{lower, Agg, Query};
use lance_graph_sap::binder::{CatsBinder, TABLE};

const IS_BILLABLE: MethodKey = MethodKey([0x31; 16]);
const BILLABLE_HOURS: MethodKey = MethodKey([0x32; 16]);
const BILLABLE_COUNT: MethodKey = MethodKey([0x33; 16]);
const DEDUP_COUNT: MethodKey = MethodKey([0x34; 16]);

const V: Step<'static> = Step::V(BILLABLE_WORK_ENTRY);

fn has(field: &'static str, lit: &'static str) -> Step<'static> {
    Step::Has {
        field,
        pred: Pred::Eq(lit),
    }
}

fn spec<'a>(key: MethodKey, body: MethodBody<'a>) -> MethodSpec<'a> {
    MethodSpec {
        key,
        concept: BILLABLE_WORK_ENTRY,
        schema: seal(),
        reading: READING,
        effect: EffectClass::Pure,
        body,
    }
}

/// FAILS IF: a composed traversal does not lower to the program of the same
/// question asked directly in Quack, or the composed methods disagree with
/// the oracle or with each other's frontends.
#[test]
fn a_composed_traversal_is_the_direct_quack_query() {
    let is_billable = compose(BILLABLE_WORK_ENTRY, &[V, has("billable", "true")]).unwrap();
    let uses = Step::Where(MethodRef::Label("is_billable"));
    let hours = compose(
        BILLABLE_WORK_ENTRY,
        &[V, uses, Step::Values("duration"), Step::Sum],
    )
    .unwrap();
    let count = compose(BILLABLE_WORK_ENTRY, &[V, uses, Step::Count]).unwrap();
    let dedup = compose(BILLABLE_WORK_ENTRY, &[V, uses, Step::Dedup, Step::Count]).unwrap();
    // The bodies, as data: what `compose` produced is plain conditions and
    // folds over canonical fields.
    assert_eq!(
        is_billable.body(),
        MethodBody::Select {
            filter: &[Cond::Eq {
                field: "billable",
                literal: "true"
            }]
        }
    );
    assert_eq!(
        hours.body(),
        MethodBody::Query {
            filter: &[Cond::Method(MethodRef::Label("is_billable"))],
            folds: &[Fold::Sum("duration")],
            law: Law::Values,
        }
    );
    for n in SIZES {
        let batch = batch(n);
        let cats = CatsBinder::new(&batch);
        let (basis, claims) = (canonical_fields(), with_duration());
        let specs = [
            ("is_billable", spec(IS_BILLABLE, is_billable.body())),
            ("billable_hours", spec(BILLABLE_HOURS, hours.body())),
            ("billable_count", spec(BILLABLE_COUNT, count.body())),
            ("dedup_count", spec(DEDUP_COUNT, dedup.body())),
        ];
        let t = MethodTable::prepare(&glove(&basis, &cats, &claims), &specs);
        // Direct Quack: the developer's own query against the glove.
        let tid = lance_graph_quack::bind::Binder::table(&cats, TABLE).unwrap();
        let hours_col = lance_graph_quack::bind::Binder::field(&cats, tid, "hours_logged")
            .unwrap()
            .col;
        let filter = table(TABLE)
            .where_eq("billable", "true")
            .count()
            .bind(&cats)
            .unwrap()
            .filter;
        let direct = |agg| {
            lower(&Query {
                filter: filter.clone(),
                agg,
            })
            .unwrap()
        };
        let prepared = |k| t.get(k).unwrap().as_ref().unwrap();
        assert_eq!(
            prepared(BILLABLE_HOURS).programs(),
            [direct(Agg::SumI32(hours_col))]
        );
        assert_eq!(prepared(BILLABLE_COUNT).programs(), [direct(Agg::Count)]);
        let (c, s) = count_sum(&batch, |code| code == 1);
        for f in [Frontend::Quack, Frontend::R2il] {
            assert_eq!(
                call(&t, &batch, BILLABLE_HOURS, f).unwrap(),
                scalars(&[s]),
                "n={n}"
            );
            assert_eq!(
                call(&t, &batch, BILLABLE_COUNT, f).unwrap(),
                scalars(&[c]),
                "n={n}"
            );
            // On the anchor every row is a distinct vertex: dedup is the
            // identity, and the count is the same number.
            assert_eq!(
                call(&t, &batch, DEDUP_COUNT, f).unwrap(),
                scalars(&[c]),
                "n={n}"
            );
        }
    }
}

/// The multiplicity contract of this slice, stated as refusals: nothing that
/// would change a bag into a set, or read through a relation, is lowered.
#[test]
fn steps_outside_the_anchor_contract_are_refused() {
    let refuse = |steps: &[Step<'static>]| compose(BILLABLE_WORK_ENTRY, steps).unwrap_err();
    // A hop: the glove declares no relation carrier, in either direction.
    for hop in [
        Step::Out("performed_by"),
        Step::In("performed_by"),
        Step::Both("performed_by"),
    ] {
        assert_eq!(
            refuse(&[V, hop, Step::Count]),
            GremlinRefusal::NoRelationCarrier("performed_by".into())
        );
    }
    // dedup of values is a distinct-value count: a different multiset.
    assert_eq!(
        refuse(&[V, Step::Values("duration"), Step::Dedup, Step::Count]),
        GremlinRefusal::DedupOfValues
    );
    // The anchor must be the concept's population.
    assert_eq!(
        refuse(&[Step::V(DOCUMENT), Step::Count]),
        GremlinRefusal::AnchorMismatch {
            step: DOCUMENT,
            concept: BILLABLE_WORK_ENTRY
        }
    );
    assert_eq!(refuse(&[Step::Count]), GremlinRefusal::NoStart);
    assert_eq!(refuse(&[V, V, Step::Count]), GremlinRefusal::NoStart);
    // A sum of vertices, a stream of values with no fold, a filter on values,
    // and anything after the terminal.
    assert_eq!(refuse(&[V, Step::Sum]), GremlinRefusal::SumOfVertices);
    assert_eq!(
        refuse(&[V, Step::Values("duration")]),
        GremlinRefusal::NoTerminal
    );
    assert_eq!(
        refuse(&[
            V,
            Step::Values("duration"),
            has("billable", "true"),
            Step::Count
        ]),
        GremlinRefusal::FilterAfterValues
    );
    assert_eq!(
        refuse(&[V, Step::Count, Step::Count]),
        GremlinRefusal::AfterTerminal
    );
    // Silence twin: the same steps in a lawful order compose.
    assert!(compose(
        BILLABLE_WORK_ENTRY,
        &[V, has("billable", "true"), Step::Dedup, Step::Count]
    )
    .is_ok());
    assert!(compose(
        BILLABLE_WORK_ENTRY,
        &[V, Step::Values("duration"), Step::Sum]
    )
    .is_ok());
}

/// `values(f).count()` is `COUNT(f)`, which differs from `COUNT(*)` once `f`
/// can be null. The method layer has no field-reading count fold, so it is
/// refused rather than lowered as `count()` (which would read no field and
/// skip the field's claim and nullability checks).
#[test]
fn counting_values_is_refused_not_rewritten() {
    assert_eq!(
        compose(
            BILLABLE_WORK_ENTRY,
            &[V, Step::Values("duration"), Step::Count]
        )
        .unwrap_err(),
        GremlinRefusal::CountOfValues
    );
}
