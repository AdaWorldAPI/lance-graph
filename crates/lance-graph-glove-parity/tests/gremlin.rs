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

// =====================================================================
// The same compositions, loaded as text at run time.
// =====================================================================

use lance_graph_glove_parity::decl::{parse_decls, DeclError};
use lance_graph_glove_parity::method::PrepareError;

fn seal_hex() -> String {
    format!("{:016x}", seal().0)
}

fn lego_text() -> String {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/lego.methods");
    let text = std::fs::read_to_string(path).expect("fixture");
    assert!(
        text.contains(&seal_hex()),
        "re-seal the fixture: {}",
        seal_hex()
    );
    text
}

fn line(label: &str, key: u8, body: &str) -> String {
    format!(
        "billable_work_entry::{label} key={} schema={} reading=1 := {body}\n",
        format!("{key:02x}").repeat(16),
        seal_hex()
    )
}

/// FAILS IF: the Lego methods loaded from text do not prepare to the same
/// programs as the compiled composition, or disagree with the oracle.
#[test]
fn the_lego_methods_load_from_text_and_match_the_compiled_ones() {
    let decls = parse_decls(&lego_text()).expect("parses");
    let is_billable = compose(BILLABLE_WORK_ENTRY, &[V, has("billable", "true")]).unwrap();
    let uses = Step::Where(MethodRef::Label("compiled_is_billable"));
    let hours = compose(
        BILLABLE_WORK_ENTRY,
        &[V, uses, Step::Values("duration"), Step::Sum],
    )
    .unwrap();
    for n in SIZES {
        let batch = batch(n);
        let cats = CatsBinder::new(&batch);
        let (basis, claims) = (canonical_fields(), with_duration());
        let compiled = [
            (
                "compiled_is_billable",
                spec(IS_BILLABLE, is_billable.body()),
            ),
            ("compiled_hours", spec(BILLABLE_HOURS, hours.body())),
        ];
        let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &compiled, &decls);
        let get = |l: &str| t.get(t.resolve(l).unwrap()).unwrap().as_ref().unwrap();
        assert_eq!(
            get("billable_hours").programs(),
            get("compiled_hours").programs()
        );
        let (c, s) = count_sum(&batch, |code| code == 1);
        for f in [Frontend::Quack, Frontend::R2il] {
            let key = |l| t.resolve(l).unwrap();
            assert_eq!(
                call(&t, &batch, key("billable_hours"), f).unwrap(),
                scalars(&[s])
            );
            assert_eq!(
                call(&t, &batch, key("billable_count"), f).unwrap(),
                scalars(&[c])
            );
        }
    }
}

/// FAILS IF: a method loaded later cannot name a predicate loaded earlier or
/// compiled in, or the text outlives nothing it should.
#[test]
fn a_later_declaration_reuses_an_earlier_predicate() {
    let mut decls = parse_decls(&lego_text()).unwrap();
    // Written after the build and after the first load; it names the loaded
    // `is_billable`, and a second predicate it declares itself.
    let later = [
        line(
            "is_not_billed",
            0xC1,
            "V(billable_work_entry).has(billable, false)",
        ),
        line(
            "unbilled_hours",
            0xC2,
            "V(billable_work_entry).where(is_not_billed()).values(duration).sum()",
        ),
        line(
            "billable_known_count",
            0xC3,
            "V(billable_work_entry).where(is_billable()).has(billable, neq(false)).count()",
        ),
    ]
    .concat();
    decls.extend(parse_decls(&later).unwrap());
    drop(later);
    let batch = batch(4097);
    let cats = CatsBinder::new(&batch);
    let (basis, claims) = (canonical_fields(), with_duration());
    let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &[], &decls);
    drop(decls);
    let (_, unbilled) = count_sum(&batch, |c| c == 0);
    let (billable, _) = count_sum(&batch, |c| c == 1);
    for f in [Frontend::Quack, Frontend::R2il] {
        let key = |l| t.resolve(l).unwrap();
        assert_eq!(
            call(&t, &batch, key("unbilled_hours"), f).unwrap(),
            scalars(&[unbilled])
        );
        assert_eq!(
            call(&t, &batch, key("billable_known_count"), f).unwrap(),
            scalars(&[billable])
        );
    }
}

/// FAILS IF: loading text can express an effect, or a form outside the
/// slice loads instead of being refused by name.
#[test]
fn effects_and_unsupported_steps_are_refused_at_load() {
    let head = "V(billable_work_entry)";
    for step in [
        "addV(x)",
        "addE(x)",
        "property(x, 1)",
        "drop()",
        "mergeV(x)",
        "mergeE(x)",
        "sideEffect(x)",
        "aggregate(x)",
        "store(x)",
        "sack(x)",
        "inject(1)",
        "io(x)",
        "call(x)",
    ] {
        let name = step.split('(').next().unwrap();
        match parse_decls(&line("m", 0xE0, &format!("{head}.{step}"))) {
            Err(DeclError::Effect { line: 1, step: s }) => assert_eq!(s, name),
            other => panic!("{step}: {other:?}"),
        }
    }
    for (step, what) in [
        ("groupCount()", "groupCount"),
        ("order()", "order"),
        ("repeat(out(x))", "repeat"),
        ("path()", "path"),
        ("select(a)", "select"),
        ("as(a)", "as"),
        ("has(duration, gt(1))", "gt"),
        ("where(not(is_billable()))", "not"),
        ("mean()", "mean"),
        ("fold()", "fold"),
    ] {
        match parse_decls(&line("m", 0xE1, &format!("{head}.{step}"))) {
            Err(DeclError::Unsupported { line: 1, what: w }) => assert_eq!(w, what, "{step}"),
            other => panic!("{step}: {other:?}"),
        }
    }
    // Lawful steps outside the anchor contract: refused by the composer.
    for (body, refusal) in [
        (
            format!("{head}.out(performed_by).count()"),
            GremlinRefusal::NoRelationCarrier("performed_by".into()),
        ),
        (
            format!("{head}.values(duration).dedup().count()"),
            GremlinRefusal::DedupOfValues,
        ),
        (
            "V(document).count()".to_owned(),
            GremlinRefusal::AnchorMismatch {
                step: DOCUMENT,
                concept: BILLABLE_WORK_ENTRY,
            },
        ),
    ] {
        assert_eq!(
            parse_decls(&line("m", 0xE2, &body)),
            Err(DeclError::Refused { line: 1, refusal }),
            "{body}"
        );
    }
    // Silence twin: every loaded method is pure.
    for d in parse_decls(&lego_text()).unwrap() {
        assert_eq!(d.effect(), EffectClass::Pure);
    }
}

/// FAILS IF: references between loaded methods bypass the preparation
/// checks: a cycle, a missing predicate, a reference to a fold.
#[test]
fn references_between_loaded_methods_are_checked() {
    let text = [
        line("a", 0xF1, "V(billable_work_entry).where(b())"),
        line("b", 0xF2, "V(billable_work_entry).where(a())"),
        line("c", 0xF3, "V(billable_work_entry).where(missing()).count()"),
        line("d", 0xF4, "V(billable_work_entry).count()"),
        line("e", 0xF5, "V(billable_work_entry).where(d()).count()"),
    ]
    .concat();
    let decls = parse_decls(&text).unwrap();
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let (basis, claims) = (canonical_fields(), with_duration());
    let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &[], &decls);
    let refused = |l: &str| t.get(t.resolve(l).unwrap()).unwrap().clone().unwrap_err();
    assert_eq!(
        refused("a"),
        PrepareError::MethodCycle(MethodKey([0xF1; 16]))
    );
    assert_eq!(
        refused("c"),
        PrepareError::UnknownMethodRef("missing".into())
    );
    assert_eq!(
        refused("e"),
        PrepareError::MethodRefNotPredicate(MethodKey([0xF4; 16]))
    );
    assert!(t.get(t.resolve("d").unwrap()).unwrap().is_ok());
}

/// FAILS IF: a label reference reaches a method whose key another method
/// also carries. Both holders of the key are refused; a third method must
/// not inline either one's predicate through its label.
#[test]
fn a_label_reference_to_a_duplicated_key_is_refused() {
    let text = [
        line("x", 0xD1, "V(billable_work_entry).has(billable, true)"),
        line("y", 0xD1, "V(billable_work_entry).has(billable, false)"),
        line("z", 0xD2, "V(billable_work_entry).where(y()).count()"),
    ]
    .concat();
    let decls = parse_decls(&text).unwrap();
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let (basis, claims) = (canonical_fields(), with_duration());
    let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &[], &decls);
    assert_eq!(
        t.get(MethodKey([0xD2; 16])).unwrap().clone().unwrap_err(),
        PrepareError::DuplicateKey(MethodKey([0xD1; 16]))
    );
    // Silence twin: a unique key behind the label still inlines.
    let ok = [
        line("x", 0xD1, "V(billable_work_entry).has(billable, true)"),
        line("z", 0xD2, "V(billable_work_entry).where(x()).count()"),
    ]
    .concat();
    let decls = parse_decls(&ok).unwrap();
    let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &[], &decls);
    assert!(t.get(MethodKey([0xD2; 16])).unwrap().is_ok());
}

/// FAILS IF: a comma or parenthesis inside a quoted literal is read as
/// syntax. The literal must reach preparation whole.
#[test]
fn quoted_literals_keep_their_commas_and_parentheses() {
    let batch = batch(2);
    let cats = CatsBinder::new(&batch);
    let (basis, claims) = (canonical_fields(), with_duration());
    for lit in ["North, America", "a)b", "(x"] {
        let body = format!("V(billable_work_entry).has(billable, \"{lit}\").count()");
        let decls = parse_decls(&line("q", 0xD3, &body)).unwrap_or_else(|e| panic!("{lit}: {e:?}"));
        let t = MethodTable::prepare_with(&glove(&basis, &cats, &claims), &[], &decls);
        assert_eq!(
            t.get(MethodKey([0xD3; 16])).unwrap().clone().unwrap_err(),
            PrepareError::UnknownValue {
                field: "billable".into(),
                literal: lit.into()
            },
            "{lit}"
        );
    }
}
