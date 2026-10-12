//! D-CMH-1: methods of `BillableWorkEntry` (`0x0103`) bound once against the
//! SAP glove and compiled into Quack algebra, with R2IL as the parity frontend.
//!
//! Every method below is DATA declared in this file. `src/` knows no method
//! by name; adding one here adds no Rust (the falsifier for the claim that a
//! new method is a binding task, not a Rust task).
//!
//! The method keys and labels below are FIXTURES. Method identity is minted by
//! the method's owner (OGAR); nothing here mints one.
use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::method::{
    Answer, CallError, Cond, EffectClass, Fold, FoldValue, Frontend, Glove, Law, MethodBody,
    MethodKey, MethodSpec, MethodTable, PrepareError, ReadingId, SchemaSeal, Universe,
};
use lance_graph_mask_risc::{LaneRef, Planes};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_quack::bind::{table, Binder, BoundField, TableId};
use lance_graph_quack::{lower, Col, Mask};
use lance_graph_sap::bind::{billable_code, CatsBatch, ACTIVITY, BILLABLE, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

const COUNT_BILLABLE: MethodKey = MethodKey([0xB1; 16]);
const COUNT_BY_DURATION: MethodKey = MethodKey([0xD0; 16]);
const SUBMIT: MethodKey = MethodKey([0xA0; 16]);
const NOT_BILLABLE: MethodKey = MethodKey([0xB0; 16]);
const BILLABLE_HOURS: MethodKey = MethodKey([0xB2; 16]);

const fn query(filter: &'static [Cond], folds: &'static [Fold]) -> MethodBody {
    MethodBody::Query {
        filter,
        folds,
        law: Law::Values,
    }
}

/// FIXTURES: the reading id and population id a real glove is issued.
const READING: ReadingId = ReadingId(1);
const UNIVERSE: Universe = Universe(42);

fn pure(key: MethodKey, body: MethodBody) -> MethodSpec {
    MethodSpec {
        key,
        concept: BILLABLE_WORK_ENTRY,
        schema: SchemaSeal::of(&canonical_fields()),
        reading: READING,
        effect: EffectClass::Pure,
        body,
    }
}

fn scalars(v: &[i64]) -> Answer {
    Answer::Values(v.iter().map(|&n| FoldValue::Scalar(n)).collect())
}

const BILLABLE_TRUE: Cond = Cond::Eq {
    field: "billable",
    literal: "true",
};

fn specs() -> Vec<(&'static str, MethodSpec)> {
    vec![
        (
            "count_billable",
            pure(COUNT_BILLABLE, query(&[BILLABLE_TRUE], &[Fold::Count])),
        ),
        (
            "count_by_duration",
            pure(
                COUNT_BY_DURATION,
                query(
                    &[Cond::Eq {
                        field: "duration",
                        literal: "1",
                    }],
                    &[Fold::Count],
                ),
            ),
        ),
        (
            "submit_for_approval",
            MethodSpec {
                effect: EffectClass::ExternalMutate,
                ..pure(
                    SUBMIT,
                    MethodBody::Action {
                        action: "submit_for_approval",
                    },
                )
            },
        ),
        // Two more shapes, still data: a negated condition, and a body with
        // two folds over one selection.
        (
            "count_not_billable",
            pure(
                NOT_BILLABLE,
                query(
                    &[Cond::Ne {
                        field: "billable",
                        literal: "true",
                    }],
                    &[Fold::Count],
                ),
            ),
        ),
        (
            "billable_hours",
            pure(
                BILLABLE_HOURS,
                query(&[BILLABLE_TRUE], &[Fold::Count, Fold::Sum("duration")]),
            ),
        ),
    ]
}

/// Billing indicators cycle through the three documented values and one the
/// DTO does not document, which is neither billable nor not.
const INDICATORS: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Unknown"];
const HOURS_TEXT: [&str; 3] = ["8.50", "0.125", "2"];

fn indicator(i: usize) -> &'static str {
    INDICATORS[(i * 7 + i / 3) % 4]
}

fn batch(n: usize) -> CatsBatch {
    let base: Vec<Option<&str>> = include_str!("../../lance-graph-sap/fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let mut input: [Vec<Option<&str>>; FIELD_COUNT] = std::array::from_fn(|i| vec![base[i]; n]);
    for i in 0..n {
        input[BILLING][i] = Some(indicator(i));
        input[HOURS][i] = Some(HOURS_TEXT[i % 3]);
    }
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}

/// The oracle reads the indicator text and the DTO's documented values, never
/// the lane a method binds.
fn billable_rows(n: usize) -> impl Iterator<Item = usize> {
    (0..n).filter(|&i| billable_code(indicator(i)) == 1)
}

/// A test-local claim table that also maps `duration` onto SAP's decimal
/// hours (at the batch's own scale). FIXTURE: SAP's real claim on `duration`
/// is `Hypothesized`; a `Converted` grade needs the unit proof.
fn with_duration() -> Vec<FieldMap> {
    SAP_FIELDS
        .iter()
        .map(|m| FieldMap {
            grade: if m.canonical == "duration" {
                Grade::Exact
            } else {
                m.grade
            },
            ..*m
        })
        .collect()
}

fn glove<'a>(basis: &'a [String], binder: &'a dyn Binder, claims: &'a [FieldMap]) -> Glove<'a> {
    Glove {
        concept: BILLABLE_WORK_ENTRY,
        basis,
        claims,
        binder,
        table: TABLE,
        reading: READING,
        universe: UNIVERSE,
    }
}

fn table_for(binder: &dyn Binder, claims: &[FieldMap]) -> MethodTable {
    MethodTable::prepare(&glove(&canonical_fields(), binder, claims), &specs())
}

fn call(
    t: &MethodTable,
    batch: &CatsBatch,
    key: MethodKey,
    f: Frontend,
) -> Result<Answer, CallError> {
    let n = batch.len();
    let live = live_words(n);
    let lanes = batch.lanes();
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    t.call(key, &planes, UNIVERSE, f)
}

/// FAILS IF: either frontend stops counting exactly the rows the DTO documents
/// as billable, or the two frontends disagree.
#[test]
fn count_billable_agrees_across_quack_r2il_and_the_oracle() {
    for n in [1, 64, 65, 4097] {
        let batch = batch(n);
        let t = table_for(&CatsBinder::new(&batch), SAP_FIELDS);
        let key = t.resolve("count_billable").expect("label");
        let want = scalars(&[billable_rows(n).count() as i64]);
        assert_eq!(
            call(&t, &batch, key, Frontend::Quack).unwrap(),
            want,
            "n={n}"
        );
        assert_eq!(
            call(&t, &batch, key, Frontend::R2il).unwrap(),
            want,
            "n={n}"
        );
    }
    // Non-vacuity: some rows are billable, some are not, some are unknown.
    let o = billable_rows(4097).count();
    assert!(o > 0 && o < 4097);
}

/// FAILS IF: a new method declared only as data does not run, or its two
/// frontends disagree with the oracle. `count_not_billable` is a negated
/// condition; `billable_hours` is two folds over one selection.
#[test]
fn methods_declared_as_data_run_without_new_rust() {
    for n in [1, 65, 4097] {
        let batch = batch(n);
        let t = table_for(&CatsBinder::new(&batch), &with_duration());
        // `!= true` keeps the non-billable rows AND the undocumented ones.
        let not_billable = (0..n).filter(|&i| billable_code(indicator(i)) != 1).count() as i64;
        let LaneRef::I32(hours) = batch.lanes()[HOURS] else {
            panic!("hours is an I32 lane");
        };
        let billable_sum: i64 = billable_rows(n).map(|i| i64::from(hours[i])).sum();
        let cases = [
            ("count_not_billable", scalars(&[not_billable])),
            (
                "billable_hours",
                scalars(&[billable_rows(n).count() as i64, billable_sum]),
            ),
        ];
        for (label, want) in cases {
            let key = t.resolve(label).expect(label);
            for f in [Frontend::Quack, Frontend::R2il] {
                assert_eq!(
                    call(&t, &batch, key, f).unwrap(),
                    want,
                    "{label} {f:?} n={n}"
                );
            }
        }
    }
    // Non-vacuity: hours vary across billable rows, so a sum of the wrong
    // lane or of every row would differ.
    let batch = batch(65);
    let LaneRef::I32(hours) = batch.lanes()[HOURS] else {
        panic!()
    };
    let all: i64 = hours.iter().map(|&h| i64::from(h)).sum();
    let billable: i64 = billable_rows(65).map(|i| i64::from(hours[i])).sum();
    assert!(billable > 0 && billable < all);
}

/// The method's Quack query is the query a developer writes against the glove
/// directly, so both lower to one program.
#[test]
fn the_bound_query_is_the_one_quack_binds_from_names() {
    let batch = batch(65);
    let cats = CatsBinder::new(&batch);
    let t = table_for(&cats, SAP_FIELDS);
    let m = t.get(COUNT_BILLABLE).unwrap().as_ref().unwrap();
    let by_name = table(TABLE)
        .where_eq("billable", "true")
        .count()
        .bind(&cats)
        .unwrap();
    assert_eq!(m.queries(), vec![by_name.clone()]);
    assert_eq!(lower(&m.queries()[0]).unwrap(), lower(&by_name).unwrap());
}

/// The field mask is over the canonical basis, not the glove's lanes.
#[test]
fn required_names_the_canonical_positions() {
    let batch = batch(2);
    let t = table_for(&CatsBinder::new(&batch), &with_duration());
    let pos = |f: &str| canonical_fields().iter().position(|c| c == f).unwrap() as u8;
    let m = t.get(COUNT_BILLABLE).unwrap().as_ref().unwrap();
    assert!(m.required.has(pos("billable")));
    assert_eq!(m.required.count(), 1);
    let m = t.get(BILLABLE_HOURS).unwrap().as_ref().unwrap();
    assert!(m.required.has(pos("billable")) && m.required.has(pos("duration")));
    assert_eq!(m.required.count(), 2);
}

/// FAILS IF: a method whose field has only a hypothesized claim binds anyway.
/// `duration` is claimed by SAP `hours_logged`, graded `Hypothesized`.
#[test]
fn a_hypothesized_field_is_refused_at_preparation() {
    let batch = batch(2);
    let t = table_for(&CatsBinder::new(&batch), SAP_FIELDS);
    for key in [COUNT_BY_DURATION, BILLABLE_HOURS] {
        assert_eq!(
            t.get(key),
            Some(&Err(PrepareError::UnmappedRequired("duration")))
        );
    }
    assert!(matches!(
        call(&t, &batch, COUNT_BY_DURATION, Frontend::Quack),
        Err(CallError::Refused(PrepareError::UnmappedRequired(
            "duration"
        )))
    ));
    // Twin: with the claim proven, the refusal moves on to binding (a decimal
    // lane is not a code), so the availability check was what refused.
    let t = table_for(&CatsBinder::new(&batch), &with_duration());
    assert_eq!(
        t.get(COUNT_BY_DURATION),
        Some(&Err(PrepareError::KindMismatch("duration")))
    );
}

/// An action prepares, so the table can name it, and never runs as a pure call.
#[test]
fn an_action_is_named_and_never_run() {
    let batch = batch(2);
    let t = table_for(&CatsBinder::new(&batch), SAP_FIELDS);
    let key = t.resolve("submit_for_approval").unwrap();
    assert!(t.get(key).unwrap().is_ok());
    for f in [Frontend::Quack, Frontend::R2il] {
        assert!(matches!(call(&t, &batch, key, f), Err(CallError::NotPure)));
    }
}

#[test]
fn labels_resolve_once_and_unknown_names_resolve_to_nothing() {
    let batch = batch(2);
    let t = table_for(&CatsBinder::new(&batch), SAP_FIELDS);
    assert_eq!(t.resolve("count_billable"), Some(COUNT_BILLABLE));
    assert_eq!(t.resolve("billable"), None, "a field name is not a method");
    assert!(matches!(
        call(&t, &batch, MethodKey([0; 16]), Frontend::Quack),
        Err(CallError::UnknownMethod)
    ));
}

#[test]
fn malformed_specs_are_refused_by_name() {
    let batch = batch(2);
    let wrong_concept = MethodSpec {
        concept: BILLABLE_WORK_ENTRY + 1,
        ..pure(MethodKey([1; 16]), query(&[BILLABLE_TRUE], &[Fold::Count]))
    };
    let no_fold = pure(MethodKey([2; 16]), query(&[BILLABLE_TRUE], &[]));
    let unknown_field = pure(
        MethodKey([3; 16]),
        query(
            &[Cond::Eq {
                field: "colour",
                literal: "red",
            }],
            &[Fold::Count],
        ),
    );
    let unknown_value = pure(
        MethodKey([4; 16]),
        query(
            &[Cond::Eq {
                field: "billable",
                literal: "maybe",
            }],
            &[Fold::Count],
        ),
    );
    let pure_action = MethodSpec {
        effect: EffectClass::Pure,
        ..pure(
            MethodKey([5; 16]),
            MethodBody::Action {
                action: "submit_for_approval",
            },
        )
    };
    let specs = [
        ("wrong_concept", wrong_concept),
        ("no_fold", no_fold),
        ("unknown_field", unknown_field),
        ("unknown_value", unknown_value),
        ("pure_action", pure_action),
    ];
    let cats = CatsBinder::new(&batch);
    let t = MethodTable::prepare(&glove(&canonical_fields(), &cats, SAP_FIELDS), &specs);
    let refused = |k: u8| t.get(MethodKey([k; 16])).unwrap().clone().unwrap_err();
    assert!(matches!(refused(1), PrepareError::WrongConcept { .. }));
    assert_eq!(refused(2), PrepareError::NoFold);
    assert_eq!(refused(3), PrepareError::UnknownField("colour"));
    assert_eq!(
        refused(4),
        PrepareError::UnknownValue {
            field: "billable",
            literal: "maybe"
        }
    );
    assert_eq!(refused(5), PrepareError::EffectMismatch);
}

/// A binder that serves `billable` from the activity lane (also `U32`).
struct Swapped<'a>(CatsBinder<'a>);

impl Binder for Swapped<'_> {
    fn table(&self, name: &str) -> Option<TableId> {
        self.0.table(name)
    }
    fn live(&self, t: TableId) -> Mask {
        self.0.live(t)
    }
    fn field(&self, t: TableId, name: &str) -> Option<BoundField> {
        let mut f = self.0.field(t, name)?;
        if name == "billable" {
            f.col = Col(ACTIVITY as u16);
        }
        Some(f)
    }
    fn code(&self, t: TableId, col: Col, literal: &str) -> Option<u32> {
        // Keep the literal's billable code so only the lane differs.
        self.0
            .code(t, Col(BILLABLE as u16), literal)
            .filter(|_| col.0 == ACTIVITY as u16)
    }
}

/// FAILS IF: the answer does not depend on which lane the method bound. The
/// same method bound to the wrong lane must disagree with the oracle on both
/// frontends.
#[test]
fn a_swapped_lane_breaks_parity() {
    let n = 4097;
    let batch = batch(n);
    let t = table_for(&Swapped(CatsBinder::new(&batch)), SAP_FIELDS);
    let want = scalars(&[billable_rows(n).count() as i64]);
    for f in [Frontend::Quack, Frontend::R2il] {
        assert_ne!(call(&t, &batch, COUNT_BILLABLE, f).unwrap(), want, "{f:?}");
    }
}
