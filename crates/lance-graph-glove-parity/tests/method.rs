//! D-CMH-1, first slice: a method of `BillableWorkEntry` (`0x0103`) bound once
//! against the SAP glove and run through Quack and R2IL.
//!
//! The method keys and labels below are FIXTURES. Method identity is minted by
//! the method's owner (OGAR); nothing here mints one.
use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::method::{
    CallError, EffectClass, Frontend, MethodBody, MethodKey, MethodSpec, MethodTable, PrepareError,
};
use lance_graph_mask_risc::Planes;
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_quack::bind::{table, Binder, BoundField, TableId};
use lance_graph_quack::{lower, Col, Mask};
use lance_graph_sap::bind::{billable_code, CatsBatch, ACTIVITY, BILLABLE, BILLING};
use lance_graph_sap::binder::{live_words, CatsBinder, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

const COUNT_BILLABLE: MethodKey = MethodKey([0xB1; 16]);
const COUNT_BY_DURATION: MethodKey = MethodKey([0xD0; 16]);
const SUBMIT: MethodKey = MethodKey([0xA0; 16]);

fn specs() -> [(&'static str, MethodSpec); 3] {
    [
        (
            "count_billable",
            MethodSpec {
                key: COUNT_BILLABLE,
                concept: BILLABLE_WORK_ENTRY,
                effect: EffectClass::Pure,
                body: MethodBody::CountWhereEq {
                    field: "billable",
                    literal: "true",
                },
            },
        ),
        (
            "count_by_duration",
            MethodSpec {
                key: COUNT_BY_DURATION,
                concept: BILLABLE_WORK_ENTRY,
                effect: EffectClass::Pure,
                body: MethodBody::CountWhereEq {
                    field: "duration",
                    literal: "1",
                },
            },
        ),
        (
            "submit_for_approval",
            MethodSpec {
                key: SUBMIT,
                concept: BILLABLE_WORK_ENTRY,
                effect: EffectClass::ExternalMutate,
                body: MethodBody::Action {
                    action: "submit_for_approval",
                },
            },
        ),
    ]
}

/// Billing indicators cycle through the three documented values and one the
/// DTO does not document, which is neither billable nor not.
const INDICATORS: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Unknown"];

fn batch(n: usize) -> CatsBatch {
    let base: Vec<Option<&str>> = include_str!("../../lance-graph-sap/fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let mut input: [Vec<Option<&str>>; FIELD_COUNT] = std::array::from_fn(|i| vec![base[i]; n]);
    for (i, v) in input[BILLING].iter_mut().enumerate() {
        *v = Some(INDICATORS[(i * 7 + i / 3) % 4]);
    }
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}

/// The oracle reads the indicator text and the DTO's documented values, never
/// the lane the method binds.
fn oracle(n: usize) -> i64 {
    (0..n)
        .filter(|&i| billable_code(INDICATORS[(i * 7 + i / 3) % 4]) == 1)
        .count() as i64
}

fn table_for(binder: &dyn Binder, claims: &[FieldMap]) -> MethodTable {
    MethodTable::prepare(
        BILLABLE_WORK_ENTRY,
        &canonical_fields(),
        claims,
        binder,
        TABLE,
        &specs(),
    )
}

fn call(t: &MethodTable, batch: &CatsBatch, key: MethodKey, f: Frontend) -> Result<i64, CallError> {
    let n = batch.len();
    let live = live_words(n);
    let lanes = batch.lanes();
    let planes = Planes {
        n_rows: n,
        masks: &[live.as_slice()],
        lanes: &lanes,
    };
    t.call(key, &planes, f)
}

/// FAILS IF: either frontend stops counting exactly the rows the DTO documents
/// as billable, or the two frontends disagree.
#[test]
fn count_billable_agrees_across_quack_r2il_and_the_oracle() {
    for n in [1, 64, 65, 4097] {
        let batch = batch(n);
        let t = table_for(&CatsBinder::new(&batch), SAP_FIELDS);
        let key = t.resolve("count_billable").expect("label");
        let want = oracle(n);
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
    let o = oracle(4097);
    assert!(o > 0 && o < 4097);
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
    assert_eq!(m.query().unwrap(), by_name);
    assert_eq!(
        lower(&m.query().unwrap()).unwrap(),
        lower(&by_name).unwrap()
    );
}

/// The field mask is over the canonical basis, not the glove's lanes.
#[test]
fn required_names_the_canonical_position() {
    let batch = batch(2);
    let t = table_for(&CatsBinder::new(&batch), SAP_FIELDS);
    let m = t.get(COUNT_BILLABLE).unwrap().as_ref().unwrap();
    let pos = canonical_fields()
        .iter()
        .position(|f| f == "billable")
        .unwrap() as u8;
    assert!(m.required.has(pos));
    assert_eq!(m.required.count(), 1);
}

/// FAILS IF: a method whose field has only a hypothesized claim binds anyway.
/// `duration` is claimed by SAP `hours_logged`, graded `Hypothesized`.
#[test]
fn a_hypothesized_field_is_refused_at_preparation() {
    let batch = batch(2);
    let t = table_for(&CatsBinder::new(&batch), SAP_FIELDS);
    assert_eq!(
        t.get(COUNT_BY_DURATION),
        Some(&Err(PrepareError::UnmappedRequired("duration")))
    );
    assert!(matches!(
        call(&t, &batch, COUNT_BY_DURATION, Frontend::Quack),
        Err(CallError::Refused(PrepareError::UnmappedRequired(
            "duration"
        )))
    ));
    // Twin: promote the claim and the refusal moves on to binding, so the
    // availability check, not the binder, was what refused.
    let promoted: Vec<FieldMap> = SAP_FIELDS
        .iter()
        .map(|m| FieldMap {
            grade: if m.canonical == "duration" {
                Grade::Exact
            } else {
                m.grade
            },
            ..*m
        })
        .collect();
    let t = table_for(&CatsBinder::new(&batch), &promoted);
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
fn a_spec_of_another_concept_is_refused() {
    let batch = batch(2);
    let mut s = specs();
    s[0].1.concept = BILLABLE_WORK_ENTRY + 1;
    let t = MethodTable::prepare(
        BILLABLE_WORK_ENTRY,
        &canonical_fields(),
        SAP_FIELDS,
        &CatsBinder::new(&batch),
        TABLE,
        &s,
    );
    assert!(matches!(
        t.get(COUNT_BILLABLE),
        Some(Err(PrepareError::WrongConcept { .. }))
    ));
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
    for f in [Frontend::Quack, Frontend::R2il] {
        assert_ne!(
            call(&t, &batch, COUNT_BILLABLE, f).unwrap(),
            oracle(n),
            "{f:?}"
        );
    }
}
