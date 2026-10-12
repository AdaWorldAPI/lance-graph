//! Shared fixture for `tests/compose.rs` and `tests/gremlin.rs`: the SAP CATS
//! glove of `billable_work_entry`, the same batch the other suites build, and
//! a row-loop oracle that reads the indicator text, never a bound lane.
#![allow(dead_code)]

use lance_graph_glove_parity::basis::{canonical_fields, FieldMap, Grade, SAP_FIELDS};
use lance_graph_glove_parity::method::{
    Answer, CallError, FoldValue, Frontend, Glove, MethodKey, MethodTable, ReadingId, SchemaSeal,
    Universe,
};
use lance_graph_mask_risc::{LaneRef, Planes};
use lance_graph_ogar::ogar_vocab::class_ids::BILLABLE_WORK_ENTRY;
use lance_graph_quack::bind::Binder;
use lance_graph_sap::bind::{billable_code, CatsBatch, BILLING, HOURS};
use lance_graph_sap::binder::{live_words, TABLE};
use lance_graph_sap::schema::{CatsSchema, FIELD_COUNT};

/// FIXTURES: the reading id and population id a real glove is issued.
pub const READING: ReadingId = ReadingId(1);
pub const UNIVERSE: Universe = Universe(42);
pub const SIZES: [usize; 4] = [1, 64, 65, 4097];

const INDICATORS: [&str; 4] = ["Billable", "Non-Billable", "Internal Cost", "Unknown"];
const HOURS_TEXT: [&str; 3] = ["8.50", "0.125", "2"];

pub fn indicator(i: usize) -> &'static str {
    INDICATORS[(i * 7 + i / 3) % 4]
}

pub fn batch(n: usize) -> CatsBatch {
    let base: Vec<Option<&str>> = include_str!("../../../lance-graph-sap/fixtures/cats.txt")
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

/// FIXTURE: proves SAP's `Hypothesized` claim on `duration`, as the other
/// suites do. Nothing in `src/` does this.
pub fn with_duration() -> Vec<FieldMap> {
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

pub fn seal() -> SchemaSeal {
    SchemaSeal::of(&canonical_fields())
}

pub fn glove<'a>(basis: &'a [String], binder: &'a dyn Binder, claims: &'a [FieldMap]) -> Glove<'a> {
    Glove {
        concept: BILLABLE_WORK_ENTRY,
        basis,
        claims,
        binder,
        table: TABLE,
        reading: READING,
        universe: UNIVERSE,
        relations: &[],
    }
}

pub fn call(
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

pub fn scalars(v: &[i64]) -> Answer {
    Answer::Values(v.iter().map(|&n| FoldValue::Scalar(n)).collect())
}

/// The oracle's rows: `(billing code from the indicator text, hours)`.
pub fn rows(batch: &CatsBatch) -> Vec<(u32, i64)> {
    let LaneRef::I32(hours) = batch.lanes()[HOURS] else {
        panic!("hours is an I32 lane")
    };
    (0..batch.len())
        .map(|i| (billable_code(indicator(i)), i64::from(hours[i])))
        .collect()
}

/// `(count, sum of hours)` over rows whose code satisfies `keep`.
pub fn count_sum(batch: &CatsBatch, keep: impl Fn(u32) -> bool) -> (i64, i64) {
    rows(batch)
        .into_iter()
        .filter(|(c, _)| keep(*c))
        .fold((0, 0), |(n, s), (_, h)| (n + 1, s + h))
}
