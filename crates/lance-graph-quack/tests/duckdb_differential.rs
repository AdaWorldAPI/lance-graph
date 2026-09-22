//! The DuckDB ↔ Quack differential harness.
//!
//! DuckDB is the semantic ORACLE — `tests/duckdb/oracle.py` runs the SQL in
//! `tests/duckdb/cases.tsv` against the committed fixture and writes the
//! `expected` column. Every test below is the same query lowered through
//! [`lance_graph_quack`] and executed by `lance_graph_mask_risc::execute`,
//! encoded the same way, and compared. **Expected values are never
//! hand-edited** — see `tests/duckdb/README.txt`.
//!
//! `join_group_sum_country` closes the last open seam: `SUM(l.amount)
//! GROUP BY p.country` is a group-sum key resolved through a FOREIGN
//! table (`p.country`, reached via `l.partner_id`) —
//! `Terminal::GroupSumViaI32` over `ndarray::simd::masked_group_sum_i32_via`,
//! ONE program reading `partner.country` as a caller-owned
//! [`Foreign::lanes`] entry, no two-hop materialisation and no K-program
//! spelling. `join_sum_country` and `join_count_docs_with_posted` are the
//! other two join cases: the former lowers a real `Filter::EqU32Via` — the
//! fk join predicate in FACTORED form, `p.country` read straight through
//! `l.partner_id` in ONE program, no foreign plane, no gathered mask; the
//! latter lowers `Agg::ScatterOrU32` (the one-to-many hop back) into a
//! caller-owned mask, then counts it with a second tiny program.
//! `group_sum_cc` runs the one-terminal `Agg::GroupSumI32`
//! (`Terminal::GroupSumI32`, ONE program) alongside the pre-existing
//! K-program `lower_group_by` reading, printing both METRIC lines so the
//! fold is visible.

#[path = "duckdb/fixture.rs"]
mod fixture;

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::{
    execute_into, materialize_rows, scratch_words_for, words_for, ExecError, Foreign, GroupFold,
    LaneRef, Out, Planes, Scratch, Terminal, Value,
};
use lance_graph_quack::{
    lower, lower_group_by, Agg, Cmp, Col, Filter, ForeignLane, GroupAddr, GroupAgg, GroupBy, Query,
};

use fixture::col::{AMOUNT, COST_CENTER, DOC_ID, DOC_ID_U32, GL_ACCOUNT, PARTNER_ID, QTY, STATUS};

// ---------------------------------------------------------------------
// The counting allocator — `no_alloc.rs`'s pattern (lance-graph-mask-risc),
// reused rather than reinvented. It counts every byte `alloc` hands out,
// for the WHOLE process; each case reads the counter before and after its
// own `execute` call(s) and reports the delta, never the running total.
// ---------------------------------------------------------------------

struct Counting;

static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn alloc_delta(before: usize) -> usize {
    BYTES.load(Ordering::Relaxed) - before
}

// ---------------------------------------------------------------------
// cases.tsv — hand-rolled TSV, no serde in this crate's dependency graph.
// ---------------------------------------------------------------------

struct Case {
    id: String,
    sql: String,
    expected: String,
}

fn load_cases() -> Vec<Case> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/duckdb/cases.tsv");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cases.tsv unreadable at {}: {e}", path.display()));
    let mut lines = content.lines();
    let header = lines.next().expect("cases.tsv has a header row");
    assert_eq!(
        header, "id\tsql\texpected",
        "cases.tsv header drifted from the id/sql/expected contract"
    );
    lines
        .filter(|l| !l.is_empty())
        .map(|line| {
            let mut parts = line.splitn(3, '\t');
            let id = parts
                .next()
                .unwrap_or_else(|| panic!("missing id column in row: {line:?}"))
                .to_string();
            let sql = parts
                .next()
                .unwrap_or_else(|| panic!("missing sql column in row: {line:?}"))
                .to_string();
            let expected = parts.next().unwrap_or("").to_string();
            Case { id, sql, expected }
        })
        .collect()
}

/// Assert `actual` matches the oracle's `expected` for `id`, with enough
/// context to diagnose a divergence without re-running anything.
fn assert_case(cases: &[Case], id: &str, actual: &str) {
    let case = cases
        .iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("case {id} is not in cases.tsv"));
    assert_eq!(
        actual, case.expected,
        "case {id} diverged from the DuckDB oracle\n  sql:     {}\n  rust:    {actual}\n  duckdb:  {}",
        case.sql, case.expected
    );
}

// ---------------------------------------------------------------------
// Execution + encoding — one place each, so the differential is actually
// comparing the same shape of answer on both sides.
// ---------------------------------------------------------------------

struct CaseMetrics {
    ops: usize,
    scratch_words: usize,
    /// The scratch slot width [`Scratch::for_program`] actually carved —
    /// `tile_words_for(n_rows)`, never `words_for(n_rows)`. For a
    /// multi-phase case this is the FIRST (filter/phase-1) scratch's width;
    /// `scratch_words` stays the sum over every phase, as before.
    tile_words: usize,
    alloc_bytes_exec: usize,
    rows_materialized: usize,
    index_vec_len: usize,
    out_bytes: usize,
    /// `Some(K)` for a grouped case — `ops`/`scratch_words`/`alloc_bytes_exec`
    /// are then the SUM over the phase-1 filter and all K phase-2 programs,
    /// per the spec's "record ops/allocs as the SUM over the K runs".
    programs: Option<usize>,
    /// Bytes an N×M pair-relation object would have cost — always `0` here.
    /// No such object is ever built: a join is a `Gather`/`ScatterOrU32`
    /// read against a caller-owned foreign mask, never a materialized
    /// row-pair list. Named explicitly (rather than left implicit) so the
    /// absence is a printed fact, not an inference from a missing field.
    pair_relation_bytes: usize,
    /// Bytes of POPULATION-SIZED state a fold carried that is neither the
    /// tile scratch nor the encoded answer — the accumulator of a
    /// `ScatterCountU32` (one bit per key of the universe). Printed so the
    /// one case that carries such state says so in numbers; every other
    /// case is `0`. The law's "no population intermediate" is checked
    /// against THIS field, not against `alloc_bytes_exec`.
    population_state_bytes: usize,
    /// Bytes of population-sized state the TEST built to hand the fold a
    /// view it does not have resident — a reordered copy of the lanes. This
    /// is materialisation in the harness, not in the terminal, and it is
    /// printed so a `population_state_bytes=0` next to it cannot be read as
    /// a zero-materialisation proof: the terminal is O(1) WHEN GIVEN that
    /// view; producing the view is the open question. `0` everywhere else.
    fixture_view_bytes: usize,
}

fn print_metric(id: &str, m: &CaseMetrics) {
    let scratch_bytes = m.scratch_words * std::mem::size_of::<u64>();
    match m.programs {
        Some(k) => eprintln!(
            "METRIC case={id} ops={} scratch_words={} tile_words={} scratch_bytes={} \
             alloc_bytes_exec={} rows_materialized={} index_vec_len={} out_bytes={} \
             pair_relation_bytes={} population_state_bytes={} fixture_view_bytes={} programs={k}",
            m.ops,
            m.scratch_words,
            m.tile_words,
            scratch_bytes,
            m.alloc_bytes_exec,
            m.rows_materialized,
            m.index_vec_len,
            m.out_bytes,
            m.pair_relation_bytes,
            m.population_state_bytes,
            m.fixture_view_bytes,
        ),
        None => eprintln!(
            "METRIC case={id} ops={} scratch_words={} tile_words={} scratch_bytes={} \
             alloc_bytes_exec={} rows_materialized={} index_vec_len={} out_bytes={} \
             pair_relation_bytes={} population_state_bytes={} fixture_view_bytes={}",
            m.ops,
            m.scratch_words,
            m.tile_words,
            scratch_bytes,
            m.alloc_bytes_exec,
            m.rows_materialized,
            m.index_vec_len,
            m.out_bytes,
            m.pair_relation_bytes,
            m.population_state_bytes,
            m.fixture_view_bytes,
        ),
    }
}

/// A plain scalar/projection query: lower, size scratch exactly, execute,
/// encode. The encoding matches `oracle.py`'s exactly — see
/// `tests/duckdb/README.txt`.
fn run_query(id: &str, planes: &Planes<'_>, filter: Filter, agg: Agg) -> (String, CaseMetrics) {
    let program = lower(&Query { filter, agg }).expect("lowers");
    let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
    let tile_words = scratch.words();
    let scratch_words = scratch_words_for(tile_words, scratch.slots()).expect("sized");

    // `Terminal::Keep` under tiled execution demands a caller `Out::Mask`
    // buffer — the kept bits land THERE, never in a scratch slot the caller
    // chases down by `Value::Mask`'s operand (see D-LGJ… the mask-risc
    // tiling note). Every other terminal folds into its own `Value`, so
    // `Out::None` is enough for it.
    let mut kept_buf = vec![0u64; words_for(planes.n_rows)];
    let out = if matches!(program.terminal, Terminal::Keep { .. }) {
        Out::Mask(&mut kept_buf)
    } else {
        Out::None
    };

    let before = BYTES.load(Ordering::Relaxed);
    let value = execute_into(&program, planes, &Foreign::NONE, &mut scratch, out).expect("runs");
    let alloc_bytes_exec = alloc_delta(before);

    let (encoded, rows_materialized, index_vec_len) = match value {
        Value::Count(c) => (c.to_string(), 0, 0),
        Value::Bool(b) => (u8::from(b).to_string(), 0, 0),
        Value::SumI64(s) => (s.to_string(), 0, 0),
        Value::OptI32(Some(x)) => (x.to_string(), 0, 0),
        Value::OptI32(None) => {
            panic!("case {id}: min/max over an empty set — fixture is not what the case assumes")
        }
        Value::Mask(_) => {
            let rows = materialize_rows(&kept_buf, planes.n_rows);
            let n = rows.len();
            let s = rows
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            (s, n, n)
        }
        Value::Blended => panic!("case {id}: no BlendI32 case in this suite"),
        Value::Scattered => panic!("case {id}: no ScatterOrU32 case in this suite"),
        Value::GroupSummed => panic!(
            "case {id}: run_query doesn't handle GroupSumI32 — see group_sum_cc's one-terminal arm"
        ),
        Value::GroupReduced => {
            panic!("case {id}: run_query doesn't handle GroupReduce — see run_group_reduce")
        }
    };
    let out_bytes = encoded.len();
    (
        encoded,
        CaseMetrics {
            ops: program.ops.len(),
            scratch_words,
            tile_words,
            alloc_bytes_exec,
            rows_materialized,
            index_vec_len,
            out_bytes,
            programs: None,
            pair_relation_bytes: 0,
            population_state_bytes: 0,
            fixture_view_bytes: 0,
        },
    )
}

/// A `GROUP BY`: run the two-phase [`lance_graph_quack::GroupPlan`] by hand
/// (there is no execution helper in `lance_graph_quack` — this crate only
/// LOWERS, per its own crate doc) — phase 1's kept mask becomes plane 0 of a
/// widened [`Planes`], then each of the K group programs runs against it.
fn run_group(
    id: &str,
    planes: &Planes<'_>,
    filter: Filter,
    key: Col,
    groups: u32,
    agg: Agg,
) -> (String, CaseMetrics) {
    let g = GroupBy {
        filter,
        key,
        groups,
        agg,
    };
    // `planes.masks` is empty in this suite, so plane 0 is free — the kept
    // mask lands there and nothing else needs to shift.
    let plan = lower_group_by(&g, 0).expect("lowers");
    assert_eq!(
        plan.filter_plane, 0,
        "case {id}: filter_plane drifted from where we place the kept mask"
    );

    let mut f_scratch = Scratch::for_program(&plan.filter, planes.n_rows).expect("carves");
    let f_tile_words = f_scratch.words();
    let mut scratch_words_total =
        scratch_words_for(f_tile_words, f_scratch.slots()).expect("sized");

    // Phase 1 is a `Terminal::Keep` — its kept bits land in this owned
    // `Out::Mask` buffer, never in a scratch slot (see `run_query`'s note).
    let mut kept_bits = vec![0u64; words_for(planes.n_rows)];
    let before = BYTES.load(Ordering::Relaxed);
    let kept = execute_into(
        &plan.filter,
        planes,
        &Foreign::NONE,
        &mut f_scratch,
        Out::Mask(&mut kept_bits),
    )
    .expect("runs");
    let mut alloc_bytes_exec = alloc_delta(before);

    match kept {
        Value::Mask(_) => {}
        other => panic!("case {id}: group filter did not Keep a mask: {other:?}"),
    }
    let widened_masks: Vec<&[u64]> = std::iter::once(kept_bits.as_slice()).collect();
    let widened = Planes {
        n_rows: planes.n_rows,
        masks: &widened_masks,
        lanes: planes.lanes,
    };

    let mut ops = plan.filter.ops.len();
    let mut pairs = Vec::with_capacity(plan.groups.len());
    for (k, prog) in plan.groups.iter().enumerate() {
        let mut g_scratch = Scratch::for_program(prog, planes.n_rows).expect("carves");
        scratch_words_total +=
            scratch_words_for(g_scratch.words(), g_scratch.slots()).expect("sized");

        let before = BYTES.load(Ordering::Relaxed);
        let v =
            execute_into(prog, &widened, &Foreign::NONE, &mut g_scratch, Out::None).expect("runs");
        alloc_bytes_exec += alloc_delta(before);
        ops += prog.ops.len();

        let val_str = match v {
            Value::Count(c) => c.to_string(),
            Value::SumI64(s) => s.to_string(),
            other => panic!("case {id}: unexpected group-{k} value: {other:?}"),
        };
        pairs.push(format!("{k}:{val_str}"));
    }
    let encoded = pairs.join(";");
    let out_bytes = encoded.len();
    (
        encoded,
        CaseMetrics {
            ops,
            scratch_words: scratch_words_total,
            tile_words: f_tile_words,
            alloc_bytes_exec,
            rows_materialized: 0,
            index_vec_len: 0,
            out_bytes,
            programs: Some(plan.groups.len()),
            pair_relation_bytes: 0,
            population_state_bytes: 0,
            fixture_view_bytes: 0,
        },
    )
}

/// A keyed reduction (`COUNT`/`MIN`/`MAX ... GROUP BY`) as ONE program:
/// `Agg::GroupReduce` → `Terminal::GroupReduce`, a K-slot `Out::I64` sink.
/// An empty MIN/MAX group still holds its [`GroupFold::seed`] and is encoded
/// as SQL `NULL`, exactly as `oracle.py` encodes DuckDB's `NULL`.
fn run_group_reduce(
    id: &str,
    planes: &Planes<'_>,
    foreign: &Foreign<'_>,
    filter: Filter,
    key: GroupAddr,
    agg: GroupAgg,
    groups: usize,
) -> (String, CaseMetrics) {
    let program = lower(&Query {
        filter,
        agg: Agg::GroupReduce { key, agg },
    })
    .expect("lowers");
    let fold = match program.terminal {
        Terminal::GroupReduce { fold, .. } => fold,
        other => panic!("case {id}: lowered onto {other:?}, not GroupReduce"),
    };
    let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
    let tile_words = scratch.words();
    let scratch_words = scratch_words_for(tile_words, scratch.slots()).expect("sized");
    let mut out = vec![0x5a5a_i64; groups];
    let before = BYTES.load(Ordering::Relaxed);
    let value =
        execute_into(&program, planes, foreign, &mut scratch, Out::I64(&mut out)).expect("runs");
    let alloc_bytes_exec = alloc_delta(before);
    assert_eq!(value, Value::GroupReduced, "case {id}");
    let empty_is_null = !matches!(fold, GroupFold::Count);
    let encoded = out
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            if empty_is_null && v == fold.seed() {
                format!("{k}:NULL")
            } else {
                format!("{k}:{v}")
            }
        })
        .collect::<Vec<_>>()
        .join(";");
    let out_bytes = encoded.len();
    (
        encoded,
        CaseMetrics {
            ops: program.ops.len(),
            scratch_words,
            tile_words,
            alloc_bytes_exec,
            rows_materialized: 0,
            index_vec_len: 0,
            out_bytes,
            programs: Some(1),
            pair_relation_bytes: 0,
            population_state_bytes: 0,
            fixture_view_bytes: 0,
        },
    )
}

// ---------------------------------------------------------------------
// The fixture drift guard.
// ---------------------------------------------------------------------

/// FAILS IF: the generator's bytes and the committed `tests/duckdb/data/`
/// CSVs have drifted apart — a change to `fixture.rs`'s generation order,
/// ranges, or CSV formatting that wasn't followed by regenerating the
/// committed files (and re-running `oracle.py`).
#[test]
fn committed_fixture_matches_generator() {
    let fx = fixture::generate();
    let partner_committed = include_bytes!("duckdb/data/partner.csv");
    let doc_committed = include_bytes!("duckdb/data/doc.csv");
    let line_committed = include_bytes!("duckdb/data/line.csv");
    assert_eq!(
        fx.partner_csv(),
        partner_committed,
        "partner.csv drifted from the generator"
    );
    assert_eq!(
        fx.doc_csv(),
        doc_committed,
        "doc.csv drifted from the generator"
    );
    assert_eq!(
        fx.line_csv(),
        line_committed,
        "line.csv drifted from the generator"
    );
}

/// Not run by default — regenerates `tests/duckdb/data/*.csv` on disk. Run
/// once with `cargo test -p lance-graph-quack --test duckdb_differential \
/// write_fixture -- --ignored` after any change to `fixture.rs`'s generation
/// logic, then re-run `oracle.py` (see `tests/duckdb/README.txt`) before
/// committing.
#[test]
#[ignore = "regenerates the committed fixture CSVs on disk; run explicitly"]
fn write_fixture() {
    let fx = fixture::generate();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/duckdb/data");
    std::fs::create_dir_all(&dir).expect("data dir");
    fx.write_csv(&dir).expect("write fixture CSVs");
}

// ---------------------------------------------------------------------
// The 11 scalar/projection cases.
// ---------------------------------------------------------------------

#[test]
fn sel_count_posted() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::cmp(STATUS, Cmp::EqU32(1));
    let (actual, m) = run_query("sel_count_posted", &planes, filter, Agg::Count);
    print_metric("sel_count_posted", &m);
    assert_case(&cases, "sel_count_posted", &actual);
}

#[test]
fn conj_count() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    // `cost_center < 4` over a lane restricted to `0..8` (3 bits) is exactly
    // "bit 2 clear" — `Cmp::MatchU32 { pattern: 0, care: 0b100 }`. There is
    // no ordered comparison over a `u32` lane in the IR (see `fixture.rs`'s
    // doc on `LineTable::doc_id`), so a ternary match is the honest spelling
    // here, not a workaround.
    let filter = Filter::and([
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        Filter::cmp(AMOUNT, Cmp::GtI32(1000)),
        Filter::cmp(
            COST_CENTER,
            Cmp::MatchU32 {
                pattern: 0,
                care: 0b100,
            },
        ),
    ]);
    let (actual, m) = run_query("conj_count", &planes, filter, Agg::Count);
    print_metric("conj_count", &m);
    assert_case(&cases, "conj_count", &actual);
}

#[test]
fn disj_count() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::or([
        Filter::cmp(STATUS, Cmp::EqU32(2)),
        Filter::cmp(AMOUNT, Cmp::LtI32(0)),
    ]);
    let (actual, m) = run_query("disj_count", &planes, filter, Agg::Count);
    print_metric("disj_count", &m);
    assert_case(&cases, "disj_count", &actual);
}

#[test]
fn not_count() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::negate(Filter::cmp(STATUS, Cmp::EqU32(1)));
    let (actual, m) = run_query("not_count", &planes, filter, Agg::Count);
    print_metric("not_count", &m);
    assert_case(&cases, "not_count", &actual);
}

#[test]
fn in_count() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::in_u32(GL_ACCOUNT, [400_000, 600_000]);
    let (actual, m) = run_query("in_count", &planes, filter, Agg::Count);
    print_metric("in_count", &m);
    assert_case(&cases, "in_count", &actual);
}

#[test]
fn sum_posted() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::cmp(STATUS, Cmp::EqU32(1));
    let (actual, m) = run_query("sum_posted", &planes, filter, Agg::SumI32(AMOUNT));
    print_metric("sum_posted", &m);
    assert_case(&cases, "sum_posted", &actual);
}

#[test]
fn min_posted_cc3() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::and([
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        Filter::cmp(COST_CENTER, Cmp::EqU32(3)),
    ]);
    let (actual, m) = run_query("min_posted_cc3", &planes, filter, Agg::MinI32(AMOUNT));
    print_metric("min_posted_cc3", &m);
    assert_case(&cases, "min_posted_cc3", &actual);
}

#[test]
fn max_posted_cc3() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::and([
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        Filter::cmp(COST_CENTER, Cmp::EqU32(3)),
    ]);
    let (actual, m) = run_query("max_posted_cc3", &planes, filter, Agg::MaxI32(AMOUNT));
    print_metric("max_posted_cc3", &m);
    assert_case(&cases, "max_posted_cc3", &actual);
}

#[test]
fn exists_neg() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::cmp(AMOUNT, Cmp::LtI32(-4990));
    let (actual, m) = run_query("exists_neg", &planes, filter, Agg::Any);
    print_metric("exists_neg", &m);
    assert_case(&cases, "exists_neg", &actual);
}

#[test]
fn rows_proj() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::and([
        Filter::cmp(STATUS, Cmp::EqU32(2)),
        Filter::cmp(QTY, Cmp::GtI32(45)),
    ]);
    let (actual, m) = run_query("rows_proj", &planes, filter, Agg::Rows);
    print_metric("rows_proj", &m);
    assert_case(&cases, "rows_proj", &actual);
}

#[test]
fn range_docid() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    // Two compares (`GeI32`/`LtI32`), not `Cmp::Range`: `Cmp::Range` is
    // minted only by `Filter::prefix_facet` from a validated
    // `OrderedLaneWitness` (D-DIAMOND-1 R2) — it is not a general-purpose
    // range spelling this suite is licensed to construct by hand, and
    // `doc_id` here is a plain generated index, not a sealed ordered lane.
    let filter = Filter::and([
        Filter::cmp(DOC_ID, Cmp::GeI32(128)),
        Filter::cmp(DOC_ID, Cmp::LtI32(256)),
    ]);
    let (actual, m) = run_query("range_docid", &planes, filter, Agg::Count);
    print_metric("range_docid", &m);
    assert_case(&cases, "range_docid", &actual);
}

// ---------------------------------------------------------------------
// The 2 GROUP BY cases.
// ---------------------------------------------------------------------

#[test]
fn group_count_cc() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::cmp(STATUS, Cmp::EqU32(1));

    // Folded: ONE program, `Terminal::GroupReduce { fold: Count }`.
    let (folded, m) = run_group_reduce(
        "group_count_cc",
        &planes,
        &Foreign::NONE,
        filter.clone(),
        GroupAddr::Local(COST_CENTER),
        GroupAgg::Count,
        8,
    );
    print_metric("group_count_cc", &m);
    assert_case(&cases, "group_count_cc", &folded);

    // The K-program forest, kept as the comparison METRIC.
    let (actual_k, m_k) = run_group(
        "group_count_cc",
        &planes,
        filter,
        COST_CENTER,
        8,
        Agg::Count,
    );
    print_metric("group_count_cc_kprogram", &m_k);
    assert_case(&cases, "group_count_cc", &actual_k);
}

/// `MIN(amount) GROUP BY cost_center` over posted lines — one
/// `Terminal::GroupReduce { fold: MinI32 }` program.
#[test]
fn group_min_cc() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let (actual, m) = run_group_reduce(
        "group_min_cc",
        &planes,
        &Foreign::NONE,
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        GroupAddr::Local(COST_CENTER),
        GroupAgg::MinI32(AMOUNT),
        8,
    );
    print_metric("group_min_cc", &m);
    assert_case(&cases, "group_min_cc", &actual);
}

/// `MAX(amount) GROUP BY cost_center` over posted lines.
#[test]
fn group_max_cc() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let (actual, m) = run_group_reduce(
        "group_max_cc",
        &planes,
        &Foreign::NONE,
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        GroupAddr::Local(COST_CENTER),
        GroupAgg::MaxI32(AMOUNT),
        8,
    );
    print_metric("group_max_cc", &m);
    assert_case(&cases, "group_max_cc", &actual);
}

/// `MAX(amount) GROUP BY cost_center` over lines with `status=2`, `qty>45`
/// and `cost_center IN (0,1,2)` — a filter tight enough that some groups are EMPTY,
/// so the SQL `NULL` encoding of an empty MIN/MAX group is exercised against
/// the fold's seed rather than assumed.
#[test]
fn group_max_cc_sparse() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let (actual, m) = run_group_reduce(
        "group_max_cc_sparse",
        &planes,
        &Foreign::NONE,
        Filter::and([
            Filter::cmp(STATUS, Cmp::EqU32(2)),
            Filter::cmp(QTY, Cmp::GtI32(45)),
            Filter::or([
                Filter::cmp(COST_CENTER, Cmp::EqU32(0)),
                Filter::cmp(COST_CENTER, Cmp::EqU32(1)),
                Filter::cmp(COST_CENTER, Cmp::EqU32(2)),
            ]),
        ]),
        GroupAddr::Local(COST_CENTER),
        GroupAgg::MaxI32(AMOUNT),
        8,
    );
    assert!(
        actual.contains(":NULL"),
        "fixture must leave some group empty or this case tests nothing: {actual}"
    );
    print_metric("group_max_cc_sparse", &m);
    assert_case(&cases, "group_max_cc_sparse", &actual);
}

#[test]
fn group_sum_cc() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::cmp(STATUS, Cmp::EqU32(1));

    // The one-terminal spelling: ONE program, `Terminal::GroupSumI32`, no
    // K-program loop.
    let program = lower(&Query {
        filter: filter.clone(),
        agg: Agg::GroupSumI32 {
            key: COST_CENTER,
            val: AMOUNT,
        },
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
    let tile_words = scratch.words();
    let scratch_words = scratch_words_for(tile_words, scratch.slots()).expect("sized");
    let mut out = [0i64; 8];
    let before = BYTES.load(Ordering::Relaxed);
    let value = execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::I64(&mut out),
    )
    .expect("runs");
    let alloc_bytes_exec = alloc_delta(before);
    assert_eq!(value, Value::GroupSummed);
    let encoded = out
        .iter()
        .enumerate()
        .map(|(k, v)| format!("{k}:{v}"))
        .collect::<Vec<_>>()
        .join(";");
    let out_bytes = encoded.len();
    let m = CaseMetrics {
        ops: program.ops.len(),
        scratch_words,
        tile_words,
        alloc_bytes_exec,
        rows_materialized: 0,
        index_vec_len: 0,
        out_bytes,
        programs: Some(1),
        pair_relation_bytes: 0,
        population_state_bytes: 0,
        fixture_view_bytes: 0,
    };
    print_metric("group_sum_cc", &m);
    assert_case(&cases, "group_sum_cc", &encoded);

    // The pre-existing K-program (`lower_group_by`) spelling, run and
    // printed alongside so the fold from K programs to one is visible in
    // the METRIC output rather than only in the source diff.
    let (actual_k, m_k) = run_group(
        "group_sum_cc",
        &planes,
        filter,
        COST_CENTER,
        8,
        Agg::SumI32(AMOUNT),
    );
    print_metric("group_sum_cc_kprogram", &m_k);
    assert_case(&cases, "group_sum_cc", &actual_k);
}

// ---------------------------------------------------------------------
// The join cases. `join_sum_country` (the fk join predicate, factored),
// `join_count_docs_with_posted` (the one-to-many hop back), and
// `join_group_sum_country` (the fk-keyed group-sum) are all real,
// ONE-program lowerings now — no open seam, no forbidden intermediate
// mask remains.
// ---------------------------------------------------------------------

/// `SELECT SUM(l.amount) FROM line l JOIN partner p ON p.rid=l.partner_id
/// WHERE l.status=1 AND p.country=3` — ONE program: `p.country` is read
/// straight through `l.partner_id` by [`Filter::EqU32Via`]
/// (`Pred::EqU32Via`'s `foreign.lanes[key][fk[i]] == v`), so no partner-side
/// `Program` is ever run, no partner kept-mask is ever materialised, and no
/// `ForeignPlane`/`MaskOp::Gather` exists anywhere in this case — a mask
/// produced only because another fold needed it is forbidden intermediate
/// state.
#[test]
fn join_sum_country() {
    let cases = load_cases();
    let fx = fixture::generate();

    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let country_lane = [LaneRef::U32(&fx.partner.country)];
    let foreign = Foreign {
        planes: &[],
        lanes: &country_lane,
    };

    let line_filter = Filter::and([
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        Filter::eq_u32_via(PARTNER_ID, ForeignLane(0), 3),
    ]);
    let line_program = lower(&Query {
        filter: line_filter,
        agg: Agg::SumI32(AMOUNT),
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&line_program, planes.n_rows).expect("carves");
    let tile_words = scratch.words();
    let scratch_words = scratch_words_for(tile_words, scratch.slots()).expect("sized");
    let before = BYTES.load(Ordering::Relaxed);
    let value =
        execute_into(&line_program, &planes, &foreign, &mut scratch, Out::None).expect("runs");
    let alloc_bytes = alloc_delta(before);

    let encoded = match value {
        Value::SumI64(s) => s.to_string(),
        other => panic!("expected a sum, got {other:?}"),
    };
    let out_bytes = encoded.len();
    let m = CaseMetrics {
        ops: line_program.ops.len(),
        scratch_words,
        tile_words,
        alloc_bytes_exec: alloc_bytes,
        rows_materialized: 0,
        index_vec_len: 0,
        out_bytes,
        programs: Some(1),
        // No N×M pair-relation object exists anywhere in this case: the fk
        // resolves through one `Pred::EqU32Via` read of the foreign VALUE
        // lane per line row, never a materialized `(line, partner)` pair
        // list — and, since the partner-side plane the old two-program
        // shape built is gone too, no foreign MASK ever exists either.
        pair_relation_bytes: 0,
        population_state_bytes: 0,
        fixture_view_bytes: 0,
    };
    print_metric("join_sum_country", &m);
    assert_case(&cases, "join_sum_country", &encoded);
}

/// `SELECT COUNT(*) FROM doc d WHERE EXISTS(SELECT 1 FROM line l WHERE
/// l.doc_id=d.rid AND l.status=1)` — `COUNT(DISTINCT doc_id)` over the
/// posted lines. ONE spelling: [`Agg::CountDistinctOrderedU32`] →
/// `Terminal::CountKeyRunsU32`, whose precondition is a key lane in key
/// order — the T0 address projection of lines under their doc.
///
/// - On the fixture AS GENERATED (`doc_id` random, no resident doc-major
///   projection) the executor REFUSES the program
///   (`ExecError::LaneNotOrdered`). That is the ruled outcome: the logical
///   query is valid, this physical lowering is not, and no seen-set is
///   allocated in its place. The `1,2,1` falsifier lives in mask-risc
///   `tests/distinct.rs`.
/// - GIVEN a doc-ORDERED view of the same lines, the fold answers the same
///   DuckDB 511 with two words of state. The view is built by the test
///   (`fixture_view_bytes`, a reordered copy): a semantic proof of the
///   terminal, not a zero-materialisation proof of the query on this
///   fixture.
///
/// The old shape — `ScatterOrU32` into a doc bitmap read back by a second
/// `Count` program — is gone, and so is its one-program successor
/// (`ScatterCountU32`): a population seen-set is not the fallback for a
/// lane that merely happens to be unordered.
#[test]
fn join_count_docs_with_posted() {
    let cases = load_cases();
    let fx = fixture::generate();

    // ── Arm 1: the generated (unordered) layout is REFUSED. ──
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let runs = lower(&Query {
        filter: Filter::cmp(STATUS, Cmp::EqU32(1)),
        agg: Agg::CountDistinctOrderedU32 { key: DOC_ID_U32 },
    })
    .expect("lowers");
    assert!(matches!(runs.terminal, Terminal::CountKeyRunsU32 { .. }));
    let mut r_scratch = Scratch::for_program(&runs, planes.n_rows).expect("carves");
    let refused = execute_into(&runs, &planes, &Foreign::NONE, &mut r_scratch, Out::None);
    assert_eq!(
        refused,
        Err(ExecError::LaneNotOrdered { lane: DOC_ID_U32.0 }),
        "an unordered key lane must be refused, never folded through a seen-set"
    );
    eprintln!(
        "REFUSED case=join_count_docs_with_posted terminal=CountKeyRunsU32 \
         reason=LaneNotOrdered population_state_bytes=0"
    );

    // ── Arm 2: an ORDERED view handed to CountKeyRunsU32. The reorder below
    // is the test's own materialisation, counted as `fixture_view_bytes`. ──
    let mut order: Vec<usize> = (0..fx.line.doc_id_u32.len()).collect();
    order.sort_by_key(|&i| fx.line.doc_id_u32[i]); // stable: runs, not a resort
    let by = |v: &[u32]| -> Vec<u32> { order.iter().map(|&i| v[i]).collect() };
    let by_i = |v: &[i32]| -> Vec<i32> { order.iter().map(|&i| v[i]).collect() };
    let ordered = fixture::LineTable {
        doc_id: by_i(&fx.line.doc_id),
        doc_id_u32: by(&fx.line.doc_id_u32),
        partner_id: by(&fx.line.partner_id),
        amount: by_i(&fx.line.amount),
        qty: by_i(&fx.line.qty),
        status: by(&fx.line.status),
        cost_center: by(&fx.line.cost_center),
        gl_account: by(&fx.line.gl_account),
    };
    let c_lanes = ordered.lanes();
    let c_planes = c_lanes.planes();
    let mut c_scratch = Scratch::for_program(&runs, c_planes.n_rows).expect("carves");
    let c_tile_words = c_scratch.words();
    let c_scratch_words = scratch_words_for(c_tile_words, c_scratch.slots()).expect("sized");
    let before = BYTES.load(Ordering::Relaxed);
    let c_value =
        execute_into(&runs, &c_planes, &Foreign::NONE, &mut c_scratch, Out::None).expect("runs");
    let c_alloc = alloc_delta(before);
    let c_encoded = match c_value {
        Value::Count(c) => c.to_string(),
        other => panic!("expected a count, got {other:?}"),
    };
    let m = CaseMetrics {
        ops: runs.ops.len(),
        scratch_words: c_scratch_words,
        tile_words: c_tile_words,
        alloc_bytes_exec: c_alloc,
        rows_materialized: 0,
        index_vec_len: 0,
        out_bytes: c_encoded.len(),
        programs: Some(1),
        pair_relation_bytes: 0,
        population_state_bytes: 0,
        // What the TEST built to hand the fold an ordered view: the
        // permutation plus eight reordered 4-byte lanes. Not the terminal's
        // state — and not evidence that a resident doc-major view exists.
        fixture_view_bytes: order.len() * (std::mem::size_of::<usize>() + 8 * 4),
    };
    print_metric("join_count_docs_with_posted_ordered_view_given", &m);
    // A distinct count is permutation-invariant: same DuckDB row, same answer.
    assert_case(&cases, "join_count_docs_with_posted", &c_encoded);
}

/// `SELECT p.country, SUM(l.amount) FROM line l JOIN partner p ON
/// p.rid=l.partner_id WHERE l.status=1 GROUP BY p.country ORDER BY
/// p.country` — a group-sum whose KEY lives on the FOREIGN table
/// (`p.country`, reached through `l.partner_id`). ONE program:
/// `Agg::GroupSumViaI32` lowers to `Terminal::GroupSumViaI32`, reading
/// `partner.country` directly as a caller-owned [`Foreign::lanes`] entry —
/// no partner-side filter program, no K-program loop, and no materialised
/// remapped key lane between the two hops.
#[test]
fn join_group_sum_country() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();

    let program = lower(&Query {
        filter: Filter::cmp(STATUS, Cmp::EqU32(1)),
        agg: Agg::GroupSumViaI32 {
            fk: PARTNER_ID,
            key: ForeignLane(0),
            val: AMOUNT,
        },
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
    let tile_words = scratch.words();
    let scratch_words = scratch_words_for(tile_words, scratch.slots()).expect("sized");

    let country_lane = [LaneRef::U32(&fx.partner.country)];
    let foreign = Foreign {
        planes: &[],
        lanes: &country_lane,
    };
    let mut out = [0i64; 8];
    let before = BYTES.load(Ordering::Relaxed);
    let value = execute_into(
        &program,
        &planes,
        &foreign,
        &mut scratch,
        Out::I64(&mut out),
    )
    .expect("runs");
    let alloc_bytes_exec = alloc_delta(before);
    assert_eq!(value, Value::GroupSummed);

    let encoded = out
        .iter()
        .enumerate()
        .map(|(k, v)| format!("{k}:{v}"))
        .collect::<Vec<_>>()
        .join(";");
    let out_bytes = encoded.len();
    let m = CaseMetrics {
        ops: program.ops.len(),
        scratch_words,
        tile_words,
        alloc_bytes_exec,
        rows_materialized: 0,
        index_vec_len: 0,
        out_bytes,
        programs: Some(1),
        // No N×M pair-relation object anywhere: the fk resolves through
        // ONE `masked_group_sum_i32_via` call per selected row, reading
        // `partner.country` in place — never a materialized `(line,
        // partner)` pair list or a remapped key lane.
        pair_relation_bytes: 0,
        population_state_bytes: 0,
        fixture_view_bytes: 0,
    };
    print_metric("join_group_sum_country", &m);
    assert_case(&cases, "join_group_sum_country", &encoded);
}

/// `COUNT(*) GROUP BY p.country` over posted lines — the key reached through
/// `l.partner_id`, fused in ONE `GroupReduce` program (no remapped key lane).
#[test]
fn join_group_count_country() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let country_lane = [LaneRef::U32(&fx.partner.country)];
    let foreign = Foreign {
        planes: &[],
        lanes: &country_lane,
    };
    let (actual, m) = run_group_reduce(
        "join_group_count_country",
        &planes,
        &foreign,
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        GroupAddr::Via {
            fk: PARTNER_ID,
            key: ForeignLane(0),
        },
        GroupAgg::Count,
        8,
    );
    print_metric("join_group_count_country", &m);
    assert_case(&cases, "join_group_count_country", &actual);
}

/// `MIN(l.amount) GROUP BY p.country` — the via-keyed MIN fold.
#[test]
fn join_group_min_country() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let country_lane = [LaneRef::U32(&fx.partner.country)];
    let foreign = Foreign {
        planes: &[],
        lanes: &country_lane,
    };
    let (actual, m) = run_group_reduce(
        "join_group_min_country",
        &planes,
        &foreign,
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        GroupAddr::Via {
            fk: PARTNER_ID,
            key: ForeignLane(0),
        },
        GroupAgg::MinI32(AMOUNT),
        8,
    );
    print_metric("join_group_min_country", &m);
    assert_case(&cases, "join_group_min_country", &actual);
}
