//! The DuckDB ↔ Quack differential harness.
//!
//! DuckDB is the semantic ORACLE — `tests/duckdb/oracle.py` runs the SQL in
//! `tests/duckdb/cases.tsv` against the committed fixture and writes the
//! `expected` column. Every test below is the same query lowered through
//! [`lance_graph_quack`] and executed by `lance_graph_mask_risc::execute`,
//! encoded the same way, and compared. **Expected values are never
//! hand-edited** — see `tests/duckdb/README.txt`.
//!
//! `join_*_is_the_open_seam` tests are the honest exception: Quack has no
//! join lowering (`src/lib.rs`'s own status section names it — `src_mask →
//! hop → dst_mask` is `lance-graph-mask-risc`'s PR5 and there is no `hop` op
//! to lower to yet). Those three still load their oracle-computed `expected`
//! value — proving the oracle side is real and committed — then `todo!()`.
//! They are `#[should_panic]` by design, not disabled and not deleted.

#[path = "duckdb/fixture.rs"]
mod fixture;

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::{
    execute, materialize_rows, scratch_words_for, words_for, Operand, Planes, Scratch, Value,
};
use lance_graph_quack::{lower, lower_group_by, Agg, Cmp, Col, Filter, GroupBy, Query};

use fixture::col::{AMOUNT, COST_CENTER, DOC_ID, GL_ACCOUNT, QTY, STATUS};

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

/// Load `id`'s expected value without asserting anything — the `join_*`
/// cases use this to prove the oracle side is real before `todo!()`-ing the
/// Rust side.
fn expected_only<'a>(cases: &'a [Case], id: &str) -> &'a str {
    cases
        .iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("case {id} is not in cases.tsv"))
        .expected
        .as_str()
}

// ---------------------------------------------------------------------
// Execution + encoding — one place each, so the differential is actually
// comparing the same shape of answer on both sides.
// ---------------------------------------------------------------------

struct CaseMetrics {
    ops: usize,
    scratch_words: usize,
    alloc_bytes_exec: usize,
    rows_materialized: usize,
    index_vec_len: usize,
    out_bytes: usize,
    /// `Some(K)` for a grouped case — `ops`/`scratch_words`/`alloc_bytes_exec`
    /// are then the SUM over the phase-1 filter and all K phase-2 programs,
    /// per the spec's "record ops/allocs as the SUM over the K runs".
    programs: Option<usize>,
}

fn print_metric(id: &str, m: &CaseMetrics) {
    match m.programs {
        Some(k) => eprintln!(
            "METRIC case={id} ops={} scratch_words={} alloc_bytes_exec={} \
             rows_materialized={} index_vec_len={} out_bytes={} programs={k}",
            m.ops,
            m.scratch_words,
            m.alloc_bytes_exec,
            m.rows_materialized,
            m.index_vec_len,
            m.out_bytes
        ),
        None => eprintln!(
            "METRIC case={id} ops={} scratch_words={} alloc_bytes_exec={} \
             rows_materialized={} index_vec_len={} out_bytes={}",
            m.ops,
            m.scratch_words,
            m.alloc_bytes_exec,
            m.rows_materialized,
            m.index_vec_len,
            m.out_bytes
        ),
    }
}

/// Copy a kept mask's bits out of wherever `Value::Mask` says they landed —
/// a scratch slot, or (for a bare-plane filter) the plane itself.
fn mask_bits(planes: &Planes<'_>, scratch: &Scratch<'_>, op: Operand) -> Vec<u64> {
    match op {
        Operand::Scratch(i) => scratch.slot(i).expect("Keep names a written slot").to_vec(),
        Operand::Plane(p) => planes.masks[usize::from(p)].to_vec(),
    }
}

/// A plain scalar/projection query: lower, size scratch exactly, execute,
/// encode. The encoding matches `oracle.py`'s exactly — see
/// `tests/duckdb/README.txt`.
fn run_query(id: &str, planes: &Planes<'_>, filter: Filter, agg: Agg) -> (String, CaseMetrics) {
    let program = lower(&Query { filter, agg }).expect("lowers");
    let words = words_for(planes.n_rows);
    let slots = program.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(words, slots).expect("sized")];
    let scratch_words = buf.len();
    let mut scratch = Scratch::over(&mut buf, words, slots).expect("carves");

    let before = BYTES.load(Ordering::Relaxed);
    let value = execute(&program, planes, &mut scratch, None).expect("runs");
    let alloc_bytes_exec = alloc_delta(before);

    let (encoded, rows_materialized, index_vec_len) = match value {
        Value::Count(c) => (c.to_string(), 0, 0),
        Value::Bool(b) => (u8::from(b).to_string(), 0, 0),
        Value::SumI64(s) => (s.to_string(), 0, 0),
        Value::OptI32(Some(x)) => (x.to_string(), 0, 0),
        Value::OptI32(None) => {
            panic!("case {id}: min/max over an empty set — fixture is not what the case assumes")
        }
        Value::Mask(op) => {
            let bits = mask_bits(planes, &scratch, op);
            let rows = materialize_rows(&bits, planes.n_rows);
            let n = rows.len();
            let s = rows
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            (s, n, n)
        }
        Value::Blended => panic!("case {id}: no BlendI32 case in this suite"),
    };
    let out_bytes = encoded.len();
    (
        encoded,
        CaseMetrics {
            ops: program.ops.len(),
            scratch_words,
            alloc_bytes_exec,
            rows_materialized,
            index_vec_len,
            out_bytes,
            programs: None,
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

    let words = words_for(planes.n_rows);
    let f_slots = plan.filter.scratch_slots as usize;
    let mut f_buf = vec![0u64; scratch_words_for(words, f_slots).expect("sized")];
    let mut scratch_words_total = f_buf.len();
    let mut f_scratch = Scratch::over(&mut f_buf, words, f_slots).expect("carves");

    let before = BYTES.load(Ordering::Relaxed);
    let kept = execute(&plan.filter, planes, &mut f_scratch, None).expect("runs");
    let mut alloc_bytes_exec = alloc_delta(before);

    let kept_bits = match kept {
        Value::Mask(op) => mask_bits(planes, &f_scratch, op),
        other => panic!("case {id}: group filter did not Keep a mask: {other:?}"),
    };
    let widened_masks: Vec<&[u64]> = std::iter::once(kept_bits.as_slice()).collect();
    let widened = Planes {
        n_rows: planes.n_rows,
        masks: &widened_masks,
        lanes: planes.lanes,
    };

    let mut ops = plan.filter.ops.len();
    let mut pairs = Vec::with_capacity(plan.groups.len());
    for (k, prog) in plan.groups.iter().enumerate() {
        let g_slots = prog.scratch_slots as usize;
        let mut g_buf = vec![0u64; scratch_words_for(words, g_slots).expect("sized")];
        scratch_words_total += g_buf.len();
        let mut g_scratch = Scratch::over(&mut g_buf, words, g_slots).expect("carves");

        let before = BYTES.load(Ordering::Relaxed);
        let v = execute(prog, &widened, &mut g_scratch, None).expect("runs");
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
            alloc_bytes_exec,
            rows_materialized: 0,
            index_vec_len: 0,
            out_bytes,
            programs: Some(plan.groups.len()),
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
    let (actual, m) = run_group(
        "group_count_cc",
        &planes,
        filter,
        COST_CENTER,
        8,
        Agg::Count,
    );
    print_metric("group_count_cc", &m);
    assert_case(&cases, "group_count_cc", &actual);
}

#[test]
fn group_sum_cc() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let filter = Filter::cmp(STATUS, Cmp::EqU32(1));
    let (actual, m) = run_group(
        "group_sum_cc",
        &planes,
        filter,
        COST_CENTER,
        8,
        Agg::SumI32(AMOUNT),
    );
    print_metric("group_sum_cc", &m);
    assert_case(&cases, "group_sum_cc", &actual);
}

// ---------------------------------------------------------------------
// The 3 join cases — the open seam. Quack lowers filters and one-table
// aggregates; it has no join lowering (`src/lib.rs`'s `# Status` section
// names the gap explicitly). Each of these loads its oracle-computed
// `expected` — proving the DuckDB side is real and committed — then
// `todo!()`s, so the case is red until join lowering exists rather than
// silently absent from the suite.
// ---------------------------------------------------------------------

#[test]
#[should_panic(expected = "NO JOIN LOWERING")]
fn join_sum_country_is_the_open_seam() {
    let cases = load_cases();
    let expected = expected_only(&cases, "join_sum_country");
    assert!(
        !expected.is_empty(),
        "join_sum_country has no oracle value — run oracle.py first"
    );
    todo!("NO JOIN LOWERING: join_sum_country");
}

#[test]
#[should_panic(expected = "NO JOIN LOWERING")]
fn join_count_docs_with_posted_is_the_open_seam() {
    let cases = load_cases();
    let expected = expected_only(&cases, "join_count_docs_with_posted");
    assert!(
        !expected.is_empty(),
        "join_count_docs_with_posted has no oracle value — run oracle.py first"
    );
    todo!("NO JOIN LOWERING: join_count_docs_with_posted");
}

#[test]
#[should_panic(expected = "NO JOIN LOWERING")]
fn join_group_sum_country_is_the_open_seam() {
    let cases = load_cases();
    let expected = expected_only(&cases, "join_group_sum_country");
    assert!(
        !expected.is_empty(),
        "join_group_sum_country has no oracle value — run oracle.py first"
    );
    todo!("NO JOIN LOWERING: join_group_sum_country");
}
