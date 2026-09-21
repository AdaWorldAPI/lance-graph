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
//! other two join cases: the former lowers a real `Filter::Semijoin` (the
//! fk gather) against a real partner-side `Program`; the latter lowers
//! `Agg::ScatterOrU32` (the one-to-many hop back) into a caller-owned
//! mask, then counts it with a second tiny program. `group_sum_cc` runs
//! the one-terminal `Agg::GroupSumI32` (`Terminal::GroupSumI32`, ONE
//! program) alongside the pre-existing K-program `lower_group_by` reading,
//! printing both METRIC lines so the fold is visible.

#[path = "duckdb/fixture.rs"]
mod fixture;

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::{
    execute_into, materialize_rows, scratch_words_for, words_for, Foreign, ForeignPlane, LaneRef,
    Out, Planes, Scratch, Terminal, Value,
};
use lance_graph_quack::{
    lower, lower_group_by, Agg, Cmp, Col, Filter, ForeignLane, ForeignMask, GroupBy, Query,
};

use fixture::col::{
    partner::COUNTRY, AMOUNT, COST_CENTER, DOC_ID, DOC_ID_U32, GL_ACCOUNT, PARTNER_ID, QTY, STATUS,
};
use fixture::DOC_ROWS;

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
}

fn print_metric(id: &str, m: &CaseMetrics) {
    let scratch_bytes = m.scratch_words * std::mem::size_of::<u64>();
    match m.programs {
        Some(k) => eprintln!(
            "METRIC case={id} ops={} scratch_words={} tile_words={} scratch_bytes={} \
             alloc_bytes_exec={} rows_materialized={} index_vec_len={} out_bytes={} \
             pair_relation_bytes={} programs={k}",
            m.ops,
            m.scratch_words,
            m.tile_words,
            scratch_bytes,
            m.alloc_bytes_exec,
            m.rows_materialized,
            m.index_vec_len,
            m.out_bytes,
            m.pair_relation_bytes,
        ),
        None => eprintln!(
            "METRIC case={id} ops={} scratch_words={} tile_words={} scratch_bytes={} \
             alloc_bytes_exec={} rows_materialized={} index_vec_len={} out_bytes={} \
             pair_relation_bytes={}",
            m.ops,
            m.scratch_words,
            m.tile_words,
            scratch_bytes,
            m.alloc_bytes_exec,
            m.rows_materialized,
            m.index_vec_len,
            m.out_bytes,
            m.pair_relation_bytes,
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
        Value::Scattered => panic!(
            "case {id}: run_query doesn't handle ScatterOrU32 — see join_count_docs_with_posted"
        ),
        Value::GroupSummed => panic!(
            "case {id}: run_query doesn't handle GroupSumI32 — see group_sum_cc's one-terminal arm"
        ),
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
// The join cases. `join_sum_country` (fk gather),
// `join_count_docs_with_posted` (the one-to-many hop back), and
// `join_group_sum_country` (the fk-keyed group-sum) are all real,
// one-or-two-program lowerings now — no open seam remains.
// ---------------------------------------------------------------------

/// `SELECT SUM(l.amount) FROM line l JOIN partner p ON p.rid=l.partner_id
/// WHERE l.status=1 AND p.country=3` — two programs: the partner-side
/// filter kept as a mask, handed to the line-side program as a
/// [`ForeignPlane`] its `Filter::Semijoin` gathers through.
#[test]
fn join_sum_country() {
    let cases = load_cases();
    let fx = fixture::generate();

    // Phase 1 (partner side): `country == 3` -> Keep.
    let partner_lanes = fx.partner.lanes();
    let partner_planes = partner_lanes.planes();
    let partner_program = lower(&Query {
        filter: Filter::cmp(COUNTRY, Cmp::EqU32(3)),
        agg: Agg::Rows,
    })
    .expect("lowers");
    let mut p_scratch =
        Scratch::for_program(&partner_program, partner_planes.n_rows).expect("carves");
    let p_tile_words = p_scratch.words();
    let p_scratch_words = scratch_words_for(p_tile_words, p_scratch.slots()).expect("sized");
    // Phase 1 is a `Terminal::Keep` — its kept bits land in this owned
    // `Out::Mask` buffer, never in a scratch slot.
    let mut partner_bits = vec![0u64; words_for(partner_planes.n_rows)];
    let before1 = BYTES.load(Ordering::Relaxed);
    let p_value = execute_into(
        &partner_program,
        &partner_planes,
        &Foreign::NONE,
        &mut p_scratch,
        Out::Mask(&mut partner_bits),
    )
    .expect("runs");
    let alloc1 = alloc_delta(before1);
    match p_value {
        Value::Mask(_) => {}
        other => panic!("partner-side filter did not Keep a mask: {other:?}"),
    }

    let foreign_plane = ForeignPlane {
        words: &partner_bits,
        rows: fixture::PARTNER_ROWS,
    };
    let foreign = Foreign {
        planes: &[foreign_plane],
        lanes: &[],
    };

    // Phase 2 (line side): `status == 1 AND Semijoin(partner_id, 0)` -> SUM.
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let line_filter = Filter::and([
        Filter::cmp(STATUS, Cmp::EqU32(1)),
        Filter::semijoin(PARTNER_ID, ForeignMask(0)),
    ]);
    let line_program = lower(&Query {
        filter: line_filter,
        agg: Agg::SumI32(AMOUNT),
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&line_program, planes.n_rows).expect("carves");
    let scratch_words = scratch_words_for(scratch.words(), scratch.slots()).expect("sized");
    let before2 = BYTES.load(Ordering::Relaxed);
    let value =
        execute_into(&line_program, &planes, &foreign, &mut scratch, Out::None).expect("runs");
    let alloc2 = alloc_delta(before2);

    let encoded = match value {
        Value::SumI64(s) => s.to_string(),
        other => panic!("expected a sum, got {other:?}"),
    };
    let out_bytes = encoded.len();
    let m = CaseMetrics {
        ops: partner_program.ops.len() + line_program.ops.len(),
        scratch_words: p_scratch_words + scratch_words,
        tile_words: p_tile_words,
        alloc_bytes_exec: alloc1 + alloc2,
        rows_materialized: 0,
        index_vec_len: 0,
        out_bytes,
        programs: Some(2),
        // No N×M pair-relation object exists anywhere in this case: the fk
        // resolves through one `MaskOp::Gather` read of the foreign plane
        // per line row, never a materialized `(line, partner)` pair list.
        pair_relation_bytes: 0,
    };
    print_metric("join_sum_country", &m);
    assert_case(&cases, "join_sum_country", &encoded);
}

/// `SELECT COUNT(*) FROM doc d WHERE EXISTS(SELECT 1 FROM line l WHERE
/// l.doc_id=d.rid AND l.status=1)` — the one-to-many hop BACK:
/// `Agg::ScatterOrU32` sets bit `doc_id` of an 8-word `Out::Mask` buffer for
/// every posted line, then a tiny doc-side program counts the buffer as a
/// resident plane.
#[test]
fn join_count_docs_with_posted() {
    let cases = load_cases();
    let fx = fixture::generate();
    let lanes = fx.line.lanes();
    let planes = lanes.planes();

    // Phase 1 (line side): `status == 1` -> ScatterOrU32(doc_id, DOC_ROWS).
    let line_program = lower(&Query {
        filter: Filter::cmp(STATUS, Cmp::EqU32(1)),
        agg: Agg::ScatterOrU32 {
            fk: DOC_ID_U32,
            out_rows: DOC_ROWS as u32,
        },
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&line_program, planes.n_rows).expect("carves");
    let tile_words = scratch.words();
    let scratch_words = scratch_words_for(tile_words, scratch.slots()).expect("sized");
    let mut scattered = vec![0u64; words_for(DOC_ROWS)];
    let before1 = BYTES.load(Ordering::Relaxed);
    let value1 = execute_into(
        &line_program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::Mask(&mut scattered),
    )
    .expect("runs");
    let alloc1 = alloc_delta(before1);
    assert_eq!(value1, Value::Scattered);
    // The demanded terminal RESULT of this phase is the 8-word buffer
    // itself — count it in bytes here rather than the final scalar answer.
    let scattered_bytes = scattered.len() * std::mem::size_of::<u64>();

    // Phase 2 (doc side): a tiny program reading `scattered` as a resident
    // plane and counting it — a second `Terminal::Count`, per the module
    // doc's "say which": this repo's own executor over a plain `Filter::Plane`,
    // not a bespoke `ndarray::simd::popcount_batch_u64` call, so the count
    // goes through the SAME validated, differential-tested path as every
    // other case in this suite.
    let doc_masks: [&[u64]; 1] = [&scattered];
    let doc_planes = Planes {
        n_rows: DOC_ROWS,
        masks: &doc_masks,
        lanes: &[],
    };
    let doc_program = lower(&Query {
        filter: Filter::plane(lance_graph_quack::Mask(0)),
        agg: Agg::Count,
    })
    .expect("lowers");
    let mut d_scratch = Scratch::for_program(&doc_program, DOC_ROWS).expect("carves");
    let d_scratch_words = scratch_words_for(d_scratch.words(), d_scratch.slots()).expect("sized");
    let before2 = BYTES.load(Ordering::Relaxed);
    let value2 = execute_into(
        &doc_program,
        &doc_planes,
        &Foreign::NONE,
        &mut d_scratch,
        Out::None,
    )
    .expect("runs");
    let alloc2 = alloc_delta(before2);

    let encoded = match value2 {
        Value::Count(c) => c.to_string(),
        other => panic!("expected a count, got {other:?}"),
    };
    let out_bytes = scattered_bytes + encoded.len();
    let m = CaseMetrics {
        ops: line_program.ops.len() + doc_program.ops.len(),
        scratch_words: scratch_words + d_scratch_words,
        tile_words,
        alloc_bytes_exec: alloc1 + alloc2,
        rows_materialized: 0,
        index_vec_len: 0,
        out_bytes,
        programs: Some(2),
        pair_relation_bytes: 0,
    };
    print_metric("join_count_docs_with_posted", &m);
    assert_case(&cases, "join_count_docs_with_posted", &encoded);
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
    };
    print_metric("join_group_sum_country", &m);
    assert_case(&cases, "join_group_sum_country", &encoded);
}
