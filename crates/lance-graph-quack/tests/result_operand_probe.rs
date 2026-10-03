//! Probe: can a completed group result be read per row by a later pass over
//! the source population, using only existing machinery?
//!
//! # The question
//!
//! Phase 1 folds `line` by `partner_id` into a K-slot Count sink. Phase 2
//! runs over `line` again and needs, for every line, the count of its own
//! partner (`count[partner_id[i]]`). Is the only missing piece that the sink
//! cannot be handed to phase 2 as a lane, or does phase 2 need more structure
//! (key binding, lifetime, shape metadata, a handle)?
//!
//! # The knife
//!
//! Between the phases this test makes ONE deliberate, test-only copy: the
//! `i64` Count sink is narrowed, with a checked conversion, into a
//! caller-owned `[u32; 64]`. That puts the produced values in a lane kind the
//! IR already accepts as a foreign lane, so the test asks only:
//!
//! > once the values sit in a supported lane, does phase 2 compose?
//!
//! It deliberately does NOT test scalar width (`i64` lanes) or provenance:
//! the copy discards both. Those are classified separately in the report
//! (`.claude/plans/population-law-crosscheck-v1.md`).
//!
//! # Phase 2, two consumers, both existing
//!
//! - **predicate**: `Filter::EqU32Via { fk: partner_id, key: <count lane>, v }`
//!   — "lines whose partner has exactly v posted lines";
//! - **key**: `GroupAddr::Via { fk: partner_id, key: <count lane> }` with
//!   `GroupAgg::Count` — the histogram of lines by their partner's count, in
//!   ONE program, so no host loop over lines or over values is involved.
//!
//! Both are checked against a plain host oracle that never touches Quack or
//! mask-risc.

#[path = "duckdb/fixture.rs"]
#[allow(dead_code)]
mod fixture;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Scratch, Value};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignLane, GroupAddr, GroupAgg, Query};

use fixture::col::{PARTNER_ID, STATUS};
use fixture::{LINE_ROWS, PARTNER_ROWS};

// ---------------------------------------------------------------------
// A counting allocator, so phase 2's own allocations can be measured.
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

// ---------------------------------------------------------------------
// The pieces.
// ---------------------------------------------------------------------

/// The posted-line filter both phases share: `status = 1`.
fn posted() -> Filter {
    Filter::cmp(STATUS, Cmp::EqU32(1))
}

/// The one foreign lane of phase 2: the narrowed count sink.
const COUNT_LANE: ForeignLane = ForeignLane(0);

/// Phase 1: `COUNT(*) GROUP BY partner_id` over posted lines, into the
/// normal `i64` Count sink. Existing machinery only.
fn phase1(fx: &fixture::Fixture) -> [i64; PARTNER_ROWS] {
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let program = lower(&Query {
        filter: posted(),
        agg: Agg::GroupReduce {
            key: GroupAddr::Local(PARTNER_ID),
            agg: GroupAgg::Count,
        },
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
    let mut sink = [0i64; PARTNER_ROWS];
    let v = execute_into(
        &program,
        &planes,
        &Foreign::NONE,
        &mut scratch,
        Out::I64(&mut sink),
    )
    .expect("runs");
    assert_eq!(v, Value::GroupReduced);
    sink
}

/// THE KNIFE: a checked, test-only narrowing of the sink into a `u32` lane.
/// K-sized. Panics rather than truncates if a count does not fit.
fn narrow(sink: &[i64; PARTNER_ROWS]) -> [u32; PARTNER_ROWS] {
    sink.map(|c| u32::try_from(c).expect("a count fits u32 on this fixture"))
}

/// Phase 2, predicate consumer: how many posted lines have a partner whose
/// posted-line count is exactly `v`.
fn phase2_eq(fx: &fixture::Fixture, count_lane: &[u32], v: u32) -> u64 {
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let flanes = [LaneRef::U32(count_lane)];
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let program = lower(&Query {
        filter: Filter::and([posted(), Filter::eq_u32_via(PARTNER_ID, COUNT_LANE, v)]),
        agg: Agg::Count,
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
    match execute_into(&program, &planes, &foreign, &mut scratch, Out::None).expect("runs") {
        Value::Count(c) => c as u64,
        other => panic!("expected a count, got {other:?}"),
    }
}

/// Phase 2, key consumer: the histogram `h[c]` = number of posted lines whose
/// partner has posted-line count `c`, in ONE program (`GroupKey::Via` keyed by
/// the produced lane). `out` is the caller-owned sink; its length is the
/// histogram universe.
fn phase2_hist(fx: &fixture::Fixture, count_lane: &[u32], out: &mut [i64]) -> usize {
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let flanes = [LaneRef::U32(count_lane)];
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    let program = lower(&Query {
        filter: posted(),
        agg: Agg::GroupReduce {
            key: GroupAddr::Via {
                fk: PARTNER_ID,
                key: COUNT_LANE,
            },
            agg: GroupAgg::Count,
        },
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
    let before = BYTES.load(Ordering::Relaxed);
    let v = execute_into(&program, &planes, &foreign, &mut scratch, Out::I64(out)).expect("runs");
    let alloc = BYTES.load(Ordering::Relaxed) - before;
    assert_eq!(v, Value::GroupReduced);
    alloc
}

// ---------------------------------------------------------------------
// The oracle: plain loops over the fixture's vectors.
// ---------------------------------------------------------------------

struct Oracle {
    per_partner: [u64; PARTNER_ROWS],
    /// `lines_with[c]` = posted lines whose partner has count `c`.
    lines_with: Vec<u64>,
}

fn oracle(fx: &fixture::Fixture) -> Oracle {
    let mut per_partner = [0u64; PARTNER_ROWS];
    for i in 0..LINE_ROWS {
        if fx.line.status[i] == 1 {
            per_partner[fx.line.partner_id[i] as usize] += 1;
        }
    }
    let max = *per_partner.iter().max().expect("non-empty") as usize;
    let mut lines_with = vec![0u64; max + 1];
    for i in 0..LINE_ROWS {
        if fx.line.status[i] == 1 {
            lines_with[per_partner[fx.line.partner_id[i] as usize] as usize] += 1;
        }
    }
    Oracle {
        per_partner,
        lines_with,
    }
}

// ---------------------------------------------------------------------
// The probe.
// ---------------------------------------------------------------------

/// FAILS IF: the produced counts, once in a `u32` lane, cannot be read per
/// line through `partner_id` by the existing predicate and key consumers, or
/// either consumer disagrees with the host oracle.
#[test]
fn a_completed_group_result_composes_into_a_later_pass_over_the_source() {
    let fx = fixture::generate();
    let o = oracle(&fx);

    // Phase 1 and the knife.
    let sink = phase1(&fx);
    for (p, (&got, &want)) in sink.iter().zip(&o.per_partner).enumerate() {
        assert_eq!(got as u64, want, "phase 1 count of partner {p}");
    }
    let lane = narrow(&sink);

    // Anti-vacuity: the read must discriminate. Partners must not all share one
    // count, or reading the wrong partner's slot could still look right.
    let mut distinct: Vec<u32> = lane.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() > 10,
        "only {} distinct counts",
        distinct.len()
    );

    // Phase 2a: the predicate consumer, for every count that occurs and for
    // one that does not.
    let mut covered = 0u64;
    for &v in &distinct {
        let got = phase2_eq(&fx, &lane, v);
        assert_eq!(
            got, o.lines_with[v as usize],
            "lines whose partner has count {v}"
        );
        covered += got;
    }
    let absent = (0..)
        .find(|v| !distinct.contains(v))
        .expect("some count is absent");
    assert_eq!(
        phase2_eq(&fx, &lane, absent),
        0,
        "count {absent} occurs nowhere"
    );
    let posted_lines = (0..LINE_ROWS).filter(|&i| fx.line.status[i] == 1).count() as u64;
    assert_eq!(
        covered, posted_lines,
        "every posted line has exactly one partner count"
    );

    // Phase 2b: the key consumer, one program.
    let mut hist = vec![0i64; o.lines_with.len()];
    let alloc = phase2_hist(&fx, &lane, &mut hist);
    for (c, (&got, &want)) in hist.iter().zip(&o.lines_with).enumerate() {
        assert_eq!(got as u64, want, "histogram bucket {c}");
    }
    eprintln!(
        "METRIC result_operand_probe N={LINE_ROWS} K={PARTNER_ROWS} distinct_counts={} \
         sink_bytes={} lane_bytes={} hist_bytes={} phase2_exec_alloc_bytes={alloc}",
        distinct.len(),
        std::mem::size_of_val(&sink),
        std::mem::size_of_val(&lane),
        hist.len() * 8,
    );
}

/// Load-bearing check 1: phase 2 really reads the produced lane through the
/// fk. Handing it a lane with the same values in a different order (each
/// partner gets its neighbour's count) must change the answer.
#[test]
fn the_via_read_is_load_bearing() {
    let fx = fixture::generate();
    let o = oracle(&fx);
    let lane = narrow(&phase1(&fx));
    let mut rotated = lane;
    rotated.rotate_left(1);
    assert_ne!(rotated, lane, "rotation must move at least one count");

    let mut hist = vec![0i64; o.lines_with.len()];
    phase2_hist(&fx, &rotated, &mut hist);
    let mismatched = hist
        .iter()
        .zip(&o.lines_with)
        .filter(|(&g, &w)| g as u64 != w)
        .count();
    assert!(
        mismatched > 0,
        "a lane in the wrong partner order went unnoticed"
    );
}

/// Load-bearing check 2: the oracle comparison can fail. Corrupting one
/// partner's count in the lane must move at least one histogram bucket.
#[test]
fn a_wrong_partner_count_is_caught() {
    let fx = fixture::generate();
    let o = oracle(&fx);
    let mut lane = narrow(&phase1(&fx));
    lane[0] += 1;
    let mut hist = vec![0i64; o.lines_with.len() + 1];
    phase2_hist(&fx, &lane, &mut hist);
    let mismatched = (0..hist.len())
        .filter(|&c| hist[c] as u64 != o.lines_with.get(c).copied().unwrap_or(0))
        .count();
    assert!(mismatched > 0, "a corrupted count went unnoticed");
}

/// The fk column must be the only route from a line to a count: grouping by a
/// different local column through the same lane gives a different answer.
/// Guards against a probe that would pass because the lane's index happens to
/// line up with some other column.
#[test]
fn the_route_is_the_partner_fk() {
    let fx = fixture::generate();
    let o = oracle(&fx);
    let lane = narrow(&phase1(&fx));
    let lanes = fx.line.lanes();
    let planes = lanes.planes();
    let flanes = [LaneRef::U32(&lane)];
    let foreign = Foreign {
        planes: &[],
        lanes: &flanes,
    };
    // `cost_center` (0..8) through the count lane: an fk into the wrong table.
    let wrong_fk: Col = fixture::col::COST_CENTER;
    let program = lower(&Query {
        filter: posted(),
        agg: Agg::GroupReduce {
            key: GroupAddr::Via {
                fk: wrong_fk,
                key: COUNT_LANE,
            },
            agg: GroupAgg::Count,
        },
    })
    .expect("lowers");
    let mut scratch = Scratch::for_program(&program, planes.n_rows).expect("carves");
    let mut hist = vec![0i64; o.lines_with.len()];
    execute_into(
        &program,
        &planes,
        &foreign,
        &mut scratch,
        Out::I64(&mut hist),
    )
    .expect("runs");
    let mismatched = hist
        .iter()
        .zip(&o.lines_with)
        .filter(|(&g, &w)| g as u64 != w)
        .count();
    assert!(mismatched > 0, "a wrong fk reproduced the right histogram");
}
