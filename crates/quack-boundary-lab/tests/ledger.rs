//! The materialisation ledger: per workload, what was written where.
//!
//! Each quantity is taken on its own, because none stands in for another:
//!
//! - `alloc`: bytes allocated on the executing thread inside the run.
//! - `scratch`: bytes of the caller's scratch (tile-wide slots plus bitmap).
//! - `demanded`: bytes of the sink the query asked for.
//! - `intermediate`: bytes one program wrote for the next program to read.
//! - `passes`: lane and mask passes the programs spend, terminals included.
//!
//! The executor borrows `Planes` shared, so the only memory a run can write
//! is the scratch and the sink it is handed. Lanes are checked unmoved and
//! unchanged after every run.
//!
//! No time is measured here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::{words_for, Foreign, LaneRef, Out, Program, Scratch, Value};
use lance_graph_quack::{
    lower, lower_group_by_auto, Agg, Cmp, Col, Filter, GroupBy, GroupLowering, Query,
};
use quack_boundary_lab::{
    admitted, oracle, unbound, Prepared, RowOrder, Source, World, AMOUNT, FIELDS, QTY, REGION,
    REGIONS, ROWS,
};

struct Counting;

static BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURING.try_with(Cell::get).unwrap_or(false) {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
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

/// Bytes allocated by this thread while `f` runs.
fn measure<R>(f: impl FnOnce() -> R) -> (usize, R) {
    let before = BYTES.load(Ordering::Relaxed);
    MEASURING.with(|m| m.set(true));
    let r = f();
    MEASURING.with(|m| m.set(false));
    (BYTES.load(Ordering::Relaxed) - before, r)
}

const T: i32 = 25;
const K: usize = REGIONS as usize;

fn col(field: u8) -> Col {
    Col(u16::from(field))
}

fn amount_gt() -> Filter {
    Filter::cmp(col(AMOUNT), Cmp::GtI32(T))
}

/// Lane and mask passes of one program, its terminal counted as one.
fn passes(p: &Program) -> usize {
    let h = p.op_histogram();
    h.predicates + h.mask_passes() + 1
}

fn scratch_bytes(s: &Scratch<'_>) -> usize {
    (s.words() * s.slots() + s.slots().div_ceil(64)) * 8
}

#[derive(Debug, PartialEq, Eq)]
struct Line {
    name: &'static str,
    programs: usize,
    passes: usize,
    alloc: usize,
    scratch: usize,
    demanded: usize,
    intermediate: usize,
}

/// FAILS IF: a scalar or folded workload allocates, writes anything
/// population-sized that was not demanded, or moves or changes a lane; or
/// the forest's intermediate goes unreported; or the counter is inert.
#[test]
fn the_ledger() {
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    let binding = w.bind(&s).unwrap();
    let addresses = s.lane_addresses();
    let checksum = s.checksum();
    let mask_bytes = words_for(ROWS) * 8;
    let mut ledger = Vec::new();

    // The lanes the admitted path reads are the source's own allocations.
    let mut seen = 0;
    for item in binding
        .borrow_all(&w.store, &w.view, s.frame_store(), &s)
        .unwrap()
    {
        let (field, lane) = item.unwrap();
        let start = match lane {
            LaneRef::I32(v) => v.as_ptr() as usize,
            LaneRef::U32(v) => v.as_ptr() as usize,
            other => panic!("unexpected lane kind {other:?}"),
        };
        assert_eq!((start, lane.len()), addresses[usize::from(field)]);
        seen += 1;
    }
    assert_eq!(seen, FIELDS);

    // The counter moves.
    let (bytes, v) = measure(|| vec![0u64; 64]);
    assert!(bytes >= 512, "the allocation counter is live");
    drop(v);

    // One admitted run of one program, measured.
    let mut admitted_line =
        |name: &'static str, program: Program, sink: &mut [i64], mask: &mut [u64]| {
            let p = Prepared::under(program, &s);
            let mut scratch = Scratch::for_program(&p.program, ROWS).unwrap();
            // Scratch is one tile wide whatever the population is.
            assert!(scratch.words() * 64 < ROWS, "{name}");
            let (demanded, out) = if !mask.is_empty() {
                (std::mem::size_of_val(mask), Out::Mask(mask))
            } else if !sink.is_empty() {
                (std::mem::size_of_val(sink), Out::I64(sink))
            } else {
                (0, Out::None)
            };
            let (alloc, value) =
                measure(|| admitted(&p, &w, &w.view, &binding, &s, &mut scratch, out));
            ledger.push(Line {
                name,
                programs: 1,
                passes: passes(&p.program),
                alloc,
                scratch: scratch_bytes(&scratch),
                demanded,
                intermediate: 0,
            });
            value.unwrap_or_else(|e| panic!("{name}: {e:?}"))
        };

    // 1. COUNT and SUM over a direct filter.
    let count = lower(&Query {
        filter: amount_gt(),
        agg: Agg::Count,
    })
    .unwrap();
    assert_eq!(
        admitted_line("1a count", count, &mut [], &mut []),
        Value::Count(oracle::count_amount_gt(&s, T))
    );
    let sum = lower(&Query {
        filter: amount_gt(),
        agg: Agg::SumI32(col(QTY)),
    })
    .unwrap();
    assert_eq!(
        admitted_line("1b sum", sum, &mut [], &mut []),
        Value::SumI64(oracle::sum_qty_amount_gt(&s, T))
    );

    // 2. Grouped COUNT and SUM through the existing folded lowering.
    let want = oracle::by_region_amount_gt(&s, T, u32::MAX);
    for (name, agg, pick) in [
        ("2a group count", Agg::Count, 0usize),
        ("2b group sum", Agg::SumI32(col(QTY)), 1usize),
    ] {
        let g = GroupBy {
            filter: amount_gt(),
            key: col(REGION),
            groups: REGIONS,
            agg,
        };
        let GroupLowering::Folded { program, .. } = lower_group_by_auto(&g, 0).unwrap() else {
            panic!("{name} folds")
        };
        let mut sink = vec![0i64; K];
        admitted_line(name, program, &mut sink, &mut []);
        let expect: Vec<i64> = want
            .iter()
            .map(|g| if pick == 0 { g.0 } else { g.1 })
            .collect();
        assert_eq!(sink, expect, "{name}");
    }

    // 3. A demanded Keep mask.
    let keep = lower(&Query {
        filter: amount_gt(),
        agg: Agg::Rows,
    })
    .unwrap();
    let mut mask = vec![0u64; words_for(ROWS)];
    admitted_line("3 keep", keep, &mut [], &mut mask);
    assert_eq!(mask, oracle::mask_amount_gt(&s, T));

    // The contrast: a grouped ANY has no grouped terminal, so it is a forest
    // whose first program writes a mask for the others to read. It can only
    // run unbound.
    let g = GroupBy {
        filter: amount_gt(),
        key: col(REGION),
        groups: REGIONS,
        agg: Agg::Any,
    };
    let GroupLowering::Forest(plan) = lower_group_by_auto(&g, 0).unwrap() else {
        panic!("a grouped ANY stays a forest")
    };
    let mut kept = vec![0u64; words_for(ROWS)];
    let mut phase1 = Scratch::for_program(&plan.filter, ROWS).unwrap();
    let mut phase2 = Scratch::for_program(&plan.groups[0], ROWS).unwrap();
    let lanes = s.raw_lanes();
    let (alloc, any) = measure(|| {
        unbound(
            &plan.filter,
            &lanes,
            &[],
            &Foreign::NONE,
            &mut phase1,
            Out::Mask(&mut kept),
        )
        .unwrap();
        let mut any = [false; K];
        for (g, program) in plan.groups.iter().enumerate() {
            any[g] = unbound(
                program,
                &lanes,
                &[&kept],
                &Foreign::NONE,
                &mut phase2,
                Out::None,
            ) == Ok(Value::Bool(true));
        }
        any
    });
    assert_eq!(
        any.to_vec(),
        want.iter().map(|g| g.0 > 0).collect::<Vec<_>>()
    );
    ledger.push(Line {
        name: "forest group any",
        programs: 1 + plan.groups.len(),
        passes: passes(&plan.filter) + plan.groups.iter().map(passes).sum::<usize>(),
        alloc,
        scratch: scratch_bytes(&phase1) + scratch_bytes(&phase2),
        demanded: K,
        intermediate: std::mem::size_of_val(kept.as_slice()),
    });

    // Lanes: not moved, not written.
    assert_eq!(s.lane_addresses(), addresses);
    assert_eq!(s.checksum(), checksum);

    for l in &ledger {
        println!(
            "{:<18} programs {:>2}  passes {:>2}  alloc {:>4} B  scratch {:>5} B  demanded {:>5} B  intermediate {:>5} B",
            l.name, l.programs, l.passes, l.alloc, l.scratch, l.demanded, l.intermediate
        );
    }

    // One tile-wide slot and one bitmap word.
    let one_slot = (256 + 1) * 8;
    let line = |name, programs, passes, scratch, demanded, intermediate| Line {
        name,
        programs,
        passes,
        alloc: 0,
        scratch,
        demanded,
        intermediate,
    };
    assert_eq!(
        ledger,
        vec![
            line("1a count", 1, 2, one_slot, 0, 0),
            line("1b sum", 1, 2, one_slot, 0, 0),
            line("2a group count", 1, 2, one_slot, K * 8, 0),
            line("2b group sum", 1, 2, one_slot, K * 8, 0),
            line("3 keep", 1, 2, one_slot, mask_bytes, 0),
            line(
                "forest group any",
                1 + K,
                2 + 2 * K,
                2 * one_slot,
                K,
                mask_bytes
            ),
        ]
    );
    // Zero allocation on every line, the forest included: an allocation
    // count says nothing about the intermediate column.
    assert!(ledger.iter().all(|l| l.alloc == 0));
    assert_eq!(mask_bytes, 5_008);
}
