//! D-MAPPING-POLYFILL-1: XVERWEIS by value as a masked read (#1963 §9).
//!
//! `SELECT name WHERE mail = key`, composed from two existing terminals over
//! one set of borrowed lanes:
//!
//! 1. `Agg::Count` over `Filter::cmp(MAIL, Cmp::EqU32(key))`: 0 is #NV, 1 is
//!    unique, more than 1 is a duplicate and is refused, never resolved to a
//!    first match.
//! 2. Only when the count is 1: `Agg::Rows` over the same filter, whose kept
//!    mask has exactly one bit; the return lane is read at that row.
//!
//! No index, no tuple and no host loop over rows: both programs fold the key
//! column, so the work is linear in the population (a column scan), and the
//! read of the return field is one aligned load. This composition is the
//! recipe, not an admitted method: no `MethodRef` exists for it, and a
//! same-program unique terminal does not exist (report-rs #33 ledger item 8).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_mask_risc::{
    execute, execute_into, words_for, Foreign, LaneRef, Out, Planes, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, Query};

struct Counting;

thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|a| a.set(a.get() + 1));
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

const MAIL: Col = Col(0);

#[derive(Debug, PartialEq, Eq)]
enum Lookup {
    Found(u32),
    NotFound,
    Duplicate(usize),
}

/// Lanes: `mail` codes and `name` values, one row per user. Code 7_007 is
/// held by rows 3 and 7 (a duplicate); every other code is unique.
fn lanes(n: usize) -> (Vec<u32>, Vec<u32>) {
    let mail: Vec<u32> = (0..n as u32)
        .map(|r| if r == 3 || r == 7 { 7_007 } else { 10_000 + r })
        .collect();
    let name: Vec<u32> = (0..n as u32).map(|r| 500 + r).collect();
    (mail, name)
}

/// The scalar oracle.
fn oracle(mail: &[u32], name: &[u32], key: u32) -> Lookup {
    let hits: Vec<usize> = (0..mail.len()).filter(|&r| mail[r] == key).collect();
    match hits.as_slice() {
        [] => Lookup::NotFound,
        [r] => Lookup::Found(name[*r]),
        many => Lookup::Duplicate(many.len()),
    }
}

/// Prepared once per key shape: the two programs, their scratch and the kept
/// mask. Built outside the measured read.
struct Prepared {
    count: lance_graph_mask_risc::Program,
    rows: lance_graph_mask_risc::Program,
}

fn prepare(key: u32) -> Prepared {
    let filter = Filter::cmp(MAIL, Cmp::EqU32(key));
    Prepared {
        count: lower(&Query {
            filter: filter.clone(),
            agg: Agg::Count,
        })
        .unwrap(),
        rows: lower(&Query {
            filter,
            agg: Agg::Rows,
        })
        .unwrap(),
    }
}

/// The read. Returns the answer and the number of programs run.
fn xverweis(
    p: &Prepared,
    planes: &Planes<'_>,
    s_count: &mut Scratch<'_>,
    s_rows: &mut Scratch<'_>,
    kept: &mut [u64],
    name: &[u32],
    skip_uniqueness: bool,
) -> (Lookup, u32) {
    let mut programs = 0;
    if !skip_uniqueness {
        programs += 1;
        match execute(&p.count, planes, s_count, None).unwrap() {
            Value::Count(0) => return (Lookup::NotFound, programs),
            Value::Count(1) => {}
            Value::Count(k) => return (Lookup::Duplicate(k), programs),
            other => panic!("count program returned {other:?}"),
        }
    }
    programs += 1;
    execute_into(&p.rows, planes, &Foreign::NONE, s_rows, Out::Mask(kept)).unwrap();
    let row = kept
        .iter()
        .enumerate()
        .find(|(_, w)| **w != 0)
        .map(|(i, w)| i * 64 + w.trailing_zeros() as usize);
    match row {
        Some(r) => (Lookup::Found(name[r]), programs),
        None => (Lookup::NotFound, programs),
    }
}

fn run(n: usize, key: u32, skip_uniqueness: bool) -> (Lookup, u32, u64) {
    let (mail, name) = lanes(n);
    let lanes = [LaneRef::U32(&mail)];
    let masks: [&[u64]; 0] = [];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let p = prepare(key);
    let mut s_count = Scratch::for_program(&p.count, n).unwrap();
    let mut s_rows = Scratch::for_program(&p.rows, n).unwrap();
    let mut kept = vec![0u64; words_for(n)];
    let a0 = allocs();
    let (got, programs) = xverweis(
        &p,
        &planes,
        &mut s_count,
        &mut s_rows,
        &mut kept,
        &name,
        skip_uniqueness,
    );
    let a = allocs() - a0;
    if !skip_uniqueness {
        assert_eq!(got, oracle(&mail, &name, key), "oracle");
    }
    (got, programs, a)
}

#[test]
fn a_unique_key_reads_the_return_field_in_two_programs_without_allocating() {
    let (got, programs, allocs) = run(512, 10_000 + 300, false);
    assert_eq!(got, Lookup::Found(800));
    assert_eq!(programs, 2);
    assert_eq!(allocs, 0, "the prepared read allocates nothing");
}

#[test]
fn an_absent_key_is_not_found_in_one_program() {
    let (got, programs, _) = run(512, 42, false);
    assert_eq!(got, Lookup::NotFound);
    assert_eq!(programs, 1);
}

#[test]
fn a_duplicate_key_is_refused_never_the_first_match() {
    let (got, programs, _) = run(512, 7_007, false);
    assert_eq!(got, Lookup::Duplicate(2));
    assert_eq!(programs, 1, "the return field is never read");
}

/// The disable arm: without the count, the same mask resolves the duplicate
/// to its lowest row (3), which is exactly the first-match behaviour the
/// contract forbids. The oracle refuses it.
#[test]
fn without_the_count_a_duplicate_silently_resolves_to_the_first_row() {
    let (got, _, _) = run(512, 7_007, true);
    assert_eq!(got, Lookup::Found(503), "row 3 wins");
    let (mail, name) = lanes(512);
    assert_eq!(oracle(&mail, &name, 7_007), Lookup::Duplicate(2));
}

/// Growth: the same lookup over 4x the rows runs the same two programs and
/// still allocates nothing in the read; the fold itself is a column scan
/// whose cost grows with the rows (no index is claimed).
#[test]
fn population_growth_keeps_the_program_count_and_zero_allocation() {
    let small = run(512, 10_000 + 300, false);
    let large = run(2048, 10_000 + 300, false);
    assert_eq!(small.0, large.0);
    assert_eq!((small.1, small.2), (large.1, large.2));
    assert_eq!(large.2, 0);
}
