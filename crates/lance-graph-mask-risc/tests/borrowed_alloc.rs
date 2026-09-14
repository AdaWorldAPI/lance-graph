//! One caller-owned buffer, many row counts, ZERO allocation.
//!
//! This is the property `Scratch::over` exists for, and it is strictly
//! stronger than `no_alloc.rs`'s: that one pins `execute` against a fixed
//! shape, this one pins the whole arena story against a CHANGING one. A cache
//! keyed by exact size — the obvious implementation — passes every fixed-shape
//! test and fails here, because allocation would become a function of the
//! population's history rather than of its maximum.
//!
//! Own binary, own counting allocator: the counter is process-global and a
//! second concurrent `#[test]` pollutes the delta.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::{
    execute, scratch_words_for, words_for, LaneRef, MaskOp, Operand, Planes, Pred, Program,
    Scratch, Terminal, Value,
};

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

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

struct Fx {
    n: usize,
    mask: Vec<u64>,
    lane: Vec<i32>,
}

impl Fx {
    fn new(n: usize) -> Self {
        let mut s = 0xA11CEu64 ^ n as u64;
        let mut mask: Vec<u64> = (0..words_for(n))
            .map(|_| lcg(&mut s) & lcg(&mut s))
            .collect();
        if !n.is_multiple_of(64) && !mask.is_empty() {
            let last = mask.len() - 1;
            mask[last] &= (1u64 << (n % 64)) - 1;
        }
        let lane = (0..n).map(|_| (lcg(&mut s) % 2000) as i32 - 1000).collect();
        Self { n, mask, lane }
    }
}

fn program() -> Program {
    Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: 100 },
                under: None,
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LtI32 { lane: 0, t: 800 },
                under: Some(Operand::Scratch(0)),
                dst: 1,
            },
            MaskOp::And {
                a: Operand::Plane(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(2),
        },
    )
}

/// FAILS IF: any row count costs an allocation once the buffer is grown —
/// including `n = 999`, which is deliberately absent from the warm-up and is
/// what a size-keyed cache would allocate for on first sight.
///
/// Four-sided. The zero-byte assertion is the property; the warm-up establishes
/// that the buffer is grown ONCE and never again; every result is checked
/// against the owned arena so a path that allocates nothing by computing
/// nothing cannot pass; and the counter itself is proven live at the end.
#[test]
fn one_growing_buffer_serves_every_row_count_without_allocating() {
    let p = program();
    let slots = p.scratch_slots as usize;

    // EVERY fixture built before the baseline — building one allocates, and
    // that allocation is the caller's, not the arena's.
    let warm: Vec<Fx> = [64usize, 4096].iter().map(|&n| Fx::new(n)).collect();
    let sweep: Vec<Fx> = [0usize, 1, 63, 64, 65, 127, 128, 999, 1000, 4096]
        .iter()
        .map(|&n| Fx::new(n))
        .collect();

    // Expected answers from the OWNED arena, also before the baseline.
    let expected: Vec<Value> = sweep
        .iter()
        .map(|f| {
            let masks: [&[u64]; 1] = [&f.mask];
            let lanes = [LaneRef::I32(&f.lane)];
            let planes = Planes {
                n_rows: f.n,
                masks: &masks,
                lanes: &lanes,
            };
            let mut s = Scratch::for_program(&p, f.n).expect("addressable");
            execute(&p, &planes, &mut s, None).expect("runs")
        })
        .collect();

    // One buffer, grown once to the largest shape the sweep will ask for.
    let cap = scratch_words_for(words_for(4096), slots).expect("fits");
    let mut buf = vec![0u64; cap];

    for f in &warm {
        let masks: [&[u64]; 1] = [&f.mask];
        let lanes = [LaneRef::I32(&f.lane)];
        let planes = Planes {
            n_rows: f.n,
            masks: &masks,
            lanes: &lanes,
        };
        let mut s = Scratch::over_for_program(&mut buf, &p, f.n).expect("prefix fits");
        execute(&p, &planes, &mut s, None).expect("runs");
    }

    let before = BYTES.load(Ordering::Relaxed);
    // A fixed array, not a `Vec`: the measured region must allocate nothing of
    // its own, or the gate measures the harness instead of the arena.
    let mut results = [Value::Blended; 10];
    for (i, f) in sweep.iter().enumerate() {
        let masks: [&[u64]; 1] = [&f.mask];
        let lanes = [LaneRef::I32(&f.lane)];
        let planes = Planes {
            n_rows: f.n,
            masks: &masks,
            lanes: &lanes,
        };
        let mut s = Scratch::over_for_program(&mut buf, &p, f.n).expect("prefix fits");
        results[i] = execute(&p, &planes, &mut s, None).expect("runs");
    }
    let after = BYTES.load(Ordering::Relaxed);

    assert_eq!(
        after - before,
        0,
        "the borrowed arena allocated {} bytes across ten row counts",
        after - before
    );
    for (i, f) in sweep.iter().enumerate() {
        assert_eq!(
            results[i], expected[i],
            "n={}: borrowed arena disagrees with the owned one",
            f.n
        );
    }

    // can-it-fire: the counter must move on a real allocation, or every
    // zero above is an instrument that measures nothing.
    let mark = BYTES.load(Ordering::Relaxed);
    let probe = std::hint::black_box(vec![0u8; 4096]);
    assert!(BYTES.load(Ordering::Relaxed) - mark >= probe.len());
}
