//! [`execute_sel`] allocates nothing, on the accepting and on the refusing
//! path (law L1). Own file: the counting allocator is process-global.

use lance_graph_mask_risc::{execute_sel, ExecError, LaneRef, Planes, SelFold, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

static BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

// SAFETY: forwards to `System` unchanged; it only counts.
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

fn measure(f: impl FnOnce()) -> usize {
    let before = BYTES.load(Ordering::Relaxed);
    MEASURING.with(|m| m.set(true));
    f();
    MEASURING.with(|m| m.set(false));
    BYTES.load(Ordering::Relaxed) - before
}

/// FAILS IF: a selection fold or its refusal path allocates, or the counter
/// is inert (the last assertion).
#[test]
fn a_thousand_selection_folds_allocate_nothing() {
    let values: Vec<i32> = (0..10_000).map(|i| i % 97 - 40).collect();
    let positions: Vec<u32> = (0..10_000u32).filter(|p| p.is_multiple_of(37)).collect();
    let mut bad = positions.clone();
    bad.push(10_000);
    let lanes = [LaneRef::I32(&values)];
    let planes = Planes {
        n_rows: values.len(),
        masks: &[],
        lanes: &lanes,
    };

    let mut seen = 0i64;
    let mut refused = 0usize;
    let bytes = measure(|| {
        for t in 0..1000 {
            if let Ok(Value::Count(c)) = execute_sel(
                SelFold::CountGtI32 { lane: 0, t: t % 50 },
                &planes,
                &positions,
            ) {
                seen += c as i64;
            }
            if let Ok(Value::SumI64(s)) =
                execute_sel(SelFold::SumI32 { lane: 0 }, &planes, &positions)
            {
                seen += s;
            }
            if let Err(ExecError::SelOutOfRange { .. }) =
                execute_sel(SelFold::SumI32 { lane: 0 }, &planes, &bad)
            {
                refused += 1;
            }
        }
    });
    assert_eq!(bytes, 0, "selection folds allocated {bytes} bytes");
    assert_ne!(seen, 0, "the folds ran");
    assert_eq!(refused, 1000, "the refusing path ran");

    // Can it fire: the same window around an allocation is not zero.
    let fired = measure(|| {
        std::hint::black_box(vec![0u8; 4096]);
    });
    assert!(fired >= 4096, "the counter is inert");
}
