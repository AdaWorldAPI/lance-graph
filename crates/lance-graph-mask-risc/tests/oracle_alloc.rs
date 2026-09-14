//! The oracle refuses an unaddressable slot count WITHOUT first sizing a
//! buffer from it.
//!
//! Its own test binary, with its own counting allocator, because the
//! measurement is a process-global counter: a second `#[test]` in the same
//! process runs concurrently by default and pollutes the delta. That is also
//! why `tests/no_alloc.rs` — which pins law L1 with the same technique —
//! stays a one-test file rather than gaining this one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::{
    reference_execute, reference_scratch, ExecError, Operand, Planes, Program, Terminal, Value,
};

struct Counting;

static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
// `alloc_zeroed` is deliberately NOT overridden — the trait's default routes
// it through `self.alloc`, which is what makes a `vec![0u64; n]` visible here.
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

/// A refusal must cost less than this. The pre-fix code sized the
/// read-before-write bitmap straight from the declared count, so `u32::MAX`
/// slots asked for `(u32::MAX / 64) * 8` = 512 MiB before the check that
/// rejects the program ever ran. One MiB is three orders of magnitude under
/// that and far above any incidental allocation on the refusal path (which
/// is, in fact, zero).
const REFUSAL_BUDGET: usize = 1 << 20;

/// FAILS IF: either oracle entry point sizes a buffer from `scratch_slots`
/// before checking it against the ceiling — the byte budget catches that
/// directly, where asserting only the returned error cannot, since the
/// pre-fix code returns the SAME error after paying for the allocation.
///
/// Three-sided. The refusal half proves the guard fires and names the right
/// error; the budget proves it fires BEFORE the allocator is asked; the
/// silence half runs the identical program at its computed slot count and
/// requires a real answer, so the guard cannot pass by rejecting everything.
#[test]
fn an_unaddressable_slot_count_is_refused_before_a_buffer_is_sized_from_it() {
    let alpha = [0xFFFF_FFFF_FFFF_FFFFu64];
    let masks: [&[u64]; 1] = [&alpha];
    let planes = Planes {
        n_rows: 64,
        masks: &masks,
        lanes: &[],
    };
    // No op, and a terminal over a PLANE: the program names no
    // `Operand::Scratch` anywhere, so every per-operand bound check is
    // structurally blind to it and only the declared-count check can refuse.
    let mut p = Program::new(
        vec![],
        Terminal::Count {
            mask: Operand::Plane(0),
        },
    );
    assert_eq!(p.scratch_slots, 0, "fixture names no scratch operand");

    // silence half, FIRST: at its own computed count the program is valid and
    // answers. Whatever the refusal half observes below is the declared count
    // alone, not a second defect in the fixture.
    assert_eq!(
        reference_execute(&p, &planes, None),
        Ok(Value::Count(64)),
        "the fixture must be a program the oracle can actually run"
    );

    p.scratch_slots = u32::MAX;

    let before = BYTES.load(Ordering::Relaxed);
    let refused = reference_execute(&p, &planes, None);
    let after = BYTES.load(Ordering::Relaxed);
    assert_eq!(
        refused,
        Err(ExecError::ScratchSlotsUnaddressable { declared: u32::MAX }),
        "a count past the ceiling must be refused by the ceiling, not by anything downstream"
    );
    assert!(
        after - before < REFUSAL_BUDGET,
        "reference_execute allocated {} bytes refusing an unaddressable program",
        after - before
    );

    // the same for the other entry point, which had its own copy of the
    // allocation and would not have been covered by testing one of them
    let before = BYTES.load(Ordering::Relaxed);
    let refused = reference_scratch(&p, &planes);
    let after = BYTES.load(Ordering::Relaxed);
    assert_eq!(
        refused,
        Err(ExecError::ScratchSlotsUnaddressable { declared: u32::MAX })
    );
    assert!(
        after - before < REFUSAL_BUDGET,
        "reference_scratch allocated {} bytes refusing an unaddressable program",
        after - before
    );

    // can-it-fire: the counter itself must move, or every budget above is
    // satisfied by an instrument that measures nothing.
    let mark = BYTES.load(Ordering::Relaxed);
    let probe = std::hint::black_box(vec![0u8; 4096]);
    assert!(BYTES.load(Ordering::Relaxed) - mark >= probe.len());
}
