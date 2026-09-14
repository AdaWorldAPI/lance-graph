//! Law L1: `execute` never allocates. A counting global allocator measures
//! 1000 executes of a six-op program; the delta must be zero bytes. The
//! can-it-fire half proves the counter itself moves.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use lance_graph_mask_risc::exec::{execute, Scratch};
use lance_graph_mask_risc::{LaneRef, MaskOp, Operand, Planes, Pred, Program, Terminal, Value};

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

/// FAILS IF: any op, terminal, or the validation path allocates — or the
/// allocator counter is inert (the second assertion).
#[test]
fn a_thousand_executes_allocate_nothing() {
    let n = 65_536;
    let words = n / 64;
    let mut seed = 7u64;
    let lane: Vec<i32> = (0..n)
        .map(|_| (lcg(&mut seed) % 2000) as i32 - 1000)
        .collect();
    let alpha: Vec<u64> = (0..words)
        .map(|_| lcg(&mut seed) & lcg(&mut seed))
        .collect();
    let masks: [&[u64]; 1] = [&alpha];
    let lanes = [LaneRef::I32(&lane)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let p = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: 600 },
                under: None,
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LeI32 { lane: 0, t: 900 },
                under: Some(Operand::Scratch(0)),
                dst: 1,
            },
            MaskOp::And {
                a: Operand::Scratch(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
            MaskOp::Ternlog {
                imm: 0x80,
                a: Operand::Plane(0),
                b: Operand::Scratch(2),
                c: Operand::Scratch(1),
                dst: 3,
            },
            MaskOp::Not {
                a: Operand::Scratch(3),
                dst: 4,
            },
            MaskOp::Or {
                a: Operand::Scratch(4),
                b: Operand::Scratch(2),
                dst: 5,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(5),
        },
    );
    let mut scratch = Scratch::for_program(&p, n);
    let warm = execute(&p, &planes, &mut scratch, None);
    assert!(matches!(warm, Ok(Value::Count(_))));

    let before = BYTES.load(Ordering::Relaxed);
    for _ in 0..1000 {
        let v = execute(&p, &planes, &mut scratch, None);
        assert_eq!(v, warm);
    }
    let after = BYTES.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "execute allocated {} bytes over 1000 runs",
        after - before
    );

    // can-it-fire: the counter must see a real allocation
    let probe = vec![0u8; 4096];
    assert!(BYTES.load(Ordering::Relaxed) - after >= probe.len());
}
