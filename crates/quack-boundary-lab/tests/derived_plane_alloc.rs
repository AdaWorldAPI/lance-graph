//! D-QUACK-DERIVED-PLANE-PROBE, P4: the warm derived-plane path allocates
//! nothing. One test in this binary, so the counter sees one thread.

use lance_graph_mask_risc::{MaskOp, Operand, Out, Program, Scratch, Terminal, Value};
use quack_boundary_lab::derived::{admitted_planes, check_planes, DerivedPlane, Recipe};
use quack_boundary_lab::{Prepared, RowOrder, Source, World, READER, REGION, ROWS};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
// SAFETY: forwards to the system allocator unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

/// FAILS IF: checking or executing over resident derived planes allocates,
/// or deriving them does not (the counter would then be blind).
#[test]
fn the_warm_plane_path_allocates_nothing() {
    let w = World::new();
    let s = Source::new(RowOrder::ById);
    let b = w.bind_fields(&s, &[REGION], READER, s.space()).unwrap();
    let before = ALLOCS.load(Relaxed);
    let planes = [2u8, 1, 0].map(|k| DerivedPlane::bit(&w, &w.view, &b, &s, REGION, k).unwrap());
    assert!(
        ALLOCS.load(Relaxed) - before >= 3,
        "the counter sees the derivation"
    );

    let expects = [2u8, 1, 0].map(|k| Recipe::Bit { field: REGION, k });
    let refs = [&planes[0], &planes[1], &planes[2]];
    let program = Program::new(
        vec![MaskOp::Ternlog {
            imm: 0b0010_0110,
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            c: Operand::Plane(2),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let mut scratch = Scratch::for_program(&program, ROWS).unwrap();
    let p = Prepared::over(program, &s, &b);
    let want = (0..ROWS)
        .filter(|&r| (0b0010_0110u8 >> s.row(r).region) & 1 == 1)
        .count();
    let mut run = || {
        admitted_planes(
            &p,
            &expects,
            &refs,
            &w,
            &w.view,
            &b,
            &s,
            &mut scratch,
            Out::None,
        )
    };
    assert_eq!(run(), Ok(Value::Count(want)));

    let before = ALLOCS.load(Relaxed);
    let checked = check_planes(&expects, &refs, &b, &s);
    let got = run();
    let after = ALLOCS.load(Relaxed);
    assert_eq!(checked, Ok(()));
    assert_eq!(got, Ok(Value::Count(want)));
    assert_eq!(after - before, 0);
}
