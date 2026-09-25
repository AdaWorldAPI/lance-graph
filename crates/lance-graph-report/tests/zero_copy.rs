//! Allocation falsifiers: what scales with the POPULATION, and what doesn't.
//!
//! A per-thread counting allocator measures bytes allocated inside a closure.
//! Two populations 64× apart must allocate the SAME bytes for execution (F8:
//! memory scales with result cardinality, never with rows) and for rotation
//! (F6/F7/A4/A5: rotation is metadata). Lanes are checked by address before
//! and after (F11: no population copy).

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use common::*;
use lance_graph_report::*;

struct Counting;
thread_local! {
    static BYTES: Cell<u64> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        BYTES.with(|b| b.set(b.get() + l.size() as u64));
        // SAFETY: forwarded verbatim to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: forwarded verbatim to the system allocator.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn allocated<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = BYTES.with(Cell::get);
    let t = f();
    (t, BYTES.with(Cell::get) - before)
}

const RA: FieldId = FieldId(0);
const RB: FieldId = FieldId(1);
const V: FieldId = FieldId(100);

fn pivot() -> ReportPlan {
    with_four(
        ReportPlan::over(src())
            .filter(Selection::cmp(V, CmpOp::Gt, Scalar::Int(-20)))
            .pivot(&[CoordSpec::Field(RA)], &[CoordSpec::Field(RB)]),
        V,
    )
}

#[test]
fn execution_memory_is_independent_of_population_size() {
    // The small population must already fill one scheduling tile: scratch is
    // `slots × min(words, TILE_WORDS)`, so below one tile it legitimately
    // grows with the rows, and the claim is only about populations past it.
    const SMALL: usize = 64 * lance_graph_mask_risc::TILE_WORDS;
    let small = synthetic(SMALL, &[6, 9], 31).batch();
    let big = synthetic(SMALL * 64, &[6, 9], 31).batch();
    let pol = PlannerPolicy {
        reuse_mask_min_programs: u64::MAX,
        ..PlannerPolicy::default()
    };
    let addr = big.lane_addr(V).unwrap();
    let ((rs, _), a_small) = allocated(|| pivot().execute(&small, &pol).unwrap());
    let ((rb, sb), a_big) = allocated(|| pivot().execute(&big, &pol).unwrap());
    assert_eq!(
        a_small, a_big,
        "execution allocates the same bytes at N and 64·N"
    );
    assert_eq!(big.lane_addr(V).unwrap(), addr, "source lane never moved");
    assert_eq!(sb.mask_materializations, 0);
    // Tile-sized: a few slots of at most `TILE_WORDS` words, bytes. Stated
    // against the constant so a change of the default tile does not have to
    // re-derive a literal here.
    assert!(
        sb.scratch_bytes_peak <= (8 * lance_graph_mask_risc::TILE_WORDS * 8) as u64,
        "scratch is tile-sized: {}",
        sb.scratch_bytes_peak
    );
    assert_eq!(rs.space().stored_cells(), rb.space().stored_cells());
    // Can-fire twin: the reused-mask carrier IS population-sized, and the
    // same probe sees it.
    let reuse = PlannerPolicy::default();
    let (_, r_small) = allocated(|| pivot().execute(&small, &reuse).unwrap());
    let (_, r_big) = allocated(|| pivot().execute(&big, &reuse).unwrap());
    assert_eq!(
        r_big - r_small,
        ((SMALL * 64 - SMALL) / 64 * 8) as u64,
        "exactly the mask's growth"
    );
}

#[test]
fn rotation_and_role_changes_allocate_no_population_or_cell_bytes() {
    for n in [4_096usize, 4_096 * 64] {
        let batch = synthetic(n, &[6, 9], 32).batch();
        let (res, _) = pivot().execute(&batch, &PlannerPolicy::default()).unwrap();
        let rot_plan = pivot().rotate();
        let (rot, bytes) = allocated(|| res.rotate());
        let (rot2, bytes2) = allocated(|| res.reinterpret(&rot_plan).unwrap());
        assert!(
            bytes < 64 && bytes2 < 256,
            "view metadata only: {bytes} / {bytes2}"
        );
        assert!(bytes < res.space().accumulator_bytes() as u64);
        assert_eq!(rot.space().payload_addr(), res.space().payload_addr());
        assert_eq!(rot2.space().payload_addr(), res.space().payload_addr());
    }
}
