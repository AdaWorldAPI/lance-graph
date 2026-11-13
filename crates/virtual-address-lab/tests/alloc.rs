//! H3 and H10, the allocation half. One test in this binary, so the counter
//! sees one thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use virtual_address_lab::cube::{axis, count_coupled, FULL};
use virtual_address_lab::grid::{Grid, World};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::route::plan_sum_where_eq;
use virtual_address_lab::view::{Trace, VirtualTable};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static LARGEST: AtomicU64 = AtomicU64::new(0);
// SAFETY: forwards to the system allocator unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        LARGEST.fetch_max(l.size() as u64, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

/// FAILS IF: composing recipes, forming row objects, demanding a value or
/// running a planned fold allocates; or counting the coupled relation
/// allocates at all; or the counter is blind (the materialised view must
/// allocate).
#[test]
fn the_warm_virtual_route_allocates_nothing() {
    let grid = Grid::new(32, 4_000, |l, p| (l * 10_000 + p) as i32);
    let world = World::new(32);
    let binding = world.bind(&grid).unwrap();
    let trace = Trace::default();
    let base = VirtualTable::bound(&world, &binding, &grid, Recipe::TRANSPOSE, &trace).unwrap();
    let table = base.through(Recipe::normalize(&[Recipe::ROTATE90, Recipe::ROTATE270]));
    let mut plan = plan_sum_where_eq(&table, 5, 7, 70_123).unwrap();
    assert_eq!(plan.sum(&table), Ok(50_123));

    let before = ALLOCS.load(Relaxed);
    let normal = Recipe::normalize(&[Recipe::ROTATE90, Recipe::MIRROR_X, Recipe::ROTATE270]);
    let row = table.row(1_234).unwrap();
    let value = row.get(17);
    let column = table.column(3);
    let sum = plan.sum(&table);
    let after = ALLOCS.load(Relaxed);
    assert_eq!(normal, Recipe::MIRROR_Y);
    assert_eq!(value, Some(171_234));
    assert!(column.is_some());
    assert_eq!(sum, Ok(50_123));
    assert_eq!(after - before, 0);

    let low = axis(|i| i < 10);
    let before = ALLOCS.load(Relaxed);
    let all = count_coupled(&FULL, &FULL, &FULL);
    let some = count_coupled(&low, &FULL, &low);
    let after = ALLOCS.load(Relaxed);
    assert_eq!((all, some), (65_536, 100));
    assert_eq!(after - before, 0);

    LARGEST.store(0, Relaxed);
    let built = table.materialize();
    assert_eq!(built.len(), 32 * 4_000);
    assert!(LARGEST.load(Relaxed) >= 32 * 4_000 * 4);
}
