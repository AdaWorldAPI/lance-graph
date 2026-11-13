//! The allocation half of the ledger. One test in this binary, so the
//! counter sees one thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use lance_graph_mask_risc::Value;
use lance_graph_report::bound_sel::BoundFold;
use virtual_address_lab::cube::{count_coupled, FULL};
use virtual_address_lab::grid::{Grid, World};
use virtual_address_lab::physical::{apply, Moved};
use virtual_address_lab::recipe::Recipe;
use virtual_address_lab::view::{fold_property, Ledger, RowObject};

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

/// FAILS IF: normalising a chain, forming a row object, demanding a value or
/// the fold over the source-issued order allocates once warm; or counting
/// the coupled relation allocates; or the counter is blind (building a view
/// must allocate).
#[test]
fn the_warm_virtual_route_allocates_nothing() {
    let mut grid = Grid::new(32, 4_000, |l, p| (l * 10_000 + p) as i32);
    let world = World::new(32);
    let sel = grid.issue_order(9).unwrap();
    let binding = world.bind(&grid).unwrap();
    let ledger = Ledger::default();
    let chain = [Recipe::TRANSPOSE, Recipe::ROTATE90, Recipe::ROTATE270];
    let want: i64 = (0..4_000i64).map(|p| 50_000 + p).sum();
    assert_eq!(
        fold_property(&world, &binding, &grid, &chain, 5, sel, BoundFold::SumI32),
        Ok(Value::SumI64(want))
    );

    let before = ALLOCS.load(Relaxed);
    let normal = Recipe::normalize(&chain);
    let row = RowObject {
        recipe: normal,
        row: 1_234,
    };
    let value = row.get(&world, &binding, &grid, 17, &ledger);
    let sum = fold_property(&world, &binding, &grid, &chain, 5, sel, BoundFold::SumI32);
    let coupled = count_coupled(&FULL, &FULL, &FULL);
    let after = ALLOCS.load(Relaxed);
    assert_eq!(normal, Recipe::TRANSPOSE);
    assert_eq!(value, Ok(171_234));
    assert_eq!(sum, Ok(Value::SumI64(want)));
    assert_eq!(coupled, 65_536);
    assert_eq!(after - before, 0);

    let before = ALLOCS.load(Relaxed);
    let mut moved = Moved::default();
    let built = apply(&grid.matrix(), Recipe::TRANSPOSE, &mut moved);
    assert_eq!(built[1_234][17], 171_234);
    assert!(ALLOCS.load(Relaxed) - before > 4_000);
}
