mod common;
use lance_graph_sap::query::CatsQuery;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! { static ACTIVE: Cell<bool> = const { Cell::new(false) }; static COUNT: Cell<usize> = const { Cell::new(0) }; }
struct Counter;
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ACTIVE.with(|a| {
            if a.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ACTIVE.with(|a| {
            if a.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        System.realloc(ptr, layout, size)
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
#[test]
fn the_entire_bound_query_allocates_zero_bytes() {
    let input = common::fixture(4097);
    let batch = common::bind(&input);
    let mut query = CatsQuery::prepare(&batch, "00000042", "2026-09-01", "2026-09-30").unwrap();
    let mut sums = vec![0; query.groups()];
    ACTIVE.with(|a| a.set(true));
    let result = query.execute_into(&mut sums).map(|mask| mask.len());
    ACTIVE.with(|a| a.set(false));
    assert!(result.is_ok());
    assert_eq!(COUNT.with(Cell::get), 0);
    assert_eq!(sums[1], 4097 * 85);
}
