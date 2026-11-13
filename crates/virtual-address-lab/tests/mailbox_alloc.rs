//! Plan section 10.2 item 9. One test in this binary, so the counter sees one
//! thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use causal_edge::CausalEdge64;
use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::cognitive_shader::MetaWord;
use virtual_address_lab::grid::World;
use virtual_address_lab::mailbox::{visit_selected, Cell64, MailboxLanes, FIELDS};
use virtual_address_lab::recipe::Recipe;

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

/// FAILS IF: normalising the chain, the witnessed borrow or the 64 reads
/// allocate once the binding exists; or the counter is blind (collecting the
/// values into a `Vec` must allocate).
#[test]
fn the_warm_selected_read_allocates_nothing() {
    const N: usize = 64;
    let mut mailbox: Box<MailboxSoA<N>> = Box::new(MailboxSoA::new(7, 3, 1.0));
    mailbox.set_populated(N);
    for r in 0..N {
        mailbox.set_meta(r, MetaWord((r as u32 * 7919) % 1_000));
        mailbox.set_edge(r, CausalEdge64(r as u64 * 3));
    }
    let mut source = MailboxLanes::new(&*mailbox);
    let sel = source.issue_order_by_meta().unwrap();
    let w = World::named(&FIELDS);
    let binding = w.bind_source(&source, source.space()).unwrap();
    let chain = [Recipe::TRANSPOSE, Recipe::ROTATE90, Recipe::ROTATE270];

    let before = ALLOCS.load(Relaxed);
    let mut sum = 0u64;
    let read = visit_selected(
        &w.store,
        &w.view,
        &binding,
        &source,
        source.rows(),
        &chain,
        0,
        sel,
        |_, c| {
            if let Cell64::U32(v) = c {
                sum += u64::from(v);
            }
        },
    );
    let after = ALLOCS.load(Relaxed);
    let want: u64 = (0..N as u64).map(|r| (r * 7919) % 1_000).sum();
    assert_eq!(read, Ok(N));
    assert_eq!(sum, want);
    assert_eq!(after - before, 0);

    let before = ALLOCS.load(Relaxed);
    let mut kept = Vec::new();
    let _ = visit_selected(
        &w.store,
        &w.view,
        &binding,
        &source,
        source.rows(),
        &chain,
        0,
        sel,
        |p, c| kept.push((p, c)),
    );
    assert!(ALLOCS.load(Relaxed) - before > 0);
}
