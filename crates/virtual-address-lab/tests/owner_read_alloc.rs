//! P0, allocation half. One test in this binary.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use cognitive_shader_driver::mailbox_soa::MailboxSoA;
use lance_graph_contract::cognitive_shader::MetaWord;
use virtual_address_lab::owner::{Reads, Value, VirtualProperty};
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

/// FAILS IF: normalising the chain or the read allocates; or the counter is
/// blind (boxing the value must count).
#[test]
fn the_read_allocates_nothing() {
    const N: usize = 64;
    let mut mailbox: Box<MailboxSoA<N>> = Box::new(MailboxSoA::new(7, 3, 1.0));
    mailbox.set_populated(N);
    mailbox.set_meta(21, MetaWord(4_242));
    let reads = Reads::default();
    let chain = [Recipe::TRANSPOSE, Recipe::ROTATE90, Recipe::ROTATE270];
    let p = VirtualProperty {
        row: 21,
        property: 0,
        chain: &chain,
    };
    let before = ALLOCS.load(Relaxed);
    let got = p.read(&*mailbox, &reads);
    let after = ALLOCS.load(Relaxed);
    assert_eq!(got, Ok(Value::U32(4_242)));
    assert_eq!(after - before, 0);

    let before = ALLOCS.load(Relaxed);
    let boxed = Box::new(p.read(&*mailbox, &reads));
    assert!(ALLOCS.load(Relaxed) - before > 0);
    assert!(boxed.is_ok());
}
