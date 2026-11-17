//! Fold terminals allocate the mask and nothing per row; materialization
//! allocates per kept row. Measured with a counting global allocator, so this
//! is its own test binary, and the tests run in one thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use vrl_recipe_probe::{Recipe, Store};

struct Counting;
static BYTES: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        CALLS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwards the caller's layout unchanged.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from `alloc` above with this layout.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn measured<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    let (b, c) = (BYTES.load(Ordering::Relaxed), CALLS.load(Ordering::Relaxed));
    let v = f();
    (
        v,
        BYTES.load(Ordering::Relaxed) - b,
        CALLS.load(Ordering::Relaxed) - c,
    )
}

fn store(n: usize) -> Store {
    Store::new(
        (0..n as i64).collect(),
        vec![
            ("status", (0..n as i64).map(|i| i % 2).collect()),
            ("amount", vec![7; n]),
        ],
    )
}

const SRC: &str =
    ".n = .key\n.a = .properties.amount.value\nif .properties.status.value != 1 { abort }";

/// `count` and `sum` make exactly one allocation, the `n.div_ceil(64)`-word
/// mask, at every size; `materialize_rows` allocates at least once per kept
/// row. Disable it catches: a fold that counts by materializing rows.
#[test]
fn folds_allocate_only_the_mask() {
    for &n in &[64usize, 512, 4096, 65536] {
        let s = store(n);
        let r = Recipe::compile(SRC, &s).unwrap();
        let words = n.div_ceil(64);
        let (count, bytes, calls) = measured(|| r.count(&s).unwrap());
        assert_eq!(count, n as u64 / 2);
        assert_eq!((bytes, calls), (8 * words, 1), "count at n {n}");
        let (sum, bytes, calls) = measured(|| r.sum(&s, "a").unwrap());
        assert_eq!(sum, 7 * n as i64 / 2);
        assert_eq!((bytes, calls), (8 * words, 1), "sum at n {n}");
        let (rows, _, calls) = measured(|| r.materialize_rows(&s).unwrap());
        assert_eq!(rows.len(), n / 2);
        assert!(
            calls > rows.len(),
            "materialize at n {n}: {calls} allocations"
        );
    }
}
