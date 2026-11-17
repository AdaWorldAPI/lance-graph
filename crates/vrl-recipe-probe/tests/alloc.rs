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
        (0..n as i32).collect(),
        vec![
            ("status", (0..n as i32).map(|i| i % 2).collect()),
            ("amount", vec![7; n]),
        ],
    )
}

const SRC: &str =
    ".n = .key\n.a = .properties.amount.value\nif .properties.status.value != 1 { abort }";

/// `count` and `sum` allocate the mask and one scratch mask, `⌈n/64⌉` words
/// each, at every size. `materialize_rows` makes five allocations at every
/// size (mask, scratch, position table, names, lanes): only the position
/// table grows with kept rows, by 4 bytes a row, and no string or event is
/// built per row. Disables it catches: a fold that counts by materializing,
/// and rows materialized as per-row events.
#[test]
fn folds_allocate_only_masks_and_rows_only_positions() {
    for &n in &[64usize, 512, 4096, 65536] {
        let s = store(n);
        let r = Recipe::compile(SRC, &s).unwrap();
        let words = n.div_ceil(64);
        let (count, bytes, calls) = measured(|| r.count(&s).unwrap());
        assert_eq!(count, n as u64 / 2);
        assert_eq!((bytes, calls), (16 * words, 2), "count at n {n}");
        let (sum, bytes, calls) = measured(|| r.sum(&s, "a").unwrap());
        assert_eq!(sum, 7 * n as i64 / 2);
        assert_eq!((bytes, calls), (16 * words, 2), "sum at n {n}");
        let kept = n / 2;
        let (rows, bytes, calls) = measured(|| r.materialize_rows(&s).unwrap());
        assert_eq!(rows.len(), kept);
        assert_eq!(calls, 5, "materialize at n {n}");
        assert!(
            bytes <= 16 * words + 4 * kept + 256,
            "materialize at n {n}: {bytes} bytes"
        );
    }
}
