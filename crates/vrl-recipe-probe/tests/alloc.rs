//! The terminals allocate only n-independent bookkeeping plus masks of
//! `⌈n/64⌉` words, and materialization adds only the position table.
//! Measured with a counting global allocator, so this is its own test binary
//! and the tests run in one thread.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{all_fields, bind, witness, Source, VIEW_CLASS};
use vrl_recipe_probe::Recipe;

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

const SRC: &str =
    ".n = .key\n.a = .properties.amount.value\nif .properties.status.value != 1 { abort }";

/// At n = 64, 512, 4096 and 65536, the bytes each terminal allocates are
/// measured against what it must hold:
///
/// - `count` and `sum`: mask-risc's scratch, five slots of one tile at most
///   (`5 · 8 · min(⌈n/64⌉, 256)` bytes, so at most 10 KiB at any n), plus
///   bookkeeping that is the same at every n.
/// - `materialize_rows`: the same, plus the kept mask (`8 · ⌈n/64⌉`) and the
///   position table (4 bytes a kept row), plus bookkeeping that is the same
///   at every n.
///
/// The allocation count is the same at every n for all three. No string or
/// event is built per row. Disables it catches: a fold that counts by
/// materializing, and a string built per materialized row.
#[test]
fn terminals_allocate_tile_scratch_mask_and_positions_only() {
    let mut calls_at: Vec<(usize, usize, usize)> = Vec::new();
    let mut fixed_at: Vec<(usize, usize, usize)> = Vec::new();
    for &n in &[64usize, 512, 4096, 65536] {
        let source = Source::from_lanes([
            (0..n as i32).collect(),
            (0..n as i32).map(|i| i % 2).collect(),
            vec![0; n],
            vec![7; n],
        ]);
        let world = bind(&source, &all_fields());
        let w = witness(&world, &source);
        let r = Recipe::compile(SRC, &common::View::new(), VIEW_CLASS).unwrap();
        let words = n.div_ceil(64);
        // Five slots: kept, dropped, term, absent, unknown (three-valued
        // program, EXO-294).
        let scratch = 5 * 8 * words.min(256);
        let kept = n / 2;
        let (count, cb, cc) = measured(|| r.count(&w).unwrap());
        assert_eq!(count, kept as u64);
        let (sum, sb, sc) = measured(|| r.sum(&w, "a").unwrap());
        assert_eq!(sum, 7 * kept as i64);
        let (rows, mb, mc) = measured(|| r.materialize_rows(&w).unwrap());
        assert_eq!(rows.len(), kept);
        let held = scratch + 8 * words + 4 * kept;
        assert!(
            cb >= scratch && sb >= scratch && mb >= held,
            "n {n}: {cb} {sb} {mb}"
        );
        calls_at.push((cc, sc, mc));
        fixed_at.push((cb - scratch, sb - scratch, mb - held));
    }
    assert!(
        calls_at.windows(2).all(|p| p[0] == p[1]),
        "allocation calls vary with n: {calls_at:?}"
    );
    assert!(
        fixed_at.windows(2).all(|p| p[0] == p[1]),
        "bookkeeping bytes vary with n: {fixed_at:?}"
    );
    // Measured 1573 bytes (count) and 3316 (materialize) for this one-term
    // recipe: the prepared program's ops (12 for one term in the
    // three-valued program) and the reads list. materialize prepares two
    // programs, keep and the unknown count it reports in `Rows::unknown`,
    // over one scratch. It grows with terms, never with rows; the equality
    // above is what pins that.
    let (c, _, m) = fixed_at[0];
    assert!(
        c < 2048 && m < 4096,
        "bookkeeping is not small: {fixed_at:?}"
    );
}

/// The accepting path of a prepared recipe allocates nothing: with the
/// program prepared and the scratch and kept-mask buffers sized once,
/// `run_into` takes 0 bytes and 0 calls at every n, for `Count`, `Sum` and
/// `Keep`. The rule #2045's `admitted` set (fixed lane array, caller-owned
/// scratch and output). Disables it catches: lanes collected into a `Vec`,
/// a program or scratch built inside `run_into`.
#[test]
fn prepared_accepting_path_allocates_nothing() {
    use lance_graph_mask_risc::{Out, Value};
    for &n in &[64usize, 512, 4096, 65536] {
        let source = Source::from_lanes([
            (0..n as i32).collect(),
            (0..n as i32).map(|i| i % 2).collect(),
            vec![0; n],
            vec![7; n],
        ]);
        let world = bind(&source, &all_fields());
        let w = witness(&world, &source);
        let r = Recipe::compile(SRC, &common::View::new(), VIEW_CLASS).unwrap();
        let rows = r.n_rows(&w).unwrap();
        let (pc, ps, pk) = (
            r.prepare_count(rows).unwrap(),
            r.prepare_sum(rows, "a").unwrap(),
            r.prepare_keep(rows).unwrap(),
        );
        let (mut sc, mut ss, mut sk) = (
            pc.scratch().unwrap(),
            ps.scratch().unwrap(),
            pk.scratch().unwrap(),
        );
        let mut mask = vec![0u64; n.div_ceil(64)];
        let kept = n / 2;
        let (v, b, c) = measured(|| r.run_into(&pc, &w, &mut sc, Out::None).unwrap());
        assert_eq!(v, Value::Count(kept as _), "n {n}");
        assert_eq!((b, c), (0, 0), "count allocated at n {n}");
        let (v, b, c) = measured(|| r.run_into(&ps, &w, &mut ss, Out::None).unwrap());
        assert_eq!(v, Value::SumI64(7 * kept as i64), "n {n}");
        assert_eq!((b, c), (0, 0), "sum allocated at n {n}");
        let (_, b, c) = measured(|| r.run_into(&pk, &w, &mut sk, Out::Mask(&mut mask)).unwrap());
        assert_eq!((b, c), (0, 0), "keep allocated at n {n}");
        let ones: u32 = mask.iter().map(|w| w.count_ones()).sum();
        assert_eq!(ones as usize, kept, "keep wrote the wrong mask at n {n}");
    }
}
