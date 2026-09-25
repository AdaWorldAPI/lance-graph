//! What does one `ShaderDriver::dispatch` materialize, and does it grow with
//! the population it looks at?
//!
//! The driver's module doc claims "no allocations beyond top-k + edges". A
//! counting global allocator measures one dispatch (after a warm-up) at
//! several population sizes. Everything the cycle materializes shows up here;
//! nothing is inferred from reading the code.
//!
//! Before the fix: 23 KB / 44 allocations at 16 rows, 5.3 MB / 530 at 256,
//! for an answer of at most 8 hits. Now: a constant allocation count, and bytes
//! grow only per surviving row (its prefilter entry and its SPOFC candidate
//! record), never per pair.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bgz17::base17::Base17;
use bgz17::palette::Palette;
use bgz17::palette_semiring::PaletteSemiring;

use cognitive_shader_driver::bindspace::BindSpace;
use cognitive_shader_driver::driver::CognitiveShaderBuilder;
use cognitive_shader_driver::engine_bridge::ingest_codebook_indices;
use cognitive_shader_driver::{
    CognitiveShaderDriver, ColumnWindow, MetaFilter, ShaderDispatch, StyleSelector,
};

struct Counting;

static BYTES: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);

// SAFETY: a pure pass-through to `System`; the counters are the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        COUNT.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn palette_256() -> PaletteSemiring {
    let entries: Vec<Base17> = (0..256)
        .map(|i| {
            let mut dims = [0i16; 17];
            dims[0] = (i * 100 % 3400) as i16;
            dims[1] = ((i * 37) % 200) as i16;
            Base17 { dims }
        })
        .collect();
    PaletteSemiring::build(&Palette { entries })
}

fn planes_chain() -> [[u64; 64]; 8] {
    let mut planes = [[0u64; 64]; 8];
    for (i, row) in planes[0].iter_mut().enumerate().take(63) {
        *row |= 1u64 << (i + 1);
    }
    for (i, row) in planes[2].iter_mut().enumerate() {
        *row |= 1u64 << i;
    }
    planes
}

/// Bytes and allocation count of one dispatch over `n` rows.
fn measure(n: u32) -> (usize, usize, u16) {
    let mut bs = BindSpace::zeros(n as usize);
    let indices: Vec<u16> = (0..n as u16).collect();
    ingest_codebook_indices(&mut bs, &indices, 1, 1000, 0);
    let driver = CognitiveShaderBuilder::new()
        .bindspace(Arc::new(bs))
        .semiring(Arc::new(palette_256()))
        .planes(planes_chain())
        .build();
    let req = ShaderDispatch {
        rows: ColumnWindow::new(0, n),
        meta_prefilter: MetaFilter::ALL,
        layer_mask: 0xFF,
        radius: u16::MAX,
        style: StyleSelector::Auto,
        max_cycles: u16::MAX / 4,
        ..Default::default()
    };
    let _warm = driver.dispatch(&req);
    let (b0, c0) = (BYTES.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed));
    let crystal = driver.dispatch(&req);
    let (b1, c1) = (BYTES.load(Ordering::Relaxed), COUNT.load(Ordering::Relaxed));
    (b1 - b0, c1 - c0, crystal.bus.resonance.hit_count)
}

#[test]
fn trace_dispatch_materialization() {
    println!(
        "{:>6} {:>12} {:>12} {:>10}",
        "rows", "bytes", "allocs", "hit_count"
    );
    let sizes = [16u32, 32, 64, 128, 256];
    let rows: Vec<(u32, usize, usize, u16)> = sizes
        .iter()
        .map(|&n| {
            let (bytes, allocs, hits) = measure(n);
            println!("{n:>6} {bytes:>12} {allocs:>12} {hits:>10}");
            (n, bytes, allocs, hits)
        })
        .collect();
    let (n0, b0, a0, _) = rows[0];
    for &(n, bytes, allocs, hits) in &rows {
        assert!(hits > 0, "fixture must produce hits at {n} rows");
        // The cycle keeps at most 8 hits and 4 cascade results per row, so the
        // number of allocations must not depend on how many rows it looks at.
        assert_eq!(
            allocs, a0,
            "{n} rows: {allocs} allocations vs {a0} at {n0} rows"
        );
        // Per surviving row: the prefilter's row list (4 bytes) and one SPOFC
        // candidate record carrying that row's aggregated support (24 bytes).
        // Nothing grows with the number of PAIRS.
        let per_row_limit = 28 * (n - n0) as usize;
        assert!(
            bytes - b0 <= per_row_limit,
            "{n} rows: {bytes} bytes, {} more than at {n0} rows (limit {per_row_limit})",
            bytes - b0
        );
    }
}
