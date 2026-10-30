//! Cold and hot costs, reported, not gated.
//!
//! `cargo run --release --example cost` (the Jina rows need
//! `python3 tools/fetch_jina_slice.py` first; they are skipped and said so
//! otherwise).
//!
//! Cold: source inventory (raw protobuf), Tract import + into_typed, admission
//! (pattern match + recipe preparation). Hot: one native recipe run, against
//! Tract's own runnable of the same typed graph. Allocations are counted per
//! hot call.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ndarray::{Array2, ArrayD, IxDyn};
use tract_harvest_probe::admit;
use tract_harvest_probe::source::SourceInventory;
use tract_onnx::prelude::*;

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn counted<T>(f: impl FnOnce() -> T) -> (T, u64, u64) {
    let (a, b) = (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    );
    let r = f();
    (
        r,
        ALLOCS.load(Ordering::Relaxed) - a,
        BYTES.load(Ordering::Relaxed) - b,
    )
}

/// Median of `runs` timings, in µs (cold) or ns (hot).
fn median(runs: usize, mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_nanos() as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[runs / 2]
}

fn cold(label: &str, bytes: &[u8]) {
    let src_us = median(51, || {
        black_box(SourceInventory::from_bytes(black_box(bytes)).unwrap());
    }) / 1e3;
    let typed_us = median(51, || {
        black_box(admit::typed(black_box(bytes)).unwrap());
    }) / 1e3;
    println!("{label:>6} cold: source inventory {src_us:9.1} µs | Tract import+into_typed {typed_us:9.1} µs");
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");

    // ── fixture A ──
    let tiny = std::fs::read(format!("{dir}/tiny/div3.onnx")).unwrap();
    cold("tiny", &tiny);
    let src = SourceInventory::from_bytes(&tiny).unwrap();
    let typed = admit::typed(&tiny).unwrap();
    let admit_us = median(51, || {
        black_box(admit::admit_tiny(&src, &typed).unwrap());
    }) / 1e3;
    let r = admit::admit_tiny(&src, &typed).unwrap();
    let x = vec![3, -1, 4, -1, 5, -9, 2];
    let f = Array2::from_shape_vec((2, 3), vec![1.5f32, -0.25, 3.0, 0.1, 7.0, -2.5]).unwrap();
    let mut scratch = vec![0u64; 1];
    let native_ns = median(20001, || {
        black_box(r.run(black_box(&x), black_box(&f), &mut scratch).unwrap());
    });
    let (_, na, nb) = counted(|| r.run(&x, &f, &mut scratch).unwrap());
    let plan = typed.clone().into_runnable().unwrap();
    let inputs = || -> TVec<TValue> {
        tvec![
            tract_ndarray::Array1::from(x.clone())
                .into_dyn()
                .into_tensor()
                .into(),
            f.clone().into_dyn().into_tensor().into()
        ]
    };
    let tract_ns = median(20001, || {
        black_box(plan.run(inputs()).unwrap());
    });
    let (_, ta, tb) = counted(|| plan.run(inputs()).unwrap());
    println!("  tiny admission {admit_us:7.1} µs | hot native {native_ns:7.0} ns ({na} allocs, {nb} B) | Tract typed runnable {tract_ns:7.0} ns ({ta} allocs, {tb} B, incl. input tensors)");

    // ── fixture B ──
    let slice = match std::fs::read(format!("{dir}/jina/slice.onnx")) {
        Ok(b) => b,
        Err(_) => {
            println!("  jina NOT RUN: fixtures/jina absent (python3 tools/fetch_jina_slice.py)");
            return;
        }
    };
    cold("jina", &slice);
    let src = SourceInventory::from_bytes(&slice).unwrap();
    let typed = admit::typed(&slice).unwrap();
    let admit_us = median(51, || {
        black_box(admit::admit_rmsnorm(&src, &typed).unwrap());
    }) / 1e3;
    let r = admit::admit_rmsnorm(&src, &typed).unwrap();
    let raw = std::fs::read(format!("{dir}/jina/gather_rows.f32")).unwrap();
    let xs: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let rows = xs.len() / 1024;
    let x = ArrayD::from_shape_vec(IxDyn(&[1, rows, 1024]), xs).unwrap();
    let native_ns = median(2001, || {
        black_box(r.run(black_box(&x)).unwrap());
    });
    let (_, na, nb) = counted(|| r.run(&x).unwrap());
    let plan = typed.clone().into_runnable().unwrap();
    let tract_ns = median(2001, || {
        black_box(plan.run(tvec![x.clone().into_tensor().into()]).unwrap());
    });
    let decl = typed.into_decluttered().unwrap().into_runnable().unwrap();
    let decl_ns = median(2001, || {
        black_box(decl.run(tvec![x.clone().into_tensor().into()]).unwrap());
    });
    println!(
        "  jina admission {admit_us:7.1} µs | hot native {native_ns:7.0} ns ({na} allocs, {nb} B) for {rows}x1024 \
         | Tract typed {tract_ns:7.0} ns | Tract decluttered (fused RmsNorm) {decl_ns:7.0} ns"
    );
}
