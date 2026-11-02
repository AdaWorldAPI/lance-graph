//! Cold and hot costs, reported, not gated.
//!
//! `cargo run --release --example cost` (the Jina rows need
//! `python3 tools/fetch_jina_slice.py` first; they are skipped and said so
//! otherwise).
//!
//! Cold: source inventory (raw protobuf), Tract typing (facts), laying the
//! graph onto the DR#3 tile and resolving every node against the registry.
//! Hot: one plan run (routing each node to its DO), against Tract's own
//! runnable of the same typed graph. Allocations are counted per hot call.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use dr_math_binding_probe::FrameStore;
use ndarray::{ArrayD, IxDyn};
use tract_harvest_probe::dr3::{tract_facts, OnnxGraph};
use tract_harvest_probe::plan::{resolve, revisions};
use tract_harvest_probe::registry::registry;
use tract_harvest_probe::source::SourceInventory;
use tract_harvest_probe::value::Value;
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

fn typed(bytes: &[u8]) -> TypedModel {
    let onnx = tract_onnx::onnx();
    onnx.model_for_proto_model(&onnx.proto_model_for_read(&mut &*bytes).unwrap())
        .unwrap()
        .into_typed()
        .unwrap()
}

/// Cold costs, then the hot plan run against Tract's typed runnable.
fn measure(
    label: &str,
    bytes: &[u8],
    pins: &[(usize, DatumType, Vec<usize>)],
    inputs: Vec<(&str, Value)>,
    tract_in: impl Fn() -> TVec<TValue>,
    runs: usize,
) {
    let reg = registry();
    let src_us = median(51, || {
        black_box(SourceInventory::from_bytes(black_box(bytes)).unwrap());
    }) / 1e3;
    let facts_us = median(51, || {
        black_box(tract_facts(black_box(bytes), pins).unwrap());
    }) / 1e3;
    let src = SourceInventory::from_bytes(bytes).unwrap();
    let facts = tract_facts(bytes, pins).unwrap();
    let resolve_us = median(51, || {
        let g = OnnxGraph::build(&src, &facts).unwrap();
        let store = FrameStore::new(revisions(&src, &reg, &g));
        black_box(resolve(&src, g, &reg, &store).unwrap());
    }) / 1e3;
    let g = OnnxGraph::build(&src, &facts).unwrap();
    let store = FrameStore::new(revisions(&src, &reg, &g));
    let p = resolve(&src, g, &reg, &store).unwrap();
    let hot_ns = median(runs, || {
        black_box(p.run(&store, black_box(&inputs)).unwrap());
    });
    let (_, na, nb) = counted(|| p.run(&store, &inputs).unwrap());
    let plan = typed(bytes).into_runnable().unwrap();
    let tract_ns = median(runs, || {
        black_box(plan.run(tract_in()).unwrap());
    });
    let retained: usize = p
        .folded
        .iter()
        .flat_map(|b| p.graph.outs[b.node].iter())
        .filter_map(|&o| p.graph.consts[o].as_ref())
        .count();
    println!("{label:>5} cold: source {src_us:8.1} µs | Tract facts {facts_us:8.1} µs | tile+fold+resolve {resolve_us:7.1} µs ({} steps, {} folded, {retained} folded constants)", p.steps.len(), p.folded.len());
    println!("{label:>5} hot:  graph plan {hot_ns:9.0} ns ({na} allocs, {nb} B) | Tract typed runnable {tract_ns:9.0} ns (incl. input tensors)");
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");

    // ── fixture A ──
    let tiny = std::fs::read(format!("{dir}/tiny/div3.onnx")).unwrap();
    let x = vec![3, -1, 4, -1, 5, -9, 2];
    let f = vec![1.5f32, -0.25, 3.0, 0.1, 7.0, -2.5];
    let ins = vec![
        (
            "X",
            Value::I32(ArrayD::from_shape_vec(IxDyn(&[7]), x.clone()).unwrap()),
        ),
        (
            "F",
            Value::F32(ArrayD::from_shape_vec(IxDyn(&[2, 3]), f.clone()).unwrap()),
        ),
    ];
    measure(
        "tiny",
        &tiny,
        &[],
        ins,
        || {
            tvec![
                tract_ndarray::Array1::from(x.clone())
                    .into_dyn()
                    .into_tensor()
                    .into(),
                tract_ndarray::Array2::from_shape_vec((2, 3), f.clone())
                    .unwrap()
                    .into_dyn()
                    .into_tensor()
                    .into()
            ]
        },
        20001,
    );

    // ── shape folding (D-TRACT-SHAPE-1) ──
    let shape = std::fs::read(format!("{dir}/tiny/shape_fold.onnx")).unwrap();
    measure(
        "shape",
        &shape,
        &[],
        vec![(
            "F",
            Value::F32(ArrayD::from_shape_vec(IxDyn(&[2, 3]), f.clone()).unwrap()),
        )],
        || {
            tvec![tract_ndarray::Array2::from_shape_vec((2, 3), f.clone())
                .unwrap()
                .into_dyn()
                .into_tensor()
                .into()]
        },
        20001,
    );
    let erf = std::fs::read(format!("{dir}/tiny/erf_sweep.onnx")).unwrap();
    let ex: Vec<f32> = (0..4096).map(|i| -6.0 + 12.0 * i as f32 / 4095.0).collect();
    measure(
        "erf",
        &erf,
        &[(0, DatumType::F32, vec![ex.len()])],
        vec![(
            "X",
            Value::F32(ArrayD::from_shape_vec(IxDyn(&[ex.len()]), ex.clone()).unwrap()),
        )],
        || {
            tvec![tract_ndarray::Array1::from(ex.clone())
                .into_dyn()
                .into_tensor()
                .into()]
        },
        20001,
    );

    // ── checked int64 shape arithmetic (registry v4) ──
    let fin = || {
        tvec![tract_ndarray::Array2::from_shape_vec((2, 3), f.clone())
            .unwrap()
            .into_dyn()
            .into_tensor()
            .into()]
    };
    let fval = || Value::F32(ArrayD::from_shape_vec(IxDyn(&[2, 3]), f.clone()).unwrap());
    let i64a = std::fs::read(format!("{dir}/tiny/i64_arith.onnx")).unwrap();
    measure("i64", &i64a, &[], vec![("F", fval())], fin, 20001);
    let i64d = std::fs::read(format!("{dir}/tiny/i64_data.onnx")).unwrap();
    measure(
        "i64rt",
        &i64d,
        &[],
        vec![
            ("F", fval()),
            (
                "I",
                Value::I64(ArrayD::from_shape_vec(IxDyn(&[2]), vec![-4, 5]).unwrap()),
            ),
        ],
        || {
            let mut t = fin();
            t.push(
                tract_ndarray::Array1::from(vec![-4i64, 5])
                    .into_dyn()
                    .into_tensor()
                    .into(),
            );
            t
        },
        20001,
    );

    // ── fixture B ──
    let slice = match std::fs::read(format!("{dir}/jina/slice.onnx")) {
        Ok(b) => b,
        Err(_) => {
            println!("  jina NOT RUN: fixtures/jina absent (python3 tools/fetch_jina_slice.py)");
            return;
        }
    };
    let raw = std::fs::read(format!("{dir}/jina/gather_rows.f32")).unwrap();
    let xs: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let rows = xs.len() / 1024;
    let xa = ArrayD::from_shape_vec(IxDyn(&[1, rows, 1024]), xs).unwrap();
    let src = SourceInventory::from_bytes(&slice).unwrap();
    let input = src.inputs[0].clone();
    measure(
        "jina",
        &slice,
        &[(0, DatumType::F32, vec![1, rows, 1024])],
        vec![(input.as_str(), Value::F32(xa.clone()))],
        || tvec![xa.clone().into_tensor().into()],
        2001,
    );
}
