//! Focused reporting benchmarks (not a dashboard). Run:
//!
//! ```sh
//! cargo run --release -p lance-graph-report --example report_bench
//! ```
//!
//! Prints the SIMD tier the binary was built for, because a timing without
//! its target is an anecdote. Every row reports bytes allocated during the
//! measured call (a counting allocator), which is the number the zero-copy
//! claims rest on; the timings are context.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use lance_graph_report::boundary::{CamLabels, Catalog, MemKv};
use lance_graph_report::render::Terminal;
use lance_graph_report::*;

struct Counting;
static BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        // SAFETY: forwarded verbatim to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: forwarded verbatim to the system allocator.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn bench<T>(name: &str, reps: u32, mut f: impl FnMut() -> T) -> T {
    let mut out = f();
    let b0 = BYTES.load(Ordering::Relaxed);
    let t = Instant::now();
    for _ in 0..reps {
        out = f();
    }
    let ns = t.elapsed().as_nanos() as f64 / f64::from(reps);
    let bytes = (BYTES.load(Ordering::Relaxed) - b0) / u64::from(reps);
    println!("{name:<44} {:>12.0} ns   {:>10} B alloc/call", ns, bytes);
    out
}

fn main() {
    let n: usize = 4 << 20;
    let (fa, fb, fv) = (FieldId(0), FieldId(1), FieldId(2));
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let a: Arc<[u32]> = (0..n).map(|_| (next() % 32) as u32).collect();
    let b: Arc<[u32]> = (0..n).map(|_| (next() % 12) as u32).collect();
    let v: Arc<[i32]> = (0..n).map(|_| (next() % 1000) as i32).collect();
    let focus: Arc<[u64]> = (0..n.div_ceil(64)).map(|_| next() & next()).collect();
    let batch = AbiBatch::new(SourceId(1), 1, n)
        .with_column(Column::coordinate(fa, a, 32))
        .unwrap()
        .with_column(Column::coordinate(fb, b, 12))
        .unwrap()
        .with_column(Column::value(fv, LaneData::I32(v)))
        .unwrap()
        .with_mask(MaskId(1), focus)
        .unwrap();
    println!(
        "rows {n}   avx512f={} avx2={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    let pol = PlannerPolicy::default();
    let src = SourceRef {
        id: SourceId(1),
        generation: 1,
    };
    let range = Selection::Range(RowRange {
        lo: 1000,
        hi: (n - 1000) as u32,
    });
    let base = ReportPlan::over(src);
    let sum = Measure::of(MeasureKind::Sum, fv);

    let p1 = base.clone().filter(range.clone()).measure(Measure::count());
    bench("1 range → count", 20, || {
        p1.execute(&batch, &pol).unwrap()
    });
    let p2 = base.clone().filter(range.clone()).measure(sum.clone());
    bench("2 range → sum", 20, || p2.execute(&batch, &pol).unwrap());
    let p3 = base
        .clone()
        .filter(Selection::Mask(MaskId(1)))
        .measure(sum.clone());
    bench("3 resident mask → sum", 20, || {
        p3.execute(&batch, &pol).unwrap()
    });
    let p4 = base
        .clone()
        .filter(Selection::Mask(MaskId(1)))
        .filter(range)
        .measure(sum.clone());
    bench("4 mask ∧ range → sum", 20, || {
        p4.execute(&batch, &pol).unwrap()
    });
    let p5 = base
        .clone()
        .axis(CoordSpec::Field(fa), AxisRole::Row)
        .measure(sum.clone());
    bench("5 grouped sum (32 groups)", 20, || {
        p5.execute(&batch, &pol).unwrap()
    });
    let p6 = base
        .clone()
        .pivot(&[CoordSpec::Field(fa)], &[CoordSpec::Field(fb)])
        .measure(sum.clone())
        .filter(Selection::cmp(fv, CmpOp::Ge, Scalar::Int(100)));
    let (res, st) = bench("6 2D pivot fold 32×12, filtered", 5, || {
        p6.execute(&batch, &pol).unwrap()
    });
    println!(
        "  passes/scans {}  mask materializations {}  accumulator {} B",
        st.population_scans, st.mask_materializations, st.accumulator_bytes
    );
    let rp = p6.clone().rotate();
    bench("7a pivot rotation (result view)", 100_000, || res.rotate());
    bench("7b pivot rotation (plan reinterpret)", 100_000, || {
        res.reinterpret(&rp).unwrap()
    });
    let (cam, kv, cat) = (CamLabels::default(), MemKv::default(), Catalog::default());
    let t = Terminal {
        cam: &cam,
        kv: &kv,
        catalog: &cat,
    };
    bench("8 JSON materialization (384 cells)", 1000, || t.json(&res));
}
