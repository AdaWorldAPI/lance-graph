//! D-DT64-GEMM-SHAPE-1/2 benchmark: 64 x 64 and 256 x 256 virtual objects.
//!
//! `cargo run --release --bin shape`. Every arm is checked against the
//! scalar oracle before timing and after the timed runs. Times are medians
//! of warm runs; allocations and peak temporary bytes come from a counting
//! global allocator, measured on one extra run of each operation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use dr_isa_stream_lab::bind::{World, SPACE_A};
use dr_isa_stream_lab::fixture::Fixture;
use dr_isa_stream_lab::shape::{
    cblas, direct, oracle, staged, BOperand, PreparedB, Selection, VirtualObject,
};
use dr_isa_stream_lab::source::{Bf16Lanes, Socket};
use lance_graph_dt64::Extent;
use ndarray::simd::bf16_tile_gemm_tier;

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        let now = LIVE.fetch_add(l.size(), Relaxed) + l.size();
        PEAK.fetch_max(now, Relaxed);
        // SAFETY: forwards the caller's layout to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        // SAFETY: `p` came from `alloc` with the same layout.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// (allocations, peak temporary bytes above the live set) of one call.
fn mem<R>(f: impl FnOnce() -> R) -> (usize, usize, R) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let a0 = ALLOCS.load(Relaxed);
    let r = f();
    (ALLOCS.load(Relaxed) - a0, PEAK.load(Relaxed) - base, r)
}

const RUNS: usize = 41;

fn median_ns(mut f: impl FnMut()) -> f64 {
    f();
    let mut t: Vec<f64> = (0..RUNS)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_nanos() as f64
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[RUNS / 2]
}

fn fmt(ns: f64) -> String {
    if ns < 10_000.0 {
        format!("{ns:.0} ns")
    } else {
        format!("{:.1} µs", ns / 1000.0)
    }
}

fn cache_info() -> String {
    let mut out = Vec::new();
    for idx in 0..8 {
        let base = format!("/sys/devices/system/cpu/cpu0/cache/index{idx}");
        let read =
            |f: &str| std::fs::read_to_string(format!("{base}/{f}")).map(|s| s.trim().to_string());
        if let (Ok(level), Ok(kind), Ok(size)) = (read("level"), read("type"), read("size")) {
            out.push(format!("L{level} {kind} {size}"));
        }
    }
    if out.is_empty() {
        "not reported".into()
    } else {
        out.join(", ")
    }
}

struct Row {
    metric: &'static str,
    v: [String; 2],
}

fn main() {
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap().trim().to_string())
        })
        .unwrap_or_default();
    println!(
        "Host: {model}; {} logical CPUs; cache (cpu0): {}",
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        cache_info()
    );
    println!(
        "Tile tier: {}. Single-threaded. {RUNS} warm runs per row (median).\n",
        bf16_tile_gemm_tier()
    );

    let n = 65_536;
    let b = BOperand {
        seed: 0xB0B,
        rev: 1,
    };
    let mut rows: Vec<Row> = Vec::new();
    let mut push = |metric: &'static str, col: usize, v: String| {
        if let Some(r) = rows.iter_mut().find(|r| r.metric == metric) {
            r.v[col] = v;
        } else {
            let mut r = Row {
                metric,
                v: [String::new(), String::new()],
            };
            r.v[col] = v;
            rows.push(r);
        }
    };

    for (col, dim) in [64usize, 256].into_iter().enumerate() {
        let fx = Fixture {
            n,
            props: dim,
            seed: 0x5A9E,
        };
        let extent = if dim == 64 {
            Extent::DR64_CUBE
        } else {
            Extent::DR256_SQ_X64
        };
        let world = World::with_extent(dim, "isa:f", extent);
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, dim);
        let fields: Vec<u8> = (0..dim).map(|f| f as u8).collect();
        let sel = Selection::region(SPACE_A, 4096, dim);
        let want: Vec<f32> = oracle(
            &sel,
            &fields,
            dim,
            |i, p| f64::from(fx.value(i, p)),
            |i, j| f64::from(b.value(i, j)),
        )
        .iter()
        .map(|&v| v as f32)
        .collect();

        let bind = || VirtualObject::bind(&world, &socket, &fields, dim, sel.clone()).unwrap();
        let obj = bind();
        let (prep, prep_rep) = PreparedB::prepare(&obj, &b);
        let lanes_r = obj.resolve(&world, &socket).unwrap();
        let mut c = vec![0f32; obj.c_len()];
        let mut scratch = Vec::new();
        let (mut am, mut bm) = (Vec::new(), Vec::new());

        // Correctness before timing.
        let dr = direct(&obj, &lanes_r, &prep, b.rev, &mut c).unwrap();
        assert_eq!(c, want, "direct {dim}");
        let sr = staged(&obj, &lanes_r, &prep, b.rev, &mut scratch, &mut c).unwrap();
        assert_eq!(c, want, "staged {dim}");
        let cr = cblas(&obj, &lanes_r, &b, &mut am, &mut bm, &mut c);
        assert_eq!(c, want, "cblas {dim}");

        let t_bind = median_ns(|| {
            black_box(bind());
        });
        let t_resolve = median_ns(|| {
            black_box(obj.resolve(&world, &socket).unwrap());
        });
        let t_resolve_pf = median_ns(|| {
            black_box(obj.resolve_per_field(&world, &socket).unwrap());
        });
        let t_prep = median_ns(|| {
            black_box(PreparedB::prepare(&obj, &b));
        });
        let t_direct = median_ns(|| {
            direct(&obj, &lanes_r, &prep, b.rev, &mut c).unwrap();
        });
        assert_eq!(c, want);
        // Same values and region, pitch n + 32 (not a multiple of 4 KiB): does
        // the power-of-two row stride of the 64K universe cost the direct read?
        let fx2 = Fixture { n: n + 32, ..fx };
        let lanes2 = Bf16Lanes::slab_in_order(&fx2).unwrap();
        let socket2 = Socket::in_order(&lanes2, dim);
        let obj2 = VirtualObject::bind(&world, &socket2, &fields, dim, sel.clone()).unwrap();
        let (prep2, _) = PreparedB::prepare(&obj2, &b);
        let l2 = obj2.resolve(&world, &socket2).unwrap();
        direct(&obj2, &l2, &prep2, b.rev, &mut c).unwrap();
        assert_eq!(c, want, "direct, odd pitch {dim}");
        let t_direct_odd = median_ns(|| {
            direct(&obj2, &l2, &prep2, b.rev, &mut c).unwrap();
        });
        assert_eq!(c, want);
        let t_staged = median_ns(|| {
            staged(&obj, &lanes_r, &prep, b.rev, &mut scratch, &mut c).unwrap();
        });
        assert_eq!(c, want);
        let t_cblas = median_ns(|| {
            cblas(&obj, &lanes_r, &b, &mut am, &mut bm, &mut c);
        });
        assert_eq!(c, want);
        let t_warm = median_ns(|| {
            let l = obj.resolve(&world, &socket).unwrap();
            direct(&obj, &l, &prep, b.rev, &mut c).unwrap();
        });
        assert_eq!(c, want);
        let t_cold = median_ns(|| {
            let o = bind();
            let (p, _) = PreparedB::prepare(&o, &b);
            let l = o.resolve(&world, &socket).unwrap();
            let mut cc = vec![0f32; o.c_len()];
            direct(&o, &l, &p, b.rev, &mut cc).unwrap();
            black_box(cc);
        });
        // 100 repetitions with unchanged selection: one prepare, 100 x (resolve + direct).
        let s = Instant::now();
        let (p, _) = PreparedB::prepare(&obj, &b);
        for _ in 0..100 {
            let l = obj.resolve(&world, &socket).unwrap();
            direct(&obj, &l, &p, b.rev, &mut c).unwrap();
        }
        let t_rep = s.elapsed().as_nanos() as f64 / 100.0;
        assert_eq!(c, want);

        let (a_bind, m_bind, _) = mem(bind);
        let (a_res, m_res, _) = mem(|| obj.resolve(&world, &socket).unwrap());
        let (a_prep, m_prep, _) = mem(|| PreparedB::prepare(&obj, &b));
        let (a_dir, m_dir, _) = mem(|| direct(&obj, &lanes_r, &prep, b.rev, &mut c).unwrap());
        let (a_stg, m_stg, _) =
            mem(|| staged(&obj, &lanes_r, &prep, b.rev, &mut Vec::new(), &mut c).unwrap());
        let (a_cbl, m_cbl, _) =
            mem(|| cblas(&obj, &lanes_r, &b, &mut Vec::new(), &mut Vec::new(), &mut c));

        let macs = (dim * dim * dim) as f64;
        push(
            "Output tiles / K=32 dot-product steps",
            col,
            format!("{} / {}", dr.tiles, dr.dp_steps),
        );
        push(
            "DT64 bind (cold)",
            col,
            format!("{} · {a_bind} allocs", fmt(t_bind)),
        );
        push(
            "Resolve: one check + borrows (warm)",
            col,
            format!("{} · {a_res} allocs · {m_res} B", fmt(t_resolve)),
        );
        push("Resolve: `lane()` per field (warm)", col, fmt(t_resolve_pf));
        push(
            "B preparation (VNNI panels)",
            col,
            format!(
                "{} · {} B written · {} B alloc",
                fmt(t_prep),
                prep_rep.bytes_written,
                m_prep
            ),
        );
        push(
            "Direct AMX (A in place)",
            col,
            format!(
                "{} · {:.1} GMAC/s · {a_dir} allocs · {m_dir} B tmp",
                fmt(t_direct),
                macs / t_direct
            ),
        );
        push(
            "Direct AMX, pitch 65,568 (not 2^k)",
            col,
            format!("{} · {:.1} GMAC/s", fmt(t_direct_odd), macs / t_direct_odd),
        );
        push(
            "Staged AMX (dense A tiles)",
            col,
            format!("{} · {a_stg} allocs · {m_stg} B tmp", fmt(t_staged)),
        );
        push(
            "ndarray CBLAS BF16",
            col,
            format!("{} · {a_cbl} allocs · {m_cbl} B tmp", fmt(t_cblas)),
        );
        push(
            "Source bytes copied (direct / staged / CBLAS)",
            col,
            format!("{} / {} / {}", dr.a_copied, sr.a_copied, cr.a_copied),
        );
        push(
            "CBLAS B materialized",
            col,
            format!("{} B", cr.b_materialized),
        );
        push(
            "FP32 accumulator (C)",
            col,
            format!("{} B", obj.c_len() * 4),
        );
        push(
            "End-to-end warm (resolve + direct, B reused)",
            col,
            fmt(t_warm),
        );
        push(
            "End-to-end cold (bind + prepare + resolve + direct)",
            col,
            fmt(t_cold),
        );
        push(
            "Warm repetitions (100, per run, one prepare)",
            col,
            fmt(t_rep),
        );
        let _ = (m_bind, a_prep);
    }

    // Sparse demand: 40 scattered identities, 256 fields x 256 columns.
    {
        let dim = 256;
        let fx = Fixture {
            n,
            props: dim,
            seed: 0x5A9E,
        };
        let world = World::with_extent(dim, "isa:f", Extent::DR256_SQ_X64);
        let lanes = Bf16Lanes::slab_in_order(&fx).unwrap();
        let socket = Socket::in_order(&lanes, dim);
        let fields: Vec<u8> = (0..dim).map(|f| f as u8).collect();
        let sel = Selection::mask(SPACE_A, n, (0..40).map(|k| (k * 1637 + 11) % n));
        let want: Vec<f32> = oracle(
            &sel,
            &fields,
            dim,
            |i, p| f64::from(fx.value(i, p)),
            |i, j| f64::from(b.value(i, j)),
        )
        .iter()
        .map(|&v| v as f32)
        .collect();
        let obj = VirtualObject::bind(&world, &socket, &fields, dim, sel.clone()).unwrap();
        let (prep, _) = PreparedB::prepare(&obj, &b);
        let l = obj.resolve(&world, &socket).unwrap();
        let mut c = vec![0f32; obj.c_len()];
        let mut scratch = Vec::new();
        let r = staged(&obj, &l, &prep, b.rev, &mut scratch, &mut c).unwrap();
        assert_eq!(c, want);
        let t = median_ns(|| {
            staged(&obj, &l, &prep, b.rev, &mut scratch, &mut c).unwrap();
        });
        assert_eq!(c, want);
        println!(
            "Sparse demand (256 fields x 256 cols, 40 scattered identities): logical selection {} · K padded {} · lanes borrowed {} · A bytes copied {} · dense dot-product steps {} · staged AMX {}\n",
            sel.len(),
            sel.k_pad(),
            dim,
            r.a_copied,
            r.dp_steps,
            fmt(t)
        );
    }

    println!("| Metric | 64×64 | 256×256 |\n|---|---|---|");
    for r in rows {
        println!("| {} | {} | {} |", r.metric, r.v[0], r.v[1]);
    }
}
