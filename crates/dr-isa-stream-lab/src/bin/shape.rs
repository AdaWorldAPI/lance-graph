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
use dr_isa_stream_lab::int8;
use dr_isa_stream_lab::shape::{
    cblas, direct, oracle, staged, BOperand, BTable, PreparedB, Selection, VirtualObject,
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
    // B read from a table built once, so B preparation times data movement
    // and not the synthetic operand's hash (which #1987's first run did).
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
        let bt = BTable::new(&b, n + 64, dim);
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
        let (prep, prep_rep) = PreparedB::prepare(&obj, &bt).unwrap();
        let lanes_r = obj.resolve(&world, &socket).unwrap();
        let mut c = vec![0f32; obj.c_len()];
        let mut scratch = Vec::new();
        let (mut am, mut bm) = (Vec::new(), Vec::new());

        // Correctness before timing.
        let dr = direct(&obj, &lanes_r, &prep, &bt, &mut c).unwrap();
        assert_eq!(c, want, "direct {dim}");
        let sr = staged(&obj, &lanes_r, &prep, &bt, &mut scratch, &mut c).unwrap();
        assert_eq!(c, want, "staged {dim}");
        let cr = cblas(&obj, &lanes_r, &bt, &mut am, &mut bm, &mut c).unwrap();
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
            black_box(PreparedB::prepare(&obj, &bt).unwrap());
        });
        let t_direct = median_ns(|| {
            direct(&obj, &lanes_r, &prep, &bt, &mut c).unwrap();
        });
        assert_eq!(c, want);
        // Same values and region, pitch n + 32 (not a multiple of 4 KiB): does
        // the power-of-two row stride of the 64K universe cost the direct read?
        let fx2 = Fixture { n: n + 32, ..fx };
        let lanes2 = Bf16Lanes::slab_in_order(&fx2).unwrap();
        let socket2 = Socket::in_order(&lanes2, dim);
        let obj2 = VirtualObject::bind(&world, &socket2, &fields, dim, sel.clone()).unwrap();
        let (prep2, _) = PreparedB::prepare(&obj2, &bt).unwrap();
        let l2 = obj2.resolve(&world, &socket2).unwrap();
        direct(&obj2, &l2, &prep2, &bt, &mut c).unwrap();
        assert_eq!(c, want, "direct, odd pitch {dim}");
        let t_direct_odd = median_ns(|| {
            direct(&obj2, &l2, &prep2, &bt, &mut c).unwrap();
        });
        assert_eq!(c, want);
        let t_staged = median_ns(|| {
            staged(&obj, &lanes_r, &prep, &bt, &mut scratch, &mut c).unwrap();
        });
        assert_eq!(c, want);
        let t_cblas = median_ns(|| {
            cblas(&obj, &lanes_r, &bt, &mut am, &mut bm, &mut c).unwrap();
        });
        assert_eq!(c, want);
        let t_warm = median_ns(|| {
            let l = obj.resolve(&world, &socket).unwrap();
            direct(&obj, &l, &prep, &bt, &mut c).unwrap();
        });
        assert_eq!(c, want);
        let t_cold = median_ns(|| {
            let o = bind();
            let (p, _) = PreparedB::prepare(&o, &bt).unwrap();
            let l = o.resolve(&world, &socket).unwrap();
            let mut cc = vec![0f32; o.c_len()];
            direct(&o, &l, &p, &bt, &mut cc).unwrap();
            black_box(cc);
        });
        // 100 repetitions with unchanged selection: one prepare, 100 x (resolve + direct).
        let s = Instant::now();
        let (p, _) = PreparedB::prepare(&obj, &bt).unwrap();
        for _ in 0..100 {
            let l = obj.resolve(&world, &socket).unwrap();
            direct(&obj, &l, &p, &bt, &mut c).unwrap();
        }
        let t_rep = s.elapsed().as_nanos() as f64 / 100.0;
        assert_eq!(c, want);

        let (a_bind, m_bind, _) = mem(bind);
        let (a_res, m_res, _) = mem(|| obj.resolve(&world, &socket).unwrap());
        let (a_prep, m_prep, _) = mem(|| PreparedB::prepare(&obj, &bt).unwrap());
        let (a_dir, m_dir, _) = mem(|| direct(&obj, &lanes_r, &prep, &bt, &mut c).unwrap());
        let (a_stg, m_stg, _) =
            mem(|| staged(&obj, &lanes_r, &prep, &bt, &mut Vec::new(), &mut c).unwrap());
        let (a_cbl, m_cbl, _) = mem(|| {
            cblas(
                &obj,
                &lanes_r,
                &bt,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut c,
            )
            .unwrap()
        });

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

        // INT8: the same object over u8 lanes, u8 x i8 -> i32. ndarray's INT8
        // facade takes dense A and packs B inside each call, so the only arm
        // is staged; the stage costs are timed apart from the GEMM.
        let u8l = int8::U8Lanes::slab_in_order(&fx);
        let u8s = int8::U8Socket::in_order(&u8l, dim);
        let o8 = VirtualObject::bind(&world, &u8s, &fields, dim, sel.clone()).unwrap();
        let want8: Vec<i32> = int8::oracle(&fx, &sel, &fields, dim, &b)
            .iter()
            .map(|&v| i32::try_from(v).unwrap())
            .collect();
        let l8 = int8::resolve(&o8, &world, &u8s).unwrap();
        let (mut a8, mut b8, mut c8) = (Vec::new(), Vec::new(), vec![0i32; o8.c_len()]);
        let r8 = int8::staged(&o8, &l8, &bt, &mut a8, &mut b8, &mut c8).unwrap();
        assert_eq!(c8, want8, "int8 {dim}");
        let t8_resolve = median_ns(|| {
            black_box(int8::resolve(&o8, &world, &u8s).unwrap());
        });
        let t8_a = median_ns(|| {
            int8::stage_a(&o8, &l8, &mut a8).unwrap();
        });
        let t8_a_pv = median_ns(|| {
            int8::stage_a_per_value(&o8, &l8, &mut a8).unwrap();
        });
        let t8_b_inplace = median_ns(|| {
            black_box(int8::b_operand(&o8, &bt, &mut b8));
        });
        let t8_b = median_ns(|| {
            int8::stage_b(&o8, &bt, &mut b8);
        });
        let t8_gemm = median_ns(|| int8::gemm(&o8, &a8, &b8, &mut c8));
        assert_eq!(c8, want8);
        // The B packing gemm_i8 does inside every call, timed on its own:
        // per 16-column band, gather the band (K x 16) and VNNI-pack it with
        // ndarray's own `vnni_pack_i8` (int8_gemm_amx_tiled_rb's steps, minus
        // its three buffer allocations, which the GEMM row reports).
        let k8 = r8.k_pad;
        assert_eq!(b8.len(), k8 * dim, "B materialized row-major");
        let (mut band, mut packed) = (vec![0i8; k8 * 16], vec![0i8; k8 * 16]);
        let t8_pack = median_ns(|| {
            for j in (0..dim).step_by(16) {
                for kk in 0..k8 {
                    band[kk * 16..kk * 16 + 16]
                        .copy_from_slice(&b8[kk * dim + j..kk * dim + j + 16]);
                }
                ndarray::hpc::amx_matmul::vnni_pack_i8(&band, &mut packed, k8, 16);
                black_box(&packed);
            }
        });
        let t8_all = median_ns(|| {
            let l = int8::resolve(&o8, &world, &u8s).unwrap();
            int8::staged(&o8, &l, &bt, &mut a8, &mut b8, &mut c8).unwrap();
        });
        assert_eq!(c8, want8);
        let (a8_allocs, m8_gemm, _) = mem(|| int8::gemm(&o8, &a8, &b8, &mut c8));
        push("INT8 resolve (u8 lanes, one check)", col, fmt(t8_resolve));
        push(
            "INT8 stage A (region: one copy per lane)",
            col,
            format!("{} · {} B", fmt(t8_a), r8.a_copied),
        );
        push("INT8 stage A, per-value walk (baseline)", col, fmt(t8_a_pv));
        push(
            "INT8 B for a region, in place from the table",
            col,
            format!(
                "{} · {} B written",
                fmt(t8_b_inplace),
                int8::b_operand(&o8, &bt, &mut b8).1
            ),
        );
        push(
            "INT8 materialize B (row-major i8)",
            col,
            format!("{} · {} B", fmt(t8_b), int8::stage_b(&o8, &bt, &mut b8)),
        );
        push(
            "INT8 GEMM `backend::gemm_i8` (packs B per call)",
            col,
            format!(
                "{} · {:.1} GMAC/s · {a8_allocs} allocs · {m8_gemm} B tmp",
                fmt(t8_gemm),
                macs / t8_gemm
            ),
        );
        push(
            "INT8 B band gather + VNNI pack (done inside each gemm_i8 call)",
            col,
            format!(
                "{} · {:.0}% of the GEMM row",
                fmt(t8_pack),
                100.0 * t8_pack / t8_gemm
            ),
        );
        push(
            "INT8 end-to-end warm (resolve + stage + GEMM)",
            col,
            fmt(t8_all),
        );
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
        let bt = BTable::new(&b, n, dim);
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
        let (prep, _) = PreparedB::prepare(&obj, &bt).unwrap();
        let l = obj.resolve(&world, &socket).unwrap();
        let mut c = vec![0f32; obj.c_len()];
        let mut scratch = Vec::new();
        let r = staged(&obj, &l, &prep, &bt, &mut scratch, &mut c).unwrap();
        assert_eq!(c, want);
        let t = median_ns(|| {
            staged(&obj, &l, &prep, &bt, &mut scratch, &mut c).unwrap();
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
