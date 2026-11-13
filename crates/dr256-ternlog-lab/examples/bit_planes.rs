//! A one-byte temperature code held as eight resident bit-planes, against the
//! widened `i32` lane and a scalar pass, over one ERA5 timestep. Every arm's
//! answer is checked against a row-at-a-time oracle before it is timed.
//!
//! Field: `ERA5_T2M_F32=<path>` names 721 x 1440 little-endian `f32` kelvin
//! (one `2m_temperature` timestep). Without it a synthetic field is used and
//! every line says so.
//!
//! Pass conditions: `.claude/board/entries/2026-11-13-weather-bit-planes.md`.

use dr256_ternlog_lab::bit_planes::{and, band_program, gt, lt, plane_op, BitPlanes};
use dr256_ternlog_lab::{scratch_bytes_of, AND3};
use lance_graph_mask_risc::{
    execute_into, fuse_program, BoolExpr, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred,
    Program, Scratch, Terminal, Value,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;
use weather_poc::floor::calibrate;

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
// SAFETY: forwards to the system allocator unchanged; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

const LAT: usize = 721;
const LON: usize = 1440;
const GRID: usize = LAT * LON;
const LO_K: f64 = 273.15;
const HI_K: f64 = 283.15;

/// 35N to 70N, 10W to 40E. Latitude index 0 is 90N; longitude index 0 is 0E.
fn in_box(cell: usize) -> bool {
    let (lat, lon) = (cell / LON, cell % LON);
    (80..=220).contains(&lat) && (lon <= 160 || lon >= 1400)
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn field() -> (Vec<f32>, &'static str) {
    if let Some(path) = std::env::var_os("ERA5_T2M_F32") {
        let bytes = std::fs::read(&path).expect("the field file reads");
        assert_eq!(bytes.len(), GRID * 4, "721 x 1440 f32");
        let v = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        return (v, "era5");
    }
    let v = (0..GRID)
        .map(|i| {
            let lat = (i / LON) as f64 / 720.0;
            (245.0
                + 55.0 * (std::f64::consts::PI * lat).sin()
                + (mix(i as u64) % 1500) as f64 / 100.0) as f32
        })
        .collect();
    (v, "SYNTHETIC")
}

fn plane_words(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; n.div_ceil(64)];
    for i in (0..n).filter(|&i| set(i)) {
        w[i / 64] |= 1 << (i % 64);
    }
    w
}

struct Row {
    arm: &'static str,
    layout: &'static str,
    /// Upper bound of source bytes one query reads.
    source: usize,
    /// Scratch held for the query's duration.
    temp: usize,
    /// Bindings checked per query.
    bindings: usize,
}

/// Median of 21 samples of `k` calls, in nanoseconds per call; also the
/// allocations of one warm call.
fn measure<T: PartialEq + std::fmt::Debug>(
    rows: usize,
    k: u32,
    want: &T,
    row: Row,
    f: &mut dyn FnMut() -> T,
) -> f64 {
    assert_eq!(&f(), want, "arm {}", row.arm);
    for _ in 0..3 {
        black_box(f());
    }
    let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    black_box(f());
    let (allocs, bytes) = (ALLOCS.load(Relaxed) - a0, BYTES.load(Relaxed) - b0);
    let mut samples = Vec::with_capacity(21);
    for _ in 0..21 {
        let t = Instant::now();
        for _ in 0..k {
            black_box(f());
        }
        samples.push(t.elapsed().as_nanos() as f64 / f64::from(k));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = samples[10];
    println!(
        "{rows}\t{}\t{:.1}\t{:.4}\t{allocs}\t{bytes}\t{}\t{}\t{}\t{}",
        row.arm,
        med / 1000.0,
        med / rows as f64,
        row.source,
        row.temp,
        row.bindings,
        row.layout
    );
    med
}

fn count_of(v: Value) -> usize {
    match v {
        Value::Count(c) => c,
        other => panic!("not a count: {other:?}"),
    }
}

fn sum_of(v: Value) -> i64 {
    match v {
        Value::SumI64(s) => s,
        other => panic!("not a sum: {other:?}"),
    }
}

/// Bounds gated under the region (the form `Fold::between` compiles).
fn gated_ops(q_lo: i32, q_hi: i32) -> Vec<MaskOp> {
    vec![
        MaskOp::Pred {
            pred: Pred::GtI32 { lane: 0, t: q_lo },
            under: Some(Operand::Plane(0)),
            dst: 0,
        },
        MaskOp::Pred {
            pred: Pred::LtI32 { lane: 0, t: q_hi },
            under: Some(Operand::Scratch(0)),
            dst: 1,
        },
        MaskOp::Ternlog {
            imm: AND3,
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            c: Operand::Scratch(1),
            dst: 2,
        },
    ]
}

fn main() {
    let (kelvin, source) = field();
    println!(
        "# field={source} avx512f={} avx2={} popcnt_avx512={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512vpopcntdq")
    );
    let n = GRID;
    let sample: Vec<f64> = kelvin.iter().map(|&k| f64::from(k)).collect();
    let floor = calibrate(&sample).expect("a finite field calibrates");
    let (f_lo, f_hi) = floor.bounds();
    let span = f_hi - f_lo;
    let bucket = span / 256.0;
    println!(
        "# floor v{}: [{f_lo:.3}, {f_hi:.3}] K, bucket {bucket:.4} K, band (32 buckets) {:.3} K",
        floor.floor_version(),
        32.0 * bucket
    );
    let codes: Vec<u8> = sample.iter().map(|&k| floor.quantize(k)).collect();
    let region: Vec<bool> = (0..n).map(in_box).collect();
    let (q_lo8, q_hi8) = (floor.quantize(LO_K), floor.quantize(HI_K));
    let (q_lo, q_hi) = (i32::from(q_lo8), i32::from(q_hi8));
    println!("# codes ({q_lo}, {q_hi}) exclusive");

    // ------------------------------------------------------------- oracles
    let want_count = codes
        .iter()
        .enumerate()
        .filter(|&(i, &c)| in_box(i) && q_lo < i32::from(c) && i32::from(c) < q_hi)
        .count();
    let want_sum: i64 = codes
        .iter()
        .enumerate()
        .filter(|&(i, &c)| in_box(i) && q_lo < i32::from(c) && i32::from(c) < q_hi)
        .map(|(_, &c)| i64::from(c))
        .sum();
    assert!(want_count > 0, "the query selects nothing");
    let (lo_b, hi_b) = (q_lo8 >> 5, q_hi8 >> 5);
    let subset = (lo_b..=hi_b).fold(0u8, |s, b| s | (1 << b));
    let want_band = codes
        .iter()
        .enumerate()
        .filter(|&(i, &c)| in_box(i) && (lo_b..=hi_b).contains(&(c >> 5)))
        .count();
    println!(
        "# range: rows={want_count} sum={want_sum}; band subset {subset:#04x} (bands {lo_b}..={hi_b}) rows={want_band}"
    );
    let edge = |code: u32| f_lo + f64::from(code) / 256.0 * span;
    println!(
        "# the band query selects {:.3} K to {:.3} K (the range query is ({:.3}, {:.3}) K)",
        edge(u32::from(lo_b) * 32),
        edge((u32::from(hi_b) + 1) * 32),
        edge(q_lo as u32),
        edge(q_hi as u32)
    );

    // ---------------------------------------------------------- preparation
    let t = Instant::now();
    let bp = BitPlanes::of(&codes);
    let build = t.elapsed();
    let region_w = plane_words(n, |i| region[i]);
    let lane = n * 4;
    let plane = n.div_ceil(64) * 8;
    let widened: Vec<i32> = codes.iter().map(|&c| i32::from(c)).collect();
    let valid_w = plane_words(n, |_| true);
    println!(
        "# cold: eight bit-planes built in {:.2} ms; held {} B planes + {} B region = {} B, widened i32 lane {} B ({:.2} x)",
        build.as_secs_f64() * 1e3,
        bp.bytes(),
        plane,
        bp.bytes() + plane,
        4 * n,
        (bp.bytes() + plane) as f64 / (4 * n) as f64
    );

    let k = (4_000_000 / n as u32).clamp(1, 32);
    println!("rows\tarm\tmedian_us\tns_per_row\tallocs\talloc_bytes\tsource_bytes_max\ttemp_bytes\tbindings\tlayout");
    let none = Foreign {
        planes: &[],
        lanes: &[],
    };

    // ------------------------------------------------------------- scalar
    measure(
        n,
        k,
        &want_count,
        Row {
            arm: "scalar_u8_range_count",
            layout: "contiguous",
            source: 2 * n,
            temp: 0,
            bindings: 0,
        },
        &mut || {
            let mut c = 0usize;
            for i in 0..n {
                let v = i32::from(codes[i]);
                c += usize::from(region[i] & (q_lo < v) & (v < q_hi));
            }
            c
        },
    );

    // ----------------------------------------------- widened i32, mask-RISC
    let lanes = [LaneRef::I32(&widened)];
    let i32_masks: [&[u64]; 2] = [&region_w, &valid_w];
    let i32_planes = Planes {
        n_rows: n,
        masks: &i32_masks,
        lanes: &lanes,
    };
    let m = Operand::Scratch(2);
    let gated_count = Program::new(gated_ops(q_lo, q_hi), Terminal::Count { mask: m });
    let gated_sum = Program::new(
        gated_ops(q_lo, q_hi),
        Terminal::MaskedSumI32 { mask: m, lane: 0 },
    );
    let ungated = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::GtI32 { lane: 0, t: q_lo },
                under: None,
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LtI32 { lane: 0, t: q_hi },
                under: None,
                dst: 1,
            },
            MaskOp::Ternlog {
                imm: AND3,
                a: Operand::Plane(0),
                b: Operand::Scratch(0),
                c: Operand::Scratch(1),
                dst: 2,
            },
            MaskOp::And {
                a: Operand::Scratch(2),
                b: Operand::Plane(1),
                dst: 3,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(3),
        },
    );

    let mut s_b = Scratch::for_program(&gated_count, n).unwrap();
    let gated_ns = measure(
        n,
        k,
        &want_count,
        Row {
            arm: "i32_gated_range_count",
            layout: "contiguous",
            source: 2 * lane + 2 * plane,
            temp: scratch_bytes_of(&gated_count, n),
            bindings: 0,
        },
        &mut || {
            count_of(execute_into(&gated_count, &i32_planes, &none, &mut s_b, Out::None).unwrap())
        },
    );
    let mut s_c = Scratch::for_program(&ungated, n).unwrap();
    let ungated_ns = measure(
        n,
        k,
        &want_count,
        Row {
            arm: "i32_ungated_range_count",
            layout: "contiguous",
            source: 2 * lane + 2 * plane,
            temp: scratch_bytes_of(&ungated, n),
            bindings: 0,
        },
        &mut || count_of(execute_into(&ungated, &i32_planes, &none, &mut s_c, Out::None).unwrap()),
    );

    // ----------------------------------------------------------- bit-planes
    let mut masks: Vec<&[u64]> = bp.planes.iter().map(|p| p.as_slice()).collect();
    masks.push(&region_w);
    let bit_planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let region_leaf = || BoolExpr::Leaf(Operand::Plane(8));
    let range = and(
        gt(q_lo8, plane_op).expect("lower bound below 255"),
        lt(q_hi8, plane_op).expect("upper bound above 0"),
    );
    let selection = and(range, region_leaf());
    let range_prog = fuse_program(&selection, 0, |mask| Terminal::Count { mask }).unwrap();
    println!("# fused range program: {} ops", range_prog.ops.len());
    let mut s_d = Scratch::for_program(&range_prog, n).unwrap();
    let bit_range_ns = measure(
        n,
        k,
        &want_count,
        Row {
            arm: "bitplane_range_count",
            layout: "contiguous",
            source: 9 * plane,
            temp: scratch_bytes_of(&range_prog, n),
            bindings: 0,
        },
        &mut || {
            count_of(execute_into(&range_prog, &bit_planes, &none, &mut s_d, Out::None).unwrap())
        },
    );

    let band_prog = band_program(subset, Operand::Plane(8));
    let mut s_e = Scratch::for_program(&band_prog, n).unwrap();
    let band_ns = measure(
        n,
        k,
        &want_band,
        Row {
            arm: "bitplane_band_count",
            layout: "contiguous",
            source: 4 * plane,
            temp: scratch_bytes_of(&band_prog, n),
            bindings: 0,
        },
        &mut || {
            count_of(execute_into(&band_prog, &bit_planes, &none, &mut s_e, Out::None).unwrap())
        },
    );

    let sum_progs: Vec<Program> = (0..8)
        .map(|b| {
            let e = and(selection.clone(), BoolExpr::Leaf(plane_op(b)));
            fuse_program(&e, 0, |mask| Terminal::Count { mask }).unwrap()
        })
        .collect();
    let mut sum_scratch: Vec<Scratch> = sum_progs
        .iter()
        .map(|p| Scratch::for_program(p, n).unwrap())
        .collect();
    let sum_temp: usize = sum_progs.iter().map(|p| scratch_bytes_of(p, n)).sum();
    let bit_sum_ns = measure(
        n,
        k,
        &want_sum,
        Row {
            arm: "bitplane_sum_8_counts",
            layout: "contiguous",
            source: 8 * 9 * plane,
            temp: sum_temp,
            bindings: 0,
        },
        &mut || {
            let mut sum = 0i64;
            for (b, (p, s)) in sum_progs.iter().zip(sum_scratch.iter_mut()).enumerate() {
                let c = count_of(execute_into(p, &bit_planes, &none, s, Out::None).unwrap());
                sum += (c as i64) << b;
            }
            sum
        },
    );

    let mut s_g = Scratch::for_program(&gated_sum, n).unwrap();
    let gated_sum_ns = measure(
        n,
        k,
        &want_sum,
        Row {
            arm: "i32_gated_sum",
            layout: "contiguous",
            source: 3 * lane + 2 * plane,
            temp: scratch_bytes_of(&gated_sum, n),
            bindings: 0,
        },
        &mut || sum_of(execute_into(&gated_sum, &i32_planes, &none, &mut s_g, Out::None).unwrap()),
    );

    // ------------------------------------------------------------- verdicts
    println!(
        "# ratios at {n}: bitplane_range/i32_gated {:.2}, bitplane_range/i32_ungated {:.2}, bitplane_band/i32_gated {:.2}, bitplane_sum/i32_gated_sum {:.2}",
        bit_range_ns / gated_ns,
        bit_range_ns / ungated_ns,
        band_ns / gated_ns,
        bit_sum_ns / gated_sum_ns
    );
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    println!(
        "# cost: band count {:.1} us <= gated i32 count {:.1} us: {}",
        band_ns / 1000.0,
        gated_ns / 1000.0,
        verdict(band_ns <= gated_ns)
    );
    println!(
        "# cost: bit-plane range count {:.1} us <= ungated i32 count {:.1} us: {}",
        bit_range_ns / 1000.0,
        ungated_ns / 1000.0,
        verdict(bit_range_ns <= ungated_ns)
    );
}
