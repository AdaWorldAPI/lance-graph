//! `WHERE region AND valid AND lo < t < hi` with count, sum, min and max over
//! one ERA5 timestep, as DR256 slabs folded by one ternlog and a masked
//! terminal. Every arm's answer is checked against a row-at-a-time oracle
//! before it is timed.
//!
//! Field: `ERA5_T2M_F32=<path>` names 721 x 1440 little-endian `f32` kelvin
//! (one `2m_temperature` timestep of ARCO-ERA5). Without it a synthetic field
//! is used and every line says so.
//!
//! Pass conditions: `.claude/board/entries/2026-11-13-weather-dr256-ternlog.md`.

use dr256_ternlog_lab::*;
use lance_graph_dt64::dr::Scope;
use lance_graph_mask_risc::{
    execute_into, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Scratch, Terminal,
    Value,
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
const VALUE: u8 = person::AGE;
const READER: Scope = Scope { id: 7 };
/// 0.0 C and 10.0 C in tenths of a degree Celsius, and in kelvin.
const LO_TENTHS: i32 = 0;
const HI_TENTHS: i32 = 100;
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct Stats {
    count: u64,
    sum: i64,
    min: Option<i32>,
    max: Option<i32>,
}

/// Row at a time over the lane, region recomputed per row.
fn oracle(first: usize, value: &[i32], valid: &[bool], lo: i32, hi: i32) -> Stats {
    let mut sum = 0i128;
    let (mut count, mut min, mut max) = (0u64, None::<i32>, None::<i32>);
    for (i, &v) in value.iter().enumerate() {
        if in_box((first + i) % GRID) && valid[i] && lo < v && v < hi {
            count += 1;
            sum += i128::from(v);
            min = Some(min.map_or(v, |m| m.min(v)));
            max = Some(max.map_or(v, |m| m.max(v)));
        }
    }
    Stats {
        count,
        sum: sum as i64,
        min,
        max,
    }
}

/// The scalar arm that is timed: one pass, region and validity prebuilt.
fn scalar(value: &[i32], region: &[bool], valid: &[bool], lo: i32, hi: i32) -> Stats {
    let (mut count, mut sum, mut min, mut max) = (0u64, 0i64, i32::MAX, i32::MIN);
    for i in 0..value.len() {
        let v = value[i];
        if region[i] & valid[i] & (lo < v) & (v < hi) {
            count += 1;
            sum += i64::from(v);
            min = min.min(v);
            max = max.max(v);
        }
    }
    Stats {
        count,
        sum,
        min: (count > 0).then_some(min),
        max: (count > 0).then_some(max),
    }
}

fn plane_words(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; n.div_ceil(64)];
    for i in (0..n).filter(|&i| set(i)) {
        w[i / 64] |= 1 << (i % 64);
    }
    w
}

const AGGS: [Agg; 4] = [
    Agg::Count,
    Agg::Sum(VALUE),
    Agg::Min(VALUE),
    Agg::Max(VALUE),
];

fn collect(answers: [Answer; 4]) -> Stats {
    let [Answer::Count(count), Answer::Sum(sum), Answer::Min(min), Answer::Max(max)] = answers
    else {
        panic!("answers out of order");
    };
    Stats {
        count,
        sum,
        min,
        max,
    }
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

/// Median of 21 samples of `k` calls; also the allocations of one warm call.
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

/// All arms over `tenths`, whose row 0 is grid cell `first`.
fn run(label: &str, first: usize, tenths: &[i32], valid: &[bool]) {
    let n = tenths.len();
    let key: Vec<u32> = vec![0; n];
    let region: Vec<bool> = (0..n).map(|i| in_box((first + i) % GRID)).collect();
    let want = oracle(first, tenths, valid, LO_TENTHS, HI_TENTHS);
    assert!(want.count > 0, "the query selects nothing at n = {n}");
    let k = (4_000_000 / n as u32).clamp(1, 32);
    println!(
        "# {label}: rows={n} selected={} sum={} min={:?} max={:?}",
        want.count, want.sum, want.min, want.max
    );

    let world = World::new();
    let table = Table::new(
        PERSON,
        1,
        n,
        1,
        READER,
        &[
            (person::ID, LaneRef::U32(&key)),
            (VALUE, LaneRef::I32(tenths)),
        ],
    );
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let slabs = table.slabs();
    let lane = n * 4;
    let plane = n.div_ceil(64) * 8;

    // -------------------------------------------------------- preparation
    let t = Instant::now();
    let prepared = prepare_nodes(&world, &table, &fields, |i| region[i], |i| valid[i]).unwrap();
    let prep_planes = t.elapsed();
    // The predicate planes as residents: plane 1 = valid & t > lo, plane 2 = t < hi.
    let t = Instant::now();
    let mut resident = prepare_nodes(
        &world,
        &table,
        &fields,
        |i| region[i],
        |i| valid[i] && tenths[i] > LO_TENTHS,
    )
    .unwrap();
    for s in 0..slabs {
        let base = s * SLAB_ROWS;
        let p = Plane::from_fn(table.space(s), table.slab_rows(s), |i| {
            tenths[base + i] < HI_TENTHS
        });
        resident.set_plane(s, 2, p);
    }
    let prep_resident = t.elapsed();
    println!(
        "# cold: bind + region and valid planes {:.2} ms; bind + three resident planes {:.2} ms; planes held {} B",
        prep_planes.as_secs_f64() * 1e3,
        prep_resident.as_secs_f64() * 1e3,
        resident.plane_bytes()
    );

    // ------------------------------------------------------------- scalar
    let base = measure(
        n,
        k,
        &want,
        Row {
            arm: "scalar_one_pass_4_stats",
            layout: "contiguous",
            source: lane + 2 * n,
            temp: 0,
            bindings: 0,
        },
        &mut || scalar(tenths, &region, valid, LO_TENTHS, HI_TENTHS),
    );

    // -------------------------------------- contiguous mask-RISC, no slabs
    let region_w = plane_words(n, |i| region[i]);
    let valid_w = plane_words(n, |i| valid[i]);
    // Bounds gated under the region (the form `Fold::between` compiles),
    // and the same query with both bounds compared over every row.
    let gated = || {
        vec![
            MaskOp::Pred {
                pred: Pred::GtI32 {
                    lane: 0,
                    t: LO_TENTHS,
                },
                under: Some(Operand::Plane(0)),
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LtI32 {
                    lane: 0,
                    t: HI_TENTHS,
                },
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
    };
    let ungated = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::GtI32 {
                    lane: 0,
                    t: LO_TENTHS,
                },
                under: None,
                dst: 0,
            },
            MaskOp::Pred {
                pred: Pred::LtI32 {
                    lane: 0,
                    t: HI_TENTHS,
                },
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
    let ops = gated;
    let m = Operand::Scratch(2);
    let programs = [
        Program::new(ops(), Terminal::Count { mask: m }),
        Program::new(ops(), Terminal::MaskedSumI32 { mask: m, lane: 0 }),
        Program::new(ops(), Terminal::MaskedMinI32 { mask: m, lane: 0 }),
        Program::new(ops(), Terminal::MaskedMaxI32 { mask: m, lane: 0 }),
    ];
    let masks: [&[u64]; 2] = [&region_w, &valid_w];
    let lanes = [LaneRef::I32(tenths)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let none = Foreign {
        planes: &[],
        lanes: &[],
    };
    let mut scratch2 = Scratch::for_program(&programs[0], n).unwrap();
    let mut flat_stats = || {
        let mut go =
            |p: &Program| execute_into(p, &planes, &none, &mut scratch2, Out::None).unwrap();
        let (Value::Count(c), Value::SumI64(s), Value::OptI32(lo), Value::OptI32(hi)) = (
            go(&programs[0]),
            go(&programs[1]),
            go(&programs[2]),
            go(&programs[3]),
        ) else {
            panic!()
        };
        Stats {
            count: c as u64,
            sum: s,
            min: lo,
            max: hi,
        }
    };
    let flat_src = 4 * (2 * lane + 2 * plane) + 3 * lane;
    let flat4 = measure(
        n,
        k,
        &want,
        Row {
            arm: "maskrisc_contiguous_4_programs",
            layout: "contiguous",
            source: flat_src,
            temp: 0,
            bindings: 0,
        },
        &mut flat_stats,
    );
    let mut scratch3 = Scratch::for_program(&programs[0], n).unwrap();
    measure(
        n,
        k,
        &want.count,
        Row {
            arm: "maskrisc_contiguous_count",
            layout: "contiguous",
            source: 2 * lane + 2 * plane,
            temp: 0,
            bindings: 0,
        },
        &mut || match execute_into(&programs[0], &planes, &none, &mut scratch3, Out::None).unwrap()
        {
            Value::Count(c) => c as u64,
            _ => unreachable!(),
        },
    );

    let mut scratch4 = Scratch::for_program(&ungated, n).unwrap();
    measure(
        n,
        k,
        &want.count,
        Row {
            arm: "maskrisc_contiguous_count_ungated",
            layout: "contiguous",
            source: 2 * lane + 2 * plane,
            temp: 0,
            bindings: 0,
        },
        &mut || match execute_into(&ungated, &planes, &none, &mut scratch4, Out::None).unwrap() {
            Value::Count(c) => c as u64,
            _ => unreachable!(),
        },
    );

    // ---------------------------------------- slabs, predicate from the lane
    let mut between: Vec<Fold> = AGGS
        .iter()
        .map(|&a| Fold::between(&prepared, VALUE, LO_TENTHS, HI_TENTHS, a).unwrap())
        .collect();
    let temp = between[0].scratch_bytes();
    let slab4 = measure(
        n,
        k,
        &want,
        Row {
            arm: "slab_between_4_folds",
            layout: "slab",
            source: flat_src,
            temp,
            bindings: 4 * slabs,
        },
        &mut || {
            collect([0, 1, 2, 3].map(|a| between[a].answer(&world, &table, &prepared).unwrap()))
        },
    );
    let mut count_fold = Fold::between(&prepared, VALUE, LO_TENTHS, HI_TENTHS, Agg::Count).unwrap();
    measure(
        n,
        k,
        &want.count,
        Row {
            arm: "slab_between_count",
            layout: "slab",
            source: 2 * lane + 2 * plane,
            temp,
            bindings: slabs,
        },
        &mut || count_fold.count(&world, &table, &prepared).unwrap(),
    );

    // ------------------------------------------- slabs, three resident planes
    let mut and3: Vec<Fold> = AGGS
        .iter()
        .map(|&a| Fold::and3(&resident, a).unwrap())
        .collect();
    let temp3 = and3[0].scratch_bytes();
    let res4 = measure(
        n,
        k,
        &want,
        Row {
            arm: "slab_and3_resident_4_folds",
            layout: "slab",
            source: 4 * 3 * plane + 3 * lane,
            temp: temp3,
            bindings: 4 * slabs,
        },
        &mut || collect([0, 1, 2, 3].map(|a| and3[a].answer(&world, &table, &resident).unwrap())),
    );
    let mut and3_count = Fold::count_and3().unwrap();
    let res1 = measure(
        n,
        k * 8,
        &want.count,
        Row {
            arm: "slab_and3_resident_count",
            layout: "slab",
            source: 3 * plane,
            temp: temp3,
            bindings: slabs,
        },
        &mut || and3_count.count(&world, &table, &resident).unwrap(),
    );

    // ------------------------------ contiguous three resident planes, count
    let gt_w = plane_words(n, |i| valid[i] && tenths[i] > LO_TENTHS);
    let lt_w = plane_words(n, |i| tenths[i] < HI_TENTHS);
    let three: [&[u64]; 3] = [&region_w, &gt_w, &lt_w];
    let p3 = Program::new(
        vec![MaskOp::Ternlog {
            imm: AND3,
            a: Operand::Plane(0),
            b: Operand::Plane(1),
            c: Operand::Plane(2),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let planes3 = Planes {
        n_rows: n,
        masks: &three,
        lanes: &[],
    };
    let mut s3 = Scratch::for_program(&p3, n).unwrap();
    let flat1 = measure(
        n,
        k * 8,
        &want.count,
        Row {
            arm: "maskrisc_contiguous_and3_count",
            layout: "contiguous",
            source: 3 * plane,
            temp: 0,
            bindings: 0,
        },
        &mut || match execute_into(&p3, &planes3, &none, &mut s3, Out::None).unwrap() {
            Value::Count(c) => c as u64,
            _ => unreachable!(),
        },
    );

    // ----------------------------- borrowed 256-position selections, count
    measure(
        n,
        k * 8,
        &want.count,
        Row {
            arm: "selections_256_and3_count",
            layout: "borrowed",
            source: 3 * plane,
            temp: 0,
            bindings: 0,
        },
        &mut || {
            let mut c = 0u64;
            for s in 0..slabs {
                let (a, b, d) = (
                    resident.plane(s, 0),
                    resident.plane(s, 1),
                    resident.plane(s, 2),
                );
                for ((x, y), z) in a.selections().zip(b.selections()).zip(d.selections()) {
                    for ((p, q), r) in x.words().iter().zip(y.words()).zip(z.words()) {
                        c += u64::from((p & q & r).count_ones());
                    }
                }
            }
            c
        },
    );

    println!(
        "# ratios at {n}: scalar/flat4 {:.2}, scalar/slab4 {:.2}, scalar/resident4 {:.2}, slab4/flat4 {:.2}, slab_and3_count/flat_and3_count {:.2}",
        base / flat4, base / slab4, base / res4, slab4 / flat4, res1 / flat1
    );
}

fn main() {
    let (kelvin, source) = field();
    println!(
        "# field={source} avx512f={} avx2={} popcnt_avx512={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512vpopcntdq")
    );
    let valid: Vec<bool> = kelvin.iter().map(|k| k.is_finite()).collect();
    let tenths: Vec<i32> = kelvin
        .iter()
        .map(|&k| ((f64::from(k) - 273.15) * 10.0).round() as i32)
        .collect();

    // --------------------------------------------- full-precision oracle
    let (mut n, mut sum, mut min, mut max) = (0u64, 0f64, f64::MAX, f64::MIN);
    for (i, &k) in kelvin.iter().enumerate() {
        let k = f64::from(k);
        if in_box(i) && valid[i] && LO_K < k && k < HI_K {
            n += 1;
            sum += k;
            min = min.min(k);
            max = max.max(k);
        }
    }
    let mean = sum / n as f64;
    println!(
        "# f32 oracle: count={n} mean={mean:.4} K min={min:.4} max={max:.4} invalid={}",
        valid.iter().filter(|v| !**v).count()
    );

    // -------------------------------------------- exact integer statistics
    let exact = oracle(0, &tenths, &valid, LO_TENTHS, HI_TENTHS);
    let mean_tenths = mean_round_half_up(exact.sum, exact.count).unwrap();
    println!(
        "# tenths: count={} (f32 oracle {n}) sum={} min={:?} max={:?} mean={} tenths C = {:.2} K (f32 oracle {mean:.4} K)",
        exact.count, exact.sum, exact.min, exact.max, mean_tenths,
        mean_tenths as f64 / 10.0 + 273.15
    );

    // ------------------------------------------------- quantized u8 codes
    let sample: Vec<f64> = kelvin.iter().map(|&k| f64::from(k)).collect();
    let floor = calibrate(&sample).expect("a finite field calibrates");
    let (f_lo, f_hi) = floor.bounds();
    let bucket = (f_hi - f_lo) / 256.0;
    let codes: Vec<i32> = sample
        .iter()
        .map(|&k| i32::from(floor.quantize(k)))
        .collect();
    let (q_lo, q_hi) = (
        i32::from(floor.quantize(LO_K)),
        i32::from(floor.quantize(HI_K)),
    );
    let world = World::new();
    let key = vec![0u32; GRID];
    let table = Table::new(
        PERSON,
        1,
        GRID,
        1,
        READER,
        &[
            (person::ID, LaneRef::U32(&key)),
            (VALUE, LaneRef::I32(&codes)),
        ],
    );
    let fields = select(&world.view, PERSON, &["age"]).unwrap();
    let prepared = prepare_nodes(&world, &table, &fields, in_box, |i| valid[i]).unwrap();
    let q = collect(AGGS.map(|a| {
        Fold::between(&prepared, VALUE, q_lo, q_hi, a)
            .unwrap()
            .answer(&world, &table, &prepared)
            .unwrap()
    }));
    assert_eq!(
        q,
        oracle(0, &codes, &valid, q_lo, q_hi),
        "code space is exact"
    );
    let decode = |c: f64| f_lo + (c + 0.5) / 256.0 * (f_hi - f_lo);
    let boundary = (0..GRID)
        .filter(|&i| in_box(i) && valid[i] && (codes[i] == q_lo || codes[i] == q_hi))
        .count() as i64;
    let saturated = sample.iter().filter(|&&k| k <= f_lo || k >= f_hi).count();
    let (q_mean, q_min, q_max) = (
        decode(q.sum as f64 / q.count as f64),
        decode(f64::from(q.min.unwrap())),
        decode(f64::from(q.max.unwrap())),
    );
    let verdict = |ok: bool| if ok { "PASS" } else { "FAIL" };
    println!(
        "# u8 floor v{}: [{f_lo:.3}, {f_hi:.3}] K, bucket {bucket:.4} K, codes ({q_lo}, {q_hi}) exclusive, {saturated} of {GRID} cells saturate",
        floor.floor_version()
    );
    let dc = q.count as i64 - n as i64;
    println!(
        "# u8 count {} vs {n}: diff {dc}, bound {boundary} rows in the two bound buckets: {}",
        q.count,
        verdict(dc.abs() <= boundary)
    );
    println!(
        "# u8 mean {q_mean:.4} vs {mean:.4}: err {:+.4} K = {:+.3} buckets: {}",
        q_mean - mean,
        (q_mean - mean) / bucket,
        verdict((q_mean - mean).abs() <= bucket)
    );
    println!(
        "# u8 min {q_min:.4} vs {min:.4}: err {:+.4} K: {}",
        q_min - min,
        verdict((q_min - min).abs() <= bucket)
    );
    println!(
        "# u8 max {q_max:.4} vs {max:.4}: err {:+.4} K: {}",
        q_max - max,
        verdict((q_max - max).abs() <= bucket)
    );
    // Global extrema go through the saturating ends of the floor.
    let (g_min, g_max) = sample
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &k| (a.min(k), b.max(k)));
    println!(
        "# u8 global extrema: min {:.3} vs {g_min:.3} (err {:+.3} K), max {:.3} vs {g_max:.3} (err {:+.3} K), bucket {bucket:.3} K",
        decode(0.0), decode(0.0) - g_min, decode(255.0), decode(255.0) - g_max
    );

    // ------------------------------------------------------------ timings
    println!("rows\tarm\tmedian_us\tns_per_row\tallocs\talloc_bytes\tsource_bytes_max\ttemp_bytes\tbindings\tlayout");
    // One slab: the slab holding the most selected rows.
    let best = (0..GRID / SLAB_ROWS)
        .max_by_key(|s| {
            (s * SLAB_ROWS..(s + 1) * SLAB_ROWS)
                .filter(|&i| in_box(i))
                .count()
        })
        .unwrap();
    let at = best * SLAB_ROWS;
    run(
        "one slab",
        at,
        &tenths[at..at + SLAB_ROWS],
        &valid[at..at + SLAB_ROWS],
    );
    run("one timestep, 16 slabs", 0, &tenths, &valid);
    let copies = 16;
    let tiled: Vec<i32> = tenths.iter().copied().cycle().take(copies * GRID).collect();
    let tiled_valid: Vec<bool> = valid.iter().copied().cycle().take(copies * GRID).collect();
    run(
        "16 tiled copies of the timestep, 254 slabs",
        0,
        &tiled,
        &tiled_valid,
    );
}
