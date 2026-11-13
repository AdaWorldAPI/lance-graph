//! Cost of admitting a calibrated bit-plane range count over bound planes,
//! against the unbound forms, at one slab and at one ERA5 timestep (16 slabs).
//! Every arm's answer is checked against a row-at-a-time oracle before it is
//! timed.
//!
//! Field: `ERA5_T2M_F32=<path>` names 721 x 1440 little-endian `f32` kelvin
//! (one `2m_temperature` timestep). Without it a synthetic field is used and
//! every line says so.
//!
//! The interpretation method is a registered DT64 method whose guard carries a
//! digest of (floor version, code thresholds); admission compares the caller's
//! description with the bound guard.
//!
//! Pass conditions (Cost):
//! `.claude/board/entries/2026-11-13-wx-calibrated-plane-admission.md`.

use dr256_ternlog_lab::bit_planes::BitPlanes;
use dr256_ternlog_lab::bound_planes::{range_program, BoundBitPlanes};
use dr256_ternlog_lab::{person, scratch_bytes_of, select, Table, View, World, PERSON, SLAB_ROWS};
use lance_graph_contract::class_view::WideFieldMask;
use lance_graph_contract::set_algebra::{Logic, SetExpr};
use lance_graph_contract::set_rules::{Population, Truth};
use lance_graph_dt64::dr::Scope;
use lance_graph_dt64::{Guard, MethodBinding, MethodId, SigRev, ValueDemand};
use lance_graph_mask_risc::{execute_into, Foreign, LaneRef, Out, Planes, Program, Scratch, Value};
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
const METHOD: MethodId = MethodId(1);

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

/// FNV-1a over the bytes of `s`.
fn fnv1a(s: &str) -> u32 {
    let mut h = 0x811C_9DC5u32;
    for b in s.bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// The caller's description of the interpretation: a guard over a symbol that
/// is the digest of the floor version and the two code thresholds.
fn describe(version: &str, q_lo: u8, q_hi: u8) -> Guard {
    let token = fnv1a(&format!("{version}|{q_lo}|{q_hi}"));
    Guard::new(SetExpr::Sym(token), Logic::TwoValued)
}

/// Whether the method bound in `world` carries exactly the guard the caller
/// describes.
fn admit_method(world: &World, version: &str, q_lo: u8, q_hi: u8) -> bool {
    let Ok(mref) = world.store.method_ref(PERSON, METHOD) else {
        return false;
    };
    let Ok(bound) = world.store.admit(&mref) else {
        return false;
    };
    bound.when.as_ref().map(|g| g.witness()) == Some(describe(version, q_lo, q_hi).witness())
}

struct Row {
    arm: &'static str,
    layout: &'static str,
    /// Scratch held for the query's duration.
    temp: usize,
    /// Slabs checked per call.
    bindings: usize,
}

/// Median of 21 samples of `k` calls, in nanoseconds per call, and the
/// allocations of one warm call.
fn measure(rows: usize, k: u32, want: u64, row: Row, f: &mut dyn FnMut() -> u64) -> (f64, u64) {
    assert_eq!(f(), want, "arm {}", row.arm);
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
        "{rows}\t{}\t{:.3}\t{:.4}\t{allocs}\t{bytes}\t{}\t{}\t{}",
        row.arm,
        med / 1000.0,
        med / rows as f64,
        row.temp,
        row.bindings,
        row.layout
    );
    (med, allocs)
}

struct Query {
    lo: u8,
    hi: u8,
    version: String,
    program: Program,
}

fn run(label: &str, codes: &[i32], world: &World, fields: &WideFieldMask, q: &Query) {
    let n = codes.len();
    println!("# {label}: {n} rows");
    let key = vec![0u32; n];
    let table = Table::new(
        PERSON,
        1,
        n,
        1,
        Scope { id: 7 },
        &[
            (person::ID, LaneRef::U32(&key)),
            (person::AGE, LaneRef::I32(codes)),
        ],
    );
    let (lo, hi) = (i32::from(q.lo), i32::from(q.hi));
    let want = codes.iter().filter(|&&c| lo < c && c < hi).count() as u64;
    assert!(want > 0, "the query selects nothing");

    let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
    let t = Instant::now();
    let bp = BoundBitPlanes::derive(world, &table, fields, person::AGE).expect("derives");
    let cold = t.elapsed();
    let (da, db) = (ALLOCS.load(Relaxed) - a0, BYTES.load(Relaxed) - b0);
    let slabs = bp.slabs();
    println!(
        "# cold: derive {:.2} ms, {da} allocations, {db} B allocated; resident planes {} B, widened i32 lane {} B ({:.2} x); rows selected {want}",
        cold.as_secs_f64() * 1e3,
        bp.bytes(),
        4 * n,
        bp.bytes() as f64 / (4 * n) as f64
    );

    let codes8: Vec<u8> = codes.iter().map(|&c| c as u8).collect();
    let flat = BitPlanes::of(&codes8);
    let masks: Vec<&[u64]> = flat.planes.iter().map(|p| p.as_slice()).collect();
    let flat_planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let none = Foreign {
        planes: &[],
        lanes: &[],
    };
    let mut s_flat = Scratch::for_program(&q.program, n).expect("scratch");
    let mut s_slab = Scratch::for_program(&q.program, SLAB_ROWS).expect("scratch");

    let k = (4_000_000 / n as u32).clamp(1, 32);
    println!("rows\tarm\tmedian_us\tns_per_row\tallocs\talloc_bytes\ttemp_bytes\tbindings\tlayout");

    let (a_ns, _) = measure(
        n,
        k,
        want,
        Row {
            arm: "unbound_contiguous_range_count",
            layout: "contiguous",
            temp: scratch_bytes_of(&q.program, n),
            bindings: 0,
        },
        &mut || match execute_into(&q.program, &flat_planes, &none, &mut s_flat, Out::None)
            .expect("executes")
        {
            Value::Count(c) => c as u64,
            other => panic!("not a count: {other:?}"),
        },
    );
    let (b_ns, _) = measure(
        n,
        k,
        want,
        Row {
            arm: "unbound_slab_range_count",
            layout: "per-slab",
            temp: scratch_bytes_of(&q.program, SLAB_ROWS),
            bindings: 0,
        },
        &mut || bp.count_unchecked(&q.program, &mut s_slab).expect("counts"),
    );
    assert!(bp.check(world, &table).is_ok(), "the planes check");
    let (c_ns, _) = measure(
        n,
        k * 64,
        0,
        Row {
            arm: "checks_only",
            layout: "per-slab",
            temp: 0,
            bindings: slabs,
        },
        &mut || {
            bp.check(world, &table).expect("checks");
            0
        },
    );
    assert!(
        admit_method(world, &q.version, q.lo, q.hi),
        "the method admits"
    );
    let (d_ns, _) = measure(
        n,
        k * 64,
        0,
        Row {
            arm: "method_admission_only",
            layout: "none",
            temp: 0,
            bindings: 0,
        },
        &mut || {
            assert!(admit_method(world, &q.version, q.lo, q.hi));
            0
        },
    );
    let (e_ns, e_allocs) = measure(
        n,
        k,
        want,
        Row {
            arm: "admitted_range_count",
            layout: "per-slab",
            temp: scratch_bytes_of(&q.program, SLAB_ROWS),
            bindings: slabs,
        },
        &mut || {
            assert!(admit_method(world, &q.version, q.lo, q.hi));
            bp.count(world, &table, &q.program, &mut s_slab)
                .expect("counts")
        },
    );

    // The same admission with the method reference and the described
    // witness resolved once, outside the query.
    let mref = world.store.method_ref(PERSON, METHOD).expect("method");
    let described = describe(&q.version, q.lo, q.hi).witness();
    let prepared = || world.store.admit(&mref).is_ok() && mref.guard.as_ref() == Some(&described);
    let (d2_ns, _) = measure(
        n,
        k * 64,
        0,
        Row {
            arm: "method_admission_prepared",
            layout: "none",
            temp: 0,
            bindings: 0,
        },
        &mut || {
            assert!(prepared());
            0
        },
    );
    let (e2_ns, e2_allocs) = measure(
        n,
        k,
        want,
        Row {
            arm: "admitted_range_count_prepared",
            layout: "per-slab",
            temp: scratch_bytes_of(&q.program, SLAB_ROWS),
            bindings: slabs,
        },
        &mut || {
            assert!(prepared());
            bp.count(world, &table, &q.program, &mut s_slab)
                .expect("counts")
        },
    );
    println!(
        "# prepared: method admission {d2_ns:.0} ns; admitted / unbound contiguous {:.2}; warm allocations {e2_allocs}: {}",
        e2_ns / a_ns,
        if e2_allocs == 0 { "PASS" } else { "FAIL" }
    );

    println!(
        "# e - b: {:.0} ns per call, {:.0} ns per slab ({slabs} slabs)",
        e_ns - b_ns,
        (e_ns - b_ns) / slabs as f64
    );
    println!(
        "# checks alone: {:.0} ns per slab; method admission alone: {d_ns:.0} ns",
        c_ns / slabs as f64
    );
    println!(
        "# admitted / unbound contiguous (e / a): {:.2}",
        e_ns / a_ns
    );
    println!(
        "# warm allocations of admitted_range_count: {e_allocs}: {}",
        if e_allocs == 0 { "PASS" } else { "FAIL" }
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
    let sample: Vec<f64> = kelvin.iter().map(|&k| f64::from(k)).collect();
    let floor = calibrate(&sample).expect("a finite field calibrates");
    let (f_lo, f_hi) = floor.bounds();
    let version = floor.floor_version().to_string();
    println!("# floor v{version}: [{f_lo:.3}, {f_hi:.3}] K");
    let codes: Vec<i32> = sample
        .iter()
        .map(|&k| i32::from(floor.quantize(k)))
        .collect();
    let (q_lo, q_hi) = (floor.quantize(LO_K), floor.quantize(HI_K));
    println!("# codes ({q_lo}, {q_hi}) exclusive");
    let program = range_program(q_lo, q_hi).expect("the range is satisfiable");

    let fields = select(&View::default(), PERSON, &["age"]).expect("age is a field");
    let world = World::with_person_methods(vec![MethodBinding {
        id: METHOD,
        sig_rev: SigRev(1),
        name: "t_between".into(),
        on: fields.clone(),
        when: Some(describe(&version, q_lo, q_hi)),
        reads: fields.clone(),
        effects: Vec::new(),
        population: Population::Distinct,
        truth: Truth::TwoValued,
        demand: ValueDemand::Values,
    }]);
    let q = Query {
        lo: q_lo,
        hi: q_hi,
        version,
        program,
    };

    let (lo, hi) = (i32::from(q_lo), i32::from(q_hi));
    let slab = codes
        .chunks(SLAB_ROWS)
        .max_by_key(|c| c.iter().filter(|&&v| lo < v && v < hi).count())
        .expect("the field has rows");
    run(
        "one slab (the one with most selected rows)",
        slab,
        &world,
        &fields,
        &q,
    );
    run("one timestep", &codes, &world, &fields, &q);
}
