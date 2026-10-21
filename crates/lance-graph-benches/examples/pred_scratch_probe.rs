//! D-CMR-2b, phase 1 (lab only): what does a residual comparison's scratch
//! actually cost, and does a scratch-free fold win?
//!
//! `fused_ternlog` declines `Pred`, so a program with a comparison runs tiled:
//! each op writes its destination slot for one tile (`TILE_WORDS` words),
//! later ops re-read it, and the terminal counts the last slot. Scratch is
//! caller-owned and tile-sized (`slots × TILE_WORDS` words), never
//! population-sized, so the question is not heap memory. It is whether the
//! per-op tile write-back and re-read, and the separate `And` + count passes,
//! cost anything measurable against a fold that compares into chunks and
//! counts them in one fused `mask_ternlog_popcount` call.
//!
//! Arms, on the same data and the same programs:
//!
//! - **A** `execute_extent` with a `Scratch` built once outside the timing.
//! - **A+alloc** the same with `Scratch::for_program` once per call, inside the
//!   timing (what `fold::set_lower`'s `SetPlan::count` does: one scratch per
//!   call, then one `execute_extent` per run).
//! - **B** a lab fold: per chunk of at most `TILE_WORDS` words, each comparison
//!   into a fresh stack chunk via the same `ndarray::simd` compare primitive
//!   the executor uses, then ONE `mask_ternlog_popcount` over the chunks (and
//!   a resident plane, shifted into a window when a run starts mid-word).
//! - **B′** the same fold over chunks reused across calls, which separates the
//!   value of the fused count from the cost of initialising the chunks.
//! - **Ac** arm A with the program compiled once outside the timing
//!   (`Program::compile` + `execute_compiled`), which separates the per-call
//!   recognition cost of `execute_extent` from the tile work itself.
//!
//! Shapes (`a = lane0 > ta`, `b = lane1 < tb`, `p = plane 0`):
//!
//! - `S1` Count(a)
//! - `S2` Count(a ∧ b)
//! - `S2u` Count(a ∧ b) with `b` gated `under` `a` (the executor's own fused
//!   compare-under form; arm B uses `*_to_mask_under` likewise)
//! - `S3` Count(a ∧ b ∧ p)
//!
//! Two workloads:
//!
//! - **whole**: one extent `0..n` (n = 16k, 1M, 4M).
//! - **runs**: `set_lower`'s shape. Over n = 1M, K non-overlapping runs of
//!   length L (200, 3,000, 40,000 rows) at unaligned starts, together about a
//!   quarter of the rows. Arm A calls `execute_extent` once per run; arm B folds
//!   each run.
//!
//! Every configuration is checked against a bit-serial row oracle and arm A
//! before anything is timed. Timing is interleaved per rep in a rotating
//! order, the median of 101 per-rep ratios is reported, and a null arm (A timed
//! twice) gives the method's own noise.
//!
//! `cargo run --release -p lance-graph-benches --example pred_scratch_probe`

use std::cell::RefCell;
use std::ops::Range;
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, execute_extent, Scratch, TILE_WORDS};
use lance_graph_mask_risc::{
    words_for, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Terminal, Value,
};
use ndarray::simd::{gt_i32_to_mask, lt_i32_to_mask, lt_i32_to_mask_under, mask_ternlog_popcount};

/// `a & b & c` in the VPTERNLOG input convention.
const AND3: i32 = 0x80;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn time<T>(f: &mut impl FnMut() -> T) -> f64 {
    let t = Instant::now();
    std::hint::black_box(f());
    t.elapsed().as_nanos() as f64
}

fn med(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Interleaved, paired timing: every rep runs A, A again (the null arm),
/// A+alloc, B, B′ and Ac back to back, in an order that rotates per rep, so
/// drift and frequency changes hit all arms alike. Returns the median ns of A
/// and the medians of the per-rep ratios A′/A, A+alloc/A, B/A, B′/A and Ac/A.
fn paired(
    reps: usize,
    mut a: impl FnMut() -> usize,
    mut aa: impl FnMut() -> usize,
    mut b: impl FnMut() -> usize,
    mut b2: impl FnMut() -> usize,
    mut ac: impl FnMut() -> usize,
) -> (f64, [f64; 5]) {
    let mut ta = vec![];
    let mut ratios: [Vec<f64>; 5] = Default::default();
    for r in 0..reps {
        let mut t = [0f64; 6];
        for k in 0..6 {
            let i = (k + r) % 6;
            t[i] = match i {
                0 | 1 => time(&mut a),
                2 => time(&mut aa),
                3 => time(&mut b),
                4 => time(&mut b2),
                _ => time(&mut ac),
            };
        }
        ta.push(t[0]);
        for j in 0..5 {
            ratios[j].push(t[j + 1] / t[0]);
        }
    }
    let [n, al, b, b2, ac] = ratios;
    (med(ta), [med(n), med(al), med(b), med(b2), med(ac)])
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    S1,
    S2,
    S2u,
    S3,
}

const SHAPES: [Shape; 4] = [Shape::S1, Shape::S2, Shape::S2u, Shape::S3];

fn program(shape: Shape, ta: i32, tb: i32) -> Program {
    let a = Pred::GtI32 { lane: 0, t: ta };
    let b = Pred::LtI32 { lane: 1, t: tb };
    let pred = |pred, under, dst| MaskOp::Pred { pred, under, dst };
    let and = |a, b, dst| MaskOp::And { a, b, dst };
    let (ops, last) = match shape {
        Shape::S1 => (vec![pred(a, None, 0)], 0),
        Shape::S2 => (
            vec![
                pred(a, None, 0),
                pred(b, None, 1),
                and(Operand::Scratch(0), Operand::Scratch(1), 2),
            ],
            2,
        ),
        Shape::S2u => (
            vec![pred(a, None, 0), pred(b, Some(Operand::Scratch(0)), 1)],
            1,
        ),
        Shape::S3 => (
            vec![
                pred(a, None, 0),
                pred(b, None, 1),
                and(Operand::Scratch(0), Operand::Scratch(1), 2),
                and(Operand::Scratch(2), Operand::Plane(0), 3),
            ],
            3,
        ),
    };
    Program::new(
        ops,
        Terminal::Count {
            mask: Operand::Scratch(last),
        },
    )
}

/// The bit-serial reference over the rows of `runs`.
fn oracle(
    shape: Shape,
    l0: &[i32],
    l1: &[i32],
    plane: &[u64],
    ta: i32,
    tb: i32,
    runs: &[Range<usize>],
) -> usize {
    runs.iter()
        .flat_map(|r| r.clone())
        .filter(|&r| {
            let a = l0[r] > ta;
            let b = l1[r] < tb;
            let p = plane[r / 64] >> (r % 64) & 1 == 1;
            match shape {
                Shape::S1 => a,
                Shape::S2 | Shape::S2u => a && b,
                Shape::S3 => a && b && p,
            }
        })
        .count()
}

/// Chunks a fold works in: two comparison masks and a plane window.
struct Chunks {
    ca: [u64; TILE_WORDS],
    cb: [u64; TILE_WORDS],
    cp: [u64; TILE_WORDS],
}

impl Chunks {
    fn new() -> Self {
        Chunks {
            ca: [0; TILE_WORDS],
            cb: [0; TILE_WORDS],
            cp: [0; TILE_WORDS],
        }
    }
}

/// `plane`'s bits `r0..r0 + 64·out.len()`, shifted to start at bit 0.
fn window(plane: &[u64], r0: usize, out: &mut [u64]) {
    let (w0, s) = (r0 / 64, r0 % 64);
    for (i, o) in out.iter_mut().enumerate() {
        let lo = plane.get(w0 + i).copied().unwrap_or(0) >> s;
        let hi = if s == 0 {
            0
        } else {
            plane.get(w0 + i + 1).copied().unwrap_or(0) << (64 - s)
        };
        *o = lo | hi;
    }
}

/// Arm B / B′: the fold over the rows of `runs`, in chunks of at most
/// `TILE_WORDS` words. A comparison mask has zero bits past its chunk's rows,
/// so the plane window needs no tail masking under `AND3`.
#[allow(clippy::too_many_arguments)]
fn fold(
    shape: Shape,
    l0: &[i32],
    l1: &[i32],
    plane: &[u64],
    ta: i32,
    tb: i32,
    runs: &[Range<usize>],
    c: &mut Chunks,
) -> usize {
    let mut total = 0u64;
    for run in runs {
        let mut r0 = run.start;
        while r0 < run.end {
            let r1 = (r0 + TILE_WORDS * 64).min(run.end);
            let tw = words_for(r1 - r0);
            let (ca, cb) = (&mut c.ca[..tw], &mut c.cb[..tw]);
            gt_i32_to_mask(&l0[r0..r1], ta, ca);
            total += match shape {
                Shape::S1 => mask_ternlog_popcount::<AND3>(ca, ca, ca),
                Shape::S2 => {
                    lt_i32_to_mask(&l1[r0..r1], tb, cb);
                    mask_ternlog_popcount::<AND3>(ca, cb, ca)
                }
                Shape::S2u => {
                    lt_i32_to_mask_under(&l1[r0..r1], tb, ca, cb);
                    mask_ternlog_popcount::<AND3>(cb, cb, cb)
                }
                Shape::S3 => {
                    lt_i32_to_mask(&l1[r0..r1], tb, cb);
                    if r0.is_multiple_of(64) {
                        mask_ternlog_popcount::<AND3>(ca, cb, &plane[r0 / 64..r0 / 64 + tw])
                    } else {
                        let cp = &mut c.cp[..tw];
                        window(plane, r0, cp);
                        mask_ternlog_popcount::<AND3>(ca, cb, cp)
                    }
                }
            };
            r0 = r1;
        }
    }
    total as usize
}

/// Op passes over one chunk: (arm A, arm B).
fn passes(shape: Shape) -> (usize, usize) {
    match shape {
        Shape::S1 => (2, 2),
        Shape::S2 => (4, 3),
        Shape::S2u => (3, 3),
        Shape::S3 => (5, 3),
    }
}

/// K non-overlapping runs of length `len` at unaligned starts over `0..n`,
/// together about a quarter of the rows.
fn runs_for(n: usize, len: usize, seed: &mut u64) -> Vec<Range<usize>> {
    let k = (n / 4 / len).max(1);
    let slot = n / k;
    (0..k)
        .map(|i| {
            let start = (i * slot + lcg(seed) as usize % (slot - len)) | 1;
            start..(start + len).min(n)
        })
        .collect()
}

struct Data {
    n: usize,
    l0: Vec<i32>,
    l1: Vec<i32>,
    plane: Vec<u64>,
}

fn data(n: usize, seed: &mut u64) -> Data {
    let l0 = (0..n).map(|_| (lcg(seed) % 1000) as i32).collect();
    let l1 = (0..n).map(|_| (lcg(seed) % 1000) as i32).collect();
    let mut plane: Vec<u64> = (0..words_for(n))
        .map(|_| lcg(seed) ^ lcg(seed) << 21)
        .collect();
    if !n.is_multiple_of(64) {
        let last = plane.len() - 1;
        plane[last] &= (1u64 << (n % 64)) - 1;
    }
    Data { n, l0, l1, plane }
}

/// Check every arm against the oracle, then time them; print one row.
fn measure(label: &str, d: &Data, runs: &[Range<usize>], shape: Shape, ta: i32, tb: i32, sel: u32) {
    let masks: [&[u64]; 1] = [&d.plane];
    let lanes = [LaneRef::I32(&d.l0), LaneRef::I32(&d.l1)];
    let planes = Planes {
        n_rows: d.n,
        masks: &masks,
        lanes: &lanes,
    };
    let p = program(shape, ta, tb);
    let want = oracle(shape, &d.l0, &d.l1, &d.plane, ta, tb, runs);
    let run_a = |s: &mut Scratch<'_>| -> usize {
        runs.iter()
            .map(|r| {
                match execute_extent(&p, &planes, &Foreign::NONE, s, Out::None, r.clone())
                    .expect("execute")
                {
                    Value::Count(c) => c,
                    v => panic!("unexpected {v:?}"),
                }
            })
            .sum()
    };
    let compiled = p.compile();
    let run_ac = |s: &mut Scratch<'_>| -> usize {
        runs.iter()
            .map(|r| {
                match execute_compiled(&compiled, &planes, &Foreign::NONE, s, Out::None, r.clone())
                    .expect("execute")
                {
                    Value::Count(c) => c,
                    v => panic!("unexpected {v:?}"),
                }
            })
            .sum()
    };
    let scratch = RefCell::new(Scratch::for_program(&p, d.n).expect("scratch"));
    let chunks = RefCell::new(Chunks::new());
    let fold_fresh = || {
        let mut c = Chunks::new();
        fold(shape, &d.l0, &d.l1, &d.plane, ta, tb, runs, &mut c)
    };
    let fold_reused = || {
        fold(
            shape,
            &d.l0,
            &d.l1,
            &d.plane,
            ta,
            tb,
            runs,
            &mut chunks.borrow_mut(),
        )
    };
    assert_eq!(
        run_a(&mut scratch.borrow_mut()),
        want,
        "A {label} {shape:?} sel={sel}"
    );
    assert_eq!(
        run_ac(&mut scratch.borrow_mut()),
        want,
        "Ac {label} {shape:?} sel={sel}"
    );
    assert_eq!(fold_fresh(), want, "B {label} {shape:?} sel={sel}");
    assert_eq!(fold_reused(), want, "B' {label} {shape:?} sel={sel}");

    let (a_ns, [null, alloc, b, b2, ac]) = paired(
        101,
        || run_a(&mut scratch.borrow_mut()),
        || {
            let mut s = Scratch::for_program(&p, d.n).expect("scratch");
            run_a(&mut s)
        },
        fold_fresh,
        fold_reused,
        || run_ac(&mut scratch.borrow_mut()),
    );
    let scratch_b = p.scratch_slots as usize * TILE_WORDS.min(words_for(d.n)) * 8;
    let (pa, pb) = passes(shape);
    println!(
        "{label:>14} {:>4} {sel:>4} | {scratch_b:>9} | {pa:>2}/{pb:<2} | {a_ns:>11.0} | {null:>6.3} {alloc:>8.3} {b:>6.3} {b2:>6.3} {ac:>6.3}",
        format!("{shape:?}")
    );
}

fn main() {
    let tier = if cfg!(target_feature = "avx512f") {
        "avx512f"
    } else if cfg!(target_feature = "avx2") {
        "avx2"
    } else {
        "scalar/other"
    };
    println!("D-CMR-2b phase 1 probe; build tier: {tier}; TILE_WORDS = {TILE_WORDS}");
    println!(
        "{:>14} {:>4} {:>4} | {:>9} | {:>5} | {:>11} | {:>6} {:>8} {:>6} {:>6} {:>6}",
        "workload",
        "shp",
        "sel%",
        "scratch_B",
        "pass",
        "A_ns",
        "null",
        "alloc/A",
        "B/A",
        "B'/A",
        "Ac/A"
    );
    // Thresholds giving roughly 1 %, 50 % and 99 % per comparison.
    let sels = [(989, 10, 1), (499, 500, 50), (9, 990, 99)];
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for &n in &[16_384usize, 1 << 20, 4 << 20] {
        let d = data(n, &mut seed);
        let whole = 0..n;
        for &(ta, tb, sel) in &sels {
            for shape in SHAPES {
                measure(
                    &format!("whole {n}"),
                    &d,
                    std::slice::from_ref(&whole),
                    shape,
                    ta,
                    tb,
                    sel,
                );
            }
        }
    }
    let d = data(1 << 20, &mut seed);
    for &len in &[200usize, 3_000, 40_000] {
        let runs = runs_for(d.n, len, &mut seed);
        let label = format!("{}x{len}", runs.len());
        for &(ta, tb, sel) in &sels {
            for shape in SHAPES {
                measure(&label, &d, &runs, shape, ta, tb, sel);
            }
        }
    }
}
