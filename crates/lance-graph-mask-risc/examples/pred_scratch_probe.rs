//! D-CMR-2b, phase 1 (lab only): what does a residual comparison's scratch
//! actually cost, and does a scratch-free fold win?
//!
//! `fused_ternlog` declines `Pred`, so a program with a comparison runs tiled:
//! each op writes its destination slot for one tile ([`TILE_WORDS`] words),
//! later ops re-read it, and the terminal counts the last slot. Scratch is
//! caller-owned and tile-sized (`slots × TILE_WORDS` words), never
//! population-sized, so the question is not heap memory. It is whether the
//! per-op tile write-back and re-read, and the separate `And` + count passes,
//! cost anything measurable against a fold that compares into stack chunks and
//! counts them in one fused `mask_ternlog_popcount` call.
//!
//! Arms, on the same data and the same programs:
//!
//! - **A** `execute_extent` with a `Scratch` built once outside the timing
//!   (what a caller that reuses its scratch pays).
//! - **A+alloc** the same call with `Scratch::for_program` inside the timing
//!   (what `fold::set_lower` pays per call today).
//! - **B** a lab fold: per tile, each comparison into a stack chunk via the
//!   same `ndarray::simd` compare primitive the executor uses, then ONE
//!   `mask_ternlog_popcount` over the chunks (and a resident plane). No heap
//!   scratch, no `And` pass, no count pass of its own.
//!
//! Shapes (`a = lane0 > ta`, `b = lane1 < tb`, `p = plane 0`):
//!
//! - `S1` Count(a)
//! - `S2` Count(a ∧ b)
//! - `S2u` Count(a ∧ b) with `b` gated `under` `a` (the executor's own fused
//!   compare-under form; arm B uses `*_to_mask_under` likewise)
//! - `S3` Count(a ∧ b ∧ p)
//!
//! Every configuration is checked against a bit-serial row oracle and arm A
//! before anything is timed. Reported: scratch bytes, the op passes each arm
//! makes per tile, median ns of 101 and the B/A ratio.
//!
//! `cargo run --release -p lance-graph-mask-risc --example pred_scratch_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, Scratch, TILE_WORDS};
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
/// A+alloc, B and B' back to back, in an order that rotates per rep, so drift
/// and frequency changes hit all arms alike. Returns the median ns of A and
/// the medians of the per-rep ratios A'/A (the method's own noise),
/// A+alloc/A, B/A and B'/A.
fn paired(
    reps: usize,
    mut a: impl FnMut() -> usize,
    mut aa: impl FnMut() -> usize,
    mut b: impl FnMut() -> usize,
    mut b2: impl FnMut() -> usize,
) -> (f64, [f64; 4]) {
    let mut ta = vec![];
    let mut ratios: [Vec<f64>; 4] = Default::default();
    for r in 0..reps {
        let mut t = [0f64; 5];
        for k in 0..5 {
            let i = (k + r) % 5;
            t[i] = match i {
                0 | 1 => time(&mut a),
                2 => time(&mut aa),
                3 => time(&mut b),
                _ => time(&mut b2),
            };
        }
        ta.push(t[0]);
        for j in 0..4 {
            ratios[j].push(t[j + 1] / t[0]);
        }
    }
    let [n, al, b, b2] = ratios;
    (med(ta), [med(n), med(al), med(b), med(b2)])
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    S1,
    S2,
    S2u,
    S3,
}

fn program(shape: Shape, ta: i32, tb: i32) -> Program {
    let a = Pred::GtI32 { lane: 0, t: ta };
    let b = Pred::LtI32 { lane: 1, t: tb };
    let (ops, last) = match shape {
        Shape::S1 => (
            vec![MaskOp::Pred {
                pred: a,
                under: None,
                dst: 0,
            }],
            0,
        ),
        Shape::S2 => (
            vec![
                MaskOp::Pred {
                    pred: a,
                    under: None,
                    dst: 0,
                },
                MaskOp::Pred {
                    pred: b,
                    under: None,
                    dst: 1,
                },
                MaskOp::And {
                    a: Operand::Scratch(0),
                    b: Operand::Scratch(1),
                    dst: 2,
                },
            ],
            2,
        ),
        Shape::S2u => (
            vec![
                MaskOp::Pred {
                    pred: a,
                    under: None,
                    dst: 0,
                },
                MaskOp::Pred {
                    pred: b,
                    under: Some(Operand::Scratch(0)),
                    dst: 1,
                },
            ],
            1,
        ),
        Shape::S3 => (
            vec![
                MaskOp::Pred {
                    pred: a,
                    under: None,
                    dst: 0,
                },
                MaskOp::Pred {
                    pred: b,
                    under: None,
                    dst: 1,
                },
                MaskOp::And {
                    a: Operand::Scratch(0),
                    b: Operand::Scratch(1),
                    dst: 2,
                },
                MaskOp::And {
                    a: Operand::Scratch(2),
                    b: Operand::Plane(0),
                    dst: 3,
                },
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

/// The bit-serial reference.
fn oracle(shape: Shape, l0: &[i32], l1: &[i32], plane: &[u64], ta: i32, tb: i32) -> usize {
    (0..l0.len())
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

/// Arm B: compare into fresh stack chunks, one fused count per tile.
fn fold_b(shape: Shape, l0: &[i32], l1: &[i32], plane: &[u64], ta: i32, tb: i32) -> usize {
    let mut ca = [0u64; TILE_WORDS];
    let mut cb = [0u64; TILE_WORDS];
    fold_into(shape, l0, l1, plane, ta, tb, &mut ca, &mut cb)
}

/// The fold itself, over chunks the caller provides. Arm B' calls it with
/// chunks reused across calls (as arm A reuses its scratch), which isolates
/// the value of the fused count from the cost of initialising the chunks.
#[allow(clippy::too_many_arguments)]
fn fold_into(
    shape: Shape,
    l0: &[i32],
    l1: &[i32],
    plane: &[u64],
    ta: i32,
    tb: i32,
    ca: &mut [u64; TILE_WORDS],
    cb: &mut [u64; TILE_WORDS],
) -> usize {
    let n = l0.len();
    let words = words_for(n);
    let mut total = 0u64;
    let mut w = 0;
    while w < words {
        let tw = TILE_WORDS.min(words - w);
        let rows = w * 64..((w + tw) * 64).min(n);
        let (ca, cb) = (&mut ca[..tw], &mut cb[..tw]);
        gt_i32_to_mask(&l0[rows.clone()], ta, ca);
        total += match shape {
            Shape::S1 => mask_ternlog_popcount::<AND3>(ca, ca, ca),
            Shape::S2 => {
                lt_i32_to_mask(&l1[rows], tb, cb);
                mask_ternlog_popcount::<AND3>(ca, cb, ca)
            }
            Shape::S2u => {
                lt_i32_to_mask_under(&l1[rows], tb, ca, cb);
                mask_ternlog_popcount::<AND3>(cb, cb, cb)
            }
            Shape::S3 => {
                lt_i32_to_mask(&l1[rows], tb, cb);
                mask_ternlog_popcount::<AND3>(ca, cb, &plane[w..w + tw])
            }
        };
        w += tw;
    }
    total as usize
}

/// Op passes over one tile: (arm A, arm B).
fn passes(shape: Shape) -> (usize, usize) {
    match shape {
        // A: compare, count. B: compare, fused count.
        Shape::S1 => (2, 2),
        // A: compare, compare, and, count. B: compare, compare, fused count.
        Shape::S2 => (4, 3),
        // A: compare, compare-under, count. B: the same three.
        Shape::S2u => (3, 3),
        // A: compare, compare, and, and, count. B: compare, compare, fused count.
        Shape::S3 => (5, 3),
    }
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
        "{:>9} {:>4} {:>5} | {:>10} {:>8} | {:>6} | {:>11} | {:>7} {:>9} {:>7} {:>7}",
        "n_rows",
        "shp",
        "sel%",
        "scratch_B",
        "stack_B",
        "pass",
        "A_ns",
        "null",
        "alloc/A",
        "B/A",
        "B'/A"
    );
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for &n in &[16_384usize, 1 << 20, 4 << 20] {
        let l0: Vec<i32> = (0..n).map(|_| (lcg(&mut seed) % 1000) as i32).collect();
        let l1: Vec<i32> = (0..n).map(|_| (lcg(&mut seed) % 1000) as i32).collect();
        let mut plane: Vec<u64> = (0..words_for(n))
            .map(|_| lcg(&mut seed) ^ lcg(&mut seed) << 21)
            .collect();
        if n % 64 != 0 {
            let last = plane.len() - 1;
            plane[last] &= (1u64 << (n % 64)) - 1;
        }
        let masks: [&[u64]; 1] = [&plane];
        let lanes = [LaneRef::I32(&l0), LaneRef::I32(&l1)];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        // Thresholds giving roughly 1 %, 50 % and 99 % per comparison.
        for &(ta, tb, sel) in &[(989, 10, 1), (499, 500, 50), (9, 990, 99)] {
            for shape in [Shape::S1, Shape::S2, Shape::S2u, Shape::S3] {
                let p = program(shape, ta, tb);
                let want = oracle(shape, &l0, &l1, &plane, ta, tb);
                let mut scratch = Scratch::for_program(&p, n).expect("scratch");
                let run_a = |s: &mut Scratch<'_>| match execute_extent(
                    &p,
                    &planes,
                    &Foreign::NONE,
                    s,
                    Out::None,
                    0..n,
                )
                .expect("execute")
                {
                    Value::Count(c) => c,
                    v => panic!("unexpected {v:?}"),
                };
                assert_eq!(run_a(&mut scratch), want, "A {shape:?} n={n} sel={sel}");
                assert_eq!(
                    fold_b(shape, &l0, &l1, &plane, ta, tb),
                    want,
                    "B {shape:?} n={n} sel={sel}"
                );

                let reps = 101;
                let scratch = std::cell::RefCell::new(scratch);
                let chunks = std::cell::RefCell::new(([0u64; TILE_WORDS], [0u64; TILE_WORDS]));
                {
                    let (ca, cb) = &mut *chunks.borrow_mut();
                    assert_eq!(
                        fold_into(shape, &l0, &l1, &plane, ta, tb, ca, cb),
                        want,
                        "B' {shape:?}"
                    );
                }
                let (a_ns, [null, alloc, ratio, ratio2]) = paired(
                    reps,
                    || run_a(&mut scratch.borrow_mut()),
                    || {
                        let mut s = Scratch::for_program(&p, n).expect("scratch");
                        run_a(&mut s)
                    },
                    || fold_b(shape, &l0, &l1, &plane, ta, tb),
                    || {
                        let (ca, cb) = &mut *chunks.borrow_mut();
                        fold_into(shape, &l0, &l1, &plane, ta, tb, ca, cb)
                    },
                );
                let slots = p.scratch_slots as usize;
                let tw = TILE_WORDS.min(words_for(n));
                let scratch_b = slots * tw * 8;
                let stack_b = match shape {
                    Shape::S1 => 1,
                    _ => 2,
                } * TILE_WORDS
                    * 8;
                let (pa, pb) = passes(shape);
                println!(
                    "{:>9} {:>4} {:>5} | {:>10} {:>8} | {:>2}/{:<3} | {:>11.0} | {:>7.3} {:>9.3} {:>7.3} {:>7.3}",
                    n,
                    format!("{shape:?}"),
                    sel,
                    scratch_b,
                    stack_b,
                    pa,
                    pb,
                    a_ns,
                    null,
                    alloc,
                    ratio,
                    ratio2
                );
            }
        }
    }
}
