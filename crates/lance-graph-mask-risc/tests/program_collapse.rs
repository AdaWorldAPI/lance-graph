//! A composed Boolean PROGRAM collapses before execution and terminates in
//! Count/Any without its membership ever becoming bits.
//!
//! Two routes reach the same fold:
//!
//! - the existing fuser: a `BoolExpr` over at most three distinct planes is
//!   lowered by `fuse_program` to ONE `Ternlog`, which the fold consumes;
//! - a raw op sequence (`And` → `Or` → `Not` → …) is interpreted symbolically
//!   by `Program::fused_ternlog` as an 8-bit truth table over its (at most
//!   three) plane leaves, and that table is the fold's immediate.
//!
//! Every case is checked three ways over the same absolute rows: the fold
//! (run with a zero-slot scratch, so it CANNOT write a derived word), the
//! MATERIALIZE arm (the same ops with a `Keep` terminal, which is never fused
//! and runs the ordinary tiled path), and the crate's row-by-row reference
//! oracle.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::fuse::{fuse_program, BoolExpr};
use lance_graph_mask_risc::{
    reference_execute, Foreign, MaskOp, Operand, Out, Planes, Program, Terminal, Value,
};

struct Counting;

thread_local! {
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn bytes() -> usize {
    BYTES.with(Cell::get)
}

// SAFETY: a pure pass-through to `System`; the counter is the only addition.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = BYTES.try_with(|b| b.set(b.get() + layout.size()));
        // SAFETY: same layout, same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

fn random_plane(n: usize, seed: &mut u64, sparse: bool) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for r in 0..n {
        let x = lcg(seed);
        if (sparse && x.is_multiple_of(11)) || (!sparse && x & 1 == 1) {
            w[r / 64] |= 1u64 << (r % 64);
        }
    }
    w
}

fn extents(n: usize) -> Vec<(usize, usize)> {
    let mut v = vec![(0, n), (0, 0)];
    for k in [1usize, 63, 64, 65, 129] {
        if k <= n {
            v.push((k, n));
            v.push((0, k));
        }
    }
    if n >= 3 {
        v.push((n / 3, 2 * n / 3 + 1));
    }
    v.retain(|&(lo, hi)| lo <= hi && hi <= n);
    v
}

/// The same ops with the terminal swapped for `Keep` (never fused), counted
/// over the extent — the tiled, scratch-writing execution of the program.
fn keep_count(p: &Program, planes: &Planes<'_>, lo: usize, hi: usize) -> usize {
    let mask = match p.terminal {
        Terminal::Count { mask } | Terminal::Any { mask } => mask,
        _ => unreachable!("fixtures only use Count/Any"),
    };
    let keep = Program::new(p.ops.clone(), Terminal::Keep { mask });
    assert!(keep.fused_ternlog().is_none(), "Keep is never fused");
    let mut sc = Scratch::for_program(&keep, planes.n_rows).expect("scratch");
    let mut out = vec![0u64; words_for(planes.n_rows)];
    execute_extent(
        &keep,
        planes,
        &Foreign::NONE,
        &mut sc,
        Out::Mask(&mut out),
        lo..hi,
    )
    .expect("keep arm");
    out.iter().map(|w| w.count_ones() as usize).sum()
}

/// Check one Count program (and its Any twin) over every extent, three ways.
fn check(p_count: &Program, planes: &Planes<'_>, what: &str) {
    let mask = match p_count.terminal {
        Terminal::Count { mask } => mask,
        _ => unreachable!(),
    };
    let p_any = Program::new(p_count.ops.clone(), Terminal::Any { mask });
    assert!(p_count.fused_ternlog().is_some(), "{what}: must collapse");
    assert!(p_any.fused_ternlog().is_some(), "{what}: Any must collapse");
    for (lo, hi) in extents(planes.n_rows) {
        let kept = keep_count(p_count, planes, lo, hi);
        let mut s = Scratch::new(0, 0);
        let folded = execute_extent(p_count, planes, &Foreign::NONE, &mut s, Out::None, lo..hi)
            .expect("fold count");
        assert_eq!(folded, Value::Count(kept), "{what} count [{lo},{hi})");
        let any = execute_extent(&p_any, planes, &Foreign::NONE, &mut s, Out::None, lo..hi)
            .expect("fold any");
        assert_eq!(any, Value::Bool(kept > 0), "{what} any [{lo},{hi})");
    }
    // The whole-population answer agrees with the independent row oracle too.
    let whole = reference_execute(p_count, planes, None).expect("oracle");
    let mut s = Scratch::new(0, 0);
    let folded = execute_extent(
        p_count,
        planes,
        &Foreign::NONE,
        &mut s,
        Out::None,
        0..planes.n_rows,
    )
    .expect("fold");
    assert_eq!(folded, whole, "{what} vs reference oracle");
}

fn leaf(i: u16) -> Box<BoolExpr> {
    Box::new(BoolExpr::Leaf(Operand::Plane(i)))
}

/// FAILS IF: the existing fuser's output for a ≤3-leaf expression does not
/// reach the fold — i.e. `fuse_program` → one `Ternlog` → `Count`/`Any` still
/// writes a mask — or folds to a wrong answer.
#[test]
fn a_fused_boolexpr_reaches_the_fold_end_to_end() {
    let exprs: [(&str, BoolExpr); 4] = [
        (
            "(a & b) | !c",
            BoolExpr::Or(
                Box::new(BoolExpr::And(leaf(0), leaf(1))),
                Box::new(BoolExpr::Not(leaf(2))),
            ),
        ),
        (
            "(a & b) | (a & !b)",
            BoolExpr::Or(
                Box::new(BoolExpr::And(leaf(0), leaf(1))),
                Box::new(BoolExpr::And(leaf(0), Box::new(BoolExpr::Not(leaf(1))))),
            ),
        ),
        (
            "!(a ^ b)",
            BoolExpr::Not(Box::new(BoolExpr::Xor(leaf(0), leaf(1)))),
        ),
        (
            "((a | b) & c) ^ (a & c)",
            BoolExpr::Xor(
                Box::new(BoolExpr::And(
                    Box::new(BoolExpr::Or(leaf(0), leaf(1))),
                    leaf(2),
                )),
                Box::new(BoolExpr::And(leaf(0), leaf(2))),
            ),
        ),
    ];
    for n in [1usize, 65, 1317] {
        let mut seed = 0xE2E_u64 ^ n as u64;
        let m = [
            random_plane(n, &mut seed, false),
            random_plane(n, &mut seed, true),
            random_plane(n, &mut seed, false),
        ];
        let masks: [&[u64]; 3] = [&m[0], &m[1], &m[2]];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        for (what, e) in &exprs {
            let p = fuse_program(e, 0, |mask| Terminal::Count { mask }).expect("fuse");
            assert_eq!(p.ops.len(), 1, "{what}: the fuser emits ONE ternlog");
            assert!(matches!(p.ops[0], MaskOp::Ternlog { .. }), "{what}");
            check(&p, &planes, what);
        }
    }
}

/// A hand-written catalogue of raw sequences the recogniser must collapse,
/// each shaped the way a lowering pass emits ops (not through `BoolExpr`).
fn catalogue() -> Vec<(&'static str, Program)> {
    let (a, b, c) = (Operand::Plane(0), Operand::Plane(1), Operand::Plane(2));
    let s = Operand::Scratch;
    let count = |x: u16| Terminal::Count {
        mask: Operand::Scratch(x),
    };
    vec![
        (
            "and→not→or",
            Program::new(
                vec![
                    MaskOp::And { a, b, dst: 0 },
                    MaskOp::Not { a: c, dst: 1 },
                    MaskOp::Or {
                        a: s(0),
                        b: s(1),
                        dst: 2,
                    },
                ],
                count(2),
            ),
        ),
        (
            "slot reuse in place",
            Program::new(
                vec![
                    MaskOp::And { a, b, dst: 0 },
                    MaskOp::Or {
                        a: s(0),
                        b: c,
                        dst: 0,
                    },
                    MaskOp::Not { a: s(0), dst: 0 },
                ],
                count(0),
            ),
        ),
        (
            "andnot chain with a repeated leaf",
            Program::new(
                vec![
                    MaskOp::AndNot { a, b, dst: 0 },
                    MaskOp::AndNot {
                        a: s(0),
                        b: c,
                        dst: 1,
                    },
                    MaskOp::Or {
                        a: s(1),
                        b: a,
                        dst: 2,
                    },
                ],
                count(2),
            ),
        ),
        (
            "ternlog over a derived input",
            Program::new(
                vec![
                    MaskOp::Xor { a, b, dst: 0 },
                    MaskOp::Ternlog {
                        imm: 0xE8,
                        a: s(0),
                        b,
                        c,
                        dst: 1,
                    },
                ],
                count(1),
            ),
        ),
        (
            "dead op beside the result",
            Program::new(
                vec![
                    MaskOp::And { a, b, dst: 0 },
                    MaskOp::Or { a: b, b: c, dst: 1 },
                ],
                count(0),
            ),
        ),
        (
            "double complement",
            Program::new(
                vec![MaskOp::Not { a, dst: 0 }, MaskOp::Not { a: s(0), dst: 1 }],
                count(1),
            ),
        ),
        (
            "contradiction a & !a",
            Program::new(
                vec![
                    MaskOp::Not { a, dst: 0 },
                    MaskOp::And { a, b: s(0), dst: 1 },
                ],
                count(1),
            ),
        ),
    ]
}

/// FAILS IF: a raw op sequence over at most three planes fails to collapse, or
/// the collapsed fold disagrees with the tiled path or the oracle.
#[test]
fn raw_op_sequences_collapse_and_agree_with_the_tiled_path() {
    for n in [1usize, 63, 64, 65, 133, 1317] {
        for sparse in [false, true] {
            let mut seed = 0xC011_A95E ^ n as u64 ^ u64::from(sparse);
            let m = [
                random_plane(n, &mut seed, sparse),
                random_plane(n, &mut seed, false),
                random_plane(n, &mut seed, sparse),
            ];
            let masks: [&[u64]; 3] = [&m[0], &m[1], &m[2]];
            let planes = Planes {
                n_rows: n,
                masks: &masks,
                lanes: &[],
            };
            for (what, p) in catalogue() {
                check(&p, &planes, what);
            }
        }
    }
}

/// FAILS IF: ANY randomly generated chain of the six Boolean ops over three
/// planes, with slot reuse and derived operands, collapses to a wrong answer —
/// or fails to collapse. 400 chains, length 1..=8, slots 0..6.
#[test]
fn random_chains_collapse_and_agree_with_the_tiled_path() {
    let n = 333;
    let mut seed = 0xBAD_C0DE_u64;
    let m = [
        random_plane(n, &mut seed, false),
        random_plane(n, &mut seed, true),
        random_plane(n, &mut seed, false),
    ];
    let masks: [&[u64]; 3] = [&m[0], &m[1], &m[2]];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let mut distinct_results = std::collections::HashSet::new();
    for _ in 0..400 {
        let len = 1 + (lcg(&mut seed) % 8) as usize;
        let mut written: Vec<u16> = Vec::new();
        let mut ops = Vec::new();
        let pick = |seed: &mut u64, written: &Vec<u16>| {
            if !written.is_empty() && lcg(seed).is_multiple_of(2) {
                Operand::Scratch(written[(lcg(seed) as usize) % written.len()])
            } else {
                Operand::Plane((lcg(seed) % 3) as u16)
            }
        };
        let mut last = 0u16;
        for _ in 0..len {
            let (x, y, z) = (
                pick(&mut seed, &written),
                pick(&mut seed, &written),
                pick(&mut seed, &written),
            );
            let dst = (lcg(&mut seed) % 6) as u16;
            ops.push(match lcg(&mut seed) % 6 {
                0 => MaskOp::And { a: x, b: y, dst },
                1 => MaskOp::Or { a: x, b: y, dst },
                2 => MaskOp::Xor { a: x, b: y, dst },
                3 => MaskOp::AndNot { a: x, b: y, dst },
                4 => MaskOp::Not { a: x, dst },
                _ => MaskOp::Ternlog {
                    imm: (lcg(&mut seed) & 0xFF) as u8,
                    a: x,
                    b: y,
                    c: z,
                    dst,
                },
            });
            if !written.contains(&dst) {
                written.push(dst);
            }
            last = dst;
        }
        let p = Program::new(
            ops,
            Terminal::Count {
                mask: Operand::Scratch(last),
            },
        );
        check(&p, &planes, "random chain");
        let mut s = Scratch::new(0, 0);
        if let Ok(Value::Count(c)) =
            execute_extent(&p, &planes, &Foreign::NONE, &mut s, Out::None, 0..n)
        {
            distinct_results.insert(c);
        }
    }
    // Anti-vacuity: the generator must produce genuinely different functions,
    // not 400 copies of a constant.
    assert!(
        distinct_results.len() >= 20,
        "only {} distinct counts",
        distinct_results.len()
    );
}

/// FAILS IF: a chain over FOUR planes is collapsed (it cannot fit one 3-input
/// table) or runs to a wrong answer on the path it falls back to. This is the
/// boundary where the compact algebra stops, pinned rather than papered over.
#[test]
fn a_four_plane_chain_stays_on_the_tiled_path_and_is_correct() {
    let n = 1000;
    let mut seed = 0x4_u64;
    let m: Vec<Vec<u64>> = (0..4).map(|_| random_plane(n, &mut seed, false)).collect();
    let masks: [&[u64]; 4] = [&m[0], &m[1], &m[2], &m[3]];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let (a, b, c, d) = (
        Operand::Plane(0),
        Operand::Plane(1),
        Operand::Plane(2),
        Operand::Plane(3),
    );
    let p = Program::new(
        vec![
            MaskOp::And { a, b, dst: 0 },
            MaskOp::Or { a: c, b: d, dst: 1 },
            MaskOp::Xor {
                a: Operand::Scratch(0),
                b: Operand::Scratch(1),
                dst: 2,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(2),
        },
    );
    assert!(p.fused_ternlog().is_none());
    assert!(p.requires_scratch());
    let mut sc = Scratch::for_program(&p, n).expect("scratch");
    let got = execute_extent(&p, &planes, &Foreign::NONE, &mut sc, Out::None, 0..n).expect("tiled");
    assert_eq!(got, reference_execute(&p, &planes, None).expect("oracle"));
}

/// FAILS IF: a collapsed chain carves a slot, writes a scratch word, or
/// allocates. The twin half proves the poison probe can see a carve.
#[test]
fn a_collapsed_chain_writes_nothing_and_allocates_nothing() {
    let n = 4133;
    let mut seed = 0xA11_u64;
    let m = [
        random_plane(n, &mut seed, false),
        random_plane(n, &mut seed, false),
        random_plane(n, &mut seed, true),
    ];
    let masks: [&[u64]; 3] = [&m[0], &m[1], &m[2]];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    for (what, p) in catalogue() {
        assert!(!p.requires_scratch(), "{what}");
        let mut poison = vec![u64::MAX; 8 * words_for(n)];
        {
            let mut sc = Scratch::over_for_program(&mut poison, &p, n).expect("arena");
            let before = bytes();
            let v = execute_extent(&p, &planes, &Foreign::NONE, &mut sc, Out::None, 7..n - 3)
                .expect("fold");
            assert_eq!(bytes(), before, "{what} allocated");
            assert!(matches!(v, Value::Count(_)));
        }
        assert!(
            poison.iter().all(|&w| w == u64::MAX),
            "{what} wrote scratch"
        );
    }
    let keep = Program::new(
        catalogue()[0].1.ops.clone(),
        Terminal::Keep {
            mask: Operand::Scratch(2),
        },
    );
    let mut poison = vec![u64::MAX; 8 * words_for(n)];
    drop(Scratch::over_for_program(&mut poison, &keep, n).expect("arena"));
    assert!(
        poison.iter().any(|&w| w != u64::MAX),
        "probe must see a carve"
    );
}
