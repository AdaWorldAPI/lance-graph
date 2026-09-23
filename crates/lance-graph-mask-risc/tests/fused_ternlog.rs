//! A Boolean membership over resident planes ends in Count/Any with no mask.
//!
//! `MaskOp::{And, Or, Xor, AndNot, Ternlog}` over `Operand::Plane`s, folded by
//! `Count` or `Any` of its own `dst`, is one truth table. The executor folds it
//! with `ndarray::simd::mask_ternlog_{popcount,any}` over the borrowed planes'
//! words, and combines the (at most two) words an extent cuts in a one-word
//! register. No scratch slot is carved and no membership word is written.
//!
//! Every case checks the FOLD against the MATERIALIZE arm (the same op with a
//! `Keep` terminal, which is never fused) and against a bit-serial scalar
//! oracle over the same absolute rows.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{
    Foreign, FusedFold, MaskOp, Operand, Out, Planes, Program, Terminal, Value,
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

/// A conforming plane of `n` rows (tail bits clear).
fn plane(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; words_for(n)];
    for r in 0..n {
        if set(r) {
            w[r / 64] |= 1u64 << (r % 64);
        }
    }
    w
}

fn bit(p: &[u64], r: usize) -> u8 {
    (p[r / 64] >> (r % 64) & 1) as u8
}

/// The shapes under test: each op kind with its truth table (the ORACLE's
/// spelling — independent of the crate's own constants).
#[derive(Clone, Copy, Debug)]
enum Shape {
    And,
    Or,
    Xor,
    AndNot,
    Ternlog(u8),
}

impl Shape {
    fn table(self) -> u8 {
        match self {
            Shape::Ternlog(t) => t,
            // Evaluate the two-input function on each (a, b, c) index.
            s => (0u8..8).fold(0u8, |acc, i| {
                let (a, b) = (i >> 2 & 1, i >> 1 & 1);
                let v = match s {
                    Shape::And => a & b,
                    Shape::Or => a | b,
                    Shape::Xor => a ^ b,
                    Shape::AndNot => a & (1 - b),
                    Shape::Ternlog(_) => unreachable!(),
                };
                acc | (v << i)
            }),
        }
    }

    fn op(self, dst: u16) -> MaskOp {
        let (a, b, c) = (Operand::Plane(0), Operand::Plane(1), Operand::Plane(2));
        match self {
            Shape::And => MaskOp::And { a, b, dst },
            Shape::Or => MaskOp::Or { a, b, dst },
            Shape::Xor => MaskOp::Xor { a, b, dst },
            Shape::AndNot => MaskOp::AndNot { a, b, dst },
            Shape::Ternlog(imm) => MaskOp::Ternlog { imm, a, b, c, dst },
        }
    }
}

fn program(s: Shape, terminal: Terminal) -> Program {
    Program::new(vec![s.op(0)], terminal)
}

fn count_p(s: Shape) -> Program {
    program(
        s,
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    )
}

fn any_p(s: Shape) -> Program {
    program(
        s,
        Terminal::Any {
            mask: Operand::Scratch(0),
        },
    )
}

/// Bit-serial oracle over the ABSOLUTE rows `[lo, hi)`.
fn oracle(s: Shape, m: &[Vec<u64>; 3], lo: usize, hi: usize) -> usize {
    let t = s.table();
    // Two-input ops read `b` as their third input, matching how the op is
    // defined (the table ignores `c`, so either choice is exact).
    (lo..hi)
        .filter(|&r| {
            let idx = bit(&m[0], r) << 2 | bit(&m[1], r) << 1 | bit(&m[2], r);
            t >> idx & 1 == 1
        })
        .count()
}

fn fold(p: &Program, planes: &Planes<'_>, lo: usize, hi: usize) -> Value {
    let mut s = Scratch::new(0, 0);
    execute_extent(p, planes, &Foreign::NONE, &mut s, Out::None, lo..hi).expect("fold arm")
}

/// The MATERIALIZE arm: the same op kept as a mask over the extent, counted.
fn keep_count(s: Shape, planes: &Planes<'_>, lo: usize, hi: usize) -> usize {
    let p = program(
        s,
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    assert!(p.fused_ternlog().is_none(), "Keep is never fused");
    let mut sc = Scratch::for_program(&p, planes.n_rows).expect("scratch");
    let mut out = vec![0u64; words_for(planes.n_rows)];
    execute_extent(
        &p,
        planes,
        &Foreign::NONE,
        &mut sc,
        Out::Mask(&mut out),
        lo..hi,
    )
    .expect("keep arm");
    out.iter().map(|w| w.count_ones() as usize).sum()
}

fn extents(n: usize) -> Vec<(usize, usize)> {
    let mut v = vec![(0, n), (0, 0), (n, n)];
    for k in [1usize, 63, 64, 65, 127, 128, 129] {
        if k <= n {
            v.push((k, n));
            v.push((0, k));
        }
    }
    if n >= 3 {
        v.push((n / 3, 2 * n / 3 + 1));
        v.push((n / 2, n / 2 + 1));
    }
    v.retain(|&(lo, hi)| lo <= hi && hi <= n);
    v
}

fn planes_for(n: usize, seed: u64) -> Vec<[Vec<u64>; 3]> {
    let mut s = seed;
    let rnd = |s: &mut u64, dense: bool| {
        let bits: Vec<bool> = (0..n)
            .map(|_| {
                let x = lcg(s);
                if dense {
                    x & 1 == 1
                } else {
                    x.is_multiple_of(13)
                }
            })
            .collect();
        plane(n, |r| bits[r])
    };
    vec![
        [rnd(&mut s, true), rnd(&mut s, true), rnd(&mut s, true)],
        [rnd(&mut s, false), rnd(&mut s, false), rnd(&mut s, true)],
        [
            plane(n, |_| false),
            plane(n, |_| false),
            plane(n, |_| false),
        ],
        [plane(n, |_| true), plane(n, |_| true), plane(n, |_| false)],
    ]
}

fn check(s: Shape, n: usize, m: &[Vec<u64>; 3]) {
    let masks: [&[u64]; 3] = [&m[0], &m[1], &m[2]];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let (cp, ap) = (count_p(s), any_p(s));
    assert_eq!(
        cp.fused_ternlog().map(|f| f.fold),
        Some(FusedFold::Count),
        "{s:?}"
    );
    assert_eq!(
        ap.fused_ternlog().map(|f| f.fold),
        Some(FusedFold::Any),
        "{s:?}"
    );
    for (lo, hi) in extents(n) {
        let want = oracle(s, m, lo, hi);
        let kept = keep_count(s, &planes, lo, hi);
        assert_eq!(kept, want, "keep arm vs oracle {s:?} n={n} [{lo},{hi})");
        assert_eq!(
            fold(&cp, &planes, lo, hi),
            Value::Count(want),
            "count {s:?} n={n} [{lo},{hi})"
        );
        assert_eq!(
            fold(&ap, &planes, lo, hi),
            Value::Bool(want > 0),
            "any {s:?} n={n} [{lo},{hi})"
        );
    }
}

/// FAILS IF: the fold disagrees with the kept mask or the scalar oracle for any
/// op kind × population × plane shape × absolute extent — including an empty
/// extent, one row, cuts at 63/64/65/127/128/129, a sub-64-row population tail
/// and the whole population.
#[test]
fn the_fold_agrees_with_keep_and_the_oracle_for_every_two_input_op() {
    for n in [1usize, 63, 64, 65, 133, 1317, 4133] {
        for m in planes_for(n, 0xF0_1D ^ n as u64) {
            for s in [Shape::And, Shape::Or, Shape::Xor, Shape::AndNot] {
                check(s, n, &m);
            }
        }
    }
}

/// FAILS IF: any of the 256 truth tables folds wrongly — an odd table (true of
/// all-zero inputs) in particular, whose dead population-tail bits must never
/// be counted.
#[test]
fn the_fold_agrees_for_all_256_tables() {
    for n in [65usize, 133] {
        for m in planes_for(n, 0x7AB1E ^ n as u64) {
            for imm in 0u8..=255 {
                check(Shape::Ternlog(imm), n, &m);
            }
        }
    }
}

/// FAILS IF: an odd table's dead tail bits leak into the count. NOR over
/// all-zero planes is true on every LIVE row: 70 rows, never 128.
#[test]
fn an_odd_table_never_counts_the_population_tail() {
    for n in [1usize, 63, 64, 65, 70, 1000] {
        let z = plane(n, |_| false);
        let masks: [&[u64]; 3] = [&z, &z, &z];
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &[],
        };
        let nor = Shape::Ternlog(0x01);
        assert_eq!(fold(&count_p(nor), &planes, 0, n), Value::Count(n), "n={n}");
        assert_eq!(fold(&any_p(nor), &planes, 0, n), Value::Bool(true), "n={n}");
        // A partial extent ending short of the tail counts its own rows only.
        assert_eq!(
            fold(&count_p(nor), &planes, 0, n / 2),
            Value::Count(n / 2),
            "n={n}"
        );
    }
}

/// FAILS IF: the fold arm materializes anything — carves a slot, writes a
/// scratch word, or allocates — or needs a scratch at all.
#[test]
fn the_fold_writes_no_membership_and_allocates_nothing() {
    let n = 4133;
    let m = &planes_for(n, 0xA11C)[0];
    let masks: [&[u64]; 3] = [&m[0], &m[1], &m[2]];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    for s in [
        Shape::And,
        Shape::Xor,
        Shape::Ternlog(0xEA),
        Shape::Ternlog(0x01),
    ] {
        for p in [count_p(s), any_p(s)] {
            assert!(!p.requires_scratch(), "{s:?} must need no scratch");
            // A poisoned arena. `over_for_program` zero-fills whatever it
            // carves, so any carved slot — and any derived word written into
            // one — shows up as a changed word.
            let mut poison = vec![u64::MAX; 4 * words_for(n)];
            {
                let mut sc = Scratch::over_for_program(&mut poison, &p, n).expect("arena");
                let before = bytes();
                let v = execute_extent(&p, &planes, &Foreign::NONE, &mut sc, Out::None, 7..n - 3)
                    .expect("fold");
                assert_eq!(bytes(), before, "{s:?} allocated during the fold");
                assert!(matches!(v, Value::Count(_) | Value::Bool(_)));
            }
            assert!(
                poison.iter().all(|&w| w == u64::MAX),
                "{s:?} wrote into scratch"
            );
        }
    }
    // Twin half: the probe CAN see a carve — the same op kept as a mask
    // carves a slot out of the same kind of arena.
    let keep = program(
        Shape::And,
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
    );
    let mut poison = vec![u64::MAX; 4 * words_for(n)];
    drop(Scratch::over_for_program(&mut poison, &keep, n).expect("arena"));
    assert!(
        poison.iter().any(|&w| w != u64::MAX),
        "the poison probe must be able to see a carve"
    );
}

/// FAILS IF: the recogniser admits a shape it cannot fold — a derived
/// (scratch) operand, a complement, a non-scalar terminal — or refuses the
/// shapes it can. The admit half keeps this from passing vacuously.
#[test]
fn the_recogniser_admits_exactly_resident_single_op_count_and_any() {
    assert!(count_p(Shape::And).fused_ternlog().is_some());
    assert!(any_p(Shape::Ternlog(0x96)).fused_ternlog().is_some());

    let scratch_operand = Program::new(
        vec![
            MaskOp::And {
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                dst: 0,
            },
            MaskOp::Or {
                a: Operand::Scratch(0),
                b: Operand::Plane(2),
                dst: 1,
            },
        ],
        Terminal::Count {
            mask: Operand::Scratch(1),
        },
    );
    assert!(scratch_operand.fused_ternlog().is_none(), "derived operand");
    assert!(scratch_operand.requires_scratch());

    let not = Program::new(
        vec![MaskOp::Not {
            a: Operand::Plane(0),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    assert!(
        not.fused_ternlog().is_none(),
        "Not clears the tail: a different shape"
    );

    for t in [
        Terminal::Keep {
            mask: Operand::Scratch(0),
        },
        Terminal::All {
            mask: Operand::Scratch(0),
        },
    ] {
        let p = program(Shape::Or, t);
        assert!(p.fused_ternlog().is_none(), "only Count/Any fold");
        assert!(p.requires_scratch());
    }
}

/// A fusable shape whose slot sits past the fused path's validation bitmap
/// must still RUN — on the tiled path — never be rejected as a read before a
/// write. Both folds are covered: the ternlog fold and the #1268 range fold
/// share the same no-scratch validation, and a slot the bitmap cannot mark
/// once made `Count(Scratch(64))` fail with `ScratchReadBeforeWrite`.
#[test]
fn a_fusable_shape_on_a_high_slot_still_executes() {
    use lance_graph_mask_risc::reference::reference_execute;
    use lance_graph_mask_risc::Pred;
    let n = 1000usize;
    let words = n.div_ceil(64);
    let mut seed = 0xB16_u64;
    let mut mk = || -> Vec<u64> {
        (0..words)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                seed
            })
            .collect()
    };
    let (mut pa, mut pb) = (mk(), mk());
    let tail = !0u64 >> (64 * words - n);
    pa[words - 1] &= tail;
    pb[words - 1] &= tail;
    let masks: [&[u64]; 2] = [&pa, &pb];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    for dst in [0u16, 31, 32, 63, 64, 65, 1000] {
        let shapes = [
            MaskOp::And {
                a: Operand::Plane(0),
                b: Operand::Plane(1),
                dst,
            },
            MaskOp::Pred {
                pred: Pred::Range { lo: 3, hi: 777 },
                under: Some(Operand::Plane(0)),
                dst,
            },
        ];
        for op in shapes {
            for terminal in [
                Terminal::Count {
                    mask: Operand::Scratch(dst),
                },
                Terminal::Any {
                    mask: Operand::Scratch(dst),
                },
            ] {
                let p = Program::new(vec![op], terminal);
                let want = reference_execute(&p, &planes, None).expect("oracle");
                let mut s = Scratch::for_program(&p, n).expect("scratch");
                let got = execute_extent(&p, &planes, &Foreign::NONE, &mut s, Out::None, 0..n)
                    .unwrap_or_else(|e| panic!("dst {dst} {op:?}: {e:?}"));
                assert_eq!(got, want, "dst {dst} {op:?}");
            }
        }
    }
}
