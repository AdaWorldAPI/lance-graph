//! D-HOLO H6: closure of address restrictions as interval sets.
//!
//! Pre-registered in `.claude/plans/d-holo-address-indexed-algebra-v1.md` (H6).
//!
//! On a dense tile the row is the Morton code, so a prefix is one row
//! interval and a box [x0, x1) × [y0, y1) is a union of aligned quadtree
//! intervals. Formulas over boxes and prefixes are evaluated three ways:
//!
//! - S: symbolically, on interval endpoints, never touching a row.
//! - M: in mask-risc, each leaf as `Range` ops ORed into a slot, then the
//!   formula's `And` / `Or` / `Not`, then `Count`.
//! - V: in mask-risc, x and y as value lanes with `GeI32` / `LtI32` per box
//!   bound: no intervals at all.
//!
//! `cargo run --release -p lance-graph-mask-risc --example holo_interval_probe`
//! `cargo test -p lance-graph-mask-risc --example holo_interval_probe`

use std::hint::black_box;
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, Scratch};
use lance_graph_mask_risc::{
    CompiledProgram, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Terminal, Value,
};

const N: u32 = 65_536;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "v4 (avx512f)"
    } else if cfg!(target_feature = "avx2") {
        "v3 (avx2): timings not reportable"
    } else {
        "below v3: timings not reportable"
    }
}

fn gather(v: u32) -> u32 {
    let mut x = v & 0x5555;
    x = (x | (x >> 1)) & 0x3333;
    x = (x | (x >> 2)) & 0x0F0F;
    (x | (x >> 4)) & 0x00FF
}

/// A sorted set of disjoint, non-adjacent half-open intervals.
type Set = Vec<(u32, u32)>;

fn push(out: &mut Set, lo: u32, hi: u32) {
    if lo >= hi {
        return;
    }
    if let Some(last) = out.last_mut() {
        if last.1 >= lo {
            last.1 = last.1.max(hi);
            return;
        }
    }
    out.push((lo, hi));
}

#[derive(Clone, Copy, Debug)]
struct Box2 {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

/// The box as maximal aligned quadtree intervals of Morton codes.
fn box_set(b: Box2) -> Set {
    fn rec(b: Box2, x: u32, y: u32, side: u32, base: u32, out: &mut Set) {
        let (bx1, by1) = (x + side, y + side);
        if bx1 <= b.x0 || x >= b.x1 || by1 <= b.y0 || y >= b.y1 {
            return;
        }
        if x >= b.x0 && bx1 <= b.x1 && y >= b.y0 && by1 <= b.y1 {
            push(out, base, base + side * side);
            return;
        }
        let h = side / 2;
        let q = h * h;
        // Child order by Morton digit: x low bit first, then y.
        rec(b, x, y, h, base, out);
        rec(b, x + h, y, h, base + q, out);
        rec(b, x, y + h, h, base + 2 * q, out);
        rec(b, x + h, y + h, h, base + 3 * q, out);
    }
    let mut out = Vec::new();
    rec(b, 0, 0, 256, 0, &mut out);
    out
}

fn prefix_set(p: u32, pv: u32) -> Set {
    let span = 1u32 << (4 * (4 - p));
    vec![(pv * span, (pv + 1) * span)]
}

fn union(a: &Set, b: &Set) -> Set {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let take_a = j >= b.len() || (i < a.len() && a[i].0 <= b[j].0);
        let (lo, hi) = if take_a { a[i] } else { b[j] };
        if take_a {
            i += 1;
        } else {
            j += 1;
        }
        push(&mut out, lo, hi);
    }
    out
}

fn intersect(a: &Set, b: &Set) -> Set {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let lo = a[i].0.max(b[j].0);
        let hi = a[i].1.min(b[j].1);
        push(&mut out, lo, hi);
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn complement(a: &Set) -> Set {
    let mut out = Vec::with_capacity(a.len() + 1);
    let mut at = 0;
    for &(lo, hi) in a {
        push(&mut out, at, lo);
        at = hi;
    }
    push(&mut out, at, N);
    out
}

fn measure(a: &Set) -> u64 {
    a.iter().map(|&(lo, hi)| u64::from(hi - lo)).sum()
}

fn well_formed(a: &Set) -> bool {
    a.iter().all(|&(lo, hi)| lo < hi && hi <= N) && a.windows(2).all(|w| w[0].1 < w[1].0)
}

#[derive(Clone, Copy)]
enum F {
    /// B1 ∩ P
    F1,
    /// B1 ∪ B2
    F2,
    /// ¬B1 ∩ P
    F3,
    /// (B1 ∪ B2) ∩ ¬B3
    F4,
}

struct Case {
    b: [Box2; 3],
    p: (u32, u32),
}

/// The leaves of a case as interval sets: three boxes, then the prefix.
fn leaves(c: &Case) -> [Set; 4] {
    [
        box_set(c.b[0]),
        box_set(c.b[1]),
        box_set(c.b[2]),
        prefix_set(c.p.0, c.p.1),
    ]
}

/// Evaluate a formula over already-decomposed leaves.
fn eval(f: F, l: &[Set; 4]) -> Set {
    match f {
        F::F1 => intersect(&l[0], &l[3]),
        F::F2 => union(&l[0], &l[1]),
        F::F3 => intersect(&complement(&l[0]), &l[3]),
        F::F4 => intersect(&union(&l[0], &l[1]), &complement(&l[2])),
    }
}

fn symbolic(f: F, c: &Case) -> Set {
    eval(f, &leaves(c))
}

fn in_box(b: Box2, r: u32) -> bool {
    let (x, y) = (gather(r), gather(r >> 1));
    x >= b.x0 && x < b.x1 && y >= b.y0 && y < b.y1
}

fn oracle(f: F, c: &Case) -> u64 {
    let pset = prefix_set(c.p.0, c.p.1)[0];
    let in_p = |r: u32| r >= pset.0 && r < pset.1;
    (0..N)
        .filter(|&r| match f {
            F::F1 => in_box(c.b[0], r) && in_p(r),
            F::F2 => in_box(c.b[0], r) || in_box(c.b[1], r),
            F::F3 => !in_box(c.b[0], r) && in_p(r),
            F::F4 => (in_box(c.b[0], r) || in_box(c.b[1], r)) && !in_box(c.b[2], r),
        })
        .count() as u64
}

// ---------- M: Range ops in mask-risc ----------

/// Emit `set` into slot `dst`, using `tmp` for each further interval.
fn emit_set(ops: &mut Vec<MaskOp>, set: &Set, dst: u16, tmp: u16) {
    for (i, &(lo, hi)) in set.iter().enumerate() {
        let to = if i == 0 { dst } else { tmp };
        ops.push(MaskOp::Pred {
            pred: Pred::Range { lo, hi },
            under: None,
            dst: to,
        });
        if i > 0 {
            ops.push(MaskOp::Or {
                a: Operand::Scratch(dst),
                b: Operand::Scratch(tmp),
                dst,
            });
        }
    }
    if set.is_empty() {
        ops.push(MaskOp::Pred {
            pred: Pred::Range { lo: 0, hi: 0 },
            under: None,
            dst,
        });
    }
}

fn mask_program(f: F, c: &Case) -> Program {
    let mut ops = Vec::new();
    let s = |i: usize| box_set(c.b[i]);
    let p = prefix_set(c.p.0, c.p.1);
    let sc = Operand::Scratch;
    match f {
        F::F1 => {
            emit_set(&mut ops, &s(0), 0, 4);
            emit_set(&mut ops, &p, 1, 4);
            ops.push(MaskOp::And {
                a: sc(0),
                b: sc(1),
                dst: 0,
            });
        }
        F::F2 => {
            emit_set(&mut ops, &s(0), 0, 4);
            emit_set(&mut ops, &s(1), 1, 4);
            ops.push(MaskOp::Or {
                a: sc(0),
                b: sc(1),
                dst: 0,
            });
        }
        F::F3 => {
            emit_set(&mut ops, &s(0), 0, 4);
            emit_set(&mut ops, &p, 1, 4);
            ops.push(MaskOp::AndNot {
                a: sc(1),
                b: sc(0),
                dst: 0,
            });
        }
        F::F4 => {
            emit_set(&mut ops, &s(0), 0, 4);
            emit_set(&mut ops, &s(1), 1, 4);
            emit_set(&mut ops, &s(2), 2, 4);
            ops.push(MaskOp::Or {
                a: sc(0),
                b: sc(1),
                dst: 0,
            });
            ops.push(MaskOp::AndNot {
                a: sc(0),
                b: sc(2),
                dst: 0,
            });
        }
    }
    Program::new(ops, Terminal::Count { mask: sc(0) })
}

// ---------- V: value sweep over x / y lanes ----------

fn emit_box(ops: &mut Vec<MaskOp>, b: Box2, dst: u16) {
    let preds = [
        Pred::GeI32 {
            lane: 0,
            t: b.x0 as i32,
        },
        Pred::LtI32 {
            lane: 0,
            t: b.x1 as i32,
        },
        Pred::GeI32 {
            lane: 1,
            t: b.y0 as i32,
        },
        Pred::LtI32 {
            lane: 1,
            t: b.y1 as i32,
        },
    ];
    for (i, pred) in preds.into_iter().enumerate() {
        ops.push(MaskOp::Pred {
            pred,
            under: if i == 0 {
                None
            } else {
                Some(Operand::Scratch(5 + (i as u16 - 1) % 2))
            },
            dst: if i == 3 { dst } else { 5 + (i as u16 % 2) },
        });
    }
}

fn value_program(f: F, c: &Case) -> Program {
    let mut ops = Vec::new();
    let sc = Operand::Scratch;
    let p = prefix_set(c.p.0, c.p.1)[0];
    let prefix = |ops: &mut Vec<MaskOp>, dst: u16| {
        ops.push(MaskOp::Pred {
            pred: Pred::Range { lo: p.0, hi: p.1 },
            under: None,
            dst,
        })
    };
    match f {
        F::F1 => {
            emit_box(&mut ops, c.b[0], 0);
            prefix(&mut ops, 1);
            ops.push(MaskOp::And {
                a: sc(0),
                b: sc(1),
                dst: 0,
            });
        }
        F::F2 => {
            emit_box(&mut ops, c.b[0], 0);
            emit_box(&mut ops, c.b[1], 1);
            ops.push(MaskOp::Or {
                a: sc(0),
                b: sc(1),
                dst: 0,
            });
        }
        F::F3 => {
            emit_box(&mut ops, c.b[0], 0);
            prefix(&mut ops, 1);
            ops.push(MaskOp::AndNot {
                a: sc(1),
                b: sc(0),
                dst: 0,
            });
        }
        F::F4 => {
            emit_box(&mut ops, c.b[0], 0);
            emit_box(&mut ops, c.b[1], 1);
            emit_box(&mut ops, c.b[2], 2);
            ops.push(MaskOp::Or {
                a: sc(0),
                b: sc(1),
                dst: 0,
            });
            ops.push(MaskOp::AndNot {
                a: sc(0),
                b: sc(2),
                dst: 0,
            });
        }
    }
    Program::new(ops, Terminal::Count { mask: sc(0) })
}

fn count(v: Value) -> u64 {
    match v {
        Value::Count(n) => n as u64,
        other => panic!("expected a count, got {other:?}"),
    }
}

struct Lanes {
    x: Vec<i32>,
    y: Vec<i32>,
}

fn lanes() -> Lanes {
    Lanes {
        x: (0..N).map(|r| gather(r) as i32).collect(),
        y: (0..N).map(|r| gather(r >> 1) as i32).collect(),
    }
}

/// A program, recognised once, with its own scratch.
struct Prepared {
    prog: CompiledProgram,
    scratch: Scratch<'static>,
}

fn prepare(p: Program) -> Prepared {
    let scratch = Scratch::for_program(&p, N as usize).expect("scratch");
    Prepared {
        prog: CompiledProgram::new(p),
        scratch,
    }
}

fn run(p: &mut Prepared, l: &Lanes) -> u64 {
    let ls = [LaneRef::I32(&l.x), LaneRef::I32(&l.y)];
    let planes = Planes {
        n_rows: N as usize,
        masks: &[],
        lanes: &ls,
    };
    let v = execute_compiled(
        &p.prog.compiled(),
        &planes,
        &Foreign::NONE,
        &mut p.scratch,
        Out::None,
        0..N as usize,
    );
    count(v.expect("run"))
}

fn time<F: FnMut()>(mut f: F) -> f64 {
    f();
    let mut v: Vec<f64> = (0..11)
        .map(|_| {
            let t = Instant::now();
            let mut n = 0u32;
            while n < 3 || t.elapsed().as_micros() < 2_000 {
                f();
                n += 1;
            }
            t.elapsed().as_nanos() as f64 / f64::from(n)
        })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[5]
}

fn bx(x0: u32, x1: u32, y0: u32, y1: u32) -> Box2 {
    Box2 { x0, x1, y0, y1 }
}

fn cases() -> Vec<(&'static str, Case)> {
    let sq = |s: u32| bx(0, s, 0, s);
    vec![
        (
            "aligned 16",
            Case {
                b: [sq(16), bx(16, 32, 0, 16), bx(0, 8, 0, 8)],
                p: (1, 0),
            },
        ),
        (
            "aligned 64",
            Case {
                b: [sq(64), bx(64, 128, 0, 64), bx(0, 32, 0, 32)],
                p: (1, 0),
            },
        ),
        (
            "aligned 256",
            Case {
                b: [sq(256), sq(256), bx(0, 128, 0, 128)],
                p: (1, 0),
            },
        ),
        (
            "unaligned",
            Case {
                b: [bx(3, 61, 5, 59), bx(40, 99, 30, 77), bx(10, 20, 10, 20)],
                p: (1, 0),
            },
        ),
        (
            "thin 256x1",
            Case {
                b: [bx(0, 256, 7, 8), bx(0, 256, 9, 10), bx(0, 128, 7, 8)],
                p: (2, 1),
            },
        ),
        (
            "thin 1x256",
            Case {
                b: [bx(7, 8, 0, 256), bx(9, 10, 0, 256), bx(7, 8, 0, 128)],
                p: (2, 1),
            },
        ),
    ]
}

fn main() {
    println!("tier: {}", tier());
    let l = lanes();
    println!(
        "shape\tformula\tintervals_in\tintervals_out\tS_eval_ns\tS_total_ns\tM_ns\tV_ns\tM_ops\tanswer"
    );
    for (name, c) in cases() {
        let lv = leaves(&c);
        let n_in: usize = lv.iter().map(Vec::len).sum();
        for (fname, f) in [("F1", F::F1), ("F2", F::F2), ("F3", F::F3), ("F4", F::F4)] {
            let sym = symbolic(f, &c);
            let want = oracle(f, &c);
            assert!(
                well_formed(&sym),
                "{name} {fname} left the interval-set form"
            );
            assert_eq!(measure(&sym), want, "{name} {fname} S");
            let mprog = mask_program(f, &c);
            let m_ops = mprog.ops.len();
            let mut mp = prepare(mprog);
            let mut vp = prepare(value_program(f, &c));
            assert_eq!(run(&mut mp, &l), want, "{name} {fname} M");
            assert_eq!(run(&mut vp, &l), want, "{name} {fname} V");
            let se = time(|| {
                black_box(measure(&eval(f, black_box(&lv))));
            });
            let st = time(|| {
                black_box(measure(&symbolic(f, black_box(&c))));
            });
            let mn = time(|| {
                black_box(run(&mut mp, &l));
            });
            let vn = time(|| {
                black_box(run(&mut vp, &l));
            });
            println!(
                "{name}\t{fname}\t{n_in}\t{}\t{se:.0}\t{st:.0}\t{mn:.0}\t{vn:.0}\t{m_ops}\t{want}",
                sym.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spread(v: u32) -> u32 {
        let mut x = v & 0xFF;
        x = (x | (x << 4)) & 0x0F0F;
        x = (x | (x << 2)) & 0x3333;
        (x | (x << 1)) & 0x5555
    }

    /// FAILS IF: the x / y lanes are not the Morton decoding of the row, so
    /// the V arm and the oracle would describe a different layout than S.
    #[test]
    fn gather_inverts_morton() {
        for x in 0..256 {
            for y in 0..256 {
                let r = spread(x) | (spread(y) << 1);
                assert_eq!((gather(r), gather(r >> 1)), (x, y));
            }
        }
    }

    /// FAILS IF: a box's interval set does not contain exactly the box's
    /// cells, or is not well formed.
    #[test]
    fn box_sets_are_exact() {
        for b in [
            bx(0, 16, 0, 16),
            bx(3, 61, 5, 59),
            bx(0, 256, 7, 8),
            bx(7, 8, 0, 256),
            bx(255, 256, 255, 256),
        ] {
            let set = box_set(b);
            assert!(well_formed(&set));
            let cells = (0..N).filter(|&r| in_box(b, r)).count() as u64;
            assert_eq!(measure(&set), cells);
            for &(lo, hi) in &set {
                assert!((lo..hi).all(|r| in_box(b, r)));
            }
        }
        assert_eq!(box_set(bx(0, 16, 0, 16)).len(), 1);
        assert!(
            box_set(bx(7, 8, 0, 256)).len() >= 128,
            "thin boxes fragment"
        );
    }

    /// FAILS IF: any formula, in any arm, differs from the row oracle, or the
    /// symbolic result leaves the interval-set form.
    #[test]
    fn every_formula_matches_the_oracle_in_every_arm() {
        let l = lanes();
        for (_, c) in cases() {
            for f in [F::F1, F::F2, F::F3, F::F4] {
                let sym = symbolic(f, &c);
                let want = oracle(f, &c);
                assert!(well_formed(&sym));
                assert_eq!(measure(&sym), want);
                assert_eq!(run(&mut prepare(mask_program(f, &c)), &l), want);
                assert_eq!(run(&mut prepare(value_program(f, &c)), &l), want);
            }
        }
    }

    /// FAILS IF: the fixture is degenerate: some formula answers 0 or N
    /// everywhere, so complement and intersection would not be exercised.
    #[test]
    fn formulas_discriminate() {
        for (name, c) in cases() {
            for f in [F::F1, F::F3, F::F4] {
                let n = oracle(f, &c);
                assert!(n < u64::from(N), "{name}: a formula covers the whole tile");
            }
            assert!(oracle(F::F3, &c) > 0 || oracle(F::F1, &c) > 0, "{name}");
        }
    }
}
