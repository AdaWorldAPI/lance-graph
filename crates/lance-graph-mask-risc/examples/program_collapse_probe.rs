//! Multi-op Boolean chain → Count/Any at N = 1M rows: collapsed fold vs tiled.
//!
//! The same raw op sequence, three physical endings:
//!
//! - FOLD: `Program::fused_ternlog` interprets the chain symbolically and
//!   lowers it onto ONE `mask_ternlog_{popcount,any}` pass; no slot is carved
//!   and no derived word is written.
//! - TILED: the identical chain with its terminal routed through slot
//!   [`FUSED_SLOT_CAP`] — the recogniser refuses (a bound of the recogniser,
//!   never of the semantics), so every op writes its tile slot and the
//!   terminal reads the last one back. Same answer, today's scratch path.
//! - KEEP: the chain with a `Keep` terminal into an `Out::Mask`, then
//!   `popcount_batch_u64` / `mask_any` over the kept words.
//!
//! Reported per chain × extent: median ns per arm and the derived words the
//! tiled arm writes (`ops × touched words`; the fold writes 0). A four-plane
//! chain is measured on the tiled path only: it is outside the algebra (one
//! ternlog addresses three inputs), and its row records where the collapse
//! stops. Every result is checked against a bit-serial scalar oracle.
//!
//! `cargo run --release -p lance-graph-mask-risc --example program_collapse_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{
    touched_words, Foreign, MaskOp, Operand, Out, Planes, Program, Terminal, Value, FUSED_SLOT_CAP,
};
use ndarray::simd::{mask_any, popcount_batch_u64};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn median<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut ts = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        last = Some(std::hint::black_box(f()));
        ts.push(t.elapsed().as_nanos() as f64);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    (ts[reps / 2], last.expect("reps > 0"))
}

/// Re-address the op writing `from` (and every later read of it) to `to`.
fn retarget(ops: &[MaskOp], from: u16, to: u16) -> Vec<MaskOp> {
    let r = |o: Operand| match o {
        Operand::Scratch(s) if s == from => Operand::Scratch(to),
        o => o,
    };
    let d = |x: u16| if x == from { to } else { x };
    ops.iter()
        .map(|op| match *op {
            MaskOp::And { a, b, dst } => MaskOp::And {
                a: r(a),
                b: r(b),
                dst: d(dst),
            },
            MaskOp::Or { a, b, dst } => MaskOp::Or {
                a: r(a),
                b: r(b),
                dst: d(dst),
            },
            MaskOp::Xor { a, b, dst } => MaskOp::Xor {
                a: r(a),
                b: r(b),
                dst: d(dst),
            },
            MaskOp::AndNot { a, b, dst } => MaskOp::AndNot {
                a: r(a),
                b: r(b),
                dst: d(dst),
            },
            MaskOp::Not { a, dst } => MaskOp::Not {
                a: r(a),
                dst: d(dst),
            },
            MaskOp::Ternlog { imm, a, b, c, dst } => MaskOp::Ternlog {
                imm,
                a: r(a),
                b: r(b),
                c: r(c),
                dst: d(dst),
            },
            other => other,
        })
        .collect()
}

fn main() {
    let n = 1usize << 20;
    let words = n.div_ceil(64);
    let mut seed = 0xC0_11A5_u64;
    let mut mk = |modulus: u64| {
        let mut w = vec![0u64; words];
        for r in 0..n {
            if lcg(&mut seed).is_multiple_of(modulus) {
                w[r / 64] |= 1 << (r % 64);
            }
        }
        w
    };
    let (pa, pb, pc) = (mk(2), mk(3), mk(5));
    // d = !a (n is a multiple of 64, so no tail bits): `a & d` is empty BY
    // DATA over a non-constant table, so Any has no first-block exit and the
    // fold must genuinely scan. (A chain that is empty by ALGEBRA, e.g.
    // `a & b & !a`, lowers to the constant table 0x00, and the monomorphized
    // fold never reads the planes at all.)
    let pd: Vec<u64> = pa.iter().map(|w| !w).collect();
    let masks: [&[u64]; 4] = [&pa, &pb, &pc, &pd];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let bit = |p: &[u64], r: usize| p[r / 64] >> (r % 64) & 1 == 1;
    let (a, b, c, d) = (
        Operand::Plane(0),
        Operand::Plane(1),
        Operand::Plane(2),
        Operand::Plane(3),
    );
    let s = Operand::Scratch;
    type Oracle = fn(bool, bool, bool, bool) -> bool;
    let chains: [(&str, Vec<MaskOp>, u16, Oracle); 5] = [
        (
            "(a&b)|!c",
            vec![
                MaskOp::And { a, b, dst: 0 },
                MaskOp::Not { a: c, dst: 1 },
                MaskOp::Or {
                    a: s(0),
                    b: s(1),
                    dst: 2,
                },
            ],
            2,
            |a, b, c, _| (a && b) || !c,
        ),
        (
            "((a^b)&!c)|(a&c)",
            vec![
                MaskOp::Xor { a, b, dst: 0 },
                MaskOp::AndNot {
                    a: s(0),
                    b: c,
                    dst: 1,
                },
                MaskOp::And { a, b: c, dst: 2 },
                MaskOp::Or {
                    a: s(1),
                    b: s(2),
                    dst: 0,
                },
            ],
            0,
            |a, b, c, _| ((a ^ b) && !c) || (a && c),
        ),
        (
            "maj(a,b,c)^a",
            vec![
                MaskOp::Ternlog {
                    imm: 0xE8,
                    a,
                    b,
                    c,
                    dst: 0,
                },
                MaskOp::Xor {
                    a: s(0),
                    b: a,
                    dst: 1,
                },
            ],
            1,
            |a, b, c, _| ((a as u8 + b as u8 + c as u8) >= 2) ^ a,
        ),
        (
            // Empty by data: Any cannot exit early on either path, so this
            // row measures the scan itself rather than a first-block exit.
            "empty:a&d",
            vec![MaskOp::And { a, b: d, dst: 0 }],
            0,
            |a, _, _, d| a && d,
        ),
        (
            "4p:(a&b)|(c&d)",
            vec![
                MaskOp::And { a, b, dst: 0 },
                MaskOp::And { a: c, b: d, dst: 1 },
                MaskOp::Or {
                    a: s(0),
                    b: s(1),
                    dst: 2,
                },
            ],
            2,
            |a, b, c, d| (a && b) || (c && d),
        ),
    ];
    let mid = n / 2 + 17;
    let extents = [("1%", mid, mid + n / 100), ("whole", 0, n)];
    let cap = FUSED_SLOT_CAP as u16;
    println!(
        "{:>17} {:>5} {:>5} {:>3} {:>10} {:>10} {:>10} {:>6} {:>6} {:>9}",
        "chain", "ext", "term", "ops", "fold_ns", "tiled_ns", "keep_ns", "t/f", "k/f", "tiled_wr"
    );
    for (name, ops, last, oracle) in chains {
        let term = |mask| (Terminal::Count { mask }, Terminal::Any { mask });
        let (tc, ta) = term(s(last));
        let fold = [Program::new(ops.clone(), tc), Program::new(ops.clone(), ta)];
        let tiled_ops = retarget(&ops, last, cap);
        let (ttc, tta) = term(s(cap));
        let tiled = [
            Program::new(tiled_ops.clone(), ttc),
            Program::new(tiled_ops, tta),
        ];
        let keep = Program::new(ops.clone(), Terminal::Keep { mask: s(last) });
        let collapses = fold[0].fused_ternlog().is_some();
        assert_eq!(collapses, fold[1].fused_ternlog().is_some());
        assert_eq!(
            collapses,
            !name.starts_with("4p"),
            "{name}: collapse boundary"
        );
        assert!(tiled.iter().all(|p| p.fused_ternlog().is_none()));
        let mut ts = Scratch::for_program(&tiled[0], n).expect("scratch");
        let mut ks = Scratch::for_program(&keep, n).expect("scratch");
        let mut out = vec![0u64; words];
        for (ename, lo, hi) in extents {
            let want = (lo..hi)
                .filter(|&r| oracle(bit(&pa, r), bit(&pb, r), bit(&pc, r), bit(&pd, r)))
                .count();
            let reps = if hi - lo > 100_000 { 41 } else { 1001 };
            let span = touched_words(lo as u32, hi as u32);
            for (k, tname) in ["Count", "Any"].into_iter().enumerate() {
                let expect = if k == 0 {
                    Value::Count(want)
                } else {
                    Value::Bool(want > 0)
                };
                let (fns, fv) = if collapses {
                    median(reps, || {
                        let mut s0 = Scratch::new(0, 0);
                        execute_extent(
                            &fold[k],
                            &planes,
                            &Foreign::NONE,
                            &mut s0,
                            Out::None,
                            lo..hi,
                        )
                        .expect("fold")
                    })
                } else {
                    (f64::NAN, expect)
                };
                let (tns, tv) = median(reps, || {
                    execute_extent(
                        &tiled[k],
                        &planes,
                        &Foreign::NONE,
                        &mut ts,
                        Out::None,
                        lo..hi,
                    )
                    .expect("tiled")
                });
                let (kns, kv) = median(reps, || {
                    out[span.clone()].fill(0);
                    execute_extent(
                        &keep,
                        &planes,
                        &Foreign::NONE,
                        &mut ks,
                        Out::Mask(&mut out),
                        lo..hi,
                    )
                    .expect("keep");
                    if k == 0 {
                        Value::Count(popcount_batch_u64(&out[span.clone()]) as usize)
                    } else {
                        Value::Bool(mask_any(&out[span.clone()]))
                    }
                });
                assert_eq!(fv, expect, "fold {name} {ename} {tname}");
                assert_eq!(tv, expect, "tiled {name} {ename} {tname}");
                assert_eq!(kv, expect, "keep {name} {ename} {tname}");
                println!(
                    "{name:>17} {ename:>5} {tname:>5} {:>3} {fns:>10.0} {tns:>10.0} {kns:>10.0} {:>6.2} {:>6.2} {:>9}",
                    ops.len(),
                    tns / fns,
                    kns / fns,
                    ops.len() * span.len()
                );
            }
        }
    }
    println!(
        "(n = {n}, {words} population words; tiled_wr = derived words the tiled path \
         writes, ops × touched words; the fold writes 0; fold_ns NaN = not collapsible)"
    );
}
