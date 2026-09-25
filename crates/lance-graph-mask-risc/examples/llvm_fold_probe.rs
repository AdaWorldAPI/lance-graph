//! How much does compiling the WHOLE stack for a fixed chain buy?
//!
//! The same Boolean chains, five ways, each checked against a bit-serial
//! oracle:
//!
//! - `tiled`: `execute_extent` on a program the fold declines (today's
//!   fallback: one facade call per op per 256-word tile, every intermediate
//!   written to scratch).
//! - `compiled`: `execute_compiled` on the folded program (lowering recognised
//!   once; per call: validation, extent checks, the 256-arm immediate match).
//! - `const_imm`: `ndarray::simd::mask_ternlog_popcount::<IMM>` called with the
//!   immediate as a literal: what a code generator would emit for a known
//!   three-plane chain. No validation, no dispatch.
//! - `inlined`: one plain loop per chain, `f(a[i], b[i], c[i], d[i])
//!   .count_ones()` summed, with `f` written out as ordinary Rust so LLVM
//!   inlines and vectorises the whole chain. Handles any plane count.
//!   **Lab arm only**: production SIMD comes from `ndarray::simd`, never from
//!   an autovectorised loop. It is here to bound what whole-stack compilation
//!   can buy, not to ship.
//! - For `Keep`: the tiled path with an `Out::Mask`, against one inlined loop
//!   that writes each output word once.
//!
//! Build with the target's native features so LLVM may use VPTERNLOG /
//! VPOPCNTQ in the lab arm:
//! `RUSTFLAGS="-C target-cpu=native" cargo run --release -p lance-graph-mask-risc --example llvm_fold_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_compiled, execute_extent, Scratch};
use lance_graph_mask_risc::{
    Foreign, MaskOp, Operand, Out, Planes, Program, Terminal, Value, FUSED_SLOT_CAP,
};
use ndarray::simd::mask_ternlog_popcount;

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

/// Move the chain's result slot to `to`, so the recogniser declines and the
/// tiled path runs.
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
            other => other,
        })
        .collect()
}

/// The whole-stack lab arm: one loop, the chain inlined as plain Rust. The
/// five planes are re-sliced to one common length and walked by iterator so
/// LLVM can prove every access in bounds; a per-index `p[k][i]` form keeps a
/// bounds check per load and does not vectorise.
#[inline(never)]
fn inlined_count(p: &[&[u64]; 5], f: impl Fn(u64, u64, u64, u64, u64) -> u64) -> u64 {
    let n = p[0].len();
    let (a, b, c, d, e) = (&p[0][..n], &p[1][..n], &p[2][..n], &p[3][..n], &p[4][..n]);
    a.iter()
        .zip(b)
        .zip(c)
        .zip(d)
        .zip(e)
        .map(|((((&a, &b), &c), &d), &e)| u64::from(f(a, b, c, d, e).count_ones()))
        .sum()
}

#[inline(never)]
fn inlined_keep(p: &[&[u64]; 5], out: &mut [u64], f: impl Fn(u64, u64, u64, u64, u64) -> u64) {
    let n = out.len();
    let (a, b, c, d, e) = (&p[0][..n], &p[1][..n], &p[2][..n], &p[3][..n], &p[4][..n]);
    for (i, o) in out.iter_mut().enumerate() {
        *o = f(a[i], b[i], c[i], d[i], e[i]);
    }
}

fn main() {
    let n = 1usize << 20;
    let words = n / 64;
    let mut seed = 0x11_F01D_u64;
    let mut mk = |m: u64| {
        let mut w = vec![0u64; words];
        for r in 0..n {
            if lcg(&mut seed).is_multiple_of(m) {
                w[r / 64] |= 1 << (r % 64);
            }
        }
        w
    };
    let ps: Vec<Vec<u64>> = [2u64, 3, 5, 7, 11].iter().map(|&m| mk(m)).collect();
    let masks: [&[u64]; 5] = [&ps[0], &ps[1], &ps[2], &ps[3], &ps[4]];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let bit = |p: &[u64], r: usize| p[r / 64] >> (r % 64) & 1 == 1;
    let (a, b, c, d, e) = (
        Operand::Plane(0),
        Operand::Plane(1),
        Operand::Plane(2),
        Operand::Plane(3),
        Operand::Plane(4),
    );
    let s = Operand::Scratch;
    type Word = fn(u64, u64, u64, u64, u64) -> u64;
    type Bitf = fn(bool, bool, bool, bool, bool) -> bool;
    // (name, ops, result slot, word function, bit oracle, literal imm if 3-plane)
    /// One probe row: name, ops, the slot the chain leaves its result in, the
    /// literal word function, the per-bit oracle, and the collapsed immediate.
    type Chain = (&'static str, Vec<MaskOp>, u16, Word, Bitf, Option<u8>);
    let chains: Vec<Chain> = vec![
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
            |a, b, c, _, _| (a & b) | !c,
            |a, b, c, _, _| (a && b) || !c,
            Some(0xD5),
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
            |a, b, c, _, _| ((a ^ b) & !c) | (a & c),
            |a, b, c, _, _| ((a ^ b) && !c) || (a && c),
            Some(0xB4),
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
            |a, b, c, d, _| (a & b) | (c & d),
            |a, b, c, d, _| (a && b) || (c && d),
            None,
        ),
        (
            "5p:((a|b)&c)^(d&!e)",
            vec![
                MaskOp::Or { a, b, dst: 0 },
                MaskOp::And {
                    a: s(0),
                    b: c,
                    dst: 1,
                },
                MaskOp::AndNot { a: d, b: e, dst: 2 },
                MaskOp::Xor {
                    a: s(1),
                    b: s(2),
                    dst: 3,
                },
            ],
            3,
            |a, b, c, d, e| ((a | b) & c) ^ (d & !e),
            |a, b, c, d, e| ((a || b) && c) ^ (d && !e),
            None,
        ),
    ];
    let cap = FUSED_SLOT_CAP as u16;
    println!(
        "{:>22} {:>10} {:>10} {:>10} {:>10} {:>7} | {:>10} {:>10} {:>10} {:>7}",
        "chain",
        "tiled",
        "compiled",
        "const_imm",
        "inlined",
        "t/inl",
        "keep_tile",
        "keep_fused",
        "keep_inl",
        "kf/inl"
    );
    for (name, ops, last, word, oracle, imm) in chains {
        let want = (0..n)
            .filter(|&r| {
                oracle(
                    bit(masks[0], r),
                    bit(masks[1], r),
                    bit(masks[2], r),
                    bit(masks[3], r),
                    bit(masks[4], r),
                )
            })
            .count();
        let reps = 41;
        // tiled
        let tp = Program::new(retarget(&ops, last, cap), Terminal::Count { mask: s(cap) });
        assert!(tp.fused_ternlog().is_none());
        let mut ts = Scratch::for_program(&tp, n).expect("scratch");
        let (tns, tv) = median(reps, || {
            execute_extent(&tp, &planes, &Foreign::NONE, &mut ts, Out::None, 0..n).expect("tiled")
        });
        assert_eq!(tv, Value::Count(want), "{name} tiled");
        // compiled fold (3-plane chains only)
        let fp = Program::new(ops.clone(), Terminal::Count { mask: s(last) });
        let (cns, ins) = if let Some(lit) = imm {
            let f = fp.fused_ternlog().expect("folds");
            assert_eq!(
                f.imm, lit,
                "{name}: literal immediate matches the recogniser"
            );
            let compiled = fp.compile();
            let (cns, cv) = median(reps, || {
                let mut s0 = Scratch::new(0, 0);
                execute_compiled(&compiled, &planes, &Foreign::NONE, &mut s0, Out::None, 0..n)
                    .expect("compiled")
            });
            assert_eq!(cv, Value::Count(want), "{name} compiled");
            let (ins, iv) = median(reps, || match lit {
                0xD5 => mask_ternlog_popcount::<0xD5>(masks[0], masks[1], masks[2]),
                0xB4 => mask_ternlog_popcount::<0xB4>(masks[0], masks[1], masks[2]),
                _ => unreachable!(),
            });
            assert_eq!(iv as usize, want, "{name} const_imm");
            (cns, ins)
        } else {
            assert!(
                fp.fused_ternlog().is_none(),
                "{name}: >3 planes never folds"
            );
            (f64::NAN, f64::NAN)
        };
        // whole-stack inlined (lab)
        // Literal closures, not the `word` fn pointer: through a fn pointer
        // every word is an indirect call LLVM cannot inline, which measures
        // call overhead, not whole-stack compilation.
        let (lns, lv) = median(reps, || match name {
            "(a&b)|!c" => inlined_count(&masks, |a, b, c, _, _| (a & b) | !c),
            "((a^b)&!c)|(a&c)" => inlined_count(&masks, |a, b, c, _, _| ((a ^ b) & !c) | (a & c)),
            "4p:(a&b)|(c&d)" => inlined_count(&masks, |a, b, c, d, _| (a & b) | (c & d)),
            _ => inlined_count(&masks, |a, b, c, d, e| ((a | b) & c) ^ (d & !e)),
        });
        // The fn-pointer table must still agree with the literal closures.
        assert_eq!(
            inlined_count(&masks, word),
            lv,
            "{name}: fn table vs literal"
        );
        assert_eq!(lv as usize, want, "{name} inlined");
        // Keep: tiled (forced past the fold) vs fused (the recognised
        // lowering, when the chain has at most three planes) vs inlined.
        let ktp = Program::new(retarget(&ops, last, cap), Terminal::Keep { mask: s(cap) });
        assert!(ktp.fused_keep().is_none());
        let mut kts = Scratch::for_program(&ktp, n).expect("scratch");
        let mut out = vec![0u64; words];
        let (kns, _) = median(reps, || {
            execute_extent(
                &ktp,
                &planes,
                &Foreign::NONE,
                &mut kts,
                Out::Mask(&mut out),
                0..n,
            )
            .expect("keep")
        });
        let kept: usize = out.iter().map(|w| w.count_ones() as usize).sum();
        assert_eq!(kept, want, "{name} keep tiled");
        let kp = Program::new(ops.clone(), Terminal::Keep { mask: s(last) });
        let kfns = if kp.fused_keep().is_some() {
            let mut ks = Scratch::new(0, 0);
            let mut outf = vec![0u64; words];
            let (t, _) = median(reps, || {
                execute_extent(
                    &kp,
                    &planes,
                    &Foreign::NONE,
                    &mut ks,
                    Out::Mask(&mut outf),
                    0..n,
                )
                .expect("keep fused")
            });
            assert_eq!(outf, out, "{name} keep fused == tiled");
            t
        } else {
            f64::NAN
        };
        let mut out2 = vec![0u64; words];
        let (kins, _) = median(reps, || match name {
            "(a&b)|!c" => inlined_keep(&masks, &mut out2, |a, b, c, _, _| (a & b) | !c),
            "((a^b)&!c)|(a&c)" => {
                inlined_keep(&masks, &mut out2, |a, b, c, _, _| ((a ^ b) & !c) | (a & c))
            }
            "4p:(a&b)|(c&d)" => inlined_keep(&masks, &mut out2, |a, b, c, d, _| (a & b) | (c & d)),
            _ => inlined_keep(&masks, &mut out2, |a, b, c, d, e| ((a | b) & c) ^ (d & !e)),
        });
        assert_eq!(out2, out, "{name} keep inlined == tiled");
        println!(
            "{name:>22} {tns:>10.0} {cns:>10.0} {ins:>10.0} {lns:>10.0} {:>7.2} | {kns:>10.0} {kfns:>10.0} {kins:>10.0} {:>7.2}",
            tns / lns,
            kfns / kins
        );
    }
    println!(
        "(n = {n}, {words} words, median ns over 41 runs; NaN = chain has more than 3 planes, \
         so no single ternlog exists; t/inl = tiled Count over the inlined lab loop, kf/inl = fused Keep over the inlined lab loop)"
    );
}
