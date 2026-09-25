//! Mississippi Queen scheduling: can op-by-op execution get closer to the
//! whole-stack ceiling if intermediates only exist in a small window that
//! moves along the rows — laid down ahead of the frontier, picked up behind
//! it — instead of in a 256-word scratch tile?
//!
//! Chains over 4–5 planes (no single ternlog exists for them) plus the two
//! 3-plane chains for reference, each run as:
//!
//! - `tiled`: `execute_extent` today — one facade call per op per 256-word
//!   tile; every intermediate is written to scratch (2 KB per slot, in L1).
//! - `win<K>`: a register-window interpreter over `as_chunks::<8>` of the
//!   planes. A window is `K` × `U64x8` (8·K words); for each window the op
//!   list is interpreted once, every intermediate living in a `[[U64x8; K];
//!   SLOTS]` register file that is dropped when the window moves on. The
//!   interpretation cost (one `match` per op per window) is paid per `8·K`
//!   words instead of per 256.
//! - `inlined`: the literal-Rust ceiling from `llvm_fold_probe`.
//!
//! Every arm is checked against the tiled result (Count) and word-for-word
//! (Keep). **Lab only**: all SIMD in the window arm comes from
//! `ndarray::simd::U64x8`; nothing here ships.
//!
//! `array_windows` is deliberately NOT an arm: it yields OVERLAPPING windows
//! (stride 1), so for an elementwise Boolean op it recomputes 7 of every 8
//! words — the right tool only for an op that reads a neighbour word (a
//! run carry, a cross-word shift), which no op here does.
//!
//! `RUSTFLAGS="-C target-cpu=native" cargo run --release -p lance-graph-mask-risc --example window_sched_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{ternlog_dispatch, ternlog_popcount_dispatch};
use lance_graph_mask_risc::{Foreign, MaskOp, Operand, Out, Planes, Program, Terminal, Value};
use ndarray::simd::U64x8;

const SLOTS: usize = 8;

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

/// Run `ops` over one window of `K` vectors starting at chunk `j`, returning
/// the result slot's registers. The register file is local: it exists for
/// this window only.
#[inline(always)]
fn window<const K: usize>(ops: &[MaskOp], res: u16, pc: &[&[[u64; 8]]], j: usize) -> [U64x8; K] {
    let mut regs = [[U64x8::splat(0); K]; SLOTS];
    let load = |o: Operand, regs: &[[U64x8; K]; SLOTS]| -> [U64x8; K] {
        match o {
            Operand::Plane(p) => {
                let c = &pc[usize::from(p)][j..j + K];
                core::array::from_fn(|k| U64x8::from_array(c[k]))
            }
            Operand::Scratch(s) => regs[usize::from(s)],
        }
    };
    for op in ops {
        let (v, dst) = match *op {
            MaskOp::And { a, b, dst } => {
                let (x, y) = (load(a, &regs), load(b, &regs));
                (core::array::from_fn(|k| x[k] & y[k]), dst)
            }
            MaskOp::Or { a, b, dst } => {
                let (x, y) = (load(a, &regs), load(b, &regs));
                (core::array::from_fn(|k| x[k] | y[k]), dst)
            }
            MaskOp::Xor { a, b, dst } => {
                let (x, y) = (load(a, &regs), load(b, &regs));
                (core::array::from_fn(|k| x[k] ^ y[k]), dst)
            }
            MaskOp::AndNot { a, b, dst } => {
                let (x, y) = (load(a, &regs), load(b, &regs));
                (core::array::from_fn(|k| x[k] & !y[k]), dst)
            }
            MaskOp::Not { a, dst } => {
                let x = load(a, &regs);
                (core::array::from_fn(|k| !x[k]), dst)
            }
            _ => unreachable!("lab probe: Boolean ops only"),
        };
        regs[usize::from(dst)] = v;
    }
    regs[usize::from(res)]
}

#[inline(never)]
fn win_count<const K: usize>(ops: &[MaskOp], res: u16, pc: &[&[[u64; 8]]]) -> usize {
    let chunks = pc[0].len();
    let mut n = 0u64;
    for j in (0..chunks).step_by(K) {
        for v in window::<K>(ops, res, pc, j) {
            n += v
                .to_array()
                .iter()
                .map(|w| u64::from(w.count_ones()))
                .sum::<u64>();
        }
    }
    n as usize
}

#[inline(never)]
fn win_keep<const K: usize>(ops: &[MaskOp], res: u16, pc: &[&[[u64; 8]]], out: &mut [[u64; 8]]) {
    for j in (0..out.len()).step_by(K) {
        let r = window::<K>(ops, res, pc, j);
        for (k, v) in r.iter().enumerate() {
            out[j + k] = v.to_array();
        }
    }
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

/// The inner ternlog: `(imm, a, b, c)` plane indices.
type Inner = (u8, usize, usize, usize);
/// The outer ternlog over the inner result: `(imm, d, e)` plane indices.
type Outer = (u8, usize, usize);

/// Two-level ternlog fold with EXISTING kernels, chunk by chunk: `t =
/// tern1(a, b, c)` into a `T`-word window, then `popcount(tern2(t, d, e))`.
/// One slot, two facade calls per window, whatever the op count.
#[inline(never)]
fn tern2_count<const T: usize>(m: &[&[u64]; 5], (i1, a, b, c): Inner, (i2, d, e): Outer) -> usize {
    let mut t = [0u64; T];
    let mut n = 0u64;
    for w in (0..m[0].len()).step_by(T) {
        let r = w..w + T;
        ternlog_dispatch(
            i1,
            &m[a][r.clone()],
            &m[b][r.clone()],
            &m[c][r.clone()],
            &mut t,
        );
        n += ternlog_popcount_dispatch(i2, &t, &m[d][r.clone()], &m[e][r]);
    }
    n as usize
}

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

fn main() {
    let n = 1usize << 20;
    let words = n / 64;
    assert_eq!(words % 64, 0, "lab probe: no tail handling");
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
    let pc: Vec<&[[u64; 8]]> = masks.iter().map(|m| m.as_chunks::<8>().0).collect();
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let (a, b, c, d, e) = (
        Operand::Plane(0),
        Operand::Plane(1),
        Operand::Plane(2),
        Operand::Plane(3),
        Operand::Plane(4),
    );
    let s = Operand::Scratch;
    type Word = fn(u64, u64, u64, u64, u64) -> u64;
    let chains: Vec<(&str, Vec<MaskOp>, u16, Word)> = vec![
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
        ),
    ];
    let reps = 41;
    println!(
        "{:>22} | {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} {:>8}",
        "chain",
        "tiled",
        "win1",
        "win2",
        "win4",
        "win8",
        "inlined",
        "k_tiled",
        "k_win2",
        "k_win4",
        "k_win8"
    );
    for (name, ops, res, word) in chains {
        // Force the tiled path by moving the result slot past the fold's
        // cap (no extra op, so no extra pass).
        let cap = lance_graph_mask_risc::FUSED_SLOT_CAP as u16;
        let tops = retarget(&ops, res, cap);
        let tp = Program::new(tops.clone(), Terminal::Count { mask: s(cap) });
        let mut ts = Scratch::for_program(&tp, n).expect("scratch");
        let (tns, tv) = median(reps, || {
            execute_extent(&tp, &planes, &Foreign::NONE, &mut ts, Out::None, 0..n).expect("tiled")
        });
        let Value::Count(want) = tv else {
            panic!("count")
        };
        let (w1, v1) = median(reps, || win_count::<1>(&ops, res, &pc));
        let (w2, v2) = median(reps, || win_count::<2>(&ops, res, &pc));
        let (w4, v4) = median(reps, || win_count::<4>(&ops, res, &pc));
        let (w8, v8) = median(reps, || win_count::<8>(&ops, res, &pc));
        // Literal closures per chain: a `fn` pointer here would be an
        // indirect call per word (the trap `llvm_fold_probe` records).
        let (il, vi) = median(reps, || match name {
            "(a&b)|!c" => inlined_count(&masks, |a, b, c, _, _| (a & b) | !c),
            "4p:(a&b)|(c&d)" => inlined_count(&masks, |a, b, c, d, _| (a & b) | (c & d)),
            _ => inlined_count(&masks, |a, b, c, d, e| ((a | b) & c) ^ (d & !e)),
        });
        assert_eq!(
            vi,
            inlined_count(&masks, word),
            "{name}: literal vs fn table"
        );
        // leaf tables: first operand 0xF0, second 0xCC, third 0xAA.
        let (l1, l2): (Inner, Outer) = match name {
            "(a&b)|!c" => ((0xF0 & 0xCC, 0, 1, 0), (0xF0 | !0xAAu8, 0, 2)),
            "4p:(a&b)|(c&d)" => ((0xF0 & 0xCC, 0, 1, 0), (0xF0 | (0xCC & 0xAA), 2, 3)),
            _ => (
                ((0xF0 | 0xCC) & 0xAA, 0, 1, 2),
                (0xF0 ^ (0xCC & !0xAAu8), 3, 4),
            ),
        };
        let (t8, x8) = median(reps, || tern2_count::<8>(&masks, l1, l2));
        let (t64, x64) = median(reps, || tern2_count::<64>(&masks, l1, l2));
        let (t256, x256) = median(reps, || tern2_count::<256>(&masks, l1, l2));
        for v in [x8, x64, x256] {
            assert_eq!(v, want, "{name}: tern2 disagrees");
        }
        println!("{name:>22}   tern2@8 {t8:.0}  tern2@64 {t64:.0}  tern2@256 {t256:.0}");
        for v in [v1, v2, v4, v8, vi as usize] {
            assert_eq!(v, want, "{name}: arm disagrees with tiled");
        }
        // Keep
        let kp = Program::new(tops, Terminal::Keep { mask: s(cap) });
        let mut ks = Scratch::for_program(&kp, n).expect("scratch");
        let mut kout = vec![0u64; words];
        let (kt, _) = median(reps, || {
            execute_extent(
                &kp,
                &planes,
                &Foreign::NONE,
                &mut ks,
                Out::Mask(&mut kout),
                0..n,
            )
            .expect("keep")
        });
        let mut wout = vec![[0u64; 8]; words / 8];
        let mut kw = [0f64; 3];
        for (i, f) in [
            win_keep::<2> as fn(&[MaskOp], u16, &[&[[u64; 8]]], &mut [[u64; 8]]),
            win_keep::<4>,
            win_keep::<8>,
        ]
        .iter()
        .enumerate()
        {
            wout.iter_mut().for_each(|c| *c = [0; 8]);
            kw[i] = median(reps, || f(&ops, res, &pc, &mut wout)).0;
            assert_eq!(
                wout.as_flattened(),
                &kout[..],
                "{name}: window keep disagrees"
            );
        }
        println!(
            "{name:>22} | {tns:>8.0} {w1:>8.0} {w2:>8.0} {w4:>8.0} {w8:>8.0} {il:>8.0} | {kt:>8.0} {:>8.0} {:>8.0} {:>8.0}",
            kw[0], kw[1], kw[2]
        );
    }
    println!("(n = {n}, {words} words, median ns over {reps} runs)");
}
