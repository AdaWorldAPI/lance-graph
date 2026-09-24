//! Multi-op Boolean chain → Count/Any at N = 1M rows: collapsed fold vs tiled.
//!
//! The same raw op sequence, four physical endings:
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
//! - BULK: the same op sequence evaluated ONCE per op over the extent's
//!   whole touched-word span, into preallocated whole-population scratch
//!   buffers (allocated once per chain, outside every timed closure) — the
//!   same word WRITES the tiled arm pays, but with no per-tile interpreter
//!   dispatch: one `ndarray::simd` facade call per op, full span wide,
//!   instead of one call per op per tile. Two derived columns report the
//!   MEASURED GAPS between these paths, not isolated costs:
//!   `wr_ns = bulk_ns - fold_ns` is the gap between one fused pass that
//!   reads at most three planes and writes nothing, and one pass PER OP that
//!   reads and writes whole-span buffers; it mixes write cost with the
//!   pass-count and read difference. `disp_ns = tiled_ns - bulk_ns` is the
//!   gap between the same ops run per 8-word tile and run once over the
//!   whole span; it mixes per-call overhead with batch-size effects (cache
//!   residency, loop setup). Attributing either gap to one cause would need
//!   matched-work controls this probe does not have.
//!
//! Reported per chain × extent: median ns per arm (fold/bulk/tiled/keep),
//! the derived words the tiled arm writes (`ops × touched words`; the fold
//! writes 0), and the two BULK-derived columns above. A four-plane chain is
//! measured on the bulk and tiled paths only: it is outside the fold algebra
//! (one ternlog addresses three inputs), and its row records where the
//! collapse stops. Every result is checked against a bit-serial scalar
//! oracle.
//!
//! `cargo run --release -p lance-graph-mask-risc --example program_collapse_probe`

use std::ops::Range;
use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::ternlog_dispatch::ternlog_dispatch;
use lance_graph_mask_risc::{
    touched_words, Foreign, MaskOp, Operand, Out, Planes, Program, Terminal, Value, FUSED_SLOT_CAP,
};
use ndarray::simd::{
    mask_and, mask_andnot, mask_any, mask_not, mask_or, mask_xor, popcount_batch_u64,
};

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

/// The scratch slot an op writes — every [`MaskOp`] variant has one.
fn op_dst(op: &MaskOp) -> u16 {
    match *op {
        MaskOp::Pred { dst, .. }
        | MaskOp::And { dst, .. }
        | MaskOp::Or { dst, .. }
        | MaskOp::Xor { dst, .. }
        | MaskOp::AndNot { dst, .. }
        | MaskOp::Not { dst, .. }
        | MaskOp::Ternlog { dst, .. }
        | MaskOp::Gather { dst, .. } => dst,
    }
}

/// The bits of word `w` that fall inside `[lo, hi)` (absolute rows) — the
/// BULK arm's own copy of `exec::edge_mask` (private to that module), used
/// to restrict the terminal's first and last touched word to the extent
/// exactly as `exec::run_fused`/`run_fused_ternlog` do.
fn edge_mask(w: usize, lo: usize, hi: usize) -> u64 {
    let base = w * 64;
    let from = lo.saturating_sub(base).min(64);
    let to = (hi - base).min(64);
    let upper = if to == 64 { u64::MAX } else { (1u64 << to) - 1 };
    upper & (u64::MAX << from)
}

/// Every BULK scratch buffer except the one an op is writing — this arm's
/// analogue of `exec::Slots`, over per-slot `Vec<u64>` buffers rather than a
/// flat tiled arena.
struct BulkSlots<'s> {
    /// Buffers `[0, hole)`.
    left: &'s [Vec<u64>],
    /// Buffers `(hole, bufs.len())`.
    right: &'s [Vec<u64>],
    hole: usize,
}

impl BulkSlots<'_> {
    fn get(&self, i: usize, n: usize) -> &[u64] {
        debug_assert!(
            i != self.hole,
            "bulk arm: slot {i} read while it is this op's write target"
        );
        if i < self.hole {
            &self.left[i][..n]
        } else {
            &self.right[i - self.hole - 1][..n]
        }
    }
}

/// Evaluate `ops` ONCE per op over `span` (a word range, e.g. from
/// [`touched_words`]) into `bufs`, one preallocated whole-population buffer
/// per scratch slot the chain uses — no per-tile dispatch, one
/// `ndarray::simd` facade call per op over the whole span. Leaf operands
/// (`Operand::Plane`) read `leaves[i][span]` directly, no copy. `bufs` must
/// have one entry per distinct scratch slot `ops` addresses, each at least
/// `span.len()` words long; nothing is allocated here.
///
/// No edge masking happens here: bits outside the extent `[lo, hi)` but
/// inside `span`'s first/last word are left as whatever the ops computed
/// them to be. That is sound because those positions are masked OUT by
/// [`bulk_terminal`] purely by WORD POSITION, so their VALUE never reaches
/// the reported result — the same reasoning `exec::run_fused_ternlog` relies
/// on for an odd ternlog's tail bits.
fn bulk_eval(ops: &[MaskOp], leaves: &[&[u64]; 4], bufs: &mut [Vec<u64>], span: Range<usize>) {
    let n = span.len();
    for op in ops {
        let dst = usize::from(op_dst(op));
        let (left, rest) = bufs.split_at_mut(dst);
        let (mid, right) = rest.split_at_mut(1);
        let slots = BulkSlots {
            left: &*left,
            right: &*right,
            hole: dst,
        };
        let rd = |o: Operand| -> &[u64] {
            match o {
                Operand::Plane(i) => &leaves[usize::from(i)][span.clone()],
                Operand::Scratch(i) => slots.get(usize::from(i), n),
            }
        };
        let d = &mut mid[0][..n];
        match *op {
            MaskOp::And { a, b, .. } => mask_and(rd(a), rd(b), d),
            MaskOp::Or { a, b, .. } => mask_or(rd(a), rd(b), d),
            MaskOp::Xor { a, b, .. } => mask_xor(rd(a), rd(b), d),
            MaskOp::AndNot { a, b, .. } => mask_andnot(rd(a), rd(b), d),
            // `n * 64` as the tail-clear bound disables `mask_not`'s own
            // clipping (it is a no-op exactly at a word boundary, which `n *
            // 64` always is) — any population-edge clipping is `bulk_eval`'s
            // caller's job via `bulk_terminal`, not an interior op's.
            MaskOp::Not { a, .. } => mask_not(rd(a), n * 64, d),
            MaskOp::Ternlog { imm, a, b, c, .. } => ternlog_dispatch(imm, rd(a), rd(b), rd(c), d),
            other => panic!("bulk arm: unsupported op {other:?}"),
        }
    }
}

/// Fold a BULK terminal slot (`buf`, the `span.len()`-word result of
/// [`bulk_eval`], `span` starting at absolute word `w0`) into `Count` or
/// `Any` over `[lo, hi)`, restricting the first and last word with
/// [`edge_mask`] exactly as `exec::run_fused`/`run_fused_ternlog` do — the
/// interior words are read as they are, since [`touched_words`] guarantees
/// every word strictly between the first and last is wholly inside the
/// extent.
fn bulk_terminal(buf: &[u64], w0: usize, lo: usize, hi: usize, want_count: bool) -> Value {
    let n = buf.len();
    if n == 0 {
        return if want_count {
            Value::Count(0)
        } else {
            Value::Bool(false)
        };
    }
    if n == 1 {
        let w = buf[0] & edge_mask(w0, lo, hi);
        return if want_count {
            Value::Count(w.count_ones() as usize)
        } else {
            Value::Bool(w != 0)
        };
    }
    let head = [buf[0] & edge_mask(w0, lo, hi)];
    let tail = [buf[n - 1] & edge_mask(w0 + n - 1, lo, hi)];
    let interior = &buf[1..n - 1];
    if want_count {
        let c =
            popcount_batch_u64(&head) + popcount_batch_u64(interior) + popcount_batch_u64(&tail);
        Value::Count(c as usize)
    } else {
        Value::Bool(mask_any(&head) || mask_any(interior) || mask_any(&tail))
    }
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
        "{:>17} {:>5} {:>5} {:>3} {:>10} {:>10} {:>10} {:>10} {:>6} {:>6} {:>9} {:>10} {:>10}",
        "chain",
        "ext",
        "term",
        "ops",
        "fold_ns",
        "bulk_ns",
        "tiled_ns",
        "keep_ns",
        "t/f",
        "k/f",
        "tiled_wr",
        "wr_ns",
        "disp_ns"
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
        // One buffer per distinct scratch slot `ops` addresses, each sized
        // to the LARGEST span either extent below touches (`words`, the
        // "whole" extent's span) — allocated once per chain, reused across
        // both extents and both terminals by slicing to the live span.
        let max_slot = ops.iter().map(op_dst).max().unwrap_or(0);
        let mut bufs: Vec<Vec<u64>> = (0..=max_slot).map(|_| vec![0u64; words]).collect();
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
                let (bns, bv) = median(reps, || {
                    bulk_eval(&ops, &masks, &mut bufs, span.clone());
                    bulk_terminal(
                        &bufs[usize::from(last)][..span.len()],
                        span.start,
                        lo,
                        hi,
                        k == 0,
                    )
                });
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
                assert_eq!(bv, expect, "bulk {name} {ename} {tname}");
                assert_eq!(tv, expect, "tiled {name} {ename} {tname}");
                assert_eq!(kv, expect, "keep {name} {ename} {tname}");
                // NaN propagates automatically: when `fns` is NaN (the 4p
                // chain, uncollapsible) `wr_ns` is NaN too, matching the
                // fold_ns column's own NaN-means-not-collapsible convention.
                let wr_ns = bns - fns;
                let disp_ns = tns - bns;
                println!(
                    "{name:>17} {ename:>5} {tname:>5} {:>3} {fns:>10.0} {bns:>10.0} {tns:>10.0} {kns:>10.0} {:>6.2} {:>6.2} {:>9} {wr_ns:>10.0} {disp_ns:>10.0}",
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
         writes, ops × touched words; the fold writes 0; fold_ns NaN = not collapsible; \
         wr_ns = bulk_ns - fold_ns and disp_ns = tiled_ns - bulk_ns are measured \
         gaps between paths, not isolated write or dispatch costs)"
    );
}
