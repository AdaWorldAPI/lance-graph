//! Classify the tiled path's bill before changing any architecture.
//!
//! Two measurements, each answering one question:
//!
//! **A — is the tile physics, or a scheduling choice?** The SIMD microtile is
//! one 512-bit vector (8 `u64`). The SCHEDULING tile is how many words one
//! pass of the op loop covers, and it is set by the caller's `Scratch` width
//! (`execute_extent` walks `extent_tiles(n_rows, scratch.words, ..)`), so it
//! can be swept without touching the executor. For each scheduling tile `T`
//! (in words) the probe reports median ns, ns per population word, tiles per
//! op (the number of times the op loop runs), and the scratch footprint
//! `slots × T × 8` bytes. `T = 8` is today's default (`TILE_WORDS`).
//!
//! **B — at small extents, is the fold paying for folding, or for
//! re-recognising the fold?** `execute_extent` calls
//! `Program::fused_ternlog()` on every execution. The probe times that
//! recognition alone, and the whole fused `execute_extent` call, over
//! extents from one word to the whole population. The difference between
//! the two is everything else the call does. It is split further: the
//! range-fold recogniser (`fused_terminal()`, which `execute_extent` also
//! asks on every call), the fold kernel alone over the same span
//! (`ternlog_popcount_dispatch`), and the remainder (validation and call
//! plumbing). Extents are word-aligned so the kernel arm is the whole fold.
//!
//! Every result is checked against a bit-serial oracle.
//!
//! `cargo run --release -p lance-graph-mask-risc --example tile_sweep_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{
    ternlog_popcount_dispatch, Foreign, MaskOp, Operand, Out, Planes, Program, Terminal, Value,
    FUSED_SLOT_CAP,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

/// Median over `reps` samples of the mean over `k` back-to-back calls: at
/// sub-100-ns scales one `Instant` pair costs as much as the call, so part B
/// amortises the timer over `k` calls per sample.
fn median_k<T>(reps: usize, k: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut ts = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..k {
            last = Some(std::hint::black_box(f()));
        }
        ts.push(t.elapsed().as_nanos() as f64 / k as f64);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    (ts[reps / 2], last.expect("reps > 0"))
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

/// Move the chain's result slot to `to`, so the recogniser declines (a bound
/// of the recogniser, never of the semantics) and the tiled path runs.
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

fn main() {
    let n = 1usize << 20;
    let words = n / 64;
    let mut seed = 0x5EE9_u64;
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
    let masks: [&[u64]; 3] = [&pa, &pb, &pc];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let bit = |p: &[u64], r: usize| p[r / 64] >> (r % 64) & 1 == 1;
    let (a, b, c) = (Operand::Plane(0), Operand::Plane(1), Operand::Plane(2));
    let s = Operand::Scratch;
    type Oracle = fn(bool, bool, bool) -> bool;
    let chains: [(&str, Vec<MaskOp>, u16, Oracle); 2] = [
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
            |a, b, c| (a && b) || !c,
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
            |a, b, c| ((a ^ b) && !c) || (a && c),
        ),
    ];
    let cap = FUSED_SLOT_CAP as u16;

    println!("== A: scheduling-tile sweep (whole population, Count, tiled path)");
    println!(
        "{:>17} {:>6} {:>10} {:>8} {:>9} {:>10} {:>7}",
        "chain", "T_w", "ns", "ns/word", "tiles", "scratch_B", "vs_T8"
    );
    for (name, ops, last, oracle) in &chains {
        let want = (0..n)
            .filter(|&r| oracle(bit(&pa, r), bit(&pb, r), bit(&pc, r)))
            .count();
        let prog = Program::new(retarget(ops, *last, cap), Terminal::Count { mask: s(cap) });
        assert!(prog.fused_ternlog().is_none());
        let slots = prog.scratch_slots as usize;
        let mut base = f64::NAN;
        let mut rows = Vec::new();
        for t in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, words] {
            let mut sc = Scratch::new(t, slots);
            let (ns, v) = median(41, || {
                execute_extent(&prog, &planes, &Foreign::NONE, &mut sc, Out::None, 0..n)
                    .expect("tiled")
            });
            assert_eq!(v, Value::Count(want), "{name} T={t}");
            if t == 8 {
                base = ns;
            }
            rows.push((t, ns));
        }
        for (t, ns) in rows {
            println!(
                "{name:>17} {t:>6} {ns:>10.0} {:>8.3} {:>9} {:>10} {:>7.2}",
                ns / words as f64,
                words.div_ceil(t),
                slots * t * 8,
                ns / base
            );
        }
    }

    println!();
    println!("== B: fold recognition vs the whole fused call, by extent (Count)");
    println!(
        "{:>17} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "chain", "rows", "call_ns", "rec_tern", "rec_rng", "kernel", "rest_ns", "recog%"
    );
    for (name, ops, last, oracle) in &chains {
        let prog = Program::new(ops.clone(), Terminal::Count { mask: s(*last) });
        assert!(prog.fused_ternlog().is_some());
        for len in [64usize, 512, 4096, 10_496, 104_896, n] {
            let lo = if len == n { 0 } else { n / 2 };
            let hi = lo + len;
            let want = (lo..hi)
                .filter(|&r| oracle(bit(&pa, r), bit(&pb, r), bit(&pc, r)))
                .count();
            let (reps, k) = if len > 100_000 { (101, 1) } else { (201, 1000) };
            let (rns, _) = median_k(reps, k, || prog.fused_ternlog());
            // `execute_extent` asks `fused_terminal()` first, on every call.
            let (tns, _) = median_k(reps, k, || prog.fused_terminal());
            // The fold kernel alone over the same (word-aligned) span.
            let f = prog.fused_ternlog().expect("fused");
            let (w0, w1) = (lo / 64, hi / 64);
            let (kns, kc) = median_k(reps, k, || {
                ternlog_popcount_dispatch(
                    f.imm,
                    &masks[usize::from(f.a)][w0..w1],
                    &masks[usize::from(f.b)][w0..w1],
                    &masks[usize::from(f.c)][w0..w1],
                )
            });
            assert_eq!(kc as usize, want, "{name} kernel len={len}");
            let (cns, v) = median_k(reps, k, || {
                let mut s0 = Scratch::new(0, 0);
                execute_extent(&prog, &planes, &Foreign::NONE, &mut s0, Out::None, lo..hi)
                    .expect("fold")
            });
            assert_eq!(v, Value::Count(want), "{name} len={len}");
            println!(
                "{name:>17} {len:>8} {cns:>9.0} {rns:>9.0} {tns:>9.0} {kns:>9.0} {:>9.0} {:>6.1}%",
                cns - rns - tns - kns,
                100.0 * (rns + tns) / cns
            );
        }
    }
    println!(
        "(n = {n}, {words} words; T_w = scheduling tile in u64 words, today's default is 8; \
         tiles = op-loop iterations per op; scratch_B = slots × T × 8; median ns. \
         Part B small extents average 1000 back-to-back calls per sample, so the timer's own cost is amortised; a residual within ~±10 ns of zero is noise.)"
    );
}
