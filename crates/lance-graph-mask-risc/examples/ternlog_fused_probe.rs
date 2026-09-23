//! Boolean membership → Count/Any at N = 1M rows: the fold vs the kept mask.
//!
//! One op over three resident planes, two physical endings:
//!
//! - FOLD: `Count` / `Any` of the op's own slot — `Program::fused_ternlog`
//!   lowers it onto `ndarray::simd::mask_ternlog_{popcount,any}`; no slot is
//!   carved and no membership word is written.
//! - MATERIALIZE: the same op with a `Keep` terminal into an `Out::Mask`, then
//!   `popcount_batch_u64` / `mask_any` over the kept words — the
//!   bitmap-then-reduce shape the fold replaces.
//!
//! Reported per shape × extent: median ns for each arm and the membership
//! words the materializing arm writes (the fold writes none). Every result is
//! checked against a bit-serial scalar oracle over the same absolute rows.
//!
//! `cargo run --release -p lance-graph-mask-risc --example ternlog_fused_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{
    touched_words, Foreign, MaskOp, Operand, Out, Planes, Program, Terminal, Value,
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

fn main() {
    let n = 1usize << 20;
    let words = n.div_ceil(64);
    let mut seed = 0x7E4F_u64;
    let mut mk = |dense: bool| {
        let mut w = vec![0u64; words];
        for r in 0..n {
            let x = lcg(&mut seed);
            if (dense && !x.is_multiple_of(3)) || (!dense && x.is_multiple_of(29)) {
                w[r / 64] |= 1 << (r % 64);
            }
        }
        w
    };
    let (pa, pb, pc) = (mk(true), mk(true), mk(false));
    let masks: [&[u64]; 3] = [&pa, &pb, &pc];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &[],
    };
    let (a, b, c) = (Operand::Plane(0), Operand::Plane(1), Operand::Plane(2));
    let shapes: [(&str, MaskOp, u8); 4] = [
        ("And", MaskOp::And { a, b, dst: 0 }, 0xC0),
        ("Xor", MaskOp::Xor { a, b, dst: 0 }, 0x3C),
        (
            "MAJ3",
            MaskOp::Ternlog {
                imm: 0xE8,
                a,
                b,
                c,
                dst: 0,
            },
            0xE8,
        ),
        (
            "NOR3",
            MaskOp::Ternlog {
                imm: 0x01,
                a,
                b,
                c,
                dst: 0,
            },
            0x01,
        ),
    ];
    let mid = n / 2 + 17;
    let extents = [
        ("1 row", mid, mid + 1),
        ("1%", mid, mid + n / 100),
        ("whole", 0, n),
    ];
    let bit = |p: &[u64], r: usize| (p[r / 64] >> (r % 64) & 1) as u8;
    println!(
        "{:>5} {:>6} {:>5} {:>11} {:>11} {:>7} {:>10}",
        "op", "extent", "term", "fold_ns", "keep_ns", "k/f", "keep_wr"
    );
    for (name, op, table) in shapes {
        let count = Program::new(
            vec![op],
            Terminal::Count {
                mask: Operand::Scratch(0),
            },
        );
        let any = Program::new(
            vec![op],
            Terminal::Any {
                mask: Operand::Scratch(0),
            },
        );
        let keep = Program::new(
            vec![op],
            Terminal::Keep {
                mask: Operand::Scratch(0),
            },
        );
        assert!(count.fused_ternlog().is_some() && any.fused_ternlog().is_some());
        let mut ks = Scratch::for_program(&keep, n).expect("scratch");
        let mut out = vec![0u64; words];
        for (ename, lo, hi) in extents {
            let want = (lo..hi)
                .filter(|&r| table >> (bit(&pa, r) << 2 | bit(&pb, r) << 1 | bit(&pc, r)) & 1 == 1)
                .count();
            let reps = if hi - lo > 100_000 { 41 } else { 2001 };
            let span = touched_words(lo as u32, hi as u32);
            for (term, p) in [("Count", &count), ("Any", &any)] {
                let (fns, fv) = median(reps, || {
                    let mut s = Scratch::new(0, 0);
                    execute_extent(p, &planes, &Foreign::NONE, &mut s, Out::None, lo..hi)
                        .expect("fold")
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
                    if term == "Count" {
                        Value::Count(popcount_batch_u64(&out[span.clone()]) as usize)
                    } else {
                        Value::Bool(mask_any(&out[span.clone()]))
                    }
                });
                let expect = if term == "Count" {
                    Value::Count(want)
                } else {
                    Value::Bool(want > 0)
                };
                assert_eq!(fv, expect, "fold {name} {ename} {term}");
                assert_eq!(kv, expect, "keep {name} {ename} {term}");
                println!(
                    "{name:>5} {ename:>6} {term:>5} {fns:>11.0} {kns:>11.0} {:>7.2} {:>10}",
                    kns / fns,
                    span.len()
                );
            }
        }
    }
    println!("(n = {n}, {words} population words; the fold writes 0 membership words)");
}
