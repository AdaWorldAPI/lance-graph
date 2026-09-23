//! Measure `Range[lo, hi) ∩ resident plane → Count` two ways over the SAME
//! membership relation:
//!
//! - MATERIALIZED: `Pred::Range → Scratch(0)`, `And(Scratch(0), Plane(0)) →
//!   Scratch(1)`, `Count(Scratch(1))` — the ops write derived membership one
//!   tile at a time before the terminal counts it.
//! - FUSED: `Pred::Range` gated by the plane, `Count` — the executor folds the
//!   plane's touched words and two register-masked edge words, writing none.
//!
//! Reported separately, never collapsed into one "speedup": latency (median
//! of repeats), derived membership words written, and plane words read. The
//! word counts are DERIVED from the executor's contract (the tiled path
//! writes every tile of every slot; the fused path reads `touched_words`), and
//! the fused count is cross-checked against the materialized one each run.
//!
//! `cargo run --release -p lance-graph-mask-risc --example range_fused_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_into, Scratch};
use lance_graph_mask_risc::{
    touched_words, Foreign, MaskOp, Operand, Out, Planes, Pred, Program, Terminal, Value,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
}

fn plane(n: usize, set: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; n.div_ceil(64)];
    for r in 0..n {
        if set(r) {
            w[r / 64] |= 1u64 << (r % 64);
        }
    }
    w
}

fn median_ns(mut f: impl FnMut() -> Value, reps: usize) -> (f64, Value) {
    let mut times = Vec::with_capacity(reps);
    let mut last = Value::Count(0);
    for _ in 0..reps {
        let t = Instant::now();
        last = std::hint::black_box(f());
        times.push(t.elapsed().as_nanos() as f64);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    (times[reps / 2], last)
}

fn main() {
    println!(
        "{:>8} {:>10} {:>18} {:>12} {:>12} {:>10} {:>10} {:>9} {:>9}",
        "N", "range", "plane", "mat_ns", "fused_ns", "mat_wr", "fused_wr", "mat_rd", "fused_rd"
    );
    let mut seed = 0xfeed_u64;
    for n in [4_096usize, 65_536, 1_048_576] {
        let scattered: Vec<bool> = (0..n).map(|_| lcg(&mut seed).is_multiple_of(97)).collect();
        let shapes: [(&str, Vec<u64>); 3] = [
            ("dense", plane(n, |r| r % 4 != 0)),
            ("sparse-clustered", plane(n, |r| (r / 4096) % 50 == 7)),
            ("sparse-scattered", plane(n, |r| scattered[r])),
        ];
        let nn = n as u32;
        let ranges: [(&str, u32, u32); 5] = [
            ("tiny", nn / 2, nn / 2 + 3),
            ("one-tile", 1000.min(nn - 600), 1000.min(nn - 600) + 512),
            ("1%", nn / 3, nn / 3 + nn / 100),
            ("25%", nn / 5, nn / 5 + nn / 4),
            ("whole", 0, nn),
        ];
        let words = n.div_ceil(64);
        for (pname, p) in &shapes {
            let masks: [&[u64]; 1] = [p];
            let planes = Planes {
                n_rows: n,
                masks: &masks,
                lanes: &[],
            };
            for (rname, lo, hi) in ranges {
                let materialized = Program::new(
                    vec![
                        MaskOp::Pred {
                            pred: Pred::Range { lo, hi },
                            under: None,
                            dst: 0,
                        },
                        MaskOp::And {
                            a: Operand::Scratch(0),
                            b: Operand::Plane(0),
                            dst: 1,
                        },
                    ],
                    Terminal::Count {
                        mask: Operand::Scratch(1),
                    },
                );
                let fused = Program::new(
                    vec![MaskOp::Pred {
                        pred: Pred::Range { lo, hi },
                        under: Some(Operand::Plane(0)),
                        dst: 0,
                    }],
                    Terminal::Count {
                        mask: Operand::Scratch(0),
                    },
                );
                assert!(materialized.fused_terminal().is_none());
                assert!(fused.fused_terminal().is_some());
                let mut ms = Scratch::for_program(&materialized, n).expect("scratch");
                let mut fs = Scratch::for_program(&fused, n).expect("scratch");
                let reps = if n > 100_000 { 31 } else { 201 };
                let (mat_ns, mv) = median_ns(
                    || {
                        execute_into(&materialized, &planes, &Foreign::NONE, &mut ms, Out::None)
                            .expect("materialized")
                    },
                    reps,
                );
                let (fused_ns, fv) = median_ns(
                    || {
                        execute_into(&fused, &planes, &Foreign::NONE, &mut fs, Out::None)
                            .expect("fused")
                    },
                    reps,
                );
                assert_eq!(mv, fv, "the two arms disagree: {pname} {rname} N={n}");
                // Materialized: the range slot and the And slot are each
                // written over every tile; the plane is read once over the
                // whole population. Fused: nothing written; the plane is read
                // over the touched span only.
                let (mat_wr, mat_rd) = (2 * words, words);
                let fused_rd = touched_words(lo, hi).len();
                println!(
                    "{n:>8} {rname:>10} {pname:>18} {mat_ns:>12.0} {fused_ns:>12.0} \
                     {mat_wr:>10} {:>10} {mat_rd:>9} {fused_rd:>9}",
                    0
                );
            }
        }
    }
}
