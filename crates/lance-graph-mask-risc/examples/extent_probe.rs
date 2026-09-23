//! Dispatch locality of an absolute execution extent at N = 1M rows.
//!
//! One program family, three physical shapes, each run over extents of
//! 1 row, 64 rows, 512 rows, 1 %, 25 % and the whole population:
//!
//! - FUSED: `Range ∩ plane → Count` (the #1268 fold, intersected with the
//!   extent — writes nothing);
//! - TILED: `Range → And(plane) → Count` (writes two scratch slots per tile);
//! - LANE: `EqU32(lane) under plane → MaskedSumI32` (reads two value lanes).
//!
//! Reported separately: median latency, population words visited, lane
//! elements visited, and derived words written. The word counts are DERIVED
//! from the extent's semantic span (`touched_words`), which the executor's
//! tile plan is test-pinned to cover exactly — not guessed; every extent result is cross-checked
//! against a scalar oracle over the same absolute rows.
//!
//! `cargo run --release -p lance-graph-mask-risc --example extent_probe`

use std::time::Instant;

use lance_graph_mask_risc::exec::{execute_extent, Scratch};
use lance_graph_mask_risc::{
    touched_words, Foreign, LaneRef, MaskOp, Operand, Out, Planes, Pred, Program, Terminal, Value,
    TILE_WORDS,
};

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 11
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
    let n = 1usize << 20;
    let mut seed = 0xE7E_u64;
    let bits: Vec<bool> = (0..n).map(|_| !lcg(&mut seed).is_multiple_of(3)).collect();
    let mut pl = vec![0u64; n.div_ceil(64)];
    for (r, &b) in bits.iter().enumerate() {
        if b {
            pl[r / 64] |= 1 << (r % 64);
        }
    }
    let keys: Vec<u32> = (0..n).map(|_| (lcg(&mut seed) % 8) as u32).collect();
    let vals: Vec<i32> = (0..n).map(|_| (lcg(&mut seed) % 1000) as i32).collect();
    let masks: [&[u64]; 1] = [&pl];
    let lanes = [LaneRef::U32(&keys), LaneRef::I32(&vals)];
    let planes = Planes {
        n_rows: n,
        masks: &masks,
        lanes: &lanes,
    };
    let (plo, phi) = (1000u32, n as u32 - 1000);
    let fused = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::Range { lo: plo, hi: phi },
            under: Some(Operand::Plane(0)),
            dst: 0,
        }],
        Terminal::Count {
            mask: Operand::Scratch(0),
        },
    );
    let tiled = Program::new(
        vec![
            MaskOp::Pred {
                pred: Pred::Range { lo: plo, hi: phi },
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
    let lane = Program::new(
        vec![MaskOp::Pred {
            pred: Pred::EqU32 { lane: 0, v: 3 },
            under: Some(Operand::Plane(0)),
            dst: 0,
        }],
        Terminal::MaskedSumI32 {
            mask: Operand::Scratch(0),
            lane: 1,
        },
    );
    let mid = n / 2 + 17; // deliberately unaligned
    let extents: [(&str, usize, usize); 6] = [
        ("1 row", mid, mid + 1),
        ("64 rows", mid, mid + 64),
        ("512 rows", mid, mid + 512),
        ("1%", mid, mid + n / 100),
        ("25%", n / 5 + 3, n / 5 + 3 + n / 4),
        ("whole", 0, n),
    ];
    println!(
        "{:>6} {:>9} {:>10} {:>12} {:>12} {:>14}",
        "shape", "extent", "median_ns", "words_read", "lane_elems", "derived_wr"
    );
    for (shape, p, slots) in [
        ("fused", &fused, 0usize),
        ("tiled", &tiled, 2),
        ("lane", &lane, 1),
    ] {
        let mut s = Scratch::for_program(p, n).expect("scratch");
        for (name, lo, hi) in extents {
            // Scalar oracle over the same ABSOLUTE rows.
            let want = match shape {
                "lane" => Value::SumI64(
                    (lo..hi)
                        .filter(|&r| bits[r] && keys[r] == 3)
                        .map(|r| i64::from(vals[r]))
                        .sum(),
                ),
                _ => Value::Count(
                    (lo..hi)
                        .filter(|&r| bits[r] && (plo as usize..phi as usize).contains(&r))
                        .count(),
                ),
            };
            let reps = if hi - lo > 100_000 { 31 } else { 2001 };
            let (ns, got) = median_ns(
                || {
                    execute_extent(p, &planes, &Foreign::NONE, &mut s, Out::None, lo..hi)
                        .expect("extent")
                },
                reps,
            );
            assert_eq!(got, want, "{shape} {name}");
            // Words the extent touches: the semantic span, which the executor's
            // tile plan covers exactly (pinned in-crate by `extent_tile_tests`).
            let tiles_words = touched_words(lo as u32, hi as u32).len();
            let (words_read, lane_elems, derived) = match shape {
                "fused" => {
                    let (a, b) = ((plo as usize).max(lo), (phi as usize).min(hi));
                    let span = if a < b {
                        touched_words(a as u32, b as u32).len()
                    } else {
                        0
                    };
                    (span, 0, 0)
                }
                "tiled" => (tiles_words, 0, slots * tiles_words),
                _ => (tiles_words, 2 * tiles_words * 64, slots * tiles_words),
            };
            println!(
                "{shape:>6} {name:>9} {ns:>10.0} {words_read:>12} {lane_elems:>12} {derived:>14}"
            );
        }
    }
    println!(
        "(n = {n}, {} population words, scratch tile = {} words)",
        n / 64,
        TILE_WORDS
    );
}
