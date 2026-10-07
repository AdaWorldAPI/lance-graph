//! D-GATED-GATHER-0: isolate the semijoin (gather) stage and ask what its
//! cost actually is — work spent on rows the gate already rejected, or the
//! per-row branch inside the kernel.
//!
//! Setup: 1M rows, a u32 foreign-key lane `fk` into a 1 024-row foreign table
//! whose resident plane marks half its rows (a hash, so the per-row bit is an
//! unpredictable branch). A GATE mask of the requested density (the first
//! conjunct's survivors) is built once, outside every timed region. Every
//! route computes the same answer: `popcount(gate ∧ gather(fk, foreign))`.
//!
//! Routes:
//! - `prod`        : the production path — `quack::lower` of
//!   `plane(gate) ∧ semijoin(fk)` → mask-risc → `ndarray::simd::mask_gather_u32`
//!   over every row, ANDed with the gate afterwards;
//! - `full-bl`     : full-population gather, branchless bit test (isolates the
//!   branch cost from the population cost);
//! - `word-gated`  : skip gate words that are 0; for a live word, gather all
//!   64 lanes branchlessly and AND with the gate word (what
//!   `Gather { under }` would do over today's kernel shape);
//! - `bit-gated`   : skip zero words; inside a live word visit only set bits;
//! - `ordinal`     : materialise the gate's set bits as `Vec<u32>`, then gather
//!   per ordinal (the selection-vector carrier).
//!
//! Counters per route: foreign-plane loads issued, index-lane 64-byte lines
//! touched, ordinal bytes materialised. The gate's own construction cost is
//! reported once as `gate_us` (it is the same for every route).
//!
//! ```bash
//! cargo run --release -p lance-graph-benches --example gated_gather_probe
//! ```

use std::time::Instant;

use lance_graph_mask_risc::{
    execute_into, words_for, Foreign, ForeignPlane, LaneRef, Out, Planes, Program, Scratch, Value,
};
use lance_graph_quack::{lower, Agg, Cmp, Col, Filter, ForeignPlane as FP, Mask, Query};

const N: usize = 1_000_000;
const F: usize = 1024;
const SEL: Col = Col(0);
const FK: Col = Col(1);
const LIVE: Mask = Mask(0);
const GATE: Mask = Mask(1);

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn ones(n: usize) -> Vec<u64> {
    let mut w = vec![u64::MAX; words_for(n)];
    if !n.is_multiple_of(64) {
        *w.last_mut().unwrap() = (1u64 << (n % 64)) - 1;
    }
    w
}

fn median_us(reps: usize, mut f: impl FnMut() -> u64) -> (f64, u64) {
    let mut last = 0;
    let mut v: Vec<u128> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            last = std::hint::black_box(f());
            t0.elapsed().as_nanos()
        })
        .collect();
    v.sort_unstable();
    (v[v.len() / 2] as f64 / 1e3, last)
}

/// The bit test every hand route uses: branchless, AND it keeps the kernel's
/// out-of-range = false contract (`i >= rows` reads as 0), so no route gets a
/// cheaper test than production's.
#[inline]
fn bit(words: &[u64], i: usize) -> u64 {
    let ok = (i < F) as u64;
    let j = if ok == 1 { i } else { 0 };
    ((words[j / 64] >> (j % 64)) & 1) & ok
}

/// Counters for one route.
#[derive(Default, Clone, Copy)]
struct Work {
    loads: u64,
    index_lines: u64,
    ordinal_bytes: u64,
}

struct World {
    sel: Vec<i32>,
    fk: Vec<u32>,
    live: Vec<u64>,
    foreign: Vec<u64>,
}

impl World {
    fn new(clustered: bool) -> Self {
        let sel = (0..N as u64)
            .map(|i| {
                if clustered {
                    i as i32
                } else {
                    (mix(i) % N as u64) as i32
                }
            })
            .collect();
        let fk = (0..N as u64)
            .map(|i| (mix(i ^ 0xF1) % F as u64) as u32)
            .collect();
        let mut foreign = vec![0u64; words_for(F)];
        for r in (0..F).filter(|r| mix(*r as u64 ^ 0x51).is_multiple_of(2)) {
            foreign[r / 64] |= 1 << (r % 64);
        }
        World {
            sel,
            fk,
            live: ones(N),
            foreign,
        }
    }

    fn exec(&self, p: &Program, masks: &[&[u64]], out: Out<'_>) -> Value {
        let lanes = [LaneRef::I32(&self.sel), LaneRef::U32(&self.fk)];
        let planes = Planes {
            n_rows: N,
            masks,
            lanes: &lanes,
        };
        let fp = [ForeignPlane {
            words: &self.foreign,
            rows: F,
        }];
        let foreign = Foreign {
            planes: &fp,
            lanes: &[],
        };
        let mut scratch = Scratch::for_program(p, N).expect("scratch");
        execute_into(p, &planes, &foreign, &mut scratch, out).expect("executes")
    }

    /// The gate: `sel < t`, kept as a mask by the production lowering.
    fn gate(&self, t: i32) -> Vec<u64> {
        let p = lower(&Query {
            filter: Filter::and([Filter::plane(LIVE), Filter::cmp(SEL, Cmp::LtI32(t))]),
            agg: Agg::Rows,
        })
        .unwrap();
        let mut m = vec![0u64; words_for(N)];
        self.exec(&p, &[&self.live], Out::Mask(&mut m));
        m
    }

    fn prod_program() -> Program {
        lower(&Query {
            filter: Filter::and([Filter::plane(GATE), Filter::semijoin(FK, FP(0))]),
            agg: Agg::Count,
        })
        .unwrap()
    }

    fn prod(&self, p: &Program, gate: &[u64]) -> u64 {
        match self.exec(p, &[&self.live, gate], Out::None) {
            Value::Count(c) => c as u64,
            v => panic!("{v:?}"),
        }
    }

    fn full_branchless(&self, gate: &[u64]) -> u64 {
        let mut c = 0u64;
        for (w, &g) in gate.iter().enumerate() {
            let base = w * 64;
            let mut acc = 0u64;
            for lane in 0..64.min(N - base) {
                acc |= bit(&self.foreign, self.fk[base + lane] as usize) << lane;
            }
            c += (acc & g).count_ones() as u64;
        }
        c
    }

    fn word_gated(&self, gate: &[u64]) -> u64 {
        let mut c = 0u64;
        for (w, &g) in gate.iter().enumerate() {
            if g == 0 {
                continue;
            }
            let base = w * 64;
            let mut acc = 0u64;
            for lane in 0..64.min(N - base) {
                acc |= bit(&self.foreign, self.fk[base + lane] as usize) << lane;
            }
            c += (acc & g).count_ones() as u64;
        }
        c
    }

    fn bit_gated(&self, gate: &[u64]) -> u64 {
        let mut c = 0u64;
        for (w, &g) in gate.iter().enumerate() {
            let mut b = g;
            while b != 0 {
                let r = w * 64 + b.trailing_zeros() as usize;
                b &= b - 1;
                c += bit(&self.foreign, self.fk[r] as usize);
            }
        }
        c
    }

    fn ordinal(&self, gate: &[u64]) -> u64 {
        let mut rows: Vec<u32> = Vec::new();
        for (w, &g) in gate.iter().enumerate() {
            let mut b = g;
            while b != 0 {
                rows.push((w * 64) as u32 + b.trailing_zeros());
                b &= b - 1;
            }
        }
        rows.iter()
            .map(|&r| bit(&self.foreign, self.fk[r as usize] as usize))
            .sum()
    }

    fn oracle(&self, gate: &[u64]) -> u64 {
        (0..N)
            .filter(|&r| {
                (gate[r / 64] >> (r % 64)) & 1 == 1 && bit(&self.foreign, self.fk[r] as usize) == 1
            })
            .count() as u64
    }
}

/// Exact work counters (not timed).
fn work(gate: &[u64]) -> [Work; 5] {
    let live_words = gate.iter().filter(|&&g| g != 0).count() as u64;
    let survivors: u64 = gate.iter().map(|g| g.count_ones() as u64).sum();
    // 16 u32 index entries per 64-byte line; a 64-row word spans 4 lines.
    let all_lines = (N as u64).div_ceil(16);
    let mut bit_lines = 0u64;
    for g in gate {
        for q in 0..4 {
            if (g >> (16 * q)) & 0xFFFF != 0 {
                bit_lines += 1;
            }
        }
    }
    let full = Work {
        loads: N as u64,
        index_lines: all_lines,
        ordinal_bytes: 0,
    };
    [
        full,
        full,
        Work {
            loads: 64 * live_words,
            index_lines: 4 * live_words,
            ordinal_bytes: 0,
        },
        Work {
            loads: survivors,
            index_lines: bit_lines,
            ordinal_bytes: 0,
        },
        Work {
            loads: survivors,
            index_lines: bit_lines,
            ordinal_bytes: 4 * survivors,
        },
    ]
}

fn main() {
    let reps: usize = std::env::var("PROBE_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let names = ["prod", "full-bl", "word-gated", "bit-gated", "ordinal"];
    println!("layout\tdensity\tsurvivors\tlive_words\tgate_us\troute\tus\tforeign_loads\tindex_lines\tordinal_bytes");
    let prod_p = World::prod_program();
    for clustered in [false, true] {
        let w = World::new(clustered);
        for density in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001] {
            let t = (N as f64 * density) as i32;
            let (gate_us, _) = median_us(reps, || w.gate(t)[0]);
            let gate = w.gate(t);
            let want = w.oracle(&gate);
            let live_words = gate.iter().filter(|&&g| g != 0).count();
            let survivors: u64 = gate.iter().map(|g| g.count_ones() as u64).sum();
            let counters = work(&gate);
            let runs: [Box<dyn Fn() -> u64>; 5] = [
                Box::new(|| w.prod(&prod_p, &gate)),
                Box::new(|| w.full_branchless(&gate)),
                Box::new(|| w.word_gated(&gate)),
                Box::new(|| w.bit_gated(&gate)),
                Box::new(|| w.ordinal(&gate)),
            ];
            for (i, f) in runs.iter().enumerate() {
                let (us, got) = median_us(reps, f);
                assert_eq!(got, want, "{} at density {density}", names[i]);
                let k = counters[i];
                println!(
                    "{}\t{density}\t{survivors}\t{live_words}\t{gate_us:.1}\t{}\t{us:.1}\t{}\t{}\t{}",
                    if clustered { "clustered" } else { "uniform" },
                    names[i],
                    k.loads,
                    k.index_lines,
                    k.ordinal_bytes,
                );
            }
        }
    }
}
