//! **D-RPF-TABLE-1, axis 4: what does the revision table buy in throughput?**
//!
//! ```text
//! cargo run -p lance-graph-planner --example d_rpf_table_1_bench --release
//! ```
//!
//! Compares direct confidence-aware revision (A, `CausalEdge64::revision`)
//! with `NarsTables::build(n).revise` for n in {1, 2, 4, 8, 16}, at two
//! levels:
//!
//! - **micro**: one revision per input pair, inputs read from a 1 Mi-entry
//!   array so nothing constant-folds, results summed into a sink.
//! - **replay**: the real `replay_step` for tables; for A, the same step with
//!   the table lookup replaced by `CausalEdge64::revision`
//!   (`tests/d_rpf_table_1.rs` proves that step bit-exact against
//!   `replay_step` for every table). `dep` is one dependent 64-step chain
//!   looped (latency-bound, the W0 shape); `batch` interleaves 256 chains so
//!   independent steps can overlap (throughput-bound).
//!
//! Two input distributions: `uniform` (whole grid) and `w0` (f 128..=255,
//! c 128..=227, the distribution `dcr_w0_replay_budget` sized the kernel on).
//! `cold` scrubs a 512 MiB buffer before a short run, so the first table
//! accesses miss every cache level this machine has.
//!
//! **Anti-vacuity:** `A-slow` adds a deliberate f64 recompute to every A
//! revision. The harness must show it slower than A, or it cannot tell
//! kernels apart at all.
//!
//! Each number is the median of 7 runs.

use std::hint::black_box;
use std::time::Instant;

use causal_edge::edge::InferenceType;
use causal_edge::tables::{unpack_c, unpack_f, NarsTables};
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_planner::chain_replay::{replay_step, ComposeTables};

const N: usize = 1 << 20;
const RUNS: usize = 7;
const LEVELS: [usize; 5] = [1, 2, 4, 8, 16];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
    fn byte(&mut self, lo: u8, hi: u8) -> u8 {
        lo + (self.next() % (hi as u64 - lo as u64 + 1)) as u8
    }
}

#[derive(Clone, Copy)]
enum Dist {
    Uniform,
    W0,
}

impl Dist {
    fn pick(self, r: &mut Lcg) -> (u8, u8) {
        match self {
            Dist::Uniform => (r.byte(0, 255), r.byte(0, 255)),
            Dist::W0 => (r.byte(128, 255), r.byte(128, 227)),
        }
    }
}

const OPS: [InferenceType; 4] = [
    InferenceType::Deduction,
    InferenceType::Induction,
    InferenceType::Abduction,
    InferenceType::Revision,
];

#[allow(deprecated)] // v2 `pack` ignores temporal
fn edge(f: u8, c: u8, op: InferenceType) -> CausalEdge64 {
    CausalEdge64::pack(
        1,
        2,
        3,
        f,
        c,
        CausalMask::SPO,
        0,
        op,
        PlasticityState::ALL_HOT,
        0,
    )
}

fn pairs(d: Dist, seed: u64) -> Vec<(CausalEdge64, CausalEdge64)> {
    let mut r = Lcg(seed);
    (0..N)
        .map(|_| {
            let (f1, c1) = d.pick(&mut r);
            let (f2, c2) = d.pick(&mut r);
            (
                edge(f1, c1, InferenceType::Revision),
                edge(f2, c2, InferenceType::Revision),
            )
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The revision a step uses: A, A-slow, or a table.
#[derive(Clone, Copy)]
enum Law<'a> {
    A,
    ASlow,
    Table(&'a NarsTables),
}

#[inline(always)]
fn revise(law: Law<'_>, a: CausalEdge64, b: CausalEdge64) -> (u8, u8) {
    match law {
        Law::A => {
            let r = a.revision(b);
            (r.frequency_u8(), r.confidence_u8())
        }
        Law::ASlow => {
            let r = a.revision(b);
            // Deliberate extra work: an f64 recompute of the same law.
            let (c1, c2) = (
                a.confidence_u8() as f64 / 255.0,
                b.confidence_u8() as f64 / 255.0,
            );
            let mut x = c1 / (1.0 - c1 + 1e-9) + c2 / (1.0 - c2 + 1e-9);
            for _ in 0..16 {
                x = black_box(x).sqrt() * x.cbrt();
            }
            let bump = u8::from(x.is_nan());
            (r.frequency_u8() ^ bump, r.confidence_u8())
        }
        Law::Table(t) => {
            let p = t.revise(
                a.frequency_u8(),
                a.confidence_u8(),
                b.frequency_u8(),
                b.confidence_u8(),
            );
            (unpack_f(p), unpack_c(p))
        }
    }
}

fn micro(law: Law<'_>, input: &[(CausalEdge64, CausalEdge64)]) -> f64 {
    let ns: Vec<f64> = (0..RUNS)
        .map(|_| {
            let t0 = Instant::now();
            let mut sink = 0u64;
            for &(a, b) in input {
                let (f, c) = revise(law, black_box(a), black_box(b));
                sink = sink.wrapping_add(f as u64 + ((c as u64) << 8));
            }
            black_box(sink);
            t0.elapsed().as_secs_f64() * 1e9 / input.len() as f64
        })
        .collect();
    median(ns)
}

/// The replay step for `law`: `replay_step` for tables, the same body with
/// `revision` for A.
#[inline(always)]
fn step(
    law: Law<'_>,
    running: CausalEdge64,
    w: CausalEdge64,
    c: ComposeTables<'_>,
) -> CausalEdge64 {
    match law {
        Law::Table(t) => replay_step(running, w, t, c).expect("executable"),
        _ => {
            let (f, cc) = revise(law, running, w);
            let mut out = running.forward(w, c.s, c.p, c.o).expect("executable");
            out.set_frequency_u8(f);
            out.set_confidence_u8(cc);
            out
        }
    }
}

fn weights(d: Dist, n: usize, seed: u64) -> Vec<CausalEdge64> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let (f, c) = d.pick(&mut r);
            edge(f, c, OPS[(r.next() % 4) as usize])
        })
        .collect()
}

/// One dependent chain over 64 cyclic weights (the W0 shape).
fn replay_dep(law: Law<'_>, w: &[CausalEdge64], seed: CausalEdge64, c: ComposeTables<'_>) -> f64 {
    let iters = N;
    let ns: Vec<f64> = (0..RUNS)
        .map(|_| {
            let mut running = seed;
            let t0 = Instant::now();
            for i in 0..iters {
                running = step(law, running, black_box(w[i & 63]), c);
            }
            black_box(running.0);
            t0.elapsed().as_secs_f64() * 1e9 / iters as f64
        })
        .collect();
    median(ns)
}

/// 256 independent chains advanced in lockstep.
fn replay_batch(
    law: Law<'_>,
    w: &[CausalEdge64],
    seeds: &[CausalEdge64],
    c: ComposeTables<'_>,
) -> f64 {
    let rounds = N / seeds.len();
    let ns: Vec<f64> = (0..RUNS)
        .map(|_| {
            let mut run = seeds.to_vec();
            let t0 = Instant::now();
            for r in 0..rounds {
                for (j, x) in run.iter_mut().enumerate() {
                    *x = step(law, *x, black_box(w[(r * 7 + j) % w.len()]), c);
                }
            }
            black_box(run[0].0);
            t0.elapsed().as_secs_f64() * 1e9 / (rounds * seeds.len()) as f64
        })
        .collect();
    median(ns)
}

/// First accesses after a 512 MiB scrub, so the table is out of every cache.
fn cold(law: Law<'_>, input: &[(CausalEdge64, CausalEdge64)]) -> f64 {
    let mut scrub = vec![0u8; 512 << 20];
    let ns: Vec<f64> = (0..RUNS)
        .map(|k| {
            for (i, b) in scrub.iter_mut().enumerate().step_by(64) {
                *b = b.wrapping_add((i + k) as u8);
            }
            black_box(&scrub);
            let n = 4096;
            let t0 = Instant::now();
            let mut sink = 0u64;
            for &(a, b) in &input[k * n..(k + 1) * n] {
                let (f, c) = revise(law, black_box(a), black_box(b));
                sink = sink.wrapping_add(f as u64 + ((c as u64) << 8));
            }
            black_box(sink);
            t0.elapsed().as_secs_f64() * 1e9 / n as f64
        })
        .collect();
    median(ns)
}

fn compose_tables() -> [Box<[u8; 256 * 256]>; 3] {
    let mut r = Lcg(0x9E37_79B9_7F4A_7C15);
    core::array::from_fn(|_| {
        let mut t = Box::new([0u8; 256 * 256]);
        for v in t.iter_mut() {
            *v = r.next() as u8;
        }
        t
    })
}

fn main() {
    let ct = compose_tables();
    let compose = ComposeTables {
        s: &ct[0],
        p: &ct[1],
        o: &ct[2],
    };
    let built: Vec<NarsTables> = LEVELS.iter().map(|&n| NarsTables::build(n)).collect();
    let mut laws: Vec<(String, Law<'_>, usize)> =
        vec![("A".into(), Law::A, 0), ("A-slow".into(), Law::ASlow, 0)];
    for (n, t) in LEVELS.iter().zip(&built) {
        laws.push((format!("B{n}"), Law::Table(t), t.byte_size()));
    }

    println!(
        "{:<7} {:>10} | {:>13} {:>13} | {:>13} {:>13} | {:>13} {:>13} | {:>13}",
        "law",
        "bytes",
        "micro unif",
        "micro w0",
        "replay dep u",
        "replay dep w0",
        "replay bat u",
        "replay bat w0",
        "cold unif"
    );
    for d in [Dist::Uniform, Dist::W0] {
        // Warm the inputs once so page faults are not charged to the first law.
        black_box(pairs(d, 1).len());
    }
    let pu = pairs(Dist::Uniform, 0xA11C_E5ED);
    let pw = pairs(Dist::W0, 0xB0B_5EED);
    let wu = weights(Dist::Uniform, 64, 0x51DE);
    let ww = weights(Dist::W0, 64, 0x051E_D270_B5A1_11E5);
    let su: Vec<_> = pairs(Dist::Uniform, 7)
        .iter()
        .take(256)
        .map(|p| p.0)
        .collect();
    let sw: Vec<_> = pairs(Dist::W0, 7).iter().take(256).map(|p| p.0).collect();
    for (name, law, bytes) in &laws {
        println!(
            "{:<7} {:>10} | {:>10.2} ns {:>10.2} ns | {:>10.2} ns {:>10.2} ns | {:>10.2} ns {:>10.2} ns | {:>10.2} ns",
            name,
            bytes,
            micro(*law, &pu),
            micro(*law, &pw),
            replay_dep(*law, &wu, su[0], compose),
            replay_dep(*law, &ww, sw[0], compose),
            replay_batch(*law, &wu, &su, compose),
            replay_batch(*law, &ww, &sw, compose),
            cold(*law, &pu),
        );
    }
}
