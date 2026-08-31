//! **D-DCR-0 (W0) — the replay budget, measured.**
//!
//! `.claude/plans/dismech-causal-replay-v1.md` §3 W0. Three numbers the
//! deferred p64 64x64 ALU wave states its BUY threshold in, plus the
//! pre-registered KILL check for W5 (frontier scheduling).
//!
//! ```text
//! cargo run -p lance-graph-contract --example dcr_w0_replay_budget --release
//! ```
//!
//! # What is measured, and what it is measured ON
//!
//! **The corpus scale is NOT invented here.** `dismech-causality-v3-v1.md`
//! §3a/§11 already measured the real supervision corpus three independent
//! ways: **2,449** oracle edges (`INDIRECT_KNOWN_INTERMEDIATES` that actually
//! name a mediator) over **534** diseases, a **4,076**-row restraint arm
//! (4,150 minus the 74 that contradict their own label) and **361** `UNKNOWN`
//! rows kept as a third arm. This probe therefore reports its per-scale
//! numbers against those magnitudes rather than against a round number
//! somebody liked.
//!
//! 1. **Step throughput** — one replay step is a `NarsTruth::revision`
//!    (evidence fusion) plus the `EvidenceMask` intersect that keeps the
//!    candidate set live. Both are contract-level and already shipped; this
//!    times the composition, never a new kernel.
//! 2. **Branching shrink** — how much a single evidence item reduces a
//!    candidate set, at 10^3 / 10^4 / 10^5 chains. This is the Mengenlehre
//!    half: `intersection` for support, `difference` for refute.
//! 3. **KILL check (pre-registered, two-sided)** — if a FULL SCAN of the real
//!    corpus costs less than one frontier decision, the scheduler is
//!    decoration at DisMech scale and W5 is descoped *for this corpus*, with
//!    the crossover size reported so a bigger corpus can re-open it.
//!
//! # Non-carriers
//!
//! `Bits` below is an EXAMPLE-LOCAL fixture implementing the existing
//! [`EvidenceMask`] trait for sets wider than the shipped `u64` impl. The
//! trait exists to be implemented; a probe fixture is not a new carrier and
//! nothing outside this file may use it (`F-RLR-2` is about the production
//! path). The step kernel is `NarsTruth` + `EvidenceMask` — `CausalEdge64`'s
//! packed form lives planner-side (`causal-edge`), one dependency layer out
//! from this zero-dep crate, and is deliberately NOT pulled in to time it.

use std::time::Instant;

use lance_graph_contract::exploration::NarsTruth;
use lance_graph_contract::revision::EvidenceMask;

/// Example-local wide bitset (see the module doc's non-carrier note).
#[derive(Clone, PartialEq, Eq, Debug)]
struct Bits(Vec<u64>);

impl Bits {
    fn with_capacity(n: usize) -> Self {
        Bits(vec![0u64; n.div_ceil(64)])
    }
    fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1u64 << (i % 64);
    }
    fn count(&self) -> usize {
        self.0.iter().map(|w| w.count_ones() as usize).sum()
    }
}

impl EvidenceMask for Bits {
    fn empty() -> Self {
        Bits(Vec::new())
    }
    fn is_empty(&self) -> bool {
        self.0.iter().all(|w| *w == 0)
    }
    fn union(&self, other: &Self) -> Self {
        Bits(zip_words(&self.0, &other.0, |a, b| a | b))
    }
    fn intersection(&self, other: &Self) -> Self {
        Bits(zip_words(&self.0, &other.0, |a, b| a & b))
    }
    fn difference(&self, other: &Self) -> Self {
        Bits(zip_words(&self.0, &other.0, |a, b| a & !b))
    }
    fn is_subset_of(&self, other: &Self) -> bool {
        self.0
            .iter()
            .enumerate()
            .all(|(i, w)| w & !other.0.get(i).copied().unwrap_or(0) == 0)
    }
}

fn zip_words(a: &[u64], b: &[u64], f: impl Fn(u64, u64) -> u64) -> Vec<u64> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            f(
                a.get(i).copied().unwrap_or(0),
                b.get(i).copied().unwrap_or(0),
            )
        })
        .collect()
}

/// Deterministic PRNG — a probe must be replayable (that is this plan's whole
/// keystone), so no `rand`, no clock seeding.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n
    }
}

/// The real supervision-corpus magnitudes, measured in
/// `dismech-causality-v3-v1.md` §3a/§11 — never re-derived here.
const ORACLE_EDGES: usize = 2_449;
const RESTRAINT_ROWS: usize = 4_076;
const UNKNOWN_ROWS: usize = 361;
const ORACLE_DISEASES: usize = 534;

/// One replay step: fuse this step's evidence into the running truth, and
/// keep the candidate set live. Both halves are shipped contract surface.
#[inline]
fn replay_step(
    running: NarsTruth,
    step_evidence: NarsTruth,
    live: &Bits,
    support: &Bits,
) -> (NarsTruth, Bits) {
    (running.revision(&step_evidence), live.intersection(support))
}

fn measure_step_throughput(chain_len: usize, chains: usize, set_bits: usize) -> (f64, f64) {
    let mut live = Bits::with_capacity(set_bits);
    for i in 0..set_bits {
        live.set(i);
    }
    let mut rng = Lcg(0x051E_D270_B5A1_11E5);
    // Pre-build per-step support masks so the timed loop measures the STEP,
    // not fixture construction.
    // 2/3 density: evidence that halves the candidate set at EVERY step is
    // unrealistically strong and collapses the set within ~12 steps, which
    // would put a clone-reset inside the timed loop and measure the fixture.
    let supports: Vec<Bits> = (0..chain_len)
        .map(|_| {
            let mut m = Bits::with_capacity(set_bits);
            for i in 0..set_bits {
                if rng.below(3) != 0 {
                    m.set(i);
                }
            }
            m
        })
        .collect();
    let ev = NarsTruth::new(0.8, 0.6);

    let t0 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..chains {
        let mut running = NarsTruth::prior();
        let mut cur = live.clone();
        for s in &supports {
            let (r, c) = replay_step(running, ev, &cur, s);
            running = r;
            cur = c;
        }
        sink += cur.count() + (running.expectation() > 0.0) as usize;
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    let steps = (chain_len * chains) as f64;
    (steps / dt / 1000.0, dt * 1000.0) // (steps per ms, total ms)
}

/// The two halves of a step, timed apart. This is the question the deferred
/// 64x64 ALU wave actually asks: a tile is 4096 bits = one node's budget, so
/// it accelerates the MASK half. If revision dominates, the tile is aimed at
/// the wrong half and the BUY is not merely early, it is misdirected.
fn measure_kernel_split(set_bits: usize, iters: usize) -> (f64, f64) {
    let mut rng = Lcg(0x0000_DDBA_11C0_FFEE);
    let mut a = Bits::with_capacity(set_bits);
    let mut b = Bits::with_capacity(set_bits);
    for i in 0..set_bits {
        if rng.below(3) != 0 {
            a.set(i);
        }
        if rng.below(3) != 0 {
            b.set(i);
        }
    }
    let (x, y) = (NarsTruth::new(0.7, 0.5), NarsTruth::new(0.4, 0.3));

    let t0 = Instant::now();
    let mut acc = NarsTruth::prior();
    for _ in 0..iters {
        acc = std::hint::black_box(&acc).revision(std::hint::black_box(&x));
        acc = std::hint::black_box(&acc).revision(std::hint::black_box(&y));
    }
    std::hint::black_box(acc.expectation());
    let revision_per_ms = (iters * 2) as f64 / (t0.elapsed().as_secs_f64() * 1000.0);

    let t1 = Instant::now();
    let mut sink = 0usize;
    for _ in 0..iters {
        sink += std::hint::black_box(&a)
            .intersection(std::hint::black_box(&b))
            .count();
    }
    std::hint::black_box(sink);
    let mask_per_ms = iters as f64 / (t1.elapsed().as_secs_f64() * 1000.0);

    (revision_per_ms, mask_per_ms)
}

/// One frontier decision, priced honestly: score EVERY candidate observation
/// by the candidate-set split it would produce. That is the W4/W5 shape
/// (expected information gain), so the KILL check compares like with like.
fn measure_frontier_decision(candidates: usize, observations: usize) -> (f64, usize) {
    let mut live = Bits::with_capacity(candidates);
    for i in 0..candidates {
        live.set(i);
    }
    let mut rng = Lcg(0x0D15_EC40_11EA_DBEE);
    let obs: Vec<Bits> = (0..observations)
        .map(|_| {
            let mut m = Bits::with_capacity(candidates);
            for i in 0..candidates {
                if rng.below(4) != 0 {
                    m.set(i);
                }
            }
            m
        })
        .collect();

    let t0 = Instant::now();
    let mut best = (usize::MAX, 0usize);
    let total = live.count();
    for (i, o) in obs.iter().enumerate() {
        let kept = live.intersection(o).count();
        let dropped = live.difference(o).count();
        // Balanced split = most informative; |kept - dropped| is its proxy.
        let imbalance = kept.abs_diff(dropped);
        if imbalance < best.0 {
            best = (imbalance, i);
        }
        std::hint::black_box((kept, dropped, total));
    }
    (t0.elapsed().as_secs_f64() * 1000.0, best.1)
}

/// A full scan: replay every chain once, no frontier at all.
fn measure_full_scan(chains: usize, chain_len: usize, set_bits: usize) -> f64 {
    let (_, ms) = measure_step_throughput(chain_len, chains, set_bits);
    ms
}

fn main() {
    println!("D-DCR-0 (W0) — replay budget");
    println!("corpus (measured in dismech-causality-v3-v1 §3a/§11, not re-derived here):");
    println!("  oracle arm    {ORACLE_EDGES} edges over {ORACLE_DISEASES} diseases");
    println!("  restraint arm {RESTRAINT_ROWS} rows");
    println!("  unknown arm   {UNKNOWN_ROWS} rows");

    // ── 1. Step throughput ───────────────────────────────────────────────
    println!("\n1. STEP THROUGHPUT (NarsTruth::revision + EvidenceMask::intersection)");
    println!(
        "   {:>10} {:>10} {:>14} {:>12}",
        "cand.set", "chain len", "steps/ms", "ns/step"
    );
    let mut steps_per_ms_at_2k = 0.0f64;
    for &set_bits in &[64usize, 1_024, 4_096] {
        for &chain_len in &[4usize, 16] {
            let chains = 2_000;
            let (spms, _) = measure_step_throughput(chain_len, chains, set_bits);
            if set_bits == 4_096 && chain_len == 16 {
                steps_per_ms_at_2k = spms;
            }
            println!(
                "   {set_bits:>10} {chain_len:>10} {spms:>14.1} {:>12.1}",
                1_000_000.0 / spms
            );
        }
    }

    // ── 2. Branching shrink ──────────────────────────────────────────────
    println!("\n2. BRANCHING SHRINK (one evidence item, support ∩ / refute ∖)");
    println!("   NOTE: densities are FIXTURE-SET (2/3 support, 1/10 refute), so this");
    println!("   measures the MECHANISM's cost and scaling — never the corpus's real");
    println!("   discriminative power. That needs the frozen oracle/restraint TSVs");
    println!("   (D-CV3-0..2, consumer-side).");
    println!(
        "   {:>10} {:>12} {:>12} {:>10}",
        "chains", "after ∩", "after ∖", "factor"
    );
    for &n in &[1_000usize, 10_000, 100_000] {
        let mut live = Bits::with_capacity(n);
        for i in 0..n {
            live.set(i);
        }
        let mut rng = Lcg(0xA5A5_1234_DEAD_C0DE);
        let mut support = Bits::with_capacity(n);
        let mut refute = Bits::with_capacity(n);
        for i in 0..n {
            if rng.below(3) != 0 {
                support.set(i);
            }
            if rng.below(10) == 0 {
                refute.set(i);
            }
        }
        let after_and = live.intersection(&support);
        let after_diff = after_and.difference(&refute);
        println!(
            "   {n:>10} {:>12} {:>12} {:>10.2}x",
            after_and.count(),
            after_diff.count(),
            n as f64 / after_diff.count().max(1) as f64
        );
    }

    // ── 3. KILL check ────────────────────────────────────────────────────
    println!("\n3. KILL CHECK (pre-registered): is a full scan cheaper than one decision?");
    let scan_ms = measure_full_scan(ORACLE_EDGES, 4, 4_096);
    let (dec_ms, _pick) = measure_frontier_decision(ORACLE_EDGES, 64);
    println!("   full scan of {ORACLE_EDGES} chains (len 4): {scan_ms:.3} ms");
    println!("   one frontier decision over 64 observations: {dec_ms:.3} ms");
    let verdict = if scan_ms <= dec_ms {
        "KILL FIRES — scheduling is decoration at DisMech scale; W5 descoped for THIS corpus"
    } else {
        "KILL does not fire — a decision is cheaper than a scan; W5 stays live"
    };
    println!("   => {verdict}");
    // Crossover: how many chains before a scan costs a decision?
    let per_chain_ms = scan_ms / ORACLE_EDGES as f64;
    println!(
        "   crossover: a scan costs one decision at ~{:.0} chains ({:.1}x the oracle arm)",
        dec_ms / per_chain_ms,
        (dec_ms / per_chain_ms) / ORACLE_EDGES as f64
    );

    // ── 4. Kernel split — which half a 64x64 tile would even touch ───────
    println!("\n4. KERNEL SPLIT (the ALU wave's actual question)");
    let (rev_pm, mask_pm) = measure_kernel_split(4_096, 200_000);
    println!(
        "   NarsTruth::revision alone : {rev_pm:>10.0} ops/ms ({:>6.1} ns)",
        1_000_000.0 / rev_pm
    );
    println!(
        "   4096-bit intersect+count  : {mask_pm:>10.0} ops/ms ({:>6.1} ns)",
        1_000_000.0 / mask_pm
    );
    let (dominant, factor) = if mask_pm < rev_pm {
        ("MASK", rev_pm / mask_pm)
    } else {
        ("REVISION", mask_pm / rev_pm)
    };
    println!("   => {dominant} dominates by {factor:.1}x — a 64x64 tile accelerates the MASK half");

    // ── 5. ALU BUY threshold, stated in the measured numbers ─────────────
    println!("\n5. ALU BUY THRESHOLD (deferred p64 64x64 wave)");
    println!("   measured step rate (4096-wide set, len 16): {steps_per_ms_at_2k:.0} steps/ms");
    let corpus_ms = ORACLE_EDGES as f64 * 16.0 / steps_per_ms_at_2k;
    println!("   => whole oracle arm at chain len 16: {corpus_ms:.2} ms");
    println!(
        "   BUY only when a workload needs > {:.0} steps/ms sustained, i.e. > {:.0}x this corpus\n   in one budget — until then the scalar path is not the bottleneck.",
        steps_per_ms_at_2k * 10.0,
        10.0
    );
}
