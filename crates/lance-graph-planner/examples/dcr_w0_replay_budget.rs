//! **D-DCR-0 (W0) — the replay budget, measured.** `dismech-causal-replay-v1` §3.
//!
//! ```text
//! cargo run -p lance-graph-planner --example dcr_w0_replay_budget --release
//! ```
//!
//! # ⊘ CORRECTED after codex review on #1118 — three findings, all valid
//!
//! The first version of this probe lived in `lance-graph-contract` and was
//! wrong in three ways that all pushed the same direction (they inflated the
//! mask half and measured a kernel the plan never named):
//!
//! 1. **It benchmarked a substitute kernel.** §3 W0 defines an eval as
//!    *"a `NarsTables` lookup + `CausalEdge64` revision"*. The v1 probe timed
//!    `NarsTruth::revision` (f32, contract-side) instead, because the real
//!    kernel lives in `causal-edge`, one dependency layer outside the zero-dep
//!    contract crate. Disclosing the substitution in a doc-comment did not make
//!    the headline number the promised measurement. **Fix: the probe moved
//!    HERE** (planner path-deps `causal-edge`) and now times
//!    [`NarsTables::revise`] + [`CausalEdge64::forward`].
//! 2. **It put a heap allocation inside the timed mask loop.** The v1 fixture
//!    was `Bits(Vec<u64>)`, so every `intersection` ran `…collect()`. The
//!    workspace already ships `impl<const N: usize> EvidenceMask for [u64; N]`
//!    (`revision.rs:70`), which is allocation-free and is the shape the p64
//!    surface actually has (`[u64; 64]` = 4096 bits = one node's budget). The
//!    v1 "MASK dominates by 3.8x" therefore charged `malloc` to the arithmetic
//!    a tile would accelerate — the exact claim it was used to support. **Fix:
//!    both are timed below and the allocation share is REPORTED**, so the
//!    correction is visible rather than quietly swapped.
//! 3. **It moved the pre-registered goalpost.** §3's KILL gate names 10^5
//!    chains verbatim; v1 substituted the 2,449-edge oracle arm and recorded
//!    the verdict from that. A pre-registered gate must be evaluated as
//!    pre-registered. **Fix: BOTH scales are run**, at one candidate width.
//!
//! Recorded rather than silently rewritten (append-only convention): the v1
//! numbers are not deleted from the board, they are superseded there with this
//! reason.
//!
//! # Corpus scale is READ, never re-derived
//!
//! `dismech-causality-v3-v1.md` §3a/§11 measured the supervision corpus three
//! independent ways: **2,449** oracle edges over **534** diseases, **4,076**
//! restraint rows, **361** `UNKNOWN` rows.

use std::time::Instant;

use causal_edge::edge::InferenceType;
use causal_edge::tables::{unpack_c, unpack_f, NarsTables};
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_contract::revision::EvidenceMask;

/// Measured in `dismech-causality-v3-v1.md` §3a/§11 — never re-derived here.
const ORACLE_EDGES: usize = 2_449;
const ORACLE_DISEASES: usize = 534;
const RESTRAINT_ROWS: usize = 4_076;
const UNKNOWN_ROWS: usize = 361;

/// The pre-registered KILL scale, verbatim from the plan's §3 W0 gate.
const PREREGISTERED_SCAN_CHAINS: usize = 100_000;

/// 4096 bits = one node's value budget = the p64 tile shape (`[u64; 64]`).
const MASK_WORDS: usize = 64;
type Mask = [u64; MASK_WORDS];

/// Deterministic PRNG — a probe for a *replay* plan must itself be replayable,
/// so no `rand`, no clock seeding.
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

/// A mask with roughly `(one_in - 1) / one_in` of its bits set.
/// `one_in == 1` would set NOTHING (`x % 1 == 0` always), so an all-ones set
/// has its own constructor rather than a degenerate parameter value — the
/// first draft of this probe hit exactly that trap and measured a frontier
/// decision against an EMPTY live set.
fn dense_mask(rng: &mut Lcg, one_in: usize) -> Mask {
    assert!(one_in >= 2, "use all_ones() for a full set");
    let mut m = [0u64; MASK_WORDS];
    for w in m.iter_mut() {
        let mut word = 0u64;
        for b in 0..64 {
            if rng.below(one_in) != 0 {
                word |= 1u64 << b;
            }
        }
        *w = word;
    }
    m
}

fn all_ones() -> Mask {
    [u64::MAX; MASK_WORDS]
}

fn popcount(m: &Mask) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

/// **The promised chain step** (`dismech-causal-replay-v1` §3 W0, verbatim):
/// a `NarsTables` lookup plus a `CausalEdge64` revision. `forward` IS that
/// revision — palette compose + NARS truth propagation in one packed edge.
#[inline]
fn promised_step(
    running: CausalEdge64,
    weight: CausalEdge64,
    tables: &NarsTables,
    cs: &[u8; 256 * 256],
    cp: &[u8; 256 * 256],
    co: &[u8; 256 * 256],
) -> CausalEdge64 {
    // 1. the table lookup half
    let revised = tables.revise(
        running.frequency_u8(),
        running.confidence_u8(),
        weight.frequency_u8(),
        weight.confidence_u8(),
    );
    // 2. the packed-edge half
    let out = running.forward(weight, cs, cp, co);
    // fold the lookup back in so neither half can be optimised away
    CausalEdge64(out.0 ^ ((unpack_f(revised) as u64) << 32) ^ (unpack_c(revised) as u64))
}

fn build_compose() -> Box<[[u8; 256 * 256]; 3]> {
    let mut t = Box::new([[0u8; 256 * 256]; 3]);
    for (k, tab) in t.iter_mut().enumerate() {
        for i in 0..256usize {
            for j in 0..256usize {
                tab[i * 256 + j] = ((i + j + k) % 256) as u8;
            }
        }
    }
    t
}

fn edge(rng: &mut Lcg, infer: InferenceType) -> CausalEdge64 {
    CausalEdge64::pack(
        rng.below(256) as u8,
        rng.below(256) as u8,
        rng.below(256) as u8,
        (128 + rng.below(128)) as u8,
        (128 + rng.below(100)) as u8,
        CausalMask::PO,
        0b101,
        infer,
        PlasticityState::S_HOT,
        0,
    )
}

/// Time the promised kernel alone. Returns (steps/ms, ns/step).
fn measure_promised_step(iters: usize) -> (f64, f64) {
    let tables = NarsTables::build(1);
    let c = build_compose();
    let mut rng = Lcg(0x051E_D270_B5A1_11E5);
    let weights: Vec<CausalEdge64> = (0..64)
        .map(|_| edge(&mut rng, InferenceType::Deduction))
        .collect();
    let mut running = edge(&mut rng, InferenceType::Revision);

    let t0 = Instant::now();
    for i in 0..iters {
        let w = weights[i & 63];
        running = promised_step(running, w, &tables, &c[0], &c[1], &c[2]);
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(running.0);
    (iters as f64 / dt / 1000.0, dt * 1e9 / iters as f64)
}

/// Time the mask half BOTH ways: the shipped allocation-free `[u64; 64]`, and
/// the v1 `Vec<u64>` shape, so the allocation share the codex review named is
/// reported rather than assumed.
fn measure_mask_both(iters: usize) -> (f64, f64, f64, f64) {
    let mut rng = Lcg(0x0000_DDBA_11C0_FFEE);
    let a = dense_mask(&mut rng, 3);
    let b = dense_mask(&mut rng, 3);

    let t0 = Instant::now();
    let mut sink = 0u32;
    for _ in 0..iters {
        let m = std::hint::black_box(&a).intersection(std::hint::black_box(&b));
        sink += popcount(&m);
    }
    let fixed_ns = t0.elapsed().as_secs_f64() * 1e9 / iters as f64;
    std::hint::black_box(sink);

    let (va, vb) = (a.to_vec(), b.to_vec());
    let t1 = Instant::now();
    let mut sink2 = 0u32;
    for _ in 0..iters {
        // the v1 shape: a fresh Vec per intersection
        let m: Vec<u64> = std::hint::black_box(&va)
            .iter()
            .zip(std::hint::black_box(&vb).iter())
            .map(|(x, y)| x & y)
            .collect();
        sink2 += m.iter().map(|w| w.count_ones()).sum::<u32>();
    }
    let vec_ns = t1.elapsed().as_secs_f64() * 1e9 / iters as f64;
    std::hint::black_box(sink2);

    (1e6 / fixed_ns, fixed_ns, 1e6 / vec_ns, vec_ns)
}

/// A full scan: replay every chain once, promised kernel, no frontier.
fn measure_full_scan(chains: usize, chain_len: usize, step_ns: f64) -> f64 {
    (chains * chain_len) as f64 * step_ns / 1e6 // ms
}

/// One frontier decision at a FIXED candidate width: score every candidate
/// observation by the split it would produce (the W4/W5 shape).
fn measure_frontier_decision(observations: usize) -> f64 {
    let mut rng = Lcg(0x0D15_EC40_11EA_DBEE);
    let live = all_ones();
    let obs: Vec<Mask> = (0..observations).map(|_| dense_mask(&mut rng, 4)).collect();

    let t0 = Instant::now();
    let mut best = (u32::MAX, 0usize);
    for (i, o) in obs.iter().enumerate() {
        let kept = popcount(&live.intersection(o));
        let dropped = popcount(&live.difference(o));
        let imbalance = kept.abs_diff(dropped);
        if imbalance < best.0 {
            best = (imbalance, i);
        }
    }
    std::hint::black_box(best);
    t0.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    println!("D-DCR-0 (W0) — replay budget  [CORRECTED after codex review on #1118]");
    println!("corpus (read from dismech-causality-v3-v1 §3a/§11, not re-derived):");
    println!("  oracle {ORACLE_EDGES} edges / {ORACLE_DISEASES} diseases · restraint {RESTRAINT_ROWS} · unknown {UNKNOWN_ROWS}");

    // ── 1. the PROMISED kernel ───────────────────────────────────────────
    println!("\n1. PROMISED CHAIN STEP — NarsTables::revise + CausalEdge64::forward");
    let (steps_per_ms, step_ns) = measure_promised_step(2_000_000);
    println!("   {steps_per_ms:>10.0} steps/ms   ({step_ns:.1} ns/step)");
    println!("   (v1 timed NarsTruth::revision f32 instead — a substitute the plan never named)");

    // ── 2. the mask half, both shapes ────────────────────────────────────
    println!("\n2. MASK HALF at 4096 bits — shipped [u64; 64] vs the v1 Vec shape");
    let (fixed_pm, fixed_ns, vec_pm, vec_ns) = measure_mask_both(1_000_000);
    println!(
        "   [u64; 64] (EvidenceMask, alloc-free): {fixed_pm:>9.0} ops/ms  ({fixed_ns:>6.1} ns)"
    );
    println!("   Vec<u64>  (v1 shape, allocates)     : {vec_pm:>9.0} ops/ms  ({vec_ns:>6.1} ns)");
    println!(
        "   => allocation was {:.1}x of the v1 mask cost — charged to arithmetic a tile would accelerate",
        vec_ns / fixed_ns
    );

    // ── 3. kernel split, re-derived from the corrected numbers ───────────
    println!("\n3. KERNEL SPLIT (the ALU wave's actual question), corrected");
    let (dominant, factor) = if fixed_ns > step_ns {
        ("MASK", fixed_ns / step_ns)
    } else {
        ("STEP", step_ns / fixed_ns)
    };
    println!("   promised step {step_ns:.1} ns  vs  alloc-free 4096-bit mask {fixed_ns:.1} ns");
    println!("   => {dominant} dominates by {factor:.2}x");
    println!("   v1 reported MASK by 3.8x on the allocating shape — SUPERSEDED by this line.");

    // ── 4. KILL check at BOTH scales, one candidate width ────────────────
    println!(
        "\n4. KILL CHECK — pre-registered 10^5 AND the real corpus, at {} candidate bits",
        MASK_WORDS * 64
    );
    let dec_ms = measure_frontier_decision(64);
    for (label, chains) in [
        ("pre-registered", PREREGISTERED_SCAN_CHAINS),
        ("real oracle arm", ORACLE_EDGES),
    ] {
        let scan_ms = measure_full_scan(chains, 4, step_ns);
        let fires = scan_ms <= dec_ms;
        println!(
            "   {label:>16}: scan {chains:>6} chains = {scan_ms:>9.3} ms  vs  decision {dec_ms:.3} ms  => KILL {}",
            if fires { "FIRES" } else { "does not fire" }
        );
    }
    let per_chain_ms = measure_full_scan(1, 4, step_ns);
    println!(
        "   crossover: a scan costs one decision at ~{:.0} chains",
        dec_ms / per_chain_ms
    );

    // ── 5. ALU BUY threshold from the corrected numbers ──────────────────
    println!("\n5. ALU BUY THRESHOLD (deferred p64 64x64 wave)");
    let corpus_ms = measure_full_scan(ORACLE_EDGES, 16, step_ns);
    println!(
        "   whole oracle arm at chain len 16: {corpus_ms:.2} ms  ({steps_per_ms:.0} steps/ms)"
    );
    println!(
        "   BUY only when a workload sustains > {:.0} steps/ms (10x this corpus in one budget).",
        steps_per_ms * 10.0
    );
    if dominant == "STEP" {
        println!("   ⚠ DIRECTION: the STEP half now dominates, so a mask-only tile is NOT the");
        println!("     first lever — the packed-edge/table path is. Re-aim before any BUY.");
    } else {
        println!("   DIRECTION: the mask half dominates even allocation-free — a 64x64 tile is aimed right.");
    }
}
