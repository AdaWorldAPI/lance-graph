//! `multihop_witness_confidence` — W-C on the **V3 substrate**: a fact's Truth is
//! NARS-revised across its **multi-hop witness chain**, using the u8 NARS truth
//! ALREADY packed in `CausalEdge64` + the `WitnessTable` W-slot arc. Richer than
//! the single-fact re-observation of `self_correcting_kg`: a belief supported by a
//! *longer chain of independent witnesses* is more confident. The shared
//! predicate:object is the **episodic-basin anchor** (`part_of:is_a`); the witness
//! entries are its **supporting edges**.
//!
//! Operator direction: *"episodic basins get supporting edges from witness entries
//! and implement both anchors and nars Truth/frequency value already present in
//! causaledge64 … Truth > multi-hop witness confidence … richer expressiveness in
//! v3."*
//!
//! ```sh
//! cargo run -p lance-graph --example multihop_witness_confidence
//! ```
//!
//! Scope: composes the SHIPPED primitives (`CausalEdge64` u8 truth + `WitnessTable`
//! + NARS revision) to prove the multi-hop-witness → Truth mechanism. The
//! contract-level wiring (edge→W-slot in the `MailboxSoA` emission path) is the
//! gated V3 slice; here the arc is keyed by fact index (feature-agnostic).

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_contract::witness_table::{WitnessEntry, WitnessTable};

/// One NARS revision step over f32 truth (read from each edge's u8 truth via
/// `.frequency()` / `.confidence()`): independent evidence raises confidence
/// `w/(w+k)`. Frequencies weight-average by evidence weight `c/(1-c)`.
fn revise(f1: f32, c1: f32, f2: f32, c2: f32) -> (f32, f32) {
    let w1 = c1 / (1.0 - c1 + f32::EPSILON);
    let w2 = c2 / (1.0 - c2 + f32::EPSILON);
    let w = w1 + w2;
    let f = if w > f32::EPSILON {
        (w1 * f1 + w2 * f2) / w
    } else {
        0.5
    };
    (f, w / (w + 1.0))
}

fn main() {
    // One episodic basin: shared predicate(20):object(30) = the part_of:is_a rail.
    // Each fact is a CausalEdge64 carrying a MODEST single-observation truth in its
    // OWN u8 freq/conf (≈1.0 / ≈0.5). Fact i is witnessed by fact i-1 — the W-slot
    // arc; `WitnessEntry.spo_fact_ref` is the hop back to the prior belief.
    const N: usize = 6;
    let freq0: u8 = 255; // frequency ≈ 1.0
    let conf0: u8 = 128; // confidence ≈ 0.5 (a single observation)

    let mut facts: Vec<CausalEdge64> = Vec::new();
    let mut witnesses: WitnessTable<64> = WitnessTable::new();
    for i in 0..N {
        let prior = if i == 0 { None } else { Some((i - 1) as u64) };
        witnesses
            .set(
                i as u8,
                WitnessEntry {
                    mailbox_ref: 1000 + i as u32,
                    spo_fact_ref: prior,
                },
            )
            .expect("slot in cohort range");
        facts.push(CausalEdge64::pack(
            10 + i as u8, // distinct subject
            20,           // shared predicate ┐ episodic-basin anchor
            30,           // shared object    ┘ (part_of:is_a)
            freq0,
            conf0,
            CausalMask::SPO,
            0, // direction
            InferenceType::Revision,
            PlasticityState::from_bits(0b111), // all hot
            0,
        ));
    }

    // Walk the multi-hop witness arc back to the root, NARS-revising the truth with
    // each witnessing fact. THIS is "Truth from multi-hop witness confidence": more
    // independent witnesses in the chain → higher belief. The truth read at every
    // hop is the u8 value already in that `CausalEdge64`.
    let multihop = |start: usize| -> (f32, usize) {
        let (mut f, mut c) = (facts[start].frequency(), facts[start].confidence());
        let (mut slot, mut hops) = (start, 1usize);
        while let Some(prior) = witnesses.get(slot as u8).and_then(|e| e.spo_fact_ref) {
            let p = prior as usize;
            let (nf, nc) = revise(f, c, facts[p].frequency(), facts[p].confidence());
            f = nf;
            c = nc;
            slot = p;
            hops += 1;
        }
        (c, hops)
    };

    println!("── multihop_witness_confidence : one episodic basin (p:o = 20:30) ──");
    println!(
        "each fact carries u8 NARS truth in its own CausalEdge64 (freq≈{:.2}, conf≈{:.3})\n",
        facts[0].frequency(),
        facts[0].confidence()
    );
    println!("  fact   single-obs conf   multi-hop conf   witness hops");
    for i in 0..N {
        let single = facts[i].confidence();
        let (mc, hops) = multihop(i);
        println!("  F{i}      {single:.3}             {mc:.3}            {hops}");
    }

    println!("\n── the point (endgame: Truth > multi-hop witness confidence, richer in v3) ──");
    let (c_root, _) = multihop(0);
    let (c_deep, h_deep) = multihop(N - 1);
    println!(
        "A belief witnessed only by itself (F0) stays at conf {c_root:.3}; the SAME belief supported"
    );
    println!(
        "by a {h_deep}-deep witness chain (F{}) rises to conf {c_deep:.3} — reading the basin's supporting",
        N - 1
    );
    println!(
        "witnesses makes the Truth *surer*. Truth lives in CausalEdge64's u8 freq/conf (already"
    );
    println!(
        "present); the WitnessTable W-slot arc is the multi-hop support; the shared p:o is the"
    );
    println!(
        "part_of:is_a basin anchor. This is the richer v3 expressiveness over `self_correcting_kg`'s"
    );
    println!(
        "single-fact revision — the same NARS math, now driven by the episodic witness chain."
    );
}
