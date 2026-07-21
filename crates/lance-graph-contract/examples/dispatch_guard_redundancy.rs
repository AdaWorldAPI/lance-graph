//! `dispatch_guard_redundancy` — MEASURE the two organ resolutions as genuinely
//! independent gates: single-pass structural BINDING vs the multipass Markov
//! STANDING WAVE (causal-vs-coincidental persistence). Operator ruling: the
//! grounding gate must be the literal Markov-chain resolution (the standing
//! wave), NOT a coarse scalar shadow.
//!
//! Two windows over the SAME rich locus set (every named locus bound), differing
//! ONLY in the chain structure the standing wave reads:
//!   * `causal`       — chains terminate → the wave settles → loci Causal;
//!   * `coincidental` — chains re-extend out of the `±8` window → the wave never
//!     settles → loci Coincidental (bound, but causally hollow).
//!
//! The single-pass binding gate sees BOTH windows as identical (all loci bound →
//! would fire every recipe). The standing wave separates them. The count of
//! recipes whose verdict FLIPS `Fires → WaveCatch` between the two windows IS the
//! wave's independent discriminating power — the higher-confidence redundancy a
//! coarse pre-filter could never provide.
//!
//! ```sh
//! cargo run -p lance-graph-contract --example dispatch_guard_redundancy
//! ```

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::dispatch_guard::{guard, GateOutcome};
use lance_graph_contract::recipe_loci::loci_disqualifier;

/// A register binding every named locus to `off`.
fn all_loci(off: i8) -> CausalWitnessFacet {
    let mut w = CausalWitnessFacet::ZERO;
    for &l in Locus::ALL.iter() {
        w = w.with(l, off);
    }
    w
}

fn main() {
    println!("dispatch_guard_redundancy — single-pass binding vs multipass Markov standing wave\n");

    // Both focals bind ALL 16 named loci → the single-pass gate fires every
    // recipe in BOTH windows. Only the chain structure (what the standing wave
    // reads) differs.
    // causal: offset +1 to a terminal peer → the wave settles.
    let causal_focal = all_loci(1);
    let causal_peer = CausalWitnessFacet::ZERO; // no rebind → terminal
    let causal = [(0usize, causal_focal), (1usize, causal_peer)];
    // coincidental: offset +7 to a peer that rebinds +7 → chain leaves ±8, never settles.
    let coinc_focal = all_loci(7);
    let coinc_peer = all_loci(7);
    let coinc = [(0usize, coinc_focal), (7usize, coinc_peer)];

    let passes = 8u8;

    // Single-pass gate: identical on both windows (all loci bound).
    let single_pass_fires_causal = (1..=34u8)
        .filter(|&id| loci_disqualifier(&causal_focal, id).is_none())
        .count() as u32;
    let single_pass_fires_coinc = (1..=34u8)
        .filter(|&id| loci_disqualifier(&coinc_focal, id).is_none())
        .count() as u32;

    let mut fires_causal = 0u32;
    let mut fires_coinc = 0u32;
    let mut wavecatch_coinc = 0u32;
    let mut flipped = 0u32; // Fires in causal, WaveCatch in coincidental
    for id in 1..=34u8 {
        let vc = guard(None, &causal, 0, id, passes).outcome;
        let vx = guard(None, &coinc, 0, id, passes).outcome;
        if vc == GateOutcome::Fires {
            fires_causal += 1;
        }
        match vx {
            GateOutcome::Fires => fires_coinc += 1,
            GateOutcome::WaveCatch => wavecatch_coinc += 1,
            GateOutcome::Unbound => {}
        }
        if vc == GateOutcome::Fires && vx == GateOutcome::WaveCatch {
            flipped += 1;
        }
    }

    println!("── single-pass structural binding gate (loci_disqualifier) ──");
    println!("  causal window:       fires {single_pass_fires_causal}/34");
    println!("  coincidental window: fires {single_pass_fires_coinc}/34   (IDENTICAL — binding can't tell them apart)");
    println!("\n── + multipass Markov standing wave gate ──");
    println!("  causal window:       fires {fires_causal}/34   (chains settle → Causal)");
    println!("  coincidental window: fires {fires_coinc}/34, WaveCatch {wavecatch_coinc}/34   (chains escalate → Coincidental)");
    println!("\n  → verdict flips Fires→WaveCatch between the two windows: {flipped}/34");
    println!("    (the wave's independent discrimination — a coarse pre-filter gives 0)");

    // ═══ registered gates ═══
    println!("\n── gates ──");
    let mut green = true;

    // G1: the single-pass gate is BLIND to the causal/coincidental difference
    //     (fires the same count on both windows) — the coarse gate's limit.
    let g1 = single_pass_fires_causal == single_pass_fires_coinc && single_pass_fires_causal > 0;
    println!(
        "[{}] G1 single-pass binding is identical on both windows (blind to persistence)",
        pf(g1)
    );
    green &= g1;

    // G2: the standing wave DISCRIMINATES — recipes that fire on the causal
    //     window are WaveCaught on the coincidental one (flips > 0).
    let g2 = flipped > 0;
    println!(
        "[{}] G2 the standing wave flips {flipped} recipes Fires→WaveCatch (independent catch)",
        pf(g2)
    );
    green &= g2;

    // G3: on the causal window the two gates AGREE (bound + settled → fires) —
    //     the wave adds no false blocks when the chain really is causal.
    let g3 = fires_causal == single_pass_fires_causal;
    println!(
        "[{}] G3 no false blocks: causal-window fires == single-pass fires ({fires_causal})",
        pf(g3)
    );
    green &= g3;

    // G4: on the coincidental window the wave blocks what binding would fire
    //     (wavecatch == single-pass fires there) — the higher-confidence gate.
    let g4 = wavecatch_coinc == single_pass_fires_coinc && wavecatch_coinc > 0;
    println!("[{}] G4 coincidental-window: wave blocks all the single-pass would fire ({wavecatch_coinc})", pf(g4));
    green &= g4;

    println!(
        "\n{}",
        if green {
            "ALL GATES GREEN"
        } else {
            "GATE FAILURE"
        }
    );
    assert!(green, "dispatch-guard redundancy gates failed");
}

fn pf(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
