//! `dispatch_guard_redundancy` — MEASURE the two organ resolutions as genuinely
//! independent gates: single-pass structural BINDING vs the multipass Markov
//! STANDING WAVE (local-causal vs beyond-the-reference-horizon persistence).
//! Operator ruling: the grounding gate must be the literal Markov-chain resolution
//! (the standing wave), NOT a coarse scalar shadow — and the `±8` window is only
//! the REFERENCE HORIZON, so a chain that leaves it is not coincidental, it
//! ESCALATES to search causality over time (Romeo & Juliet's death is still caused
//! by the distant feud; `E-HORIZON-NOT-BOUND-1`).
//!
//! Two windows over the SAME rich locus set (every named locus bound), differing
//! ONLY in the chain structure the standing wave reads:
//!   * `local`  — chains terminate inside `±8` → the wave settles → loci Causal;
//!   * `beyond` — chains re-extend out of the `±8` reference horizon → the wave
//!     ESCALATES (bound, but the cause is non-local → search over time / the
//!     absolute AriGraph basin).
//!
//! The single-pass binding gate sees BOTH windows as identical (all loci bound →
//! would fire every recipe). The standing wave separates them. The count of
//! recipes whose verdict FLIPS `Fires → Escalate` between the two windows IS the
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
    // local: offset +1 to a terminal peer → the wave settles inside ±8.
    let local_focal = all_loci(1);
    let local_peer = CausalWitnessFacet::ZERO; // no rebind → terminal
    let local = [(0usize, local_focal), (1usize, local_peer)];
    // beyond: offset +7 to a peer that rebinds +7 → chain leaves ±8 → escalate.
    let beyond_focal = all_loci(7);
    let beyond_peer = all_loci(7);
    let beyond = [(0usize, beyond_focal), (7usize, beyond_peer)];

    let passes = 8u8;

    // Single-pass gate: identical on both windows (all loci bound).
    let single_pass_fires_local = (1..=34u8)
        .filter(|&id| loci_disqualifier(&local_focal, id).is_none())
        .count() as u32;
    let single_pass_fires_beyond = (1..=34u8)
        .filter(|&id| loci_disqualifier(&beyond_focal, id).is_none())
        .count() as u32;

    let mut fires_local = 0u32;
    let mut fires_beyond = 0u32;
    let mut escalate_beyond = 0u32;
    let mut flipped = 0u32; // Fires in local, Escalate in beyond
    for id in 1..=34u8 {
        let vl = guard(None, &local, 0, id, passes).outcome;
        let vb = guard(None, &beyond, 0, id, passes).outcome;
        if vl == GateOutcome::Fires {
            fires_local += 1;
        }
        match vb {
            GateOutcome::Fires => fires_beyond += 1,
            GateOutcome::Escalate => escalate_beyond += 1,
            GateOutcome::Unbound => {}
        }
        if vl == GateOutcome::Fires && vb == GateOutcome::Escalate {
            flipped += 1;
        }
    }

    println!("── single-pass structural binding gate (loci_disqualifier) ──");
    println!("  local window:  fires {single_pass_fires_local}/34");
    println!("  beyond window: fires {single_pass_fires_beyond}/34   (IDENTICAL — binding can't tell them apart)");
    println!("\n── + multipass Markov standing wave gate ──");
    println!("  local window:  fires {fires_local}/34   (chains settle inside ±8 → Causal)");
    println!("  beyond window: fires {fires_beyond}/34, Escalate {escalate_beyond}/34   (chains leave the horizon → search over time)");
    println!("\n  → verdict flips Fires→Escalate between the two windows: {flipped}/34");
    println!("    (the wave's independent discrimination — a coarse pre-filter gives 0)");

    // ═══ registered gates ═══
    println!("\n── gates ──");
    let mut green = true;

    // G1: the single-pass gate is BLIND to the local/beyond difference
    //     (fires the same count on both windows) — the coarse gate's limit.
    let g1 = single_pass_fires_local == single_pass_fires_beyond && single_pass_fires_local > 0;
    println!(
        "[{}] G1 single-pass binding is identical on both windows (blind to persistence)",
        pf(g1)
    );
    green &= g1;

    // G2: the standing wave DISCRIMINATES — recipes that fire on the local window
    //     escalate on the beyond one (flips > 0).
    let g2 = flipped > 0;
    println!(
        "[{}] G2 the standing wave flips {flipped} recipes Fires→Escalate (independent catch)",
        pf(g2)
    );
    green &= g2;

    // G3: on the local window the two gates AGREE (bound + settled → fires) —
    //     the wave adds no false escalations when the chain really is local.
    let g3 = fires_local == single_pass_fires_local;
    println!(
        "[{}] G3 no false escalation: local-window fires == single-pass fires ({fires_local})",
        pf(g3)
    );
    green &= g3;

    // G4: on the beyond window the wave escalates what binding would fire
    //     (escalate == single-pass fires there) — the higher-confidence gate.
    let g4 = escalate_beyond == single_pass_fires_beyond && escalate_beyond > 0;
    println!(
        "[{}] G4 beyond-window: wave escalates all the single-pass would fire ({escalate_beyond})",
        pf(g4)
    );
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
