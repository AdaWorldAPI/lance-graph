//! `F-PARALLEL-RUNG-1` — the constructive half: does ONE problem context carry
//! meaningful, simultaneously-observable activity at several distinct rungs?
//!
//! # What this establishes, and what it does not
//!
//! **Establishes (representation):** one `BeliefArena` — one problem, one
//! sealed context — holds beliefs at rungs 0, 1 and 2 at the same instant
//! (measured 4 / 3 / 3), all readable, none demoted by the arrival of a
//! higher one.
//!
//! Measured note, recorded rather than smoothed over: the 5-node chain admits
//! a rung-3 derivation of `1->5` (via `1->4` at rung 2), but the closure
//! reaches `1->5` first through `1->3` + `3->5` (both rung 1), so it is
//! admitted at rung 2 and `S2` uniqueness keeps it there. The observed
//! ceiling is therefore 2, not 3. The gates assert what ran.
//!
//! **Does NOT establish (execution):** wall-clock or thread parallelism.
//! `close_transitive` runs a sequential fixpoint on one thread. The claim here
//! is SEMANTIC coexistence — the state contains work at several rungs — not
//! concurrent execution. Nothing here measures CPU concurrency and nothing
//! here should be cited as if it did.
//!
//! # Why the arena, and not two `ShaderDriver`s
//!
//! Two drivers with two `RungElevator.level` values would only prove
//! LOCALITY (a scalar field is per-instance). That is the wrong evidence: it
//! says nothing about whether one problem can hold multiple rungs. The
//! dominant shipped representation is the per-item rung tag, so the fixture
//! is the per-item tag: `Belief::rung` (`nars/belief.rs:98`), Tarski-shaped —
//! *"0 observed; derived = `max(premise rungs)+1`, fixed at creation —
//! revision does NOT change it"*.
//!
//! # Production types only
//!
//! `BeliefArena`, `CStmt`, `Copula`, `Stamp`, `TruthValue`, `observe`,
//! `revise_at`, `close_transitive`. No new scheduler, no layer stack, no
//! attention type, no mask ABI.

use lance_graph_planner::nars::belief::{BeliefArena, CStmt, Copula, Stamp};
use lance_graph_planner::nars::truth::TruthValue;

/// `s -> p` as an Inheritance statement (the copula that transits).
fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}

/// Every `(statement, rung, truth)` currently in the arena, sorted for a
/// stable byte-comparable snapshot.
fn snapshot(a: &BeliefArena) -> Vec<((u16, u16), u32, (f32, f32))> {
    let mut v: Vec<_> = a
        .entries()
        .iter()
        .map(|b| {
            (
                (b.stmt.s, b.stmt.p),
                b.rung,
                (b.truth.frequency, b.truth.confidence),
            )
        })
        .collect();
    v.sort_by_key(|(k, r, _)| (*k, *r));
    v
}

fn rungs_present(a: &BeliefArena) -> Vec<u32> {
    let mut r: Vec<u32> = a.entries().iter().map(|b| b.rung).collect();
    r.sort_unstable();
    r.dedup();
    r
}

fn main() {
    let mut gates: Vec<(&str, bool, String)> = Vec::new();

    // ── The fixture: a 5-link observed chain, all at rung 0 ────────────
    // 1->2->3->4->5. Transitive closure derives 1->3 / 2->4 / 3->5 at rung 1,
    // then 1->4 / 2->5 / 1->5 at rung 2 (1->5 via the two rung-1 halves, which
    // is why the ceiling is 2 — see the module note).
    let mut arena = BeliefArena::new();
    for (i, (s, p)) in [(1u16, 2u16), (2, 3), (3, 4), (4, 5)].iter().enumerate() {
        arena.observe(
            inh(*s, *p),
            TruthValue::new(1.0, 0.9),
            Stamp::source(i as u32),
        );
    }
    let observed_snapshot = snapshot(&arena);
    let observed_count = arena.entries().len();

    arena.close_transitive(16);

    let after = snapshot(&arena);
    let present = rungs_present(&arena);

    // ── G1: at least three distinct rungs coexist in ONE arena ────────
    gates.push((
        "G1 one problem context holds >= 3 distinct rungs simultaneously",
        present.len() >= 3,
        format!("rungs present = {present:?}"),
    ));

    // ── G2: anti-vacuity — the spread is real, not two adjacent rungs ──
    let spread = present.last().copied().unwrap_or(0) - present.first().copied().unwrap_or(0);
    gates.push((
        "G2 the rung spread is >= 2 (not a trivial 0/1 pair)",
        spread >= 2,
        format!(
            "min={} max={} spread={spread}",
            present.first().copied().unwrap_or(0),
            present.last().copied().unwrap_or(0)
        ),
    ));

    // ── G3: each rung band carries MEANINGFUL content (non-empty) ─────
    let per_band: Vec<(u32, usize)> = present
        .iter()
        .map(|r| (*r, arena.entries().iter().filter(|b| b.rung == *r).count()))
        .collect();
    gates.push((
        "G3 every present rung band is non-empty (meaningful, not a label)",
        per_band.iter().all(|(_, n)| *n >= 1) && per_band.len() >= 3,
        format!("{per_band:?}"),
    ));

    // ── G4: the rung-0 contributions survive the higher rungs' arrival ─
    // Requirement 5: creating higher-rung work must NOT delete, overwrite,
    // invalidate, or demote the lower-rung work.
    let observed_still: Vec<_> = after.iter().filter(|(_, r, _)| *r == 0).cloned().collect();
    gates.push((
        "G4 every rung-0 belief survives byte-identical after closure",
        observed_still == observed_snapshot && observed_still.len() == observed_count,
        format!(
            "{} observed before, {} identical rung-0 entries after",
            observed_count,
            observed_still.len()
        ),
    ));

    // ── G5: revision at a low rung does not move that belief's rung ────
    // The S2 rung-inflation fix, exercised rather than quoted.
    let id = arena
        .entries()
        .iter()
        .position(|b| b.rung == 0)
        .expect("a rung-0 belief exists") as u32;
    let rung_before = arena.entries()[id as usize].rung;
    let _ = arena.revise_at(id, TruthValue::new(0.8, 0.85), Stamp::source(99));
    let rung_after = arena.entries()[id as usize].rung;
    gates.push((
        "G5 revising a rung-0 belief leaves its rung at 0 (no inflation, no demotion)",
        rung_before == 0 && rung_after == 0,
        format!("rung {rung_before} -> {rung_after}"),
    ));

    // ── G6: the high-rung band is still there after that revision ──────
    let present_after_revision = rungs_present(&arena);
    gates.push((
        "G6 revising a low rung does not collapse the high-rung bands",
        present_after_revision.len() >= 3,
        format!("rungs after revision = {present_after_revision:?}"),
    ));

    // ── G7: CAN-IT-STAY-SILENT — a single observation is single-rung ───
    // Without this, G1 would be true of any arena and would carry no
    // information (the workspace falsifiability rule).
    let mut lone = BeliefArena::new();
    lone.observe(inh(7, 8), TruthValue::new(1.0, 0.9), Stamp::source(0));
    lone.close_transitive(16);
    let lone_rungs = rungs_present(&lone);
    gates.push((
        "G7 control: a single observation yields exactly ONE rung (G1 is not vacuous)",
        lone_rungs == vec![0],
        format!("lone arena rungs = {lone_rungs:?}"),
    ));

    println!("═══ F-PARALLEL-RUNG-1 (constructive) ═══\n");
    println!("  one BeliefArena, one problem, {observed_count} observed links");
    for (r, n) in &per_band {
        println!("    rung {r}: {n} belief(s)");
    }
    println!();

    let mut all_green = true;
    for (name, pass, detail) in &gates {
        println!(
            "  [{}] {name} — {detail}",
            if *pass { "PASS" } else { "FAIL" }
        );
        all_green &= *pass;
    }
    println!(
        "\nSCOPE: representation permits coexistence — PROVEN here.\n\
         Concurrent EXECUTION (threads/wall-clock) — NOT measured, not claimed.\n\
         close_transitive is a sequential fixpoint on one thread."
    );
    assert!(all_green, "F-PARALLEL-RUNG-1: a gate failed — see above");
    println!("\nF-PARALLEL-RUNG-1: ALL GATES GREEN");
}
