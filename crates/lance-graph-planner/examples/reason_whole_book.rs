//! `reason_whole_book` — run the whole-book KJV SPO stream through the ACTUAL
//! dialectic reasoning layer (`nars::belief` + `nars::tactics`), and measure
//! what the thinking does at book scale.
//!
//! This is the reasoning half of the SoC seam
//! (`E-DEEPNSM-V2-IS-INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1`): the inbound
//! leg (`deepnsm-v2`, `bible_wave --export`) emits the whole-book belief stream;
//! THIS example (the reasoning layer) consumes it into a `BeliefArena` and runs
//! copula-gated transitive closure + the five tactics.
//!
//! Usage (deepnsm-v2 is a workspace-EXCLUDED crate → `--manifest-path`, not `-p`):
//! ```text
//! cargo run --release --manifest-path crates/deepnsm-v2/Cargo.toml \
//!     --example bible_wave -- /tmp/pg10.txt --export /tmp/kjv_spo.tsv
//! cargo run -p lance-graph-planner --release --example reason_whole_book -- /tmp/kjv_spo.tsv
//! ```
//!
//! The falsifiers (what a whole-book run tests that the unit tests cannot):
//! - **F1 — copula gating at scale (S3):** verbs (`Rel`) must NOT transit; only
//!   `is/was/are` (`Inh`) compose. If closure derived verb chains, S3 failed.
//! - **F2 — termination + no explosion:** `close_transitive` (uncapped) must
//!   reach a fixed point at book scale, not blow up. If it doesn't, the arena
//!   needs a derivation-horizon cap (the `reason.rs` `derive_transitive_capped`
//!   lesson) — a real finding.
//! - **F3 — tactics + gaps:** the S5 throttle must keep the RCR abductive
//!   frontier finite; `ReasoningGap`s should surface where word-level SPO lacks
//!   the concept structure the tactics need (the E-HERMENEUTIK concept-lift).

use lance_graph_planner::nars::{
    cas_abstract, rcr_abduce, BeliefArena, CStmt, Copula, GapKind, Stamp, Throttle, TruthValue,
};
use std::collections::HashMap;
use std::time::Instant;

/// Copular predicates map to `Inh` (is_a, transitive); everything else is a
/// `Rel` verb (stored, NEVER transitive — the S3 gate).
fn is_copular(word: &str) -> bool {
    matches!(
        word,
        "is" | "was"
            | "are"
            | "were"
            | "be"
            | "been"
            | "being"
            | "am"
            | "art"
            | "wast"
            | "become"
            | "became"
            | "becometh"
    )
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: reason_whole_book <kjv_spo.tsv>  (from bible_wave --export)");
    let raw = std::fs::read_to_string(&path).expect("read SPO tsv");

    // ── ingest the belief stream into the arena ──
    let mut arena = BeliefArena::new();
    let (mut n_inh, mut n_rel) = (0u64, 0u64);
    let mut subj_degree: HashMap<u16, u32> = HashMap::new();
    let t_ingest = Instant::now();
    for line in raw.lines() {
        let mut f = line.split('\t');
        let (Some(s), Some(_sw), Some(_pid), Some(pw), Some(o), Some(_ow), Some(v)) = (
            f.next(),
            f.next(),
            f.next(),
            f.next(),
            f.next(),
            f.next(),
            f.next(),
        ) else {
            continue;
        };
        let (Ok(s), Ok(o), Ok(v)) = (s.parse::<u16>(), o.parse::<u16>(), v.parse::<u32>()) else {
            continue;
        };
        let cop = if is_copular(pw) {
            n_inh += 1;
            Copula::Inh
        } else {
            n_rel += 1;
            // a verb term id — kept, but S3 forbids it from ever composing.
            Copula::Rel(_pid.parse::<u16>().unwrap_or(0))
        };
        *subj_degree.entry(s).or_default() += 1;
        // Observed fact: asserted (freq 1.0), moderate confidence; stamp = verse.
        arena.observe(
            CStmt { s, cop, p: o },
            TruthValue::new(1.0, 0.9),
            Stamp::source(v),
        );
    }
    let observed = arena.entries().len();
    println!("── ingest ──");
    println!(
        "  {observed} distinct observed statements ({n_inh} is_a rows, {n_rel} verb rows) in {:?}",
        t_ingest.elapsed()
    );

    // ── F1 + F2: copula-gated transitive closure over the whole book ──
    println!("── close_transitive (S3 copula-gated) ──");
    let t_close = Instant::now();
    arena.close_transitive(64);
    let after = arena.entries().len();
    let derived = after - observed;
    // Every derived belief must be Inh (verbs never transit — F1).
    let derived_non_inh = arena.entries()[observed..]
        .iter()
        .filter(|b| b.stmt.cop != Copula::Inh)
        .count();
    let max_rung = arena.entries().iter().map(|b| b.rung).max().unwrap_or(0);
    println!(
        "  derived {derived} new statements (arena {observed} -> {after}) in {:?}",
        t_close.elapsed()
    );
    println!(
        "  passes={}  reached_fixed_point={}  max_rung={max_rung}",
        arena.passes, arena.reached_fixed_point
    );
    println!(
        "  F1 copula gate: {derived_non_inh} derived non-Inh statements (MUST be 0 — verbs never transit)"
    );
    println!(
        "  F2 termination: {}",
        if arena.reached_fixed_point {
            "PASS — reached a true fixed point (no explosion)"
        } else {
            "KILL — hit the pass cap; book-scale closure needs a horizon cap"
        }
    );

    // A few derived is_a chains (rung >= 2 = genuinely multi-hop reasoning).
    println!("── sample multi-hop derivations (rung >= 2) ──");
    let mut shown = 0;
    for b in arena.entries()[observed..].iter() {
        if b.rung >= 2 && shown < 8 {
            println!(
                "  [{}]  {} is {}  (premises {:?}, expectation {:.3})",
                b.rung,
                b.stmt.s,
                b.stmt.p,
                b.premises,
                b.truth.expectation()
            );
            shown += 1;
        }
    }
    if shown == 0 {
        println!("  (none — the is_a subset has no 2-hop chains; concept-lift needed)");
    }

    // ── F3: the RCR abductive frontier under the S5 throttle ──
    println!("── RCR abduction (S5 throttle: c_min 0.05, budget 20000, hub>32 barred) ──");
    let t_rcr = Instant::now();
    let fr = rcr_abduce(&arena, &Throttle::new(0.05, 20_000, 32));
    let mut gap_counts: HashMap<&str, usize> = HashMap::new();
    for g in &fr.gaps {
        *gap_counts.entry(gap_name(g.kind)).or_default() += 1;
    }
    println!(
        "  {} candidates in {:?}; gaps: {:?}",
        fr.candidates.len(),
        t_rcr.elapsed(),
        gap_counts
    );

    // ── F3: CAS abstraction over the highest-degree subjects (S5-throttled) ──
    println!("── CAS abstraction over the top-10 highest-degree subjects (S5 throttle) ──");
    let mut top: Vec<(u16, u32)> = subj_degree.into_iter().collect();
    top.sort_unstable_by_key(|b| std::cmp::Reverse(b.1));
    let cas_throttle = Throttle::new(0.05, 20_000, 32);
    let (mut cas_cands, mut cas_hub, mut cas_budget) = (0usize, 0usize, 0usize);
    let t_cas = Instant::now();
    for &(s, _) in top.iter().take(10) {
        let fr = cas_abstract(&arena, s, &cas_throttle);
        cas_cands += fr.candidates.len();
        cas_hub += fr
            .gaps
            .iter()
            .filter(|g| g.kind == GapKind::HubExcluded)
            .count();
        cas_budget += fr
            .gaps
            .iter()
            .filter(|g| g.kind == GapKind::BudgetExhausted)
            .count();
    }
    println!(
        "  {cas_cands} candidates across the top-10 subjects in {:?} (was ~2M unthrottled); {cas_hub} hub parents barred, {cas_budget} budget-capped",
        t_cas.elapsed()
    );

    println!("── done — the whole book ran through the actual reasoning layer ──");
}

fn gap_name(k: GapKind) -> &'static str {
    match k {
        GapKind::NoSharedMiddle => "NoSharedMiddle",
        GapKind::NoSibling => "NoSibling",
        GapKind::NoAbstraction => "NoAbstraction",
        GapKind::HubExcluded => "HubExcluded",
        GapKind::BudgetExhausted => "BudgetExhausted",
    }
}
