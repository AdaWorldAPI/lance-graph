//! PROBE recipe_loci_walk — Door C (the 24-dimension organ gate) over a
//! **Maslow rung ascent**, measured against the shipped Door B (the surprise
//! selector).
//!
//! Answers, on runnable code: *does gating the 34 recipes on the real
//! [`CausalWitnessFacet`](lance_graph_contract::causal_witness) 24-loci organ
//! (a) reach recipes the selector cannot, (b) carry lower-rung awareness UP so a
//! deeper recipe re-derives nothing, and (c) let higher thinking prune lower
//! related work* — the #780 Axis-B close on the dispatch path.
//!
//! ## What it measures (nothing planted)
//!
//! 1. **Door B reach (MEASURED):** sweep the shipped
//!    [`materialize::select_tactic`](lance_graph_contract::materialize::select_tactic)
//!    across a grid of `ThoughtCtx` (free_energy × dissonance × sd-gate × rung)
//!    and collect the distinct winners — the selector's real reachable set.
//! 2. **Door C reach (MEASURED):** a **rung ascent** — a sequence of witnesses
//!    that bind progressively more loci as the Maslow climb deepens (Surface
//!    grounds the shallow loci; the counterfactual apex grounds Kausal +
//!    Contradiction + S/P/O). At each step: how many recipes the organ reaches,
//!    the carried-awareness set, and the active set after higher-prunes-lower.
//!
//! ## Registered gates (fixed before first run)
//!
//! 1. Door B reaches a STRICT SUBSET of 34 (the selector shadows the rest —
//!    `E-RECIPE-SELECTOR-REACHABILITY-1`).
//! 2. Door C at full grounding reaches ALL 34 (the organ, not the style label).
//! 3. Carry is monotone: the carried-awareness set never shrinks as the climb
//!    deepens.
//! 4. Prune fires: under full grounding the active set is a strict subset of the
//!    reachable set (higher thinking subsumes lower related), and the apex ICR
//!    #31 is never pruned.
//!
//! KILL: any gate fails (recorded loudly).
//!
//! ## Run
//! ```bash
//! cargo run -p lance-graph-contract --example recipe_loci_walk
//! ```

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::materialize::select_tactic;
use lance_graph_contract::recipe_kernels::ThoughtCtx;
use lance_graph_contract::recipe_loci::{
    active_after_prune, carried_awareness, reachable, required_loci, rung_level,
};
use std::collections::HashSet;

/// A witness that binds every locus at or below a given Maslow depth. The ascent
/// grounds loci in cognitive order: shallow observation/meaning first, then
/// consensus, then the counterfactual causal apex — so a deeper step is a strict
/// superset of a shallower one (the #778 "more loci bound, shape unchanged").
fn witness_for_depth(depth: usize) -> CausalWitnessFacet {
    // loci grouped by the pyramid level at which they first become available.
    let tiers: [&[Locus]; 5] = [
        &[
            Locus::Temporal,
            Locus::SMeaning,
            Locus::PMeaning,
            Locus::OMeaning,
        ], // observe/ground
        &[
            Locus::BasinAnchor,
            Locus::Supports,
            Locus::Antecedent,
            Locus::Lokal,
        ], // relate/basin
        &[
            Locus::MeaningLevel,
            Locus::RunbookEvidence,
            Locus::QualiaReference,
            Locus::Modal,
        ], // meta/texture
        &[Locus::Quorum, Locus::SupportedBy],   // consensus
        &[Locus::Kausal, Locus::Contradiction], // causal / counterfactual apex
    ];
    let mut w = CausalWitnessFacet::ZERO;
    for tier in tiers.iter().take(depth + 1) {
        for &l in *tier {
            w = w.with(l, -1); // one event back (bound)
        }
    }
    w
}

fn main() {
    // ── Door B (MEASURED): sweep the shipped selector ───────────────────────
    let mut selector_reach: HashSet<u8> = HashSet::new();
    for fe in [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0] {
        for diss in [0.0f32, 0.5, 0.9] {
            for sd in [-1.0f32, 0.0, 1.0] {
                for rung in [0u8, 3, 5, 7, 9] {
                    let ctx = ThoughtCtx {
                        sd,
                        free_energy: fe,
                        dissonance: diss,
                        temperature: 0.5,
                        confidence: 0.5,
                        rung,
                        candidates: vec![0.5],
                        beliefs: vec![(7, 0.9, 0.8)],
                    };
                    selector_reach.insert(select_tactic(&ctx));
                }
            }
        }
    }
    println!(
        "Door B (surprise selector, MEASURED over {} ctx): reaches {}/34 distinct recipes {:?}",
        6 * 3 * 3 * 5,
        selector_reach.len(),
        {
            let mut v: Vec<u8> = selector_reach.iter().copied().collect();
            v.sort_unstable();
            v
        }
    );

    // ── Door C (MEASURED): the Maslow rung ascent over the organ ────────────
    println!("\nDoor C (24-loci organ gate) — the Maslow rung ascent:");
    println!(
        "{:<7} {:>6} {:>9} {:>7} {:>7}   newly-carried loci",
        "depth", "bound", "reachable", "carried", "active"
    );
    let mut prev_carried: HashSet<Locus> = HashSet::new();
    let mut carry_monotone = true;
    let mut full_reach = 0usize;
    let mut full_active = 0usize;
    let mut apex_active = false;
    for depth in 0..5 {
        let w = witness_for_depth(depth);
        let bound = Locus::ALL.iter().filter(|&&l| w.is_bound(l)).count();
        let reach = reachable(&w);
        // carried awareness at the deepest rung present (rung 9 = whole pyramid).
        let carried: Vec<Locus> = carried_awareness(&w, 9);
        let carried_set: HashSet<Locus> = carried.iter().copied().collect();
        let active = active_after_prune(&w);
        // monotonicity of carry across the ascent
        if !prev_carried.is_subset(&carried_set) {
            carry_monotone = false;
        }
        let newly: Vec<&str> = carried
            .iter()
            .filter(|l| !prev_carried.contains(l))
            .map(|l| l.label())
            .collect();
        println!(
            "{:<7} {:>6} {:>9} {:>7} {:>7}   {}",
            depth,
            bound,
            reach.len(),
            carried.len(),
            active.len(),
            newly.join(",")
        );
        prev_carried = carried_set;
        if depth == 4 {
            full_reach = reach.len();
            full_active = active.len();
            apex_active = active.contains(&31);
        }
    }

    // ── Maslow rung naming of a few recipes ─────────────────────────────────
    println!("\nMaslow rung of sample recipes (shipped RungLevel vocabulary):");
    for id in [5u8, 25, 4, 31] {
        println!(
            "  recipe #{id:<2} → rung_level {:?}  (reads {:?})",
            rung_level(id),
            required_loci(id)
        );
    }

    // ── Registered gates ────────────────────────────────────────────────────
    println!("\n== gates ==");
    let g1 = selector_reach.len() < 34;
    let g2 = full_reach == 34;
    let g3 = carry_monotone;
    let g4 = full_active < 34 && apex_active;
    println!("gate 1 selector shadows some (reach < 34): {g1}");
    println!("gate 2 organ reaches all 34 when grounded:  {g2}");
    println!("gate 3 carry is monotone up the pyramid:    {g3}");
    println!("gate 4 prune fires + apex ICR#31 survives:  {g4}");
    assert!(g1, "KILL gate 1: selector unexpectedly reaches all 34");
    assert!(
        g2,
        "KILL gate 2: organ gate does not reach 34 when fully grounded"
    );
    assert!(
        g3,
        "KILL gate 3: carried awareness shrank up the climb (rediscovery)"
    );
    assert!(
        g4,
        "KILL gate 4: prune inert, or the apex recipe was pruned"
    );

    println!(
        "\nPASS — Door C gates the Maslow climb on the REAL 24-loci organ: the selector\n\
         shadows {}/34; the organ reaches all 34 when grounded; lower-rung awareness is\n\
         carried UP (monotone, no rediscovery); higher thinking prunes lower-related.",
        34 - selector_reach.len()
    );
}
