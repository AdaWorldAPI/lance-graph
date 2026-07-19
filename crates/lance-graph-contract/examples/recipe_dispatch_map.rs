//! `recipe_dispatch_map` — the MEASURED "when do the 34 recipes fire?" map.
//!
//! Enumerates the awareness-state space the selector actually reads — surprise
//! band (`free_energy`) × CollapseGate state (`sd`) × rung tier × contradiction
//! flag (`dissonance`) — and records which of the 34 tactic recipes
//! [`select_tactic`] dispatches in every cell, which recipes are therefore
//! REACHABLE via the surprise selector at all, and which fire only through the
//! other live wire (the planner's style→mechanism fan). Companion generator for
//! `docs/NARS_RECIPES_DISPATCH.md` — the doc's tables are this output.
//!
//! ```sh
//! cargo run -p lance-graph-contract --example recipe_dispatch_map
//! ```

use std::collections::BTreeSet;

use lance_graph_contract::materialize::{awareness_is_causal, materialize, select_tactic};
use lance_graph_contract::recipe_kernels::{
    all_kernels, ThoughtCtx, ThoughtField, SD_BLOCK, SD_FLOW,
};
use lance_graph_contract::recipes::{recipe, Mechanism, Recipe, RECIPES};

/// Render a tactic's input checklist (`requires()` mask) as short field names.
fn mask_line(k: &dyn lance_graph_contract::recipe_kernels::Tactic) -> String {
    const FIELDS: [(ThoughtField, &str); 8] = [
        (ThoughtField::Sd, "sd"),
        (ThoughtField::FreeEnergy, "F"),
        (ThoughtField::Dissonance, "diss"),
        (ThoughtField::Temperature, "temp"),
        (ThoughtField::Confidence, "conf"),
        (ThoughtField::Rung, "rung"),
        (ThoughtField::Candidates, "cand"),
        (ThoughtField::Beliefs, "belief"),
    ];
    let m = k.requires();
    FIELDS
        .iter()
        .filter(|(f, _)| m.has(*f))
        .map(|(_, n)| *n)
        .collect::<Vec<_>>()
        .join("+")
}

/// The gate state implied by dispersion (`ThoughtCtx::gate_state` is private;
/// the thresholds are public — same derivation as `materialize::gate_of`).
fn gate_name(sd: f32) -> &'static str {
    if sd < SD_FLOW {
        "FLOW "
    } else if sd <= SD_BLOCK {
        "HOLD "
    } else {
        "BLOCK"
    }
}

fn main() {
    // ── §1 the catalogue + each kernel's declared input checklist. ──
    println!(
        "── §1 the 34 recipes (catalogue + requires() checklist, from the live registry) ──\n"
    );
    println!("  id  code  name                                    tier            mechanism             bucket    2³        reads");
    for k in all_kernels() {
        let r = k.meta();
        println!(
            "  {:>2}  {:<4}  {:<38}  {:<14}  {:<20}  {:<8}  {:<8}  {}",
            r.id,
            r.code,
            r.name,
            format!("{:?}", r.tier),
            format!("{:?}", r.mechanism),
            format!("{:?}", r.bucket),
            format!("{:?}", r.spo2cubed),
            mask_line(k),
        );
    }

    // ── §2 the dispatch map: which recipe wins in every awareness cell. ──
    // Representative values per axis (the selector reads only band membership):
    // F: <0.33 routine / 0.33..0.66 inference / ≥0.66 leap.
    // sd: <SD_FLOW FLOW / ≤SD_BLOCK HOLD / >SD_BLOCK BLOCK (bucket Datapath/Control/Gate).
    // rung: <4 CrossTier / 4..=6 Hard / ≥7 ExtremelyHard.  dissonance: <0.5 / ≥0.5.
    let f_bands = [(0.15f32, "F<.33"), (0.50, ".33-.66"), (0.80, "F≥.66")];
    let gates = [(0.10f32, "FLOW"), (0.25, "HOLD"), (0.45, "BLOCK")];
    let rungs = [(1u8, "R1-3"), (5, "R4-6"), (8, "R7-9")];
    let dissos = [(0.1f32, "coherent"), (0.7, "contradicted")];

    println!("\n── §2 the dispatch map (54 awareness cells → the recipe select_tactic fires) ──\n");
    println!("  gate   rung   dissonance    F<.33 routine        .33-.66 inference     F≥.66 leap");
    let mut winners: BTreeSet<u8> = BTreeSet::new();
    for (sd, gname) in gates {
        for (rung, rname) in rungs {
            for (diss, dname) in dissos {
                let mut cells = Vec::new();
                for (fe, _) in f_bands {
                    let mut ctx = ThoughtCtx::new(vec![0.9, 0.6, 0.3]);
                    ctx.sd = sd;
                    ctx.rung = rung;
                    ctx.dissonance = diss;
                    ctx.free_energy = fe;
                    let id = select_tactic(&ctx);
                    winners.insert(id);
                    let r = recipe(id).expect("id in 1..=34");
                    cells.push(format!("#{:<2} {:<16}", r.id, r.code));
                }
                println!(
                    "  {}  {}   {:<12}  {}",
                    gname,
                    rname,
                    dname,
                    cells.join("  ")
                );
            }
        }
        debug_assert_eq!(gate_name(sd).trim(), gname);
    }

    // ── §3 reachability: who can the surprise selector EVER dispatch? ──
    println!("\n── §3 selector reachability (measured over all 54 cells) ──\n");
    let reached: Vec<String> = winners
        .iter()
        .map(|&id| format!("#{} {}", id, recipe(id).expect("valid").code))
        .collect();
    println!(
        "  reachable via select_tactic: {} of 34 — {}",
        winners.len(),
        reached.join(", ")
    );
    for (mech, label) in [
        (Mechanism::ParallelIndependence, "ParallelIndependence"),
        (Mechanism::TruthAwareInference, "TruthAwareInference "),
        (Mechanism::StructuralDivergence, "StructuralDivergence"),
        (Mechanism::Infrastructure, "Infrastructure      "),
    ] {
        let (r, s): (Vec<&Recipe>, Vec<&Recipe>) = RECIPES
            .iter()
            .filter(|x| x.mechanism == mech)
            .partition(|x| winners.contains(&x.id));
        println!(
            "  {}  reached {:<12} shadowed/unreachable {}",
            label,
            r.iter()
                .map(|x| format!("#{}", x.id))
                .collect::<Vec<_>>()
                .join(","),
            s.iter()
                .map(|x| format!("#{}{}", x.id, x.code))
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    let infra_reached = RECIPES
        .iter()
        .filter(|x| x.mechanism == Mechanism::Infrastructure && winners.contains(&x.id))
        .count();
    println!(
        "\n  Infrastructure recipes reachable via the surprise selector: {infra_reached} (mechanism\n  match scores +5; the max non-match score is bucket+tier+reconcile = 4 — Infrastructure\n  is never the wanted mechanism, so these 14 fire ONLY via the style→mechanism wire)."
    );
    println!(
        "  NOTE #31 ICR (causal-lattice Covered) shares band/bucket/tier with #4 RCR and loses\n  the lowest-id tie — the counterfactual recipe is selector-shadowed by the backward-\n  causality one. The style wire (or a direct kernel(31) call) is ICR's only route."
    );

    // ── §4 the loop + the materialization predicate (sanity, measured). ──
    let mut ctx = ThoughtCtx::new(vec![0.9, 0.6, 0.3]);
    ctx.sd = 0.45; // BLOCK: the gate itself is the problem
    ctx.dissonance = 0.7;
    ctx.free_energy = 0.8;
    let trace = materialize(&mut ctx, 32);
    let path: Vec<String> = trace
        .steps
        .iter()
        .map(|s| format!("#{}{}", s.tactic_id, if s.fired { "" } else { "(blocked)" }))
        .collect();
    println!("\n── §4 one full F→34→F run from the worst cell (BLOCK, contradicted, F=0.8) ──\n");
    println!(
        "  path {} → rested={} conf={:.2} F={:.2}",
        path.join(" → "),
        trace.rested,
        trace.final_confidence,
        trace.final_free_energy
    );
    let probe = {
        let mut c = ThoughtCtx::new(vec![0.9, 0.6, 0.3]);
        c.sd = 0.25;
        c
    };
    println!(
        "  awareness_is_causal(F 0.1 vs 0.9): {} — perturbing surprise changes the tactic",
        awareness_is_causal(&probe, 0.1, 0.9)
    );
}
