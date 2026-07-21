//! `recipe_ladder_over_substrate` — the whole wiring, end to end: the three real
//! tenants → projected [`ThoughtCtx`] → the rung-ordered, NaN-gated **causal
//! ladder** of the 34 recipes. This is the answer to the audit's root cause
//! (recipes ran on a scalar proxy): now they run on **SPO + 24 witness edges +
//! qualia**.
//!
//! Demonstrates, on REAL tenant registers (not hand-set scalars):
//!   1. the escalation ladder (shallow deduction → deep counterfactual), keyed by
//!      inference type, with each recipe's rung + NaN-disqualifier state;
//!   2. **qualia is additive & stakes-only** — swap the qualia register and ONLY
//!      the affective temperature moves; the dispatch (the logic) is identical;
//!   3. **causality is a witness edge** — the KAUSAL locus places the cause;
//!   4. **NaN-disqualification** (the unmeasurable conjugate) — drop a tenant and
//!      the recipes that required it are skipped, not read off noise.
//!
//! ```sh
//! cargo run -p lance-graph-contract --example recipe_ladder_over_substrate
//! ```

use lance_graph_contract::awareness_facet::SpoFacet;
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::qualia::{QualiaI4_16D, QUALIA_I4_LABELS};
use lance_graph_contract::recipe_dispatch::{inference, ladder, RecipeStep};
use lance_graph_contract::recipe_substrate::{affective_temperature, SubstrateView};

fn q(label: &str) -> usize {
    QUALIA_I4_LABELS.iter().position(|&l| l == label).unwrap()
}

/// A grounded row: a real SPO triple, a witness with the KAUSAL cause 3 events
/// back, an agreeing quorum peer and a preserved contradiction, and hot qualia.
fn grounded_row() -> SubstrateView {
    let spo = SpoFacet::from_register([12, 40, 7, 88, 30, 5, 12, 41, 7, 90, 31, 6]);
    let witness = CausalWitnessFacet::ZERO
        .with(Locus::Kausal, -3) // the cause sits 3 events back
        .with(Locus::Antecedent, -1) // coreference antecedent 1 back
        .with(Locus::SMeaning, 2)
        .with(Locus::Quorum, 4) // an agreeing peer 4 ahead
        .with(Locus::Contradiction, -5) // a preserved dissenter 5 back
        .with(Locus::MeaningLevel, 3); // rung-3 content (the 34-runbook tier)
    let qualia = QualiaI4_16D::ZERO
        .with(q("arousal"), 6)
        .with(q("tension"), 4);
    SubstrateView::new(spo, witness, qualia)
}

fn print_ladder(title: &str, steps: &[RecipeStep]) {
    println!("── {title} ──");
    let mut cur = 0u8;
    for s in steps {
        if s.rung != cur {
            cur = s.rung;
            println!("  rung {cur}:");
        }
        let status = match s.disqualified_by {
            None => "fire".to_string(),
            Some(f) => format!("SKIP (NaN: {f:?})"),
        };
        println!(
            "     #{:<2} {:<15} trig={} {}",
            s.id,
            inference(s.id).tag(),
            s.trigger,
            status
        );
    }
}

fn main() {
    let row = grounded_row();
    let ctx = row.project();

    println!(
        "recipe_ladder_over_substrate — 34 recipes over real SPO + 24 witness edges + qualia\n"
    );
    println!("projected ThoughtCtx from tenants:");
    println!(
        "  confidence {:.3} (SPO+quorum/contradiction edges)  free_energy {:.3} (witness)  dissonance {:.3}",
        ctx.confidence, ctx.free_energy, ctx.dissonance
    );
    println!(
        "  temperature {:.3} (qualia: additive stakes)  rung {} (meaning-level edge)  candidates {} (bound edges)",
        ctx.temperature,
        ctx.rung,
        ctx.candidates.len()
    );
    println!(
        "  cause (KAUSAL edge) = {} events; antecedent = {}; agreement w/ self = {}",
        row.witness.cause(),
        row.witness.antecedent(),
        row.witness.agreement_count(row.witness)
    );
    println!();

    let steps = ladder(&ctx);
    print_ladder("causal ladder (grounded — everything fires)", &steps);

    // ── 2. qualia is additive & stakes-only: swap qualia, logic unchanged ──
    let cool = SubstrateView::new(
        row.spo,
        row.witness,
        QualiaI4_16D::ZERO.with(q("groundedness"), 6),
    );
    let cool_ctx = cool.project();
    println!("\n── qualia swap (hot → cool): ONLY temperature moves ──");
    println!(
        "  temperature {:.3} → {:.3}  |  confidence {:.3} → {:.3}  free_energy {:.3} → {:.3}",
        ctx.temperature,
        cool_ctx.temperature,
        ctx.confidence,
        cool_ctx.confidence,
        ctx.free_energy,
        cool_ctx.free_energy
    );

    // ── 4. NaN-disqualification: drop the SPO + witness tenants ──
    let bare = SubstrateView::new(SpoFacet::default(), CausalWitnessFacet::ZERO, row.qualia);
    let bare_ctx = bare.project();
    let bare_steps = ladder(&bare_ctx);
    let skipped = bare_steps
        .iter()
        .filter(|s| s.disqualified_by.is_some())
        .count();
    println!("\n── tenants dropped (SPO+witness absent): the unmeasurable conjugates ──");
    println!(
        "  confidence NaN={}  free_energy NaN={}  candidates empty={}  → {} of 34 recipes disqualified",
        bare_ctx.confidence.is_nan(),
        bare_ctx.free_energy.is_nan(),
        bare_ctx.candidates.is_empty(),
        skipped
    );

    // ═══ gates ═══
    println!("\n── gates ──");
    let mut green = true;

    // G1: grounded row fires every recipe (all checklists covered).
    let g1 = steps.iter().all(|s| s.disqualified_by.is_none());
    println!(
        "[{}] G1 grounded row fires all 34 (full checklist coverage)",
        pf(g1)
    );
    green &= g1;

    // G2: qualia is additive & stakes-only — temperature moves, logic frozen.
    let g2 = (ctx.temperature - cool_ctx.temperature).abs() > 1e-6
        && ctx.confidence == cool_ctx.confidence
        && ctx.free_energy.to_bits() == cool_ctx.free_energy.to_bits()
        && ctx.candidates == cool_ctx.candidates;
    println!(
        "[{}] G2 qualia swap moves ONLY temperature (additive, stakes-only)",
        pf(g2)
    );
    green &= g2;

    // G3: the ladder is rung-ordered (the causal chain escalates shallow→deep).
    let g3 = steps.windows(2).all(|w| w[0].rung <= w[1].rung);
    println!(
        "[{}] G3 dispatch is rung-ordered (causal chain shallow→deep)",
        pf(g3)
    );
    green &= g3;

    // G4: dropping the tenants disqualifies real recipes (NaN conjugate skipped).
    let g4 = skipped > 0 && bare_ctx.confidence.is_nan() && bare_ctx.free_energy.is_nan();
    println!(
        "[{}] G4 absent tenants ⇒ NaN inputs ⇒ recipes disqualified ({} skipped)",
        pf(g4),
        skipped
    );
    green &= g4;

    // G5: causality is on the witness edge (not qualia) and reads back.
    let g5 = row.witness.cause() == -3 && affective_temperature(row.qualia) > 0.5;
    println!(
        "[{}] G5 causality = KAUSAL witness edge; qualia = additive stakes",
        pf(g5)
    );
    green &= g5;

    println!(
        "\n{}",
        if green {
            "ALL GATES GREEN"
        } else {
            "GATE FAILURE"
        }
    );
    assert!(green, "ladder-over-substrate gates failed");
}

fn pf(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
