//! `disorder_gate_probe` — confirm the NaN-blindness DEFECT in `select_tactic`,
//! then show the `dispatch_mode` router catches it. Run over REAL projected
//! `SubstrateView` ctxs (grounded → partial → ungrounded), never planted scalars.
//!
//! The defect (E-DISORDER-GATE-1): `select_tactic` reads `free_energy >= 0.66 /
//! >= 0.33 / else`. `NaN >= x` is always `false`, so an ungrounded state (a
//! marker a missing tenant could not ground) falls into the `else` and is
//! dispatched as the LOWEST-surprise **routine** band — an undefined state read
//! as a calm one. The router runs first and routes it to `FieldGather` instead.
//!
//! ```sh
//! cargo run -p lance-graph-contract --example disorder_gate_probe
//! ```

use lance_graph_contract::awareness_facet::SpoFacet;
use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::dispatch_mode::{classify, is_ungrounded, route, DispatchMode, Domain};
use lance_graph_contract::materialize::select_tactic;
use lance_graph_contract::mul::DkPosition;
use lance_graph_contract::qualia::QualiaI4_16D;
use lance_graph_contract::recipe_substrate::SubstrateView;
use lance_graph_contract::recipes::{recipe, Mechanism};

/// Build a real SubstrateView with the given tenant presence, then project it.
/// `spo_present` / `witness_edges` control which markers the projection can
/// ground; an absent tenant emits NaN/empty — the honest ungrounded state.
fn view(spo_present: bool, witness_edges: &[(Locus, i8)]) -> SubstrateView {
    let spo = if spo_present {
        SpoFacet::from_register([12, 40, 7, 88, 30, 5, 12, 41, 7, 90, 31, 6])
    } else {
        SpoFacet::default()
    };
    let mut w = CausalWitnessFacet::ZERO;
    for &(l, off) in witness_edges {
        w = w.with(l, off);
    }
    SubstrateView::new(spo, w, QualiaI4_16D::ZERO)
}

fn main() {
    println!(
        "disorder_gate_probe — NaN-blindness defect in select_tactic + the dispatch_mode fix\n"
    );

    // ── the four real substrate states ──
    // GROUNDED: SPO + rich witness (quorum + kausal + s-meaning).
    let grounded = view(
        true,
        &[
            (Locus::Quorum, 4),
            (Locus::Kausal, -3),
            (Locus::SMeaning, 2),
        ],
    );
    // COMPLEX: SPO + witness with a bound contradiction ≠ quorum.
    let complex = view(true, &[(Locus::Quorum, 2), (Locus::Contradiction, -5)]);
    // UNGROUNDED: no SPO, no witness edges → free_energy NaN, candidates empty.
    let ungrounded = view(false, &[]);

    let g = grounded.project();
    let x = complex.project();
    let u = ungrounded.project();

    // ═══ Leg A — CONFIRM THE DEFECT (real code, real projected ctx) ═══
    println!("── Leg A: the defect (select_tactic on the ungrounded ctx) ──");
    let picked = select_tactic(&u);
    let r = recipe(picked).unwrap();
    println!(
        "  ungrounded ctx: free_energy={:?} candidates={} → select_tactic → #{} {} ({:?})",
        u.free_energy,
        u.candidates.len(),
        picked,
        r.code,
        r.mechanism
    );
    // The defect: NaN free_energy reads as the routine (ParallelIndependence) band.
    let defect_confirmed = u.free_energy.is_nan() && r.mechanism == Mechanism::ParallelIndependence;
    println!(
        "  → NaN surprise silently read as ROUTINE ParallelIndependence: {}",
        if defect_confirmed {
            "DEFECT CONFIRMED"
        } else {
            "not reproduced"
        }
    );

    // ═══ Leg B — THE ROUTER CATCHES IT ═══
    println!("\n── Leg B: the dispatch_mode router (runs BEFORE select_tactic) ──");
    for (name, ctx) in [("grounded", &g), ("complex", &x), ("ungrounded", &u)] {
        let rt = route(ctx, None);
        println!(
            "  {:<11} ungrounded={:<5} → {:?} → {:?}",
            name, rt.was_ungrounded, rt.domain, rt.mode
        );
    }
    let routed = route(&u, None);
    println!(
        "  → the ungrounded ctx routes to {:?} (ground the fields), NOT routine dispatch",
        routed.mode
    );

    // ═══ Leg C — the MUL MountStupid veto (circle-of-competence at the mode level) ═══
    println!("\n── Leg C: MUL veto — a Dunning-Kruger novice cannot saccade a 'Clear' state ──");
    let clear_dom = classify(&g_clear(), None);
    let vetoed_dom = classify(&g_clear(), Some(DkPosition::MountStupid));
    println!("  Clear state: expert → {clear_dom:?} ; MountStupid novice → {vetoed_dom:?}");

    // ═══ registered gates ═══
    println!("\n── gates ──");
    let mut green = true;

    // G1: the defect is real — NaN surprise dispatches routine ParallelIndependence.
    println!(
        "[{}] G1 defect reproduced: NaN free_energy → select_tactic routine band",
        pf(defect_confirmed)
    );
    green &= defect_confirmed;

    // G2: the router routes the SAME ungrounded ctx to FieldGather, not routine.
    let g2 = routed.mode == DispatchMode::FieldGather && routed.domain == Domain::Confused;
    println!(
        "[{}] G2 router sends the ungrounded ctx to FieldGather (Confused), not routine",
        pf(g2)
    );
    green &= g2;

    // G3: domain distribution is non-degenerate on the real states (≥3 distinct
    //     domains occupied) — the classifier is a real function of state, not a label.
    let doms = [
        route(&g, None).domain,
        route(&x, None).domain,
        route(&u, None).domain,
        classify(&g_clear(), None),
    ];
    let distinct = {
        let mut v: Vec<Domain> = doms.to_vec();
        v.dedup_by(|a, b| a == b);
        v.sort_by_key(|d| *d as u8);
        v.dedup();
        v.len()
    };
    let g3 = distinct >= 3;
    println!(
        "[{}] G3 domain distribution non-degenerate: {} distinct domains on real states",
        pf(g3),
        distinct
    );
    green &= g3;

    // G4: the MUL veto is causal — MountStupid downgrades Clear to Complicated.
    let g4 = clear_dom == Domain::Clear && vetoed_dom == Domain::Complicated;
    println!(
        "[{}] G4 MountStupid veto downgrades Clear→Complicated (circle-of-competence)",
        pf(g4)
    );
    green &= g4;

    // G5: election is causal — a grounded/ungrounded pair elects different modes.
    let g5 =
        route(&g, None).mode != route(&u, None).mode && is_ungrounded(&u) && !is_ungrounded(&g);
    println!(
        "[{}] G5 election is causal: grounded vs ungrounded elect different modes",
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
    assert!(green, "disorder-gate probe gates failed");
}

/// A grounded, obviously-Clear ctx (FLOW gate, low surprise, no contradiction).
fn g_clear() -> lance_graph_contract::recipe_kernels::ThoughtCtx {
    let mut c = lance_graph_contract::recipe_kernels::ThoughtCtx::new(vec![0.5, 0.5]);
    c.free_energy = 0.1;
    c.sd = 0.05;
    c.dissonance = 0.0;
    c.confidence = 0.7;
    c
}

fn pf(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
