//! `tactic_select_confusion` — the D-DIA-V2-B falsifier for the S8
//! GraphBias→tactic LUT (`nars::tactic_select::tactic_for_bias`).
//!
//! The claim under test: the bias-selected tactic is the one whose intended work
//! matches the graph condition the bias names — so it FIRES on a graph
//! exhibiting that bias, and (for the structural tactics) mismatched tactics
//! return a `ReasoningGap`. This is the E-BASIN-WIDTH discipline applied to
//! SELECTION: a LUT that "picks the right tactic" must beat a bias-blind null.
//!
//! Five structural fixtures, one per bias-relevant graph condition:
//!   Explore  → shared-predicate pair      (RCR abduces)
//!   Adapt    → a similarity sibling        (TR diverges)
//!   Exploit  → an `is_a` chain             (CAS abstracts)
//!   Stagnant → a lone challengeable belief (ASC self-critiques)
//!   Resolve  → a lone belief to synthesize (CR revises)
//!
//! HONEST SCOPE (anti-overclaim): the three FRONTIER tactics (RCR/TR/CAS) are
//! *structurally* selective — they gap when their premise-structure is absent,
//! so their selection is an APPLICABILITY claim. The two REVISION tactics
//! (ASC/CR) are *broadly* applicable — any existing belief can be challenged or
//! synthesized — so their selection is NORMATIVE (which revision is appropriate
//! to the graph condition), not structural. The probe measures both, and does
//! not pretend ASC/CR discriminate structurally.
//!
//! Usage: `cargo run -p lance-graph-planner --example tactic_select_confusion`

use lance_graph_contract::sensorium::GraphBias;
use lance_graph_planner::nars::{
    asc_challenge, cas_abstract, challenge_target, cr_synthesize, rcr_abduce, tactic_for_bias,
    tr_diverge, AscOutcome, BeliefArena, CStmt, Copula, ReviseOutcome, Stamp, TacticChoice,
    Throttle, TruthValue,
};

fn inh(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Inh,
        p,
    }
}
fn sim(s: u16, p: u16) -> CStmt {
    CStmt {
        s,
        cop: Copula::Sim,
        p,
    }
}

/// A structural fixture: how to build its arena, plus the canonical entry points
/// each tactic needs (focus statement, focus subject, revision target/thesis).
struct Fixture {
    bias: GraphBias,
    name: &'static str,
    build: fn() -> BeliefArena,
    focus: CStmt,       // TR
    focus_subject: u16, // CAS
    target: CStmt,      // ASC / CR (an existing statement)
    target_truth: TruthValue,
}

/// Does `choice`'s tactic do its intended work on a fresh copy of `fx`?
/// (Frontier tactics: ≥1 candidate. Revision tactics: the revision took —
/// `Revised`, not a fresh `Admitted`.) Fixture is rebuilt per trial because
/// `BeliefArena` is not `Clone` and the revision tactics mutate.
fn fires(choice: TacticChoice, fx: &Fixture) -> bool {
    let mut arena = (fx.build)();
    let counter_stamp = Stamp::source(9999); // disjoint from every fixture source
    match choice {
        TacticChoice::Rcr => !rcr_abduce(&arena, &Throttle::permissive())
            .candidates
            .is_empty(),
        TacticChoice::Tr => !tr_diverge(&arena, fx.focus).candidates.is_empty(),
        TacticChoice::Cas => !cas_abstract(&arena, fx.focus_subject, &Throttle::permissive())
            .candidates
            .is_empty(),
        TacticChoice::Asc => matches!(
            asc_challenge(
                &mut arena,
                fx.target,
                challenge_target(fx.target_truth),
                counter_stamp,
            ),
            AscOutcome::Revised { .. }
        ),
        TacticChoice::Cr => matches!(
            cr_synthesize(
                &mut arena,
                fx.target,
                TruthValue::new(0.2, 0.8),
                counter_stamp
            ),
            ReviseOutcome::Revised { .. }
        ),
    }
}

fn main() {
    let fixtures = [
        // Explore → RCR: two is_a beliefs share predicate 9 (a non-hub middle).
        Fixture {
            bias: GraphBias::Explore,
            name: "shared-predicate pair",
            build: || {
                let mut a = BeliefArena::new();
                a.observe(inh(1, 9), TruthValue::new(0.9, 0.8), Stamp::source(0));
                a.observe(inh(2, 9), TruthValue::new(0.8, 0.7), Stamp::source(1));
                a
            },
            focus: inh(1, 9),
            focus_subject: 1,
            target: inh(1, 9),
            target_truth: TruthValue::new(0.9, 0.8),
        },
        // Adapt → TR: a similarity sibling (1~2) beside an is_a to transfer.
        Fixture {
            bias: GraphBias::Adapt,
            name: "similarity sibling",
            build: || {
                let mut a = BeliefArena::new();
                a.observe(inh(1, 9), TruthValue::new(0.9, 0.8), Stamp::source(0));
                a.observe(sim(1, 2), TruthValue::new(0.9, 0.8), Stamp::source(1));
                a
            },
            focus: inh(1, 9),
            focus_subject: 1,
            target: inh(1, 9),
            target_truth: TruthValue::new(0.9, 0.8),
        },
        // Exploit → CAS: an is_a chain 1→2→3 to abstract/deduce over.
        Fixture {
            bias: GraphBias::Exploit,
            name: "is_a chain",
            build: || {
                let mut a = BeliefArena::new();
                a.observe(inh(1, 2), TruthValue::new(0.9, 0.9), Stamp::source(0));
                a.observe(inh(2, 3), TruthValue::new(0.9, 0.9), Stamp::source(1));
                a
            },
            focus: inh(1, 2),
            focus_subject: 1,
            target: inh(1, 2),
            target_truth: TruthValue::new(0.9, 0.9),
        },
        // Stagnant → ASC: a lone belief to self-critique with independent counter.
        Fixture {
            bias: GraphBias::Stagnant,
            name: "lone belief (challenge)",
            build: || {
                let mut a = BeliefArena::new();
                a.observe(inh(1, 2), TruthValue::new(0.8, 0.8), Stamp::source(0));
                a
            },
            focus: inh(1, 2),
            focus_subject: 1,
            target: inh(1, 2),
            target_truth: TruthValue::new(0.8, 0.8),
        },
        // Resolve → CR: a lone belief to synthesize thesis+antithesis onto.
        Fixture {
            bias: GraphBias::Resolve,
            name: "lone belief (synthesize)",
            build: || {
                let mut a = BeliefArena::new();
                a.observe(inh(1, 2), TruthValue::new(0.9, 0.8), Stamp::source(0));
                a
            },
            focus: inh(1, 2),
            focus_subject: 1,
            target: inh(1, 2),
            target_truth: TruthValue::new(0.9, 0.8),
        },
    ];

    let all = [
        TacticChoice::Rcr,
        TacticChoice::Tr,
        TacticChoice::Cas,
        TacticChoice::Asc,
        TacticChoice::Cr,
    ];
    let structural = [TacticChoice::Rcr, TacticChoice::Tr, TacticChoice::Cas];

    println!("=== S8 GraphBias→tactic confusion matrix (rows = fixture/bias, cols = tactic) ===");
    print!("{:<26}", "fixture \\ tactic");
    for t in all {
        print!("{:>6?}", t);
    }
    println!("   selected");

    // Build the matrix and evaluate the gates.
    let mut diagonal_fires = 0usize;
    let mut structural_on = 0usize; // structural tactic fires on ITS OWN fixture
    let mut structural_off = 0usize; // structural tactic fires on a DIFFERENT fixture
    let mut constant_cover = [0usize; 5]; // how many fixtures each constant policy would fire on

    for fx in &fixtures {
        let selected = tactic_for_bias(fx.bias);
        print!("{:<26}", fx.name);
        for (ti, &t) in all.iter().enumerate() {
            let f = fires(t, fx);
            print!("{:>6}", if f { "●" } else { "·" });
            if f {
                constant_cover[ti] += 1;
            }
            // structural discrimination accounting
            if structural.contains(&t) {
                let is_own = t == selected;
                if f && is_own {
                    structural_on += 1;
                } else if f && !is_own {
                    structural_off += 1;
                }
            }
        }
        let sel_fires = fires(selected, fx);
        if sel_fires {
            diagonal_fires += 1;
        }
        println!("   {selected:?} {}", if sel_fires { "✓" } else { "✗ MISS" });
    }

    let n = fixtures.len();
    println!("\n=== gates ===");

    // G1 — the LUT-selected tactic fires on every fixture (applicability).
    let g1 = diagonal_fires == n;
    println!(
        "G1 diagonal applicability : {diagonal_fires}/{n} selected-tactic fires  → {}",
        pass(g1)
    );

    // G2 — the three structural tactics fire ONLY on their own fixture.
    // on-structure should be 3 (RCR@Explore, TR@Adapt, CAS@Exploit); off = 0.
    let g2 = structural_on == 3 && structural_off == 0;
    println!(
        "G2 structural discrimination: on-structure {structural_on}/3, off-structure {structural_off}/… → {}",
        pass(g2)
    );

    // G3 — no CONSTANT single-tactic policy matches the LUT: either it fires on
    // <5 fixtures (misses some), or it fires on 5 with zero structural
    // discrimination (the broad revision tactics). The LUT alone gets 5/5
    // applicable AND keeps the 3 structural discriminations.
    let best_constant = *constant_cover.iter().max().unwrap();
    let structural_constant_max = constant_cover[0]
        .max(constant_cover[1])
        .max(constant_cover[2]);
    // a constant structural tactic covers at most 1 fixture; a constant revision
    // tactic covers all 5 but discriminates 0 → neither matches the LUT.
    let g3 = g1 && structural_constant_max <= 1;
    println!(
        "G3 beats constant policy   : best-constant cover {best_constant}/{n} (structural ≤ {structural_constant_max}); LUT = {n}/{n} applicable + 3 discriminations → {}",
        pass(g3)
    );

    println!(
        "\nHONEST NOTE: ASC/CR are broadly applicable (constant cover {}/{n}, {}/{n}); their\n\
         selection is NORMATIVE (which revision suits the condition), not structural. The\n\
         structural discrimination is carried by RCR/TR/CAS. The LUT is falsifiable via G1:\n\
         a mis-mapping (e.g. Explore→CAS) would GAP on its fixture and drop the diagonal.",
        constant_cover[3], constant_cover[4]
    );

    let verdict = g1 && g2 && g3;
    println!(
        "\n=== VERDICT: {} ===",
        if verdict { "PASS" } else { "FAIL" }
    );
    assert!(verdict, "D-DIA-V2-B falsifier failed — see gates above");
}

fn pass(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}
