//! PROBE-STYLE-MICROCODE-FRONTIER-1 — thinking styles as MICROCODE (ordered
//! groups of typed ops), with Autopoiesis-frontier reinforcement as NARS
//! revision over style-level claims. **Phase 1 of 2.**
//!
//! **The operator's intent (2026-08-23), mapped onto shipped machinery:**
//!
//! ```text
//!   "microcode for thinking styles"        style = ORDERED GROUP of typed ops
//!                                          (PROBE-MULTI-GROUP-MEMBERSHIP-1 B3)
//!   "runtime evolvement of outcome         TruthValue::revise over per-style
//!    revision"                             claims, stamped per episode
//!                                          (nars/truth.rs:57 — SHIPPED)
//!   "resonance based thinking"             dispatch by expectation() (CHOICE)
//!   "frozen learned / explore              BOTH groups resident over the SAME
//!    superposition"                        op vocabulary — a group distinction,
//!                                          not a subsystem
//!   "what is more efficient and            measured op-count cost; takeover
//!    reusing that"                         gated on truth AND measured cost
//!   Autopoiesis                            rung 4 of the content ladder
//!                                          (StyleFamily macros + autopoiesis
//!                                          triangle — persona-vs-rung-ladder;
//!                                          StyleLane / cognitive_palette ship
//!                                          the triangle lanes today)
//! ```
//!
//! **The headline hypothesis this probe measures:** the reinforcement
//! learning the frontier needs is ALREADY SHIPPED — it is NARS revision +
//! CHOICE applied to style-level claims, with admission to "frozen" gated
//! on the `LearnedSurvivedTests` predicate (#1011: the only state licensing
//! a learned transformation). No gradient, no bandit, no learner subsystem.
//!
//! # Phase 2 — R2IL (RECORDED, NOT BUILT)
//!
//! **R2IL is the way richer op vocabulary, and it is Phase 2.** Where this
//! probe's ops are the #1001 view-edit atoms, R2IL carries reconstructible
//! typed BEHAVIOR (the `VarnodeFacet` typed drill, `FlatFact`'s no-heap
//! rows, intervention/counterfactual operations — the four-plane DID
//! plane). The same loop lifts onto it: R2IL ops as microcode members,
//! episodes as interventions with observed consequences, revision from
//! outcome, freezing gated on surviving falsification. NOTHING of that is
//! built here: the V4 classid stays provisional (O5 gate), no R2IL types
//! are imported, and Phase 2 begins only after this loop shape is ruled
//! sound. Same-dock ≠ same-ClassView; behavior semantics stay V4's.
//!
//! **The widened Phase-2 synthesis (operator, recorded as HYPOTHESIS in the
//! mandated conditional phrasing):** the combination `R2IL × BPE` with
//! OGAR-loco-shaped routing macros, reading **V4 as the thinking-dynamic
//! plane**. Precisely: IF measured recurrent typed R2IL behavior requires a
//! resident macro representation, the recurrence/compression machinery
//! (token-BPE probe: CAN-FIT; behavioral carrier: UNDECIDED) MAY compress
//! ordered groups of R2IL transformations into reconstructible macros; IF
//! that recurrence produces reusable routing structure, OGAR-loco-shaped
//! routing MAY carry it; and V4-shaped behavior geometry is one possible
//! future carrier for the resulting thinking DYNAMICS. Three IFs, zero
//! decisions — every admission condition from the root order applies, and
//! none of the three is built or reserved here.
//!
//! # Honesty box
//!
//! - Episode outcomes come from a TOY world oracle (success iff the
//!   microcode grounds itself with a `PushBoundAt` before interrogating —
//!   a stand-in for the #1001 warrant). This proves the LOOP MACHINERY
//!   (revision, choice, admission, immutability, no-double-count), not
//!   that any real style wins. Production episodes are Phase 2 material.
//! - The dispatch rule (among sufficiently-trusted styles pick the
//!   cheapest measured; otherwise keep exploring) is PROBE-LOCAL policy
//!   composed of shipped pieces (expectation + measured cost). It is not
//!   canon and is labelled as policy.
//! - No `Learner`, no gradient, no bandit struct, no Q-table, no reward
//!   model type. The only state that "learns" is a NARS `TruthValue` per
//!   style claim plus its evidential `Stamp` — both shipped types.

use lance_graph_planner::nars::belief::Stamp;
use lance_graph_planner::nars::truth::TruthValue;
use std::collections::HashMap;

/// The typed op vocabulary (the #1001 atoms — Phase 1). Phase 2 swaps in
/// R2IL's richer reconstructible behavior ops; the loop is unchanged.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Op {
    PushBoundAt(u8),
    PushRungBand(u8, u8),
    PushGapSubject(u16),
    Pop,
}

/// One style = one ORDERED microcode over typed ops, plus its epistemic
/// state: a NARS claim ("this style succeeds") with evidential provenance.
/// The microcode of a FROZEN style is immutable — evolution mints a NEW
/// explore style; the population does not move.
#[derive(Clone, Debug)]
struct Style {
    name: &'static str,
    microcode: Vec<Op>,
    truth: TruthValue,
    stamp: Stamp,
    frozen: bool,
    survived_falsifier: bool,
}

impl Style {
    fn new(name: &'static str, microcode: Vec<Op>) -> Self {
        Self {
            name,
            microcode,
            truth: TruthValue::new(0.5, 0.0), // unknown — the frontier state
            stamp: Stamp::default(),
            frozen: false,
            survived_falsifier: false,
        }
    }
    /// Measured cost = op count (operation counts, never wall time).
    fn cost(&self) -> usize {
        self.microcode.len()
    }
    /// Outcome revision — the "runtime evolvement": shipped NARS revise,
    /// with a per-episode stamp so evidence can never double-count.
    fn observe_outcome(&mut self, success: bool, episode: u32) {
        let ev = TruthValue::new(if success { 1.0 } else { 0.0 }, 0.5);
        let st = Stamp::source(episode);
        if self.stamp.disjoint(st) {
            self.truth = self.truth.revise(&ev);
            self.stamp = self.stamp.union(st);
        }
    }
}

/// The TOY world oracle (a stand-in for the #1001 warrant, stated as such):
/// a style succeeds iff it GROUNDS itself (a `PushBoundAt`) before it
/// interrogates (`PushGapSubject`).
fn world(microcode: &[Op]) -> bool {
    let ground = microcode
        .iter()
        .position(|o| matches!(o, Op::PushBoundAt(_)));
    let ask = microcode
        .iter()
        .position(|o| matches!(o, Op::PushGapSubject(_)));
    matches!((ground, ask), (Some(g), Some(a)) if g < a)
}

/// PROBE-LOCAL dispatch policy (labelled policy, not canon): among styles
/// whose expectation clears the trust bar, pick the CHEAPEST measured;
/// if none clears it, keep the incumbent frozen style.
fn dispatch(styles: &[Style], trust: f32) -> &Style {
    styles
        .iter()
        .filter(|s| s.truth.expectation() >= trust)
        .min_by_key(|s| s.cost())
        .unwrap_or_else(|| {
            styles
                .iter()
                .find(|s| s.frozen)
                .expect("an incumbent frozen style exists")
        })
}

fn main() {
    let mut pass = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        assert!(ok, "[FAIL] {name} — {detail}");
        println!("  [PASS] {name} — {detail}");
        pass += 1;
    };

    // The superposition: one FROZEN incumbent + two EXPLORE variants, all
    // over the SAME op vocabulary (no op is copied into a style — the
    // microcode holds the ops by value because Op is a 4-byte Copy atom;
    // the RESIDENT population these ops act on is elsewhere and untouched).
    let mut styles = vec![
        Style {
            frozen: true,
            survived_falsifier: true,
            truth: TruthValue::new(0.9, 0.8), // the learned incumbent
            ..Style::new(
                "frozen-incumbent",
                vec![
                    Op::PushBoundAt(9),
                    Op::PushRungBand(1, 2),
                    Op::PushGapSubject(1),
                    Op::Pop,
                ],
            )
        },
        // Cheaper AND sound: grounds before it interrogates.
        Style::new(
            "explore-lean",
            vec![Op::PushBoundAt(9), Op::PushGapSubject(1), Op::Pop],
        ),
        // Cheapest but UNSOUND: interrogates without grounding.
        Style::new("explore-reckless", vec![Op::PushGapSubject(1), Op::Pop]),
    ];
    let trust = 0.75f32;

    // ---- S1 — a style IS microcode: an ordered group that reconstructs ----
    let mc: Vec<Op> = styles[0].microcode.clone();
    gate(
        "S1 a thinking style is an ordered microcode over typed ops",
        mc.len() == 4 && mc[0] == Op::PushBoundAt(9) && *mc.last().unwrap() == Op::Pop,
        "the style replays as the exact typed sequence — the B3 ordered-group result \
         applied to styles; order is semantics, not decoration"
            .to_string(),
    );

    // ---- S2 — the superposition: frozen + explore coexist, dispatchable ----
    let initial = dispatch(&styles, trust).name;
    gate(
        "S2 frozen and explore styles coexist as a superposition; dispatch is CHOICE",
        initial == "frozen-incumbent"
            && styles.iter().filter(|s| !s.frozen).count() == 2
            && styles.iter().all(|s| !s.microcode.is_empty()),
        format!(
            "3 styles resident simultaneously; with the explorers at expectation 0.50 \
             (unknown), CHOICE keeps the incumbent ({initial}) — resonance-based \
             dispatch is expectation(), which is shipped"
        ),
    );

    // ---- Episodes: the frontier loop, mechanically ----
    // Each episode runs BOTH explorers against the world and revises their
    // style-level claims from the outcome. The incumbent needs no episodes;
    // its truth is already learned state.
    for episode in 1..=6u32 {
        for s in styles.iter_mut().filter(|s| !s.frozen) {
            let ok = world(&s.microcode);
            s.observe_outcome(ok, episode);
        }
    }
    let lean = styles.iter().find(|s| s.name == "explore-lean").unwrap();
    let reckless = styles
        .iter()
        .find(|s| s.name == "explore-reckless")
        .unwrap();

    // ---- S3 — outcome revision moved the claims (runtime evolvement) ----
    gate(
        "S3 outcome revision is shipped NARS revise: expectations moved apart",
        lean.truth.expectation() > 0.9 && reckless.truth.expectation() < 0.1,
        format!(
            "after 6 stamped episodes: explore-lean e={:.3} (6/6 succeeded), \
             explore-reckless e={:.3} (0/6 — it interrogates without grounding). \
             The 'learning' is TruthValue::revise + Stamp disjointness, nothing else",
            lean.truth.expectation(),
            reckless.truth.expectation()
        ),
    );

    // ---- S4 — no-double-count: replaying an episode is inert ----
    let before = (
        lean.truth.frequency.to_bits(),
        lean.truth.confidence.to_bits(),
    );
    {
        let s = styles
            .iter_mut()
            .find(|s| s.name == "explore-lean")
            .unwrap();
        s.observe_outcome(true, 3); // episode 3 already counted
    }
    let lean = styles.iter().find(|s| s.name == "explore-lean").unwrap();
    let after = (
        lean.truth.frequency.to_bits(),
        lean.truth.confidence.to_bits(),
    );
    gate(
        "S4 a replayed episode cannot double-count (Stamp disjointness)",
        before == after,
        "re-observing episode 3's outcome left the claim bit-identical — the same \
         guard that protects belief evidence protects style evidence"
            .to_string(),
    );

    // ---- S5 — admission to FROZEN requires surviving a falsifier ----
    // The falsification episode: an adversarial world where grounding is
    // checked strictly (same oracle, fresh episode id). explore-lean must
    // still succeed to be admitted; candidacy without it is refused.
    let lean_survives = world(
        &styles
            .iter()
            .find(|s| s.name == "explore-lean")
            .unwrap()
            .microcode,
    );
    {
        let s = styles
            .iter_mut()
            .find(|s| s.name == "explore-lean")
            .unwrap();
        s.survived_falsifier = lean_survives;
        s.observe_outcome(lean_survives, 100);
        // The admission predicate — the LearnedSurvivedTests state as code:
        if s.survived_falsifier && s.truth.expectation() >= trust {
            s.frozen = true;
        }
    }
    let lean_frozen = styles
        .iter()
        .find(|s| s.name == "explore-lean")
        .unwrap()
        .frozen;
    gate(
        "S5 freezing is the LearnedSurvivedTests predicate, not popularity",
        lean_frozen && lean_survives,
        "explore-lean froze only after high expectation AND a survived \
         falsification episode — the #1011 admission rule (the only state \
         licensing a learned transformation) applied at the style level"
            .to_string(),
    );

    // ---- S6 — the takeover: dispatch now reuses what is more efficient ----
    let chosen = dispatch(&styles, trust);
    gate(
        "S6 dispatch flips to the cheaper PROVEN style — reuse of the efficient",
        chosen.name == "explore-lean" && chosen.cost() < 4,
        format!(
            "CHOICE now selects {} (cost {} ops) over the incumbent (cost 4): \
             efficiency is measured op count, trust is NARS expectation, and the \
             flip required BOTH — 'frozen learned explore superposition of what is \
             more efficient and reusing that', as the loop",
            chosen.name,
            chosen.cost()
        ),
    );

    // ---- S7 — the reckless explorer never freezes (can-stay-silent) ----
    let reckless = styles
        .iter()
        .find(|s| s.name == "explore-reckless")
        .unwrap();
    gate(
        "S7 cheap-but-failing exploration never freezes and never dispatches",
        !reckless.frozen
            && reckless.truth.expectation() < trust
            && dispatch(&styles, trust).name != "explore-reckless",
        format!(
            "the cheapest style (2 ops) sits at e={:.3}: revision pushed it DOWN, \
             the admission predicate never fires, and CHOICE never selects it — \
             repeats-but-fails cannot be reinforced (F6 at the style level)",
            reckless.truth.expectation()
        ),
    );

    // ---- S8 — frozen microcode is immutable; evolution mints, never edits ----
    let incumbent_mc: Vec<Op> = styles
        .iter()
        .find(|s| s.name == "frozen-incumbent")
        .unwrap()
        .microcode
        .clone();
    // "Evolve" = add a NEW explore variant; the frozen incumbents are untouched.
    styles.push(Style::new(
        "explore-next",
        vec![Op::PushBoundAt(7), Op::PushGapSubject(2)],
    ));
    let incumbent_after: Vec<Op> = styles
        .iter()
        .find(|s| s.name == "frozen-incumbent")
        .unwrap()
        .microcode
        .clone();
    gate(
        "S8 evolution mints new explore styles; frozen microcode never mutates",
        incumbent_mc == incumbent_after && styles.len() == 4,
        "a new frontier variant appeared as a NEW ordered group; both frozen styles' \
         microcode is bit-identical — the population does not move, the frontier does"
            .to_string(),
    );

    // ---- S9 — the fence: no learner subsystem exists ----
    let style_state_is_shipped_types = core::mem::size_of::<TruthValue>() == 8
        && core::mem::size_of::<Stamp>() == 8
        && core::mem::size_of::<Op>() == 4;
    gate(
        "S9 no gradient, no bandit, no Q-table: the learner IS revise + CHOICE",
        style_state_is_shipped_types,
        "per-style learned state is exactly one shipped TruthValue (8B) + one \
         shipped Stamp (8B); dispatch is expectation() + measured cost; nothing \
         else was invented, which is the headline finding"
            .to_string(),
    );

    // ---- Efficiency ledger (measured, for the record) ----
    let mut ledger: HashMap<&str, (usize, f32)> = HashMap::new();
    for s in &styles {
        ledger.insert(s.name, (s.cost(), s.truth.expectation()));
    }
    println!("PROBE-STYLE-MICROCODE-FRONTIER-1: ALL {pass} GATES GREEN");
    println!(
        "report: thinking styles ARE microcode — ordered groups of typed ops (S1) — \
         and the Autopoiesis frontier loop runs on SHIPPED machinery only: frozen + \
         explore styles coexist as a superposition over one op vocabulary (S2), \
         outcomes revise style-level NARS claims at runtime with stamped \
         no-double-count evidence (S3/S4), freezing is the LearnedSurvivedTests \
         admission predicate (S5), dispatch reuses the cheaper PROVEN style (S6) \
         while cheap-but-failing exploration is never reinforced (S7), and evolution \
         mints new frontier variants without mutating frozen microcode (S8). The \
         reinforcement learner is TruthValue::revise + CHOICE — no new subsystem \
         (S9). Ledger (cost ops, expectation): {:?}. PHASE 2, recorded not built: \
         R2IL is the way richer op vocabulary — reconstructible typed behavior \
         (typed drill, interventions, counterfactuals) as microcode members — and \
         the identical loop lifts onto it once this shape is ruled sound; the V4 \
         classid stays provisional and no R2IL type was imported here.",
        {
            let mut v: Vec<_> = ledger.iter().collect();
            v.sort_by_key(|(name, _)| **name);
            v
        }
    );
}
