//! D-PEARL-IO-0: Pearl's ladder executed in and out of `CausalEdge64`.
//!
//! An edge enters with one `EpistemicState5`, the Pearl projection in bits
//! 40..42 selects which existing operator runs, the operator measures, and the
//! measurement — not the projection, not the caller — decides whether bits
//! 59..63 change.
//!
//! ```text
//! CausalEdge64 ──bits 40..42──► operator ──► Measured ──revise──► bits 59..63
//!                                  │
//!     SO  ► P7a observational folds (certify_observational)
//!     PO  ► P7a executed randomized arms (Model::causes)
//!     SPO ► dismech counterfactual replay (counterfactual_replay)
//!     SP  ► SO direction vs PO direction (CausalMask::simpsons_paradox_risk)
//! ```
//!
//! # Nothing new underneath
//!
//! Every operator is existing code: the P7a sealed model and its folds
//! (`shared/certification_model.rs`, moved unchanged from
//! `relational_certification_probe`), `lance_graph_planner::chain_counterfactual`
//! (one replay path for both arms), `CausalMask::simpsons_paradox_risk`, and
//! the canonical `EpistemicState5` declarations. What this probe adds is the
//! dispatch (`pearl::reason`), the write-back rule (`pearl::revise`) and the
//! topology gate (`pearl::hydrate`).
//!
//! # The write-back rule
//!
//! - **Certification only rises, and only to what the operator computed.**
//!   SO can earn up to `CausalCandidate` (its folds do not include `causes`).
//!   PO can earn `Causes`, from the executed arms. SPO and SP earn nothing: a
//!   counterfactual reaction and a confounding diagnostic are evidence beside
//!   the certification, not certifications.
//! - **Topology changes only through `hydrate`**, from `IndirectUnknown` to
//!   `IndirectKnown`, when both bindings `A → B` and `B → Y` exist in the
//!   sealed chain. Proposing a candidate B is not evidence.
//! - **Only bits 59..63 move.** The projection, mantissa, frequency and
//!   confidence of the edge are inputs, never outputs.
//!
//! # What the measurements showed
//!
//! - The counterfactual replay's verdict is a NARS revision over the chain's
//!   step frequencies, so a cut can move the truth in either direction:
//!   cutting a weak step can RAISE the terminal frequency. The reaction is
//!   therefore reported with both frequencies and its sign, never as "flipped"
//!   alone. In the sweep that chose the fixture, the frequency did not depend
//!   on the steps' endpoints.
//! - One-at-a-time removal is not the causal test. Under Y = A or B, removing
//!   A where B holds changes nothing, removing A under the contingency "B
//!   disabled" changes Y, and the randomized arms certify `Causes`. The
//!   dismech chain replay is linear and cannot express the OR; the contingency
//!   is a mask over the same unit population the P7a folds read.
//!
//! # Not decided here
//!
//! - Production placement: done (D-PEARL-PROD-0). The dispatch is
//!   `lance_graph_planner::pearl`, the certification folds are
//!   `lance_graph_contract::certification`; the falsifier tests live with them.
//! - Whole-node removal of B: in a two-step chain B's only mechanisms are the
//!   two routes, so route cuts are run and node removal is not.
//! - A responsibility framework beyond the single contingency.
//!
//! Run: `cargo run -p cognitive-shader-driver --features with-planner --example pearl_ladder_probe`

use causal_edge::edge::{CausalEdge64, InferenceType};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;
use causal_edge::tables::NarsTables;
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::causal_audit::SupportBasis;
#[cfg(test)]
use lance_graph_contract::epistemic_state5::{fact, facts_population};
use lance_graph_contract::epistemic_state5::{Certification3, Topology2};
use lance_graph_planner::chain_counterfactual::{CutContext, DEFAULT_FREQUENCY_BAR};
use lance_graph_planner::chain_replay::{ChainStep, ComposeTables};
use lance_graph_planner::pearl::{hydrate, reason, revise, Chain, Edit, Evidence, Reading};

#[path = "shared/certification_model.rs"]
mod certification_model;
#[path = "shared/certification_reading.rs"]
mod certification_reading;

use certification_model::*;
use certification_reading::*;

// ── Fixtures ──────────────────────────────────────────────────────────────

const A: u8 = 10;
const B: u8 = 20;
const Y: u8 = 30;
const PREDICATE: u8 = 0x91;

fn edge(s: u8, o: u8, freq: u8, conf: u8, mask: CausalMask) -> CausalEdge64 {
    CausalEdge64::pack(
        s,
        PREDICATE,
        o,
        freq,
        conf,
        mask,
        0b101,
        InferenceType::Deduction,
        PlasticityState::S_HOT,
        0,
    )
}

/// The queried relation `A → Y` under projection `mask`, at a given state.
fn query(mask: CausalMask, contract: Contract, topology: Topology2) -> CausalEdge64 {
    stamp(edge(A, Y, 200, 200, mask), contract, topology)
}

fn with_mask(mut e: CausalEdge64, mask: CausalMask) -> CausalEdge64 {
    e.set_causal_mask(mask);
    e
}

/// A → Y associated in the pooled population and after the robustness mask,
/// but lowered in stratum 1: observationally `Related`, not `Contributes`.
fn related_population() -> Builder {
    let mut b = Builder::new();
    b.cell(0, true, 5, 4);
    b.cell(0, false, 5, 1);
    b.cell(1, true, 3, 1);
    b.cell(1, false, 3, 2);
    b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
    b
}

/// The mediated chain `A → B → Y`. Measured with `counterfactual_replay`:
/// the weak first step and strong second step put the factual terminal
/// frequency above the bar; cutting `B → Y` drops below it, cutting `A → B`
/// raises it.
fn chain_steps() -> Vec<ChainStep> {
    vec![
        (PREDICATE, edge(A, B, 40, 220, CausalMask::SPO)),
        (PREDICATE, edge(B, Y, 250, 220, CausalMask::SPO)),
    ]
}

fn compose_tables() -> [Box<[u8; 256 * 256]>; 3] {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    core::array::from_fn(|_| {
        let mut t = Box::new([0u8; 256 * 256]);
        for v in t.iter_mut() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((x >> 33) % 256) as u8;
        }
        t
    })
}

/// Everything the replay needs, owned once.
struct Replay {
    tables: NarsTables,
    compose: [Box<[u8; 256 * 256]>; 3],
    steps: Vec<ChainStep>,
}

impl Replay {
    fn new() -> Self {
        Replay {
            tables: NarsTables::build(1),
            compose: compose_tables(),
            steps: chain_steps(),
        }
    }

    fn chain(&self) -> Chain<'_> {
        Chain {
            steps: &self.steps,
            seed: edge(A, Y, 200, 200, CausalMask::SPO),
            cx: CutContext {
                tables: &self.tables,
                compose: ComposeTables {
                    s: &self.compose[0],
                    p: &self.compose[1],
                    o: &self.compose[2],
                },
                owner: 3,
                base_seq: 100,
                bar: DEFAULT_FREQUENCY_BAR,
            },
        }
    }
}

/// Y = A or B, per unit, as unit masks. The counterfactual edit is a mask;
/// the base masks are never modified.
struct OrGate {
    a: u64,
    b: u64,
}

impl OrGate {
    fn y(a: u64, b: u64) -> u64 {
        a | b
    }

    /// Does removing A at `unit` change Y there, with B as it is?
    fn but_for(&self, unit: u64) -> bool {
        (Self::y(self.a, self.b) ^ Self::y(self.a & !unit, self.b)) & unit != 0
    }

    /// The same removal under the contingency `disabled` (B masked off).
    fn but_for_given(&self, unit: u64, disabled: u64) -> bool {
        let b = self.b & !disabled;
        (Self::y(self.a, b) ^ Self::y(self.a & !unit, b)) & unit != 0
    }
}

/// The randomized trial behind [`OrGate`]: A assigned in arms with and
/// without B, two intervention sources.
fn overdetermined_trial() -> (Model, OrGate) {
    let mut b = Builder::new();
    let (a_and_b, _) = b.arm(true, 3, 3);
    let (a_only, _) = b.arm(true, 3, 3);
    let (b_only, _) = b.arm(false, 3, 3);
    b.arm(false, 3, 0);
    b.sources(SupportBasis::InterventionBacked, &[10, 11]);
    (
        b.build(),
        OrGate {
            a: a_and_b | a_only,
            b: a_and_b | b_only,
        },
    )
}

fn certification(e: CausalEdge64) -> Certification3 {
    let c = read(
        &declarations(),
        CERT_CLASS,
        e,
        lance_graph_contract::band_reading::EdgeProvenance::V2Stamped,
    )
    .expect("declared class, valid code");
    c.certification()
}

fn topology(e: CausalEdge64) -> Topology2 {
    Topology2::from_ordinal(e.epistemic_raw5() & 0b11).expect("2-bit ordinal")
}

// ── Candidate semantic upper-half reading ─────────────────────────────────
//
// D-CE64-SEM-0 is deliberately probe-local. It does NOT rename the shipped
// fields yet. The question is whether the current physical carriers can support
// one coherent transition loop without inventing another object:
//
//   40..42 Pearl question
//   43..45 orientation / syntax
//   46..49 signed activation (reaction, not opcode)
//   50..52 novelty / epistemic entropy proxy
//   53..58 belief-state update breadcrumb
//   59..63 canonical EpistemicState5
//
// If these falsifiers survive, a later PR can decide which readings deserve
// contract names. Until then the production accessors keep their current names.
#[cfg(test)]
mod semantic_upper_half {
    use super::*;
    use causal_edge::layout::{PLAST_MASK, PLAST_SHIFT};
    use lance_graph_planner::pearl::Reaction;

    /// Read bits 50..52 as one probe-local 3-bit novelty ordinal.
    pub fn novelty3(edge: CausalEdge64) -> u8 {
        ((edge.0 & PLAST_MASK) >> PLAST_SHIFT) as u8
    }

    /// Write the probe-local novelty reading without changing any other bit.
    pub fn with_novelty3(edge: CausalEdge64, novelty: u8) -> CausalEdge64 {
        CausalEdge64((edge.0 & !PLAST_MASK) | ((u64::from(novelty & 0b111)) << PLAST_SHIFT))
    }

    /// Quantize the maximum entropy of the surviving EpistemicState5
    /// population: ceil(log2(popcount)).  One surviving state is settled (0);
    /// all 24 meaningful V1 states fit in 5.  Integer-only and deterministic.
    ///
    /// `None` when no state survives: the required facts contradict each
    /// other (for example `DIRECT | INDIRECT`). That is not a settled belief,
    /// so it never shares the ordinal `0` with one surviving state.
    pub fn population_novelty(required: u32) -> Option<u8> {
        match facts_population(required).count_ones() {
            0 => None,
            1 => Some(0),
            n => Some((u32::BITS - (n - 1).leading_zeros()) as u8),
        }
    }

    /// The signed i4 is a REACTION reading here, not the operation code.
    /// Positive means the edit raised the terminal frequency; negative means
    /// it lowered it.  Quantize |delta| in 32-point buckets into 1..=7.
    pub fn activation(reaction: Reaction) -> i8 {
        let (factual, counterfactual) = match reaction {
            Reaction::LoadBearing {
                factual,
                counterfactual,
            }
            | Reaction::TruthOnly {
                factual,
                counterfactual,
            } => (factual, counterfactual),
            Reaction::Inert { .. } => return 0,
        };
        let delta = i16::from(counterfactual) - i16::from(factual);
        if delta == 0 {
            return 0;
        }
        let magnitude = (if delta < 0 { -delta } else { delta }) as u16;
        let q = magnitude.div_ceil(32).clamp(1, 7) as i8;
        if delta > 0 {
            q
        } else {
            -q
        }
    }

    pub fn with_activation(edge: CausalEdge64, reaction: Reaction) -> CausalEdge64 {
        edge.with_inference_mantissa(activation(reaction))
    }

    /// W records a BELIEF-STATE UPDATE.  A thought that produced no
    /// EpistemicState5 transition does not steal the breadcrumb.
    pub fn record_belief_update(
        before: CausalEdge64,
        after: CausalEdge64,
        witness: u8,
    ) -> CausalEdge64 {
        if before.epistemic_raw5() == after.epistemic_raw5() {
            after.with_w_slot(before.w_slot())
        } else {
            after.with_w_slot(witness & 0x3f)
        }
    }
}

#[cfg(test)]
use semantic_upper_half::*;

// ── The script ────────────────────────────────────────────────────────────

fn main() {
    let decl = declarations();
    let rd = Reading {
        declarations: &decl,
        class: CERT_CLASS,
        rail: RAIL,
        provenance: EdgeProvenance::V2Stamped,
    };
    let rp = Replay::new();
    let mut with_trial = related_population();
    add_positive_trial(&mut with_trial);
    let model = with_trial.build();
    let ev = Evidence {
        model: &model,
        chain: Some(rp.chain()),
    };
    let show = |label: &str, e: CausalEdge64| {
        println!(
            "  {label:<46} {:?} x {:?}  (bits 40..42 {:?}, raw5 {})",
            topology(e),
            certification(e),
            e.causal_mask(),
            e.epistemic_raw5()
        );
    };
    println!("D-PEARL-IO-0: A -> Y through a candidate B (CUI BONO)");

    let start = query(
        CausalMask::SO,
        Contract::Related,
        Topology2::IndirectUnknown,
    );
    show("START", start);

    let m = reason(start, &ev, Edit::None);
    let e = revise(&m, rd).unwrap();
    println!("  {:?} measured: earned {:?}", m.operation(), m.earned());
    show("after OBSERVATION", e);

    let (same, h) = hydrate(e, (A, 99, Y), &rp.steps, rd).unwrap();
    println!("  candidate 99 proposed: {h:?}");
    show("after PROPOSING an unbound candidate", same);

    let (e, h) = hydrate(e, (A, B, Y), &rp.steps, rd).unwrap();
    println!("  candidate B: {h:?}");
    show("after HYDRATING A -> B -> Y", e);

    let e = with_mask(e, CausalMask::PO);
    let m = reason(e, &ev, Edit::None);
    let e = revise(&m, rd).unwrap();
    println!(
        "  PO measured: earned {:?}, ungrounded {:?}",
        m.earned(),
        m.ungrounded()
    );
    show("after INTERVENTION", e);

    let e = with_mask(e, CausalMask::SPO);
    for (cut, route) in [(1usize, "B -> Y"), (0, "A -> B")] {
        let m = reason(e, &ev, Edit::CutStep(cut));
        let after = revise(&m, rd).unwrap();
        println!(
            "  SPO mask {route}: reaction {:?}, earned {:?}, cf terminal mantissa {:?}",
            m.reaction(),
            m.earned(),
            m.counterfactual_terminal().map(|t| t.inference_mantissa())
        );
        show("after COUNTERFACTUAL", after);
    }

    let e = with_mask(e, CausalMask::SP);
    let m = reason(e, &ev, Edit::None);
    println!("  SP measured: confounded {:?}", m.confounded());

    println!("redundancy, Y = A or B:");
    let (trial, gate) = overdetermined_trial();
    let unit = (gate.a & gate.b).isolate_lowest_one();
    println!(
        "  remove A at a unit with B: reaction {}; with B disabled: reaction {}",
        gate.but_for(unit),
        gate.but_for_given(unit, gate.b)
    );
    let q = query(CausalMask::PO, Contract::Open, Topology2::Direct);
    let m = reason(
        q,
        &Evidence {
            model: &trial,
            chain: None,
        },
        Edit::None,
    );
    show("  randomized arms, PO", revise(&m, rd).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_planner::pearl::{Hydration, Measured};

    fn evidence<'a>(model: &'a Model, rp: &'a Replay) -> Evidence<'a> {
        Evidence {
            model,
            chain: Some(rp.chain()),
        }
    }

    fn rd(decl: &lance_graph_contract::epistemic_state5::Epi5Declarations) -> Reading<'_> {
        Reading {
            declarations: decl,
            class: CERT_CLASS,
            rail: RAIL,
            provenance: EdgeProvenance::V2Stamped,
        }
    }

    fn step(
        e: CausalEdge64,
        mask: CausalMask,
        ev: &Evidence<'_>,
        edit: Edit,
    ) -> (CausalEdge64, Measured) {
        let decl = declarations();
        let m = reason(with_mask(e, mask), ev, edit);
        (revise(&m, rd(&decl)).expect("declared"), m)
    }

    /// D-CE64-SEM-0. One real CUI-BONO trajectory crosses the proposed
    /// upper-half reading without inventing another carrier. Hydration and PO
    /// move Epi5 and therefore advance W; SPO produces a measured activation
    /// but no new belief-state breadcrumb because Epi5 does not move.
    #[test]
    fn semantic_loop_records_only_real_belief_state_updates() {
        let decl = declarations();
        let rp = Replay::new();
        let mut b = related_population();
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);

        let mut start = query(
            CausalMask::SO,
            Contract::Related,
            Topology2::IndirectUnknown,
        )
        .with_w_slot(0);
        start = with_novelty3(
            start,
            population_novelty(fact::IND_UNKNOWN | fact::RELATED).expect("consistent facts"),
        );
        assert_eq!(novelty3(start), 2, "four certification cells still survive");

        // A real A -> B -> Y hydration moves topology, so it earns breadcrumb 1.
        let (hydrated, h) = hydrate(start, (A, B, Y), &rp.steps, rd(&decl)).unwrap();
        assert_eq!(h, Hydration::Hydrated);
        let mut hydrated = record_belief_update(start, hydrated, 1);
        hydrated = with_novelty3(
            hydrated,
            population_novelty(fact::IND_KNOWN | fact::RELATED).expect("consistent facts"),
        );
        assert_eq!(hydrated.w_slot(), 1);
        assert_eq!(novelty3(hydrated), 2);
        assert_eq!(
            (topology(hydrated), certification(hydrated)),
            (Topology2::IndirectKnown, Certification3::Related)
        );

        // The randomized PO arms earn Causes. That is another epistemic move,
        // so W advances and the surviving population collapses to one state.
        let (causal, m) = step(hydrated, CausalMask::PO, &ev, Edit::None);
        assert_eq!(m.earned(), Some(Certification3::Causes));
        let mut causal = record_belief_update(hydrated, causal, 2);
        causal = with_novelty3(
            causal,
            population_novelty(fact::IND_KNOWN | fact::CAUSES).expect("consistent facts"),
        );
        assert_eq!(causal.w_slot(), 2);
        assert_eq!(novelty3(causal), 0, "one epistemic state survives");
        assert_eq!(
            (topology(causal), certification(causal)),
            (Topology2::IndirectKnown, Certification3::Causes)
        );

        // The SPO cut is mechanically real but earns no new Epi5 state.
        // Its measured reaction becomes activation; the belief-update
        // breadcrumb stays on the PO transition that actually changed Epi5.
        let (spo, m) = step(causal, CausalMask::SPO, &ev, Edit::CutStep(1));
        let reaction = m.reaction().expect("counterfactual replay measured");
        let spo = record_belief_update(causal, with_activation(spo, reaction), 3);
        assert_eq!(activation(reaction), -3);
        assert_eq!(spo.inference_mantissa(), -3);
        assert_eq!(
            spo.w_slot(),
            2,
            "no epistemic transition, no new breadcrumb"
        );
        assert_eq!(spo.epistemic_raw5(), causal.epistemic_raw5());
    }

    /// The i4 activation is a measured reaction, not a disguised SPO opcode:
    /// the SAME counterfactual operator yields opposite signs for two cuts.
    #[test]
    fn activation_is_reaction_not_operation_echo() {
        let rp = Replay::new();
        let model = related_population().build();
        let ev = evidence(&model, &rp);
        let q = query(CausalMask::SPO, Contract::Related, Topology2::IndirectKnown);

        let (_, down) = step(q, CausalMask::SPO, &ev, Edit::CutStep(1));
        let (_, up) = step(q, CausalMask::SPO, &ev, Edit::CutStep(0));
        let down = down.reaction().expect("cut 1 measured");
        let up = up.reaction().expect("cut 0 measured");

        assert_eq!(activation(down), -3, "185 -> 120 inhibits");
        assert_eq!(activation(up), 2, "185 -> 225 excites");
        assert_ne!(
            activation(down),
            InferenceType::Counterfactual.to_mantissa(),
            "activation is not the -6 counterfactual opcode"
        );
        assert_ne!(activation(down).signum(), activation(up).signum());
    }

    /// Novelty is the width of the still-legal epistemic population, not a
    /// synonym for causality.  Hydrating the intermediate changes topology but
    /// leaves four certification possibilities; the intervention collapses
    /// those possibilities to one.
    #[test]
    fn novelty_measures_remaining_epistemic_population() {
        let all = facts_population(0);
        let unknown_related = facts_population(fact::IND_UNKNOWN | fact::RELATED);
        let known_related = facts_population(fact::IND_KNOWN | fact::RELATED);
        let known_causes = facts_population(fact::IND_KNOWN | fact::CAUSES);

        assert_eq!(all.count_ones(), 24);
        assert_eq!(unknown_related.count_ones(), 4);
        assert_eq!(known_related.count_ones(), 4);
        assert_eq!(known_causes.count_ones(), 1);

        assert_eq!(population_novelty(0), Some(5));
        assert_eq!(
            population_novelty(fact::IND_UNKNOWN | fact::RELATED),
            Some(2)
        );
        assert_eq!(population_novelty(fact::IND_KNOWN | fact::RELATED), Some(2));
        assert_eq!(population_novelty(fact::IND_KNOWN | fact::CAUSES), Some(0));

        // Codex #1393: contradictory facts leave no state at all. That is not
        // the settled single state above, so it must not read as 0.
        assert_eq!(facts_population(fact::DIRECT | fact::INDIRECT), 0);
        assert_eq!(population_novelty(fact::DIRECT | fact::INDIRECT), None);
    }

    /// Orientation is orthogonal to operator selection and certification in
    /// this fixture. Every 3-bit direction reaches the same PO measurement,
    /// and revision preserves the chosen orientation bit-for-bit.
    ///
    /// Ported to #1391's API: a `Measured` now carries the edge it measured,
    /// so the comparison is over what the operator computed, not the whole
    /// value (the edges differ by their direction bits by construction).
    #[test]
    fn orientation_does_not_secretly_select_the_pearl_operator() {
        let decl = declarations();
        let rp = Replay::new();
        let mut b = related_population();
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);

        let baseline_q = query(CausalMask::PO, Contract::Related, Topology2::IndirectKnown);
        let baseline = reason(baseline_q, &ev, Edit::None);
        let computed = |m: &Measured| {
            (
                m.operation(),
                m.earned(),
                m.ungrounded(),
                m.reaction(),
                m.confounded(),
                m.counterfactual_terminal(),
            )
        };

        for direction in 0u8..8 {
            let mut q = baseline_q;
            q.set_direction(direction);
            let measured = reason(q, &ev, Edit::None);
            assert_eq!(
                computed(&measured),
                computed(&baseline),
                "direction {direction:03b}"
            );
            assert_eq!(measured.edge(), q);
            let out = revise(&measured, rd(&decl)).unwrap();
            assert_eq!(out.direction(), direction);
            assert_eq!(certification(out), Certification3::Causes);
        }
    }
}
