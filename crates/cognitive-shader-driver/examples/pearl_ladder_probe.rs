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
//! `relational_certification_probe`), `lance_graph_planner::dismech_counterfactual`
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
//! - Production placement of `reason` / `revise`. This is a probe.
//! - Whole-node removal of B: in a two-step chain B's only mechanisms are the
//!   two routes, so route cuts are run and node removal is not.
//! - A responsibility framework beyond the single contingency.
//!
//! Run: `cargo run -p cognitive-shader-driver --features with-planner --example pearl_ladder_probe`
//! Tests: `cargo test -p cognitive-shader-driver --features with-planner --example pearl_ladder_probe`

use causal_edge::edge::{CausalEdge64, InferenceType};
use causal_edge::pearl::CausalMask;
use causal_edge::plasticity::PlasticityState;
use causal_edge::tables::NarsTables;
use lance_graph_contract::causal_audit::SupportBasis;
use lance_graph_contract::epistemic_state5::{fact, facts_population, Certification3, Topology2};
use lance_graph_planner::dismech_counterfactual::{CutContext, DEFAULT_FREQUENCY_BAR};
use lance_graph_planner::dismech_replay::{ChainStep, ComposeTables};

#[path = "shared/certification_model.rs"]
mod certification_model;
#[path = "shared/certification_reading.rs"]
mod certification_reading;

use certification_model::*;
use certification_reading::*;

// ── The operator layer ────────────────────────────────────────────────────

/// Dispatch, measurement and write-back. A separate module so that a
/// [`pearl::Measured`] can only be produced by running an operator: its
/// fields are private, so no caller outside this module can assert that an
/// intervention or a counterfactual passed.
mod pearl {
    use super::certification_model::{compare, Model, NotGrounded};
    use super::certification_reading::{declarations, stamp, Contract, Refusal, CERT_CLASS, RAIL};
    use causal_edge::edge::CausalEdge64;
    use causal_edge::pearl::CausalMask;
    use lance_graph_contract::band_reading::EdgeProvenance;
    use lance_graph_contract::epistemic_state5::{Certification3, Epi5Gen, Topology2};
    use lance_graph_planner::dismech_counterfactual::{counterfactual_replay, CutContext, Verdict};
    use lance_graph_planner::dismech_replay::ChainStep;

    /// The operator bits 40..42 select.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Operation {
        /// `SO`: observational folds.
        Association,
        /// `PO`: executed randomized arms.
        Intervention,
        /// `SPO`: counterfactual replay with one route cut.
        Counterfactual,
        /// `SP`: observational direction against interventional direction.
        ConfounderCheck,
        /// A projection with no operator here (`None`, `O`, `P`, `S`).
        NoOperator(CausalMask),
    }

    impl Operation {
        pub fn of(mask: CausalMask) -> Self {
            match mask {
                CausalMask::SO => Operation::Association,
                CausalMask::PO => Operation::Intervention,
                CausalMask::SPO => Operation::Counterfactual,
                CausalMask::SP => Operation::ConfounderCheck,
                other => Operation::NoOperator(other),
            }
        }
    }

    /// Why an operator could not decide.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Ungrounded {
        /// A P7a fold lacked what it needs.
        Model(NotGrounded),
        /// A counterfactual was asked for without a sealed chain.
        NoChain,
        /// A counterfactual was asked for without an edit.
        NoEdit,
        /// The edit names no step of the chain.
        CutOutOfRange(usize),
        /// The replay refused (sequence reservation).
        Replay,
        /// One direction of the confounder check is a tie.
        NoDirection,
        /// The projection selects no operator.
        NoOperator,
    }

    /// What one route cut did to the chain's answer, with both frequencies.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Reaction {
        /// The verdict changed.
        LoadBearing { factual: u8, counterfactual: u8 },
        /// The truth moved, the verdict did not.
        TruthOnly { factual: u8, counterfactual: u8 },
        /// Nothing moved.
        Inert { frequency: u8 },
    }

    /// A counterfactual edit. The base generation is never modified; the
    /// operator replays the same chain with the edit applied.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Edit {
        None,
        /// Mask one route (one step) of the chain.
        CutStep(usize),
    }

    /// The sealed evidence an operator may read. Read-only.
    pub struct Evidence<'a> {
        pub model: &'a Model,
        pub chain: Option<Chain<'a>>,
    }

    /// A sealed chain for the counterfactual replay.
    #[derive(Clone, Copy)]
    pub struct Chain<'a> {
        pub steps: &'a [ChainStep],
        pub seed: CausalEdge64,
        pub cx: CutContext<'a>,
    }

    /// The result of running one operator. Only [`reason`] constructs it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Measured {
        operation: Operation,
        earned: Option<Certification3>,
        ungrounded: Option<Ungrounded>,
        reaction: Option<Reaction>,
        confounded: Option<bool>,
        /// The counterfactual arm's terminal edge, kept as witness. Never
        /// written back.
        counterfactual_terminal: Option<CausalEdge64>,
    }

    impl Measured {
        pub fn operation(&self) -> Operation {
            self.operation
        }
        pub fn earned(&self) -> Option<Certification3> {
            self.earned
        }
        pub fn ungrounded(&self) -> Option<Ungrounded> {
            self.ungrounded
        }
        pub fn reaction(&self) -> Option<Reaction> {
            self.reaction
        }
        pub fn confounded(&self) -> Option<bool> {
            self.confounded
        }
        pub fn counterfactual_terminal(&self) -> Option<CausalEdge64> {
            self.counterfactual_terminal
        }

        fn new(operation: Operation) -> Self {
            Measured {
                operation,
                earned: None,
                ungrounded: None,
                reaction: None,
                confounded: None,
                counterfactual_terminal: None,
            }
        }
    }

    /// Run the operator the edge's projection selects. Reads bits 40..42 and
    /// nothing else of the edge: the epistemic state, mantissa and truth are
    /// not inputs to the choice.
    pub fn reason(edge: CausalEdge64, ev: &Evidence<'_>, edit: Edit) -> Measured {
        let op = Operation::of(edge.causal_mask());
        let mut out = Measured::new(op);
        let m = ev.model;
        match op {
            Operation::Association => match m.sourced() {
                Err(e) => out.ungrounded = Some(Ungrounded::Model(e)),
                Ok(()) => {
                    let c = m.certify_observational();
                    if c != Contract::Open {
                        out.earned = Some(c.certification());
                    }
                }
            },
            Operation::Intervention => match m.causes() {
                Ok(true) => out.earned = Some(Certification3::Causes),
                Ok(false) => {}
                Err(e) => out.ungrounded = Some(Ungrounded::Model(e)),
            },
            Operation::Counterfactual => {
                let Some(chain) = ev.chain else {
                    out.ungrounded = Some(Ungrounded::NoChain);
                    return out;
                };
                let Edit::CutStep(i) = edit else {
                    out.ungrounded = Some(Ungrounded::NoEdit);
                    return out;
                };
                match counterfactual_replay(chain.steps, i, chain.seed, chain.cx) {
                    None => out.ungrounded = Some(Ungrounded::CutOutOfRange(i)),
                    Some(Err(_)) => out.ungrounded = Some(Ungrounded::Replay),
                    Some(Ok(cf)) => {
                        let f = cf.factual.last().map_or(0, |r| r.edge.frequency_u8());
                        let c = cf
                            .counterfactual
                            .last()
                            .map_or(0, |r| r.edge.frequency_u8());
                        out.reaction = Some(if cf.role.factual != cf.role.counterfactual {
                            Reaction::LoadBearing {
                                factual: f,
                                counterfactual: c,
                            }
                        } else if f != c || cf.factual.len() != cf.counterfactual.len() {
                            Reaction::TruthOnly {
                                factual: f,
                                counterfactual: c,
                            }
                        } else {
                            Reaction::Inert { frequency: f }
                        });
                        out.counterfactual_terminal = cf.counterfactual.last().map(|r| r.edge);
                        debug_assert!(matches!(
                            cf.role.factual,
                            Verdict::Consistent | Verdict::Inconsistent
                        ));
                    }
                }
            }
            Operation::ConfounderCheck => {
                let obs = direction(m.outcome, m.universe & m.exposed, m.universe & !m.exposed);
                let trial = direction(m.outcome, m.trial & m.assigned, m.trial & !m.assigned);
                match (obs, trial) {
                    (Ok(Some(so)), Ok(Some(po))) => {
                        out.confounded = Some(CausalMask::simpsons_paradox_risk(so, po));
                    }
                    (Err(e), _) | (_, Err(e)) => out.ungrounded = Some(Ungrounded::Model(e)),
                    _ => out.ungrounded = Some(Ungrounded::NoDirection),
                }
            }
            Operation::NoOperator(_) => out.ungrounded = Some(Ungrounded::NoOperator),
        }
        out
    }

    /// The direction triad `simpsons_paradox_risk` reads: bit 2 set when the
    /// outcome is worse (here: rarer) in the exposed group. `None` on a tie.
    fn direction(outcome: u64, a: u64, b: u64) -> Result<Option<u8>, NotGrounded> {
        if compare(outcome, a, b, true)? {
            Ok(Some(0b000))
        } else if compare(outcome, b, a, true)? {
            Ok(Some(0b100))
        } else {
            Ok(None)
        }
    }

    /// Write a measurement back. Certification rises to what the operator
    /// earned when that is strictly stronger; nothing else moves, and nothing
    /// is ever lowered.
    pub fn revise(edge: CausalEdge64, m: &Measured) -> Result<CausalEdge64, Refusal> {
        let decl = declarations();
        let state = decl
            .project_state5(
                CERT_CLASS,
                RAIL,
                Epi5Gen::V1,
                edge.epistemic_raw5(),
                EdgeProvenance::V2Stamped,
            )
            .map_err(Refusal::Reading)?;
        let current = Contract::from_certification(state.certification());
        let next = match m.earned.map(Contract::from_certification) {
            Some(c) if c != current && c.entails(current) => c,
            _ => current,
        };
        Ok(stamp(edge, next, state.topology()))
    }

    /// What a candidate intermediate B turned out to be.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Hydration {
        /// No binding of B to the path exists in the sealed chain.
        ProposedOnly,
        /// One of `A → B`, `B → Y` exists.
        Partial,
        /// Both bindings exist.
        Hydrated,
    }

    /// The topology gate. `IndirectUnknown` becomes `IndirectKnown` only when
    /// both bindings `a → b` and `b → y` are present in the sealed chain.
    /// Certification never moves here.
    pub fn hydrate(
        edge: CausalEdge64,
        (a, b, y): (u8, u8, u8),
        sealed: &[ChainStep],
    ) -> Result<(CausalEdge64, Hydration), Refusal> {
        let has = |s: u8, o: u8| sealed.iter().any(|(_, e)| e.s_idx() == s && e.o_idx() == o);
        let found = match (has(a, b), has(b, y)) {
            (true, true) => Hydration::Hydrated,
            (false, false) => Hydration::ProposedOnly,
            _ => Hydration::Partial,
        };
        let state = declarations()
            .project_state5(
                CERT_CLASS,
                RAIL,
                Epi5Gen::V1,
                edge.epistemic_raw5(),
                EdgeProvenance::V2Stamped,
            )
            .map_err(Refusal::Reading)?;
        let topology =
            if found == Hydration::Hydrated && state.topology() == Topology2::IndirectUnknown {
                Topology2::IndirectKnown
            } else {
                state.topology()
            };
        let contract = Contract::from_certification(state.certification());
        Ok((stamp(edge, contract, topology), found))
    }
}

use pearl::*;

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
mod semantic_upper_half {
    use super::*;
    use causal_edge::layout::{PLAST_MASK, PLAST_SHIFT};

    /// Read bits 50..52 as one probe-local 3-bit novelty ordinal.
    pub fn novelty3(edge: CausalEdge64) -> u8 {
        ((edge.0 & PLAST_MASK) >> PLAST_SHIFT) as u8
    }

    /// Write the probe-local novelty reading without changing any other bit.
    pub fn with_novelty3(edge: CausalEdge64, novelty: u8) -> CausalEdge64 {
        CausalEdge64(
            (edge.0 & !PLAST_MASK) | ((u64::from(novelty & 0b111)) << PLAST_SHIFT),
        )
    }

    /// Quantize the maximum entropy of the surviving EpistemicState5
    /// population: ceil(log2(popcount)).  One surviving state is settled (0);
    /// all 24 meaningful V1 states fit in 5.  Integer-only and deterministic.
    pub fn population_novelty(required: u32) -> u8 {
        let n = facts_population(required).count_ones();
        if n <= 1 {
            0
        } else {
            (u32::BITS - (n - 1).leading_zeros()) as u8
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
        let q = ((magnitude + 31) / 32).clamp(1, 7) as i8;
        if delta > 0 { q } else { -q }
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

use semantic_upper_half::*;

// ── The script ────────────────────────────────────────────────────────────

fn main() {
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
    let e = revise(start, &m).unwrap();
    println!("  {:?} measured: earned {:?}", m.operation(), m.earned());
    show("after OBSERVATION", e);

    let (same, h) = hydrate(e, (A, 99, Y), &rp.steps).unwrap();
    println!("  candidate 99 proposed: {h:?}");
    show("after PROPOSING an unbound candidate", same);

    let (e, h) = hydrate(e, (A, B, Y), &rp.steps).unwrap();
    println!("  candidate B: {h:?}");
    show("after HYDRATING A -> B -> Y", e);

    let e = with_mask(e, CausalMask::PO);
    let m = reason(e, &ev, Edit::None);
    let e = revise(e, &m).unwrap();
    println!(
        "  PO measured: earned {:?}, ungrounded {:?}",
        m.earned(),
        m.ungrounded()
    );
    show("after INTERVENTION", e);

    let e = with_mask(e, CausalMask::SPO);
    for (cut, route) in [(1usize, "B -> Y"), (0, "A -> B")] {
        let m = reason(e, &ev, Edit::CutStep(cut));
        let after = revise(e, &m).unwrap();
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
    show("  randomized arms, PO", revise(q, &m).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_edge::layout::EPISTEMIC_MASK;

    fn evidence<'a>(model: &'a Model, rp: &'a Replay) -> Evidence<'a> {
        Evidence {
            model,
            chain: Some(rp.chain()),
        }
    }

    /// Run one operator and write it back.
    fn step(
        e: CausalEdge64,
        mask: CausalMask,
        ev: &Evidence<'_>,
        edit: Edit,
    ) -> (CausalEdge64, Measured) {
        let e = with_mask(e, mask);
        let m = reason(e, ev, edit);
        (revise(e, &m).expect("declared"), m)
    }

    /// The whole script, pinned: the edge leaves with a different state only
    /// where an operator earned it.
    #[test]
    fn cui_bono_script_moves_the_state_only_where_evidence_earned_it() {
        let rp = Replay::new();
        let mut b = related_population();
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);
        let start = query(
            CausalMask::SO,
            Contract::Related,
            Topology2::IndirectUnknown,
        );

        let (e, m) = step(start, CausalMask::SO, &ev, Edit::None);
        assert_eq!(m.earned(), Some(Certification3::Related));
        assert_eq!(
            (topology(e), certification(e)),
            (Topology2::IndirectUnknown, Certification3::Related)
        );

        let (e, h) = hydrate(e, (A, B, Y), &rp.steps).unwrap();
        assert_eq!(h, Hydration::Hydrated);
        assert_eq!(
            (topology(e), certification(e)),
            (Topology2::IndirectKnown, Certification3::Related)
        );

        let (e, m) = step(e, CausalMask::PO, &ev, Edit::None);
        assert_eq!(m.earned(), Some(Certification3::Causes));
        assert_eq!(
            (topology(e), certification(e)),
            (Topology2::IndirectKnown, Certification3::Causes)
        );

        let (after, m) = step(e, CausalMask::SPO, &ev, Edit::CutStep(1));
        assert_eq!(
            m.reaction(),
            Some(Reaction::LoadBearing {
                factual: 185,
                counterfactual: 120
            })
        );
        assert_eq!(after.epistemic_raw5(), e.epistemic_raw5());
        let (_, m) = step(e, CausalMask::SPO, &ev, Edit::CutStep(0));
        assert_eq!(
            m.reaction(),
            Some(Reaction::TruthOnly {
                factual: 185,
                counterfactual: 225
            })
        );
    }

    /// F1. Observation alone never reaches `Causes`, even when the sealed
    /// model also holds a positive trial the SO operator could have read.
    #[test]
    fn f1_observation_alone_never_promotes_to_causes() {
        let mut fixtures = vec![
            related_population(),
            robust_association(),
            simpson_population(),
        ];
        let mut many = robust_association();
        many.sources(SupportBasis::DirectlyObserved, &(3..40).collect::<Vec<_>>());
        fixtures.push(many);
        let rp = Replay::new();
        for mut b in fixtures {
            add_positive_trial(&mut b);
            let model = b.build();
            assert_eq!(
                model.certify(),
                Contract::Causes,
                "the trial is there to be ignored"
            );
            let ev = evidence(&model, &rp);
            let (e, m) = step(
                query(CausalMask::SO, Contract::Open, Topology2::Direct),
                CausalMask::SO,
                &ev,
                Edit::None,
            );
            assert_ne!(m.earned(), Some(Certification3::Causes));
            assert_ne!(certification(e), Certification3::Causes);
            assert!(
                m.earned().is_some(),
                "anti-vacuity: SO must earn something here"
            );
        }
    }

    /// F2. Setting `SPO` certifies nothing: with no chain the operator is not
    /// grounded and the state is unchanged.
    #[test]
    fn f2_setting_the_projection_certifies_nothing() {
        let model = related_population().build();
        let ev = Evidence {
            model: &model,
            chain: None,
        };
        for c in Contract::ALL {
            let q = query(CausalMask::SPO, c, Topology2::IndirectUnknown);
            let m = reason(q, &ev, Edit::CutStep(1));
            assert_eq!(m.ungrounded(), Some(Ungrounded::NoChain));
            assert_eq!(revise(q, &m).unwrap(), q);
        }
    }

    /// F3. A load-bearing counterfactual reaction does not become `Causes`.
    #[test]
    fn f3_a_load_bearing_cut_is_not_causes() {
        let rp = Replay::new();
        let model = related_population().build();
        let ev = evidence(&model, &rp);
        for c in [Contract::Open, Contract::Related, Contract::CausalCandidate] {
            let q = query(CausalMask::SPO, c, Topology2::IndirectKnown);
            let m = reason(q, &ev, Edit::CutStep(1));
            assert!(matches!(m.reaction(), Some(Reaction::LoadBearing { .. })));
            assert_eq!(m.earned(), None);
            assert_eq!(certification(revise(q, &m).unwrap()), c.certification());
        }
    }

    /// F4. The counterfactual arm is tagged −6 and is never written back as
    /// observed truth: the revised edge keeps its own frequency, confidence
    /// and mantissa.
    #[test]
    fn f4_the_counterfactual_branch_is_not_written_back() {
        let rp = Replay::new();
        let model = related_population().build();
        let ev = evidence(&model, &rp);
        let q = query(CausalMask::SPO, Contract::Related, Topology2::IndirectKnown);
        let m = reason(q, &ev, Edit::CutStep(1));
        let t = m.counterfactual_terminal().expect("replayed");
        assert_eq!(
            t.inference_mantissa(),
            InferenceType::Counterfactual.to_mantissa()
        );
        assert_ne!(
            t.frequency_u8(),
            q.frequency_u8(),
            "anti-vacuity: the arm differs"
        );
        let out = revise(q, &m).unwrap();
        assert_eq!(out.frequency_u8(), q.frequency_u8());
        assert_eq!(out.confidence_u8(), q.confidence_u8());
        assert_eq!(out.inference_mantissa(), q.inference_mantissa());
    }

    /// F5. Proposing B changes nothing; half a path changes nothing; only both
    /// bindings move the topology, and never the certification.
    #[test]
    fn f5_a_proposed_beneficiary_is_not_evidence() {
        let rp = Replay::new();
        let q = query(
            CausalMask::SO,
            Contract::Related,
            Topology2::IndirectUnknown,
        );
        let (e, h) = hydrate(q, (A, 99, Y), &rp.steps).unwrap();
        assert_eq!((e, h), (q, Hydration::ProposedOnly));
        let half = &rp.steps[..1];
        let (e, h) = hydrate(q, (A, B, Y), half).unwrap();
        assert_eq!((e, h), (q, Hydration::Partial));
        let (e, h) = hydrate(q, (A, B, Y), &rp.steps).unwrap();
        assert_eq!(h, Hydration::Hydrated);
        assert_eq!(topology(e), Topology2::IndirectKnown);
        assert_eq!(certification(e), Certification3::Related);
    }

    /// F6. Simpson: SO and PO disagree in direction, SP reports it, and the
    /// diagnostic does not demote an intervention-certified `Causes`.
    #[test]
    fn f6_simpson_does_not_demote_intervention_causes() {
        let rp = Replay::new();
        let mut b = Builder::new();
        // Pooled: exposed rarer outcome (4/10 vs 7/10); trial: exposed better.
        b.cell(0, true, 8, 2);
        b.cell(0, false, 2, 0);
        b.cell(1, true, 2, 2);
        b.cell(1, false, 8, 7);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);
        let q = query(CausalMask::PO, Contract::Open, Topology2::Direct);
        let (e, _) = step(q, CausalMask::PO, &ev, Edit::None);
        assert_eq!(certification(e), Certification3::Causes);
        let (after, m) = step(e, CausalMask::SP, &ev, Edit::None);
        assert_eq!(m.confounded(), Some(true));
        assert_eq!(certification(after), Certification3::Causes);
        // And SO on the same population does not lower it either.
        let (after, _) = step(after, CausalMask::SO, &ev, Edit::None);
        assert_eq!(certification(after), Certification3::Causes);
    }

    /// F6 silence twin: same-direction populations are not reported
    /// confounded.
    #[test]
    fn f6_agreeing_directions_are_not_confounded() {
        let rp = Replay::new();
        let mut b = robust_association();
        add_positive_trial(&mut b);
        let model = b.build();
        let (_, m) = step(
            query(CausalMask::SP, Contract::Open, Topology2::Direct),
            CausalMask::SP,
            &evidence(&model, &rp),
            Edit::None,
        );
        assert_eq!(m.confounded(), Some(false));
    }

    /// F7. Y = A or B: removing A where B holds shows no reaction, the
    /// contingency "B disabled" shows one, and the trial certifies `Causes`.
    /// A null removal never demotes.
    #[test]
    fn f7_one_at_a_time_removal_is_not_the_definition_of_causality() {
        let (trial, gate) = overdetermined_trial();
        let unit = (gate.a & gate.b).isolate_lowest_one();
        assert_ne!(unit, 0);
        assert!(!gate.but_for(unit), "B keeps Y: no reaction");
        assert!(
            gate.but_for_given(unit, gate.b),
            "with B disabled, removing A removes Y"
        );
        let a_only = (gate.a & !gate.b).isolate_lowest_one();
        assert!(
            gate.but_for(a_only),
            "anti-vacuity: removal can react at all"
        );

        let rp = Replay::new();
        let ev = Evidence {
            model: &trial,
            chain: Some(rp.chain()),
        };
        let (e, _) = step(
            query(CausalMask::PO, Contract::Open, Topology2::Direct),
            CausalMask::PO,
            &ev,
            Edit::None,
        );
        assert_eq!(certification(e), Certification3::Causes);
        // A counterfactual with no verdict change on this edge must not lower it.
        let (after, m) = step(e, CausalMask::SPO, &ev, Edit::CutStep(0));
        assert!(matches!(m.reaction(), Some(Reaction::TruthOnly { .. })));
        assert_eq!(certification(after), Certification3::Causes);
    }

    /// F8. Every operator writes bits 59..63 only.
    #[test]
    fn f8_only_bits_59_to_63_move() {
        let rp = Replay::new();
        let mut b = related_population();
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);
        let mut moved = 0;
        for v in 0u8..8 {
            let mask = CausalMask::from_bits(v);
            for mantissa in [-6i8, 0, 6] {
                let q = query(mask, Contract::Open, Topology2::IndirectUnknown)
                    .with_inference_mantissa(mantissa);
                let m = reason(q, &ev, Edit::CutStep(1));
                let out = revise(q, &m).unwrap();
                assert_eq!((out.0 ^ q.0) & !EPISTEMIC_MASK, 0, "{mask:?} {mantissa}");
                moved += usize::from(out != q);
            }
        }
        assert!(moved > 0, "anti-vacuity: some operator must move the state");
    }

    /// F9. Same sealed evidence, same edit: same measurement, same bits.
    #[test]
    fn f9_replay_is_deterministic() {
        let rp = Replay::new();
        let mut b = related_population();
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);
        for v in 0u8..8 {
            let q = query(
                CausalMask::from_bits(v),
                Contract::Related,
                Topology2::IndirectKnown,
            );
            for edit in [Edit::None, Edit::CutStep(0), Edit::CutStep(1)] {
                let (m1, m2) = (reason(q, &ev, edit), reason(q, &ev, edit));
                assert_eq!(m1, m2);
                assert_eq!(revise(q, &m1).unwrap(), revise(q, &m2).unwrap());
            }
        }
    }

    /// F10. "The intervention passed" cannot be asserted: intervention
    /// receipts without executed arms are not grounded, and executed arms
    /// without an effect earn nothing. (`Measured` has private fields, so a
    /// caller cannot construct a passing result either.)
    #[test]
    fn f10_a_pass_must_be_executed_not_claimed() {
        let rp = Replay::new();
        let mut b = related_population();
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
        let claimed = b.build();
        let (e, m) = step(
            query(CausalMask::PO, Contract::Related, Topology2::Direct),
            CausalMask::PO,
            &evidence(&claimed, &rp),
            Edit::None,
        );
        assert_eq!(
            m.ungrounded(),
            Some(Ungrounded::Model(NotGrounded::EmptyArm))
        );
        assert_eq!(certification(e), Certification3::Related);

        let mut b = related_population();
        b.arm(true, 6, 3);
        b.arm(false, 6, 3);
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
        let null = b.build();
        let (e, m) = step(
            query(CausalMask::PO, Contract::Related, Topology2::Direct),
            CausalMask::PO,
            &evidence(&null, &rp),
            Edit::None,
        );
        assert_eq!((m.earned(), m.ungrounded()), (None, None));
        assert_eq!(certification(e), Certification3::Related);
    }

    /// F11. Bits 59..63 do not select the operation and carry no reasoning
    /// band: two edges differing only there get the same measurement.
    #[test]
    fn f11_bits_59_to_63_are_epistemic_output_not_a_reasoning_band() {
        let rp = Replay::new();
        let mut b = related_population();
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);
        for v in 0u8..8 {
            let mask = CausalMask::from_bits(v);
            let base = reason(
                query(mask, Contract::Open, Topology2::Direct),
                &ev,
                Edit::CutStep(1),
            );
            for c in Contract::ALL {
                for t in Topology2::ALL {
                    assert_eq!(reason(query(mask, c, t), &ev, Edit::CutStep(1)), base);
                }
            }
        }
        let src = include_str!("pearl_ladder_probe.rs");
        assert!(!src.contains(concat!("Reasoning", "Band")));
    }

    /// F12. Projection, mantissa and epistemic state are separate: the
    /// operation follows the projection whatever the mantissa says, and
    /// revision keeps both.
    #[test]
    fn f12_projection_mantissa_and_state_are_not_aliases() {
        let rp = Replay::new();
        let mut b = related_population();
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);
        for v in 0u8..8 {
            let mask = CausalMask::from_bits(v);
            for mantissa in [
                InferenceType::Intervention.to_mantissa(),
                InferenceType::Counterfactual.to_mantissa(),
                0,
            ] {
                let q = query(mask, Contract::Open, Topology2::Direct)
                    .with_inference_mantissa(mantissa);
                let m = reason(q, &ev, Edit::CutStep(1));
                assert_eq!(m.operation(), Operation::of(mask));
                let out = revise(q, &m).unwrap();
                assert_eq!(out.causal_mask(), mask);
                assert_eq!(out.inference_mantissa(), mantissa);
            }
        }
        // The counterfactual mantissa on an SO edge still runs association.
        let q = query(CausalMask::SO, Contract::Open, Topology2::Direct)
            .with_inference_mantissa(InferenceType::Counterfactual.to_mantissa());
        assert_eq!(
            reason(q, &ev, Edit::None).operation(),
            Operation::Association
        );
    }

    /// D-CE64-SEM-0. One real CUI-BONO trajectory crosses the proposed
    /// upper-half reading without inventing another carrier. Hydration and PO
    /// move Epi5 and therefore advance W; SPO produces a measured activation
    /// but no new belief-state breadcrumb because Epi5 does not move.
    #[test]
    fn semantic_loop_records_only_real_belief_state_updates() {
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
            population_novelty(fact::IND_UNKNOWN | fact::RELATED),
        );
        assert_eq!(novelty3(start), 2, "four certification cells still survive");

        // A real A -> B -> Y hydration moves topology, so it earns breadcrumb 1.
        let (hydrated, h) = hydrate(start, (A, B, Y), &rp.steps).unwrap();
        assert_eq!(h, Hydration::Hydrated);
        let mut hydrated = record_belief_update(start, hydrated, 1);
        hydrated = with_novelty3(
            hydrated,
            population_novelty(fact::IND_KNOWN | fact::RELATED),
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
            population_novelty(fact::IND_KNOWN | fact::CAUSES),
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
        assert_eq!(spo.w_slot(), 2, "no epistemic transition, no new breadcrumb");
        assert_eq!(spo.epistemic_raw5(), causal.epistemic_raw5());
    }

    /// The i4 activation is a measured reaction, not a disguised SPO opcode:
    /// the SAME counterfactual operator yields opposite signs for two cuts.
    #[test]
    fn activation_is_reaction_not_operation_echo() {
        let rp = Replay::new();
        let model = related_population().build();
        let ev = evidence(&model, &rp);
        let q = query(
            CausalMask::SPO,
            Contract::Related,
            Topology2::IndirectKnown,
        );

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

        assert_eq!(population_novelty(0), 5);
        assert_eq!(population_novelty(fact::IND_UNKNOWN | fact::RELATED), 2);
        assert_eq!(population_novelty(fact::IND_KNOWN | fact::RELATED), 2);
        assert_eq!(population_novelty(fact::IND_KNOWN | fact::CAUSES), 0);
    }

    /// Orientation is orthogonal to operator selection and certification in
    /// this fixture. Every 3-bit direction reaches the same PO measurement,
    /// and revision preserves the chosen orientation bit-for-bit.
    #[test]
    fn orientation_does_not_secretly_select_the_pearl_operator() {
        let rp = Replay::new();
        let mut b = related_population();
        add_positive_trial(&mut b);
        let model = b.build();
        let ev = evidence(&model, &rp);

        let baseline_q = query(CausalMask::PO, Contract::Related, Topology2::IndirectKnown);
        let baseline = reason(baseline_q, &ev, Edit::None);

        for direction in 0u8..8 {
            let mut q = baseline_q;
            q.set_direction(direction);
            let measured = reason(q, &ev, Edit::None);
            assert_eq!(measured, baseline, "direction {direction:03b}");
            let out = revise(q, &measured).unwrap();
            assert_eq!(out.direction(), direction);
            assert_eq!(certification(out), Certification3::Causes);
        }
    }

}
