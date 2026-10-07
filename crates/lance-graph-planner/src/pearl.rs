//! **D-PEARL-PROD-0 — Pearl's ladder, in and out of `CausalEdge64`.**
//!
//! An edge enters with one `EpistemicState5` (bits 59..63). Its Pearl
//! projection (bits 40..42) selects which existing operator runs. The operator
//! measures, and only the measurement decides whether bits 59..63 change.
//!
//! ```text
//! CausalEdge64 ──bits 40..42──► reason ──► Measured ──revise──► bits 59..63
//!
//!     SO  ► observational folds   (certification::CertificationModel::observational_certification)
//!     PO  ► executed trial arms   (CertificationModel::causes)
//!     SPO ► counterfactual replay (crate::chain_counterfactual::counterfactual_replay)
//!     SP  ► SO direction vs PO direction (causal_edge::CausalMask::simpsons_paradox_risk)
//! ```
//!
//! It lives in this crate because this is the crate that depends on both
//! `lance-graph-contract` (the certification obligations, `EpistemicState5`)
//! and `causal-edge` (`CausalEdge64`), the same reason
//! [`crate::chain_counterfactual`] lives here.
//!
//! # The write-back rule
//!
//! - **Certification only rises, and only to what the operator computed.** SO
//!   can earn at most `CausalCandidate`. PO is the only path to `Causes`, and
//!   only from executed randomized arms. SPO and SP earn nothing: a
//!   counterfactual reaction and a confounding diagnostic are evidence beside
//!   the certification, not certifications.
//! - **Topology changes only through [`hydrate`]**: `IndirectUnknown` becomes
//!   `IndirectKnown` when both bindings `A → B` and `B → Y` exist in the sealed
//!   chain. Proposing a candidate is not evidence.
//! - **Only bits 59..63 move.** The projection, mantissa, frequency and
//!   confidence of the edge are inputs, never outputs.
//! - **A pass cannot be asserted.** [`Measured`] has private fields; only
//!   [`reason`] produces one, by running the operator.
//!
//! # Measured on the probe that shaped this module
//!
//! `cognitive-shader-driver/examples/pearl_ladder_probe.rs` (D-PEARL-IO-0):
//! the counterfactual replay's verdict is a NARS revision over step
//! frequencies, so a cut can raise the terminal frequency; the reaction keeps
//! both frequencies. One-at-a-time removal is not the causal test: under
//! Y = A or B, removing A where B holds changes nothing while the trial
//! certifies `Causes`.

use causal_edge::{CausalEdge64, CausalMask};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::certification::{compare, CertificationModel, NotGrounded};
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5ReadError, EpistemicState5, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;

use crate::chain_counterfactual::{counterfactual_replay, CutContext};
use crate::chain_replay::ChainStep;

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
    /// A projection with no operator (`None`, `O`, `P`, `S`).
    NoOperator(CausalMask),
}

impl Operation {
    /// The operator a projection selects.
    #[must_use]
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
    /// A certification fold lacked what it needs.
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

/// A counterfactual edit. The base chain is never modified; the operator
/// replays it with the edit applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edit {
    /// No edit.
    None,
    /// Mask one route (one step) of the chain.
    CutStep(usize),
}

/// The sealed evidence an operator may read. Read-only.
pub struct Evidence<'a> {
    /// The sealed population, with any executed randomized arms.
    pub model: &'a CertificationModel,
    /// The sealed chain, for counterfactual replay.
    pub chain: Option<Chain<'a>>,
}

/// A sealed chain for the counterfactual replay.
#[derive(Clone, Copy)]
pub struct Chain<'a> {
    /// The recorded steps.
    pub steps: &'a [ChainStep],
    /// The replay's seed edge.
    pub seed: CausalEdge64,
    /// Tables, owner, sequence base and verdict bar, shared by both arms.
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
    counterfactual_terminal: Option<CausalEdge64>,
}

impl Measured {
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

    /// The operator that ran.
    #[must_use]
    pub fn operation(&self) -> Operation {
        self.operation
    }
    /// The certification the operator computed, if any.
    #[must_use]
    pub fn earned(&self) -> Option<Certification3> {
        self.earned
    }
    /// Why the operator could not decide, if it could not.
    #[must_use]
    pub fn ungrounded(&self) -> Option<Ungrounded> {
        self.ungrounded
    }
    /// The counterfactual reaction (`SPO` only).
    #[must_use]
    pub fn reaction(&self) -> Option<Reaction> {
        self.reaction
    }
    /// Whether the observational and interventional directions disagree
    /// (`SP` only).
    #[must_use]
    pub fn confounded(&self) -> Option<bool> {
        self.confounded
    }
    /// The counterfactual arm's terminal edge, tagged −6. Witness only;
    /// [`revise`] never writes it back.
    #[must_use]
    pub fn counterfactual_terminal(&self) -> Option<CausalEdge64> {
        self.counterfactual_terminal
    }
}

/// Run the operator the edge's projection selects. Reads bits 40..42 and
/// nothing else of the edge: the epistemic state, mantissa and truth are not
/// inputs to the choice.
#[must_use]
pub fn reason(edge: CausalEdge64, ev: &Evidence<'_>, edit: Edit) -> Measured {
    let op = Operation::of(edge.causal_mask());
    let mut out = Measured::new(op);
    let m = ev.model;
    match op {
        Operation::Association => match m.sourced() {
            Err(e) => out.ungrounded = Some(Ungrounded::Model(e)),
            Ok(()) => {
                let c = m.observational_certification();
                if c != Certification3::Open {
                    out.earned = Some(c);
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
                    out.reaction = Some(if cf.role.is_load_bearing() {
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
/// outcome is rarer in the exposed group. `None` on a tie.
fn direction(outcome: u64, a: u64, b: u64) -> Result<Option<u8>, NotGrounded> {
    if compare(outcome, a, b, true)? {
        Ok(Some(0b000))
    } else if compare(outcome, b, a, true)? {
        Ok(Some(0b100))
    } else {
        Ok(None)
    }
}

/// The declared reading an edge's bits 59..63 are projected through: whose
/// class and rail declare `EpistemicState5`, and where the edge came from.
#[derive(Clone, Copy)]
pub struct Reading<'a> {
    /// The declarations.
    pub declarations: &'a Epi5Declarations,
    /// The edge's class.
    pub class: ClassId,
    /// The rail the class declared the reading on.
    pub rail: RailAxis,
    /// Where the edge's bits came from.
    pub provenance: EdgeProvenance,
}

impl Reading<'_> {
    fn project(&self, edge: CausalEdge64) -> Result<EpistemicState5, Epi5ReadError> {
        self.declarations.project_state5(
            self.class,
            self.rail,
            Epi5Gen::V1,
            edge.epistemic_raw5(),
            self.provenance,
        )
    }
}

fn stamp(edge: CausalEdge64, state: EpistemicState5) -> CausalEdge64 {
    edge.with_epistemic_raw5(state.raw())
}

/// Write a measurement back. Certification rises to what the operator earned
/// when that is strictly stronger; nothing else moves, and nothing is ever
/// lowered.
///
/// # Errors
///
/// The edge's bits 59..63 do not project under `reading`.
pub fn revise(
    edge: CausalEdge64,
    m: &Measured,
    reading: Reading<'_>,
) -> Result<CausalEdge64, Epi5ReadError> {
    let state = reading.project(edge)?;
    let current = state.certification();
    let next = match m.earned {
        Some(c) if c != current && c.entails(current) => c,
        _ => current,
    };
    Ok(stamp(edge, state.with_certification(next)))
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

/// The topology gate. `IndirectUnknown` becomes `IndirectKnown` only when both
/// bindings `a → b` and `b → y` are present in the sealed chain. Certification
/// never moves here.
///
/// # Errors
///
/// The edge's bits 59..63 do not project under `reading`.
pub fn hydrate(
    edge: CausalEdge64,
    (a, b, y): (u8, u8, u8),
    sealed: &[ChainStep],
    reading: Reading<'_>,
) -> Result<(CausalEdge64, Hydration), Epi5ReadError> {
    let has = |s: u8, o: u8| sealed.iter().any(|(_, e)| e.s_idx() == s && e.o_idx() == o);
    let found = match (has(a, b), has(b, y)) {
        (true, true) => Hydration::Hydrated,
        (false, false) => Hydration::ProposedOnly,
        _ => Hydration::Partial,
    };
    let state = reading.project(edge)?;
    if found == Hydration::Hydrated && state.topology() == Topology2::IndirectUnknown {
        Ok((
            stamp(edge, state.with_topology(Topology2::IndirectKnown)),
            found,
        ))
    } else {
        Ok((edge, found))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain_counterfactual::DEFAULT_FREQUENCY_BAR;
    use crate::chain_replay::ComposeTables;
    use causal_edge::edge::InferenceType;
    use causal_edge::layout::EPISTEMIC_MASK;
    use causal_edge::tables::NarsTables;
    use causal_edge::PlasticityState;
    use lance_graph_contract::causal_audit::SupportBasis;
    use lance_graph_contract::certification::ModelBuilder;
    use lance_graph_contract::epistemic_state5::Epi5Reading;

    const A: u8 = 10;
    const B: u8 = 20;
    const Y: u8 = 30;
    const PREDICATE: u8 = 0x91;
    const CLASS: ClassId = 0x0902;
    const RAIL: RailAxis = RailAxis::Taxonomy;

    fn declarations() -> Epi5Declarations {
        let mut d = Epi5Declarations::new();
        d.declare(
            CLASS,
            RAIL,
            Epi5Reading {
                generation: Epi5Gen::V1,
            },
        );
        d
    }

    fn reading(d: &Epi5Declarations) -> Reading<'_> {
        Reading {
            declarations: d,
            class: CLASS,
            rail: RAIL,
            provenance: EdgeProvenance::V2Stamped,
        }
    }

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
    fn query(mask: CausalMask, c: Certification3, t: Topology2) -> CausalEdge64 {
        stamp(
            edge(A, Y, 200, 200, mask),
            EpistemicState5::new(Epi5Gen::V1, t, c),
        )
    }

    fn with_mask(mut e: CausalEdge64, mask: CausalMask) -> CausalEdge64 {
        e.set_causal_mask(mask);
        e
    }

    fn state(e: CausalEdge64) -> (Topology2, Certification3) {
        let s = EpistemicState5::decode(Epi5Gen::V1, e.epistemic_raw5()).expect("valid code");
        (s.topology(), s.certification())
    }

    /// Associated pooled and after the robustness mask, lowered in stratum 1:
    /// observationally `Related`, not `Supports`.
    fn related_population() -> ModelBuilder {
        let mut b = ModelBuilder::new();
        b.cell(0, true, 5, 4);
        b.cell(0, false, 5, 1);
        b.cell(1, true, 3, 1);
        b.cell(1, false, 3, 2);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        b
    }

    fn robust_association() -> ModelBuilder {
        let mut b = ModelBuilder::new();
        b.cell(0, true, 5, 4);
        b.cell(0, false, 5, 1);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        b
    }

    fn add_positive_trial(b: &mut ModelBuilder) {
        b.arm(true, 6, 5);
        b.arm(false, 6, 1);
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
    }

    fn with_trial(mut b: ModelBuilder) -> CertificationModel {
        add_positive_trial(&mut b);
        b.build()
    }

    /// The mediated chain `A → B → Y`, measured: weak first step, strong
    /// second. Cutting `B → Y` drops below the bar; cutting `A → B` raises it.
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

        fn evidence<'a>(&'a self, model: &'a CertificationModel) -> Evidence<'a> {
            Evidence {
                model,
                chain: Some(self.chain()),
            }
        }
    }

    /// Run one operator under `mask` and write it back.
    fn step(
        e: CausalEdge64,
        mask: CausalMask,
        ev: &Evidence<'_>,
        edit: Edit,
        d: &Epi5Declarations,
    ) -> (CausalEdge64, Measured) {
        let e = with_mask(e, mask);
        let m = reason(e, ev, edit);
        (revise(e, &m, reading(d)).expect("declared"), m)
    }

    /// Y = A or B per unit; the counterfactual edit is a mask.
    struct OrGate {
        a: u64,
        b: u64,
    }

    impl OrGate {
        fn but_for(&self, unit: u64, disabled: u64) -> bool {
            let b = self.b & !disabled;
            ((self.a | b) ^ ((self.a & !unit) | b)) & unit != 0
        }
    }

    fn overdetermined_trial() -> (CertificationModel, OrGate) {
        let mut b = ModelBuilder::new();
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

    /// The CUI BONO script: the edge leaves with a different state only where
    /// an operator earned it.
    #[test]
    fn cui_bono_script_moves_the_state_only_where_evidence_earned_it() {
        let d = declarations();
        let rp = Replay::new();
        let model = with_trial(related_population());
        let ev = rp.evidence(&model);
        let start = query(
            CausalMask::SO,
            Certification3::Related,
            Topology2::IndirectUnknown,
        );

        let (e, m) = step(start, CausalMask::SO, &ev, Edit::None, &d);
        assert_eq!(m.earned(), Some(Certification3::Related));
        assert_eq!(
            state(e),
            (Topology2::IndirectUnknown, Certification3::Related)
        );

        let (e, h) = hydrate(e, (A, B, Y), &rp.steps, reading(&d)).unwrap();
        assert_eq!(h, Hydration::Hydrated);
        assert_eq!(
            state(e),
            (Topology2::IndirectKnown, Certification3::Related)
        );

        let (e, m) = step(e, CausalMask::PO, &ev, Edit::None, &d);
        assert_eq!(m.earned(), Some(Certification3::Causes));
        assert_eq!(state(e), (Topology2::IndirectKnown, Certification3::Causes));

        let (after, m) = step(e, CausalMask::SPO, &ev, Edit::CutStep(1), &d);
        assert_eq!(
            m.reaction(),
            Some(Reaction::LoadBearing {
                factual: 185,
                counterfactual: 120
            })
        );
        assert_eq!(after.epistemic_raw5(), e.epistemic_raw5());
        let (_, m) = step(e, CausalMask::SPO, &ev, Edit::CutStep(0), &d);
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
        let d = declarations();
        let rp = Replay::new();
        let mut many = robust_association();
        many.sources(SupportBasis::DirectlyObserved, &(3..40).collect::<Vec<_>>());
        for b in [related_population(), robust_association(), many] {
            let model = with_trial(b);
            assert_eq!(model.certification(), Certification3::Causes);
            let (e, m) = step(
                query(CausalMask::SO, Certification3::Open, Topology2::Direct),
                CausalMask::SO,
                &rp.evidence(&model),
                Edit::None,
                &d,
            );
            assert!(m.earned().is_some(), "anti-vacuity: SO must earn something");
            assert_ne!(m.earned(), Some(Certification3::Causes));
            assert_ne!(state(e).1, Certification3::Causes);
        }
    }

    /// F2. Setting `SPO` certifies nothing: with no chain the operator is not
    /// grounded and the edge is unchanged.
    #[test]
    fn f2_setting_the_projection_certifies_nothing() {
        let d = declarations();
        let model = related_population().build();
        let ev = Evidence {
            model: &model,
            chain: None,
        };
        for c in Certification3::ALL {
            let q = query(CausalMask::SPO, c, Topology2::IndirectUnknown);
            let m = reason(q, &ev, Edit::CutStep(1));
            assert_eq!(m.ungrounded(), Some(Ungrounded::NoChain));
            assert_eq!(revise(q, &m, reading(&d)).unwrap(), q);
        }
    }

    /// F3. A load-bearing counterfactual reaction does not become `Causes`.
    #[test]
    fn f3_a_load_bearing_cut_is_not_causes() {
        let d = declarations();
        let rp = Replay::new();
        let model = related_population().build();
        let ev = rp.evidence(&model);
        for c in [
            Certification3::Open,
            Certification3::Related,
            Certification3::CausalCandidate,
        ] {
            let q = query(CausalMask::SPO, c, Topology2::IndirectKnown);
            let m = reason(q, &ev, Edit::CutStep(1));
            assert!(matches!(m.reaction(), Some(Reaction::LoadBearing { .. })));
            assert_eq!(m.earned(), None);
            assert_eq!(state(revise(q, &m, reading(&d)).unwrap()).1, c);
        }
    }

    /// F4. The counterfactual arm is tagged −6 and never written back.
    #[test]
    fn f4_the_counterfactual_branch_is_not_written_back() {
        let d = declarations();
        let rp = Replay::new();
        let model = related_population().build();
        let q = query(
            CausalMask::SPO,
            Certification3::Related,
            Topology2::IndirectKnown,
        );
        let m = reason(q, &rp.evidence(&model), Edit::CutStep(1));
        let t = m.counterfactual_terminal().expect("replayed");
        assert_eq!(
            t.inference_mantissa(),
            InferenceType::Counterfactual.to_mantissa()
        );
        assert_ne!(t.frequency_u8(), q.frequency_u8(), "anti-vacuity");
        let out = revise(q, &m, reading(&d)).unwrap();
        assert_eq!(out.frequency_u8(), q.frequency_u8());
        assert_eq!(out.confidence_u8(), q.confidence_u8());
        assert_eq!(out.inference_mantissa(), q.inference_mantissa());
    }

    /// F5. Proposing B changes nothing; half a path changes nothing; both
    /// bindings move the topology and never the certification.
    #[test]
    fn f5_a_proposed_beneficiary_is_not_evidence() {
        let d = declarations();
        let rp = Replay::new();
        let q = query(
            CausalMask::SO,
            Certification3::Related,
            Topology2::IndirectUnknown,
        );
        assert_eq!(
            hydrate(q, (A, 99, Y), &rp.steps, reading(&d)).unwrap(),
            (q, Hydration::ProposedOnly)
        );
        assert_eq!(
            hydrate(q, (A, B, Y), &rp.steps[..1], reading(&d)).unwrap(),
            (q, Hydration::Partial)
        );
        let (e, h) = hydrate(q, (A, B, Y), &rp.steps, reading(&d)).unwrap();
        assert_eq!(h, Hydration::Hydrated);
        assert_eq!(
            state(e),
            (Topology2::IndirectKnown, Certification3::Related)
        );
    }

    /// F6. SO and PO disagree in direction; SP reports it and does not demote
    /// an intervention-certified `Causes`. Silence twin: agreeing directions
    /// are not confounded.
    #[test]
    fn f6_simpson_does_not_demote_intervention_causes() {
        let d = declarations();
        let rp = Replay::new();
        let mut b = ModelBuilder::new();
        b.cell(0, true, 8, 2);
        b.cell(0, false, 2, 0);
        b.cell(1, true, 2, 2);
        b.cell(1, false, 8, 7);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        let model = with_trial(b);
        let ev = rp.evidence(&model);
        let (e, _) = step(
            query(CausalMask::PO, Certification3::Open, Topology2::Direct),
            CausalMask::PO,
            &ev,
            Edit::None,
            &d,
        );
        assert_eq!(state(e).1, Certification3::Causes);
        let (after, m) = step(e, CausalMask::SP, &ev, Edit::None, &d);
        assert_eq!(m.confounded(), Some(true));
        assert_eq!(state(after).1, Certification3::Causes);
        let (after, _) = step(after, CausalMask::SO, &ev, Edit::None, &d);
        assert_eq!(state(after).1, Certification3::Causes);

        let agreeing = with_trial(robust_association());
        let (_, m) = step(
            query(CausalMask::SP, Certification3::Open, Topology2::Direct),
            CausalMask::SP,
            &rp.evidence(&agreeing),
            Edit::None,
            &d,
        );
        assert_eq!(m.confounded(), Some(false));
    }

    /// F7. Y = A or B: removing A where B holds shows no reaction, removing A
    /// with B disabled does, and the trial certifies `Causes`. A null reaction
    /// never demotes.
    #[test]
    fn f7_one_at_a_time_removal_is_not_the_definition_of_causality() {
        let d = declarations();
        let (trial, gate) = overdetermined_trial();
        let unit = (gate.a & gate.b).isolate_lowest_one();
        assert!(!gate.but_for(unit, 0));
        assert!(gate.but_for(unit, gate.b));
        assert!(gate.but_for((gate.a & !gate.b).isolate_lowest_one(), 0));

        let rp = Replay::new();
        let ev = rp.evidence(&trial);
        let (e, _) = step(
            query(CausalMask::PO, Certification3::Open, Topology2::Direct),
            CausalMask::PO,
            &ev,
            Edit::None,
            &d,
        );
        assert_eq!(state(e).1, Certification3::Causes);
        let (after, m) = step(e, CausalMask::SPO, &ev, Edit::CutStep(0), &d);
        assert!(matches!(m.reaction(), Some(Reaction::TruthOnly { .. })));
        assert_eq!(state(after).1, Certification3::Causes);
    }

    /// F8. Every operator writes bits 59..63 only.
    #[test]
    fn f8_only_bits_59_to_63_move() {
        let d = declarations();
        let rp = Replay::new();
        let model = with_trial(related_population());
        let ev = rp.evidence(&model);
        let mut moved = 0;
        for v in 0u8..8 {
            let mask = CausalMask::from_bits(v);
            for mantissa in [-6i8, 0, 6] {
                let q = query(mask, Certification3::Open, Topology2::IndirectUnknown)
                    .with_inference_mantissa(mantissa);
                let out = revise(q, &reason(q, &ev, Edit::CutStep(1)), reading(&d)).unwrap();
                assert_eq!((out.0 ^ q.0) & !EPISTEMIC_MASK, 0, "{mask:?} {mantissa}");
                moved += usize::from(out != q);
            }
        }
        assert!(moved > 0, "anti-vacuity");
    }

    /// F9. Same sealed evidence, same edit: same measurement, same bits.
    #[test]
    fn f9_replay_is_deterministic() {
        let d = declarations();
        let rp = Replay::new();
        let model = with_trial(related_population());
        let ev = rp.evidence(&model);
        for v in 0u8..8 {
            let q = query(
                CausalMask::from_bits(v),
                Certification3::Related,
                Topology2::IndirectKnown,
            );
            for edit in [Edit::None, Edit::CutStep(0), Edit::CutStep(1)] {
                let (m1, m2) = (reason(q, &ev, edit), reason(q, &ev, edit));
                assert_eq!(m1, m2);
                assert_eq!(revise(q, &m1, reading(&d)), revise(q, &m2, reading(&d)));
            }
        }
    }

    /// F10. A pass must be executed: intervention receipts without arms are
    /// not grounded, and executed arms without an effect earn nothing.
    #[test]
    fn f10_a_pass_must_be_executed_not_claimed() {
        let d = declarations();
        let rp = Replay::new();
        let mut b = related_population();
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
        let claimed = b.build();
        let (e, m) = step(
            query(CausalMask::PO, Certification3::Related, Topology2::Direct),
            CausalMask::PO,
            &rp.evidence(&claimed),
            Edit::None,
            &d,
        );
        assert_eq!(
            m.ungrounded(),
            Some(Ungrounded::Model(NotGrounded::EmptyArm))
        );
        assert_eq!(state(e).1, Certification3::Related);

        let mut b = related_population();
        b.arm(true, 6, 3);
        b.arm(false, 6, 3);
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
        let null = b.build();
        let (e, m) = step(
            query(CausalMask::PO, Certification3::Related, Topology2::Direct),
            CausalMask::PO,
            &rp.evidence(&null),
            Edit::None,
            &d,
        );
        assert_eq!((m.earned(), m.ungrounded()), (None, None));
        assert_eq!(state(e).1, Certification3::Related);
    }

    /// F11. Bits 59..63 do not select the operation.
    #[test]
    fn f11_bits_59_to_63_are_output_not_a_selector() {
        let rp = Replay::new();
        let model = with_trial(related_population());
        let ev = rp.evidence(&model);
        for v in 0u8..8 {
            let mask = CausalMask::from_bits(v);
            let base = reason(
                query(mask, Certification3::Open, Topology2::Direct),
                &ev,
                Edit::CutStep(1),
            );
            for c in Certification3::ALL {
                for t in Topology2::ALL {
                    assert_eq!(reason(query(mask, c, t), &ev, Edit::CutStep(1)), base);
                }
            }
        }
    }

    /// F12. Projection, mantissa and epistemic state are not aliases.
    #[test]
    fn f12_projection_mantissa_and_state_are_not_aliases() {
        let d = declarations();
        let rp = Replay::new();
        let model = with_trial(related_population());
        let ev = rp.evidence(&model);
        for v in 0u8..8 {
            let mask = CausalMask::from_bits(v);
            for mantissa in [
                InferenceType::Intervention.to_mantissa(),
                InferenceType::Counterfactual.to_mantissa(),
                0,
            ] {
                let q = query(mask, Certification3::Open, Topology2::Direct)
                    .with_inference_mantissa(mantissa);
                let m = reason(q, &ev, Edit::CutStep(1));
                assert_eq!(m.operation(), Operation::of(mask));
                let out = revise(q, &m, reading(&d)).unwrap();
                assert_eq!(out.causal_mask(), mask);
                assert_eq!(out.inference_mantissa(), mantissa);
            }
        }
    }

    /// An undeclared class is refused, not read through a guess.
    #[test]
    fn an_undeclared_reading_is_refused() {
        let d = Epi5Declarations::new();
        let model = related_population().build();
        let q = query(CausalMask::SO, Certification3::Open, Topology2::Direct);
        let m = reason(
            q,
            &Evidence {
                model: &model,
                chain: None,
            },
            Edit::None,
        );
        assert!(revise(q, &m, reading(&d)).is_err());
    }
}
