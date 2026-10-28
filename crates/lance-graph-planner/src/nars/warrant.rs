//! `warrant`: which planner tactic may run on which grounded statements, and
//! what it may conclude (D-EPI-COT C3).
//!
//! A tactic runs only on [`Marker`]s. A marker is a statement whose evidence
//! comes from the [`DerivationLedger`] (truth and source roots) and whose
//! bits 59..63 project under a declared reading to an [`EpistemicState5`].
//! [`fire`] checks every premise against the tactic's [`RecipeRule`], runs the
//! planner's own tactic function over the markers, admits each candidate into
//! the ledger, and returns [`Judgment`]s.
//!
//! # What a judgment may claim
//!
//! - **Truth:** the truth of the ledger proof through the new derivation,
//!   returned with that proof's roots. It equals the tactic's own truth
//!   unless a premise's evidence depends on the conclusion itself: the
//!   ledger's cycle guard drops that evidence from the proof, and the
//!   judgment reports what the proof actually rests on.
//! - **Roots:** the source roots of the proof through the new derivation,
//!   which are the union of the premises' roots. A thought that replays the
//!   same premises adds no root.
//! - **Epistemic state:** always `Unknown × Open`. A term-logic step earns no
//!   certification and fixes no topology. Certification rises only through
//!   `crate::pearl::reason` + `revise` (executed trial arms for `Causes`), and
//!   topology only through `crate::pearl::hydrate`. Two `Causes` premises do
//!   not make their deduction a cause: causation is not transitive in general.
//! - **Breadcrumb:** the [`ThoughtRef`] the step ran in (the 53..58 slot and
//!   its generation). The operation is not written into the result.
//!
//! # The tactic law (policy pins)
//!
//! | recipe | tactic | requires | forbids |
//! |---|---|---|---|
//! | 4 | RCR abduction | `ASSOCIATED` | `INTERMEDIATE_KNOWN` |
//! | 6 | TR analogy | `ASSOCIATED` | — |
//! | 7 | ASC self-critique | `ASSOCIATED` | — |
//! | 8 | CAS induction / deduction | `ASSOCIATED` | — |
//! | 11 | CR revision | — | — |
//!
//! - DECISION: a Datapath tactic composes statements that have at least been
//!   associated from independent sources; an `Open` statement has nothing to
//!   compose.
//! - DECISION: RCR does not abduce a middle term for a relation whose mediator
//!   is already known (the lab rule `HYDRATE_INTERMEDIATE` forbids the same).
//! - SCOPE: [`fire`] runs the Datapath tactics (4, 6, 8). ASC and CR are
//!   revisions; in the ledger they are new observations, pooled by
//!   [`DerivationLedger::pooled`].
//! - BASIS: none measured; these are the smallest rules that make the
//!   eligibility boundary testable.
//! - REVISIT WHEN: a tactic produces a measured wrong conclusion that a
//!   stricter rule would have refused, or refuses a step a fixture needs.

use std::collections::BTreeSet;

use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    fact, Certification3, Epi5Declarations, Epi5Gen, Epi5ReadError, EpistemicState5, RecipeRule,
    Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;

use super::belief::{BeliefArena, CStmt, Stamp};
use super::support::{
    Ablation, ChainId, DerivationId, DerivationLedger, NotAFigure, Refusal, SourceId,
};
use super::tactics::{
    cas_abstract, rcr_abduce, tr_diverge, Throttle, ASC_ID, CAS_ID, CR_ID, RCR_ID, TR_ID,
};
use super::truth::TruthValue;

/// The thought a step ran in: the CE64 bits 53..58 slot (0 = WORLD) and the
/// slot's generation, so a reused slot is not mistaken for the old thought.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThoughtRef {
    /// Slot `0..=63`.
    pub slot: u8,
    /// Generation of the slot.
    pub generation: u32,
}

/// Highest thought slot (six bits).
pub const MAX_SLOT: u8 = 63;

/// The tactic law: a rule per recipe id. Recipes outside the table have no
/// rule and are never eligible.
#[must_use]
pub const fn tactic_rule(recipe: u8) -> Option<RecipeRule> {
    match recipe {
        RCR_ID => Some(RecipeRule::new(fact::ASSOCIATED, fact::INTERMEDIATE_KNOWN)),
        TR_ID | ASC_ID | CAS_ID => Some(RecipeRule::new(fact::ASSOCIATED, 0)),
        CR_ID => Some(RecipeRule::new(0, 0)),
        _ => None,
    }
}

/// Where a statement's bits 59..63 came from and how they are declared.
#[derive(Clone, Copy)]
pub struct Declared<'a> {
    /// The declarations.
    pub declarations: &'a Epi5Declarations,
    /// The statement's class.
    pub class: ClassId,
    /// The rail the class declared the reading on.
    pub rail: RailAxis,
    /// Where the bits came from.
    pub provenance: EdgeProvenance,
    /// The raw bits 59..63.
    pub raw5: u8,
}

/// A statement grounded in the ledger with a declared epistemic state.
/// Built only by [`Marker::ground`].
#[derive(Debug, Clone, PartialEq)]
pub struct Marker {
    stmt: CStmt,
    truth: TruthValue,
    roots: BTreeSet<SourceId>,
    state: EpistemicState5,
    thought: ThoughtRef,
}

/// Why no marker was built.
#[derive(Debug, Clone, PartialEq)]
pub enum MarkerRefusal {
    /// The ledger has no evidence for the statement.
    NoEvidence(CStmt),
    /// The ledger refused to evaluate.
    Ledger(Refusal),
    /// Bits 59..63 do not project under the declaration.
    Epi5(Epi5ReadError),
    /// The slot does not fit in six bits.
    Slot(u8),
}

impl Marker {
    /// Ground `stmt`: truth and roots from the ledger's pooled evidence,
    /// the state from the declared projection.
    ///
    /// # Errors
    /// A [`MarkerRefusal`].
    pub fn ground(
        ledger: &DerivationLedger,
        stmt: CStmt,
        declared: Declared<'_>,
        thought: ThoughtRef,
    ) -> Result<Self, MarkerRefusal> {
        if thought.slot > MAX_SLOT {
            return Err(MarkerRefusal::Slot(thought.slot));
        }
        let state = declared
            .declarations
            .project_state5(
                declared.class,
                declared.rail,
                Epi5Gen::V1,
                declared.raw5,
                declared.provenance,
            )
            .map_err(MarkerRefusal::Epi5)?;
        let pooled = ledger
            .pooled(stmt, &Ablation::none())
            .map_err(MarkerRefusal::Ledger)?
            .ok_or(MarkerRefusal::NoEvidence(stmt))?;
        Ok(Self {
            stmt,
            truth: pooled.truth,
            roots: pooled.roots,
            state,
            thought,
        })
    }

    /// The statement.
    #[must_use]
    pub fn stmt(&self) -> CStmt {
        self.stmt
    }
    /// Its pooled truth when grounded.
    #[must_use]
    pub fn truth(&self) -> TruthValue {
        self.truth
    }
    /// Its source roots when grounded.
    #[must_use]
    pub fn roots(&self) -> &BTreeSet<SourceId> {
        &self.roots
    }
    /// Its declared epistemic state.
    #[must_use]
    pub fn state(&self) -> EpistemicState5 {
        self.state
    }
    /// The thought it was grounded in.
    #[must_use]
    pub fn thought(&self) -> ThoughtRef {
        self.thought
    }
}

/// A Datapath tactic step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// RCR: abduction over every shared predicate (recipe 4).
    Rcr,
    /// TR: analogy from a focus statement (recipe 6).
    Tr(CStmt),
    /// CAS: induction and deduction around a focus subject (recipe 8).
    Cas(u16),
}

impl Step {
    /// The recipe id the step implements.
    #[must_use]
    pub fn recipe(self) -> u8 {
        match self {
            Step::Rcr => RCR_ID,
            Step::Tr(_) => TR_ID,
            Step::Cas(_) => CAS_ID,
        }
    }
}

/// What a step concluded.
#[derive(Debug, Clone, PartialEq)]
pub struct Judgment {
    /// The conclusion.
    pub stmt: CStmt,
    /// The truth of the proof through [`Self::derivation`].
    pub truth: TruthValue,
    /// The derivation admitted into the ledger.
    pub derivation: DerivationId,
    /// Roots of the proof through that derivation.
    pub roots: BTreeSet<SourceId>,
    /// The thought the step ran in.
    pub thought: ThoughtRef,
    /// Always `Unknown × Open`: see the module documentation.
    pub state: EpistemicState5,
}

/// Why a step did not run, or stopped.
#[derive(Debug, Clone, PartialEq)]
pub enum FireRefusal {
    /// No markers.
    NoPremises,
    /// Two markers name the same statement.
    DuplicatePremise(CStmt),
    /// The recipe has no rule.
    NoRule(u8),
    /// A premise's state does not satisfy the rule.
    Ineligible {
        /// The premise.
        stmt: CStmt,
        /// Its state.
        state: EpistemicState5,
        /// The recipe.
        recipe: u8,
    },
    /// A premise's ledger evidence changed after its marker was built.
    Stale(CStmt),
    /// The ledger refused.
    Ledger(Refusal),
    /// The ledger refused a candidate's figure.
    Figure(NotAFigure),
    /// No proof uses the admitted derivation.
    NoProof(CStmt),
}

/// The state every judgment carries.
pub const JUDGMENT_STATE: EpistemicState5 =
    EpistemicState5::new(Epi5Gen::V1, Topology2::Unknown, Certification3::Open);

/// Run `step` over `markers` in `chain`, as thought `thought`.
///
/// Refuses before anything is admitted when a premise is ineligible or stale.
///
/// # Errors
/// A [`FireRefusal`].
pub fn fire(
    ledger: &mut DerivationLedger,
    chain: ChainId,
    thought: ThoughtRef,
    step: Step,
    markers: &[Marker],
) -> Result<Vec<Judgment>, FireRefusal> {
    if markers.is_empty() {
        return Err(FireRefusal::NoPremises);
    }
    let recipe = step.recipe();
    let rule = tactic_rule(recipe).ok_or(FireRefusal::NoRule(recipe))?;
    let mut arena = BeliefArena::new();
    let mut seen: Vec<CStmt> = Vec::new();
    for m in markers {
        if seen.contains(&m.stmt) {
            return Err(FireRefusal::DuplicatePremise(m.stmt));
        }
        seen.push(m.stmt);
        if !rule.admits(m.state) {
            return Err(FireRefusal::Ineligible {
                stmt: m.stmt,
                state: m.state,
                recipe,
            });
        }
        let now = ledger
            .pooled(m.stmt, &Ablation::none())
            .map_err(FireRefusal::Ledger)?;
        if now.as_ref().map(|p| (p.truth, &p.roots)) != Some((m.truth, &m.roots)) {
            return Err(FireRefusal::Stale(m.stmt));
        }
        let stamp = m
            .roots
            .iter()
            .fold(Stamp(0), |s, &r| s.union(Stamp::source(r)));
        arena.observe(m.stmt, m.truth, stamp);
    }

    let throttle = Throttle::permissive();
    let frontier = match step {
        Step::Rcr => rcr_abduce(&arena, &throttle),
        Step::Tr(focus) => tr_diverge(&arena, focus),
        Step::Cas(subject) => cas_abstract(&arena, subject, &throttle),
    };

    let mut out = Vec::with_capacity(frontier.candidates.len());
    for c in &frontier.candidates {
        let derivation = ledger
            .admit(chain, &arena, c)
            .map_err(FireRefusal::Figure)?;
        let proof = ledger
            .proofs(c.stmt, &Ablation::none())
            .map_err(FireRefusal::Ledger)?
            .into_iter()
            .find(|p| p.uses.contains(&derivation))
            .ok_or(FireRefusal::NoProof(c.stmt))?;
        out.push(Judgment {
            stmt: c.stmt,
            truth: proof.truth,
            derivation,
            roots: proof.roots,
            thought,
            state: JUDGMENT_STATE,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nars::belief::Copula;
    use crate::pearl::{reason, revise, Edit, Evidence, Reading};
    use causal_edge::edge::{CausalEdge64, InferenceType};
    use causal_edge::{CausalMask, PlasticityState};
    use lance_graph_contract::causal_audit::SupportBasis;
    use lance_graph_contract::certification::ModelBuilder;
    use lance_graph_contract::epistemic_state5::Epi5Reading;

    const CLASS: ClassId = 0x0902;
    const RAIL: RailAxis = RailAxis::Taxonomy;
    const THOUGHT: ThoughtRef = ThoughtRef {
        slot: 1,
        generation: 1,
    };

    fn inh(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Inh,
            p,
        }
    }

    fn decls() -> Epi5Declarations {
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

    fn raw(t: Topology2, c: Certification3) -> u8 {
        EpistemicState5::new(Epi5Gen::V1, t, c).raw()
    }

    fn declared(d: &Epi5Declarations, raw5: u8) -> Declared<'_> {
        Declared {
            declarations: d,
            class: CLASS,
            rail: RAIL,
            provenance: EdgeProvenance::V2Stamped,
            raw5,
        }
    }

    /// `1 → 9` from source 1 and `2 → 9` from source 2: a shared predicate.
    fn ledger() -> DerivationLedger {
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 9), TruthValue::new(0.9, 0.9), 1);
        l.observe(inh(2, 9), TruthValue::new(0.8, 0.9), 2);
        l
    }

    fn markers(l: &DerivationLedger, d: &Epi5Declarations, raw5: u8) -> Vec<Marker> {
        [inh(1, 9), inh(2, 9)]
            .into_iter()
            .map(|s| Marker::ground(l, s, declared(d, raw5), THOUGHT).expect("grounded"))
            .collect()
    }

    /// The acceptance path: RCR runs on two grounded markers, its candidates
    /// enter the ledger, and each judgment carries the tactic's truth, the
    /// union of the premises' roots, the thought, and `Unknown × Open`.
    #[test]
    fn a_tactic_consumes_grounded_markers_and_keeps_their_roots() {
        let d = decls();
        let mut l = ledger();
        let ms = markers(&l, &d, raw(Topology2::Direct, Certification3::Associated));
        let js = fire(&mut l, ChainId(1), THOUGHT, Step::Rcr, &ms).expect("eligible");

        assert_eq!(js.len(), 2, "1→2 and 2→1");
        assert_eq!(l.derivations().len(), 2);
        let both: BTreeSet<SourceId> = [1, 2].into_iter().collect();
        for j in &js {
            assert_eq!(j.roots, both);
            assert_eq!(j.thought, THOUGHT);
            assert_eq!(j.state, JUDGMENT_STATE);
            let pooled = l.pooled(j.stmt, &Ablation::none()).unwrap().unwrap();
            assert_eq!(pooled.truth, j.truth, "the ledger agrees with the tactic");
        }
        let (rule, obs) = (TruthValue::new(0.8, 0.9), TruthValue::new(0.9, 0.9));
        let j12 = js.iter().find(|j| j.stmt == inh(1, 2)).expect("1→2");
        assert_eq!(j12.truth, rule.abduction(&obs));
    }

    /// Eligibility follows the declared facts only: `Open` refuses, the same
    /// topology at `Associated` runs; for RCR a known mediator refuses where
    /// an unknown one runs, while CAS takes both. A refusal admits nothing.
    #[test]
    fn eligibility_follows_the_declared_facts() {
        let d = decls();
        let cases = [
            (Step::Rcr, Topology2::Direct, Certification3::Open, false),
            (
                Step::Rcr,
                Topology2::Direct,
                Certification3::Associated,
                true,
            ),
            (
                Step::Rcr,
                Topology2::IndirectKnown,
                Certification3::Associated,
                false,
            ),
            (
                Step::Rcr,
                Topology2::IndirectUnknown,
                Certification3::Associated,
                true,
            ),
            (
                Step::Cas(1),
                Topology2::IndirectKnown,
                Certification3::Associated,
                true,
            ),
            (
                Step::Cas(1),
                Topology2::Unknown,
                Certification3::Open,
                false,
            ),
        ];
        for (step, t, c, ok) in cases {
            let mut l = ledger();
            let ms = markers(&l, &d, raw(t, c));
            let r = fire(&mut l, ChainId(1), THOUGHT, step, &ms);
            match (ok, r) {
                (true, Ok(_)) => {}
                (false, Err(FireRefusal::Ineligible { recipe, .. })) => {
                    assert_eq!(recipe, step.recipe());
                    assert!(l.derivations().is_empty(), "a refusal admits nothing");
                }
                (_, other) => panic!("{step:?} on {t:?} × {c:?}: {other:?}"),
            }
        }
    }

    /// No evidence, an undeclared class, untrusted provenance or an
    /// out-of-range slot: no marker.
    #[test]
    fn a_marker_needs_evidence_and_a_declared_reading() {
        let d = decls();
        let l = ledger();
        let ok = raw(Topology2::Direct, Certification3::Associated);
        assert_eq!(
            Marker::ground(&l, inh(3, 9), declared(&d, ok), THOUGHT),
            Err(MarkerRefusal::NoEvidence(inh(3, 9)))
        );
        let mut undeclared = declared(&d, ok);
        undeclared.class = 0x0903;
        assert_eq!(
            Marker::ground(&l, inh(1, 9), undeclared, THOUGHT),
            Err(MarkerRefusal::Epi5(Epi5ReadError::UndeclaredClass(0x0903)))
        );
        let mut legacy = declared(&d, ok);
        legacy.provenance = EdgeProvenance::V1Legacy;
        assert!(matches!(
            Marker::ground(&l, inh(1, 9), legacy, THOUGHT),
            Err(MarkerRefusal::Epi5(Epi5ReadError::UnknownProvenance(_)))
        ));
        let far = ThoughtRef {
            slot: 64,
            generation: 0,
        };
        assert_eq!(
            Marker::ground(&l, inh(1, 9), declared(&d, ok), far),
            Err(MarkerRefusal::Slot(64))
        );
    }

    /// Premises at `Causes` do not make a judgment a cause: every eligible
    /// premise state yields `Unknown × Open`. The only promotion is an
    /// executed trial through `pearl`: observation alone leaves the edge
    /// `Open`, and the trial arms raise it.
    #[test]
    fn a_judgment_is_never_promoted_by_its_premises() {
        let d = decls();
        let rule = tactic_rule(RCR_ID).unwrap();
        for t in Topology2::ALL {
            for c in Certification3::ALL {
                let state = EpistemicState5::new(Epi5Gen::V1, t, c);
                if !rule.admits(state) {
                    continue;
                }
                let mut l = ledger();
                let ms = markers(&l, &d, state.raw());
                for j in fire(&mut l, ChainId(1), THOUGHT, Step::Rcr, &ms).unwrap() {
                    assert_eq!(j.state, JUDGMENT_STATE, "premises at {t:?} × {c:?}");
                }
            }
        }

        let edge = CausalEdge64::pack(
            1,
            0x91,
            2,
            200,
            200,
            CausalMask::PO,
            causal_edge::Orientation::from_code(0b101),
            InferenceType::Deduction,
            PlasticityState::S_HOT,
            0,
        )
        .with_epistemic_raw5(JUDGMENT_STATE.raw());
        let reading = Reading {
            declarations: &d,
            class: CLASS,
            rail: RAIL,
            provenance: EdgeProvenance::V2Stamped,
        };
        let mut b = ModelBuilder::new();
        b.cell(0, true, 5, 4);
        b.cell(0, false, 5, 1);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        let observed = b.build();
        let m = reason(
            edge,
            &Evidence {
                model: &observed,
                chain: None,
            },
            Edit::None,
        );
        let after = revise(&m, reading).unwrap();
        assert_eq!(after.epistemic_raw5(), JUDGMENT_STATE.raw());

        b.arm(true, 6, 5);
        b.arm(false, 6, 1);
        b.sources(SupportBasis::InterventionBacked, &[10, 11]);
        let trial = b.build();
        let m = reason(
            edge,
            &Evidence {
                model: &trial,
                chain: None,
            },
            Edit::None,
        );
        let after = revise(&m, reading).unwrap();
        assert_eq!(
            EpistemicState5::decode(Epi5Gen::V1, after.epistemic_raw5())
                .unwrap()
                .certification(),
            Certification3::Causes
        );
    }

    /// A premise whose pooled evidence runs through the conclusion: `1 → 9`
    /// is observed weakly and also derived through `1 → 2`. RCR then
    /// proposes `1 → 2` from `1 → 9` and `2 → 9`. The admitted proof cannot
    /// use the derivation through `1 → 2` (it would support itself), so the
    /// judgment carries the proof's weaker truth and its roots, not the
    /// snapshot truth.
    #[test]
    fn a_judgment_reports_the_truth_its_roots_support() {
        let d = decls();
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 2), TruthValue::new(0.9, 0.9), 1);
        l.observe(inh(2, 9), TruthValue::new(0.9, 0.9), 2);
        l.observe(inh(1, 9), TruthValue::new(0.1, 0.1), 3);
        l.derive(
            ChainId(0),
            crate::nars::support::Law::Deduction,
            [inh(1, 2), inh(2, 9)],
            inh(1, 9),
        )
        .unwrap();
        let st = raw(Topology2::Direct, Certification3::Associated);
        let ms: Vec<Marker> = [inh(1, 9), inh(2, 9)]
            .into_iter()
            .map(|s| Marker::ground(&l, s, declared(&d, st), THOUGHT).unwrap())
            .collect();
        let js = fire(&mut l, ChainId(1), THOUGHT, Step::Rcr, &ms).unwrap();
        let j = js.iter().find(|j| j.stmt == inh(1, 2)).expect("1→2");
        let snapshot = ms[1].truth().abduction(&ms[0].truth());
        assert!(j.truth.confidence < snapshot.confidence / 10.0);
        let proof = l
            .proofs(inh(1, 2), &Ablation::none())
            .unwrap()
            .into_iter()
            .find(|p| p.uses.contains(&j.derivation))
            .unwrap();
        assert_eq!(j.truth, proof.truth);
        assert_eq!(j.roots, proof.roots);
        assert!(
            !j.roots.contains(&1),
            "the conclusion's own source is not a root"
        );
    }

    /// The same step in 63 thoughts admits 63 derivations, and the
    /// conclusion's pooled truth and roots stay those of one.
    #[test]
    fn replaying_a_step_in_many_thoughts_adds_no_evidence() {
        let d = decls();
        let mut l = ledger();
        let ms = markers(&l, &d, raw(Topology2::Direct, Certification3::Associated));
        let once = fire(&mut l, ChainId(1), THOUGHT, Step::Rcr, &ms).unwrap();
        let target = once[0].stmt;
        let first = l.pooled(target, &Ablation::none()).unwrap().unwrap();
        for slot in 2..=MAX_SLOT {
            let t = ThoughtRef {
                slot,
                generation: 1,
            };
            fire(&mut l, ChainId(u32::from(slot)), t, Step::Rcr, &ms).unwrap();
        }
        assert_eq!(l.derivations().len(), 2 * 63);
        let after = l.pooled(target, &Ablation::none()).unwrap().unwrap();
        assert_eq!(after.truth, first.truth);
        assert_eq!(after.roots, first.roots);
    }

    /// A marker built before its statement's evidence changed is refused,
    /// and nothing is admitted.
    #[test]
    fn a_stale_marker_is_refused() {
        let d = decls();
        let mut l = ledger();
        let ms = markers(&l, &d, raw(Topology2::Direct, Certification3::Associated));
        l.observe(inh(1, 9), TruthValue::new(0.1, 0.9), 3);
        assert_eq!(
            fire(&mut l, ChainId(1), THOUGHT, Step::Rcr, &ms),
            Err(FireRefusal::Stale(inh(1, 9)))
        );
        assert!(l.derivations().is_empty());
    }

    /// CAS on a focus subject admits induction and deduction, each a valid
    /// figure, with the tactic's truth.
    #[test]
    fn cas_deduction_runs_on_grounded_markers() {
        let d = decls();
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 5), TruthValue::new(0.9, 0.9), 1);
        l.observe(inh(5, 7), TruthValue::new(0.8, 0.9), 2);
        let st = raw(Topology2::Direct, Certification3::Associated);
        let ms: Vec<Marker> = [inh(1, 5), inh(5, 7)]
            .into_iter()
            .map(|s| Marker::ground(&l, s, declared(&d, st), THOUGHT).unwrap())
            .collect();
        let js = fire(&mut l, ChainId(1), THOUGHT, Step::Cas(1), &ms).unwrap();
        let ded = js
            .iter()
            .find(|j| j.stmt == inh(1, 7))
            .expect("1→7 by deduction");
        assert_eq!(
            ded.truth,
            TruthValue::new(0.8, 0.9).deduction(&TruthValue::new(0.9, 0.9))
        );
        assert_eq!(ded.roots, [1, 2].into_iter().collect());
    }

    #[test]
    fn a_duplicate_premise_is_refused() {
        let d = decls();
        let mut l = ledger();
        let mut ms = markers(&l, &d, raw(Topology2::Direct, Certification3::Associated));
        ms.push(ms[0].clone());
        assert_eq!(
            fire(&mut l, ChainId(1), THOUGHT, Step::Rcr, &ms),
            Err(FireRefusal::DuplicatePremise(inh(1, 9)))
        );
        assert_eq!(
            fire(&mut l, ChainId(1), THOUGHT, Step::Rcr, &[]),
            Err(FireRefusal::NoPremises)
        );
    }
}
