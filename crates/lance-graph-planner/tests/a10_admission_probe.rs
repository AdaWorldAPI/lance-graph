//! A10: a typed epistemic admission probe. TEST-LOCAL; nothing here is a
//! production writer, and nothing here allocates a CE64 bit.
//!
//! Every write of bits 59..63 must name what kind of evidence licenses it.
//! The four kinds reuse the existing owners:
//!
//! | kind | requires | owner reused | may reach |
//! |---|---|---|---|
//! | Given | an identified source and the proof hydration ran | givens layer cap (membership: Associated; resolved relation: Related) | at most Related |
//! | Observed | an identified source and a new independent root | `GadamerRevision` + `BasisView` | no change: eligibility is not acceptance |
//! | Derived | a rule identity, premises, inherited roots only | `GadamerRevision` (inherited roots never count as new) | min of the premises, never above Related |
//! | Measured | an executed operator result | `pearl::reason` + `pearl::revise` | what the operator earned (SO ≤ CausalCandidate; only PO reaches Causes) |
//!
//! Two gaps are explicit, not papered over:
//!
//! - **Rule identity.** No contract type carries the identity of a
//!   derivation rule. [`RuleIdentityGap`] stands in for it here, and a
//!   derivation without one is refused.
//! - **The production admission boundary.** Production code writes bits
//!   59..63 through `pearl::revise` / `pearl::hydrate`, `edge_v3::rehydrate`
//!   (a restore) and `deepnsm_v2::wire` (its own observational rule). None of
//!   them goes through an admission check like this one; this probe does not
//!   change that.
//!
//! Retraction (F4) is closed by `pearl::retract`, the lowering counterpart of
//! `pearl::revise`.
//!
//! The "Derived never above Related" line is the pre-registered F5 bar
//! (`a10-finding.md`): Supports is earned by an observational fold, and a
//! formally valid derivation is neither observation nor causal evidence.

use causal_edge::edge::InferenceType;
use causal_edge::{CausalEdge64, CausalMask, PlasticityState};
use lance_graph_contract::band_reading::EdgeProvenance;
use lance_graph_contract::causal_audit::{EvidenceSourceId, SupportBasis};
use lance_graph_contract::certification::{CertificationModel, ModelBuilder};
use lance_graph_contract::class_view::ClassId;
use lance_graph_contract::epistemic_state5::{
    Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5, Topology2,
};
use lance_graph_contract::rail_geometry::RailAxis;
use lance_graph_contract::revision::{
    BasisView, CodebookId, EncounterEvidence, EvidentialEffect, GadamerRevision, GrammarId,
    HorizonId, InterpretiveHorizon, LanguageId, LensId, QuestionId, RevisionPolicy,
};
use lance_graph_planner::pearl::{self, Edit, Evidence, Reading};

const CLASS: ClassId = 0x0902;
const RAIL: RailAxis = RailAxis::Taxonomy;

/// Stand-in for the missing rule-identity carrier. Its existence here is the
/// gap: no contract type names which rule produced a derivation.
#[derive(Clone, Copy)]
struct RuleIdentityGap(#[allow(dead_code)] u32);

#[derive(Clone, Copy)]
enum GivenProof {
    None,
    Membership,
    ResolvedRelation,
}

enum Admission<'a> {
    Given {
        source: u64,
        proof: GivenProof,
        topology: Topology2,
    },
    Observed {
        roots: u64,
    },
    Derived {
        rule: Option<RuleIdentityGap>,
        premises: &'a [EpistemicState5],
        inherited: u64,
        independent: u64,
    },
    Measured {
        edge: CausalEdge64,
        evidence: &'a Evidence<'a>,
    },
}

#[derive(Debug, PartialEq, Eq)]
enum Refusal {
    Unidentified,
    NoNewRoot,
    MintsRoot,
    RuleIdentityGap,
    NoPremises,
    Ungrounded,
}

fn horizon() -> InterpretiveHorizon<(), u64> {
    InterpretiveHorizon {
        id: HorizonId(0),
        awareness: (),
        question: QuestionId(0),
        language: LanguageId(0),
        grammar: GrammarId(0),
        codebook: CodebookId(0),
        lens: LensId(0),
        projected_claims: 0b1,
        independent_roots: 0,
        inherited_roots: 0,
        unresolved_tension: 0,
        revision_index: 0,
    }
}

fn root_effect(independent: u64, inherited: u64, ancestry: u64) -> EvidentialEffect {
    let enc = EncounterEvidence {
        proposed_claims: 0b1,
        independent_roots: independent,
        inherited_roots: inherited,
        resistance: 0,
        contradictions: 0,
        affected_parts: 0,
    };
    let basis = BasisView {
        ancestry_independent_roots: ancestry,
        ancestry_derived_roots: 0,
        ancestor_claims: 0,
        closes_cycle: false,
    };
    GadamerRevision
        .revise(&horizon(), &enc, &basis)
        .evidential_effect
}

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

fn state(e: CausalEdge64) -> EpistemicState5 {
    EpistemicState5::decode(Epi5Gen::V1, e.epistemic_raw5()).expect("valid code")
}

/// Decide what one admission may write, given the current state and the
/// ancestry's independent roots. Returns the new edge.
fn admit(
    current: CausalEdge64,
    a: &Admission<'_>,
    ancestry_roots: u64,
    d: &Epi5Declarations,
) -> Result<CausalEdge64, Refusal> {
    let now = state(current);
    let write = |s: EpistemicState5| current.with_epistemic_raw5(s.raw());
    match a {
        Admission::Given {
            source,
            proof,
            topology,
        } => {
            if *source == 0 {
                return Err(Refusal::Unidentified);
            }
            let c = match proof {
                GivenProof::None => Certification3::Open,
                GivenProof::Membership => Certification3::Associated,
                GivenProof::ResolvedRelation => Certification3::Related,
            };
            Ok(write(EpistemicState5::new(Epi5Gen::V1, *topology, c)))
        }
        Admission::Observed { roots } => {
            if *roots == 0 {
                return Err(Refusal::Unidentified);
            }
            if root_effect(*roots, 0, ancestry_roots) != EvidentialEffect::IncreaseEligible {
                return Err(Refusal::NoNewRoot);
            }
            // A new root is eligibility, not certification.
            Ok(current)
        }
        Admission::Derived {
            rule,
            premises,
            inherited,
            independent,
        } => {
            if rule.is_none() {
                return Err(Refusal::RuleIdentityGap);
            }
            if *independent != 0 {
                return Err(Refusal::MintsRoot);
            }
            let Some(weakest) = premises
                .iter()
                .map(|p| p.certification())
                .min_by_key(|c| c.code())
            else {
                return Err(Refusal::NoPremises);
            };
            debug_assert_ne!(
                root_effect(0, *inherited, ancestry_roots),
                EvidentialEffect::IncreaseEligible
            );
            let capped = if weakest.code() > Certification3::Related.code() {
                Certification3::Related
            } else {
                weakest
            };
            Ok(write(now.with_certification(capped)))
        }
        Admission::Measured { edge, evidence } => {
            let m = pearl::reason(*edge, evidence, Edit::None);
            if m.ungrounded().is_some() {
                return Err(Refusal::Ungrounded);
            }
            let reading = Reading {
                declarations: d,
                class: CLASS,
                rail: RAIL,
                provenance: EdgeProvenance::V2Stamped,
            };
            pearl::revise(&m, reading).map_err(|_| Refusal::Ungrounded)
        }
    }
}

fn edge(mask: CausalMask, s: EpistemicState5) -> CausalEdge64 {
    CausalEdge64::pack(
        10,
        0x91,
        30,
        200,
        200,
        mask,
        causal_edge::Orientation::from_code(0b101),
        InferenceType::Deduction,
        PlasticityState::S_HOT,
        0,
    )
    .with_epistemic_raw5(s.raw())
}

fn at(t: Topology2, c: Certification3) -> EpistemicState5 {
    EpistemicState5::new(Epi5Gen::V1, t, c)
}

/// Observational population: `Related`, not `Supports` (pearl's own fixture).
fn observational() -> ModelBuilder {
    let mut b = ModelBuilder::new();
    b.cell(0, true, 5, 4);
    b.cell(0, false, 5, 1);
    b.cell(1, true, 3, 1);
    b.cell(1, false, 3, 2);
    b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
    b
}

fn with_trial(mut b: ModelBuilder) -> CertificationModel {
    b.arm(true, 6, 5);
    b.arm(false, 6, 1);
    b.sources(SupportBasis::InterventionBacked, &[10, 11]);
    b.build()
}

fn evidence(model: &CertificationModel) -> Evidence<'_> {
    Evidence { model, chain: None }
}

// ── each kind needs positive evidence ─────────────────────────────────

#[test]
fn given_requires_an_identified_source_and_is_capped_by_its_proof() {
    let d = declarations();
    let e = edge(CausalMask::SO, at(Topology2::Unknown, Certification3::Open));
    let given = |source, proof| Admission::Given {
        source,
        proof,
        topology: Topology2::Direct,
    };
    assert_eq!(
        admit(e, &given(0, GivenProof::ResolvedRelation), 0, &d),
        Err(Refusal::Unidentified)
    );
    for (proof, want) in [
        (GivenProof::None, Certification3::Open),
        (GivenProof::Membership, Certification3::Associated),
        (GivenProof::ResolvedRelation, Certification3::Related),
    ] {
        let out = admit(e, &given(1, proof), 0, &d).expect("identified");
        assert_eq!(state(out).certification(), want);
    }
}

#[test]
fn an_observation_needs_a_new_root_and_earns_no_certification() {
    let d = declarations();
    let s = at(Topology2::Direct, Certification3::Associated);
    let e = edge(CausalMask::SO, s);
    let out = admit(e, &Admission::Observed { roots: 0b10 }, 0, &d).expect("new root");
    assert_eq!(out, e, "eligibility is not acceptance");
    assert_eq!(
        admit(e, &Admission::Observed { roots: 0b10 }, 0b10, &d),
        Err(Refusal::NoNewRoot)
    );
}

#[test]
fn a_derivation_without_rule_identity_is_refused() {
    let d = declarations();
    let e = edge(CausalMask::SO, at(Topology2::Direct, Certification3::Open));
    let premises = [at(Topology2::Direct, Certification3::Related)];
    let derived = |rule| Admission::Derived {
        rule,
        premises: &premises,
        inherited: 0b110,
        independent: 0,
    };
    assert_eq!(
        admit(e, &derived(None), 0, &d),
        Err(Refusal::RuleIdentityGap)
    );
    assert!(admit(e, &derived(Some(RuleIdentityGap(7))), 0, &d).is_ok());
}

#[test]
fn a_derivation_cannot_mint_an_independent_root() {
    let d = declarations();
    let e = edge(CausalMask::SO, at(Topology2::Direct, Certification3::Open));
    let premises = [at(Topology2::Direct, Certification3::Related)];
    let a = Admission::Derived {
        rule: Some(RuleIdentityGap(7)),
        premises: &premises,
        inherited: 0b110,
        independent: 0b1000,
    };
    assert_eq!(admit(e, &a, 0, &d), Err(Refusal::MintsRoot));
}

// ── F5: a valid derivation is not causal evidence ─────────────────────

#[test]
fn f5_a_derivation_from_causal_premises_stays_below_causal() {
    let d = declarations();
    let e = edge(
        CausalMask::SO,
        at(Topology2::IndirectKnown, Certification3::Open),
    );
    let causes = [
        at(Topology2::Direct, Certification3::Causes),
        at(Topology2::Direct, Certification3::Causes),
    ];
    let out = admit(
        e,
        &Admission::Derived {
            rule: Some(RuleIdentityGap(1)),
            premises: &causes,
            inherited: 0b11,
            independent: 0,
        },
        0,
        &d,
    )
    .expect("derived");
    assert_eq!(state(out).certification(), Certification3::Related);
    assert!(state(out).certification().code() < Certification3::Supports.code());

    // Twin: an executed randomized trial on the same relation does reach
    // Causes, so the cap is about the evidence kind, not the relation.
    let model = with_trial(observational());
    let ev = evidence(&model);
    let po = edge(
        CausalMask::PO,
        at(Topology2::IndirectKnown, Certification3::Related),
    );
    let measured = admit(
        po,
        &Admission::Measured {
            edge: po,
            evidence: &ev,
        },
        0,
        &d,
    )
    .expect("trial ran");
    assert_eq!(state(measured).certification(), Certification3::Causes);
}

#[test]
fn a_measured_observation_is_not_an_intervention() {
    let d = declarations();
    let model = with_trial(observational());
    let ev = evidence(&model);
    let so = edge(
        CausalMask::SO,
        at(Topology2::IndirectKnown, Certification3::Open),
    );
    let out = admit(
        so,
        &Admission::Measured {
            edge: so,
            evidence: &ev,
        },
        0,
        &d,
    )
    .expect("observational fold ran");
    let c = state(out).certification();
    assert!(
        c.code() <= Certification3::CausalCandidate.code(),
        "an observational measurement reached {c:?}"
    );
    assert_ne!(c, Certification3::Open, "the fold earned something");
}

// ── F2: elimination (Sudoku) does not raise certification ─────────────

#[test]
fn f2_ruling_out_rivals_does_not_raise_the_survivors_certification() {
    let d = declarations();
    // Eight eliminated rivals, each eliminated by a Related constraint,
    // leave one candidate. The survivor is derived from those constraints.
    let constraints = [at(Topology2::Direct, Certification3::Related); 8];
    let survivor = edge(
        CausalMask::SO,
        at(Topology2::Direct, Certification3::Associated),
    );
    let elim = Admission::Derived {
        rule: Some(RuleIdentityGap(9)),
        premises: &constraints,
        inherited: 0xFF,
        independent: 0,
    };
    let once = admit(survivor, &elim, 0, &d).expect("derived");
    assert_eq!(state(once).certification(), Certification3::Related);
    // The derived cell may not carry a root of its own.
    let minted = Admission::Derived {
        rule: Some(RuleIdentityGap(9)),
        premises: &constraints,
        inherited: 0xFF,
        independent: 0x100,
    };
    assert_eq!(admit(survivor, &minted, 0, &d), Err(Refusal::MintsRoot));
    // Maturation may move; certification does not.
    let matured = once.with_novelty_wisdom(7);
    assert_eq!(matured.epistemic_raw5(), once.epistemic_raw5());
}

/// F2 silent twin: two givens from different sources stay independent; the
/// second is a new root, not a duplicate of the first.
#[test]
fn f2_two_givens_from_different_sources_stay_independent() {
    let d = declarations();
    let e = edge(CausalMask::SO, at(Topology2::Direct, Certification3::Open));
    assert!(admit(e, &Admission::Observed { roots: 0b01 }, 0, &d).is_ok());
    assert!(admit(e, &Admission::Observed { roots: 0b10 }, 0b01, &d).is_ok());
}

// ── F4: retraction ───────────────────────────────────────────────────

#[test]
fn f4_withdrawn_sources_block_new_certification_and_retract_lowers_the_old() {
    let d = declarations();
    let mut model = observational().build();
    let so = edge(
        CausalMask::SO,
        at(Topology2::IndirectKnown, Certification3::Open),
    );
    let before = admit(
        so,
        &Admission::Measured {
            edge: so,
            evidence: &evidence(&model),
        },
        0,
        &d,
    )
    .expect("two observational sources");
    assert_ne!(state(before).certification(), Certification3::Open);

    assert_eq!(model.ledger.withdraw_source(EvidenceSourceId(1)), 1);
    // A fresh measurement no longer grounds: one source left.
    let fresh = edge(
        CausalMask::SO,
        at(Topology2::IndirectKnown, Certification3::Open),
    );
    assert_eq!(
        admit(
            fresh,
            &Admission::Measured {
                edge: fresh,
                evidence: &evidence(&model)
            },
            0,
            &d
        ),
        Err(Refusal::Ungrounded)
    );
    // The code written before the withdrawal no longer stands: re-measuring
    // with the edge's own operator and `pearl::retract` lowers it to Open.
    let ev = evidence(&model);
    let m = pearl::reason(before, &ev, Edit::None);
    let reading = Reading {
        declarations: &d,
        class: CLASS,
        rail: RAIL,
        provenance: EdgeProvenance::V2Stamped,
    };
    let after = pearl::retract(&m, reading).expect("declared");
    assert_eq!(state(after).certification(), Certification3::Open);
    assert_eq!(state(after).topology(), state(before).topology());
}

/// F4 silent twin: withdrawing a source the ledger never held changes
/// nothing; the same measurement earns the same code.
#[test]
fn f4_withdrawing_an_unrelated_source_changes_nothing() {
    let d = declarations();
    let mut model = observational().build();
    let so = edge(
        CausalMask::SO,
        at(Topology2::IndirectKnown, Certification3::Open),
    );
    let before = admit(
        so,
        &Admission::Measured {
            edge: so,
            evidence: &evidence(&model),
        },
        0,
        &d,
    );
    assert_eq!(model.ledger.withdraw_source(EvidenceSourceId(99)), 0);
    let after = admit(
        so,
        &Admission::Measured {
            edge: so,
            evidence: &evidence(&model),
        },
        0,
        &d,
    );
    assert_eq!(before, after);
    assert!(after.is_ok());
}
