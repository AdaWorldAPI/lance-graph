//! `mediator`: the seam from the derivation ledger to `pearl::hydrate`
//! (D-EPI-COT C3a).
//!
//! The ledger (#1840) can hold a grounded deduction `A → Y` through a
//! mediator `B`. That is a reasoning result, not a source-attested chain.
//! [`hydrate_from_ledger`] hands the step to the existing topology gate,
//! [`crate::pearl::hydrate`], and leaves the decision there:
//!
//! - the ledger supplies the receipt: which derivations go through `B`, their
//!   source roots, and what removing them does to `A → Y`;
//! - the sealed chain decides the topology. `IndirectUnknown` becomes
//!   `IndirectKnown` only when both `A → B` and `B → Y` are bindings in the
//!   sealed chain. A ledger path whose links are not in the chain changes
//!   nothing;
//! - certification never moves here.
//!
//! The adapter refuses when the ledger has no grounded path through `B`, so a
//! hydration it reports always has a ledger receipt behind it. It reads only
//! the subject and object of each chain step, as `pearl::hydrate` does; the
//! step's `u8` tag is not a predicate (the withdrawn #1572 guard).

use std::collections::BTreeSet;

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::epistemic_state5::Epi5ReadError;

use super::belief::CStmt;
use super::support::{
    Ablation, DerivationId, DerivationLedger, Law, Refusal, Sensitivity, SourceId,
};
use crate::chain_replay::ChainStep;
use crate::pearl::{hydrate, Hydration, Reading};

/// What the ledger and the sealed chain said about `A → Y` through `B`.
#[derive(Debug, Clone, PartialEq)]
pub struct MediatorReceipt {
    /// Grounded deductions of `A → Y` through `B`.
    pub derivations: Vec<DerivationId>,
    /// Source roots of the proofs through them.
    pub roots: BTreeSet<SourceId>,
    /// What `pearl::hydrate` found in the sealed chain.
    pub hydration: Hydration,
    /// The edge after `pearl::hydrate`.
    pub edge: CausalEdge64,
    /// `A → Y` with and without the derivations through `B`.
    pub mediated: Sensitivity,
}

/// Why no receipt was produced.
#[derive(Debug, Clone, PartialEq)]
pub enum MediatorRefusal {
    /// A term does not fit the edge's 8-bit index.
    TermRange(u16),
    /// The edge is not the relation `A → Y`.
    EdgeMismatch,
    /// The ledger has no grounded deduction of `A → Y` through `B`.
    NoLedgerPath,
    /// The ledger refused to evaluate.
    Ledger(Refusal),
    /// The edge's bits 59..63 do not project.
    Epi5(Epi5ReadError),
}

fn idx(term: u16) -> Result<u8, MediatorRefusal> {
    u8::try_from(term).map_err(|_| MediatorRefusal::TermRange(term))
}

/// Hand a ledger path `A → B → Y` to `pearl::hydrate` for the edge `A → Y`.
///
/// # Errors
/// A [`MediatorRefusal`].
pub fn hydrate_from_ledger(
    ledger: &DerivationLedger,
    edge: CausalEdge64,
    target: CStmt,
    mediator: u16,
    sealed: &[ChainStep],
    reading: Reading<'_>,
) -> Result<MediatorReceipt, MediatorRefusal> {
    let (a, b, y) = (idx(target.s)?, idx(mediator)?, idx(target.p)?);
    if edge.s_idx() != a || edge.o_idx() != y {
        return Err(MediatorRefusal::EdgeMismatch);
    }

    let first = CStmt {
        s: target.s,
        cop: target.cop,
        p: mediator,
    };
    let second = CStmt {
        s: mediator,
        cop: target.cop,
        p: target.p,
    };
    let through: Vec<DerivationId> = ledger
        .derivations()
        .iter()
        .enumerate()
        .filter(|(_, d)| {
            d.law == Law::Deduction
                && d.conclusion == target
                && (d.premises == [first, second] || d.premises == [second, first])
        })
        .map(|(i, _)| DerivationId(i as u32))
        .collect();

    let proofs = ledger
        .proofs(target, &Ablation::none())
        .map_err(MediatorRefusal::Ledger)?;
    let mut derivations = Vec::new();
    let mut roots = BTreeSet::new();
    for id in through {
        if let Some(p) = proofs.iter().find(|p| p.uses.contains(&id)) {
            derivations.push(id);
            roots.extend(p.roots.iter().copied());
        }
    }
    if derivations.is_empty() {
        return Err(MediatorRefusal::NoLedgerPath);
    }

    let (edge, hydration) =
        hydrate(edge, (a, b, y), sealed, reading).map_err(MediatorRefusal::Epi5)?;
    let mediated = ledger
        .sensitivity(target, &Ablation::derivations(&derivations))
        .map_err(MediatorRefusal::Ledger)?;
    Ok(MediatorReceipt {
        derivations,
        roots,
        hydration,
        edge,
        mediated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nars::belief::Copula;
    use crate::nars::support::ChainId;
    use crate::nars::truth::TruthValue;
    use causal_edge::edge::InferenceType;
    use causal_edge::{CausalMask, PlasticityState};
    use lance_graph_contract::band_reading::EdgeProvenance;
    use lance_graph_contract::class_view::ClassId;
    use lance_graph_contract::epistemic_state5::{
        Certification3, Epi5Declarations, Epi5Gen, Epi5Reading, EpistemicState5, Topology2,
    };
    use lance_graph_contract::rail_geometry::RailAxis;

    const CLASS: ClassId = 0x0902;
    const RAIL: RailAxis = RailAxis::Taxonomy;
    const A: u16 = 10;
    const B: u16 = 20;
    const C: u16 = 25;
    const Y: u16 = 30;

    fn imp(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Impl,
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

    fn reading(d: &Epi5Declarations) -> Reading<'_> {
        Reading {
            declarations: d,
            class: CLASS,
            rail: RAIL,
            provenance: EdgeProvenance::V2Stamped,
        }
    }

    fn edge(s: u16, o: u16, t: Topology2, c: Certification3) -> CausalEdge64 {
        CausalEdge64::pack(
            s as u8,
            0x91,
            o as u8,
            200,
            200,
            CausalMask::SO,
            causal_edge::Orientation::from_code(0b101),
            InferenceType::Deduction,
            PlasticityState::S_HOT,
            0,
        )
        .with_epistemic_raw5(EpistemicState5::new(Epi5Gen::V1, t, c).raw())
    }

    fn state(e: CausalEdge64) -> (Topology2, Certification3) {
        let s = EpistemicState5::decode(Epi5Gen::V1, e.epistemic_raw5()).unwrap();
        (s.topology(), s.certification())
    }

    /// `A → B` from source 1, `B → Y` from source 2, and their deduction.
    fn ledger_through(m: u16) -> DerivationLedger {
        let mut l = DerivationLedger::new();
        l.observe(imp(A, m), TruthValue::new(0.9, 0.9), 1);
        l.observe(imp(m, Y), TruthValue::new(0.9, 0.9), 2);
        l.derive(
            ChainId(1),
            Law::Deduction,
            [imp(m, Y), imp(A, m)],
            imp(A, Y),
        )
        .unwrap();
        l
    }

    /// Sealed steps; the `u8` tag is deliberately not the predicate.
    fn sealed(links: &[(u16, u16)]) -> Vec<ChainStep> {
        links
            .iter()
            .map(|&(s, o)| {
                (
                    0xFF,
                    edge(s, o, Topology2::Direct, Certification3::Associated),
                )
            })
            .collect()
    }

    fn query() -> CausalEdge64 {
        edge(A, Y, Topology2::IndirectUnknown, Certification3::Causes)
    }

    /// Ledger and chain agree: `IndirectUnknown × Causes` (raw5 22) becomes
    /// `IndirectKnown × Causes` (raw5 21), with the ledger's roots, and the
    /// mediated path carries the conclusion.
    #[test]
    fn a_ledger_path_witnessed_by_the_chain_hydrates() {
        let d = decls();
        let l = ledger_through(B);
        let r = hydrate_from_ledger(
            &l,
            query(),
            imp(A, Y),
            B,
            &sealed(&[(A, B), (B, Y)]),
            reading(&d),
        )
        .unwrap();
        assert_eq!(r.hydration, Hydration::Hydrated);
        assert_eq!(
            state(r.edge),
            (Topology2::IndirectKnown, Certification3::Causes)
        );
        assert_eq!(r.edge.epistemic_raw5(), 21);
        assert_eq!(r.roots, [1, 2].into_iter().collect());
        assert!(!r.mediated.survives(), "the only path goes through B");
    }

    /// A ledger path whose links are not both in the sealed chain does not
    /// hydrate: the edge keeps raw5 22.
    #[test]
    fn a_ledger_path_alone_does_not_hydrate() {
        let d = decls();
        let l = ledger_through(B);
        for links in [&[(A, B)][..], &[][..]] {
            let r = hydrate_from_ledger(&l, query(), imp(A, Y), B, &sealed(links), reading(&d))
                .unwrap();
            assert_ne!(r.hydration, Hydration::Hydrated);
            assert_eq!(r.edge, query());
        }
    }

    /// No grounded ledger path through `B`: refused, even when the chain has
    /// both links. Covers no derivation, a derivation through another
    /// mediator, and a derivation whose premise has no evidence.
    #[test]
    fn the_adapter_needs_a_grounded_ledger_path_through_the_mediator() {
        let d = decls();
        let chain = sealed(&[(A, B), (B, Y)]);
        let run = |l: &DerivationLedger| {
            hydrate_from_ledger(l, query(), imp(A, Y), B, &chain, reading(&d))
        };

        assert_eq!(
            run(&DerivationLedger::new()),
            Err(MediatorRefusal::NoLedgerPath)
        );
        assert_eq!(run(&ledger_through(C)), Err(MediatorRefusal::NoLedgerPath));

        let mut ungrounded = DerivationLedger::new();
        ungrounded.observe(imp(B, Y), TruthValue::new(0.9, 0.9), 2);
        ungrounded
            .derive(
                ChainId(1),
                Law::Deduction,
                [imp(B, Y), imp(A, B)],
                imp(A, Y),
            )
            .unwrap();
        assert_eq!(run(&ungrounded), Err(MediatorRefusal::NoLedgerPath));
    }

    /// With an independent direct observation of `A → Y`, removing the
    /// mediated path leaves the conclusion: the path through `B` is not
    /// individually necessary, though it still hydrates.
    #[test]
    fn a_redundant_mediated_path_is_not_reported_as_necessary() {
        let d = decls();
        let mut l = ledger_through(B);
        l.observe(imp(A, Y), TruthValue::new(0.8, 0.9), 3);
        let r = hydrate_from_ledger(
            &l,
            query(),
            imp(A, Y),
            B,
            &sealed(&[(A, B), (B, Y)]),
            reading(&d),
        )
        .unwrap();
        assert_eq!(r.hydration, Hydration::Hydrated);
        assert!(r.mediated.survives());
        assert_eq!(r.mediated.after.unwrap().roots, [3].into_iter().collect());
    }

    #[test]
    fn the_edge_must_be_the_target_relation() {
        let d = decls();
        let l = ledger_through(B);
        let other = edge(A, B, Topology2::IndirectUnknown, Certification3::Causes);
        assert_eq!(
            hydrate_from_ledger(
                &l,
                other,
                imp(A, Y),
                B,
                &sealed(&[(A, B), (B, Y)]),
                reading(&d)
            ),
            Err(MediatorRefusal::EdgeMismatch)
        );
        assert_eq!(
            hydrate_from_ledger(&l, query(), imp(A, Y), 300, &[], reading(&d)),
            Err(MediatorRefusal::TermRange(300))
        );
    }
}
