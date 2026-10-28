//! `scent`: which candidate mediator to investigate next for `A → Y`
//! (D-EPI-COT C3b).
//!
//! An edge at `IndirectUnknown × Causes` says an intervention certified that
//! `A` causes `Y` and that the mechanism is open. [`scent`] ranks candidate
//! mediators `B` by whether one investigation could settle them, reading only
//! the ledger and the sealed chain. It decides nothing:
//!
//! - it never writes an edge, a truth value or a certification;
//! - `pearl::hydrate` alone moves the topology, and only on a sealed chain
//!   that holds both `A → B` and `B → Y`;
//! - a score is a reason to look, never evidence.
//!
//! # The ranking
//!
//! A lead is **testable** when exactly one of its two links is missing from
//! the sealed chain and the other is in the chain *and* grounded in the
//! ledger. One observation of the missing link then decides whether `B`
//! hydrates. A lead with both links missing needs two discoveries, and a
//! lead with an unanchored link has no grounded mechanism behind it.
//!
//! Testable leads come first; within each group, higher uncertainty of the
//! missing link first; then mediator id, for determinism.
//!
//! Uncertainty is the binary entropy of the NARS expectation of a link's
//! pooled evidence, and 1 for a link without evidence. On its own it cannot
//! tell a lead worth testing from noise: every lead with no evidence scores
//! the maximum. The D-CE64-STAUNEN-0 probe measured the same failure for raw
//! surprise (0.649 resolved against 0.964 for round-robin).
//!
//! # Settled is not closed
//!
//! A lead is [`LeadStatus::Settled`] when both links are in the chain and both
//! have pooled frequency at least 0.5 (a policy pin). Independent
//! contradicting evidence on a link reopens it as [`LeadStatus::Contested`]
//! without touching the edge: re-examining a mature relation does not demote
//! its certification or topology.
//!
//! Nothing is stored between calls, so repeated calls and replayed
//! observations cannot advance anything (S3, S7).

use causal_edge::edge::CausalEdge64;
use lance_graph_contract::epistemic_state5::Epi5ReadError;

use super::belief::CStmt;
use super::support::{Ablation, DerivationLedger, Refusal};
use super::truth::TruthValue;
use crate::chain_replay::ChainStep;
use crate::pearl::{hydrate, Hydration, Reading};

/// Pooled frequency at or above which a link reads as holding. Policy pin.
pub const HOLDS_FREQUENCY: f32 = 0.5;

/// What one link of `A → B → Y` looks like now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Link {
    /// The link's pooled evidence in the ledger, if any.
    pub truth: Option<TruthValue>,
    /// Whether the sealed chain binds it.
    pub in_chain: bool,
}

impl Link {
    /// Binary entropy of the expectation; 1 without evidence.
    #[must_use]
    pub fn uncertainty(&self) -> f32 {
        self.truth.map_or(1.0, |t| binary_entropy(t.expectation()))
    }

    fn holds(&self) -> bool {
        self.in_chain && self.truth.is_some_and(|t| t.frequency >= HOLDS_FREQUENCY)
    }

    fn anchored(&self) -> bool {
        self.in_chain && self.truth.is_some()
    }
}

fn binary_entropy(p: f32) -> f32 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -(p * p.log2() + (1.0 - p) * (1.0 - p).log2())
}

/// Where a lead stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeadStatus {
    /// Both links bound and holding.
    Settled,
    /// Both links bound, but the evidence on one no longer holds.
    Contested,
    /// One link missing, the other bound and grounded: one observation decides.
    Testable,
    /// Neither link anchored, or both missing.
    Open,
}

/// One candidate mediator.
#[derive(Debug, Clone, PartialEq)]
pub struct Lead {
    /// The candidate `B`.
    pub mediator: u16,
    /// `A → B`.
    pub first: Link,
    /// `B → Y`.
    pub second: Link,
    /// What `pearl::hydrate` reports for this path.
    pub hydration: Hydration,
    /// The lead's status.
    pub status: LeadStatus,
    /// Uncertainty of the link to investigate (the missing or weaker one).
    pub uncertainty: f32,
}

/// Why nothing was ranked.
#[derive(Debug, Clone, PartialEq)]
pub enum ScentRefusal {
    /// A term does not fit the edge's 8-bit index.
    TermRange(u16),
    /// The edge is not `A → Y`.
    EdgeMismatch,
    /// The ledger refused.
    Ledger(Refusal),
    /// The edge's bits 59..63 do not project.
    Epi5(Epi5ReadError),
}

fn idx(term: u16) -> Result<u8, ScentRefusal> {
    u8::try_from(term).map_err(|_| ScentRefusal::TermRange(term))
}

fn rank(status: LeadStatus) -> u8 {
    match status {
        LeadStatus::Testable => 0,
        LeadStatus::Contested => 1,
        LeadStatus::Open => 2,
        LeadStatus::Settled => 3,
    }
}

/// Rank `candidates` as mediators of `target` (`A → Y`, the relation `edge`
/// carries). Reads only; returns leads, most worth investigating first.
///
/// # Errors
/// A [`ScentRefusal`].
pub fn scent(
    ledger: &DerivationLedger,
    edge: CausalEdge64,
    target: CStmt,
    candidates: &[u16],
    sealed: &[ChainStep],
    reading: Reading<'_>,
) -> Result<Vec<Lead>, ScentRefusal> {
    let (a, y) = (idx(target.s)?, idx(target.p)?);
    if edge.s_idx() != a || edge.o_idx() != y {
        return Err(ScentRefusal::EdgeMismatch);
    }
    let bound = |s: u8, o: u8| sealed.iter().any(|(_, e)| e.s_idx() == s && e.o_idx() == o);
    let link = |s: u16, p: u16| -> Result<Link, ScentRefusal> {
        let stmt = CStmt {
            s,
            cop: target.cop,
            p,
        };
        let truth = ledger
            .pooled(stmt, &Ablation::none())
            .map_err(ScentRefusal::Ledger)?
            .map(|p| p.truth);
        Ok(Link {
            truth,
            in_chain: bound(idx(s)?, idx(p)?),
        })
    };

    let mut leads = Vec::with_capacity(candidates.len());
    for &m in candidates {
        let b = idx(m)?;
        let first = link(target.s, m)?;
        let second = link(m, target.p)?;
        let (_, hydration) =
            hydrate(edge, (a, b, y), sealed, reading).map_err(ScentRefusal::Epi5)?;
        let status = match (first.in_chain, second.in_chain) {
            (true, true) if first.holds() && second.holds() => LeadStatus::Settled,
            (true, true) => LeadStatus::Contested,
            (true, false) if first.anchored() => LeadStatus::Testable,
            (false, true) if second.anchored() => LeadStatus::Testable,
            _ => LeadStatus::Open,
        };
        let uncertainty = match (first.in_chain, second.in_chain) {
            (true, false) => second.uncertainty(),
            (false, true) => first.uncertainty(),
            _ => first.uncertainty().max(second.uncertainty()),
        };
        leads.push(Lead {
            mediator: m,
            first,
            second,
            hydration,
            status,
            uncertainty,
        });
    }
    leads.sort_by(|x, y| {
        rank(x.status)
            .cmp(&rank(y.status))
            .then(y.uncertainty.total_cmp(&x.uncertainty))
            .then(x.mediator.cmp(&y.mediator))
    });
    Ok(leads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nars::belief::Copula;
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
    const Y: u16 = 30;
    /// The true mediator: `A → B` is known, `B → Y` is not yet.
    const TRUE_B: u16 = 20;
    /// Noise candidates: nothing known about them. Lower ids, so an
    /// uncertainty-only order with an id tie-break visits them first.
    const NOISE: [u16; 3] = [11, 12, 13];

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

    fn step(s: u16, o: u16) -> ChainStep {
        (
            0xFF,
            edge(s, o, Topology2::Direct, Certification3::Associated),
        )
    }

    fn query() -> CausalEdge64 {
        edge(A, Y, Topology2::IndirectUnknown, Certification3::Causes)
    }

    fn candidates() -> Vec<u16> {
        let mut c = NOISE.to_vec();
        c.push(TRUE_B);
        c
    }

    /// The world the oracle knows: `A → TRUE_B → Y`. The noise candidates
    /// have no links.
    struct World {
        ledger: DerivationLedger,
        sealed: Vec<ChainStep>,
        next_source: u32,
    }

    impl World {
        fn new() -> Self {
            let mut ledger = DerivationLedger::new();
            ledger.observe(imp(A, TRUE_B), TruthValue::new(0.9, 0.9), 1);
            World {
                ledger,
                sealed: vec![step(A, TRUE_B)],
                next_source: 2,
            }
        }

        /// Investigate a lead: observe its first missing link. A link that
        /// exists in the world is recorded; one that does not leaves nothing.
        fn investigate(&mut self, m: u16) {
            let exists = |s: u16, p: u16| (s, p) == (A, TRUE_B) || (s, p) == (TRUE_B, Y);
            for (s, p) in [(A, m), (m, Y)] {
                if self
                    .sealed
                    .iter()
                    .any(|(_, e)| e.s_idx() == s as u8 && e.o_idx() == p as u8)
                {
                    continue;
                }
                if exists(s, p) {
                    self.ledger
                        .observe(imp(s, p), TruthValue::new(0.9, 0.9), self.next_source);
                    self.next_source += 1;
                    self.sealed.push(step(s, p));
                }
                return;
            }
        }

        fn hydrated(&self, d: &Epi5Declarations) -> bool {
            hydrate(
                query(),
                (A as u8, TRUE_B as u8, Y as u8),
                &self.sealed,
                reading(d),
            )
            .unwrap()
            .1 == Hydration::Hydrated
        }
    }

    /// Investigations a policy needs before the true mediator hydrates. A
    /// policy picks from the leads it has not investigated yet.
    fn investigations(d: &Epi5Declarations, order: impl Fn(&[Lead]) -> u16) -> usize {
        let mut w = World::new();
        let mut visited: Vec<u16> = Vec::new();
        for n in 1..=10 {
            let leads: Vec<Lead> = scent(
                &w.ledger,
                query(),
                imp(A, Y),
                &candidates(),
                &w.sealed,
                reading(d),
            )
            .unwrap()
            .into_iter()
            .filter(|l| !visited.contains(&l.mediator))
            .collect();
            let pick = order(&leads);
            visited.push(pick);
            w.investigate(pick);
            if w.hydrated(d) {
                return n;
            }
        }
        usize::MAX
    }

    /// S4. All four candidates have the same uncertainty on the link to
    /// investigate (no evidence, 1.0). Only the true mediator has a grounded,
    /// chain-bound first link, so only it can be decided by one observation.
    /// Scent finds it in one investigation; ordering by uncertainty alone
    /// spends three on noise first.
    #[test]
    fn s4_a_testable_lead_outranks_equally_uncertain_noise() {
        let d = decls();
        let w = World::new();
        let leads = scent(
            &w.ledger,
            query(),
            imp(A, Y),
            &candidates(),
            &w.sealed,
            reading(&d),
        )
        .unwrap();
        assert!(leads.iter().all(|l| (l.uncertainty - 1.0).abs() < 1e-6));
        assert_eq!(leads[0].mediator, TRUE_B);
        assert_eq!(leads[0].status, LeadStatus::Testable);
        assert!(leads[1..].iter().all(|l| l.status == LeadStatus::Open));

        let scented = investigations(&d, |l| l[0].mediator);
        let by_uncertainty = investigations(&d, |l| {
            let mut v = l.to_vec();
            v.sort_by(|x, y| {
                y.uncertainty
                    .total_cmp(&x.uncertainty)
                    .then(x.mediator.cmp(&y.mediator))
            });
            v[0].mediator
        });
        assert_eq!(scented, 1);
        assert_eq!(by_uncertainty, 4);
    }

    /// S1/S8. A settled lead (both links bound and holding) reopens as
    /// contested when independent evidence contradicts a link. The edge is
    /// untouched: scent writes nothing.
    #[test]
    fn s1_a_settled_lead_reopens_on_independent_contradiction() {
        let d = decls();
        let mut l = DerivationLedger::new();
        l.observe(imp(A, TRUE_B), TruthValue::new(0.9, 0.9), 1);
        l.observe(imp(TRUE_B, Y), TruthValue::new(0.9, 0.9), 2);
        let sealed = vec![step(A, TRUE_B), step(TRUE_B, Y)];
        let known = edge(A, Y, Topology2::IndirectKnown, Certification3::Causes);
        let before = scent(&l, known, imp(A, Y), &[TRUE_B], &sealed, reading(&d)).unwrap();
        assert_eq!(before[0].status, LeadStatus::Settled);

        l.observe(imp(TRUE_B, Y), TruthValue::new(0.0, 0.95), 9);
        let after = scent(&l, known, imp(A, Y), &[TRUE_B], &sealed, reading(&d)).unwrap();
        assert_eq!(after[0].status, LeadStatus::Contested);
        assert!(after[0].second.truth.unwrap().frequency < HOLDS_FREQUENCY);
        assert_eq!(
            known.epistemic_raw5(),
            EpistemicState5::new(
                Epi5Gen::V1,
                Topology2::IndirectKnown,
                Certification3::Causes
            )
            .raw()
        );
    }

    /// S3. Replaying one source's observation 63 times leaves the lead's
    /// uncertainty where one observation puts it.
    #[test]
    fn s3_replay_does_not_change_the_scent() {
        let d = decls();
        let sealed = vec![step(A, TRUE_B)];
        let mut once = DerivationLedger::new();
        once.observe(imp(A, TRUE_B), TruthValue::new(0.9, 0.9), 1);
        once.observe(imp(TRUE_B, Y), TruthValue::new(0.7, 0.6), 2);
        let mut many = once.clone();
        for _ in 0..63 {
            many.observe(imp(TRUE_B, Y), TruthValue::new(0.7, 0.6), 2);
        }
        let a = scent(&once, query(), imp(A, Y), &[TRUE_B], &sealed, reading(&d)).unwrap();
        let b = scent(&many, query(), imp(A, Y), &[TRUE_B], &sealed, reading(&d)).unwrap();
        assert_eq!(a, b);
        assert!(a[0].uncertainty < 1.0);
    }

    /// S7. Ranking again changes nothing: no state advances with compute.
    #[test]
    fn s7_repeated_scenting_advances_nothing() {
        let d = decls();
        let w = World::new();
        let first = scent(
            &w.ledger,
            query(),
            imp(A, Y),
            &candidates(),
            &w.sealed,
            reading(&d),
        )
        .unwrap();
        for _ in 0..100 {
            assert_eq!(
                scent(
                    &w.ledger,
                    query(),
                    imp(A, Y),
                    &candidates(),
                    &w.sealed,
                    reading(&d)
                )
                .unwrap(),
                first
            );
        }
    }

    /// An unanchored half-link (bound in the chain, no ledger evidence) is
    /// not testable: there is no grounded mechanism behind it.
    #[test]
    fn a_chain_link_without_evidence_is_not_an_anchor() {
        let d = decls();
        let l = DerivationLedger::new();
        let sealed = vec![step(A, TRUE_B)];
        let leads = scent(&l, query(), imp(A, Y), &[TRUE_B], &sealed, reading(&d)).unwrap();
        assert_eq!(leads[0].status, LeadStatus::Open);
    }

    #[test]
    fn the_edge_must_be_the_target_relation() {
        let d = decls();
        let l = DerivationLedger::new();
        let other = edge(
            A,
            TRUE_B,
            Topology2::IndirectUnknown,
            Certification3::Causes,
        );
        assert_eq!(
            scent(&l, other, imp(A, Y), &[TRUE_B], &[], reading(&d)),
            Err(ScentRefusal::EdgeMismatch)
        );
        assert_eq!(
            scent(&l, query(), imp(A, Y), &[300], &[], reading(&d)),
            Err(ScentRefusal::TermRange(300))
        );
    }
}
