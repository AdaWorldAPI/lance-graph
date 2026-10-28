//! `support` — every derivation of a statement, with its source roots, so a
//! conclusion's load-bearing premises can be found by ablation (D-EPI-COT C1/C2).
//!
//! [`BeliefArena`](super::belief::BeliefArena) keeps one entry per statement.
//! A derived entry holds a single premise pair, the one with the highest
//! expectation, and an empty stamp. That is the right shape for closure, but
//! it cannot answer the question this module exists for: *which observations
//! does a conclusion actually need?* Two chains that derive the same
//! conclusion collapse into one entry, and the observations under a derived
//! entry are not recorded on it.
//!
//! [`DerivationLedger`] is the append-only record beside the arena. It keeps
//! every observation with its source, and every derivation with its law, its
//! premises and the chain it belongs to. From that record it computes, for
//! any statement and any [`Ablation`]:
//!
//! - its proofs: one per surviving observation, one per derivation whose
//!   premises have evidence. A derivation reads its premises' *pooled*
//!   evidence, as NARS derives from revised beliefs; no statement supports
//!   itself on one path. Every proof carries its exact source roots;
//! - the pooled NARS truth: proofs with disjoint roots are revised together,
//!   a proof that shares a root with what is already pooled is skipped. So
//!   two chains, or 63 thought slots, replaying one observation count once;
//! - which chains still derive the statement.
//!
//! [`DerivationLedger::load`] compares the baseline with single and joint
//! ablations and classifies two candidate premises as
//! [`Load::FirstCritical`] / [`Load::SecondCritical`], [`Load::Redundant`] (overdetermined: either single
//! ablation leaves the conclusion, the joint one removes it),
//! [`Load::Synergistic`] (each single ablation removes it) or
//! [`Load::NotLoadBearing`].
//!
//! # What this does not decide
//!
//! - **Outcome or trigger dependence.** Losing a proof is derivation
//!   dependence. Whether an *outcome* changes needs a method/effect model
//!   (HIRO ON/WHEN/DO, C4); whether a *method selection* changes needs the
//!   guard. Neither is computed here.
//! - **Causation.** Removing a source is not Pearl's `do(X)`. Certification
//!   stays with `lance_graph_contract::certification`; nothing here writes
//!   `EpistemicState5`.
//! - **Truth arithmetic.** Every truth value comes from [`TruthValue`]'s laws.
//!   Premise order is the law's: `premises[0].law(premises[1])`.
//! - **Figure.** The ledger evaluates the steps it is given; it does not
//!   check that a step's terms fit its law (a shared middle term for
//!   deduction). The tactic that proposes the step owns that.
//!
//! Evaluation is not memoized (the path guard makes a statement's evidence
//! depend on how it was reached), so a deep lattice of alternative
//! derivations costs exponential time. It is bounded by [`MAX_STEPS`]; past it
//! the ledger refuses ([`Truncated`]) instead of returning a partial answer.

use super::belief::CStmt;
use super::truth::TruthValue;
use std::collections::BTreeSet;

/// The source of one observation. Exact, unlike [`super::belief::Stamp`],
/// which folds sources modulo 64.
pub type SourceId = u32;

/// Statement evaluations per query before the ledger refuses. Policy pin.
pub const MAX_STEPS: usize = 100_000;

/// Evaluation passed [`MAX_STEPS`]; no partial answer is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truncated;

/// The NARS law a derivation applies, as [`TruthValue`] implements it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Law {
    /// `{A→B, B→C} ⊢ A→C`.
    Deduction,
    /// `{A→B, A→C} ⊢ B→C`.
    Induction,
    /// `{A→B, C→B} ⊢ A→C`.
    Abduction,
    /// `{S cop P, S↔S′} ⊢ S′ cop P`.
    Analogy,
}

impl Law {
    /// The conclusion's truth from the premises' truths, in premise order.
    #[must_use]
    pub fn apply(self, first: TruthValue, second: TruthValue) -> TruthValue {
        match self {
            Law::Deduction => first.deduction(&second),
            Law::Induction => first.induction(&second),
            Law::Abduction => first.abduction(&second),
            Law::Analogy => first.analogy(&second),
        }
    }
}

/// A chain of thought: the derivations one line of reasoning made. A chain
/// lives in a thought; the caller owns that mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChainId(pub u32);

/// The index of a derivation in its ledger. Stable: the ledger only appends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DerivationId(pub u32);

/// One observation of a statement from one source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    pub stmt: CStmt,
    pub truth: TruthValue,
    pub source: SourceId,
}

/// One derivation step a chain made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Derivation {
    pub chain: ChainId,
    pub law: Law,
    pub premises: [CStmt; 2],
    pub conclusion: CStmt,
}

/// What to remove before evaluating: observations from these sources, and
/// these derivations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ablation {
    pub sources: BTreeSet<SourceId>,
    pub derivations: BTreeSet<DerivationId>,
}

impl Ablation {
    /// Nothing removed: the baseline.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Remove every observation from `sources`.
    #[must_use]
    pub fn sources(sources: &[SourceId]) -> Self {
        Self {
            sources: sources.iter().copied().collect(),
            derivations: BTreeSet::new(),
        }
    }

    /// Remove these derivations.
    #[must_use]
    pub fn derivations(ids: &[DerivationId]) -> Self {
        Self {
            sources: BTreeSet::new(),
            derivations: ids.iter().copied().collect(),
        }
    }

    /// Both ablations at once.
    #[must_use]
    pub fn joint(&self, other: &Self) -> Self {
        Self {
            sources: self.sources.union(&other.sources).copied().collect(),
            derivations: self
                .derivations
                .union(&other.derivations)
                .copied()
                .collect(),
        }
    }
}

/// One proof of a statement: an observation, or a derivation over its
/// premises' pooled evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct Proof {
    /// Its truth: the observation's, or the law applied to the premises.
    pub truth: TruthValue,
    /// Every source it rests on.
    pub roots: BTreeSet<SourceId>,
    /// Every derivation it uses, transitively.
    pub uses: BTreeSet<DerivationId>,
}

/// A statement's evidence under one ablation.
#[derive(Debug, Clone, PartialEq)]
pub struct Pooled {
    /// Proofs with pairwise-disjoint roots revised together; the rest skipped.
    pub truth: TruthValue,
    /// Roots of the proofs that were pooled.
    pub roots: BTreeSet<SourceId>,
    /// Derivations used by the proofs that were pooled.
    pub uses: BTreeSet<DerivationId>,
    /// Number of proofs found.
    pub proofs: usize,
    /// Number of proofs revised into `truth` (the rest shared a root).
    pub pooled: usize,
    /// Largest `|f₁ − f₂|` between two proofs: the contradiction kept, not erased.
    pub contradiction: f32,
    /// Chains that still derive the statement on their own.
    pub chains: BTreeSet<ChainId>,
}

/// How two candidate premises bear on a conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Load {
    /// Only the first is needed: removing it alone removes the conclusion.
    FirstCritical,
    /// Only the second is needed.
    SecondCritical,
    /// Either alone removes the conclusion: they are needed together.
    Synergistic,
    /// Neither alone removes it, both together do: independent supports.
    Redundant,
    /// The conclusion survives every ablation tried.
    NotLoadBearing,
    /// The conclusion does not hold in the baseline.
    Absent,
}

/// The append-only record of observations and derivations.
#[derive(Debug, Default, Clone)]
pub struct DerivationLedger {
    observations: Vec<Observation>,
    derivations: Vec<Derivation>,
}

impl DerivationLedger {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an observation.
    pub fn observe(&mut self, stmt: CStmt, truth: TruthValue, source: SourceId) {
        self.observations.push(Observation {
            stmt,
            truth,
            source,
        });
    }

    /// Record a derivation step and return its id. The same step recorded in
    /// two chains is two records over the same roots; pooling counts it once.
    pub fn derive(
        &mut self,
        chain: ChainId,
        law: Law,
        premises: [CStmt; 2],
        conclusion: CStmt,
    ) -> DerivationId {
        let id = DerivationId(self.derivations.len() as u32);
        self.derivations.push(Derivation {
            chain,
            law,
            premises,
            conclusion,
        });
        id
    }

    /// Every derivation, in recording order.
    #[must_use]
    pub fn derivations(&self) -> &[Derivation] {
        &self.derivations
    }

    /// Number of derivations recorded; a mark for [`Self::grounded_since`].
    #[must_use]
    pub fn mark(&self) -> usize {
        self.derivations.len()
    }

    /// Conclusions of derivations recorded at or after `mark` that have
    /// evidence. A step that changed only a scalar (a confidence, a candidate
    /// score) recorded nothing and returns nothing.
    ///
    /// # Errors
    /// [`Truncated`] past [`MAX_STEPS`].
    pub fn grounded_since(&self, mark: usize) -> Result<Vec<CStmt>, Truncated> {
        let mut out = Vec::new();
        for d in self.derivations.iter().skip(mark) {
            if !out.contains(&d.conclusion)
                && self.pooled(d.conclusion, &Ablation::none())?.is_some()
            {
                out.push(d.conclusion);
            }
        }
        Ok(out)
    }

    /// The proofs of `stmt` under `ablation`: one per surviving observation,
    /// and one per derivation whose premises both have evidence. A
    /// derivation's truth comes from its premises' *pooled* evidence, so a
    /// premise revised by new evidence moves every conclusion drawn from it.
    ///
    /// # Errors
    /// [`Truncated`] past [`MAX_STEPS`].
    pub fn proofs(&self, stmt: CStmt, ablation: &Ablation) -> Result<Vec<Proof>, Truncated> {
        let mut eval = Eval::new(self, ablation);
        eval.proofs(stmt)
    }

    /// The pooled evidence for `stmt` under `ablation`, or `None` when it has
    /// none.
    ///
    /// Proofs are taken in descending confidence (ties broken by their roots,
    /// so the result does not depend on recording order). Each is revised in
    /// ([`TruthValue::revise`]) only if its roots are disjoint from every root
    /// already pooled; otherwise it is skipped, as the arena's CHOICE does.
    ///
    /// # Errors
    /// [`Truncated`] past [`MAX_STEPS`].
    pub fn pooled(&self, stmt: CStmt, ablation: &Ablation) -> Result<Option<Pooled>, Truncated> {
        let mut eval = Eval::new(self, ablation);
        eval.pooled(stmt)
    }

    /// How `first` and `second` bear on `stmt`: single ablations of each,
    /// then the joint ablation, against the baseline.
    ///
    /// # Errors
    /// [`Truncated`] past [`MAX_STEPS`].
    pub fn load(
        &self,
        stmt: CStmt,
        first: &Ablation,
        second: &Ablation,
    ) -> Result<Load, Truncated> {
        let holds =
            |a: &Ablation| -> Result<bool, Truncated> { Ok(self.pooled(stmt, a)?.is_some()) };
        if !holds(&Ablation::none())? {
            return Ok(Load::Absent);
        }
        let without_first = holds(first)?;
        let without_second = holds(second)?;
        Ok(match (without_first, without_second) {
            (false, false) => Load::Synergistic,
            (false, true) => Load::FirstCritical,
            (true, false) => Load::SecondCritical,
            (true, true) if !holds(&first.joint(second))? => Load::Redundant,
            (true, true) => Load::NotLoadBearing,
        })
    }
}

/// One evaluation under one ablation: the path guard and the step budget.
struct Eval<'a> {
    ledger: &'a DerivationLedger,
    ablation: &'a Ablation,
    path: Vec<CStmt>,
    steps: usize,
}

impl<'a> Eval<'a> {
    fn new(ledger: &'a DerivationLedger, ablation: &'a Ablation) -> Self {
        Self {
            ledger,
            ablation,
            path: Vec::new(),
            steps: 0,
        }
    }

    fn proofs(&mut self, stmt: CStmt) -> Result<Vec<Proof>, Truncated> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            return Err(Truncated);
        }
        let mut out: Vec<Proof> = self
            .ledger
            .observations
            .iter()
            .filter(|o| o.stmt == stmt && !self.ablation.sources.contains(&o.source))
            .map(|o| Proof {
                truth: o.truth,
                roots: BTreeSet::from([o.source]),
                uses: BTreeSet::new(),
            })
            .collect();
        if self.path.contains(&stmt) {
            // A statement may not support itself on one path.
            return Ok(Vec::new());
        }
        self.path.push(stmt);
        let result = self.derived(stmt, &mut out);
        self.path.pop();
        result.map(|()| out)
    }

    fn derived(&mut self, stmt: CStmt, out: &mut Vec<Proof>) -> Result<(), Truncated> {
        for (i, d) in self.ledger.derivations.iter().enumerate() {
            let id = DerivationId(i as u32);
            if d.conclusion != stmt || self.ablation.derivations.contains(&id) {
                continue;
            }
            let Some(first) = self.pooled(d.premises[0])? else {
                continue;
            };
            let Some(second) = self.pooled(d.premises[1])? else {
                continue;
            };
            let mut uses: BTreeSet<DerivationId> =
                first.uses.union(&second.uses).copied().collect();
            uses.insert(id);
            out.push(Proof {
                truth: d.law.apply(first.truth, second.truth),
                roots: first.roots.union(&second.roots).copied().collect(),
                uses,
            });
        }
        Ok(())
    }

    fn pooled(&mut self, stmt: CStmt) -> Result<Option<Pooled>, Truncated> {
        let mut proofs = self.proofs(stmt)?;
        if proofs.is_empty() {
            return Ok(None);
        }
        proofs.sort_by(|a, b| {
            b.truth
                .confidence
                .total_cmp(&a.truth.confidence)
                .then_with(|| a.roots.cmp(&b.roots))
                .then_with(|| a.truth.frequency.total_cmp(&b.truth.frequency))
        });
        let mut truth = proofs[0].truth;
        let mut roots = proofs[0].roots.clone();
        let mut uses = proofs[0].uses.clone();
        let mut pooled = 1;
        for p in &proofs[1..] {
            if roots.is_disjoint(&p.roots) {
                truth = truth.revise(&p.truth);
                roots.extend(p.roots.iter().copied());
                uses.extend(p.uses.iter().copied());
                pooled += 1;
            }
        }
        let mut contradiction = 0.0f32;
        for (i, a) in proofs.iter().enumerate() {
            for b in &proofs[i + 1..] {
                contradiction = contradiction.max((a.truth.frequency - b.truth.frequency).abs());
            }
        }
        let chains = proofs.iter().filter_map(|p| self.single_chain(p)).collect();
        Ok(Some(Pooled {
            truth,
            roots,
            uses,
            proofs: proofs.len(),
            pooled,
            contradiction,
            chains,
        }))
    }

    /// The chain a proof belongs to, if every derivation it uses is from one
    /// chain. A bare observation belongs to no chain.
    fn single_chain(&self, proof: &Proof) -> Option<ChainId> {
        let mut chains = proof
            .uses
            .iter()
            .map(|id| self.ledger.derivations[id.0 as usize].chain);
        let first = chains.next()?;
        chains.all(|c| c == first).then_some(first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nars::belief::Copula;

    fn inh(s: u16, p: u16) -> CStmt {
        CStmt {
            s,
            cop: Copula::Inh,
            p,
        }
    }

    const A: ChainId = ChainId(1);
    const B: ChainId = ChainId(2);
    fn t(f: f32, c: f32) -> TruthValue {
        TruthValue::new(f, c)
    }

    /// Conclusion 1→3 derived twice: chain A through middle term 2, chain B
    /// through middle term 4. `a_src` / `b_src` are the sources of each chain's
    /// two premises.
    fn two_chains(a_src: [SourceId; 2], b_src: [SourceId; 2]) -> DerivationLedger {
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 2), t(0.9, 0.9), a_src[0]);
        l.observe(inh(2, 3), t(0.9, 0.9), a_src[1]);
        l.observe(inh(1, 4), t(0.9, 0.9), b_src[0]);
        l.observe(inh(4, 3), t(0.9, 0.9), b_src[1]);
        l.derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], inh(1, 3));
        l.derive(B, Law::Deduction, [inh(1, 4), inh(4, 3)], inh(1, 3));
        l
    }

    /// F1: two chains that rest on the same root pool to the confidence of one
    /// proof; two chains on disjoint roots pool above it.
    #[test]
    fn f1_shared_root_is_not_double_counted() {
        let shared = two_chains([1, 1], [1, 1]).pooled(inh(1, 3), &Ablation::none());
        let shared = shared.unwrap().unwrap();
        let disjoint = two_chains([1, 2], [3, 4]).pooled(inh(1, 3), &Ablation::none());
        let disjoint = disjoint.unwrap().unwrap();
        let one = t(0.9, 0.9).deduction(&t(0.9, 0.9));
        assert_eq!(shared.proofs, 2);
        assert_eq!(shared.pooled, 1, "a proof sharing a root is skipped");
        assert_eq!(shared.truth, one);
        assert_eq!(disjoint.pooled, 2);
        assert!(
            disjoint.truth.confidence > one.confidence,
            "independent roots pool"
        );
        assert_eq!(disjoint.roots, BTreeSet::from([1, 2, 3, 4]));
    }

    /// F1, the thought-slot form: one derivation replayed in 63 chains over the
    /// same observation is still one proof's worth of evidence.
    #[test]
    fn sixty_three_replays_of_one_root_count_once() {
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 2), t(0.9, 0.9), 7);
        l.observe(inh(2, 3), t(0.9, 0.9), 7);
        for chain in 1..=63 {
            l.derive(
                ChainId(chain),
                Law::Deduction,
                [inh(1, 2), inh(2, 3)],
                inh(1, 3),
            );
        }
        let p = l.pooled(inh(1, 3), &Ablation::none()).unwrap().unwrap();
        assert_eq!(p.proofs, 63);
        assert_eq!(p.pooled, 1);
        assert_eq!(p.truth, t(0.9, 0.9).deduction(&t(0.9, 0.9)));
        assert_eq!(p.chains.len(), 63);
    }

    /// F2: removing chain A's premise invalidates chain A, and the conclusion
    /// still holds through chain B. Proof dependence, not conclusion dependence.
    #[test]
    fn f2_a_lost_proof_is_not_a_lost_conclusion() {
        let l = two_chains([1, 2], [3, 4]);
        let base = l.pooled(inh(1, 3), &Ablation::none()).unwrap().unwrap();
        assert_eq!(base.chains, BTreeSet::from([A, B]));
        let after = l
            .pooled(inh(1, 3), &Ablation::sources(&[1]))
            .unwrap()
            .unwrap();
        assert_eq!(after.chains, BTreeSet::from([B]), "chain A invalidated");
        assert!(
            after.truth.confidence < base.truth.confidence,
            "evidence lost"
        );
        assert_eq!(
            l.load(
                inh(1, 3),
                &Ablation::sources(&[1]),
                &Ablation::sources(&[3])
            ),
            Ok(Load::Redundant)
        );
    }

    /// F3: each single ablation leaves the conclusion; the joint one removes it.
    /// And the silence twin: two premises of one chain are each critical.
    #[test]
    fn f3_joint_ablation_exposes_overdetermination() {
        let l = two_chains([1, 2], [3, 4]);
        let c = inh(1, 3);
        assert!(!l.proofs(c, &Ablation::sources(&[1])).unwrap().is_empty());
        assert!(!l.proofs(c, &Ablation::sources(&[3])).unwrap().is_empty());
        assert!(l.proofs(c, &Ablation::sources(&[1, 3])).unwrap().is_empty());
        assert_eq!(
            l.load(c, &Ablation::sources(&[1]), &Ablation::sources(&[3])),
            Ok(Load::Redundant)
        );
        // Sources 1 and 2 are the two premises of chain A. With chain B gone,
        // each is needed: synergistic.
        let mut only_a = DerivationLedger::new();
        only_a.observe(inh(1, 2), t(0.9, 0.9), 1);
        only_a.observe(inh(2, 3), t(0.9, 0.9), 2);
        only_a.derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], c);
        assert_eq!(
            only_a.load(c, &Ablation::sources(&[1]), &Ablation::sources(&[2])),
            Ok(Load::Synergistic)
        );
        // A source nothing uses bears no load.
        assert_eq!(
            l.load(c, &Ablation::sources(&[99]), &Ablation::sources(&[98])),
            Ok(Load::NotLoadBearing)
        );
    }

    /// Ablating a derivation, not a source: the step is removed, its premises
    /// stay observed.
    #[test]
    fn ablating_a_derivation_keeps_its_premises() {
        let l = two_chains([1, 2], [3, 4]);
        let after = l
            .pooled(inh(1, 3), &Ablation::derivations(&[DerivationId(0)]))
            .unwrap()
            .unwrap();
        assert_eq!(after.chains, BTreeSet::from([B]));
        assert!(!l
            .proofs(inh(1, 2), &Ablation::derivations(&[DerivationId(0)]))
            .unwrap()
            .is_empty());
    }

    /// F10: a new, independently sourced observation against a premise lowers
    /// the conclusion and the contradiction is kept; ablating that source
    /// replays the original exactly.
    #[test]
    fn f10_a_new_contradiction_revises_and_the_original_replays() {
        let mut l = two_chains([1, 2], [3, 4]);
        let before = l.pooled(inh(1, 3), &Ablation::none()).unwrap().unwrap();
        l.observe(inh(2, 3), t(0.05, 0.9), 50);
        // The premise is revised (sources 2 and 50 are disjoint) and keeps
        // the contradiction; chain A derives from the revised premise.
        let premise = l.pooled(inh(2, 3), &Ablation::none()).unwrap().unwrap();
        let revised = t(0.9, 0.9).revise(&t(0.05, 0.9));
        assert_eq!(premise.truth, revised);
        assert!((premise.contradiction - 0.85).abs() < 1e-6);
        let chain_a = t(0.9, 0.9).deduction(&revised);
        let chain_b = t(0.9, 0.9).deduction(&t(0.9, 0.9));
        let after = l.pooled(inh(1, 3), &Ablation::none()).unwrap().unwrap();
        assert_eq!(after.truth, chain_a.revise(&chain_b), "{after:?}");
        assert!(after.truth.frequency < before.truth.frequency);
        assert!((after.contradiction - (chain_b.frequency - chain_a.frequency)).abs() < 1e-6);
        let replay = l
            .pooled(inh(1, 3), &Ablation::sources(&[50]))
            .unwrap()
            .unwrap();
        assert_eq!(replay, before, "the original chain replays");
    }

    /// A cycle cannot support itself: 1→2 derived from 2→1 and back.
    #[test]
    fn a_statement_does_not_support_itself() {
        let mut l = DerivationLedger::new();
        l.derive(A, Law::Deduction, [inh(2, 1), inh(1, 1)], inh(1, 2));
        l.derive(A, Law::Deduction, [inh(1, 2), inh(2, 2)], inh(2, 1));
        assert!(l.proofs(inh(1, 2), &Ablation::none()).unwrap().is_empty());
        assert_eq!(
            l.load(inh(1, 2), &Ablation::none(), &Ablation::none()),
            Ok(Load::Absent)
        );
    }

    /// F9: a step that changes only a scalar records no grounded statement; a
    /// tactic candidate recorded as a derivation does.
    #[test]
    fn f9_only_a_grounded_derivation_is_productive() {
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 2), t(0.9, 0.9), 1);
        l.observe(inh(3, 2), t(0.9, 0.9), 2);
        let mark = l.mark();
        let mut confidence = 0.5f32;
        confidence += 0.1; // what a scalar kernel does
        assert!(confidence > 0.5);
        assert!(l.grounded_since(mark).unwrap().is_empty());
        // An ungrounded derivation (premise never observed) is not productive.
        l.derive(A, Law::Abduction, [inh(1, 2), inh(9, 2)], inh(1, 9));
        assert!(l.grounded_since(mark).unwrap().is_empty());
        l.derive(A, Law::Abduction, [inh(1, 2), inh(3, 2)], inh(1, 3));
        assert_eq!(l.grounded_since(mark).unwrap(), vec![inh(1, 3)]);
    }

    /// The bound refuses rather than returning a partial answer: a lattice
    /// where every level is derived two ways costs 2^depth evaluations.
    #[test]
    fn evaluation_refuses_past_the_bound() {
        // Terms are arbitrary here: the ledger does not check figures.
        let lattice = |depth: u16| {
            let mut l = DerivationLedger::new();
            l.observe(inh(0, 1), t(0.9, 0.9), 0);
            for k in 1..depth {
                for side in 0..2u16 {
                    let helper = inh(k, 1000 + 2 * k + side);
                    l.observe(helper, t(0.9, 0.9), u32::from(2 * k + side));
                    l.derive(A, Law::Deduction, [inh(k - 1, k), helper], inh(k, k + 1));
                }
            }
            l
        };
        assert!(lattice(8)
            .pooled(inh(7, 8), &Ablation::none())
            .unwrap()
            .is_some());
        assert_eq!(
            lattice(24).pooled(inh(23, 24), &Ablation::none()),
            Err(Truncated)
        );
    }
}
