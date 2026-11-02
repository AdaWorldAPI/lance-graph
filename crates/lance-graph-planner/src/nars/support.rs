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
//! every observation with its source, and every admitted derivation with its
//! law, its premises and the chain it belongs to. From that record it
//! computes, for any statement and any [`Ablation`]:
//!
//! - its proofs: one per surviving observation, one per derivation whose
//!   premises have evidence. A derivation reads its premises' *pooled*
//!   evidence, as NARS derives from revised beliefs; no statement supports
//!   itself on one path. Every proof carries its exact source roots;
//! - the pooled NARS truth, under the selection policy in [`Eval::select`]:
//!   the root-disjoint set of proofs with the most total evidence. Proofs that
//!   share a root are never revised together, so two chains, or 63 thought
//!   slots, replaying one observation count once;
//! - which chains still derive the statement.
//!
//! Two reads compare an ablation with the baseline:
//!
//! - [`DerivationLedger::load`]: does the conclusion still hold? Classifies two
//!   candidate premises as [`Load::FirstCritical`] / [`Load::SecondCritical`],
//!   [`Load::Synergistic`] (each single ablation removes it),
//!   [`Load::Redundant`] (only the joint one does) or [`Load::NotLoadBearing`].
//! - [`DerivationLedger::sensitivity`]: how much does it change? Δfrequency,
//!   Δconfidence, the chains and roots lost, the contradiction before and after.
//!   A conclusion can survive an ablation while its evidence collapses; `load`
//!   says `NotLoadBearing` there and `sensitivity` shows the collapse.
//!
//! # Admission
//!
//! A step is admitted only when its terms fit its law's figure
//! ([`Law::conclusion`]), the figures the planner tactics use:
//!
//! | law | premises | conclusion | tactic |
//! |---|---|---|---|
//! | deduction | `M cop P`, `S cop M` (either order), `cop` ∈ {Inh, Sim} | `S cop P` | CAS down, `close_transitive` |
//! | induction | `S→G`, `S→P` | `G→P` | CAS up |
//! | abduction | `P→M`, `S→M` | `S→P` | RCR |
//! | analogy | `S cop P`, `S↔S′` | `S′ cop P` | TR |
//!
//! [`DerivationLedger::admit`] records a tactic's [`Candidate`] directly, so the
//! tactics stay the authority that proposes steps; the ledger only keeps what
//! was admitted.
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
//! - **Truth arithmetic.** Every truth value comes from [`TruthValue`]'s laws,
//!   with its fixed evidence horizon k = 1. Premise order is the law's:
//!   `premises[0].law(premises[1])`.
//!
//! # Bounds
//!
//! Evaluation is not memoized (the path guard makes a statement's evidence
//! depend on how it was reached), so a deep lattice of alternative derivations
//! costs exponential time. Three bounds refuse instead of returning a partial
//! answer: [`MAX_STEPS`] evaluations, [`MAX_DEPTH`] statements on one path, and
//! [`MAX_POOL`] candidate proofs per statement for the exact selection.

use super::belief::{BeliefArena, CStmt, Copula};
use super::tactics::{Candidate, Tactic};
use super::truth::TruthValue;
use std::collections::BTreeSet;

/// The source of one observation. Exact, unlike [`super::belief::Stamp`],
/// which folds sources modulo 64.
pub type SourceId = u32;

/// Statement evaluations per query before the ledger refuses. Policy pin.
pub const MAX_STEPS: usize = 100_000;

/// Statements on one evaluation path before the ledger refuses. Bounds the
/// recursion depth, which [`MAX_STEPS`] does not. Policy pin.
pub const MAX_DEPTH: usize = 64;

/// Candidate proofs of one statement, after dropping dominated ones, that the
/// exact selection searches. Policy pin.
pub const MAX_POOL: usize = 20;

/// Why the ledger returned no answer. Never a partial one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// More than [`MAX_STEPS`] evaluations.
    Steps,
    /// A path longer than [`MAX_DEPTH`].
    Depth,
    /// More than [`MAX_POOL`] undominated proofs of one statement.
    Pool,
}

/// Whether an explicit deduction step may chain two statements of `cop`.
///
/// Wider than [`Copula::transits`], which governs blanket closure: `Impl` is
/// never closed transitively, but an admitted step may chain it, as the
/// copula's own documentation requires ("causal chaining goes through explicit
/// tactics"). The step earns NARS truth only; it certifies no cause. `Rel`
/// never composes.
fn composes(cop: Copula) -> bool {
    matches!(cop, Copula::Inh | Copula::Sim | Copula::Impl)
}

/// The NARS law a derivation applies, as [`TruthValue`] implements it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Law {
    /// `{M→P, S→M} ⊢ S→P` (premises in either order).
    Deduction,
    /// `{S→G, S→P} ⊢ G→P`.
    Induction,
    /// `{P→M, S→M} ⊢ S→P`.
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

    /// The conclusion this law draws from `premises`, or `None` when their
    /// terms do not fit its figure. A conclusion whose subject equals its
    /// predicate is never drawn.
    #[must_use]
    pub fn conclusion(self, premises: [CStmt; 2]) -> Option<CStmt> {
        let [a, b] = premises;
        let drawn = match self {
            Law::Deduction => {
                if a.cop != b.cop || !composes(a.cop) {
                    return None;
                }
                if a.s == b.p {
                    // {M→P, S→M}: a = M→P, b = S→M.
                    CStmt {
                        s: b.s,
                        cop: a.cop,
                        p: a.p,
                    }
                } else if a.p == b.s {
                    // {S→M, M→P}.
                    CStmt {
                        s: a.s,
                        cop: a.cop,
                        p: b.p,
                    }
                } else {
                    return None;
                }
            }
            Law::Induction => {
                if a.cop != Copula::Inh || b.cop != Copula::Inh || a.s != b.s {
                    return None;
                }
                CStmt {
                    s: a.p,
                    cop: Copula::Inh,
                    p: b.p,
                }
            }
            Law::Abduction => {
                if a.cop != Copula::Inh || b.cop != Copula::Inh || a.p != b.p {
                    return None;
                }
                CStmt {
                    s: b.s,
                    cop: Copula::Inh,
                    p: a.s,
                }
            }
            Law::Analogy => {
                if b.cop != Copula::Sim {
                    return None;
                }
                let sibling = if b.s == a.s {
                    b.p
                } else if b.p == a.s {
                    b.s
                } else {
                    return None;
                };
                CStmt {
                    s: sibling,
                    cop: a.cop,
                    p: a.p,
                }
            }
        };
        (drawn.s != drawn.p).then_some(drawn)
    }

    /// The law a planner tactic applies.
    #[must_use]
    pub fn of(tactic: Tactic) -> Self {
        match tactic {
            Tactic::Rcr => Law::Abduction,
            Tactic::Tr => Law::Analogy,
            Tactic::CasUp => Law::Induction,
            Tactic::CasDown => Law::Deduction,
        }
    }
}

/// A step whose terms do not fit its law: refused, not recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotAFigure {
    pub law: Law,
    pub premises: [CStmt; 2],
    pub conclusion: CStmt,
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

/// One admitted derivation step of a chain.
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
    /// The selected proofs revised together.
    pub truth: TruthValue,
    /// Roots of the selected proofs.
    pub roots: BTreeSet<SourceId>,
    /// Derivations used by the selected proofs.
    pub uses: BTreeSet<DerivationId>,
    /// Number of proofs found.
    pub proofs: usize,
    /// Number of proofs selected (the rest shared a root with the selection).
    pub pooled: usize,
    /// Largest `|f₁ − f₂|` between two proofs: the contradiction kept, not erased.
    pub contradiction: f32,
    /// Chains that still derive the statement on their own.
    pub chains: BTreeSet<ChainId>,
}

/// How two candidate premises bear on a conclusion's *existence*. For how
/// much its truth moves, see [`Sensitivity`].
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

/// A statement's evidence before and after one ablation.
#[derive(Debug, Clone, PartialEq)]
pub struct Sensitivity {
    pub before: Option<Pooled>,
    pub after: Option<Pooled>,
}

impl Sensitivity {
    /// Held in the baseline and still holds.
    #[must_use]
    pub fn survives(&self) -> bool {
        self.before.is_some() && self.after.is_some()
    }

    /// Frequency after minus before; `None` unless it survives.
    #[must_use]
    pub fn d_frequency(&self) -> Option<f32> {
        Some(self.after.as_ref()?.truth.frequency - self.before.as_ref()?.truth.frequency)
    }

    /// Confidence after minus before; `None` unless it survives.
    #[must_use]
    pub fn d_confidence(&self) -> Option<f32> {
        Some(self.after.as_ref()?.truth.confidence - self.before.as_ref()?.truth.confidence)
    }

    /// Chains that derived it before and no longer do.
    #[must_use]
    pub fn chains_lost(&self) -> BTreeSet<ChainId> {
        let empty = BTreeSet::new();
        let before = self.before.as_ref().map_or(&empty, |p| &p.chains);
        let after = self.after.as_ref().map_or(&empty, |p| &p.chains);
        before.difference(after).copied().collect()
    }

    /// Selected roots before that are no longer selected.
    #[must_use]
    pub fn roots_lost(&self) -> BTreeSet<SourceId> {
        let empty = BTreeSet::new();
        let before = self.before.as_ref().map_or(&empty, |p| &p.roots);
        let after = self.after.as_ref().map_or(&empty, |p| &p.roots);
        before.difference(after).copied().collect()
    }
}

/// The append-only record of observations and admitted derivations.
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

    /// Admit a derivation step whose conclusion is the one its law draws from
    /// its premises. The same step admitted in two chains is two records over
    /// the same roots; selection counts it once.
    ///
    /// # Errors
    /// [`NotAFigure`] when the terms do not fit the law; nothing is recorded.
    pub fn derive(
        &mut self,
        chain: ChainId,
        law: Law,
        premises: [CStmt; 2],
        conclusion: CStmt,
    ) -> Result<DerivationId, NotAFigure> {
        if law.conclusion(premises) != Some(conclusion) {
            return Err(NotAFigure {
                law,
                premises,
                conclusion,
            });
        }
        let id = DerivationId(self.derivations.len() as u32);
        self.derivations.push(Derivation {
            chain,
            law,
            premises,
            conclusion,
        });
        Ok(id)
    }

    /// Admit a planner tactic's candidate into `chain`. Its premises are
    /// `arena` indices, as the tactic produced them.
    ///
    /// # Errors
    /// [`NotAFigure`] when the candidate's terms do not fit its law.
    ///
    /// # Panics
    /// When a premise index is outside `arena`, which a candidate from that
    /// arena never has.
    pub fn admit(
        &mut self,
        chain: ChainId,
        arena: &BeliefArena,
        candidate: &Candidate,
    ) -> Result<DerivationId, NotAFigure> {
        let premise = |i: u32| arena.entries()[i as usize].stmt;
        self.derive(
            chain,
            Law::of(candidate.tactic),
            [
                premise(candidate.premises[0]),
                premise(candidate.premises[1]),
            ],
            candidate.stmt,
        )
    }

    /// Every derivation, in admission order.
    #[must_use]
    pub fn derivations(&self) -> &[Derivation] {
        &self.derivations
    }

    /// Number of derivations admitted; a mark for [`Self::grounded_since`].
    #[must_use]
    pub fn mark(&self) -> usize {
        self.derivations.len()
    }

    /// Conclusions of derivations admitted at or after `mark` that have
    /// evidence. A step that changed only a scalar (a confidence, a candidate
    /// score) admitted nothing and returns nothing.
    ///
    /// # Errors
    /// A [`Refusal`] from evaluation.
    pub fn grounded_since(&self, mark: usize) -> Result<Vec<CStmt>, Refusal> {
        let mut out = Vec::new();
        for (i, d) in self.derivations.iter().enumerate().skip(mark) {
            let id = DerivationId(i as u32);
            if !out.contains(&d.conclusion)
                && self
                    .proofs(d.conclusion, &Ablation::none())?
                    .iter()
                    .any(|p| p.uses.contains(&id))
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
    /// A [`Refusal`] from evaluation.
    pub fn proofs(&self, stmt: CStmt, ablation: &Ablation) -> Result<Vec<Proof>, Refusal> {
        let mut eval = Eval::new(self, ablation);
        eval.proofs(stmt)
    }

    /// The pooled evidence for `stmt` under `ablation`, or `None` when it has
    /// none. The selection policy is [`Eval::select`]'s.
    ///
    /// # Errors
    /// A [`Refusal`] from evaluation.
    pub fn pooled(&self, stmt: CStmt, ablation: &Ablation) -> Result<Option<Pooled>, Refusal> {
        let mut eval = Eval::new(self, ablation);
        let Some(mut pooled) = eval.pooled(stmt)? else {
            return Ok(None);
        };
        pooled.chains = self.chains_deriving(stmt, ablation)?;
        Ok(Some(pooled))
    }

    /// The chains that derive `stmt` on their own: with every other chain's
    /// derivations removed (observations stay, they belong to no chain), a
    /// proof of `stmt` through at least one derivation remains. Judged per
    /// chain, never from the pooled proof's combined provenance, which mixes
    /// chains wherever a premise has proofs from more than one.
    fn chains_deriving(
        &self,
        stmt: CStmt,
        ablation: &Ablation,
    ) -> Result<BTreeSet<ChainId>, Refusal> {
        let all: BTreeSet<ChainId> = self.derivations.iter().map(|d| d.chain).collect();
        let mut out = BTreeSet::new();
        for chain in all {
            let mut only = ablation.clone();
            only.derivations.extend(
                self.derivations
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| d.chain != chain)
                    .map(|(i, _)| DerivationId(i as u32)),
            );
            let mut eval = Eval::new(self, &only);
            if eval.proofs(stmt)?.iter().any(|p| !p.uses.is_empty()) {
                out.insert(chain);
            }
        }
        Ok(out)
    }

    /// `stmt`'s evidence in the baseline and under `ablation`.
    ///
    /// # Errors
    /// A [`Refusal`] from either evaluation.
    pub fn sensitivity(&self, stmt: CStmt, ablation: &Ablation) -> Result<Sensitivity, Refusal> {
        Ok(Sensitivity {
            before: self.pooled(stmt, &Ablation::none())?,
            after: self.pooled(stmt, ablation)?,
        })
    }

    /// How `first` and `second` bear on whether `stmt` holds: single
    /// ablations of each, then the joint ablation, against the baseline.
    ///
    /// # Errors
    /// A [`Refusal`] from evaluation.
    pub fn load(&self, stmt: CStmt, first: &Ablation, second: &Ablation) -> Result<Load, Refusal> {
        let holds = |a: &Ablation| -> Result<bool, Refusal> { Ok(self.pooled(stmt, a)?.is_some()) };
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

/// One evaluation under one ablation: the path guard and the bounds.
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

    fn proofs(&mut self, stmt: CStmt) -> Result<Vec<Proof>, Refusal> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            return Err(Refusal::Steps);
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
        if self.path.len() >= MAX_DEPTH {
            return Err(Refusal::Depth);
        }
        self.path.push(stmt);
        let result = self.derived(stmt, &mut out);
        self.path.pop();
        result.map(|()| out)
    }

    fn derived(&mut self, stmt: CStmt, out: &mut Vec<Proof>) -> Result<(), Refusal> {
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

    fn pooled(&mut self, stmt: CStmt) -> Result<Option<Pooled>, Refusal> {
        let proofs = self.proofs(stmt)?;
        if proofs.is_empty() {
            return Ok(None);
        }
        let chosen = Self::select(&proofs)?;
        let mut truth = proofs[chosen[0]].truth;
        let mut roots = proofs[chosen[0]].roots.clone();
        let mut uses = proofs[chosen[0]].uses.clone();
        for &i in &chosen[1..] {
            truth = truth.revise(&proofs[i].truth);
            roots.extend(proofs[i].roots.iter().copied());
            uses.extend(proofs[i].uses.iter().copied());
        }
        let mut contradiction = 0.0f32;
        for (i, a) in proofs.iter().enumerate() {
            for b in &proofs[i + 1..] {
                contradiction = contradiction.max((a.truth.frequency - b.truth.frequency).abs());
            }
        }
        Ok(Some(Pooled {
            truth,
            roots,
            uses,
            proofs: proofs.len(),
            pooled: chosen.len(),
            contradiction,
            chains: BTreeSet::new(),
        }))
    }

    /// The selection policy: the set of proofs, pairwise root-disjoint, with
    /// the largest total evidence weight `Σ c/(1−c)` (what revision sums).
    /// Ties go to the set found first in descending-weight order, which is
    /// independent of admission order.
    ///
    /// A greedy pick by single-proof confidence is not this policy: a proof
    /// over roots {1, 2} can outweigh each proof over {1} or {2} alone and
    /// still carry less evidence than the two together.
    ///
    /// A proof is dropped first when another has a subset of its roots and at
    /// least its weight (it can never be the better choice). If more than
    /// [`MAX_POOL`] remain, the search refuses.
    fn select(proofs: &[Proof]) -> Result<Vec<usize>, Refusal> {
        let weight = |p: &Proof| p.truth.evidence_weight().min(1e12);
        let mut order: Vec<usize> = (0..proofs.len()).collect();
        order.sort_by(|&a, &b| {
            weight(&proofs[b])
                .total_cmp(&weight(&proofs[a]))
                .then_with(|| proofs[a].roots.cmp(&proofs[b].roots))
                .then_with(|| {
                    proofs[a]
                        .truth
                        .frequency
                        .total_cmp(&proofs[b].truth.frequency)
                })
        });
        let mut kept: Vec<usize> = Vec::new();
        for &i in &order {
            let dominated = kept.iter().any(|&k| {
                proofs[k].roots.is_subset(&proofs[i].roots)
                    && weight(&proofs[k]) >= weight(&proofs[i])
            });
            if !dominated {
                kept.push(i);
            }
        }
        if kept.len() > MAX_POOL {
            return Err(Refusal::Pool);
        }
        let mut best: (f32, Vec<usize>) = (f32::NEG_INFINITY, Vec::new());
        let mut current = Vec::new();
        Self::search(proofs, &kept, 0, &mut current, 0.0, &mut best, &weight);
        Ok(best.1)
    }

    fn search(
        proofs: &[Proof],
        kept: &[usize],
        from: usize,
        current: &mut Vec<usize>,
        total: f32,
        best: &mut (f32, Vec<usize>),
        weight: &dyn Fn(&Proof) -> f32,
    ) {
        if !current.is_empty() && total > best.0 {
            *best = (total, current.clone());
        }
        let rest: f32 = kept[from..].iter().map(|&k| weight(&proofs[k])).sum();
        if total + rest <= best.0 {
            return;
        }
        for (j, &k) in kept.iter().enumerate().skip(from) {
            if current
                .iter()
                .all(|&c| proofs[c].roots.is_disjoint(&proofs[k].roots))
            {
                current.push(k);
                Self::search(
                    proofs,
                    kept,
                    j + 1,
                    current,
                    total + weight(&proofs[k]),
                    best,
                    weight,
                );
                current.pop();
            }
        }
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
        l.derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], inh(1, 3))
            .unwrap();
        l.derive(B, Law::Deduction, [inh(1, 4), inh(4, 3)], inh(1, 3))
            .unwrap();
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

    /// A chain that derives a conclusion on its own is credited even when
    /// one of its premises also has a proof from another chain. A derives
    /// 1→3 through 2 and then 1→5 through 3; B derives 1→3 through 4. Only
    /// A reaches 1→5.
    #[test]
    fn a_chain_is_credited_when_a_premise_is_shared_with_another_chain() {
        let mut l = DerivationLedger::new();
        for (i, (s, p)) in [(1, 2), (2, 3), (1, 4), (4, 3), (3, 5)]
            .into_iter()
            .enumerate()
        {
            l.observe(inh(s, p), t(0.9, 0.9), i as SourceId + 1);
        }
        l.derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], inh(1, 3))
            .unwrap();
        l.derive(A, Law::Deduction, [inh(1, 3), inh(3, 5)], inh(1, 5))
            .unwrap();
        l.derive(B, Law::Deduction, [inh(1, 4), inh(4, 3)], inh(1, 3))
            .unwrap();
        let p = l.pooled(inh(1, 5), &Ablation::none()).unwrap().unwrap();
        assert_eq!(p.chains, BTreeSet::from([A]));
        let both = l.pooled(inh(1, 3), &Ablation::none()).unwrap().unwrap();
        assert_eq!(both.chains, BTreeSet::from([A, B]));
    }

    /// A step is productive only if its own premises are grounded, not
    /// merely because its conclusion already has support.
    #[test]
    fn an_ungrounded_step_is_not_productive_beside_existing_support() {
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 3), t(0.9, 0.9), 1);
        let mark = l.mark();
        l.derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], inh(1, 3))
            .unwrap();
        assert!(l.grounded_since(mark).unwrap().is_empty());
        l.observe(inh(1, 2), t(0.9, 0.9), 2);
        l.observe(inh(2, 3), t(0.9, 0.9), 3);
        assert_eq!(l.grounded_since(mark).unwrap(), vec![inh(1, 3)]);
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
            )
            .unwrap();
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
        only_a
            .derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], c)
            .unwrap();
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

    /// A cycle cannot support itself: 1→2 from {1→3, 3→2}, and 1→3 from
    /// {1→2, 2→3}, with 3→2 and 2→3 observed and nothing else.
    #[test]
    fn a_statement_does_not_support_itself() {
        let mut l = DerivationLedger::new();
        l.observe(inh(3, 2), t(0.9, 0.9), 1);
        l.observe(inh(2, 3), t(0.9, 0.9), 2);
        l.derive(A, Law::Deduction, [inh(1, 3), inh(3, 2)], inh(1, 2))
            .unwrap();
        l.derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], inh(1, 3))
            .unwrap();
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
        l.derive(A, Law::Abduction, [inh(1, 2), inh(9, 2)], inh(9, 1))
            .unwrap();
        assert!(l.grounded_since(mark).unwrap().is_empty());
        l.derive(A, Law::Abduction, [inh(1, 2), inh(3, 2)], inh(3, 1))
            .unwrap();
        assert_eq!(l.grounded_since(mark).unwrap(), vec![inh(3, 1)]);
    }

    /// The step bound refuses rather than returning a partial answer: when
    /// every level is admitted twice (two chains), evaluation costs 2^depth.
    #[test]
    fn evaluation_refuses_past_the_step_bound() {
        let lattice = |depth: u16| {
            let mut l = DerivationLedger::new();
            l.observe(inh(0, 1), t(0.9, 0.9), 0);
            for k in 1..depth {
                l.observe(inh(k, k + 1), t(0.9, 0.9), u32::from(k));
                for chain in [A, B] {
                    l.derive(
                        chain,
                        Law::Deduction,
                        [inh(0, k), inh(k, k + 1)],
                        inh(0, k + 1),
                    )
                    .unwrap();
                }
            }
            l
        };
        assert!(lattice(8)
            .pooled(inh(0, 8), &Ablation::none())
            .unwrap()
            .is_some());
        assert_eq!(
            lattice(24).pooled(inh(0, 24), &Ablation::none()),
            Err(Refusal::Steps)
        );
    }

    /// D: a chain deeper than MAX_DEPTH is refused before the stack is at
    /// risk; one just inside the bound is answered.
    #[test]
    fn evaluation_refuses_past_the_depth_bound() {
        let chain = |len: u16| {
            let mut l = DerivationLedger::new();
            l.observe(inh(0, 1), t(0.9, 0.9), 0);
            for k in 1..len {
                l.observe(inh(k, k + 1), t(0.9, 0.9), u32::from(k));
                l.derive(A, Law::Deduction, [inh(0, k), inh(k, k + 1)], inh(0, k + 1))
                    .unwrap();
            }
            l
        };
        let inside = MAX_DEPTH as u16;
        assert!(chain(inside)
            .pooled(inh(0, inside), &Ablation::none())
            .unwrap()
            .is_some());
        let past = MAX_DEPTH as u16 + 2;
        assert_eq!(
            chain(past).pooled(inh(0, past), &Ablation::none()),
            Err(Refusal::Depth)
        );
    }

    /// L8 / A: a step whose terms do not fit its law is refused and nothing
    /// is recorded; the same premises with the drawn conclusion are admitted.
    #[test]
    fn l8_an_invalid_figure_is_not_admitted() {
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 2), t(0.9, 0.9), 1);
        l.observe(inh(3, 2), t(0.9, 0.9), 2);
        // Deduction needs a shared middle: 1→2 and 3→2 have none.
        assert!(l
            .derive(A, Law::Deduction, [inh(1, 2), inh(3, 2)], inh(1, 3))
            .is_err());
        // Abduction draws 3→1 from them, not 1→3.
        assert!(l
            .derive(A, Law::Abduction, [inh(1, 2), inh(3, 2)], inh(1, 3))
            .is_err());
        // Rel never composes.
        let rel = |s, p| CStmt {
            s,
            cop: Copula::Rel(7),
            p,
        };
        assert!(l
            .derive(A, Law::Deduction, [rel(1, 2), rel(2, 3)], rel(1, 3))
            .is_err());
        // An explicit implication step chains, though closure never would.
        let imp = |s, p| CStmt {
            s,
            cop: Copula::Impl,
            p,
        };
        let mut chained = DerivationLedger::new();
        assert!(chained
            .derive(A, Law::Deduction, [imp(1, 2), imp(2, 3)], imp(1, 3))
            .is_ok());
        assert!(l.derivations().is_empty());
        assert!(l.grounded_since(0).unwrap().is_empty());
        assert!(l
            .derive(A, Law::Abduction, [inh(1, 2), inh(3, 2)], inh(3, 1))
            .is_ok());
        assert_eq!(l.grounded_since(0).unwrap(), vec![inh(3, 1)]);
    }

    /// Every law draws what its planner tactic draws, for the tactics'
    /// own premise order.
    #[test]
    fn figures_match_the_planner_tactics() {
        let sim = |s, p| CStmt {
            s,
            cop: Copula::Sim,
            p,
        };
        // CAS down: {G→P, S→G} ⊢ S→P.
        assert_eq!(
            Law::Deduction.conclusion([inh(5, 9), inh(1, 5)]),
            Some(inh(1, 9))
        );
        // close_transitive: {A→B, B→C} ⊢ A→C.
        assert_eq!(
            Law::Deduction.conclusion([inh(1, 5), inh(5, 9)]),
            Some(inh(1, 9))
        );
        // CAS up: {S→G, S→P} ⊢ G→P.
        assert_eq!(
            Law::Induction.conclusion([inh(1, 5), inh(1, 9)]),
            Some(inh(5, 9))
        );
        // RCR: {P→M, S→M} ⊢ S→P.
        assert_eq!(
            Law::Abduction.conclusion([inh(9, 2), inh(1, 2)]),
            Some(inh(1, 9))
        );
        // TR: {S cop P, S↔S′} ⊢ S′ cop P, either side of the similarity.
        assert_eq!(
            Law::Analogy.conclusion([inh(1, 9), sim(1, 4)]),
            Some(inh(4, 9))
        );
        assert_eq!(
            Law::Analogy.conclusion([inh(1, 9), sim(4, 1)]),
            Some(inh(4, 9))
        );
        // No trivial S→S.
        assert_eq!(Law::Deduction.conclusion([inh(1, 2), inh(2, 1)]), None);
    }

    /// A: candidates from the planner tactics are admitted through `admit`.
    #[test]
    fn tactic_candidates_are_admitted() {
        use crate::nars::belief::{BeliefArena, Stamp};
        use crate::nars::tactics::{cas_abstract, rcr_abduce, Throttle};
        let mut arena = BeliefArena::new();
        arena.observe(inh(1, 2), t(0.9, 0.9), Stamp::source(1));
        arena.observe(inh(3, 2), t(0.9, 0.9), Stamp::source(2));
        arena.observe(inh(1, 5), t(0.9, 0.9), Stamp::source(3));
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 2), t(0.9, 0.9), 1);
        l.observe(inh(3, 2), t(0.9, 0.9), 2);
        l.observe(inh(1, 5), t(0.9, 0.9), 3);
        let mut candidates = rcr_abduce(&arena, &Throttle::permissive()).candidates;
        candidates.extend(cas_abstract(&arena, 1, &Throttle::permissive()).candidates);
        assert!(candidates.len() >= 3, "{candidates:?}");
        for c in &candidates {
            l.admit(A, &arena, c).unwrap();
            let pooled = l.pooled(c.stmt, &Ablation::none()).unwrap().unwrap();
            assert_eq!(pooled.truth, c.truth, "{c:?}");
        }
    }

    /// C: the selection takes the root-disjoint set with the most evidence. A
    /// derived proof over roots {1, 2} has the highest single confidence, but
    /// the two observations over {1} and {2} carry more evidence together.
    #[test]
    fn selection_prefers_more_evidence_over_the_single_strongest_proof() {
        let mut l = DerivationLedger::new();
        l.observe(inh(1, 2), t(1.0, 0.95), 1);
        l.observe(inh(2, 3), t(1.0, 0.95), 2);
        l.derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], inh(1, 3))
            .unwrap();
        l.observe(inh(1, 3), t(0.6, 0.9), 1);
        l.observe(inh(1, 3), t(0.6, 0.9), 2);
        let derived = t(1.0, 0.95).deduction(&t(1.0, 0.95));
        assert!(
            derived.confidence > 0.9,
            "the single strongest proof: {derived:?}"
        );
        let both = t(0.6, 0.9).revise(&t(0.6, 0.9));
        assert!(both.evidence_weight() > derived.evidence_weight());
        let p = l.pooled(inh(1, 3), &Ablation::none()).unwrap().unwrap();
        assert_eq!(p.truth, both);
        assert_eq!(p.pooled, 2);
        assert_eq!(p.proofs, 3);
    }

    /// C, the bound: more undominated proofs than MAX_POOL is refused.
    #[test]
    fn selection_refuses_past_the_pool_bound() {
        let mut l = DerivationLedger::new();
        for s in 0..=MAX_POOL as u32 {
            l.observe(inh(1, 2), t(0.9, 0.9), s);
        }
        assert_eq!(l.pooled(inh(1, 2), &Ablation::none()), Err(Refusal::Pool));
        let mut ok = DerivationLedger::new();
        for s in 0..MAX_POOL as u32 {
            ok.observe(inh(1, 2), t(0.9, 0.9), s);
        }
        assert_eq!(
            ok.pooled(inh(1, 2), &Ablation::none())
                .unwrap()
                .unwrap()
                .pooled,
            MAX_POOL
        );
    }

    /// L6 / B: the conclusion survives removing source 1, so `load` reports
    /// no loss, and its evidence still drops: `sensitivity` shows how much.
    #[test]
    fn l6_a_surviving_conclusion_can_still_lose_evidence() {
        // Chain B rests on weaker evidence than chain A.
        let mut weak = DerivationLedger::new();
        weak.observe(inh(1, 2), t(0.9, 0.9), 1);
        weak.observe(inh(2, 3), t(0.9, 0.9), 2);
        weak.observe(inh(1, 4), t(0.6, 0.5), 3);
        weak.observe(inh(4, 3), t(0.6, 0.5), 4);
        weak.derive(A, Law::Deduction, [inh(1, 2), inh(2, 3)], inh(1, 3))
            .unwrap();
        weak.derive(B, Law::Deduction, [inh(1, 4), inh(4, 3)], inh(1, 3))
            .unwrap();
        let c = inh(1, 3);
        assert_eq!(
            weak.load(c, &Ablation::sources(&[1]), &Ablation::sources(&[99])),
            Ok(Load::NotLoadBearing)
        );
        let s = weak.sensitivity(c, &Ablation::sources(&[1])).unwrap();
        assert!(s.survives());
        let chain_a = t(0.9, 0.9).deduction(&t(0.9, 0.9));
        let chain_b = t(0.6, 0.5).deduction(&t(0.6, 0.5));
        let before = chain_a.revise(&chain_b);
        assert_eq!(s.before.as_ref().unwrap().truth, before);
        assert_eq!(s.after.as_ref().unwrap().truth, chain_b);
        assert!((s.d_frequency().unwrap() - (chain_b.frequency - before.frequency)).abs() < 1e-6);
        assert!(s.d_frequency().unwrap() < -0.3);
        assert!(s.d_confidence().unwrap() < -0.5);
        assert_eq!(s.chains_lost(), BTreeSet::from([A]));
        assert_eq!(s.roots_lost(), BTreeSet::from([1, 2]));
        // Silence twin: ablating an unused source changes nothing.
        let none = weak.sensitivity(c, &Ablation::sources(&[99])).unwrap();
        assert_eq!(none.d_frequency(), Some(0.0));
        assert_eq!(none.d_confidence(), Some(0.0));
        assert!(none.chains_lost().is_empty());
    }
}
