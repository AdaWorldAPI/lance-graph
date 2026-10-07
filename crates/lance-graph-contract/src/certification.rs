// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `certification` — the obligations behind each `Certification3` code
//! (D-GSO-7a / P7a, #1369; production home since D-PEARL-PROD-0).
//!
//! A [`CertificationModel`] is one sealed population held as unit masks: who
//! was exposed, who had the outcome, the declared strata, the robustness mask,
//! the ordering, and a randomized trial. Each certification is an integer fold
//! over those masks:
//!
//! | code | `Certification3` | obligation |
//! |---|---|---|
//! | 1 | `Associated` | exposure raises the outcome in some declared population, from at least [`MIN_SOURCES`] distinct observational sources |
//! | 2 | `Related` | the association holds in a declared population and after the robustness mask |
//! | 3 | `Supports` | in every declared stratum exposure does not lower the outcome, before and after the mask, and raises it in at least one |
//! | 4 | `CausalCandidate` | `Supports`, and exposure precedes outcome on every co-occurring unit |
//! | 5 | `Causes` | at least [`MIN_SOURCES`] distinct `InterventionBacked` sources, and the executed randomized arms show the treated rate higher |
//!
//! Thresholds are policy pins. A receipt recorded after the model's seal does
//! not count. [`CertificationModel::observational_certification`] cannot return
//! `Causes`; only the executed arms can. The robustness mask can only refute.
//! Removal is not the causal test: under overdetermination the trial still
//! certifies `Causes` where removing one cause changes nothing.
//!
//! # Any population size, one set of folds
//!
//! The folds use only intersection, difference and a population count, so the
//! model is generic over [`PopulationMask`]. `u64` (64 units) is the default,
//! which keeps every existing caller unchanged; `[u64; N]` holds `64 * N`
//! units — `[u64; 1024]` is one 64k-row cycle. The obligations do not change
//! with the carrier: the same fixture certifies identically in every width
//! (pinned by `every_carrier_width_certifies_a_fixture_identically`).
//!
//! The probe that measured these obligations is
//! `cognitive-shader-driver/examples/relational_certification_probe.rs`.

use crate::causal_audit::{EvidenceSourceId, SupportBasis, SupportLedger, SupportReceipt};
use crate::epistemic_state5::Certification3;
use crate::revision::EvidenceMask;
use crate::scheduler::DatasetVersion;

// ── The sealed model and its satisfaction folds ───────────────────────────

/// Why a proposition could not be decided in the model. Not a third truth
/// value: the model lacks what the decision needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotGrounded {
    EmptyArm,
    TooFewSources,
    NoIdentificationDesign,
    NoDeclaredStratum,
}

/// Satisfied (`Ok(true)`), refuted (`Ok(false)`), or not grounded.
pub type Sat = Result<bool, NotGrounded>;

pub const OBSERVATION: &[SupportBasis] =
    &[SupportBasis::DirectlyObserved, SupportBasis::TextAttested];
pub const INTERVENTION: &[SupportBasis] = &[SupportBasis::InterventionBacked];
/// Distinct sources a contract needs. Policy pin.
pub const MIN_SOURCES: usize = 2;

/// A unit mask the certification folds can count: [`EvidenceMask`]'s set
/// algebra plus a population count, a full mask and single units.
///
/// A sub-trait rather than new [`EvidenceMask`] methods, so a mask that only
/// needs the set algebra (candidate masking, revision) is not asked to count.
pub trait PopulationMask: EvidenceMask {
    /// How many units the mask can hold.
    const CAPACITY: u32;
    /// Every unit set.
    fn full() -> Self;
    /// The number of units set.
    fn count(&self) -> u64;
    /// The mask holding only unit `i`, or `None` past [`Self::CAPACITY`].
    fn unit(i: u32) -> Option<Self>;
}

impl PopulationMask for u64 {
    const CAPACITY: u32 = 64;
    fn full() -> Self {
        u64::MAX
    }
    fn count(&self) -> u64 {
        u64::from(self.count_ones())
    }
    fn unit(i: u32) -> Option<Self> {
        1u64.checked_shl(i)
    }
}

impl<const N: usize> PopulationMask for [u64; N] {
    const CAPACITY: u32 = 64 * N as u32;
    fn full() -> Self {
        [u64::MAX; N]
    }
    fn count(&self) -> u64 {
        self.iter().map(|w| u64::from(w.count_ones())).sum()
    }
    fn unit(i: u32) -> Option<Self> {
        let word = (i / 64) as usize;
        if word >= N {
            return None;
        }
        let mut m = [0; N];
        m[word] = 1u64 << (i % 64);
        Some(m)
    }
}

/// A satisfaction fold over a sealed model.
type Evaluator<M> = fn(&CertificationModel<M>) -> Sat;

/// One sealed model: unit masks over at most [`PopulationMask::CAPACITY`]
/// units of `M`.
#[derive(Debug, Clone)]
pub struct CertificationModel<M: PopulationMask = u64> {
    pub seal: DatasetVersion,
    pub universe: M,
    pub exposed: M,
    pub outcome: M,
    /// The declared comparison partition, fixed before the data are read.
    pub strata: [M; 4],
    /// Units whose exposure precedes their outcome.
    pub ordered: M,
    /// Units the robustness mask keeps.
    pub clean: M,
    /// Randomized identification design.
    pub trial: M,
    pub assigned: M,
    pub ledger: SupportLedger,
    /// Which bases count as this model's evidence for the association
    /// contracts.
    pub bases: &'static [SupportBasis],
}

/// The number of units in `m`.
pub fn count<M: PopulationMask>(m: &M) -> u64 {
    m.count()
}

/// rate(Y | a) compared with rate(Y | b) by cross-multiplication.
pub fn compare<M: PopulationMask>(outcome: &M, a: &M, b: &M, strict: bool) -> Sat {
    let (na, nb) = (a.count(), b.count());
    if na == 0 || nb == 0 {
        return Err(NotGrounded::EmptyArm);
    }
    let (lhs, rhs) = (
        outcome.intersection(a).count() * nb,
        outcome.intersection(b).count() * na,
    );
    Ok(if strict { lhs > rhs } else { lhs >= rhs })
}

/// Distinct sources among `bases`, counting only receipts recorded at or
/// before `seal`: a sealed model is certified from the evidence it held, so a
/// later receipt cannot change what an earlier seal replays to.
pub fn distinct_sources(
    ledger: &SupportLedger,
    bases: &[SupportBasis],
    seal: DatasetVersion,
) -> usize {
    let mut seen: Vec<EvidenceSourceId> = Vec::new();
    for r in ledger
        .receipts()
        .iter()
        .filter(|r| r.at <= seal && bases.contains(&r.basis))
    {
        if !seen.contains(&r.source) {
            seen.push(r.source);
        }
    }
    seen.len()
}

/// Existential fold: satisfied if any grounded case is, refuted if every
/// grounded case is, not grounded if none is.
fn any(results: impl Iterator<Item = Sat>) -> Sat {
    let mut grounded = false;
    let mut first_err = NotGrounded::EmptyArm;
    for r in results {
        match r {
            Ok(true) => return Ok(true),
            Ok(false) => grounded = true,
            Err(e) => first_err = e,
        }
    }
    if grounded {
        Ok(false)
    } else {
        Err(first_err)
    }
}

impl<M: PopulationMask> CertificationModel<M> {
    pub fn sourced(&self) -> Result<(), NotGrounded> {
        if distinct_sources(&self.ledger, self.bases, self.seal) >= MIN_SOURCES {
            Ok(())
        } else {
            Err(NotGrounded::TooFewSources)
        }
    }

    /// Exposed against unexposed within population `p`.
    fn arms_in(&self, p: &M) -> (M, M) {
        (p.intersection(&self.exposed), p.difference(&self.exposed))
    }

    pub fn assoc_in(&self, p: &M) -> Sat {
        let (a, b) = self.arms_in(p);
        compare(&self.outcome, &a, &b, true)
    }

    pub fn declared(&self) -> impl Iterator<Item = M> + '_ {
        core::iter::once(self.universe.clone()).chain(
            self.strata
                .iter()
                .filter(|s| !s.is_empty())
                .map(move |s| s.intersection(&self.universe)),
        )
    }

    pub fn associated(&self) -> Sat {
        self.sourced()?;
        any(self.declared().map(|p| self.assoc_in(&p)))
    }

    /// The marginal variant, kept to measure where the chain breaks.
    pub fn associated_marginal(&self) -> Sat {
        self.sourced()?;
        self.assoc_in(&self.universe)
    }

    /// The robustness mask can only refute: the association must hold in the
    /// full population and after the mask.
    pub fn related(&self) -> Sat {
        self.sourced()?;
        any(self.declared().map(|p| {
            let full = self.assoc_in(&p)?;
            let kept = self.assoc_in(&p.intersection(&self.clean))?;
            Ok(full && kept)
        }))
    }

    pub fn contributes(&self) -> Sat {
        self.sourced()?;
        let mut strict = false;
        let mut any_stratum = false;
        for s in self.strata.iter().filter(|s| !s.is_empty()) {
            any_stratum = true;
            let s = s.intersection(&self.universe);
            let (a, b) = self.arms_in(&s);
            let (ka, kb) = (a.intersection(&self.clean), b.intersection(&self.clean));
            let y = &self.outcome;
            if !compare(y, &a, &b, false)? || !compare(y, &ka, &kb, false)? {
                return Ok(false);
            }
            strict |= compare(y, &a, &b, true)? && compare(y, &ka, &kb, true)?;
        }
        if !any_stratum {
            return Err(NotGrounded::NoDeclaredStratum);
        }
        Ok(strict)
    }

    pub fn causal_candidate(&self) -> Sat {
        if !self.contributes()? {
            return Ok(false);
        }
        let co = self
            .universe
            .intersection(&self.exposed)
            .intersection(&self.outcome);
        Ok(co.is_subset_of(&self.ordered))
    }

    pub fn causes(&self) -> Sat {
        match distinct_sources(&self.ledger, INTERVENTION, self.seal) {
            0 => return Err(NotGrounded::NoIdentificationDesign),
            n if n < MIN_SOURCES => return Err(NotGrounded::TooFewSources),
            _ => {}
        }
        compare(
            &self.outcome,
            &self.trial.intersection(&self.assigned),
            &self.trial.difference(&self.assigned),
            true,
        )
    }

    /// The randomized arms read as a population of their own.
    pub fn trial_view(&self) -> CertificationModel<M> {
        CertificationModel {
            universe: self.trial.clone(),
            exposed: self.assigned.clone(),
            strata: [self.trial.clone(), M::empty(), M::empty(), M::empty()],
            ordered: self.trial.clone(),
            bases: INTERVENTION,
            ..self.clone()
        }
    }

    /// The strongest certification the model satisfies: `Causes` from the
    /// executed randomized arms, otherwise the strongest observational one.
    pub fn certification(&self) -> Certification3 {
        if self.causes() == Ok(true) {
            return Certification3::Causes;
        }
        self.observational_certification()
    }

    /// The strongest certification the observational folds alone satisfy.
    /// `causes` is not in the list, so no amount of observation returns
    /// `Causes`.
    pub fn observational_certification(&self) -> Certification3 {
        let rungs: [(Certification3, Evaluator<M>); 4] = [
            (Certification3::CausalCandidate, Self::causal_candidate),
            (Certification3::Supports, Self::contributes),
            (Certification3::Related, Self::related),
            (Certification3::Associated, Self::associated),
        ];
        rungs
            .iter()
            .find(|(_, f)| f(self) == Ok(true))
            .map_or(Certification3::Open, |(c, _)| *c)
    }
}

// ── Builder ───────────────────────────────────────────────────────────────

/// Builds a [`CertificationModel`] unit by unit; [`ModelBuilder::units`]
/// panics past the mask's [`PopulationMask::CAPACITY`].
pub struct ModelBuilder<M: PopulationMask = u64> {
    next: u32,
    m: CertificationModel<M>,
}

/// `new()` builds the default 64-unit width, so it infers without annotation
/// (the `HashMap::new` / `HashMap::default` split); `default()` builds any
/// width, e.g. `ModelBuilder::<[u64; 16]>::default()`.
impl ModelBuilder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<M: PopulationMask> Default for ModelBuilder<M> {
    fn default() -> Self {
        ModelBuilder {
            next: 0,
            m: CertificationModel {
                seal: DatasetVersion(1),
                universe: M::empty(),
                exposed: M::empty(),
                outcome: M::empty(),
                strata: [M::empty(), M::empty(), M::empty(), M::empty()],
                ordered: M::empty(),
                clean: M::full(),
                trial: M::empty(),
                assigned: M::empty(),
                ledger: SupportLedger::new(),
                bases: OBSERVATION,
            },
        }
    }
}

impl<M: PopulationMask> ModelBuilder<M> {
    /// `n` fresh units, the first `hits` of them with the outcome.
    pub fn units(&mut self, n: u32, hits: u32) -> (M, M) {
        assert!(
            hits <= n && self.next + n <= M::CAPACITY,
            "fixture exceeds {} units",
            M::CAPACITY
        );
        let mut all = M::empty();
        let mut hit = M::empty();
        for i in 0..n {
            let bit = M::unit(self.next + i).expect("checked against CAPACITY");
            all = all.union(&bit);
            if i < hits {
                hit = hit.union(&bit);
            }
        }
        self.next += n;
        self.m.outcome = self.m.outcome.union(&hit);
        (all, hit)
    }

    /// Skip `n` units, so the next cell starts further into the mask.
    pub fn skip(&mut self, n: u32) -> &mut Self {
        assert!(
            self.next + n <= M::CAPACITY,
            "skip exceeds {} units",
            M::CAPACITY
        );
        self.next += n;
        self
    }

    /// Observational cell: `n` units in `stratum`, `hits` with the outcome.
    pub fn cell(&mut self, stratum: usize, exposed: bool, n: u32, hits: u32) -> (M, M) {
        let (all, hit) = self.units(n, hits);
        let m = &mut self.m;
        m.universe = m.universe.union(&all);
        m.strata[stratum] = m.strata[stratum].union(&all);
        m.ordered = m.ordered.union(&all);
        if exposed {
            m.exposed = m.exposed.union(&all);
        }
        (all, hit)
    }

    pub fn arm(&mut self, assigned: bool, n: u32, hits: u32) -> (M, M) {
        let (all, hit) = self.units(n, hits);
        self.m.trial = self.m.trial.union(&all);
        if assigned {
            self.m.assigned = self.m.assigned.union(&all);
        }
        (all, hit)
    }

    pub fn sources(&mut self, basis: SupportBasis, ids: &[u64]) -> &mut Self {
        let at = self.m.seal;
        self.sources_at(basis, ids, at)
    }

    pub fn sources_at(
        &mut self,
        basis: SupportBasis,
        ids: &[u64],
        at: DatasetVersion,
    ) -> &mut Self {
        for id in ids {
            self.m.ledger.record(SupportReceipt {
                basis,
                source: EvidenceSourceId(*id),
                at,
                strength: 128,
            });
        }
        self
    }

    /// The model under construction, for masks the cell and arm helpers do
    /// not set (the robustness mask, ordering).
    pub fn model_mut(&mut self) -> &mut CertificationModel<M> {
        &mut self.m
    }

    pub fn build(&self) -> CertificationModel<M> {
        self.m.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every fold of a model, so two models can be compared fold by fold.
    fn folds<M: PopulationMask>(m: &CertificationModel<M>) -> [Sat; 6] {
        [
            m.associated(),
            m.associated_marginal(),
            m.related(),
            m.contributes(),
            m.causal_candidate(),
            m.causes(),
        ]
    }

    /// The fixtures the P7a probe measured, written once over any width.
    /// `offset` units are skipped first, so a fixture can sit past unit 64.
    fn fixtures<M: PopulationMask>(offset: u32) -> Vec<CertificationModel<M>> {
        let mut out = Vec::new();
        // Robust association, two observational sources.
        let mut b = ModelBuilder::<M>::default();
        b.skip(offset);
        b.cell(0, true, 5, 4);
        b.cell(0, false, 5, 1);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        out.push(b.build());
        // The same with one source: not grounded.
        let mut b = ModelBuilder::<M>::default();
        b.skip(offset);
        b.cell(0, true, 5, 4);
        b.cell(0, false, 5, 1);
        b.sources(SupportBasis::DirectlyObserved, &[1]);
        out.push(b.build());
        // Simpson: positive marginally, reversed inside each stratum.
        let mut b = ModelBuilder::<M>::default();
        b.skip(offset);
        b.cell(0, true, 8, 6);
        b.cell(0, false, 2, 2);
        b.cell(1, true, 2, 0);
        b.cell(1, false, 8, 1);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        out.push(b.build());
        // Executed arms with two intervention sources.
        let mut b = ModelBuilder::<M>::default();
        b.skip(offset);
        b.cell(0, true, 4, 2);
        b.cell(0, false, 4, 2);
        b.arm(true, 6, 5);
        b.arm(false, 6, 1);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        b.sources(SupportBasis::InterventionBacked, &[3, 4]);
        out.push(b.build());
        out
    }

    #[test]
    fn every_carrier_width_certifies_a_fixture_identically() {
        let narrow = fixtures::<u64>(0);
        let two = fixtures::<[u64; 2]>(0);
        let cycle = fixtures::<[u64; 1024]>(0);
        let mut seen = Vec::new();
        for ((a, b), c) in narrow.iter().zip(&two).zip(&cycle) {
            assert_eq!(folds(a), folds(b));
            assert_eq!(folds(a), folds(c));
            assert_eq!(a.certification(), b.certification());
            assert_eq!(a.certification(), c.certification());
            seen.push(a.certification());
        }
        // Anti-vacuity: the fixtures span several certifications, so equal
        // results are not equal because everything is `Open`.
        assert_eq!(
            seen,
            [
                Certification3::CausalCandidate,
                Certification3::Open,
                // Simpson holds marginally and after the mask, so `Related`;
                // the reversal inside the strata stops it short of `Supports`.
                Certification3::Related,
                Certification3::Causes,
            ]
        );
    }

    #[test]
    fn units_past_the_first_word_are_counted() {
        // The same fixtures, placed entirely beyond unit 64 (and, in the
        // cycle-sized mask, beyond unit 60 000). A count that read only the
        // first word would see empty arms and return `EmptyArm`.
        let base = fixtures::<u64>(0);
        let far = fixtures::<[u64; 2]>(64);
        let farther = fixtures::<[u64; 1024]>(60_000);
        for ((a, b), c) in base.iter().zip(&far).zip(&farther) {
            assert_eq!(folds(a), folds(b));
            assert_eq!(folds(a), folds(c));
        }
        assert!(far[0].universe[0] == 0 && far[0].universe[1] != 0);
    }

    #[test]
    fn a_population_wider_than_64_units_is_decided() {
        // 60 exposed (45 hits) against 60 unexposed (15 hits): 120 units, which
        // a `u64` model cannot hold.
        let mut b = ModelBuilder::<[u64; 2]>::default();
        b.cell(0, true, 60, 45);
        b.cell(0, false, 60, 15);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        assert_eq!(b.build().associated(), Ok(true));
        // Silence twin: equal rates over the same 120 units.
        let mut b = ModelBuilder::<[u64; 2]>::default();
        b.cell(0, true, 60, 30);
        b.cell(0, false, 60, 30);
        b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
        assert_eq!(b.build().associated(), Ok(false));
    }

    #[test]
    fn units_stop_at_capacity() {
        assert_eq!(<u64 as PopulationMask>::unit(63), Some(1 << 63));
        assert_eq!(<u64 as PopulationMask>::unit(64), None);
        assert_eq!(<[u64; 2] as PopulationMask>::unit(64), Some([0, 1]));
        assert_eq!(<[u64; 2] as PopulationMask>::unit(128), None);
    }

    #[test]
    #[should_panic(expected = "fixture exceeds 128 units")]
    fn the_builder_refuses_past_capacity() {
        let mut b = ModelBuilder::<[u64; 2]>::default();
        b.cell(0, true, 100, 50);
        b.cell(0, false, 29, 10);
    }
}
