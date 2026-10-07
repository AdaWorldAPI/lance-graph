//! The P7a sealed model (D-GSO-7a, #1369): unit-mask populations, the
//! satisfaction folds that decide each certification contract, and the
//! fixture builder. Moved here unchanged from `relational_certification_probe`
//! (where it is tested) so that `pearl_ladder_probe` executes the same folds
//! instead of a copy. Items are `pub` only so the including examples can
//! reach them; each uses a subset, hence the `dead_code` allowance.
#![allow(dead_code)]

use crate::certification_reading::Contract;
use lance_graph_contract::causal_audit::{
    EvidenceSourceId, SupportBasis, SupportLedger, SupportReceipt,
};
use lance_graph_contract::scheduler::DatasetVersion;

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

/// A satisfaction fold over a sealed model.
pub type Evaluator = fn(&Model) -> Sat;

/// One sealed model: unit masks over at most 64 units.
#[derive(Debug, Clone)]
pub struct Model {
    pub seal: DatasetVersion,
    pub universe: u64,
    pub exposed: u64,
    pub outcome: u64,
    /// The declared comparison partition, fixed before the data are read.
    pub strata: [u64; 4],
    /// Units whose exposure precedes their outcome.
    pub ordered: u64,
    /// Units the robustness mask keeps.
    pub clean: u64,
    /// Randomized identification design.
    pub trial: u64,
    pub assigned: u64,
    pub ledger: SupportLedger,
    /// Which bases count as this model's evidence for the association
    /// contracts.
    pub bases: &'static [SupportBasis],
}

pub fn count(m: u64) -> u64 {
    u64::from(m.count_ones())
}

/// rate(Y | a) compared with rate(Y | b) by cross-multiplication.
pub fn compare(outcome: u64, a: u64, b: u64, strict: bool) -> Sat {
    let (na, nb) = (count(a), count(b));
    if na == 0 || nb == 0 {
        return Err(NotGrounded::EmptyArm);
    }
    let (lhs, rhs) = (count(outcome & a) * nb, count(outcome & b) * na);
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
pub fn any(results: impl Iterator<Item = Sat>) -> Sat {
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

impl Model {
    pub fn sourced(&self) -> Result<(), NotGrounded> {
        if distinct_sources(&self.ledger, self.bases, self.seal) >= MIN_SOURCES {
            Ok(())
        } else {
            Err(NotGrounded::TooFewSources)
        }
    }

    pub fn assoc_in(&self, p: u64) -> Sat {
        compare(self.outcome, p & self.exposed, p & !self.exposed, true)
    }

    pub fn declared(&self) -> impl Iterator<Item = u64> + '_ {
        core::iter::once(self.universe).chain(
            self.strata
                .iter()
                .filter(|s| **s != 0)
                .map(move |s| s & self.universe),
        )
    }

    pub fn associated(&self) -> Sat {
        self.sourced()?;
        any(self.declared().map(|p| self.assoc_in(p)))
    }

    /// The marginal variant, kept to measure where the chain breaks.
    pub fn associated_marginal(&self) -> Sat {
        self.sourced()?;
        self.assoc_in(self.universe)
    }

    /// The robustness mask can only refute: the association must hold in the
    /// full population and after the mask.
    pub fn related(&self) -> Sat {
        self.sourced()?;
        any(self.declared().map(|p| {
            let full = self.assoc_in(p)?;
            let kept = self.assoc_in(p & self.clean)?;
            Ok(full && kept)
        }))
    }

    pub fn contributes(&self) -> Sat {
        self.sourced()?;
        let mut strict = false;
        let mut any_stratum = false;
        for s in self.strata.iter().filter(|s| **s != 0) {
            any_stratum = true;
            let s = s & self.universe;
            let (a, b) = (s & self.exposed, s & !self.exposed);
            let (ka, kb) = (a & self.clean, b & self.clean);
            if !compare(self.outcome, a, b, false)? || !compare(self.outcome, ka, kb, false)? {
                return Ok(false);
            }
            strict |= compare(self.outcome, a, b, true)? && compare(self.outcome, ka, kb, true)?;
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
        let co = self.universe & self.exposed & self.outcome;
        Ok(co & !self.ordered == 0)
    }

    pub fn causes(&self) -> Sat {
        match distinct_sources(&self.ledger, INTERVENTION, self.seal) {
            0 => return Err(NotGrounded::NoIdentificationDesign),
            n if n < MIN_SOURCES => return Err(NotGrounded::TooFewSources),
            _ => {}
        }
        compare(
            self.outcome,
            self.trial & self.assigned,
            self.trial & !self.assigned,
            true,
        )
    }

    /// The randomized arms read as a population of their own.
    pub fn trial_view(&self) -> Model {
        Model {
            universe: self.trial,
            exposed: self.assigned,
            strata: [self.trial, 0, 0, 0],
            ordered: self.trial,
            bases: INTERVENTION,
            ..self.clone()
        }
    }

    /// The strongest contract the model satisfies.
    pub fn certify(&self) -> Contract {
        if self.causes() == Ok(true) {
            return Contract::Causes;
        }
        self.certify_observational()
    }

    /// The strongest contract the observational folds alone satisfy. `causes`
    /// is not in the list, so no amount of observation returns `Causes`.
    pub fn certify_observational(&self) -> Contract {
        let rungs: [(Contract, Evaluator); 4] = [
            (Contract::CausalCandidate, Model::causal_candidate),
            (Contract::Contributes, Model::contributes),
            (Contract::Related, Model::related),
            (Contract::Associated, Model::associated),
        ];
        rungs
            .iter()
            .find(|(_, f)| f(self) == Ok(true))
            .map_or(Contract::Open, |(c, _)| *c)
    }
}

// ── Fixture builder ───────────────────────────────────────────────────────

pub struct Builder {
    pub next: u32,
    pub m: Model,
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            next: 0,
            m: Model {
                seal: DatasetVersion(1),
                universe: 0,
                exposed: 0,
                outcome: 0,
                strata: [0; 4],
                ordered: 0,
                clean: u64::MAX,
                trial: 0,
                assigned: 0,
                ledger: SupportLedger::new(),
                bases: OBSERVATION,
            },
        }
    }

    pub fn units(&mut self, n: u32, hits: u32) -> (u64, u64) {
        assert!(hits <= n && self.next + n <= 64, "fixture exceeds 64 units");
        let mut all = 0;
        let mut hit = 0;
        for i in 0..n {
            let bit = 1u64 << (self.next + i);
            all |= bit;
            if i < hits {
                hit |= bit;
            }
        }
        self.next += n;
        self.m.outcome |= hit;
        (all, hit)
    }

    /// Observational cell: `n` units in `stratum`, `hits` with the outcome.
    pub fn cell(&mut self, stratum: usize, exposed: bool, n: u32, hits: u32) -> (u64, u64) {
        let (all, hit) = self.units(n, hits);
        self.m.universe |= all;
        self.m.strata[stratum] |= all;
        self.m.ordered |= all;
        if exposed {
            self.m.exposed |= all;
        }
        (all, hit)
    }

    pub fn arm(&mut self, assigned: bool, n: u32, hits: u32) -> (u64, u64) {
        let (all, hit) = self.units(n, hits);
        self.m.trial |= all;
        if assigned {
            self.m.assigned |= all;
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

    pub fn build(&self) -> Model {
        self.m.clone()
    }
}

/// A robust single-stratum association: 4/5 exposed vs 1/5 unexposed.
pub fn robust_association() -> Builder {
    let mut b = Builder::new();
    b.cell(0, true, 5, 4);
    b.cell(0, false, 5, 1);
    b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
    b
}

/// Within each stratum A raises Y; pooled, A lowers it (Simpson).
pub fn simpson_population() -> Builder {
    let mut b = Builder::new();
    // stratum 0: mostly exposed, low base rate
    b.cell(0, true, 8, 2); // 0.25
    b.cell(0, false, 2, 0); // 0.00
                            // stratum 1: mostly unexposed, high base rate
    b.cell(1, true, 2, 2); // 1.00
    b.cell(1, false, 8, 6); // 0.75
    b.sources(SupportBasis::DirectlyObserved, &[1, 2]);
    b
}

/// A positive randomized trial with two independent intervention sources.
pub fn add_positive_trial(b: &mut Builder) {
    b.arm(true, 6, 5);
    b.arm(false, 6, 1);
    b.sources(SupportBasis::InterventionBacked, &[10, 11]);
}
