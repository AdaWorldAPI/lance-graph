//! Typed causal-relation audit — four orthogonal axes, never one merged carrier.
//!
//! ## Why this module exists
//!
//! The substrate carries Pearl's *vocabulary* (`RungLevel::Counterfactual`,
//! `InferenceType::Intervention`) at several layers, but an edge asserting
//! "X causes Y" has, until now, been **untyped**: nothing distinguishes a
//! causal claim a corpus merely *reports* from one the system *observed*, or a
//! claim about the world from one about its own derivations. An audit that
//! cannot make those distinctions will happily promote a sentence into an
//! interventional fact.
//!
//! ## The four axes, and why they must stay four
//!
//! Causality has several independent geometries. Merging any two of them
//! produces a carrier that looks tidy and silently discards a dimension:
//!
//! | Axis | Question | Type |
//! |---|---|---|
//! | **Kind** | causal at all, or correlational / definitional / temporal? | [`RelationClassification`] |
//! | **Locus** | *where in the architecture* does the relation operate? | [`CausalLocus`] |
//! | **Domain** | *what subject matter* does it concern? | [`WorldDomain`] |
//! | **Scope** | a general regularity, or this particular episode? | [`CausalScope`] |
//! | **Support** | what evidence backs it, and of which kinds? | [`SupportLedger`] |
//!
//! **Locus is not domain.** `accusative marker → parser selects object role`
//! concerns physical text, a social language convention, and a formal grammar
//! all at once — yet its causal locus is unambiguously
//! [`Interpretive`](CausalLocus::Interpretive). `recipe 17 + rail → belief P
//! admitted` is [`Derivational`](CausalLocus::Derivational) whether P is about
//! physics or politics. Classifying only by domain reproduces the original
//! category error under prettier names.
//!
//! **Scope is not grammatical voice.** "The outage was caused by cable damage"
//! is passive and [`Token`](CausalScope::Token); "Smoking causes cancer" is
//! active and [`Type`](CausalScope::Type). Voice does not predict scope, so the
//! axis is named for what it measures.
//!
//! **Support is many-of, not one-of.** A single edge can be simultaneously
//! text-attested, linguistically asserted, derivationally traced, and
//! cross-environment invariant — those are not competing alternatives. A
//! single-valued `support_basis` field would force a later process to elect one
//! and silently discard the rest, which is exactly the fold this module is
//! built to prevent. Hence [`SupportLedger`], a receipt list.
//!
//! ## Illegal states are unrepresentable
//!
//! [`RelationClassification`] is a sum type, not a bag of `Option`s: a
//! non-causal relation cannot carry a `CausalLocus`, and an unclassified edge
//! has [`Unclassified`](RelationClassification::Unclassified) to sit in rather
//! than being coerced into a half-typed causal claim. An audit that cannot say
//! "not yet classified" will invent classifications.
//!
//! ## Classification and support are separable, and stay separable
//!
//! [`AuditedRelation`] holds the two side by side rather than copying support
//! into every classification branch, because their lifecycles differ: support
//! **accumulates** while classification is still `Unclassified`, and
//! classification can be **revised** without rewriting historical receipts.

use crate::scheduler::DatasetVersion;

/// Where in the architecture a causal relation operates.
///
/// Orthogonal to [`WorldDomain`] (what it is *about*) — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalLocus {
    /// Out in the modelled world: physical, social, or institutional
    /// mechanisms the system did not produce.
    World,
    /// In the act of interpretation: a parse decision, a role assignment, a
    /// disambiguation. The cause operates on *reading*, not on the world.
    Interpretive,
    /// In the system's own inference: a recipe, a rail, a rule admitted this
    /// belief from those premises. The provenance of a conclusion.
    Derivational,
    /// In the system's own processing history: what it saw, in what order,
    /// under what load — causes that operate on the experiencing substrate.
    Experiential,
    /// Not yet determined. Distinct from "no locus": this says *unknown*, and
    /// must never be defaulted into `World`.
    Unknown,
}

/// What subject matter a causal relation concerns.
///
/// Primarily meaningful under [`CausalLocus::World`], though it may annotate
/// the content processed by any locus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorldDomain {
    /// Physical mechanism.
    Physical,
    /// Social dynamics between agents.
    Social,
    /// Deliberate action by an agent with intent.
    Intentional,
    /// Rules, policies, organisations — causes that hold because a body says so.
    Institutional,
    /// Within a formal model or calculus, where "cause" means derivation under
    /// the model's own rules.
    FormalModel,
    /// Not yet determined.
    Unknown,
}

/// Whether a causal claim is general or particular.
///
/// NOT grammatical voice — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalScope {
    /// A general regularity: "smoking causes cancer".
    Type,
    /// This particular episode: "the cable damage caused Tuesday's outage".
    /// Halpern-Pearl *actual* causality.
    Token,
}

/// A relation that is not a causal claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NonCausalKind {
    /// Co-varies, with no direction asserted.
    Correlational,
    /// True by definition or stipulation — "a bachelor is unmarried".
    Definitional,
    /// Ordered in time, with no mechanism claimed. The most common thing
    /// mistaken for causal.
    Temporal,
    /// Part-of / member-of structure.
    Mereological,
    /// A relation the classifier can name but that fits no bucket above.
    Other,
}

/// An opaque handle for a relation the audit has not classified yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct RelationId(pub u64);

/// What kind of relation this is — a sum type, so a non-causal relation
/// *cannot* carry a causal locus and an unclassified one need not pretend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationClassification {
    /// Classified, and not a causal claim.
    NonCausal { kind: NonCausalKind },
    /// Classified as causal. `locus` and `scope` are REQUIRED — a causal claim
    /// that cannot say where it operates or whether it is general is not
    /// classified, it is [`Unclassified`](RelationClassification::Unclassified).
    /// `world_domain` is optional because it is only fully meaningful under
    /// [`CausalLocus::World`].
    Causal {
        /// Where in the architecture the relation operates.
        locus: CausalLocus,
        /// What subject matter it concerns, when that is known.
        world_domain: Option<WorldDomain>,
        /// General regularity or particular episode.
        scope: CausalScope,
    },
    /// Not yet classified — an honest resting place. Support may accumulate
    /// against this edge for as long as it sits here.
    Unclassified { raw_relation: RelationId },
}

impl RelationClassification {
    /// Is this a causal claim? `false` for both `NonCausal` and `Unclassified`
    /// — an unclassified edge is NOT provisionally causal.
    #[inline]
    #[must_use]
    pub const fn is_causal(&self) -> bool {
        matches!(self, Self::Causal { .. })
    }

    /// The locus, when this is a classified causal relation.
    #[inline]
    #[must_use]
    pub const fn locus(&self) -> Option<CausalLocus> {
        match self {
            Self::Causal { locus, .. } => Some(*locus),
            _ => None,
        }
    }
}

/// One *kind* of evidential support. An edge normally has several at once —
/// this is a receipt category, never a whole verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SupportBasis {
    /// A source *states* the causal claim. The corpus attests that someone
    /// wrote it; it does not witness the mechanism.
    TextAttested = 0,
    /// The system observed both relata occurring. Observation of occurrence,
    /// still not of mechanism.
    DirectlyObserved = 1,
    /// Only the ordering is known. The weakest basis that still looks causal.
    TemporalOrderOnly = 2,
    /// Carried by causative wording ("because", "led to", a causative verb) —
    /// a grammatical signal, not evidence about the world.
    LinguisticallyAsserted = 3,
    /// Holds across environments/contexts that vary other factors.
    CrossEnvironmentInvariant = 4,
    /// Changing the antecedent under controlled mechanism conditions changed
    /// the consequent. The only basis that earns interventional standing —
    /// and the one a text corpus can never produce.
    InterventionBacked = 5,
    /// Reproduced in simulation, under the simulation's own assumptions.
    SimulationOnly = 6,
    /// A derivation admitted it: recipe, rule, or rail, with a traceable path.
    DerivationalTrace = 7,
    /// Provenance not recorded.
    Unknown = 8,
}

impl SupportBasis {
    /// Bit position in a [`SupportProfile`] mask.
    #[inline]
    #[must_use]
    pub const fn bit(self) -> u16 {
        1u16 << (self as u8)
    }

    /// Does this basis, on its own, license treating the relation as
    /// interventionally established? Only
    /// [`InterventionBacked`](SupportBasis::InterventionBacked).
    ///
    /// Deliberately narrow: `CrossEnvironmentInvariant` is strong evidence and
    /// still not an intervention, and the gap between them is the whole reason
    /// this enum has nine variants instead of four.
    #[inline]
    #[must_use]
    pub const fn is_intervention_grade(self) -> bool {
        matches!(self, Self::InterventionBacked)
    }
}

/// An opaque, stable identity for an evidence source.
///
/// NOT a bit position. Arbitrary and sparse — a term id, corpus id, witness
/// id, or hash. Mapping it to a dense local slot is a registry's job, never an
/// arithmetic accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct SourceId(pub u64);

/// One piece of evidence for a relation: which kind, from whom, when, how
/// strong.
///
/// Receipts are the **source of truth** for provenance; [`SupportProfile`] is a
/// derived projection of them. That direction matters — three independent
/// text attestations and one attestation counted three times produce identical
/// masks but must never produce identical strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SupportReceipt {
    /// Which kind of support this is.
    pub basis: SupportBasis,
    /// Who supplied it — a stable external identity.
    pub source: SourceId,
    /// When it was recorded.
    ///
    /// **Known gap:** this is a storage revision, NOT an epistemic view. It
    /// answers "which dataset version" but not "which version was this observer
    /// permitted to see, under which read mode". The planner-side ledger should
    /// carry the richer `QueryReference` once that type is reachable from the
    /// contract; until then this field is deliberately the weaker identity and
    /// is labelled as such rather than silently standing in for the stronger
    /// one.
    pub at: DatasetVersion,
    /// Weight of this individual receipt, `0..=255`. Per-receipt, never a
    /// pre-aggregated score.
    pub strength: u8,
}

/// The receipt ledger for one relation — the canonical provenance record.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SupportLedger {
    receipts: Vec<SupportReceipt>,
}

impl SupportLedger {
    /// An empty ledger — the correct starting state for a relation nobody has
    /// evidenced yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            receipts: Vec::new(),
        }
    }

    /// Record a receipt. Append-only: evidence accumulates, and a later
    /// classification never rewrites it.
    pub fn record(&mut self, receipt: SupportReceipt) {
        self.receipts.push(receipt);
    }

    /// Every receipt, in the order recorded.
    #[inline]
    #[must_use]
    pub fn receipts(&self) -> &[SupportReceipt] {
        &self.receipts
    }

    /// Is there no evidence at all?
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }

    /// Withdraw every receipt from `source`, returning how many were removed.
    ///
    /// This is why receipts are canonical and a mask is not: withdrawal
    /// requires knowing *which* evidence came from whom, and a bitmask cannot
    /// answer that.
    pub fn withdraw_source(&mut self, source: SourceId) -> usize {
        let before = self.receipts.len();
        self.receipts.retain(|r| r.source != source);
        before - self.receipts.len()
    }

    /// How many DISTINCT sources back this relation with `basis`.
    ///
    /// Distinct-source counting, not receipt counting: one source repeating
    /// itself is not corroboration. Linear scan — ledgers are small, and the
    /// hot path reads [`SupportProfile`], not this.
    #[must_use]
    pub fn distinct_sources_for(&self, basis: SupportBasis) -> usize {
        let mut seen: Vec<SourceId> = Vec::new();
        for r in self.receipts.iter().filter(|r| r.basis == basis) {
            if !seen.contains(&r.source) {
                seen.push(r.source);
            }
        }
        seen.len()
    }

    /// Does any receipt license interventional standing?
    #[must_use]
    pub fn has_intervention_grade(&self) -> bool {
        self.receipts
            .iter()
            .any(|r| r.basis.is_intervention_grade())
    }

    /// Project to the compact [`SupportProfile`] for the SIMD / fixed-width path.
    #[must_use]
    pub fn profile(&self) -> SupportProfile {
        let mut p = SupportProfile::default();
        for r in &self.receipts {
            p.basis_mask |= r.basis.bit();
            let slot = r.basis as usize;
            p.receipt_counts[slot] = p.receipt_counts[slot].saturating_add(1);
            p.strength[slot] = p.strength[slot].saturating_add(r.strength);
        }
        for basis in SupportBasis::ALL {
            p.distinct_sources[basis as usize] =
                u8::try_from(self.distinct_sources_for(basis)).unwrap_or(u8::MAX);
        }
        p
    }
}

impl SupportBasis {
    /// Every variant, for exhaustive projection.
    pub const ALL: [SupportBasis; 9] = [
        Self::TextAttested,
        Self::DirectlyObserved,
        Self::TemporalOrderOnly,
        Self::LinguisticallyAsserted,
        Self::CrossEnvironmentInvariant,
        Self::InterventionBacked,
        Self::SimulationOnly,
        Self::DerivationalTrace,
        Self::Unknown,
    ];
}

/// Fixed-width projection of a [`SupportLedger`] — derived, never authoritative.
///
/// Keeps `receipt_counts` and `distinct_sources` SEPARATE on purpose: one
/// source attesting three times and three sources attesting once each share a
/// `basis_mask` and must not share a corroboration reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SupportProfile {
    /// One bit per [`SupportBasis`] present.
    pub basis_mask: u16,
    /// Receipts recorded per basis.
    pub receipt_counts: [u8; 9],
    /// DISTINCT sources per basis — the corroboration reading.
    pub distinct_sources: [u8; 9],
    /// Summed receipt strength per basis (saturating).
    pub strength: [u8; 9],
}

impl SupportProfile {
    /// Is `basis` present at all?
    #[inline]
    #[must_use]
    pub const fn has(&self, basis: SupportBasis) -> bool {
        self.basis_mask & basis.bit() != 0
    }

    /// How many distinct bases back this relation.
    #[inline]
    #[must_use]
    pub const fn basis_diversity(&self) -> u32 {
        self.basis_mask.count_ones()
    }
}

/// A relation with its classification and its evidence, held separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditedRelation {
    /// What kind of relation this is.
    pub classification: RelationClassification,
    /// What backs it.
    pub support: SupportLedger,
}

impl AuditedRelation {
    /// A relation entering the audit with no classification and no evidence.
    #[must_use]
    pub fn unclassified(raw: RelationId) -> Self {
        Self {
            classification: RelationClassification::Unclassified { raw_relation: raw },
            support: SupportLedger::new(),
        }
    }

    /// Revise the classification, leaving the receipt ledger untouched.
    ///
    /// The whole point of keeping the two apart: re-reading an edge as
    /// `Derivational` rather than `World` must not disturb the record of who
    /// attested it.
    pub fn reclassify(&mut self, classification: RelationClassification) {
        self.classification = classification;
    }

    /// May this relation be treated as interventionally established?
    ///
    /// Requires BOTH a causal classification AND an intervention-grade receipt.
    /// A corpus-derived edge fails this no matter how many text attestations it
    /// accumulates — which is the guarantee this module exists to provide.
    #[must_use]
    pub fn is_intervention_established(&self) -> bool {
        self.classification.is_causal() && self.support.has_intervention_grade()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(basis: SupportBasis, source: u64, strength: u8) -> SupportReceipt {
        SupportReceipt {
            basis,
            source: SourceId(source),
            at: DatasetVersion(1),
            strength,
        }
    }

    /// The load-bearing invariant: support is MANY-of. An edge carrying four
    /// distinct bases must keep all four — no election, no discard.
    #[test]
    fn support_is_many_of_not_one_of() {
        let mut led = SupportLedger::new();
        for b in [
            SupportBasis::TextAttested,
            SupportBasis::LinguisticallyAsserted,
            SupportBasis::DerivationalTrace,
            SupportBasis::CrossEnvironmentInvariant,
        ] {
            led.record(receipt(b, 7, 10));
        }
        let p = led.profile();
        assert_eq!(p.basis_diversity(), 4, "all four bases survive projection");
        assert!(p.has(SupportBasis::TextAttested));
        assert!(p.has(SupportBasis::DerivationalTrace));
    }

    /// Three independent attestations and one attestation repeated three times
    /// share a `basis_mask` — and MUST NOT read as equally corroborated.
    /// This is the compression the receipt ledger exists to refuse.
    #[test]
    fn repeated_source_is_not_corroboration() {
        let mut independent = SupportLedger::new();
        for src in [1, 2, 3] {
            independent.record(receipt(SupportBasis::TextAttested, src, 10));
        }

        let mut repeated = SupportLedger::new();
        for _ in 0..3 {
            repeated.record(receipt(SupportBasis::TextAttested, 1, 10));
        }

        let (a, b) = (independent.profile(), repeated.profile());
        assert_eq!(a.basis_mask, b.basis_mask, "masks are identical…");
        assert_eq!(a.receipt_counts, b.receipt_counts, "…and so are raw counts");
        assert_eq!(a.distinct_sources[SupportBasis::TextAttested as usize], 3);
        assert_eq!(b.distinct_sources[SupportBasis::TextAttested as usize], 1);
        assert_ne!(
            a.distinct_sources, b.distinct_sources,
            "corroboration must distinguish them"
        );
    }

    /// A corpus edge cannot reach interventional standing by piling on text.
    #[test]
    fn text_attestation_never_becomes_intervention() {
        let mut r = AuditedRelation::unclassified(RelationId(9));
        r.reclassify(RelationClassification::Causal {
            locus: CausalLocus::World,
            world_domain: Some(WorldDomain::Physical),
            scope: CausalScope::Type,
        });
        for src in 0..50 {
            r.support
                .record(receipt(SupportBasis::TextAttested, src, 255));
            r.support
                .record(receipt(SupportBasis::LinguisticallyAsserted, src, 255));
        }
        assert!(
            !r.is_intervention_established(),
            "100 receipts, zero interventions"
        );

        // …and ONE genuine intervention flips it.
        r.support
            .record(receipt(SupportBasis::InterventionBacked, 999, 1));
        assert!(r.is_intervention_established());
    }

    /// Classification is revisable; receipts are not disturbed by revision.
    #[test]
    fn support_survives_reclassification() {
        let mut r = AuditedRelation::unclassified(RelationId(1));
        r.support
            .record(receipt(SupportBasis::DerivationalTrace, 4, 30));
        r.support.record(receipt(SupportBasis::TextAttested, 5, 20));
        let before = r.support.clone();

        r.reclassify(RelationClassification::Causal {
            locus: CausalLocus::Derivational,
            world_domain: None,
            scope: CausalScope::Token,
        });
        assert_eq!(r.support, before, "revision must not rewrite history");

        r.reclassify(RelationClassification::NonCausal {
            kind: NonCausalKind::Temporal,
        });
        assert_eq!(r.support, before);
    }

    /// An unclassified edge is NOT provisionally causal, and support may
    /// accumulate against it while it waits.
    #[test]
    fn unclassified_is_not_causal_but_still_collects_evidence() {
        let mut r = AuditedRelation::unclassified(RelationId(3));
        assert!(!r.classification.is_causal());
        assert_eq!(r.classification.locus(), None);

        r.support.record(receipt(SupportBasis::TextAttested, 1, 10));
        assert!(!r.support.is_empty());
        assert!(
            !r.is_intervention_established(),
            "no classification, no standing"
        );
    }

    /// Withdrawal is per-source and exact — the operation a bitmask cannot do.
    #[test]
    fn withdrawal_removes_exactly_one_sources_receipts() {
        let mut led = SupportLedger::new();
        led.record(receipt(SupportBasis::TextAttested, 1, 10));
        led.record(receipt(SupportBasis::DerivationalTrace, 1, 10));
        led.record(receipt(SupportBasis::TextAttested, 2, 10));

        assert_eq!(led.withdraw_source(SourceId(1)), 2);
        assert_eq!(led.receipts().len(), 1);
        assert_eq!(led.receipts()[0].source, SourceId(2));
        assert_eq!(
            led.withdraw_source(SourceId(42)),
            0,
            "absent source is a no-op"
        );
    }

    /// Locus and domain vary INDEPENDENTLY — the orthogonality receipt.
    ///
    /// Two witnesses, both non-trivial: locus changes while domain holds, and
    /// domain changes while locus holds. One-directional variation would mean
    /// one axis is derived from the other and should not be a stored field.
    #[test]
    fn locus_and_domain_are_independently_variable() {
        let causal = |locus, domain| RelationClassification::Causal {
            locus,
            world_domain: Some(domain),
            scope: CausalScope::Type,
        };

        // Witness 1: locus varies, domain fixed (Social).
        assert_ne!(
            causal(CausalLocus::World, WorldDomain::Social),
            causal(CausalLocus::Interpretive, WorldDomain::Social)
        );
        // Witness 2: domain varies, locus fixed (Interpretive).
        assert_ne!(
            causal(CausalLocus::Interpretive, WorldDomain::Social),
            causal(CausalLocus::Interpretive, WorldDomain::FormalModel)
        );
    }

    /// Scope varies independently of BOTH locus and domain.
    #[test]
    fn scope_is_independent_of_locus_and_domain() {
        let at = |scope| RelationClassification::Causal {
            locus: CausalLocus::World,
            world_domain: Some(WorldDomain::Physical),
            scope,
        };
        assert_ne!(at(CausalScope::Type), at(CausalScope::Token));
    }
}
