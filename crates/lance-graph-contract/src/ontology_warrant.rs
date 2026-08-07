//! `ontology_warrant` — grading a factfinder's ungraded verdict, without ever
//! letting the grade leak back into the fact.
//!
//! # The two rungs, kept apart by the type system
//!
//! A factfinder (OGAR's `ogar-elk` and siblings) answers **exactly**: this
//! subsumption is entailed, or it is not. That answer is rung 1 — retrieval —
//! and it is mandatory that it stay exact. Nothing above may make it
//! probabilistic.
//!
//! What genuinely IS graded is a different question: *how well warranted is a
//! claim, given how many independent sources speak to it.* That is rung 2, and
//! it is computed **over** facts rather than replacing them.
//!
//! This module keeps the two apart structurally rather than by convention:
//! [`Quorum`] carries the counts, [`Quorum::warrant`] returns a graded
//! [`NarsTruth`], and there is deliberately **no** method that turns a
//! `NarsTruth` back into an entailment. A caller that needs the exact answer
//! asks the factfinder; a caller that needs the warrant asks here; neither
//! surface can be mistaken for the other.
//!
//! # Silence is abstention, not dissent — the load-bearing rule
//!
//! When two independently authored ontologies are compared, a claim falls into
//! one of three states, and **conflating the second with the third inverts the
//! result**:
//!
//! | state | meaning | evidence |
//! |---|---|---|
//! | **corroborating** | the other source asserts the same relation | positive |
//! | **silent** | the other source asserts nothing about this pair | **none** |
//! | **conflicting** | the other source asserts the opposite direction | negative |
//!
//! A source that has no path between two classes has **not** denied the
//! relation — it has said nothing. Counting silence as dissent turns "the other
//! ontology is sparser than this one" into "the other ontology disagrees",
//! which is the opposite finding from the same data. So [`Quorum::warrant`]
//! lets `silent` affect **confidence only through what it is not** — it never
//! pushes frequency down.
//!
//! This is not a modelling preference. It was measured: on a real cross-ontology
//! comparison, reading silence as dissent reported ~50 % disagreement where the
//! sources that actually both spoke agreed 99.8 % of the time.
//!
//! # Zero-dep, and knows no factfinder
//!
//! This module names no ontology, no vocabulary and no producer crate. It takes
//! three counts. Any factfinder that can classify its comparisons into those
//! three buckets can feed it, and the contract crate stays dependency-free.

use crate::exploration::NarsTruth;

/// How many independent sources speak to one claim, and what they say.
///
/// Counts, not sources: this type deliberately cannot name who said what. The
/// provenance belongs with the factfinder that produced the comparison; what
/// reaches the grading layer is how many, in which direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Quorum {
    /// Sources asserting the same relation.
    pub corroborating: u16,
    /// Sources with no opinion on this pair. **Abstention, never dissent.**
    pub silent: u16,
    /// Sources asserting the opposite direction.
    pub conflicting: u16,
}

/// Evidence weight of one source that has spoken.
///
/// A single source is worth `1.0` evidence unit; NARS confidence for `n` units
/// is `n / (n + k)` with `k = 1`. Two independent sources agreeing therefore
/// give `2/3` rather than `1/2` — the whole reason a quorum is worth computing.
const EVIDENCE_PER_SOURCE: f32 = 1.0;

impl Quorum {
    /// A quorum from the three counts.
    #[must_use]
    pub const fn new(corroborating: u16, silent: u16, conflicting: u16) -> Self {
        Self {
            corroborating,
            silent,
            conflicting,
        }
    }

    /// How many sources actually asserted something. **Silence is excluded** —
    /// it is the denominator's job to count opinions, not participants.
    #[must_use]
    pub const fn speaking(self) -> u16 {
        self.corroborating.saturating_add(self.conflicting)
    }

    /// Whether any source spoke at all. `false` means the warrant is a prior
    /// and carries no evidence — a caller must not read it as agreement.
    #[must_use]
    pub const fn has_evidence(self) -> bool {
        self.speaking() > 0
    }

    /// **Rung 2 — the graded warrant.**
    ///
    /// Frequency is the share of *speaking* sources that corroborate; confidence
    /// grows with how many spoke. Silence enters neither: it is not counted as
    /// corroboration (which would inflate agreement) and not as conflict (which
    /// would manufacture disagreement). It simply leaves the claim less
    /// attested, which is exactly what a lower confidence means.
    ///
    /// With nothing speaking this returns [`NarsTruth::prior`] — a weak prior,
    /// not agreement. [`Self::has_evidence`] is how a caller tells the two
    /// apart, because `expectation()` alone cannot.
    #[must_use]
    pub fn warrant(self) -> NarsTruth {
        let speaking = f32::from(self.speaking());
        if speaking == 0.0 {
            return NarsTruth::prior();
        }
        let frequency = f32::from(self.corroborating) / speaking;
        let evidence = speaking * EVIDENCE_PER_SOURCE;
        let confidence = evidence / (evidence + 1.0);
        NarsTruth::new(frequency, confidence)
    }

    /// Fold another source's verdict in. Equivalent to incrementing the matching
    /// counter — provided so a caller accumulating over sources does not have to
    /// remember which bucket silence belongs in.
    #[must_use]
    pub const fn observe(self, v: SourceVerdict) -> Self {
        match v {
            SourceVerdict::Corroborates => Self {
                corroborating: self.corroborating.saturating_add(1),
                ..self
            },
            SourceVerdict::Silent => Self {
                silent: self.silent.saturating_add(1),
                ..self
            },
            SourceVerdict::Conflicts => Self {
                conflicting: self.conflicting.saturating_add(1),
                ..self
            },
        }
    }
}

/// What one source says about one claim.
///
/// Three variants, no `Unknown`: a source that was not consulted is not a
/// source. [`SourceVerdict::Silent`] means consulted **and** having no opinion,
/// which is a different fact and the one that matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceVerdict {
    /// Asserts the same relation.
    Corroborates,
    /// Consulted; asserts nothing about this pair.
    Silent,
    /// Asserts the opposite direction.
    Conflicts,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The load-bearing rule, as a test.** Silence must not move frequency.
    ///
    /// Two quorums with identical speaking sources but wildly different silence
    /// must agree on frequency exactly. Without this, a change that "counted
    /// abstentions" would pass every other test in this file.
    #[test]
    fn silence_does_not_move_frequency() {
        let quiet = Quorum::new(1, 0, 0).warrant();
        let loud = Quorum::new(1, 500, 0).warrant();
        assert_eq!(
            quiet.frequency, loud.frequency,
            "500 abstentions must not change what the one speaker said"
        );
        assert_eq!(
            quiet.confidence, loud.confidence,
            "…nor how well attested it is"
        );
    }

    /// …and the twin: a CONFLICT does move it. A rule that ignored everything
    /// would satisfy the test above.
    #[test]
    fn a_conflict_does_move_frequency() {
        let agreed = Quorum::new(2, 0, 0).warrant();
        let split = Quorum::new(1, 0, 1).warrant();
        assert!(
            split.frequency < agreed.frequency,
            "a dissenting source must lower frequency ({} vs {})",
            split.frequency,
            agreed.frequency
        );
        assert!((split.frequency - 0.5).abs() < 1e-6, "one for, one against");
    }

    /// Corroboration raises confidence — the reason a quorum is computed at all.
    /// A confidence that ignored source count would fail here.
    #[test]
    fn more_speaking_sources_raise_confidence() {
        let one = Quorum::new(1, 0, 0).warrant();
        let two = Quorum::new(2, 0, 0).warrant();
        let five = Quorum::new(5, 0, 0).warrant();
        assert!(two.confidence > one.confidence);
        assert!(five.confidence > two.confidence);
        assert_eq!(one.frequency, two.frequency, "agreement stays agreement");
        assert!(five.confidence < 1.0, "confidence never reaches certainty");
    }

    /// No evidence yields a PRIOR, and the caller can tell. `expectation()`
    /// alone cannot distinguish "nobody spoke" from "opinion split down the
    /// middle" — both sit at 0.5 — so `has_evidence` is the discriminator and
    /// this test is what keeps it honest.
    #[test]
    fn silence_from_everyone_is_a_prior_not_agreement() {
        let nobody = Quorum::new(0, 7, 0);
        assert!(!nobody.has_evidence());
        let w = nobody.warrant();
        assert!((w.expectation() - 0.5).abs() < 0.05);

        let split = Quorum::new(1, 0, 1);
        assert!(split.has_evidence(), "a split IS evidence, unlike silence");
        assert!(
            (split.warrant().expectation() - w.expectation()).abs() < 0.05,
            "…and expectation alone cannot tell them apart — hence has_evidence"
        );
    }

    /// `observe` routes each verdict to the right bucket. A folder that put
    /// silence in `corroborating` would show up as a frequency change.
    #[test]
    fn observe_routes_each_verdict_to_its_own_bucket() {
        let q = Quorum::default()
            .observe(SourceVerdict::Corroborates)
            .observe(SourceVerdict::Silent)
            .observe(SourceVerdict::Silent)
            .observe(SourceVerdict::Conflicts);
        assert_eq!(q, Quorum::new(1, 2, 1));
        assert_eq!(q.speaking(), 2, "silence is not a speaker");
        assert!((q.warrant().frequency - 0.5).abs() < 1e-6);
    }

    /// **The measured scenario, as a regression.** On the real comparison the
    /// sources that both spoke agreed 1,730 : 3, while 1,693 were silent.
    /// Reading silence as dissent would report ~50 % agreement; reading it as
    /// abstention reports 99.8 %. Both numbers are computed here so the
    /// difference is visible rather than asserted.
    #[test]
    fn the_measured_cross_ontology_case_reads_as_agreement() {
        let q = Quorum::new(1_730, 1_693, 3);
        let w = q.warrant();
        assert!(
            w.frequency > 0.998,
            "sources that spoke agreed overwhelmingly, got {}",
            w.frequency
        );
        // What the WRONG reading would have produced, computed rather than
        // claimed: silence folded into the dissent bucket.
        let wrong = Quorum::new(1_730, 0, 1_693 + 3).warrant();
        assert!(
            wrong.frequency < 0.52,
            "the inverted reading lands near a coin flip, got {}",
            wrong.frequency
        );
    }
}
