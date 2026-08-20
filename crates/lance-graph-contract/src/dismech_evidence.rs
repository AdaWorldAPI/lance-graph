//! `dismech_evidence` — the compact evidence vocabulary measured on the real
//! DisMech corpus, plus the typed citation sidecar.
//!
//! **Source-side only.** This module encodes what the DisMech YAML *says*. It
//! deliberately does NOT reference `CausalEdge64`: the durable causal overlay
//! must not become a pile of hot reasoning registers (operator ruling). The
//! mapping [`DismechTopology`] → CE64 bits 59..60 happens at HYDRATION, in the
//! consumer, and is a 1:1 read of [`DismechTopology::to_bits_2`].
//!
//! # Measured, not designed (upstream `monarch-initiative/dismech`, 2,100 files)
//!
//! Every cardinality below was counted on the real corpus before this module
//! existed; each enum is exhaustively round-tripped in the tests.
//!
//! | field | states | bits | occurrences |
//! |---|---|---|---|
//! | `causal_link_type` | 4 | 2 | **17,998** |
//! | `supports` | 4 | 2 | ~89,800 |
//! | `evidence_source` | 5 | 3 | ~79,200 |
//! | `modifier` | 7 | 3 | 9,926 |
//! | `frequency` | 19 | 5 | 11,767 |
//!
//! `causal_link_type`'s four values are *exactly* the four `CausalTopology`
//! states, so the mapping is **source-authoritative** — never inferred from
//! confidence, edge count, or predicate name.
//!
//! ## Why every parse FAILS CLOSED
//!
//! An unrecognised source token returns `None`. It must never fold into a
//! neighbouring state and must never default to `Unknown`: `UNKNOWN` is a value
//! the corpus actually asserts (408 topology rows), so silently minting it from
//! a parse failure would forge an assertion the source never made.
//!
//! # The citation sidecar
//!
//! Measured: **131,904** reference-title occurrences over **31,361** distinct
//! titles (4.21× reuse); 12.12 MB inline → 3.05 MB deduped. But titles are the
//! WRONG key — the corpus is LLM-generated, so wording drifts between
//! regenerations while the citation identity does not. The corpus already
//! carries stable identifiers on ~104,700 occurrences: PMID (dominant), DOI,
//! ORPHA, CGGV, ClinicalTrials, URL.
//!
//! So a [`CitationKey`] is `(namespace, id)`, and the title is COLD content
//! reached by [`ContentId`]. Where no stable identifier exists,
//! [`CitationKey::ContentAddressed`] says so **explicitly** — a fake
//! bibliographic identity is never synthesised from a title.

use crate::content_store::ContentId;

/// Source `causal_link_type` — the four states, 2 bits.
///
/// Maps 1:1 onto the CE64 `CausalTopology` ordinal at hydration; the mapping is
/// source-authoritative and is never inferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DismechTopology {
    /// `DIRECT` — measured 9,073.
    Direct = 0,
    /// `INDIRECT_KNOWN_INTERMEDIATES` — measured 3,978. The hidden-mediator
    /// oracle population: the source names the intermediates, so they can be
    /// hidden and recovery measured.
    IndirectKnownIntermediates = 1,
    /// `INDIRECT_UNKNOWN_INTERMEDIATES` — measured 4,539. The genuine
    /// known-unknown population and the epistemic-restraint control: the source
    /// itself does not know the mediator, so a reasoner that "recovers" one is
    /// hallucinating closure.
    IndirectUnknownIntermediates = 2,
    /// `UNKNOWN` — measured 408. An asserted value, never a parse fallback.
    Unknown = 3,
}

impl DismechTopology {
    /// All four, ordinal order.
    pub const ALL: [Self; 4] = [
        Self::Direct,
        Self::IndirectKnownIntermediates,
        Self::IndirectUnknownIntermediates,
        Self::Unknown,
    ];

    /// Parse the source token. **Fails closed** — no folding, no default.
    #[must_use]
    pub fn from_source(s: &str) -> Option<Self> {
        Some(match s {
            "DIRECT" => Self::Direct,
            "INDIRECT_KNOWN_INTERMEDIATES" => Self::IndirectKnownIntermediates,
            "INDIRECT_UNKNOWN_INTERMEDIATES" => Self::IndirectUnknownIntermediates,
            "UNKNOWN" => Self::Unknown,
            _ => return None,
        })
    }

    /// The exact source token — round-trips [`from_source`](Self::from_source).
    #[must_use]
    pub const fn as_source(self) -> &'static str {
        match self {
            Self::Direct => "DIRECT",
            Self::IndirectKnownIntermediates => "INDIRECT_KNOWN_INTERMEDIATES",
            Self::IndirectUnknownIntermediates => "INDIRECT_UNKNOWN_INTERMEDIATES",
            Self::Unknown => "UNKNOWN",
        }
    }

    /// The 2-bit ordinal. Consumers map this onto CE64 bits 59..60; this module
    /// never does, so the source bake stays free of hot registers.
    #[must_use]
    pub const fn to_bits_2(self) -> u8 {
        self as u8
    }

    /// Inverse of [`to_bits_2`](Self::to_bits_2). Fails closed above 3.
    #[must_use]
    pub const fn from_bits_2(b: u8) -> Option<Self> {
        Some(match b {
            0 => Self::Direct,
            1 => Self::IndirectKnownIntermediates,
            2 => Self::IndirectUnknownIntermediates,
            3 => Self::Unknown,
            _ => return None,
        })
    }

    /// Does the SOURCE claim to know the intermediates?
    ///
    /// True only for [`IndirectKnownIntermediates`](Self::IndirectKnownIntermediates).
    /// This is what separates the oracle population from the restraint control:
    /// for the other three a recovered mediator is not a success, it is an
    /// unsupported claim.
    #[must_use]
    pub const fn source_knows_intermediates(self) -> bool {
        matches!(self, Self::IndirectKnownIntermediates)
    }

    /// Is the mediator slot legitimately unresolved in the SOURCE?
    ///
    /// **[`Unknown`](Self::Unknown) is deliberately NOT included**, and the
    /// distinction is epistemic rather than cosmetic:
    ///
    /// ```text
    /// IndirectUnknownIntermediates    A -> ? -> B
    ///     the path is known indirect, the mediator ROLE is established,
    ///     only its IDENTITY is missing  => SeekMediator is warranted
    ///
    /// Unknown                         A  ?  B
    ///     direct-vs-indirect is itself undecided, so the mediator role is
    ///     not established at all       => SeekMediator would MINT a schema
    ///                                     slot the source never asserted
    /// ```
    ///
    /// Conflating them lets `Unknown` dispatch a mediator search for a hole
    /// nobody claimed exists. Ask [`topology_unresolved`](Self::topology_unresolved)
    /// for that case instead.
    ///
    /// **This predicate answers what the SOURCE asserts, never whether a
    /// mediator is actually present**, and on real data those differ sharply.
    ///
    /// Measured on `monarch-initiative/dismech` (2,100 disorder files),
    /// **scoped to `pathophysiology[].downstream[]`** — the edge list that
    /// actually carries `intermediate_mechanisms`. This is a SUBSET of the
    /// corpus-wide `causal_link_type` census in the module header (which counts
    /// the field wherever it occurs), so the totals differ by scope, not by
    /// disagreement: 3,844 here against 3,978 corpus-wide.
    ///
    /// | declared | with ≥1 intermediate | with ZERO |
    /// |---|---|---|
    /// | `INDIRECT_KNOWN_INTERMEDIATES` (3,844) | 2,497 | **1,347 = 35.0%** |
    /// | `INDIRECT_UNKNOWN_INTERMEDIATES` (4,325) | 92 (contradictory) | 4,233 |
    ///
    /// So a third of the edges that CLAIM known intermediates list none, and a
    /// few that claim none list some. A consumer building the hidden-mediator
    /// oracle must filter on the ACTUAL intermediate list, never on the topology
    /// ordinal alone: the usable population is **2,497 distinct
    /// `(disease, source, target, mediators)` tuples across 539 diseases**
    /// (deduped — the raw and distinct counts coincide, so no many-to-many
    /// mention inflation), not 3,844.
    #[must_use]
    pub const fn mediator_unresolved(self) -> bool {
        matches!(self, Self::IndirectUnknownIntermediates)
    }

    /// Is the TOPOLOGY ITSELF unresolved — i.e. the source has not decided
    /// direct vs indirect, so no mediator role is established?
    ///
    /// The companion to [`mediator_unresolved`](Self::mediator_unresolved):
    /// together they partition the two genuinely-unresolved states without
    /// collapsing them. Measured 403 `UNKNOWN` against 4,325
    /// `INDIRECT_UNKNOWN_INTERMEDIATES` — a ~10.7x population difference that a
    /// merged predicate would hide.
    #[must_use]
    pub const fn topology_unresolved(self) -> bool {
        matches!(self, Self::Unknown)
    }
}

/// Source `supports` — 4 states, 2 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Supports {
    /// `SUPPORT`.
    Support = 0,
    /// `PARTIAL`.
    Partial = 1,
    /// `REFUTE`.
    Refute = 2,
    /// `NO_EVIDENCE`.
    NoEvidence = 3,
}

impl Supports {
    /// All four.
    pub const ALL: [Self; 4] = [Self::Support, Self::Partial, Self::Refute, Self::NoEvidence];

    /// Parse; fails closed.
    #[must_use]
    pub fn from_source(s: &str) -> Option<Self> {
        Some(match s {
            "SUPPORT" => Self::Support,
            "PARTIAL" => Self::Partial,
            "REFUTE" => Self::Refute,
            "NO_EVIDENCE" => Self::NoEvidence,
            _ => return None,
        })
    }

    /// The exact source token.
    #[must_use]
    pub const fn as_source(self) -> &'static str {
        match self {
            Self::Support => "SUPPORT",
            Self::Partial => "PARTIAL",
            Self::Refute => "REFUTE",
            Self::NoEvidence => "NO_EVIDENCE",
        }
    }

    /// 2-bit ordinal.
    #[must_use]
    pub const fn to_bits_2(self) -> u8 {
        self as u8
    }
}

/// Source `evidence_source` — 5 states, 3 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EvidenceSource {
    /// `HUMAN_CLINICAL`.
    HumanClinical = 0,
    /// `MODEL_ORGANISM`.
    ModelOrganism = 1,
    /// `IN_VITRO`.
    InVitro = 2,
    /// `COMPUTATIONAL`.
    Computational = 3,
    /// `OTHER`.
    Other = 4,
}

impl EvidenceSource {
    /// All five.
    pub const ALL: [Self; 5] = [
        Self::HumanClinical,
        Self::ModelOrganism,
        Self::InVitro,
        Self::Computational,
        Self::Other,
    ];

    /// Parse; fails closed. `OTHER` is an asserted value, never a fallback.
    #[must_use]
    pub fn from_source(s: &str) -> Option<Self> {
        Some(match s {
            "HUMAN_CLINICAL" => Self::HumanClinical,
            "MODEL_ORGANISM" => Self::ModelOrganism,
            "IN_VITRO" => Self::InVitro,
            "COMPUTATIONAL" => Self::Computational,
            "OTHER" => Self::Other,
            _ => return None,
        })
    }

    /// The exact source token.
    #[must_use]
    pub const fn as_source(self) -> &'static str {
        match self {
            Self::HumanClinical => "HUMAN_CLINICAL",
            Self::ModelOrganism => "MODEL_ORGANISM",
            Self::InVitro => "IN_VITRO",
            Self::Computational => "COMPUTATIONAL",
            Self::Other => "OTHER",
        }
    }

    /// 3-bit ordinal.
    #[must_use]
    pub const fn to_bits_3(self) -> u8 {
        self as u8
    }
}

/// Where a citation's identity comes from.
///
/// The namespace is part of the KEY, not a hint — `PMID:123` and `ORPHA:123`
/// are different citations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum CitationNamespace {
    /// PubMed id — dominant in the corpus.
    Pmid = 0,
    /// DOI.
    Doi = 1,
    /// Orphanet.
    Orpha = 2,
    /// ClinicalTrials.gov (`NCT…`).
    ClinicalTrials = 3,
    /// ClinGen/CGGV.
    Cggv = 4,
    /// A bare URL.
    Url = 5,
}

impl CitationNamespace {
    /// All six.
    pub const ALL: [Self; 6] = [
        Self::Pmid,
        Self::Doi,
        Self::Orpha,
        Self::ClinicalTrials,
        Self::Cggv,
        Self::Url,
    ];

    /// Parse a CURIE prefix (case-insensitive); fails closed.
    #[must_use]
    pub fn from_prefix(p: &str) -> Option<Self> {
        Some(match p.to_ascii_uppercase().as_str() {
            "PMID" => Self::Pmid,
            "DOI" => Self::Doi,
            "ORPHA" => Self::Orpha,
            "NCT" | "CLINICALTRIALS" => Self::ClinicalTrials,
            "CGGV" => Self::Cggv,
            "URL" | "HTTP" | "HTTPS" => Self::Url,
            _ => return None,
        })
    }

    /// The canonical prefix.
    #[must_use]
    pub const fn prefix(self) -> &'static str {
        match self {
            Self::Pmid => "PMID",
            Self::Doi => "DOI",
            Self::Orpha => "ORPHA",
            Self::ClinicalTrials => "NCT",
            Self::Cggv => "CGGV",
            Self::Url => "URL",
        }
    }
}

/// The stable identity of one cited source.
///
/// **Identity never derives from the title.** The corpus is LLM-generated, so
/// title wording drifts between regenerations while the citation does not —
/// pinned by `reference_identity_survives_a_title_rewording`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CitationKey {
    /// A real, stable external identifier.
    Identified {
        /// Which identifier space.
        namespace: CitationNamespace,
        /// The bare id, prefix stripped (e.g. `"25252"`).
        id: String,
    },
    /// No stable identifier exists — the content address of the citation text,
    /// **explicitly marked as such**. Never a synthesised bibliographic id.
    ContentAddressed(ContentId),
}

impl CitationKey {
    /// Parse a source `reference` value such as `"PMID:25252"`.
    ///
    /// Falls back to [`ContentAddressed`](Self::ContentAddressed) when the value
    /// carries no recognised namespace — deliberately explicit, so a consumer
    /// can always tell a real citation from a hashed string.
    ///
    /// Returns `None` for a blank or whitespace-only `raw`. **Absence is made
    /// structurally unrepresentable**, because the alternative is worse than it
    /// looks: `fnv1a(b"")` is the nonzero FNV offset basis
    /// `0xcbf2_9ce4_8422_2325`, so hashing `""` would mint a perfectly
    /// well-formed [`ContentAddressed`](Self::ContentAddressed) key meaning
    /// "no citation supplied" — indistinguishable on replay from a real
    /// content-addressed citation, and colliding every uncited row onto one
    /// shared identity. `ContentId(0)` is the reserved sentinel
    /// ([`ContentId::is_sentinel`]), and `of_str("")` does not produce it.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        let t = raw.trim();
        if t.is_empty() {
            return None;
        }
        if let Some((p, rest)) = t.split_once(':') {
            if let Some(ns) = CitationNamespace::from_prefix(p) {
                let id = if matches!(ns, CitationNamespace::Url) {
                    t.to_string()
                } else {
                    rest.trim().to_string()
                };
                if !id.is_empty() {
                    return Some(Self::Identified { namespace: ns, id });
                }
            }
        }
        Some(Self::ContentAddressed(ContentId::of_str(t)))
    }

    /// Does this key rest on a real external identifier?
    #[must_use]
    pub const fn is_identified(&self) -> bool {
        matches!(self, Self::Identified { .. })
    }
}

/// One bibliography row: a stable key, with the title held as COLD content.
///
/// The title is never stored inline — 131,904 occurrences over 31,361 distinct
/// titles measured, so inlining costs 12.12 MB to say 3.05 MB.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BibliographyRecord {
    /// Stable identity.
    pub key: CitationKey,
    /// CAM handle for the title text, or the reserved sentinel `ContentId(0)`
    /// (see [`ContentId::is_sentinel`]) when no title was supplied.
    ///
    /// The sentinel must be written explicitly: `of_str("")` hashes the empty
    /// byte string to the nonzero FNV offset basis, so an untitled record would
    /// otherwise carry a real-looking content address that no store can resolve.
    pub title: ContentId,
}

impl BibliographyRecord {
    /// Build from a raw `reference` and its (possibly drifting) title.
    ///
    /// `None` when `reference` is blank — a bibliography row with no citation
    /// is not a row (see [`CitationKey::parse`]). A blank `title` is legal and
    /// lands on the `ContentId(0)` sentinel rather than on `fnv1a(b"")`.
    #[must_use]
    pub fn new(reference: &str, title: &str) -> Option<Self> {
        Some(Self {
            key: CitationKey::parse(reference)?,
            title: if title.trim().is_empty() {
                ContentId(0)
            } else {
                ContentId::of_str(title)
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every source token round-trips, and the ordinal is stable.
    #[test]
    fn topology_round_trips_every_source_value() {
        for t in DismechTopology::ALL {
            assert_eq!(DismechTopology::from_source(t.as_source()), Some(t));
            assert_eq!(DismechTopology::from_bits_2(t.to_bits_2()), Some(t));
        }
        // anti-vacuity: the four are genuinely distinct ordinals
        let mut bits: Vec<u8> = DismechTopology::ALL.iter().map(|t| t.to_bits_2()).collect();
        bits.sort_unstable();
        bits.dedup();
        assert_eq!(bits.len(), 4, "topology ordinals collided");
    }

    /// FAIL CLOSED — and specifically, never mint `Unknown`, which the corpus
    /// asserts 408 times and which would forge an assertion.
    #[test]
    fn unrecognised_topology_fails_closed_and_never_becomes_unknown() {
        for bad in [
            "",
            "direct",
            "INDIRECT",
            "INDIRECT_KNOWN",
            "UNKNOWN_INTERMEDIATES",
            "unknown",
        ] {
            assert_eq!(
                DismechTopology::from_source(bad),
                None,
                "{bad:?} must not parse"
            );
        }
        assert_eq!(DismechTopology::from_bits_2(4), None);
        // the silence half: the real token still parses
        assert_eq!(
            DismechTopology::from_source("UNKNOWN"),
            Some(DismechTopology::Unknown)
        );
    }

    /// The two experimental populations are distinguishable, which is what makes
    /// the hidden-mediator test and its restraint control separable.
    #[test]
    fn only_indirect_known_claims_the_source_knows_intermediates() {
        assert!(DismechTopology::IndirectKnownIntermediates.source_knows_intermediates());
        for t in [
            DismechTopology::Direct,
            DismechTopology::IndirectUnknownIntermediates,
            DismechTopology::Unknown,
        ] {
            assert!(!t.source_knows_intermediates(), "{t:?} must not claim it");
        }
        assert!(DismechTopology::IndirectUnknownIntermediates.mediator_unresolved());
        assert!(!DismechTopology::Direct.mediator_unresolved());
    }

    #[test]
    fn supports_and_evidence_source_round_trip_every_value() {
        for s in Supports::ALL {
            assert_eq!(Supports::from_source(s.as_source()), Some(s));
            assert!(s.to_bits_2() < 4);
        }
        for e in EvidenceSource::ALL {
            assert_eq!(EvidenceSource::from_source(e.as_source()), Some(e));
            assert!(e.to_bits_3() < 8);
        }
        assert_eq!(Supports::from_source("MAYBE"), None);
        assert_eq!(EvidenceSource::from_source("HUMAN"), None);
    }

    /// THE LLM-DRIFT FALSIFIER: identity is the citation, not the wording.
    #[test]
    fn reference_identity_survives_a_title_rewording() {
        let a = BibliographyRecord::new(
            "PMID:25252",
            "Insulin resistance in type 2 diabetes mellitus",
        )
        .expect("a real citation");
        let b = BibliographyRecord::new(
            "PMID:25252",
            "Insulin Resistance in Type 2 Diabetes Mellitus: A Review",
        )
        .expect("a real citation");
        assert_eq!(a.key, b.key, "citation identity must survive a re-wording");
        assert_ne!(a.title, b.title, "the titles genuinely differ");
        // …and a different citation is a different key even with the same title
        let c = BibliographyRecord::new(
            "PMID:99999",
            "Insulin resistance in type 2 diabetes mellitus",
        )
        .expect("a real citation");
        assert_ne!(a.key, c.key);
        assert_eq!(a.title, c.title);
    }

    /// The namespace is part of the key — same digits, different source.
    #[test]
    fn namespace_is_part_of_the_identity() {
        assert_ne!(
            CitationKey::parse("PMID:123").unwrap(),
            CitationKey::parse("ORPHA:123").unwrap()
        );
        for ns in CitationNamespace::ALL {
            assert_eq!(CitationNamespace::from_prefix(ns.prefix()), Some(ns));
        }
        assert_eq!(CitationNamespace::from_prefix("PUBMED"), None);
    }

    /// An unidentified citation says so, rather than pretending to a
    /// bibliographic id synthesised from its title.
    #[test]
    fn unidentified_citations_are_explicitly_content_addressed() {
        let k = CitationKey::parse("Smith et al, personal communication")
            .expect("non-blank input parses");
        assert!(
            !k.is_identified(),
            "must not claim a bibliographic identity"
        );
        assert!(matches!(k, CitationKey::ContentAddressed(_)));
        // deterministic
        assert_eq!(
            k,
            CitationKey::parse("Smith et al, personal communication").unwrap()
        );
        assert!(CitationKey::parse("PMID:25252").unwrap().is_identified());
    }
    /// P1 FALSIFIER — `Unknown` must NOT dispatch a mediator search.
    ///
    /// `IndirectUnknownIntermediates` establishes the mediator ROLE and leaves
    /// its identity open; `Unknown` leaves direct-vs-indirect itself undecided,
    /// so treating it as an unresolved mediator MINTS a schema slot the source
    /// never asserted. Two-sided, and the silence half is the point.
    #[test]
    fn unknown_topology_is_not_an_unresolved_mediator() {
        assert!(DismechTopology::IndirectUnknownIntermediates.mediator_unresolved());
        assert!(
            !DismechTopology::Unknown.mediator_unresolved(),
            "Unknown must not claim a mediator hole the source never established"
        );
        // …and the companion predicate covers it instead, exclusively.
        assert!(DismechTopology::Unknown.topology_unresolved());
        assert!(!DismechTopology::IndirectUnknownIntermediates.topology_unresolved());
        // the two resolved states answer NEITHER
        for t in [
            DismechTopology::Direct,
            DismechTopology::IndirectKnownIntermediates,
        ] {
            assert!(!t.mediator_unresolved(), "{t:?} is not mediator-unresolved");
            assert!(!t.topology_unresolved(), "{t:?} is not topology-unresolved");
        }
        // anti-vacuity: the two predicates are disjoint and neither is constant
        let any_med = DismechTopology::ALL
            .iter()
            .filter(|t| t.mediator_unresolved())
            .count();
        let any_top = DismechTopology::ALL
            .iter()
            .filter(|t| t.topology_unresolved())
            .count();
        assert_eq!(
            (any_med, any_top),
            (1, 1),
            "each predicate names exactly one state"
        );
    }

    /// P1 FALSIFIER — absence must not become a well-formed content address.
    ///
    /// `fnv1a(b"")` is the nonzero FNV offset basis, so hashing a blank would
    /// mint a real-looking citation meaning "none supplied" and collide every
    /// uncited row onto it.
    #[test]
    fn a_blank_citation_is_absent_not_content_addressed() {
        for blank in ["", "   ", "\t\n "] {
            assert_eq!(
                CitationKey::parse(blank),
                None,
                "blank {blank:?} must not parse to a citation"
            );
            assert_eq!(BibliographyRecord::new(blank, "a title"), None);
        }
        // the trap this guards, stated as a measurement rather than a memory:
        assert_ne!(
            ContentId::of_str("").0,
            0,
            "of_str(\"\") is the FNV offset basis, NOT the sentinel"
        );
        // a blank TITLE is legal, and lands on the sentinel rather than the basis
        let r = BibliographyRecord::new("PMID:25252", "   ").expect("real citation");
        assert!(
            r.title.is_sentinel(),
            "an untitled record uses ContentId(0)"
        );
        assert_ne!(r.title, ContentId::of_str(""), "must not be the empty hash");
        // anti-vacuity: a real title is NOT the sentinel
        let t = BibliographyRecord::new("PMID:25252", "A real title").expect("real citation");
        assert!(!t.title.is_sentinel());
    }
}
