// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `epistemic_state5` — the canonical reading of `CausalEdge64` bits 59..63
//! (`D-EPI-CANON-0`, operator decision 2026-10-07).
//!
//! # One field, one reading
//!
//! Bits 59..63 are ONE 5-bit field: an `EpistemicState5` code, a dense
//! codebook over valid epistemic conjunctions (grounding × assertion). The
//! code is semantically opaque on its own — **raw ordinal + declared reading
//! = meaning** — so every read goes through a declaration keyed by
//! `(classid, rail)` and a codebook generation:
//!
//! ```text
//! (classid, rail) → Epi5Reading { generation }
//!                 → project_state5(raw5, provenance) → EpistemicState5 | refusal
//! ```
//!
//! The physical read is `CausalEdge64::epistemic_raw5()` (equal to
//! `(spare() << 2) | truth_raw()`), the physical write
//! `CausalEdge64::with_epistemic_raw5(state.raw())`. This crate is zero-dep and
//! never touches the edge layout; it takes and returns raw ordinals, like
//! [`band_reading`](crate::band_reading).
//!
//! # The codebook is also the affordance law's input
//!
//! A code's [facts](fact) decide which statistical/microcode recipes are legal
//! on the edge (`CausalEdge64 × RecipeLaw[64] → EligibleRecipes: u64`, the
//! D-GSO-AFF-0 measurement). For example `25 = Indirect × IntermediateUnknown ×
//! Related` admits hydration of the intermediate but never a known-intermediate
//! mechanism fold; `30 = Indirect × IntermediateKnown × Supports` admits the
//! mechanism fold and causal identification. Eligibility is transient
//! measurement; nothing here stores a capability.
//!
//! # Supersedes the split reading
//!
//! [`band_reading`](crate::band_reading) declared 59..60 (`TruthLens`) and
//! 61..63 (`BandPresence`) as two fields. That split no longer owns these bits.
//! The P7a certification band (#1369), which wrote 61..63 alone, survives as a
//! [legacy](legacy) translation into this codebook; the historical
//! `ReasoningBand` and `TrustTexture` readings have **no** declared
//! translation and refuse (their producers never declared conjunction
//! semantics, and the bits cannot say which reading wrote them).
//!
//! # Refusals
//!
//! [`Epi5Declarations::project_state5`] refuses, never guesses: undeclared
//! `(classid, rail)`, a generation other than the one requested, untrusted
//! provenance (`V1Legacy` / `Unknown` — on a v1 row these bits are old
//! `temporal`), and a code the generation does not declare.

use crate::band_reading::EdgeProvenance;
use crate::class_view::ClassId;
use crate::rail_geometry::RailAxis;

/// Number of raw codes the 5-bit field can carry.
pub const CODES: usize = 32;
const _: () = assert!(CODES == 1 << 5);

/// A conjunction of epistemic facts — the meaning of one declared code.
pub type Facts = u32;

/// The fact vocabulary. A code's meaning is a conjunction of these; the
/// affordance law states each recipe's obligations in the same vocabulary.
pub mod fact {
    use super::Facts;

    /// The relation is grounded directly (no intermediate).
    pub const DIRECT: Facts = 1 << 0;
    /// The relation runs through at least one intermediate.
    pub const INDIRECT: Facts = 1 << 1;
    /// An intermediate is part of the claim.
    pub const INTERMEDIATE_PRESENT: Facts = 1 << 2;
    /// The intermediate is not known (hydration is the open work).
    pub const INTERMEDIATE_UNKNOWN: Facts = 1 << 3;
    /// The intermediate is known (a mechanism can be folded/tested).
    pub const INTERMEDIATE_KNOWN: Facts = 1 << 4;
    /// The relation was observed on units, not only asserted.
    pub const OBSERVED: Facts = 1 << 5;
    /// Associated in a declared population.
    pub const ASSOCIATED: Facts = 1 << 6;
    /// Related: the association survives the robustness mask.
    pub const RELATED: Facts = 1 << 7;
    /// Supports / contributes in every declared stratum.
    pub const SUPPORTS: Facts = 1 << 8;
    /// Certified causes (intervention-backed).
    pub const CAUSES: Facts = 1 << 9;

    /// `Indirect × IntermediateUnknown`.
    pub const IND_UNKNOWN: Facts = INDIRECT | INTERMEDIATE_PRESENT | INTERMEDIATE_UNKNOWN;
    /// `Indirect × IntermediateKnown`.
    pub const IND_KNOWN: Facts = INDIRECT | INTERMEDIATE_PRESENT | INTERMEDIATE_KNOWN;
    /// `Associated × Related`.
    pub const UP_TO_RELATED: Facts = ASSOCIATED | RELATED;
    /// `Associated × Related × Supports`.
    pub const UP_TO_SUPPORTS: Facts = UP_TO_RELATED | SUPPORTS;
}

/// The V1 codebook: code → facts, `None` = not a code of this generation.
///
/// Adopted unchanged from the D-GSO-AFF-0 measurement (#1370). Deliberately
/// not ordered by strength (`Causes` sits at 1, `Open` at 0, `Related` at 20):
/// the codebook is dense over conjunctions, never an ordinal scale.
pub const CODEBOOK_V1: [Option<Facts>; CODES] = {
    use fact::*;
    let mut t = [None; CODES];
    t[0] = Some(0); // Open
    t[3] = Some(DIRECT | OBSERVED);
    t[7] = Some(DIRECT | OBSERVED | ASSOCIATED);
    t[12] = Some(DIRECT | ASSOCIATED);
    t[20] = Some(DIRECT | UP_TO_RELATED);
    t[25] = Some(IND_UNKNOWN | UP_TO_RELATED);
    t[5] = Some(IND_KNOWN | UP_TO_RELATED);
    t[30] = Some(IND_KNOWN | UP_TO_SUPPORTS);
    t[1] = Some(DIRECT | UP_TO_SUPPORTS | CAUSES);
    t[17] = Some(IND_KNOWN | UP_TO_SUPPORTS | CAUSES);
    t
};

/// Named V1 codes. The facts are the content; the names are pointers.
pub mod code_v1 {
    /// No fact asserted. Also what `pack_v2` / `compose` leave behind.
    pub const OPEN: u8 = 0;
    /// `Direct × Supports × Causes`.
    pub const DIRECT_CAUSES: u8 = 1;
    /// `Direct × Observed`.
    pub const DIRECT_OBSERVED: u8 = 3;
    /// `Indirect × IntermediateKnown × Related`.
    pub const IND_KNOWN_RELATED: u8 = 5;
    /// `Direct × Observed × Associated`.
    pub const DIRECT_OBSERVED_ASSOCIATED: u8 = 7;
    /// `Direct × Associated`.
    pub const DIRECT_ASSOCIATED: u8 = 12;
    /// `Indirect × IntermediateKnown × Supports × Causes`.
    pub const IND_KNOWN_CAUSES: u8 = 17;
    /// `Direct × Related`.
    pub const DIRECT_RELATED: u8 = 20;
    /// `Indirect × IntermediateUnknown × Related`.
    pub const IND_UNKNOWN_RELATED: u8 = 25;
    /// `Indirect × IntermediateKnown × Supports`.
    pub const IND_KNOWN_SUPPORTS: u8 = 30;
}

/// A codebook generation. A generation names the whole declared reading;
/// changing any code's facts is a new generation, never an edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Epi5Gen {
    /// [`CODEBOOK_V1`].
    #[default]
    V1,
}

impl Epi5Gen {
    /// The generation's codebook — the ONE authoritative code → facts table.
    #[must_use]
    pub const fn codebook(self) -> &'static [Option<Facts>; CODES] {
        match self {
            Epi5Gen::V1 => &CODEBOOK_V1,
        }
    }
}

/// A code validated under a generation. Only obtainable through
/// [`EpistemicState5::decode`] / [`Epi5Declarations::project_state5`] /
/// a [legacy] translation, so holding one means the code is declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpistemicState5 {
    code: u8,
    generation: Epi5Gen,
}

impl EpistemicState5 {
    /// Validate `raw5` against `generation`'s codebook. No provenance or
    /// class check: use [`Epi5Declarations::project_state5`] to read an edge.
    /// This is the producer-side constructor.
    pub const fn decode(generation: Epi5Gen, raw5: u8) -> Result<Self, Epi5ReadError> {
        if raw5 as usize >= CODES || generation.codebook()[raw5 as usize].is_none() {
            return Err(Epi5ReadError::UndeclaredCode(raw5));
        }
        Ok(Self {
            code: raw5,
            generation,
        })
    }

    /// The raw 5-bit code, for `CausalEdge64::with_epistemic_raw5`.
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.code
    }

    /// The generation the code was validated under.
    #[must_use]
    pub const fn generation(self) -> Epi5Gen {
        self.generation
    }

    /// The code's facts.
    #[must_use]
    pub const fn facts(self) -> Facts {
        match self.generation.codebook()[self.code as usize] {
            Some(f) => f,
            // Unreachable: construction validated the code.
            None => 0,
        }
    }

    /// Does the state assert every fact in `required`?
    #[must_use]
    pub const fn asserts(self, required: Facts) -> bool {
        self.facts() & required == required
    }
}

/// A class's declaration for bits 59..63.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Epi5Reading {
    /// The codebook generation the class's producers write.
    pub generation: Epi5Gen,
}

/// Why a projection refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Epi5ReadError {
    /// No declaration for this `(classid, rail)`.
    UndeclaredClass(ClassId),
    /// The class declares a different generation than the reader understands.
    GenerationMismatch {
        /// What the class's producers write.
        declared: Epi5Gen,
        /// What the reader asked for.
        requested: Epi5Gen,
    },
    /// `V1Legacy` or `Unknown` provenance: the bits may be old `temporal`.
    UnknownProvenance(EdgeProvenance),
    /// The code is not declared by the generation.
    UndeclaredCode(u8),
}

/// `(classid, rail) → Epi5Reading`, caller-populated (an OGAR mint / bake
/// decision populates it; this crate never pre-fills a class).
#[derive(Debug, Clone, Default)]
pub struct Epi5Declarations {
    entries: Vec<((ClassId, RailAxis), Epi5Reading)>,
}

impl Epi5Declarations {
    /// An empty table.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Declare (or re-declare) a class's reading. Returns `true` if this
    /// replaced an existing declaration.
    pub fn declare(&mut self, class: ClassId, rail: RailAxis, reading: Epi5Reading) -> bool {
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find(|((c, r), _)| *c == class && *r == rail)
        {
            slot.1 = reading;
            true
        } else {
            self.entries.push(((class, rail), reading));
            false
        }
    }

    /// The audit read: `None` = never declared.
    #[must_use]
    pub fn get(&self, class: ClassId, rail: RailAxis) -> Option<Epi5Reading> {
        self.entries
            .iter()
            .find(|((c, r), _)| *c == class && *r == rail)
            .map(|(_, b)| *b)
    }

    /// The canonical projection of bits 59..63. Check order: declaration,
    /// provenance, generation, code — untrusted bits fail before their
    /// meaning is asked.
    pub fn project_state5(
        &self,
        class: ClassId,
        rail: RailAxis,
        requested: Epi5Gen,
        raw5: u8,
        provenance: EdgeProvenance,
    ) -> Result<EpistemicState5, Epi5ReadError> {
        let reading = self
            .get(class, rail)
            .ok_or(Epi5ReadError::UndeclaredClass(class))?;
        if !provenance.trusted() {
            return Err(Epi5ReadError::UnknownProvenance(provenance));
        }
        if reading.generation != requested {
            return Err(Epi5ReadError::GenerationMismatch {
                declared: reading.generation,
                requested,
            });
        }
        EpistemicState5::decode(requested, raw5)
    }
}

/// Legacy compatibility: the split readings, as translations INTO and
/// projections OUT OF the canonical codebook. Transitional — new code reads
/// [`Epi5Declarations::project_state5`] and writes a declared state.
///
/// Only one legacy producer has declared conjunction semantics: P7a
/// certification (#1369) under a legacy `CausalTopology` grounding. Its
/// translation is an explicit table; a cell without a canonical code refuses.
/// A translation never manufactures a fact: `Causes` comes only from a
/// certified `Causes`.
pub mod legacy {
    use super::{code_v1, fact, Epi5Gen, EpistemicState5};

    /// The legacy 2-bit grounding reading. Ordinals mirror
    /// `causal_edge::layout::CausalTopology` (doc pointer, never an import).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum LegacyTopology {
        /// `CausalTopology::Direct` (0).
        Direct,
        /// `CausalTopology::IndirectKnownIntermediates` (1).
        IndirectKnown,
        /// `CausalTopology::IndirectUnknownIntermediates` (2).
        IndirectUnknown,
        /// `CausalTopology::Unknown` (3): grounding not known.
        Unknown,
    }

    impl LegacyTopology {
        /// All four.
        pub const ALL: [LegacyTopology; 4] = [
            LegacyTopology::Direct,
            LegacyTopology::IndirectKnown,
            LegacyTopology::IndirectUnknown,
            LegacyTopology::Unknown,
        ];

        /// The legacy ordinal.
        #[must_use]
        pub const fn ordinal(self) -> u8 {
            self as u8
        }

        /// From the legacy ordinal (`0..=3`); `None` above.
        #[must_use]
        pub const fn from_ordinal(o: u8) -> Option<Self> {
            match o {
                0 => Some(LegacyTopology::Direct),
                1 => Some(LegacyTopology::IndirectKnown),
                2 => Some(LegacyTopology::IndirectUnknown),
                3 => Some(LegacyTopology::Unknown),
                _ => None,
            }
        }
    }

    /// The P7a certification contract (#1369) that used to own bits 61..63.
    /// Codes 6 and 7 were reserved there.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum LegacyCertification {
        /// No obligation met.
        Open,
        /// Associated in a declared population, ≥ 2 sources.
        Associated,
        /// Associated and robust under the mask.
        Related,
        /// Does not lower Y in any stratum, raises it in one (= `SUPPORTS`).
        Contributes,
        /// `Contributes` plus ordering. No canonical fact yet.
        CausalCandidate,
        /// Intervention-certified.
        Causes,
    }

    impl LegacyCertification {
        /// All six, in P7a's ordinal order.
        pub const ALL: [LegacyCertification; 6] = [
            LegacyCertification::Open,
            LegacyCertification::Associated,
            LegacyCertification::Related,
            LegacyCertification::Contributes,
            LegacyCertification::CausalCandidate,
            LegacyCertification::Causes,
        ];

        /// P7a's 3-bit code.
        #[must_use]
        pub const fn code(self) -> u8 {
            self as u8
        }

        /// From P7a's code; 6 and 7 are reserved and refuse.
        pub const fn from_code(raw: u8) -> Result<Self, LegacyError> {
            Ok(match raw {
                0 => LegacyCertification::Open,
                1 => LegacyCertification::Associated,
                2 => LegacyCertification::Related,
                3 => LegacyCertification::Contributes,
                4 => LegacyCertification::CausalCandidate,
                5 => LegacyCertification::Causes,
                other => return Err(LegacyError::ReservedCertification(other)),
            })
        }
    }

    /// Why a legacy translation refused.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum LegacyError {
        /// The legacy pair has no canonical conjunction in this generation.
        NoCanonicalState {
            /// The declared grounding.
            topology: LegacyTopology,
            /// The declared certification.
            certification: LegacyCertification,
        },
        /// P7a's reserved codes 6 / 7.
        ReservedCertification(u8),
        /// The state asserts no grounding, so a grounding-relative update
        /// has nothing to keep.
        NoGrounding,
    }

    /// Translate a declared legacy pair into the canonical code. The ONE
    /// translation table; every cell not listed refuses.
    pub const fn translate(
        generation: Epi5Gen,
        topology: LegacyTopology,
        certification: LegacyCertification,
    ) -> Result<EpistemicState5, LegacyError> {
        use LegacyCertification as C;
        use LegacyTopology as T;
        let code = match generation {
            Epi5Gen::V1 => match (topology, certification) {
                (_, C::Open) => Some(code_v1::OPEN),
                (T::Direct, C::Associated) => Some(code_v1::DIRECT_ASSOCIATED),
                (T::Direct, C::Related) => Some(code_v1::DIRECT_RELATED),
                (T::Direct, C::Causes) => Some(code_v1::DIRECT_CAUSES),
                (T::IndirectKnown, C::Related) => Some(code_v1::IND_KNOWN_RELATED),
                (T::IndirectKnown, C::Contributes) => Some(code_v1::IND_KNOWN_SUPPORTS),
                (T::IndirectKnown, C::Causes) => Some(code_v1::IND_KNOWN_CAUSES),
                (T::IndirectUnknown, C::Related) => Some(code_v1::IND_UNKNOWN_RELATED),
                _ => None,
            },
        };
        match code {
            Some(c) => match EpistemicState5::decode(generation, c) {
                Ok(s) => Ok(s),
                Err(_) => Err(LegacyError::NoCanonicalState {
                    topology,
                    certification,
                }),
            },
            None => Err(LegacyError::NoCanonicalState {
                topology,
                certification,
            }),
        }
    }

    /// Project a canonical state onto the legacy grounding. `None` if the
    /// state asserts none (`Open`). Observed states project to `Direct`
    /// (lossy: `Observed` has no legacy spelling).
    #[must_use]
    pub const fn project_topology(state: EpistemicState5) -> Option<LegacyTopology> {
        let f = state.facts();
        if f & fact::DIRECT != 0 {
            Some(LegacyTopology::Direct)
        } else if f & fact::INTERMEDIATE_KNOWN != 0 {
            Some(LegacyTopology::IndirectKnown)
        } else if f & fact::INTERMEDIATE_UNKNOWN != 0 {
            Some(LegacyTopology::IndirectUnknown)
        } else {
            None
        }
    }

    /// Project a canonical state onto the strongest legacy certification it
    /// asserts.
    #[must_use]
    pub const fn project_certification(state: EpistemicState5) -> LegacyCertification {
        let f = state.facts();
        if f & fact::CAUSES != 0 {
            LegacyCertification::Causes
        } else if f & fact::SUPPORTS != 0 {
            LegacyCertification::Contributes
        } else if f & fact::RELATED != 0 {
            LegacyCertification::Related
        } else if f & fact::ASSOCIATED != 0 {
            LegacyCertification::Associated
        } else {
            LegacyCertification::Open
        }
    }

    /// Compatibility writer for a legacy certification update: keep the
    /// state's grounding, replace the certification, resolve to a canonical
    /// code — or refuse. Never a 3-bit overwrite.
    pub const fn recertify(
        state: EpistemicState5,
        certification: LegacyCertification,
    ) -> Result<EpistemicState5, LegacyError> {
        match project_topology(state) {
            Some(t) => translate(state.generation(), t, certification),
            None => match certification {
                LegacyCertification::Open => Ok(state),
                _ => Err(LegacyError::NoGrounding),
            },
        }
    }

    /// Compatibility writer for a legacy topology update: keep the state's
    /// certification, replace the grounding, resolve — or refuse. Never a
    /// 2-bit overwrite.
    pub const fn retopologize(
        state: EpistemicState5,
        topology: LegacyTopology,
    ) -> Result<EpistemicState5, LegacyError> {
        translate(state.generation(), topology, project_certification(state))
    }
}

#[cfg(test)]
mod tests {
    use super::legacy::*;
    use super::*;

    const CLASS: ClassId = 0x0902;
    const RAIL: RailAxis = RailAxis::Taxonomy;

    fn decl() -> Epi5Declarations {
        let mut d = Epi5Declarations::new();
        d.declare(CLASS, RAIL, Epi5Reading::default());
        d
    }

    #[test]
    fn v1_declares_exactly_the_ten_measured_codes() {
        let declared: Vec<u8> = (0..32u8)
            .filter(|c| CODEBOOK_V1[*c as usize].is_some())
            .collect();
        assert_eq!(declared, vec![0, 1, 3, 5, 7, 12, 17, 20, 25, 30]);
    }

    /// The four operator examples, pinned by facts.
    #[test]
    fn the_operator_examples_mean_what_they_say() {
        use fact::*;
        let f = |c: u8| EpistemicState5::decode(Epi5Gen::V1, c).unwrap().facts();
        assert_eq!(f(25), IND_UNKNOWN | ASSOCIATED | RELATED);
        assert_eq!(f(30), IND_KNOWN | ASSOCIATED | RELATED | SUPPORTS);
        assert_eq!(f(7), DIRECT | OBSERVED | ASSOCIATED);
        assert_eq!(f(17), IND_KNOWN | ASSOCIATED | RELATED | SUPPORTS | CAUSES);
    }

    /// F1 (contract half): every declared code projects to itself.
    #[test]
    fn projection_round_trips_every_declared_code() {
        let d = decl();
        for c in (0..32u8).filter(|c| CODEBOOK_V1[*c as usize].is_some()) {
            let s = d
                .project_state5(CLASS, RAIL, Epi5Gen::V1, c, EdgeProvenance::V2Stamped)
                .unwrap();
            assert_eq!(s.raw(), c);
            assert_eq!(s.generation(), Epi5Gen::V1);
        }
    }

    #[test]
    fn projection_refuses_every_failure_and_admits_the_good_case() {
        let d = decl();
        let ok = d.project_state5(CLASS, RAIL, Epi5Gen::V1, 25, EdgeProvenance::V3Register);
        assert!(ok.is_ok());
        assert_eq!(
            d.project_state5(0x0999, RAIL, Epi5Gen::V1, 25, EdgeProvenance::V2Stamped),
            Err(Epi5ReadError::UndeclaredClass(0x0999))
        );
        for p in [EdgeProvenance::V1Legacy, EdgeProvenance::Unknown] {
            assert_eq!(
                d.project_state5(CLASS, RAIL, Epi5Gen::V1, 25, p),
                Err(Epi5ReadError::UnknownProvenance(p))
            );
        }
        assert_eq!(
            d.project_state5(CLASS, RAIL, Epi5Gen::V1, 2, EdgeProvenance::V2Stamped),
            Err(Epi5ReadError::UndeclaredCode(2))
        );
        assert_eq!(
            EpistemicState5::decode(Epi5Gen::V1, 32),
            Err(Epi5ReadError::UndeclaredCode(32))
        );
    }

    /// Every legacy cell that translates comes back through the legacy
    /// projections unchanged (Open carries no grounding).
    #[test]
    fn legacy_translation_round_trips_through_the_projections() {
        let mut mapped = 0;
        for t in LegacyTopology::ALL {
            for c in LegacyCertification::ALL {
                if let Ok(s) = translate(Epi5Gen::V1, t, c) {
                    mapped += 1;
                    assert_eq!(project_certification(s), c, "{t:?} {c:?}");
                    if c != LegacyCertification::Open {
                        assert_eq!(project_topology(s), Some(t), "{t:?} {c:?}");
                    }
                }
            }
        }
        // 4 Open cells + 7 grounded cells.
        assert_eq!(mapped, 11);
    }

    /// F3: a certified Causes is canonical Causes, never Related.
    #[test]
    fn legacy_causes_translates_to_a_causes_code() {
        for t in [LegacyTopology::Direct, LegacyTopology::IndirectKnown] {
            let s = translate(Epi5Gen::V1, t, LegacyCertification::Causes).unwrap();
            assert!(s.asserts(fact::CAUSES));
            assert!(!matches!(s.raw(), code_v1::DIRECT_RELATED));
        }
        assert_eq!(
            translate(
                Epi5Gen::V1,
                LegacyTopology::IndirectKnown,
                LegacyCertification::Causes
            )
            .unwrap()
            .raw(),
            17
        );
    }

    /// F4: an undeclared legacy combination refuses.
    #[test]
    fn undeclared_legacy_combinations_refuse() {
        for (t, c) in [
            (LegacyTopology::Unknown, LegacyCertification::Causes),
            (LegacyTopology::IndirectUnknown, LegacyCertification::Causes),
            (LegacyTopology::Direct, LegacyCertification::Contributes),
            (LegacyTopology::Direct, LegacyCertification::CausalCandidate),
            (
                LegacyTopology::IndirectKnown,
                LegacyCertification::Associated,
            ),
        ] {
            assert_eq!(
                translate(Epi5Gen::V1, t, c),
                Err(LegacyError::NoCanonicalState {
                    topology: t,
                    certification: c
                })
            );
        }
        assert_eq!(
            LegacyCertification::from_code(6),
            Err(LegacyError::ReservedCertification(6))
        );
    }

    /// F2 + F9: the compatibility writers resolve a whole state or refuse,
    /// where the raw half-writes would land on another valid code (1 → 3,
    /// 12 → 20).
    #[test]
    fn compatibility_writers_never_produce_an_accidental_code() {
        let causes = EpistemicState5::decode(Epi5Gen::V1, code_v1::DIRECT_CAUSES).unwrap();
        // Raw: topology half := 3 gives code 3 (Direct × Observed). Wrapper:
        assert_eq!(
            retopologize(causes, LegacyTopology::Unknown),
            Err(LegacyError::NoCanonicalState {
                topology: LegacyTopology::Unknown,
                certification: LegacyCertification::Causes
            })
        );
        let assoc = EpistemicState5::decode(Epi5Gen::V1, code_v1::DIRECT_ASSOCIATED).unwrap();
        // Raw: band half := 5 gives code 20 = Related, a claim nobody made.
        // The wrapper with the legacy meaning of 5 (Causes) resolves to 1.
        assert_eq!(
            recertify(assoc, LegacyCertification::Causes).unwrap().raw(),
            code_v1::DIRECT_CAUSES
        );
        // A legal grounding move keeps the certification.
        let rel = EpistemicState5::decode(Epi5Gen::V1, code_v1::DIRECT_RELATED).unwrap();
        assert_eq!(
            retopologize(rel, LegacyTopology::IndirectUnknown)
                .unwrap()
                .raw(),
            code_v1::IND_UNKNOWN_RELATED
        );
        // Open has no grounding to keep.
        let open = EpistemicState5::decode(Epi5Gen::V1, 0).unwrap();
        assert_eq!(
            recertify(open, LegacyCertification::Related),
            Err(LegacyError::NoGrounding)
        );
        assert_eq!(recertify(open, LegacyCertification::Open), Ok(open));
    }
}
