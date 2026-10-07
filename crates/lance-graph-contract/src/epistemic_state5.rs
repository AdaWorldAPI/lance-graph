// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `epistemic_state5` — the canonical reading of `CausalEdge64` bits 59..63
//! (`D-EPI-CANON-0`, operator decision 2026-10-07).
//!
//! # One field, two coordinates
//!
//! Bits 59..63 are ONE 5-bit field, `EpistemicState5`, and it is a dense
//! PRODUCT space, not a hand-assigned codebook:
//!
//! ```text
//! bits 59..60 = Topology2       00 Direct  01 IndirectKnown  10 IndirectUnknown  11 Unknown
//! bits 61..63 = Certification3  0 Open  1 Associated  2 Related  3 Supports  4 CausalCandidate  5 Causes  6,7 reserved
//!
//! raw5 = topology | (certification << 2)        4 × 6 = 24 meaningful states, 8 reserved
//! ```
//!
//! Both ordinals are the ones already on the wire: `Topology2` is
//! `causal_edge::layout::CausalTopology` ordinal for ordinal, and
//! `Certification3` is the P7a certification contract (#1369) code for code.
//! The two coordinates are orthogonal on purpose: `IndirectUnknown × Causes`
//! (a randomized intervention certified the relation, the mechanism is still
//! open) and `Unknown × Causes` (causality known, topology unresolved) are
//! meaningful states. Whether a *next step* is legal is the affordance law's
//! question, never this layout's — validity is only `certification < 6`.
//!
//! # Layout decides coordinates, law decides affordances
//!
//! A state's [facts](fact) are `TOPOLOGY_FACTS[t] | CERTIFICATION_FACTS[c]`.
//! The affordance law (`CausalEdge64 × RecipeLaw[64] → EligibleRecipes: u64`)
//! compiles from those factors; a `[u64; 32]` table is a hot-path artifact
//! DERIVED from them ([`COMPILED_FACTS_V1`]), never semantic authority.
//! Eligibility is transient measurement; nothing here stores a capability.
//!
//! Observation is NOT a coordinate. That a transition was earned by
//! observation lives in the evidence path (witness, `SupportLedger`,
//! revision input); "observed vs asserted" belongs to witness/provenance,
//! never to one of these 32 cells.
//!
//! # Declared, never inferred
//!
//! The code is opaque without a declaration — **raw ordinal + declared reading
//! = meaning** — so every read goes through `(classid, rail) → Epi5Reading`
//! and [`Epi5Declarations::project_state5`], which refuses an undeclared
//! class, a generation mismatch, untrusted provenance (`V1Legacy` /
//! `Unknown`: on a v1 row these bits are old `temporal`) and a reserved
//! certification. The physical read/write is `CausalEdge64::epistemic_raw5` /
//! `with_epistemic_raw5(state.raw())`; this crate is zero-dep and never touches
//! the edge layout.
//!
//! # Legacy readings (compatibility only)
//!
//! [`band_reading`](crate::band_reading) declared 59..60 and 61..63 as two
//! fields. A producer that declared **topology + P7a certification** wrote
//! exactly this product (identity packing, no translation table). A producer
//! that declared the `TrustTexture` lens on 59..60 or the historical
//! `ReasoningBand` on 61..63 wrote OTHER meanings into the same bits; no
//! explicit mapping exists for either, so its bits are never numerically
//! reinterpreted here — such a class has no `Epi5Reading` and projection
//! refuses. The declaration is the migration firewall.

use crate::band_reading::EdgeProvenance;
use crate::class_view::ClassId;
use crate::rail_geometry::RailAxis;

/// Number of raw codes the 5-bit field can carry.
pub const CODES: usize = 32;
/// Meaningful V1 states: 4 topologies × 6 certifications.
pub const MEANINGFUL_STATES: usize = 24;
const _: () = assert!(CODES == 1 << 5 && MEANINGFUL_STATES == 4 * 6);
// Topology and certification facts are disjoint, so recipe eligibility
// compiles per factor (`state = topology & certification`).
const _: () = assert!(fact::TOPOLOGY_MASK & fact::CERTIFICATION_MASK == 0);

/// A conjunction of epistemic facts.
pub type Facts = u32;

/// The fact vocabulary. A state's meaning is the union of its two factors'
/// facts; the affordance law states each recipe's obligations in the same
/// vocabulary.
pub mod fact {
    use super::Facts;

    /// Topology: the relation is grounded directly (no intermediate).
    pub const DIRECT: Facts = 1 << 0;
    /// Topology: the relation runs through at least one intermediate.
    pub const INDIRECT: Facts = 1 << 1;
    /// Topology: an intermediate is part of the claim.
    pub const INTERMEDIATE_PRESENT: Facts = 1 << 2;
    /// Topology: the intermediate is not known (hydration is open work).
    pub const INTERMEDIATE_UNKNOWN: Facts = 1 << 3;
    /// Topology: the intermediate is known (a mechanism can be folded/tested).
    pub const INTERMEDIATE_KNOWN: Facts = 1 << 4;
    /// Topology: not resolved at all.
    pub const TOPOLOGY_UNKNOWN: Facts = 1 << 10;
    /// Certification: associated in a declared population.
    pub const ASSOCIATED: Facts = 1 << 6;
    /// Certification: the association survives the robustness mask.
    pub const RELATED: Facts = 1 << 7;
    /// Certification: supports / contributes in every declared stratum.
    pub const SUPPORTS: Facts = 1 << 8;
    /// Certification: supports, and exposure precedes outcome.
    pub const CAUSAL_CANDIDATE: Facts = 1 << 11;
    /// Certification: intervention-certified.
    pub const CAUSES: Facts = 1 << 9;

    /// `Indirect × IntermediateUnknown`.
    pub const IND_UNKNOWN: Facts = INDIRECT | INTERMEDIATE_PRESENT | INTERMEDIATE_UNKNOWN;
    /// `Indirect × IntermediateKnown`.
    pub const IND_KNOWN: Facts = INDIRECT | INTERMEDIATE_PRESENT | INTERMEDIATE_KNOWN;
    /// `Associated × Related`.
    pub const UP_TO_RELATED: Facts = ASSOCIATED | RELATED;
    /// `Associated × Related × Supports`.
    pub const UP_TO_SUPPORTS: Facts = UP_TO_RELATED | SUPPORTS;
    /// Every topology fact.
    pub const TOPOLOGY_MASK: Facts = DIRECT | IND_UNKNOWN | IND_KNOWN | TOPOLOGY_UNKNOWN;
    /// Every certification fact.
    pub const CERTIFICATION_MASK: Facts = UP_TO_SUPPORTS | CAUSAL_CANDIDATE | CAUSES;
}

/// The topology coordinate (bits 59..60). Ordinals equal
/// `causal_edge::layout::CausalTopology` (doc pointer, never an import).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topology2 {
    /// `00`
    Direct,
    /// `01` — `IndirectKnownIntermediates`.
    IndirectKnown,
    /// `10` — `IndirectUnknownIntermediates`.
    IndirectUnknown,
    /// `11`
    Unknown,
}

impl Topology2 {
    /// All four, in ordinal order.
    pub const ALL: [Topology2; 4] = [
        Topology2::Direct,
        Topology2::IndirectKnown,
        Topology2::IndirectUnknown,
        Topology2::Unknown,
    ];

    /// The 2-bit ordinal.
    #[must_use]
    pub const fn ordinal(self) -> u8 {
        self as u8
    }

    /// From the 2-bit ordinal; `None` above 3.
    #[must_use]
    pub const fn from_ordinal(o: u8) -> Option<Self> {
        match o {
            0 => Some(Topology2::Direct),
            1 => Some(Topology2::IndirectKnown),
            2 => Some(Topology2::IndirectUnknown),
            3 => Some(Topology2::Unknown),
            _ => None,
        }
    }

    /// This coordinate's facts.
    #[must_use]
    pub const fn facts(self) -> Facts {
        TOPOLOGY_FACTS[self as usize]
    }
}

/// The certification coordinate (bits 61..63). Codes equal the P7a
/// certification contract (#1369); 6 and 7 are reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Certification3 {
    /// `0` — no obligation met.
    Open,
    /// `1`
    Associated,
    /// `2`
    Related,
    /// `3` — P7a `Contributes`.
    Supports,
    /// `4`
    CausalCandidate,
    /// `5`
    Causes,
}

impl Certification3 {
    /// All six, in code order.
    pub const ALL: [Certification3; 6] = [
        Certification3::Open,
        Certification3::Associated,
        Certification3::Related,
        Certification3::Supports,
        Certification3::CausalCandidate,
        Certification3::Causes,
    ];

    /// The 3-bit code.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// From the 3-bit code; `None` for the reserved 6 and 7 (and above).
    #[must_use]
    pub const fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(Certification3::Open),
            1 => Some(Certification3::Associated),
            2 => Some(Certification3::Related),
            3 => Some(Certification3::Supports),
            4 => Some(Certification3::CausalCandidate),
            5 => Some(Certification3::Causes),
            _ => None,
        }
    }

    /// This coordinate's facts (cumulative: each certification asserts every
    /// weaker one, the chain P7a measured monotone).
    #[must_use]
    pub const fn facts(self) -> Facts {
        CERTIFICATION_FACTS[self as usize]
    }

    /// Does holding `self` license asserting `required`? By facts, never by
    /// comparing codes.
    #[must_use]
    pub const fn entails(self, required: Certification3) -> bool {
        self.facts() & required.facts() == required.facts()
    }
}

/// Topology factor → facts.
pub const TOPOLOGY_FACTS: [Facts; 4] = {
    use fact::*;
    [DIRECT, IND_KNOWN, IND_UNKNOWN, TOPOLOGY_UNKNOWN]
};

/// Certification factor → facts.
pub const CERTIFICATION_FACTS: [Facts; 6] = {
    use fact::*;
    [
        0,
        ASSOCIATED,
        UP_TO_RELATED,
        UP_TO_SUPPORTS,
        UP_TO_SUPPORTS | CAUSAL_CANDIDATE,
        UP_TO_SUPPORTS | CAUSAL_CANDIDATE | CAUSES,
    ]
};

/// The V1 facts of a raw code, from the two factors; `None` if the
/// certification is reserved (or `raw5 > 31`).
#[must_use]
pub const fn facts_v1(raw5: u8) -> Option<Facts> {
    if raw5 as usize >= CODES {
        return None;
    }
    match Certification3::from_code(raw5 >> 2) {
        Some(c) => Some(TOPOLOGY_FACTS[(raw5 & 0b11) as usize] | c.facts()),
        None => None,
    }
}

/// `facts_v1` over all 32 codes — a DERIVED hot-path table, not authority.
pub const COMPILED_FACTS_V1: [Option<Facts>; CODES] = {
    let mut t = [None; CODES];
    let mut c = 0;
    while c < CODES {
        t[c] = facts_v1(c as u8);
        c += 1;
    }
    t
};

/// A set of raw5 codes as one machine word: bit `r` stands for the state
/// whose raw5 is `r`. The 5-bit field says where ONE edge stands; a
/// population says which part of the whole 32-code space satisfies a
/// condition. Transient — computed, never stored on an edge.
pub type Population = u32;

/// The V1 codes whose facts include every fact in `required`.
///
/// Conjunction is intersection: `facts_population_v1(a | b) ==
/// facts_population_v1(a) & facts_population_v1(b)`. Reserved codes (24..31)
/// never appear — they have no facts, so they satisfy nothing, not even
/// `required == 0` (whose population is the 24 meaningful states). A
/// requirement no state meets (e.g. `DIRECT | INDIRECT`) yields `0`.
#[must_use]
pub const fn facts_population_v1(required: Facts) -> Population {
    let mut population = 0;
    let mut raw = 0;
    while raw < CODES {
        if let Some(facts) = COMPILED_FACTS_V1[raw] {
            if facts & required == required {
                population |= 1 << raw;
            }
        }
        raw += 1;
    }
    population
}

/// A reading generation. V1 is the 2 × 3 product above.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Epi5Gen {
    /// `Topology2 × Certification3`.
    #[default]
    V1,
}

/// A coordinate pair validated under a generation. Only constructible
/// validated, so holding one means the state is meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpistemicState5 {
    topology: Topology2,
    certification: Certification3,
    generation: Epi5Gen,
}

impl EpistemicState5 {
    /// The state at `(topology, certification)`. Every pair is meaningful in V1.
    #[must_use]
    pub const fn new(
        generation: Epi5Gen,
        topology: Topology2,
        certification: Certification3,
    ) -> Self {
        Self {
            topology,
            certification,
            generation,
        }
    }

    /// Decode a raw code: `topology = raw5 & 0b11`, `certification = raw5 >> 2`.
    /// Refuses a reserved certification. No provenance or class check: use
    /// [`Epi5Declarations::project_state5`] to read an edge.
    pub const fn decode(generation: Epi5Gen, raw5: u8) -> Result<Self, Epi5ReadError> {
        if raw5 as usize >= CODES {
            return Err(Epi5ReadError::UndeclaredCode(raw5));
        }
        let topology = match Topology2::from_ordinal(raw5 & 0b11) {
            Some(t) => t,
            None => return Err(Epi5ReadError::UndeclaredCode(raw5)),
        };
        match Certification3::from_code(raw5 >> 2) {
            Some(c) => Ok(Self::new(generation, topology, c)),
            None => Err(Epi5ReadError::UndeclaredCode(raw5)),
        }
    }

    /// The raw 5-bit code, for `CausalEdge64::with_epistemic_raw5`.
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.topology.ordinal() | (self.certification.code() << 2)
    }

    /// The topology coordinate.
    #[must_use]
    pub const fn topology(self) -> Topology2 {
        self.topology
    }

    /// The certification coordinate.
    #[must_use]
    pub const fn certification(self) -> Certification3 {
        self.certification
    }

    /// The generation the state was validated under.
    #[must_use]
    pub const fn generation(self) -> Epi5Gen {
        self.generation
    }

    /// `TOPOLOGY_FACTS[t] | CERTIFICATION_FACTS[c]`.
    #[must_use]
    pub const fn facts(self) -> Facts {
        self.topology.facts() | self.certification.facts()
    }

    /// This state's bit in a [`Population`]: `1 << raw()`.
    #[must_use]
    pub const fn bit(self) -> Population {
        1 << self.raw()
    }

    /// Does the state assert every fact in `required`?
    #[must_use]
    pub const fn asserts(self, required: Facts) -> bool {
        self.facts() & required == required
    }

    /// Factor update: change the topology, hold the certification. A legal
    /// product-space transition; moves only bits 59..60 once written.
    #[must_use]
    pub const fn with_topology(self, topology: Topology2) -> Self {
        Self { topology, ..self }
    }

    /// Factor update: change the certification, hold the topology. Moves only
    /// bits 61..63 once written. Whether the CHANGE is earned is the evidence
    /// path's business; this only keeps the coordinate well-formed.
    #[must_use]
    pub const fn with_certification(self, certification: Certification3) -> Self {
        Self {
            certification,
            ..self
        }
    }
}

/// A class's declaration for bits 59..63.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Epi5Reading {
    /// The generation the class's producers write.
    pub generation: Epi5Gen,
}

impl Epi5Reading {
    /// Project a raw code under THIS declaration (no table lookup, no
    /// allocation): provenance, then generation, then code.
    pub const fn project(
        self,
        requested: Epi5Gen,
        raw5: u8,
        provenance: EdgeProvenance,
    ) -> Result<EpistemicState5, Epi5ReadError> {
        if !provenance.trusted() {
            return Err(Epi5ReadError::UnknownProvenance(provenance));
        }
        if !matches!((self.generation, requested), (Epi5Gen::V1, Epi5Gen::V1)) {
            return Err(Epi5ReadError::GenerationMismatch {
                declared: self.generation,
                requested,
            });
        }
        EpistemicState5::decode(requested, raw5)
    }
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
    /// A reserved certification (codes 24..31) or a value above 31.
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

    /// The canonical projection of bits 59..63: declaration, then provenance,
    /// generation and code ([`Epi5Reading::project`]).
    pub fn project_state5(
        &self,
        class: ClassId,
        rail: RailAxis,
        requested: Epi5Gen,
        raw5: u8,
        provenance: EdgeProvenance,
    ) -> Result<EpistemicState5, Epi5ReadError> {
        self.get(class, rail)
            .ok_or(Epi5ReadError::UndeclaredClass(class))?
            .project(requested, raw5, provenance)
    }
}

#[cfg(test)]
mod tests {
    use super::fact::*;
    use super::*;
    use Certification3 as C;
    use Topology2 as T;

    const CLASS: ClassId = 0x0902;
    const RAIL: RailAxis = RailAxis::Taxonomy;

    fn decl() -> Epi5Declarations {
        let mut d = Epi5Declarations::new();
        d.declare(CLASS, RAIL, Epi5Reading::default());
        d
    }

    fn code(t: T, c: C) -> u8 {
        EpistemicState5::new(Epi5Gen::V1, t, c).raw()
    }

    /// The layout is the product: `raw5 = t | c << 2`, 24 meaningful, 8
    /// reserved, and decode inverts it on every code.
    #[test]
    fn the_layout_is_the_two_by_three_product() {
        let mut meaningful = 0;
        for raw in 0u8..32 {
            match EpistemicState5::decode(Epi5Gen::V1, raw) {
                Ok(s) => {
                    meaningful += 1;
                    assert_eq!(s.raw(), raw);
                    assert_eq!(s.topology().ordinal(), raw & 0b11);
                    assert_eq!(s.certification().code(), raw >> 2);
                }
                Err(e) => {
                    assert!(raw >= 24, "only certifications 6, 7 are reserved");
                    assert_eq!(e, Epi5ReadError::UndeclaredCode(raw));
                }
            }
        }
        assert_eq!(meaningful, MEANINGFUL_STATES);
    }

    /// The operator's example coordinates.
    #[test]
    fn the_operator_examples_land_where_stated() {
        let pins = [
            (T::Direct, C::Open, 0),
            (T::Direct, C::Associated, 4),
            (T::Direct, C::Related, 8),
            (T::Direct, C::Supports, 12),
            (T::Direct, C::CausalCandidate, 16),
            (T::Direct, C::Causes, 20),
            (T::IndirectKnown, C::Related, 9),
            (T::IndirectKnown, C::Supports, 13),
            (T::IndirectKnown, C::CausalCandidate, 17),
            (T::IndirectKnown, C::Causes, 21),
            (T::IndirectUnknown, C::Related, 10),
            (T::IndirectUnknown, C::Supports, 14),
            (T::IndirectUnknown, C::Causes, 22),
        ];
        for (t, c, want) in pins {
            assert_eq!(code(t, c), want, "{t:?} × {c:?}");
        }
    }

    /// Facts are the union of the factors' facts, pinned by hand on a few
    /// states so the factor tables cannot drift silently.
    #[test]
    fn facts_are_the_union_of_the_factors() {
        let f = |raw| facts_v1(raw).unwrap();
        assert_eq!(f(10), IND_UNKNOWN | ASSOCIATED | RELATED);
        assert_eq!(f(13), IND_KNOWN | UP_TO_SUPPORTS);
        assert_eq!(f(4), DIRECT | ASSOCIATED);
        assert_eq!(
            f(21),
            IND_KNOWN | UP_TO_SUPPORTS | CAUSAL_CANDIDATE | CAUSES
        );
        for raw in 0u8..24 {
            let s = EpistemicState5::decode(Epi5Gen::V1, raw).unwrap();
            assert_eq!(s.facts() & TOPOLOGY_MASK, s.topology().facts());
            assert_eq!(s.facts() & CERTIFICATION_MASK, s.certification().facts());
            assert_eq!(COMPILED_FACTS_V1[raw as usize], Some(s.facts()));
        }
        assert!(COMPILED_FACTS_V1[24..].iter().all(Option::is_none));
    }

    /// Orthogonality: an intervention-certified relation with an unresolved
    /// mechanism or topology is a meaningful state.
    #[test]
    fn causes_is_meaningful_under_every_topology() {
        for t in T::ALL {
            let s = EpistemicState5::decode(Epi5Gen::V1, code(t, C::Causes)).unwrap();
            assert!(s.asserts(CAUSES));
        }
        assert!(EpistemicState5::decode(Epi5Gen::V1, 22)
            .unwrap()
            .asserts(IND_UNKNOWN | CAUSES));
        assert!(EpistemicState5::decode(Epi5Gen::V1, 23)
            .unwrap()
            .asserts(TOPOLOGY_UNKNOWN | CAUSES));
    }

    /// Certification entailment is the cumulative chain (P7a's `>=` on 0..=5).
    #[test]
    fn certification_entailment_is_the_measured_chain() {
        for a in C::ALL {
            for b in C::ALL {
                assert_eq!(a.entails(b), a.code() >= b.code(), "{a:?} vs {b:?}");
            }
        }
    }

    /// A factor update moves only its own coordinate.
    #[test]
    fn factor_updates_move_only_their_coordinate() {
        for t in T::ALL {
            for c in C::ALL {
                let s = EpistemicState5::new(Epi5Gen::V1, t, c);
                for t2 in T::ALL {
                    let m = s.with_topology(t2);
                    assert_eq!(m.certification(), c);
                    assert_eq!((m.raw() ^ s.raw()) & !0b11, 0);
                }
                for c2 in C::ALL {
                    let m = s.with_certification(c2);
                    assert_eq!(m.topology(), t);
                    assert_eq!((m.raw() ^ s.raw()) & 0b11, 0);
                }
            }
        }
    }

    #[test]
    fn projection_refuses_every_failure_and_admits_the_good_case() {
        let d = decl();
        let s = d
            .project_state5(CLASS, RAIL, Epi5Gen::V1, 10, EdgeProvenance::V3Register)
            .unwrap();
        assert_eq!(
            (s.topology(), s.certification()),
            (T::IndirectUnknown, C::Related)
        );
        assert_eq!(
            d.project_state5(0x0999, RAIL, Epi5Gen::V1, 10, EdgeProvenance::V2Stamped),
            Err(Epi5ReadError::UndeclaredClass(0x0999))
        );
        for p in [EdgeProvenance::V1Legacy, EdgeProvenance::Unknown] {
            assert_eq!(
                d.project_state5(CLASS, RAIL, Epi5Gen::V1, 10, p),
                Err(Epi5ReadError::UnknownProvenance(p))
            );
        }
        for reserved in [24u8, 27, 28, 31] {
            assert_eq!(
                d.project_state5(
                    CLASS,
                    RAIL,
                    Epi5Gen::V1,
                    reserved,
                    EdgeProvenance::V2Stamped
                ),
                Err(Epi5ReadError::UndeclaredCode(reserved))
            );
        }
        assert_eq!(
            EpistemicState5::decode(Epi5Gen::V1, 32),
            Err(Epi5ReadError::UndeclaredCode(32))
        );
    }

    /// Population of `required` built the other way: enumerate the 24
    /// coordinate pairs and ask each state.
    fn population_by_states(required: Facts) -> Population {
        let mut p = 0;
        for c in C::ALL {
            for t in T::ALL {
                let s = EpistemicState5::new(Epi5Gen::V1, t, c);
                if s.asserts(required) {
                    p |= s.bit();
                }
            }
        }
        p
    }

    /// D-EPI-POP-0: the populations of the stated questions.
    #[test]
    fn populations_of_the_stated_questions() {
        let pop = facts_population_v1;
        assert_eq!(pop(CAUSES).count_ones(), 4, "the Causes column");
        assert_eq!(
            pop(RELATED).count_ones(),
            16,
            "Related-or-stronger x 4 topologies"
        );
        assert_eq!(pop(IND_UNKNOWN | RELATED).count_ones(), 4);
        assert_eq!(pop(IND_KNOWN | SUPPORTS).count_ones(), 3);
        assert_eq!(pop(TOPOLOGY_UNKNOWN | CAUSES), 1 << 23);
        assert_eq!(
            pop(CAUSES) & pop(IND_UNKNOWN),
            EpistemicState5::new(Epi5Gen::V1, T::IndirectUnknown, C::Causes).bit()
        );
        // The empty requirement selects every meaningful state and no
        // reserved one; an impossible requirement selects nothing.
        assert_eq!(pop(0), (1 << MEANINGFUL_STATES) - 1);
        assert_eq!(pop(DIRECT | INDIRECT), 0);
    }

    /// Over every requirement made of the declared facts, the operator
    /// equals the state enumeration and never selects a reserved code.
    #[test]
    fn population_equals_the_state_enumeration_for_every_requirement() {
        let all = TOPOLOGY_MASK | CERTIFICATION_MASK;
        let mut sub = all;
        let mut checked = 0;
        loop {
            let p = facts_population_v1(sub);
            assert_eq!(p, population_by_states(sub), "requirement {sub:#x}");
            assert_eq!(p >> MEANINGFUL_STATES, 0, "a reserved code appeared");
            checked += 1;
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & all;
        }
        assert_eq!(checked, 1 << all.count_ones());
    }

    /// Conjunction of requirements is intersection of populations.
    #[test]
    fn conjunction_is_intersection() {
        let all = TOPOLOGY_MASK | CERTIFICATION_MASK;
        for a in 0..Facts::BITS {
            for b in 0..Facts::BITS {
                let (fa, fb) = (1 << a & all, 1 << b & all);
                assert_eq!(
                    facts_population_v1(fa | fb),
                    facts_population_v1(fa) & facts_population_v1(fb)
                );
            }
        }
    }
}
