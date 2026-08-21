// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `band_reading` — the **59..63 reading contract** (`D-ACR-7`, council-ratified
//! spec: `.claude/plans/dacr7-band-reading-contract-v1.md`).
//!
//! Two bits (the truth tail) and three bits (the band tail) of a causal edge
//! carry FOUR shipped readings between them — `TrustTexture` vs
//! `CausalTopology` on the 2-bit field, `ReasoningBand`-present vs spare on
//! the 3-bit field — and **which reading a producer wrote is not recoverable
//! from the bits**. A consumer that reads one lens while the producer wrote
//! the other gets a plausible wrong answer, silently. This module is the
//! producer knowledge, declared per `(classid, rail)`, that the bits
//! themselves cannot carry.
//!
//! # One contract, TWO carriers
//!
//! The same two fields exist on both causal-edge carriers, and the projection
//! here takes **raw ordinals**, so it serves both identically and names
//! neither (this crate is zero-dep; so is the edge crate — measured, both
//! `Cargo.toml`s refuse the other):
//!
//! | carrier | truth field | band field | role |
//! |---|---|---|---|
//! | `CausalEdge64` (bits 59-60 / 61-63) | `truth()` | `reasoning_band()` | muscle memory — it reasons |
//! | `CausalEdgeV3` (byte 8 hi-2 / byte 9 lo-3) | `truth_raw()` | `spare_raw()` | granularity — it rehydrates INTO CE64 to reason |
//!
//! The V3 module doc states the gap this contract closes, verbatim: *"Which
//! lens the ordinal was written through is the producer's knowledge, not the
//! conversion's."*
//!
//! # The provenance doctrine (council BLOCK 1 — read before trusting a V3 register)
//!
//! Under the v1 layout, both fields were `temporal` bits: a v1 edge with
//! `temporal >= 512` reads a NON-ZERO band. And `CausalEdgeV3::from_v1` has
//! **no provenance parameter** — it raw-copies the fields from whatever CE64
//! it is handed. So the v1 trap reaches V3 **transitively through the lift**,
//! and a tainted register is indistinguishable from a clean one. Hence:
//!
//! - [`EdgeProvenance::V3Register`] means *"the caller asserts this register
//!   was minted clean"* — it is an **assertion, never an inference**. The
//!   contract cannot recover what the lift destroyed.
//! - A register of unstated origin is [`EdgeProvenance::Unknown`], and
//!   `Unknown` **refuses** (zero-fallback: absent an assertion, refuse).
//!
//! # Total lookup, fallible projection (the council's L1 split)
//!
//! Two different operations, deliberately different shapes:
//!
//! - **Declaration lookup is TOTAL** — [`ClassView::band_reading`]
//!   (`class_view.rs`) and [`BandDeclarations::reading_or_default`] return the
//!   [zero-fallback](BandReading::ZERO_FALLBACK) for an undeclared class,
//!   exactly like the sibling selectors `edge_codec_flavor` / `rail_carving` /
//!   `value_schema`. Hot-path safe, never an error. *(G5a)*
//! - **Projection is FALLIBLE** — [`BandReading::project_truth`] /
//!   [`project_band`](BandReading::project_band) return
//!   [`BandReadError`]: a lens mismatch, an absent band, or untrusted
//!   provenance must **FAIL, never return a plausible value**. *(G3′/G4′/G5b)*
//!
//! The audit distinction rides on [`BandDeclarations::get`] returning
//! `Option`: `None` = never declared; `Some(band: Absent)` = **explicitly**
//! declared band-free. Folding the two would make "opted out" and "never
//! considered" indistinguishable to a migration audit.
//!
//! # What this module does NOT do (G8)
//!
//! It declares; it never stamps. No `with_*` call, no shift/mask against any
//! edge layout, no feature gate (G9: a `#[cfg]` split re-meaning a reading
//! under one name is the exact v1-accessor anti-pattern
//! `I-LEGACY-API-FEATURE-GATED` catalogues). Temporal carries NO field here —
//! time is implicit in the epistemic pothole (Lance versions); explicit
//! temporal lives only in its three sanctioned homes (Rubikon revision window,
//! `CausalEdgeV3`'s TE byte, a future attention-v3 reading). `EdgeProvenance`
//! is **layout epoch, never time**.

use crate::class_view::ClassId;
use crate::rail_geometry::RailAxis;
use crate::recipe_kernels::Tactic;

/// States the 2-bit truth field can carry — the arity pin (G7′, compile-time,
/// F9-exempt). A 5-variant sibling (`planner::mul::trust::TrustTexture`, with
/// `Dissonant`) is **unrepresentable** here and must never be routed through
/// these bits.
pub const TRUTH_STATES: usize = 4;
/// States the 3-bit band field can carry.
pub const BAND_STATES: usize = 8;
const _: () = assert!(TRUTH_STATES == 4 && (TRUTH_STATES - 1) == 0b11);
const _: () = assert!(BAND_STATES == 8 && (BAND_STATES - 1) == 0b111);

/// WHICH projection of the 2-bit truth field this class's producers wrote.
///
/// The two shipped readings are ordinal-identical on the wire (`Crystalline` ≡
/// `Direct` … `Murky` ≡ `Unknown`), which is precisely why the bits cannot
/// disambiguate themselves. The names below are doc-comment pointers to the
/// edge crate's enums, **never imports** (both crates are zero-dep):
///
/// - [`Trust`](TruthLens::Trust) → `causal_edge::layout::TrustTexture`
///   (4 variants — NOT `contract::mul::TrustTexture`, NOT the planner's
///   5-variant one, NOT arigraph's 3-variant one; the ×4 homonym is recorded
///   in `docs/TYPE_DUPLICATION_MAP.md`).
/// - [`Topology`](TruthLens::Topology) → `causal_edge::layout::CausalTopology`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TruthLens {
    /// Epistemic-trust reading — the canonical default (`layout.rs`: "the
    /// canonical epistemic-trust reading").
    #[default]
    Trust,
    /// Causal-topology reading — the additive factual view of the same bits.
    Topology,
}

/// Whether the 3-bit field carries a `ReasoningBand` for this class at all.
///
/// `Absent` is the zero-fallback: an unstamped class declares no band, and
/// projecting one is a [refusal](BandReadError::BandAbsent), never a
/// `Surface(0)` lookalike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BandPresence {
    /// No band: the three bits are spare for this class. The default.
    #[default]
    Absent,
    /// The class's producers stamp `ReasoningBand` ordinals via the one
    /// sanctioned writer (`with_reasoning_band()` — nothing derives it).
    Present,
}

/// WHICH witness carrier discriminates evidence-KIND for this class — the
/// axis the band deliberately does not carry (frozen decision F5: *the band
/// grades; the witness reference discriminates*). A weak episodic witness and
/// a weak epistemic claim are the same band and different things; the
/// difference lives in what the row POINTS AT, costing zero bits here.
///
/// Under the two-armed trace (`known-unknown-handover-network-v1.md` §9 ⊘⊘)
/// this reference is also what a next-rung focus entry resolves through:
/// static-substrate attention lands as an alpha-layer entry, dynamic-substrate
/// attention as the row's own Lance-version history — the discriminator names
/// the kind either way. The Hole becomes a target here once `HoleV3` lands
/// (blocked on the `BoardAggregates = 15` mint).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WitnessKind {
    /// No witness reference — grading stands alone.
    #[default]
    None,
    /// A `WitnessTable`/`WitnessLens` entry (the generic register-slab
    /// machinery, `witness_table.rs`).
    Table,
    /// A `CausalWitnessFacet` locus register (`causal_witness.rs`).
    CausalFacet,
    /// An episodic basin (the AriGraph cold-path lineage; hot-path mount
    /// pending `ValueTenant::EpisodicEdges`).
    EpisodicBasin,
}

/// The layout EPOCH a raw ordinal was read under — **never a timestamp**
/// (the temporal doctrine: time is implicit in the pothole; this axis is
/// about which bit-layout wrote the field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EdgeProvenance {
    /// The edge was stamped under the v2 layout — the fields mean what the
    /// v2 accessors say.
    V2Stamped,
    /// **The caller asserts this V3 register was minted clean** (stamped by a
    /// v2-aware producer, or lifted from a `V2Stamped` edge). This is an
    /// ASSERTION, never an inference: `CausalEdgeV3::from_v1` drops
    /// provenance, so a register's own bytes cannot prove this (council
    /// BLOCK 1).
    V3Register,
    /// A v1-layout edge: both fields alias old `temporal` bits — a
    /// `temporal >= 512` reads as a non-zero band. Refused.
    V1Legacy,
    /// Origin unstated. The zero-fallback, and it REFUSES: absent an
    /// assertion, the fields are not readable. The default.
    #[default]
    Unknown,
}

impl EdgeProvenance {
    /// Are the truth/band fields trustworthy under this provenance?
    /// `V1Legacy` and `Unknown` are not — on EITHER field (`layout.rs` applies
    /// the version-gate rule to bits 59-60 as well as 61-63).
    #[inline]
    #[must_use]
    pub const fn trusted(self) -> bool {
        matches!(self, EdgeProvenance::V2Stamped | EdgeProvenance::V3Register)
    }
}

/// Why a projection refused. Every variant is a FAILURE the contract's own
/// falsifier demands — *"must FAIL, not return a plausible value."*
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandReadError {
    /// The consumer requested one lens; the class declared the other. The
    /// plausible-wrong-answer case, refused.
    LensMismatch {
        /// What the class's producers actually wrote.
        declared: TruthLens,
        /// What the consumer asked to read.
        requested: TruthLens,
    },
    /// The class declared `BandPresence::Absent` — its three bits are spare,
    /// and reading a band from spare bits is the `Surface(0)` lookalike this
    /// refuses.
    BandAbsent,
    /// `V1Legacy` or `Unknown` provenance: the fields may be stale v1
    /// `temporal` payload (directly, or transitively through `from_v1`).
    UnknownProvenance,
    /// No declaration exists for this `(classid, rail)` — the fallible path's
    /// guard (G5b). The TOTAL path folds this to the zero-fallback instead;
    /// they are different surfaces on purpose.
    UndeclaredClass(ClassId),
}

/// The reading a class's producers committed to, per `(classid, rail)` —
/// three declarations, zero bits (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BandReading {
    /// Which projection of the 2-bit truth field applies.
    pub truth_lens: TruthLens,
    /// Whether the 3-bit field carries a band at all.
    pub band: BandPresence,
    /// Which witness carrier discriminates evidence-kind (F5).
    pub witness: WitnessKind,
}

impl BandReading {
    /// The zero-fallback reading an undeclared class resolves to on the TOTAL
    /// path: the canonical `Trust` lens, **no band** (projecting one refuses),
    /// no witness reference. Identical to `Default`, named so call sites read
    /// as the ladder rung they are.
    pub const ZERO_FALLBACK: Self = Self {
        truth_lens: TruthLens::Trust,
        band: BandPresence::Absent,
        witness: WitnessKind::None,
    };

    /// The producer-side pre-write check (council L2): may a producer stamp
    /// the truth field through `requested`? `false` means the write would
    /// contradict this declaration. This DECLARES compatibility — it cannot
    /// enforce it, by construction: the edge crate is zero-dep and does not
    /// see this contract (`causal-edge/Cargo.toml:20-23`), so enforcement at
    /// the write site is that crate's follow-up, not a claim here.
    #[inline]
    #[must_use]
    pub fn admits(self, requested: TruthLens) -> bool {
        self.truth_lens == requested
    }

    /// The producer-side pre-write check for the band field: `false` for an
    /// `Absent` class — stamping a band there would re-mean spare bits.
    #[inline]
    #[must_use]
    pub fn admits_band(self) -> bool {
        self.band == BandPresence::Present
    }

    /// Project the 2-bit truth field: validates provenance, then the lens.
    /// Returns the validated raw ordinal (`0..4`) — the CONSUMER (who holds
    /// the edge crate) projects it through the declared lens's enum; this
    /// crate stays content-blind and imports neither.
    ///
    /// **Precondition:** `truth_raw` is the 2-bit field value (`< 4`) —
    /// `debug_assert`ed, part of the G7′ compile-time/precondition pin
    /// (F9-exempt, stated).
    ///
    /// Check order is doctrine: **provenance before lens** — untrustworthy
    /// bits fail before any question about their meaning is entertained.
    pub fn project_truth(
        self,
        requested: TruthLens,
        truth_raw: u8,
        provenance: EdgeProvenance,
    ) -> Result<u8, BandReadError> {
        debug_assert!(
            (truth_raw as usize) < TRUTH_STATES,
            "truth_raw must be the 2-bit field value"
        );
        if !provenance.trusted() {
            return Err(BandReadError::UnknownProvenance);
        }
        if self.truth_lens != requested {
            return Err(BandReadError::LensMismatch {
                declared: self.truth_lens,
                requested,
            });
        }
        Ok(truth_raw)
    }

    /// Project the 3-bit band field: validates provenance, then presence.
    /// Returns the validated raw ordinal (`0..8`) for the consumer to project
    /// through `ReasoningBand` — never a `ReasoningBand` here (content-blind).
    ///
    /// **Precondition:** `band_raw < 8` (`debug_assert`ed, G7′ regime).
    pub fn project_band(
        self,
        band_raw: u8,
        provenance: EdgeProvenance,
    ) -> Result<u8, BandReadError> {
        debug_assert!(
            (band_raw as usize) < BAND_STATES,
            "band_raw must be the 3-bit field value"
        );
        if !provenance.trusted() {
            return Err(BandReadError::UnknownProvenance);
        }
        if self.band != BandPresence::Present {
            return Err(BandReadError::BandAbsent);
        }
        Ok(band_raw)
    }
}

/// The declaration table — `(classid, rail) → BandReading`, caller-populated
/// (an OGAR mint / bake decision populates it; this crate never pre-fills a
/// class, so D-ACR-2's operator-gated rail mint is not pre-empted).
///
/// Both access disciplines live here, on purpose (the L1 split):
/// [`get`](Self::get) is the audit read (`Option` — never-declared vs
/// declared-`Absent` stay distinguishable), [`reading_or_default`](Self::reading_or_default)
/// is the total read (G5a), and the `project_*` pair is the fallible
/// projection under lookup (G5b fires [`BandReadError::UndeclaredClass`]).
#[derive(Debug, Clone, Default)]
pub struct BandDeclarations {
    entries: Vec<((ClassId, RailAxis), BandReading)>,
}

impl BandDeclarations {
    /// An empty table — every class undeclared.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Declare (or re-declare) a class's reading. Returns `true` if this
    /// replaced an existing declaration — a re-declaration is visible, never
    /// silent.
    pub fn declare(&mut self, class: ClassId, rail: RailAxis, reading: BandReading) -> bool {
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

    /// The AUDIT read: `None` = never declared; `Some` = declared (possibly
    /// `band: Absent` — an explicit opt-out, distinguishable from silence).
    #[must_use]
    pub fn get(&self, class: ClassId, rail: RailAxis) -> Option<BandReading> {
        self.entries
            .iter()
            .find(|((c, r), _)| *c == class && *r == rail)
            .map(|(_, b)| *b)
    }

    /// The TOTAL read (G5a): an undeclared class folds to
    /// [`BandReading::ZERO_FALLBACK`], never an error — sibling-consistent
    /// with `edge_codec_flavor` / `rail_carving`.
    #[must_use]
    pub fn reading_or_default(&self, class: ClassId, rail: RailAxis) -> BandReading {
        self.get(class, rail).unwrap_or(BandReading::ZERO_FALLBACK)
    }

    /// The fallible truth projection under lookup: declaration first
    /// (G5b — [`BandReadError::UndeclaredClass`] must fire), then delegate to
    /// [`BandReading::project_truth`] (provenance, then lens).
    pub fn project_truth(
        &self,
        class: ClassId,
        rail: RailAxis,
        requested: TruthLens,
        truth_raw: u8,
        provenance: EdgeProvenance,
    ) -> Result<u8, BandReadError> {
        self.get(class, rail)
            .ok_or(BandReadError::UndeclaredClass(class))?
            .project_truth(requested, truth_raw, provenance)
    }

    /// The fallible band projection under lookup — same discipline.
    pub fn project_band(
        &self,
        class: ClassId,
        rail: RailAxis,
        band_raw: u8,
        provenance: EdgeProvenance,
    ) -> Result<u8, BandReadError> {
        self.get(class, rail)
            .ok_or(BandReadError::UndeclaredClass(class))?
            .project_band(band_raw, provenance)
    }
}

/// The sampling filter the acceptance condition demands (F6/G6): admit a
/// tactic iff it can move `delta_conf` — **never** filter on
/// `maturity().is_production()`, which admits 31/34 where only 14/34 can
/// dissent. A sampled tactic that cannot move the confidence number is a
/// spent slot returning guaranteed agreement
/// (`E-A-WATCHER-THAT-CANNOT-DISSENT-IS-NOT-A-WATCHER-1`).
#[inline]
#[must_use]
pub fn sampling_admits(tactic: &dyn Tactic) -> bool {
    tactic.moves_confidence()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn topology_class() -> BandReading {
        BandReading {
            truth_lens: TruthLens::Topology,
            band: BandPresence::Present,
            witness: WitnessKind::CausalFacet,
        }
    }

    // ── G3′: lens mismatch AND match ─────────────────────────────────────

    #[test]
    fn g3_lens_mismatch_fails_and_match_resolves() {
        let decl = topology_class();
        // can-fire: Topology-declared, read as Trust → LensMismatch.
        assert_eq!(
            decl.project_truth(TruthLens::Trust, 2, EdgeProvenance::V2Stamped),
            Err(BandReadError::LensMismatch {
                declared: TruthLens::Topology,
                requested: TruthLens::Trust,
            })
        );
        // can-stay-silent: read as Topology → Ok, the ordinal untouched.
        assert_eq!(
            decl.project_truth(TruthLens::Topology, 2, EdgeProvenance::V2Stamped),
            Ok(2)
        );
    }

    // ── G4′: provenance, both directions ─────────────────────────────────

    #[test]
    fn g4_untrusted_provenance_refuses_and_trusted_resolves() {
        let decl = topology_class();
        for bad in [EdgeProvenance::V1Legacy, EdgeProvenance::Unknown] {
            assert_eq!(
                decl.project_truth(TruthLens::Topology, 1, bad),
                Err(BandReadError::UnknownProvenance),
                "truth field must refuse under {bad:?}"
            );
            assert_eq!(
                decl.project_band(5, bad),
                Err(BandReadError::UnknownProvenance),
                "band field must refuse under {bad:?}"
            );
        }
        for good in [EdgeProvenance::V2Stamped, EdgeProvenance::V3Register] {
            assert_eq!(decl.project_truth(TruthLens::Topology, 1, good), Ok(1));
            assert_eq!(decl.project_band(5, good), Ok(5));
        }
    }

    /// Provenance is checked BEFORE the lens: untrustworthy bits fail before
    /// any question about their meaning — a mismatched request under bad
    /// provenance reports the provenance, not the mismatch.
    #[test]
    fn provenance_outranks_the_lens_question() {
        assert_eq!(
            topology_class().project_truth(TruthLens::Trust, 0, EdgeProvenance::Unknown),
            Err(BandReadError::UnknownProvenance)
        );
    }

    /// The default provenance is `Unknown`, and `Unknown` refuses — the
    /// zero-fallback ladder applied to trust: absent an assertion, refuse.
    #[test]
    fn default_provenance_is_unknown_and_refuses() {
        assert_eq!(EdgeProvenance::default(), EdgeProvenance::Unknown);
        assert!(!EdgeProvenance::default().trusted());
    }

    // ── G5a / G5b: the total and fallible surfaces are DIFFERENT ─────────

    #[test]
    fn g5a_total_lookup_folds_undeclared_to_zero_fallback() {
        let table = BandDeclarations::new();
        let r = table.reading_or_default(0x0301, RailAxis::Taxonomy);
        assert_eq!(r, BandReading::ZERO_FALLBACK, "no error, the fold");
        // …and the fallback's own band projection still refuses (Absent):
        assert_eq!(
            r.project_band(3, EdgeProvenance::V2Stamped),
            Err(BandReadError::BandAbsent)
        );
    }

    #[test]
    fn g5b_fallible_projection_fires_on_undeclared() {
        let table = BandDeclarations::new();
        assert_eq!(
            table.project_truth(
                0x0301,
                RailAxis::Taxonomy,
                TruthLens::Trust,
                0,
                EdgeProvenance::V2Stamped
            ),
            Err(BandReadError::UndeclaredClass(0x0301))
        );
        // silent half: a declared class projects through the same surface.
        let mut t = BandDeclarations::new();
        t.declare(0x0301, RailAxis::Taxonomy, topology_class());
        assert_eq!(
            t.project_truth(
                0x0301,
                RailAxis::Taxonomy,
                TruthLens::Topology,
                3,
                EdgeProvenance::V2Stamped
            ),
            Ok(3)
        );
    }

    // ── the audit distinction (L3) ───────────────────────────────────────

    #[test]
    fn never_declared_and_declared_absent_stay_distinguishable() {
        let mut t = BandDeclarations::new();
        t.declare(
            0x0302,
            RailAxis::Mereology,
            BandReading {
                band: BandPresence::Absent,
                ..BandReading::ZERO_FALLBACK
            },
        );
        assert_eq!(t.get(0x0301, RailAxis::Mereology), None, "never declared");
        assert!(
            t.get(0x0302, RailAxis::Mereology).is_some(),
            "explicit opt-out is a declaration, not silence"
        );
        // …while the TOTAL read is identical for both — that is the fold,
        // and it is why the audit path must read the Option.
        assert_eq!(
            t.reading_or_default(0x0301, RailAxis::Mereology),
            t.reading_or_default(0x0302, RailAxis::Mereology)
        );
    }

    #[test]
    fn redeclaration_is_visible_never_silent() {
        let mut t = BandDeclarations::new();
        assert!(!t.declare(1, RailAxis::Taxonomy, topology_class()));
        assert!(t.declare(1, RailAxis::Taxonomy, BandReading::ZERO_FALLBACK));
        assert_eq!(
            t.get(1, RailAxis::Taxonomy),
            Some(BandReading::ZERO_FALLBACK)
        );
    }

    // ── BandAbsent: fire AND stay-silent ─────────────────────────────────

    #[test]
    fn band_absent_refuses_and_present_resolves() {
        let absent = BandReading::ZERO_FALLBACK;
        assert_eq!(
            absent.project_band(0, EdgeProvenance::V2Stamped),
            Err(BandReadError::BandAbsent),
            "even ordinal 0 must refuse — a refusal is not Surface(0)"
        );
        assert_eq!(
            topology_class().project_band(7, EdgeProvenance::V2Stamped),
            Ok(7)
        );
    }

    // ── the producer pre-write check (L2) ────────────────────────────────

    #[test]
    fn admits_declares_write_compatibility_both_ways() {
        let decl = topology_class();
        assert!(decl.admits(TruthLens::Topology));
        assert!(!decl.admits(TruthLens::Trust));
        assert!(decl.admits_band());
        assert!(!BandReading::ZERO_FALLBACK.admits_band());
    }

    // ── G6: the sampling filter, against the REAL kernel registry ────────

    #[test]
    fn g6_sampling_admits_the_14_and_rejects_the_20_mutes() {
        let kernels = crate::recipe_kernels::all_kernels();
        let admitted = kernels.iter().filter(|k| sampling_admits(**k)).count();
        assert_eq!(
            admitted, 14,
            "the measured delta_conf-capable count — a drift here means a \
             kernel changed capability and this contract's premise moved"
        );
        assert_eq!(kernels.len() - admitted, 20, "the mutes are rejected");
        // And the filter is NOT the maturity filter: production count differs.
        let production = kernels
            .iter()
            .filter(|k| k.maturity().is_production())
            .count();
        assert_ne!(
            admitted, production,
            "delta_conf capability and maturity must remain different questions"
        );
    }

    // ── G7′: the arity pin (compile-time regime, F9-exempt — stated) ─────

    #[test]
    fn g7_arity_pins_hold() {
        assert_eq!(TRUTH_STATES, 4, "2 bits: a 5th state is unrepresentable");
        assert_eq!(BAND_STATES, 8);
    }

    // ── the zero-fallback is what it says ────────────────────────────────

    #[test]
    fn zero_fallback_is_trust_absent_none_and_is_the_default() {
        assert_eq!(BandReading::ZERO_FALLBACK, BandReading::default());
        assert_eq!(BandReading::ZERO_FALLBACK.truth_lens, TruthLens::Trust);
        assert_eq!(BandReading::ZERO_FALLBACK.band, BandPresence::Absent);
        assert_eq!(BandReading::ZERO_FALLBACK.witness, WitnessKind::None);
    }
}
