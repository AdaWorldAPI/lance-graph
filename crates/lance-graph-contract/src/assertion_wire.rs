// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `assertion_wire` — **D-BBB-NARS-2**: the versioned canonical little-endian
//! truth DTO. The one syntax/vocabulary module the G11 fence admits for truth.
//!
//! # What this is
//!
//! The 16-byte edge facet — `classid(4) | CausalEdgeV3 payload(12)` — read as
//! a **complete assertion** rather than a bare `(frequency, confidence)` pair:
//!
//! ```text
//! Assertion = proposition reference   (target node: its CAM-PQ facet IS the SPO)
//!           × Pearl projection         (causal_mask, 3 bits)
//!           × NARS valuation           ((f, c), two u8)
//!           × causal topology          (AssertionTopology, 2 bits)
//!           × reasoning/assertion band (AssertionBand, 3 bits)
//!           × provenance               (EdgeProvenance, declared — never inferred)
//! ```
//!
//! Operator rulings, 2026-09-10 (`E-LE-IS-THE-UNIVERSAL-DTO-LAYER-TYPED-SYNTAX-MEANS-A-VERSIONED-LE-SCHEMA-1`):
//! *little-endian is the universal DTO layer of the ABI*; a bare `(f, c)` is a
//! degree, not a typed truth; `CausalTopology` and `ReasoningBand` are
//! **defining, universal** coordinates of truth — *"a field becomes defining
//! when changing or omitting it changes the proposition, not merely its
//! presentation. Every defining epistemic dimension SHALL participate in the
//! versioned canonical LE DTO; a reader lacking its declared lens or
//! provenance must refuse, never project a plausible default."* And the
//! boundary: **meaning crosses; machinery does not.**
//!
//! # What this is NOT
//!
//! - **Not arithmetic.** Nothing here computes a truth from truths — no
//!   revision, deduction, abduction. `D-BBB-NARS-1`: execution stays
//!   substrate-owned (`nars_engine`, `causal_edge::syllogize`). A consumer that
//!   holds this DTO can recognize and preserve the assertion; it cannot reason
//!   with it, by construction.
//! - **Not a new layout.** The 16 bytes are the existing V3 edge facet
//!   (`causal_edge::edge_v3::CausalEdgeV3`, 12 B) behind the existing key
//!   classid (4 B). No new bit, no `ENVELOPE_LAYOUT_VERSION` bump (D-ACR-7 F7).
//!   The byte positions below are a *mirror* of that crate's documented layout
//!   — both crates are zero-dep and cannot import each other — and the mirror
//!   is FUSED by a cross-crate parity test in `lance-graph-planner`
//!   (`cache::assertion_wire_parity`), the only crate that holds both.
//! - **Not a reader that guesses.** [`AssertionWire::read`] is fallible and
//!   refusing, exactly as [`crate::band_reading`] (D-ACR-7): provenance before
//!   lens before presence; an unstated origin, a lens the class did not
//!   declare, or an absent band is an `Err`, never `Surface(0)`.
//!
//! # The schema version
//!
//! [`ASSERTION_WIRE_SCHEMA`] names this reading of the 16 bytes. It is NOT
//! carried inside them (every byte is assigned) — it rides the envelope
//! ([`crate::soa_envelope::ENVELOPE_LAYOUT_VERSION`], which governs the
//! register-file image these facets live in) and the ABI manifest a G11 host
//! exports beside its endianness probe. A reader whose schema constant differs
//! from the producer's must refuse; the pair `(ENVELOPE_LAYOUT_VERSION,
//! ASSERTION_WIRE_SCHEMA)` is the version the LE ruling asks for.
//!
//! # The aliasing pair — this module's own falsifier
//!
//! ```text
//! (S,P,O, f,c, IndirectUnknownIntermediates, Relation)   "S and O are related; mediation is unknown."
//! (S,P,O, f,c, IndirectKnownIntermediates,   Causal)     "P causally connects S to O; the mediation is known."
//! ```
//!
//! Identical `(f, c)`, different truths. `Causal` is not "Relation with more
//! confidence"; `IndirectKnown` is not a cosmetic refinement of
//! `IndirectUnknown`. Two wires that differ only in those bits MUST read to
//! different [`AssertionView`]s and MUST survive `to_le_bytes`/`from_le_bytes`
//! distinct — flattening either to `(S,P,O,f,c)` is **epistemic aliasing**, the
//! `F-BBB-NARS-2 (LE)` failure. Pinned below and cross-crate in the planner.
//!
//! # One fence entry, not the cupboard
//!
//! Everything a G11 consumer needs to READ an assertion is reachable through
//! this module: the wire type, the two vocabularies, and re-exports of the
//! declaration types from [`crate::band_reading`]. So the G11 allowlist grows
//! by exactly ONE module (`assertion_wire`) — the "one scalpel cut, never the
//! cupboard" clause of `D-BBB-NARS-1`. Java-side admission is the
//! `lance-graph-java` brick (its `ALLOWED` list, `CLAUDE.md`, `Cargo.toml`
//! must move together); this crate only makes it admissible.

pub use crate::band_reading::{
    BandPresence, BandReadError, BandReading, EdgeProvenance, TruthLens, WitnessKind,
};

/// The schema this module reads the 16 bytes under. Bump ONLY with a
/// documented re-meaning of a byte position; never carried in the bytes.
pub const ASSERTION_WIRE_SCHEMA: u8 = 1;

/// Width of the wire: the key classid + the 96-bit V3 edge register.
pub const ASSERTION_WIRE_BYTES: usize = 16;

// ── Byte positions (LE; mirror of `causal_edge::edge_v3` — fused in the planner) ──

/// `classid` — bytes 0..4, little-endian `u32`.
pub const CLASSID_OFFSET: usize = 0;
/// NARS frequency, `255 = 1.0` — payload byte 0.
pub const FREQUENCY_OFFSET: usize = 4;
/// NARS confidence, `255 = 1.0` — payload byte 1.
pub const CONFIDENCE_OFFSET: usize = 5;
/// Pearl 2³ causal mask (low 3 bits) | direction triad (bits 3..6) — payload byte 2.
pub const KAUSAL_OFFSET: usize = 6;
/// Signed 4-bit inference mantissa (low nibble) | plasticity (bits 4..7) — payload byte 3.
pub const MANTISSA_OFFSET: usize = 7;
/// Lokal target node reference, little-endian `u16` — payload bytes 4..6.
pub const TARGET_OFFSET: usize = 8;
/// Nibble anaphora (low nibble) — payload byte 6.
pub const ANAPHORA_OFFSET: usize = 10;
/// Temporal chain offset, `i8` — payload byte 7.
pub const TEMPORAL_OFFSET: usize = 11;
/// W-slot (low 6 bits) | causal topology RAW (high 2 bits) — payload byte 8.
pub const WSLOT_TOPOLOGY_OFFSET: usize = 12;
/// Reasoning band RAW (low 3 bits) | reserved — payload byte 9.
pub const BAND_OFFSET: usize = 13;

const TOPOLOGY_SHIFT: u32 = 6;
const TOPOLOGY_MASK: u8 = 0b11;
const BAND_MASK: u8 = 0b111;
const WSLOT_MASK: u8 = 0x3F;
const KAUSAL_MASK: u8 = 0b111;

/// The 2-bit causal-topology vocabulary — the WIRE mirror of
/// `causal_edge::layout::CausalTopology` (same ordinals, same names; fused by
/// `cache::assertion_wire_parity` in the planner). *The shape of the causal
/// connection*: what kind of causal hole an edge is
/// (`entropy-closure-causal-ground-v1` §4b).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AssertionTopology {
    /// Direct causal edge, no intermediates.
    #[default]
    Direct = 0,
    /// Indirect, with known/named intermediate nodes on the causal path.
    IndirectKnownIntermediates = 1,
    /// Indirect, but the intermediate nodes are unknown/unnamed.
    IndirectUnknownIntermediates = 2,
    /// Topology not established — the unresolved causal hole.
    Unknown = 3,
}

impl AssertionTopology {
    /// Every ordinal, in wire order.
    pub const ALL: [AssertionTopology; 4] = [
        AssertionTopology::Direct,
        AssertionTopology::IndirectKnownIntermediates,
        AssertionTopology::IndirectUnknownIntermediates,
        AssertionTopology::Unknown,
    ];

    /// Decode the 2-bit field (only the low two bits are read).
    #[inline]
    #[must_use]
    pub const fn from_bits_2(v: u8) -> Self {
        match v & TOPOLOGY_MASK {
            0 => AssertionTopology::Direct,
            1 => AssertionTopology::IndirectKnownIntermediates,
            2 => AssertionTopology::IndirectUnknownIntermediates,
            _ => AssertionTopology::Unknown,
        }
    }

    /// Encode to the 2-bit field.
    #[inline]
    #[must_use]
    pub const fn to_bits_2(self) -> u8 {
        self as u8
    }

    /// The wire label — what a G11 consumer prints, never re-derives.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            AssertionTopology::Direct => "Direct",
            AssertionTopology::IndirectKnownIntermediates => "IndirectKnownIntermediates",
            AssertionTopology::IndirectUnknownIntermediates => "IndirectUnknownIntermediates",
            AssertionTopology::Unknown => "Unknown",
        }
    }
}

/// The 3-bit reasoning/assertion-band vocabulary — the WIRE mirror of
/// `causal_edge::layout::ReasoningBand` (same ordinals, same names; fused in
/// the planner). *The level of ASSERTION, Tarski permission*: what kind of
/// candidate assertion may bridge a hole — `Relation` → `Causal` is
/// relates-to → *causes* (`dismech_evidence::DISMECH_PREDICATES`, `0x90`).
/// Never Tarski derivation depth (`Belief.rung`), never `RungLevel`
/// (`E-RUNG-BAND-AND-PLASTICITY-ARE-THREE-AXES-NEVER-ONE-LEVEL-FIELD-1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum AssertionBand {
    /// Surface-level reasoning.
    #[default]
    Surface = 0,
    /// Association-level reasoning.
    Association = 1,
    /// Relation-level reasoning — "relates to".
    Relation = 2,
    /// Causal-level reasoning — "causes".
    Causal = 3,
    /// Counterfactual reasoning context.
    Counterfactual = 4,
    /// Perspective / decentration reasoning.
    Perspective = 5,
    /// Meta-cognitive reasoning (about reasoning / evidence / revision).
    Meta = 6,
    /// Highest ordinal in this band. Mechanical only.
    Transcendent = 7,
}

impl AssertionBand {
    /// Every ordinal, in wire order.
    pub const ALL: [AssertionBand; 8] = [
        AssertionBand::Surface,
        AssertionBand::Association,
        AssertionBand::Relation,
        AssertionBand::Causal,
        AssertionBand::Counterfactual,
        AssertionBand::Perspective,
        AssertionBand::Meta,
        AssertionBand::Transcendent,
    ];

    /// Decode the 3-bit field (only the low three bits are read).
    #[inline]
    #[must_use]
    pub const fn from_bits_3(v: u8) -> Self {
        match v & BAND_MASK {
            0 => AssertionBand::Surface,
            1 => AssertionBand::Association,
            2 => AssertionBand::Relation,
            3 => AssertionBand::Causal,
            4 => AssertionBand::Counterfactual,
            5 => AssertionBand::Perspective,
            6 => AssertionBand::Meta,
            _ => AssertionBand::Transcendent,
        }
    }

    /// Encode to the 3-bit field.
    #[inline]
    #[must_use]
    pub const fn to_bits_3(self) -> u8 {
        self as u8
    }

    /// The wire label — what a G11 consumer prints, never re-derives.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            AssertionBand::Surface => "Surface",
            AssertionBand::Association => "Association",
            AssertionBand::Relation => "Relation",
            AssertionBand::Causal => "Causal",
            AssertionBand::Counterfactual => "Counterfactual",
            AssertionBand::Perspective => "Perspective",
            AssertionBand::Meta => "Meta",
            AssertionBand::Transcendent => "Transcendent",
        }
    }
}

/// The 16-byte assertion wire — `classid(4, LE u32) | V3 edge payload(12)`.
///
/// `repr(transparent)` over the byte array: the in-memory image IS the wire
/// image, so a G11 host reads it with zero decode (`to_le_bytes` is a copy of
/// the bytes, not a serialization). Byte order is the CONTRACT's, never the
/// host's — the `classid` and `target` integers are little-endian by
/// definition here, whatever the host's native order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(transparent)]
pub struct AssertionWire([u8; ASSERTION_WIRE_BYTES]);

const _: () = assert!(core::mem::size_of::<AssertionWire>() == ASSERTION_WIRE_BYTES);
const _: () = assert!(core::mem::align_of::<AssertionWire>() == 1);
const _: () = assert!(BAND_OFFSET < ASSERTION_WIRE_BYTES);

impl AssertionWire {
    /// The canonical LE image, as bytes. Identity on the representation.
    #[inline]
    #[must_use]
    pub const fn from_le_bytes(b: [u8; ASSERTION_WIRE_BYTES]) -> Self {
        Self(b)
    }

    /// The canonical LE image, as bytes. Identity on the representation.
    #[inline]
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; ASSERTION_WIRE_BYTES] {
        self.0
    }

    /// Borrow the LE image (zero-copy; what an envelope column hands out).
    #[inline]
    #[must_use]
    pub const fn as_le_bytes(&self) -> &[u8; ASSERTION_WIRE_BYTES] {
        &self.0
    }

    /// Assemble from the key classid and the 12-byte V3 edge payload
    /// (`CausalEdgeV3::to_le_bytes()` on the producer side).
    #[must_use]
    pub const fn from_parts(classid: u32, payload: [u8; 12]) -> Self {
        let c = classid.to_le_bytes();
        let mut b = [0u8; ASSERTION_WIRE_BYTES];
        b[0] = c[0];
        b[1] = c[1];
        b[2] = c[2];
        b[3] = c[3];
        let mut i = 0;
        while i < 12 {
            b[4 + i] = payload[i];
            i += 1;
        }
        Self(b)
    }

    /// The key classid (LE `u32`) — the address whose `ClassView` declares how
    /// the tail bits are read ([`crate::class_view::ClassView::band_reading`]).
    #[inline]
    #[must_use]
    pub const fn classid(self) -> u32 {
        u32::from_le_bytes([self.0[0], self.0[1], self.0[2], self.0[3]])
    }

    /// The 12-byte V3 edge payload (what `CausalEdgeV3::from_le_bytes` takes).
    #[must_use]
    pub const fn payload(self) -> [u8; 12] {
        let mut p = [0u8; 12];
        let mut i = 0;
        while i < 12 {
            p[i] = self.0[4 + i];
            i += 1;
        }
        p
    }

    /// NARS frequency, `255 = 1.0`.
    #[inline]
    #[must_use]
    pub const fn frequency_u8(self) -> u8 {
        self.0[FREQUENCY_OFFSET]
    }

    /// NARS confidence, `255 = 1.0`.
    #[inline]
    #[must_use]
    pub const fn confidence_u8(self) -> u8 {
        self.0[CONFIDENCE_OFFSET]
    }

    /// Pearl 2³ causal-mask bits (3 bits: S/P/O planes) — the Pearl projection.
    #[inline]
    #[must_use]
    pub const fn causal_mask_bits(self) -> u8 {
        self.0[KAUSAL_OFFSET] & KAUSAL_MASK
    }

    /// Direction triad (3 bits).
    #[inline]
    #[must_use]
    pub const fn direction_bits(self) -> u8 {
        (self.0[KAUSAL_OFFSET] >> 3) & 0b111
    }

    /// The RAW signed 4-bit inference mantissa (−8..=7) — provenance/type
    /// grammar, never half of the truth value.
    #[inline]
    #[must_use]
    pub const fn inference_mantissa(self) -> i8 {
        let lo = self.0[MANTISSA_OFFSET] & 0x0F;
        if lo >= 8 {
            lo as i8 - 16
        } else {
            lo as i8
        }
    }

    /// Plasticity bits (3 bits).
    #[inline]
    #[must_use]
    pub const fn plasticity_bits(self) -> u8 {
        (self.0[MANTISSA_OFFSET] >> 4) & 0b111
    }

    /// The Lokal target node reference (LE `u16`) — the proposition reference:
    /// the node whose CAM-PQ facet IS this edge's SPO.
    #[inline]
    #[must_use]
    pub const fn target(self) -> u16 {
        u16::from_le_bytes([self.0[TARGET_OFFSET], self.0[TARGET_OFFSET + 1]])
    }

    /// W-slot: witness corpus root handle (6 bits, 0 = none).
    #[inline]
    #[must_use]
    pub const fn w_slot(self) -> u8 {
        self.0[WSLOT_TOPOLOGY_OFFSET] & WSLOT_MASK
    }

    /// The RAW 2-bit truth/topology ordinal. Raw on purpose: which lens it
    /// was written through is the class's declaration, not the bytes'.
    #[inline]
    #[must_use]
    pub const fn topology_raw(self) -> u8 {
        (self.0[WSLOT_TOPOLOGY_OFFSET] >> TOPOLOGY_SHIFT) & TOPOLOGY_MASK
    }

    /// The RAW 3-bit band ordinal. Raw on purpose, as above.
    #[inline]
    #[must_use]
    pub const fn band_raw(self) -> u8 {
        self.0[BAND_OFFSET] & BAND_MASK
    }

    /// **The defining read.** Project the wire into a complete
    /// [`AssertionView`] under the class's declared reading and the caller's
    /// asserted provenance. Refuses — never defaults — when:
    ///
    /// - provenance is not trusted ([`BandReadError::UnknownProvenance`]),
    /// - the class declared the `Trust` lens for the 2-bit field
    ///   ([`BandReadError::LensMismatch`] — the defining coordinate is
    ///   topology; a Trust-lensed class has no topology to assert),
    /// - the class declared no band ([`BandReadError::BandAbsent`]).
    ///
    /// Check order is doctrine (D-ACR-7): provenance before lens before
    /// presence. On `Ok`, every coordinate of the assertion is present; two
    /// views are `==` only if the assertions are the same claim.
    pub fn read(
        self,
        declared: BandReading,
        provenance: EdgeProvenance,
    ) -> Result<AssertionView, BandReadError> {
        let topology_raw =
            declared.project_truth(TruthLens::Topology, self.topology_raw(), provenance)?;
        let band_raw = declared.project_band(self.band_raw(), provenance)?;
        Ok(AssertionView {
            classid: self.classid(),
            target: self.target(),
            causal_mask_bits: self.causal_mask_bits(),
            frequency: self.frequency_u8(),
            confidence: self.confidence_u8(),
            topology: AssertionTopology::from_bits_2(topology_raw),
            band: AssertionBand::from_bits_3(band_raw),
            witness: declared.witness,
            w_slot: self.w_slot(),
        })
    }

    /// The RAW 2-bit ordinal under whichever lens the class declared — for a
    /// consumer that holds the edge crate and projects through its own enum
    /// (`TrustTexture` for `Trust`, `CausalTopology` for `Topology`). Still
    /// refuses on untrusted provenance or a lens mismatch; never a default.
    pub fn read_truth_raw(
        self,
        declared: BandReading,
        requested: TruthLens,
        provenance: EdgeProvenance,
    ) -> Result<u8, BandReadError> {
        declared.project_truth(requested, self.topology_raw(), provenance)
    }
}

/// A complete assertion, read from the wire under a declared lens and an
/// asserted provenance. Every field is a coordinate of the claim; none is
/// optional metadata. `==` is claim identity: the aliasing pair is `!=`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssertionView {
    /// The key classid the wire arrived under.
    pub classid: u32,
    /// Proposition reference — the node whose CAM-PQ facet is the SPO.
    pub target: u16,
    /// Pearl projection (S/P/O plane mask, 3 bits).
    pub causal_mask_bits: u8,
    /// NARS frequency, `255 = 1.0`.
    pub frequency: u8,
    /// NARS confidence, `255 = 1.0`.
    pub confidence: u8,
    /// The shape of the causal connection.
    pub topology: AssertionTopology,
    /// The level of assertion (Tarski permission).
    pub band: AssertionBand,
    /// Which witness carrier discriminates evidence-kind for this class
    /// (declared; F5: the band grades, the witness discriminates).
    pub witness: WitnessKind,
    /// Witness corpus root handle (6 bits, 0 = none).
    pub w_slot: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wire with the given `(f, c)`, topology and band; everything else
    /// held at a fixed non-zero pattern so a positional slip cannot hide.
    fn wire(f: u8, c: u8, topo: AssertionTopology, band: AssertionBand) -> AssertionWire {
        let mut p = [0u8; 12];
        p[0] = f;
        p[1] = c;
        p[2] = 0b101 | (0b011 << 3); // mask SO, direction 3
        p[3] = 0x0D | (0b010 << 4); // mantissa −3, plasticity 2
        p[4] = 0x34; // target 0x1234 LE
        p[5] = 0x12;
        p[6] = 0x02; // anaphora +2
        p[7] = 0xFE; // temporal −2
        p[8] = 0x2A | (topo.to_bits_2() << 6); // w_slot 42 | topology
        p[9] = band.to_bits_3();
        AssertionWire::from_parts(0x0902_0011, p)
    }

    fn topology_present() -> BandReading {
        BandReading {
            truth_lens: TruthLens::Topology,
            band: BandPresence::Present,
            witness: WitnessKind::CausalFacet,
        }
    }

    #[test]
    fn le_round_trip_is_identity_and_classid_is_little_endian() {
        let w = wire(192, 217, AssertionTopology::Direct, AssertionBand::Surface);
        assert_eq!(AssertionWire::from_le_bytes(w.to_le_bytes()), w);
        let b = w.to_le_bytes();
        assert_eq!(
            &b[0..4],
            &[0x11, 0x00, 0x02, 0x09],
            "classid must be LE on the wire"
        );
        assert_eq!(w.classid(), 0x0902_0011);
        assert_eq!(w.target(), 0x1234, "target must be LE on the wire");
        assert_eq!(w.payload()[8] & 0x3F, 42);
    }

    #[test]
    fn every_coordinate_reads_from_its_documented_position() {
        let w = wire(
            192,
            217,
            AssertionTopology::IndirectKnownIntermediates,
            AssertionBand::Causal,
        );
        assert_eq!(w.frequency_u8(), 192);
        assert_eq!(w.confidence_u8(), 217);
        assert_eq!(w.causal_mask_bits(), 0b101);
        assert_eq!(w.direction_bits(), 0b011);
        assert_eq!(w.inference_mantissa(), -3);
        assert_eq!(w.plasticity_bits(), 2);
        assert_eq!(w.w_slot(), 42);
        assert_eq!(w.topology_raw(), 1);
        assert_eq!(w.band_raw(), 3);
    }

    /// The module's own falsifier: same `(f, c)`, different truths.
    #[test]
    fn the_aliasing_pair_reads_to_different_assertions_and_stays_distinct_on_the_wire() {
        let a = wire(
            192,
            217,
            AssertionTopology::IndirectUnknownIntermediates,
            AssertionBand::Relation,
        );
        let b = wire(
            192,
            217,
            AssertionTopology::IndirectKnownIntermediates,
            AssertionBand::Causal,
        );
        // Anti-vacuity: the pair differs ONLY in the two defining fields —
        // exactly bits 6-7 of the wslot/topology byte and bits 0-2 of the band byte.
        let (ba, bb) = (a.to_le_bytes(), b.to_le_bytes());
        let diff: Vec<(usize, u8)> = (0..ASSERTION_WIRE_BYTES)
            .filter(|&i| ba[i] != bb[i])
            .map(|i| (i, ba[i] ^ bb[i]))
            .collect();
        assert_eq!(
            diff,
            vec![(WSLOT_TOPOLOGY_OFFSET, 0b11 << 6), (BAND_OFFSET, 0b001)],
            "the pair must differ only in topology and band bits"
        );
        let va = a
            .read(topology_present(), EdgeProvenance::V3Register)
            .unwrap();
        let vb = b
            .read(topology_present(), EdgeProvenance::V3Register)
            .unwrap();
        assert_eq!((va.frequency, va.confidence), (vb.frequency, vb.confidence));
        assert_ne!(
            va, vb,
            "epistemic aliasing: identical (f,c) must not be one claim"
        );
        assert_eq!(va.topology, AssertionTopology::IndirectUnknownIntermediates);
        assert_eq!(va.band, AssertionBand::Relation);
        assert_eq!(vb.topology, AssertionTopology::IndirectKnownIntermediates);
        assert_eq!(vb.band, AssertionBand::Causal);
        // Round trip through the wire keeps them distinct.
        let a2 = AssertionWire::from_le_bytes(a.to_le_bytes());
        let b2 = AssertionWire::from_le_bytes(b.to_le_bytes());
        assert_ne!(a2, b2);
        assert_eq!(
            a2.read(topology_present(), EdgeProvenance::V3Register)
                .unwrap(),
            va
        );
        assert_eq!(
            b2.read(topology_present(), EdgeProvenance::V3Register)
                .unwrap(),
            vb
        );
    }

    /// The silent twin: same claim ⇒ same view.
    #[test]
    fn identical_wires_read_identically() {
        let a = wire(10, 20, AssertionTopology::Unknown, AssertionBand::Meta);
        let b = wire(10, 20, AssertionTopology::Unknown, AssertionBand::Meta);
        assert_eq!(a, b);
        assert_eq!(
            a.read(topology_present(), EdgeProvenance::V2Stamped)
                .unwrap(),
            b.read(topology_present(), EdgeProvenance::V2Stamped)
                .unwrap()
        );
    }

    #[test]
    fn unstated_provenance_refuses_before_any_lens_question() {
        let w = wire(1, 2, AssertionTopology::Direct, AssertionBand::Causal);
        // Even a fully-declared class refuses on Unknown / V1Legacy.
        assert_eq!(
            w.read(topology_present(), EdgeProvenance::Unknown),
            Err(BandReadError::UnknownProvenance)
        );
        assert_eq!(
            w.read(topology_present(), EdgeProvenance::V1Legacy),
            Err(BandReadError::UnknownProvenance)
        );
        // Default provenance is Unknown — the zero-fallback refuses.
        assert_eq!(
            w.read(topology_present(), EdgeProvenance::default()),
            Err(BandReadError::UnknownProvenance)
        );
    }

    #[test]
    fn a_trust_lensed_class_has_no_topology_to_assert_and_refuses() {
        let w = wire(1, 2, AssertionTopology::Direct, AssertionBand::Causal);
        let trust_declared = BandReading {
            truth_lens: TruthLens::Trust,
            band: BandPresence::Present,
            witness: WitnessKind::None,
        };
        assert_eq!(
            w.read(trust_declared, EdgeProvenance::V2Stamped),
            Err(BandReadError::LensMismatch {
                declared: TruthLens::Trust,
                requested: TruthLens::Topology,
            })
        );
        // ...but the raw ordinal IS readable under the lens the class declared.
        assert_eq!(
            w.read_truth_raw(trust_declared, TruthLens::Trust, EdgeProvenance::V2Stamped),
            Ok(0)
        );
        // The zero-fallback (undeclared class) declares Trust + Absent: refuses.
        assert!(w
            .read(BandReading::ZERO_FALLBACK, EdgeProvenance::V2Stamped)
            .is_err());
    }

    #[test]
    fn an_absent_band_refuses_and_never_reads_as_surface() {
        let w = wire(1, 2, AssertionTopology::Direct, AssertionBand::Surface);
        let no_band = BandReading {
            truth_lens: TruthLens::Topology,
            band: BandPresence::Absent,
            witness: WitnessKind::None,
        };
        assert_eq!(
            w.read(no_band, EdgeProvenance::V2Stamped),
            Err(BandReadError::BandAbsent),
            "Surface(0) lookalike must be refused, not returned"
        );
        assert_eq!(
            w.read(topology_present(), EdgeProvenance::V2Stamped)
                .unwrap()
                .band,
            AssertionBand::Surface
        );
    }

    #[test]
    fn the_two_vocabularies_are_bijections_on_their_bits_with_distinct_labels() {
        for (i, t) in AssertionTopology::ALL.iter().enumerate() {
            assert_eq!(t.to_bits_2() as usize, i);
            assert_eq!(AssertionTopology::from_bits_2(i as u8), *t);
        }
        for (i, b) in AssertionBand::ALL.iter().enumerate() {
            assert_eq!(b.to_bits_3() as usize, i);
            assert_eq!(AssertionBand::from_bits_3(i as u8), *b);
        }
        // High bits are ignored, never aliased into a different ordinal.
        assert_eq!(
            AssertionTopology::from_bits_2(0b1111_1101),
            AssertionTopology::IndirectKnownIntermediates
        );
        assert_eq!(
            AssertionBand::from_bits_3(0b1111_1011),
            AssertionBand::Causal
        );
        let mut tl: Vec<&str> = AssertionTopology::ALL.iter().map(|t| t.label()).collect();
        tl.dedup();
        assert_eq!(tl.len(), 4);
        let mut bl: Vec<&str> = AssertionBand::ALL.iter().map(|b| b.label()).collect();
        bl.dedup();
        assert_eq!(bl.len(), 8);
        // The wire label is the Debug name — one vocabulary, not two spellings.
        for t in AssertionTopology::ALL {
            assert_eq!(format!("{t:?}"), t.label());
        }
        for b in AssertionBand::ALL {
            assert_eq!(format!("{b:?}"), b.label());
        }
    }

    #[test]
    fn schema_and_width_are_pinned() {
        assert_eq!(ASSERTION_WIRE_SCHEMA, 1);
        assert_eq!(ASSERTION_WIRE_BYTES, 16);
        assert_eq!(core::mem::size_of::<AssertionWire>(), 16);
        assert_eq!(BAND_OFFSET, 13);
        assert_eq!(WSLOT_TOPOLOGY_OFFSET, 12);
    }
}
