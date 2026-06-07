//! Signed discrete reading-crystal: grammar / discourse / episodic meaning field.
//!
//! ## Why this is NOT a float path
//!
//! `holograph::sentence_crystal::SemanticCrystal` already provides the correct
//! integer-first architecture (char n-gram hashing → bit rotation → majority
//! bundling → 16Kbit fingerprint). This module provides the **grammar/discourse
//! axis** that DeepNSM emits *before* the holograph fingerprint is computed.
//!
//! The two layers are complementary, not competing:
//!
//! ```text
//! DeepNSM step() → EpisodicSpoFrame
//!                → P64MeaningField      (8-lane grammar/discourse byte field)
//!                → Crystal4096          (signed 3-axis reading coordinate)
//!
//! holograph SemanticCrystal → BitpackedVector (16Kbit fingerprint)
//!
//! AriGraph basin ← XOR-bind(Crystal4096, BitpackedVector) → tombstone witness
//! ```
//!
//! ## Signed nibble axes
//!
//! A **`SignedOffset4`** encodes a local reading offset in the range −7..=+7
//! (14 values) plus one overflow/basin sentinel (15):
//!
//! ```text
//! 0..=14  → signed offset (value - 7): 0=-7, 7=0, 14=+7
//! 15      → overflow / basin-change / unknown
//! ```
//!
//! Three axes packed into 12 bits give **4096 cells** — the same cardinality as
//! the P4096 palette codebook, enabling O(1) lookup against it.
//!
//! ## Three axes
//!
//! | Axis | Name | What it encodes |
//! |------|------|-----------------|
//! | X | sentence offset | distance from current sentence (±5 window = ±5) |
//! | Y | clause offset   | intra-sentence clause position (±3 sub-clauses) |
//! | Z | basin delta     | drift from the prior basin anchor (±7 SPO hops) |
//!
//! All three stay in the −7..+7 band during normal reading; overflow (15) fires
//! on basin transitions, topic shifts, or coreference chains exceeding the window.
//!
//! ## P64 meaning field
//!
//! `P64MeaningField` carries the 8-lane grammar/semantic signal from `Cam64`
//! augmented with the NSM prime contribution. It is the same u64 substrate as
//! `Cam64` but emphasised as the *meaning-field* output (distinct semantic role):
//!
//! ```text
//! Cam64  = reading-state locality key  (NOT the truth)
//! P64    = meaning-field projection    (grammar + NSM composite)
//! ```
//!
//! They share the same bit width but carry different information and must not be
//! fused without an explicit projection step.

use crate::cam64::Cam64;
use crate::spo::NO_ROLE;

// ── HorizonPolarity (v2 stub) ─────────────────────────────────────────────────

/// Epistemic provenance of a reading horizon offset.
///
/// `SignedOffset4` encodes **where** (signed local distance −7..+7 + overflow).
/// `HorizonPolarity` encodes **why/how known** — the epistemic status of that
/// position. The two are orthogonal:
///
/// ```text
/// +1 could mean:
///   next sentence physically          (ConfirmedBackward after a step)
///   expected referent not yet seen    (ExpectedForward from left-corner trigger)
///   right-context memo already known  (InferredRight from inverse/Pika pass)
///   basin continuation projected      (Basin)
/// ```
///
/// In v1, the expectation information lives in `SentenceWindow` via
/// `ExpectedReason` and `push_expected()`. `HorizonPolarity` is the v2 type
/// that generalises this to all three reading directions (backward confirmed,
/// forward expected, inverse/right inferred) so they can be tracked uniformly
/// in `Crystal4096` metadata or a P64 lane without stealing nibble values from
/// `SignedOffset4`.
///
/// **Do NOT fold polarity into the 4-bit offset.** The clean split is:
/// - `SignedOffset4` = compact ABI, always means signed distance, nothing else.
/// - `HorizonPolarity` = caller-side metadata attached to the coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum HorizonPolarity {
    /// Ordinary prior context: backward from the current sentence. Confirmed by
    /// the sentence ring.
    #[default]
    ConfirmedBackward = 0,
    /// Left-corner forward prediction: antecedent/subject expected but not yet
    /// confirmed. Created by `push_expected()` on `SentenceWindow`.
    ExpectedForward = 1,
    /// Pika-style right-context / inverse pass: memo available from later clause
    /// material (right-to-left prepopulation). V2 — not yet wired in v1.
    InferredRight = 2,
    /// Offset is outside the local ±7 window; use basin/archetype lookup instead.
    BasinOverflow = 3,
}

impl HorizonPolarity {
    /// Pack into 2 bits (fits in any spare lane bits or metadata field).
    #[inline]
    pub fn to_bits(self) -> u8 { self as u8 }

    /// Unpack from 2 bits. Values > 3 map to `BasinOverflow`.
    #[inline]
    pub fn from_bits(b: u8) -> Self {
        match b & 0x3 {
            0 => Self::ConfirmedBackward,
            1 => Self::ExpectedForward,
            2 => Self::InferredRight,
            _ => Self::BasinOverflow,
        }
    }

    /// True if this position is known from evidence (not a prediction).
    #[inline]
    pub fn is_confirmed(self) -> bool {
        matches!(self, Self::ConfirmedBackward)
    }

    /// True if this position is a forward prediction that may not materialise.
    #[inline]
    pub fn is_prediction(self) -> bool {
        matches!(self, Self::ExpectedForward | Self::InferredRight)
    }
}

// ── SignedOffset4 ─────────────────────────────────────────────────────────────

/// A signed 4-bit reading offset: values 0..=14 encode −7..=+7; 15 = overflow.
///
/// Encoding: `raw = offset + 7` for offset in −7..=+7.
/// `raw = 15` is the overflow/basin-change sentinel.
///
/// ## Epistemic polarity is NOT encoded here
///
/// `SignedOffset4` encodes **signed local distance only**. It intentionally
/// does not encode epistemic provenance (confirmed backward context vs
/// left-corner forward expectation vs inverse/right-context prepopulation).
/// Those distinctions are carried by `HorizonPolarity` (v2) or by
/// `ExpectedReason` + `SentenceWindow::push_expected()` (v1). Callers that
/// need to distinguish "physically +1 sentence ahead" from "predicted +1
/// referent not yet seen" must track `HorizonPolarity` separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct SignedOffset4(pub u8);

impl SignedOffset4 {
    /// Overflow / basin-change / unknown sentinel.
    pub const OVERFLOW: Self = Self(15);

    /// Zero offset (raw value 7).
    pub const ZERO: Self = Self(7);

    /// Minimum representable offset (−7, raw 0).
    pub const MIN: Self = Self(0);

    /// Maximum representable offset (+7, raw 14).
    pub const MAX: Self = Self(14);

    /// Encode a signed offset. Clamps to −7..=+7; values outside produce OVERFLOW.
    #[inline]
    pub fn from_offset(offset: i8) -> Self {
        if offset < -7 || offset > 7 {
            Self::OVERFLOW
        } else {
            Self((offset + 7) as u8)
        }
    }

    /// Decode to a signed offset. Returns `None` for the overflow sentinel.
    #[inline]
    pub fn to_offset(self) -> Option<i8> {
        if self.0 == 15 {
            None
        } else {
            Some(self.0 as i8 - 7)
        }
    }

    /// Raw nibble value (0..=15).
    #[inline]
    pub fn raw(self) -> u8 {
        self.0
    }

    /// True if this is the overflow/basin sentinel.
    #[inline]
    pub fn is_overflow(self) -> bool {
        self.0 == 15
    }
}

impl Default for SignedOffset4 {
    fn default() -> Self {
        Self::ZERO
    }
}

// ── Crystal4096 ───────────────────────────────────────────────────────────────

/// A 12-bit signed reading crystal coordinate: three `SignedOffset4` axes.
///
/// Layout (little-endian nibble packing):
/// ```text
/// bits  0.. 3 = X (sentence offset)
/// bits  4.. 7 = Y (clause offset)
/// bits  8..11 = Z (basin delta)
/// bits 12..15 = reserved (always 0)
/// ```
///
/// 4096 valid cells (bit patterns 0x000..=0xFFF), directly addressable in the
/// P4096 palette codebook.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Crystal4096(pub u16);

impl Crystal4096 {
    /// Construct from three signed axes.
    #[inline]
    pub fn new(x: SignedOffset4, y: SignedOffset4, z: SignedOffset4) -> Self {
        Self(x.raw() as u16 | ((y.raw() as u16) << 4) | ((z.raw() as u16) << 8))
    }

    /// Extract the X axis (sentence offset).
    #[inline]
    pub fn x(self) -> SignedOffset4 {
        SignedOffset4((self.0 & 0xF) as u8)
    }

    /// Extract the Y axis (clause offset).
    #[inline]
    pub fn y(self) -> SignedOffset4 {
        SignedOffset4(((self.0 >> 4) & 0xF) as u8)
    }

    /// Extract the Z axis (basin delta).
    #[inline]
    pub fn z(self) -> SignedOffset4 {
        SignedOffset4(((self.0 >> 8) & 0xF) as u8)
    }

    /// Raw 12-bit coordinate (bits 0-11 used, bits 12-15 always zero).
    #[inline]
    pub fn raw(self) -> u16 {
        self.0 & 0x0FFF
    }

    /// True if any axis is the overflow sentinel.
    #[inline]
    pub fn has_overflow(self) -> bool {
        self.x().is_overflow() || self.y().is_overflow() || self.z().is_overflow()
    }

    /// XOR two coordinates — used for binding/unbinding in holograph.
    #[inline]
    pub fn xor(self, other: Crystal4096) -> Crystal4096 {
        Crystal4096(self.0 ^ other.0)
    }

    /// Hamming-style distance: count differing nibbles (0, 1, 2, or 3).
    #[inline]
    pub fn nibble_distance(self, other: Crystal4096) -> u8 {
        let xor = self.0 ^ other.0;
        ((xor & 0x00F != 0) as u8)
            + ((xor & 0x0F0 != 0) as u8)
            + ((xor & 0xF00 != 0) as u8)
    }

    /// True if both coordinates are in the same basin (no axis overflows and
    /// nibble distance ≤ 1 — at most one axis shifted by one step).
    #[inline]
    pub fn same_basin(self, other: Crystal4096) -> bool {
        !self.has_overflow() && !other.has_overflow() && self.nibble_distance(other) <= 1
    }
}

// ── P64MeaningField ───────────────────────────────────────────────────────────

/// 8-lane grammar/semantic/discourse meaning field.
///
/// Derived from `Cam64` + NSM prime mask, this is the *meaning-field projection*
/// — the composite signal that summarises what this sentence is *about*, in the
/// grammar of the reading state machine.
///
/// **`P64MeaningField` is NOT `Cam64`.** `Cam64` is a reading-state locality key
/// (fast index, not truth). `P64MeaningField` is the output of the DeepNSM
/// grammar lens onto the P64 meaning lattice — a different interpretive layer.
///
/// Lane layout (same physical encoding as Cam64 but different semantic contract):
/// ```text
/// byte 0 — primary entity bucket  (vocabulary rank >> 5)
/// byte 1 — predicate bucket
/// byte 2 — object bucket (0 if absent)
/// byte 3 — morphology / NSM prime composite (low byte)
/// byte 4 — NSM prime composite (high byte, top-16 primes)
/// byte 5 — discourse / coreference marker
/// byte 6 — causal / temporal / episodic marker
/// byte 7 — basin / novelty / wisdom / epiphany marker
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct P64MeaningField {
    pub bits: u64,
}

impl P64MeaningField {
    /// Construct from a `Cam64` locality code and the NSM prime mask.
    ///
    /// The NSM prime mask (64-bit, up to 63 primes) is folded into lanes 3-4
    /// via XOR — this makes the meaning field sensitive to semantic prime
    /// coverage without losing the grammar-lane signals.
    #[inline]
    pub fn from_cam64_and_nsm(cam: Cam64, nsm_prime_mask: u64) -> Self {
        // Fold low 16 bits of NSM mask into lanes 3-4 (the morphology lanes).
        let nsm_low  = (nsm_prime_mask & 0xFF) as u64;
        let nsm_high = ((nsm_prime_mask >> 8) & 0xFF) as u64;
        let nsm_xor  = nsm_low | (nsm_high << 8); // into bits 24-39

        Self {
            bits: cam.raw() ^ (nsm_xor << 24),
        }
    }

    /// Extract one meaning-field lane (0-7).
    #[inline]
    pub fn lane(self, i: usize) -> u8 {
        debug_assert!(i < 8);
        (self.bits >> (i * 8)) as u8
    }

    /// XOR bind with another meaning field (VSA binding).
    #[inline]
    pub fn bind(self, other: P64MeaningField) -> P64MeaningField {
        P64MeaningField { bits: self.bits ^ other.bits }
    }

    /// Popcount — number of active bits in the meaning field.
    #[inline]
    pub fn popcount(self) -> u32 {
        self.bits.count_ones()
    }

    /// Shared bits with another field (XNOR popcount = agreement measure).
    #[inline]
    pub fn agreement(self, other: P64MeaningField) -> u32 {
        64 - (self.bits ^ other.bits).count_ones()
    }

    /// Raw u64.
    #[inline]
    pub fn raw(self) -> u64 {
        self.bits
    }
}

// ── SignedSentenceCrystal ─────────────────────────────────────────────────────

/// The complete signed reading-crystal output for one sentence.
///
/// Emitted by `DeepNSM::step()` alongside `EpisodicSpoFrame`. Carries:
///
/// - `p64`: the grammar/semantic/discourse meaning field (8 × u8 lanes)
/// - `coord`: the signed 3-axis crystal coordinate (P4096 codebook key)
///
/// The `coord` is the bridge to the holograph fingerprint substrate:
/// `Crystal4096::raw()` is a direct index into the P4096 palette. The
/// holograph `BitpackedVector` for this sentence can be XOR-bound with
/// the cell prototype at that index for basin-aware resonance search.
///
/// ## Floats at the border
///
/// This struct is entirely integer. Quality annotations (confidence, novelty,
/// wisdom) remain as `f32` in `EpisodicSpoFrame` — they are boundary tools,
/// not the hot-path substrate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignedSentenceCrystal {
    /// Grammar / semantic / discourse meaning field (P64 lattice).
    pub p64: P64MeaningField,
    /// Signed 3-axis reading coordinate (P4096 codebook key, 12 bits valid).
    pub coord: Crystal4096,
}

impl SignedSentenceCrystal {
    /// Construct from a `Cam64`, NSM prime mask, and the three axis offsets.
    pub fn new(
        cam: Cam64,
        nsm_prime_mask: u64,
        sentence_offset: i8,
        clause_offset: i8,
        basin_delta: i8,
    ) -> Self {
        Self {
            p64: P64MeaningField::from_cam64_and_nsm(cam, nsm_prime_mask),
            coord: Crystal4096::new(
                SignedOffset4::from_offset(sentence_offset),
                SignedOffset4::from_offset(clause_offset),
                SignedOffset4::from_offset(basin_delta),
            ),
        }
    }

    /// True if this crystal and `other` are plausibly in the same reading basin.
    ///
    /// Combines P64 agreement (≥ 40 shared bits) with crystal coordinate
    /// proximity (nibble distance ≤ 1).
    #[inline]
    pub fn same_basin_as(&self, other: &SignedSentenceCrystal) -> bool {
        self.p64.agreement(other.p64) >= 40 && self.coord.same_basin(other.coord)
    }

    /// XOR bind two crystals — used for holograph integration.
    ///
    /// Binding combines both the meaning field and the coordinate so the result
    /// encodes the *relationship* between two reading positions, not either one
    /// alone. Pass the bound result to `holograph::XorBind` or use it as a
    /// lookup key in the P4096 codebook.
    #[inline]
    pub fn bind(&self, other: &SignedSentenceCrystal) -> SignedSentenceCrystal {
        SignedSentenceCrystal {
            p64: self.p64.bind(other.p64),
            coord: self.coord.xor(other.coord),
        }
    }
}

// ── Convenience: build from EpisodicSpoFrame fields ──────────────────────────

/// Build a `SignedSentenceCrystal` from the fields already present on an
/// `EpisodicSpoFrame` plus a sentence-window offset.
///
/// - `sentence_window_offset`: −5..+5 (from `EpisodicSpoFrame::sentence_window_offset`)
/// - `clause_idx`: 0-based clause position within the sentence (0=main, 1=first sub, …)
/// - `basin_hop_delta`: how many SPO hops this sentence is from the prior basin anchor
pub fn crystal_from_frame_context(
    cam64: Cam64,
    nsm_prime_mask: u64,
    sentence_window_offset: i8,
    clause_idx: u8,
    basin_hop_delta: i8,
) -> SignedSentenceCrystal {
    // Clause offset: clause_idx 0 = centre (0), 1 = +1, 2 = +2, capped at +3.
    let clause_offset = (clause_idx as i8).min(3);
    SignedSentenceCrystal::new(
        cam64,
        nsm_prime_mask,
        sentence_window_offset,
        clause_offset,
        basin_hop_delta,
    )
}

/// Derive the basin hop delta from two consecutive `Cam64` locality codes.
///
/// Uses the `continues_basin` predicate: if the transition continues the
/// prior basin, delta = 0; otherwise delta = +1 (or overflow if repeated
/// basin-breaks exceed the signed range).
pub fn basin_delta_from_cam(prev: Cam64, curr: Cam64) -> i8 {
    if curr.continues_basin(prev) { 0 } else { 1 }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cam64::Cam64;

    #[test]
    fn signed_offset4_encode_decode() {
        for v in -7i8..=7 {
            let s = SignedOffset4::from_offset(v);
            assert!(!s.is_overflow(), "offset {v} should not overflow");
            assert_eq!(s.to_offset(), Some(v));
        }
    }

    #[test]
    fn signed_offset4_overflow_outside_range() {
        assert_eq!(SignedOffset4::from_offset(-8), SignedOffset4::OVERFLOW);
        assert_eq!(SignedOffset4::from_offset(8),  SignedOffset4::OVERFLOW);
        assert!(SignedOffset4::OVERFLOW.is_overflow());
        assert_eq!(SignedOffset4::OVERFLOW.to_offset(), None);
    }

    #[test]
    fn signed_offset4_zero_is_seven() {
        assert_eq!(SignedOffset4::ZERO.raw(), 7);
        assert_eq!(SignedOffset4::ZERO.to_offset(), Some(0));
    }

    #[test]
    fn crystal4096_pack_unpack() {
        let x = SignedOffset4::from_offset(-3);
        let y = SignedOffset4::from_offset(0);
        let z = SignedOffset4::from_offset(5);
        let c = Crystal4096::new(x, y, z);
        assert_eq!(c.x(), x);
        assert_eq!(c.y(), y);
        assert_eq!(c.z(), z);
        assert_eq!(c.raw(), c.0 & 0x0FFF);
    }

    #[test]
    fn crystal4096_raw_fits_in_12_bits() {
        // All possible nibble combinations should fit in 4096.
        for raw in 0u16..=0x0FFF {
            let c = Crystal4096(raw);
            assert_eq!(c.raw(), raw);
        }
    }

    #[test]
    fn crystal4096_nibble_distance_same_is_zero() {
        let c = Crystal4096::new(
            SignedOffset4::ZERO,
            SignedOffset4::ZERO,
            SignedOffset4::ZERO,
        );
        assert_eq!(c.nibble_distance(c), 0);
    }

    #[test]
    fn crystal4096_nibble_distance_one_axis_differs() {
        let a = Crystal4096::new(SignedOffset4::from_offset(0), SignedOffset4::from_offset(0), SignedOffset4::from_offset(0));
        let b = Crystal4096::new(SignedOffset4::from_offset(1), SignedOffset4::from_offset(0), SignedOffset4::from_offset(0));
        assert_eq!(a.nibble_distance(b), 1);
    }

    #[test]
    fn crystal4096_same_basin_adjacent_coords() {
        let a = Crystal4096::new(SignedOffset4::ZERO, SignedOffset4::ZERO, SignedOffset4::ZERO);
        let b = Crystal4096::new(SignedOffset4::from_offset(1), SignedOffset4::ZERO, SignedOffset4::ZERO);
        assert!(a.same_basin(b));
    }

    #[test]
    fn crystal4096_different_basin_two_axes_differ() {
        let a = Crystal4096::new(SignedOffset4::ZERO, SignedOffset4::ZERO, SignedOffset4::ZERO);
        let b = Crystal4096::new(SignedOffset4::from_offset(3), SignedOffset4::from_offset(2), SignedOffset4::ZERO);
        assert!(!a.same_basin(b));
    }

    #[test]
    fn crystal4096_overflow_axis_is_not_same_basin() {
        let a = Crystal4096::new(SignedOffset4::ZERO, SignedOffset4::ZERO, SignedOffset4::ZERO);
        let b = Crystal4096::new(SignedOffset4::OVERFLOW, SignedOffset4::ZERO, SignedOffset4::ZERO);
        assert!(!a.same_basin(b));
    }

    #[test]
    fn crystal4096_xor_bind_unbind() {
        let a = Crystal4096(0x123);
        let b = Crystal4096(0x456);
        let bound = a.xor(b);
        assert_eq!(bound.xor(b), a); // XOR is self-inverse
    }

    #[test]
    fn p64_meaning_field_agreement_self() {
        let cam = Cam64::from_lanes([1, 2, 3, 4, 5, 6, 7, 8]);
        let p = P64MeaningField::from_cam64_and_nsm(cam, 0xDEAD_BEEF_0000_0000);
        assert_eq!(p.agreement(p), 64);
    }

    #[test]
    fn p64_meaning_field_agreement_differs_on_nsm() {
        let cam = Cam64::default();
        let p0 = P64MeaningField::from_cam64_and_nsm(cam, 0);
        let p1 = P64MeaningField::from_cam64_and_nsm(cam, 0xFFFF);
        // NSM folds into lanes 3-4 — agreement drops below 64.
        assert!(p0.agreement(p1) < 64);
    }

    #[test]
    fn signed_sentence_crystal_same_basin_nearby() {
        let cam = Cam64::from_lanes([10, 20, 30, 40, 50, 60, 70, 80]);
        let a = SignedSentenceCrystal::new(cam, 0, 0, 0, 0);
        let b = SignedSentenceCrystal::new(cam, 0, 0, 1, 0); // clause +1
        assert!(a.same_basin_as(&b));
    }

    #[test]
    fn signed_sentence_crystal_different_basin_far_coord() {
        let cam_a = Cam64::from_lanes([10, 20, 30, 40, 50, 60, 70, 80]);
        let cam_b = Cam64::from_lanes([200, 201, 202, 203, 204, 205, 206, 207]);
        let a = SignedSentenceCrystal::new(cam_a, 0, 0, 0, 0);
        let b = SignedSentenceCrystal::new(cam_b, 0xFFFF, -5, 3, 7);
        assert!(!a.same_basin_as(&b));
    }

    #[test]
    fn crystal_from_frame_context_zero_offsets() {
        let cam = Cam64::default();
        let c = crystal_from_frame_context(cam, 0, 0, 0, 0);
        assert_eq!(c.coord.x(), SignedOffset4::ZERO);
        assert_eq!(c.coord.y(), SignedOffset4::ZERO);
        assert_eq!(c.coord.z(), SignedOffset4::ZERO);
    }

    #[test]
    fn basin_delta_from_cam_same_basin() {
        let c = Cam64::from_lanes([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(basin_delta_from_cam(c, c), 0);
    }

    #[test]
    fn basin_delta_from_cam_different_basin() {
        let a = Cam64::from_raw(0x0000_0000_0000_0000);
        let b = Cam64::from_raw(0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(basin_delta_from_cam(a, b), 1);
    }
}
