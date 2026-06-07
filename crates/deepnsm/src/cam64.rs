//! 64-bit reading-state locality code (CAM64).
//!
//! **This is NOT semantic truth.** The `EpisodicSpoFrame` rows in
//! `episodic_spo` are the auditable witnesses. `Cam64` is a fast locality
//! key for:
//! - candidate prefetch
//! - relative-pronoun / coreference heuristics
//! - "does this sentence continue the previous story?" basin matching
//! - "does this sentence open a new basin?" detection
//!
//! The 64 bits encode 8 named lanes, one byte each.
//! The 256 values per lane give 4096 possible lane combinations,
//! which maps directly onto the CAM-PQ bucket space for O(1) lookup.
//!
//! ## Lane layout
//!
//! ```text
//! byte 0 — entity / subject state      (vocabulary-bucket of active subject)
//! byte 1 — predicate / action state    (vocabulary-bucket of active predicate)
//! byte 2 — object / complement state   (vocabulary-bucket of active object, 0 if absent)
//! byte 3 — morphology / tense / number / voice  (MorphFlags low byte)
//! byte 4 — clause structure / relative / subordination (MorphFlags high byte)
//! byte 5 — discourse / anaphora / referent stack depth + coreference flag
//! byte 6 — causal / temporal / conditional markers
//! byte 7 — episodic basin / novelty / wisdom / epiphany markers
//! ```
//!
//! Bytes 3-4 split `MorphFlags(u16)` across two lanes so all 14 morph bits are
//! represented without compression.

use crate::morphology::MorphFlags;
use crate::spo::{SpoTriple, NO_ROLE};

/// 64-bit reading-state locality code: 8 lanes × 8 bits.
///
/// Stored little-endian in a `u64`: lane 0 occupies bits 0-7, lane 7 bits 56-63.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Cam64(u64);

impl Cam64 {
    /// Construct from an explicit 8-byte lane array.
    #[inline]
    pub fn from_lanes(lanes: [u8; 8]) -> Self {
        let mut v = 0u64;
        for (i, &b) in lanes.iter().enumerate() {
            v |= (b as u64) << (i * 8);
        }
        Self(v)
    }

    /// Extract one lane (0-7).
    #[inline]
    pub fn lane(self, i: usize) -> u8 {
        debug_assert!(i < 8, "lane index out of range");
        (self.0 >> (i * 8)) as u8
    }

    /// Return a new `Cam64` with one lane replaced.
    #[inline]
    pub fn with_lane(self, i: usize, val: u8) -> Self {
        debug_assert!(i < 8, "lane index out of range");
        let mask = !(0xFFu64 << (i * 8));
        Self((self.0 & mask) | ((val as u64) << (i * 8)))
    }

    /// Raw u64 value.
    #[inline]
    pub fn raw(self) -> u64 { self.0 }

    /// Construct from raw u64.
    #[inline]
    pub fn from_raw(v: u64) -> Self { Self(v) }

    // ── Named lane accessors ─────────────────────────────────────────────────

    /// Vocabulary bucket of the active subject (lane 0).
    /// Bucket = rank >> 5 → 128 buckets of 32 adjacent vocabulary items each.
    pub fn entity_state(self)    -> u8 { self.lane(0) }
    /// Vocabulary bucket of the active predicate (lane 1).
    pub fn predicate_state(self) -> u8 { self.lane(1) }
    /// Vocabulary bucket of the active object, or 0 if absent (lane 2).
    pub fn object_state(self)    -> u8 { self.lane(2) }
    /// `MorphFlags` bits 0-7: tense, number, person, passive, negated (lane 3).
    pub fn morph_state(self)     -> u8 { self.lane(3) }
    /// `MorphFlags` bits 8-13: clause structure flags (lane 4).
    pub fn clause_state(self)    -> u8 { self.lane(4) }
    /// Discourse / anaphora: entity-stack depth (bits 0-6) + coreference flag (bit 7) (lane 5).
    pub fn discourse_state(self) -> u8 { self.lane(5) }
    /// Causal / temporal / conditional markers (lane 6).
    pub fn causal_state(self)    -> u8 { self.lane(6) }
    /// Episodic basin markers: novelty/entropy/epiphany flags (lane 7).
    pub fn basin_state(self)     -> u8 { self.lane(7) }

    // ── Construction helpers ─────────────────────────────────────────────────

    /// Build a `Cam64` from a resolved triple + morph flags + reading context.
    ///
    /// `entity_stack_depth` — number of active entities in the coreference stack (0-127).
    /// `coreference_resolved` — true if the subject was resolved from the entity stack.
    /// `has_temporal` — true if the triple carries a temporal marker.
    /// `novelty_high` — caller-supplied hint that this triple is novel (bit 0 of basin lane).
    pub fn from_triple(
        triple: &SpoTriple,
        morph: MorphFlags,
        entity_stack_depth: u8,
        coreference_resolved: bool,
        has_temporal: bool,
        novelty_high: bool,
    ) -> Self {
        // Lanes 0-2: vocabulary-bucket of each role (128 buckets of 32 ranks).
        // Adjacent vocabulary items share a bucket → helps basin-matching.
        let entity_lane   = (triple.subject() >> 5) as u8;
        let pred_lane     = (triple.predicate() >> 5) as u8;
        let obj_lane      = if triple.object() != NO_ROLE {
            (triple.object() >> 5) as u8
        } else {
            0
        };

        // Lanes 3-4: split MorphFlags across two bytes.
        let morph_bits    = morph.bits();
        let morph_lane    = (morph_bits & 0xFF) as u8;
        let clause_lane   = ((morph_bits >> 8) & 0xFF) as u8;

        // Lane 5: discourse — stack depth (bits 0-6) + coreference flag (bit 7).
        let depth_clamped = entity_stack_depth.min(127);
        let discourse_lane = depth_clamped | if coreference_resolved { 0x80 } else { 0 };

        // Lane 6: causal/temporal — bit 0 = temporal marker present.
        let causal_lane   = if has_temporal { 0x01u8 } else { 0x00 };

        // Lane 7: basin — bit 0 = novelty_high (v1 placeholder; epiphany/wisdom baked in v2).
        let basin_lane    = if novelty_high { 0x01u8 } else { 0x00 };

        Self::from_lanes([
            entity_lane, pred_lane, obj_lane,
            morph_lane, clause_lane,
            discourse_lane, causal_lane, basin_lane,
        ])
    }

    /// Return true if the entity bucket of `self` matches `other`.
    ///
    /// Used for basin-matching: "is this sentence in the same topic domain?"
    #[inline]
    pub fn same_entity_bucket(self, other: Cam64) -> bool {
        self.entity_state() == other.entity_state()
    }

    /// Return true if the discourse lane indicates a coreference was resolved.
    #[inline]
    pub fn has_coreference(self) -> bool {
        self.discourse_state() & 0x80 != 0
    }

    /// Entity stack depth encoded in the discourse lane (bits 0-6).
    #[inline]
    pub fn entity_stack_depth(self) -> u8 {
        self.discourse_state() & 0x7F
    }
}

impl core::fmt::Debug for Cam64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Cam64(entity={:#04x} pred={:#04x} obj={:#04x} morph={:#04x} \
             clause={:#04x} discourse={:#04x} causal={:#04x} basin={:#04x})",
            self.entity_state(), self.predicate_state(), self.object_state(),
            self.morph_state(), self.clause_state(), self.discourse_state(),
            self.causal_state(), self.basin_state(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::MorphFlags;
    use crate::spo::SpoTriple;

    #[test]
    fn lane_roundtrip() {
        let lanes = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let c = Cam64::from_lanes(lanes);
        for (i, &b) in lanes.iter().enumerate() {
            assert_eq!(c.lane(i), b, "lane {i}");
        }
    }

    #[test]
    fn with_lane_does_not_corrupt_others() {
        let c = Cam64::from_lanes([0xFF; 8]);
        let c2 = c.with_lane(3, 0x00);
        assert_eq!(c2.lane(3), 0x00);
        for i in [0, 1, 2, 4, 5, 6, 7] {
            assert_eq!(c2.lane(i), 0xFF, "lane {i} corrupted");
        }
    }

    #[test]
    fn from_triple_entity_bucket() {
        let t = SpoTriple::new(64, 96, 128); // bucket = rank >> 5
        let m = MorphFlags::default().set(MorphFlags::PRESENT);
        let c = Cam64::from_triple(&t, m, 3, false, false, false);
        assert_eq!(c.entity_state(), 64 >> 5);
        assert_eq!(c.predicate_state(), 96 >> 5);
        assert_eq!(c.object_state(), 128 >> 5);
    }

    #[test]
    fn from_triple_morph_split() {
        let t = SpoTriple::new(1, 2, 3);
        // Set a flag that lands in the high byte (RELATIVE_CLAUSE = bit 11)
        let m = MorphFlags::default()
            .set(MorphFlags::NEGATED)          // bit 9 → high byte bit 1
            .set(MorphFlags::RELATIVE_CLAUSE); // bit 11 → high byte bit 3
        let c = Cam64::from_triple(&t, m, 0, false, false, false);
        // morph_lane = low byte of flags
        assert_eq!(c.morph_state(), (m.bits() & 0xFF) as u8);
        // clause_lane = high byte
        assert_eq!(c.clause_state(), (m.bits() >> 8) as u8);
    }

    #[test]
    fn coreference_flag_in_discourse_lane() {
        let t = SpoTriple::new(1, 2, 3);
        let m = MorphFlags::default();
        let c_yes = Cam64::from_triple(&t, m, 5, true, false, false);
        let c_no  = Cam64::from_triple(&t, m, 5, false, false, false);
        assert!(c_yes.has_coreference());
        assert!(!c_no.has_coreference());
        assert_eq!(c_yes.entity_stack_depth(), 5);
        assert_eq!(c_no.entity_stack_depth(), 5);
    }

    #[test]
    fn temporal_sets_causal_bit0() {
        let t = SpoTriple::new(1, 2, 3);
        let m = MorphFlags::default();
        let c = Cam64::from_triple(&t, m, 0, false, true, false);
        assert_eq!(c.causal_state() & 0x01, 1);
    }

    #[test]
    fn same_entity_bucket_matching() {
        let t1 = SpoTriple::new(64, 1, 1);
        let t2 = SpoTriple::new(70, 2, 2); // same bucket (64 >> 5 == 70 >> 5 == 2)
        let t3 = SpoTriple::new(100, 3, 3); // different bucket (100 >> 5 == 3)
        let m = MorphFlags::default();
        let c1 = Cam64::from_triple(&t1, m, 0, false, false, false);
        let c2 = Cam64::from_triple(&t2, m, 0, false, false, false);
        let c3 = Cam64::from_triple(&t3, m, 0, false, false, false);
        assert!(c1.same_entity_bucket(c2));
        assert!(!c1.same_entity_bucket(c3));
    }

    #[test]
    fn raw_roundtrip() {
        let c = Cam64::from_lanes([0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78, 0x9A]);
        assert_eq!(Cam64::from_raw(c.raw()), c);
    }

    #[test]
    fn stack_depth_clamped_at_127() {
        let t = SpoTriple::new(1, 2, 3);
        let m = MorphFlags::default();
        let c = Cam64::from_triple(&t, m, 255, false, false, false);
        assert_eq!(c.entity_stack_depth(), 127);
    }
}
