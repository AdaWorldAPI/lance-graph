//! `identity_quad` — four external identifier spaces materialized in ONE
//! 96-bit facet payload, joined at bake time.
//!
//! # What this is for
//!
//! A row whose identity is asserted independently by **four external
//! identifier spaces** (each in the 10^5–10^7 range) has, today, two bad
//! options: keep four side tables and pay a join per read, or keep a
//! crosswalk chain and walk it per read. Both put work on the read path that
//! the addresses could have carried themselves.
//!
//! This module is the third option: **resolve the crosswalk ONCE, at bake
//! time, and materialize all four identities in a single V3 facet payload.**
//! Afterwards a read is a fixed-offset register read — no join, no crosswalk
//! table consulted, no walk. That is the entire point of the tenant; the
//! saving is not space, it is the disappearance of the read-time join.
//!
//! # The carving: 4 × 24 bits, CONTIGUOUS
//!
//! The payload is [`crate::legacy_outliers::LegacyOutlier::WideTriple`] (G2) —
//! `4 × u24` **contiguous**, and this module reads and writes it through that
//! type rather than duplicating the bit math.
//!
//! It is deliberately **NOT** the axis-grouped `4×(8:8:8)`
//! ([`CascadeShape::G4D3`], `le-contract.md` §3 L5). That shape is three
//! independently-meaningful bytes per slot — a rail reading, where each byte
//! is its own one-byte reference. An **exact identity** split across three
//! independently-read bytes is no longer a single invertible value, and
//! invertibility is this tenant's whole acceptance criterion (see
//! § Bijectivity). 2^24 = 16,777,216 covers every space this carving is
//! intended for with room to spare.
//!
//! [`CascadeShape::G4D3`]: crate::facet::CascadeShape::G4D3
//!
//! ## Why a wide carving is warranted here, and what it costs
//!
//! `legacy_outliers` is documented as the V1-migration waiting room, and
//! `le-contract.md` §3a is right to discourage it: a wide field is usually the
//! symptom of a class carrying too many concerns, or of a field with nothing to
//! roll over into. **This tenant is neither**, and the module doc there names
//! the two conditions explicitly, so they can be checked rather than assumed:
//!
//! - *god-object-related* — no. The carving is four slots, fixed, of one kind
//!   (an identifier ordinal). It does not grow with the class's concerns; a
//!   fifth identifier space is a second facet, never a wider field.
//! - *lacking proper bucket rollover* — no. Each slot's capacity is the size of
//!   its own codebook, and [`IdentityCodebook`] **refuses** to exceed it
//!   (`CodebookError::TooLarge`) rather than saturating silently — the same
//!   refuse-don't-widen discipline [`crate::codebook::Codebook`] uses at its own
//!   256-entry scale.
//!
//! What it genuinely gives up is the byte axis: a `u24` slot has no rail, so it
//! is not shift-addressable and `group_of` does not apply to it. That is the
//! honest cost, and it is the correct trade *only because* these slots are
//! identities rather than adjacency references. This module makes the trade
//! visible instead of quiet; whether the carving deserves promotion out of
//! `legacy_outliers` into its own sanctioned name is an operator question, not
//! one to settle in code.
//!
//! ## Related, and deliberately not reused
//!
//! [`crate::codebook::Codebook`] is the **1-byte, ≤255-entry per-family**
//! vocabulary: a family that outgrows it *splits*, never widens the byte. That
//! is the right rule at rail scale and the wrong instrument at this one — a
//! space in the 10^6 range cannot be reached by splitting into 256-entry
//! families without the split itself becoming the address. [`IdentityCodebook`]
//! is its 24-bit sibling, with the same overflow-refuses discipline, not a
//! widening of it.
//!
//! # Slots hold ordinals, never text
//!
//! A slot holds a **codebook ordinal**, never a string. Where an identifier
//! space is natively numeric the ordinal may be that number; where it is
//! textual, [`IdentityCodebook`] assigns the ordinal and is the only place the
//! text lives.
//!
//! # Bijectivity is the acceptance criterion
//!
//! This tenant is only sound if the mapping is 1:1 in **both** directions —
//! `key → ordinal → key` must be the identity for every entry in the codebook.
//! A lossy or many-to-one encoding would make two distinct external identifiers
//! indistinguishable after the bake, which is not a degraded answer but a wrong
//! one. So:
//!
//! - [`IdentityCodebook::try_new`] **rejects** a non-injective key list at
//!   construction (`CodebookError::DuplicateKey`) — the failure is impossible to
//!   observe later because it cannot be built;
//! - [`IdentityCodebook::verify_bijective`] proves the round-trip over every
//!   entry, for a caller that wants the proof at bake time rather than trusting
//!   construction.
//!
//! # Absent is not ordinal zero
//!
//! Slots are stored as `ordinal + 1`, so the raw value `0` means **absent** —
//! the CANON zero-fallback ladder's "not consulted", not "the zeroth entry".
//! Without the offset, the first entry of every codebook would be
//! indistinguishable from a slot that was never filled, and a partially-joined
//! row would silently read as fully joined. The largest ordinal a slot can hold
//! is therefore [`MAX_ORDINAL`] = `2^24 - 2`, and a book may hold one more entry
//! than that number — [`MAX_ENTRIES`] = `2^24 - 1` — because ordinals start at
//! zero. The two constants are deliberately separate: [`check_capacity`] gates
//! an **entry count**, [`MAX_ORDINAL`] bounds a **slot value**.
//!
//! # Ordinals are stable only for a FIXED key set — the bake owes a digest
//!
//! [`IdentityCodebook::try_new`] sorts and derives each ordinal from the sorted
//! position, so **an ordinal is a property of the whole key set, not of the key
//! alone.** Add one key that sorts early and every ordinal at or after it shifts
//! by one. A facet baked against the old book then resolves, through the new
//! book, to a *neighbouring* key — internally consistent on both sides, and
//! therefore invisible to [`IdentityCodebook::verify_bijective`], which proves
//! only that each book round-trips against itself.
//!
//! This matters here and not at rail scale because the quad is a **persisted
//! payload**: the ordinal outlives the process that computed it.
//!
//! The contract does **not** silently absorb this. It states the constraint and
//! gives it a witness:
//!
//! - **The constraint:** a baked [`IdentityQuad`] is meaningful only against the
//!   exact codebook that produced it. Growing a codebook is a **rebake**, not an
//!   append — there is deliberately no append-preserving constructor, because a
//!   book that preserved assignments would no longer be sorted and
//!   [`IdentityCodebook::ordinal`]'s `binary_search` would be unsound.
//! - **The witness:** [`IdentityCodebook::digest`] is a deterministic value over
//!   the ordered key list. A bake records the digest beside the rows it wrote; a
//!   later read compares, and **refuses a shifted book instead of resolving a
//!   wrong key**. The comparison is the caller's to make — the digest gives it
//!   something to compare, which is what was missing.
//!
//! Whether the join should instead bind facets to a versioned codebook at the
//! contract level is an open design question, recorded as
//! `ISS-IDENTITY-CODEBOOK-ORDINAL-STABILITY` rather than settled here.

use crate::facet::FacetCascade;
use crate::legacy_outliers::{LegacyOutlier, PAYLOAD_LEN};

/// Number of identifier slots in the payload (`4 × 24 bit = 96 bit`).
pub const IDENTITY_SLOTS: usize = 4;

/// Largest ordinal a slot can carry. One below `2^24 - 1` because slots are
/// stored as `ordinal + 1` so that raw `0` can mean *absent* (see module doc).
pub const MAX_ORDINAL: u32 = 0x00FF_FFFF - 1;

/// Why a slot write was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadError {
    /// Slot index was not in `0..4`.
    SlotOutOfRange { slot: usize },
    /// Ordinal exceeded [`MAX_ORDINAL`]. Refused, never truncated — a silently
    /// masked ordinal would address a different entity.
    OrdinalTooLarge { ordinal: u32 },
}

/// Decode one raw slot value: `0` is *absent*, anything else is `ordinal + 1`.
/// The single decode site, so a reader and a sweep can never disagree about
/// what a zero means.
#[inline]
const fn decode(raw: u32) -> Option<u32> {
    if raw == 0 {
        None
    } else {
        Some(raw - 1)
    }
}

/// Four external identities in one 96-bit content-blind payload.
///
/// Read/written through [`LegacyOutlier::WideTriple`] (G2, `4 × u24`
/// contiguous); see the module doc for why that carving and not the
/// axis-grouped `4×(8:8:8)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IdentityQuad {
    payload: [u8; PAYLOAD_LEN],
}

impl IdentityQuad {
    /// The carving this type reads and writes — stated as a value so an audit
    /// can assert it rather than infer it from the byte math.
    pub const CARVING: LegacyOutlier = LegacyOutlier::WideTriple;

    /// An all-absent quad — every slot dormant (zero-fallback), never
    /// "four zeroth entries".
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            payload: [0u8; PAYLOAD_LEN],
        }
    }

    /// Wrap raw payload bytes (facet bytes `4..16`).
    #[must_use]
    pub const fn from_payload(payload: [u8; PAYLOAD_LEN]) -> Self {
        Self { payload }
    }

    /// The raw 12-byte payload, little-endian.
    #[must_use]
    pub const fn payload(self) -> [u8; PAYLOAD_LEN] {
        self.payload
    }

    /// Build from four optional ordinals, in slot order.
    pub fn from_slots(slots: [Option<u32>; IDENTITY_SLOTS]) -> Result<Self, QuadError> {
        let mut raw = [0u32; IDENTITY_SLOTS];
        for (i, s) in slots.iter().enumerate() {
            raw[i] = match *s {
                None => 0,
                Some(ordinal) if ordinal <= MAX_ORDINAL => ordinal + 1,
                Some(ordinal) => return Err(QuadError::OrdinalTooLarge { ordinal }),
            };
        }
        Ok(Self {
            payload: LegacyOutlier::write_wide_triple(raw),
        })
    }

    /// The ordinal in `slot`, or `None` when the slot is absent (raw `0`).
    /// `None` for an out-of-range slot index as well — the caller that cares
    /// about the difference uses [`Self::try_slot`].
    #[must_use]
    pub fn slot(self, slot: usize) -> Option<u32> {
        self.try_slot(slot).ok().flatten()
    }

    /// [`Self::slot`], distinguishing "absent" from "no such slot".
    pub fn try_slot(self, slot: usize) -> Result<Option<u32>, QuadError> {
        if slot >= IDENTITY_SLOTS {
            return Err(QuadError::SlotOutOfRange { slot });
        }
        Ok(decode(LegacyOutlier::read_wide_triple(&self.payload)[slot]))
    }

    /// All four slots, in order.
    #[must_use]
    pub fn slots(self) -> [Option<u32>; IDENTITY_SLOTS] {
        let raw = LegacyOutlier::read_wide_triple(&self.payload);
        let mut out = [None; IDENTITY_SLOTS];
        for (o, r) in out.iter_mut().zip(raw) {
            *o = decode(r);
        }
        out
    }

    /// A copy with one slot replaced. The other three are carried through
    /// unchanged by construction (they are re-encoded from their own read
    /// values); the field-isolation matrix test asserts it rather than assuming
    /// it, per `I-LEGACY-API-FEATURE-GATED`.
    pub fn with_slot(self, slot: usize, ordinal: Option<u32>) -> Result<Self, QuadError> {
        if slot >= IDENTITY_SLOTS {
            return Err(QuadError::SlotOutOfRange { slot });
        }
        let mut slots = self.slots();
        slots[slot] = ordinal;
        Self::from_slots(slots)
    }

    /// How many slots carry an identity — the join's own measure of how much
    /// it resolved for this row.
    #[must_use]
    pub fn filled(self) -> usize {
        self.slots().iter().filter(|s| s.is_some()).count()
    }

    /// Attach the payload to a classid, producing the 16-byte V3 facet.
    #[must_use]
    pub fn into_facet(self, classid: u32) -> FacetCascade {
        let mut b = [0u8; 16];
        b[0..4].copy_from_slice(&classid.to_le_bytes());
        b[4..16].copy_from_slice(&self.payload);
        FacetCascade::from_bytes(&b)
    }

    /// Read the payload back out of a V3 facet.
    #[must_use]
    pub fn from_facet(facet: &FacetCascade) -> Self {
        let b = facet.to_bytes();
        let mut payload = [0u8; PAYLOAD_LEN];
        payload.copy_from_slice(&b[4..16]);
        Self { payload }
    }
}

/// Why a codebook could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodebookError {
    /// The key list was not injective. Rejected at construction: a duplicate
    /// key would make `key → ordinal → key` non-invertible, which is the one
    /// property this tenant cannot degrade gracefully.
    DuplicateKey { key: String },
    /// More entries than a 24-bit slot can address. Refused, not truncated —
    /// the same refuse-don't-widen signal [`crate::codebook::Codebook`] gives at
    /// its own scale.
    TooLarge { len: usize },
    /// A round-trip failed on an already-built codebook — should be
    /// unreachable, and [`IdentityCodebook::verify_bijective`] exists to say so
    /// out loud rather than assume it.
    NotBijective { ordinal: u32 },
}

/// A bijective `key ⇄ ordinal` codebook for one external identifier space.
///
/// Sorted and unique by construction, so `ordinal` is a `binary_search` and
/// `key` is an index. The strings live here and only here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IdentityCodebook {
    keys: Vec<String>,
}

/// How many entries a 24-bit slot can address, given that raw `0` is reserved
/// for *absent*.
pub const MAX_ENTRIES: usize = MAX_ORDINAL as usize + 1;

/// The capacity gate, factored out so it is directly testable at its boundary
/// — building a 16.7M-entry codebook to exercise the check would cost more
/// than the check is worth, and an untestable guard is the defect one level up.
///
/// Refuses rather than truncates: a codebook whose tail silently fell off the
/// end would leave real identifiers unaddressable with no signal.
pub fn check_capacity(len: usize) -> Result<(), CodebookError> {
    if len > MAX_ENTRIES {
        Err(CodebookError::TooLarge { len })
    } else {
        Ok(())
    }
}

impl IdentityCodebook {
    /// Build from a key list. Sorts, then **rejects** any duplicate (the
    /// injectivity gate) and any list larger than a 24-bit slot can address.
    ///
    /// Because ordinals come from the sorted position, the assignment depends on
    /// the **whole key set**; see the module doc's stability section and
    /// [`Self::digest`] before growing a book whose ordinals have been persisted.
    pub fn try_new(keys: impl IntoIterator<Item = String>) -> Result<Self, CodebookError> {
        let mut keys: Vec<String> = keys.into_iter().collect();
        keys.sort();
        if let Some(w) = keys.windows(2).find(|w| w[0] == w[1]) {
            return Err(CodebookError::DuplicateKey { key: w[0].clone() });
        }
        check_capacity(keys.len())?;
        Ok(Self { keys })
    }

    /// The ordinal for `key`, or `None` when the key is not in this space.
    #[must_use]
    pub fn ordinal(&self, key: &str) -> Option<u32> {
        self.keys
            .binary_search_by(|k| k.as_str().cmp(key))
            .ok()
            .map(|i| i as u32)
    }

    /// The key for `ordinal`, or `None` when out of range.
    #[must_use]
    pub fn key(&self, ordinal: u32) -> Option<&str> {
        self.keys.get(ordinal as usize).map(String::as_str)
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the space is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// A deterministic digest over the **ordered** key list — the stability
    /// witness for the constraint in the module doc.
    ///
    /// Two books with the same digest assign the same ordinal to the same key.
    /// A bake records this next to the rows it wrote; a later read compares it
    /// against the book it holds and **refuses a shifted book rather than
    /// resolving a wrong key**. It is a *witness*, not an enforcement: nothing
    /// in this type can stop a caller that declines to compare.
    ///
    /// Each key is fed length-first, so `["ab", "c"]` and `["a", "bc"]` cannot
    /// collide through concatenation. Plain FNV-1a — this is an accidental-drift
    /// detector for a build artifact, not a security primitive, and the crate is
    /// zero-dependency by contract.
    #[must_use]
    pub fn digest(&self) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = OFFSET;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        };
        for key in &self.keys {
            for b in (key.len() as u64).to_le_bytes() {
                eat(b);
            }
            for b in key.as_bytes() {
                eat(*b);
            }
        }
        h
    }

    /// Prove `ordinal → key → ordinal` is the identity for **every** entry.
    ///
    /// Construction already guarantees it; this is the explicit witness a bake
    /// runs so the guarantee is measured on the real book rather than inherited
    /// from an argument about the constructor.
    ///
    /// **What it deliberately cannot see:** this is a *within-book* property.
    /// Two revisions of a book that assign different ordinals to the same key
    /// each pass here, because each is internally consistent. That failure is
    /// the [`Self::digest`] comparison's job, not this one's.
    pub fn verify_bijective(&self) -> Result<(), CodebookError> {
        for i in 0..self.keys.len() {
            let ordinal = i as u32;
            let round = self
                .key(ordinal)
                .and_then(|k| self.ordinal(k))
                .ok_or(CodebookError::NotBijective { ordinal })?;
            if round != ordinal {
                return Err(CodebookError::NotBijective { ordinal });
            }
        }
        Ok(())
    }
}

/// The bake-time join: four codebooks, one per slot, resolving a row's four
/// external keys into a single [`IdentityQuad`].
///
/// This is where the crosswalk chain is paid. Everything downstream reads the
/// quad at a fixed offset.
#[derive(Debug, Clone, Copy)]
pub struct QuadJoin<'a> {
    books: [&'a IdentityCodebook; IDENTITY_SLOTS],
}

impl<'a> QuadJoin<'a> {
    /// Bind one codebook per slot, in slot order.
    #[must_use]
    pub const fn new(books: [&'a IdentityCodebook; IDENTITY_SLOTS]) -> Self {
        Self { books }
    }

    /// The codebook bound to `slot`.
    #[must_use]
    pub fn book(&self, slot: usize) -> Option<&'a IdentityCodebook> {
        self.books.get(slot).copied()
    }

    /// Resolve four optional keys into a quad.
    ///
    /// A key its codebook does not know resolves to **absent**, never to a
    /// fabricated ordinal — the join reports what it could not resolve by
    /// leaving the slot dormant, which [`IdentityQuad::filled`] then counts.
    pub fn resolve(&self, keys: [Option<&str>; IDENTITY_SLOTS]) -> Result<IdentityQuad, QuadError> {
        let mut slots = [None; IDENTITY_SLOTS];
        for (i, key) in keys.iter().enumerate() {
            slots[i] = key.and_then(|k| self.books[i].ordinal(k));
        }
        IdentityQuad::from_slots(slots)
    }

    /// The pull-back: render a quad's slots back to their keys.
    ///
    /// `resolve` then `explain` is the identity on any key set the codebooks
    /// know — the tenant-level statement of the bijectivity criterion.
    #[must_use]
    pub fn explain(&self, quad: IdentityQuad) -> [Option<&'a str>; IDENTITY_SLOTS] {
        let slots = quad.slots();
        let mut out = [None; IDENTITY_SLOTS];
        for (i, o) in out.iter_mut().enumerate() {
            *o = slots[i].and_then(|ordinal| self.books[i].key(ordinal));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic keys for slot `s`: `s-000000`, `s-000001`, … Deliberately not
    /// drawn from any real identifier space.
    fn synthetic_book(slot: usize, n: usize) -> IdentityCodebook {
        IdentityCodebook::try_new((0..n).map(|i| format!("{slot}-{i:06}"))).expect("unique keys")
    }

    // ── the carving ────────────────────────────────────────────────────────

    /// The carving is the contiguous G2 wide-triple, not the axis-grouped
    /// `4×(8:8:8)`. Falsifier: a future edit switching to `CascadeShape::G4D3`
    /// byte triples changes this constant and fails here.
    #[test]
    fn carving_is_the_contiguous_wide_triple() {
        assert_eq!(IdentityQuad::CARVING, LegacyOutlier::WideTriple);
        assert_eq!(IDENTITY_SLOTS * 24, PAYLOAD_LEN * 8);
    }

    /// Every slot survives the full 24-bit range, including both boundaries.
    /// Falsifier: any truncation, sign error, or byte-order slip in the u24
    /// pack/unpack makes a boundary ordinal come back different.
    #[test]
    fn slots_round_trip_across_the_whole_range() {
        for slot in 0..IDENTITY_SLOTS {
            for ordinal in [0, 1, 255, 256, 65_535, 65_536, 4_400_000, MAX_ORDINAL] {
                let q = IdentityQuad::empty()
                    .with_slot(slot, Some(ordinal))
                    .expect("in range");
                assert_eq!(q.slot(slot), Some(ordinal), "slot {slot} ordinal {ordinal}");
            }
        }
    }

    /// **Field isolation** (`I-LEGACY-API-FEATURE-GATED`): writing each slot in
    /// turn leaves the other three byte-identical. Falsifier: an off-by-three
    /// in the u24 offsets, or a write that rebuilds the payload from a stale
    /// read, corrupts a neighbour and fails here.
    #[test]
    fn writing_one_slot_leaves_the_other_three_unchanged() {
        let base =
            IdentityQuad::from_slots([Some(11), Some(2_222), Some(333_333), Some(4_444_444)])
                .expect("in range");

        for slot in 0..IDENTITY_SLOTS {
            let changed = base.with_slot(slot, Some(1_000_001)).expect("in range");
            assert_eq!(changed.slot(slot), Some(1_000_001));
            for other in 0..IDENTITY_SLOTS {
                if other == slot {
                    continue;
                }
                assert_eq!(
                    changed.slot(other),
                    base.slot(other),
                    "writing slot {slot} disturbed slot {other}"
                );
            }
        }
    }

    /// Clearing a slot is also isolated, and clearing is distinguishable from
    /// writing zero. Falsifier: an encoding without the `+1` offset makes these
    /// two assertions contradict each other.
    #[test]
    fn absent_is_distinct_from_the_zeroth_ordinal() {
        let zeroth = IdentityQuad::empty()
            .with_slot(2, Some(0))
            .expect("in range");
        let cleared = zeroth.with_slot(2, None).expect("in range");

        assert_eq!(zeroth.slot(2), Some(0), "ordinal 0 is a real identity");
        assert_eq!(cleared.slot(2), None, "absent is not ordinal 0");
        assert_ne!(zeroth.payload(), cleared.payload(), "and the bytes differ");
        assert_eq!(zeroth.filled(), 1);
        assert_eq!(cleared.filled(), 0);
    }

    /// An over-large ordinal is refused, not masked. Falsifier: a `& 0xFF_FFFF`
    /// anywhere on the write path would silently address a different entity and
    /// return `Ok` here.
    #[test]
    fn an_over_capacity_ordinal_is_refused_not_truncated() {
        let too_big = MAX_ORDINAL + 1;
        assert_eq!(
            IdentityQuad::empty().with_slot(0, Some(too_big)),
            Err(QuadError::OrdinalTooLarge { ordinal: too_big })
        );
        // …and the boundary below it is accepted, so the guard is not simply
        // rejecting everything large (can-stay-silent twin).
        assert!(IdentityQuad::empty()
            .with_slot(0, Some(MAX_ORDINAL))
            .is_ok());
    }

    /// A slot index outside the carving is named, not wrapped.
    #[test]
    fn out_of_range_slot_is_named() {
        assert_eq!(
            IdentityQuad::empty().with_slot(IDENTITY_SLOTS, Some(1)),
            Err(QuadError::SlotOutOfRange {
                slot: IDENTITY_SLOTS
            })
        );
        assert_eq!(
            IdentityQuad::empty().try_slot(9),
            Err(QuadError::SlotOutOfRange { slot: 9 })
        );
    }

    /// The quad rides a real V3 facet: classid in the prefix, payload in
    /// `4..16`, untouched by the attach. Falsifier: an off-by-four in the
    /// prefix would either clobber a slot or shift the classid.
    #[test]
    fn facet_attach_preserves_both_halves() {
        let q =
            IdentityQuad::from_slots([Some(7), None, Some(9), Some(16_777_214)]).expect("in range");
        let facet = q.into_facet(0x0301_0000);
        assert_eq!(facet.facet_classid, 0x0301_0000);
        assert_eq!(IdentityQuad::from_facet(&facet), q);
        assert_eq!(&facet.to_bytes()[4..16], &q.payload()[..]);
    }

    // ── bijectivity ────────────────────────────────────────────────────────

    /// `key → ordinal → key` is the identity for every entry, and the explicit
    /// witness agrees. Falsifier: an unsorted `keys` vector (binary_search
    /// unsound) or an off-by-one in `key`/`ordinal` breaks entries in the
    /// middle of the book, which a spot-check would miss and this sweep does
    /// not.
    #[test]
    fn codebook_round_trips_every_entry_both_directions() {
        let book = synthetic_book(0, 5_000);
        assert!(book.verify_bijective().is_ok());
        for i in 0..book.len() {
            let ordinal = i as u32;
            let key = book.key(ordinal).expect("in range");
            assert_eq!(book.ordinal(key), Some(ordinal), "ordinal {ordinal}");
        }
        // …and a key outside the space stays silent rather than colliding.
        assert_eq!(book.ordinal("0-999999"), None);
        assert_eq!(book.key(book.len() as u32), None);
    }

    /// **The negative case**: a non-injective key list is rejected at
    /// construction, so a many-to-one mapping cannot exist to be discovered
    /// later. Falsifier: dropping the duplicate check makes this `Ok`, and the
    /// duplicated key would then have two ordinals mapping back to one string —
    /// exactly the collapse the tenant may not tolerate.
    #[test]
    fn a_non_injective_key_list_is_rejected() {
        let dup = IdentityCodebook::try_new(["b", "a", "b"].into_iter().map(str::to_string));
        assert_eq!(
            dup,
            Err(CodebookError::DuplicateKey {
                key: "b".to_string()
            })
        );
        // The same list without the duplicate builds — so the guard
        // discriminates rather than refusing everything.
        assert!(IdentityCodebook::try_new(["b", "a"].into_iter().map(str::to_string)).is_ok());
    }

    /// A book larger than a slot can address is refused rather than silently
    /// losing its tail — asserted against [`check_capacity`] directly, at both
    /// sides of the boundary, because materializing 16.7M keys to exercise the
    /// same branch would cost more than the branch is worth.
    ///
    /// Falsifier: deleting the check makes the `Err` arm `Ok`; widening the
    /// bound by one makes the `MAX_ENTRIES + 1` arm pass. Both halves are
    /// asserted, so a guard that fired on everything would fail the first arm.
    #[test]
    fn the_capacity_gate_refuses_only_past_the_boundary() {
        assert!(check_capacity(0).is_ok());
        assert!(
            check_capacity(MAX_ENTRIES).is_ok(),
            "the last addressable entry"
        );
        assert_eq!(
            check_capacity(MAX_ENTRIES + 1),
            Err(CodebookError::TooLarge {
                len: MAX_ENTRIES + 1
            }),
            "one past capacity must be refused, not truncated"
        );
        // And a real (small) book goes through the same gate successfully.
        assert!(IdentityCodebook::try_new(["a".to_string()]).is_ok());
    }

    // ── ordinal stability across codebook revisions ────────────────────────

    /// **The hazard is real, and this test makes it observable.** Growing a book
    /// with a key that sorts before an existing one renumbers the existing key,
    /// so a payload baked against the old book explains as a *different* key
    /// through the new one — while both books pass `verify_bijective`.
    ///
    /// This is a can-it-fire test for a documented constraint, not an assertion
    /// that the code is correct. Falsifier: if `try_new` ever gained
    /// assignment-preserving growth, `after.ordinal("b")` would stay `0` and the
    /// mis-explanation would not occur — this test would fail, which is exactly
    /// the signal wanted if the semantics change.
    #[test]
    fn growing_a_codebook_with_an_early_key_renumbers_the_existing_ones() {
        let before = IdentityCodebook::try_new(["b".to_string()]).expect("unique");
        let after =
            IdentityCodebook::try_new(["a", "b"].into_iter().map(str::to_string)).expect("unique");

        assert_eq!(before.ordinal("b"), Some(0));
        assert_eq!(
            after.ordinal("b"),
            Some(1),
            "the existing key was renumbered"
        );

        // Both books are internally sound, which is why the whole-book witness
        // cannot detect the shift — the digest is what can.
        assert!(before.verify_bijective().is_ok());
        assert!(after.verify_bijective().is_ok());

        // A payload baked against `before` explains as the WRONG key under
        // `after`: ordinal 0 was "b", it is now "a".
        let baked = IdentityQuad::empty()
            .with_slot(0, before.ordinal("b"))
            .expect("in range");
        assert_eq!(after.key(baked.slot(0).expect("filled")), Some("a"));
    }

    /// The digest separates exactly the books that would shift ordinals, and
    /// **stays silent** on the books that would not — the can-fire/can-stay-silent
    /// pair on the same value.
    ///
    /// Falsifier: a digest that hashed an unordered set would make the first
    /// assertion fail; a digest that hashed input order rather than the sorted
    /// list would make the second fail; a digest without the length prefix would
    /// make the third fail (`"ab"+"c"` and `"a"+"bc"` concatenate identically).
    #[test]
    fn the_digest_fires_on_a_shift_and_stays_silent_on_an_equivalent_book() {
        let one = IdentityCodebook::try_new(["b".to_string()]).expect("unique");
        let grown =
            IdentityCodebook::try_new(["a", "b"].into_iter().map(str::to_string)).expect("unique");
        assert_ne!(
            one.digest(),
            grown.digest(),
            "a book that renumbers must not compare equal"
        );

        // Same key set, different input order — same book, same ordinals, and
        // therefore the digest must NOT fire.
        let forward =
            IdentityCodebook::try_new(["a", "b", "c"].into_iter().map(str::to_string)).expect("ok");
        let shuffled =
            IdentityCodebook::try_new(["c", "a", "b"].into_iter().map(str::to_string)).expect("ok");
        assert_eq!(forward, shuffled);
        assert_eq!(
            forward.digest(),
            shuffled.digest(),
            "input order is not part of the book's identity"
        );

        // Length-delimiting: these two books concatenate to the same bytes.
        let ab_c =
            IdentityCodebook::try_new(["ab", "c"].into_iter().map(str::to_string)).expect("ok");
        let a_bc =
            IdentityCodebook::try_new(["a", "bc"].into_iter().map(str::to_string)).expect("ok");
        assert_ne!(
            ab_c.digest(),
            a_bc.digest(),
            "concatenation must not collide two different key sets"
        );
    }

    // ── the join ───────────────────────────────────────────────────────────

    /// The whole chain: four key sets → one quad → back to the same four keys.
    /// Falsifier: any slot mix-up (book `i` consulted for slot `j`) returns a
    /// key from the wrong space and fails here, because the synthetic books are
    /// prefixed by slot.
    #[test]
    fn the_join_round_trips_keys_through_the_quad() {
        let books = [
            synthetic_book(0, 2_000),
            synthetic_book(1, 300),
            synthetic_book(2, 40_000),
            synthetic_book(3, 7),
        ];
        let join = QuadJoin::new([&books[0], &books[1], &books[2], &books[3]]);

        let keys = [
            Some("0-001999"),
            Some("1-000000"),
            Some("2-039999"),
            Some("3-000003"),
        ];
        let quad = join.resolve(keys).expect("all in range");
        assert_eq!(quad.filled(), 4);
        assert_eq!(join.explain(quad), keys);
    }

    /// **Anti-vacuity + can-stay-silent.** Over a row set where only some rows
    /// carry every key, the join must produce a *non-trivial* mix: genuinely
    /// multi-slot rows AND genuinely partial ones. A join that filled
    /// everything, or nothing, would satisfy "a join ran" while carrying no
    /// information.
    ///
    /// Falsifier: if `resolve` fabricated an ordinal for an unknown key, the
    /// partial count would collapse to zero; if it dropped known keys, the
    /// multi count would.
    #[test]
    fn the_join_is_non_trivial_and_refuses_to_fabricate() {
        let books = [
            synthetic_book(0, 100),
            synthetic_book(1, 100),
            synthetic_book(2, 100),
            synthetic_book(3, 100),
        ];
        let join = QuadJoin::new([&books[0], &books[1], &books[2], &books[3]]);

        // 100 rows; every third row's slot-2 key is from a space the codebook
        // does not know, and slot 3 is present only on even rows.
        let mut multi = 0usize;
        let mut partial = 0usize;
        for i in 0..100usize {
            let k0 = format!("0-{i:06}");
            let k1 = format!("1-{i:06}");
            let k2 = if i % 3 == 0 {
                "unknown-key".to_string()
            } else {
                format!("2-{i:06}")
            };
            let k3 = format!("3-{i:06}");
            let quad = join
                .resolve([
                    Some(&k0),
                    Some(&k1),
                    Some(&k2),
                    (i % 2 == 0).then_some(k3.as_str()),
                ])
                .expect("in range");
            if quad.filled() >= 2 {
                multi += 1;
            }
            if quad.filled() < IDENTITY_SLOTS {
                partial += 1;
            }
            // An unknown key is absent, never fabricated.
            if i % 3 == 0 {
                assert_eq!(quad.slot(2), None, "row {i}: unknown key must stay absent");
            } else {
                assert_eq!(quad.slot(2), Some(i as u32));
            }
        }
        assert_eq!(multi, 100, "every row resolved at least two spaces");
        // The excluded set is substantial, not a rounding artifact: rows are
        // partial unless (even AND not a multiple of three).
        assert!(
            partial * 2 > 100,
            "more than half the rows are genuinely partial, got {partial}"
        );
        assert!(partial < 100, "and not all of them — the join does resolve");
    }
}
