// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `causal_witness` — the **CausalWitnessFacet** (A9): the
//! `TekamoloWindowBinding` reading of a `12`-byte content-blind register as **24
//! signed `i4` loci** (`.claude/plans/soa-32-tenant-awareness-redundancy-v1.md`
//! §2.9).
//!
//! **Status: EXPERIMENTAL reading — NOT in the operator-locked §3 catalogue.**
//! Contract: `.claude/v3/soa_layout/witness-nibble-lane.md` (byte/nibble law,
//! value law, placement law, slot-purity fence, open proposals).
//!
//! > **⊘ CITATION CORRECTED (2026-07-28).** Earlier revisions of this header
//! > cited "le-contract §3 **L9** `G24N4`". **No such entry exists** — §3 is
//! > L1–L8, and `G24N4`/`L9` appear nowhere in `.claude/v3/soa_layout/`. The
//! > claim was wrong in KIND, not merely missing: §3 is byte-axis by
//! > construction (every tier is a *byte*, so `group_of` is a pure shift),
//! > §3a's grace carvings are *wider* than a byte and are granted no
//! > `CascadeShape` variants, and a 24×4-bit carving is **sub-byte** — a third
//! > direction the catalogue never contemplated. The canon has moreover
//! > already declined a ninth layout for a structurally identical request:
//! > *"this is NOT a ninth 12-byte layout … the sanctioned in-row refinement
//! > budget remains the turbovec 6×4-bit nibble lane."* **Sub-byte
//! > granularity's sanctioned home is a LANE** (`ValueTenant` variant +
//! > `VALUE_TENANTS` descriptor — the two places), never a §3 payload layout.
//! > `G24N4` is a lane shape NAME; `CascadeShape` gains no 24-group variant.
//!
//! This is a **reading, not a layout.** It re-labels the same 12 bytes a value
//! lane already holds — nothing here reserves, moves, or stores a byte, exactly
//! like [`awareness_facet::SpoFacet`](crate::awareness_facet::SpoFacet) re-labels
//! the register as `6×(8:8)`. The A9 reading carves those 12 bytes as `24×4-bit`
//! instead: **24 signed nibbles**, each a **locus** — a signed offset
//! `∈ [−8, +7]` naming WHERE in the `±8` `temporal.rs` Markov window that
//! awareness dimension's filler sits.
//!
//! # One register, THREE readings (PR #729 doctrine, extended)
//!
//! PR #729 shipped "one register, two readings" of the same content-blind
//! 12 bytes — the FROZEN `12×u8` palette-226 codebook and the Orchestration
//! `6×(8:8)` `style_rails_at` (V3-replayable, `E-H268-REPLAYABLE-TILE-1`). A9 is
//! the **third** ClassView-selected interpretation of the identical storage:
//! `24×4-bit` (`G24N4`). WHICH reading a class uses is an OGAR mint (the per-row/
//! class "Place 2" decision, #729), never a property of the bytes and never a
//! slot (le-contract §2 slot purity).
//!
//! # Loci, not magnitudes (operator-locked)
//!
//! Every nibble is a **context pointer**, never a strength. `0` = **unbound**
//! (zero-fallback — the dimension is not currently placed in the window). The
//! sign is **orientation**: `−` = before / antecedent-cause, `+` = after /
//! consequent. The filler's meaning is **read at the offset** (the node at
//! `self_pos + offset` in the stream), never stored here — `I-VSA-IDENTITIES`
//! clean: identity pointers, not content.
//!
//! # The 16 named dimensions + 8 reserved
//!
//! Slots `0..16` are the operator-named [`Locus`] dimensions (§2.9), grounded in
//! shipped organs. Slots `16..24` stay **reserved-empty** (RESERVE-DON'T-RECLAIM
//! — held open, never padded with a construct to reach 24). The rung level
//! occupies ZERO slots: escalation is carried by the elected ClassView, which
//! re-interprets which loci are live, never by a stored magnitude.
//!
//! # Window agreement = loci comparison
//!
//! Two co-window rows agree about a higher-order dimension when their loci
//! **converge on the same context event** — [`Self::agrees_at`] compares two
//! nibbles (bound + equal offset after normalization), never re-derives meaning.
//! [`Self::quorum`] / [`Self::contradiction`] read the agreeing / preserved-
//! dissenting peer loci directly.

/// The content-blind register width (le-contract §3): 12 bytes = 24 nibbles.
/// Same 12-byte lane [`awareness_facet::SpoFacet`](crate::awareness_facet::SpoFacet)
/// reads as `6×(8:8)`; A9 reads it as `24×4-bit` (`G24N4` — a LANE shape name,
/// never a [`CascadeShape`](crate::facet::CascadeShape) variant: that enum stays
/// byte-axis-only per le-contract §3a).
pub const WITNESS_REGISTER_BYTES: usize = 12;

/// Total loci in the A9 register (24 signed nibbles = 12 bytes).
pub const WITNESS_LOCI: usize = 24;
/// The operator-named loci (§2.9); the remaining `24 − 16 = 8` are reserved.
pub const NAMED_LOCI: usize = 16;

/// The 16 named locus dimensions (plan §2.9), in canonical slot order. Reserved
/// slots `16..24` have no name and are always read `0` unless explicitly set.
pub const LOCUS_LABELS: [&str; WITNESS_LOCI] = [
    "temporal",
    "kausal",
    "modal",
    "lokal",
    "s_meaning",
    "p_meaning",
    "o_meaning",
    "antecedent",
    "basin_anchor",
    "supported_by",
    "supports",
    "runbook_evidence",
    "qualia_reference",
    "meaning_level",
    "quorum",
    "contradiction",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "", // 8 reserved-empty
];

/// A named A9 locus dimension — a coordinate into the 24-slot register. `as usize`
/// IS the slot index (`0..16`); the 8 reserved slots have no `Locus` variant.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Locus {
    /// My time reference (`role_keys::TEMPORAL_KEY`).
    Temporal = 0,
    /// My cause — causality learning's stored answer (`role_keys::KAUSAL_KEY`).
    Kausal = 1,
    /// The manner/possibility context (`role_keys::MODAL_KEY`).
    Modal = 2,
    /// The where/context (`role_keys::LOKAL_KEY`).
    Lokal = 3,
    /// What S means here — the subject meaning-grounding event (SPO plane A1).
    SMeaning = 4,
    /// What P means here — the predicate meaning-grounding event (SPO plane A1).
    PMeaning = 5,
    /// What O means here — the object meaning-grounding event (SPO plane A1).
    OMeaning = 6,
    /// relativPronomen → its antecedent (MODIFIER/CONTEXT keys).
    Antecedent = 7,
    /// The event binding me to my AriGraph basin (`part_of:is_a`, L1).
    BasinAnchor = 8,
    /// The nested supporting basin's evidence ↓ (`hi_chain`).
    SupportedBy = 9,
    /// What I support ↑ (`lo_chain`).
    Supports = 10,
    /// Where my current runbook drew its finding (`RECIPES[34]` / A8).
    RunbookEvidence = 11,
    /// The event that set my current texture (QualiaColumn / i4-qualia).
    QualiaReference = 12,
    /// The context defining my current level of meaning (rung-content ladder 0–4).
    MeaningLevel = 13,
    /// The **agreeing peer** — the window event whose reading matches mine
    /// (NARS freq·conf, A3).
    Quorum = 14,
    /// The **disagreeing peer** — the committed contradiction PRESERVED beside
    /// the quorum (Staunen×Wisdom depth).
    Contradiction = 15,
}

impl Locus {
    /// All 16 named loci in slot order.
    pub const ALL: [Locus; NAMED_LOCI] = [
        Locus::Temporal,
        Locus::Kausal,
        Locus::Modal,
        Locus::Lokal,
        Locus::SMeaning,
        Locus::PMeaning,
        Locus::OMeaning,
        Locus::Antecedent,
        Locus::BasinAnchor,
        Locus::SupportedBy,
        Locus::Supports,
        Locus::RunbookEvidence,
        Locus::QualiaReference,
        Locus::MeaningLevel,
        Locus::Quorum,
        Locus::Contradiction,
    ];

    /// The dimension's canonical name.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        LOCUS_LABELS[self as usize]
    }
}

/// The A9 reading: a `12`-byte register carved as 24 signed `i4` loci.
///
/// Stores the raw register (like [`QualiaI4_16D`](crate::qualia::QualiaI4_16D)
/// stores its packed `u64`); each nibble is sign-extended on read, so the
/// register round-trips loss-free by construction.
///
/// ```
/// use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
/// // A row whose KAUSAL cause sits 3 events back and whose antecedent is 1 back.
/// let w = CausalWitnessFacet::ZERO
///     .with(Locus::Kausal, -3)
///     .with(Locus::Antecedent, -1);
/// assert_eq!(w.at(Locus::Kausal), -3);
/// assert_eq!(w.at(Locus::Antecedent), -1);
/// assert_eq!(w.at(Locus::Temporal), 0); // unbound (zero-fallback)
/// assert_eq!(w.to_register(), CausalWitnessFacet::from_register(w.to_register()).to_register());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct CausalWitnessFacet {
    reg: [u8; WITNESS_REGISTER_BYTES],
}

const _: () = assert!(
    core::mem::size_of::<CausalWitnessFacet>() == WITNESS_REGISTER_BYTES,
    "CausalWitnessFacet must be exactly 12 bytes (repr(transparent) over [u8; 12])"
);
const _: () = assert!(
    core::mem::align_of::<CausalWitnessFacet>()
        == core::mem::align_of::<[u8; WITNESS_REGISTER_BYTES]>(),
    "CausalWitnessFacet must share its backing register's alignment (repr(transparent) requirement)"
);

impl CausalWitnessFacet {
    /// The all-unbound register (every locus `0`).
    pub const ZERO: Self = Self {
        reg: [0u8; WITNESS_REGISTER_BYTES],
    };

    /// The canonical **borrowed** read of a raw 12-byte content-blind register
    /// as an A9 facet — a pointer reborrow, not a copy.
    ///
    /// Sound because [`CausalWitnessFacet`] is `#[repr(transparent)]` over
    /// `[u8; WITNESS_REGISTER_BYTES]` with identical size and alignment
    /// (const-asserted above), so a reference to the backing array and a
    /// reference to the facet share layout byte-for-byte — reinterpreting
    /// `&[u8; N]` as `&CausalWitnessFacet` is defined behaviour under the
    /// repr(transparent) guarantee, not merely "the compiler happens to
    /// optimize the copy away."
    ///
    /// This is THE canonical zero-cost read: it compiles to a pointer
    /// reborrow, provably nothing. [`from_register`](Self::from_register) /
    /// [`to_register`](Self::to_register) remain as **by-value conveniences**
    /// the contract does not depend on — prefer this method when the caller
    /// already holds a borrowed register and wants a provably-zero-cost view.
    #[inline]
    #[must_use]
    pub const fn from_register_ref(reg: &[u8; WITNESS_REGISTER_BYTES]) -> &Self {
        // SAFETY: `CausalWitnessFacet` is `#[repr(transparent)]` over
        // `[u8; WITNESS_REGISTER_BYTES]` (const-asserted above: identical
        // size and identical alignment), so `&[u8; N]` and `&Self` have
        // identical layout and this reinterpretation is defined behaviour.
        // No initialization, no allocation, no read of the bytes occurs here.
        unsafe { &*(reg as *const [u8; WITNESS_REGISTER_BYTES] as *const Self) }
    }

    /// Read a raw 12-byte content-blind register as an A9 facet (BY-VALUE
    /// convenience copy — see [`from_register_ref`](Self::from_register_ref)
    /// for the canonical zero-cost borrowed read the contract relies on).
    #[inline]
    #[must_use]
    pub const fn from_register(reg: [u8; WITNESS_REGISTER_BYTES]) -> Self {
        Self { reg }
    }

    /// The 12-byte register — the inverse of [`from_register`](Self::from_register)
    /// (BY-VALUE convenience copy; see [`from_register_ref`](Self::from_register_ref)
    /// for the canonical zero-cost borrowed read).
    #[inline]
    #[must_use]
    pub const fn to_register(self) -> [u8; WITNESS_REGISTER_BYTES] {
        self.reg
    }

    /// The signed offset `∈ [−8, +7]` at slot `0..24` (sign-extended nibble).
    /// Out-of-range slot returns `0` (defensive; the 16-named accessors are preferred).
    #[inline]
    #[must_use]
    pub const fn get(self, slot: usize) -> i8 {
        if slot >= WITNESS_LOCI {
            return 0;
        }
        let byte = self.reg[slot / 2];
        let nibble = if slot & 1 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        // sign-extend 4 → 8 bits (mirrors qualia::QualiaI4_16D::get)
        ((nibble << 4) as i8) >> 4
    }

    /// Set the signed offset at slot `0..24`. Clamps to `[−8, +7]`; out-of-range
    /// slot is a no-op.
    #[inline]
    pub fn set(&mut self, slot: usize, offset: i8) {
        if slot >= WITNESS_LOCI {
            return;
        }
        let v = (offset.clamp(-8, 7) as u8) & 0x0F;
        let bi = slot / 2;
        if slot & 1 == 0 {
            self.reg[bi] = (self.reg[bi] & 0xF0) | v;
        } else {
            self.reg[bi] = (self.reg[bi] & 0x0F) | (v << 4);
        }
    }

    /// Builder-shape [`set`](Self::set) on a named [`Locus`].
    #[inline]
    #[must_use]
    pub fn with(mut self, locus: Locus, offset: i8) -> Self {
        self.set(locus as usize, offset);
        self
    }

    /// The signed offset at a named [`Locus`].
    #[inline]
    #[must_use]
    pub const fn at(self, locus: Locus) -> i8 {
        self.get(locus as usize)
    }

    /// Is this locus **bound** (nonzero offset) rather than unbound (zero-fallback)?
    #[inline]
    #[must_use]
    pub const fn is_bound(self, locus: Locus) -> bool {
        self.at(locus) != 0
    }

    /// How many of the 16 named loci are bound.
    #[inline]
    #[must_use]
    pub fn bound_count(self) -> usize {
        Locus::ALL.iter().filter(|&&l| self.is_bound(l)).count()
    }

    /// Resolve a locus to a **stream position** relative to `self_pos`: the node
    /// at `self_pos + offset`, or `None` if unbound or the resolved index falls
    /// outside `0..stream_len`.
    #[inline]
    #[must_use]
    pub fn resolves_to(self, locus: Locus, self_pos: usize, stream_len: usize) -> Option<usize> {
        let off = self.at(locus);
        if off == 0 {
            return None;
        }
        let idx = self_pos as isize + off as isize;
        if idx < 0 || idx as usize >= stream_len {
            None
        } else {
            Some(idx as usize)
        }
    }

    /// **Window agreement at a locus** — do `self` and `other` place this
    /// dimension at the SAME context event? True iff both are bound AND their
    /// offsets are equal (loci convergence; never re-derives meaning).
    #[inline]
    #[must_use]
    pub const fn agrees_at(self, other: Self, locus: Locus) -> bool {
        let a = self.at(locus);
        let b = other.at(locus);
        a != 0 && a == b
    }

    /// Count of named loci on which `self` and `other` agree (the quorum measure).
    #[inline]
    #[must_use]
    pub fn agreement_count(self, other: Self) -> usize {
        Locus::ALL
            .iter()
            .filter(|&&l| self.agrees_at(other, l))
            .count()
    }

    /// The agreeing-peer offset (the [`Locus::Quorum`] pointer).
    #[inline]
    #[must_use]
    pub const fn quorum(self) -> i8 {
        self.at(Locus::Quorum)
    }

    /// The preserved-dissenting-peer offset (the [`Locus::Contradiction`] pointer).
    #[inline]
    #[must_use]
    pub const fn contradiction(self) -> i8 {
        self.at(Locus::Contradiction)
    }

    /// The KAUSAL cause pointer — causality learning's stored answer.
    #[inline]
    #[must_use]
    pub const fn cause(self) -> i8 {
        self.at(Locus::Kausal)
    }

    /// The coreference antecedent pointer (relativPronomen → antecedent).
    #[inline]
    #[must_use]
    pub const fn antecedent(self) -> i8 {
        self.at(Locus::Antecedent)
    }

    /// **Locus-mask election over the 24 slots** — return a facet holding
    /// ONLY the loci `mask` elects; every unelected slot reads `0` (UNBOUND).
    ///
    /// `mask` positions `0..24` line up 1:1 with locus slots `0..24` (the 16
    /// named [`Locus`] variants at `0..16`, the 8 reserved-empty slots at
    /// `16..24`). An unelected slot is **absent from the projection** —
    /// zeroed, not merely hidden from a subsequent read — so a projected
    /// facet is itself a valid, safe-to-share [`CausalWitnessFacet`].
    ///
    /// **Fail-closed, non-negotiable:** the caller supplies `mask` — there is
    /// no implicit fallback here. Callers that lack an election MUST pass
    /// [`crate::class_view::WideFieldMask::EMPTY`] (elects nothing), never
    /// [`crate::class_view::WideFieldMask::full_for`] — `full_for` is a
    /// **render** convenience (e.g. "show every field for debugging") and
    /// must never stand in for an absent RBAC/election mask, exactly as
    /// `full_for` must never backstop a missing RBAC field mask elsewhere in
    /// this crate. Do not add a helper that defaults an absent mask to full.
    #[inline]
    #[must_use]
    pub fn project(self, mask: &crate::class_view::WideFieldMask) -> Self {
        let mut out = Self::ZERO;
        for slot in 0..WITNESS_LOCI {
            if mask.has(slot as u8) {
                let v = self.get(slot);
                if v != 0 {
                    out.set(slot, v);
                }
            }
        }
        out
    }

    /// **The guarded single read:** `Some(offset)` iff `mask` elects `locus`
    /// **AND** the slot is bound (nonzero); `None` otherwise.
    ///
    /// The two conditions are independent — a locus can be elected but
    /// unbound (reads `None`), or bound but unelected (reads `None`). Zero
    /// maps to `None` because `0` is the register's own zero-fallback
    /// sentinel for "unbound" (§ Loci, not magnitudes, above) — it is never
    /// "offset zero, meaning self," so there is nothing here for an election
    /// to surface even when the mask admits the slot.
    ///
    /// **Fail-closed, non-negotiable:** same contract as [`Self::project`].
    /// An absent election is [`crate::class_view::WideFieldMask::EMPTY`]
    /// (elects nothing, so this always returns `None`), never `full_for`
    /// (a render convenience, not an election fallback).
    #[inline]
    #[must_use]
    pub fn elected(self, mask: &crate::class_view::WideFieldMask, locus: Locus) -> Option<i8> {
        if !mask.has(locus as u8) {
            return None;
        }
        let v = self.at(locus);
        if v == 0 {
            None
        } else {
            Some(v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_roundtrip_is_loss_free() {
        let reg = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44,
        ];
        let w = CausalWitnessFacet::from_register(reg);
        assert_eq!(w.to_register(), reg);
    }

    #[test]
    fn signed_nibble_range_is_minus8_to_plus7() {
        let mut w = CausalWitnessFacet::ZERO;
        for slot in 0..WITNESS_LOCI {
            w.set(slot, 7);
            assert_eq!(w.get(slot), 7, "max +7 at slot {slot}");
            w.set(slot, -8);
            assert_eq!(w.get(slot), -8, "min −8 at slot {slot}");
            w.set(slot, 0);
            assert_eq!(w.get(slot), 0, "unbound at slot {slot}");
        }
    }

    #[test]
    fn set_clamps_out_of_range() {
        let mut w = CausalWitnessFacet::ZERO;
        w.set(0, 100);
        assert_eq!(w.get(0), 7);
        w.set(1, -100);
        assert_eq!(w.get(1), -8);
    }

    #[test]
    fn named_accessors_hit_the_right_slots() {
        let w = CausalWitnessFacet::ZERO
            .with(Locus::Kausal, -3)
            .with(Locus::Antecedent, -1)
            .with(Locus::Quorum, 2)
            .with(Locus::Contradiction, -5);
        assert_eq!(w.at(Locus::Kausal), -3);
        assert_eq!(w.cause(), -3);
        assert_eq!(w.antecedent(), -1);
        assert_eq!(w.quorum(), 2);
        assert_eq!(w.contradiction(), -5);
        assert_eq!(w.at(Locus::Temporal), 0); // unbound
        assert_eq!(w.bound_count(), 4);
    }

    #[test]
    fn slot_index_matches_locus_discriminant() {
        for (i, l) in Locus::ALL.iter().enumerate() {
            assert_eq!(*l as usize, i, "{} slot", l.label());
            assert!(!l.label().is_empty(), "named locus has a label");
        }
    }

    #[test]
    fn reserved_slots_16_to_23_have_empty_labels() {
        for (slot, label) in LOCUS_LABELS.iter().enumerate().skip(NAMED_LOCI) {
            assert_eq!(*label, "", "slot {slot} reserved-empty");
        }
    }

    #[test]
    fn resolves_to_respects_window_and_bounds() {
        let w = CausalWitnessFacet::ZERO
            .with(Locus::Kausal, -3)
            .with(Locus::Temporal, 4);
        // self at position 5 in a 10-node stream
        assert_eq!(w.resolves_to(Locus::Kausal, 5, 10), Some(2)); // 5 + (−3)
        assert_eq!(w.resolves_to(Locus::Temporal, 5, 10), Some(9)); // 5 + 4
        assert_eq!(w.resolves_to(Locus::Modal, 5, 10), None); // unbound
                                                              // out of bounds
        assert_eq!(w.resolves_to(Locus::Kausal, 1, 10), None); // 1 + (−3) < 0
        assert_eq!(w.resolves_to(Locus::Temporal, 8, 10), None); // 8 + 4 ≥ 10
    }

    #[test]
    fn window_agreement_is_loci_convergence() {
        let a = CausalWitnessFacet::ZERO
            .with(Locus::Kausal, -3)
            .with(Locus::SMeaning, 2);
        let b = CausalWitnessFacet::ZERO
            .with(Locus::Kausal, -3) // agrees
            .with(Locus::SMeaning, 5); // disagrees (different offset)
        assert!(a.agrees_at(b, Locus::Kausal));
        assert!(!a.agrees_at(b, Locus::SMeaning));
        // unbound never agrees
        assert!(!a.agrees_at(b, Locus::Modal));
        assert_eq!(a.agreement_count(b), 1);
    }

    /// I-LEGACY-API-FEATURE-GATED field-isolation matrix: writing slot `k`
    /// must change ONLY slot `k` — every other slot (named or reserved) must
    /// read back its prior value. A real loop over all 24 slots, not a spot
    /// check — mandatory per `.claude/v3/soa_layout/witness-nibble-lane.md` §6.
    #[test]
    fn field_isolation_matrix_all_24_slots() {
        for k in 0..WITNESS_LOCI {
            let mut w = CausalWitnessFacet::ZERO;
            // Give every OTHER slot a distinctive, non-zero prior value first
            // so "unchanged" is actually falsifiable (not just "still 0").
            for other in 0..WITNESS_LOCI {
                if other != k {
                    // deterministic distinctive value in [-8, 7], never 0
                    let v = ((other as i32 % 15) - 7) as i8;
                    let v = if v == 0 { 1 } else { v };
                    w.set(other, v);
                }
            }
            let before: Vec<i8> = (0..WITNESS_LOCI).map(|s| w.get(s)).collect();

            // Write a distinctive value into slot k ONLY.
            let distinctive = if k % 2 == 0 { 7 } else { -8 };
            w.set(k, distinctive);

            assert_eq!(w.get(k), distinctive, "slot {k} took the write");
            for (other, &prior) in before.iter().enumerate() {
                if other != k {
                    assert_eq!(
                        w.get(other),
                        prior,
                        "slot {other} must be unchanged by a write to slot {k}"
                    );
                }
            }
        }
    }

    /// The borrowed view and the by-value copy path must agree on every slot
    /// for a non-trivial register.
    #[test]
    fn view_and_copy_agree_on_all_slots() {
        let reg = [
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0x33, 0x44,
        ];
        let copy = CausalWitnessFacet::from_register(reg);
        let view = CausalWitnessFacet::from_register_ref(&reg);
        assert_eq!(copy, *view, "copy and borrowed view are equal facets");
        for slot in 0..WITNESS_LOCI {
            assert_eq!(
                copy.get(slot),
                view.get(slot),
                "slot {slot}: copy and view disagree"
            );
        }
        assert_eq!(view.to_register(), reg);
    }

    /// Sign-extension edges: -8 and +7 round-trip through EVERY slot, and 0
    /// reads back as 0 (the unbound sentinel) at every slot too.
    #[test]
    fn sign_extension_edges_round_trip_every_slot() {
        for slot in 0..WITNESS_LOCI {
            let mut w = CausalWitnessFacet::ZERO;
            w.set(slot, -8);
            assert_eq!(w.get(slot), -8, "slot {slot}: -8 round-trip");
            w.set(slot, 7);
            assert_eq!(w.get(slot), 7, "slot {slot}: +7 round-trip");
            w.set(slot, 0);
            assert_eq!(w.get(slot), 0, "slot {slot}: unbound sentinel");
        }
    }

    /// Nibble order (byte/nibble law, witness-nibble-lane.md §2): writing
    /// slot `k` affects ONLY byte `k/2`; even `k` occupies the LOW nibble,
    /// odd `k` occupies the HIGH nibble. Constructed by hand against the raw
    /// bytes, not via the accessors under test.
    #[test]
    fn nibble_order_matches_byte_nibble_law() {
        // Even slot 4 -> low nibble of byte 2.
        let mut w = CausalWitnessFacet::ZERO;
        w.set(4, 5); // 5 = 0b0101, fits unsigned nibble range
        let reg = w.to_register();
        assert_eq!(reg[2], 0x05, "even slot 4 -> low nibble of byte 2");
        assert_eq!(reg[0], 0x00, "no other byte touched");
        assert_eq!(reg[1], 0x00, "no other byte touched");

        // Odd slot 5 -> high nibble of byte 2 (same byte as slot 4).
        let mut w2 = CausalWitnessFacet::ZERO;
        w2.set(5, 5);
        let reg2 = w2.to_register();
        assert_eq!(reg2[2], 0x50, "odd slot 5 -> high nibble of byte 2");

        // Both slots 4 and 5 together pack into one byte, low+high.
        let mut w3 = CausalWitnessFacet::ZERO;
        w3.set(4, 5);
        w3.set(5, 5);
        let reg3 = w3.to_register();
        assert_eq!(reg3[2], 0x55, "slots 4 (low) and 5 (high) share byte 2");

        // Hand-constructed register: byte 0 = 0xBA means slot0=0xA(low)=-6
        // sign-extended, slot1=0xB(high)=-5 sign-extended.
        let hand = CausalWitnessFacet::from_register([0xBA, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(hand.get(0), -6, "low nibble of byte 0 = slot 0");
        assert_eq!(hand.get(1), -5, "high nibble of byte 0 = slot 1");
    }

    /// Non-trivial source: EVERY named locus is bound (16 bound slots).
    fn fully_bound_witness() -> CausalWitnessFacet {
        Locus::ALL
            .iter()
            .enumerate()
            .fold(CausalWitnessFacet::ZERO, |w, (i, &l)| {
                // deterministic distinctive nonzero value in [-8, 7]
                let v = ((i as i32 % 15) - 7) as i8;
                let v = if v == 0 { 1 } else { v };
                w.with(l, v)
            })
    }

    /// **Can-fire:** an anaphora-shaped election (`Antecedent` only, slot 7)
    /// over a fully-bound source yields a projection strictly smaller than
    /// the source — asserting the ACTUAL counts, not merely "different".
    #[test]
    fn project_can_fire_anaphora_election_shrinks_bound_count() {
        let source = fully_bound_witness();
        assert_eq!(
            source.bound_count(),
            NAMED_LOCI,
            "source: all 16 named loci bound"
        );

        let anaphora_mask =
            crate::class_view::WideFieldMask::from_positions(&[Locus::Antecedent as u8]);
        let projected = source.project(&anaphora_mask);

        assert_eq!(
            projected.bound_count(),
            1,
            "projection admits exactly the one elected, bound locus"
        );
        assert!(
            projected.bound_count() < source.bound_count(),
            "projection ({}) must be strictly smaller than source ({})",
            projected.bound_count(),
            source.bound_count()
        );
        assert!(projected.is_bound(Locus::Antecedent));
        assert_eq!(
            projected.at(Locus::Antecedent),
            source.at(Locus::Antecedent)
        );
    }

    /// **Can-stay-silent (the twin, mandatory per the anti-eigenvalue rule):**
    /// the SAME anaphora election must NOT surface TEKAMOLO (0..4) or SPO
    /// (4..7) slots, even though the source has them bound. A mask that
    /// admits everything carries exactly as much information as one that
    /// admits nothing — so the silence must be checked as hard as the fire.
    #[test]
    fn project_can_stay_silent_tekamolo_and_spo_excluded() {
        let source = fully_bound_witness();
        let anaphora_mask =
            crate::class_view::WideFieldMask::from_positions(&[Locus::Antecedent as u8]);

        // TEKAMOLO: Temporal, Kausal, Modal, Lokal (slots 0..4) — all bound
        // in `source`, none elected by `anaphora_mask`.
        for locus in [Locus::Temporal, Locus::Kausal, Locus::Modal, Locus::Lokal] {
            assert!(source.is_bound(locus), "source has {} bound", locus.label());
            assert_eq!(
                source.elected(&anaphora_mask, locus),
                None,
                "{} is bound in source but NOT elected — must read None",
                locus.label()
            );
        }

        // SPO meaning-grounding: SMeaning, PMeaning, OMeaning (slots 4..7).
        for locus in [Locus::SMeaning, Locus::PMeaning, Locus::OMeaning] {
            assert!(source.is_bound(locus), "source has {} bound", locus.label());
            assert_eq!(
                source.elected(&anaphora_mask, locus),
                None,
                "{} is bound in source but NOT elected — must read None",
                locus.label()
            );
        }
    }

    /// **Fail-closed:** `WideFieldMask::EMPTY` elects nothing — the
    /// projection has zero bound slots and every `elected(...)` read is
    /// `None`, even over a fully-bound source.
    #[test]
    fn empty_mask_projects_nothing_and_elects_nothing() {
        let source = fully_bound_witness();
        let empty = crate::class_view::WideFieldMask::EMPTY;

        let projected = source.project(&empty);
        assert_eq!(projected.bound_count(), 0, "EMPTY mask elects nothing");
        assert_eq!(projected, CausalWitnessFacet::ZERO);

        for &locus in Locus::ALL.iter() {
            assert_eq!(
                source.elected(&empty, locus),
                None,
                "{} must read None under the EMPTY (fail-closed) mask",
                locus.label()
            );
        }
    }

    /// **Elected-but-unbound is independent from bound-but-unelected:** a
    /// locus that IS elected but whose slot is `0` (unbound) still reads
    /// `None` — proving `elected` checks both conditions, not just the mask.
    #[test]
    fn elected_but_unbound_locus_reads_none() {
        // Kausal is elected but left unbound (0); Antecedent is bound but
        // NOT elected — the two failure modes, side by side.
        let w = CausalWitnessFacet::ZERO.with(Locus::Antecedent, -1);
        let mask = crate::class_view::WideFieldMask::from_positions(&[Locus::Kausal as u8]);

        assert!(
            !w.is_bound(Locus::Kausal),
            "Kausal is unbound in the source"
        );
        assert_eq!(
            w.elected(&mask, Locus::Kausal),
            None,
            "elected but unbound → None"
        );
        assert_eq!(
            w.elected(&mask, Locus::Antecedent),
            None,
            "bound but unelected → None (independent failure mode)"
        );
    }
}
