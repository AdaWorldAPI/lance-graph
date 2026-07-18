// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `awareness_facet` — the **SpoFacet** reading of a `6×(8:8)` content-blind
//! register: reading **A1** of the M20 awareness assembly
//! (`.claude/plans/soa-32-tenant-awareness-redundancy-v1.md`).
//!
//! This is a **reading, not a layout.** It re-labels the same 12 bytes a value
//! lane already holds (`MailboxSoaView::style_rails_at`) or a
//! [`FacetCascade`](crate::facet::FacetCascade) payload
//! ([`crate::facet_schema`]) — nothing here reserves, moves, or stores a byte.
//! Per the `#729` per-row/class doctrine (*"the reading is ClassView-selected
//! PER ROW/CLASS, never per lane"*), WHICH class reads its register as a
//! `SpoFacet` is an OGAR mint (the "Place 2" per-app decision), never a property
//! of these bytes and never a slot in the payload (le-contract §2 slot purity).
//!
//! It reuses the shipped rail convention verbatim — `rail k = (bytes[2k],
//! bytes[2k+1])` (`soa_view.rs::style_rails_at`) — so a `SpoFacet` and a
//! `style_rails_at` reading of the same register agree pair-for-pair.
//!
//! Layout (le-contract §3 **L4**, `6×(8:8)` = `palette256²`): six
//! `(basin, identity)` centroid pairs — three for the semantic SPO triple, three
//! for the episodic-witness triple (the operator's base design: *3 SPO + 3
//! episodicwitness*). Each pair indexes the 256×256 palette distance/compose
//! tables; similarity between two pairs is one table read
//! ([`crate::distance`], Fisher-z cosine-replacement), never a float.

/// A palette256² centroid — `(basin, identity)`. The two bytes index the
/// 256×256 palette distance/compose tables (le-contract §3 L4); the distance
/// between two pairs is a single table read ([`crate::distance`]), the
/// cosine-replacement — not a float.
pub type Palette256Pair = (u8, u8);

/// The **SpoFacet** reading of a `6×(8:8)` register: three semantic-SPO
/// palette256² pairs + three episodic-witness pairs (M20 reading **A1**).
///
/// A re-labeling of six `(u8, u8)` rails, not a distinct storage (see the module
/// docs). Build from rails ([`from_rails`](Self::from_rails)) or the raw 12-byte
/// register ([`from_register`](Self::from_register)); recover either losslessly
/// with [`to_rails`](Self::to_rails) / [`to_register`](Self::to_register).
///
/// ```
/// use lance_graph_contract::awareness_facet::SpoFacet;
/// let reg = [10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61];
/// let f = SpoFacet::from_register(reg);
/// assert_eq!(f.subject, (10, 11));
/// assert_eq!(f.ew_object, (60, 61));
/// assert_eq!(f.spo(), [(10, 11), (20, 21), (30, 31)]);
/// assert_eq!(f.to_register(), reg); // the reading is loss-free
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SpoFacet {
    /// Semantic subject centroid — rail 0.
    pub subject: Palette256Pair,
    /// Semantic predicate centroid — rail 1.
    pub predicate: Palette256Pair,
    /// Semantic object centroid — rail 2.
    pub object: Palette256Pair,
    /// Episodic-witness subject centroid — rail 3.
    pub ew_subject: Palette256Pair,
    /// Episodic-witness predicate centroid — rail 4.
    pub ew_predicate: Palette256Pair,
    /// Episodic-witness object centroid — rail 5.
    pub ew_object: Palette256Pair,
}

impl SpoFacet {
    /// Label six `6×(8:8)` rails (e.g. from
    /// [`MailboxSoaView::style_rails_at`](crate::soa_view::MailboxSoaView::style_rails_at))
    /// as an `SpoFacet`. Rail order:
    /// `[subject, predicate, object, ew_subject, ew_predicate, ew_object]`.
    #[inline]
    #[must_use]
    pub const fn from_rails(rails: [Palette256Pair; 6]) -> Self {
        Self {
            subject: rails[0],
            predicate: rails[1],
            object: rails[2],
            ew_subject: rails[3],
            ew_predicate: rails[4],
            ew_object: rails[5],
        }
    }

    /// The six rails in canonical order — the inverse of
    /// [`from_rails`](Self::from_rails).
    #[inline]
    #[must_use]
    pub const fn to_rails(self) -> [Palette256Pair; 6] {
        [
            self.subject,
            self.predicate,
            self.object,
            self.ew_subject,
            self.ew_predicate,
            self.ew_object,
        ]
    }

    /// Read a raw 12-byte content-blind register as an `SpoFacet`, using the
    /// shipped rail convention `rail k = (bytes[2k], bytes[2k+1])`
    /// (`soa_view.rs`) — so this agrees pair-for-pair with `style_rails_at` on
    /// the same bytes.
    #[inline]
    #[must_use]
    pub const fn from_register(b: [u8; 12]) -> Self {
        Self {
            subject: (b[0], b[1]),
            predicate: (b[2], b[3]),
            object: (b[4], b[5]),
            ew_subject: (b[6], b[7]),
            ew_predicate: (b[8], b[9]),
            ew_object: (b[10], b[11]),
        }
    }

    /// Serialize back to the 12-byte register — the inverse of
    /// [`from_register`](Self::from_register).
    #[inline]
    #[must_use]
    pub const fn to_register(self) -> [u8; 12] {
        [
            self.subject.0,
            self.subject.1,
            self.predicate.0,
            self.predicate.1,
            self.object.0,
            self.object.1,
            self.ew_subject.0,
            self.ew_subject.1,
            self.ew_predicate.0,
            self.ew_predicate.1,
            self.ew_object.0,
            self.ew_object.1,
        ]
    }

    /// The semantic SPO triple `[subject, predicate, object]` (rails 0–2).
    #[inline]
    #[must_use]
    pub const fn spo(self) -> [Palette256Pair; 3] {
        [self.subject, self.predicate, self.object]
    }

    /// The episodic-witness triple `[ew_subject, ew_predicate, ew_object]`
    /// (rails 3–5).
    #[inline]
    #[must_use]
    pub const fn witness(self) -> [Palette256Pair; 3] {
        [self.ew_subject, self.ew_predicate, self.ew_object]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REG: [u8; 12] = [10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61];

    #[test]
    fn register_roundtrip_is_loss_free() {
        let f = SpoFacet::from_register(REG);
        assert_eq!(f.to_register(), REG);
    }

    #[test]
    fn rails_roundtrip_is_loss_free() {
        let rails = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)];
        let f = SpoFacet::from_rails(rails);
        assert_eq!(f.to_rails(), rails);
    }

    #[test]
    fn from_register_uses_the_shipped_rail_convention() {
        // rail k = (bytes[2k], bytes[2k+1]) — must match soa_view::style_rails_at.
        let f = SpoFacet::from_register(REG);
        let expect_rails = [
            (REG[0], REG[1]),
            (REG[2], REG[3]),
            (REG[4], REG[5]),
            (REG[6], REG[7]),
            (REG[8], REG[9]),
            (REG[10], REG[11]),
        ];
        assert_eq!(f.to_rails(), expect_rails);
    }

    #[test]
    fn register_and_rails_paths_agree() {
        let from_reg = SpoFacet::from_register(REG);
        let from_rails = SpoFacet::from_rails(from_reg.to_rails());
        assert_eq!(from_reg, from_rails);
    }

    #[test]
    fn spo_and_witness_split_the_six_rails() {
        let f = SpoFacet::from_register(REG);
        assert_eq!(f.spo(), [(10, 11), (20, 21), (30, 31)]);
        assert_eq!(f.witness(), [(40, 41), (50, 51), (60, 61)]);
    }

    #[test]
    fn default_is_the_zero_register() {
        assert_eq!(SpoFacet::default().to_register(), [0u8; 12]);
        assert_eq!(SpoFacet::from_register([0u8; 12]), SpoFacet::default());
    }
}
