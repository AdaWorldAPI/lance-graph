//! `promote` — writing basins as SoA rows **at TOC node keys**.
//!
//! The step `D-ACR-6` was blocked on, now that its rail is minted and the tree
//! is spawned. A basin stops being a partition held in RAM and becomes a row
//! addressed at a literary unit.
//!
//! # Why the key is the whole point
//!
//! `E-HERMENEUTIK-RUNG-LADDER-1`: a basin keyed by a bare SUBJECT is rung-0 —
//! it keeps grammatical adjacency and discards genre, pericope and canonical
//! position, leaving SIZE as the only between-basin signal (which is what three
//! independent gates then measured). Promoting at a [`TocEntry`]'s address
//! keys the basin by **literary unit**, and it needs no change to the rail,
//! because a node's cascade position lives in its own key.
//!
//! So this module writes TWO things into one row and they answer different
//! questions:
//!
//! | half | carries | question |
//! |---|---|---|
//! | the KEY | classid + the TOC path folded into HEEL/HIP/TWIG/LEAF | *where in the book* |
//! | the LANE | [`BasinRow`] — subject, count, self-code, version range | *what was observed there* |
//!
//! # The tier fold is generic, not hand-carved
//!
//! [`tiers_of`] packs a [`NiblePath`]'s nibbles **left-aligned** into the
//! 16-nibble space the canon's tiers span, so
//! `NiblePath::from_guid_prefix_v3(key).prefix(depth)` returns the original
//! path by construction. Carving book/chapter/verse into bytes by hand would
//! split a chapter across two tiers and would be a second home for an address
//! the path already holds.
//!
//! # What this module does not decide
//!
//! **The classid is a parameter.** A corpus/book concept is a MINT, and minting
//! is not this module's call — it takes the classid it is given and refuses
//! nothing about it. Likewise `self_code` is copied from the basin's own
//! [`Cam96`](crate::space::Cam96) when one exists and left zero when it does
//! not; a fabricated centroid would be worse than an honest absence.

use crate::basin::BasinCode;
use crate::toc::TocEntry;
use lance_graph_contract::canonical_node::{
    classid_read_mode, EdgeBlock, NodeGuid, NodeRow, TailVariant, ValueTenant,
};
use lance_graph_contract::episodic_basin::{BasinRow, BASIN_ROW_BYTES};
use lance_graph_contract::facet::FacetCascade;
use lance_graph_contract::hhtl::{NiblePath, MAX_DEPTH};
use lance_graph_contract::tekamolo_facet::TekamoloFacet;

/// The TEKAMOLO facet's on-row width — the 16-byte V3 4+12 facet.
pub const TEKAMOLO_BYTES: usize = 16;

/// Fold a path into the four canonical tiers, **left-aligned**.
///
/// The v3 read (`NiblePath::from_guid_prefix_v3`) assembles
/// `heel<<48 | hip<<32 | twig<<16 | leaf` as a full 16-nibble path, so a path
/// shorter than 16 must occupy the HIGH nibbles for its prefix to survive the
/// round trip. `packed()` right-aligns, hence the shift.
#[must_use]
pub fn tiers_of(path: NiblePath) -> (u16, u16, u16, u16) {
    let (bits, depth) = path.packed();
    let shift = 4 * u32::from(MAX_DEPTH - depth);
    let full = bits << shift;
    (
        ((full >> 48) & 0xFFFF) as u16,
        ((full >> 32) & 0xFFFF) as u16,
        ((full >> 16) & 0xFFFF) as u16,
        (full & 0xFFFF) as u16,
    )
}

/// The node key for a TOC address under `classid`, or `None` if `classid`
/// cannot carry one.
///
/// Minted through `mint_for(classid_read_mode(classid).tail_variant, …)` — the
/// symmetric spine — never `NodeGuid::new`.
///
/// # Why this REFUSES the V1 tail instead of minting it
///
/// The canon forbids the V1 tail for new units, and this is a promoter for new
/// units only. Refusing matters because the fallback is otherwise **silent** in
/// two stacked ways:
///
/// 1. A classid with no registry entry resolves to `ReadMode::DEFAULT`, whose
///    `tail_variant` is `V1`.
/// 2. `mint_for` itself has a `#[cfg(not(feature = "guid-v2-tail"))]` arm that
///    maps V2/V3 back onto the V1 layout "so the crate compiles".
///
/// Either path produces a key that looks fine — `Debug` prints, `Eq` works,
/// the HHTL prefix even round-trips while `family == 0` — but has dropped
/// `leaf` and written the tail as `family:u24 ++ identity:u24` at bytes 10..16,
/// so `identity_v2()` (bytes 14..16) reads something else entirely. That was
/// live in this crate until an identity assertion in `toc_hydrate` caught it.
///
/// `None` means *this concept has no mint yet* — an OGAR question, not
/// something to paper over with a V1 key.
#[must_use]
pub fn key_at(classid: u32, path: NiblePath, family: u32, identity: u32) -> Option<NodeGuid> {
    let tail = classid_read_mode(classid).tail_variant;
    if tail == TailVariant::V1 {
        return None;
    }
    let (heel, hip, twig, leaf) = tiers_of(path);
    let key = NodeGuid::mint_for(tail, classid, heel, hip, twig, leaf, family, identity);
    // The fallback arm above is a cfg, not a runtime branch, so a build without
    // the feature would return a V1 key from a V3 request and this is the only
    // place that can still see it.
    debug_assert_eq!(
        key.identity_v2(),
        u16::try_from(identity).unwrap_or(u16::MAX),
        "minted key does not read its own identity back — V1 fallback compiled in?"
    );
    Some(key)
}

/// A [`BasinRow`] from this crate's own [`BasinCode`], over a version range.
///
/// `member_count` saturates rather than wraps: a basin with more than 65_535
/// members reports the ceiling, which is visibly wrong, instead of a small
/// number that looks plausible.
#[must_use]
pub fn row_of(code: &BasinCode, version_from: u64, version_to: u64) -> BasinRow {
    BasinRow {
        subject: code.subject,
        member_count: u16::try_from(code.members).unwrap_or(u16::MAX),
        self_code: code.self_code,
        version_from,
        version_to,
    }
}

/// Write a basin into a row's [`ValueTenant::EpisodicBasin`] lane, in place.
///
/// The offset is **derived** from the tenant descriptor
/// ([`ValueTenant::value_offset`]), never written as a literal — the table's own
/// rule, after a reservation was recorded as an absolute offset three times
/// running and went stale each time.
pub fn write_lane(row: &mut NodeRow, basin: &BasinRow) {
    let off = ValueTenant::EpisodicBasin.value_offset();
    row.value[off..off + BASIN_ROW_BYTES].copy_from_slice(&basin.to_le_bytes());
}

/// Read the lane back. Returns [`BasinRow::EMPTY`] for an unwritten lane —
/// zero-fallback, so "no basin promoted here" is a value, not an error.
#[must_use]
pub fn read_lane(row: &NodeRow) -> BasinRow {
    let off = ValueTenant::EpisodicBasin.value_offset();
    let mut b = [0u8; BASIN_ROW_BYTES];
    b.copy_from_slice(&row.value[off..off + BASIN_ROW_BYTES]);
    BasinRow::from_le_bytes(&b)
}

/// Promote one basin to a full row at a TOC address.
///
/// `identity` distinguishes rows that share an address — the caller's
/// discriminator, not one this module invents.
#[must_use]
pub fn promote(entry: &TocEntry, basin: &BasinRow, classid: u32, identity: u32) -> Option<NodeRow> {
    let mut row = NodeRow {
        key: key_at(classid, entry.path, 0, identity)?,
        edges: EdgeBlock::default(),
        value: [0u8; 480],
    };
    write_lane(&mut row, basin);
    Some(row)
}

/// Promote many, pairing each basin with the TOC entry it was observed at.
///
/// Pairs are `(entry, basin)` because a basin has no opinion about where it
/// belongs — the promoter's caller decides that, and this signature makes the
/// decision explicit instead of inferring it.
#[must_use]
pub fn promote_all(pairs: &[(TocEntry, BasinRow)], classid: u32) -> Vec<NodeRow> {
    pairs
        .iter()
        .enumerate()
        .filter_map(|(i, (e, b))| promote(e, b, classid, u32::try_from(i).unwrap_or(0)))
        .collect()
}

// ── TEKAMOLO: the tenant every SoA row carries ─────────────────────────────

/// The TEKAMOLO facet for a TOC address — the *when* lane, hydrated.
///
/// `ValueTenant::Tekamolo` is a lane on EVERY SoA row, not a property of the
/// promoted basins: every node and every addressed triple carries its own
/// when/why/how/where address. This builds the one lane this crate has
/// evidence for.
///
/// - **Temporal (when)** = `[book, chapter, verse]`, the node's own place in
///   corpus reading order, read as the lane's `256:256:256` coarse→fine
///   cascade. Narrative position IS the temporal address of a scripture node,
///   and it is exact — no inference.
/// - **Kausal / Modal / Lokal** stay ALL-ZERO. The tenant's own doc fixes the
///   meaning of that: *"Zero-fallback: an all-zero facet reads as unaddressed
///   (no lane asserted), never a wrong circumstance."* This crate extracts no
///   causal connective, no modal auxiliary and no place, so asserting any of
///   the three would be a fabricated circumstance — worse than an absent one.
///
/// Returns `None` if a tier exceeds `u8` — a refusal, never a fold, matching
/// [`crate::toc`]'s rule that silently wrapping two literary units onto one
/// address is the failure to avoid.
#[must_use]
pub fn tekamolo_of(entry: &TocEntry, facet_classid: u32) -> Option<TekamoloFacet> {
    let b = u8::try_from(entry.book).ok()?;
    let c = u8::try_from(entry.chapter).ok()?;
    let v = u8::try_from(entry.verse).ok()?;
    Some(TekamoloFacet::from_lanes(
        facet_classid,
        [b, c, v], // temporal — the only lane with evidence
        [0, 0, 0], // kausal  — unaddressed
        [0, 0, 0], // modal   — unaddressed
        [0, 0, 0], // lokal   — unaddressed
    ))
}

/// Write the TEKAMOLO facet into a row's [`ValueTenant::Tekamolo`] lane.
///
/// Offset derived from the tenant descriptor, never a literal — same rule as
/// [`write_lane`].
pub fn write_tekamolo(row: &mut NodeRow, facet: &TekamoloFacet) {
    let off = ValueTenant::Tekamolo.value_offset();
    row.value[off..off + TEKAMOLO_BYTES].copy_from_slice(&facet.facet().to_bytes());
}

/// Read the TEKAMOLO lane back. An unwritten lane reads all-zero — the
/// documented zero-fallback, i.e. *unaddressed*, not an error.
#[must_use]
pub fn read_tekamolo(row: &NodeRow) -> TekamoloFacet {
    let off = ValueTenant::Tekamolo.value_offset();
    let mut b = [0u8; TEKAMOLO_BYTES];
    b.copy_from_slice(&row.value[off..off + TEKAMOLO_BYTES]);
    TekamoloFacet::new(FacetCascade::from_bytes(&b))
}

/// A row for EVERY TOC entry, each with its TEKAMOLO lane hydrated.
///
/// The basin lane is left at its zero-fallback here; [`promote`] writes that
/// for the subset of addresses a basin was measured at. TEKAMOLO is not that
/// subset — it is on every row.
#[must_use]
pub fn rows_with_tekamolo(entries: &[TocEntry], classid: u32) -> Vec<NodeRow> {
    entries
        .iter()
        .enumerate()
        .filter_map(|(i, e)| {
            let facet = tekamolo_of(e, classid)?;
            let mut row = NodeRow {
                key: key_at(classid, e.path, 0, u32::try_from(i).unwrap_or(0))?,
                edges: EdgeBlock::default(),
                value: [0u8; 480],
            };
            write_tekamolo(&mut row, &facet);
            Some(row)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::toc::{spawn, CorpusToc, TocLevel};
    use std::collections::HashSet;

    const CLASSID: u32 = NodeGuid::CLASSID_OSINT_V3;

    fn toy() -> Vec<TocEntry> {
        spawn(
            &CorpusToc {
                chapters: vec![vec![3, 2], vec![1]],
            },
            3,
        )
    }

    fn basin(subject: u16, members: u32) -> BasinRow {
        BasinRow {
            subject,
            member_count: u16::try_from(members).unwrap_or(u16::MAX),
            self_code: [9; 12],
            version_from: 4,
            version_to: 11,
        }
    }

    /// **The load-bearing claim: the key still IS the TOC address.**
    ///
    /// Folding a path into tiers and reading it back through the canon's own
    /// v3 fold must return the path. If it does not, a promoted basin is at an
    /// address nobody can ascend from — which is precisely the gap the tree was
    /// spawned to close.
    #[test]
    fn the_key_round_trips_to_the_toc_path() {
        let entries = toy();
        assert!(!entries.is_empty(), "anti-vacuity: there are entries");
        let mut checked = 0usize;
        for e in &entries {
            let key = key_at(CLASSID, e.path, 0, 1).expect("V3 classid mints");
            let back = NiblePath::from_guid_prefix_v3(&key);
            assert_eq!(
                back.prefix(e.path.depth()),
                Some(e.path),
                "{e:?}: the key does not carry its own address back"
            );
            checked += 1;
        }
        assert!(checked >= 9, "only {checked} addresses round-tripped");
    }

    /// Distinct literary units get distinct keys. Two basins sharing a key
    /// would be unreadable apart once written.
    #[test]
    fn distinct_toc_entries_get_distinct_keys() {
        let entries = toy();
        let keys: HashSet<[u8; 16]> = entries
            .iter()
            .map(|e| {
                *key_at(CLASSID, e.path, 0, 0)
                    .expect("V3 classid mints")
                    .as_bytes()
            })
            .collect();
        assert_eq!(keys.len(), entries.len(), "an address collided");
        // …and the ancestry survives the fold: a verse's key folds to a path
        // whose parent-prefix is its chapter's path.
        let verse = entries
            .iter()
            .find(|e| e.level == TocLevel::Verse)
            .expect("a verse exists");
        let chapter = entries
            .iter()
            .find(|e| {
                e.level == TocLevel::Chapter && e.book == verse.book && e.chapter == verse.chapter
            })
            .expect("its chapter exists");
        assert!(
            chapter.path.is_ancestor_of(verse.path),
            "the chapter must remain an ancestor of its verse"
        );
    }

    /// TEKAMOLO round-trips through a real row, and writing it touches NO
    /// other byte — the field-isolation half for the second lane.
    #[test]
    fn tekamolo_round_trips_and_disturbs_nothing_else() {
        let e = toy()[0];
        let f = tekamolo_of(&e, CLASSID).expect("toy tiers fit u8");
        let mut row = NodeRow {
            key: key_at(CLASSID, e.path, 0, 1).expect("V3 mints"),
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        };
        write_tekamolo(&mut row, &f);
        assert_eq!(read_tekamolo(&row), f, "the lane must read back");

        let off = ValueTenant::Tekamolo.value_offset();
        for (i, b) in row.value.iter().enumerate() {
            if i < off || i >= off + TEKAMOLO_BYTES {
                assert_eq!(*b, 0, "byte {i} outside the TEKAMOLO lane was written");
            }
        }
        // Anti-vacuity: the lane itself is NOT all-zero, so the sweep above is
        // discriminating rather than asserting a zeroed row is zeroed.
        assert!(
            row.value[off..off + TEKAMOLO_BYTES].iter().any(|b| *b != 0),
            "the lane must carry something"
        );
    }

    /// The two lanes do not overlap: writing one leaves the other intact.
    #[test]
    fn tekamolo_and_basin_lanes_are_disjoint() {
        let e = toy()[0];
        let b = basin(77, 5);
        let f = tekamolo_of(&e, CLASSID).expect("tiers fit");
        let mut row = promote(&e, &b, CLASSID, 3).expect("V3 mints");
        write_tekamolo(&mut row, &f);
        assert_eq!(read_lane(&row), b, "basin survived the TEKAMOLO write");
        assert_eq!(read_tekamolo(&row), f, "TEKAMOLO survived the basin write");
    }

    /// Only the temporal lane is asserted. The other three stay at the
    /// documented zero-fallback — "unaddressed, never a wrong circumstance".
    #[test]
    fn only_the_temporal_lane_is_asserted() {
        let entries = toy();
        let verse = entries
            .iter()
            .find(|e| e.level == TocLevel::Verse)
            .expect("a verse exists");
        let f = tekamolo_of(verse, CLASSID).expect("tiers fit");
        assert_eq!(
            f.temporal(),
            [
                u8::try_from(verse.book).unwrap(),
                u8::try_from(verse.chapter).unwrap(),
                u8::try_from(verse.verse).unwrap()
            ]
        );
        assert_eq!(f.causal(), [0, 0, 0]);
        assert_eq!(f.modal(), [0, 0, 0]);
        assert_eq!(f.local(), [0, 0, 0]);
    }

    /// The temporal lane is a REAL prefix-routing axis: two verses in the same
    /// chapter share more coarse→fine temporal prefix than two in different
    /// books. Without this the lane would be stored and meaningless.
    #[test]
    fn temporal_prefix_routing_separates_near_from_far() {
        use lance_graph_contract::tekamolo_facet::TekamoloRole;
        let mk = |b: u16, c: u16, v: u16| {
            tekamolo_of(
                &TocEntry {
                    path: toy()[0].path,
                    level: TocLevel::Verse,
                    book: b,
                    chapter: c,
                    verse: v,
                },
                CLASSID,
            )
            .expect("tiers fit")
        };
        let a = mk(1, 2, 3);
        let same_chapter = mk(1, 2, 9);
        let same_book = mk(1, 7, 3);
        let other_book = mk(5, 2, 3);
        let s_ch = a.shared(&same_chapter, TekamoloRole::Temporal);
        let s_bk = a.shared(&same_book, TekamoloRole::Temporal);
        let s_ot = a.shared(&other_book, TekamoloRole::Temporal);
        assert!(
            s_ch > s_bk && s_bk > s_ot,
            "temporal prefix must order near→far: same-chapter {s_ch} > same-book {s_bk} > other-book {s_ot}"
        );
        // Identity is the maximum, and it is reachable — the scale is not flat.
        assert_eq!(a.shared(&a, TekamoloRole::Temporal), 3);
    }

    /// A tier past `u8` is REFUSED, not folded onto a wrong address.
    #[test]
    fn a_tier_past_u8_is_refused_not_folded() {
        let bad = TocEntry {
            path: toy()[0].path,
            level: TocLevel::Verse,
            book: 0,
            chapter: 300, // > 255
            verse: 1,
        };
        assert!(tekamolo_of(&bad, CLASSID).is_none());
        // …and a legal one right at the boundary still mints, so the guard is
        // not simply always-None.
        let ok = TocEntry {
            chapter: 255,
            ..bad
        };
        assert!(tekamolo_of(&ok, CLASSID).is_some());
    }

    /// EVERY tree node gets a row with a hydrated lane — TEKAMOLO is not the
    /// basin subset.
    #[test]
    fn every_toc_entry_gets_a_hydrated_row() {
        let entries = toy();
        let rows = rows_with_tekamolo(&entries, CLASSID);
        assert_eq!(rows.len(), entries.len(), "one row per entry");
        for (r, e) in rows.iter().zip(&entries) {
            assert_eq!(
                read_tekamolo(r).temporal(),
                [
                    u8::try_from(e.book).unwrap(),
                    u8::try_from(e.chapter).unwrap(),
                    u8::try_from(e.verse).unwrap()
                ]
            );
        }
    }

    /// A classid with no registry entry resolves to the V1 tail, and the
    /// promoter must REFUSE it rather than mint the forbidden shape.
    ///
    /// This is the can-fire half; `distinct_toc_entries_get_distinct_keys`
    /// above is the can-stay-silent half — it mints on a real V3 classid and
    /// gets `Some` for every entry, so the guard is not simply always-None.
    #[test]
    fn an_unminted_classid_is_refused_not_silently_downgraded() {
        let e = toy()[0];
        // 0x0301_0000 — MONDO's block, which is exactly what this crate used
        // as a corpus stand-in. It has no read-mode entry, so it resolves to
        // ReadMode::DEFAULT (V1) and the old code minted a V1 key from it.
        let unminted = 0x0301_0000;
        assert_eq!(
            classid_read_mode(unminted).tail_variant,
            TailVariant::V1,
            "anti-vacuity: this classid really does resolve to V1"
        );
        assert!(key_at(unminted, e.path, 0, 1).is_none());
        assert!(promote(&e, &basin(1, 1), unminted, 1).is_none());
        // ...and the V3 classid the tests use is genuinely different.
        assert_ne!(classid_read_mode(CLASSID).tail_variant, TailVariant::V1);
        assert!(key_at(CLASSID, e.path, 0, 1).is_some());
    }

    /// The lane round-trips through a real row, and writing it **touches no
    /// other byte of the 480-byte slab** — the row-level field-isolation half.
    #[test]
    fn the_lane_round_trips_and_disturbs_nothing_else() {
        let e = toy()[0];
        let b = basin(77, 5);
        let row = promote(&e, &b, CLASSID, 3).expect("V3 classid mints");
        assert_eq!(
            read_lane(&row),
            b,
            "the lane must read back what was written"
        );

        // Every byte outside the lane is still zero.
        let off = ValueTenant::EpisodicBasin.value_offset();
        for (i, byte) in row.value.iter().enumerate() {
            if (off..off + BASIN_ROW_BYTES).contains(&i) {
                continue;
            }
            assert_eq!(*byte, 0, "byte {i} outside the lane was written");
        }
        // Anti-vacuity: the lane itself is NOT all zero, so the loop above is
        // not trivially true of the whole slab.
        assert!(row.value[off..off + BASIN_ROW_BYTES]
            .iter()
            .any(|b| *b != 0));

        // An unwritten row reads as "no basin", not as an error.
        let bare = NodeRow {
            key: row.key,
            edges: EdgeBlock::default(),
            value: [0u8; 480],
        };
        assert!(read_lane(&bare).is_empty());
    }

    /// `row_of` carries the basin's own Cam96 through unchanged, and saturates
    /// the member count rather than wrapping.
    #[test]
    fn row_of_carries_the_self_code_and_saturates_the_count() {
        let code = BasinCode {
            subject: 42,
            self_code: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            width: 0.5,
            members: 7,
            contradiction: 0.25,
        };
        let r = row_of(&code, 2, 8);
        assert_eq!(r.subject, 42);
        assert_eq!(r.self_code, code.self_code, "the Cam96 passes through");
        assert_eq!(r.member_count, 7);
        assert_eq!((r.version_from, r.version_to), (2, 8));

        // Saturation, not wrap: 70_000 members must not report 4_464.
        let huge = BasinCode {
            members: 70_000,
            ..code
        };
        assert_eq!(row_of(&huge, 0, 1).member_count, u16::MAX);
    }

    /// `promote_all` keeps each basin with the entry it was paired to — a
    /// reordering would put a basin at the wrong literary unit.
    #[test]
    fn promote_all_keeps_each_basin_at_its_own_entry() {
        let entries = toy();
        let pairs: Vec<(TocEntry, BasinRow)> = entries
            .iter()
            .take(4)
            .enumerate()
            .map(|(i, e)| (*e, basin(u16::try_from(i).unwrap() + 1, i as u32 + 1)))
            .collect();
        let rows = promote_all(&pairs, CLASSID);
        assert_eq!(rows.len(), pairs.len());
        for (row, (e, b)) in rows.iter().zip(&pairs) {
            assert_eq!(read_lane(row), *b, "basin landed on the wrong row");
            assert_eq!(
                NiblePath::from_guid_prefix_v3(&row.key).prefix(e.path.depth()),
                Some(e.path),
                "row is keyed at the wrong literary unit"
            );
        }
        // Anti-vacuity: the subjects really differ, so a shuffle would show.
        let subs: HashSet<u16> = pairs.iter().map(|(_, b)| b.subject).collect();
        assert_eq!(subs.len(), pairs.len());
    }
}
