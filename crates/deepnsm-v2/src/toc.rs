//! `toc` — spawning the table of contents as an HHTL tree of SoA nodes.
//!
//! Operator: *"Brutal heißt, bevor wir über das Buch nachdenken wird ein
//! Inhaltsverzeichnis als HHTL-Baum mit SoA-Knoten erstellt."* "Brutal" names
//! the SEQUENCING: the TOC is minted as the FULL tree skeleton **before** any
//! triple-level reasoning touches a verse — no incremental, content-driven
//! tree-growing.
//!
//! # Why the tree has to exist first — two reasons, not one
//!
//! 1. **Addressing.** `D-ACR-12`'s ascent primitive walks `NiblePath::parent()`
//!    until it finds content. Over an unspawned tree an ascent from a verse
//!    witness terminates at an unminted gap. Top-down construction makes every
//!    ancestor real by definition.
//! 2. **Structure-bearing basins.** `E-HERMENEUTIK-RUNG-LADDER-1`: a basin
//!    keyed by a bare SUBJECT is rung-0 — it keeps grammatical adjacency and
//!    discards genre, pericope and canonical position, leaving SIZE as the only
//!    between-basin signal (which is what three independent gates then
//!    measured). A basin promoted at a TOC node's key IS keyed by literary
//!    unit. The tree is what makes that possible; the rail already supports it
//!    with no byte change, because a node's cascade position is in its own key.
//!
//! # Two nibbles per level, and why that is the canon rather than a workaround
//!
//! `FAN_OUT = 16` — one nibble addresses 0..15. The KJV does not fit that at any
//! level: 66 books, chapters to 150 (Psalms), verses to 176 (Psalm 119). So each
//! level spends **two** nibbles (capacity 256), which is exactly the canon's
//! *"scale is the next cascade level, never field-widening"* — the answer to
//! "more than 16 children" is more levels, never a wider nibble.
//!
//! Depth cost: the basin root nibble (1) + book 2 + chapter 2 + verse 2 =
//! **7 of `MAX_DEPTH`'s 16**, so the whole KJV addresses inside one `u64` path
//! with room left over. The root nibble is easy to forget —
//! [`NiblePath::root`] already consumes depth 1, so a book lands at depth 3,
//! not 2; the depth assertion in the tests is what caught that.
//!
//! # What this module does NOT do
//!
//! It mints ADDRESSES, not rows: a `TocEntry` is a path plus the coordinate it
//! encodes. Writing SoA rows at those paths is the promoter's job, and reading
//! them is gated by the `le-contract.md` §3b jc-pillar rule like any lane.

use lance_graph_contract::hhtl::{NiblePath, FAN_OUT, MAX_DEPTH};

/// Nibbles one TOC level occupies. Two ⇒ 256 children per level, which covers
/// every KJV level with headroom (max is 176 verses in Psalm 119).
pub const NIBBLES_PER_LEVEL: u8 = 2;

/// Largest ordinal one level can address (`16² − 1`).
pub const MAX_PER_LEVEL: u16 = (FAN_OUT as u16) * (FAN_OUT as u16) - 1;

// ── Compile-time sizing guarantees ───────────────────────────────────────
// The KJV's real extremes must fit the two-nibble carve, and a verse address
// must leave depth for deeper routing. These are properties of the CONSTANTS,
// so they are asserted at compile time for every build — not in a test that
// only runs under `cargo test`. (Same precedent as `canonical_node.rs`'s
// `const _` row-size asserts.)
const _: () = assert!(66 <= MAX_PER_LEVEL, "66 books must address in one level");
const _: () = assert!(150 <= MAX_PER_LEVEL, "Psalms has 150 chapters");
const _: () = assert!(176 <= MAX_PER_LEVEL, "Psalm 119 has 176 verses");
const _: () = assert!(
    TocLevel::Verse.depth() + NIBBLES_PER_LEVEL <= MAX_DEPTH,
    "a verse address must leave depth for a level below it"
);

/// Which tier of the book tree an entry sits at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TocLevel {
    /// A book (Genesis …) — depth 3 (basin 1 + 2).
    Book,
    /// A chapter within a book — depth 5.
    Chapter,
    /// A verse within a chapter — depth 7.
    Verse,
}

impl TocLevel {
    /// Path depth in nibbles at this level, **including the basin root
    /// nibble** — the off-by-one that a depth assertion caught during the
    /// build: `NiblePath::root` is already depth 1.
    #[must_use]
    pub const fn depth(self) -> u8 {
        match self {
            TocLevel::Book => 1 + NIBBLES_PER_LEVEL,
            TocLevel::Chapter => 1 + 2 * NIBBLES_PER_LEVEL,
            TocLevel::Verse => 1 + 3 * NIBBLES_PER_LEVEL,
        }
    }
}

/// One minted node of the skeleton: its address and the coordinate it encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TocEntry {
    /// The HHTL address.
    pub path: NiblePath,
    /// Tier.
    pub level: TocLevel,
    /// 0-based book ordinal.
    pub book: u16,
    /// 1-based chapter, `0` at [`TocLevel::Book`].
    pub chapter: u16,
    /// 1-based verse, `0` above [`TocLevel::Verse`].
    pub verse: u16,
}

/// Extend `base` by `value` encoded in [`NIBBLES_PER_LEVEL`] nibbles, high
/// nibble first. `None` if `value` exceeds [`MAX_PER_LEVEL`] or the path is
/// full — a **refusal, never a fold**: silently wrapping a 256th chapter onto
/// chapter 0 would address two literary units with one node.
#[must_use]
pub fn descend(base: NiblePath, value: u16) -> Option<NiblePath> {
    if value > MAX_PER_LEVEL || base.depth() + NIBBLES_PER_LEVEL > MAX_DEPTH {
        return None;
    }
    let hi = u8::try_from(value >> 4).ok()?;
    let lo = u8::try_from(value & 0x0F).ok()?;
    base.try_child(hi)?.try_child(lo)
}

/// The corpus outline the spawn walks: per book, the verse count of each of its
/// chapters. `chapters[b][c]` = verses in book `b`, chapter `c`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CorpusToc {
    /// Per book, per chapter, the verse count.
    pub chapters: Vec<Vec<u16>>,
}

impl CorpusToc {
    /// Derive the outline from an ordered `(chapter, verse)` marker stream —
    /// what [`crate::corpus`]'s `d+:d+` markers give.
    ///
    /// **A book opens at `1:1`.** Every book starts at chapter 1 verse 1, and
    /// `(1,1)` never recurs inside a book, so that marker is exactly the book
    /// boundary. An inference from the corpus's own numbering, not an external
    /// table.
    ///
    /// **⊘ CORRECTED 2026-08-22 by the real corpus.** The first rule here was
    /// "a chapter RESET opens a book" — chapter 1 following a higher chapter.
    /// Run against the real KJV it found **61 books, not 66**, and the missing
    /// five are exactly the single-chapter books (Obadiah, Philemon, 2 John,
    /// 3 John, Jude): such a book is followed by another that also starts at
    /// chapter 1, so `prev_chapter == 1`, no reset is seen, and the two books
    /// MERGE. The merge then costs verses too (the merged chapter keeps the
    /// max verse count, not the sum), which is where 78 missing verse nodes and
    /// 88 unaddressed triples came from.
    ///
    /// The synthetic fixtures could not catch it: none of them contained a
    /// single-chapter book. It took the corpus.
    #[must_use]
    pub fn from_markers(markers: &[(u16, u16)]) -> Self {
        let mut chapters: Vec<Vec<u16>> = Vec::new();
        for &(ch, vs) in markers {
            if ch == 1 && vs == 1 {
                chapters.push(Vec::new());
            }
            let book = match chapters.last_mut() {
                Some(b) => b,
                None => {
                    chapters.push(Vec::new());
                    chapters.last_mut().expect("just pushed")
                }
            };
            while book.len() < ch as usize {
                book.push(0);
            }
            if ch >= 1 {
                let slot = &mut book[ch as usize - 1];
                *slot = (*slot).max(vs);
            }
        }
        Self { chapters }
    }

    /// Books in the outline.
    #[must_use]
    pub fn book_count(&self) -> usize {
        self.chapters.len()
    }

    /// Total verses across the outline.
    #[must_use]
    pub fn verse_count(&self) -> usize {
        self.chapters
            .iter()
            .flat_map(|b| b.iter())
            .map(|v| *v as usize)
            .sum()
    }
}

/// Spawn the FULL skeleton, top-down: every book, then its chapters, then their
/// verses — in that order, so no entry is ever emitted before its parent.
///
/// `basin` is the root nibble the whole corpus hangs under.
///
/// Entries whose ordinal exceeds [`MAX_PER_LEVEL`] are **skipped, with their
/// subtree**, rather than folded onto a sibling. A dropped node is a gap an
/// ascent can detect; a folded one is two literary units sharing an address,
/// which nothing downstream could detect.
#[must_use]
pub fn spawn(toc: &CorpusToc, basin: u8) -> Vec<TocEntry> {
    let root = NiblePath::root(basin);
    if root == NiblePath::EMPTY {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (bi, book) in toc.chapters.iter().enumerate() {
        let Ok(b) = u16::try_from(bi) else { continue };
        let Some(bp) = descend(root, b) else { continue };
        out.push(TocEntry {
            path: bp,
            level: TocLevel::Book,
            book: b,
            chapter: 0,
            verse: 0,
        });
        for (ci, &verses) in book.iter().enumerate() {
            let Ok(c) = u16::try_from(ci + 1) else {
                continue;
            };
            let Some(cp) = descend(bp, c) else { continue };
            out.push(TocEntry {
                path: cp,
                level: TocLevel::Chapter,
                book: b,
                chapter: c,
                verse: 0,
            });
            for v in 1..=verses {
                let Some(vp) = descend(cp, v) else { continue };
                out.push(TocEntry {
                    path: vp,
                    level: TocLevel::Verse,
                    book: b,
                    chapter: c,
                    verse: v,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// A small outline: 2 books, chapters with differing verse counts.
    fn toy() -> CorpusToc {
        CorpusToc {
            chapters: vec![vec![3, 2], vec![1]],
        }
    }

    /// **The reason the tree is spawned at all: every ancestor is real.**
    ///
    /// `D-ACR-12`'s ascent walks `parent()` until it finds content. This asserts
    /// the walk can never leave the minted set — from EVERY node, every ancestor
    /// up to the basin is itself minted. Over an unspawned tree this is exactly
    /// what fails.
    #[test]
    fn every_ancestor_of_every_node_is_itself_minted() {
        let entries = spawn(&toy(), 3);
        let minted: HashSet<NiblePath> = entries.iter().map(|e| e.path).collect();
        let root = NiblePath::root(3);
        assert!(
            !entries.is_empty(),
            "anti-vacuity: the spawn produced nodes"
        );

        let mut checked = 0usize;
        for e in &entries {
            let mut cur = e.path;
            while let Some(p) = cur.parent() {
                if p == root || p.depth() < 2 {
                    break;
                }
                // Only whole levels are nodes; the odd depths are interior
                // nibbles of a two-nibble level, not addresses.
                // Node depths are 3/5/7 (basin + whole levels); the even
                // depths between them are interior nibbles, not addresses.
                if (p.depth() - 1) % NIBBLES_PER_LEVEL == 0 {
                    assert!(
                        minted.contains(&p),
                        "{:?} at depth {} has an unminted ancestor at depth {}",
                        e.level,
                        e.path.depth(),
                        p.depth()
                    );
                    checked += 1;
                }
                cur = p;
            }
        }
        assert!(
            checked >= 6,
            "anti-vacuity: only {checked} ancestor links were actually checked"
        );
    }

    /// Top-down order: a parent is always emitted before any of its descendants.
    /// "Brutal" is a sequencing claim, so it gets a sequencing test.
    #[test]
    fn the_skeleton_is_emitted_parent_before_child() {
        let entries = spawn(&toy(), 1);
        let mut seen: HashSet<NiblePath> = HashSet::new();
        for e in &entries {
            if e.level != TocLevel::Book {
                let parent_depth = e.level.depth() - NIBBLES_PER_LEVEL;
                let p = e.path.prefix(parent_depth).expect("parent prefix exists");
                assert!(seen.contains(&p), "{e:?} emitted before its parent");
            }
            seen.insert(e.path);
        }
        // Anti-vacuity: there ARE children to order.
        assert!(entries.iter().any(|e| e.level == TocLevel::Verse));
    }

    /// Counts and depths match the outline exactly — no node invented, none lost.
    #[test]
    fn the_skeleton_covers_the_outline_exactly() {
        let t = toy();
        let entries = spawn(&t, 5);
        let n = |lvl: TocLevel| entries.iter().filter(|e| e.level == lvl).count();
        assert_eq!(n(TocLevel::Book), t.book_count());
        assert_eq!(n(TocLevel::Chapter), 3, "2 + 1 chapters");
        assert_eq!(n(TocLevel::Verse), t.verse_count(), "3 + 2 + 1 verses");
        for e in &entries {
            assert_eq!(e.path.depth(), e.level.depth(), "{e:?} depth mismatch");
        }
        // Every address is distinct — two literary units may never share a node.
        let uniq: HashSet<NiblePath> = entries.iter().map(|e| e.path).collect();
        assert_eq!(uniq.len(), entries.len(), "an address was reused");
    }

    /// **Refusal, never a fold.** A level ordinal past 255 has no address; the
    /// node and its subtree are skipped rather than wrapped onto a sibling.
    /// Both halves: 255 addresses, 256 refuses.
    #[test]
    fn an_ordinal_past_the_level_capacity_is_refused_not_folded() {
        let base = NiblePath::root(2);
        assert_eq!(MAX_PER_LEVEL, 255);
        assert!(descend(base, 255).is_some(), "255 is addressable");
        assert_eq!(descend(base, 256), None, "256 must refuse");
        // …and distinctly: 255 and 0 are different addresses, so nothing folded.
        assert_ne!(descend(base, 255), descend(base, 0));

        // A chapter past capacity drops its verses WITH it, rather than
        // re-pointing them at chapter 0's node.
        let over = CorpusToc {
            chapters: vec![vec![1; 300]],
        };
        let entries = spawn(&over, 2);
        let chapters = entries
            .iter()
            .filter(|e| e.level == TocLevel::Chapter)
            .count();
        assert_eq!(chapters, 255, "only the addressable chapters are minted");
        let uniq: HashSet<NiblePath> = entries.iter().map(|e| e.path).collect();
        assert_eq!(
            uniq.len(),
            entries.len(),
            "a dropped node was folded instead"
        );
    }

    /// Book boundaries come from `1:1`, and — the case the real KJV caught —
    /// **two single-chapter books in a row must still be two books.** The old
    /// chapter-reset rule merged them, losing 5 books and 78 verses on the real
    /// corpus. No synthetic fixture had a single-chapter book; this one does.
    #[test]
    fn two_single_chapter_books_in_a_row_stay_two_books() {
        // Obadiah-shaped: one chapter, then another one-chapter book.
        let t = CorpusToc::from_markers(&[(1, 1), (1, 2), (1, 1), (1, 2), (1, 3)]);
        assert_eq!(t.book_count(), 2, "the chapter-reset rule merged these");
        assert_eq!(t.chapters[0], vec![2]);
        assert_eq!(t.chapters[1], vec![3]);
        assert_eq!(t.verse_count(), 5, "no verse may be lost to a merge");
    }

    /// Book boundaries come from `1:1`, and the derivation is falsifiable: a
    /// stream with a second `1:1` yields two books, one without yields one.
    #[test]
    fn book_boundaries_come_from_chapter_one_verse_one() {
        let two = CorpusToc::from_markers(&[(1, 1), (1, 2), (2, 1), (1, 1), (1, 2)]);
        assert_eq!(
            two.book_count(),
            2,
            "the reset at the 4th marker opens a book"
        );
        assert_eq!(two.chapters[0], vec![2, 1]);
        assert_eq!(two.chapters[1], vec![2]);

        // Silence half: no reset ⇒ one book, so the rule discriminates.
        let one = CorpusToc::from_markers(&[(1, 1), (2, 1), (3, 1)]);
        assert_eq!(one.book_count(), 1);
        assert_eq!(one.verse_count(), 3);
    }

    /// The level depths are what the module claims — the off-by-one the basin
    /// root nibble causes, pinned. (The CAPACITY claims are `const _` asserts
    /// at module level: they are properties of constants, so they hold at
    /// compile time for every build, not only under `cargo test`.)
    #[test]
    fn the_level_depths_include_the_basin_root_nibble() {
        assert_eq!(TocLevel::Book.depth(), 3, "1 basin + 2, not 2");
        assert_eq!(TocLevel::Chapter.depth(), 5);
        assert_eq!(TocLevel::Verse.depth(), 7);
        // Each level is exactly NIBBLES_PER_LEVEL below the next.
        assert_eq!(
            TocLevel::Chapter.depth() - TocLevel::Book.depth(),
            NIBBLES_PER_LEVEL
        );
        assert_eq!(
            TocLevel::Verse.depth() - TocLevel::Chapter.depth(),
            NIBBLES_PER_LEVEL
        );
        // And a real spawned node agrees with the declared depth — the tie
        // between the constant and the code that emits addresses.
        let e = spawn(&toy(), 4);
        for x in &e {
            assert_eq!(x.path.depth(), x.level.depth());
        }
    }
}
