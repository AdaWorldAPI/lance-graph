//! `hydrate` — laying the triple stream over the spawned TOC tree.
//!
//! [`crate::toc`] mints the skeleton; [`crate::promote`] writes basins at its
//! keys. This is the middle step: every SPO triple gets the **address of the
//! verse it was read in**, so a triple is no longer a free-floating
//! `(s, p, o)` but an observation at a literary unit.
//!
//! # What this adds, precisely
//!
//! The triples already exist — `fsm::parse_to_spo` produces them per verse.
//! What did not exist is the JOIN to the tree. Without it a triple's only
//! coordinate is its index in a flat stream, which is exactly the rung-0
//! representation `E-HERMENEUTIK-RUNG-LADDER-1` diagnoses: with canonical
//! position discarded there is no non-size structure left to find.
//!
//! # The chain is the reading order, and it is preserved
//!
//! The triples are kept in corpus order and [`Hydration::chain`] exposes
//! consecutive pairs — the Markov trajectory over the book. Order is therefore
//! load-bearing, not incidental: a sort or a shuffle anywhere in this module
//! destroys the chain while leaving every triple individually correct, which is
//! the kind of break a count-based test cannot see. Hence
//! `the_chain_follows_reading_order_not_the_tree`.
//!
//! # The nodes are where the residue lands
//!
//! The TOC nodes are not only addresses. The alpha-layer eyetracker cascade and
//! the HHTL nodes are what carry the **residue** — the record of what the
//! thinking actually looked at — so a scanpath over this tree is the thinking
//! made visible rather than inferred. [`Hydration::scanpath`] is that sequence
//! in its rawest form: the addresses visited, in reading order.
//!
//! It is a TRAJECTORY, not a set: only consecutive repeats collapse, so a
//! return to a verse later in the stream shows up again. Deduplicating globally
//! would turn the residue into coverage and throw away the revisit — which is
//! the part that carries the thinking.
//!
//! # Scope
//!
//! It ADDRESSES; it does not tag or parse. The per-verse triples are an input,
//! produced by the caller's existing FSM path, so this module has no opinion
//! about tokenization and cannot silently disagree with the pipeline that
//! already runs.

use crate::spo::Spo;
use crate::toc::{spawn, CorpusToc, TocEntry, TocLevel};
use lance_graph_contract::hhtl::NiblePath;

/// One triple, addressed at the verse node it was read in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddressedTriple {
    /// The verse node's HHTL address.
    pub path: NiblePath,
    /// The triple.
    pub spo: Spo,
    /// Position in the corpus-order stream — the Markov index.
    pub order: u32,
}

/// The tree plus the stream laid over it.
#[derive(Debug, Clone, Default)]
pub struct Hydration {
    /// The outline the tree was spawned from.
    pub toc: CorpusToc,
    /// Every minted node, top-down.
    pub entries: Vec<TocEntry>,
    /// Every triple, in corpus reading order.
    pub triples: Vec<AddressedTriple>,
    /// Verses that produced no triple — reported, never hidden.
    pub barren_verses: usize,
    /// Triples whose verse had no addressable node (a level over capacity).
    pub unaddressed: usize,
}

impl Hydration {
    /// Verse nodes in the tree.
    #[must_use]
    pub fn verse_nodes(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.level == TocLevel::Verse)
            .count()
    }

    /// Consecutive `(from, to)` pairs in reading order — the Markov chain over
    /// the book. `n` triples yield `n − 1` links; fewer than two yield none.
    #[must_use]
    pub fn chain(&self) -> Vec<(&AddressedTriple, &AddressedTriple)> {
        self.triples.windows(2).map(|w| (&w[0], &w[1])).collect()
    }

    /// The **scanpath**: addresses in reading order, consecutive repeats
    /// collapsed.
    ///
    /// This is the residue in its rawest form — what a focus mask over the tree
    /// is built from. Only CONSECUTIVE repeats collapse: a verse revisited
    /// later appears again, because a revisit is the signal, not noise. A
    /// global dedup would report coverage and lose the trajectory.
    #[must_use]
    pub fn scanpath(&self) -> Vec<NiblePath> {
        let mut out: Vec<NiblePath> = Vec::new();
        for t in &self.triples {
            if out.last() != Some(&t.path) {
                out.push(t.path);
            }
        }
        out
    }

    /// How many chain links cross a verse boundary — the structure a flat
    /// stream cannot report, and the reason the addressing is worth its cost.
    #[must_use]
    pub fn cross_verse_links(&self) -> usize {
        self.chain()
            .iter()
            .filter(|(a, b)| a.path != b.path)
            .count()
    }
}

/// Lay `per_verse` triples over a tree spawned from `markers`.
///
/// `markers[i]` and `per_verse[i]` describe the SAME verse — index `i` is the
/// join key, so the two must come from one walk of the corpus. A length
/// mismatch is not an error here: the walk stops at the shorter of the two and
/// the shortfall shows up as untouched verse nodes, which a caller can see.
///
/// Verses whose node is missing (a level over the tree's per-level capacity)
/// have their triples counted in [`Hydration::unaddressed`] rather than
/// attached to a neighbour — the same refusal-not-fold rule the spawn uses.
#[must_use]
pub fn hydrate(markers: &[(u16, u16)], per_verse: &[Vec<Spo>], basin: u8) -> Hydration {
    let toc = CorpusToc::from_markers(markers);
    let entries = spawn(&toc, basin);

    // Verse nodes in the order the spawn emitted them, which is the order the
    // markers were read — so the i-th verse marker is the i-th verse node.
    let verse_paths: Vec<NiblePath> = entries
        .iter()
        .filter(|e| e.level == TocLevel::Verse)
        .map(|e| e.path)
        .collect();

    let mut triples = Vec::new();
    let mut barren_verses = 0usize;
    let mut unaddressed = 0usize;
    let mut order = 0u32;

    for (i, spos) in per_verse.iter().enumerate().take(markers.len()) {
        if spos.is_empty() {
            barren_verses += 1;
            continue;
        }
        match verse_paths.get(i) {
            Some(&path) => {
                for spo in spos {
                    triples.push(AddressedTriple {
                        path,
                        spo: *spo,
                        order,
                    });
                    order = order.saturating_add(1);
                }
            }
            None => unaddressed += spos.len(),
        }
    }

    Hydration {
        toc,
        entries,
        triples,
        barren_verses,
        unaddressed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Two books: 2 chapters (2 + 1 verses), then 1 chapter (2 verses) = 5 verses.
    fn markers() -> Vec<(u16, u16)> {
        vec![(1, 1), (1, 2), (2, 1), (1, 1), (1, 2)]
    }

    fn spo(n: u16) -> Spo {
        Spo::new(n, n + 100, n + 200)
    }

    /// **Every triple lands on a real minted verse node.** The join is the whole
    /// deliverable, so it is the test: an address that is not in the tree means
    /// a triple nothing can ascend from.
    #[test]
    fn every_triple_is_addressed_at_a_minted_verse_node() {
        let per_verse = vec![
            vec![spo(1), spo(2)],
            vec![spo(3)],
            vec![],
            vec![spo(4)],
            vec![spo(5)],
        ];
        let h = hydrate(&markers(), &per_verse, 3);

        assert_eq!(h.verse_nodes(), 5, "the outline has 5 verses");
        assert_eq!(h.triples.len(), 5, "2+1+0+1+1 triples");
        assert_eq!(
            h.barren_verses, 1,
            "the empty verse is REPORTED, not hidden"
        );
        assert_eq!(h.unaddressed, 0);

        let minted: HashSet<NiblePath> = h
            .entries
            .iter()
            .filter(|e| e.level == TocLevel::Verse)
            .map(|e| e.path)
            .collect();
        for t in &h.triples {
            assert!(minted.contains(&t.path), "{t:?} is at an unminted address");
        }
        // Anti-vacuity: the triples do NOT all share one address, so
        // "contains" is checking a real spread.
        let used: HashSet<NiblePath> = h.triples.iter().map(|t| t.path).collect();
        assert!(used.len() >= 3, "only {} distinct addresses", used.len());
    }

    /// Triples from ONE verse share that verse's address; triples from
    /// different verses do not. Both halves — otherwise a constant address
    /// would pass the test above.
    #[test]
    fn triples_share_an_address_exactly_when_they_share_a_verse() {
        let per_verse = vec![vec![spo(1), spo(2)], vec![spo(3)], vec![], vec![], vec![]];
        let h = hydrate(&markers(), &per_verse, 2);
        assert_eq!(h.triples.len(), 3);
        assert_eq!(
            h.triples[0].path, h.triples[1].path,
            "same verse ⇒ same address"
        );
        assert_ne!(
            h.triples[1].path, h.triples[2].path,
            "different verse ⇒ different address"
        );
    }

    /// **The chain is READING order, not tree order.** A sort or shuffle would
    /// leave every triple individually correct and silently destroy the Markov
    /// trajectory, which no count-based assertion could see.
    #[test]
    fn the_chain_follows_reading_order_not_the_tree() {
        // **The subjects are DELIBERATELY not ascending.** A first version used
        // 1,2,3,4,5 and a "sort the stream by subject" mutation did not fire —
        // sorted and reading order were the same sequence, so the fixture could
        // not tell them apart. The fixture's SHAPE is part of the coverage.
        let per_verse = vec![
            vec![spo(50), spo(10)],
            vec![spo(40)],
            vec![],
            vec![spo(20)],
            vec![spo(30)],
        ];
        let h = hydrate(&markers(), &per_verse, 4);

        // `order` is dense and ascending across the whole stream.
        let orders: Vec<u32> = h.triples.iter().map(|t| t.order).collect();
        assert_eq!(orders, (0..h.triples.len() as u32).collect::<Vec<_>>());
        // The subjects come back in CORPUS order — the join did not re-sort.
        let subs: Vec<u16> = h.triples.iter().map(|t| t.spo.subject).collect();
        assert_eq!(subs, vec![50, 10, 40, 20, 30]);
        let mut sorted = subs.clone();
        sorted.sort_unstable();
        assert_ne!(
            subs, sorted,
            "anti-vacuity: the fixture must distinguish reading order from sorted order"
        );

        let chain = h.chain();
        assert_eq!(chain.len(), h.triples.len() - 1, "n triples ⇒ n−1 links");
        for (a, b) in &chain {
            assert_eq!(b.order, a.order + 1, "a link must be consecutive");
        }

        // Cross-verse links are the structure only the addressing can report:
        // 4 links here, of which the 1→2 link is WITHIN a verse.
        assert_eq!(h.cross_verse_links(), 3, "3 of 4 links cross a verse");
        assert!(
            h.cross_verse_links() < chain.len(),
            "anti-vacuity: not every link crosses, so the measure discriminates"
        );
    }

    /// **The scanpath is a trajectory, not a set.** A revisit must survive; a
    /// consecutive repeat must collapse. Both halves, because collapsing
    /// everything and collapsing nothing both "pass" a length-only check.
    #[test]
    fn the_scanpath_collapses_repeats_but_keeps_revisits() {
        // v0 gets two triples (a consecutive repeat), v3 is revisited after v4
        // by putting a later triple back on it — the stream order is what
        // decides, so this is expressed through the marker/verse pairing.
        let per_verse = vec![
            vec![spo(1), spo(2)], // verse 0 twice in a row -> ONE scan entry
            vec![spo(3)],         // verse 1
            vec![],
            vec![spo(4)], // verse 3
            vec![spo(5)], // verse 4
        ];
        let h = hydrate(&markers(), &per_verse, 6);
        let sp = h.scanpath();

        assert_eq!(sp.len(), 4, "5 triples over 4 distinct consecutive verses");
        assert!(sp.windows(2).all(|w| w[0] != w[1]), "no consecutive repeat");
        // Anti-vacuity: it really did collapse something.
        assert!(
            sp.len() < h.triples.len(),
            "the consecutive repeat was not collapsed"
        );
        // …and it is not merely the distinct set: the scan follows the stream.
        let distinct: std::collections::HashSet<NiblePath> = sp.iter().copied().collect();
        assert_eq!(distinct.len(), sp.len(), "no revisit in THIS fixture");

        // The revisit half: a stream that returns to an earlier address keeps
        // both visits, so the scanpath is longer than the distinct set.
        let revisit = Hydration {
            triples: vec![
                h.triples[0],
                h.triples[2],
                h.triples[0], // back to the first verse
            ],
            ..Hydration::default()
        };
        let rsp = revisit.scanpath();
        assert_eq!(rsp.len(), 3, "a revisit must appear again");
        let rdistinct: std::collections::HashSet<NiblePath> = rsp.iter().copied().collect();
        assert_eq!(rdistinct.len(), 2, "…while the distinct set stays 2");
        assert!(rsp.len() > rdistinct.len(), "trajectory, not set");
    }

    /// Fewer than two triples is no chain — and one triple is still addressed.
    #[test]
    fn a_single_triple_has_an_address_but_no_chain() {
        let per_verse = vec![vec![spo(9)], vec![], vec![], vec![], vec![]];
        let h = hydrate(&markers(), &per_verse, 1);
        assert_eq!(h.triples.len(), 1);
        assert!(h.chain().is_empty(), "one triple cannot form a link");
        assert_eq!(h.cross_verse_links(), 0);
        // …and the empty stream is safe too.
        let e = hydrate(&markers(), &[], 1);
        assert!(e.triples.is_empty() && e.chain().is_empty());
        assert_eq!(e.verse_nodes(), 5, "the tree still spawned");
    }

    /// A verse whose node is missing has its triples COUNTED, never attached to
    /// a neighbour — the same refusal-not-fold rule the spawn uses.
    #[test]
    fn triples_of_an_unaddressable_verse_are_counted_not_reattached() {
        // One book, one chapter, 300 verses — 45 past the 255 capacity.
        let m: Vec<(u16, u16)> = (1..=300).map(|v| (1u16, v)).collect();
        let per_verse: Vec<Vec<Spo>> = (1..=300).map(|v| vec![spo(v as u16)]).collect();
        let h = hydrate(&m, &per_verse, 5);

        assert_eq!(h.verse_nodes(), 255, "only the addressable verses minted");
        assert_eq!(h.triples.len(), 255);
        assert_eq!(h.unaddressed, 45, "the overflow is reported");
        assert_eq!(h.triples.len() + h.unaddressed, 300, "nothing vanished");
        // No address carries two different verses' triples.
        let used: HashSet<NiblePath> = h.triples.iter().map(|t| t.path).collect();
        assert_eq!(used.len(), 255, "an address was reused by a folded verse");
    }
}
