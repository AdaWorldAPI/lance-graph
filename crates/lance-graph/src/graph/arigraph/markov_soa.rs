// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `markov_soa` — the EXPLICIT, AUDITABLE, **vocabulary-agnostic** SoA-window
//! proposer (the Markov *wave* over a window of the Markov *particle* arc).
//!
//! ## markov_soa IS AriGraph (cold path promoted to hot path)
//!
//! This is not a generic projector that merely *lives in* AriGraph — **it is
//! AriGraph**. AriGraph is a Markov chain in the **cold path**; `markov_soa` is
//! that same chain **promoted to the hot path** (the per-mailbox SoA). Same
//! object, same agnostic nature, hot instead of cold. EW64 / the `CausalEdge64`
//! W-slot → witness arc is the *particle* (discrete, addressable, exact); this
//! windowed projection is the *wave* (accumulated resonance). Both are AriGraph.
//!
//! It previously lived (wrongly) in `deepnsm`, which made the agnostic hot-path
//! graph depend on a *language sensor* — a layer inversion. Dependency flows
//! AriGraph(core) → sensor, never the reverse.
//!
//! ## AriGraph is agnostic — and is NOT necessarily English
//!
//! AriGraph holds SPO from ANY source (business, GoBD, Wikidata, English text);
//! its agnosticism is structural — the SoA row is three **opaque `u16` ranks**
//! that carry no language. The match metric is **AriGraph's own
//! `cam_pq::DistanceTables`** (the graph's native semantic distance), injected
//! as `Fn(u16, u16) -> u8` so the projector itself names no encoding.
//!
//! **The language layer stays UPSTREAM, in DeepNSM, and never reaches in here.**
//! DeepNSM / COCA-4096 / the grammar templates are the *English-language input
//! sensor*: they scan flat data (usually English), parse it, and EMIT SPO
//! triplets into AriGraph. They must stay English — the grammar templates get
//! messy the instant they are not. Injecting a COCA/language distance into this
//! hot-path graph would be the GoBD-with-Rumi error: running a *language* lens
//! over an *agnostic* graph. Don't. The injected distance here is AriGraph's
//! cam_pq, not a language table. SPO *can* be English (when DeepNSM produced it),
//! but the SoA / AriGraph mailbox-view is never *forced* into a language.
//!
//! ## Strictly a fuzzy proposer — "hybrid+ autocomplete" (Markov #2)
//!
//! Output is a **best-guess match** (System-1 priming, "feels like a Sicilian
//! with a pinch of death trap"): it proposes *where to look* / *what this
//! resembles*, **never asserts truth**. The deterministic particle chain
//! (CE64→witness arc + the 32k SPO-W triplets) ALWAYS confirms. A wrong guess
//! costs a cheap reprioritization, never a wrong answer. **Invariant: the fuzz
//! is only legitimate while leashed to the deterministic chain that confirms it
//! — an unleashed bundle degrades into "sink-in-and-pray" (Markov #3).**
//!
//! ## STATUS: verified (2026-07-01) — compiles + tests green in-sandbox
//!
//! Authored against the grounded `contract::soa_view::MailboxSoaView` surface.
//! The earlier "unverified-offline" caveat (core deps fetch from crates.io) is
//! cleared: with crates.io in the proxy allow-list, the root
//! `[patch.crates-io] ndarray` pointed at the local sibling path, and `protoc`
//! installed, `lance-graph` core compiles and this module's 4 tests pass
//! (part of the 124-green `graph::arigraph` suite). The truly-correct home is
//! still *inside the EW64-in-SoA seam* (P1+P2 of the three-Markovs ordering);
//! this module is the agnostic wave-projector that seam will host.

use lance_graph_contract::soa_view::MailboxSoaView;

/// An SPO triple as three **opaque** `u16` ranks — vocabulary-agnostic. The
/// class above the mailbox says which vocabulary decodes these (COCA / business
/// / QID); the rank itself carries no meaning (C2 agnostic register).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpoRanks {
    /// Subject rank (opaque; vocabulary resolved by the class).
    pub s: u16,
    /// Predicate rank (opaque).
    pub p: u16,
    /// Object rank (opaque; may be a no-role sentinel).
    pub o: u16,
}

/// One row's contribution to a projection, recorded for audit — the thing that
/// makes the wave NOT a black box: every fold is attributable. All integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowContribution {
    /// SoA row index that contributed.
    pub row: usize,
    /// `entity_type`/`class_id` of the row (the class that resolves its vocabulary).
    pub class_id: u16,
    /// `|delta|` from the focal row — the recency/proximity prior, integer.
    pub proximity: u32,
}

/// The provenance of a projection: the ordered list of what folded in. A
/// projection + this + the SoA fully reconstruct the wave — nothing lost.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BundleProvenance {
    /// Source mailbox id (which cohort this projection summarizes).
    pub mailbox_id: u32,
    /// Rows that contributed, in fold order.
    pub contributions: Vec<RowContribution>,
}

impl BundleProvenance {
    /// How many rows contributed a triple.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.contributions.len()
    }
}

/// A deterministic, vocabulary-agnostic projection of a SoA window: the opaque
/// rank-triples in a ±radius window + their provenance. The triples stay
/// **addressable** (no superposition destroys the register); matching is an
/// injected per-vocabulary distance, never float cosine, never a learned embed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WaveProjection {
    /// The opaque rank-triples in the window, in fold order — the explicit content.
    pub triples: Vec<SpoRanks>,
    /// The replayable construction.
    pub provenance: BundleProvenance,
}

impl WaveProjection {
    /// **Best-guess match** to another projection — the System-1 priming read.
    /// Deterministic, integer: for each triple here take the nearest triple
    /// there by mean per-role distance under the injected `dist` closure, then
    /// average. `dist(a, b)` is **AriGraph's own** `cam_pq::DistanceTables`
    /// (the graph's native semantic distance), injected so this function names
    /// no encoding. NOT a language/COCA table — language stays upstream in
    /// DeepNSM. `0.0` if either side is empty.
    #[must_use]
    pub fn best_guess_match(&self, other: &WaveProjection, dist: impl Fn(u16, u16) -> u8) -> f32 {
        if self.triples.is_empty() || other.triples.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0f32;
        for a in &self.triples {
            let mut nearest = u8::MAX;
            for b in &other.triples {
                let d = ((dist(a.s, b.s) as u16 + dist(a.p, b.p) as u16 + dist(a.o, b.o) as u16)
                    / 3) as u8;
                if d < nearest {
                    nearest = d;
                }
            }
            // similarity = 1 - normalized distance (caller's table is the metric;
            // u8::MAX = maximally dissimilar). Integer-derived, deterministic.
            acc += 1.0 - (nearest as f32 / u8::MAX as f32);
        }
        acc / self.triples.len() as f32
    }
}

/// Folds a [`MailboxSoaView`] window into the opaque rank-triples + provenance.
/// Deterministic: same SoA + focal + radius ⇒ identical triples and provenance.
/// `row_triple(row) -> Option<SpoRanks>` resolves a row to its triple (from the
/// deterministic SoA/AriGraph state); untripled rows are skipped, not recorded.
/// The projector invents nothing and names no vocabulary.
#[derive(Debug, Clone, Copy)]
pub struct SoaWavePrimer {
    /// ±window radius over mailboxes (the Markov proximity prior).
    pub radius: u32,
}

impl Default for SoaWavePrimer {
    fn default() -> Self {
        Self { radius: 5 }
    }
}

impl SoaWavePrimer {
    /// New primer with an explicit ±radius window.
    #[must_use]
    pub fn new(radius: u32) -> Self {
        Self { radius }
    }

    /// Project the window centered on `focal_row`.
    pub fn project<V, F>(&self, soa: &V, focal_row: usize, row_triple: F) -> WaveProjection
    where
        V: MailboxSoaView,
        F: Fn(usize) -> Option<SpoRanks>,
    {
        let mut triples = Vec::new();
        let mut contributions = Vec::new();
        let n = soa.n_rows();
        let r = self.radius as i32;
        let class_ids = soa.class_id();
        for d in -r..=r {
            let row_i = focal_row as i32 + d;
            if row_i < 0 || row_i as usize >= n {
                continue;
            }
            let row = row_i as usize;
            let Some(t) = row_triple(row) else { continue };
            triples.push(t);
            contributions.push(RowContribution {
                row,
                class_id: class_ids[row],
                proximity: d.unsigned_abs(),
            });
        }
        WaveProjection {
            triples,
            provenance: BundleProvenance {
                mailbox_id: soa.mailbox_id(),
                contributions,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::KanbanColumn;

    struct FakeSoa {
        entity_type: Vec<u16>,
    }
    impl MailboxSoaView for FakeSoa {
        fn mailbox_id(&self) -> MailboxId {
            42
        }
        fn n_rows(&self) -> usize {
            self.entity_type.len()
        }
        fn w_slot(&self) -> u8 {
            0
        }
        fn current_cycle(&self) -> u32 {
            0
        }
        fn phase(&self) -> KanbanColumn {
            KanbanColumn::Planning
        }
        fn energy(&self) -> &[f32] {
            &[]
        }
        fn edges_raw(&self) -> &[u64] {
            &[]
        }
        fn meta_raw(&self) -> &[u32] {
            &[]
        }
        fn entity_type(&self) -> &[u16] {
            &self.entity_type
        }
    }

    fn row_triple(row: usize) -> Option<SpoRanks> {
        Some(SpoRanks {
            s: row as u16,
            p: (row + 1) as u16,
            o: (row + 2) as u16,
        })
    }
    fn soa(n: usize) -> FakeSoa {
        FakeSoa {
            entity_type: (0..n as u16).collect(),
        }
    }

    #[test]
    fn projection_is_deterministic() {
        let s = soa(20);
        let p = SoaWavePrimer::new(3);
        let a = p.project(&s, 10, row_triple);
        let b = p.project(&s, 10, row_triple);
        assert_eq!(a, b);
        assert_eq!(a.provenance.row_count(), 7);
    }

    #[test]
    fn window_clamps_and_records_proximity() {
        let s = soa(20);
        let proj = SoaWavePrimer::new(5).project(&s, 1, row_triple);
        assert_eq!(proj.provenance.row_count(), 7); // rows 0..=6
        assert_eq!(
            proj.provenance
                .contributions
                .iter()
                .find(|c| c.row == 1)
                .unwrap()
                .proximity,
            0
        );
        assert_eq!(
            proj.provenance
                .contributions
                .iter()
                .find(|c| c.row == 6)
                .unwrap()
                .proximity,
            5
        );
        assert_eq!(proj.provenance.mailbox_id, 42);
    }

    #[test]
    fn match_uses_injected_distance_no_vocabulary_named() {
        // identity distance: equal ranks → near (0), else far (max).
        let dist = |x: u16, y: u16| -> u8 {
            if x == y {
                0
            } else {
                u8::MAX
            }
        };
        let s = soa(20);
        let p = SoaWavePrimer::new(2);
        let here = p.project(&s, 10, row_triple);
        let same = p.project(&s, 10, row_triple);
        let far = p.project(&s, 2, row_triple);
        let self_m = here.best_guess_match(&same, dist);
        let far_m = here.best_guess_match(&far, dist);
        assert!(
            self_m > far_m,
            "identical window must out-resemble a distant one"
        );
        assert!((self_m - 1.0).abs() < 1e-6, "exact-twin match = 1.0");
    }

    #[test]
    fn empty_matches_zero() {
        let dist = |_: u16, _: u16| 0u8;
        let empty = WaveProjection::default();
        let s = soa(5);
        let ne = SoaWavePrimer::new(2).project(&s, 2, row_triple);
        assert_eq!(empty.best_guess_match(&ne, dist), 0.0);
        assert_eq!(ne.best_guess_match(&empty, dist), 0.0);
    }
}
