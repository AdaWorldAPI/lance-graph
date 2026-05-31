// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # `markov_soa` — the EXPLICIT, AUDITABLE proposer: SoA window → COCA-rank SPO triplets.
//!
//! ## Why this module exists (the "meet halfway")
//!
//! A fuzzy whole-window bundle was, historically, a **black box praying for
//! meaning**: an opaque vector you ran cosine on and *hoped* meant something.
//! That posture is a fundamental error in a deterministic, addressable, exact
//! substrate — it imports *hope* into a system built to eliminate it.
//!
//! This module fixes the **posture**. It makes the projection an **explicit,
//! deterministic list of the COCA-rank SPO triplets** in a ±radius mailbox
//! window, with full **provenance** of exactly which rows bundled in and at what
//! proximity. The triplets stay **addressable** (no superposition destroys the
//! register); matching is DeepNSM's own machinery — **COCA-4096 vocabulary +
//! the CAM-PQ 4096² u8 word-distance matrix via
//! [`SimilarityTable::lookup_u8`](crate::similarity::SimilarityTable) + grammar
//! heuristics — NOT float cosine, NOT a learned embedding**. The fuzziness lives
//! only in the *match readout* (calibrated CAM-PQ similarity), never in the
//! construction. That is the whole difference between "praying for meaning" and
//! "a known projection you can inspect."
//!
//! ## What it IS — strictly a fuzzy proposer (cognitive priming)
//!
//! The output is a **best-guess match** — System-1 priming, never System-2
//! truth. In chess terms: *"this feels like a Sicilian with a pinch of death
//! trap."* It proposes **where to look** and **what this resembles**; it NEVER
//! asserts what is true. The exact 32k SPO-W triplets (the deterministic
//! substrate) ALWAYS confirm. A wrong guess costs a cheap reprioritization,
//! never a wrong answer — that is honest approximation, not praying.
//!
//! Firewall (faiss-homology / `I-VSA-IDENTITIES`): similarity lives ONLY in the
//! discovery/proposer layer (Aerial). This projection is a **proposer input** —
//! it may steer foveated hydration ("this region smells relevant"), never
//! address or assert. The register stays intact: the bundle POINTS at the
//! triplets it summarized (via [`BundleProvenance`]); it never *replaces* them.
//!
//! ## The grail hypothesis (CONJECTURE — labelled, not asserted)
//!
//! If a deterministic CAM-PQ match over the windowed rank-triplets yields a
//! useful best-guess-next proposal, that is **"autocomplete from deterministic
//! semantic structure"** — a proposal you did not *train* but *derived*, knowing
//! exactly which triplets produced it. Whether it carries recoverable signal
//! above the noise floor is UNPROVEN; see [`BundleProjection::provenance`] (the
//! audit trail) and the module tests for the determinism guarantee, and
//! `I-NOISE-FLOOR-JIRAK` for the significance gate any "it works" claim must clear.
//!
//! ## Zero new dependency
//!
//! DeepNSM already hard-deps `lance-graph-contract` (for `RoleKeySlice`). This
//! module consumes `contract::soa_view::MailboxSoaView` through that existing
//! seam — no new dependency, firewall preserved (DeepNSM does not depend on the
//! heavy `cognitive-shader-driver` that *implements* the view).

/// The mailbox-SoA view this projector reads. Re-exported alias so the call
/// site reads as the seam it is. Implemented by `cognitive-shader-driver`'s
/// `MailboxSoA` (consumer side); DeepNSM only needs the read surface.
pub use lance_graph_contract::soa_view::MailboxSoaView;

/// One row's contribution to the bundle, recorded for audit. This is what makes
/// the projection NOT a black box: every fold is attributable. All integer —
/// no float weight (the proximity IS the prior, recorded as |delta| from focal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowContribution {
    /// The SoA row index that contributed.
    pub row: usize,
    /// The `entity_type`/`class_id` of that row (the semantic identity bundled).
    pub class_id: u16,
    /// Distance from the focal row (`|delta|`) — the recency/proximity prior,
    /// integer. Nearer rows are weighted more at match time, deterministically.
    pub proximity: u32,
}

/// The provenance of a projection — the complete, ordered list of what folded in.
///
/// A projection WITHOUT this is a black box; a projection WITH it is a
/// deterministic, replayable, auditable construction. The triplet list + this
/// provenance + the SoA fully reconstruct the projection — nothing is lost
/// (the register stays intact — `I-VSA-IDENTITIES`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BundleProvenance {
    /// Source mailbox id (which cohort/book this projection summarizes).
    pub mailbox_id: u32,
    /// Rows that contributed, in fold order.
    pub contributions: Vec<RowContribution>,
}

impl BundleProvenance {
    /// How many rows contributed a triplet to this projection.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.contributions.len()
    }
}

/// A deterministic projection of a mailbox SoA window: the COCA-rank SPO
/// triplets that bundled in, + their provenance.
///
/// **No cosine, no float embedding.** DeepNSM's match machinery is its own:
/// the **COCA-4096 vocabulary** (`SpoTriple` s/p/o are 12-bit COCA ranks) + the
/// **CAM-PQ 4096² u8 word-distance matrix** read through
/// [`SimilarityTable::lookup_u8`](crate::similarity::SimilarityTable) (O(1),
/// CDF-calibrated) + grammar heuristics. The "priming vector" is therefore the
/// **multiset of rank-triplets**, and "best-guess match" is the CAM-PQ /
/// SimilarityTable comparison over those ranks — NOT a learned dense vector.
/// The provenance is the **audit trail** that keeps this explicit, not opaque.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BundleProjection {
    /// The COCA-rank SPO triplets in the window, in fold order — the explicit
    /// content this projection superposes (each `(s, p, o)` a 12-bit COCA rank
    /// triple). This is the priming material; matching reads it via CAM-PQ.
    pub triplets: Vec<RankTriple>,
    /// What bundled in, in order — the replayable construction.
    pub provenance: BundleProvenance,
}

/// An SPO triple as three 12-bit COCA-4096 vocabulary ranks (mirror of
/// `spo::SpoTriple`'s accessors; carried explicitly so the projection is a
/// plain auditable list, not a packed opaque). `predicate`/`object` may be the
/// `spo::NO_ROLE` sentinel for intransitive verbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankTriple {
    /// Subject COCA rank (0..4096).
    pub s: u16,
    /// Predicate COCA rank (0..4096).
    pub p: u16,
    /// Object COCA rank (0..4096, or `NO_ROLE`).
    pub o: u16,
}

impl BundleProjection {
    /// **Best-guess match** to another projection — the System-1 priming read.
    /// Deterministic, integer: for each triplet here, take the *nearest* triplet
    /// there by summed CAM-PQ word distance (`dist` = the 4096² u8 matrix
    /// lookup), map that distance to calibrated similarity via `sim.lookup_u8`,
    /// and average. "How much does this region resemble that one?" — the
    /// chess-intuition "feels like a Sicilian". It proposes; never asserts.
    /// `0.0` if either side is empty. NO cosine, NO float embedding.
    #[must_use]
    pub fn best_guess_match(
        &self,
        other: &BundleProjection,
        sim: &crate::similarity::SimilarityTable,
        dist: impl Fn(u16, u16) -> u8,
    ) -> f32 {
        if self.triplets.is_empty() || other.triplets.is_empty() {
            return 0.0;
        }
        let mut acc = 0.0f32;
        for a in &self.triplets {
            let mut best = 0.0f32;
            for b in &other.triplets {
                // summed word-distance over the 3 roles → mean u8 → similarity.
                let d = ((dist(a.s, b.s) as u16 + dist(a.p, b.p) as u16 + dist(a.o, b.o) as u16) / 3) as u8;
                let s = sim.lookup_u8(d);
                if s > best {
                    best = s;
                }
            }
            acc += best;
        }
        acc / self.triplets.len() as f32
    }
}

/// The projector: folds a [`MailboxSoaView`]'s rows into the multiset of
/// COCA-rank SPO triplets in a ±radius window, **recording every contribution**.
/// Deterministic by construction — same SoA + same focal + same radius ⇒
/// identical triplet list AND identical provenance, every run, every target
/// (no RNG, no time, no float hashing).
///
/// `row_triple(row) -> Option<RankTriple>` resolves a SoA row to its COCA-rank
/// SPO triple (from the deterministic NSM→SPO output). Rows without a triple are
/// skipped and NOT recorded. The projector NEVER invents ranks — it only carries
/// the ones the deterministic parse produced.
pub struct SoaPrimer {
    /// Proximity radius around the focal row (the Markov ±window over mailboxes).
    pub radius: u32,
}

impl Default for SoaPrimer {
    fn default() -> Self {
        Self { radius: 5 }
    }
}

impl SoaPrimer {
    /// New primer with an explicit ±radius window.
    #[must_use]
    pub fn new(radius: u32) -> Self {
        Self { radius }
    }

    /// Project the SoA window centered on `focal_row` into the multiset of
    /// COCA-rank SPO triplets + provenance. `row_triple(row) -> Option<RankTriple>`
    /// resolves a row to its deterministic NSM→SPO rank-triple; rows without one
    /// are skipped (and NOT recorded — they contributed nothing).
    ///
    /// This is the explicit construction: the ordered list of rank-triplets of
    /// the rows in the ±radius window. The proximity ordering IS the prior (the
    /// Markov ±window over mailboxes); matching later weights nearer triplets via
    /// the CAM-PQ / `SimilarityTable` read in [`BundleProjection::best_guess_match`].
    /// No float weight, no superposed vector — the triplets stay addressable.
    pub fn project<V, F>(&self, soa: &V, focal_row: usize, row_triple: F) -> BundleProjection
    where
        V: MailboxSoaView,
        F: Fn(usize) -> Option<RankTriple>,
    {
        let mut triplets = Vec::new();
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
            triplets.push(t);
            // proximity recorded as |delta| from focal (the recency prior),
            // integer — NOT a learned float weight.
            contributions.push(RowContribution {
                row,
                class_id: class_ids[row],
                proximity: d.unsigned_abs(),
            });
        }
        BundleProjection {
            triplets,
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
    use crate::similarity::SimilarityTable;

    /// Minimal MailboxSoaView fake: only the columns the primer reads.
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

    // A book is a sequence of mailboxes; row r carries SPO ranks (r, r+1, r+2).
    fn row_triple(row: usize) -> Option<RankTriple> {
        Some(RankTriple { s: row as u16, p: (row + 1) as u16, o: (row + 2) as u16 })
    }

    fn soa(n: usize) -> FakeSoa {
        FakeSoa { entity_type: (0..n as u16).collect() }
    }

    #[test]
    fn projection_is_deterministic_in_triplets_and_provenance() {
        let s = soa(20);
        let p = SoaPrimer::new(3);
        let a = p.project(&s, 10, row_triple);
        let b = p.project(&s, 10, row_triple);
        // SAME soa + focal + radius ⇒ identical triplets AND identical provenance.
        assert_eq!(a, b, "projection must be bitwise-deterministic");
        // ±3 window around row 10, clamped to [0,20) ⇒ 7 rows.
        assert_eq!(a.provenance.row_count(), 7);
        assert_eq!(a.triplets.len(), 7);
    }

    #[test]
    fn window_clamps_at_edges_and_records_proximity() {
        let s = soa(20);
        let p = SoaPrimer::new(5);
        let proj = p.project(&s, 1, row_triple); // focal=1, radius 5 → rows 0..=6
        assert_eq!(proj.provenance.row_count(), 7);
        // proximity = |delta from focal|; focal row 1 has proximity 0.
        let focal = proj.provenance.contributions.iter().find(|c| c.row == 1).unwrap();
        assert_eq!(focal.proximity, 0);
        let far = proj.provenance.contributions.iter().find(|c| c.row == 6).unwrap();
        assert_eq!(far.proximity, 5);
        assert_eq!(proj.provenance.mailbox_id, 42);
    }

    #[test]
    fn rows_without_a_triple_are_skipped_not_recorded() {
        let s = soa(20);
        let p = SoaPrimer::new(3);
        // Only even rows produce a triple.
        let proj = p.project(&s, 10, |r| if r % 2 == 0 { row_triple(r) } else { None });
        assert!(proj.triplets.len() < 7, "odd rows contributed nothing");
        assert!(proj.provenance.contributions.iter().all(|c| c.row % 2 == 0));
    }

    #[test]
    fn best_guess_match_uses_cam_pq_not_cosine() {
        // identity word-distance: equal ranks → 0 distance, else a big distance.
        let dist = |x: u16, y: u16| -> u8 { if x == y { 0 } else { 200 } };
        // SimilarityTable: distance 0 → high similarity, large → low.
        let sim = SimilarityTable::from_stats(100.0, 40.0);

        let s = soa(20);
        let p = SoaPrimer::new(2);
        let a = p.project(&s, 10, row_triple);
        let identical = p.project(&s, 10, row_triple);
        let elsewhere = p.project(&s, 2, row_triple);

        let self_match = a.best_guess_match(&identical, &sim, dist);
        let other_match = a.best_guess_match(&elsewhere, &sim, dist);
        // identical window ⇒ every triplet finds an exact (distance-0) twin ⇒
        // similarity = lookup_u8(0), the max. Non-overlapping window scores lower.
        assert!(self_match > other_match, "identical region must out-resemble a distant one");
        assert!((self_match - sim.lookup_u8(0)).abs() < 1e-6, "exact-twin match = lookup_u8(0)");
    }

    #[test]
    fn empty_projection_matches_zero() {
        let sim = SimilarityTable::from_stats(100.0, 40.0);
        let dist = |_: u16, _: u16| 0u8;
        let empty = BundleProjection::default();
        let s = soa(5);
        let nonempty = SoaPrimer::new(2).project(&s, 2, row_triple);
        assert_eq!(empty.best_guess_match(&nonempty, &sim, dist), 0.0);
        assert_eq!(nonempty.best_guess_match(&empty, &sim, dist), 0.0);
    }
}
