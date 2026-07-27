//! # `evidence` — the NARS disjointness guard as an EDGE READ (zero-dep).
//!
//! Replaces the withdrawn `source_registry` and the legacy `Stamp(u64)` source
//! bitset, per primer §17 (operator ruling, 2026-07-27: *"§17 dissolves it — the
//! substrate edges are the evidence record"*).
//!
//! ## Why the old shape was falsified
//!
//! `Stamp(1u64 << (id % 64))` modelled **source membership**. NARS revision needs
//! **event** identity: Wang's evidential base is a set of input *serial numbers*,
//! and the guard exists to stop ONE event being counted twice through two
//! derivation paths. Keying it on sources means **one sensor observing twice can
//! never raise confidence** — the two observations fold to the same bit — which
//! disables the most basic form of evidence accumulation. That is what falsified
//! the design, not the 64-bit ceiling the reviewers saw.
//!
//! ## What replaces it: nothing new is stored
//!
//! A belief's evidential base is **not a field** — it is the evidence edges the
//! row already carries ([`EpisodicEdges64`], 4 × `EdgeRef` slots), with temporal
//! birth on the version axis. Evidence-event identity is *the substrate address
//! of the evidencing row at its version*: nothing to mint, nothing to serialize.
//! Two observations by one sensor are two DISTINCT [`EdgeRef`]s, so repetition
//! accumulates — the exact defect above, gone by construction.
//!
//! Per §17's locality rule: **any provenance structure whose contents are
//! derivable by walking resident edges is a sidecar and is rejected on sight.**
//! This module therefore adds no storage — it is a pure read over a `u64` the
//! row already holds.
//!
//! ## Tri-state, because `bool` converts ignorance into permission
//!
//! "Not known to overlap" ≠ "known disjoint". [`EpisodicEdges64`] holds at most
//! [`EpisodicEdges64::CAPACITY`] edges and *demotes* on overflow
//! (`promote_into`), so a saturated edge word MAY have evicted the very edge that
//! would have matched. A saturated side therefore yields
//! [`EvidenceOverlap::Unknown`], never `Disjoint`. The bound is read off the
//! carrier, not chosen.

use crate::episodic_edges::{EdgeRef, EpisodicEdges64};

/// Result of the evidential-base disjointness guard.
///
/// `Unknown` is NOT a failure mode — it is the honest answer when the carrier
/// cannot support the stronger claim, and callers must treat it as "may not
/// revise" rather than "safe to revise".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvidenceOverlap {
    /// No shared evidence edge, and both sides are provably complete.
    /// Revision is admissible.
    Disjoint,
    /// At least one shared evidence edge — revision would double-count it.
    Overlap,
    /// At least one side is saturated, so an evicted edge could have matched.
    /// Conservative: refuse revision. Never widen this to `Disjoint`.
    Unknown,
}

impl EvidenceOverlap {
    /// Is revision admissible? ONLY on [`Disjoint`](EvidenceOverlap::Disjoint) —
    /// `Unknown` deliberately answers `false`, which is the whole reason the
    /// tri-state exists.
    #[inline]
    #[must_use]
    pub const fn admits_revision(self) -> bool {
        matches!(self, EvidenceOverlap::Disjoint)
    }
}

/// The guard: do two beliefs' evidence edges reach overlapping rows?
///
/// An `[a,b]`-shaped question over resident slots — exact, replay-stable, no
/// aliasing, no allocation. `O(CAPACITY²) = O(16)` comparisons, all register
/// work.
#[must_use]
pub fn evidence_overlap(a: EpisodicEdges64, b: EpisodicEdges64) -> EvidenceOverlap {
    for e in a.iter() {
        if b.contains(e) {
            return EvidenceOverlap::Overlap;
        }
    }
    // No match found — but a saturated side may have EVICTED a matching edge, so
    // the absence of a match is not proof of disjointness.
    if a.is_full() || b.is_full() {
        return EvidenceOverlap::Unknown;
    }
    EvidenceOverlap::Disjoint
}

/// The pooled evidential base of a revision — the union of both sides' edges.
///
/// Returns `None` when the union does not fit [`EpisodicEdges64::CAPACITY`]:
/// the caller must then either demote through a
/// [`DemotionSink`](crate::episodic_edges::DemotionSink) or decline the
/// revision. Silently truncating here would manufacture the same false
/// disjointness the old `% 64` fold did.
#[must_use]
pub fn pooled_base(a: EpisodicEdges64, b: EpisodicEdges64) -> Option<EpisodicEdges64> {
    let mut out = a;
    for e in b.iter() {
        if out.contains(e) {
            continue;
        }
        out = out.push(e)?;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges(locals: &[u16]) -> EpisodicEdges64 {
        let mut e = EpisodicEdges64::empty();
        for &l in locals {
            e = e
                .push(EdgeRef::intra(l).expect("valid local"))
                .expect("capacity");
        }
        e
    }

    /// THE FALSIFYING CASE that killed `Stamp`: one sensor observing TWICE.
    /// Under the old bitset both observations folded to `1 << (id % 64)`, so they
    /// looked like one event and revision could never accumulate. As distinct
    /// edges they are distinct evidence, so the guard admits the revision.
    #[test]
    fn one_sensor_observing_twice_is_two_events_and_admits_revision() {
        let first = edges(&[1]);
        let second = edges(&[2]);
        assert_eq!(evidence_overlap(first, second), EvidenceOverlap::Disjoint);
        assert!(evidence_overlap(first, second).admits_revision());
        // ...and the pooled base carries BOTH, so confidence can rise.
        assert_eq!(pooled_base(first, second).expect("fits").count(), 2);
    }

    /// The guard's actual job: the SAME event reached by two derivation paths
    /// must be caught, or it gets double-counted.
    #[test]
    fn same_event_through_two_paths_is_overlap() {
        assert_eq!(
            evidence_overlap(edges(&[7, 8]), edges(&[9, 7])),
            EvidenceOverlap::Overlap
        );
        assert!(!evidence_overlap(edges(&[7, 8]), edges(&[9, 7])).admits_revision());
    }

    /// CAN-IT-FIRE for `Unknown`: a saturated side must NOT report `Disjoint`,
    /// because an evicted edge could have matched. Without this the tri-state
    /// would be decoration.
    #[test]
    fn a_saturated_side_is_unknown_never_disjoint() {
        let full = edges(&[1, 2, 3, 4]);
        assert!(full.is_full(), "precondition: the carrier is at capacity");
        let other = edges(&[90, 91]);
        assert_eq!(evidence_overlap(full, other), EvidenceOverlap::Unknown);
        assert!(!evidence_overlap(full, other).admits_revision());
        // Symmetric — saturation on EITHER side blocks the strong claim.
        assert_eq!(evidence_overlap(other, full), EvidenceOverlap::Unknown);
    }

    /// CAN-IT-STAY-SILENT: saturation must not swallow a REAL overlap into
    /// `Unknown`. A guard that answered `Unknown` for everything full would carry
    /// as little information as one that never fired.
    #[test]
    fn saturation_does_not_mask_a_real_overlap() {
        let full = edges(&[1, 2, 3, 4]);
        assert_eq!(
            evidence_overlap(full, edges(&[3])),
            EvidenceOverlap::Overlap
        );
    }

    /// Two empty bases are disjoint and admit revision — the vacuous case, pinned
    /// so a future refactor cannot quietly turn it into `Unknown`.
    #[test]
    fn empty_bases_are_disjoint() {
        let e = EpisodicEdges64::empty();
        assert!(!e.is_full());
        assert_eq!(evidence_overlap(e, e), EvidenceOverlap::Disjoint);
    }

    /// Cross-family edges are distinct from intra-family edges with the same
    /// local index — the family nibble participates in identity.
    #[test]
    fn family_nibble_participates_in_edge_identity() {
        let intra = edges(&[5]);
        let mut cross = EpisodicEdges64::empty();
        cross = cross
            .push(EdgeRef::cross(3, 5).expect("valid cross"))
            .expect("capacity");
        assert_eq!(evidence_overlap(intra, cross), EvidenceOverlap::Disjoint);
    }

    /// The union must REFUSE rather than truncate — silent truncation is exactly
    /// the false-disjointness the `% 64` fold produced.
    #[test]
    fn pooled_base_refuses_rather_than_truncating() {
        assert_eq!(
            pooled_base(edges(&[1, 2, 3]), edges(&[4])).map(|e| e.count()),
            Some(4)
        );
        assert_eq!(pooled_base(edges(&[1, 2, 3, 4]), edges(&[5])), None);
        // A union that only re-adds shared edges still fits.
        assert_eq!(
            pooled_base(edges(&[1, 2, 3, 4]), edges(&[1, 2])).map(|e| e.count()),
            Some(4)
        );
    }
}
