//! `wave` — the Markov STANDING-WAVE RESOLUTION over the temporal stream.
//!
//! [`super::TemporalStream`] generalized v1's ±5 ring to a version-range WINDOW
//! (which triples a reader at `ref_version` can see). This module adds the piece
//! it left open: the multipass Markov standing-wave **resolution** over that
//! window — the shipped `witness_fabric::{standing_wave_grounded, resolve_chain}`,
//! producing a `Causal` / `Escalate` grounding per focal event. Operator ruling
//! `ISS-BUNDLE-RULING-SCOPE` path (b): the wave resolution as the complement to
//! the temporal window.
//!
//! ## Single-owner, no bundle (`E-NO-BUNDLE-STANDING-WAVE-1`)
//!
//! Each versioned event owns its `CausalWitnessFacet` loci; the wave **reads** a
//! version-range window and mutates nothing — there is no accumulator and no
//! shared register. The Markov property is STREAM ORDER
//! (`E-MARKOV-TEMPORAL-STREAM-1`), never a superposition into one carrier.
//!
//! ## How the version window meets the ±8 reference horizon
//!
//! Positions handed to the contract fabric are each event's ABSOLUTE stream
//! index, so the ±8 offset horizon (`E-HORIZON-NOT-BOUND-1`) resolves against
//! real positions. The version-range window decides which events are VISIBLE: a
//! chain whose target sits at a version the reader cannot see is simply absent
//! from the slice → the wave `Escalate`s (widen the version read), exactly the
//! horizon-is-a-reference-not-a-bound semantics — now bound to the version read
//! rather than a fixed ±5 ring.
//!
//! ## Paper grounding for the ±8 horizon (2026-07-22 four-paper review)
//!
//! The empirical anchor: **Manning & Carpenter 1997 (IWPT-97) Table 7** — the
//! maximum left-corner stack depth over the ENTIRE binarized WSJ Penn Treebank
//! is **8** (~99.4% of configurations ≤ 5), and their implementation carries
//! partial parses as **pointers into the move stream** with trees materialized
//! only at output — the grammar tree IS a pointer fabric over the stream. The
//! bounded-stack theorem itself (LC parsing: O(1) memory on left/right
//! branching, O(n) only on center-embedding, which natural text caps at ~3) is
//! **Abney & Johnson 1991 / Resnik 1992 / Stabler 1994** — NOT Roark & Johnson
//! 2000 or Moore 2000 (provenance corrected after review). Honest framing:
//! that theorem bounds the *open-constituent count* (which the 24-loci
//! register triples), NOT the *token span* of one attachment — a single slot
//! can span a long relative clause. The ±8 offset is therefore a **recency
//! prior rescued by escalation**: `Escalate` on OFFSET overflow (long-distance
//! attachment / invisible version) is the load-bearing channel, and Roark &
//! Johnson 2000 §3.3 predicts non-local context makes resolution *cheaper*,
//! not just more correct. Full synthesis:
//! `.claude/knowledge/left-corner-grammar-tree-pointer-fabric.md`.
//!
//! ## Scope boundary (declared, not faked)
//!
//! The loci (`CausalWitnessFacet` — antecedent / kausal / grounding offsets) are
//! the caller's input, the same shape the `jc` `l9_loci_real_text` harness
//! computes from text. Deriving loci FROM [`super::Spo`] triples is a separate
//! adapter and is intentionally NOT bundled here, so this stays a clean resolver.

use lance_graph_contract::causal_witness::{CausalWitnessFacet, Locus};
use lance_graph_contract::temporal_pov::{TemporalPov, VersionRange};
use lance_graph_contract::witness_fabric::{
    resolve_chain, standing_wave_grounded, ChainResolution, WaveGrounding,
};

/// A version-stamped, single-owner stream of loci events, resolved by the Markov
/// standing wave over a version-range window — the resolution complement to
/// [`super::TemporalStream`]'s window.
#[derive(Debug, Clone, Default)]
pub struct WitnessStream {
    /// `(version, owned register)` in append order; the append index IS the
    /// event's stream position (the ±8 offset space).
    events: Vec<(u64, CausalWitnessFacet)>,
}

impl WitnessStream {
    /// Empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one single-owner event at `version`; returns its stream position
    /// (the absolute append index the loci offsets are relative to). `version`
    /// is the caller's monotone tick (Lance-version / sentence-commit).
    pub fn push(&mut self, version: u64, register: CausalWitnessFacet) -> usize {
        let pos = self.events.len();
        self.events.push((version, register));
        pos
    }

    /// Number of events appended.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the stream is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The `(absolute stream position, register)` window a reader pinned at
    /// `ref_version` can see (`row_version ≤ ref_version`, [`TemporalPov::at`]).
    /// Positions stay the absolute append index so ±8 offsets resolve correctly.
    #[must_use]
    pub fn window_at(&self, ref_version: u64) -> Vec<(usize, CausalWitnessFacet)> {
        let pov = TemporalPov::at(ref_version, 0);
        self.events
            .iter()
            .enumerate()
            .filter(|(_, (v, _))| pov.admits(*v))
            .map(|(pos, (_, r))| (pos, *r))
            .collect()
    }

    /// The window over an explicit half-open [`VersionRange`] `[from, to)` — an
    /// arbitrary-width version span, not just the contemporary prefix.
    #[must_use]
    pub fn window_range(&self, range: VersionRange) -> Vec<(usize, CausalWitnessFacet)> {
        self.events
            .iter()
            .enumerate()
            .filter(|(_, (v, _))| range.contains(*v))
            .map(|(pos, (_, r))| (pos, *r))
            .collect()
    }

    /// Ground the event at absolute stream position `focal_pos` on `locus`, as
    /// seen by a reader at `ref_version`: the multipass standing wave over the
    /// contemporary window.
    ///
    /// - settles within the ±8 reference horizon → [`WaveGrounding::Causal`];
    /// - chain leaves the horizon OR its target is not visible at `ref_version`
    ///   → [`WaveGrounding::Escalate`] (widen the version read; a distant cause
    ///   is still a cause);
    /// - focal not visible at `ref_version`, or the locus unbound →
    ///   [`WaveGrounding::Unbound`].
    #[must_use]
    pub fn ground_at(
        &self,
        focal_pos: usize,
        locus: Locus,
        ref_version: u64,
        passes: u8,
    ) -> WaveGrounding {
        let window = self.window_at(ref_version);
        match window.iter().position(|(p, _)| *p == focal_pos) {
            Some(idx) => standing_wave_grounded(idx, &window, locus, passes),
            None => WaveGrounding::Unbound, // focal not visible in this version window
        }
    }

    /// Follow the focal event's `locus` chain (single hop budget) within the
    /// window a reader at `ref_version` sees. `escalated` is set when the chain
    /// leaves the visible window or the ±8 horizon — the signal to widen the
    /// version read. `None` if the focal is not visible at `ref_version`.
    #[must_use]
    pub fn resolve_at(
        &self,
        focal_pos: usize,
        locus: Locus,
        ref_version: u64,
        max_hops: u8,
    ) -> Option<ChainResolution> {
        let window = self.window_at(ref_version);
        window
            .iter()
            .position(|(p, _)| *p == focal_pos)
            .map(|idx| resolve_chain(idx, &window, locus, max_hops))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a register binding the given `(locus, offset)` edges.
    fn ev(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
        let mut f = CausalWitnessFacet::ZERO;
        for &(l, o) in edges {
            f = f.with(l, o);
        }
        f
    }

    #[test]
    fn window_at_admits_the_contemporary_prefix() {
        let mut s = WitnessStream::new();
        for v in 0..10u64 {
            s.push(v, ev(&[]));
        }
        // Reader at v=4 sees versions 0..=4 → 5 events, positions 0..=4.
        let w = s.window_at(4);
        assert_eq!(w.len(), 5);
        assert!(w.iter().all(|(pos, _)| *pos <= 4));
        // A future-version event is not admitted at an earlier ref.
        assert!(s.window_at(4).iter().all(|(pos, _)| *pos <= 4));
    }

    #[test]
    fn window_range_is_arbitrary_width() {
        let mut s = WitnessStream::new();
        for v in 0..10u64 {
            s.push(v, ev(&[]));
        }
        // [2,7) admits versions 2..=6 → 5 events, at their absolute positions.
        let w = s.window_range(VersionRange::new(2, 7));
        assert_eq!(w.len(), 5);
        assert_eq!(w.first().map(|(p, _)| *p), Some(2));
        assert_eq!(w.last().map(|(p, _)| *p), Some(6));
    }

    #[test]
    fn settles_local_is_causal() {
        // pos 1 → Antecedent −1 → pos 0 (a terminal, no Antecedent): the chain
        // terminates inside the ±8 horizon and inside the version window → Causal.
        let mut s = WitnessStream::new();
        s.push(0, ev(&[])); // pos 0, v0: terminal binder
        s.push(1, ev(&[(Locus::Antecedent, -1)])); // pos 1, v1: pronoun → pos 0
        assert_eq!(
            s.ground_at(1, Locus::Antecedent, 1, 4),
            WaveGrounding::Causal
        );
    }

    #[test]
    fn chain_leaves_reference_horizon_is_escalate() {
        // pos 0 → Kausal +7 → pos 7 (re-binds) → +7 = pos 14, total +14 > +7:
        // leaves the ±8 reference horizon → Escalate. All versions visible.
        let mut s = WitnessStream::new();
        s.push(0, ev(&[(Locus::Kausal, 7)])); // pos 0
        for v in 1..7u64 {
            s.push(v, ev(&[]));
        }
        s.push(7, ev(&[(Locus::Kausal, 7)])); // pos 7 re-binds
        assert_eq!(
            s.ground_at(0, Locus::Kausal, 99, 8),
            WaveGrounding::Escalate
        );
    }

    #[test]
    fn target_out_of_version_window_escalates_but_becomes_causal_when_visible() {
        // The SAME chain: pos 0 → Kausal +2 → pos 2 (terminal). pos 2 is stamped
        // at a HIGH version. A reader who cannot yet see version 5 must Escalate
        // (widen the read); a reader at version ≥ 5 sees pos 2 and settles Causal.
        let mut s = WitnessStream::new();
        s.push(0, ev(&[(Locus::Kausal, 2)])); // pos 0, v0
        s.push(1, ev(&[])); // pos 1, v1
        s.push(5, ev(&[])); // pos 2, v5: the target, only visible at ref ≥ 5
                            // Reader at v1: pos 2 (v5) not visible → target absent → Escalate.
        assert_eq!(s.ground_at(0, Locus::Kausal, 1, 4), WaveGrounding::Escalate);
        // Reader at v5: pos 2 visible, terminal in-horizon → Causal.
        assert_eq!(s.ground_at(0, Locus::Kausal, 5, 4), WaveGrounding::Causal);
    }

    #[test]
    fn focal_not_visible_at_ref_is_unbound() {
        let mut s = WitnessStream::new();
        s.push(0, ev(&[]));
        s.push(9, ev(&[(Locus::Antecedent, -1)])); // pos 1, v9
                                                   // Reader at v0 cannot see pos 1 → Unbound (not resolvable there).
        assert_eq!(
            s.ground_at(1, Locus::Antecedent, 0, 4),
            WaveGrounding::Unbound
        );
    }

    #[test]
    fn unbound_locus_is_unbound() {
        let mut s = WitnessStream::new();
        s.push(0, ev(&[(Locus::Antecedent, -1)]));
        // Focal visible + binds Antecedent, but NOT Kausal → Kausal is Unbound.
        assert_eq!(s.ground_at(0, Locus::Kausal, 0, 4), WaveGrounding::Unbound);
    }

    #[test]
    fn single_owner_events_are_independent() {
        // Each event owns its register; resolving one reads its own edge, never a
        // superposition of peers.
        let mut s = WitnessStream::new();
        s.push(0, ev(&[(Locus::Kausal, -2)])); // pos 0
        s.push(1, ev(&[(Locus::Kausal, -1)])); // pos 1 → cause at pos 0
        let r = s.resolve_at(1, Locus::Kausal, 9, 1).expect("focal visible");
        assert_eq!(r.final_offset, Some(-1));
        // The two registers are distinct owned values.
        let w = s.window_range(VersionRange::new(0, 99));
        assert_eq!(w[0].1.at(Locus::Kausal), -2);
        assert_eq!(w[1].1.at(Locus::Kausal), -1);
    }

    #[test]
    fn resolve_at_follows_multi_hop_chain() {
        // pos 0 → +2 → pos 2 (re-binds) → +2 → pos 4 (terminal): total +4.
        let mut s = WitnessStream::new();
        s.push(0, ev(&[(Locus::Kausal, 2)])); // 0
        s.push(1, ev(&[])); // 1
        s.push(2, ev(&[(Locus::Kausal, 2)])); // 2 re-binds
        s.push(3, ev(&[])); // 3
        s.push(4, ev(&[])); // 4 terminal
        let r = s
            .resolve_at(0, Locus::Kausal, 99, 5)
            .expect("focal visible");
        assert_eq!(r.final_offset, Some(4));
        assert!(!r.escalated, "chain terminates inside the horizon + window");
    }

    #[test]
    fn empty_stream_is_empty() {
        let s = WitnessStream::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(s.window_at(0).is_empty());
    }
}
