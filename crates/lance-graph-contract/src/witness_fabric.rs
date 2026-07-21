// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `witness_fabric` — make the A9 witness tenant REAL: compute the social loci
//! (quorum / contradiction) from a WINDOW of real peer rows instead of
//! hand-setting them, follow loci chains with a hop budget, and recognize a
//! persisted contradiction as an opinion. Thinking as agreement topology
//! (`E-THINKING-EXPANSION-QUEUE-1` Tier 3).
//!
//! These are FUNCTIONS over a slice of `(position, CausalWitnessFacet)` rows —
//! never a materialized `W×W` fabric struct (AGI-as-SoA: methods over the
//! existing carrier, not a new stored layer).
//!
//! # Loci converge on the SAME EVENT, not the same offset
//!
//! A locus offset is relative to its OWN row's stream position. Two rows agree
//! about a dimension iff they point at the **same absolute event**: row A at
//! `pos_a` with offset `o_a` and row B at `pos_b` with `o_b` converge iff
//! `pos_a + o_a == pos_b + o_b` (plan §2.9: "converge on the SAME context
//! events"). [`CausalWitnessFacet::agrees_at`](crate::causal_witness) compares
//! bare offsets (co-located rows); the fabric compares absolute targets.

use crate::causal_witness::{CausalWitnessFacet, Locus};

/// The named loci that carry CONTENT agreement — everything except the two
/// social loci (Quorum / Contradiction), which the fabric is COMPUTING and so
/// must not read as input (no self-reference).
const CONTENT_LOCI: [Locus; 14] = [
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
];

/// Count content loci on which two rows converge on the **same absolute event**.
#[must_use]
pub fn absolute_agreement(
    pos_a: usize,
    a: CausalWitnessFacet,
    pos_b: usize,
    b: CausalWitnessFacet,
) -> usize {
    CONTENT_LOCI
        .iter()
        .filter(|&&l| {
            a.is_bound(l)
                && b.is_bound(l)
                && (pos_a as isize + a.at(l) as isize) == (pos_b as isize + b.at(l) as isize)
        })
        .count()
}

/// The elected social peers for a focal row: the offsets to bind into its
/// Quorum / Contradiction loci, computed from the window fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PeerElection {
    /// Signed offset (`∈[−8,+7]`) to the agreeing peer, or `0` if none found.
    pub quorum_offset: i8,
    /// Signed offset to the preserved dissenting peer, or `0` if none.
    pub contradiction_offset: i8,
    /// Content-loci agreement count with the elected quorum peer.
    pub quorum_agreement: usize,
}

/// **E-QUORUM-FABRIC-1** — elect the focal row's quorum + contradiction peers
/// from the window. Quorum = the peer with the MOST content-loci convergence.
/// Contradiction = a peer that converges on ≥1 content locus but points its
/// **Kausal** (cause) edge at a DIFFERENT event — shares context, disagrees on
/// the cause (a preserved dissent, not mere unrelatedness). Offsets are the peer
/// position relative to the focal, clamped to the `±8` window (peers outside it
/// are not electable social loci).
///
/// `window` is `(stream_position, register)`; `focal_idx` indexes into it.
#[must_use]
pub fn elect_peers(focal_idx: usize, window: &[(usize, CausalWitnessFacet)]) -> PeerElection {
    let Some(&(focal_pos, focal)) = window.get(focal_idx) else {
        return PeerElection::default();
    };
    let mut best_q: Option<(usize, i8)> = None; // (agreement, offset)
    let mut best_c: Option<(usize, i8)> = None; // (agreement, offset)
    for (i, &(pos, peer)) in window.iter().enumerate() {
        if i == focal_idx {
            continue;
        }
        let delta = pos as isize - focal_pos as isize;
        if !(-8..=7).contains(&delta) {
            continue; // outside the ±8 window — not an electable social locus
        }
        let offset = delta as i8;
        let agree = absolute_agreement(focal_pos, focal, pos, peer);
        if agree == 0 {
            continue;
        }
        // Quorum candidate: maximize agreement.
        if best_q.is_none_or(|(a, _)| agree > a) {
            best_q = Some((agree, offset));
        }
        // Contradiction candidate: agrees on context BUT the Kausal cause points
        // elsewhere (both bound, different absolute target).
        let kausal_conflict = focal.is_bound(Locus::Kausal)
            && peer.is_bound(Locus::Kausal)
            && (focal_pos as isize + focal.cause() as isize)
                != (pos as isize + peer.cause() as isize);
        if kausal_conflict && best_c.is_none_or(|(a, _)| agree > a) {
            best_c = Some((agree, offset));
        }
    }
    PeerElection {
        quorum_offset: best_q.map(|(_, o)| o).unwrap_or(0),
        contradiction_offset: best_c.map(|(_, o)| o).unwrap_or(0),
        quorum_agreement: best_q.map(|(a, _)| a).unwrap_or(0),
    }
}

/// The result of following a locus chain across the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainResolution {
    /// Final signed offset from the focal to the resolved event, if resolved
    /// within budget and window; `None` if unresolved.
    pub final_offset: Option<i8>,
    /// Hops taken (0 = the focal's own locus resolved directly).
    pub hops: u8,
    /// The chain left the `±8` window or exceeded the hop budget — the signal a
    /// `temporal.rs` version-range read (`QueryReference::at`) is required. The
    /// contract emits the signal; the consumer does the read (no widening of the
    /// i4 nibble, no new witness variant).
    pub escalated: bool,
}

/// **E-LOCI-CHAIN-ESCALATE-1** — follow the focal row's `locus` chain: the
/// offset points at a peer; if that peer's SAME locus is also bound, hop again
/// (the register following its own nibbles, `E-L9-REAL-TEXT-1`), up to
/// `max_hops`. Escalate (rather than widen) when the chain leaves the `±8`
/// window or exhausts the budget.
#[must_use]
pub fn resolve_chain(
    focal_idx: usize,
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    max_hops: u8,
) -> ChainResolution {
    let Some(&(focal_pos, focal)) = window.get(focal_idx) else {
        return ChainResolution {
            final_offset: None,
            hops: 0,
            escalated: false,
        };
    };
    // Map stream position → window index for O(1) hops.
    let idx_of = |pos: isize| window.iter().position(|&(p, _)| p as isize == pos);

    let mut cur_idx = focal_idx;
    let mut cur = focal;
    let mut hops = 0u8;
    loop {
        let off = cur.at(locus);
        if off == 0 {
            // chain broke before reaching a terminal — escalate if we hopped at all
            return ChainResolution {
                final_offset: None,
                hops,
                escalated: hops > 0,
            };
        }
        let cur_pos = window[cur_idx].0 as isize;
        let target_pos = cur_pos + off as isize;
        let total = target_pos - focal_pos as isize;
        // Left the window the focal can address in one i4 → escalate.
        if !(-8..=7).contains(&total) {
            return ChainResolution {
                final_offset: None,
                hops,
                escalated: true,
            };
        }
        let Some(next_idx) = idx_of(target_pos) else {
            // target event not present in this window → resolved to the offset,
            // but its filler is out of window: escalate for the version read.
            return ChainResolution {
                final_offset: Some(total as i8),
                hops,
                escalated: true,
            };
        };
        let next = window[next_idx].1;
        // If the target does NOT re-bind this locus, the chain terminates here.
        if !next.is_bound(locus) {
            return ChainResolution {
                final_offset: Some(total as i8),
                hops,
                escalated: false,
            };
        }
        hops += 1;
        if hops >= max_hops {
            // budget exhausted mid-chain → escalate.
            return ChainResolution {
                final_offset: Some(total as i8),
                hops,
                escalated: true,
            };
        }
        cur_idx = next_idx;
        cur = next;
    }
}

/// The grounding verdict of the **multipass Markov standing wave** over a locus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveGrounding {
    /// Bound AND the standing wave SETTLES — increasing the hop budget stops
    /// changing the resolved target (per-rung persistence, D-CSW-1): the locus is
    /// **causally grounded**, not a coincidental co-occurrence.
    Causal,
    /// Bound but the wave never settles within the pass budget — the target keeps
    /// moving or escalates: a **coincidental** binding, not causal grounding.
    Coincidental,
    /// The locus is not bound at all (unbound at the source).
    Unbound,
}

/// **The multipass Markov standing wave** over one locus — the LITERAL Markov-chain
/// resolution (not a coarse scalar proxy). Runs [`resolve_chain`] at increasing hop
/// budgets `1..=passes` (the standing wave's passes) and asks whether the resolved
/// target **persists**: a causal locus settles (two successive budgets agree, chain
/// terminated), a coincidental one keeps extending or escalates. This is D-CSW-1's
/// "per-rung persistence separates causal from coincidental" applied to grounding —
/// the honest second resolution beside the single-pass structural binding
/// ([`is_bound`](crate::causal_witness::CausalWitnessFacet::is_bound)).
#[must_use]
pub fn standing_wave_grounded(
    focal_idx: usize,
    window: &[(usize, CausalWitnessFacet)],
    locus: Locus,
    passes: u8,
) -> WaveGrounding {
    let Some(&(_, focal)) = window.get(focal_idx) else {
        return WaveGrounding::Unbound;
    };
    if !focal.is_bound(locus) {
        return WaveGrounding::Unbound;
    }
    let mut last: Option<i8> = None;
    for budget in 1..=passes.max(1) {
        let r = resolve_chain(focal_idx, window, locus, budget);
        match r.final_offset {
            // settled: this budget resolved to the same target the previous did
            // (adding budget stopped moving it) and it did not escalate.
            Some(off) if !r.escalated => {
                if last == Some(off) {
                    return WaveGrounding::Causal; // the wave stood still → causal
                }
                last = Some(off);
            }
            // escalated (left the ±8 window) or unresolved → the wave has not
            // settled: coincidental / needs a temporal.rs read.
            _ => return WaveGrounding::Coincidental,
        }
    }
    // A single-hop terminal chain resolves identically at every budget, so `last`
    // is Some but the "two agree" short-circuit needs ≥2 passes to fire; treat a
    // resolved-non-escalated single pass as causal only when passes==1.
    if passes <= 1 && last.is_some() {
        WaveGrounding::Causal
    } else {
        WaveGrounding::Coincidental
    }
}

/// **E-CONTRADICTION-OPINION-1** — a stance/opinion is a row whose Contradiction
/// locus stays BOUND across successive revisions (committed-contradiction
/// persistence as first-class epistemic state). `revisions` is the same row's
/// register at successive versions (oldest→newest).
///
/// Returns how many revisions kept the contradiction bound; the row IS an
/// opinion iff that persistence covers the whole (non-empty) history.
#[must_use]
pub fn opinion_strength(revisions: &[CausalWitnessFacet]) -> usize {
    revisions
        .iter()
        .filter(|r| r.is_bound(Locus::Contradiction))
        .count()
}

/// A row is an OPINION iff its contradiction survived every revision (a preserved
/// dissent, not transient noise). Empty history is not an opinion.
#[must_use]
pub fn is_opinion(revisions: &[CausalWitnessFacet]) -> bool {
    !revisions.is_empty() && opinion_strength(revisions) == revisions.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(edges: &[(Locus, i8)]) -> CausalWitnessFacet {
        let mut f = CausalWitnessFacet::ZERO;
        for &(l, o) in edges {
            f = f.with(l, o);
        }
        f
    }

    #[test]
    fn absolute_agreement_is_same_event_not_same_offset() {
        // A at pos 5 points Kausal −3 → event 2. B at pos 7 points Kausal −5 → event 2.
        // Different OFFSETS but the SAME absolute event → they agree.
        let a = w(&[(Locus::Kausal, -3)]);
        let b = w(&[(Locus::Kausal, -5)]);
        assert_eq!(absolute_agreement(5, a, 7, b), 1);
        // Same offset but different positions → different events → no agreement.
        let c = w(&[(Locus::Kausal, -3)]);
        assert_eq!(absolute_agreement(5, a, 7, c), 0);
    }

    #[test]
    fn elect_peers_picks_the_max_agreement_quorum() {
        // focal at pos 5; two content loci to SMeaning@7, Kausal@2.
        let focal = w(&[(Locus::SMeaning, 2), (Locus::Kausal, -3)]);
        let strong = w(&[(Locus::SMeaning, 1), (Locus::Kausal, -4)]); // pos 6 → S@7, K@2: both agree
        let weak = w(&[(Locus::SMeaning, 4)]); // pos 4 → S@8: disagree
        let window = [(5usize, focal), (6, strong), (4, weak)];
        let e = elect_peers(0, &window);
        assert_eq!(e.quorum_offset, 1, "the agreeing peer is at +1");
        assert_eq!(e.quorum_agreement, 2, "both content loci converge");
    }

    #[test]
    fn elect_peers_finds_kausal_dissenter_as_contradiction() {
        // A peer that agrees on SMeaning but points Kausal elsewhere = contradiction.
        let focal = w(&[(Locus::SMeaning, 2), (Locus::Kausal, -3)]); // pos 5 → S@7, K@2
        let dissenter = w(&[(Locus::SMeaning, 1), (Locus::Kausal, 0)]); // pos 6 → S@7 agree; K unbound? no
        let dissenter = dissenter.with(Locus::Kausal, 2); // pos 6 → K@8 ≠ focal K@2
        let window = [(5usize, focal), (6, dissenter)];
        let e = elect_peers(0, &window);
        assert_eq!(
            e.contradiction_offset, 1,
            "the Kausal-dissenting peer at +1"
        );
    }

    #[test]
    fn peers_outside_the_window_are_not_electable() {
        let focal = w(&[(Locus::SMeaning, 2)]);
        let far = w(&[(Locus::SMeaning, 2)]); // agrees, but 20 away
        let window = [(5usize, focal), (25, far)];
        assert_eq!(elect_peers(0, &window).quorum_offset, 0);
    }

    #[test]
    fn resolve_chain_hops_then_escalates_on_budget() {
        // pos 3 → Kausal +2 → pos 5 (rebinds) → +2 → pos 7 (rebinds) → +2 → pos 9
        let a = w(&[(Locus::Kausal, 2)]);
        let b = w(&[(Locus::Kausal, 2)]);
        let c = w(&[(Locus::Kausal, 2)]);
        let d = w(&[]); // terminal, no Kausal
        let window = [(3usize, a), (5, b), (7, c), (9, d)];
        // budget 1 hop: after 1 hop, escalate.
        let r1 = resolve_chain(0, &window, Locus::Kausal, 1);
        assert!(r1.escalated && r1.hops == 1);
        // budget 5: chain resolves to pos 9 (total offset +6) within window.
        let r5 = resolve_chain(0, &window, Locus::Kausal, 5);
        assert_eq!(r5.final_offset, Some(6));
        assert!(!r5.escalated);
    }

    #[test]
    fn resolve_chain_escalates_when_leaving_the_window() {
        // pos 0 → Kausal +7 → pos 7 (rebinds) → +7 would be pos 14 = total +14 > 7 → escalate.
        let a = w(&[(Locus::Kausal, 7)]);
        let b = w(&[(Locus::Kausal, 7)]);
        let window = [(0usize, a), (7, b)];
        let r = resolve_chain(0, &window, Locus::Kausal, 8);
        assert!(
            r.escalated,
            "chain leaves the ±8 window → escalate to version-range read"
        );
    }

    #[test]
    fn opinion_is_a_persisted_contradiction() {
        let bound = w(&[(Locus::Contradiction, -4)]);
        let clear = w(&[(Locus::SMeaning, 1)]);
        assert!(is_opinion(&[bound, bound, bound]));
        assert!(
            !is_opinion(&[bound, clear, bound]),
            "a dropped contradiction is not a stance"
        );
        assert!(!is_opinion(&[]), "empty history is not an opinion");
        assert_eq!(opinion_strength(&[bound, clear, bound]), 2);
    }
}
