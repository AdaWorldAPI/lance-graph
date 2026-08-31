//! **Rung-aware dispatch** — the ruled work item, built as the ruled
//! COMPOSITION and nothing else.
//!
//! The reasoning-fabric crossmap pinned both the gap and its shape:
//! *"rung-aware dispatch MISSING (`PhaseCensus` deliberately rung-blind —
//! composing it with per-rung `QueryReference::at(v, rung)` readers IS the
//! ruled work item)"* — and #1077's finding 2 settled the permission
//! question: *"rung already differs by temporal horizon, not by the right to
//! think — what is missing is demonstration, not permission."*
//!
//! So this module adds NO rung type to any supervisor and NO new coordinate:
//! it hands each tunnel lane the [`QueryReference`] its own rung already
//! defines, and gates the lane's claims through
//! [`EpistemicMode::admits`](crate::temporal::EpistemicMode::admits). The
//! sentence that executes here: *"Low rungs reason strictly in the present;
//! mid rungs admit hindsight; top rungs may spoiler-read."* A refusal is
//! REPORTED with the status that caused it, never silent.

use lance_graph_contract::alpha::{AlphaAddr, AlphaClaim, AlphaError, AlphaOverlay};
use lance_graph_contract::alpha_tunnel::AlphaTunnel;
use lance_graph_contract::rung_schedule::{Wave, LEVELS};

use crate::temporal::{classify, LanceVersion, QueryReference, TemporalStatus};

/// One reader per tunnel lane, all pinned at ONE `KnowledgeHorizon`.
///
/// The rung is the only thing that differs between them — which is the whole
/// point: same world-version, ten epistemic disciplines over it.
#[must_use]
pub fn horizon_readers(ref_version: LanceVersion) -> [QueryReference; LEVELS] {
    core::array::from_fn(|r| QueryReference::at(ref_version, r as u8))
}

/// What became of one horizon-gated claim.
#[derive(Debug)]
pub enum HorizonVerdict {
    /// The lane's mode admits the row's temporal status; the claim stands.
    Claimed(AlphaClaim),
    /// The row is outside this rung's horizon — the WHY travels with the no.
    Refused(TemporalStatus),
    /// The substrate itself said no (e.g. an unallocated address).
    Substrate(AlphaError),
}

impl HorizonVerdict {
    /// Did the claim land?
    #[must_use]
    pub fn claimed(&self) -> bool {
        matches!(self, HorizonVerdict::Claimed(_))
    }
}

/// The horizon-gated claim: lane *N* may claim a row only if *N*'s epistemic
/// mode admits the row's [`TemporalStatus`] at this reader's horizon.
///
/// The order is deliberate: classify FIRST, claim only on admission — a lane
/// never touches (not even as a `visits` bump) a row its discipline may not
/// see, so a Strict lane's scanpath carries no trace of the future.
pub fn claim_admitted(
    lane: &mut AlphaOverlay<'_>,
    reader: &QueryReference,
    addr: AlphaAddr,
    row_version: LanceVersion,
    knowable_from: LanceVersion,
) -> HorizonVerdict {
    let status = classify(row_version, knowable_from, reader);
    if !reader.mode.admits(status) {
        return HorizonVerdict::Refused(status);
    }
    match lane.claim(addr, reader.rung) {
        Ok(c) => HorizonVerdict::Claimed(c),
        Err(e) => HorizonVerdict::Substrate(e),
    }
}

/// Run one wave with each lane handed ITS OWN rung's reader — the
/// composition, as one call.
///
/// The closure gets `(recipe ids at this rung, the lane, the lane's
/// reader)`; every claim it makes should go through [`claim_admitted`] with
/// that reader. The census/tunnel underneath stays rung-blind — the rung
/// enters ONLY through the reader, which is exactly the ruled shape.
pub fn run_wave_with_horizon<'a, F>(
    tunnel: &mut AlphaTunnel<'a>,
    wave: &Wave,
    ref_version: LanceVersion,
    mut f: F,
) where
    F: FnMut(&[u8], &mut AlphaOverlay<'a>, &QueryReference),
{
    let readers = horizon_readers(ref_version);
    tunnel.run_wave(wave, |rung, ids, lane| {
        if let Some(reader) = readers.get(rung as usize) {
            f(ids, lane, reader);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::EpistemicMode;
    use lance_graph_contract::alpha::AlphaAllocation;
    use lance_graph_contract::canonical_node::{EdgeBlock, NodeGuid, NodeRow};
    use lance_graph_contract::recipe_kernels::{ThoughtField, ThoughtMask};
    use lance_graph_contract::rung_schedule::schedule;

    fn base() -> Vec<NodeRow> {
        (0u32..8)
            .map(|i| NodeRow {
                key: NodeGuid::new(0x0D0D_0000 + i, 7, 8, 9, 0x44, i + 1),
                edges: EdgeBlock::default(),
                value: [0u8; 480],
            })
            .collect()
    }

    /// **The horizon sentence, executing over the tunnel.** One FUTURE row
    /// (`row_version > ref`): the Strict lane (rung 2) refuses it as
    /// `Anachronistic`, the Aware lane (rung 6) claims it, the Retro lane
    /// (rung 9) claims it as a `Spoiler` read. A CONTEMPORARY row is claimed
    /// even at rung 2, and an UNKNOWABLE row is refused even at rung 9 —
    /// both silence twins, so "the gate fires" and "the gate discriminates"
    /// are separately pinned. Disable-verified: removing the `admits` gate
    /// turns the Strict refusal into a claim and this test red.
    #[test]
    fn the_horizon_sentence_executes_over_the_tunnel() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut tunnel = AlphaTunnel::over(&alloc, 1);
        let readers = horizon_readers(100);
        assert_eq!(readers[2].mode, EpistemicMode::Strict);
        assert_eq!(readers[6].mode, EpistemicMode::Aware);
        assert_eq!(readers[9].mode, EpistemicMode::Retro);

        let future = (b[0].key, 150u64, 0u64); // row_version 150 > ref 100
        let now = (b[1].key, 90u64, 0u64);
        let unknowable = (b[2].key, 90u64, 200u64); // class not yet knowable

        for (rung, want_future, want_status) in [
            (2u8, false, TemporalStatus::Anachronistic),
            (6, true, TemporalStatus::Contemporary), // unused status for a claim
            (9, true, TemporalStatus::Contemporary),
        ] {
            let reader = readers[rung as usize];
            let lane = tunnel.lane_mut(rung).expect("lane exists");
            let v = claim_admitted(lane, &reader, future.0, future.1, future.2);
            assert_eq!(
                v.claimed(),
                want_future,
                "rung {rung} on a future row: {v:?}"
            );
            if let HorizonVerdict::Refused(st) = v {
                assert_eq!(st, want_status, "the WHY travels with the no");
            }
            assert!(
                claim_admitted(lane, &reader, now.0, now.1, now.2).claimed(),
                "a contemporary row is admitted at rung {rung}"
            );
            assert!(
                !claim_admitted(lane, &reader, unknowable.0, unknowable.1, unknowable.2).claimed(),
                "an unknowable row is refused even at rung {rung}"
            );
        }

        // The Strict lane's scanpath carries NO trace of the future row —
        // refusal happened before any visits bump.
        let strict = tunnel.lane(2).expect("lane");
        assert!(
            strict.get(future.0).is_none(),
            "no trace of the refused row"
        );
        assert!(strict.get(now.0).is_some());
    }

    /// The composition hands each lane ITS OWN rung's reader — never a
    /// shared one. Disable-verified: indexing `readers[9]` for every lane
    /// (one shared Retro reader) fails the mode assertions.
    #[test]
    fn each_lane_gets_its_own_rung_reader() {
        let b = base();
        let alloc = AlphaAllocation::over(&b);
        let mut tunnel = AlphaTunnel::over(&alloc, 2);
        let plan = schedule(ThoughtMask::of(&[
            ThoughtField::Sd,
            ThoughtField::FreeEnergy,
            ThoughtField::Dissonance,
            ThoughtField::Temperature,
            ThoughtField::Confidence,
            ThoughtField::Rung,
            ThoughtField::Candidates,
            ThoughtField::Beliefs,
        ]));
        assert!(!plan.waves.is_empty(), "a fully-grounded plan schedules");
        let mut seen = Vec::new();
        for wave in &plan.waves {
            let rungs = wave.rungs();
            run_wave_with_horizon(&mut tunnel, wave, 42, |ids, _lane, reader| {
                assert!(!ids.is_empty(), "a lane only runs with recipes to run");
                assert_eq!(reader.ref_version, 42, "one horizon for all lanes");
                assert_eq!(
                    reader.mode,
                    EpistemicMode::for_rung(reader.rung),
                    "the reader's discipline is its OWN rung's"
                );
                seen.push(reader.rung);
            });
            assert_eq!(seen.len(), rungs.len(), "every scheduled rung ran once");
            assert_eq!(seen, rungs, "…in lane order, each with its own reader");
            seen.clear();
        }
    }
}
