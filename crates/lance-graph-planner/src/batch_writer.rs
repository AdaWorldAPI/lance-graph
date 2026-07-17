//! W1b ahead-firing batch writer — intent recording, nothing else.
//!
//! `cast()` records intent moves AHEAD of any storage write completing.
//! Payload-generic: the writer never inspects `P` (DTO purity — ownership
//! rides the cast pairing, never the DTO).
//!
//! **There is no confirmation bookkeeping here — by ruling (operator,
//! 2026-07-17, E-ACK-ELIMINATED-1).** Durability evidence is the written
//! row's own `LanceVersion` in Lance, read through `crate::temporal`
//! (`QueryReference::at` + deinterlace). Crash-replay is a temporal READ —
//! compare recorded intents against what Lance holds — never a stored
//! ledger in this struct. Do not add a confirmation method under any name.
//!
//! **Zero-copy sink (operator ruling, plan Addendum-6):** `P` is a DESCRIPTOR
//! — (mailbox, dirty row-range, cycle) — never owned delta bytes. Deltas stay
//! in the SoA backing store; the sink reads them through
//! `NodeRowPacket::as_le_bytes` at flush time. The sink drains EAGERLY
//! (ASAP on cast, background), and the write masks the thinking and vice
//! versa: the thinker reports (casts) and moves on — "melden macht frei" —
//! it is NEVER refused. Stacked casts on the same row are stacked intent
//! records; the sink reads the LIVE store at flush, so one physical flush
//! coalesces all earlier intents for a row (last-state-wins; the move log
//! keeps the full ordered history).
//!
//! The kanban advance is the in-stream synchronous kanbanstep
//! (`VersionScheduler::on_version → try_advance_phase(&mut)`), fired inline
//! by whoever already holds the version — never from this module.
//!
//! Uses the REAL shipped kanban contract types
//! ([`lance_graph_contract::kanban::KanbanMove`],
//! [`lance_graph_contract::collapse_gate::MailboxId`]) — this module does not
//! mint a parallel `KanbanMove`; see the D-MBX-A6 Outcome adapter context in
//! `crate::strategy::style_strategy`.

use std::collections::{BTreeMap, HashMap};

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::KanbanMove;

/// Identity of one `cast()` — a write-ahead intent record.
///
/// `next_id` is monotonically increasing, so `CastId`'s derived `Ord` gives
/// insertion order — this is what lets `board` use `BTreeMap` and iterate
/// in cast order for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CastId(pub u64);

/// Ahead-firing batch writer: intent (`cast`) is visible immediately;
/// `resolve_owner` is the W1c delegation cache (resolve-once, cache-hit
/// thereafter). Confirmation state does not exist here (module doc).
pub struct BatchWriter<P> {
    /// Monotonic id generator for the next cast.
    next_id: u64,
    /// Intent moves recorded per cast, keyed by `CastId`, alongside the
    /// mailbox the cast was recorded on behalf of. `BTreeMap` (not
    /// `HashMap`) so iteration is in ascending `CastId` order — i.e. cast
    /// (insertion) order.
    board: BTreeMap<CastId, (MailboxId, Vec<KanbanMove>)>,
    /// W1c delegation cache: `on_behalf` mailbox -> resolved owner mailbox.
    delegation_cache: HashMap<MailboxId, MailboxId>,
    /// Payloads recorded per cast (payload-generic; the writer never inspects `P`).
    pending_payloads: Vec<(CastId, P)>,
}

impl<P> Default for BatchWriter<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> BatchWriter<P> {
    /// Construct an empty batch writer.
    pub fn new() -> Self {
        Self {
            next_id: 0,
            board: BTreeMap::new(),
            delegation_cache: HashMap::new(),
            pending_payloads: Vec::new(),
        }
    }

    /// AHEAD: records the intent (moves visible immediately). Returns the
    /// cast id.
    ///
    /// "Melden macht frei" (plan Addendum-7): casting is REPORTING, never
    /// refused — stacked casts are stacked intent records, each with its
    /// own `CastId`.
    pub fn cast(&mut self, on_behalf: MailboxId, moves: Vec<KanbanMove>, payload: P) -> CastId {
        let cast = CastId(self.next_id);
        self.next_id += 1;
        self.board.insert(cast, (on_behalf, moves));
        self.pending_payloads.push((cast, payload));
        cast
    }

    /// Recorded casts, in ascending `CastId` (cast) order.
    #[must_use]
    pub fn casts(&self) -> Vec<CastId> {
        self.board.keys().copied().collect()
    }

    /// Intent moves recorded for a cast.
    #[must_use]
    pub fn intent_moves(&self, cast: CastId) -> Option<&[KanbanMove]> {
        self.board.get(&cast).map(|(_, moves)| moves.as_slice())
    }

    /// The mailbox a cast was recorded ON BEHALF OF — the delegation-audit
    /// surface (W4a cast pairing: consumers pair the owner at cast; wardens
    /// verify the pairing here, never on the DTO).
    #[must_use]
    pub fn on_behalf_of(&self, cast: CastId) -> Option<MailboxId> {
        self.board.get(&cast).map(|(on_behalf, _)| *on_behalf)
    }

    /// W1c delegation cache: resolve an owner once, cache; returns `(owner, was_cache_hit)`.
    pub fn resolve_owner(
        &mut self,
        on_behalf: MailboxId,
        resolver: impl FnOnce(MailboxId) -> MailboxId,
    ) -> (MailboxId, bool) {
        if let Some(&owner) = self.delegation_cache.get(&on_behalf) {
            return (owner, true);
        }
        let owner = resolver(on_behalf);
        self.delegation_cache.insert(on_behalf, owner);
        (owner, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn casting_is_fire_and_forget_and_stacks_freely() {
        let mut w: BatchWriter<u8> = BatchWriter::new();

        // The thinker casts and is freed at once — nothing to wait on,
        // and further casts stack freely as independent intent records.
        let c1 = w.cast(7, vec![], 0);
        let c2 = w.cast(7, vec![], 1);
        let c3 = w.cast(7, vec![], 2);
        assert_eq!(w.casts(), vec![c1, c2, c3], "cast (insertion) order");
        assert_eq!(w.on_behalf_of(c2), Some(7));
    }

    #[test]
    fn intent_moves_stay_readable_for_replay_comparison() {
        // Crash-replay is a temporal READ elsewhere; this struct's only
        // job is that recorded intents remain readable to compare against
        // what Lance holds.
        let mut w: BatchWriter<u8> = BatchWriter::new();
        let cast = w.cast(7, vec![], 0);
        assert!(w.intent_moves(cast).is_some(), "intent remains readable");
        assert_eq!(w.intent_moves(CastId(9999)), None, "stray id: nothing");
    }

    #[test]
    fn delegation_cache_resolves_once_then_hits() {
        let mut w: BatchWriter<u8> = BatchWriter::new();
        let (owner, hit) = w.resolve_owner(3, |_| 11);
        assert_eq!((owner, hit), (11, false), "first resolve is a miss");
        let (owner, hit) = w.resolve_owner(3, |_| unreachable!("cached"));
        assert_eq!((owner, hit), (11, true), "second resolve is a cache hit");
    }
}
