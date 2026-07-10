//! W1b ahead-firing batch writer — the kanban board IS the write-ahead log (M24).
//!
//! `cast()` records intent moves AHEAD of any storage ack; `ack()` confirms
//! at the `LanceVersion` the sink assigned (the CastId↔LanceVersion join
//! wiring the WAL into the temporal classifier; see `crate::temporal`);
//! `unacked()` is the crash-replay surface. Payload-generic: the writer never
//! inspects `P` (DTO purity — ownership rides the cast pairing, never the DTO).
//!
//! **Zero-copy sink (operator ruling, plan Addendum-6):** `P` is a DESCRIPTOR
//! — (mailbox, dirty row-range, cycle) — never owned delta bytes. Deltas stay
//! in the SoA backing store; the sink reads them through
//! `NodeRowPacket::as_le_bytes` at flush time. The sink drains EAGERLY
//! (ASAP on cast, background), and the write masks the thinking and vice
//! versa: the thinker reports (casts) and moves on — "melden macht frei" —
//! it is NEVER refused because earlier casts are unacked. Stacked casts on
//! the same row are stacked WAL entries; the sink reads the LIVE store at
//! flush, so one physical flush coalesces all earlier intents for a row
//! (last-state-wins; the move log keeps the full ordered history).
//!
//! Uses the REAL shipped kanban contract types
//! ([`lance_graph_contract::kanban::KanbanMove`],
//! [`lance_graph_contract::kanban::KanbanColumn`],
//! [`lance_graph_contract::collapse_gate::MailboxId`]) — this module does not
//! mint a parallel `KanbanMove`; see the D-MBX-A6 Outcome adapter context in
//! `crate::strategy::style_strategy`.

use std::collections::{BTreeMap, HashMap};

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::KanbanMove;

use crate::temporal::LanceVersion;

/// Identity of one `cast()` — a write-ahead intent record on the kanban board.
///
/// `next_id` is monotonically increasing, so `CastId`'s derived `Ord` gives
/// insertion order — this is what lets `board`/`acked` use `BTreeMap` and
/// have `unacked()` iterate in cast order for free.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CastId(pub u64);

/// Ahead-firing batch writer: intent (`cast`) is visible on the board before
/// any ack; `unacked()` is the M24 crash-replay surface; `resolve_owner` is
/// the W1c delegation cache (resolve-once, cache-hit thereafter).
pub struct BatchWriter<P> {
    /// Monotonic id generator for the next cast.
    next_id: u64,
    /// Board: intent moves recorded per cast, keyed by `CastId`, alongside the
    /// mailbox the cast was recorded on behalf of. Visible between `cast()`
    /// and `ack()` (and beyond, for crash-replay via `unacked()`).
    /// `BTreeMap` (not `HashMap`) so iteration is in ascending `CastId` order
    /// — i.e. cast (insertion) order, matching `unacked()`'s ordering contract.
    board: BTreeMap<CastId, (MailboxId, Vec<KanbanMove>)>,
    /// Casts that have been confirmed (acked), mapped to the `LanceVersion`
    /// the sink assigned when it flushed — the CastId↔LanceVersion join
    /// wiring the WAL into the temporal classifier (see `crate::temporal`).
    /// A cast present in `board` but absent from `acked` is the crash-replay
    /// surface (`unacked()`).
    acked: BTreeMap<CastId, LanceVersion>,
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
            acked: BTreeMap::new(),
            delegation_cache: HashMap::new(),
            pending_payloads: Vec::new(),
        }
    }

    /// AHEAD: records the intent (moves visible on the board) BEFORE any ack.
    /// Returns the cast id.
    ///
    /// "Melden macht frei" (plan Addendum-7): casting is REPORTING, never
    /// refused because earlier casts on the same mailbox are still unacked —
    /// stacked casts are stacked WAL entries, each with its own `CastId`.
    pub fn cast(&mut self, on_behalf: MailboxId, moves: Vec<KanbanMove>, payload: P) -> CastId {
        let cast = CastId(self.next_id);
        self.next_id += 1;
        self.board.insert(cast, (on_behalf, moves));
        self.pending_payloads.push((cast, payload));
        cast
    }

    /// Confirmation — marks the cast acked at the Lance version the sink
    /// assigned (the CastId↔LanceVersion join wiring the WAL into the
    /// temporal classifier; see planner `temporal.rs`).
    pub fn ack(&mut self, cast: CastId, version: LanceVersion) {
        self.acked.insert(cast, version);
    }

    /// The `LanceVersion` a cast was acked at, if it has been acked.
    #[must_use]
    pub fn acked_version(&self, cast: CastId) -> Option<LanceVersion> {
        self.acked.get(&cast).copied()
    }

    /// Crash-replay surface (M24): casts recorded but not yet acked, in
    /// ascending `CastId` (cast) order.
    #[must_use]
    pub fn unacked(&self) -> Vec<CastId> {
        self.board
            .keys()
            .filter(|cast| !self.acked.contains_key(cast))
            .copied()
            .collect()
    }

    /// Board read: intent moves recorded for a cast (visible between cast and ack).
    #[must_use]
    pub fn intent_moves(&self, cast: CastId) -> Option<&[KanbanMove]> {
        self.board.get(&cast).map(|(_, moves)| moves.as_slice())
    }

    /// Board read: the mailbox a cast was recorded ON BEHALF OF — the
    /// delegation-audit surface (W4a cast pairing: consumers pair the owner
    /// at cast; wardens verify the pairing here, never on the DTO).
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
