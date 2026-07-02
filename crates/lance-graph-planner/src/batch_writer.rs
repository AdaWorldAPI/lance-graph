//! W1b ahead-firing batch writer — the kanban board IS the write-ahead log (M24).
//!
//! `cast()` records intent moves AHEAD of any storage ack; `ack()` confirms;
//! `unacked()` is the crash-replay surface. Payload-generic: the writer never
//! inspects `P` (DTO purity — ownership rides the cast pairing, never the DTO).
//!
//! **Zero-copy sink (operator ruling, plan Addendum-6):** `P` is a DESCRIPTOR
//! — (mailbox, dirty row-range, cycle) — never owned delta bytes. Deltas stay
//! in the SoA backing store; the sink reads them through
//! `NodeRowPacket::as_le_bytes` at flush time. The sink drains EAGERLY
//! (ASAP on cast, background), and the phase machine provides the mutual
//! masking: rows in sink phase are mutation-frozen until ack (the owner
//! refuses `advance_phase` re-entry), while thinking proceeds on all other
//! rows — compute masks I/O and vice versa, race-free without buffers.
//!
//! Uses the REAL shipped kanban contract types
//! ([`lance_graph_contract::kanban::KanbanMove`],
//! [`lance_graph_contract::kanban::KanbanColumn`],
//! [`lance_graph_contract::collapse_gate::MailboxId`]) — this module does not
//! mint a parallel `KanbanMove`; see the D-MBX-A6 Outcome adapter context in
//! `crate::strategy::style_strategy`.

use std::collections::HashMap;

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::KanbanMove;

/// Identity of one `cast()` — a write-ahead intent record on the kanban board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    board: HashMap<CastId, (MailboxId, Vec<KanbanMove>)>,
    /// Casts that have been confirmed (acked). A cast present in `board` but
    /// absent from `acked` is the crash-replay surface (`unacked()`).
    acked: std::collections::HashSet<CastId>,
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
            board: HashMap::new(),
            acked: std::collections::HashSet::new(),
            delegation_cache: HashMap::new(),
            pending_payloads: Vec::new(),
        }
    }

    /// AHEAD: records the intent (moves visible on the board) BEFORE any ack.
    /// Returns the cast id.
    ///
    /// Probe-first skeleton (D-V3-W1e): body pending — see
    /// `tests/w1_probes.rs::probe_ahead_update_ordering` /
    /// `probe_kill_after_cast_replay`.
    pub fn cast(&mut self, _on_behalf: MailboxId, _moves: Vec<KanbanMove>, _payload: P) -> CastId {
        todo!("W1b")
    }

    /// Confirmation — marks the cast acked.
    ///
    /// Probe-first skeleton (D-V3-W1e): body pending — see
    /// `tests/w1_probes.rs::probe_ahead_update_ordering`.
    pub fn ack(&mut self, _cast: CastId) {
        todo!("W1b")
    }

    /// Crash-replay surface (M24): casts recorded but not yet acked.
    ///
    /// Probe-first skeleton (D-V3-W1e): body pending — see
    /// `tests/w1_probes.rs::probe_kill_after_cast_replay`.
    pub fn unacked(&self) -> Vec<CastId> {
        todo!("W1b")
    }

    /// Board read: intent moves recorded for a cast (visible between cast and ack).
    ///
    /// Probe-first skeleton (D-V3-W1e): body pending — see
    /// `tests/w1_probes.rs::probe_ahead_update_ordering` /
    /// `probe_kill_after_cast_replay`.
    pub fn intent_moves(&self, _cast: CastId) -> Option<&[KanbanMove]> {
        todo!("W1b")
    }

    /// W1c delegation cache: resolve an owner once, cache; returns `(owner, was_cache_hit)`.
    ///
    /// Probe-first skeleton (D-V3-W1e): body pending — see
    /// `tests/w1_probes.rs::probe_delegation_miss_then_hit`.
    pub fn resolve_owner(
        &mut self,
        _on_behalf: MailboxId,
        _resolver: impl FnOnce(MailboxId) -> MailboxId,
    ) -> (MailboxId, bool) {
        todo!("W1c")
    }
}
