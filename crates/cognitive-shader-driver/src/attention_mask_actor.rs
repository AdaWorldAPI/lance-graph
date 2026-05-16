//! AttentionMaskActor — actor scaffold for AttentionMask SoA.
//! W6 spec §3. Trait-based; concrete ractor binding is sprint-12+ work.

use lance_graph_contract::collapse_gate::MailboxId;
// NOTE: AttentionMaskSoA from sibling W-F2 file is NOT imported here to avoid
// cross-worker lib.rs ordering issues; the actor's `inner` field uses a generic
// type parameter constrained by the AttentionMaskBackend trait below.

/// Messages the AttentionMaskActor accepts.
#[derive(Clone, Debug)]
pub enum AttentionMaskMsg {
    BindRequest { mailbox_id: MailboxId, w_slot: u8, reply_to: u32 },
    BindReply { mailbox_id: MailboxId, was_new: bool, reply_to: u32 },
    EvictionMsg { evicted: MailboxId },
    Tick,
}

/// Outcome of one actor message handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttentionMaskOutcome {
    Bound { mailbox_id: MailboxId, was_new: bool },
    Evicted { mailbox_id: MailboxId },
    Ticked,
    NoOp,
}

/// Pluggable backend: any type implementing this trait can wire to the actor.
/// W-F2's AttentionMaskSoA satisfies this trait via an impl in main-thread post-fleet aggregation.
pub trait AttentionMaskBackend {
    fn touch(&mut self, mailbox_id: MailboxId, w_slot: u8) -> bool;
    fn evict_lru(&mut self) -> Option<MailboxId>;
    fn tick(&mut self);
    fn is_active(&self, mailbox_id: MailboxId) -> bool;
}

pub struct AttentionMaskActor<B: AttentionMaskBackend> {
    inner: B,
    pending_evictions: Vec<MailboxId>,
}

impl<B: AttentionMaskBackend> AttentionMaskActor<B> {
    pub fn new(inner: B) -> Self {
        Self { inner, pending_evictions: Vec::new() }
    }

    pub fn handle(&mut self, msg: AttentionMaskMsg) -> AttentionMaskOutcome {
        match msg {
            AttentionMaskMsg::BindRequest { mailbox_id, w_slot, .. } => {
                let was_new = self.inner.touch(mailbox_id, w_slot);
                if let Some(evicted) = self.inner.evict_lru() {
                    self.pending_evictions.push(evicted);
                }
                AttentionMaskOutcome::Bound { mailbox_id, was_new }
            }
            AttentionMaskMsg::BindReply { .. } => AttentionMaskOutcome::NoOp,
            AttentionMaskMsg::EvictionMsg { evicted } => {
                self.pending_evictions.push(evicted);
                AttentionMaskOutcome::Evicted { mailbox_id: evicted }
            }
            AttentionMaskMsg::Tick => {
                self.inner.tick();
                AttentionMaskOutcome::Ticked
            }
        }
    }

    pub fn drain_pending_evictions(&mut self) -> Vec<MailboxId> {
        std::mem::take(&mut self.pending_evictions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── FakeBackend ──────────────────────────────────────────────────────────

    struct FakeBackend {
        /// mailbox_id -> w_slot of last touch
        active: HashMap<MailboxId, u8>,
        tick_count: u32,
        /// Queue of mailbox_ids to return from evict_lru, in order.
        evict_queue: Vec<MailboxId>,
    }

    impl FakeBackend {
        fn new() -> Self {
            Self {
                active: HashMap::new(),
                tick_count: 0,
                evict_queue: Vec::new(),
            }
        }

        fn with_evict_queue(evict_queue: Vec<MailboxId>) -> Self {
            Self {
                active: HashMap::new(),
                tick_count: 0,
                evict_queue,
            }
        }
    }

    impl AttentionMaskBackend for FakeBackend {
        fn touch(&mut self, mailbox_id: MailboxId, w_slot: u8) -> bool {
            let was_absent = !self.active.contains_key(&mailbox_id);
            self.active.insert(mailbox_id, w_slot);
            was_absent
        }

        fn evict_lru(&mut self) -> Option<MailboxId> {
            if self.evict_queue.is_empty() {
                None
            } else {
                Some(self.evict_queue.remove(0))
            }
        }

        fn tick(&mut self) {
            self.tick_count += 1;
        }

        fn is_active(&self, mailbox_id: MailboxId) -> bool {
            self.active.contains_key(&mailbox_id)
        }
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    #[test]
    fn test_actor_new_no_pending() {
        let mut actor = AttentionMaskActor::new(FakeBackend::new());
        assert!(actor.drain_pending_evictions().is_empty());
    }

    #[test]
    fn test_actor_bind_request_calls_touch_and_returns_was_new() {
        let mut actor = AttentionMaskActor::new(FakeBackend::new());

        // First bind: mailbox_id=1 is new → was_new == true
        let outcome = actor.handle(AttentionMaskMsg::BindRequest {
            mailbox_id: 1,
            w_slot: 3,
            reply_to: 0,
        });
        assert_eq!(outcome, AttentionMaskOutcome::Bound { mailbox_id: 1, was_new: true });

        // Second bind for the same id: already present → was_new == false
        let outcome2 = actor.handle(AttentionMaskMsg::BindRequest {
            mailbox_id: 1,
            w_slot: 3,
            reply_to: 0,
        });
        assert_eq!(outcome2, AttentionMaskOutcome::Bound { mailbox_id: 1, was_new: false });
    }

    #[test]
    fn test_actor_eviction_msg_adds_to_pending() {
        let mut actor = AttentionMaskActor::new(FakeBackend::new());

        let outcome = actor.handle(AttentionMaskMsg::EvictionMsg { evicted: 42 });
        assert_eq!(outcome, AttentionMaskOutcome::Evicted { mailbox_id: 42 });

        let pending = actor.drain_pending_evictions();
        assert_eq!(pending, vec![42u32]);
    }

    #[test]
    fn test_actor_tick_calls_inner_tick() {
        let mut actor = AttentionMaskActor::new(FakeBackend::new());
        assert_eq!(actor.inner.tick_count, 0);

        let outcome = actor.handle(AttentionMaskMsg::Tick);
        assert_eq!(outcome, AttentionMaskOutcome::Ticked);
        assert_eq!(actor.inner.tick_count, 1);

        actor.handle(AttentionMaskMsg::Tick);
        assert_eq!(actor.inner.tick_count, 2);
    }

    #[test]
    fn test_actor_bind_reply_is_noop() {
        let mut actor = AttentionMaskActor::new(FakeBackend::new());

        let outcome = actor.handle(AttentionMaskMsg::BindReply {
            mailbox_id: 7,
            was_new: true,
            reply_to: 99,
        });
        assert_eq!(outcome, AttentionMaskOutcome::NoOp);
        // No side-effects: inner untouched, no pending evictions
        assert!(actor.inner.active.is_empty());
        assert!(actor.drain_pending_evictions().is_empty());
    }

    #[test]
    fn test_drain_pending_evictions_clears_buffer() {
        // Backend that immediately evicts mailbox 100 on every touch
        let backend = FakeBackend::with_evict_queue(vec![100, 200]);
        let mut actor = AttentionMaskActor::new(backend);

        // Two BindRequests: each causes one LRU eviction from the queue
        actor.handle(AttentionMaskMsg::BindRequest { mailbox_id: 1, w_slot: 0, reply_to: 0 });
        actor.handle(AttentionMaskMsg::BindRequest { mailbox_id: 2, w_slot: 0, reply_to: 0 });

        // First drain returns both and clears
        let first_drain = actor.drain_pending_evictions();
        assert_eq!(first_drain, vec![100u32, 200u32]);

        // Second drain is empty — buffer was cleared
        let second_drain = actor.drain_pending_evictions();
        assert!(second_drain.is_empty());
    }
}
