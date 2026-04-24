//! `LanceVersionWatcher` — DM-4 of the callcenter membrane plan.
//!
//! Single-producer / many-consumer fan-out over `tokio::sync::watch`. The
//! membrane is the sole writer (one instance per session); every external
//! subscriber receives the latest `CognitiveEventRow` and skips stale
//! revisions — supabase-realtime shape with always-latest semantics.
//!
//! # BBB invariant
//!
//! The channel payload is `CognitiveEventRow`, the canonical Arrow-scalar
//! outbound DTO. `bbb_scalar_only_compile_check` in `lance_membrane.rs`
//! proves the row carries no VSA / RoleKey / NarsTruth.
//!
//! Plan: `.claude/plans/supabase-subscriber-v1.md` § DM-4.

use tokio::sync::watch;

use crate::external_intent::CognitiveEventRow;

/// Fan-out for projected cognitive events.
///
/// Wraps a `tokio::sync::watch` channel keyed on `CognitiveEventRow`.
/// Created with a sentinel initial value (default row). Each
/// `LanceMembrane::project()` call feeds the latest committed row via
/// [`bump`](Self::bump); subscribers observe it with [`subscribe`](Self::subscribe).
#[derive(Debug)]
pub struct LanceVersionWatcher {
    tx: watch::Sender<CognitiveEventRow>,
}

impl LanceVersionWatcher {
    /// Build a watcher seeded with `initial`.
    ///
    /// The first `subscribe()` call sees this value. Typical construction
    /// uses `CognitiveEventRow::default()` as the sentinel — subscribers
    /// that poll before any `project()` fire see an all-zero row.
    pub fn new(initial: CognitiveEventRow) -> Self {
        let (tx, _rx) = watch::channel(initial);
        Self { tx }
    }

    /// Publish a fresh committed row. All current subscribers observe it.
    ///
    /// Returns `true` when at least one subscriber is listening, `false`
    /// when every receiver has been dropped. The membrane ignores the
    /// return value — a session with zero subscribers is a valid state.
    pub fn bump(&self, row: CognitiveEventRow) -> bool {
        self.tx.send(row).is_ok()
    }

    /// Attach a new subscriber.
    ///
    /// The receiver sees the most recently bumped row on first
    /// `borrow()` and is woken by subsequent bumps. Per `tokio::sync::watch`
    /// semantics, a slow subscriber may skip intermediate revisions — the
    /// supabase-shape "always-latest" guarantee.
    pub fn subscribe(&self) -> watch::Receiver<CognitiveEventRow> {
        self.tx.subscribe()
    }

    /// Observer count — useful for tests and diagnostics.
    pub fn receiver_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl Default for LanceVersionWatcher {
    fn default() -> Self {
        Self::new(CognitiveEventRow::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subscribe_observes_initial() {
        let mut row = CognitiveEventRow::default();
        row.thinking = 7;
        let w = LanceVersionWatcher::new(row);
        let rx = w.subscribe();
        assert_eq!(rx.borrow().thinking, 7);
    }

    #[test]
    fn bump_delivers_latest() {
        let w = LanceVersionWatcher::default();
        let mut rx = w.subscribe();

        let mut row = CognitiveEventRow::default();
        row.free_e = 42;
        assert!(w.bump(row));

        // Manual borrow_and_update to observe the latest value.
        let snapshot = rx.borrow_and_update().clone();
        assert_eq!(snapshot.free_e, 42);
    }

    #[test]
    fn bump_without_subscribers_returns_false() {
        let w = LanceVersionWatcher::default();
        // No subscribers → send succeeds only if a receiver exists.
        // `watch::Sender::send` errors when every receiver has been
        // dropped; we model that as `bump() == false`.
        assert!(!w.bump(CognitiveEventRow::default()));
    }

    #[test]
    fn receiver_count_tracks_subscribers() {
        let w = LanceVersionWatcher::default();
        assert_eq!(w.receiver_count(), 0);
        let _rx1 = w.subscribe();
        let _rx2 = w.subscribe();
        assert_eq!(w.receiver_count(), 2);
    }
}
