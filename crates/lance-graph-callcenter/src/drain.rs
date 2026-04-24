//! `DrainTask` — DM-6 scaffold of the callcenter membrane plan.
//!
//! Future home for the `steering_intent` Lance-read → `UnifiedStep` →
//! `OrchestrationBridge::route()` pipeline. This PR ships only the
//! type shell and a `Poll::Pending` `drain()` method so that
//! `lib.rs` can re-export the name and consumers can start wiring
//! against the surface. The live drain loop lands in a follow-up.
//!
//! Plan: `.claude/plans/supabase-subscriber-v1.md` § DM-6.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

/// Background task that drains `steering_intent` rows from Lance and
/// forwards them to the `OrchestrationBridge`.
///
/// **Scaffold only.** Fields and the drain loop will be populated in
/// the follow-up PR. Ships now so that `LanceMembrane` consumers can
/// import the symbol and so the `pub mod drain` re-export in
/// `lib.rs` is honest about the type existing.
#[derive(Debug, Default)]
pub struct DrainTask {
    /// Monotonic count of rows drained (zero until DM-6b lands).
    drained: u64,
}

impl DrainTask {
    /// Build an empty drain task. The follow-up PR will add
    /// `new(dataset: &LanceDataset, bridge: Arc<dyn OrchestrationBridge>)`.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many `steering_intent` rows this task has forwarded so far.
    pub fn drained(&self) -> u64 {
        self.drained
    }

    /// Poll the drain loop.
    ///
    /// Returns `Poll::Pending` unconditionally in the scaffold; the
    /// follow-up PR replaces this with the Lance read + route pipeline.
    pub fn drain(&mut self, _cx: &mut Context<'_>) -> Poll<()> {
        Poll::Pending
    }
}

/// `Future` adapter so the scaffold composes with `tokio::spawn`
/// as soon as the drain body lands.
impl Future for DrainTask {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.as_mut().drain(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::task::{RawWaker, RawWakerVTable, Waker};

    fn noop_waker() -> Waker {
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: The vtable functions are all no-ops and never
        // dereference the pointer; null is safe here.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
    }

    #[test]
    fn scaffold_starts_at_zero() {
        let task = DrainTask::new();
        assert_eq!(task.drained(), 0);
    }

    #[test]
    fn drain_is_pending_in_scaffold() {
        let mut task = DrainTask::new();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(task.drain(&mut cx), Poll::Pending));
    }
}
