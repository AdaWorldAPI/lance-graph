//! `DrainTask` — IN-bound side of the callcenter membrane (TD-INT-12).
//!
//! Drains pending `ExternalIntent` rows into `OrchestrationBridge::route()`.
//!
//! # Phase map
//!
//! - **Phase A (this file, realtime feature):** in-memory channel
//!   (`tokio::sync::mpsc::UnboundedReceiver<ExternalIntent>`). The membrane
//!   pushes intents through a paired `UnboundedSender`; `DrainTask::poll`
//!   greedily drains the channel into the wired `OrchestrationBridge`.
//! - **Phase D (deferred):** the channel sender becomes a Lance row poller
//!   tailing the `steering_intent` dataset. Public surface is unchanged —
//!   `DrainTask` keeps consuming an `mpsc::UnboundedReceiver`; only the
//!   producer side swaps to a Lance reader.
//!
//! Without the `realtime` feature, the trivial `Poll::Pending` scaffold is
//! preserved so the non-realtime build keeps compiling — `tokio::sync::mpsc`
//! lives behind the realtime feature flag, and there is no honest IN-bound
//! drain to ship without it.
//!
//! Plan: `.claude/plans/supabase-subscriber-v1.md` § DM-6, TD-INT-12.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

#[cfg(feature = "realtime")]
use std::sync::Arc;

#[cfg(feature = "realtime")]
use tokio::sync::mpsc;

#[cfg(feature = "realtime")]
use lance_graph_contract::external_membrane::ExternalMembrane;
#[cfg(feature = "realtime")]
use lance_graph_contract::orchestration::OrchestrationBridge;

#[cfg(feature = "realtime")]
use crate::external_intent::ExternalIntent;
#[cfg(feature = "realtime")]
use crate::lance_membrane::LanceMembrane;

// ─────────────────────────────────────────────────────────────────────────────
// Realtime build — live drain loop (TD-INT-12)
// ─────────────────────────────────────────────────────────────────────────────

/// Background task that drains `ExternalIntent` rows from an in-memory
/// channel and forwards them through `LanceMembrane::ingest()` →
/// `OrchestrationBridge::route()`.
///
/// One `DrainTask` per session. Construct with [`drain_channel`], which
/// returns the paired sender and the task. Spawn the task on a tokio
/// runtime; push intents through the sender from the membrane front-end.
///
/// # Phase map
///
/// Phase A: producer is the in-process membrane front-end (HTTP / WS /
/// direct API). Phase D: producer is a Lance dataset row poller. Either
/// way, `DrainTask` itself is unchanged.
#[cfg(feature = "realtime")]
pub struct DrainTask {
    rx: mpsc::UnboundedReceiver<ExternalIntent>,
    bridge: Arc<dyn OrchestrationBridge>,
    membrane: Arc<LanceMembrane>,
    /// Monotonic count of rows drained.
    drained: u64,
}

#[cfg(feature = "realtime")]
impl DrainTask {
    /// How many `ExternalIntent` rows this task has forwarded so far.
    pub fn drained(&self) -> u64 {
        self.drained
    }
}

/// Construct an in-memory drain channel and the paired [`DrainTask`].
///
/// Push `ExternalIntent`s onto the returned `UnboundedSender`; the task
/// drains them through `membrane.ingest(intent)` to produce a
/// `UnifiedStep`, then forwards that step to `bridge.route()`.
///
/// When `bridge.route()` returns `Err`, the row is still counted as drained
/// — the canonical bridge owns retry / failure semantics; `DrainTask` is a
/// pump, not a queue manager.
///
/// # Phase D plan
///
/// The returned `UnboundedSender` is the seam where Phase D wires the
/// Lance `steering_intent` poller. Until that lands, the membrane front-end
/// keeps the sender and pushes intents directly.
#[cfg(feature = "realtime")]
pub fn drain_channel(
    bridge: Arc<dyn OrchestrationBridge>,
    membrane: Arc<LanceMembrane>,
) -> (mpsc::UnboundedSender<ExternalIntent>, DrainTask) {
    let (tx, rx) = mpsc::unbounded_channel();
    let task = DrainTask {
        rx,
        bridge,
        membrane,
        drained: 0,
    };
    (tx, task)
}

/// `Future` adapter so `DrainTask` composes with `tokio::spawn`.
///
/// Returns `Poll::Pending` while the channel is open and empty.
/// Returns `Poll::Ready(())` when the channel is closed (all senders
/// dropped) — that is the canonical shutdown signal.
#[cfg(feature = "realtime")]
impl Future for DrainTask {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        loop {
            match self.rx.poll_recv(cx) {
                Poll::Ready(Some(intent)) => {
                    let mut step = self.membrane.ingest(intent);
                    // Errors are counted as drained — the bridge owns
                    // retry / dead-letter semantics. DrainTask is a pump.
                    let _ = self.bridge.route(&mut step);
                    self.drained += 1;
                    continue;
                }
                Poll::Ready(None) => return Poll::Ready(()),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-realtime build — trivial Pending scaffold (preserves compile)
// ─────────────────────────────────────────────────────────────────────────────

/// Non-realtime scaffold: returns `Poll::Pending` forever.
///
/// The realtime feature owns the live drain loop (`tokio::sync::mpsc`
/// requires tokio). Without realtime there is no in-memory channel and
/// no Lance reader, so there is no honest IN-bound drain to ship. This
/// scaffold exists only to keep the symbol exported on non-realtime
/// builds; consumers should gate their wiring on `feature = "realtime"`.
#[cfg(not(feature = "realtime"))]
#[derive(Debug, Default)]
pub struct DrainTask {
    drained: u64,
}

#[cfg(not(feature = "realtime"))]
impl DrainTask {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn drained(&self) -> u64 {
        self.drained
    }
}

#[cfg(not(feature = "realtime"))]
impl Future for DrainTask {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        Poll::Pending
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "realtime"))]
mod realtime_tests {
    use super::*;
    use lance_graph_contract::{
        external_membrane::ExternalRole,
        nars::InferenceType,
        orchestration::{OrchestrationError, StepDomain, UnifiedStep},
        plan::ThinkingContext,
        thinking::ThinkingStyle,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::dn_path::DnPath;

    /// Mock `OrchestrationBridge` that always returns `Ok(())` and counts
    /// route invocations. Lets tests assert that the bridge actually saw
    /// each drained step.
    struct CountingBridge {
        routed: AtomicU64,
    }

    impl CountingBridge {
        fn new() -> Self {
            Self {
                routed: AtomicU64::new(0),
            }
        }

        fn routed(&self) -> u64 {
            self.routed.load(Ordering::Acquire)
        }
    }

    impl OrchestrationBridge for CountingBridge {
        fn route(&self, _step: &mut UnifiedStep) -> Result<(), OrchestrationError> {
            self.routed.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }

        fn resolve_thinking(
            &self,
            _style: ThinkingStyle,
            _inference_type: InferenceType,
        ) -> ThinkingContext {
            // Tests never call this; if they do, the panic localises the bug.
            unimplemented!("CountingBridge::resolve_thinking is not used in drain tests")
        }

        fn domain_available(&self, _domain: StepDomain) -> bool {
            true
        }
    }

    fn make_dn() -> DnPath {
        DnPath::parse("/tree/ada/heel/callcenter/hip/v1/branch/agents/twig/card/leaf/abc")
            .expect("dn path parses")
    }

    fn make_intent() -> ExternalIntent {
        ExternalIntent::seed(ExternalRole::CrewaiAgent, make_dn(), b"hello".to_vec())
    }

    fn noop_waker() -> core::task::Waker {
        use core::task::{RawWaker, RawWakerVTable, Waker};
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: vtable functions are no-ops and never deref the pointer.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
    }

    #[test]
    fn drain_processes_intent() {
        let bridge = Arc::new(CountingBridge::new());
        let membrane = Arc::new(LanceMembrane::new());
        let (tx, mut task) = drain_channel(bridge.clone(), membrane);

        // Push one intent before polling.
        tx.send(make_intent()).expect("send before poll");

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        // Channel is still open after one push, so the future drains the
        // pending row and parks at Pending waiting for more.
        let poll = Pin::new(&mut task).poll(&mut cx);
        assert!(matches!(poll, Poll::Pending), "open channel parks Pending");

        assert_eq!(task.drained(), 1, "one intent was drained");
        assert_eq!(bridge.routed(), 1, "bridge.route() saw one step");
    }

    #[test]
    fn drain_pending_when_empty() {
        let bridge = Arc::new(CountingBridge::new());
        let membrane = Arc::new(LanceMembrane::new());
        let (_tx, mut task) = drain_channel(bridge.clone(), membrane);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let poll = Pin::new(&mut task).poll(&mut cx);
        assert!(
            matches!(poll, Poll::Pending),
            "empty open channel is Pending"
        );
        assert_eq!(task.drained(), 0);
        assert_eq!(bridge.routed(), 0);
    }

    #[test]
    fn drain_completes_on_channel_close() {
        let bridge = Arc::new(CountingBridge::new());
        let membrane = Arc::new(LanceMembrane::new());
        let (tx, mut task) = drain_channel(bridge.clone(), membrane);

        // Drop the only sender — the channel closes and the future should
        // return Poll::Ready(()).
        drop(tx);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let poll = Pin::new(&mut task).poll(&mut cx);
        assert!(
            matches!(poll, Poll::Ready(())),
            "closed channel completes the drain future"
        );
    }

    #[test]
    fn drain_processes_multiple_intents_greedily() {
        let bridge = Arc::new(CountingBridge::new());
        let membrane = Arc::new(LanceMembrane::new());
        let (tx, mut task) = drain_channel(bridge.clone(), membrane);

        // Push three intents, then close the channel by dropping the sender.
        tx.send(make_intent()).unwrap();
        tx.send(make_intent()).unwrap();
        tx.send(make_intent()).unwrap();
        drop(tx);

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        // One poll should drain all three (greedy loop) and then observe
        // the closed channel for completion.
        let poll = Pin::new(&mut task).poll(&mut cx);
        assert!(matches!(poll, Poll::Ready(())), "drains all then completes");
        assert_eq!(task.drained(), 3);
        assert_eq!(bridge.routed(), 3);
    }
}

#[cfg(all(test, not(feature = "realtime")))]
mod scaffold_tests {
    use super::*;
    use core::task::{RawWaker, RawWakerVTable, Waker};

    fn noop_waker() -> Waker {
        const VTABLE: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VTABLE),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: vtable functions are no-ops and never deref the pointer.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
    }

    #[test]
    fn scaffold_starts_at_zero() {
        let task = DrainTask::new();
        assert_eq!(task.drained(), 0);
    }

    #[test]
    fn scaffold_is_pending() {
        let mut task = DrainTask::new();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let poll = Pin::new(&mut task).poll(&mut cx);
        assert!(matches!(poll, Poll::Pending));
    }
}
