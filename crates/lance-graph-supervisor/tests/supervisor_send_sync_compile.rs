//! Static `Send + Sync` compile proof for all envelope/reply types.
//!
//! Spec: pr-g2-ractor-supervisor.md §7.2 (I-2 enforcement, static_assertions).
//! This test file is a compile-time check — no runtime assertions needed.

#[cfg(feature = "supervisor")]
mod compile_checks
{
    use static_assertions::assert_impl_all;

    use lance_graph_supervisor::{ConsumerEnvelope, ConsumerReply, SupervisorErr};

    // These assertions fail at compile time if any of the types are not Send + Sync.
    assert_impl_all!(ConsumerEnvelope: Send, Sync);
    assert_impl_all!(ConsumerReply:    Send, Sync);
    assert_impl_all!(SupervisorErr:    Send, Sync);

    // LifecycleAuditEvent (always-present, no feature gate):
    use lance_graph_supervisor::LifecycleAuditEvent;
    assert_impl_all!(LifecycleAuditEvent: Send, Sync);

    #[test]
    fn static_send_sync_assertions_compile()
    {
        // Presence of this test confirms the assert_impl_all! macros above
        // did not prevent compilation — i.e., all types are Send + Sync.
    }
}
