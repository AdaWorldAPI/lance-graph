//! Test: LifecycleAuditEvent emission on actor spawn.
//!
//! Feature-gated under `supervisor-lifecycle-audit`.
//! Verifies that:
//!   1. `LifecycleAuditEvent` has a 18-byte canonical_bytes layout (CC-2 compliance).
//!   2. The event type discriminants are correct.
//!   3. The noop sink discards without panic.
//!
//! CC-2: `UnifiedAuditEvent::canonical_bytes` remains 26 bytes (regression in
//!        `lance-graph-callcenter` tests). This file only tests the SEPARATE
//!        `LifecycleAuditEvent` (18 bytes).

// LifecycleAuditEvent is always-present (no feature gate on the type itself).
use lance_graph_supervisor::{LifecycleAuditEvent, LifecycleAuditSink, LifecycleEventType, NoopLifecycleSink};

#[test]
fn lifecycle_audit_event_canonical_bytes_is_18_not_26()
{
    let ev = LifecycleAuditEvent {
        actor_id:     0xABCD_EF01_2345_6789,
        g_slot:       2,
        timestamp_us: 1_700_000_000_000_000,
        event_type:   LifecycleEventType::ActorStart,
    };
    let bytes = ev.canonical_bytes();
    assert_eq!(bytes.len(), 18, "LifecycleAuditEvent must be 18 bytes, NOT 26 (CC-2)");
    assert_eq!(&bytes[0..8],  &0xABCD_EF01_2345_6789u64.to_le_bytes());
    assert_eq!(&bytes[8..16], &1_700_000_000_000_000u64.to_le_bytes());
    assert_eq!(bytes[16], LifecycleEventType::ActorStart.as_u8());
    assert_eq!(bytes[17], 2u8);
}

#[test]
fn lifecycle_event_type_discriminants()
{
    assert_eq!(LifecycleEventType::ActorStart.as_u8(),   0);
    assert_eq!(LifecycleEventType::ActorStop.as_u8(),    1);
    assert_eq!(LifecycleEventType::ActorRestart.as_u8(), 2);
}

#[test]
fn noop_lifecycle_sink_does_not_panic()
{
    let sink = NoopLifecycleSink;
    let ev = LifecycleAuditEvent {
        actor_id:     42,
        g_slot:       4,
        timestamp_us: 999_999,
        event_type:   LifecycleEventType::ActorRestart,
    };
    sink.emit(&ev); // must not panic
}

#[test]
fn lifecycle_event_g_slot_is_at_byte_17()
{
    // Verifies field ordering in canonical_bytes.
    for g in [0u8, 1, 2, 3, 4, 5, 127, 255] {
        let ev = LifecycleAuditEvent {
            actor_id:     1,
            g_slot:       g,
            timestamp_us: 1,
            event_type:   LifecycleEventType::ActorStop,
        };
        let bytes = ev.canonical_bytes();
        assert_eq!(bytes[17], g, "g_slot must be at byte [17]");
    }
}
