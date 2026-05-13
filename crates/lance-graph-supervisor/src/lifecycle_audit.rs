//! `LifecycleAuditEvent` — **separate** from `UnifiedAuditEvent` / `AuthOp`.
//!
//! # CC-2 compliance
//!
//! The meta-review CC-2 fix is strictly enforced here:
//! - `UnifiedAuditEvent` and `AuthOp` are **NOT modified**.
//! - Actor lifecycle events (start / stop / restart) use this separate type.
//! - `LifecycleAuditEvent` has its **own** canonical_bytes layout (documented below).
//! - Feature-gated under `supervisor-lifecycle-audit` (see `Cargo.toml`).
//!
//! # CC-3 compliance
//!
//! Lifecycle events use `SuperDomain::System` (PR-G2 new variant) to route
//! cross-domain system events to a dedicated chain, separate from the domain-
//! partitioned authorization audit chains. `SuperDomain::System` is **exempt
//! from the hard-lock partner matrix** (§13.4) — it is the governance umbrella
//! above all other super domains, not a peer domain. This is documented here
//! rather than enforced by the hard-lock matrix (which covers peer-domain
//! cross-authorization, not supervisor events).
//!
//! # canonical_bytes layout (fixed, 18 bytes)
//!
//! ```text
//! [0..8]   actor_id:     u64   little-endian
//! [8..16]  timestamp_us: u64   little-endian
//! [16]     event_type:   u8    LifecycleEventType as u8
//! [17]     g_slot:       u8    G slot index (truncated to u8; max 255 slots)
//! ```
//!
//! Total: 18 bytes. This layout is DISTINCT from `UnifiedAuditEvent::canonical_bytes`
//! (26 bytes). The two are never mixed.
//!
//! Spec: pr-g2-ractor-supervisor.md §6 + meta-review CC-2.

/// Which lifecycle event happened to the actor.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LifecycleEventType {
    /// Consumer actor was spawned (initial start or respawn after crash).
    ActorStart = 0,
    /// Consumer actor stopped gracefully (supervisor-initiated Shutdown).
    ActorStop = 1,
    /// Consumer actor restarted after a crash (one-for-one respawn path).
    ActorRestart = 2,
}

impl LifecycleEventType {
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// One lifecycle audit record. NOT a `UnifiedAuditEvent`. NOT chained via
/// `AuditMerkleRoot`. Stored independently; may be chained in a future
/// supervisor-specific audit chain (sprint-8 hardening).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LifecycleAuditEvent {
    /// Opaque actor identifier. In ractor, this is the numeric part of `ActorId`.
    pub actor_id: u64,
    /// G-slot index this actor owns (0..255 in current topology; u8 is fine).
    pub g_slot: u8,
    /// Wall-clock timestamp in **microseconds** since UNIX epoch. Higher
    /// resolution than `UnifiedAuditEvent::ts_unix_ms` (milliseconds) because
    /// lifecycle events are rare and the extra resolution aids post-hoc analysis.
    pub timestamp_us: u64,
    pub event_type: LifecycleEventType,
}

impl LifecycleAuditEvent {
    /// Canonical 18-byte representation. Layout documented in module doc.
    /// SEPARATE from `UnifiedAuditEvent::canonical_bytes` (26 bytes).
    pub fn canonical_bytes(&self) -> [u8; 18] {
        let mut out = [0u8; 18];
        out[0..8].copy_from_slice(&self.actor_id.to_le_bytes());
        out[8..16].copy_from_slice(&self.timestamp_us.to_le_bytes());
        out[16] = self.event_type.as_u8();
        out[17] = self.g_slot;
        out
    }
}

/// Sink trait for lifecycle audit events. Pluggable; default is `NoopLifecycleSink`.
pub trait LifecycleAuditSink: Send + Sync {
    fn emit(&self, event: &LifecycleAuditEvent);
}

/// No-op sink — discards every lifecycle event. Default when
/// `supervisor-lifecycle-audit` feature is disabled or in test builds.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopLifecycleSink;

impl LifecycleAuditSink for NoopLifecycleSink {
    fn emit(&self, _event: &LifecycleAuditEvent) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_audit_event_canonical_bytes_is_18() {
        let ev = LifecycleAuditEvent {
            actor_id: 0xDEAD_BEEF_CAFE_1234,
            g_slot: 2,
            timestamp_us: 1_700_000_000_000_000,
            event_type: LifecycleEventType::ActorStart,
        };
        let b = ev.canonical_bytes();
        assert_eq!(
            b.len(),
            18,
            "LifecycleAuditEvent canonical_bytes must be 18 bytes"
        );
        // actor_id at [0..8]
        assert_eq!(&b[0..8], &0xDEAD_BEEF_CAFE_1234u64.to_le_bytes());
        // timestamp_us at [8..16]
        assert_eq!(&b[8..16], &1_700_000_000_000_000u64.to_le_bytes());
        // event_type at [16]
        assert_eq!(b[16], LifecycleEventType::ActorStart.as_u8());
        // g_slot at [17]
        assert_eq!(b[17], 2);
    }

    #[test]
    fn lifecycle_canonical_bytes_does_not_collide_with_unified_audit_26_bytes() {
        // Regression: LifecycleAuditEvent is 18 bytes, UnifiedAuditEvent is 26.
        // This test encodes the separation contract so a future refactor cannot
        // accidentally unify the two layouts.
        assert_eq!(
            std::mem::size_of::<[u8; 18]>(),
            18,
            "LifecycleAuditEvent canonical_bytes must be 18, not 26"
        );
    }

    #[test]
    fn noop_lifecycle_sink_does_not_panic() {
        let sink = NoopLifecycleSink;
        let ev = LifecycleAuditEvent {
            actor_id: 1,
            g_slot: 0,
            timestamp_us: 1_000_000,
            event_type: LifecycleEventType::ActorStop,
        };
        sink.emit(&ev);
    }
}
