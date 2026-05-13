//! D-SDR-4b audit sink infrastructure — the canonical sink trait.
//!
//! Defines the `AuditSink` trait, `AuditError` enum, and `NoopAuditSink`
//! used by `UnifiedBridge` and the production sinks (`LanceAuditSink`
//! columnar, `JsonlAuditSink` plain text).
//!
//! Per OQ-7-2 (locked 2026-05-13): this is the ONLY audit sink trait.
//! The earlier `UnifiedAuditSink` shim from D-SDR-4 was migrated to this
//! interface in sprint-7. `emit()` returns `Result<_, AuditError>` and
//! moves the event (not `&event`); `flush()` + `checkpoint()` provide
//! durability guarantees that the legacy trait lacked.

use crate::unified_audit::UnifiedAuditEvent;

/// Alias for readability in trait return types.
pub type MerkleRoot = u64;

/// Errors that can arise from audit sink operations.
#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("channel full: {0}")]
    ChannelFull(String),

    #[error("serialization error: {0}")]
    Serialize(String),

    #[error("schema migration blocked: {0}")]
    SchemaMigration(String),

    #[cfg(feature = "lance-sink")]
    #[error("lance write failed: {0}")]
    Lance(String),

    #[cfg(feature = "lance-sink")]
    #[error("arrow schema error: {0}")]
    Arrow(String),
}

/// Pluggable sink for `UnifiedAuditEvent`. Implementations must be
/// `Send + Sync`. The `emit()` hot path MUST NOT block on I/O for
/// more than 1 ms -- the authorize() hot path calls this synchronously.
/// Production sinks buffer asynchronously and flush on a separate task.
pub trait AuditSink: Send + Sync {
    /// Enqueue one event. Non-blocking on the hot path.
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError>;

    /// Flush buffered events to durable storage. Returns the merkle root
    /// of the last flushed event (for checkpoint chaining).
    fn flush(&self) -> Result<MerkleRoot, AuditError>;

    /// Write an atomic checkpoint (last flushed merkle root + timestamp).
    fn checkpoint(&self) -> Result<(), AuditError>;
}

/// No-op sink — discards every event. Default for `UnifiedBridge::new()`
/// when `super_domain.audit_required = false` (no compliance regime requires
/// audit), and for tests. Per OQ-7-3 (locked 2026-05-13): silent default;
/// explicit opt-in to durable sinks via `UnifiedBridge::with_jsonl_audit()` /
/// `with_audit_chain()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopAuditSink;

impl AuditSink for NoopAuditSink {
    fn emit(&self, _event: UnifiedAuditEvent) -> Result<(), AuditError> {
        Ok(())
    }
    fn flush(&self) -> Result<MerkleRoot, AuditError> {
        Ok(0)
    }
    fn checkpoint(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

pub mod composite;

#[cfg(feature = "jsonl")]
pub mod jsonl_sink;

#[cfg(feature = "lance-sink")]
pub mod lance_sink;

pub use composite::{CompositeSink, FanoutMode};

#[cfg(feature = "jsonl")]
pub use jsonl_sink::JsonlAuditSink;

#[cfg(feature = "lance-sink")]
pub use lance_sink::LanceAuditSink;
