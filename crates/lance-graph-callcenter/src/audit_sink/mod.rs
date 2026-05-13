//! D-SDR-4b audit sink infrastructure.
//!
//! Defines the `AuditSink` trait and `AuditError` enum used by both
//! `LanceAuditSink` (columnar) and `JsonlAuditSink` (plain text) as
//! production persistence sinks for `UnifiedAuditEvent`.
//!
//! The legacy `UnifiedAuditSink` in `unified_audit.rs` takes `&UnifiedAuditEvent`
//! and returns `()`. This module introduces `AuditSink` as the D-SDR-4b
//! production interface: `emit()` returns `Result<_, AuditError>`, and
//! `flush()` + `checkpoint()` provide durability guarantees.

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
