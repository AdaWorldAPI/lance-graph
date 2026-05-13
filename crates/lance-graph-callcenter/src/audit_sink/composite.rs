//! CompositeSink — broadcast writes to N child sinks with per-sink failure
//! isolation. Production canonical config (per MedCare-rs sprint-2 audit-sink
//! decision: JSONL primary + optional Lance projection):
//!
//! ```ignore
//! CompositeSink::new(vec![
//!     Box::new(JsonlAuditSink::new(&base_path)?),   // primary (durable, line-oriented)
//!     Box::new(LanceAuditSink::new(&base_path)?),   // optional projection (analytical query)
//! ], FanoutMode::BestEffort)
//! ```

use super::{AuditError, AuditSink, MerkleRoot};
use crate::unified_audit::UnifiedAuditEvent;

/// Controls how `CompositeSink` handles per-sink errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FanoutMode {
    /// First error aborts; remaining sinks NOT called. For test environments.
    FailFast,
    /// All sinks always called. Collects first error; returns Ok if all pass.
    /// Production default.
    BestEffort,
}

/// Broadcasts one `emit()` call to N child sinks with per-sink failure
/// isolation in `BestEffort` mode.
pub struct CompositeSink {
    sinks: Vec<Box<dyn AuditSink>>,
    mode: FanoutMode,
}

impl CompositeSink {
    /// Construct a `CompositeSink` with the given child sinks and fanout mode.
    pub fn new(sinks: Vec<Box<dyn AuditSink>>, mode: FanoutMode) -> Self {
        Self { sinks, mode }
    }
}

impl AuditSink for CompositeSink {
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
        match self.mode {
            FanoutMode::FailFast => {
                for sink in &self.sinks {
                    sink.emit(event)?;
                }
                Ok(())
            }
            FanoutMode::BestEffort => {
                let mut first_err: Option<AuditError> = None;
                for sink in &self.sinks {
                    if let Err(e) = sink.emit(event) {
                        if first_err.is_none() {
                            first_err = Some(e);
                        }
                    }
                }
                first_err.map_or(Ok(()), Err)
            }
        }
    }

    fn flush(&self) -> Result<MerkleRoot, AuditError> {
        let mut last_root: MerkleRoot = 0;
        let mut first_err: Option<AuditError> = None;
        for sink in &self.sinks {
            match sink.flush() {
                Ok(root) => last_root = root,
                Err(e) => {
                    if self.mode == FanoutMode::FailFast {
                        return Err(e);
                    }
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        first_err.map_or(Ok(last_root), Err)
    }

    fn checkpoint(&self) -> Result<(), AuditError> {
        // Always best-effort for checkpoint: one sink failing must not
        // suppress others.
        for sink in &self.sinks {
            if let Err(e) = sink.checkpoint() {
                log::warn!("CompositeSink::checkpoint() sink error (ignored): {e}");
            }
        }
        Ok(())
    }
}
