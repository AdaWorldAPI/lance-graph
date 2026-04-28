//! Audit log (LF-90).
//!
//! META-AGENT: enable behind `audit-log` feature; add `pub mod audit;` to lib.rs;
//! add `audit-log = []` to Cargo.toml [features].
//!
//! Append-only audit log for every RLS-rewritten query. The default
//! [`InMemoryAuditSink`] is a bounded ring buffer; production deployments
//! will swap in a Lance-backed writer in a follow-up PR.

use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// A single audit record describing one RLS-rewritten statement.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub ts_unix_ms: u64,
    pub tenant_id: String,
    pub actor_id: String,
    /// Stable hash of the rewritten LogicalPlan or its display string.
    /// Use std::hash::DefaultHasher (no extra deps).
    pub statement_hash: u64,
    pub statement_kind: StatementKind,
    pub rls_predicates_added: u8,
}

/// Coarse classification of the audited statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementKind {
    Select,
    Insert,
    Update,
    Delete,
    Other,
}

/// Append-only sink. Default impl is in-memory ring buffer; production
/// path swaps in a Lance-backed writer in a follow-up PR.
pub trait AuditSink: Send + Sync {
    fn append(&self, entry: AuditEntry);
    fn snapshot(&self) -> Vec<AuditEntry>;
}

/// In-memory bounded ring buffer used for tests and development.
#[derive(Debug)]
pub struct InMemoryAuditSink {
    entries: Mutex<Vec<AuditEntry>>,
    cap: usize,
}

impl Default for InMemoryAuditSink {
    fn default() -> Self {
        Self::with_capacity(1024)
    }
}

impl InMemoryAuditSink {
    /// Create a sink that retains at most `cap` entries (oldest dropped on overflow).
    /// A `cap` of 0 is treated as 1 to avoid a degenerate sink that drops every entry.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            entries: Mutex::new(Vec::with_capacity(cap)),
            cap,
        }
    }
}

impl AuditSink for InMemoryAuditSink {
    fn append(&self, entry: AuditEntry) {
        // F-09: recover from a poisoned mutex rather than panicking.
        let mut guard = match self.entries.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.len() >= self.cap {
            // Drop oldest (ring semantics).
            guard.remove(0);
        }
        guard.push(entry);
    }

    fn snapshot(&self) -> Vec<AuditEntry> {
        let guard = match self.entries.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.clone()
    }
}

/// Helper to compute statement_hash from any &str.
pub fn hash_statement(stmt_text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    stmt_text.hash(&mut h);
    h.finish()
}

/// Convenience: current wall-clock time in unix milliseconds.
/// Returns 0 if the clock is before the unix epoch (should not happen in practice).
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn sample_entry(tag: &str) -> AuditEntry {
        AuditEntry {
            ts_unix_ms: now_unix_ms(),
            tenant_id: format!("tenant-{tag}"),
            actor_id: format!("actor-{tag}"),
            statement_hash: hash_statement(tag),
            statement_kind: StatementKind::Select,
            rls_predicates_added: 1,
        }
    }

    #[test]
    fn append_then_snapshot_returns_entry() {
        let sink = InMemoryAuditSink::with_capacity(4);
        let e = sample_entry("a");
        let expected_hash = e.statement_hash;
        sink.append(e);
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].tenant_id, "tenant-a");
        assert_eq!(snap[0].actor_id, "actor-a");
        assert_eq!(snap[0].statement_hash, expected_hash);
        assert_eq!(snap[0].statement_kind, StatementKind::Select);
        assert_eq!(snap[0].rls_predicates_added, 1);
    }

    #[test]
    fn ring_overflow_drops_oldest() {
        let sink = InMemoryAuditSink::with_capacity(2);
        sink.append(sample_entry("a"));
        sink.append(sample_entry("b"));
        sink.append(sample_entry("c"));
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 2);
        // Oldest ("a") should have been dropped; remaining should be b, c in order.
        assert_eq!(snap[0].tenant_id, "tenant-b");
        assert_eq!(snap[1].tenant_id, "tenant-c");
    }

    #[test]
    fn poisoned_mutex_recovers() {
        let sink = Arc::new(InMemoryAuditSink::with_capacity(4));
        let sink_clone = Arc::clone(&sink);
        // Poison the mutex by panicking while holding the lock.
        let handle = thread::spawn(move || {
            let _guard = sink_clone.entries.lock().unwrap();
            panic!("intentional panic to poison mutex");
        });
        let _ = handle.join();
        assert!(sink.entries.is_poisoned());
        // append() and snapshot() must still work after poisoning.
        sink.append(sample_entry("post-poison"));
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].tenant_id, "tenant-post-poison");
    }

    #[test]
    fn hash_is_deterministic() {
        let h1 = hash_statement("SELECT * FROM calls WHERE tenant_id = 'x'");
        let h2 = hash_statement("SELECT * FROM calls WHERE tenant_id = 'x'");
        let h3 = hash_statement("SELECT * FROM calls WHERE tenant_id = 'y'");
        assert_eq!(h1, h2, "same input must hash identically within a run");
        assert_ne!(h1, h3, "different inputs should (with overwhelming prob) differ");
    }

    #[test]
    fn zero_capacity_is_normalized_to_one() {
        let sink = InMemoryAuditSink::with_capacity(0);
        sink.append(sample_entry("a"));
        sink.append(sample_entry("b"));
        let snap = sink.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].tenant_id, "tenant-b");
    }
}
