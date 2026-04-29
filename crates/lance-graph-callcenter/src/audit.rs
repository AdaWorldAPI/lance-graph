//! Audit log (LF-90).
//!
//! META-AGENT: enable behind `audit-log` feature; add `pub mod audit;` to lib.rs;
//! add `audit-log = []` to Cargo.toml [features].
//! META-AGENT: also add `datafusion-plan = ["dep:datafusion"]` feature alias to
//! Cargo.toml [features] if not already present (gates `audit_from_plan`).
//!
//! Append-only audit log for every RLS-rewritten query. The default
//! [`InMemoryAuditSink`] is a bounded ring buffer; production deployments
//! will swap in a Lance-backed writer in a follow-up PR.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// A single audit record describing one RLS-rewritten statement.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub ts_unix_ms: u64,
    pub tenant_id: String,
    pub actor_id: String,
    /// Stable hash of the rewritten LogicalPlan or its display string.
    /// Computed via [`hash_statement`] (FNV-64a, stable across Rust versions
    /// and platforms — safe to persist and compare across binaries).
    pub statement_hash: u64,
    pub statement_kind: StatementKind,
    pub rls_predicates_added: u16,
    /// Optional rewritten LogicalPlan as a display string. Allows plan replay
    /// for retroactive policy enforcement (epiphany E3 from PR #279 outlook).
    /// None for sinks that don't capture plans (e.g. error-path entries).
    pub rewritten_plan: Option<String>,
}

impl AuditEntry {
    /// Construct an audit entry that retains the rewritten plan's display string.
    /// Used by the policy layer (RlsRewriter) at the moment of plan transformation.
    pub fn with_plan(
        tenant_id: impl Into<String>,
        actor_id: impl Into<String>,
        statement_kind: StatementKind,
        plan_text: impl Into<String>,
        rls_predicates_added: u16,
    ) -> Self {
        let plan_text = plan_text.into();
        let statement_hash = hash_statement(&plan_text);
        Self {
            ts_unix_ms: now_unix_ms(),
            tenant_id: tenant_id.into(),
            actor_id: actor_id.into(),
            statement_hash,
            statement_kind,
            rls_predicates_added,
            rewritten_plan: Some(plan_text),
        }
    }
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
pub trait AuditSink: Send + Sync + std::fmt::Debug {
    fn append(&self, entry: AuditEntry);
    fn snapshot(&self) -> Vec<AuditEntry>;
}

/// In-memory bounded ring buffer used for tests and development.
///
/// Backed by a `VecDeque` so that overflow eviction is O(1) (`pop_front`)
/// rather than O(n) (`Vec::remove(0)`). Append + snapshot remain bounded
/// in time regardless of capacity.
#[derive(Debug)]
pub struct InMemoryAuditSink {
    entries: Mutex<VecDeque<AuditEntry>>,
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
            entries: Mutex::new(VecDeque::with_capacity(cap)),
            cap,
        }
    }
}

impl AuditSink for InMemoryAuditSink {
    fn append(&self, entry: AuditEntry) {
        // F-09: recover from a poisoned mutex rather than panicking.
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if guard.len() == self.cap {
            // Drop oldest (ring semantics) — O(1) on VecDeque.
            guard.pop_front();
        }
        guard.push_back(entry);
    }

    fn snapshot(&self) -> Vec<AuditEntry> {
        let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard.iter().cloned().collect()
    }
}

/// Stable FNV-64a hash of a statement's text (or display form of a LogicalPlan).
///
/// **Stability guarantee:** this is the FNV-1a 64-bit algorithm with the
/// canonical offset basis `0xcbf29ce484222325` and prime `0x100000001b3`.
/// It is byte-for-byte identical across Rust versions, target platforms,
/// and process restarts — making the resulting `statement_hash` safe to
/// persist (e.g. in a Lance-backed audit log) and compare across binaries.
///
/// The previous implementation used `std::collections::hash_map::DefaultHasher`,
/// whose output is explicitly not stable across Rust versions and therefore
/// could not be relied on for long-lived audit records.
pub fn hash_statement(stmt_text: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in stmt_text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Convenience: current wall-clock time in unix milliseconds.
/// Returns 0 if the clock is before the unix epoch (should not happen in practice).
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build an AuditEntry from a rewritten DataFusion LogicalPlan.
/// Used by RlsRewriter at the moment of plan transformation (epiphany E3 hook).
#[cfg(feature = "datafusion-plan")]
pub fn audit_from_plan(
    ctx: &crate::rls::RlsContext,
    kind: StatementKind,
    plan: &datafusion::logical_expr::LogicalPlan,
    predicates_added: u16,
) -> AuditEntry {
    let plan_str = format!("{:?}", plan);
    AuditEntry {
        ts_unix_ms: now_unix_ms(),
        tenant_id: ctx.tenant_id.clone(),
        actor_id: ctx.actor_id.clone(),
        statement_hash: hash_statement(&plan_str),
        statement_kind: kind,
        rls_predicates_added: predicates_added,
        rewritten_plan: Some(plan_str),
    }
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
            rewritten_plan: None,
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
        assert!(snap[0].rewritten_plan.is_none());
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
    fn hash_is_stable_fnv64a() {
        // Spot-check the FNV-64a stability guarantee against known vectors.
        // Empty string → offset basis.
        assert_eq!(hash_statement(""), 0xcbf29ce484222325);
        // "a" → 0xaf63dc4c8601ec8c (canonical FNV-1a 64-bit test vector).
        assert_eq!(hash_statement("a"), 0xaf63dc4c8601ec8c);
        // "foobar" → 0x85944171f73967e8 (canonical test vector).
        assert_eq!(hash_statement("foobar"), 0x85944171f73967e8);
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

    #[test]
    fn with_plan_constructor_captures_plan_text() {
        let entry = AuditEntry::with_plan(
            "tenant-x",
            "actor-x",
            StatementKind::Select,
            "Filter: tenant_id = 'tenant-x'\n  TableScan: calls",
            2,
        );
        assert_eq!(entry.tenant_id, "tenant-x");
        assert_eq!(entry.actor_id, "actor-x");
        assert_eq!(entry.statement_kind, StatementKind::Select);
        assert_eq!(entry.rls_predicates_added, 2);
        let plan = entry.rewritten_plan.expect("plan retained");
        assert!(plan.starts_with("Filter:"));
        assert_eq!(entry.statement_hash, hash_statement(&plan));
    }

    #[test]
    fn concurrent_appends_no_loss() {
        let sink = Arc::new(InMemoryAuditSink::with_capacity(10_000));
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let s = sink.clone();
                thread::spawn(move || {
                    for i in 0..100 {
                        s.append(AuditEntry {
                            ts_unix_ms: now_unix_ms(),
                            tenant_id: format!("t{}", t),
                            actor_id: format!("a{}", i),
                            statement_hash: hash_statement(&format!("q{}-{}", t, i)),
                            statement_kind: StatementKind::Select,
                            rls_predicates_added: 1,
                            rewritten_plan: None,
                        });
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(sink.snapshot().len(), 800);
    }
}
