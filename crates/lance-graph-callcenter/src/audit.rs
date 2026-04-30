//! Audit log (LF-90).
//!
//! META-AGENT: enable behind `audit-log` feature; add `pub mod audit;` to lib.rs;
//! add `audit-log = []` to Cargo.toml [features].
//! META-AGENT: also add `datafusion-plan = ["dep:datafusion"]` feature alias to
//! Cargo.toml [features] if not already present (gates `audit_from_plan`).
//!
//! Append-only audit log for every RLS-rewritten query. The default
//! [`InMemoryAuditSink`] is a bounded ring buffer; production deployments
//! use [`LanceAuditSink`] which persists entries to a Lance dataset.

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

// ---------------------------------------------------------------------------
// LanceAuditSink — Lance-backed persistent audit writer (PR-F3)
// ---------------------------------------------------------------------------

/// Lance-backed audit sink that persists [`AuditEntry`] rows to a Lance dataset.
///
/// Entries are buffered in memory via `append()`. Calling [`flush()`](LanceAuditSink::flush)
/// converts the buffer into an Arrow `RecordBatch` and appends it to the Lance
/// dataset in append mode (no overwrites). [`scan_back(n)`](LanceAuditSink::scan_back)
/// reads the last `n` entries from the persisted dataset.
///
/// # Schema
///
/// | Column                 | Arrow Type                              | Source                          |
/// |------------------------|-----------------------------------------|---------------------------------|
/// | `tenant_id`            | Utf8                                    | `AuditEntry::tenant_id`         |
/// | `actor_id`             | Utf8                                    | `AuditEntry::actor_id`          |
/// | `statement_hash`       | UInt64                                  | `AuditEntry::statement_hash`    |
/// | `timestamp`            | Timestamp(Millisecond, "UTC")           | `AuditEntry::ts_unix_ms` as i64 |
/// | `action`               | Utf8                                    | `StatementKind` display name    |
/// | `rls_predicates_added` | UInt16                                  | `AuditEntry::rls_predicates_added` |
/// | `rewritten_plan`       | Utf8 (nullable)                         | `AuditEntry::rewritten_plan`    |
///
/// `timestamp` is declared as a temporal type (millisecond precision, UTC) so
/// DataFusion temporal predicates (`>=`, `BETWEEN`, etc.) work on the column;
/// Lance still stores it as int64 underneath. The `as i64` cast at flush-time
/// is therefore safe.
///
/// # Example
///
/// ```ignore
/// let sink = LanceAuditSink::new("/tmp/audit.lance");
/// sink.append(entry);
/// sink.flush().await.unwrap();
/// let recent = sink.scan_back(10).await.unwrap();
/// ```
#[cfg(feature = "audit-log")]
#[derive(Debug)]
pub struct LanceAuditSink {
    /// Path to the Lance dataset directory.
    dataset_path: String,
    /// In-memory buffer of entries not yet flushed.
    buffer: Mutex<Vec<AuditEntry>>,
}

#[cfg(feature = "audit-log")]
impl LanceAuditSink {
    /// Create a new Lance audit sink that will write to the given dataset path.
    /// The dataset is created on first flush if it does not exist.
    pub fn new(dataset_path: impl Into<String>) -> Self {
        Self {
            dataset_path: dataset_path.into(),
            buffer: Mutex::new(Vec::new()),
        }
    }

    /// Flush buffered entries to the Lance dataset (append mode).
    ///
    /// Drains the in-memory buffer, converts entries to an Arrow RecordBatch,
    /// and appends them to the Lance dataset. If the dataset does not exist,
    /// it is created. Returns the number of entries flushed.
    pub async fn flush(&self) -> Result<usize, String> {
        use arrow::array::{StringArray, TimestampMillisecondArray, UInt16Array, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use arrow::record_batch::RecordBatch;
        use lance::dataset::{Dataset, WriteMode, WriteParams};
        use std::sync::Arc;

        let entries: Vec<AuditEntry> = {
            let mut guard = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *guard)
        };

        if entries.is_empty() {
            return Ok(0);
        }

        let n = entries.len();

        // Build columnar arrays from the buffered entries.
        let tenant_ids: Vec<&str> = entries.iter().map(|e| e.tenant_id.as_str()).collect();
        let actor_ids: Vec<&str> = entries.iter().map(|e| e.actor_id.as_str()).collect();
        let hashes: Vec<u64> = entries.iter().map(|e| e.statement_hash).collect();
        let timestamps: Vec<i64> = entries.iter().map(|e| e.ts_unix_ms as i64).collect();
        let actions: Vec<&str> = entries
            .iter()
            .map(|e| statement_kind_str(e.statement_kind))
            .collect();
        let rls_preds: Vec<u16> = entries.iter().map(|e| e.rls_predicates_added).collect();
        let plans: Vec<Option<&str>> =
            entries.iter().map(|e| e.rewritten_plan.as_deref()).collect();

        let tz: Arc<str> = Arc::from("UTC");
        let schema = Arc::new(Schema::new(vec![
            Field::new("tenant_id", DataType::Utf8, false),
            Field::new("actor_id", DataType::Utf8, false),
            Field::new("statement_hash", DataType::UInt64, false),
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Millisecond, Some(tz.clone())),
                false,
            ),
            Field::new("action", DataType::Utf8, false),
            Field::new("rls_predicates_added", DataType::UInt16, false),
            Field::new("rewritten_plan", DataType::Utf8, true),
        ]));

        let ts_array =
            TimestampMillisecondArray::from(timestamps).with_timezone(tz.clone());

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(tenant_ids)),
                Arc::new(StringArray::from(actor_ids)),
                Arc::new(UInt64Array::from(hashes)),
                Arc::new(ts_array),
                Arc::new(StringArray::from(actions)),
                Arc::new(UInt16Array::from(rls_preds)),
                Arc::new(StringArray::from(plans)),
            ],
        )
        .map_err(|e| format!("Arrow batch error: {e}"))?;

        let reader = arrow::record_batch::RecordBatchIterator::new(
            vec![Ok(batch)],
            schema,
        );

        // Determine write mode: Create if new, Append if existing.
        let mode = match Dataset::open(&self.dataset_path).await {
            Ok(_) => WriteMode::Append,
            Err(_) => WriteMode::Create,
        };

        let params = WriteParams {
            mode,
            ..Default::default()
        };

        Dataset::write(reader, &self.dataset_path, Some(params))
            .await
            .map_err(|e| format!("Lance write error: {e}"))?;

        Ok(n)
    }

    /// Read the last `n` entries from the Lance dataset.
    ///
    /// Returns entries in dataset order (oldest first among the returned set).
    /// If fewer than `n` entries exist, all entries are returned.
    pub async fn scan_back(&self, n: usize) -> Result<Vec<AuditEntry>, String> {
        use arrow::array::{
            Array, StringArray, TimestampMillisecondArray, UInt16Array, UInt64Array,
        };
        use futures::TryStreamExt;
        use lance::dataset::Dataset;

        let ds = Dataset::open(&self.dataset_path)
            .await
            .map_err(|e| format!("Lance open error: {e}"))?;

        let total_rows = ds
            .count_rows(None)
            .await
            .map_err(|e| format!("Lance count error: {e}"))?;

        let skip = total_rows.saturating_sub(n);

        // Push the limit + offset into the Lance scanner so we don't pull all
        // fragments only to slice off the tail in process. `Scanner::limit`
        // takes (limit, offset) on Lance v4. If the underlying fragment layout
        // doesn't support efficient offsets, Lance will still degrade
        // gracefully — but we get the win for typical append-only audit logs.
        let mut scanner = ds.scan();
        scanner
            .project(&[
                "tenant_id",
                "actor_id",
                "statement_hash",
                "timestamp",
                "action",
                "rls_predicates_added",
                "rewritten_plan",
            ])
            .map_err(|e| format!("Lance project error: {e}"))?;
        scanner
            .limit(Some(n as i64), Some(skip as i64))
            .map_err(|e| format!("Lance limit error: {e}"))?;

        let batches: Vec<arrow::record_batch::RecordBatch> = scanner
            .try_into_stream()
            .await
            .map_err(|e| format!("Lance stream error: {e}"))?
            .try_collect()
            .await
            .map_err(|e| format!("Lance collect error: {e}"))?;

        let mut entries = Vec::new();

        for batch in &batches {
            let tenant_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("tenant_id column type mismatch")?;
            let actor_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("actor_id column type mismatch")?;
            let hash_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or("statement_hash column type mismatch")?;
            let ts_col = batch
                .column(3)
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or("timestamp column type mismatch")?;
            let action_col = batch
                .column(4)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("action column type mismatch")?;
            let rls_col = batch
                .column(5)
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or("rls_predicates_added column type mismatch")?;
            let plan_col = batch
                .column(6)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("rewritten_plan column type mismatch")?;

            for i in 0..batch.num_rows() {
                let plan = if plan_col.is_null(i) {
                    None
                } else {
                    Some(plan_col.value(i).to_string())
                };
                entries.push(AuditEntry {
                    ts_unix_ms: ts_col.value(i) as u64,
                    tenant_id: tenant_col.value(i).to_string(),
                    actor_id: actor_col.value(i).to_string(),
                    statement_hash: hash_col.value(i),
                    statement_kind: parse_statement_kind(action_col.value(i)),
                    rls_predicates_added: rls_col.value(i),
                    rewritten_plan: plan,
                });
            }
        }

        Ok(entries)
    }
}

#[cfg(feature = "audit-log")]
impl AuditSink for LanceAuditSink {
    fn append(&self, entry: AuditEntry) {
        let mut guard = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        guard.push(entry);
    }

    fn snapshot(&self) -> Vec<AuditEntry> {
        let guard = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        guard.clone()
    }
}

/// Convert [`StatementKind`] to its string representation for the Lance schema.
fn statement_kind_str(kind: StatementKind) -> &'static str {
    match kind {
        StatementKind::Select => "Select",
        StatementKind::Insert => "Insert",
        StatementKind::Update => "Update",
        StatementKind::Delete => "Delete",
        StatementKind::Other => "Other",
    }
}

/// Parse a string back into [`StatementKind`].
fn parse_statement_kind(s: &str) -> StatementKind {
    match s {
        "Select" => StatementKind::Select,
        "Insert" => StatementKind::Insert,
        "Update" => StatementKind::Update,
        "Delete" => StatementKind::Delete,
        _ => StatementKind::Other,
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

    // ── LanceAuditSink tests ──────────────────────────────────────────────

    #[cfg(feature = "audit-log")]
    mod lance_tests {
        use super::*;

        fn lance_sample_entry(tag: &str) -> AuditEntry {
            AuditEntry {
                ts_unix_ms: 1000 + tag.len() as u64,
                tenant_id: format!("tenant-{tag}"),
                actor_id: format!("actor-{tag}"),
                statement_hash: hash_statement(tag),
                statement_kind: StatementKind::Select,
                rls_predicates_added: 2,
                rewritten_plan: None,
            }
        }

        /// Flush 10 entries → scan_back(10) → verify round-trip.
        #[tokio::test]
        async fn flush_10_then_scan_back_roundtrip() {
            let dir = std::env::temp_dir().join(format!(
                "lance_audit_test_10_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let path = dir.to_str().unwrap();

            let sink = LanceAuditSink::new(path);

            // Append 10 entries with distinct tags. The first 5 receive
            // non-default `rls_predicates_added` values so we verify the
            // column survives the Lance round-trip (and isn't silently
            // rebuilt as 0 like the original schema-dropping behaviour).
            for i in 0..10 {
                let mut entry = lance_sample_entry(&format!("e{i}"));
                if i < 5 {
                    entry.rls_predicates_added = (i as u16) + 7; // 7..=11
                }
                sink.append(entry);
            }

            // snapshot() should see the 10 buffered entries.
            assert_eq!(sink.snapshot().len(), 10);

            // Flush to Lance.
            let flushed = sink.flush().await.unwrap();
            assert_eq!(flushed, 10);

            // After flush, in-memory buffer is empty.
            assert_eq!(sink.snapshot().len(), 0);

            // Read back from Lance.
            let entries = sink.scan_back(10).await.unwrap();
            assert_eq!(entries.len(), 10);

            // Verify round-trip field integrity.
            for (i, entry) in entries.iter().enumerate() {
                let tag = format!("e{i}");
                assert_eq!(entry.tenant_id, format!("tenant-{tag}"));
                assert_eq!(entry.actor_id, format!("actor-{tag}"));
                assert_eq!(entry.statement_hash, hash_statement(&tag));
                assert_eq!(entry.statement_kind, StatementKind::Select);
                assert_eq!(entry.ts_unix_ms, 1000 + tag.len() as u64);
                let expected_preds = if i < 5 { (i as u16) + 7 } else { 2 };
                assert_eq!(
                    entry.rls_predicates_added, expected_preds,
                    "rls_predicates_added must round-trip (idx {i})"
                );
            }

            // Cleanup.
            let _ = std::fs::remove_dir_all(&dir);
        }

        /// Flush 1000 entries → verify count via scan_back.
        #[tokio::test]
        async fn flush_1000_verify_count() {
            let dir = std::env::temp_dir().join(format!(
                "lance_audit_test_1000_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let path = dir.to_str().unwrap();

            let sink = LanceAuditSink::new(path);

            // Append 1000 entries.
            for i in 0..1000 {
                sink.append(AuditEntry {
                    ts_unix_ms: i as u64,
                    tenant_id: format!("t-{i}"),
                    actor_id: format!("a-{i}"),
                    statement_hash: hash_statement(&format!("stmt-{i}")),
                    statement_kind: if i % 2 == 0 {
                        StatementKind::Select
                    } else {
                        StatementKind::Insert
                    },
                    rls_predicates_added: 1,
                    rewritten_plan: None,
                });
            }

            let flushed = sink.flush().await.unwrap();
            assert_eq!(flushed, 1000);

            // scan_back with very large n returns all 1000.
            let all = sink.scan_back(2000).await.unwrap();
            assert_eq!(all.len(), 1000);

            // scan_back(10) returns last 10.
            let last_10 = sink.scan_back(10).await.unwrap();
            assert_eq!(last_10.len(), 10);
            // The last entry should be t-999.
            assert_eq!(last_10[9].tenant_id, "t-999");
            // The first of the last 10 should be t-990.
            assert_eq!(last_10[0].tenant_id, "t-990");

            // Verify alternating action round-trip.
            assert_eq!(last_10[0].statement_kind, StatementKind::Select); // 990 is even
            assert_eq!(last_10[1].statement_kind, StatementKind::Insert); // 991 is odd

            // Cleanup.
            let _ = std::fs::remove_dir_all(&dir);
        }

        /// Multiple flushes append (don't overwrite).
        #[tokio::test]
        async fn multiple_flushes_accumulate() {
            let dir = std::env::temp_dir().join(format!(
                "lance_audit_test_accum_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let path = dir.to_str().unwrap();

            let sink = LanceAuditSink::new(path);

            // First batch.
            for i in 0..5 {
                sink.append(lance_sample_entry(&format!("batch1-{i}")));
            }
            sink.flush().await.unwrap();

            // Second batch.
            for i in 0..5 {
                sink.append(lance_sample_entry(&format!("batch2-{i}")));
            }
            sink.flush().await.unwrap();

            // Should have 10 total entries.
            let all = sink.scan_back(100).await.unwrap();
            assert_eq!(all.len(), 10);

            // Cleanup.
            let _ = std::fs::remove_dir_all(&dir);
        }

        /// Failing-test-first (loose end #1): the persisted Lance schema must declare
        /// `timestamp` as a temporal type so DataFusion temporal predicates work.
        /// On the original code this fails because the column is `Int64`.
        #[tokio::test]
        async fn test_timestamp_column_is_temporal_type() {
            use arrow::datatypes::DataType;
            use lance::dataset::Dataset;

            let dir = std::env::temp_dir().join(format!(
                "lance_audit_test_ts_temporal_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let path = dir.to_str().unwrap();

            let sink = LanceAuditSink::new(path);
            sink.append(lance_sample_entry("ts-probe"));
            sink.flush().await.unwrap();

            let ds = Dataset::open(path).await.unwrap();
            let schema = ds.schema();
            // Lance's internal Schema → Arrow Schema for the field type assertion.
            let arrow_schema: arrow::datatypes::Schema = schema.into();
            let ts_field = arrow_schema
                .field_with_name("timestamp")
                .expect("timestamp column exists");
            match ts_field.data_type() {
                DataType::Timestamp(_, _) => { /* pass */ }
                other => panic!(
                    "timestamp must be a temporal type for DataFusion predicates, got {:?}",
                    other
                ),
            }

            let _ = std::fs::remove_dir_all(&dir);
        }

        /// Failing-test-first (loose end #2): `rls_predicates_added` and `rewritten_plan`
        /// must round-trip through Lance. On the original code these were silently
        /// dropped from the schema and rebuilt as `0` / `None` on read-back.
        #[tokio::test]
        async fn test_round_trip_preserves_rls_predicates_added_and_rewritten_plan() {
            let dir = std::env::temp_dir().join(format!(
                "lance_audit_test_lossless_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let path = dir.to_str().unwrap();

            let sink = LanceAuditSink::new(path);

            // 10 entries; 5 of them with non-default rls_predicates_added + rewritten_plan.
            for i in 0..10 {
                let plan = if i < 5 {
                    Some(format!("Filter: tenant_id = 't{i}'\n  TableScan: calls"))
                } else {
                    None
                };
                let preds = if i < 5 { (i as u16) + 3 } else { 0 };
                sink.append(AuditEntry {
                    ts_unix_ms: 2000 + i as u64,
                    tenant_id: format!("t-{i}"),
                    actor_id: format!("a-{i}"),
                    statement_hash: hash_statement(&format!("stmt-{i}")),
                    statement_kind: StatementKind::Select,
                    rls_predicates_added: preds,
                    rewritten_plan: plan,
                });
            }
            sink.flush().await.unwrap();

            let entries = sink.scan_back(10).await.unwrap();
            assert_eq!(entries.len(), 10);

            for (i, e) in entries.iter().enumerate() {
                if i < 5 {
                    assert_eq!(
                        e.rls_predicates_added,
                        (i as u16) + 3,
                        "rls_predicates_added must round-trip (idx {i})"
                    );
                    let plan = e
                        .rewritten_plan
                        .as_ref()
                        .unwrap_or_else(|| panic!("rewritten_plan must round-trip (idx {i})"));
                    assert!(
                        plan.contains(&format!("tenant_id = 't{i}'")),
                        "rewritten_plan content must round-trip (idx {i}): got {plan:?}"
                    );
                } else {
                    assert_eq!(e.rls_predicates_added, 0);
                    assert!(e.rewritten_plan.is_none());
                }
            }

            let _ = std::fs::remove_dir_all(&dir);
        }

        /// Flush of empty buffer is a no-op.
        #[tokio::test]
        async fn flush_empty_is_noop() {
            let dir = std::env::temp_dir().join(format!(
                "lance_audit_test_empty_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let path = dir.to_str().unwrap();

            let sink = LanceAuditSink::new(path);
            let flushed = sink.flush().await.unwrap();
            assert_eq!(flushed, 0);

            // Cleanup (dir may not exist).
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
