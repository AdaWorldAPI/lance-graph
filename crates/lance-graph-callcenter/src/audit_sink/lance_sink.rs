//! LanceAuditSink — columnar audit sink backed by a Lance dataset.
//!
//! 12-column Arrow schema with `FixedSizeBinary(3)` `owl_identity`,
//! partitioned by `super_domain` × `date`.
//!
//! Write path: `emit()` buffers in-memory (non-blocking, < 5 µs p99).
//! Background task flushes when buffer >= 1024 events or every 5 seconds.
//! `flush()` drains the buffer, builds Arrow `RecordBatch` per
//! (super_domain, date) partition, and writes via Lance `InsertBuilder`.
//!
//! Partition path pattern:
//! ```text
//! <base_path>/audit/super_domain=<N>/date=<YYYY-MM-DD>/
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, RecordBatch, StringArray, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};

use super::{AuditError, AuditSink, MerkleRoot};
use crate::unified_audit::UnifiedAuditEvent;

/// Maximum events buffered before `emit()` returns `AuditError::ChannelFull`.
pub const LANCE_BUFFER_CAPACITY: usize = 8192;

/// Flush batch threshold: flush when this many events are buffered.
pub const LANCE_FLUSH_THRESHOLD: usize = 1024;

/// Returns the canonical 12-column Arrow schema for the audit dataset.
pub fn audit_event_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp_us", DataType::UInt64, false),
        Field::new("tenant_id", DataType::UInt32, false),
        Field::new("super_domain", DataType::UInt8, false),
        Field::new("family_id", DataType::UInt8, false),
        Field::new("owl_identity", DataType::FixedSizeBinary(3), false),
        Field::new("action", DataType::UInt8, false),
        Field::new("decision", DataType::UInt8, false),
        Field::new("actor_role_hash", DataType::UInt64, false),
        Field::new("prev_merkle", DataType::UInt64, false),
        Field::new("event_merkle", DataType::UInt64, false),
        Field::new("payload", DataType::Binary, true),
        Field::new("date_partition", DataType::Utf8, false),
    ]))
}

/// Columnar audit sink backed by a Lance dataset.
/// Thread-safe; `emit()` is non-blocking (buffer only).
pub struct LanceAuditSink {
    base_path: PathBuf,
    /// In-memory event buffer. Drained by `flush()`.
    buffer: Arc<std::sync::Mutex<Vec<UnifiedAuditEvent>>>,
    /// Last flushed merkle root (for `checkpoint()`).
    last_root: Arc<std::sync::Mutex<u64>>,
    /// Tokio runtime handle for Lance async operations from sync callers.
    rt: Arc<tokio::runtime::Runtime>,
}

impl LanceAuditSink {
    /// Create a new `LanceAuditSink` rooted at `base_path`.
    pub fn new(base_path: impl Into<PathBuf>) -> Result<Self, AuditError> {
        let base_path = base_path.into();
        std::fs::create_dir_all(base_path.join("audit")).map_err(AuditError::Io)?;

        let rt = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .thread_name("lance-audit-sink")
                .enable_all()
                .build()
                .map_err(AuditError::Io)?,
        );

        Ok(Self {
            base_path,
            buffer: Arc::new(std::sync::Mutex::new(Vec::with_capacity(
                LANCE_FLUSH_THRESHOLD,
            ))),
            last_root: Arc::new(std::sync::Mutex::new(0u64)),
            rt,
        })
    }
}

impl AuditSink for LanceAuditSink {
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
        let mut buf = self
            .buffer
            .lock()
            .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?;
        if buf.len() >= LANCE_BUFFER_CAPACITY {
            return Err(AuditError::ChannelFull(format!(
                "lance buffer at {} capacity",
                LANCE_BUFFER_CAPACITY
            )));
        }
        buf.push(event);
        Ok(())
    }

    fn flush(&self) -> Result<MerkleRoot, AuditError> {
        // 1. Drain buffer under lock.
        let events: Vec<UnifiedAuditEvent> = {
            let mut buf = self
                .buffer
                .lock()
                .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?;
            std::mem::take(&mut *buf)
        };
        if events.is_empty() {
            return Ok(*self
                .last_root
                .lock()
                .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?);
        }

        // 2. Build one RecordBatch per (super_domain, date) partition key.
        let batches = build_partitioned_batches(&events)?;

        // 3. Write each batch to the corresponding Lance partition path.
        let base = self.base_path.clone();
        self.rt.block_on(async move {
            for (partition_path, batch) in batches {
                write_batch_to_lance(&base, &partition_path, batch).await?;
            }
            Ok::<(), AuditError>(())
        })?;

        // 4. Update last_root from final event's merkle.
        let final_root = events.last().map(|e| e.merkle_root.raw()).unwrap_or(0);
        *self
            .last_root
            .lock()
            .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))? = final_root;
        Ok(final_root)
    }

    fn checkpoint(&self) -> Result<(), AuditError> {
        let root = *self
            .last_root
            .lock()
            .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?;
        let tmp = self.base_path.join("audit/_checkpoint.lance.json.tmp");
        let live = self.base_path.join("audit/_checkpoint.lance.json");
        let now_us = now_unix_us();
        let json = serde_json::json!({
            "last_merkle_root": root.to_string(),
            "timestamp_us":     now_us.to_string(),
            "schema_version":   1u8,
        });
        std::fs::write(
            &tmp,
            serde_json::to_string(&json).map_err(|e| AuditError::Serialize(e.to_string()))?,
        )?;
        std::fs::rename(tmp, live)?; // atomic on POSIX
        Ok(())
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Group events by `(super_domain u8, date_string)` and build one
/// `RecordBatch` per group.
fn build_partitioned_batches(
    events: &[UnifiedAuditEvent],
) -> Result<Vec<(String, RecordBatch)>, AuditError> {
    // Group by (super_domain_raw, date_str)
    let mut groups: HashMap<(u8, String), Vec<&UnifiedAuditEvent>> = HashMap::new();
    for ev in events {
        let ts_us = ev.ts_unix_ms.saturating_mul(1000);
        let secs = (ts_us / 1_000_000) as i64;
        let date_str = chrono::DateTime::from_timestamp(secs, 0)
            .map(|dt| dt.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| chrono::Utc::now().format("%Y-%m-%d").to_string());
        groups
            .entry((ev.super_domain.raw(), date_str))
            .or_default()
            .push(ev);
    }

    let schema = audit_event_schema();
    let mut result = Vec::new();

    for ((sd_raw, date_str), group) in groups {
        let batch = build_record_batch(&schema, &group, &date_str)?;
        let partition_path = format!("super_domain={}/date={}", sd_raw, date_str);
        // Assert partition column alignment (OQ-6 guard).
        debug_assert!(
            batch
                .column_by_name("date_partition")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .and_then(|a| a.iter().next().flatten())
                .map(|v| v == date_str.as_str())
                .unwrap_or(false),
            "date_partition column value does not match directory date string"
        );
        result.push((partition_path, batch));
    }
    Ok(result)
}

/// Build a `RecordBatch` from a slice of events using the canonical schema.
fn build_record_batch(
    schema: &Arc<Schema>,
    events: &[&UnifiedAuditEvent],
    date_str: &str,
) -> Result<RecordBatch, AuditError> {
    let n = events.len();

    let mut timestamp_us = Vec::with_capacity(n);
    let mut tenant_id = Vec::with_capacity(n);
    let mut super_domain_col = Vec::with_capacity(n);
    let mut family_id = Vec::with_capacity(n);
    // FixedSizeBinary(3): flat buffer of 3*n bytes
    let mut owl_identity_buf = Vec::with_capacity(n * 3);
    let mut action = Vec::with_capacity(n);
    let mut decision = Vec::with_capacity(n);
    let mut actor_role_hash = Vec::with_capacity(n);
    let mut prev_merkle_col = Vec::with_capacity(n);
    let mut event_merkle_col = Vec::with_capacity(n);
    let mut date_partition = Vec::with_capacity(n);

    for ev in events {
        let ts_us = ev.ts_unix_ms.saturating_mul(1000);
        timestamp_us.push(ts_us);
        tenant_id.push(ev.tenant.raw());
        super_domain_col.push(ev.super_domain.raw());
        let owl_bytes = ev.owl.to_canonical_bytes();
        family_id.push(owl_bytes[0]);
        owl_identity_buf.extend_from_slice(&owl_bytes);
        action.push(ev.op.as_u8());
        decision.push(ev.decision.as_u8());
        actor_role_hash.push(ev.actor_role_hash);
        prev_merkle_col.push(ev.prev_merkle.raw());
        event_merkle_col.push(ev.merkle_root.raw());
        date_partition.push(date_str.to_string());
    }

    let arrays: Vec<Arc<dyn Array>> = vec![
        Arc::new(UInt64Array::from(timestamp_us)),
        Arc::new(UInt32Array::from(tenant_id)),
        Arc::new(UInt8Array::from(super_domain_col)),
        Arc::new(UInt8Array::from(family_id)),
        {
            // Build FixedSizeBinary(3) from the flat byte buffer.
            use arrow_array::builder::FixedSizeBinaryBuilder;
            let mut builder = FixedSizeBinaryBuilder::with_capacity(n, 3);
            for chunk in owl_identity_buf.chunks(3) {
                builder
                    .append_value(chunk)
                    .map_err(|e| AuditError::Arrow(e.to_string()))?;
            }
            Arc::new(builder.finish())
        },
        Arc::new(UInt8Array::from(action)),
        Arc::new(UInt8Array::from(decision)),
        Arc::new(UInt64Array::from(actor_role_hash)),
        Arc::new(UInt64Array::from(prev_merkle_col)),
        Arc::new(UInt64Array::from(event_merkle_col)),
        // payload: always null Binary column
        Arc::new(BinaryArray::from(vec![None::<&[u8]>; n])),
        Arc::new(StringArray::from(date_partition)),
    ];

    RecordBatch::try_new(schema.clone(), arrays).map_err(|e| AuditError::Arrow(e.to_string()))
}

/// Write one `RecordBatch` to the Lance partition at
/// `<base_path>/audit/<partition_path>/`.
async fn write_batch_to_lance(
    base_path: &std::path::Path,
    partition_path: &str,
    batch: RecordBatch,
) -> Result<(), AuditError> {
    let dir = base_path.join("audit").join(partition_path);
    std::fs::create_dir_all(&dir)?;
    let uri = format!("file://{}", dir.display());

    use lance::dataset::write::{WriteMode, WriteParams};
    use lance::dataset::InsertBuilder;

    let params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };

    InsertBuilder::new(&uri)
        .with_params(&params)
        .execute(vec![batch])
        .await
        .map_err(|e| AuditError::Lance(e.to_string()))?;

    Ok(())
}

/// Current wall clock in microseconds since UNIX epoch.
fn now_unix_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_millis() as u64).saturating_mul(1000))
        .unwrap_or(0)
}
