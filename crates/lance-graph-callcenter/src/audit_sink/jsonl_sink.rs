//! JsonlAuditSink — plain JSONL fallback audit sink.
//!
//! One event per line, `owl_identity` as 6-char lowercase hex, u64 fields
//! as decimal strings (per pr-d3b §1.4 — avoids IEEE 754 double precision
//! loss for values > 2^53).
//!
//! File layout:
//! ```text
//! <base_path>/audit/
//!   <tenant_id>/
//!     2026-05-13.jsonl          # current day - append-only
//!     2026-05-12.jsonl.gz       # prior day - rotated + gzip compressed
//!     _checkpoint.json          # last flushed merkle root
//!     _checkpoint.json.tmp      # write target before atomic rename
//! ```

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write as IoWrite;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

use chrono::Utc;

use super::{AuditError, AuditSink, MerkleRoot};
use crate::unified_audit::UnifiedAuditEvent;

/// Maximum number of events buffered before `emit()` returns
/// `AuditError::ChannelFull`.
pub const JSONL_BUFFER_CAPACITY: usize = 4096;

/// Plain JSONL fallback sink. Thread-safe; `emit()` is non-blocking.
pub struct JsonlAuditSink {
    base_path: PathBuf,
    /// In-memory event buffer; drained on each `flush()` call.
    buffer: Arc<Mutex<VecDeque<UnifiedAuditEvent>>>,
    /// Tracks last fully-flushed checkpoint root for `checkpoint()`.
    last_root: Arc<Mutex<MerkleRoot>>,
}

impl JsonlAuditSink {
    /// Create a new `JsonlAuditSink` rooted at `base_path`.
    pub fn new(base_path: impl Into<PathBuf>) -> Result<Self, AuditError> {
        let base_path = base_path.into();
        std::fs::create_dir_all(base_path.join("audit"))?;
        Ok(Self {
            base_path,
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(256))),
            last_root: Arc::new(Mutex::new(0u64)),
        })
    }
}

impl AuditSink for JsonlAuditSink {
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
        let mut buf = self
            .buffer
            .lock()
            .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?;
        if buf.len() >= JSONL_BUFFER_CAPACITY {
            return Err(AuditError::ChannelFull(format!(
                "jsonl buffer at {} capacity",
                JSONL_BUFFER_CAPACITY
            )));
        }
        buf.push_back(event);
        Ok(())
    }

    fn flush(&self) -> Result<MerkleRoot, AuditError> {
        // 1. Drain buffer under lock.
        let events: Vec<UnifiedAuditEvent> = {
            let mut buf = self
                .buffer
                .lock()
                .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?;
            buf.drain(..).collect()
        };
        if events.is_empty() {
            return Ok(*self
                .last_root
                .lock()
                .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?);
        }

        // 2. Group by (tenant_id, UTC date from timestamp_us).
        let grouped = group_by_tenant_date(&events);
        let today = Utc::now().date_naive();

        // 3. For each group, rotate if needed, then append.
        for ((tenant_id, date), group_events) in &grouped {
            let dir = self
                .base_path
                .join("audit")
                .join(tenant_id.to_string());
            std::fs::create_dir_all(&dir)?;
            if *date < today {
                rotate_if_uncompressed(&dir, *date);
            }
            let file_path = dir.join(format!("{}.jsonl", date.format("%Y-%m-%d")));
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&file_path)?;
            for ev in group_events {
                let line = serialize_event(ev)?;
                file.write_all(line.as_bytes())?;
                file.write_all(b"\n")?;
            }
        }

        // 4. Update last_root from final event.
        let final_root = events
            .last()
            .map(|e| e.merkle_root.raw())
            .unwrap_or(0);
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
        let tmp = self.base_path.join("audit/_checkpoint.json.tmp");
        let live = self.base_path.join("audit/_checkpoint.json");
        let now_us = now_unix_us();
        let json = serde_json::json!({
            "last_merkle_root": root.to_string(),
            "timestamp_us": now_us.to_string(),
            "salt_version": 0u8,
        });
        std::fs::write(
            &tmp,
            serde_json::to_string(&json)
                .map_err(|e| AuditError::Serialize(e.to_string()))?,
        )?;
        std::fs::rename(tmp, live)?; // atomic on POSIX
        Ok(())
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Group events by `(tenant_id u32, UTC date)`.
fn group_by_tenant_date(
    events: &[UnifiedAuditEvent],
) -> HashMap<(u32, chrono::NaiveDate), Vec<&UnifiedAuditEvent>> {
    let mut map: HashMap<(u32, chrono::NaiveDate), Vec<&UnifiedAuditEvent>> = HashMap::new();
    for ev in events {
        let ts_us = ev.ts_unix_ms.saturating_mul(1000);
        let secs = (ts_us / 1_000_000) as i64;
        let date = chrono::DateTime::from_timestamp(secs, 0)
            .map(|dt| dt.date_naive())
            .unwrap_or_else(|| Utc::now().date_naive());
        map.entry((ev.tenant.raw(), date))
            .or_default()
            .push(ev);
    }
    map
}

/// Serialize one event to a JSONL line (no trailing `\n`).
pub fn serialize_event(ev: &UnifiedAuditEvent) -> Result<String, AuditError> {
    let owl_bytes = ev.owl.to_canonical_bytes();
    let owl_hex = format!(
        "{:02x}{:02x}{:02x}",
        owl_bytes[0], owl_bytes[1], owl_bytes[2]
    );
    let ts_us = ev.ts_unix_ms.saturating_mul(1000);
    let json = serde_json::json!({
        "timestamp_us":    ts_us.to_string(),
        "tenant_id":       ev.tenant.raw(),
        "super_domain":    ev.super_domain.raw(),
        "family_id":       owl_bytes[0],
        "owl_identity":    owl_hex,
        "action":          ev.op.as_u8(),
        "decision":        ev.decision.as_u8(),
        "actor_role_hash": ev.actor_role_hash.to_string(),
        "prev_merkle":     ev.prev_merkle.raw().to_string(),
        "event_merkle":    ev.merkle_root.raw().to_string(),
        "payload":         serde_json::Value::Null,
    });
    serde_json::to_string(&json).map_err(|e| AuditError::Serialize(e.to_string()))
}

/// If `YYYY-MM-DD.jsonl` exists and `.gz` does not, gzip-compress in a
/// background thread (fire-and-forget; errors are logged via `log::warn!`).
fn rotate_if_uncompressed(dir: &std::path::Path, date: chrono::NaiveDate) {
    let filename = format!("{}.jsonl", date.format("%Y-%m-%d"));
    let gz_filename = format!("{}.jsonl.gz", date.format("%Y-%m-%d"));
    let src = dir.join(&filename);
    let dst = dir.join(&gz_filename);

    if !src.exists() || dst.exists() {
        return;
    }

    let src_c = src.clone();
    let dst_c = dst.clone();
    std::thread::spawn(move || {
        use std::io::Read;
        match (std::fs::File::open(&src_c), std::fs::File::create(&dst_c)) {
            (Ok(mut input), Ok(output)) => {
                let mut encoder = flate2::write::GzEncoder::new(output, flate2::Compression::default());
                let mut buf = vec![0u8; 65536];
                loop {
                    match input.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            if encoder.write_all(&buf[..n]).is_err() {
                                log::warn!("rotate_if_uncompressed: gz write error");
                                return;
                            }
                        }
                        Err(e) => {
                            log::warn!("rotate_if_uncompressed: read error: {e}");
                            return;
                        }
                    }
                }
                if encoder.finish().is_err() {
                    log::warn!("rotate_if_uncompressed: gz finish error");
                    return;
                }
                // Remove uncompressed original on success.
                let _ = std::fs::remove_file(&src_c);
            }
            _ => {
                log::warn!("rotate_if_uncompressed: could not open files for rotation");
            }
        }
    });
}

/// Current wall clock in microseconds since UNIX epoch.
fn now_unix_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_millis() as u64).saturating_mul(1000))
        .unwrap_or(0)
}
