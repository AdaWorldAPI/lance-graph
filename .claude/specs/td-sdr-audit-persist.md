# TD-SDR-AUDIT-PERSIST-1 — Durable Audit Sink Spec

**TD-ID:** TD-SDR-AUDIT-PERSIST-1  
**Priority:** P1  
**Author:** W8 / sprint-log-4 (2026-05-13)  
**Branch:** `claude/lance-datafusion-integration-gv0BF`  
**Crate target:** `crates/lance-graph-callcenter/src/audit_sink/`

---

## 1. Problem Statement

D-SDR-4 shipped `UnifiedAuditEvent` + `AuditChain` + `NoopUnifiedAuditSink` in
`crates/lance-graph-callcenter/src/unified_audit.rs`. The chain is merkle-stamped in
memory and tamper-detectable — but on process restart, all events are lost. Production
compliance regimes (HIPAA §164.312(b), SOC-2 CC7.2) require durable audit storage
reachable by a forensic verifier.

This spec defines the full production sink stack:
- `AuditSink` trait (superseding/extending `UnifiedAuditSink`)
- `LanceAuditSink` — Arrow57 columnar, partitioned by `tenant_id/day`
- `JsonlAuditSink` — plain JSONL fallback, daily rotation with gzip-on-rotate
- `CompositeSink` — fanout with fail-fast vs. best-effort mode
- `audit-verify` binary — replay + chain validation from Lance
- Performance budget and cross-flag notes for W4 / W10

---

## 2. Existing Surface (D-SDR-4 baseline)

### 2.1 `UnifiedAuditEvent` (line 146, `unified_audit.rs`)

```rust
pub struct UnifiedAuditEvent {
    pub ts_unix_ms:      u64,           // wall-clock ms since UNIX epoch
    pub tenant:          TenantId,      // u32 newtype
    pub super_domain:    SuperDomain,   // u8 repr enum
    pub owl:             OwlIdentity,   // u16 = (family_id:u8 << 8) | slot:u8
    pub op:              AuthOp,        // u8: Read=0, Write=1, Act=2
    pub decision:        AuthDecision,  // u8: Allow=0, Deny=1, Escalate=2, BridgeError=3
    pub actor_role_hash: u64,           // FNV-1a 64-bit of role &str
    pub merkle_root:     AuditMerkleRoot, // u64 FNV-1a chain output
}
```

`canonical_bytes()` = 25 bytes (excludes `merkle_root`; merkle_root is the OUTPUT of
`AuditMerkleRoot::chain(prev_root, salt, canonical_bytes)`).

`AuditMerkleRoot` = `pub struct AuditMerkleRoot(pub u64)` — 8 bytes, FNV-1a 64-bit.
`GENESIS = 0xa5a5_a5a5_a5a5_a5a5`.

**Correction vs. brief:** The brief specified `prev_merkle [u8;32]` and `event_merkle [u8;32]`.
The actual implementation uses `u64` (FNV-1a 64-bit, not SHA-256). The Arrow schema below
uses `UInt64` for these fields accordingly.

### 2.2 Existing trait (line 258)

```rust
pub trait UnifiedAuditSink: Send + Sync {
    fn emit(&self, event: &UnifiedAuditEvent);
}
```

The existing trait has no `flush`, no `checkpoint`, no error return. We do NOT break
this interface — instead we add the new `AuditSink` trait alongside it and migrate
callsites in a follow-up (D-SDR-4b).

---

## 3. AuditSink Trait

**Location:** `crates/lance-graph-callcenter/src/audit_sink/mod.rs`

```rust
use crate::unified_audit::{AuditMerkleRoot, UnifiedAuditEvent};

/// Durable audit sink. Replaces `UnifiedAuditSink` on the `UnifiedBridge`
/// hot path once D-SDR-4b lands. The old `UnifiedAuditSink` remains for
/// backward compatibility; an `AuditSink` blanket impl covers it.
pub trait AuditSink: Send + Sync {
    /// Append one event to the sink's internal buffer. MUST NOT block on I/O.
    /// For Lance/JSONL sinks this is a lock-free channel send.
    /// Target: <10 µs p99 on the calling thread.
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError>;

    /// Drain the buffer, write all pending events to durable storage,
    /// and return the merkle root of the last-flushed event.
    /// Called by the flush task on a background Tokio runtime.
    /// Target: <50 ms per 1024-event batch (Lance RecordBatch write).
    fn flush(&self) -> Result<MerkleRoot, AuditError>;

    /// Persist the current merkle root as a checkpoint record alongside the
    /// batch so chain validation survives process restart.
    /// Typically called immediately after `flush()`.
    fn checkpoint(&self) -> Result<(), AuditError>;
}

/// The persisted merkle root type — u64 (FNV-1a 64-bit, matches AuditMerkleRoot).
pub type MerkleRoot = u64;

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("sink channel full: {0}")]
    ChannelFull(String),
    #[error("lance write error: {0}")]
    Lance(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(String),
}
```

### 3.1 Compatibility shim

```rust
/// Blanket: any AuditSink is also a UnifiedAuditSink (swallow errors, fire-and-forget).
impl<S: AuditSink> UnifiedAuditSink for S {
    fn emit(&self, event: &UnifiedAuditEvent) {
        let _ = AuditSink::emit(self, *event);
    }
}
```

This lets callers migrate incrementally without breaking the existing
`sink.emit(&event)` callsites in `UnifiedBridge::authorize()`.

---

## 4. Arrow Schema for UnifiedAuditEvent

**Location:** `crates/lance-graph-callcenter/src/audit_sink/lance_sink.rs`

```rust
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Fixed Arrow schema for UnifiedAuditEvent.
/// All integers little-endian (Arrow default).
/// Field names match canonical_bytes() field order for alignment.
pub fn audit_event_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp_us",    DataType::UInt64, false), // ts_unix_ms * 1000
        Field::new("tenant_id",       DataType::UInt32, false),
        Field::new("super_domain",    DataType::UInt8,  false),
        Field::new("family_id",       DataType::UInt8,  false), // owl >> 8
        Field::new("owl_identity",    DataType::UInt16, false), // full OwlIdentity u16
        Field::new("action",          DataType::UInt8,  false), // AuthOp as u8
        Field::new("decision",        DataType::UInt8,  false), // AuthDecision as u8
        Field::new("actor_role_hash", DataType::UInt64, false),
        Field::new("prev_merkle",     DataType::UInt64, false), // AuditChain::last_root before advance()
        Field::new("event_merkle",    DataType::UInt64, false), // AuditMerkleRoot after advance()
        Field::new("payload",         DataType::Binary, true),  // reserved; NULL for core events
    ]))
}
```

**Notes:**

- `timestamp_us` = `ts_unix_ms * 1000`. Storing in microseconds aligns with Arrow/Parquet
  timestamp conventions and makes DataFusion predicate pushdown on time ranges efficient.

- `prev_merkle` must be captured at emit time (before `AuditChain::advance()` updates
  `last_root`). The sink receives the already-chained event — the caller must pass
  `chain.last_root` BEFORE calling `advance()`, or the event struct must be extended.
  **Action for D-SDR-4b:** extend `UnifiedAuditEvent` with a `prev_merkle: AuditMerkleRoot`
  field set inside `AuditChain::advance()` before the call to `AuditMerkleRoot::chain()`.

- `payload` is `Binary` (variable-length bytes), nullable. Reserved for future extensions
  (e.g., serialized RBAC policy snapshot, OWL class label). NULL for the initial impl.

- `super_domain` and `family_id` are redundant with `owl_identity` but are stored
  separately for DataFusion predicate pushdown without bitmask arithmetic.

---

## 5. LanceAuditSink

**Location:** `crates/lance-graph-callcenter/src/audit_sink/lance_sink.rs`

### 5.1 Partitioning layout

```
<base_path>/audit/
  tenant_id=<u32>/
    yyyy=2026/
      mm=05/
        dd=13/
          part-<uuid>.lance
```

This matches Hive-style partitioning, which Lance + DataFusion understand natively.
The partition key is `(tenant_id, yyyy, mm, dd)` — all four derived from `timestamp_us`.

### 5.2 Batch size

**1024 events per `RecordBatch`.** The flush task accumulates events in a
`VecDeque<UnifiedAuditEvent>` inside a `Mutex`. When the buffer reaches 1024 or the
flush timer fires (whichever is first), it drains up to 1024 events, constructs one
`RecordBatch`, and writes it to Lance.

### 5.3 Struct

```rust
pub struct LanceAuditSink {
    base_path:    PathBuf,
    schema:       Arc<Schema>,
    buffer:       Arc<Mutex<VecDeque<UnifiedAuditEvent>>>,
    last_root:    Arc<Mutex<MerkleRoot>>,  // tracks last flushed root for checkpoint
    flush_handle: Option<JoinHandle<()>>,  // background Tokio task
}
```

### 5.4 emit() implementation

```rust
fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
    let mut buf = self.buffer.lock()
        .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?;
    if buf.len() >= 4096 {
        return Err(AuditError::ChannelFull("buffer at 4096 capacity".into()));
    }
    buf.push_back(event);
    Ok(())
}
```

**Hot path: only a Mutex lock + VecDeque push.** No I/O. Lock contention is bounded
because the flush task drains in batches of 1024, keeping `buf.len()` well below 4096
at steady state. Target: <10 µs p99 including lock acquisition on modern hardware.

For zero-contention paths, a future iteration can replace `Mutex<VecDeque>` with
`crossbeam::queue::SegQueue` or a per-thread buffer with a background collector.

### 5.5 flush() implementation (background task)

```rust
async fn flush_inner(
    base_path: &Path,
    schema: &Arc<Schema>,
    events: Vec<UnifiedAuditEvent>,
    last_root: &Arc<Mutex<MerkleRoot>>,
) -> Result<MerkleRoot, AuditError> {
    // 1. Group by (tenant_id, yyyy, mm, dd)
    let grouped = group_by_partition(events);

    for (partition_key, batch_events) in grouped {
        // 2. Build RecordBatch (up to 1024 events per batch)
        for chunk in batch_events.chunks(1024) {
            let record_batch = build_record_batch(schema, chunk)?;
            // 3. Write to Lance partition
            let path = partition_path(base_path, &partition_key);
            lance_write_batch(&path, schema, record_batch).await?;
        }
    }

    // 4. Update last_root
    let final_root = events.last().map(|e| e.merkle_root.raw()).unwrap_or(0);
    *last_root.lock().unwrap() = final_root;
    Ok(final_root)
}
```

### 5.6 checkpoint() implementation

```rust
fn checkpoint(&self) -> Result<(), AuditError> {
    let root = *self.last_root.lock().unwrap();
    let tmp_path  = self.base_path.join("audit/_checkpoint.json.tmp");
    let live_path = self.base_path.join("audit/_checkpoint.json");
    let json = serde_json::json!({
        "last_merkle_root": root.to_string(),  // decimal string (OQ-4)
        "timestamp_us": now_unix_us(),
        "salt_version": 0u8,                   // reserved for OQ-1
    });
    std::fs::write(&tmp_path, json.to_string())?;
    std::fs::rename(tmp_path, live_path)?;      // atomic on POSIX
    Ok(())
}
```

On restart, `AuditChain::resume()` reads `last_merkle_root` from this file to seed
`last_root`.

---

## 6. JsonlAuditSink

**Location:** `crates/lance-graph-callcenter/src/audit_sink/jsonl_sink.rs`

### 6.1 Design goals

- Zero dependency on Arrow/Lance — suitable for low-tooling environments
- One file per tenant per day: `<base_path>/audit/<tenant_id>/YYYY-MM-DD.jsonl`
- Gzip-on-rotate: when a day rolls over, the previous day's file is gzip-compressed
  in a background thread (`YYYY-MM-DD.jsonl.gz`)
- Schema mirrors Lance fields exactly (same field names, same types) for easy
  cross-format joins

### 6.2 JSONL record format

```json
{
  "timestamp_us":    "1747180800000000",
  "tenant_id":       42,
  "super_domain":    1,
  "family_id":       7,
  "owl_identity":    1797,
  "action":          0,
  "decision":        0,
  "actor_role_hash": "14627333968358193902",
  "prev_merkle":     "12297829382473034410",
  "event_merkle":    "9823479283742938472",
  "payload":         null
}
```

`u64` fields that may exceed 2^53 are emitted as decimal strings (see OQ-4):
`timestamp_us`, `actor_role_hash`, `prev_merkle`, `event_merkle`.
`tenant_id`, `super_domain`, `family_id`, `owl_identity`, `action`, `decision`
are safe as JSON numbers (all fit in u32 or smaller).

### 6.3 Struct

```rust
pub struct JsonlAuditSink {
    base_path:    PathBuf,
    buffer:       Arc<Mutex<VecDeque<UnifiedAuditEvent>>>,
    current_day:  Arc<Mutex<chrono::NaiveDate>>,
    last_root:    Arc<Mutex<MerkleRoot>>,
}
```

### 6.4 File rotation

`flush()` checks if `current_day` has changed (UTC). If yes:
1. Close the current file handle (flush OS buffers).
2. Spawn a background thread: compress old file with `flate2::GzEncoder`, write
   `YYYY-MM-DD.jsonl.gz`, then remove the uncompressed original.
3. Open a new file for today's date.
4. Update `current_day`.

### 6.5 emit() / flush() / checkpoint()

Same pattern as `LanceAuditSink`. `emit()` is a buffer push (< 10 µs).
`flush()` serializes each event to JSON and appends to the current day's file.
`checkpoint()` writes `_checkpoint.json` in the same tenant subdirectory using
the same atomic-rename pattern.

---

## 7. CompositeSink

**Location:** `crates/lance-graph-callcenter/src/audit_sink/composite.rs`

```rust
pub struct CompositeSink {
    sinks: Vec<Box<dyn AuditSink>>,
    mode:  FanoutMode,
}

#[derive(PartialEq)]
pub enum FanoutMode {
    /// First sink error aborts; remaining sinks are NOT called.
    FailFast,
    /// All sinks are called regardless of individual failures.
    /// Returns the first error encountered, or Ok if all succeeded.
    BestEffort,
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
                let mut first_err = None;
                for sink in &self.sinks {
                    if let Err(e) = sink.emit(event) {
                        if first_err.is_none() { first_err = Some(e); }
                    }
                }
                first_err.map_or(Ok(()), Err)
            }
        }
    }

    fn flush(&self) -> Result<MerkleRoot, AuditError> {
        let mut root = 0u64;
        let mut first_err = None;
        for sink in &self.sinks {
            match sink.flush() {
                Ok(r) => root = r,
                Err(e) => {
                    if self.mode == FanoutMode::FailFast {
                        return Err(e);
                    }
                    if first_err.is_none() { first_err = Some(e); }
                }
            }
        }
        first_err.map_or(Ok(root), Err)
    }

    fn checkpoint(&self) -> Result<(), AuditError> {
        // Always best-effort for checkpoint — don't fail the whole chain if
        // one sink can't write its checkpoint marker.
        for sink in &self.sinks {
            let _ = sink.checkpoint();
        }
        Ok(())
    }
}
```

**Canonical production configuration:**

```rust
CompositeSink {
    sinks: vec![
        Box::new(LanceAuditSink::new(&base_path)?),
        Box::new(JsonlAuditSink::new(&base_path)?),
    ],
    mode: FanoutMode::BestEffort,
}
```

Use `BestEffort` in production: if Lance is temporarily unreachable, JSONL still
captures the events and the chain remains auditable. Operators can replay JSONL
into Lance once storage recovers.

---

## 8. Replay / Verify Tool

**Binary:** `crates/lance-graph-callcenter/src/bin/audit_verify.rs`  
**Invocation:** `cargo run --bin audit-verify -- --since 2026-05-01 [--tenant 42]`

### 8.1 Algorithm

```
1. Read _checkpoint.json → seed_root (AuditMerkleRoot)
2. Scan Lance partitions matching (tenant_id, date >= since)
   ORDER BY (tenant_id, timestamp_us) ASC
3. For each event row:
   a. Reconstruct UnifiedAuditEvent from Arrow columns
   b. Recompute expected_root = AuditMerkleRoot::chain(prev_root, salt, event.canonical_bytes())
   c. Assert expected_root == event.event_merkle
      If mismatch: print "CHAIN BREAK at row <i>: tenant=<t>, ts=<ts_us>, expected=<x>, got=<y>"
   d. prev_root = expected_root
4. On success: print "OK: <N> events verified, final root = <root>"
```

### 8.2 Salt sourcing

The per-super-domain salt is embedded in `SuperDomainRegistry::merkle_salt(super_domain)`.
The verify tool reads this from the running crate's constant table — same binary, same
constants. If salt rotation is ever introduced (see OQ-1), the checkpoint file must also
store the salt version used at each batch boundary.

### 8.3 CLI interface

```
audit-verify [OPTIONS]

Options:
  --since <DATE>       ISO 8601 date; scan events from this day forward [required]
  --until <DATE>       ISO 8601 date; stop scanning here [default: today]
  --tenant <ID>        Restrict to one tenant_id [default: all tenants]
  --base-path <PATH>   Lance base path [default: $AUDIT_BASE_PATH env var]
  --seed-root <HEX>    Override checkpoint root (hex u64) [default: read checkpoint]
  --verbose            Print each row's computed vs stored root
```

### 8.4 Exit codes

- `0` — all events verified, chain intact
- `1` — chain break detected (details printed to stdout)
- `2` — I/O or schema error (details printed to stderr)

---

## 9. Performance Budget

| Operation | Target | Mechanism |
|---|---|---|
| `emit()` p99 | < 10 µs | Mutex + VecDeque push; no I/O |
| `flush()` per 1024-batch | < 50 ms | Lance RecordBatch write; background Tokio task |
| `checkpoint()` | < 5 ms | Single JSON file write + atomic rename |
| `audit-verify` scan | < 5 s / 1M events | DataFusion predicate pushdown on partition columns |

**Latency breakdown for emit() target (10 µs p99):**
- Mutex lock acquisition: ~50 ns (uncontended)
- VecDeque push_back: ~20 ns
- Total hot path: ~100 ns — well under 10 µs budget even with OS scheduling jitter
- P99 includes lock contention: if flush task holds lock for batch drain, up to
  1–2 µs added. Still within budget.

**Flush task design:** a Tokio task running in `tokio::spawn` (separate from the
`authorize()` hot path). Wakes on either:
- Channel notification (event count reached 1024)
- Timer tick (every 1 second, flush whatever is buffered)

This ensures `checkpoint()` is called at most once per second, not per event.

---

## 10. Cross-Flag Notes

### W4 — Super-Domain Subcrate Cascade (TD-SUPER-DOMAIN-SUBCRATES-1)

Each super-domain subcrate (`medcare-bridge`, `smb-bridge`, `hubspot-bridge`, etc.)
owns its own `UnifiedBridge` instance. Per W4's spec, each subcrate will expose:

```rust
pub fn make_bridge(config: &SubcrateConfig) -> UnifiedBridge {
    let sink = CompositeSink {
        sinks: vec![
            Box::new(LanceAuditSink::new(&config.audit_base_path).expect("audit sink")),
        ],
        mode: FanoutMode::BestEffort,
    };
    UnifiedBridge::new(config, Arc::new(sink))
}
```

**Implication:** each subcrate has an independent `AuditChain` starting from its own
seed (or resuming from its own `_checkpoint.json`). Cross-subcrate audit correlation
is NOT supported by design (§13.4 cross-domain unlinkability).

The `audit-verify` binary must be invoked per-subcrate base path:

```
audit-verify --base-path /data/medcare-bridge/audit --since 2026-05-01
```

### W10 — BridgeError Audit Fix (TD-SDR-BRIDGE-ERR-AUDIT-1)

The current `UnifiedBridge::authorize()` short-circuits on `BridgeError` before
calling `sink.emit()`. W10's fix must ensure:

```rust
// BEFORE returning BridgeError from authorize():
let event = UnifiedAuditEvent {
    decision: AuthDecision::BridgeError,
    // ... other fields ...
    merkle_root: AuditMerkleRoot::GENESIS,  // will be overwritten by chain.advance()
};
let stamped = chain.advance(event);
// Emit but do NOT propagate sink error as BridgeError:
if let Err(e) = sink.emit(stamped) {
    log::warn!("audit emit failed on BridgeError path: {e}");
}
// Then return the original BridgeError to the caller
```

**Probe-detection signal:** `AuthDecision::BridgeError` events in the audit log
are the primary signal for detecting systematic probe attacks (a caller triggering
repeated bridge errors to fingerprint the system). W10 must not swallow these.
The JSONL sink in particular is the fallback for low-tooling forensic review of
probe patterns.

---

## 11. Open Questions

**OQ-1: merkle salt rotation.**  
The current design uses a fixed `merkle_salt` per super-domain embedded in the
crate constant table. If salts must rotate (e.g., per-quarter for compliance
freshness), the checkpoint file must store `(last_root, salt_version)` and the
verify tool must look up the historical salt for each batch window. Deferring
salt rotation to a separate TD item — but the checkpoint schema must reserve a
`"salt_version": 0` field now (already included in §5.6) to avoid a breaking
change later.

**OQ-2: multi-process / multi-instance safety.**  
`LanceAuditSink` assumes a single writer per partition path. In a horizontally
scaled deployment (multiple `UnifiedBridge` instances in separate processes),
concurrent Lance writers to the same partition will conflict. Options:
(a) per-instance sub-partition keyed on `instance_id` (e.g., `instance_id=<uuid>/`),
(b) Lance's built-in multi-writer mode (check Lance 2.x API for `WriteMode::Append`
concurrency guarantees), (c) a dedicated audit aggregator service receiving events
over a channel. This is unresolved.

**OQ-3: Arrow UInt8 for super_domain vs. UInt32.**  
Arrow's `UInt8` type works but some DataFusion optimizations (e.g., Bloom filters)
are only implemented for `UInt32`/`UInt64`. If filter pushdown on `super_domain` is
needed for the verify tool's scan performance, promoting to `UInt32` costs 3 bytes
per event (1 → 4) but unlocks predicate pushdown. Decision deferred to perf testing.

**OQ-4: JSONL numeric precision for u64 merkle fields.**  
JSON `number` is IEEE 754 double — values above 2^53 lose precision. The FNV-1a
outputs for `actor_role_hash`, `prev_merkle`, and `event_merkle` can exceed 2^53.
This spec proposes decimal strings as default (§6.2). An alternative is hex strings
(`"event_merkle": "0x884a1b2c3d4e5f60"`). Either breaks naive JSON number parsing;
the D-SDR-4b implementer should confirm with downstream log consumers before locking
the wire format.

---

## 12. File Manifest

```
crates/lance-graph-callcenter/src/audit_sink/
  mod.rs          — AuditSink trait, AuditError, MerkleRoot type alias, compat shim
  lance_sink.rs   — LanceAuditSink + audit_event_schema() + build_record_batch()
  jsonl_sink.rs   — JsonlAuditSink + rotation + gzip logic
  composite.rs    — CompositeSink + FanoutMode
  checkpoint.rs   — read_checkpoint() / write_checkpoint() helpers

crates/lance-graph-callcenter/src/bin/
  audit_verify.rs — CLI replay + chain verifier

crates/lance-graph-callcenter/Cargo.toml additions:
  lance          = "2"
  arrow          = "57"
  tokio          (likely already present)
  serde_json     = "1"
  chrono         = { version = "0.4", features = ["serde"] }
  thiserror      = "1"
  flate2         = "1"   (gzip for JSONL rotation)
```

---

## 13. Implementation Order (for the D-SDR-4b engineer)

1. Add `audit_sink/mod.rs` with `AuditSink` trait + `AuditError` + compat shim.
2. Add `audit_event_schema()` in `lance_sink.rs`; stub `LanceAuditSink` with no-op flush.
3. Extend `UnifiedAuditEvent` with `prev_merkle: AuditMerkleRoot` (set in `AuditChain::advance()`).
4. Implement `build_record_batch()` — fill Arrow arrays from event slices.
5. Implement flush task (background Tokio task, 1024-event batching, partition path).
6. Implement `checkpoint()` with atomic rename.
7. Implement `JsonlAuditSink` + rotation + gzip.
8. Implement `CompositeSink`.
9. Implement `audit-verify` binary.
10. Wire `CompositeSink` into `UnifiedBridge::new()` — replace `NoopUnifiedAuditSink`.
11. Coordinate with W10: ensure `BridgeError` path calls `sink.emit()` before returning.
12. Coordinate with W4: ensure each subcrate calls `make_bridge()` with its own `base_path`.

---

*End of spec. Author: W8 / sprint-log-4 / 2026-05-13.*
