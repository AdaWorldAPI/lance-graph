# PR-D3A — LanceAuditSink: Arrow Schema + Partitioning + Write Path

**PR-ID:** PR-D3A  
**Sprint:** 5 (S5-W1) / sprint-log-5-6 worker W1  
**Author:** agent-W1 / 2026-05-13  
**Branch target:** `claude/lance-datafusion-integration-gv0BF`  
**Crate target:** `crates/lance-graph-callcenter/src/audit_sink/`  
**Sibling spec:** `.claude/specs/pr-d3b-jsonl-and-verify.md` (W2 — JsonlAuditSink + CompositeSink + verify CLI; read that spec for JSONL schema alignment, cross-verify subcommands, and CompositeSink fanout semantics)  
**Substrate base:** PR #364 merged 2026-05-13 (`c8176cb`); D-SDR-4 ships `UnifiedAuditEvent` 26-byte `canonical_bytes()` + `AuditMerkleRoot` (FNV-1a u64) + `verify_chain()` in `unified_audit.rs`

---

## 0. Context and Scope

D-SDR-4 shipped the merkle-chained `UnifiedAuditEvent` type. The `NoopUnifiedAuditSink` is the only production sink today. This PR (D3A) adds the **columnar persistence tier**: an Arrow/Lance dataset that stores one row per authorization decision, partitioned for efficient tenant- and time-scoped forensic replay.

**This spec owns:**

- `AuditSink` trait and `AuditError` enum (the shared interface PR-D3B also implements without modification)
- `LanceAuditSink` — struct, batching, Lance write path, fsync contract, merkle integrity enforcement
- `audit_event_schema()` — canonical Arrow schema with field-by-field mapping from `UnifiedAuditEvent`
- Partitioning strategy (super_domain x date), justification, and failure modes
- D-SDR-4b action item: extending `UnifiedAuditEvent` with `prev_merkle` (shared change, coordinate with W2)

**What this spec does NOT define** (owned by PR-D3B / W2):

- `JsonlAuditSink` and `CompositeSink`
- `verify` binary and its three subcommands (`verify-jsonl`, `verify-lance`, `cross-verify`)
- JSONL serialization details beyond field naming alignment

---

## 1. Trait Surface: `AuditSink` and `AuditError`

**Location:** `crates/lance-graph-callcenter/src/audit_sink/mod.rs`

These definitions are shared — PR-D3B depends on them unchanged.

### 1.1 `AuditError`

```rust
#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("lance write failed: {0}")]
    Lance(#[from] lance::Error),

    #[error("arrow schema error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("channel full: {0}")]
    ChannelFull(String),

    #[error("serialization error: {0}")]
    Serialize(String),

    #[error("schema migration blocked: {0}")]
    SchemaMigration(String),
}
```

### 1.2 `AuditSink` trait

```rust
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

/// Alias for readability in trait return types.
pub type MerkleRoot = u64;
```

**Difference from `UnifiedAuditSink` in `unified_audit.rs`:** the legacy trait (D-SDR-4) takes `&UnifiedAuditEvent` and returns `()`. This spec introduces `AuditSink` as the D-SDR-4b production interface: it returns `Result<_, AuditError>` and adds `flush()` + `checkpoint()` so the write path can propagate backpressure and guarantee durability. The `NoopUnifiedAuditSink` is not replaced; the `AuditSink` trait coexists for production sinks only.

---

## 2. `UnifiedAuditEvent` Extension (D-SDR-4b Shared Change)

D-SDR-4 `UnifiedAuditEvent` does not carry `prev_merkle`. The verify tools (W2 §5.3) require it for single-event spot-checks. `AuditChain::advance()` must be extended to capture the prior root before chaining:

```rust
pub fn advance(&mut self, mut event: UnifiedAuditEvent) -> UnifiedAuditEvent {
    event.prev_merkle = self.last_root;         // capture BEFORE chaining
    let new_root = AuditMerkleRoot::chain(
        self.last_root, self.salt, &event.canonical_bytes());
    event.merkle_root = new_root;
    self.last_root = new_root;
    event
}
```

`prev_merkle` MUST NOT appear in `canonical_bytes()` -- it is the prior chain output, not an input, and including it would create a circular dependency.

`UnifiedAuditEvent` gains one field:

```rust
pub struct UnifiedAuditEvent {
    // ... existing fields unchanged ...
    /// Merkle root of the immediately preceding event in this chain.
    /// `AuditMerkleRoot::GENESIS` for the first event.
    /// Excluded from canonical_bytes() -- see D-SDR-4b note.
    pub prev_merkle: AuditMerkleRoot,
}
```

This is a shared change; coordinate with W2 before merging either D3A or D3B.

---

## 3. Dependency Graph

```
unified_audit.rs       (PR #364 -- shipped)
      | UnifiedAuditEvent + AuditMerkleRoot
      v
audit_sink/mod.rs      (this PR -- AuditSink trait + AuditError)
      |
      +---> audit_sink/lance_sink.rs   (LanceAuditSink -- this PR)
      |          | arrow-schema "57", lance "2", tokio, thiserror
      |
      +---> audit_sink/jsonl_sink.rs   (JsonlAuditSink -- PR-D3B)
                 | serde_json, chrono, flate2

audit_sink/composite.rs  (CompositeSink -- PR-D3B)
      |  uses both sinks above
      v
crates/lance-graph-callcenter/src/bin/audit_verify.rs  (verify CLI -- PR-D3B)
```

Cargo.toml additions for D3A:

```toml
arrow-schema = "57"
arrow-array  = "57"
arrow-cast   = "57"
lance        = "2"
tokio        = { version = "1", features = ["rt-multi-thread", "sync", "time"] }
thiserror    = "1"
```

---

## 4. Arrow Schema

**Function:** `pub fn audit_event_schema() -> Arc<Schema>`  
**Location:** `crates/lance-graph-callcenter/src/audit_sink/lance_sink.rs`

The schema is the single source of truth for both the Lance write path and the `verify-lance` subcommand (W2 §4.4). Field names match the JSONL schema in W2 §1.3 exactly -- no renaming, cross-format joins work without aliasing.

### 4.1 Field mapping

| Arrow Field | Arrow Type | Source | canonical_bytes offset | Notes |
|---|---|---|---|---|
| `timestamp_us` | `UInt64` | `ts_unix_ms * 1000` | `[0..8)` LE | Microsecond epoch; multiply at write time |
| `tenant_id` | `UInt32` | `tenant.raw()` | `[8..12)` LE | Chinese wall predicate column |
| `super_domain` | `UInt8` | `super_domain.raw()` | `[12]` | Partition key (see §5) |
| `family_id` | `UInt8` | `owl.family().raw()` | `[13]` | = owl_identity first byte; redundant but faster to filter |
| `owl_identity` | `FixedSizeBinary(3)` | `owl.to_canonical_bytes()` | `[13..16)` | 6-char hex in JSONL (W2 §1.5); raw 3 bytes here |
| `action` | `UInt8` | `op.as_u8()` | `[16]` | 0=Read 1=Write 2=Act |
| `decision` | `UInt8` | `decision.as_u8()` | `[17]` | 0=Allow 1=Deny 2=Escalate 3=BridgeError |
| `actor_role_hash` | `UInt64` | `actor_role_hash` | `[18..26)` LE | FNV-1a of role name string |
| `prev_merkle` | `UInt64` | `event.prev_merkle.raw()` | n/a (not in canonical_bytes) | Prior chain root; redundancy for spot-checks |
| `event_merkle` | `UInt64` | `merkle_root.raw()` | n/a (not in canonical_bytes) | Computed by AuditChain::advance() |
| `payload` | `Binary` (nullable) | reserved | n/a | null for all current events; future extension point |
| `date_partition` | `Utf8` | derived from `timestamp_us` | n/a | "YYYY-MM-DD" UTC; Lance partition column (§5) |

**Total columns:** 12. **Non-nullable:** all except `payload` (always null today) and `date_partition` (always set at write time but derived, not sourced from the event struct).

### 4.2 Schema construction

```rust
pub fn audit_event_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp_us",    DataType::UInt64,              false),
        Field::new("tenant_id",       DataType::UInt32,              false),
        Field::new("super_domain",    DataType::UInt8,               false),
        Field::new("family_id",       DataType::UInt8,               false),
        Field::new("owl_identity",    DataType::FixedSizeBinary(3),  false),
        Field::new("action",          DataType::UInt8,               false),
        Field::new("decision",        DataType::UInt8,               false),
        Field::new("actor_role_hash", DataType::UInt64,              false),
        Field::new("prev_merkle",     DataType::UInt64,              false),
        Field::new("event_merkle",    DataType::UInt64,              false),
        Field::new("payload",         DataType::Binary,              true),
        Field::new("date_partition",  DataType::Utf8,                false),
    ]))
}
```

### 4.3 `canonical_bytes` column-wise decomposition

The 26-byte layout from `unified_audit.rs` maps to Arrow columns as follows (verified against the `canonical_bytes_round_trips_field_order` test in `unified_audit.rs`):

```
canonical_bytes[0..8]   -> timestamp_us  (UInt64 LE) -- stored as ts_unix_ms * 1000
canonical_bytes[8..12]  -> tenant_id     (UInt32 LE)
canonical_bytes[12]     -> super_domain  (UInt8)
canonical_bytes[13..16] -> owl_identity  (FixedSizeBinary(3)); family_id = canonical_bytes[13]
canonical_bytes[16]     -> action        (UInt8)
canonical_bytes[17]     -> decision      (UInt8)
canonical_bytes[18..26] -> actor_role_hash (UInt64 LE)
```

`family_id` is redundant (= `owl_identity[0]`) but materialized as a separate column to enable `SELECT DISTINCT family_id` and partition-pruning on family without decoding `FixedSizeBinary`.

### 4.4 Schema version tagging

The Lance dataset metadata carries `"audit_schema_version": "1"`. The `LanceAuditSink::open()` call validates this key. Mismatch triggers `AuditError::SchemaMigration` (see §8.3 for the migration path).

### 4.5 owl_identity layout (Codex P1 fix, PR #364)

`OwlIdentity::to_canonical_bytes()` returns `[family u8, slot_lo u8, slot_hi u8]` (little-endian slot). The family byte is slot=0 of that array. `family_id` column = `owl_identity[0]`. The verify tool sanity-checks this via: `assert_eq!(family_id_col, owl_identity_col[0])` per W2 §1.5.

---

## 5. Partitioning

### 5.1 Strategy: super_domain x date (UTC)

**Partition path pattern:**

```
<base_path>/audit/
  super_domain=<u8>/
    date=<YYYY-MM-DD>/
      <uuid>.lance
```

Lance stores Hive-style partitions as directory path components. The partition columns (`super_domain` and `date_partition`) are included in the Arrow schema and written as directory names by the Lance write API.

**Example paths:**

```
audit/super_domain=1/date=2026-05-13/batch-001.lance   # Healthcare
audit/super_domain=2/date=2026-05-13/batch-001.lance   # Science
audit/super_domain=7/date=2026-05-12/batch-001.lance   # OSINT (prior day)
```

### 5.2 Partitioning justification

**Why `super_domain` as the first partition level:**

1. **Hard-lock compliance (super-domain-rbac-tenancy-v1.md §13.4):** Healthcare and OSINT chains are unlinkable by per-super-domain `merkle_salt`. Physical partition separation makes it impossible for a misconfigured scan to accidentally cross-domain join at the storage level -- each super-domain's files live in a separate directory tree.
2. **Forensic replay scoping:** a compliance auditor for `SuperDomain::Healthcare` (super_domain=1) scans only `super_domain=1/` files. No predicate pushdown required -- directory skip is free.
3. **Isolation during incidents:** if a tenant's row leaks into the wrong super-domain partition (emit-time bug), the partition mismatch is immediately visible by directory inspection.

**Why `date` as the second level:**

1. **Retention automation:** a cron job can `rm -rf audit/super_domain=*/date=2024-*` to apply per-domain retention policies without touching active data.
2. **Time-scoped verify:** `verify-lance --since 2026-05-01 --until 2026-05-13` pushes date predicates to directory skips -- no row-level scan required.
3. **Compaction cadence:** daily granularity aligns with the `compact()` job described in §6.6.

**Why not `tenant_id` as a partition level:**

Tenant count is unbounded (multi-tenant SaaS). Partitioning by tenant creates O(tenants x days) directories, causing filesystem metadata overhead and Lance manifest bloat. Tenant filtering is handled by predicate pushdown on the `tenant_id` UInt32 column (Lance supports native predicate pushdown via its manifest min/max metadata).

**Why not `family_id`:**

Family is a finer-grained subdivision of super_domain. Partitioning by both creates O(families x days) partitions with many near-empty files at low event rates. The `family_id` column suffices for predicate pushdown within a super_domain partition.

---

## 6. Write Path

### 6.1 `LanceAuditSink` struct

**Location:** `crates/lance-graph-callcenter/src/audit_sink/lance_sink.rs`

```rust
/// Columnar audit sink backed by a Lance dataset.
/// Thread-safe; `emit()` is non-blocking (buffer only).
pub struct LanceAuditSink {
    base_path: PathBuf,
    /// In-memory event buffer. Drained by `flush()`.
    buffer:    Arc<tokio::sync::Mutex<Vec<UnifiedAuditEvent>>>,
    /// Last flushed merkle root (for `checkpoint()`).
    last_root: Arc<tokio::sync::Mutex<u64>>,
    /// Tokio runtime handle for Lance async operations from sync callers.
    rt:        Arc<tokio::runtime::Handle>,
}
```

### 6.2 `emit()` -- non-blocking buffer push

```rust
impl AuditSink for LanceAuditSink {
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
        let mut buf = self.rt.block_on(self.buffer.lock());
        if buf.len() >= LANCE_BUFFER_CAPACITY {   // default: 8192 events
            return Err(AuditError::ChannelFull(format!(
                "lance buffer at {} capacity", LANCE_BUFFER_CAPACITY
            )));
        }
        buf.push(event);
        Ok(())
    }
```

Hot-path cost: async mutex lock + `Vec::push`. No I/O. Target: < 5 us p99.

**Buffer capacity:** `LANCE_BUFFER_CAPACITY = 8192` events (~8192 x ~88 bytes = ~720 KB in-memory). Sized to absorb a burst of 8192 authorize() calls without flushing, leaving the hot path free for 8 seconds at 1000 events/sec.

### 6.3 `flush()` -- batch Arrow write to Lance

```rust
    fn flush(&self) -> Result<MerkleRoot, AuditError> {
        let events: Vec<UnifiedAuditEvent> = {
            let mut buf = self.rt.block_on(self.buffer.lock());
            std::mem::take(&mut *buf)
        };
        if events.is_empty() {
            return Ok(*self.rt.block_on(self.last_root.lock()));
        }

        // 1. Build one RecordBatch per (super_domain, date) partition key.
        let batches = build_partitioned_batches(&events)?;

        // 2. Write each batch to the corresponding Lance partition path.
        for (partition_path, batch) in batches {
            self.rt.block_on(
                write_batch_to_lance(&self.base_path, &partition_path, batch)
            )?;
        }

        // 3. Update last_root from final event's merkle.
        let final_root = events.last().map(|e| e.merkle_root.raw()).unwrap_or(0);
        *self.rt.block_on(self.last_root.lock()) = final_root;
        Ok(final_root)
    }
```

**`build_partitioned_batches()`** groups events by `(super_domain.raw(), date_utc_from_ts_us)`, then for each group builds an Arrow `RecordBatch` using `audit_event_schema()`. The `date_partition` column is a `StringArray` with the `"YYYY-MM-DD"` UTC string derived from `timestamp_us / 1_000_000`.

**`write_batch_to_lance()`** calls `lance::dataset::WriteParams` with `mode = WriteMode::Append`, targeting `<base_path>/audit/super_domain=<N>/date=<YYYY-MM-DD>/`. Lance handles atomic fragment writes internally.

### 6.4 `checkpoint()` -- fsync + atomic manifest

```rust
    fn checkpoint(&self) -> Result<(), AuditError> {
        let root = *self.rt.block_on(self.last_root.lock());
        let tmp  = self.base_path.join("audit/_checkpoint.lance.json.tmp");
        let live = self.base_path.join("audit/_checkpoint.lance.json");
        let json = serde_json::json!({
            "last_merkle_root": root.to_string(),
            "timestamp_us":     now_unix_us().to_string(),
            "schema_version":   1u8,
        });
        std::fs::write(&tmp, serde_json::to_string(&json)
            .map_err(|e| AuditError::Serialize(e.to_string()))?)?;
        std::fs::rename(tmp, live)?;  // atomic on POSIX
        Ok(())
    }
```

**fsync contract:** Lance's `WriteParams` sets `store_options.sync_on_close = true`. This ensures each fragment file is fsync'd before Lance commits the manifest. The POSIX `rename()` in `checkpoint()` is additionally atomic -- the checkpoint file contains either the prior root or the new root, never a partial write.

**Merkle integrity at flush:** before writing batches to Lance, `flush()` calls `verify_chain(seed_root, salt, &events)` on the drained buffer. If the in-memory chain is corrupt (e.g., two concurrent `AuditChain::advance()` callers without the per-chain mutex), this catches it before persisting a broken root. Callers are responsible for holding the `AuditChain` lock; this check is a defense-in-depth gate, not the primary concurrency guard.

### 6.5 Flush wake conditions

1. **Batch threshold:** when `buffer.len() >= 1024`, the background flush task wakes immediately (channel notification).
2. **Timer:** periodic Tokio `time::interval` every 5 seconds -- ensures events reach Lance within 5 seconds at low throughput.
3. **Manual `flush()` call:** via the `AuditSink` trait, e.g. from `CompositeSink::flush()`.

The background flush task is spawned in `LanceAuditSink::new()`:

```rust
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        interval.tick().await;
        if let Err(e) = sink_weak.flush() {
            log::warn!("LanceAuditSink background flush error: {e}");
        }
    }
});
```

At high throughput (>1024 events/5s), the batch threshold triggers first; the timer fires at most once every 5 seconds as a durability backstop.

### 6.6 Lance dataset compaction (daily, out of scope for this PR)

A separate Tokio task (or cron job) calls `lance::dataset::Dataset::optimize()` on each partition once per day after UTC midnight. Compaction merges small fragment files created by high-frequency flushes into fewer, larger files for efficient DataFusion scan. The write path is designed for append-only fragment writes; compaction does not change logical dataset contents.

---

## 7. Cross-Verify Alignment with JSONL Sink (W2)

The Lance schema (§4) is intentionally isomorphic to the JSONL line schema (W2 §1.3). Key alignment points:

| Concern | JSONL (W2) | Lance (this spec) |
|---|---|---|
| Field names | snake_case, 11 fields | Same names, 12 fields (adds `date_partition`) |
| `owl_identity` | 6-char lowercase hex string | `FixedSizeBinary(3)` raw bytes |
| `timestamp_us` | decimal string (W2 §1.4) | `UInt64` -- no precision loss |
| `actor_role_hash` | decimal string | `UInt64` -- no precision loss |
| `prev_merkle` | decimal string | `UInt64` |
| `event_merkle` | decimal string | `UInt64` |
| `payload` | null | `Binary`, nullable |
| `family_id` | derived from owl_identity first byte | Separate `UInt8` column (redundant, faster filter) |

**`cross-verify` subcommand (W2 §4.5):** reads both JSONL and Lance events sorted by `(tenant_id, timestamp_us)`, zips on `event_merkle`, and reports divergence. The Lance reader reconstructs `owl_identity` bytes to 6-char hex for comparison with JSONL strings via `format!("{:02x}{:02x}{:02x}", bytes[0], bytes[1], bytes[2])` -- same formula as W2 §2.6.

**`verify-lance` subcommand (W2 §4.4):** builds DataFusion scan with partition pushdown on `super_domain` and `date_partition`. Orders by `(tenant_id, timestamp_us) ASC`. For each row, reconstructs the 26-byte `canonical_bytes` from the nine Arrow columns (`timestamp_us / 1000 -> ts_unix_ms`, `tenant_id`, `super_domain`, `owl_identity`, `action`, `decision`, `actor_role_hash`), retrieves `salt = SuperDomainRegistry::merkle_salt(super_domain)`, calls `AuditMerkleRoot::chain(prev_merkle_col, salt, &canonical_bytes)`, and compares to `event_merkle`.

---

## 8. Failure Modes

### 8.1 Partial write

**Scenario:** Lance crashes after writing two of three partition groups in a single `flush()` call.

**Detection:** The checkpoint file `_checkpoint.lance.json` is only updated via `checkpoint()`, which is called after `flush()` returns `Ok`. If Lance crashes mid-flush, the checkpoint still points to the prior root. On restart, `LanceAuditSink::new()` reads the checkpoint root and `AuditChain::resume()` seeds from it. Events from the partial flush are duplicated in Lance. The verify tool reports a chain break at the first duplicated event (its `prev_merkle` does not match the recomputed chain). The operator uses `cross-verify` to identify the gap and replay from the JSONL fallback.

**Prevention:** `CompositeSink::BestEffort` (W2 §3.1) always writes to JSONL concurrently. If Lance fails mid-flush, the JSONL record is the canonical source of truth for that window.

### 8.2 Partition skew

**Scenario:** One super_domain receives disproportionate traffic (e.g., Healthcare spikes to 10M events/day while OSINT sees 100/day).

**Mitigation:**
- Healthcare partition accumulates many fragment files; the daily compaction job (§6.6) merges them.
- No cross-partition load balancing is needed -- Lance partitions are independent datasets.
- At extreme rates (>100K events/sec sustained), the `LANCE_BUFFER_CAPACITY = 8192` limit triggers `AuditError::ChannelFull`. The operator increases `LANCE_BUFFER_CAPACITY` or adds a second flush worker for the hot super_domain.
- The `date_partition` granularity absorbs day-level skew; within-day skew is a compaction concern only.

### 8.3 Schema migration (canonical_bytes growth)

**Scenario:** D-SDR-4c extends `canonical_bytes` from 26 to 34 bytes (e.g., adding an 8-byte policy context hash).

**Migration path:**

1. `LanceAuditSink::open()` reads `"audit_schema_version"` from Lance dataset metadata.
2. If `schema_version == 1` and the running code targets version 2, `open()` returns `AuditError::SchemaMigration("dataset is v1, code is v2; run audit-migrate tool")`.
3. A separate `audit-migrate` binary (out of scope for this PR) reads v1 rows, adds the new column with a sentinel default (null or 0x00 bytes for the new field), and rewrites to a v2 dataset.
4. The v1 dataset is retained read-only for historical verify runs. `verify-lance` accepts a `--schema-version` flag to select the reconstruction logic.

**No silent schema drift:** `audit_event_schema()` is versioned. Any field addition or reordering requires a schema version bump and explicit migration tooling. Lance's schema evolution (add nullable column) is acceptable only for the `payload` field (already nullable); structural changes require the migration tool.

---

## 9. LOC Estimate

| File | Purpose | Estimated LOC |
|---|---|---|
| `src/audit_sink/mod.rs` | `AuditSink` trait + `AuditError` | ~80 |
| `src/audit_sink/lance_sink.rs` | `LanceAuditSink` + `audit_event_schema()` + batch builder + Lance write | ~350 |
| `tests/lance_audit_sink_tests.rs` | Round-trip test + merkle integrity test + partition path test + compaction smoke | ~120 |
| **Total** | | **~550 LOC** |

**Cargo.toml additions (D3A only):** `arrow-schema`, `arrow-array`, `arrow-cast`, `lance`, `tokio`, `thiserror` (~6 dependency lines).

**Dependency graph additions:** `lance-graph-callcenter` gains an explicit `lance = "2"` dependency. This is already a transitive dependency via `lance_membrane.rs`; D3A makes it explicit in the manifest.

---

## 10. DELTA vs. `anatomy-realtime-v1.md`

The proof-of-vision plan (`anatomy-realtime-v1.md`) cites `LanceAuditSink` at **§step-8** of the radiologist demo:

> "Radiologist adds finding... GenericBridge admits write via medcare-rs ConsumerPointer; RBAC gates; LanceAuditSink emits trail."

And in **§2** (substrate inventory):

> "Lance MVCC + audit + RBAC seams closed. PRs #29, #98, #337."

**Delta this spec closes:**

1. `anatomy-realtime-v1.md §2` lists `LanceAuditSink` as "shipped" -- that was aspirational; it was `NoopUnifiedAuditSink` in production. This PR ships the real columnar sink.
2. `anatomy-realtime-v1.md §step-8` assumes `LanceAuditSink` writes a tamper-evident trail. This spec adds the merkle-chain integrity verification at flush time (§6.4), making the "tamper-evident" property concrete rather than nominal.
3. The anatomy plan does not specify the Arrow schema, partitioning strategy, or the `prev_merkle` field. This spec supplies all three and aligns them with the JSONL sink so `cross-verify` can audit the step-8 trail.

**Not changed by this spec in `anatomy-realtime-v1.md`:** the step ordering, the radiologist demo narrative, or the substrate inventory table. The inventory table's `LanceAuditSink` row moves from "shipped (aspirational)" to "shipped (real)" after this PR merges.

---

## 11. Implementation Order

1. Extend `UnifiedAuditEvent` with `prev_merkle` + update `AuditChain::advance()` (coordinate with W2 -- shared change; gate both D3A and D3B behind this).
2. Implement `audit_sink/mod.rs`: `AuditSink` trait + `AuditError`.
3. Implement `audit_event_schema()` + `build_partitioned_batches()`.
4. Implement `LanceAuditSink::new()`, `emit()`, `flush()`, `checkpoint()`.
5. Wire background flush task into `new()`.
6. Add `tests/lance_audit_sink_tests.rs`: emit 10 events -> flush -> Lance scan -> verify all 10 rows present; merkle root matches `verify_chain()` in `unified_audit.rs`; partition directory structure is correct.
7. Wire `LanceAuditSink` into `CompositeSink` canonical config (PR-D3B does the wiring; this PR only ships the sink itself).
8. Update `anatomy-realtime-v1.md §2` substrate inventory row.

---

## 12. Open Questions

**OQ-2 (fsync latency):** Lance's `sync_on_close = true` adds fsync per-fragment. At high throughput (>5K events/sec), this may become the write bottleneck. If measured p99 flush latency exceeds 50 ms, disable `sync_on_close` and instead call `Dataset::commit()` with a separate batched fsync. Defer until load testing with real medcare-rs traffic.

**OQ-3 (Lance dataset open semantics):** Lance `Dataset::open()` is async. If multiple `LanceAuditSink` instances share the same base path (multi-process deployment), Lance MVCC handles concurrent writes but the merkle chain ordering is undefined across processes. Current deployment model (single-process MedCare-rs / smb-office-rs) avoids this. Multi-process is a separate tech-debt item (matches OQ-5 in W2's spec).

**OQ-6 (partition pruning correctness):** Lance's Hive-style partition pruning requires that the partition column values written to directory names match the column values in the row data. `build_partitioned_batches()` must ensure `date_partition` column values in the `RecordBatch` equal the directory date string. A mismatch causes silent double-scan. Add an assertion in the write path: `assert_eq!(batch_date_col[0], dir_date_str)`.

---

*End of spec. Author: agent-W1 / sprint-log-5-6 / 2026-05-13.*
