# PR-D3B — JsonlAuditSink + CompositeSink + `verify` CLI

**PR-ID:** PR-D3B  
**Sprint:** 5 (S5-W8) / sprint-log-5-6 worker W2  
**Author:** agent-W2 / 2026-05-13  
**Branch target:** `claude/lance-datafusion-integration-gv0BF`  
**Crate target:** `crates/lance-graph-callcenter/src/audit_sink/`  
**Sibling spec:** `.claude/specs/pr-d3a-lance-audit-sink.md` (W1 — LanceAuditSink; read that spec first for the `AuditSink` trait, `AuditError`, Arrow schema, and `LanceAuditSink`)  
**Substrate base:** PR #364 merged 2026-05-13 (`c8176cb`); D-SDR-4 ships `UnifiedAuditEvent` 26-byte canonical_bytes + `AuditMerkleRoot` (FNV-1a u64) + `verify_chain()` in `unified_audit.rs`  

---

## 0. Context and Scope

D-SDR-4 shipped the merkle-chained `UnifiedAuditEvent` type with `canonical_bytes() -> [u8; 26]` and `verify_chain()`. The `NoopUnifiedAuditSink` is the only production sink. This PR adds:

1. **`JsonlAuditSink`** — plain JSONL fallback sink with daily rotation + gzip-on-rotate
2. **`CompositeSink`** — broadcasts writes to N sinks with per-sink failure isolation
3. **`verify` binary** — three subcommands (`verify-jsonl`, `verify-lance`, `cross-verify`) that walk the chain and report integrity

This spec is the second of a two-PR set. PR-D3A (W1 sibling) defines the `AuditSink` trait, `AuditError`, `LanceAuditSink`, and the Arrow/Lance schema. This spec extends that foundation with the JSONL path and the forensic verifier.

**What this spec does NOT define** (deferred to PR-D3A / W1):
- `AuditSink` trait definition and `AuditError` enum
- `LanceAuditSink` internals and `audit_event_schema()`
- `UnifiedAuditEvent::prev_merkle` field extension (D-SDR-4b action item in W1's spec)

---

## 1. JSONL Line Schema

### 1.1 One event per line

Each line is a JSON object terminated by `\n`. No trailing comma. UTF-8 encoding. The schema mirrors the Lance Arrow schema in PR-D3A §4 exactly — same field names, same logical types — to enable cross-format joins with no field renaming.

### 1.2 Canonical field set

```json
{
  "timestamp_us":    "1747180800000000",
  "tenant_id":       42,
  "super_domain":    1,
  "family_id":       7,
  "owl_identity":    "07051c",
  "action":          0,
  "decision":        0,
  "actor_role_hash": "14627333968358193902",
  "prev_merkle":     "12297829382473034410",
  "event_merkle":    "9823479283742938472",
  "payload":         null
}
```

### 1.3 Field naming convention

Field names match the Arrow schema from PR-D3A §4. The canonical mapping:

| Field | Source | Type | JSONL encoding |
|---|---|---|---|
| `timestamp_us` | `ts_unix_ms * 1000` | u64 | **decimal string** (see §1.4) |
| `tenant_id` | `tenant.raw()` | u32 | JSON number (fits u32, safe in JS) |
| `super_domain` | `super_domain.raw()` | u8 | JSON number |
| `family_id` | `owl.family().raw()` | u8 | JSON number |
| `owl_identity` | `owl.to_canonical_bytes()` | [u8; 3] | **lowercase hex string** (see §1.5) |
| `action` | `op.as_u8()` | u8 | JSON number |
| `decision` | `decision.as_u8()` | u8 | JSON number |
| `actor_role_hash` | `actor_role_hash` | u64 | **decimal string** (see §1.4) |
| `prev_merkle` | `prev_root` captured pre-advance | u64 | **decimal string** (see §1.4) |
| `event_merkle` | `merkle_root.raw()` | u64 | **decimal string** (see §1.4) |
| `payload` | reserved | `Option<Vec<u8>>` | `null` for all current events |

### 1.4 u64 fields as decimal strings

JSON `number` is IEEE 754 double (53-bit mantissa). The FNV-1a outputs for `timestamp_us`, `actor_role_hash`, `prev_merkle`, and `event_merkle` regularly exceed 2^53 and would silently lose precision if serialized as JSON numbers. These four fields MUST be serialized as **decimal strings** (e.g., `"14627333968358193902"`), not JSON numbers.

Rationale for decimal over hex: decimal strings parse directly to u64 with `str::parse::<u64>()` in all languages without base-prefix handling; decimal is also what Python's `json.loads` recovers when reading them back as strings and parsing to int.

**Open question (OQ-4):** downstream log consumers (e.g., Splunk, Elasticsearch) may expect numeric fields. If a consuming pipeline objects to decimal strings for the merkle fields, the alternative is hex strings (`"0x884a1b2c3d4e5f60"`) with a documented parse convention. Settle before the first production deployment; the format can change in D-SDR-4b before any on-disk data exists.

### 1.5 `owl_identity` serialization: lowercase hex

`OwlIdentity::to_canonical_bytes()` returns `[u8; 3]` = `[family, slot_lo, slot_hi]` (little-endian slot, per PR #364 Codex P1 fix).

The JSONL field `"owl_identity"` serializes these 3 bytes as a **6-character lowercase hex string** with no `0x` prefix:

```
family=0x07, slot=0x051c  ->  canonical_bytes=[0x07, 0x1c, 0x05]  ->  "071c05"
```

**Rationale:**
- Hex is compact (6 chars vs 12 for decimal-per-byte), unambiguous, and round-trips without precision loss.
- The `verify-jsonl` tool reconstructs 3 bytes via `u8::from_str_radix(&hex[0..2], 16)` etc.
- Base64 was considered but adds padding complexity and is less grep-friendly for forensics.
- The verify tool sanity-checks that `family_id` (separate JSON number) == first byte of `owl_identity` hex.

---

## 2. `JsonlAuditSink`

**Location:** `crates/lance-graph-callcenter/src/audit_sink/jsonl_sink.rs`

### 2.1 Design goals

- Zero dependency on Arrow/Lance — suitable for low-tooling, low-memory environments
- One file per tenant per day (UTC): `<base_path>/audit/<tenant_id>/YYYY-MM-DD.jsonl`
- Gzip-on-rotate: when the UTC day rolls over, compress the previous day's file in a background thread -> `YYYY-MM-DD.jsonl.gz`, then remove the uncompressed original
- POSIX-atomic checkpoint writes (write-to-tmp + rename)
- Implements the `AuditSink` trait from PR-D3A without modifications to that trait

### 2.2 Struct definition

```rust
/// Plain JSONL fallback sink. Thread-safe; emit() is non-blocking.
pub struct JsonlAuditSink {
    base_path:   PathBuf,
    /// In-memory event buffer; drained on each flush() call.
    buffer:      Arc<Mutex<VecDeque<UnifiedAuditEvent>>>,
    /// Tracks last fully-flushed checkpoint root for checkpoint().
    last_root:   Arc<Mutex<MerkleRoot>>,
    /// Tracks the current UTC date; used to detect day rotation in flush().
    current_day: Arc<Mutex<chrono::NaiveDate>>,
}
```

### 2.3 File layout

```
<base_path>/audit/
  <tenant_id>/
    2026-05-13.jsonl          # current day - append-only, uncompressed
    2026-05-12.jsonl.gz       # prior day - rotated + compressed
    _checkpoint.json          # last flushed merkle root (atomic rename write)
    _checkpoint.json.tmp      # write target before rename
```

One file per `(tenant_id, date)`. Cross-tenant events are demultiplexed at flush time: the flush task groups events by `tenant_id`, opens the appropriate file per group, and appends.

### 2.4 `emit()` — non-blocking buffer push

```rust
impl AuditSink for JsonlAuditSink {
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
        let mut buf = self.buffer.lock()
            .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?;
        if buf.len() >= JSONL_BUFFER_CAPACITY {      // default: 4096 events
            return Err(AuditError::ChannelFull(format!(
                "jsonl buffer at {} capacity", JSONL_BUFFER_CAPACITY
            )));
        }
        buf.push_back(event);
        Ok(())
    }
    // flush() and checkpoint() below
}
```

Hot-path cost: Mutex lock + `VecDeque::push_back`. No I/O. Target: < 10 us p99.

### 2.5 `flush()` — day-aware append + rotation

```rust
fn flush(&self) -> Result<MerkleRoot, AuditError> {
    // 1. Drain buffer under lock.
    let events: Vec<UnifiedAuditEvent> = {
        let mut buf = self.buffer.lock()
            .map_err(|_| AuditError::ChannelFull("lock poisoned".into()))?;
        buf.drain(..).collect()
    };
    if events.is_empty() {
        return Ok(*self.last_root.lock().unwrap());
    }
    // 2. Group by (tenant_id, UTC date from timestamp_us).
    let grouped = group_by_tenant_date(&events);
    let today = Utc::now().date_naive();

    // 3. For each group, rotate if needed, then append.
    for ((tenant_id, date), group_events) in &grouped {
        let dir = self.base_path.join("audit").join(tenant_id.to_string());
        std::fs::create_dir_all(&dir)?;
        if *date < today {
            rotate_if_uncompressed(&dir, *date)?;  // gzip in background thread
        }
        let file_path = dir.join(format!("{}.jsonl", date.format("%Y-%m-%d")));
        let mut file = OpenOptions::new().create(true).append(true).open(&file_path)?;
        for ev in group_events {
            let line = serialize_event(ev)?;
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
        }
    }
    // 4. Update last_root from final event.
    let final_root = events.last().map(|e| e.merkle_root.raw()).unwrap_or(0);
    *self.last_root.lock().unwrap() = final_root;
    Ok(final_root)
}
```

`rotate_if_uncompressed(dir, date)`: if `YYYY-MM-DD.jsonl` exists and `.gz` does not, spawns a `std::thread` that gzip-compresses the file via `flate2::GzEncoder` then removes the original. Fire-and-forget; errors are logged via `log::warn!` but do not propagate.

### 2.6 `serialize_event()` — JSONL line production

```rust
fn serialize_event(ev: &UnifiedAuditEvent) -> Result<String, AuditError> {
    let owl_bytes = ev.owl.to_canonical_bytes();
    let owl_hex = format!("{:02x}{:02x}{:02x}",
        owl_bytes[0], owl_bytes[1], owl_bytes[2]);
    let prev = ev.prev_merkle.raw();  // requires D-SDR-4b UnifiedAuditEvent extension
    let json = serde_json::json!({
        "timestamp_us":    ev.ts_unix_ms.saturating_mul(1000).to_string(),
        "tenant_id":       ev.tenant.raw(),
        "super_domain":    ev.super_domain.raw(),
        "family_id":       owl_bytes[0],
        "owl_identity":    owl_hex,
        "action":          ev.op.as_u8(),
        "decision":        ev.decision.as_u8(),
        "actor_role_hash": ev.actor_role_hash.to_string(),
        "prev_merkle":     prev.to_string(),
        "event_merkle":    ev.merkle_root.raw().to_string(),
        "payload":         serde_json::Value::Null,
    });
    serde_json::to_string(&json).map_err(|e| AuditError::Serialize(e.to_string()))
}
```

### 2.7 `checkpoint()` — atomic POSIX write

```rust
fn checkpoint(&self) -> Result<(), AuditError> {
    let root = *self.last_root.lock().unwrap();
    let tmp  = self.base_path.join("audit/_checkpoint.json.tmp");
    let live = self.base_path.join("audit/_checkpoint.json");
    let json = serde_json::json!({
        "last_merkle_root": root.to_string(),
        "timestamp_us":     now_unix_us().to_string(),
        "salt_version":     0u8,   // reserved for OQ-1 (salt rotation)
    });
    std::fs::write(&tmp, serde_json::to_string(&json)
        .map_err(|e| AuditError::Serialize(e.to_string()))?)?;
    std::fs::rename(tmp, live)?;  // atomic on POSIX
    Ok(())
}
```

---

## 3. `CompositeSink`

**Location:** `crates/lance-graph-callcenter/src/audit_sink/composite.rs`

### 3.1 Purpose

Broadcasts one `emit()` call to N child sinks. Production canonical configuration:

```rust
CompositeSink::new(vec![
    Box::new(LanceAuditSink::new(&base_path)?),    // primary: columnar, indexed
    Box::new(JsonlAuditSink::new(&base_path)?),    // fallback: plain text
], FanoutMode::BestEffort)
```

`BestEffort` is the production default: if Lance is temporarily unreachable, JSONL still captures events and the chain remains auditable.

### 3.2 Struct and `FanoutMode`

```rust
pub struct CompositeSink {
    sinks: Vec<Box<dyn AuditSink>>,
    mode:  FanoutMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FanoutMode {
    /// First error aborts; remaining sinks NOT called. For test environments.
    FailFast,
    /// All sinks always called. Collects first error; returns Ok if all pass.
    /// Production default.
    BestEffort,
}
```

### 3.3 Ordering guarantees

`emit()` calls child sinks in **declaration order** (index 0 first). Lance is declared first in the canonical config because it has the stronger durability guarantee. In `BestEffort` mode, a Lance failure does NOT skip the JSONL sink.

### 3.4 `AuditSink` implementation

```rust
impl AuditSink for CompositeSink {
    fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
        match self.mode {
            FanoutMode::FailFast => {
                for sink in &self.sinks { sink.emit(event)?; }
                Ok(())
            }
            FanoutMode::BestEffort => {
                let mut first_err: Option<AuditError> = None;
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
        let mut last_root: MerkleRoot = 0;
        let mut first_err: Option<AuditError> = None;
        for sink in &self.sinks {
            match sink.flush() {
                Ok(root) => last_root = root,
                Err(e) => {
                    if self.mode == FanoutMode::FailFast { return Err(e); }
                    if first_err.is_none() { first_err = Some(e); }
                }
            }
        }
        first_err.map_or(Ok(last_root), Err)
    }

    fn checkpoint(&self) -> Result<(), AuditError> {
        // Always best-effort for checkpoint: one sink failing must not suppress others.
        for sink in &self.sinks {
            if let Err(e) = sink.checkpoint() {
                log::warn!("CompositeSink::checkpoint() sink error (ignored): {e}");
            }
        }
        Ok(())
    }
}
```

### 3.5 Per-sink failure isolation

In `BestEffort` mode:
- A dead Lance cluster does NOT suppress JSONL writes.
- A full JSONL disk does NOT suppress Lance writes.
- `emit()` returns the first error for logging, but MUST NOT propagate as `AccessDecision::Deny` in the authorize() hot path. The bridge logs a warning and continues (per W10 spec, TD-SDR-BRIDGE-ERR-AUDIT-1).

---

## 4. `verify` CLI

**Binary:** `crates/lance-graph-callcenter/src/bin/audit_verify.rs`  
**Cargo command:** `cargo run -p lance-graph-callcenter --bin audit-verify -- <subcommand> [OPTIONS]`

### 4.1 Subcommands

| Subcommand | Input | Purpose |
|---|---|---|
| `verify-jsonl` | JSONL file(s) | Walk JSONL audit log, recompute chain, report first break |
| `verify-lance` | Lance dataset | Walk Lance columnar data, recompute chain, report first break |
| `cross-verify` | JSONL + Lance | Compare two representations for event-by-event agreement |

### 4.2 Global options (all subcommands)

```
--since <DATE>       ISO 8601 date YYYY-MM-DD; scan from this day [required]
--until <DATE>       ISO 8601 date; stop scanning [default: today UTC]
--tenant <ID>        Restrict to one tenant_id u32 [default: all]
--seed-root <HEX>    Override checkpoint root (hex u64, no 0x prefix)
                     [default: read _checkpoint.json]
--verbose            Print each row: computed root, stored root, MATCH/FAIL
--base-path <PATH>   Audit base path [default: $AUDIT_BASE_PATH env var]
```

### 4.3 `verify-jsonl` subcommand

```
audit-verify verify-jsonl [GLOBAL OPTIONS]
  --file <PATH>      Explicit JSONL file (overrides --base-path discovery)
```

**Algorithm:**

```
1. Determine seed root:
   a. --seed-root hex present: parse hex -> u64.
   b. Else read <base_path>/audit/_checkpoint.json -> "last_merkle_root" decimal string -> u64.
   c. Else default: AuditMerkleRoot::GENESIS.0 (0xa5a5_a5a5_a5a5_a5a5).

2. For each JSONL file matching (tenant_id, date range), in chronological order:
   a. Read lines in file order (emission order = timestamp_us ascending).
   b. For each line:
      i.  Parse JSON -> extract all fields.
      ii. Parse "owl_identity" hex -> [u8; 3].
      iii. Parse decimal strings -> u64 for timestamp_us, actor_role_hash, prev_merkle, event_merkle.
      iv. Reconstruct canonical_bytes() from fields (26 bytes, same layout as unified_audit.rs).
      v.  salt = SuperDomainRegistry::merkle_salt(super_domain_value).
      vi. expected = AuditMerkleRoot::chain(prev_root, salt, &canonical_bytes).
      vii. If expected.raw() != event_merkle_field: report BREAK.
      viii. prev_root = expected (advance regardless of match to show downstream breaks).

3. Print summary.
```

**Canonical bytes reconstruction from JSONL:**

```rust
fn jsonl_to_canonical_bytes(r: &JsonlRecord) -> Result<[u8; 26], VerifyError> {
    let mut out = [0u8; 26];
    let ts_ms = r.timestamp_us_str.parse::<u64>()? / 1000;
    out[0..8].copy_from_slice(&ts_ms.to_le_bytes());
    out[8..12].copy_from_slice(&(r.tenant_id as u32).to_le_bytes());
    out[12] = r.super_domain;
    let owl = parse_owl_hex(&r.owl_identity)?;  // 3 bytes from hex string
    out[13..16].copy_from_slice(&owl);
    out[16] = r.action;
    out[17] = r.decision;
    let role_hash = r.actor_role_hash_str.parse::<u64>()?;
    out[18..26].copy_from_slice(&role_hash.to_le_bytes());
    Ok(out)
}
```

### 4.4 `verify-lance` subcommand

```
audit-verify verify-lance [GLOBAL OPTIONS]
```

**Algorithm:** Open Lance dataset at `<base_path>/audit/`. Build DataFusion scan with partition pushdown (tenant_id, date range). ORDER BY (tenant_id, timestamp_us) ASC. For each row: reconstruct canonical_bytes from Arrow columns, read `prev_merkle` column for prior root, retrieve salt from `SuperDomainRegistry::merkle_salt(super_domain)`, recompute and compare. Report breaks with row index and field values.

Lance provides native timestamp_us predicate pushdown via UInt64 partition columns (Arrow schema from PR-D3A §4).

### 4.5 `cross-verify` subcommand

```
audit-verify cross-verify [GLOBAL OPTIONS]
  --jsonl-path <PATH>   JSONL base path (may differ from Lance base path)
  --lance-path <PATH>   Lance base path
```

**Algorithm:**

```
1. Collect JSONL events -> Vec<EventRecord> sorted (tenant, ts_us).
2. Collect Lance events -> Vec<EventRecord> sorted (tenant, ts_us).
3. Zip-compare event_merkle at each position.
4. Report:
   - N total JSONL events, M total Lance events.
   - K events in both with matching merkle (OK).
   - P JSONL-only events (Lance write failed during incident).
   - Q Lance-only events (JSONL write failed).
   - R events in both with mismatched merkle (data corruption).
```

`cross-verify` is the primary forensic tool after a partial sink failure (e.g., Lance was unreachable for 30 minutes while JSONL kept writing).

### 4.6 Output format

**Per-event (--verbose):**

```
[OK]   tenant=42 ts=2026-05-13T14:23:01.000Z owl=07051c op=0 dec=0 expected=9823479283742938472 got=9823479283742938472
[FAIL] tenant=42 ts=2026-05-13T14:23:05.000Z owl=07051c op=1 dec=1 expected=1234567890123456789 got=9999999999999999999  <- CHAIN BREAK
```

**Summary (always printed):**

```
verify-jsonl: 1024 events checked (tenant=42, 2026-05-13..2026-05-13)
  OK:    1023
  BREAK: 1  (first at row 512, ts=2026-05-13T14:23:05.000Z)
  final root: 9823479283742938472
```

### 4.7 Exit codes

| Code | Meaning |
|---|---|
| `0` | All events verified, chain intact |
| `1` | One or more chain breaks detected (details to stdout) |
| `2` | I/O error, schema mismatch, or missing checkpoint (details to stderr) |
| `3` | cross-verify only: JSONL and Lance event sets diverge (not just merkle breaks) |

---

## 5. Merkle Integrity Check Algorithm

### 5.1 Chain walk

```
prev_root = seed_root   (GENESIS or checkpoint)
for each event E in emission order (ascending timestamp_us per tenant):
  salt     = SuperDomainRegistry::merkle_salt(E.super_domain)
  canonical = E.canonical_bytes()    // 26 bytes, merkle_root excluded
  expected  = AuditMerkleRoot::chain(prev_root, salt, &canonical)
  if expected != E.merkle_root:
    report BREAK at E
  prev_root = expected               // advance regardless (shows downstream breaks)
```

This mirrors `verify_chain()` in `unified_audit.rs` (PR #364) but operates on deserialized storage rows rather than in-memory structs.

### 5.2 Tamper detection properties

The FNV-1a chaining detects:
- **Field mutation:** Any change to ts, tenant, owl, op, decision, or actor_role_hash breaks the chain at that event.
- **Event insertion:** A forged event shifts all subsequent `prev_merkle` values; breaks at the insertion point.
- **Event deletion:** Deletes cause a `prev_merkle` gap; break at the next event.
- **Root forgery:** Changing `event_merkle` in storage without changing `canonical_bytes()` — recomputed root does not match stored root.
- **Cross-domain correlation prevention:** Per-super-domain salt (from `AuditChain::salt`) ensures Healthcare and SMB chains are unlinkable (§13.4 from super-domain-rbac-tenancy-v1.md).

### 5.3 `prev_merkle` field sourcing

The D-SDR-4 `UnifiedAuditEvent` does not yet carry `prev_merkle`. Per PR-D3A §4 and td-sdr-audit-persist.md §4.2, `AuditChain::advance()` must be extended:

```rust
pub fn advance(&mut self, mut event: UnifiedAuditEvent) -> UnifiedAuditEvent {
    event.prev_merkle = self.last_root;            // capture BEFORE chaining
    let new_root = AuditMerkleRoot::chain(
        self.last_root, self.salt, &event.canonical_bytes());
    event.merkle_root = new_root;
    self.last_root = new_root;
    event
}
```

`prev_merkle` MUST NOT appear in `canonical_bytes()` — it is the prior chain output, not an input, and including it would create a circular dependency.

**Sequential fallback:** The verify tool can walk without `prev_merkle` stored per-event by treating `event_merkle[i-1]` as `prev_root` for `event[i]`. The stored `prev_merkle` column adds redundancy for single-event spot-checks without scanning from genesis.

---

## 6. Performance Envelope

| Operation | Target | Mechanism |
|---|---|---|
| `emit()` p99 (JsonlAuditSink) | < 10 us | Mutex + VecDeque push; no I/O |
| `emit()` p99 (CompositeSink, 2 sinks) | < 20 us | Two sequential lock+push; no I/O |
| `flush()` per 1024 events (JSONL) | < 100 ms | Sequential file appends; no Arrow overhead |
| `flush()` per 1024 events (Lance) | < 50 ms | Single RecordBatch write (PR-D3A) |
| `checkpoint()` (JSONL) | < 5 ms | One JSON file + atomic rename |
| `verify-jsonl` scan | < 10 s / 1M events | Line-by-line parse + FNV-1a; I/O bound |
| `verify-lance` scan | < 5 s / 1M events | DataFusion pushdown + columnar read |
| `cross-verify` | < 15 s / 1M events | Two scans + zip comparison |

### 6.1 Backpressure

**Buffer capacity:** `JSONL_BUFFER_CAPACITY = 4096` events. When full, `emit()` returns `AuditError::ChannelFull`. In `CompositeSink::BestEffort` mode, a full JSONL buffer does NOT block Lance writes.

**Flush wake conditions:**
1. Buffer count >= 1024 (batch threshold notification)
2. Periodic timer tick every 1 second (ensures events reach disk at low throughput)

At steady state (>1024 events/second), buffer stays below capacity. At low throughput, events reach disk within 1 second of emit.

**Backpressure isolation:** `AuditError::ChannelFull` returned from `emit()` MUST NOT propagate as `AccessDecision::Deny`. Audit-sink backpressure must never become an availability attack vector.

---

## 7. DELTA vs. Adjacent Specs

### 7.1 vs. `td-sdr-audit-persist.md` (sprint-4 W8 foundational sketch)

| Topic | Sprint-4 sketch | This spec (PR-D3B) |
|---|---|---|
| `owl_identity` serialization | Not specified | Lowercase 6-char hex, §1.5 |
| u64 JSON precision | OQ-4 left open | Decimal strings mandated; OQ-4 status documented |
| verify subcommands | Single binary `--since` | Three subcommands: verify-jsonl / verify-lance / cross-verify |
| `prev_merkle` sourcing | "extend UnifiedAuditEvent" action item | Sequential fallback path + `advance()` change specified, §5.3 |
| Exit codes | 0/1/2 | 0/1/2/3 (exit 3 for cross-verify divergence) |
| Backpressure | Mentioned | Per-sink capacity, flush wake conditions, BestEffort isolation, §6.1 |

### 7.2 vs. PR-D3A (`pr-d3a-lance-audit-sink.md`, sibling W1)

PR-D3A owns: `AuditSink` trait, `AuditError`, `LanceAuditSink`, Arrow schema, Tokio batch flush.

This spec (PR-D3B) owns: `JsonlAuditSink`, `CompositeSink`, `verify` binary. The JSONL schema (§1) intentionally mirrors the Arrow schema in PR-D3A §4 so cross-format joins require no field renaming.

**Not redefined here:** the `AuditSink` trait from PR-D3A. Both sinks implement it without modification.

### 7.3 vs. `anatomy-realtime-v1.md` §step-8

The proof-of-vision plan (Sprint-2 W12) cites "LanceAuditSink emits trail" at demo step 8 (radiologist write -> RBAC gates -> audit). This spec delivers the JSONL fallback and the `cross-verify` subcommand a compliance auditor runs to prove the step-8 trail is intact when Lance was temporarily unreachable during the demo.

---

## 8. File Manifest

```
crates/lance-graph-callcenter/src/audit_sink/
  mod.rs          -- AuditSink trait, AuditError, MerkleRoot alias (owned by PR-D3A)
  jsonl_sink.rs   -- JsonlAuditSink, serialize_event(), rotate_if_uncompressed()
  composite.rs    -- CompositeSink, FanoutMode

crates/lance-graph-callcenter/src/bin/
  audit_verify.rs -- CLI: verify-jsonl / verify-lance / cross-verify subcommands

Cargo.toml additions (this PR):
  serde_json = "1"
  chrono     = { version = "0.4", features = ["serde"] }
  flate2     = "1"
  clap       = { version = "4", features = ["derive"] }
  log        = "0.4"
  # tokio and thiserror assumed present from PR-D3A
```

---

## 9. Open Questions

**OQ-1 (salt rotation):** Checkpoint JSON reserves `"salt_version": 0`. If per-quarter rotation is required, `SuperDomainRegistry::merkle_salt()` becomes versioned and `--salt-version <N>` is added to the verify CLI. Non-breaking given the reserved field.

**OQ-4 (u64 precision):** Decimal strings mandated (§1.4). Confirm against first consuming pipeline (Splunk / Elasticsearch) before locking format. Non-breaking through D-SDR-4b since no on-disk data exists yet.

**OQ-5 (multi-process write safety):** `JsonlAuditSink` assumes single writer per `(tenant_id, date)` file. Multi-process: use per-process file suffix or OS-level append-mode lock. Matches current single-process MedCare-rs / smb-office-rs deployment model; multi-process is a separate TD item.

---

## 10. Implementation Order

1. Land PR-D3A first (AuditSink trait + LanceAuditSink).
2. Extend `UnifiedAuditEvent` with `prev_merkle` field + update `AuditChain::advance()` per §5.3 (coordinate with W1 — shared change).
3. Implement `jsonl_sink.rs`: `JsonlAuditSink::new()`, `emit()`, `flush()`, `checkpoint()`, `serialize_event()`.
4. Implement `rotate_if_uncompressed()` — background gzip thread.
5. Implement `composite.rs`: `CompositeSink`, `FanoutMode`.
6. Implement `audit_verify.rs`: `verify-jsonl` subcommand first (no DataFusion dep).
7. Add `verify-lance` subcommand (DataFusion scan).
8. Add `cross-verify` subcommand.
9. Wire `CompositeSink` into `UnifiedBridge::new()` replacing `NoopUnifiedAuditSink`.
10. Test gates: JSONL round-trip (emit -> flush -> parse -> canonical_bytes -> verify), CompositeSink BestEffort isolation, `verify-jsonl` tamper detection, `cross-verify` agreement on clean run.

---

*End of spec. Author: agent-W2 / sprint-log-5-6 / 2026-05-13.*
