# TD: SDR Family Hydration
**Sprint:** sprint-log-4  
**Owner:** W9-retry  
**Branch:** `claude/lance-datafusion-integration-gv0BF`  
**Date:** 2026-05-13  
**Status:** DRAFT

---

## Background

`FAMILY_TO_SUPER_DOMAIN` at `crates/lance-graph-callcenter/src/super_domain.rs:315` is a `static [SuperDomain::Unknown; 256]` array that is never written at runtime. `super_domain_for_family()` reads it directly with no hydration path. `UnifiedBridge::new()` has no hydration step; the comment referencing `NamespaceRegistry::seed_defaults()` is stale dead code. The regression test at line 667 documents the bug explicitly: `assert_eq!(events[0].super_domain, SuperDomain::Unknown)`. This spec describes the complete engineering plan to fix the silent mis-classification.

---

## 1. Hydration Source

The canonical hydration source is a TOML file bundled into the binary via `include_str!`:

```
crates/lance-graph-contract/data/family_to_super_domain.toml
```

This file maps integer family IDs (0–254) to `SuperDomain` variant names. It is the **seed** — always present, ships with the binary, requires no network. Example schema:

```toml
# family_to_super_domain.toml — seed mapping, ~256 rows at GA
# family_id (u8) => super_domain variant name (string)
# Unclassified families are omitted; absence implies Unknown.

[[family]]
id = 1
domain = "Healthcare"

[[family]]
id = 2
domain = "Finance"

[[family]]
id = 3
domain = "Legal"

[[family]]
id = 4
domain = "Government"

[[family]]
id = 5
domain = "Education"

[[family]]
id = 6
domain = "Retail"

[[family]]
id = 7
domain = "Osint"

[[family]]
id = 10
domain = "Healthcare"

[[family]]
id = 11
domain = "Finance"
```

Two optional overlay layers sit above the seed, applied in order:

1. **Lance table overlay** — a `lance::Dataset` at a path supplied via `BridgeConfig::lance_overlay_path`. Scanned at startup and on reload. Rows share the same `(id u8, domain String)` schema; they overwrite seed entries.
2. **HTTP overlay** — a JSON endpoint at `BridgeConfig::http_overlay_url`. Fetched with a hard 5-second timeout. On timeout or non-2xx the overlay is skipped silently; the seed (+ any Lance overlay) is used instead.

The TOML seed is parsed with `toml::from_str` at startup inside `hydration::load_seed()`. The Lance and HTTP overlays are parsed in `hydration::load_overlays()`. Both functions return `Result<HashMap<u8, SuperDomain>, HydrationError>`.

---

## 2. Bootstrap Sequence

`UnifiedBridge::new_hydrated(config: BridgeConfig) -> Result<(UnifiedBridge, BridgeHandle), HydrationError>` replaces bare `new()` as the entry point for production code. The sequence is:

1. **Parse seed** — call `hydration::load_seed()`. On parse failure return `Err(HydrationError::SeedParseFailed)`. This is a hard failure; the binary must exit 1.
2. **Apply Lance overlay** — if `config.lance_overlay_path` is `Some`, open the dataset and merge. Failures are logged as WARN and skipped.
3. **Apply HTTP overlay** — if `config.http_overlay_url` is `Some`, issue a GET with a **5-second timeout** via `reqwest::blocking`. Failures (timeout, DNS, non-2xx) are logged as WARN and skipped. This is intentional: network availability must not gate startup.
4. **Sanity gate** — call `hydration::sanity_gate(&merged_map)`. Returns `Err(HydrationError::InsufficientDomains { found, required })` if fewer than N=5 distinct non-Unknown/Unhydrated domains are represented. On error the binary exits 1.
5. **Commit to global** — write merged entries into the `OnceLock<Arc<RwLock<[SuperDomain; 256]>>>` (see section 3). This is the only writer during startup.
6. **Spawn background refresher** — a `tokio::task` (or `std::thread` in sync builds) wakes every **1 hour**, re-runs steps 2-4, and swaps the table under `RwLock` write-lock. If the refresh fails sanity, it logs ERROR and keeps the previous table.
7. **Return** — `(UnifiedBridge, BridgeHandle)`. `BridgeHandle` carries the channel for triggering manual reloads (section 4).

`new()` is preserved for tests but marked `#[cfg(test)]` and emits a compile-time warning via `#[deprecated]` in non-test builds.

---

## 3. Sentinel Handling

### New variant

```rust
pub enum SuperDomain {
    Unknown       = 0,   // explicitly unclassified -- family known, domain not assigned
    Healthcare    = 1,
    Finance       = 2,
    Legal         = 3,
    Government    = 4,
    Education     = 5,
    Retail        = 6,
    Osint         = 7,
    // future variants occupy 8-254
    Unhydrated    = 255, // table not yet loaded -- transient startup sentinel
}
```

`Unhydrated = 255` is the sentinel for "we have not loaded the table yet." It must never appear in a committed audit event. `Unknown = 0` is retained as a legitimate classification meaning "this family ID is known to the registry but has not been assigned a super-domain."

### Resolution API

```rust
pub fn try_resolve(family_id: u8) -> Result<SuperDomain, HydrationError> {
    let table = FAMILY_TABLE.get().ok_or(HydrationError::TableNotInitialized)?;
    let guard = table.read();
    let sd = guard[family_id as usize];
    if sd == SuperDomain::Unhydrated {
        Err(HydrationError::TableNotInitialized)
    } else {
        Ok(sd)
    }
}
```

The existing `super_domain_for_family(family_id: u8) -> SuperDomain` becomes a thin shim over `try_resolve`, returning `SuperDomain::Unknown` on error, for backward-compat callers that cannot propagate errors. All new call-sites must use `try_resolve`.

### `HydrationError`

```rust
pub enum HydrationError {
    SeedParseFailed(toml::de::Error),
    InsufficientDomains { found: usize, required: usize },
    TableNotInitialized,
    UnknownVariant(String),
}
```

---

## 4. Hot-Reload Contract

`BridgeHandle` exposes:

```rust
impl BridgeHandle {
    pub async fn reload_family_table(&self) -> Result<HydrationRefreshSummary, HydrationError>;
}
```

Calling `reload_family_table()` triggers an out-of-cycle refresh (same logic as the background task in section 2, steps 2-4). On success it:

1. Swaps the `RwLock<[SuperDomain; 256]>` under write-lock.
2. Emits a `UnifiedAuditEvent::HydrationRefresh` event to the bridge's audit channel:

```rust
UnifiedAuditEvent::HydrationRefresh {
    updated_count: u16,   // number of family IDs whose domain changed
    timestamp: DateTime<Utc>,
    source: HydrationSource, // Seed | LanceOverlay | HttpOverlay | Manual
}
```

3. Returns `HydrationRefreshSummary { updated_count, previous_sanity_ok, new_sanity_ok }`.

The hot-reload path must be atomic from the perspective of concurrent readers: a reader holding the `RwLock` read-guard during a swap sees a consistent table (either old or new, never torn). Writers (the reload task) acquire the write-lock, build the new array off-lock, then swap under the lock for minimal contention.

---

## 5. Sanity Gate

```rust
pub fn sanity_gate(map: &HashMap<u8, SuperDomain>) -> Result<(), HydrationError> {
    const REQUIRED_DISTINCT: usize = 5;
    let distinct: HashSet<SuperDomain> = map.values()
        .filter(|sd| **sd != SuperDomain::Unknown && **sd != SuperDomain::Unhydrated)
        .cloned()
        .collect();
    if distinct.len() < REQUIRED_DISTINCT {
        return Err(HydrationError::InsufficientDomains {
            found: distinct.len(),
            required: REQUIRED_DISTINCT,
        });
    }
    Ok(())
}
```

The gate is applied both at startup (section 2 step 4) and after every background or manual reload (section 4). At startup, `Err` causes the binary entrypoint to print a human-readable message to stderr and call `std::process::exit(1)`. In library mode (no binary entrypoint), the caller receives the `Err` and decides.

The threshold N=5 is a compile-time constant `REQUIRED_DISTINCT` in `hydration.rs`. It can be overridden via `BridgeConfig::min_distinct_domains: Option<usize>`. The rationale: fewer than 5 distinct domains strongly suggests the TOML seed is truncated or the overlay returned garbage.

---

## 6. Test Plan

### Unit Tests (in `crates/lance-graph-callcenter/src/hydration.rs`)

**U1 - seed parse round-trip:** Parse the bundled `family_to_super_domain.toml` via `load_seed()`. Assert every variant 1-7 appears at least once. Assert no `Unhydrated` entry.

**U2 - sanity gate passes with >= 5 domains:** Build a `HashMap` with 5 distinct non-Unknown domains. Assert `sanity_gate` returns `Ok(())`.

**U3 - sanity gate fails with 4 domains:** Build a `HashMap` with 4 distinct domains. Assert `sanity_gate` returns `Err(HydrationError::InsufficientDomains { found: 4, .. })`.

**U4 - `try_resolve` before init returns `Err(TableNotInitialized)`:** Call `try_resolve(1)` without initializing the `OnceLock`. Assert the error variant.

**U5 - `try_resolve` after init returns correct domain:** Call `new_hydrated(BridgeConfig::default())`. Assert `try_resolve(1) == Ok(SuperDomain::Healthcare)` (using the seed mapping from the bundled TOML).

**U6 - backward-compat shim returns `Unknown` on uninitialized:** Call the old `super_domain_for_family(42)` before init. Assert it returns `SuperDomain::Unknown` (not a panic, not `Unhydrated`).

### Integration Tests (in `crates/lance-graph-callcenter/tests/hydration_integration.rs`)

**I1 - audit event emitted on reload:** Construct a `UnifiedBridge` via `new_hydrated`. Call `handle.reload_family_table()`. Read the audit channel. Assert a `UnifiedAuditEvent::HydrationRefresh` event was received with `updated_count >= 0` and a valid timestamp.

**I2 - HTTP overlay timeout does not block startup:** Configure `BridgeConfig` with an `http_overlay_url` pointing to `http://192.0.2.1` (TEST-NET, guaranteed unreachable) and `http_timeout = Duration::from_millis(200)`. Assert `new_hydrated` completes within 1 second and returns `Ok`. The overlay is silently skipped.

---

## 7. W4 Cross-Flag

Subcrates (e.g. `lance-graph-telephony`, `lance-graph-medical`) call `register_families()` on a shared `FamilyRegistry` before `new_hydrated()` is invoked. This registers the family IDs those subcrates own. The hydration bootstrap sequence must:

- Accept a reference to the `FamilyRegistry` in `BridgeConfig`.
- After applying all overlay layers, validate that no seed or overlay entry claims a family ID that is registered to a subcrate **unless** the domain assignment matches what that subcrate declared.
- The seed file must NOT pre-claim family IDs in the range reserved for subcrate dynamic registration. The range `200-254` is reserved; the seed ships only entries in `0-199`.

If a conflict is detected (seed or overlay disagrees with a subcrate's declared domain for one of its own family IDs), the bridge logs WARN and defers to the subcrate declaration. This policy prevents the seed from silently overriding subcrate-local knowledge. The conflict count is included in the `HydrationRefresh` audit event as `conflict_count: u16`.

---

## 8. Open Questions

**OQ-1 - `AtomicU8` vs `RwLock` for hot-path perf.**  
The current plan uses `RwLock<[SuperDomain; 256]>`. An alternative is 256 `AtomicU8` values (one per family ID), which allows lock-free reads. The trade-off: `AtomicU8` requires `unsafe` transmute between `u8` and `SuperDomain` (unless we store raw discriminants), and a torn mid-reload window where some slots are updated and others are not. `RwLock` gives a consistent snapshot. Decision deferred to performance profiling; the abstraction behind `try_resolve()` means the storage model can be swapped without changing callers.

**OQ-2 - `const`-eval vs runtime TOML parse.**  
The seed TOML is currently planned as a runtime `toml::from_str` call inside `load_seed()`. An alternative is a build-script (`build.rs`) that parses the TOML at compile time and emits a `const [SuperDomain; 256]` array via `include!` or `phf`. This eliminates the runtime parse and the `toml` dependency from the final binary. Downside: the build script adds complexity and makes the seed harder to patch without a rebuild. Decision: use runtime parse for v1; migrate to `build.rs` const-eval in v2 if binary size or cold-start time is a concern.

**OQ-3 - Overlay conflict resolution policy.**  
When the Lance overlay and the HTTP overlay both specify a domain for the same family ID, who wins? Current default: HTTP overlay wins (last writer). This may be wrong if the Lance table represents curated ground truth and the HTTP endpoint represents a live feed that may have data-quality issues. Alternative: Lance overlay wins; HTTP overlay only fills gaps. This policy should be declared in `BridgeConfig::overlay_priority: OverlayPriority` with variants `HttpWins` (default for now) and `LanceWins`. The spec leaves the default as `HttpWins` pending a product decision.

---

## New Files Enumerated

| File | Action | Key additions |
|------|---------|---------------|
| `crates/lance-graph-callcenter/src/super_domain.rs` | Modify | Add `Unhydrated = 255`, `HydrationError`, `try_resolve()`, `sanity_gate()`, `FAMILY_TABLE: OnceLock<Arc<RwLock<[SuperDomain;256]>>>` |
| `crates/lance-graph-callcenter/src/unified_bridge.rs` | Modify | Add `new_hydrated()`, `BridgeHandle`, `BridgeConfig`, `BridgeConfig::lance_overlay_path`, `BridgeConfig::http_overlay_url`, `BridgeConfig::min_distinct_domains` |
| `crates/lance-graph-callcenter/src/unified_audit.rs` | Modify | Add `UnifiedAuditEvent::HydrationRefresh { updated_count, timestamp, source, conflict_count }`, `HydrationSource` enum |
| `crates/lance-graph-callcenter/src/hydration.rs` | **New** | `load_seed()`, `load_overlays()`, `sanity_gate()`, `REQUIRED_DISTINCT`, background refresh task |
| `crates/lance-graph-callcenter/tests/hydration_integration.rs` | **New** | I1, I2 integration tests |
| `crates/lance-graph-contract/data/family_to_super_domain.toml` | **New** | Seed mapping family_id to super_domain for all known families in range 0-199 |

---

*End of spec. Total estimated implementation: ~600 lines of Rust + ~256-line TOML seed.*
