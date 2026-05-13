# PR-D4 — FAMILY_TO_SUPER_DOMAIN Hydration from TTL

> **Sprint:** sprint-5 / sprint-log-5-6 W3
> **Worker:** W3 (Sonnet 4.6)
> **Date:** 2026-05-13
> **Branch target:** `claude/lance-datafusion-integration-gv0BF`
> **Status:** DRAFT — awaiting engineer execution
> **Prior art:** `.claude/specs/td-sdr-family-hydration.md` (tech-debt stub, W9-retry sprint-4);
>               `.claude/plans/super-domain-rbac-tenancy-v1.md §3.4, §8 D-SDR-4, §9.1`
> **Extends:** `super-domain-rbac-tenancy-v1.md §6` (federation policy) — specifically §9.1
>             (single-member assignment) and the deferred hydration bullet in PR #364 Locked.

---

## 0 — Problem Statement

Commit `e23ce89` (PR #364, codex P2 fix) correctly routes `emit_audit` through
`AuditChain.super_domain()` instead of the static `FAMILY_TO_SUPER_DOMAIN` table, but the
table itself remains a `static [SuperDomain::Unknown; 256]` that is **never populated at runtime**:

```rust
// crates/lance-graph-callcenter/src/super_domain.rs:316
static FAMILY_TO_SUPER_DOMAIN: [SuperDomain; 256] = [SuperDomain::Unknown; 256];
```

`super_domain_for_family()` reads this directly. Any code path that calls `super_domain_for_family`
without going through a correctly-configured `AuditChain` silently returns `SuperDomain::Unknown`
for every basin. The regression test at `unified_bridge.rs:667` documents this explicitly.

`FAMILY_TO_SUPER_DOMAIN` is now vestigial fallback (per PR #364 "Locked" annotation), but it is
still the only table that `super_domain_for_family()` reads and it is still uniformly wrong.

This spec describes how to hydrate it from TTL files at boot, with a TTL-refresh path.

---

## 1 — TTL File Format

### 1.1 Reuse the existing ontology TTL schema

`lance-graph-ontology` already ships `parse_ttl_directory_with_provenance()` (used by
`OntologyRegistry::hydrate_once_sync`). It parses standard Turtle (`.ttl`) files and emits
`MappingProposal` rows carrying `namespace` and `OgitUri`.

The family→super-domain mapping does NOT need a separate ad-hoc file format. Instead, OGIT
family-namespace TTL files carry two custom predicates:

```turtle
# Example: ogit/NTO/Healthcare/Healthcare.ttl
@prefix ogit:      <http://www.purl.org/ogit/> .
@prefix ogit.meta: <http://www.purl.org/ogit/meta/> .
@prefix xsd:       <http://www.w3.org/2001/XMLSchema#> .

ogit.Healthcare:
    a ogit:FamilyNamespace ;
    ogit.meta:superDomain "Healthcare" ;
    ogit.meta:familyId    "1"^^xsd:unsignedByte .
```

| Predicate | Value type | Meaning |
|---|---|---|
| `ogit.meta:superDomain` | `xsd:string` | Maps this namespace to a `SuperDomain` enum variant name |
| `ogit.meta:familyId` | `xsd:unsignedByte` | The `u8` basin id (`OgitFamily.raw()`) |

**Rationale for TTL over TOML:** The hydration machinery in `registry.rs` is already battle-tested
(idempotent via SHA-256 checksum, error-accumulated via `HydrationReport`, provenance-aware via
`dcterms:source`). The `td-sdr-family-hydration.md` draft proposed a TOML file, which would add
a second ingest mechanism plus a `toml` crate dependency. The TTL route keeps one ingest surface
and zero new dependencies.

### 1.2 Inline seed: `data/family_registry.ttl`

For binary distributions without a separate TTL root directory, a seed file ships inline:

```
crates/lance-graph-callcenter/data/family_registry.ttl
```

Compiled into the binary via `include_str!` inside `hydration::SEED_TTL`. Covers the 8 starter
super-domains and their ~75 family assignments at GA. Unclassified families are omitted; absence
implies `SuperDomain::Unknown`. Size budget: ~75 entries × ~120 bytes = ~9 KB.

---

## 2 — Boot-Time Hydration Sequence

### 2.1 Where in the call graph

Hydration fires inside `UnifiedBridge::new_hydrated(config: BridgeConfig)`, a new constructor
replacing bare `UnifiedBridge::new()` for production use:

```
binary entrypoint / consumer::setup()
  └─ UnifiedBridge::new_hydrated(config)
       │
       ├─ 1. hydration::load_seed(SEED_TTL)
       │       └─ parse_ttl_bytes → HashMap<u8, SuperDomain>
       │
       ├─ 2. hydration::load_overlay(config.ttl_overlay_dir)  [optional]
       │       └─ parse_ttl_directory_with_provenance → merge into map
       │
       ├─ 3. hydration::sanity_gate(&merged_map)
       │       └─ Err → fail-hard or fail-soft per BridgeConfig::hydration_policy
       │
       ├─ 4. hydration::commit(&merged_map)
       │       └─ write into FAMILY_TABLE: OnceLock<Arc<RwLock<FamilyTableInner>>>
       │
       ├─ 5. spawn_refresh_task(config)   [if config.ttl_refresh_interval.is_some()]
       │
       └─ 6. return (UnifiedBridge, BridgeHandle)
```

`UnifiedBridge::new()` is preserved for unit tests (`#[cfg(test)]`) and emits a `#[deprecated]`
warning in non-test builds so callers migrate without a hard break.

### 2.2 Failure semantics when TTL is missing or malformed

| Scenario | Fail mode | Rationale |
|---|---|---|
| Seed parse failed (malformed inline TTL) | **Hard fail** `Err(HydrationError::SeedParseFailed)` | Inline seed ships with the binary; a parse failure is a release bug. |
| Overlay directory missing | **Soft warn + continue on seed alone** | Overlay is optional; absence is normal in minimal deployments. |
| Overlay TTL malformed (one file) | **Soft warn + skip malformed file** | Partial overlays beat total startup failure. |
| Sanity gate fails (< 5 distinct non-Unknown domains) | **Hard fail in binary mode; soft warn in library mode** | Binary mode: misconfiguration. Library mode: consumer controls policy. |
| Seed empty (no family entries after parse) | **Soft warn + leave table all-Unknown** | Pre-boot reads of `super_domain_for_family` return `Unknown` — documented pre-hydration value. No crash. |

Surfaced via `BridgeConfig::hydration_policy`:

```rust
pub enum HydrationPolicy {
    /// Fail constructor if sanity gate fails. Default for binaries.
    RequireMinDomains { min: usize },
    /// Log WARN and continue with available seed. Default for tests/library.
    BestEffort,
}
```

---

## 3 — TTL Refresh Path

### 3.1 Hot reload vs restart-only

**Hot reload is supported; restart-only is the safe default.**

Background: `emit_audit` calls `AuditChain.super_domain()` (the P2 fix). That reads a field on
the `AuditChain` struct set at bridge construction — it does NOT touch `FAMILY_TABLE`. Hot-swapping
the fallback table has no correctness risk for correctly-wired `AuditChain` callers.

Hot reload is useful for operators who add a new OGIT basin TTL file without restarting the
callcenter service. It is opt-in:

```rust
pub struct BridgeConfig {
    /// None = restart-only. Some(d) = background refresh every d. Default: None.
    pub ttl_refresh_interval: Option<Duration>,
    pub ttl_overlay_dir: Option<PathBuf>,
    pub hydration_policy: HydrationPolicy,
}
```

### 3.2 Versioning

Each hydration run increments a `generation: u64` counter in `FamilyTableInner`:

```rust
struct FamilyTableInner {
    table:      [SuperDomain; 256],
    generation: u64,
    loaded_at:  std::time::Instant,
    source:     HydrationSourceSet,  // Seed | Overlay(path) | Manual
}
```

`BridgeHandle::family_table_generation() -> u64` exposes this for change-detection.

### 3.3 Refresh atomicity

1. Build new `[SuperDomain; 256]` array off-lock (I/O happens here; no lock held).
2. Acquire `RwLock` write-guard.
3. Swap array + increment `generation` + update `loaded_at` and `source`.
4. Release write-guard.
5. Emit `HydrationRefreshAudit` event (see §unified_audit.rs change).

Read side (`super_domain_for_family`) acquires the `RwLock` read-guard, copies the one byte
discriminant, releases immediately. Each reader sees either all-old or all-new — no torn state.

---

## 4 — Concurrency Model

### 4.1 Storage primitive

```rust
// crates/lance-graph-callcenter/src/super_domain.rs
// Replaces:  static FAMILY_TO_SUPER_DOMAIN: [SuperDomain; 256] = [SuperDomain::Unknown; 256];
static FAMILY_TABLE: OnceLock<Arc<RwLock<FamilyTableInner>>> = OnceLock::new();
```

`OnceLock` replaces the immutable compile-time static. `Arc` lets the background refresh task
share the lock without requiring `'static` lifetime on the bridge.

### 4.2 Effect on `emit_audit` hot path

`emit_audit` → `AuditChain.super_domain()` → struct field read. This is a field dereference with
no lock, no atomic, no table lookup. The `FAMILY_TABLE` is not involved. **Zero hot-path impact.**

`FAMILY_TABLE` is the fallback path only. Operators who configure `AuditChain` correctly before
the first audit event never touch `FAMILY_TABLE` in steady state.

### 4.3 RwLock vs AtomicU8 array

`RwLock<FamilyTableInner>` is preferred over 256 × `AtomicU8` for two reasons:

1. **Consistency during reload**: `RwLock` ensures readers see a full-generation snapshot.
   256 atomic stores during reload leave a torn window.
2. **`unsafe`-free**: `AtomicU8` → `SuperDomain` requires `unsafe transmute`; `RwLock` does not.

Tradeoff: `RwLock` read-acquisition is ~10 ns; `AtomicU8` load is ~1 ns. Since `super_domain_for_family`
is the **fallback** path, the 10 ns cost is immaterial. Revisit if promoted to hot path.

---

## 5 — Compatibility: Keep FAMILY_TO_SUPER_DOMAIN as Populated Fallback

**Decision: retain as populated fallback — do NOT hard-fail on pre-hydration reads.**

PR #364 "Locked" states `FAMILY_TO_SUPER_DOMAIN`'s purpose narrows to a "fallback / future
hydration mechanism." This spec fulfills the future-hydration clause. Post-hydration, the table
returns TTL-derived values. Pre-hydration (raw unit tests, before `new_hydrated`), it returns
`SuperDomain::Unknown` — same as before, no regression.

Hard-failing on pre-hydration reads would break existing tests that call `super_domain_for_family`
without going through `new_hydrated`. `BestEffort` policy preserves backward compat.

The `try_resolve()` API is added for new call sites:

```rust
/// Returns Err(HydrationError::TableNotInitialized) if new_hydrated has not yet run.
pub fn try_resolve(family: OgitFamily) -> Result<SuperDomain, HydrationError> {
    let inner = FAMILY_TABLE
        .get()
        .ok_or(HydrationError::TableNotInitialized)?;
    Ok(inner.read().unwrap().table[family.raw() as usize])
}
```

The old `super_domain_for_family` becomes a shim:

```rust
#[inline]
pub fn super_domain_for_family(family: OgitFamily) -> SuperDomain {
    try_resolve(family).unwrap_or(SuperDomain::Unknown)
}
```

---

## 6 — Delta vs super-domain-rbac-tenancy-v1.md §6

This spec extends `super-domain-rbac-tenancy-v1.md` at three points:

| Plan section | What this spec adds |
|---|---|
| **§3.4** `FAMILY_TO_SUPER_DOMAIN` comment `/* baked at hydration */` | Specifies *when* and *how* the bake happens: TTL-driven at `new_hydrated()`, `OnceLock<Arc<RwLock<FamilyTableInner>>>` backed, generation-versioned. The "at hydration" comment is now a concrete boot-time sequence. |
| **§8 D-SDR-4** static table with ~75 mappings | Upgrades the static-only approach to boot-time TTL hydration with optional overlay directory and hot-reload path. The static array (`[SuperDomain::Unknown; 256]`) becomes the pre-hydration default; the runtime path populates it via `OnceLock`. |
| **§9.1** single-member super-domain assignment | Preserved. The `ogit.meta:superDomain` predicate accepts exactly one value per namespace. Cross-cutting basins (HPO/MONDO) assign to the primary domain; the secondary domain is discoverable via `SuperDomainEntry.basins` reverse table (unchanged from plan). |

The federation policy in **§6** (pure Chinese wall / k-anonymity escape) is untouched —
hydration does not affect cross-tenant authorization logic.

---

## 7 — New Files Enumerated

| File | Action | Purpose |
|---|---|---|
| `crates/lance-graph-callcenter/data/family_registry.ttl` | **New** | Inline seed: ~75 family→super-domain mappings in Turtle. `include_str!`-d at compile time. ~9 KB. |
| `crates/lance-graph-callcenter/src/hydration.rs` | **New** | `load_seed()`, `load_overlay()`, `commit()`, `sanity_gate()`, `FAMILY_TABLE`, `FamilyTableInner`, `HydrationError`, `HydrationPolicy`, `HydrationSourceSet`. ~300 LOC. |
| `crates/lance-graph-callcenter/src/unified_bridge.rs` | Modify | Add `BridgeConfig`, `new_hydrated()`, `BridgeHandle`, `spawn_refresh_task()`. ~120 new LOC. |
| `crates/lance-graph-callcenter/src/super_domain.rs` | Modify | Replace `static FAMILY_TO_SUPER_DOMAIN` with `FAMILY_TABLE` OnceLock. Add `try_resolve()`. Convert `super_domain_for_family` to shim. ~+35 / −1 LOC. |
| `crates/lance-graph-callcenter/src/unified_audit.rs` | Modify | Add `HydrationRefreshAudit { generation, updated_count, source }` event variant. ~+15 LOC. |
| `crates/lance-graph-callcenter/tests/hydration_integration.rs` | **New** | Tests I1-I4. ~80 LOC. |

**Total estimate:** ~450 LOC new Rust + ~9 KB TTL seed. Zero new crate dependencies (TTL parsing
reuses `lance-graph-ontology::ttl_parse`; no `toml` crate needed).

---

## 8 — Test Plan

### Unit Tests (`crates/lance-graph-callcenter/src/hydration.rs`)

**U1 — seed parse round-trip:** Parse bundled TTL via `load_seed(SEED_TTL)`. Assert every
`SuperDomain` variant 1-7 appears at least once. Assert family 1 → `Healthcare`.

**U2 — sanity gate passes with ≥ 5 domains:** Build map with 5 distinct non-Unknown domains.
Assert `sanity_gate` returns `Ok(())`.

**U3 — sanity gate fails with 4 domains:** Build map with 4 distinct domains.
Assert `Err(HydrationError::InsufficientDomains { found: 4, .. })`.

**U4 — `try_resolve` before init returns `Err(TableNotInitialized)`:** Call `try_resolve(OgitFamily(1))`
without calling `new_hydrated`. Assert `Err(HydrationError::TableNotInitialized)`.

**U5 — backward-compat shim returns `Unknown` on uninitialized table:** Call
`super_domain_for_family(OgitFamily(42))` before init. Assert `SuperDomain::Unknown` (no panic).

**U6 — `try_resolve` after init returns seed-declared domain:** Call
`new_hydrated(BridgeConfig::default())`. Assert `try_resolve(OgitFamily(1)) == Ok(SuperDomain::Healthcare)`.

### Integration Tests (`crates/lance-graph-callcenter/tests/hydration_integration.rs`)

**I1 — audit event emitted on manual reload:** Construct via `new_hydrated`. Call
`handle.reload_family_table()`. Assert `HydrationRefreshAudit` event with `generation == 2`.

**I2 — overlay directory missing does not crash:** Configure non-existent `ttl_overlay_dir`.
Assert `new_hydrated` returns `Ok`. Assert table reflects seed values only.

**I3 — overlay overrides seed:** Create temp TTL reassigning family 1 to `Science`. Call
`new_hydrated` with overlay path. Assert `try_resolve(OgitFamily(1)) == Ok(SuperDomain::Science)`.

**I4 — background refresh updates generation:** Configure `ttl_refresh_interval = 50ms`. Modify
overlay file. Wait 100ms. Assert `BridgeHandle::family_table_generation()` incremented.

---

## 9 — Open Question

**OQ-1 — Parser extension boundary.** `hydration::load_overlay` will call
`parse_ttl_directory_with_provenance` from `lance-graph-ontology`. That function currently emits
`MappingProposal` rows — we need to extract two custom predicates (`ogit.meta:superDomain`,
`ogit.meta:familyId`). Options:

(a) Extend the parser to pass through unrecognized predicates as raw `(subject, predicate, object)`
triples alongside the `MappingProposal` stream. Clean but touches the existing parser surface.

(b) `hydration::load_overlay` reads the raw Oxigraph `MemoryStore` directly after TTL load,
bypassing the proposal layer entirely. Self-contained but bypasses provenance accounting.

(c) Add a thin separate `ttl_parse::parse_family_registry(ttl_bytes)` entry point in
`lance-graph-ontology` that only looks for `ogit.meta:superDomain` and `ogit.meta:familyId`.
Cleanest separation; new public API surface.

**Decision needed before implementation begins.** Option (c) is the recommendation: smallest
surface, no impact on the existing proposal path, and the function name self-documents its scope.

---

*End of spec. Estimated implementation: ~450 LOC Rust + ~9 KB TTL seed. One PR.*
