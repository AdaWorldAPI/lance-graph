# Handover — super-domain-rbac-tenancy-v1: Tier A complete, Tier B+ pending

**Date:** 2026-05-13 08:52 UTC
**From:** Opus 4.7 session (continued from compaction)
**To:** next session picking up the integration plan
**Spec:** `.claude/plans/super-domain-rbac-tenancy-v1.md` (~1387 lines, §1-§19)
**Source PR (merged):** [`AdaWorldAPI/lance-graph#363`](https://github.com/AdaWorldAPI/lance-graph/pull/363) — D-SDR-1 + D-SDR-2 + spec + Codex canonical-name fix; merged 2026-05-13T07:24Z at sha `421e71e`.
**Working branch:** `claude/lance-datafusion-integration-gv0BF` (5 commits ahead of `main` post-#363; no follow-up PR opened yet).

---

## TL;DR

- **Tier A (D-SDR-1..5) is fully implemented and pushed**, but only D-SDR-1+2 are merged via PR #363. D-SDR-3, D-SDR-4, D-SDR-5 (and a knowledge-doc commit) sit on the working branch **unmerged** and need a follow-up PR.
- **Consumer wiring (medcare-analytics, smb-bridge) committed locally** to the matching branch in each repo but **not pushed/PR'd**.
- **Tier B (TTL namespaces) onward is unstarted** and partly blocked on `AdaWorldAPI/OGIT` + `AdaWorldAPI/MedCare` + `AdaWorldAPI/MedCareV2` being out of the current MCP scope.
- Board governance files (`LATEST_STATE.md`, `STATUS_BOARD.md`, `PR_ARC_INVENTORY.md`, `INTEGRATION_PLANS.md`) **do not yet record D-SDR-3..5** — they should be updated as part of the follow-up PR per the Mandatory Board-Hygiene Rule.

---

## 1. Spec map (§1-§19)

| Section | Topic | Status |
|---|---|---|
| §1 | Why this exists — 4-level OGIT hierarchy, 6 bytes/row, RBAC + Chinese walls | Stable |
| §2 | The 4-level hierarchy (MetaAnchors → SuperDomain → OgitBasin → WithinBasinSlot) | Stable |
| §3 | Core DTOs (3.1..3.10) | Stable; mostly shipped in D-SDR-1..3 |
| §4 | Concrete consumer-to-basin mapping | Stable |
| §5 | OSLC absorption decision | Stable (absorb-with-lineage) |
| §6 | Cross-tenant federation policy | Stable (3 options A/B/C) |
| §7 | Substrate citations | Stable |
| §8 | Deliverables (Tier A-E originals) | Tier A complete; B+ pending |
| §9 | Tradeoffs flagged | Stable |
| §10 | Open questions for next session | Several resolved by §13-§18 |
| §11 | Status | Updated 2026-05-13 |
| §12 | One-line summary | — |
| §13 | Refinements — PolicyRewriter chain reuse, LanceDB encryption, merkle audit, hard-locks, researcher anonymization | Adds D-SDR-13..17 |
| §14 | Harvest + templates + cross-language migration | Adds D-SDR-18..23 |
| §15 | Multi-implementation drift detection | Superseded by §16+§17; preserved as design arc |
| §16 | Zone 3 drift boundary + two-track migration; CRITICAL crypto correction (single 3DES, ECB-equivalent, zero IV) | Adds D-SDR-24..30 (revisions) |
| §17 | DataFusion SQL inside LanceDB as unified persistence; Arrow Flight SQL replaces custom IDL | Adds D-SDR-31..34 |
| §18 | Empirical reality check — MedCare + MedCareV2 inspection; LanceProbe IS the drift bridge; Argon2-backfill replaces 3DES rewrap | Adds D-SDR-35..39 |
| §19 | Build invariants + SIMD strategy (`ndarray::simd` canonical path; rust 1.94.1 / lance =4.0.0 / lancedb 0.27.2) | Stable |

---

## 2. Deliverable status (D-SDR-1 through D-SDR-39)

### Tier A — DTOs + bridge surface (lance-graph workspace)

| # | Title | Status | Commit | Notes |
|---|---|---|---|---|
| **D-SDR-1** | `UnifiedBridge` + `TenantId` + `AuthError` + `OgitFamily` + `OwlIdentity` newtypes | ✅ MERGED via #363 | `f627ef1` (in #363) | ~360 LOC + 4 tests |
| **D-SDR-2** | `SuperDomain` + `MetaAnchors` + `DolceMarker` + `ComplianceRegime` + `SUPER_DOMAINS` + `FAMILY_TO_SUPER_DOMAIN` | ✅ MERGED via #363 | `17987ce` (in #363) | ~290 LOC + 7 tests |
| **D-SDR-2 fix** | Codex P2: authorize against canonical entity type via `bridge.row().ogit_uri.name()`, not bridge-side alias | ✅ MERGED via #363 | `421e71e` (in #363) | +2 regression tests |
| **D-SDR-3** | `OgitFamilyTable` + `FamilyEntry` + `PerFamilyCodebook` + `OwlCharacteristics` + `SchemaKind` inline (NOT sidecar) | ✅ COMMITTED, ⏳ unmerged | `2c3e87d` | `crates/lance-graph-callcenter/src/family_table.rs`, ~330 LOC + 9 tests |
| **D-SDR-4** | `UnifiedAuditEvent` + `AuditChain` (merkle FNV-1a chain) + `AuditMerkleRoot` + `verify_chain` + `NoopUnifiedAuditSink` | ✅ COMMITTED, ⏳ unmerged | `1d0157f` (+ `dabd510` lockfile) | `crates/lance-graph-callcenter/src/unified_audit.rs`, ~340 LOC + 10 tests |
| **D-SDR-5** | Wire `authorize_read/write/act` through `Policy::evaluate` + emit one chained audit event per call; `with_audit_chain`/`with_audit_chain_resume`/`audit_root` builders; `BridgeError` short-circuits before audit | ✅ COMMITTED, ⏳ unmerged | `dc9e081` | +5 audit tests, 96/96 lib tests green, clippy `-D warnings` clean |

**Tier A test count:** 96 lib tests in `lance-graph-callcenter` (was 85 pre-D-SDR-3).

### Tier B — TTL namespaces (AdaWorldAPI/OGIT fork PRs)

| # | Title | Status | Blocker |
|---|---|---|---|
| **D-SDR-6** | `OGIT/NTO/Hiro/{entities,verbs}/*.ttl` (15-30 entities, 10-20 verbs; absorbs OSLC-* with lineage) | ⏳ NOT STARTED | `AdaWorldAPI/OGIT` repo out of current MCP scope |
| **D-SDR-7** | `OGIT/NTO/HubSpot/{entities,verbs}/*.ttl` (15-25 entities, 8-15 verbs) | ⏳ NOT STARTED | Same MCP scope blocker |

### Tier C — Consumer crate scaffolding

| # | Title | Status |
|---|---|---|
| **D-SDR-8** | `/home/user/hiro-rs` new crate; `HiroBridge::from_registry()` + round-trip integration test | ⏳ NOT STARTED |
| **D-SDR-9** | `/home/user/hubspot-rs` new crate; `HubspotBridge::from_registry()` + round-trip test | ⏳ NOT STARTED |

### Tier D — Compliance + audit surface

| # | Title | Status | Notes |
|---|---|---|---|
| **D-SDR-10** | `AuditEntry` JSON schema + `JsonLinesAuditSink` impl + retention policy doc | ⏳ NOT STARTED | D-SDR-4 lays the in-memory chain; D-SDR-10 = JSON persistence sink |
| **D-SDR-11** | Compliance regime certification doc (HIPAA §164.312 / SOX §404 / PCI-DSS Reqs 3+7+10) | ⏳ NOT STARTED | Doc-only, ~200 lines |

### Tier E — Cross-tenant federation gate (Phase 2)

| # | Title | Status |
|---|---|---|
| **D-SDR-12** | `FederationPolicy::KAnonymityAggregate` (HLL cardinality gate) — **deferred until customer demands** | ⏳ DEFERRED |

### Tier additions from §13 (merkle/hard-lock/DP/encrypted-view)

| # | Title | Status |
|---|---|---|
| **D-SDR-13** | `merkle_salt` on `SuperDomainEntry` + per-super-domain HKDF for `TenantContext::encryption_key` (hard-lock crypto barrier, §13.4) | ⏳ NOT STARTED — partly anticipated by D-SDR-4's per-super-domain `merkle_salt: u64` field on `AuditChain` |
| **D-SDR-14** | `AuditEntry` updated schema (merkle + ClamPath + salt) + JSONL sink with replay-time integrity verification | ⏳ NOT STARTED — D-SDR-4's `verify_chain` is the in-memory primitive; D-SDR-14 = persistent format + replay |
| **D-SDR-15** | `PolicyKind::DifferentialPrivacy` for `researcher` role; aggregate-only + ε-bounded noise + k-anonymity floor | ⏳ NOT STARTED |
| **D-SDR-16** | `EncryptedViewAggregate` federation policy (LanceDB transparent encrypted view bridge) | ⏳ NOT STARTED |
| **D-SDR-17** | Hard-lock partner matrix as static table + predicate-time enforcement in `authorize()` | ⏳ NOT STARTED — Healthcare ↔ OSINT + 3 others, per §13.4 |

### Tier F — Harvest + meta-bridge + migration (from §14)

| # | Title | Status |
|---|---|---|
| **D-SDR-18** | Archaeology pass on `medcare_bridge.rs`/`sharepoint_bridge.rs`/`woa_bridge.rs`; extract fix-commits as named tests in `meta_bridge::tests` | ⏳ NOT STARTED |
| **D-SDR-19** | `MetaBridge` trait + `BridgeFromRegistry` extension absorbing harvested patterns | ⏳ NOT STARTED |
| **D-SDR-20** | ❌ **SUPERSEDED by §17.2** (Arrow Flight SQL replaces custom Protobuf IDL) |
| **D-SDR-21** | `MedCare-rs` retrofit to `MetaBridge` (zero behavior change) | ⏳ NOT STARTED |
| **D-SDR-22** | `smb-office-rs` retrofit (zero behavior change) | ⏳ NOT STARTED |
| **D-SDR-23** | `MedCareV2 C#` aligned to `MetaBridge` via Arrow Flight SQL client | ⏳ NOT STARTED — but §18 reframes this as HTTP+JSON first, Flight SQL Phase 5 |

### Tier (from §15-§16) — drift detection, MySQL one-shot import, crypto migration

| # | Title | Status |
|---|---|---|
| **D-SDR-24** | ❌ **DROPPED in §16/§17** — `MySQLAdapterBridge` replaced by D-SDR-27 one-shot import |
| **D-SDR-25** | `DriftDetectionBridge` impl + `JsonLinesDriftSink` — **demoted to Phase 2-3 cutover window only**; retires after Phase 4 | ⏳ NOT STARTED |
| **D-SDR-26** | Determinism rule test suite — **reduced from 12 to ~3 rules** (decimal + timestamp + FP aggregate), then further to **6 named tests** in §18 (date / decimal / bool / soft-delete / pwd / timestamp) | ⏳ NOT STARTED |
| **D-SDR-27** | One-shot MySQL+3DES → LanceDB import tool (**scope reduced** in §18: ~80 LOC + 2 integration tests; **drops 3DES→AES-GCM rewrap entirely** — carry ciphertext forward, Argon2-backfill on login) | ⏳ NOT STARTED, partly blocked on `AdaWorldAPI/MedCare` MCP scope for column inventory |
| **D-SDR-28** | MerkleRoot-beside-ciphertext storage layout (lance =4.0.0 schema) | ⏳ NOT STARTED |
| **D-SDR-29** | Per-row encryption envelope wiring (Phase 2 cutover) | ⏳ NOT STARTED |
| **D-SDR-30** | 3DES key destroy step (post-Argon2-backfill completion, Phase M6) | ⏳ NOT STARTED |

### Tier (from §17) — DataFusion-on-LanceDB convergence

| # | Title | Status |
|---|---|---|
| **D-SDR-31** | Arrow Flight SQL endpoint exposing DataFusion catalog (lancedb 0.27.2) | ⏳ NOT STARTED — **§18 update:** HTTP+JSON over JWT is M2-M6; Flight SQL is Phase 5+ |
| **D-SDR-32** | Substrait dialect mapping (DataFusion → C# client) | ⏳ NOT STARTED |
| **D-SDR-33** | Substrait extension types for `OwlIdentity` + `MerkleRoot` + `SuperDomain` | ⏳ NOT STARTED |
| **D-SDR-34** | Cross-language schema fixtures + golden tests | ⏳ NOT STARTED |

### Tier H — medcare-rs LanceProbe wiring (from §18.7)

| # | Title | Status |
|---|---|---|
| **D-SDR-35** | `POST /api/__parity/csharp` ingest endpoint (DriftEvent JSON → Lance table) — blocks LanceProbe M5 | ⏳ NOT STARTED |
| **D-SDR-36** | `GET /api/__parity` dashboard endpoint (cross-session aggregation) — blocks M5 | ⏳ NOT STARTED |
| **D-SDR-37** | `_dto_contracts.md` for 5 pilot endpoints + 40+ FUTURE_STACK_ADMIN routes — **blocks M2** | ⏳ NOT STARTED |
| **D-SDR-38** | `legacy-tripledes-fallback` feature flag with Argon2 backfill-on-login — blocks M5a | ⏳ NOT STARTED |
| **D-SDR-39** | `/api/__parity/telemetry` endpoint — blocks M6 | ⏳ NOT STARTED |

---

## 3. Concrete implementation state

### 3.1 lance-graph workspace (this repo)

Branch `claude/lance-datafusion-integration-gv0BF`, 5 commits ahead of merged `main`:

```
dc9e081 feat(lance-graph-callcenter): D-SDR-5 wire authorize_* through Policy with chained audit emission
dabd510 chore(deps): Cargo.lock after D-SDR-4
1d0157f feat(lance-graph-callcenter): D-SDR-4 merkle-chained audit log for UnifiedBridge
2c3e87d feat(lance-graph-callcenter): D-SDR-3 per-family codebook table (OgitFamilyTable + FamilyEntry)
3e94a27 knowledge: log E1-E6 splat + formal-grounding epiphanies (inbox doc, separate from active spec)
```

**Modules in `crates/lance-graph-callcenter/src/`:**

| File | LOC | What it defines |
|---|---|---|
| `unified_bridge.rs` | ~750 | `UnifiedBridge<B: NamespaceBridge>`, `TenantId`, `OgitFamily`, `OwlIdentity`, `AuthError`, builders, audit emission, 11 tests (incl. 5 D-SDR-5 audit tests) |
| `super_domain.rs` | ~290 | `SuperDomain`, `MetaAnchors`, `DolceMarker`, `ComplianceRegime`, `SuperDomainEntry`, `SUPER_DOMAINS`, `FAMILY_TO_SUPER_DOMAIN`, 7 tests |
| `family_table.rs` | ~330 | `OgitFamilyTable`, `FamilyEntry`, `PerFamilyCodebook`, `OwlCharacteristics`, `SchemaKind`, 9 tests |
| `unified_audit.rs` | ~340 | `UnifiedAuditEvent`, `AuthOp`, `AuthDecision`, `AuditMerkleRoot`, `AuditChain`, `UnifiedAuditSink` trait, `NoopUnifiedAuditSink`, `verify_chain`, 10 tests |
| `lib.rs` | re-exports all of the above |

**Test count:** `cargo test -p lance-graph-callcenter --lib` → **96/96 passing**. `cargo clippy -p lance-graph-callcenter --tests --no-deps -- -D warnings` → clean on Tier A code. (Pre-existing `assert!(true)` warning in `tests/zone_serialize_check.rs:31` is unrelated to this work, commit `8528161`.)

**Codex P2 fix (in #363, commit `421e71e`):** policy evaluates against canonical OGIT entity type (`row.ogit_uri.name()`, e.g. `Order`), not the bridge-side alias (`public_name`, e.g. `WorkOrder`). Regression tests: `unified_bridge_evaluates_policy_against_canonical_entity_type` and `unified_bridge_does_not_honor_alias_keyed_policy`.

### 3.2 MedCare-rs (consumer)

Branch `claude/lance-datafusion-integration-gv0BF`, 1 commit ahead of origin (NOT pushed):

```
31e999b feat(medcare-analytics): wire UnifiedBridge<MedcareBridge> for F2 RBAC entry-point auth
```

File: `crates/medcare-analytics/src/unified_bridge_wiring.rs` (107 LOC, gated on `lance-phase2-rbac` feature). Constructor: `medcare_unified_bridge(registry, actor_role, tenant) -> Result<UnifiedBridge<MedcareBridge>, _>`. Uses `smb_policy()` as the placeholder Policy until D-SDR-2's medcare-specific roles ship.

**Action needed:** push and open consumer PR after the lance-graph follow-up PR lands.

### 3.3 smb-office-rs (consumer)

Branch `claude/lance-datafusion-integration-gv0BF`, 1 commit ahead of origin (NOT pushed):

```
342f601 feat(smb-bridge): wire UnifiedBridge<OgitBridge> for F2 RBAC entry-point auth
```

File: `crates/smb-bridge/src/unified_bridge_wiring.rs` (90 LOC, gated on `auth` feature).

**Action needed:** same as MedCare-rs — push + open consumer PR.

---

## 4. Architectural reality (the controlling shape)

Per §17.7 + §18.9 (after all refinements), the **net architecture** is:

```
Per-row data in LanceDB:
    tenant_id u32 + owl_id u16 + ciphertext + merkle_root(cleartext)
    Identity total: 6 bytes per row; merkle ~8 bytes; payload variable.

Access layer (single plan, two clients):
    Cypher / SPARQL / Gremlin / SQL → DataFusion logical plan → LanceDB scan
    Rust direct AND C# via HTTP+JSON (Phase M2-M6) / Flight SQL (Phase 5+).

RBAC enforcement (single masked predicate):
    PolicyRewriter chain in lance-graph-callcenter::policy
    (RowFilter + ColumnMask + RowEncryption + DifferentialPrivacy + Audit)
    composed by UnifiedBridge::authorize() 4-stage flow.

Drift detection:
    Phase 2-3 only — ArrowBatchDriftSignal across Rust + C# clients.
    Retires to CI gate after Phase 4 cutover.

Migration (one-shot, throwaway):
    D-SDR-27..30 — MySQL+3DES → LanceDB. 3DES keys destroyed post-import.
    NOT a sustained 3DES→AES-GCM rewrap; password column is Argon2-backfill-on-login.

Bridge harvest:
    medcare_bridge.rs (HIPAA) + sharepoint_bridge.rs (SMB) as pattern source.
    woa_bridge retrofit; hubspot_bridge + hiro_bridge new templates.
    ~45 LOC each after MetaBridge extraction.

Cross-language IDL:
    No custom Protobuf. Substrait extension types over Arrow Flight SQL.
    Phase 5+; HTTP+JSON over JWT is the immediate path.
```

### Key crypto correction (§18.5)

What the MedCare C# code calls "3DES" is **NOT** standard 3DES. It is:
- Single 3DES cipher (not a 3-cipher chain)
- 128-bit truncated key derived from a broken KDF
- ECB-equivalent mode (no IV chaining)
- Zero IV
- Hardcoded password table indexed by the first character of the ciphertext

This affects **only** the `u_pwd` column (likely; pending a focused grep of `MySQL_Connect.cs` per §18.10 open question). D-SDR-27's scope drops from "decrypt-3DES + rewrap-with-AES-GCM" to **"carry ciphertext forward unchanged; backfill Argon2 on successful legacy login"**.

### Hard-lock matrix (§13.4)

Healthcare ↔ OSINT (plus 3 other pairs to be enumerated in D-SDR-17) are crypto-barrier-locked: cross-domain audit chains are unlinkable via per-super-domain `merkle_salt`. D-SDR-4 already carries the `merkle_salt: u64` field on `AuditChain`; D-SDR-13 wires it into `TenantContext::encryption_key` via HKDF.

---

## 5. Build invariants (§19)

**Pinned versions** (already set in workspace `Cargo.toml`):
- `rust 1.94.1` (stable; no `#![feature(...)]` anywhere)
- `lance =4.0.0`
- `lancedb 0.27.2`

**SIMD policy:** `ndarray::simd` is the canonical SIMD path. The `LazyLock<Tier>` dispatch pattern is already shipped — just import. Tier A LOC reduction story did not materialize (per §19.7 correction) because the D-SDR-1..3 hot path is per-row scalar, not batch.

**Clippy gate:** `cargo clippy -- -D warnings` is the merge gate. Every `unsafe` block needs a `// SAFETY:` comment.

---

## 6. Pending work (ordered by what unblocks what)

### Immediately actionable (no external blockers)

1. **Open follow-up PR for D-SDR-3 + D-SDR-4 + D-SDR-5** on `AdaWorldAPI/lance-graph`. The five unmerged commits (`3e94a27`, `2c3e87d`, `1d0157f`, `dabd510`, `dc9e081`) cleanly stack on the merged #363 base.
2. **Push + open consumer PRs:**
   - `AdaWorldAPI/medcare-rs` (commit `31e999b` already local; branch up to date)
   - `AdaWorldAPI/smb-office-rs` (commit `342f601` already local)
3. **Update governance board** (per Mandatory Board-Hygiene Rule) **in the same follow-up PR**:
   - `.claude/board/LATEST_STATE.md` — add D-SDR-3..5 modules to Contract Inventory
   - `.claude/board/STATUS_BOARD.md` — D-SDR-1..5 = Shipped (post-merge), D-SDR-6..39 = Queued
   - `.claude/board/PR_ARC_INVENTORY.md` — PREPEND entries for #363 (already merged) and the upcoming follow-up PR
   - `.claude/board/INTEGRATION_PLANS.md` — PREPEND status update
4. **D-SDR-13** (per-super-domain HKDF in `TenantContext::encryption_key`) — wires the `merkle_salt` field already shipped on `AuditChain` (D-SDR-4) into key derivation. Self-contained.
5. **D-SDR-17** (hard-lock partner matrix) — static table + predicate-time check in `authorize()`. ~60 LOC + 4 tests. Self-contained.
6. **D-SDR-10** (`JsonLinesAuditSink`) — JSON persistence for `UnifiedAuditEvent`. D-SDR-4 already exposes the `UnifiedAuditSink` trait + `NoopUnifiedAuditSink`. ~80 LOC + 1 integration test.

### Blocked on MCP scope expansion

7. **D-SDR-6 + D-SDR-7** (TTL namespaces on `AdaWorldAPI/OGIT` fork) — need OGIT repo in MCP scope.
8. **D-SDR-27** column inventory — needs `AdaWorldAPI/MedCare` + `AdaWorldAPI/MedCareV2` in MCP scope to grep `MySQL_Connect.cs` for `EncryptMessage()`/`DecryptMessage()` callsites.
9. **D-SDR-35..39** (medcare-rs LanceProbe endpoints) — coordination doc lives at `MedCare-rs/docs/CSHARP_HANDOFF_PROMPT.md` on branch `claude/csharp-handoff-docs-L3DF0`; should be merged or its content referenced from this spec.

### Larger blocks

10. **D-SDR-18 + D-SDR-19** (Tier F harvest + MetaBridge extraction) — substantive work; produces the trait the consumer migrations (D-SDR-21..23) retrofit onto.
11. **D-SDR-31..34** (Tier from §17) — Arrow Flight SQL endpoint + Substrait extension types. **Phase 5+** per §18 — HTTP+JSON over JWT precedes this.
12. **D-SDR-15 + D-SDR-16** (DP role + encrypted view aggregate) — Phase 2 federation; gates on D-SDR-12 demand signal first.

### Deferred indefinitely

- **D-SDR-12** — federation Phase 2, deferred until a customer demands it.

---

## 7. Knowledge / context recovery

- **Spec master:** `.claude/plans/super-domain-rbac-tenancy-v1.md` (§1-§19). Read §17.7 + §18.9 + §19 first — those are the controlling sections; §15-§16 are preserved design arcs.
- **PR #363 description:** the most concise narrative of what shipped in Tier A starter. Read it to understand the gap between starter (#363 merged) and full Tier A (D-SDR-3..5 unmerged on branch).
- **Per-deliverable LOC estimates:** §8 in the spec gives original LOC budgets; §13.8 / §14.4 / §15.3 / §16.6 / §17.5 / §18.7 carry the revisions.
- **CRITICAL crypto correction:** §18.5. Without reading this you will design the wrong rewrap pipeline for D-SDR-27.
- **Build pins + SIMD canonical path:** §19. Without this you may reach for nightly features or write scalar fallbacks that should delegate to `ndarray::simd`.
- **Consumer-side coordination:** `MedCare-rs/docs/CSHARP_HANDOFF_PROMPT.md` (branch `claude/csharp-handoff-docs-L3DF0`). Defines LanceProbe milestones M1-M6 against D-SDR-35..39.

## 8. Things explicitly NOT done (do not redo)

- **Do NOT design a custom Protobuf IDL** — Arrow Flight SQL replaces it (§17.3, dropped in §17.6).
- **Do NOT design a 3DES→AES-GCM rewrap pipeline** — Argon2 backfill on login is the correct approach (§18.5, D-SDR-27 scope refinement in §18.6).
- **Do NOT reshape MedCareV2 C#** — it is an overlay (LanceProbe) on top of a copy of MedCare; do not refactor (§18.9).
- **Do NOT introduce a parallel enforcement path** alongside `PolicyRewriter` — D-SDR-5 composes onto the shipped chain (§13.1, the ~30% LOC reduction lever).
- **Do NOT add unstable Rust features** — workspace is stable-only per §19.1.

---

## 9. Open questions still standing (post §18.10)

1. Which columns besides `u_pwd` call `EncryptMessage()`/`DecryptMessage()` in `MySQL_Connect.cs`? Needs grep of the 721 KB file. Likely very few or none.
2. DTO contracts for the 40+ additional routes per `FUTURE_STACK_ADMIN.md §4` — D-SDR-37 needs to enumerate them. Pending an agent pass.
3. `MedCare-rs/docs/AUTH_LEGACY_TRIPLEDES_MIGRATION.md` is DRAFT — what blocks promotion to Active? Likely D-SDR-38 implementation + a human security review.
4. Hard-lock partner matrix completeness — defer enumeration until the billing/ticket basins are concretized (per §18.10).
5. Per-super-domain DP epsilon defaults — not blocking (Phase 2 federation territory).

---

**End of handover.**

The natural next move is item (1) under §6: open the follow-up PR for D-SDR-3..5 with the board-hygiene updates folded in, then push the two consumer PRs in parallel.
