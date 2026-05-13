# PR-E2 — smb-office-rs UnifiedBridge Retrofit

> **Sprint:** 6 (S6-W3)
> **Author:** W7 (claude-sonnet-4-6), session 2026-05-13
> **Repo:** `AdaWorldAPI/smb-office-rs`
> **Size estimate:** ~480 LOC net (Batches A+B+C, excluding ~80 LOC tests)
> **Prior plan extended:** `super-domain-rbac-tenancy-v1.md` §14 (D-SDR-22)
> **Status:** SPEC READY — awaiting D-SDR-22 unlock (see §8 blockers)

---

## 0 — What PR-C Shipped (the baseline)

smb-office-rs#31 (merged 2026-05-13, +111 LOC) wired **one file**:

```
crates/smb-bridge/src/unified_bridge_wiring.rs
```

It provides `smb_unified_bridge(registry, namespace, actor_role, tenant) ->
Result<UnifiedBridge<OgitBridge>>`. One test (`smb_unified_bridge_errors_on_unhydrated_registry`)
exercises the error path. The constructor is **never called** by any production
path inside smb-office-rs. Everything else — `MongoConnector`, `LanceConnector`,
`SmbOrchestrator`, the login flow, the WoA binary skeleton — still bypasses
`UnifiedBridge` entirely.

PR-E2 is the **wider sweep** that makes smb-office actually *use* the bridge
it already knows how to build.

---

## 1 — Retrofit Scope: Call Sites That Still Bypass UnifiedBridge

Five paths inside smb-office-rs reach storage or orchestration decisions without
consulting `UnifiedBridge` after PR-C:

### 1.1 MongoConnector (crates/smb-bridge/src/mongo.rs)

`EntityStore::get` and `EntityWriter::upsert` issue BSON queries directly via
`mongodb::Client`. No call to `UnifiedBridge::authorize_read` or
`::authorize_write`. The doc comment acknowledges: "Tenant filtering — RLS
injection is upstream's job" — but that upstream gate is never invoked.

**Replacement:** `MongoConnector` gains a `bridge: Arc<UnifiedBridge<OgitBridge>>`
(optional in Batch A, mandatory in Batch C). Every read gates on
`bridge.authorize_read(owl_id, ctx)` before the BSON query; every write gates on
`bridge.authorize_write(owl_id, ctx)` before the BSON write.

| Method | UnifiedBridge replacement |
|---|---|
| `EntityStore::get` | `authorize_read` before BSON query |
| `EntityWriter::upsert` | `authorize_write` before BSON write |
| `EntityStore::scan` (future) | `authorize_read` + RLS predicate injection |

### 1.2 LanceConnector (crates/smb-bridge/src/lance.rs)

Same shape as `MongoConnector`. Appends to a Lance dataset with no authorization
gate. Doc comment: "What this connector does NOT do — Tenant filtering."

**Replacement:** Mirror the `MongoConnector` pattern. `LanceConnector` gains
`bridge: Arc<UnifiedBridge<OgitBridge>>`. Every `get` / `upsert` gates on the
bridge before the Lance dataset operation.

### 1.3 SmbOrchestrator::route (crates/smb-bridge/src/orchestration.rs)

`SmbOrchestrator::route` validates `smb.<entity>.<action>` and sets
`step.status = StepStatus::Completed`. It never calls `authorize_read` /
`authorize_write`. The existing `// TODO(F6 follow-up)` comment is the exact
slot where this gate must land.

**Replacement:** `SmbOrchestrator` gains `bridge: Arc<UnifiedBridge<OgitBridge>>`.
Before transitioning to `Completed`, `route()` calls:
- `bridge.authorize_write(smb_owl_id_for(entity), ctx)` for mutating actions
  (`upsert`, `send`, `submit`, `delete`, `create`, `update`)
- `bridge.authorize_read(smb_owl_id_for(entity), ctx)` for non-mutating actions
  (`lookup`, `scan`, `export`, `get`, `list`)

`smb_owl_id_for(entity)` maps entity names to `OwlIdentity` via new
`smb_owl_ids.rs` (see §5.2).

### 1.4 customer-woa-bin main.rs (crates/customer-woa-bin/src/main.rs)

Skeleton binary — prints a message and exits. Never constructs
`UnifiedBridge`. The constructor `smb_unified_bridge()` is never called
in any production path.

**Replacement (Batch C):**

1. Hydrate `OntologyRegistry` against `"WorkOrder"` namespace (Steuerberater domain).
2. Call `smb_unified_bridge(registry, "WorkOrder", SmbRole::Accountant, tenant)`.
3. Thread `Arc<UnifiedBridge<OgitBridge>>` into `SmbOrchestrator::new(bridge)`,
   `MongoConnector::new_with_bridge(client, fp, bridge.clone())`, and
   `LanceConnector::new_with_bridge(root, fp, bridge.clone())`.

### 1.5 smb-woa login flow (crates/smb-woa/src/auth/login_flow.rs)

`authenticate()` validates credentials and mints a Phase-1 JWT. Never calls
`smb_unified_bridge()` — the resulting `ActorContext` is produced with no
OGIT-level audit trace.

**Replacement:** After successful credential validation, optionally emit a
`UnifiedAuditEvent::Auth` to record the authentication event in the audit chain.
Bridge is injected via `LoginFlowConfig { bridge: Option<Arc<UnifiedBridge<OgitBridge>>>, .. }`.
The credential check remains the auth gate; `authorize_read` is NOT called on
login (no entity is accessed). This is an audit hook only.

---

## 2 — Audit Emission: Which Operations Should Emit UnifiedAuditEvents

`UnifiedAuditEvent` (26 bytes, FNV-1a merkle-chained, D-SDR-4 via lance-graph#364)
must be emitted for the following operations:

### 2.1 Super-domain classification

Per `super-domain-rbac-tenancy-v1.md` §4.2:

| Consumer / tenant | Super domain | Notes |
|---|---|---|
| smb-office-rs Steuerberater | `WorkOrderBilling` | Tax / invoice / customer CRUD |
| smb-woa WoA (IT work-orders) | `WorkOrderBilling` + `Networking` | CRUD = WorkOrderBilling; route-handlers = Networking |

`UnifiedAuditEvent.super_domain` MUST stamp `SuperDomain::WorkOrderBilling` for
all Steuerberater-domain operations, and `SuperDomain::Networking` for WoA
route-handler operations.

**Note:** `Networking` discriminant is not yet assigned in `super_domain.rs`.
See §8.1.

### 2.2 Audit emission table

| Operation | Location | Event type | super_domain |
|---|---|---|---|
| `MongoConnector::get` authorized | mongo.rs | `UnifiedAuditEvent::Read` | `WorkOrderBilling` |
| `MongoConnector::upsert` authorized | mongo.rs | `UnifiedAuditEvent::Write` | `WorkOrderBilling` |
| `LanceConnector::get` authorized | lance.rs | `UnifiedAuditEvent::Read` | `WorkOrderBilling` |
| `LanceConnector::upsert` authorized | lance.rs | `UnifiedAuditEvent::Write` | `WorkOrderBilling` |
| `SmbOrchestrator::route` read action | orchestration.rs | `UnifiedAuditEvent::Read` | `WorkOrderBilling` |
| `SmbOrchestrator::route` mutating action | orchestration.rs | `UnifiedAuditEvent::Write` | `WorkOrderBilling` |
| `authenticate()` success | login_flow.rs | `UnifiedAuditEvent::Auth` | `WorkOrderBilling` |
| WoA route handler (future WT-21+) | customer-woa-bin | `UnifiedAuditEvent::Read/Write` | `Networking` |

Audit events are emitted via `bridge.audit_chain().append(event)`. The
`AuditChain` is already shipped in `lance-graph-callcenter` (D-SDR-4).
No new sink or channel is required.

### 2.3 What is NOT audited

- Internal BSON-level schema fingerprint comparisons (too fine-grained)
- Password hashing operations (pre-auth, no data access)
- WAL drain heartbeats (use structured logs)
- Test fixtures (guard with `#[cfg(not(test))]`)

---

## 3 — Backwards-Compat Surface During Retrofit

PR-E2 must be **zero behavior change** for existing call sites (per D-SDR-22).

### 3.1 Incremental strategy (3 independently-mergeable batches)

**Batch A** — Add `bridge: Option<Arc<UnifiedBridge<OgitBridge>>>` to both connectors.
When `None` (existing usage), connectors behave exactly as today. No existing call
sites break; existing test suite is unchanged.

**Batch B** — Wire the orchestrator with `Option<Arc<UnifiedBridge<OgitBridge>>>`.
When `None`, `SmbOrchestrator` routes as today. Add `smb_owl_ids.rs`.

**Batch C** — Replace `Option<Arc<...>>` with `Arc<...>` (mandatory). Update
`customer-woa-bin/src/main.rs` startup. Tighten `smb_unified_bridge()` to accept
`SmbRole` instead of `&'static str`. Add `LoginFlowConfig` in `login_flow.rs`.
The only production caller of the connectors + orchestrator is the binary skeleton,
which is updated in the same commit.

### 3.2 Feature-flag gating (rejected)

Gating bridge wiring behind a Cargo feature would allow silent bypass of the
auth gate. Rejected — the bridge is not optional for production code.

### 3.3 SmbRole ↔ actor_role type tightening (Batch C)

`unified_bridge_wiring.rs` accepts `actor_role: &'static str`. Batch C changes
this to `actor_role: SmbRole`. The existing test passes `"accountant"`, which
becomes `SmbRole::Accountant`. Call sites that pass hardcoded strings get a
compile error that forces them to use the typed catalogue.

---

## 4 — LOC Estimate Per Retrofit Batch

| Batch | Files touched | Net LOC |
|---|---|---|
| **A** — Bridge Option on connectors; authorize + audit gates | `mongo.rs`, `lance.rs`, `lib.rs` | ~160 |
| **B** — Bridge Option on orchestrator; `smb_owl_ids.rs` | `orchestration.rs`, new `smb_owl_ids.rs` | ~120 |
| **C** — Remove Option; binary startup; SmbRole tightening; LoginFlowConfig | `main.rs`, `login_flow.rs`, `unified_bridge_wiring.rs`, `lib.rs` | ~120 |
| **Tests** — deny/allow paths; audit round-trip; integration | test blocks in above files | ~80 |
| **Total** | | **~480 LOC** |

---

## 5 — Registry Hydration and smb_owl_id_for()

### 5.1 Namespace conventions

- Steuerberater tenant (back-office CRUD): namespace `"WorkOrder"` → `WorkOrderBilling`
- WoA tenant (IT work-order service): namespace `"Network"` → `Networking`

Binary entry-point (Batch C) hydrates the registry against both namespaces and
constructs one bridge per tenant context.

### 5.2 smb_owl_ids.rs — new file (Batch B)

`crates/smb-bridge/src/smb_owl_ids.rs`:

```rust
// SMB_FAMILY: WorkOrderBilling family byte.
// PLACEHOLDER 0 — replace with real discriminant once §8.1 resolved.
const SMB_FAMILY: u8 = 0; // TODO(PR-E2 §8.1)

pub fn smb_owl_id_for(entity: &str) -> Option<OwlIdentity> {
    let slot: u16 = match entity {
        "customer" | "kunde" => 1,
        "rechnung"           => 2,
        "mahnung"            => 3,
        "dokument"           => 4,
        "bank"               => 5,
        "fibu"               => 6,
        "steuer"             => 7,
        "lieferant"          => 8,
        "mitarbeiter"        => 9,
        "auftrag"            => 10,
        "angebot"            => 11,
        "zahlung"            => 12,
        "schuldner"          => 13,
        _                    => return None,
    };
    Some(OwlIdentity { family: SMB_FAMILY, slot })
}

pub fn is_mutating(action: &str) -> bool {
    matches!(action, "upsert" | "send" | "submit" | "delete" | "create" | "update")
}
```

The 13 entities mirror `SmbOrchestrator::ACCEPTED_ENTITIES` exactly. Any new
entity added to that list MUST get a slot here.

### 5.3 OwlIdentity uses 3-byte canonical form

Per PR #364 Codex P1 fix: `OwlIdentity { family: u8, slot: u16 }`, canonical
wire = `[family, slot_lo, slot_hi]`. The slot field is `u16` (not `u8`). The
`SMB_FAMILY` byte remains the high byte of the `OwlIdentity` discriminant.

---

## 6 — Dependencies

### 6.1 Sprint-5 W3 / W9 — Family Hydration (td-sdr-family-hydration.md)

`td-sdr-family-hydration.md` defines `new_hydrated(BridgeConfig)` as the production
constructor, plus the TOML seed at `crates/lance-graph-contract/data/family_to_super_domain.toml`.

**PR-E2 Batch C depends on this landing.** Batches A and B can ship using
the existing `new()` constructor.

Canonical spec filename from sprint roadmap: `.claude/specs/pr-d4-family-hydration.md`.
That spec is listed as sprint-log-5-6 W3's deliverable.

### 6.2 td-super-domain-subcrates.md — SmallBizSuperDomain Phase 2

`td-super-domain-subcrates.md` §5 Phase 2 is the blueprint; PR-E2 implements D-SDR-22.
The spec's full dependency chain for `SmallBizSuperDomain::UnifiedBridgeImpl` conformance
(W3→W6→W8→Phase 2) is **beyond PR-E2's scope**. PR-E2 stops at authorize gates + audit
emission using the existing `AuditChain` in `UnifiedBridge` (D-SDR-4), without requiring
`CognitiveStack::for_domain` (W6) or `default_lance_sink` (W8).

### 6.3 SuperDomain discriminant assignment (§8.1 — BLOCKER for Batch C)

`WorkOrderBilling` and `Networking` discriminants must be assigned in
`lance-graph-callcenter::super_domain.rs` before Batch C ships.
Batches A+B can use placeholder `SMB_FAMILY = 0u8`.

### 6.4 SmbBridge dedicated bridge (post-PR-E2)

`unified_bridge_wiring.rs` notes `UnifiedBridge<OgitBridge>` is temporary until a
dedicated `SmbBridge` ships in `lance-graph-ontology::bridges`. That swap is post-PR-E2;
call sites are unchanged because the type parameter is hidden behind `smb_unified_bridge()`.

---

## 7 — DELTA vs super-domain-rbac-tenancy-v1.md §14

### Sub-items EXTENDED by PR-E2 (not new)

| §14 item | PR-E2 role |
|---|---|
| §14.3 "smb-office-rs → retrofit (same)" | PR-E2 IS this retrofit |
| §14.5 D-SDR-22 "zero behavior change" | Implemented via 3-batch incremental plan |
| §14.1 harvest from smb_bridge.rs | PR-E2 validates smb_bridge as harvest source; maps bypass sites |
| §14.2 `woa_bridge.rs` retrofit | §1.5 (login flow) + §2.1 (Networking); woa-rs extraction proper is PR-E3 |

### Sub-items GENUINELY NEW in PR-E2

| New item | Section |
|---|---|
| 5-site bypass inventory with per-site replacement spec | §1 |
| Audit emission table (event type + super_domain per operation) | §2.2 |
| WorkOrderBilling vs Networking cross-domain split for WoA paths | §2.1 |
| smb_owl_id_for() entity→OwlIdentity mapping (13 entities) | §5.2 |
| is_mutating() action classifier for orchestrator | §5.2 |
| SmbRole ↔ actor_role compile-time tightening | §3.3 |
| 3-batch incremental plan with per-batch LOC | §3.1, §4 |
| LoginFlowConfig audit-hook pattern | §1.5 |
| Rejected feature-flag alternative + rationale | §3.2 |
| OwlIdentity u16 slot (3-byte canonical per P1 fix) | §5.3 |

---

## 8 — Open Issues / Blockers

### 8.1 SuperDomain discriminants not assigned (BLOCKER for Batch C)

`super_domain.rs` in `lance-graph-callcenter` must assign discriminants for:
- `WorkOrderBilling` (listed at position 6 in §4.2 but NOT yet in enum code)
- `Networking` (not yet in enum at all)

`SMB_FAMILY` placeholder in `smb_owl_ids.rs` and `super_domain` in every audit
event remain wrong until these are assigned. Resolution: lance-graph-callcenter
owner opens a small PR; PR-E2 Batch C picks up the constants.

### 8.2 smb.ttl OWL file not yet authored

`td-super-domain-subcrates.md` §11 requires `smb-office-rs/ontology/smb.ttl`.
It does not exist. The `smb_owl_id_for()` slot assignments are provisional;
reconcile against `smb.ttl` when it ships.

### 8.3 pr-d4-family-hydration.md execution spec not yet written

Sprint-log-5-6 W3 is responsible for this spec. Batch C of PR-E2 depends on the
implementation, not the spec file — but the spec should exist before an engineer
picks up Batch C.

---

## 9 — File Map

### New files

| File | Batch | Purpose |
|---|---|---|
| `crates/smb-bridge/src/smb_owl_ids.rs` | B | `smb_owl_id_for()` + `is_mutating()` helpers |

### Modified files

| File | Batch | Change summary |
|---|---|---|
| `crates/smb-bridge/src/mongo.rs` | A | Add bridge Option; gate get/upsert; emit `UnifiedAuditEvent` |
| `crates/smb-bridge/src/lance.rs` | A | Mirror of mongo.rs change |
| `crates/smb-bridge/src/orchestration.rs` | B | Add bridge Option; call `smb_owl_id_for` + `authorize_*`; emit audit |
| `crates/smb-bridge/src/lib.rs` | A+B | Expose `smb_owl_ids` module; update re-exports |
| `crates/smb-bridge/src/unified_bridge_wiring.rs` | C | Accept `SmbRole` not `&'static str`; integrate `new_hydrated` path |
| `crates/customer-woa-bin/src/main.rs` | C | Startup: hydrate registry, build bridge, thread to connectors + orchestrator |
| `crates/smb-woa/src/auth/login_flow.rs` | C | `LoginFlowConfig`; emit `UnifiedAuditEvent::Auth` on success |

---

## 10 — Test Plan

**T1** — `MongoConnector` with mock bridge that denies → `BridgeError::AuthorizationDenied`,
no BSON round-trip.

**T2** — `MongoConnector` with mock bridge that allows → `UnifiedAuditEvent::Read`
emitted with `super_domain = WorkOrderBilling`.

**T3** — `LanceConnector` authorize_write gate (mirror T1 for `upsert`).

**T4** — `SmbOrchestrator::route` with deny bridge → `OrchestrationError::RoutingFailed`.

**T5** — `smb_owl_id_for`: all 13 `ACCEPTED_ENTITIES` return `Some`; unknown returns `None`.

**T6** — `authenticate` success with `LoginFlowConfig` → `UnifiedAuditEvent::Auth` emitted.

**T7** — `smb_unified_bridge` accepts all 5 `SmbRole` variants (Batch C).

**I1** — Integration: `smb_unified_bridge` happy path with hydrated registry returns `Ok`.

**I2** — Integration: `SmbOrchestrator` + real bridge → `smb.rechnung.lookup` completes,
exactly one `UnifiedAuditEvent::Read` in the chain.

---

## 11 — PR Sequencing

```
.claude/specs/pr-d4-family-hydration.md (S5-W3 impl)
          │
          ├─ PR-E2 Batch A  (mongo.rs + lance.rs)      ← no blocker
          ├─ PR-E2 Batch B  (orchestration.rs + smb_owl_ids.rs)  ← no blocker, parallel with A
          │
 [§8.1: WorkOrderBilling + Networking discriminants assigned in super_domain.rs]
          │
          ▼
 PR-E2 Batch C  (main.rs + login_flow.rs + SmbRole tightening)
          │
          ▼
 pr-e3-woa-rs-extract (S6-W4)  — woa-rs standalone crate, same bridge pattern
```

---

## 12 — Acceptance Criteria

- [ ] All 5 bypass sites (§1.1–§1.5) gate on `authorize_read` / `authorize_write`
- [ ] `UnifiedAuditEvent` emitted for every operation row in §2.2
- [ ] `super_domain` in events is `WorkOrderBilling` or `Networking`, never `Unknown`
      (requires §8.1)
- [ ] `cargo test -p smb-bridge` green (existing suite unchanged)
- [ ] T1–T7 unit tests pass; I1–I2 integration tests pass
- [ ] `smb_unified_bridge()` accepts `SmbRole`, not `&'static str`
- [ ] `Option<Arc<UnifiedBridge>>` wrappers removed in Batch C
- [ ] `SMB_FAMILY` is the real `WorkOrderBilling` discriminant, not `0` (requires §8.1)
- [ ] `smb_unified_bridge_errors_on_unhydrated_registry` (existing) still passes

---

*End of spec. Estimated ~560 LOC total (480 net + 80 tests).*
*Assign Batches A+B immediately; Batch C after §8.1 + pr-d4-family-hydration land.*
