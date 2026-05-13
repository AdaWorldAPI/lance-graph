# PR-G2: CallcenterSupervisor — ractor actor tree for unified bridge fan-out

**Sprint:** sprint-6 | **Worker:** W11 | **Pattern:** F (compile-time-consumer-binding-v1.md)
**Tech-debt anchor:** TD-RACTOR-SUPERVISOR-5
**Immediate upstream:** PR-G1 (manifest-modules, D-MANIFEST-MODULES — must merge first)
**Downstream consumer:** PR-H5 (SIMD callcenter batch retrofit, vsa_udfs.rs)
**Target crate:** `crates/lance-graph-callcenter/`
**Spec version:** 1 (2026-05-13, W11 sprint-log-5-6)

---

## 1. What this PR replaces

Today `lance-graph-callcenter` routes all consumer dispatch through a single
`UnifiedBridge<B>` generic monomorph (see `src/unified_bridge.rs`). Each
consumer crate (`MedcareBridge`, `OgitBridge`, …) wires its own bridge
independently; there is no shared supervisor, no crash isolation between
consumers, no per-consumer mailbox, and no structured restart strategy.

Concretely: if the medcare consumer's actor panics, the entire callcenter
membrane needs manual restart. If smb-office receives a burst of authorization
requests while medcare is slow, there is no per-consumer backpressure — both
sit behind a single lock (the `AuditChain` `Mutex` in `UnifiedBridge`).

PR-G2 replaces this with a `ractor`-supervised actor tree:

```
CallcenterSupervisor (root, one-for-one supervisor)
    ├── ConsumerActor<MedcareBridge>    (G=2, HEALTHCARE_V1)
    ├── ConsumerActor<OgitBridge>       (G=4, SMB_V1)
    ├── ConsumerActor<WoaBridge>        (G=3, GOTHAM_V1, once wired)
    └── [inert G slots: DOLCE G=0, FMA G=5 — registered, no actor spawned]
```

---

## 2. Actor topology

### 2.1 Granularity

One actor per **active consumer G slot** — not per tenant, not per bridge
instance, not per step-domain. Rationale:

- The `MODULE_TABLE` emitted by PR-G1's build-script indexes consumers by
  `(G, version)` tuple. The supervisor's state keyed on this tuple maps
  cleanly — O(1) lookup on every dispatch.
- Per-tenant actors would require N × |G| actors; N is unbounded at runtime.
  Tenant isolation is already handled by `TenantId` in `UnifiedBridge` and the
  RBAC `Policy::evaluate` chain. Actors do not need to duplicate this.
- Per-step-domain actors are sub-actors within a consumer's own crate, not the
  supervisor's concern. The supervisor owns lifecycle; consumers own computation.

### 2.2 Inert slots

DOLCE (G=0) and FMA (G=5) are registered in OGIT and traversable via
SPARQL/Cypher but have `consumer_pointer = None` in the manifest (per
`inert_when_consumer_absent: true`). The supervisor **skips** inert slots
during spawn (Option A from PR-F-1 CORRECTION, 2026-05-12). A
`Route { g: 5, .. }` for an inert G returns `SupervisorErr::InertG(5)`,
not a panic. SPARQL queries against FMA triples route through the
`OntologyRegistry` directly, bypassing the actor mesh entirely.

### 2.3 Supervision strategy: one-for-one

Each child actor is supervised independently. A panicking medcare actor does
not affect the smb-office actor. This is the correct default for N < 50
consumers (current trajectory: 6 active, see MODULE_TABLE). The plan
documents "restart-all-on-crash" as the v1 simplification; this spec
upgrades that to one-for-one because:

1. The `AuditChain` inside each `UnifiedBridge` is per-bridge-instance; a
   restart wipes only that bridge's merkle chain, not the supervisor's.
2. `ractor`'s `SupervisionEvent::ActorTerminated` already identifies the
   terminated cell — the supervisor can isolate and respawn only that cell.
3. Cross-consumer crash cascade is architecturally unacceptable: MedCare
   carries §73 SGB V / BtM regulatory audit requirements that must remain
   available even if smb-office crashes.

**Decision locked:** one-for-one. Documented as DELTA from
compile-time-consumer-binding-v1.md §3 Open Q 6 (which deferred this choice).

### 2.4 Actor naming

Each child is named `consumer_g_{G}` (e.g. `consumer_g_2` for Healthcare).
This naming survives respawn: a restarted Healthcare actor is still
`consumer_g_2`. The supervisor reverse-index maps `ActorId → G` for the
`handle_supervisor_evt` path.

---

## 3. Message types crossing actor boundaries

### 3.1 Supervisor-level messages

```rust
// crates/lance-graph-callcenter/src/supervisor.rs

pub enum SupervisorMsg {
    /// Route a typed envelope to the actor owning G.
    DispatchToG {
        g: u32,
        version: u32,
        envelope: ConsumerEnvelope,
        reply: ractor::RpcReplyPort<Result<ConsumerReply, SupervisorErr>>,
    },
    /// Health check — returns a summary of all live children.
    Health {
        reply: ractor::RpcReplyPort<SupervisorHealthSummary>,
    },
    /// Graceful shutdown — drains all child mailboxes then stops.
    Shutdown,
    /// Internal: supervisor respawns a dead child by G.
    RespawnG { g: u32, version: u32, crash_count: u32 },
}
```

### 3.2 Per-consumer envelope (the crossing payload)

The envelope carries the gRPC-shaped payload without the tonic wrapper:

```rust
// crates/lance-graph-callcenter/src/consumer_msg.rs

pub enum ConsumerEnvelope {
    Dispatch(DispatchRequest),
    Ingest(IngestRequest),
    Health,
    Qualia(QualiaRequest),
    Styles(StylesRequest),
    // Lab-only arms (behind `--features lab`):
    Tensors(TensorsRequest),
    Calibrate(CalibrateRequest),
    Probe(ProbeRequest),
}

pub enum ConsumerReply {
    Crystal(CrystalResponse),
    Ingest(IngestAck),
    Health(HealthStatus),
    Qualia(Qualia17DResponse),
    Styles(StyleList),
    // Lab-only:
    Tensors(TensorsResponse),
    Calibrate(CalibrateResponse),
    Probe(ProbeResponse),
}
```

These are **not** `UnifiedStep` / `UnifiedAuditEvent`. The crossing payload
is the gRPC request/response pair stripped of its tonic wrapper, not the
internal SPO/semiring substrate shape. The `UnifiedStep` lives further inward;
`ConsumerEnvelope` is the external-membrane-facing message.

### 3.3 Audit events at actor boundaries

`UnifiedAuditEvent` is **not** routed through the supervisor's mailbox.
Audit emission happens inside the `UnifiedBridge::authorize_*` call chain
(D-SDR-5, already shipped in PR #364). The supervisor receives no audit
responsibility; it merely routes the envelope to the child actor that owns
the bridge with the wired `AuditChain`.

**Consequence:** audit events are emitted synchronously, inline, before the
actor's handler returns its reply. This satisfies the §73 SGB V requirement
that authorization decisions be auditable before effect, not after.

### 3.4 What does NOT cross actor boundaries

- `Vsa10k` / `Vsa16kF32` / `RoleKey` / `SemiringChoice` — internal substrate
  types; never in any mailbox. The BBB (blood-brain barrier) invariant from
  callcenter-membrane-v1.md §3 applies: Arrow scalars only cross the membrane.
- `Box<dyn ConsumerActorMsg>` (from PR-F-1 sprint-3 sketch) — dropped in this
  PR in favor of the typed `ConsumerEnvelope` enum. The trait-object approach
  adds ~40 ns per dispatch for no gain once the envelope enum is fixed.

---

## 4. Backpressure and per-actor inbox sizing

### 4.1 Default inbox capacity

**Bounded mailboxes.** Default: **1024 messages per consumer actor.**

Rationale: unbounded mailboxes are an availability footgun under sustained
load. A slow medcare actor must not become an unbounded accumulation site for
Healthcare route requests; the producer (e.g. DrainTask) must receive a
backpressure signal (mailbox-full error) so it can shed load or pause.

### 4.2 Per-consumer override via manifest

The `manifest.yaml` `stack_profile` block gains a `mailbox_capacity` field:

```yaml
# /modules/medcare/manifest.yaml (extension)
stack_profile:
  audit_retention_days: 3650
  requires_fail_closed: true
  escalation: llm
  mailbox_capacity: 512      # override default 1024; tighter for regulated consumer
```

The build-script (PR-G1) emits this as a `u32` constant per `ModuleEntry`;
the supervisor reads it at spawn time via:

```rust
let cap = entry.stack_profile.mailbox_capacity.unwrap_or(DEFAULT_MAILBOX_CAPACITY);
Actor::spawn_linked_with_options(name, actor, args, parent, SpawnOptions::with_mailbox_capacity(cap))
```

### 4.3 Supervisor inbox

The supervisor itself uses an **unbounded** inbox. Rationale: the supervisor
receives at most one message per consumer per request (O(G_active) messages
at once, where G_active <= 50); its handler is trivially fast (lookup +
forward). An unbounded supervisor inbox with bounded child inboxes is the
canonical Erlang/OTP pattern.

### 4.4 Backpressure error surface

When a child mailbox is full, `DispatchToG` returns
`SupervisorErr::MailboxFull(g)` via the `RpcReplyPort`. The caller (typically
`DrainTask`) is responsible for deciding whether to retry, shed, or escalate.
No implicit buffering at the supervisor level.

---

## 5. Failure handling

### 5.1 One-for-one restart

When a child actor terminates (crash or panic), `handle_supervisor_evt`
receives `SupervisionEvent::ActorTerminated(cell, _, reason)`:

```rust
async fn handle_supervisor_evt(
    &self,
    myself: ActorRef<Self::Msg>,
    evt: SupervisionEvent,
    state: &mut Self::State,
) -> Result<(), ActorProcessingErr> {
    if let SupervisionEvent::ActorTerminated(cell, _, reason) = evt {
        if let Some(&(g, version)) = state.reverse_index.get(&cell.get_id()) {
            tracing::warn!(g, ?reason, "consumer actor terminated; scheduling respawn");
            let crash_count = state.slots.get(&(g, version))
                .map(|s| s.crash_count + 1)
                .unwrap_or(1);
            state.slots.remove(&(g, version));
            state.reverse_index.remove(&cell.get_id());
            myself.cast(SupervisorMsg::RespawnG { g, version, crash_count })?;
        }
    }
    Ok(())
}
```

### 5.2 Exponential backoff

Crashes (panics) trigger backoff: 100 ms initial, doubling, capped at 30 s.
Normal terminations (supervisor-initiated `Shutdown` to a child) are NOT
subject to backoff.

Backoff state is per-G in `supervisor::State`:

```rust
pub struct ConsumerSlot {
    pub actor_ref: ActorRef<ConsumerEnvelope>,
    pub crash_count: u32,
    pub last_crash_ts: Option<std::time::Instant>,
}
```

Backoff formula: `min(100ms * 2^crash_count, 30s)`. After the cap is hit and
the actor still crashes, the supervisor emits a `SupervisorErr::ConsumerUnhealthy`
event to a configurable unhealthy hook (default: tracing::error) and stops
retrying until a `ResetCrashCount { g }` message arrives (operator action).

### 5.3 What escalates

**Escalation (to operator):**
- `crash_count > 10` within a 5-minute window on any single G.
- Supervisor's own `pre_start` fails (no registry, no MODULE_TABLE).

**Does not escalate:**
- Single child crash (restarts silently per one-for-one).
- `DispatchToG` for an inert G (returns typed error to caller).
- Child mailbox full (returns backpressure error to caller).

### 5.4 Supervisor crash

If the supervisor itself crashes (which should not happen — its handler
contains no panic paths except the `pre_start` loop), the entire callcenter
restarts. The calling binary (`lance-membrane.rs` outbound boundary) is
responsible for respawning the supervisor. The `DrainTask` and
`LanceVersionWatcher` are downstream of the supervisor; they detect supervisor
absence via the `ActorRef` becoming dead and emit a reconnect request.

---

## 6. Audit integration — actor lifecycle events

### 6.1 Actor lifecycle → UnifiedAuditEvent

Actor start/stop/restart events **do** emit `UnifiedAuditEvent` records, but
only as lifecycle audit entries, not as authorization decisions.

A new `AuthOp` variant is added:

```rust
// crates/lance-graph-callcenter/src/unified_audit.rs (extension)

pub enum AuthOp {
    Read,
    Write,
    Act,
    // Lifecycle events (new in PR-G2):
    ActorStart,    // consumer actor spawned
    ActorStop,     // consumer actor stopped gracefully
    ActorRestart,  // consumer actor restarted after crash
}
```

Lifecycle events are emitted through a dedicated audit chain attached to the
supervisor itself (not to a `UnifiedBridge` instance). This chain uses
`super_domain = SuperDomain::System` and a separate salt.

### 6.2 Which super_domain?

Lifecycle events use `SuperDomain::System` (a new variant, added in PR-G2).
Authorization events continue to use the super_domain wired into each
consumer's `AuditChain` at construction (e.g. `SuperDomain::Healthcare` for
medcare-rs, `SuperDomain::WorkOrderBilling` for woa-rs).

**Rationale:** lifecycle events (actor start/stop) are cross-domain system
events, not domain-specific authorization decisions. Routing them to
`SuperDomain::System` keeps the domain-partitioned audit chains clean.

### 6.3 Lifecycle audit gate

Lifecycle audit emission is controlled by a feature flag:

```toml
# lance-graph-callcenter Cargo.toml
[features]
supervisor-lifecycle-audit = ["audit-log"]
```

Default is **off** to avoid audit noise in test / development environments.
Production deployments enable it via feature flag. The supervisor's
`emit_lifecycle_event` method is a no-op when the feature is disabled (zero
overhead via conditional compilation).

---

## 7. ractor crate: version and feature selection

### 7.1 Version

```toml
ractor = { version = "0.14", default-features = false, features = ["tokio-runtime"] }
```

Notes:
- ractor 0.10 (referenced in compile-time-consumer-binding-v1.md and
  pr-f-1-ractor-supervisor.md) is the prior context; the crate has since
  advanced. As of 2026-05-13, ractor 0.14.x is the latest stable. Version
  constraint: `"0.14"` (minor-compatible pinned).
- `default-features = false` strips the `cluster` feature (distributed actor
  cluster, not needed; adds significant deps).
- `tokio-runtime` is required because ractor's async executor backend is
  tokio-based even in "sync mode" usage. The I-2 invariant is enforced not by
  eliminating tokio from the ractor runtime, but by ensuring the **consumer
  handler bodies** do not import `tokio::spawn` / `tokio::select!` / etc.
  directly (enforced via clippy `disallowed-types`).

### 7.2 I-2 enforcement mechanism

The plan's I-2 invariant ("tokio outbound only, sync ractor inbound") is
enforced mechanically:

1. **clippy.toml** (workspace root, scoped to `lance-graph-callcenter`):
   ```toml
   [[disallowed-types]]
   path = "tokio::sync::Mutex"
   reason = "I-2: use std::sync::Mutex inside the membrane"

   [[disallowed-types]]
   path = "tokio::task::spawn"
   reason = "I-2: spawn only at the outbound boundary (lance_membrane.rs)"

   [[disallowed-types]]
   path = "tokio::time::sleep"
   reason = "I-2: use std::thread::sleep or ractor backoff inside handlers"
   ```
2. **Static assertions** (compile-time):
   ```rust
   // crates/lance-graph-callcenter/tests/supervisor_send_sync_compile.rs
   static_assertions::assert_impl_all!(SupervisorMsg: Send, Sync);
   static_assertions::assert_impl_all!(ConsumerEnvelope: Send, Sync);
   static_assertions::assert_impl_all!(ConsumerReply: Send, Sync);
   ```

### 7.3 Additional new Cargo.toml entries

```toml
# Supervisor feature gate (not always-on; enables the ractor dep)
[features]
supervisor = ["dep:ractor", "dep:static_assertions"]

[dependencies]
ractor            = { version = "0.14", optional = true, default-features = false, features = ["tokio-runtime"] }
static_assertions = { version = "1",    optional = true }
```

The `supervisor` feature is not included in `default`. Callers opt in:
`lance-graph-callcenter = { …, features = ["supervisor"] }`.

---

## 8. File layout

| File | New / Modified | LOC est. | Notes |
|---|---|---|---|
| `src/supervisor.rs` | **NEW** | ~220 | `CallcenterSupervisor`, `SupervisorMsg`, `ConsumerSlot`, `SupervisorState`, `SupervisorHealthSummary`, spawn/respawn/backoff logic |
| `src/consumer_msg.rs` | **NEW** | ~80 | `ConsumerEnvelope` + `ConsumerReply` enums; typed payload crossing boundary |
| `src/actors/mod.rs` | **NEW** | ~30 | Module map for per-consumer actor types; `ConsumerActorMsg` re-export |
| `src/actors/medcare_actor.rs` | **NEW** | ~130 | First concrete `Consumer::Actor` impl (medcare proof-of-concept) |
| `src/lib.rs` | **MODIFIED** | +20 | Add `pub mod supervisor; pub mod consumer_msg; pub mod actors;` + feature gate |
| `Cargo.toml` | **MODIFIED** | +10 | Add `ractor`, `static_assertions` optional deps + `supervisor` feature |
| `../lance-graph-contract/src/consumer.rs` | **MODIFIED** | +30 | `Consumer` trait gets `type Actor: ConsumerActorMsg` + `spawn_actor()` method |
| `../lance-graph-contract/src/consumer_actor_msg.rs` | **NEW** | ~40 | `ConsumerActorMsg` marker trait + `ConsumerMsgKind` enum |
| `clippy.toml` (workspace) | **MODIFIED** | +15 | `disallowed-types` for I-2 scoped to callcenter |
| `tests/supervisor_spawn_active_consumers.rs` | **NEW** | ~60 | Registry seeded with 3 active G; assert 3 children spawned |
| `tests/supervisor_inert_g_denies.rs` | **NEW** | ~30 | `DispatchToG { g: 999 }` → `InertG(999)` error |
| `tests/supervisor_one_for_one_restart.rs` | **NEW** | ~60 | Child panic → respawn; siblings unaffected |
| `tests/supervisor_dispatch_round_trip.rs` | **NEW** | ~40 | End-to-end `ConsumerEnvelope::Health` → `ConsumerReply::Health` |
| `tests/supervisor_send_sync_compile.rs` | **NEW** | ~10 | `static_assertions::assert_impl_all!` on all envelope/reply types |
| `tests/supervisor_lifecycle_audit.rs` | **NEW** | ~40 | With `supervisor-lifecycle-audit` feature: ActorStart events emitted |
| `src/unified_audit.rs` | **MODIFIED** | +25 | New `AuthOp::{ActorStart, ActorStop, ActorRestart}` variants + `SuperDomain::System` |

**Total estimated LOC:** ~820

The estimate is conservative. Subsequent consumer ports after medcare (smb-office,
woa-rs) drop to ~100 LOC each because supervisor + envelope plumbing is already
in place. The plan's 770 LOC estimate (compile-time-consumer-binding-v1.md §2.2)
was for the supervisor alone; this spec includes the lifecycle audit extension
(+25 LOC) and an extra test (`supervisor_lifecycle_audit.rs`, ~40 LOC).

---

## 9. Acceptance criteria

- [ ] `CallcenterSupervisor` actor + `SupervisorMsg` enum land in `supervisor.rs`.
- [ ] `ConsumerEnvelope` + `ConsumerReply` typed enums land in `consumer_msg.rs`.
- [ ] `ConsumerActorMsg` marker trait lands in `lance-graph-contract/src/consumer_actor_msg.rs`.
- [ ] `Consumer` trait in `lance-graph-contract` gains `type Actor: ConsumerActorMsg` + `spawn_actor()`.
- [ ] One-for-one restart confirmed via `supervisor_one_for_one_restart.rs` test.
- [ ] Inert G returns typed `SupervisorErr::InertG` (not panic) via `supervisor_inert_g_denies.rs`.
- [ ] Exponential backoff applies on consecutive crashes (100 ms, doubling, cap 30 s).
- [ ] `Send + Sync` compile proof via `static_assertions::assert_impl_all!` on all envelope/reply types.
- [ ] All 6 test files green in CI under `cargo test -p lance-graph-callcenter --features supervisor`.
- [ ] `clippy.toml` `disallowed-types` rule for `tokio::sync::Mutex` + `tokio::task::spawn` passes.
- [ ] Existing gRPC service trait (`crates/cognitive-shader-driver/src/grpc.rs`) unchanged; `cargo build --features grpc --bin shader-grpc` still compiles.
- [ ] `cargo test -p lance-graph-callcenter` (default features, no `supervisor`) remains green.
- [ ] With `--features supervisor-lifecycle-audit`: `ActorStart` events emitted on spawn; `ActorRestart` on respawn.

---

## 10. LOC estimate

| Concern | LOC |
|---|---|
| supervisor.rs (core supervisor actor) | ~220 |
| consumer_msg.rs (envelope + reply enums) | ~80 |
| actors/ (module + medcare proof actor) | ~160 |
| contract changes (ConsumerActorMsg + Consumer trait extension) | ~70 |
| Cargo.toml + clippy.toml deltas | ~25 |
| unified_audit.rs lifecycle extensions | ~25 |
| 6 test files | ~240 |
| **Total** | **~820 LOC** |

---

## 11. DELTA from reference documents

### 11.1 vs compile-time-consumer-binding-v1.md Pattern F (D-RACTOR-SUPERVISOR)

| Claim in plan | This spec's resolution |
|---|---|
| "restart-all simplest first" (§2.2 sketch) | **Changed to one-for-one** (§5.1 above). Open Q 6 in the plan said "probably yes for N > 10 consumers"; this spec commits one-for-one for v1 given the regulatory audit isolation requirement. |
| `Box<dyn ConsumerActorMsg>` via dynamic dispatch (§3 Open Q 6) | **Changed to typed `ConsumerEnvelope` enum.** Box overhead (~40 ns) is eliminated; the envelope enum is fixed at compile time over the gRPC-shaped arms. |
| `(G, version)` routing key | **Retained.** Both `G` and `version` travel in `DispatchToG`. The supervisor state key is `(u32, u32)`. |
| `Vec::find` reverse lookup in `handle_supervisor_evt` noted as O(N) risk | **Replaced by `HashMap<ActorId, (u32, u32)>`** in `SupervisorState.reverse_index`. The plan flagged this as "fine for N < 50; swap if N grows". This spec makes the O(1) version the baseline. |
| ractor sync mode (feature investigation deferred to engineer) | **Resolved:** `ractor = "0.14"` with `features = ["tokio-runtime"]`. I-2 is enforced via clippy `disallowed-types` scoped to handler bodies, not by removing tokio from the runtime layer. |

### 11.2 vs callcenter-membrane-v1.md

| callcenter-membrane-v1 claim | This spec |
|---|---|
| `DrainTask` routes `UnifiedStep` through `OrchestrationBridge` directly (§D architecture) | Retained. `DrainTask` still feeds `OrchestrationBridge`. The supervisor is a parallel path for inbound external dispatch (`ConsumerEnvelope`), not a replacement for the drain path. |
| `UnifiedBridge<B>` as the single consumer entry point | After PR-G2, `UnifiedBridge<B>` is still the typed authorization surface, but it is **owned by** the per-consumer actor rather than constructed ad-hoc at each call site. |
| Backpressure not specified in DM-6 (DrainTask) | **Specified here:** bounded mailboxes (default 1024); `SupervisorErr::MailboxFull(g)` returned to DrainTask; DrainTask must implement shedding or retry. |

### 11.3 vs pr-f-1-ractor-supervisor.md (sprint-3 spec)

PR-G2 is the sprint-6 execution of Pattern F. Key differences from the sprint-3 spec:

1. **Typed envelope over trait object.** PR-F-1 used `Box<dyn ConsumerActorMsg>`; this spec uses the closed `ConsumerEnvelope` + `ConsumerReply` enums.
2. **Lifecycle audit integration** (§6 above) is new — not in PR-F-1.
3. **`SuperDomain::System`** is a new variant; the sprint-3 spec predated the SuperDomain layer (PR #364, shipped 2026-05-13).
4. **`mailbox_capacity` in manifest** is new — PR-G1 (manifest-modules) was not shipped when PR-F-1 was written.
5. **Test `supervisor_skips_inert_bundles_and_spawns_consumers`** from the PR-F-1 CORRECTION (2026-05-12) is incorporated as `supervisor_spawn_active_consumers.rs` here (with DOLCE + FMA as inert fixtures).

---

## 12. PR dependency graph

```
PR-G1 (manifest-modules, D-MANIFEST-MODULES)
    └── PR-G2 (this PR — CallcenterSupervisor ractor port)
            ├── PR-H5 (SIMD callcenter batch retrofit, vsa_udfs.rs)
            ├── PR-G3 (future: smb-office consumer actor, ~100 LOC)
            └── PR-G4 (future: woa-rs consumer actor, ~100 LOC)
```

PR-G2 also requires as runtime context (already shipped):
- PR #364: `UnifiedAuditEvent` 26-byte canonical, `SuperDomain` type, `AuditChain`
- `crates/lance-graph-ontology`: `OntologyRegistry` + `NamespaceBridge` trait + per-tenant bridges

---

## 13. Open questions for the engineer

1. **ractor 0.14 exact feature matrix.** Verify `cargo tree -p ractor -f "{p} {f}"` to confirm no hidden `cluster` or `remote` transitive deps.

2. **`SuperDomain::System` placement.** Confirm no exhaustive match in existing code breaks. The `SUPER_DOMAINS` static array in `super_domain.rs` needs an entry for `System`.

3. **`ConsumerEnvelope` lab arms.** Decide whether to gate `Tensors / Calibrate / Probe` arms on `--features lab` at the enum level (conditional compilation) or keep always-present. Recommend: always-present, document as lab-only in doc comments.

4. **Medcare audit chain initialization.** The medcare actor needs `UnifiedBridge<MedcareBridge>.with_audit_chain(SuperDomain::Healthcare, salt, sink)`. For v1: accept env var `MEDCARE_AUDIT_SALT`. Sprint-7 hardening PR wires HSM.

5. **`static_assertions` in no-std contexts.** Not relevant here (callcenter is not no-std), but confirm the crate compiles without issues in the workspace CI matrix.

---

## Cross-references

- `.claude/plans/compile-time-consumer-binding-v1.md` — D-RACTOR-SUPERVISOR (originating deliverable, §2.2)
- `.claude/plans/callcenter-membrane-v1.md` — membrane architecture supervised here
- `.claude/specs/pr-f-1-ractor-supervisor.md` — sprint-3 precursor spec (incorporated + extended)
- `.claude/specs/pr-g1-manifest-modules.md` — required upstream (MODULE_TABLE, `mailbox_capacity`)
- `.claude/board/TECH_DEBT.md` — TD-RACTOR-SUPERVISOR-5 (canonical anchor)
- `.claude/board/SINGLE_BINARY_TOPOLOGY.md` — I-2 invariant ("tokio outbound only")
- `crates/cognitive-shader-driver/src/grpc.rs` — 345 LOC proof-shape (message arm origin)
- `crates/lance-graph-callcenter/src/unified_audit.rs` — `AuthOp` + `AuditChain` extended in §6
- `crates/lance-graph-callcenter/src/super_domain.rs` — `SuperDomain::System` added in §6.2
- `.claude/board/sprint-log-5-6/agents/agent-W11.md` — this worker's scratchpad
