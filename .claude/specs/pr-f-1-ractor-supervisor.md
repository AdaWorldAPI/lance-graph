# PR-F-1: ractor supervisor port from gRPC service trait shape

**Tier-2 implementation spec — Pattern F canonical (post-PR #359 letter assignment).**
**Tech-debt anchor:** TD-RACTOR-SUPERVISOR-5.
**Sprint-3 owner:** W6 (this spec) -> engineer pickup.

---

## Goal

Port the proven actor-message shape of
`crates/cognitive-shader-driver/src/grpc.rs` (~345 LOC tonic service trait)
into a **ractor**-supervised tree of per-consumer actors that lives inside
`lance-graph-callcenter`.

- One ractor actor per active consumer `G` (OGIT slot).
- Children supervised one-for-one by `CallcenterSupervisor`.
- Sync mode (per topology invariant **I-2**: tokio is outbound-only and
  must not touch the cognitive substrate or the membrane).
- The supervisor enumerates active `G` slots from the
  `OntologyRegistry` (PR-B-1) and reads each `ConsumerPointer`
  (PR-C-1) at compile time to know which actor type to spawn.
- The gRPC service trait stays exactly where it is — it is **L3 outbound
  lab surface**, not the canonical consumer surface. The supervisor is
  **L2 in-process** and runs in every binary, lab and production.

Once this lands, the callcenter is no longer a single Mutex-guarded
`ShaderDriver` — it is an addressable mesh of per-consumer actors with
crash isolation, mailbox back-pressure, and supervised restart.

---

## Why this shape

The gRPC service trait already proved out the message-arm enumeration
under live testing in the Claude Code backend. Each gRPC handler maps
one-for-one to a ractor `Msg` arm — same payload type in, same response
type out, same error semantics. Mechanical port, not a redesign.

### The shape mapping (from gRPC trait to ractor handler)

| gRPC handler (tonic) | ractor handler arm |
|---|---|
| `dispatch(DispatchRequest) -> CrystalResponse` | `ShaderMsg::Dispatch { req, reply: oneshot<CrystalResponse> }` |
| `ingest(IngestRequest) -> IngestResponse` | `ShaderMsg::Ingest { req, reply }` |
| `health(HealthRequest) -> HealthResponse` | `ShaderMsg::Health { reply }` |
| `qualia(QualiaRequest) -> QualiaResponse` | `ShaderMsg::Qualia { row, reply }` |
| `styles(StylesRequest) -> StylesResponse` | `ShaderMsg::Styles { reply }` |
| `tensors(TensorsRequest) -> TensorsResponse` (lab) | `ShaderMsg::Tensors { req, reply }` (lab-only feature) |
| `calibrate / probe` (lab) | `ShaderMsg::Codec*` (lab-only feature) |

All handler arms ride `oneshot::Sender<T>` (not `tokio::sync::oneshot` —
ractor's own `RpcReplyPort` or `crossbeam_channel::bounded(1)` per **I-2**).

---

## Files to touch

| File | Change |
|---|---|
| `crates/lance-graph-callcenter/src/supervisor.rs` | **NEW** — `CallcenterSupervisor` ractor actor + `SupervisorMsg` enum |
| `crates/lance-graph-callcenter/src/actors/mod.rs` | **NEW** — actor module map; re-exports the per-consumer actor types |
| `crates/lance-graph-callcenter/src/actors/shader_actor.rs` | **NEW** — per-G shader actor (the first concrete `Consumer::Actor`) |
| `crates/lance-graph-callcenter/src/lib.rs` | Add `pub mod supervisor; pub mod actors;` and re-export `CallcenterSupervisor` + `SupervisorMsg` at crate root |
| `crates/lance-graph-callcenter/Cargo.toml` | Add `ractor = { version = "0.10", default-features = false, features = ["async-std"] }` (or sync executor — see Open Q 1); also `static_assertions = "1"` for the Send+Sync compile proof |
| `crates/lance-graph-contract/src/consumer.rs` | `Consumer` trait grows associated `type Actor: ConsumerActorMsg` (the type-system bridge from PR-C-1 ConsumerPointer to the spawned actor) |
| `crates/lance-graph-contract/src/consumer_actor_msg.rs` | **NEW** — `ConsumerActorMsg` marker trait that the per-consumer Msg enums implement; carries `Send + Sync + 'static` plus a `kind()` method for tracing |
| `clippy.toml` (workspace root) | Add `disallowed-types = ["tokio::sync::Mutex", "tokio::sync::oneshot::Sender", ...]` for the `lance-graph-callcenter` crate to enforce I-2 mechanically |

---

## API sketch

```rust
// crates/lance-graph-contract/src/consumer_actor_msg.rs
pub trait ConsumerActorMsg: Send + Sync + 'static {
    /// Static tag for tracing / metrics. One arm per gRPC method
    /// in the proof shape.
    fn kind(&self) -> ConsumerMsgKind;
}

#[derive(Debug, Clone, Copy)]
pub enum ConsumerMsgKind {
    Dispatch,
    Ingest,
    Health,
    Qualia,
    Styles,
    Tensors,   // lab-only
    CodecCal,  // lab-only
    CodecProbe // lab-only
}
```

```rust
// crates/lance-graph-contract/src/consumer.rs (extension)
use crate::consumer_actor_msg::ConsumerActorMsg;

pub trait Consumer: 'static {
    type Actor: ConsumerActorMsg;
    fn pointer(&self) -> &ConsumerPointer;
    fn spawn_actor(&self) -> Box<dyn ActorFactory<Msg = Self::Actor>>;
}
```

```rust
// crates/lance-graph-callcenter/src/supervisor.rs
use std::collections::HashMap;
use std::sync::Arc;
use ractor::{Actor, ActorRef, ActorProcessingErr, SupervisionEvent};
use lance_graph_ontology::OntologyRegistry;
use lance_graph_contract::consumer_actor_msg::ConsumerActorMsg;

pub struct CallcenterSupervisor {
    pub registry: Arc<OntologyRegistry>,
}

pub enum SupervisorMsg {
    DispatchToG {
        g: u32,
        msg: Box<dyn ConsumerActorMsg>,
        reply: ractor::RpcReplyPort<Result<(), SupervisorErr>>,
    },
    Health { reply: ractor::RpcReplyPort<HealthSummary> },
    Shutdown,
}

#[derive(Debug, thiserror::Error)]
pub enum SupervisorErr {
    #[error("inert G slot: {0}")]
    InertG(u32),
    #[error("child mailbox full: g={0}")]
    MailboxFull(u32),
    #[error("ractor: {0}")]
    Ractor(#[from] ractor::MessagingErr<()>),
}

#[ractor::async_trait]
impl Actor for CallcenterSupervisor {
    type Msg = SupervisorMsg;
    type State = HashMap<u32, ActorRef<Box<dyn ConsumerActorMsg>>>;
    type Arguments = Arc<OntologyRegistry>;

    async fn pre_start(
        &self,
        myself: ActorRef<Self::Msg>,
        registry: Self::Arguments,
    ) -> Result<Self::State, ActorProcessingErr> {
        let mut children = HashMap::new();
        // Walk every active G in the registry. For each one, look up the
        // ConsumerPointer (PR-C-1), instantiate its actor type, and spawn
        // it linked to ourselves so we get SupervisionEvent on crash.
        for g in registry.active_g_list() {
            let bundle = registry.resolve(g)
                .ok_or_else(|| ActorProcessingErr::from(format!("inert g={}", g)))?;
            let pointer = bundle.consumer_pointer.as_ref()
                .ok_or_else(|| ActorProcessingErr::from(format!("g={} has no ConsumerPointer", g)))?;
            let factory = pointer.actor_factory.clone();
            let (actor_ref, _join) = Actor::spawn_linked(
                Some(format!("consumer_g_{}", g)),
                factory.actor(),
                factory.args(),
                myself.get_cell(),
            ).await?;
            children.insert(g, actor_ref);
        }
        Ok(children)
    }

    async fn handle(
        &self,
        _myself: ActorRef<Self::Msg>,
        msg: Self::Msg,
        children: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match msg {
            SupervisorMsg::DispatchToG { g, msg, reply } => {
                if let Some(child) = children.get(&g) {
                    child.cast(msg).map_err(|_| SupervisorErr::MailboxFull(g))?;
                    reply.send(Ok(())).ok();
                } else {
                    reply.send(Err(SupervisorErr::InertG(g))).ok();
                }
                Ok(())
            }
            SupervisorMsg::Health { reply } => {
                reply.send(HealthSummary::from_children(children)).ok();
                Ok(())
            }
            SupervisorMsg::Shutdown => Ok(()),
        }
    }

    async fn handle_supervisor_evt(
        &self,
        myself: ActorRef<Self::Msg>,
        evt: SupervisionEvent,
        children: &mut Self::State,
    ) -> Result<(), ActorProcessingErr> {
        match evt {
            SupervisionEvent::ActorTerminated(cell, _, _) => {
                tracing::warn!(actor = ?cell.get_id(), "consumer actor terminated; respawning");
                self.respawn_dead_child(myself, children, cell).await
            }
            SupervisionEvent::ActorPanicked(cell, panic_msg) => {
                tracing::error!(actor = ?cell.get_id(), panic = %panic_msg,
                    "consumer actor panicked; respawning with backoff");
                tokio_or_std_sleep(self.backoff_for(cell.get_id())).await;
                self.respawn_dead_child(myself, children, cell).await
            }
            _ => Ok(()),
        }
    }
}
```

The `respawn_dead_child` helper looks the dead actor's `G` up in
`children` (reverse lookup by cell id), drops the stale `ActorRef`,
re-resolves the `ConsumerPointer` from the registry (in case a hot
reload happened in between), and `Actor::spawn_linked`s a fresh
child. If the registry no longer reports the `G` as active, the slot
is left empty — that is the explicit "withdraw a consumer at runtime"
path.

---

## I-2 enforcement (mechanical, not just docstring)

Per `SINGLE_BINARY_TOPOLOGY.md`, **tokio is outbound only** — no
`async fn` in the cognitive substrate or the callcenter membrane.
Two enforcement mechanisms:

1. **Compile-time:** `static_assertions::assert_impl_all!(SupervisorMsg: Send, Sync);`
   plus a `tests/supervisor_send_sync_compile.rs` that asserts every
   per-consumer `Msg` enum is `Send + Sync` without depending on a
   tokio runtime.
2. **Lint-time:** `clippy.toml` workspace addition listing
   `tokio::sync::*` types in the `disallowed-types` set scoped to
   `lance-graph-callcenter`. Anyone reaching for tokio inside the
   membrane gets a clippy error in CI.

(Tokio still appears in the *outbound* WebSocket / Postgrest / Lance
sink modules of `lance-graph-callcenter`, behind the existing
`realtime` feature flag. Those modules are downstream of the supervisor
— the supervisor itself never awaits tokio.)

---

## Test plan

| Test file (under `crates/lance-graph-callcenter/tests/`) | What it proves |
|---|---|
| `supervisor_spawn_active_consumers.rs` | Registry seeded with 3 active G; `CallcenterSupervisor::pre_start` returns a `State` with 3 children, each registered under `consumer_g_<N>` name |
| `supervisor_inert_g_denies.rs` | `DispatchToG { g: 999, .. }` for a G not in the registry returns `SupervisorErr::InertG(999)` via the reply oneshot; supervisor remains running |
| `supervisor_one_for_one_restart.rs` | Spawn 3 children; cause child 2 to panic; assert (a) supervisor receives `ActorPanicked`, (b) a fresh child for G=2 is spawned, (c) children 1 and 3 are unaffected (same `ActorId`) |
| `supervisor_dispatch_round_trip.rs` | Send a `ShaderMsg::Health` to the supervisor's `DispatchToG`; receive a `HealthResponse` via the reply oneshot end-to-end |
| `supervisor_send_sync_compile.rs` | `static_assertions::assert_impl_all!(SupervisorMsg: Send, Sync); assert_impl_all!(ShaderActorMsg: Send, Sync);` — compile fails if anyone introduces a `!Send` or `!Sync` field. Type-system proof of I-2. |

**Plus** the existing gRPC service-trait callers (`shader-grpc` lab
binary, `examples/grpc_dispatch.rs`) MUST stay green — the gRPC
trait does not change shape; the supervisor just gives us a second,
in-process consumer of the same handler arms.

---

## Dependencies (PR-graph upstream of this PR)

| Dep | What it provides for PR-F-1 |
|---|---|
| **PR-B-1** (W3) | `ContextBundle` + `OntologyRegistry::resolve(g)`. The supervisor enumerates active G slots through the registry. |
| **PR-C-1** (W4) | `ConsumerPointer` (which carries the `actor_factory` field) + `GenericBridge`. The supervisor reads the pointer to know what concrete actor to spawn for each G. |
| **PR-E-1** (W5) | `manifest.yaml` per-consumer metadata (mailbox capacity, restart policy, backoff). The supervisor reads `stack_profile` out of the manifest at startup. |
| **External:** `ractor = "0.10"` (sync mode features, see Open Q 1). |
| **External:** `static_assertions = "1"` for compile-time Send+Sync proof. |

Downstream of this PR: PR-D-1 (W9 FMA OWL hydrator) wires its first
real consumer into the supervisor; the consumer template (W8) uses
the supervisor as its scaffolding target.

---

## Acceptance criteria

- [ ] `CallcenterSupervisor` actor + `SupervisorMsg` enum + `ConsumerActorMsg` marker trait land in their target paths.
- [ ] One-for-one restart on child crash with bounded exponential backoff (default: 100 ms, doubling, capped at 30 s, configurable per consumer via `manifest.yaml`).
- [ ] `Send + Sync` compile-time proof via `static_assertions::assert_impl_all!` on every `Msg` enum (per **I-2**).
- [ ] Five test files all green in CI.
- [ ] Existing gRPC service trait + lab binary unchanged in shape (regression-free at L3).
- [ ] Backwards-compat: `cargo run --features grpc --bin shader-grpc` still serves the proof-shape protobuf interface.
- [ ] `clippy.toml` `disallowed-types` rule for `tokio::sync::*` inside the membrane lands and is exercised in CI.

---

## Effort

Large — ~400 LOC across supervisor.rs (~180), shader_actor.rs (~120),
mod.rs glue (~20), ConsumerActorMsg trait (~30), Cargo + clippy
deltas (~20), plus 5 test files (~150 LOC combined). ~3 engineer
days, dominated by ractor 0.10 sync-mode familiarisation and
shader-actor message arm porting.

---

## Open questions for the engineer

1. **ractor sync mode vs tokio mode.** ractor 0.10 has both. Per **I-2**
   the supervisor must run in a sync executor (no tokio in the
   membrane). Verify ractor 0.10 features cleanly disable the tokio
   dependency. If not — fall back to a hand-rolled `crossbeam`-channel
   actor harness keyed off the same `Msg` enum (the message shape is
   what we are committing to; the runtime is replaceable).
2. **Send+Sync compile proof.** Use `static_assertions::assert_impl_all!`
   in a `tests/supervisor_send_sync_compile.rs` module. Trait-bound
   tests (e.g., `fn assert_send<T: Send + Sync>() {} fn _t() { assert_send::<SupervisorMsg>(); }`)
   are equivalent but `static_assertions` reads better and surfaces
   the failure in `cargo test`'s output without a runtime path.
3. **Restart strategy: one-for-one (default) vs all-for-one (cascade).**
   Recommend **one-for-one**: a panicking medcare consumer must not
   take down the unrelated smb-office consumer. That is the BEAM /
   Erlang heritage and matches the per-consumer crash isolation we
   designed Pattern E around.
4. **Mailbox capacity: bounded vs unbounded.** Recommend **bounded**
   per consumer (default: 1024 messages), configurable in
   `manifest.yaml` `stack_profile.mailbox_capacity`. Unbounded
   mailboxes are an availability footgun under load.
5. **Tracing: per-message span vs aggregated counter.** Recommend
   **feature-gated** under a `tracing-supervisor` feature in
   `lance-graph-callcenter`. Per-message spans are invaluable
   when debugging actor deadlocks but cost ~1 µs per dispatch in
   hot paths. Default to off; flip on for staging / debug builds.
6. **`Box<dyn ConsumerActorMsg>` vs typed dispatch.** The sketch above
   uses `Box<dyn ConsumerActorMsg>` so the supervisor can route to
   any consumer's actor. A per-G typed dispatch would skip the box
   but require the supervisor to know every consumer crate's `Msg`
   type at compile time, defeating the dynamic registry. The box
   allocation cost is ~40 ns per message — negligible against the
   actor mailbox push.
7. **Hot reload of `ConsumerPointer`.** The current sketch re-resolves
   the pointer on respawn but not on every dispatch. If the registry
   swaps a consumer's pointer at runtime (PR-D-1 OWL hydrator
   reload), in-flight messages still reach the old actor. Decide
   whether to drain the old mailbox on swap (preferred) or let it
   complete (simpler). Recommend draining once the supervisor reads
   the new pointer.

---

## Cross-references

- `.claude/plans/compile-time-consumer-binding-v1.md` — D-RACTOR-SUPERVISOR (the deliverable that called this spec out).
- `.claude/board/TECH_DEBT.md` — TD-RACTOR-SUPERVISOR-5 (the canonical TD anchor).
- `crates/cognitive-shader-driver/src/grpc.rs` — the ~345 LOC tonic service trait that proves the actor message shape.
- `.claude/board/SINGLE_BINARY_TOPOLOGY.md` — **I-2** invariant ("tokio outbound only") that this spec enforces mechanically.
- `.claude/specs/pr-b-1-context-bundle.md` (W3 sister; required upstream).
- `.claude/specs/pr-c-1-generic-bridge.md` (W4 sister; required upstream).
- `.claude/specs/pr-e-1-manifest-modules.md` (W5 sister; required upstream).
- `.claude/specs/sprint-3-execution-plan.md` (W1 master).
- `.claude/board/sprint-log-3/agents/agent-W6.md` (this agent's log).

---

## CORRECTION (2026-05-12, PR #360 review)

**Defect:** The original `pre_start` loop sketched in this spec iterates over `registry.active_g_list()` and unwraps `bundle.consumer_pointer` — but inert bundles (DOLCE G=0, FMA G=5) have `consumer_pointer = None` by design. Per the W11 smoke test spec, DOLCE must remain registered as inert context (no actor) while Healthcare spawns its actor. The original loop would either panic on `unwrap()` or return `ActorProcessingErr` and abort `pre_start` before any consumer actor spawns.

**Fix:** Skip inert bundles in the supervisor's spawn loop. Two equivalent options:

### Option A — explicit filter inside `pre_start` (recommended)

```rust
async fn pre_start(
    &self,
    myself: ActorRef<Self::Msg>,
    registry: Self::Arguments,
) -> Result<Self::State, ActorProcessingErr> {
    let mut children = HashMap::new();
    for g in registry.all_registered_g() {
        let bundle = registry.resolve(g).expect("registered g must resolve");

        // SKIP inert bundles — DOLCE / FMA / unconsumed ontologies are
        // queryable via SPARQL/Cypher but have no executable behavior.
        let pointer = match bundle.consumer_pointer.as_ref() {
            Some(p) => p,
            None => {
                tracing::debug!("g={} is inert (no consumer_pointer); skipping spawn", g);
                continue;
            }
        };

        let (actor_ref, _handle) = Actor::spawn_linked(
            Some(format!("consumer_g_{}", g)),
            pointer.actor_type.spawn(),
            (),
            myself.get_cell(),
        ).await?;
        children.insert(g, actor_ref);
    }
    Ok(children)
}
```

### Option B — narrow the iterator's contract

Rename `active_g_list()` to `active_consumer_g_list()` and have it return ONLY G slots whose bundle has `consumer_pointer.is_some()`. The supervisor loop becomes:

```rust
for g in registry.active_consumer_g_list() {
    let bundle = registry.resolve(g).unwrap();
    let pointer = bundle.consumer_pointer.as_ref().unwrap();  // safe by iterator contract
    // ... spawn
}
```

Plus a sibling iterator `inert_g_list()` for SPARQL/Cypher consumers who need read access to all G (active + inert).

**Recommendation:** Option A — explicit filter — surfaces the inert-vs-active distinction at the spawn site (debugging clarity > iterator API minimalism).

**New test for the fix** (extends PR-F-1 test plan):

```rust
#[tokio::test]
async fn supervisor_skips_inert_bundles_and_spawns_consumers() {
    // Registry seeded with: DOLCE (inert), Healthcare (active), FMA (inert)
    let registry = test_registry_with_inert_and_active();
    let (sup_ref, _handle) = Actor::spawn(
        Some("test_sup".into()),
        CallcenterSupervisor { registry: registry.clone(), children: HashMap::new() },
        registry.clone(),
    ).await.unwrap();

    // Supervisor MUST be running (not aborted)
    assert_eq!(sup_ref.get_status(), ActorStatus::Running);

    // Healthcare actor MUST exist; DOLCE / FMA actors MUST NOT exist
    assert!(supervisor_has_g(&sup_ref, OGIT::HEALTHCARE_V1.0).await);
    assert!(!supervisor_has_g(&sup_ref, OGIT::DOLCE_V1.0).await);
    assert!(!supervisor_has_g(&sup_ref, OGIT::FMA_V1.0).await);
}
```

This test also covers W11's smoke-test expectation that DOLCE is queryable but not spawned.

**Provenance:** flagged by user during PR #360 review.
