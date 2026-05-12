# Compile-Time Consumer Binding — Tier-2 Sub-Plan (Patterns E + F)

**Status:** Active (sprint-2, 2026-05-12)
**Worker:** W11
**Branch:** `claude/unified-ogit-architecture-synthesis`
**Tier:** 2 (depends on W10 Tier-1: `ogit-g-context-bundle-v1.md`)
**Master plan:** `unified-ogit-architecture-v1.md` (W1)
**Sibling proof plan:** `anatomy-realtime-v1.md` (W12 — consumes both deliverables here)
**TECH_DEBT rows:** TD-MANIFEST-MODULES-4, TD-RACTOR-SUPERVISOR-5 (W5)

---

## 1. Motivation

W10's Tier-1 (SPO-G slot + `ContextBundle` + `GenericBridge`) makes it *possible* to register a consumer as runtime data: a `ConsumerPointer` lives in the OGIT registry, every triple carries a G-tag, and a single generic dispatch surface fans out to whichever consumer owns G. That is necessary but not sufficient. Three load-bearing questions remain:

1. **How does a consumer self-declare its `(G, version, entity_types, rbac_policy, action_capabilities, stack_profile, actor_type, thinking_styles)` bundle at *compile time*, without hand-edited registry boilerplate in `lance-graph-contract`?** Today every new consumer requires touching the contract crate to add constants, register entity-type ranges, wire RBAC defaults, and patch the dispatch enum. That is the dependency-inversion violation Pattern E exists to fix.

2. **How do consumer actors get supervised at runtime — with crash isolation, restart strategy, and a well-typed message contract — without dragging tokio into the actor handler bodies (topology invariant I-2: tokio outbound only, sync ractor inbound)?** Pattern F: ractor sync mode + a `CallcenterSupervisor` that enumerates consumers from the compile-time manifest table generated in Pattern E and routes through `GenericBridge`.

3. **How is schema evolution handled without a stop-the-world flag day?** `(G, version)` tuples: `HEALTHCARE_V1` and `HEALTHCARE_V2` coexist in OGIT; consumers compiled against v1 keep working; new consumers target v2; deprecations are loud and explicit at the version-constant level.

The shape is not new. PostNuke (2002-2008) shipped `/modules/<name>/manifest.yaml` as a directory-as-module declaration; Drupal `.info.yml`, Symfony bundles, Cargo workspaces, and even VSCode extensions all converged on the same shape. We adopt it. The naming `lance-graph-callcenter` was deliberate from day one: telephony switching, supervised processes, per-line crash isolation — that crate already *is* the supervisor's home; we just need to make the supervision concrete.

The gRPC service trait in `crates/cognitive-shader-driver/src/grpc.rs` (345 LOC) already proves the actor message shape. Each tonic service method (`dispatch`, `ingest`, `qualia`, `styles`, `health`, `tensors`, `calibrate`, `probe`) maps 1:1 to a ractor handler arm; the request/response structs become typed messages. The translation is mechanical.

---

## 2. Deliverables

Two independently shippable deliverables, each with its own PR. D-MANIFEST-MODULES must merge first because D-RACTOR-SUPERVISOR enumerates consumers from the compile-time table the manifest build-script emits.

### 2.1 D-MANIFEST-MODULES — PostNuke-style `/modules/<name>/manifest.yaml` + build-script

**Goal:** Make adding a new consumer = drop a manifest under `/modules/` + add a Cargo dep + write `~30 LOC` of consumer-crate glue (`impl Consumer for FooActor`). Zero edits to `lance-graph-contract`'s hand-written source after this lands.

**File layout (workspace root):**

```
/modules/
  dolce/manifest.yaml         # G=0,    DOLCE_V0,        root context, always-present, inert=false
  medcare/manifest.yaml       # G=2,    HEALTHCARE_V1,   active (medcare-rs crate present)
  q2-cockpit/manifest.yaml    # G=3,    GOTHAM_V1,       active (q2 workspace)
  smb-office/manifest.yaml    # G=4,    SMB_V1,          active (smb-office-rs)
  fma/manifest.yaml           # G=5,    FMA_V1,          INERT (no consumer crate; triples + SPARQL only)
  hubspo/manifest.yaml        # G=6,    CRM_V1,          future, Claude-Code-reverse-engineered, currently inert
```

**Manifest schema** (one example, all six follow the same shape, ~30 LOC each):

```yaml
# /modules/medcare/manifest.yaml
ogit_g: HEALTHCARE
version: 1
domain_name: medcare
inert_when_consumer_absent: false      # if true: register G but skip actor spawn

entity_types:
  Patient:      { code: 100, parent: dolce.Person }
  Diagnosis:    { code: 101, parent: dolce.SocialObject }
  LabResult:    { code: 102, parent: dolce.Quality }
  Prescription: { code: 103, parent: dolce.SocialObject }
  Anamnese:     { code: 104, parent: dolce.Information }
  Ueberweisung: { code: 105, parent: dolce.SocialAct }

rbac_policy: medcare_policy
stack_profile:
  audit_retention_days: 3650
  requires_fail_closed: true
  escalation: llm

action_capabilities:
  finalize_diagnosis:     escalate
  issue_btm_prescription: escalate
  anonymize_patient:      escalate
  read_lab:               permit
  read_anamnese:          permit_with_audit

actor:
  crate: medcare-rs
  type: MedCareActor
  message_type: MedCareMessage

thinking_styles_inherited_from: dolce
thinking_styles_added: [differential, evidence_based, risk_stratified]
```

**Build-script** (`crates/lance-graph-contract/build.rs`, target ~150 LOC):

1. **Scan** `${CARGO_MANIFEST_DIR}/../../modules/*/manifest.yaml` (canonical glob).
2. **Parse** each via `serde_yaml::from_str::<ManifestSchema>(...)` with `#[serde(deny_unknown_fields)]` so a typo fails compile, never silently misregisters.
3. **Validate**: G uniqueness across modules; entity-type code uniqueness within G; `actor.crate` resolvable as a workspace member when `inert_when_consumer_absent: false`; `parent` references resolve to a known DOLCE class or an entity in an earlier-loaded G.
4. **Emit** `${OUT_DIR}/ogit_modules_gen.rs`:
   ```rust
   pub mod OGIT {
       pub const DOLCE_V0:      (u32, u32) = (0, 0);
       pub const HEALTHCARE_V1: (u32, u32) = (2, 1);
       pub const GOTHAM_V1:     (u32, u32) = (3, 1);
       pub const SMB_V1:        (u32, u32) = (4, 1);
       pub const FMA_V1:        (u32, u32) = (5, 1);
       pub const CRM_V1:        (u32, u32) = (6, 1);
   }
   pub const MODULE_TABLE: &[ModuleEntry] = &[ /* … */ ];
   ```
   plus `Consumer` trait registration shims keyed by `(G, version)`. The consumer crate's `lib.rs` provides the actor body; the contract crate generates the *registration shim* that hands the `ConsumerPointer` to the supervisor.
5. **Detect peers**: walk the workspace `Cargo.toml` to determine which `actor.crate` values are actually members. Emit per-G `ACTIVE` vs `INERT` markers. `FMA_V1` ships INERT in v1 of this plan; `CRM_V1` ships INERT pending HubSpo Pattern.
6. **Idempotency**: re-running with no manifest change must produce a byte-identical `ogit_modules_gen.rs` (`cargo:rerun-if-changed=../../modules/`).

**Effort:** ~150 LOC build-script + 6 × ~30 LOC manifests + ~80 LOC test fixtures = ~410 LOC + 5 unit tests + 1 integration test that verifies a synthetic 7th manifest registers cleanly.

**Acceptance for D-MANIFEST-MODULES:**

- `cargo build -p lance-graph-contract` produces `ogit_modules_gen.rs` with all 6 manifests parsed.
- `cargo test -p lance-graph-contract --test manifest_parse` covers: deny_unknown_fields, duplicate-G rejection, entity-code collision rejection, inert-crate-missing rejection.
- Re-build with no changes hits the cache: `cargo:rerun-if-changed` confirmed via `CARGO_LOG=cargo::core::compiler::fingerprint=info`.
- A 7th manifest (`/modules/test-fixture/manifest.yaml`) added in a test workspace builds without source edits to `lance-graph-contract`.

### 2.2 D-RACTOR-SUPERVISOR — Port gRPC service trait shape to ractor

**Goal:** Replace the 1:N hand-wired call sites in `lance-graph-callcenter` with a single `CallcenterSupervisor` ractor that owns the consumer registry, spawns each active consumer on boot (per the compile-time manifest table), and routes typed messages to the right addr — all in ractor sync mode (I-2 enforced).

**Mapping from `crates/cognitive-shader-driver/src/grpc.rs` to ractor:**

| gRPC method (tonic)                              | ractor handler arm                              |
|--------------------------------------------------|-------------------------------------------------|
| `dispatch(DispatchRequest) → CrystalResponse`    | `handle(Dispatch{..}) → Reply<Crystal>`         |
| `ingest(IngestRequest) → IngestResponse`         | `handle(Ingest{..}) → Reply<IngestAck>`         |
| `qualia(QualiaRequest) → QualiaResponse`         | `handle(Qualia{..}) → Reply<Qualia17DResponse>` |
| `styles(StylesRequest) → StylesResponse`         | `handle(Styles{..}) → Reply<StyleList>`         |
| `health(HealthRequest) → HealthResponse`         | `handle(Health) → Reply<HealthStatus>`          |
| `tensors / calibrate / probe`                    | corresponding `Tensors / Calibrate / Probe` arms|

The translation is mechanical because the gRPC trait was already designed sync-shaped: every method takes an owned request, returns a `Result<Response<T>, Status>`, no streaming, no long-lived state borrow. ractor sync mode satisfies the same constraints.

**Sketch (`crates/lance-graph-callcenter/src/supervisor.rs`, ~400 LOC):**

```rust
use ractor::{Actor, ActorRef, ActorProcessingErr};
use std::collections::HashMap;
use std::sync::Arc;
use lance_graph_contract::ogit::{OGIT, MODULE_TABLE, ModuleEntry, ConsumerPointer};

pub struct CallcenterSupervisor {
    registry: Arc<OntologyRegistry>,
    bridge:   Arc<GenericBridge>,           // from W10 Tier-1
}

pub enum SupervisorMsg {
    SpawnConsumers,
    Route { g: u32, version: u32, msg: ConsumerEnvelope, reply: ractor::RpcReplyPort<ConsumerReply> },
    Shutdown,
    ConsumerCrashed { g: u32, version: u32, err: String },
}

pub struct CallcenterState {
    consumers: HashMap<(u32, u32), ActorRef<ConsumerEnvelope>>,
    inert:     Vec<(u32, u32)>,            // registered in OGIT, no actor spawned
}

impl Actor for CallcenterSupervisor {
    type Msg   = SupervisorMsg;
    type State = CallcenterState;
    type Arguments = ();

    async fn pre_start(&self, _myself: ActorRef<Self::Msg>, _args: ()) -> Result<Self::State, ActorProcessingErr> {
        Ok(CallcenterState { consumers: HashMap::new(), inert: Vec::new() })
    }

    async fn handle(&self, myself: ActorRef<Self::Msg>, msg: Self::Msg, state: &mut Self::State)
        -> Result<(), ActorProcessingErr>
    {
        match msg {
            SupervisorMsg::SpawnConsumers => {
                for entry in MODULE_TABLE {
                    if entry.inert {
                        state.inert.push((entry.g, entry.version));
                        continue;
                    }
                    let cp = self.registry.lookup_consumer_pointer(entry.g, entry.version)?;
                    let (addr, _join) = ractor::Actor::spawn_linked(
                        Some(cp.domain_name.to_string()),
                        ConsumerActorFactory::for_pointer(cp.clone()),
                        cp.initial_state(),
                        myself.get_cell(),
                    ).await?;
                    state.consumers.insert((entry.g, entry.version), addr);
                }
                Ok(())
            }
            SupervisorMsg::Route { g, version, msg, reply } => {
                let addr = state.consumers.get(&(g, version))
                    .ok_or_else(|| ActorProcessingErr::from(format!("no consumer for ({g},{version})")))?;
                // GenericBridge does the typed cast; supervisor just routes.
                self.bridge.dispatch_via(addr, msg, reply)?;
                Ok(())
            }
            SupervisorMsg::ConsumerCrashed { g, version, err } => {
                tracing::warn!(?g, ?version, ?err, "consumer crashed; restarting");
                state.consumers.remove(&(g, version));
                myself.cast(SupervisorMsg::SpawnConsumers)?;  // restart-all simplest first
                Ok(())
            }
            SupervisorMsg::Shutdown => {
                for (_, addr) in state.consumers.drain() {
                    addr.stop(Some("supervisor shutdown".into()));
                }
                Ok(())
            }
        }
    }

    async fn handle_supervisor_evt(&self, _myself: ActorRef<Self::Msg>, evt: ractor::SupervisionEvent, state: &mut Self::State)
        -> Result<(), ActorProcessingErr>
    {
        if let ractor::SupervisionEvent::ActorTerminated(cell, _, reason) = evt {
            if let Some(((g, v), _)) = state.consumers.iter().find(|(_, a)| a.get_id() == cell.get_id()).map(|(k, _)| (*k, ())) {
                let _ = _myself.cast(SupervisorMsg::ConsumerCrashed { g, version: v, err: reason.unwrap_or_default() });
            }
        }
        Ok(())
    }
}
```

**Sync-mode enforcement.** ractor's `concurrency::sync` feature gate is used; the consumer actor handler bodies do *not* import `tokio::spawn`, `tokio::select!`, or any tokio runtime primitive. Outbound I/O (HTTP, gRPC client calls, Postgres queries) is done from the supervisor's outbound side via a dedicated tokio runtime owned by the `lance_membrane.rs` boundary — never inside the consumer handler. CI lints this with a clippy `disallowed-types` rule scoped to the `consumers::*` modules.

**Effort:** ~400 LOC supervisor + ~250 LOC first consumer port (medcare, as the proof) + ~120 LOC test harness = ~770 LOC. Subsequent consumer ports (`q2-cockpit`, `smb-office`) drop to ~100 LOC each because the supervisor + envelope plumbing is already there.

**Acceptance for D-RACTOR-SUPERVISOR:**

- `cargo build -p lance-graph-callcenter` produces the supervisor + medcare consumer actor.
- Integration test `roundtrip_dispatch`: send `Dispatch{g: HEALTHCARE_V1.0, version: HEALTHCARE_V1.1, ..}` → receive `Crystal` reply within ractor sync timeout.
- Crash test `consumer_crashes_restarts`: medcare actor panics inside handler → supervisor receives `ActorTerminated` → emits `ConsumerCrashed` → re-spawn succeeds; integration test asserts second dispatch returns 200.
- Lint test `no_tokio_in_handlers`: clippy run with `disallowed-types = ["tokio::runtime::*"]` scoped to `consumers/` passes.
- Inert manifest test `fma_traversable_no_actor`: SPARQL query against `FMA_V1` triples returns rows; `Route { g: FMA_V1.0, .. }` returns a typed `NoConsumer` error, not a panic.

---

## 3. Open design questions

1. **Build-script home: `lance-graph-contract` or new `lance-graph-modules` crate?**
   *Recommendation: contract.* Every consumer already depends on `lance-graph-contract`; spinning up a new crate just to host one build-script is over-modular. The build-script is ~150 LOC and its outputs are co-located with the consumer-pointer type definitions that consumers already import. Risk: contract's clean build graph picks up `serde_yaml` as a build-dep — acceptable, build-deps don't leak into the runtime closure.

2. **Manifest schema validation — strictness vs. evolvability.**
   *Recommendation: `#[serde(deny_unknown_fields)]` on every manifest struct, but bump `version:` for any breaking field addition.* A typo today must fail compile; a deliberate schema change tomorrow must bump the manifest's own meta-schema version (separate field from `version:` which is the *ontology* version). Two-axis versioning. Annoying once a year; saves silent-misregistration class of bugs forever.

3. **Versioning rollover — coexistence vs. flag-day.**
   *Recommendation: coexistence is mandatory.* When `HEALTHCARE_V1` → `HEALTHCARE_V2` lands, both `(2, 1)` and `(2, 2)` stay registered in the OGIT table; consumers compiled against v1 keep dispatching to the v1 actor; new consumers target v2 via the const tuple. Deprecation lands as a separate PR that removes the v1 manifest after all consumers have migrated. The `(G, version)` tuple in `MODULE_TABLE` makes the migration window explicit and queryable: `SELECT * FROM ogit_modules WHERE g = 2 ORDER BY version DESC`.

4. **Inert manifest semantics.**
   *Recommendation: inert = registered in OGIT, traversable via SPARQL, no actor spawned.* `FMA_V1` ships inert in this plan: its entity-types (Anatomy.* classes, FMA IDs) populate the triple store, queries against them succeed, but `Route { g: 5, .. }` returns `NoConsumer`. When a consumer crate for FMA later ships, only the manifest's `inert_when_consumer_absent: false` flag and `actor:` block need flipping. Tier-3 / Pattern K (JIT compile a consumer from manifest) is out of scope here — see TD-CIRCULAR-COMPILATION-7.

5. **Hot-reload — explicitly out of scope.**
   The build-script-emitted `MODULE_TABLE` is a `const &[ModuleEntry]`; runtime mutation is impossible without re-link. That's the *design*, not a bug. Hot-reload of consumer logic belongs in Tier-4 (Pattern K, circular compilation, dyn-loader on `libsmb_office_v2.so`). Filed as TD-CIRCULAR-COMPILATION-7 for after Tier-3 ships.

6. **Supervisor restart strategy — one-for-one vs. one-for-all.**
   The sketch above uses `restart-all-on-any-crash` (cheapest, smallest blast radius for v1 since spawn is fast). Open question: do we want per-consumer one-for-one once consumer counts exceed ~10? Probably yes, but deferred until the second consumer ships and we measure spawn cost. Captured as a note on TD-RACTOR-SUPERVISOR-5.

---

## 4. Acceptance criteria (plan-level)

- **Two deliverables**, each independently shippable as separate PRs in order: D-MANIFEST-MODULES first, D-RACTOR-SUPERVISOR second.
- **Backwards compatibility.** Existing newtype gates in PR #29 (callcenter newtype wrappers) and PR #98 (cognitive-shader-driver bindspace gates) keep working as ergonomic wrappers — they call `GenericBridge::dispatch` under the hood; the supervisor sits *above* them and routes by G. Nothing in PR #29 or PR #98 needs to be reverted; both become thin facades.
- **Tests** required for plan acceptance:
  - build-script idempotency (no manifest change ⇒ byte-identical `ogit_modules_gen.rs`)
  - manifest parse correctness (5 unit tests covering deny_unknown_fields, duplicate-G, code collision, inert+missing-crate, parent-link resolution)
  - actor spawn + route + reply roundtrip (medcare consumer)
  - sync-mode I-2 enforcement (clippy `disallowed-types` lint scoped to `consumers/`)
  - inert manifest path (SPARQL traverses FMA_V1, route returns typed NoConsumer error)
  - crash + restart (panic in handler triggers supervisor restart; second dispatch succeeds)
- **Ergonomics target met.** After this plan ships, adding a new consumer = ~30 LOC: one `manifest.yaml`, one `Cargo.toml` workspace member entry, one `impl Consumer for FooActor` in the consumer crate. Zero edits to `lance-graph-contract` source. Zero edits to `lance-graph-callcenter` source. Verified end-to-end by the synthetic 7th-manifest test fixture.

---

## 5. Dependencies and cross-references

- **Hard dependency on W10's Tier-1** (`ogit-g-context-bundle-v1.md`):
  - `ContextBundle` is the in-memory shape of the per-G runtime context the supervisor hands to each consumer at spawn.
  - `ConsumerPointer` is what `MODULE_TABLE[i]` resolves to at supervisor boot.
  - `GenericBridge` is the typed dispatch surface the supervisor uses to fan out to consumers — without it the supervisor would have to know every consumer's message enum statically, defeating the whole pattern.
- **Master plan** (`unified-ogit-architecture-v1.md`, W1): this is Tier 2 of the four-tier architecture (Tier 1 = data model, Tier 2 = binding, Tier 3 = TBD, Tier 4 = JIT/Pattern K).
- **Sibling proof plan** (`anatomy-realtime-v1.md`, W12): consumes both deliverables to demonstrate end-to-end FMA anatomy + realtime dispatch. W12 will reference `OGIT::FMA_V1` and the supervisor's `Route` arm as already-shipped APIs.
- **TECH_DEBT rows owned by W5:** TD-MANIFEST-MODULES-4 (this plan, D-MANIFEST-MODULES), TD-RACTOR-SUPERVISOR-5 (this plan, D-RACTOR-SUPERVISOR). Also adjacent: TD-OGIT-G-SLOT-1, TD-CONTEXT-BUNDLE-2, TD-GENERIC-BRIDGE-3 (W10's rows, blocking deps).
- **Plan-index entry:** W8 indexed this plan-doc in `INTEGRATION_PLANS.md` (2026-05-07 section) under Active status.
- **Reframes** the following pre-existing plans (cited per W8's index):
  - `callcenter-membrane-v1.md` — the supervisor here replaces its ad-hoc `lance_membrane.rs` dispatch fan-out with structured ractor supervision.
  - `ogit-cascade-supabase-callcenter-v1.md` — the cascade's "OGIT registry → callcenter consumer" wiring is exactly the (manifest → supervisor → actor) chain this plan formalizes.
  - `palantir-parity-cascade-v2.md` — its consumer-stack-profile concept becomes the `stack_profile:` block in the manifest schema.
- **Ledger reframe.** PR #29 + PR #98 are not retroactively in scope for revision; both shipped before this plan and remain merged. After this plan lands, *new* consumer registrations bypass the newtype-gate pattern entirely; the gates persist as a documented legacy path for the two consumers that pre-date this work.

---

## 6. Brutally honest self-review

**Risks I see and have not eliminated:**

- The build-script + `OUT_DIR`-emitted-code path is well-trodden (proc-macro-server, prost-build, tonic-build) but every project that uses it eventually hits the "rust-analyzer doesn't see the generated symbols" papercut. We accept that — the workaround is well-known (`include!(concat!(env!("OUT_DIR"), "/ogit_modules_gen.rs"));` from a `mod modules;` declaration). It is a developer-ergonomics tax, not a correctness risk.
- The `Vec::find` in `handle_supervisor_evt` is O(N) over consumer count. Fine for N < 50; if N grows to hundreds, swap to a reverse-index `HashMap<ActorId, (u32, u32)>`. Not blocking for v1.
- Sync-mode ractor + a tokio-using `lance_membrane.rs` outbound boundary requires careful boundary discipline. The clippy lint enforces "no tokio in `consumers/`" but the supervisor itself sits at the boundary and *does* hold an `Arc<TokioRuntime>` for outbound dispatch. That's the I-2 design (tokio outbound only), not a violation — but it's the kind of nuance the next worker who reads this will get wrong without the lint catching them. Adding a doctest in `supervisor.rs` that demonstrates the legal pattern is cheap insurance; logging it as a sub-task of D-RACTOR-SUPERVISOR.
- The plan assumes W10's Tier-1 lands first. If W10 slips, D-MANIFEST-MODULES can still ship with stub `ConsumerPointer` / `ContextBundle` types defined locally in `lance-graph-contract`, and D-RACTOR-SUPERVISOR blocks. Sequencing is in the master plan.
- No reverse-engineering work on HubSpo / CRM_V1 is in scope of *this* plan — only the inert manifest. The actual Claude-Code reverse-engineering of HubSpo's surface is owned elsewhere.

**What I deliberately did not do:**

- Did not specify the consumer-side `impl Consumer for FooActor` trait shape in detail — that's W10's `ContextBundle` / `GenericBridge` surface and belongs in Tier-1.
- Did not enumerate every entity-type for every G — that's the manifest authors' job once the schema is fixed.
- Did not touch Pattern K / circular compilation / JIT — Tier-4, deferred, TD-CIRCULAR-COMPILATION-7.
- Did not write code, only spec + sketches. The 770 LOC effort estimate is a *commitment to write*, not a commitment to land in this sprint.

---
