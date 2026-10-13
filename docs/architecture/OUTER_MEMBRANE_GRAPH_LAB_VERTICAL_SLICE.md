# Outer membrane + graph lab: evidence, ownership, and a falsifiable vertical slice

**Status:** architecture note / proposal, **not** a shipped end-to-end integration  
**Date:** 2026-10-13  
**Scope:** LanceGraph callcenter/supervisor, shader lab, DN/Redis model, HHTL, Lance/S3 cache lifecycle, planner/Quack, and existing graph-flow/z8run integrations  
**Rule:** *LanceGraph owns universal mechanisms; each consumer owns the meaning of its ClassIds. OGAR is one provider of that meaning.*

This note records a documentation-only decision and a demonstrable integration target. It does **not** declare an existing Redis service, a production REST gateway, full actor dispatch, or whole-chain D-CMR normalization. Paths below were read on `main` during preparation; links are navigational, not substitutes for compatibility or CI tests.

## Companion: epoch-indexed graph timeline research

The detailed temporal proposal and its **20 falsifiable exploration questions** live in [Epoch-indexed temporal graph: research agenda](EPOCH_TEMPORAL_GRAPH_RESEARCH_AGENDA.md). It covers existing `TemporalPov` and `VersionedGraph` machinery, the prior September delta/tombstone audit, three distinct clocks, half-open interval masks, sparse deltas versus checkpoints, NAL/kanban revision history, Rust/SIMD cost tests, authorization/backpressure, and KGExplorer-inspired time-slider projection. **No temporal graph implementation is claimed in this docs PR.**

## New implementation target: assimilate ONA as folding IR methods

[**ONA → LanceGraph NAL/R2IL fold-method assimilation**](ONA_NAL_R2IL_FOLD_METHOD_ASSIMILATION.md) defines the next executable objective, preserving OGAR's Loco/r2sleigh ownership: ONA truth laws, derivation stamps, premise matching, temporal projection, goal scoring, method-shaped R2IL lowering, and pinned semantic oracles. The W3–W9 coding waves are *proposed*: **this PR does not implement or finalize those operations**. Implementation completion depends on independent truth/provenance parity, correct retained non-scalar outputs, and the existing authorized write/action boundaries.

## 1. Decision: coexistence, not blanket R2IL relocation

- **OGAR remains authoritative for its `ogar-loco` / `ogar-r2il` toolchain and r2sleigh-oriented use cases.** The fact that its generic pieces are reusable does not authorize moving both crates into LanceGraph. Preserve the shared ABI / wire identities and avoid parallel incompatible definitions.
- **`lance-graph-r2il` already exists** as LanceGraph's global fold/computation path into `mask-risc`. **`lance-graph-nal` already exists** for global NAL operations. Their richer “thinking IR” is a *direction*, not evidence that a new IR or complete NAL inference frontend has shipped. Audit their present OGAR imports before changing dependency edges.
- LanceGraph's `contract` owns domain-agnostic descriptions; Quack and R2IL own their **respective** lowering paths; `mask-risc → ndarray` remains the shared physical execution route. No forced `contract → mask-risc` dependency simply to make execution visible.
- The global Rubicon/Kanban transition contract already lives in [`lance-graph-contract/src/kanban.rs`](../../crates/lance-graph-contract/src/kanban.rs); [`lance-graph-planner`](../../crates/lance-graph-planner/src/lib.rs) plans; [`lance-graph-supervisor`](../../crates/lance-graph-supervisor/src/lib.rs) has the sealed-cycle application and writer under optional features. Consolidate **decision authority and transitions**, not by copying a second state machine.
- [`graph-flow-lance`](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/crates/graph-flow-lance/src/lib.rs) and [`z8run-lance`](https://github.com/AdaWorldAPI/z8run/blob/main/crates/z8run-lance/src/lib.rs) are *existing* handle-based adapters. A possible `lance-graph-flow` name would be a **proposed** home for generic cognitive bindings, *not* a duplicate session runner. A possible `lance-graph-rig` would likewise be an **optional adapter**, not a new model/tool engine.
- A fold is not necessarily scalar. Preserve range/mask/vector/other typed results until algebraic normalization and physical lowering determine whether a move, SIMD Boolean operation, or scalar reduction is necessary. Whole-chain D-CMR Set-IR remains **proposed**; current execution must not be described as if it already eliminates entire chains.

**Ownership criterion:** Can the mechanism be specified and tested without interpreting a consumer's ClassId? Then it is eligible for global LanceGraph ownership. Consumer-specific vocabularies, semantics, method applicability, and bindings remain with the consumer (including OGAR). Opaque ClassIds and global Loco/R2IL wire IDs can appear in global APIs without converting them into consumer semantics.

## 2. Actual outer surface: Supabase-inspired, not another HTTP server

[`lance-graph-callcenter`](../../crates/lance-graph-callcenter/src/lib.rs) was designed as a **partial Supabase-shape transcode**, replacing Postgres with Lance/DataFusion rather than implementing a Supabase clone. The [original membrane plan](../../.claude/plans/callcenter-membrane-v1.md) names [`scaraude/supabase-realtime-rs`](https://github.com/scaraude/supabase-realtime-rs) (Phoenix topic/messages, heartbeat) and [`xylex-group/supabase_rs`](https://github.com/xylex-group/supabase_rs) (PostgREST filter conventions) as inspirations.

Canonical boundary:

```text
HTTP / WebSocket / agent / z8run consumer
    |
    v
callcenter external membrane
    |  auth, tenant / RBAC, audited decision, typed external intent
    v
ExternalIntent -> UnifiedStep -> OrchestrationBridge
    |
    +--> planner / NAL / R2IL
    +--> Quack / mask-risc / ndarray
    +--> canonical graph write/read path
    |
    v
versioned projected external results -> z8run / graph view
```

The existing [`ExternalMembrane`](../../crates/lance-graph-callcenter/src/lance_membrane.rs) separates external `ingest`, `project` and `subscribe`. The “blood-brain barrier” is a typed *projection rule*, not a license to put arbitrary internal semiring objects in an opaque binary field. External DTOs, metadata and access control must remain explicit.

**Do not mistake feature gates for wired endpoints.** [`postgrest.rs`](../../crates/lance-graph-callcenter/src/postgrest.rs) identifies itself as request/response and pure dispatcher scaffolding, not a live PostgREST server. [`drain.rs`](../../crates/lance-graph-callcenter/src/drain.rs) implements an in-memory, realtime-feature drain into `OrchestrationBridge`; durable `steering_intent` polling is deferred. [`version_watcher.rs`](../../crates/lance-graph-callcenter/src/version_watcher.rs) does implement **always-latest** in-process subscriber notifications: a slow subscriber may skip intermediate revisions. Never use that signal itself as a lossless CDC or durable work queue.

## 3. Concurrency: present controls versus missing admission

[`lance-graph-supervisor`](../../crates/lance-graph-supervisor/src/supervisor.rs) uses an optional **ractor** actor tree and one-for-one crash recovery, keyed by active G/version slots. [`supervisor/Cargo.toml`](../../crates/lance-graph-supervisor/Cargo.toml) declares the fork dependency and feature gating. Source-observed boundaries:

| Capability | Current evidence | Confidence / gap |
|---|---|---|
| Per-consumer actor slots, health, restart/backoff | `supervisor.rs` | Implemented skeleton; confirm runtime integration before claiming production behavior |
| Default mailbox target `1024` / per-module override | `DEFAULT_MAILBOX_CAPACITY`, `ModuleEntry.mailbox_capacity` | **Declared**, but `spawn_consumer_actor` currently calls `Actor::spawn_linked` without applying that capacity |
| Normal `DispatchToG` forwarding | `supervisor.rs::handle` | **Not implemented** for non-`Health` envelopes: returns `DispatchNotImplemented` |
| Inbound `DrainTask` queue | `callcenter/src/drain.rs` | **Unbounded** Tokio mpsc in realtime build, even though child mailbox backpressure is a design goal |
| Outbound watch | `callcenter/src/version_watcher.rs` | Always-latest fanout, not lossless stream |
| End-to-end HTTP admission: rate, size, deadline, cancellation | not established on the callcenter ingress | **Not demonstrated** by supervisor existence |

**Useful comparison:** [StellarDB's `src/server/admission.rs`](https://github.com/vadim-su/stellardb/blob/main/src/server/admission.rs) implements request concurrency and per-client throttling; [`server/capabilities.rs`](https://github.com/vadim-su/stellardb/blob/main/src/server/capabilities.rs) reports explicitly that some advertised features (e.g. cancellation, audit log) are unavailable. These are reference patterns for a callcenter-facing **admission layer and honest capabilities discovery**, not a reason to replace `ractor`.

**Target, not current guarantee:** HTTP admission (auth, body limits, per-tenant quotas, deadlines) → bounded queue/backpressure with fail-closed responses → per-G actor supervision → authorized canonical bridge. Do not make dispatch success, audit durability, or completed writes implicit from a `Health` reply. Queue overflow should be visible and retryable where safe, not silently buffered without bound. Actor crash restart is different from request cancellation and persistence.

## 4. Transport and address shapes that already exist

| Surface | Shipped in source | Missing / not equivalent |
|---|---|---|
| [Shader lab Axum server](../../crates/cognitive-shader-driver/src/serve.rs) | `/v1/shader/{route,ingest,dispatch,plan,tensors,calibrate,probe,...}`; `UnifiedStep` conversion | **LAB-ONLY**, not a production external gateway; direct remote access requires review of binding, auth, limits and JIT/file access |
| [`dn_redis` model](../../crates/lance-graph-cognitive/src/container_bs/dn_redis.rs) | Keys `ada:dn:{hex}` / `ada:spine:{hex}`, `RedisCommand`, `RedisPipeline`, `CogRecord` bytes | No Redis server, RESP parser, listener, backend executor or automatic graph commit. See [protocol note](../DN_REDIS_KEY_SHAPE_PROTOCOL.md) |
| [`PackedDn`](../../crates/lance-graph-cognitive/src/container_bs/adjacency.rs) | 7 hierarchy components stored in a `u64` as byte-sized value+1 fields; `from_path` hashes path segments to bytes | Can collide; **not** a canonical NodeGuid and **not** the same encoding as HHTL |
| [HHTL `NiblePath`](../../crates/lance-graph-contract/src/hhtl.rs) | Domain-agnostic fan-out 16, up to 16 nibbles, derivation from canonical identity where the documented preconditions apply | Neither arbitrary external colon paths nor `PackedDn` convert into semantic ancestry without explicit resolution |

An example external path `path:research:project:atlas` is a **demo input convention**; the existing `dn_redis` keys are `ada:dn:...` with `PackedDn` encoding. Treat external text as a lookup/ingestion alias:

```text
external text path
  -> parse / validate / namespace / permission
  -> stable canonical NodeGuid (identity)
  -> separately resolved HHTL NiblePath (hierarchical route)
  -> typed, witnessed semantic edge declarations
  -> authorized graph commit
```

**Non-negotiable:** lexical key prefix, HHTL route, NodeGuid identity, and SPO adjacency are four different assertions. No `CAUSES` / `OWNER` / `IS_A` edge is inferred merely because strings share a prefix. Explicit relationships need schema, witness/provenance and direction; retrying ingestion must not mint duplicate IDs/edges. `PackedDn::from_path` uses narrow hashed components, so a collision/overdepth counterexample must fail closed or be disambiguated by a canonical registry. Keep the existing Redis Cluster `MGET` hash-slot caveat from [the protocol note](../DN_REDIS_KEY_SHAPE_PROTOCOL.md).

## 5. Storage is not HTTP, and Moka is not a new authority

The [generic hydration crate](../../crates/lance-graph-hydrate/src/lib.rs) already implements raw object-store copies, staging/publish, checksum-pinned single-file hydration and idle page-cache release. [`VersionedGraph`](../../crates/lance-graph/src/graph/versioned.rs) exposes a `hydrate_from` integration path. The intended lifecycle remains:

```text
S3-compatible object store (remote/versioned hydration source)
       -> verified staging -> local published Lance dataset
       -> graph access / Quack queries / active working set
       -> explicit safe release (never discard sole dirty copy)
```

The [hydration lifecycle doctrine](../../.claude/knowledge/s3-hydration-lifecycle.md) describes the local-store strategy. The older [cache audit](../../.claude/knowledge/lance-cache-surface.md) explicitly leaves cache-content/capacity/RSS tests open **and disputes literal mmap claims** in other doctrine; it describes a previous Lance version. Current workspace dependencies have since advanced. **Re-audit the current Lance dependency's read/cache behavior** before claiming mmap, zero-copy pages, bounded RSS or Moka hit ratios. [`surreal_container/src/cache.rs`](../../crates/surreal_container/src/cache.rs) is a TODO for a separate bounded L2, not a shipped cognitive Moka cache.

AWS-compatible `object_store` and Axum do not compete: Axum is the transport/control plane, object store hydrates datasets, and Lance/Quack reads the canonical store. Never turn Redis, Moka or HTTP result materialization into a second durable source of truth. Prefer dataset and result **handles**, not serialized populations.

## 6. Small, falsifiable demonstration proposal

**Goal:** Two input aliases produce two identifiable nodes and **one explicitly supplied semantic edge**, with the same identities visible in a graph projection and in a Quack-backed z8run flow. No second evaluator, Redis service or Neo4j database required.

**Illustrative request DTO only (not an existing API):**

```json
{
  "namespace": "demo",
  "nodes": [
    {"external_path": "path:research:project:atlas", "type": "project"},
    {"external_path": "path:research:people:ada", "type": "person"}
  ],
  "edges": [
    {
      "from": "path:research:project:atlas",
      "predicate": "owner",
      "to": "path:research:people:ada",
      "witness": "demo-explicit-assertion-1"
    }
  ],
  "idempotency_key": "graph-lab-demo-001"
}
```

The test harness should make *one narrow connection at a time*:

1. **Ingest:** pass through an authenticated/admitted membrane, validate aliases, assign/retrieve stable canonical identities, and record the namespace mapping. Confirm repeated input keeps NodeGuid stable. Use `dn_redis` only as the existing key/command **type model**, not a fake claim of RESP support.
2. **Resolve:** derive HHTL routes only with the correct type/length/witness preconditions. Verify that different lexical strings or PackedDn collisions cannot masquerade as the same canonical node.
3. **Commit:** use one canonical graph writer to commit two nodes plus the asserted edge. Preserve edge direction, predicate, witness and dataset version; repeat must not duplicate an edge.
4. **Query:** use existing graph tables/adapters and Quack/`mask-risc` for a bounded relation query/fold. Compare with a plain reference traversal; do not require Quack or NAL to pretend semantic graph traversal is implemented if the bridge is still absent.
5. **Plan:** run a legal Rubicon/Kanban transition through existing contract/planner/sealed-cycle semantics; do not advance phase merely because a visualization tick arrived.
6. **Project:** feed the *result handles and displayed subset* to [`z8run-lance`](https://github.com/AdaWorldAPI/z8run/blob/main/crates/z8run-lance/src/lib.rs) and optionally [`graph-flow-lance`](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/crates/graph-flow-lance/src/lib.rs). Their current design uses `lance-abi` envelopes, `handle` and `generation`. Stale generations must fail closed; only a terminal explicit materializer exports rows.
7. **Visualize:** show nodes/edges as a **derived projection** with canonical IDs, edge labels, provenance, source version, and optional NAL truth data *only when computed*. [StellarDB Station's projection](https://github.com/vadim-su/stellardb/blob/main/station/src/lib/station-graph.ts) and [2D/3D graph UI](https://github.com/vadim-su/stellardb/blob/main/station/src/components/station-graph.tsx) are useful reference designs; do not import its Fjall engine or generate synthetic persistent identities from array positions.

### Falsifiers and negative controls

| Test | Must demonstrate |
|---|---|
| Same idempotency key retried | Exactly the same nodes, relations and committed version/result semantics (or an explicitly defined no-op version behavior) |
| Different strings colliding under `PackedDn::from_path` | Distinct canonical identities or typed refusal; never silent merge |
| Unasserted semantic relation | No invented `OWNER` / `CAUSES` adjacency |
| Unauthorized namespace / actor | No graph mutation, read, or tool invocation; denied action can be audited |
| Actor mailbox saturated / HTTP concurrency exceeded | Bounded resource consumption and explicit rejection, **after** admission/backpressure wiring is proven |
| Supervisor child termination | Failed/in-flight operations are surfaced; no phantom success and no loss of sealed durable intent |
| Slow watcher subscriber | May see latest only; ledger query remains source of truth for historical events |
| S3 unavailable / dirty artifact | Hydration fails explicitly or existing pinned local version remains usable; never discard uncommitted dirty state |
| Old `lance-abi` generation | Reject stale handle; never silently resolve to a newer population |
| Non-scalar fold | Retain mask/range/vector intermediate results; correct scalar oracle and SIMD parity by tested CPU tier |

### Acceptance and sequencing

- **P0:** Mark this document as a proposal; inventory exact integration call sites; no fake success path.
- **P1:** Prove bounded front-door admission + real actor forwarding and backpressure, or run the demonstration **strictly isolated** with these disabled features clearly labeled.
- **P2:** Prove canonical path/identity/HHTL mapping and relation ingestion with idempotent replay + negative controls.
- **P3:** Connect actual Quack/graph result handles to existing graph-flow/z8run bindings; capture complete trace and version.
- **P4:** Optional graph UI inspired by StellarDB; then measure latency, RSS, queue saturation and S3 hydration cold/warm behavior on the current pinned Lance release.

No implementation PR should mix all stages. A successful demo is evidence of **one integrated substrate with several familiar interfaces**, not evidence that every suggested interface or cognitive capability is already production-ready.

## References / source checkpoints

- [Original callcenter plan](../../.claude/plans/callcenter-membrane-v1.md) and [callcenter crate](../../crates/lance-graph-callcenter/src/lib.rs)
- [Supervisor actor implementation](../../crates/lance-graph-supervisor/src/supervisor.rs), [callcenter drain](../../crates/lance-graph-callcenter/src/drain.rs), [watcher](../../crates/lance-graph-callcenter/src/version_watcher.rs)
- [LAB REST bridge](../../crates/cognitive-shader-driver/src/serve.rs) and [DN Redis protocol note](../DN_REDIS_KEY_SHAPE_PROTOCOL.md)
- [HHTL contract](../../crates/lance-graph-contract/src/hhtl.rs), [PackedDn and adjacency](../../crates/lance-graph-cognitive/src/container_bs/adjacency.rs)
- [Hydration crate](../../crates/lance-graph-hydrate/src/lib.rs), [Lance cache open questions](../../.claude/knowledge/lance-cache-surface.md)
- [Kanban contract](../../crates/lance-graph-contract/src/kanban.rs), [planner](../../crates/lance-graph-planner/src/lib.rs), [graph-flow-lance](https://github.com/AdaWorldAPI/rs-graph-llm/blob/main/crates/graph-flow-lance/src/lib.rs), [z8run-lance](https://github.com/AdaWorldAPI/z8run/blob/main/crates/z8run-lance/src/lib.rs)
- [StellarDB](https://github.com/vadim-su/stellardb): [admission](https://github.com/vadim-su/stellardb/blob/main/src/server/admission.rs), [capabilities](https://github.com/vadim-su/stellardb/blob/main/src/server/capabilities.rs), [graph projection](https://github.com/vadim-su/stellardb/blob/main/station/src/lib/station-graph.ts)
