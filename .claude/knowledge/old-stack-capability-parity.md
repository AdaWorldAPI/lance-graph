# substrate-b consumer integration — NEW-stack capability shape + plans

**READ BY:** integration-lead, truth-architect, anyone wiring a lance-graph + ractor + surrealdb integration (the substrate-b shape, per `lance-graph-callcenter`'s `ExternalMembrane` + `LanceVersionWatcher` + `Timeline` view).

**Status:** preventive documentation + capability roadmap for substrate-b consumers. Captures the design pattern + the built-today honesty + the planned capability roadmap.

**Companion to:** `.claude/knowledge/lab-vs-canonical-surface.md` (the canonical surface rule), `.claude/knowledge/hollow-wire-failure-modes.md` (the failure mode when the canonical wire is incomplete), `.claude/knowledge/encoding-ecosystem.md` (the full codec ledger).

---

## 1. The substrate-b shape (the integration pattern this doc serves)

A substrate-b integration of lance-graph composes seven cooperating capabilities into a single binary:

1. **Storage** — `lance-graph` (Lance columnar) + Lance versions as the temporal axis (`checkout_version(V)` is the time-travel primitive)
2. **Distributed KV** — `surrealdb` with `kv-lance` feature (PRs `AdaWorldAPI/surrealdb#35` + `#36` merged) OR external TiKV cluster
3. **Search** — Tantivy (Rust-native full-text)
4. **Analytics / OLAP** — DataFusion over Lance versions
5. **Actors / dispatch** — ractor + the mailbox-as-owner topology + `MessagingErr::Saturated` for backpressure (PR `AdaWorldAPI/ractor#1` merged)
6. **In-process event bus** — `lance-graph-callcenter::version_watcher::LanceVersionWatcher` (std::sync Condvar; never tokio per the I-2 invariant)
7. **OIDC / IAM** — external Zitadel (retained as the IdP boundary) + in-proc JWT validation via `auth-plug`

The `lance-graph-planner` (16 strategies + 12 thinking styles + NARS dispatch + the AutocompleteCache) is the compute / rule-evaluation layer that sits on top.

## 2. The three load-bearing primitives substrate-b consumers must understand

### 2.1 Lance versions are the multi-purpose temporal primitive

A single primitive — Lance versions — serves three distinct capabilities a substrate-b consumer would otherwise build separately:

- **Point-in-time query** = `dataset.checkout_version(V_ref)` — pin an immutable snapshot at any version
- **Time-series** = the version log itself — every commit is a versioned event with a timestamp
- **Immutable audit** = append-only by construction — versions never disappear; the log IS the audit trail

This is the substrate-b efficiency claim: three capabilities, one primitive, no separate storage. The implications are captured in `STANDING_WAVE_ARCHITECTURE.md` §1 (in substrate-b consumer repos).

### 2.2 Per-element auth = palette256 + Hamming popcount on Binary16K

The substrate-b hot-path auth primitive is bit-op-per-element via `Binary16K = [u64; 256]` (Hamming-compare format) with palette256 as the codebook. The per-vertex `_effectiveReaders` / `_effectiveDevices` bitmap is materialised on write; the per-read check is a Hamming popcount / bit-intersection (uncached, immediate-effect by construction — auth changes take effect at the next commit, no separate cache invalidation).

This is the substrate-b shape for fine-grained per-element authorization. Any consumer wiring auth onto lance-graph entities should use this primitive (per `vsa-switchboard-architecture.md` Layer-2 catalogue + `encoding-ecosystem.md`); do NOT introduce a separate ACL store.

### 2.3 ractor Actor + Lance-version-as-state-machine = the Rubicon phase machine

A substrate-b actor models its lifecycle as a ractor `Actor` whose epistemic state transitions are mirrored by Lance commits. The pattern:

- **State enum** (typed) on the ractor Actor — one variant per epistemic state (e.g. Contemporary / Anachronistic / Spoiler for an awareness-mode dispatch)
- **State-enter side-effect** — entering the Decision state fires the Lance commit; the version bump is the state-transition record
- **Defer-until-transition** — events arriving before the Decision are held; replayed after the version bump
- **Per-state timeouts** — SLA-bounded waits per state; expiry routes through `MessagingErr::Saturated` for backpressure

The architectural payoff: the actor's state history is materialised in the Lance version log (no separate state-machine event store), and the determinism firewall is preserved (each actor commits to its own dataset; no cross-mailbox state coupling). Full pattern in `STANDING_WAVE_ARCHITECTURE.md` §6 in substrate-b consumer repos.

## 3. Capability roadmap — built today / partial / not-yet

Honest accounting for substrate-b consumers planning integration sequencing.

| Capability | Where | Status |
|---|---|---|
| Lance versions (point-in-time + time-series + audit) | `lance` / `lance-graph` | **built** |
| `LanceVersionWatcher` (in-proc event bus) | `lance-graph-callcenter::version_watcher` | **built** (std::sync Condvar, never tokio) |
| `ractor::MessagingErr::Saturated` (backpressure) | `AdaWorldAPI/ractor#1` | **built** (merged) |
| surrealdb `kv-lance` feature + Lance backend struct | `AdaWorldAPI/surrealdb#35` + `#36` | **built** (merged) |
| `lance-graph-planner` 16 strategies + 12 thinking styles + NARS | `lance-graph-planner` | **built** |
| `auth-plug` in-proc JWT validation | `auth-plug` | **built** |
| palette256 + Hamming popcount primitives | `lance-graph-contract` + `bgz17` | **built** |
| `cognitive-shader-driver` co-located dispatch | `cognitive-shader-driver` | **built** (canonical surface) |
| `EpisodicEdges64` 4-slot MRU + `DemotionSink` (Phase A) | `lance-graph-contract::episodic_edges` (#446/#447/#448) | **built** |
| OGAR `MappingProposal` → `OntologyRegistry` | `AdaWorldAPI/OGAR#5/#6/#7/#8` | **built** (Sprint 5/6 shipped; Sprint 7 muscle memory documented) |
| lance-graph traversal (consumer surface) | `lance-graph` + `lance-graph-planner` | **partial** — need a stable consumer surface for substrate-b |
| DataFusion OLAP surface (consumer endpoint) | `lance-graph` + DataFusion integration | **partial** |
| Tantivy search wiring | (not yet wired) | **not-yet** |
| OGAR `ogar-runtime` Sprint 7 (ClassActor + KanbanMailbox) | `AdaWorldAPI/OGAR` | **not-yet** (holding for protoc-build access + signal) |
| Peer-Raft consensus (openraft / surreal-cluster / TiKV) | (pick-one deferred) | **not-yet** |
| Migration endpoint contract (the dual-stack ground-truth surface) | (substrate-b consumer concern) | **not-yet** (consumer side) |
| WebSocket / gRPC actor mailbox (Layer-3 outbound) | (deferred per I-2 invariant) | **not-yet** |
| Cold-tier `DemotionSink` impl (Phase C, surreal-LIVE wingman) | `lance-graph-contract::episodic_edges::DemotionSink` (seam shipped; impl gated on OQ-11.6) | **gated** |
| `EpisodicWitness64` SoA column (Phase D) | `soa_view.rs:77 episodic_witness()` accessor | **gated** (offline; needs `cognitive-shader-driver`'s `MailboxSoA<N>`) |

## 4. The migration endpoint contract — substrate-b's dual-stack ground-truth surface

A substrate-b consumer that's replacing an external system needs a uniform API surface exposed by BOTH the substrate-b binary AND the system being replaced, so the same workload can be replayed against each and compared deterministically.

The contract substrate-b consumers should expose (one minimal shape; consumers extend per workload):

```
POST   /v1/entity            create vertex
GET    /v1/entity/:id        read vertex
GET    /v1/entity/:id?at=V   read vertex at Lance version V (historisation)
PUT    /v1/entity/:id        update vertex
DELETE /v1/entity/:id        soft-delete (lifecycle markers, never hard delete)
POST   /v1/edge              create edge
POST   /v1/traverse          multi-hop traversal (lance-graph-planner dispatch)
POST   /v1/query             point query
POST   /v1/graphql           tabular output (DataFusion SQL over Lance versions)
GET    /v1/audit             audit query (DataFusion over Lance version log)
WS     /v1/stream            real-time subscription (LanceVersionWatcher → WS push, Layer-3 tokio)
POST   /v1/dispatch          cognitive dispatch (substrate-b-specific; no comparison)
```

Properties to preserve at every endpoint:
- **Version-pinnable reads** — every read can specify `?at=V` for point-in-time
- **Lifecycle markers, not hard delete** — `revokedAt` / `archivedAt` timestamps; never DROP
- **Real-time subscription via the in-proc bus** — `WS /v1/stream` proxies `LanceVersionWatcher::subscribe()` through to the WebSocket client (Layer-3 tokio outbound)
- **DataFusion is the cold-path SQL** — graphql + audit go through DataFusion over Lance versions; the hot path stays on `lance-graph-planner` strategy dispatch

Substrate-b consumers run dual-stack workload replay against this contract; the §14 acceptance gate (in substrate-b consumer repos' `docs/MIGRATION-COMPARISON-HARNESS.md`) produces a per-endpoint verdict (PASS / DIVERGENT-RECONCILABLE / DIVERGENT-FAULTY / INDETERMINATE).

## 5. Integration patterns that fall out of this shape

### 5.1 Three OLD components collapse to one when substrate-b is the target

A consumer migrating from a separate Historisation layer + a separate time-series database + a separate signed-audit-log service replaces all three with Lance versions. This is the design-pattern claim; it follows from §2.1. Substrate-b consumers should NOT introduce separate stores for these three capabilities.

### 5.2 ACL changes take effect immediately, by construction

Because the per-element auth check is a bit-op against the per-vertex bitmap (no separate cache), an ACL change at Lance version V is in effect at every read at version >= V. There is no auth-cache invalidation step; substrate-b consumers should NOT introduce one. The materialisation of `_effectiveReaders` on write is the only side-effect.

### 5.3 The actor's state history is the Lance version log

A ractor Actor following the Rubicon pattern (§2.3) does NOT need a separate state-machine event store. The Lance version log on the actor's dataset IS the state history. Substrate-b consumers should NOT introduce a side-table of state transitions; queries against the version log answer "what state was this actor in at version V?" via `checkout_version(V)`.

### 5.4 In-process events use std::sync, not tokio

The `LanceVersionWatcher` (lance-graph-callcenter) uses `std::sync::{Arc, RwLock, Mutex, Condvar, AtomicUsize}` and never `tokio::sync` — per the I-2 invariant documented in that module. Substrate-b consumers wiring real-time event subscription must follow the same rule for in-process listeners; tokio is reserved for Layer-3 outbound sinks (PhoenixServer, PostgRestHandler, WebSocket push, gRPC remote actor transport).

This is a `hollow-wire-failure-modes.md` failure-mode magnet: a consumer that introduces tokio for in-process subscription violates I-2 and creates the exact bug `version_watcher.rs`'s migration history note (2026-05-06) records as already-corrected upstream.

## 6. The OGAR carrier — substrate-b's data-model entry point

OGAR (`AdaWorldAPI/OGAR`) is the substrate-b carrier for class identity + schema + Active-Record semantics → `lance-graph-ontology::OntologyRegistry`. Substrate-b consumers feeding domain models into lance-graph go through OGAR's `MappingProposal` shape (PR `AdaWorldAPI/OGAR#5` shipped via owned-mirror workaround; Sprint 6 registry integration shipped; Sprint 7 ClassActor + KanbanMailbox muscle-memory documented and gated on the upstream symbol-layout decisions).

Consumer integration sequence (when wiring a new domain model):

1. Author the domain model as an OGAR `Class` (the IR; OGAR Sprint 1 vocab + Sprint 4.5 SurrealQL adapter shipping)
2. Feed `MappingProposal` into `OntologyRegistry` (OGAR Sprint 5/6 path)
3. The lance-graph-planner picks up the registered class for strategy dispatch
4. The `lance-graph-callcenter::LanceMembrane` (sole writer per session) projects domain events into Arrow-scalar `CognitiveEventRow` (BBB invariant: no VSA / RoleKey / NarsTruth crosses the membrane)
5. `LanceVersionWatcher` fans out to in-proc subscribers; the bus is the version pointer

No new contracts are needed for steps 1-5; the existing OGAR + lance-graph-ontology + lance-graph-callcenter surfaces compose. See `OGAR_ELIXIR_HIRO_MUSCLE_MEMORY.md` and `STANDING_WAVE_ARCHITECTURE.md` §1.6 in substrate-b consumer repos for the muscle memory.

## 7. Process rule for substrate-b consumers

Before proposing a new lance-graph integration trait / contract / coordination surface:

1. Check the §3 capability roadmap — does an existing built-or-partial primitive serve the capability?
2. Check `.claude/knowledge/encoding-ecosystem.md` — does an existing codec serve the data shape?
3. Check `.claude/knowledge/lab-vs-canonical-surface.md` — is the canonical bridge the right place (vs the LAB transport)?
4. Check `.claude/knowledge/hollow-wire-failure-modes.md` — am I about to wire the lab surface without plugging the canonical bridge?

If steps 1-4 don't dissolve the proposal, then a new structure is warranted. Otherwise, the existing surfaces already name what to wire against.

---

## Cross-references

### Companion lance-graph knowledge docs
- `.claude/knowledge/lab-vs-canonical-surface.md` — the canonical surface rule (MANDATORY before any REST/gRPC/Wire DTO work)
- `.claude/knowledge/hollow-wire-failure-modes.md` — DRAFT-INERT / sealed-but-shadowed / feature-flag mismatch (companion failure-mode catalogue)
- `.claude/knowledge/encoding-ecosystem.md` — the full codec ledger (MANDATORY before any codec work)
- `.claude/knowledge/vsa-switchboard-architecture.md` — the three-layer carrier (Switchboard / Domain role catalogues / Content stores)

### Upstream contributions in this lineage (substrate-b correspondence)
- `AdaWorldAPI/lance-graph#452` (Lance-append-Raft dovetail), `#453` (cluster asymmetry), `#454` (post-merge corrections), `#455` (dn_redis as key-shape protocol), `#456`/`#457`/`#458` (Pearl junctions classifier + post-merge corrections) — MERGED
- `AdaWorldAPI/lance-graph#464` (hollow-wire failure modes catalogue) — OPEN
- `AdaWorldAPI/surrealdb#35` (kv-lance SDK feature), `#36` (Lance backend struct + endpoint helper) — MERGED
- `AdaWorldAPI/ractor#1` (`MessagingErr::Saturated` variant + `From<TrySendError>` split) — MERGED
- `AdaWorldAPI/OGAR#5` (Sprint 5a owned-mirror), `#6` (cross-session coordination), `#7` (temporal time-travel + std::sync correction), `#8` (Sprint 7 muscle memory absorbed) — MERGED
