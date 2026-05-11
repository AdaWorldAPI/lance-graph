# LF Integration Mapping — v1

> **Status:** Active (2026-04-25)
> **Owner:** @integration-lead, @scenario-world (newly added)
> **Scope:** Comprehensive map of all 41 LF + 4 W chunks shipped or queued
> across the lance-graph workspace, with status, commit refs, residence,
> and cross-stage dependencies. Mirrors the SMB-side `foundry-parity-checklist.md`
> (commit `bf7c05e` and onward) but lives here so future lance-graph sessions
> see the producer-side state at a glance.
> **Mirrors:** `smb-office-rs/docs/foundry-parity-checklist.md` (consumer side)
> **Companion:** `.claude/agents/scenario-world.md` (decision rationale for LF-70..72)

---

## Reading guide

| Status | Meaning |
|---|---|
| ✅ DONE | Shipped on `main`; SMB session can VERIFY by integration |
| 🟢 IN-PR | On a feature branch awaiting merge |
| 🟡 QUEUED | Specced, awaiting REQUEST or implementation slot |
| 🔵 DEFERRED | Skipped intentionally (with reason) |
| 🔴 REDESIGN | Original spec rejected; counter-proposal queued |
| ⚫ FUTURE | Out-of-scope for current cycle, no design pass yet |

**Residence:**

| Layer | Crate | Visibility |
|---|---|---|
| L0 — DTO contract | `lance-graph-contract` | zero-dep, every consumer can pull |
| L1 — substrate | `lance-graph` | core query + dataset versioning |
| L2 — boundary | `lance-graph-callcenter` | BBB, REST/WS, RLS, audit |
| L3 — cognition | `lance-graph-cognitive`, `lance-graph-archetype` | Pearl Rung 3, archetype World |
| L4 — orchestration | `lance-graph-planner` | strategies, MUL, scenarios |
| OUTER | future `lance-graph-connectors` | unified-data-layer DTO |

The **inside-BBB** vs **outside-BBB** axis is enforced by the typed boundary
in `contract::external_membrane`: every Tier 1 / Tier 2 chunk lands either
strictly outside the BBB (DTO additions, REST surface, connectors) or as
an **additive column on the four BindSpace SoAs** (FingerprintColumns /
QualiaColumn / MetaColumn / EdgeColumn). No new struct wraps the SoA; no
new layer hides it. Per CLAUDE.md "AGI is the glove, not the oracle."

---

## Tier 0 — Pre-existing baseline (no work, do not duplicate)

| Type | Where | Status |
|---|---|---|
| `OrchestrationBridge` + `UnifiedStep` | `contract::orchestration` | ✅ shipped |
| `Blackboard` + `BlackboardEntry` (a2a) | `contract::a2a_blackboard` | ✅ shipped |
| `CrystalFingerprint` + `Vsa16kF32` algebra | `contract::crystal::fingerprint` | ✅ shipped |
| Existing role-key catalogue (47 keys, [0..10000)) | `contract::grammar::role_keys` | ✅ shipped (pre-LF-2) |
| `BindSpace` SoA + 4 columns | `cognitive-shader-driver::bindspace` | ✅ shipped |
| `Pearl Rung 3 intervention` (do-calculus) | `lance-graph-cognitive::world::counterfactual` | ✅ shipped (5 tests) |
| `VersionedGraph` (Lance-native time-travel + diff + tag) | `lance-graph::graph::versioned` | ✅ shipped |
| `WorldModelDto` (Self/User/Field/Context + qualia + proprioception) | `contract::world_model` | ✅ shipped |
| `archetype::World` scaffold (URI + tick) | `lance-graph-archetype::world` | ✅ shipped (was stub, now wired in this branch) |

---

## Tier 1 — SMB feature parity (8 + 4 chunks, ALL ✅ DONE)

The minimum-viable cut for the SMB session to consume. Every item shipped
across PRs #262 / #263 / #264 (3 merged PRs in two days). Detailed
commits:

| LF / W | Commit | Lands in | What | Verified by SMB? |
|---|---|---|---|---|
| **LF-1** | `474d3eb` (PR #262) | `contract::orchestration` | `StepDomain::Smb` variant + `from_step_type("smb")` | VERIFY-PENDING (F6 OrchestrationBridge) |
| **LF-2** | `56f2695` (PR #264) | `contract::grammar::role_keys` | `VSA_DIMS 10000 → 16384` + 8 SMB role keys (KUNDE/SCHULDNER/MAHNUNG/RECHNUNG/DOKUMENT/BANK/FIBU/STEUER) at `[10000..14096)`, 512 dims each, headroom `[14096..16384)` | VERIFY-PENDING (F5 ontology) |
| **LF-3** | `c7310ec` (PR #264) | `contract::auth` + `callcenter::{auth, rls}` | `ActorContext { actor_id: String, tenant_id, roles }` + `JwtMiddleware` (Phase 1, no sig verification) + `RlsRewriter` (DataFusion `OptimizerRule` injecting tenant + actor_id predicates on TableScan) | VERIFY-PENDING (F8 RBAC) |
| **LF-4** | `2857a03` (PR #262) | `contract::property::EntityStore` | `scan_stream<'a>(...) -> Result<Self::Stream, Self::Error>` associated-type streaming reads | VERIFY-PENDING (F4 LanceConnector) |
| **LF-5** | `2857a03` (PR #262) | `contract::property::EntityWriter` | `upsert_with_lineage(..., LineageHandle)` returning row count | VERIFY-PENDING (F4 connectors) |
| **LF-6** | `474d3eb` (PR #262) | `contract::property::PropertySpec` | `Marking::{Public, Internal, Pii, Financial, Restricted}` (ordered by restrictiveness) | ✅ VERIFIED `smb-office-rs::514f58a` (Default = Internal; ordering test) |
| **LF-7** | `474d3eb` (PR #262) | `contract::property` | `LineageHandle { entity_type, entity_id, version, source_system, timestamp_ms }` const ctor | ✅ VERIFIED `smb-office-rs::514f58a` |
| **LF-8** | `474d3eb` (PR #262) | `contract::a2a_blackboard::ExpertCapability` | `Smb{EntityValidation, LineageTracking, ComplianceCheck}` variants 10/11/12 | VERIFY-PENDING (F6) |
| **LF-21** | `76a7237` (PR #263) | `contract::property` | `enum SemanticType { Iban, Date(DatePrecision), TaxId, CustomerId, InvoiceNumber, Currency(IsoCode), Geo(GeoFormat), File(MimeType), ... }` on PropertySpec | ✅ VERIFIED — German predicates iban/kdnr/geburtsdatum/steuer-id mapped |
| **LF-22** | `76a7237` (PR #263) | `contract::property::Schema` | `ObjectView { card: Vec<&str>, detail: Vec<&str>, summary_template: &str }` | ✅ VERIFIED — fits firma/kdnr/ort customer card |
| **LF-90** | `76a7237` (PR #263) | `contract::property` | `AuditEntry` + `AuditLog` (immutable append-only audit trail with 64-byte signature placeholder) | ✅ VERIFIED — `AuditAction::Create` + predicate target test |
| **LF-91** | `e70f944` (PR #263) | `contract::sla` | `SlaPolicy { max_latency_ms, min_freshness_ms, priority }` + `SlaPriority::{Background, Standard, Interactive, Urgent}` + `STANDARD` / `INTERACTIVE` consts | ✅ shipped, awaiting downstream integration |
| **LF-92** | `e70f944` (PR #263) | `contract::sla` | `TenantId = u64`, `TenantScope::{Single, Multi, All}` (`Default = All`); composes with `MembraneGate`/`CommitFilter` | ✅ shipped, awaiting downstream integration |
| **W-1** | `6d3016c` (PR #262) | `contract::property::LineageHandle` | `merge(a, b)` — order-independent, picks higher version + max timestamp + source-of-newer | ✅ VERIFIED — Mongo v3 + IMAP v5 → v5 from IMAP |
| **W-2** | `6d3016c` (PR #262) | `contract::property::Marking` | `most_restrictive(slice)` — empty → Public, fold over slice | ✅ VERIFIED — `[Internal, Financial, Pii, Pii] → Financial` |
| **W-3+W-4** | `6d3016c` (PR #262) | `contract::property::mock_store::VecStore` | In-memory `EntityStore` + `EntityWriter` impls for integration testing | ✅ VERIFIED — round-trip + version increment tests |

**Total: 16 chunks shipped. 9 verified by SMB-side `contract_verify.rs` (14 tests passing); 7 awaiting downstream stage execution.**

---

## Tier 2 — Foundry-equivalent surface (28 chunks across 8 stages)

Status as of `claude/scenario-world-facade` branch (commit `521f946`).
Most chunks are 🟡 QUEUED awaiting REQUEST from SMB session or
implementation slot.

### Stage 1 — Data Integration (LF-10..14)

External sources (PostgreSQL, Mongo, MS Graph, Drive, SAP, SIEM, LLM
APIs incl. xAI gRPC) live on the **outer-membrane unified data-layer
DTO** per the architectural decision captured in PR #264 bus discussion.
Lands in a new `lance-graph-connectors` crate, NOT inside-BBB.

| LF | Chunk | Status | Effort | Residence | Notes |
|---|---|---|---|---|---|
| **LF-10** | `Connector` registry + `S3Connector` impl of `EntityStore` | 🟡 QUEUED | M | new `lance-graph-connectors` | Trait shape inferable from existing `EntityStore` (LF-4); registry pattern matches `OrchestrationBridge` slot |
| **LF-11** | `PostgresConnector` impl | 🟡 QUEUED | M | `lance-graph-connectors` | sqlx or postgres-types based; SMB uses Mongo today, Postgres queued for payroll/Lohnabrechnung phase |
| **LF-12** | `Pipeline` DAG + topological scheduler — `UnifiedStep.depends_on: Vec<StepId>` + executor | 🟡 QUEUED | L (split: schema / executor / cron) | mostly `lance-graph-planner` + `contract::orchestration` field add | High leverage — unblocks the connector tier orchestration |
| **LF-13** | `Schedule` trait + cron expression parser | 🟡 QUEUED | M | `lance-graph-planner::schedule` (new module) | Apache Temporal **NOT** chosen — see scenario-world agent card §"Apache Temporal" |
| **LF-14** | Per-row column-level lineage: `LineageEdge { from: (batch_id, row_idx), to: (batch_id, row_idx), step_id }` | 🟡 QUEUED | L (split: edge type / capture / query) | `contract::lineage` (new module) | Extends LF-7 `LineageHandle` to the row-level graph |

**Sequencing:** LF-10 first (registry), then LF-11 (one concrete connector
proves the shape), then LF-12 (orchestration), LF-13 (scheduling), LF-14
(row-level lineage). LF-15+ (Mongo/MS Graph/Drive/SAP/SIEM/LLM) follow
the same template once LF-11 lands.

### Stage 2 — Ontology (LF-20..23)

Two of the four chunks already shipped. Pure DTO additions to
`contract::ontology` and `contract::property`.

| LF | Chunk | Status | Effort | Residence | Notes |
|---|---|---|---|---|---|
| **LF-20** | `FunctionSpec { name, signature, body }` trait in `contract::ontology` | 🟡 QUEUED | M | `contract::ontology::function` | Foundry Functions equivalent — pure transforms over object sets |
| **LF-21** | `SemanticType` enum on `PropertySpec` | ✅ DONE (PR #263 `76a7237`) | M | `contract::property` | Iban / Date / Currency / Geo / File / Email / Phone / TaxId / CustomerId / InvoiceNumber |
| **LF-22** | `ObjectView` (card / detail / summary_template) on `Schema` | ✅ DONE (PR #263 `76a7237`) | S | `contract::property::Schema` | const ctor + 6 ergonomic builder methods |
| **LF-23** | `NotificationSpec { trigger: ActionTrigger, recipients, template }` | 🟡 QUEUED | M | `contract::ontology::notification` | Event-driven; reuses existing `ActionSpec.on_commit` hook |

### Stage 3 — Storage v2 (LF-30..33)

Touches the Lance dataset layer. Below the BBB; no internal cognition
changes needed.

| LF | Chunk | Status | Effort | Residence | Notes |
|---|---|---|---|---|---|
| **LF-30** | MVCC support on `EntityWriter::upsert` — `version: u64, prior_version: Option<u64>` | 🟡 QUEUED | L (split: schema / write / conflict resolution) | `contract::property::EntityWriter` + Lance impl | Optimistic concurrency for concurrent writers |
| **LF-31** | Time-travel queries: `EntityStore::scan_as_of(..., timestamp: SystemTime)` | 🟡 QUEUED | M | `contract::property::EntityStore` + `lance-graph::graph::versioned` | **Already partially exists** via `VersionedGraph::at_version` — this chunk = expose through trait |
| **LF-32** | Sharded write paths: per-shard write coordinator | 🔵 DEFERRED | L | — | Not needed below ~10 GB tables; SMB workload << this. Revisit when first tenant scales |
| **LF-33** | Secondary index trait: `SecondaryIndex { kind: BTree | Inverted | Hash, predicate: &str }` + Lance wiring | 🟡 QUEUED | M | `contract::index` (new module) + Lance impl | CAM-PQ already covers similarity; this covers predicate filters |

### Stage 4 — Search (LF-40..42)

Outside-BBB query side. Tantivy-based.

| LF | Chunk | Status | Effort | Residence | Notes |
|---|---|---|---|---|---|
| **LF-40** | Full-text search: `Searchable` predicate flag on `PropertySpec` + Tantivy-backed inverted index | 🟡 QUEUED | L (split: trait / index build / query) | `contract::property::PropertySpec` + new `lance-graph-search` crate | Sub-second object search |
| **LF-41** | Faceted aggregation API: `Search { filters, facets, sort, page }` over `EntityStore` | 🟡 QUEUED | M | `contract::search` (new module) | Standard "filter + facet + sort + page" shape |
| **LF-42** | Fuzzy / typo-tolerant search via Levenshtein on inverted index | 🟡 QUEUED | M | `lance-graph-search::fuzzy` | Builds on LF-40 |

### Stage 5 — Models (LF-50..53)

Generic model artifact + provider tier. xAI gRPC, OpenAI, Anthropic,
Ollama all fit through `LlmProvider`.

| LF | Chunk | Status | Effort | Residence | Notes |
|---|---|---|---|---|---|
| **LF-50** | `ModelRegistry { models: Vec<ModelArtifact>, versions, deployments }` trait | 🟡 QUEUED | M | new `lance-graph-models` crate | Foundry Model Registry equivalent |
| **LF-51** | `ModelDeployment { model_id, version, status, endpoint }` lifecycle | 🟡 QUEUED | M | `lance-graph-models` | Model deployment state machine |
| **LF-52** | LLM endpoint wrapper trait: `LlmProvider::generate(prompt, hints) -> Stream<Token>` | 🟡 QUEUED | S | `contract::llm` (new module) | Matches existing `Reasoner` trait shape; multi-provider via enum dispatch |
| **LF-53** | "AIP Logic" equivalent: visual blackboard composition UI | 🔵 DEFERRED | L | Q2 UI project, NOT lance-graph | Out-of-repo scope; depends on UI infrastructure that doesn't exist yet |

### Stage 6 — Decisions (LF-60..62)

Workflow over `ActionSpec` + `Blackboard`. NARS revision wires the
human-in-the-loop loop into the cognitive substrate.

| LF | Chunk | Status | Effort | Residence | Notes |
|---|---|---|---|---|---|
| **LF-60** | `Approval { action_id, requested_by, approvers: Vec<RoleId>, status }` workflow trait | 🟡 QUEUED | M | `contract::approval` (new module) | Wraps `ActionSpec` with approval gating |
| **LF-61** | Decision capture: user corrections flow back as `NarsRevision` on the corrected SPO triple | 🟡 QUEUED | M | `contract::nars` + AriGraph hook | Already wired-but-dormant — `NarsInference::Revision` exists; needs `correct_triple()` API |
| **LF-62** | Webhook trigger from `ActionSpec.on_commit` | 🟡 QUEUED | S | `lance-graph-callcenter::webhook` (new module) | Reuses existing `ActionTrigger` enum |

### Stage 7 — Scenarios (LF-70..72) — 🔴 REDESIGNED in this branch

Original LF-70/71/72 spec proposed inserting a `scenario_id` column into
every BindSpace SoA + SPO row — would have widened the SIMD sweep by
8 bytes/row and duplicated Lance's native versioning.

**This branch (`claude/scenario-world-facade`, commit `521f946`) ships
the architecturally correct counter-proposal:**

| LF | Original | Redesign | Status | Where |
|---|---|---|---|---|
| **LF-70** | `World::fork(branch_name)` full impl | Same; thin wrapper composing `archetype::World::fork` (now wired) + `VersionedGraph::tag_version` + RNG seed capture | 🟢 IN-PR | `contract::scenario::ScenarioBranch::new` + `archetype::world::World::fork` |
| **LF-71** | `scenario_id` column on every BindSpace SoA + SPO row | **DROPPED.** Scenario identity = role-bind in trajectory + dataset-path branch identity. No new column, no SIMD widening. Composable with grammar/persona/callcenter role catalogues. | 🔴 REDESIGN | rationale in `.claude/agents/scenario-world.md` §"Why not LF-71 column" |
| **LF-72** | `diff(base, fork) -> Vec<RowDiff>` | Same shape; `ScenarioDiff` composes three resolutions: graph-node-diff (`VersionedGraph::diff`) + fingerprint-diff (`worlds_differ`) + gestalt-diff (`WorldModelDto.field_state.dissonance`) | 🟢 IN-PR | `contract::scenario::ScenarioDiff` |

**New chunks added in the same branch (Tier-2.5 — finish wiring):**

| LF | Chunk | Status | Where |
|---|---|---|---|
| **LF-73** | `ScenarioWorld::simulate_forward(branch, steps, model)` — wires the dormant `NarsInference::CounterfactualSynthesis` (key slot exists, never called) into a forward-walking loop, optionally consulting an ONNX `ModelBinding` per step | 🟡 QUEUED | `lance-graph-cognitive::world` impl of `ScenarioWorld` trait |
| **LF-74** | `ScenarioWorld::forecast_palette(branch, depth)` — Chronos-extracted **method** (not crate): chained `compose[a][b]→c` lookups over the existing 256-archetype palette + ComposeTable. ~2 ns/step. | 🟡 QUEUED | `lance-graph-cognitive::world` impl |
| **LF-75** | `ScenarioWorld::replay(branch)` — Apache-Temporal-extracted **method**: deterministic replay using the captured `fork_seed`. Reproducibility by construction. | 🟡 QUEUED | `lance-graph-cognitive::world` impl |

**Key decisions documented in `.claude/agents/scenario-world.md`:**

1. **Lance versioning ≠ explicit branching.** Versioning is read-as-of (immutable, monotonic); branching is write-divergent (named, mutable forward).
2. **`scenario_id` column rejected.** Per CLAUDE.md `I-VSA-IDENTITIES`: VSA carries identity, not content state. Scenario identity belongs as a role-bind, not a row column.
3. **Pearl Rung 3 already shipped.** `lance-graph-cognitive::world::counterfactual::intervene()` does do-calculus on fingerprint world states. ScenarioBranch composes this, doesn't replace it.
4. **Chronos as method, not crate.** Adopt the palette-compose-chain forecast idea; do NOT pull in the model. We already have the palette + ComposeTable.
5. **Apache Temporal as method, not infra.** Adopt deterministic replay (fork_seed); do NOT adopt durable-execution workflow runtime. That's wrong for simulation.
6. **Archetype as scenario prior.** 144 archetype identity fingerprints already exist (12 families × 12 voice channels in ndarray::hpc::audio). Bundling an archetype into a branch's trajectory biases all forward inference toward that archetype — for free, no new code.

### Stage 8 — Marketplace (LF-80..81)

| LF | Chunk | Status | Effort | Residence | Notes |
|---|---|---|---|---|---|
| **LF-80** | `OntologyBundle { ontology, schemas, examples, version, signature }` + signing | ⚫ FUTURE | M | `contract::bundle` (new) | Useful when first regulated-industry tenant arrives (DATEV, GoBD, BaFin) — until then, speculative |
| **LF-81** | Cross-tenant install API: `Bundle::install(target_namespace, role_mapping)` | ⚫ FUTURE | M | `lance-graph-callcenter::bundle` | Depends on LF-80 |

### Cross-cutting (LF-90..92) — ALL ✅ DONE

Already covered in Tier 1 above. Summary:

| LF | Status | Commit |
|---|---|---|
| **LF-90** AuditEntry + AuditLog | ✅ DONE | `76a7237` (PR #263) |
| **LF-91** SlaPolicy + SlaPriority | ✅ DONE | `e70f944` (PR #263) |
| **LF-92** TenantId + TenantScope | ✅ DONE | `e70f944` (PR #263) |

---

## Status summary by stage

| Stage | Done | In-PR | Queued | Deferred | Future | Total |
|---|---|---|---|---|---|---|
| Tier 1 (LF-1..8) | 8 | 0 | 0 | 0 | 0 | 8 |
| Stage 1 — Data Integration | 0 | 0 | 5 | 0 | 0 | 5 |
| Stage 2 — Ontology | 2 | 0 | 2 | 0 | 0 | 4 |
| Stage 3 — Storage v2 | 0 | 0 | 3 | 1 | 0 | 4 |
| Stage 4 — Search | 0 | 0 | 3 | 0 | 0 | 3 |
| Stage 5 — Models | 0 | 0 | 3 | 1 | 0 | 4 |
| Stage 6 — Decisions | 0 | 0 | 3 | 0 | 0 | 3 |
| Stage 7 — Scenarios | 0 | 2 | 3 | 0 | 0 | 5 (LF-70/72 IN-PR; LF-71 redesigned; LF-73/74/75 new) |
| Stage 8 — Marketplace | 0 | 0 | 0 | 0 | 2 | 2 |
| Cross-cutting (LF-90..92) | 3 | 0 | 0 | 0 | 0 | 3 |
| Wishlist (W-1..4) | 4 | 0 | 0 | 0 | 0 | 4 |
| **TOTAL** | **17** | **2** | **22** | **2** | **2** | **45** |

**38% shipped (17/45)**, **44% queued with REQUEST**, **18% deferred or future**.

---

## Sequencing — what to ship next

Ordered by leverage (unblocks the most downstream work first):

1. **Merge `claude/scenario-world-facade` PR** — closes Stage 7 (LF-70/72); ships rationale agent; LF-73/74/75 placeholders documented.

2. **LF-12 Pipeline DAG** (Stage 1 keystone) — adds `UnifiedStep.depends_on: Vec<StepId>` field + a topological executor in `lance-graph-planner`. Once shipped, LF-10/11/13/14 all become straightforward additions.

3. **LF-50 + LF-52 ModelRegistry + LlmProvider** (Stage 5) — small (S effort each), unlocks generic model dispatch for xAI gRPC + OpenAI + Anthropic + Ollama through one trait.

4. **LF-20 FunctionSpec** (Stage 2) — pure DTO addition to `contract::ontology`. Foundry Functions equivalent. Useful for SMB defining German tax-rule transforms.

5. **LF-23 NotificationSpec** (Stage 2) — small. Reuses `ActionTrigger`.

6. **LF-31 scan_as_of** (Stage 3) — almost free; just exposes the existing `VersionedGraph::at_version` through the `EntityStore` trait.

7. **LF-10 Connector registry** (Stage 1) — establishes the connector tier; LF-11 (Postgres) follows immediately.

8. **LF-61 NARS-revision-on-correction** (Stage 6) — wires existing `NarsInference::Revision` slot. Closes the human-in-the-loop loop.

9. **LF-73 simulate_forward** (Stage 7) — wires `CounterfactualSynthesis` inference into the ScenarioBranch facade.

10. **LF-40 Tantivy full-text** (Stage 4) — large but high SMB value (customer search).

After this sequence, **Tier 2 is ~85% complete** with only the four
deferred/future chunks (LF-32 sharded, LF-53 UI, LF-80/81 marketplace)
unbuilt — all of which have explicit "wait for trigger" markers.

---

## Cross-references

- **SMB-side mirror:** `smb-office-rs::docs/foundry-parity-checklist.md` (pulled via direct git protocol to `/tmp/sources/smb-office-rs/`).
- **Bus log of every shipped chunk:** `.claude/board/CROSS_SESSION_BROADCAST.md` (claude/blackboard branch).
- **Architectural rationale (LF-70..75):** `.claude/agents/scenario-world.md`.
- **Ripple model (gestalt theory):** `.claude/knowledge/user-agent-topic-ripple-model.md`.
- **VSA-as-Layer-2-catalogue iron rule:** `CLAUDE.md` § I-VSA-IDENTITIES.
- **Lance-native versioning (the substrate this all sits on):** `crates/lance-graph/src/graph/versioned.rs`.

---

## Open questions

1. **LF-12 split.** Schema field add + executor + cron-trigger as one PR, or three? Recommend: **one PR for schema + executor**, separate PR for cron (LF-13 territory).
2. **LF-32 trigger threshold.** What table size triggers "now we need sharded writes"? Recommend documenting "≥10 GB sustained per-tenant" as the empirical threshold; revisit when first tenant approaches it.
3. **LF-53 ownership.** Out-of-repo for sure, but does Q2 UI plan to consume `contract::scenario::ScenarioBranch` directly, or via a REST shim in `lance-graph-callcenter`? Recommend REST shim — UI shouldn't depend on the contract crate types.
4. **LF-80/81 trigger.** Which regulated-industry tenant first signs (BaFin? GoBD? DATEV?) determines the bundle signature scheme. Defer until concrete.

