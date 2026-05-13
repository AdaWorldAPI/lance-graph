# Q2 Foundry-Equivalent Integration Plan — v1

> **Status:** Proposed (2026-04-24)
> **Owner:** @integration-lead, @truth-architect
> **Scope:** Q2 = user interface for the cognitive stack; SMB = first tenant / testbed
> **Depends on:** 8 PRs merged this session (#253-#260)

---

## Vision

Q2 is the Palantir Gotham/Workshop/Vertex equivalent: graph exploration,
property scrolling, Cypher queries, live neural-debug overlay, action
triggers, scenario branching. SMB (small business — Steuerberater data)
is the first tenant consuming Q2, providing the reality-check testbed
for the entire cognitive stack.

Foundry's frame: "end-to-end data operating system."
Our frame: **"end-to-end cognitive operating system"** — same layers but
with active inference, NARS truth, and CausalEdge64 as substrate.

---

## Architecture: Firefly Stack

```
┌─────────────────────────────────────────────────────────────────┐
│ Q2 UI (TypeScript / React / Solid)                              │
│   Object Explorer — property scrolling (PrefetchDepth L0→L3)    │
│   Cypher Console — polyglot query input (GQL/Cypher/SPARQL)     │
│   Neural Debug Overlay — live BindSpace health + NeuronState     │
│   Action Panel — trigger ActionSpec (Manual/Auto/Suggested)      │
│   Scenario Viewer — World::fork() branches + diff               │
├─────────────────────────────────────────────────────────────────┤
│ Transport Layer                                                  │
│   JSON REST (:3001)      — queries, encode, dispatch, admin      │
│   Arrow Flight SQL (:3002) — bulk graph traversals, streaming    │
│   SSE/WebSocket (:3001/subscribe) — realtime push from watcher   │
├─────────────────────────────────────────────────────────────────┤
│ Wire Schema (serde DTOs — protocol-agnostic)                     │
│   WireUnifiedStep → OrchestrationBridge::route()                 │
│   WireDispatch → ShaderDriver::dispatch()                        │
│   WireEncode → DeepNSM → BindSpace content row                  │
│   WireBlackboard → A2A Blackboard + NARS reasoning rounds       │
├─────────────────────────────────────────────────────────────────┤
│ Cognitive Engine (Rust)                                          │
│   BindSpace 4096 (0x000..0xFFF) — content/cycle/topic/angle     │
│   ShaderDriver — meta prefilter + Hamming cascade + palette      │
│   CypherBridge — lg.cypher → parse → SPO commit/query            │
│   OrchestrationBridge — nd.*/lg.* domain routing                 │
│   RBAC Policy — PermissionSpec × PrefetchDepth × ActionSpec      │
├─────────────────────────────────────────────────────────────────┤
│ Ontology Layer (contract crate, zero-dep)                        │
│   PropertySchema (Required/Optional/Free × CodecRoute)           │
│   LinkSpec (typed edges, Cardinality)                             │
│   ActionSpec (Manual/Auto/Suggested triggers)                    │
│   ModelBinding + ModelHealth (NARS-based monitoring)              │
│   SimulationSpec (World::fork() what-if parameters)              │
├─────────────────────────────────────────────────────────────────┤
│ Storage (Lance + AriGraph)                                       │
│   SPO triple store — CausalEdge64 per edge (Pearl 2³ masks)      │
│   Lance versioned datasets — World::fork() = branch              │
│   CAM-PQ compressed search — 6-byte fingerprints, O(1)          │
│   Episodic memory — ±5 Markov window, VSA trajectory             │
├─────────────────────────────────────────────────────────────────┤
│ Firefly Repository                                               │
│   Ballista — distributed DataFusion execution                    │
│   Dragonfly — fast-path CPU lane (BindSpace sweep, JIT kernels)  │
│   GEL — Graph Execution Language (ArenaIR → JIT → native)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deliverables — 4 Phases

### Phase 1: Make the stack demo-able (Q2 MVP)

Target: browser-accessible Cypher console + graph viz + property scroll
against real SMB data. Minimum viable Gotham.

| D-id | Deliverable | Effort | Blocked by |
|---|---|---|---|
| Q2-1.1 | **Cypher Console UI** — textarea → POST /v1/shader/route → render result | small | nothing |
| Q2-1.2 | **Object Explorer** — node click → GET properties at PrefetchDepth L0→L3 | small | Q2-1.1 |
| Q2-1.3 | **Neural Debug Overlay** — wire neural-debug into /v1/shader/health, render NeuronState per row | medium | nothing |
| Q2-1.4 | **Graph Viz** — force-directed graph from LinkSpec edges, node = entity, edge = predicate | medium | Q2-1.1 |
| Q2-1.5 | **SMB seed data** — load Customer/Invoice/TaxDeclaration schemas + sample data via /v1/shader/encode | small | nothing |
| Q2-1.6 | **RBAC gate on endpoints** — Policy.evaluate() middleware on Axum routes | small | nothing |
| Q2-1.7 | **Cypher Phase 2** — wire real parser (add CREATE to core), SPO commit on CREATE, BindSpace search on MATCH | medium | CypherBridge PR #258 |

### Phase 2: Make the stack operational (Q2 workflows)

Target: actions, decision capture, writeback, streaming. Gotham → Workshop equivalent.

| D-id | Deliverable | Effort | Blocked by |
|---|---|---|---|
| Q2-2.1 | **Action Panel** — trigger ActionSpec from UI, POST /v1/shader/route step_type="lg.action" | medium | Q2-1.2 |
| Q2-2.2 | **Decision Capture** — user corrections flow back as NARS revisions on affected triples | medium | Q2-2.1 |
| Q2-2.3 | **SSE/WebSocket subscribe** — /v1/shader/subscribe endpoint wired to LanceVersionWatcher (DM-4) | medium | PR #255 (merged) |
| Q2-2.4 | **Streaming property updates** — Q2 UI subscribes to entity changes, updates in-place | medium | Q2-2.3 |
| Q2-2.5 | **DrainTask runtime** — DM-6 scaffold → real drain of steering_intent via OrchestrationBridge | medium | PR #255 (merged) |
| Q2-2.6 | **FunctionSpec** — callable logic bound to entity types (Foundry Functions equivalent) | small | nothing |
| Q2-2.7 | **Orchestration DAG** — UnifiedStep.depends_on + topological scheduler | medium | nothing |

### Phase 3: Make the stack intelligent (Q2 reasoning)

Target: Blackboard endpoint, NARS multi-expert reasoning, FreeEnergy-gated
auto-commit, scenario branching. The AGI glove fully wired.

| D-id | Deliverable | Effort | Blocked by |
|---|---|---|---|
| Q2-3.1 | **Blackboard REST endpoint** — POST/GET /v1/shader/blackboard/{post,read,advance} | medium | nothing |
| Q2-3.2 | **NARS reasoning rounds** — blackboard/advance triggers deduction/revision/abduction across entries | large | Q2-3.1 |
| Q2-3.3 | **FreeEnergy-gated auto-commit** — dispatch + F < 0.2 → auto-commit SPO triple to AriGraph | medium | PR #259 (merged, resonance works) |
| Q2-3.4 | **Scenario branching UI** — World::fork() from Q2, show base vs fork diff, SimulationSpec params | medium | Q2-1.4 |
| Q2-3.5 | **Content-to-SPO pipeline** — /v1/shader/encode → dispatch → FreeEnergy → Resolution → AriGraph commit (end-to-end) | large | Q2-3.3 |
| Q2-3.6 | **Scenario-aware ontology** — scenario_id column on BindSpace/SPO so rows know which branch they're in | medium | Q2-3.4 |
| Q2-3.7 | **CausalEdge64 explorer** — Q2 UI shows Pearl 2³ mask, NARS truth, inference type per edge | small | Q2-1.4 |

### Phase 4: Make the stack fly (Q2 at scale)

Target: Arrow Flight for bulk queries, Ballista for distributed execution,
GEL for graph-executable JIT. The Firefly Repository.

| D-id | Deliverable | Effort | Blocked by |
|---|---|---|---|
| Q2-4.1 | **Arrow Flight SQL server** — :3002, FlightSqlService impl wrapping OrchestrationBridge | large | nothing |
| Q2-4.2 | **Ballista integration** — distributed DataFusion queries across nodes | large | Q2-4.1 |
| Q2-4.3 | **GEL IR** — formal graph-executable language spec, ArenaIR → GEL → JIT | large | nothing |
| Q2-4.4 | **Dragonfly CPU lane** — fast-path BindSpace sweep optimized for single-node hot queries | medium | nothing |
| Q2-4.5 | **Data lineage** — LineageEdge tying output rows to input rows + transform step_ids | medium | Q2-2.7 |
| Q2-4.6 | **Data connectors** — EntityStore impls: S3, Postgres, Mongo, DATEV (SMB-specific) | N × small | nothing |
| Q2-4.7 | **Marketplace** — signed ontology bundles shareable across tenants | large | Q2-4.6 |

---

## SMB as Testbed

SMB (small business / Steuerberater) exercises every layer:

| Q2 feature | SMB reality check |
|---|---|
| Object Explorer | Scroll customer → see Required tax_id, Optional address, Free notes |
| Cypher Console | `MATCH (c:Customer)-[:issued]->(i:Invoice) WHERE i.total > 10000` |
| Actions | "Approve invoice" → ActionSpec(Manual) → NARS revise status triple |
| Decision Capture | Accountant corrects OCR'd tax_id → NARS frequency 0.4 → 0.95 |
| Neural Debug | "Why is this customer flagged?" → NeuronState shows which rows resonate |
| RBAC | Accountant sees Detail (L1), Auditor sees Full (L3), Admin sees + writes |
| Streaming | New invoice arrives → LanceVersionWatcher bumps → Q2 updates in-place |
| Scenarios | "What if this customer switches to quarterly invoicing?" → World::fork() |
| FreeEnergy | Missing Required property → F > 0.8 → FailureTicket → UI shows alert |
| CausalEdge64 | Invoice → payment → customer satisfaction: direct cause (001) with NARS 0.7/0.9 |

---

## What we have ON TOP of Foundry (differentiators to protect)

1. Active inference as dispatch mechanism (not bolt-on)
2. NARS truth values as primary data (not metadata)
3. CausalEdge64 with Pearl 2³ masks (8 causal types per edge)
4. VSA algebra on role-indexed identities (coreference via unbind)
5. CAM-PQ O(1) compressed similarity
6. JIT-compiled sensor lenses (10μs/sentence, no ONNX runtime)
7. Zero-dep contract crate (~500 LOC, any consumer)
8. Polyglot query routing (Cypher/GQL/Gremlin/SPARQL → same IR)
9. 12 thinking styles as dispatch parameters
10. Three-layer agent coordination (teleport / file / branch pub-sub)
11. Content Hamming cascade (PR #259) — style-threshold-gated similarity
12. Chapman-Kolmogorov Markov by construction (VSA bundling)

---

## Non-goals (explicit)

- NOT replicating Foundry's 200+ connectors. Start with 3: S3, Postgres, DATEV.
- NOT building a full Workshop low-code builder. Start with Cypher console + graph viz.
- NOT multi-tenant SaaS in Phase 1. Single tenant (SMB testbed). Multi-tenant in Phase 4.
- NOT replacing Neo4j. Augmenting with causal typing + NARS + active inference that Neo4j can't do.

---

## Session 2026-04-24 — what shipped (foundation for this plan)

| PR | What | Phase it enables |
|---|---|---|
| #253 | Vsa16kF32 carrier + algebra | All (substrate) |
| #254 | Archetype scaffold | Phase 2 (scenarios) |
| #255 | Supabase subscriber (DM-4/6) | Phase 2 (streaming) |
| #256 | Cycle f32 migration | All (substrate) |
| #257 | SMB traits + Ontology + RBAC + encode | Phase 1 (demo) |
| #258 | CypherBridge | Phase 1 (queries) |
| #259 | Hamming content cascade | Phase 3 (reasoning) |
| #260 | AGENT_LOG split/gitignore | Governance |
