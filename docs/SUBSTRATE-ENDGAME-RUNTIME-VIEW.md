# Substrate Endgame — runtime view (lance-graph slice)

> **Purpose.** Tailored view onto the substrate-b endgame architecture
> from `AdaWorldAPI/lance-graph`'s perspective. The master doc lives in
> `AdaWorldAPI/OGAR/docs/SUBSTRATE-ENDGAME.md` (full five-rooms map);
> this doc highlights the runtime-side slice and cross-references the PRs
> in this repo that land each room's runtime dependencies.
>
> **Why a tailored view.** The endgame architecture spans four repos
> (OGAR, lance-graph, ractor_actors, openproject-nexgen-rs) plus the
> ruff fork producers and the surrealdb fork DDL builders. The master
> doc in OGAR is comprehensive but spans all of it. This view lets a
> lance-graph session pick up the runtime-side concerns without reading
> the full master (which they should still read for context).
>
> Companion: `OGAR/docs/ARCHITECTURAL-DECISIONS-2026-06-04.md` (ADR-style
> backward-looking session capture). ADR-008 (`commit_event` sibling)
> and ADR-009 (`temporal` two-axis engine) are the lance-graph-owned
> decisions; ADR-018 (Kanban polyglot dispatcher) is the runtime-side
> architecture this view foregrounds.
>
> Status: **CARVED v0** (2026-06-05). Mirror of `OGAR/docs/SUBSTRATE-
> ENDGAME.md` for runtime-side concerns. Master doc is authoritative;
> this view stays consistent with it on per-quarter review.

## 0. Five-rooms map — what lance-graph owns

```
ROOM 1 (TODAY)             What lance-graph ships
─────────────────────      ─────────────────────────────────────────
substrate primitives  ──►  lance-graph-contract (canonical types)
                           lance-graph-ontology (OntologyRegistry,
                             MappingProposal, SchemaSource, TTL
                             hydrators incl. wikidata_hhtl)
                           lance-graph-callcenter (LanceMembrane +
                             commit_event sibling — PR #467, merged)
                           lance-graph-planner::temporal (classify +
                             deinterlace + EpistemicMode + HLC-aware
                             QueryReference + DependsClosure trait
                             — PR #468, open)
                           lance-graph-supervisor (ractor-supervised
                             callcenter actor tree)
                           cognitive-shader-driver (CognitiveEventRow,
                             Markov ±5 trajectory)

ROOM 2 (MIGRATION SCAFFOLD)   What lance-graph adds
─────────────────────────     ─────────────────────────────────────
Kanban polyglot interface ──► formalize the work-item trait that admits
                              multiple executable forms (native ractor
                              handler / BEAM call / interpreter / HTTP
                              sidecar / CRuby / reflection-dump)
                              — Kanban primitive in SOA-IMPLEMENTATION §5.2
                              — new work-item-form-registration trait

§14 oracle harness        ──► record-replay protocol for per-actor
                              graduation verification:
                              record(migration form) -> tape ->
                              replay(native candidate) -> verdict
                              bucket (PASS / DIVERGENT-RECONCILABLE /
                              DIVERGENT-FAULTY / INDETERMINATE)

HTTP-sidecar bridge        ──► Rust HTTP client issuing per-action POSTs
                              to a Rails Puma sidecar; decoding responses
                              to Transition<State> + commit row

BEAM bridge (optional)     ──► Erlang Port or NIF for HIRO/Bardioc-side
                              migration; alternative: tiny Elixir AST
                              interpreter crate

ROOM 3 (OP-AS-OPERATOR-PANE)  What lance-graph enables
─────────────────────────     ─────────────────────────────────────
OP hosted on substrate     ──► callcenter actors per OGAR Class
                              (WorkPackage, Project, Status, etc.)
                              Hotwire / version_watcher push stream
                              for live UI updates

Workflow → live Rubicon    ──► RubiconWriter Phase 2 supports dynamic
                              regeneration when Workflow table rows
                              change (operator admin UI edits propagate
                              to in-process Rubicon machines without
                              restart)

ROOM 4 (VISUALIZATION)        What lance-graph emits
─────────────────────         ─────────────────────────────────────
OpenTelemetry exposition  ──► Lance commits/sec, ractor mailbox depth,
                              Rubicon transition counters, temporal
                              classify ratios (CONTEMPORARY /
                              ANACHRONISTIC / SPOILER), HLC drift,
                              §14 verdict ratios — standard Prometheus
                              metric exposition

Substrate event stream    ──► version_watcher → WebSocket / SSE for
                              live operator-UI updates; cognitive-event-
                              row stream for sexy-tier visualizations

ROOM 5 (SDK ENDGAME)          What lance-graph contributes
─────────────────────         ─────────────────────────────────────
Stable public API         ──► SemVer-pin lance-graph-contract /
                              lance-graph-ontology / lance-graph-planner /
                              lance-graph-callcenter (each ratchets to 1.0
                              on their own timeline)

Runtime documentation    ──► getting-started for substrate runtime
                              consumers (separate from OGAR's
                              producer-side getting-started)

Reference deployment     ──► substrate-b hosting OP in production;
                              prove the runtime side end-to-end
```

## 1. What lance-graph owns in each room (detail)

### 1.1 Room 1 — current floor (shipped on `main`)

- **`lance-graph-contract`** — zero-dep canonical types
  (`Schema`, `LinkSpec`, `SemanticType`, `Marking`, `PropertySpec`,
  `Cardinality`, `CodecRoute`, `ExternalMembrane`). Implementation
  crates depend on this; this crate depends on nothing. Status: stable.
- **`lance-graph-ontology`** — `OntologyRegistry` + `MappingProposal`
  + `SchemaSource` trait + TTL hydrators (SKOS, PROV-O, schema.org,
  FIBO, Odoo, ZUGFeRD, SKR03/04, **`wikidata_hhtl`**) + 47KB Lance
  dictionary cache. `odoo_blueprint` provides 15-lane typed
  `OdooEntity` consts (OGAR's runtime-IR equivalent for Odoo source).
  Status: stable; `lance-bind` Sprint-5b is the receiver of OGAR-side
  `MappingProposal` flow.
- **`lance-graph-callcenter`** — `ExternalMembrane` impl;
  `LanceMembrane::commit_event(row: CognitiveEventRow) -> u64`
  sibling shipped in **PR #467, merged**. Pairs with
  `ExternalMembrane::project()` (cognitive-cycle path). Action-commit
  path skips `ShaderBus`. Per ADR-008.
- **`lance-graph-planner::temporal`** — the deinterlace engine, in
  **PR #468 (open as of 2026-06-04 → may be merged by future
  sessions)**. Surface: `EpistemicMode {Strict, Aware, Retro}` +
  `QueryReference {server_id, ref_version, hlc_tick: Option<u64>,
  mode, rung}` + `classify(row_version, knowable_from, v_ref) ->
  Classification` + `deinterlace(rows, v_ref, deps)` +
  `DependsClosure` trait (TIME-causal `hlc_tick` + DATA-causal
  `DependsClosure` axes). Per ADR-009.
- **`lance-graph-supervisor`** — ractor-supervised callcenter actor
  tree (PR-G2 / TD-RACTOR-SUPERVISOR-5).
- **`cognitive-shader-driver`** — cognitive-cycle row producer
  (the substrate's other write path; co-located with Lance).

### 1.2 Room 2 — migration-scaffold runtime side

Three pieces lance-graph adds for the Kanban-as-polyglot-dispatcher
pattern (per ADR-018):

1. **Work-item-form trait** — the Kanban dispatcher's interface that
   admits multiple executable forms (native ractor handler / BEAM
   call / interpreter / HTTP RPC / CRuby FFI / reflection-dump-only).
   Probably lives in `lance-graph-callcenter` next to the existing
   actor registration. Architecture pinned in OGAR PR #20
   (`SUBSTRATE-ENDGAME.md §2`); implementation pending.

2. **HTTP-sidecar bridge** — Rust client issuing `POST /api/v3/...`
   to a Rails Puma deployment, decoding JSON responses to
   `Transition<State>` + commit row. Engineering: standard HTTP
   client + retry/auth/telemetry; new crate or module in callcenter.

3. **§14 oracle harness** — record-replay infrastructure for
   per-actor graduation verification. Per `SUBSTRATE-ENDGAME.md §2.4`:
   record tape against migration form, replay against native
   candidate, compare provenance-normalized, emit verdict bucket.

Optional fourth: **BEAM bridge** (Erlang Port or NIF) for HIRO/Bardioc
migration. Alternative: tiny Elixir AST interpreter crate.

### 1.3 Room 3 — OP-as-operator-pane runtime side

When OP graduates onto substrate-b per Room 2:

- **Callcenter actors per OGAR Class** — OP's WorkPackage, Project,
  Status, User, Role, Workflow each get a ractor actor in the
  substrate's supervision tree.
- **Hotwire / version_watcher push stream** — OP's existing
  WebSocket / Hotwire stream wires through to
  `lance-graph-callcenter::version_watcher` so the UI updates from
  substrate state directly.
- **RubiconWriter Phase 2 dynamic regeneration** — when OP's
  Workflow table rows change (operator admin UI), the in-process
  Rubicon machine for the affected class regenerates without
  restart. Requires Rubicon-from-OGAR codegen to support dynamic
  emission; runtime session's Phase 2 work.

### 1.4 Room 4 — visualization tier-stack runtime side

The substrate is observable by design; runtime exposes the metrics:

- **OpenTelemetry / Prometheus exposition** from existing
  instrumentation crates. Standard `tracing-opentelemetry` setup;
  emit per-actor and per-class metrics, Lance-dataset stats,
  deinterlace ratios.
- **Cognitive-event-row stream** for sexy-tier visualizations.
  Existing `cognitive-shader-driver` row producer; expose via
  WebSocket / SSE for the four-frame deinterlace visualizer + the
  cognitive trajectory animation.
- **Actor-tree topology API** — runtime endpoint exposing the live
  supervision tree shape for 3D-topology visualizations
  (CytoscapeJS / three.js).

### 1.5 Room 5 — SDK endgame runtime side

- **Stable public API across lance-graph crates** — each crate
  ratchets to 1.0 on its own timeline; SemVer pinning; deprecation
  paths documented; cross-crate version compatibility matrix.
- **Runtime getting-started doc** — for substrate consumers
  (separate from OGAR's producer-side getting-started). Cover:
  setting up a Lance dataset, registering classes via SchemaSource,
  dispatching work-items through callcenter, reading via temporal
  classify, etc.
- **Reference deployment** — substrate-b hosting OP in production;
  the visible run-time validation point.

## 2. Cross-references

### 2.1 Master doc (OGAR)

- `AdaWorldAPI/OGAR/docs/SUBSTRATE-ENDGAME.md` — the comprehensive
  five-rooms architecture; this view is the runtime slice.
- `AdaWorldAPI/OGAR/docs/ARCHITECTURAL-DECISIONS-2026-06-04.md` —
  ADR-style backward-looking session capture; ADR-008 (commit_event),
  ADR-009 (temporal two-axis), ADR-018 (Kanban polyglot) are the
  lance-graph-touching decisions.
- `AdaWorldAPI/OGAR/docs/OGAR-AST-CONTRACT.md` — the typed surface
  callcenter actors lower onto; §3 binding references temporal +
  CommitHook + commit_event.
- `AdaWorldAPI/OGAR/docs/SURREAL-AST-AS-ADAPTER.md` — structural-vs-
  behavioral decision (ADR-016); §6 covers migration scaffold
  counterpoint.

### 2.2 Companion runtime references

- `AdaWorldAPI/ractor_actors` — `feat/state-machine-actor @ 38a71a4`
  is the canonical `StateMachine` crate Rubicon-from-OGAR binds onto
  (per ADR-007).
- `AdaWorldAPI/lance-graph/PR #467` — `LanceMembrane::commit_event`
  sibling (merged).
- `AdaWorldAPI/lance-graph/PR #468` — `temporal::classify` +
  `deinterlace` + `DependsClosure` (open as of 2026-06-04).
- `AdaWorldAPI/bardioc/CROSS_SESSION_COORDINATION.md` — the
  authoritative cross-session coord doc (runtime-session-owned).
  `knowable_from` meet-point pin (per ADR-010) mirrors from
  `OGAR/docs/OPENPROJECT-TRANSCODING.md §10.3`.

## 3. Open items lance-graph owns

In addition to the in-flight items listed in OGAR's master doc:

- **Cross-server HLC merge policy** — `QueryReference.hlc_tick: Option<u64>`
  is HLC-aware in signature; cross-server merge policy is deferred
  body work. Lands when peer-Raft / cluster bus comes online.
- **`temporal::deinterlace` postpone-replay-ordering test at scale** —
  single-actor FIFO is verified; multi-actor concurrent postpone-
  queue interaction at scale is the harder property.
- **Work-item-form trait + per-actor registration table** — Room 2
  primary lance-graph work item.
- **§14 oracle harness** — record-replay infrastructure. Pairs with
  the OGAR-side producer; runtime owns the storage + replay layer.
- **RubiconWriter Phase 2** (Rubicon's durable home + `KvLanceWriter`
  backend) — pairs with `LanceMembraneWriter`; both share the same
  Lance 7.0.0 commit contract.

## 4. Doc lifecycle

- **Author:** OGAR session 2026-06-04 (placed cross-repo 2026-06-05).
- **Status:** Mirror view; master in OGAR.
- **Update cadence:** when the master doc (`OGAR/docs/SUBSTRATE-
  ENDGAME.md`) updates, mirror relevant changes here. When
  lance-graph-side items graduate (e.g. Room 2 work-item trait
  lands), add a one-line update to the relevant section.
- **Authority:** master in OGAR. This view is for navigation;
  decisions cite OGAR docs.

## 5. Compact map — runtime-side dependencies

For the runtime session to pick up where this view leaves off, in
approximate priority order:

1. **`lance-graph PR #468` lands `temporal` on main** — unblocks
   downstream consumers (Rubicon's repointing from local placeholder
   to the canonical `temporal::classify`).
2. **Rubicon's durable home + Phase 2** — `RubiconWriter` two
   backends (`LanceMembraneWriter` + `KvLanceWriter`).
3. **Work-item-form trait in callcenter** — enables Room 2.
4. **HTTP-sidecar bridge** — first migration scaffold variant.
5. **§14 oracle harness** — verifies graduation.
6. **First OP per-actor graduation end-to-end** — `WorkPackage#save`
   via HTTP sidecar → native ractor handler.
7. **OpenTelemetry exposition + Grafana panels** — Boring tier
   visualization.
8. **Stable public API + SemVer pinning** — Room 5 prep.

Each unlocks the next; total timeline per master doc §6.7 is
12-24 months from Room 1 to demonstrable Room 5.
