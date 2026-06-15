# Integrated Cognitive Planner — meticulous reference map (v1, 2026-06-15)

## READ BY / AGENT TARGET (mandatory)

This is THE reference doc for the integrated-cognitive-planner arc. Any savant
(expansion) or brutally-honest (hardening) agent **MUST cite this map by
`file:line`** and may not assert architecture not grounded here — if a claim
isn't in §1/§7, it's a hypothesis, not a fact. Grounded by a 5-agent research
sweep (2026-06-15, integrated-planner axes) + a 3-agent external-pattern sweep
(Google Pinpoint / TiKV / addressing). Capture-before-dilution of a long design
session (post-#495 merge; #495's `ValueSchema`/`EdgeCodecFlavor` ride #496).

**Verdict of the research:** ~90% of the substrate EXISTS across #437–#492 +
the unmerged `jolly-cori-clnf9`. The remaining work is **6 small additive
seams + the addressing/ScopedReference + the cognitive-cycle sequencer** — NOT
a new build. The recurring failure mode this doc prevents: re-deriving what
already ships.

---

## P0 — The architecture (4 layers). FORGET LADYBUG.

```
 SurrealDB  (BUILDING BLOCK — orchestrates the META AST/Elixir level; GOOD at it)
   │  returns the AST via the contract → ExecTarget::{SurrealQl, Elixir}
   │  NEVER thinks, NEVER writes. PROJECTS a read-view (kanban time-series).
   ▼
 lance-graph-planner :: PlannerAwareness   = the SHARED COORDINATION contract
   │  orchestrate → ThinkingContext (style / rung / ExecTarget)
   │  emits Vec<KanbanMove>(ExecTarget); rubicon-DAG lifecycle
   │  DELEGATES the thinking ▼            (it COORDINATES; it does not think)
   ▼
 thinking-engine → P64 → cognitive-shader-driver   = THE COGNITION
   │  Φ StreamDto → Ψ ResonanceDto → B BusDto  (thinking-engine/src/dto.rs)
   │  via p64-bridge convergence; driver.rs:581 dispatch + engine_bridge.rs:29
   │  ⚠ NOT `StepDomain::Ladybug` — that is legacy ladybug-rs (forget it;
   │    only residue is in lance-graph-cognitive, minimal impact).
   ▼
 ractor (lance-graph-supervisor) DRIVES the KanbanMove lifecycle
   │  MailboxSoaOwner::try_advance_phase (Planning→CognitiveWork→Evaluation→…)
   ▼  on Commit
 lance-graph-callcenter WRITES = the OUTER BOUNDARY
      lance_membrane.rs:315 commit_event → audit_sink/lance_sink.rs:292 (RocksDB/kvs-lance/ORM)
 SurrealDB PROJECTS the kanban columns read-only.

 temporal-aware throughout: Lance-version read-as-of + temporal.rs deinterlace (HLC).
```

---

## §1 — GROUNDED CURRENT STATE (what EXISTS, `file:line`)

**Planner core**
- `lance-graph-planner/src/lib.rs:99` `PlannerAwareness { strategies, selector }`; entry points `plan_full` (:171), `plan_auto` (:264); impls `OrchestrationBridge` (`orchestration_impl.rs`).
- 16 strategies (lib.rs:21 table), `selector.rs`, `compose.rs`, `pipeline.rs` (DAG), `cache/` (autocomplete), `ir/` (Polars-arena `LogicalPlan`, `ir/mod.rs:92`).
- `strategy/style_strategy.rs` (Strategy #18, **D-MBX-A6-P3a**): `reliability_of` (:168) over recipe kernels; `plan()` (:138) is a deliberate **pure pass-through — emits NO KanbanMove** (:144-151, "faking one here would be theatre").
- `PlanResult.emitted_edges: Vec<u64>` (lib.rs:123) — **always `Vec::new()`** at every construction site (:216/254/312 + api.rs).

**Temporal (the deinterlace = "klares zeitliches Bewusstsein")**
- `lance-graph-planner/src/temporal.rs` — 4-frame deinterlace (lance-version / surrealql `knowable_from` / ractor `V_ref` / cognitive trajectory) via HLC: `EpistemicMode{Strict,Aware,Retro}`+`for_rung` (:63-87), `TemporalStatus` (:90), `QueryReference{server_id,ref_version,hlc_tick,mode,rung}`+`::at` (:107-138), `classify` (:147), `DependsClosure`/`classify_ready`/`NoDeps` (:218-269), `deinterlace` (:301). Tested; **ZERO non-test callers** (only `temporal.rs` + lib.rs doc reference it).
- Lance read-as-of lives on the OTHER side of a dep wall: `lance-graph/src/graph/versioned.rs:419` `VersionedGraph::at_version → checkout_version` (real, tested). Planner deps `contract` NOT `lance-graph` core (anti-circular) → cannot reach it.
- Granger is a SEPARATE module: `lance-graph-cognitive/src/search/temporal.rs::granger_effect`.
- `prediction/{mod,scenario,temporal,ingestion}.rs` = simulated-time NARS rounds (abstract, not Lance versions).

**Kanban / ExecTarget (contract, #437 A6-P1 + #439 A6-P2, both merged)**
- `lance-graph-contract/src/kanban.rs:32` `KanbanColumn` Rubicon DAG (`next_phases`/`can_transition_to`/`is_absorbing`); `:112` `KanbanMove{mailbox,from,to,witness_chain_position,libet_offset_us,exec}` (`Copy`, ≤16 B); `:136` `ExecTarget{Native,Jit,SurrealQl,Elixir}`.
- `soa_view.rs:112` `MailboxSoaOwner::try_advance_phase` → `KanbanMove` | `RubiconTransitionError`.
- `scheduler.rs:46` `NextPhaseScheduler::on_version`; concrete `LanceVersionScheduler` deferred (scheduler.rs:20).

**Escalation / NARS / MUL (the cycle primitives)**
- `escalation.rs`: `CollapseHint{Flow,Fanout,RungElevate}` (:45), `fanout_width` (:59), `rung_delta(emergence,coherence)` (:75), `InnerCouncil::deliberate` (:144, 3 archetypes + split-amplify), `EpiphanyDetector::observe` (:241, surprise>baseline×1.5 ∧ window≥4), `GhostEcho{…,Staunen,Wisdom,…}` (:289), `WisdomMarker` (:312, decays to FLOOR 0.1). **Consumed** by `planner::mul::escalation`.
- `nars::InferenceType{Deduction,Induction,Abduction,Revision,Synthesis}`; carried on `ThinkingContext.inference_type` (plan.rs:19).
- `mul/` `MulAssessment` (Dunning-Kruger / trust / compass / homeostasis).
- `cognitive_shader.rs:157` `RungLevel(0..9)`; `ThinkingContext.rung` (plan.rs:22).

**Cognition pipeline (thinking-engine > P64 > shader-driver)**
- `thinking-engine/src/dto.rs`: Φ `StreamDto` (:40), Ψ `ResonanceDto` (:59), B `BusDto` (:120). (ResonanceDto rename in progress — `TD-RESONANCEDTO-DUP-1`.)
- `p64-bridge/src/lib.rs` convergence (CausalEdge64/palette).
- `cognitive-shader-driver/src/driver.rs:581` `CognitiveShaderDriver::dispatch`; `engine_bridge.rs:29` wires thinking-engine ↔ ShaderBus; deps planner (lab) + p64-bridge + (opt) thinking-engine.

**The loop (jolly-cori-clnf9 ONLY — unmerged)**
- `cognitive-shader-driver/src/mailbox_soa.rs:349/397` `impl MailboxSoaView + MailboxSoaOwner for MailboxSoA<N>` + driving test `:648` (`463d71bd`, +149 LOC). On main/this branch `MailboxSoA` has **no owner impl** → loop can't run in-tree.

**Surreal + callcenter write**
- Contract correctly declares surreal=project-read-only, callcenter=commit (kanban.rs:1-21).
- `lance-graph-callcenter/src/lance_membrane.rs:315` `commit_event` (sole writer, version tick) → `audit_sink/lance_sink.rs:292` real Lance `InsertBuilder…Append`.
- `surreal_container` ~all stub, BLOCKED on lance-7 fork (`TD-SURREALDB-KVLANCE-LANCE7`).

**Canonical node + addressing (contract; #489/#490 + #495-mine)**
- `canonical_node.rs:35` `NodeGuid` = `classid(u32)|HEEL(u16)|HIP(u16)|TWIG(u16)|family(u24)|identity(u24)`; `local_key()` (:106) = trailing 6 B; zero-fallback ladder. `EdgeBlock` (:181) 12+4. `EdgeCodecFlavor` (:207, #495-mine). `ValueSchema`/`ValueTenant`/`VALUE_TENANTS` (#495-mine — unmerged, rides #496).
- `hhtl.rs` `NiblePath` radix walk (`child`, `is_ancestor_of`).

---

## §2 — THE 6 SEAMS (gap → tracking → FOLD `file:line`)

1. **Planner emits `KanbanMove`** — gap: `emitted_edges` always empty; planner imports none of `kanban`/`soa_view`. Tracking: **D-MBX-A6-P3** (`D-MBX-COMPLETION-MAP.md`). FOLD: `Outcome/reliability → KanbanMove{exec}` adapter in `compose.rs`/`plan_full` (lib.rs:207); change `emitted_edges` → `emitted_moves: Vec<KanbanMove>`; `use lance_graph_contract::kanban::{KanbanMove, ExecTarget}`.
2. **temporal.rs unconsumed + dep-wall** — gap: `deinterlace`/`QueryReference` 0 callers; planner ⊥ lance-core. FOLD: relocate `temporal.rs` into zero-dep `contract` (both sides import) OR a thin `lance-graph` `temporal_read.rs` that calls `VersionedGraph::at_version(T)` → `deinterlace`; add `ref_version`/`rung` to `PlanContext` (traits.rs:69).
3. **Loop only on jolly** — gap: `MailboxSoaOwner for MailboxSoA` is +149 LOC on `463d71bd`, unmerged. FOLD: cherry-pick `mailbox_soa.rs:349-460` + test `:648` (purely additive, traits already on main).
4. **Rung inert** — gap: `RungLevel::Surface` hardcoded everywhere (orchestration_impl.rs:151, api.rs:178, pipeline.rs:593, thinking/mod.rs:86); `rung_delta` imported, never called; Staunen/Wisdom orphaned. FOLD: drive `RungLevel` from `escalation::rung_delta` + sustained `GateDecision::BLOCK`; bind `WisdomMarker` → `ShaderDispatch.rung`.
5. **Think-delegation (thinking-engine>P64>shader-driver, NOT Ladybug)** — gap: planner doesn't call shader-driver; driver→planner is lab-only (`planner_bridge.rs`); p64-convergence stubbed (`cache/convergence.rs:22`). FOLD: a bridge (third crate or shader-driver-owned) implementing the route `ThinkingContext → ShaderDispatch(carrying rung) → CognitiveShaderDriver::dispatch → ShaderBus.emitted_edges → PlanResult`. **Route via the cognition chain, never `StepDomain::Ladybug`.**
6. **Write mis-framing (doc)** — gap: `plan.rs:42-44` "the vart/surreal seam persists". FOLD (zero-code): → "callcenter (`commit_event` → `LanceAuditSink`) calcifies; surreal projects read-only."

---

## §3 — ADDRESSING: `identity / ScopedReference / (hhtl-guid):path:documentid`

Resolves left→right, mirroring `NodeGuid`:
```
(hhtl-guid)                 : path                     : documentid
classid|HEEL|HIP|TWIG       : VALUE_TENANTS offset      : family|identity (local_key)
= routing prefix (radix)    : intra-row value-slab path : the leaf
↔ TiKV Region-prefix / Pinpoint collection            ↔ Pinpoint document-id
```
- **identity** = `NodeGuid` itself (structured address, not a handle). Pinpoint's alias→Knowledge-Graph-MID collapse IS `family|identity` (one stable id per real entity; surface forms resolve to it).
- **ScopedReference** (the genuinely NEW piece; "ticket" — but **NOT** named `ticket`: collides with `grammar::ticket::FailureTicket`) = `(NiblePath scope, QueryReference as-of)` = a TiKV-TSO snapshot-handle scoped to a key-range = "this subtree, as-of Lance version T". `QueryReference::at(ref_version,rung)` is already the as-of half.
- **Bardioc's** "which export-restriction at data-window T" = `deinterlace(rows, QueryReference::at(T,rung), deps)` → rows Contemporary-at-T.

**ADOPT** — TiKV: prefix-routes-to-placement (`NiblePath::is_ancestor_of` = range-containment); snapshot-as-handle (never implicit "latest"); coprocessor pushdown ↔ `DependsClosure`/`deinterlace` filter-at-source; a **batched monotonic ticket-oracle = TSO over Lance versions** (but keep our decentralized HLC `Option<u64>`, no central oracle). Pinpoint: entity-MID = `identity`; cross-doc entities = `EdgeBlock`/`MaterializedEdges` (entities are EDGES between leaves, not a sidecar index); content-addressed `documentid` (free dedup); as-of-T first-class from day one (their post-2023-08-01 date-epoch trap = our `I-LEGACY-API` violation to avoid).
**DON'T** — Pinpoint filename-as-id / no-dedup / no-versioning; TiKV central TSO bottleneck.

---

## §4 — THE COGNITIVE CYCLE (8 steps → primitives → Rubicon phases)

| Step | Primitive (`file:line`) | Rubicon phase |
|---|---|---|
| fanout | `escalation::CollapseHint::Fanout` + `fanout_width` (:45/:59) | Planning |
| consolidate | `escalation::InnerCouncil::deliberate` (:144) | Planning |
| induction | `nars::InferenceType::Induction` | Planning→Σ |
| synthesize insights | `escalation::EpiphanyDetector::observe` (:241) + `InferenceType::Synthesis` | CognitiveWork |
| think | `CognitiveShaderDriver::dispatch` (driver.rs:581; thinking-engine>P64) | CognitiveWork ← seam #5 |
| deduction | `nars::InferenceType::Deduction` | CognitiveWork |
| meta awareness | `mul::MulAssessment` (DK/trust/compass) | Evaluation |
| abduction | `nars::InferenceType::Abduction` | Evaluation→{Commit\|Plan\|Prune} |

Spiral: abduction → `KanbanColumn::Plan` → re-deliberate → next fanout (Peirce abductive-inductive-deductive loop, gated by EpiphanyDetector synthesis + MUL anti-Mount-Stupid). **Add = a `CognitiveCycle` sequencer** (method on the integrated Planner) that drives the 8 steps through the kanban phases, setting `ThinkingContext.inference_type` per step. Everything it calls EXISTS; the sequencer is the only new code — and it is the consumer that closes seams #1 (emit per phase), #2 (as-of read per step), #4 (rung drive), #5 (think delegation).

---

## §5 — PROBES (measure-first; falsifiable, declared before running)

- **P-DEDUP-ASOF**: ingest the same doc at versions T₁,T₂ → assert one `identity` and as-of-T₁ excludes the T₂ copy (collapses Pinpoint's no-dedup + no-versioning into one invariant).
- **P-TICKET-SNAPSHOT**: one `ScopedReference` per session → two reads at the same ticket see byte-identical as-of-T snapshots across split basins (TiKV snapshot-isolation, mapped).
- **P-SCOPE-CLASSIFY**: `ScopedReference` admits a row iff `scope.is_ancestor_of(row.niblepath) && classify(...)==Contemporary` (needs the `NiblePath`↔`classid|HEEL|HIP|TWIG` byte↔nibble bijection — currently unwritten, `hhtl.rs:48`).
- **P-RUNG-VARIES** (exists, `style_strategy.rs:264`): reliability varies by style → a rung/style gate is non-cosmetic.
- **P-CYCLE-SPIRAL**: abduction → Plan → fanout actually changes the next cycle's candidate set (else the spiral is decorative).

---

## §6 — OPEN QUESTIONS / DEFERRALS

- OQ-11.6 surreal external trigger (`Notification→KanbanMove` `LanceVersionScheduler`) — fork-blocked (`TD-SURREALDB-KVLANCE-LANCE7`).
- OQ-11.7 planner DTO cutover scope (feature-gated vs clean break).
- The jolly→main merge of the loop (`463d71bd`).
- `temporal.rs` relocation (planner→contract) vs a lance-core `temporal_read` (the dep-wall decision).
- ResonanceDto rename (`TD-RESONANCEDTO-DUP-1`).
- `StepDomain::Ladybug` → mark deprecated (forget Ladybug).

---

## §7 — REFERENCE INDEX (the file:line grounding agents MUST target)

- Planner: `lance-graph-planner/src/{lib.rs:99/123/171, strategy/style_strategy.rs:138/168, temporal.rs:107/147/301, orchestration_impl.rs:151, traits.rs:69, ir/mod.rs:92}`
- Contract: `lance-graph-contract/src/{plan.rs:16/42/144, orchestration.rs:37/56/390, kanban.rs:32/112/136, jit.rs:48, escalation.rs:45/144/241/289/312, scheduler.rs:46, soa_view.rs:112, canonical_node.rs:35/106/181/207, hhtl.rs, cognitive_shader.rs:157}`
- Cognition: `thinking-engine/src/dto.rs:40/59/120`, `p64-bridge/src/lib.rs`, `cognitive-shader-driver/src/{driver.rs:581, engine_bridge.rs:29, mailbox_soa.rs:349(jolly)}`
- Write boundary: `lance-graph-callcenter/src/lance_membrane.rs:315`, `audit_sink/lance_sink.rs:292`
- Lance versioning: `lance-graph/src/graph/versioned.rs:419`
- External patterns: Google Pinpoint (entity-MID, collection-ACL, no-dedup/no-version traps); TiKV (key→region→store via PD, MVCC/TSO snapshot reads, coprocessor pushdown).
