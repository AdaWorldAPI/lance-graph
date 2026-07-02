# V3 Substrate Integration Plan — waves W0–W6

> **Status:** ACTIVE (W0 ships with this PR; W1–W6 sequenced below)
> **Date:** 2026-07-02 · **Authority:** operator rulings of 2026-07-02
> (board entries E-MAILBOX-KANBAN-NO-COLLAPSEGATE, E-COMPILED-THINKING-TEMPLATES,
> E-DTO-LADDER-OWNERSHIP-SPLIT, E-TWO-RESONANCES-SPLIT, E-V3-MARKER-IS-A-MONITOR,
> E-CLASSID-* arc) under standing full authorization.
> **Index entry:** `.claude/board/INTEGRATION_PLANS.md` (prepended same commit).
> **Deliverable dashboard:** STATUS_BOARD rows `D-V3-*` + the existing
> D-MBX-A6 / D-PERT-1 / D-CC-* / D-VCW-* / D-CCF-4 rows this plan adopts.

The organizing principle is unchanged from v3-convergence-wiring: **wire,
don't invent** — every wave below wires machinery that already exists
(envelope, kanban types, DSL triple, graph-flow, classid helpers) into the
ruled model. Anything that looks like a new layer is a defect in this plan.

---

## Wave map (dependency order)

```
W0 ratify/document ──► W1 envelope+ownership ──► W2 kanban executors ──► W5 consumers
                              │                        │
                              └──► W4 DTO ladder       └──► W3 templates ──► (catalogue ⤳ W6/P4)
                                                                 W6 monitor & retirement (P4 = operator checkpoint)
```

Rule of thumb: **W1 is the keystone** — the cast pairing
(`cast(on_behalf = envelope.mailbox_owner(), payload = BusDto)`) is what
every later wave consumes. W4 can run any time (rename + doc). W6's scanner
can be built early; its retirements only fire on proof.

---

## W0 — Ratify & document (THIS PR)

| D-id | Deliverable | State |
|---|---|---|
| D-V3-W0a | `.claude/v3/` tree (README, this plan, COMPONENT-MAP, ENTROPY-MILESTONES, MODULE-TABLE, soa_layout/*) | this commit |
| D-V3-W0b | V3 awareness layer: knowledge docs, 4 agent cards (`v3-*`), `/v3` skill, `/v3-audit` command, CLAUDE.md + BOOT.md entrypoints | this commit |

Gate: board hygiene same-commit (INTEGRATION_PLANS prepend, STATUS_BOARD
rows, AGENT_LOG, handover pointer).

## W1 — Envelope & ownership (the keystone)

| D-id | Deliverable | Notes |
|---|---|---|
| D-V3-W1a | `SoaEnvelope::mailbox_owner()` stamp | **SHIPPED** (this branch) — default 0 = bootstrap, zero-fallback ladder |
| D-V3-W1b | **Batch writer**: `cast(on_behalf, payload)` pairing + AHEAD kanban update fired at cast (never at write-ack) | new module, planner-adjacent; the ONLY sanctioned online write shape |
| D-V3-W1c | **Delegation cache**: cast id vs envelope stamp; hit = proceed, miss = resolve once + cache | lives in the batch writer; consumers never re-implement |
| D-V3-W1d | MailboxId minting path (non-zero owners; `debug_assert` uniqueness in default basin) | per CANON zero-fallback ladder |
| D-V3-W1e | Probe: ahead-update ordering test (kanban board reflects intent BEFORE Lance ack) + delegation-miss test | probe-first; W2 consumes |

Gates: `v3-envelope-auditor` LAYOUT-CLEAN; `v3-mailbox-warden` OWNED on the
writer's own paths; no serialization on the hot path (zero-copy invariant).

## W2 — Kanban executors (two arms + structural owner)

| D-id | Deliverable | Notes |
|---|---|---|
| D-MBX-A6 | `Outcome → KanbanMove` adapter emit in `lance-graph-planner strategy/style_strategy.rs` (arm #1) | adopts existing STATUS_BOARD row; converged/cycle_count from BusDto feed it |
| D-V3-W2a | **Per-mailbox kanban board as TENANT** (type + lane; sibling of per-row `KanbanTenant`) | envelope-auditor gate: field-isolation matrix |
| D-V3-W2b | Supervisor wiring: `kanban_actor.rs` applies moves via `MailboxSoaOwner::advance_phase` (sole mutator); ractor stays spawn-only | structural-owner proof unchanged |
| D-V3-W2c | symbiont arm (SurrealDB-on-kv-lance): kanban updates as KV transactions | **CORRECTED 2026-07-02:** coordinates RESOLVED 2026-06-16 — the remaining block is the deliberate cold-build gate (`BlockedColdBuild`, surreal_container Cargo.toml dep commented out to keep `cargo check` fast); arm #2 = one dependency-uncomment + ~10 min cold build away; POC = `symbiont/kanban_loop.rs`; read glove `SurrealMailboxView` compiles today |
| D-V3-W2d | 550 ms budget: load-balancing hooks into planner `elevation/` (extend, don't shadow) + budget instrumentation | 64k–256k SoA prioritization |
| D-V3-W2e | **Sub-µs dispatch speed probe**: rs-graph-llm (graph-flow) vs SurrealQL-on-kv-lance, same kanban/thinking hot path, measured | operator 2026-07-02: "the planner is too slow for sub-microsecond"; winner owns the hot-path `ExecTarget` (Native/Jit/SurrealQl/Elixir already coded, kanban.rs); planner = slow/plan path either way; truth-architect reviews the numbers |

Gate: W1b/W1c shipped (moves fire off casts). Standing rule: updates
reprioritize, never gate — a missing update must not deadlock a cycle.

## W3 — Compiled templates

| D-id | Deliverable | Notes |
|---|---|---|
| D-V3-W3a | `StepMask` in contract (sibling of FieldMask; selection, not control flow) | zero-dep |
| D-V3-W3b | ElixirTemplate → graph-flow `GraphBuilder` adapter; Session threads the MailboxId (ownership inheritance) | rs-graph-llm seam; Iron-Rule-5-style contract pull |
| D-V3-W3c | Rig oracle node: FailureTicket → oracle run → `template-equivalence` gate → `cognitive-compiler` compile-down into catalogue | D-VCW-7 lineage; deterministic-first |
| D-V3-W3d | Catalogue keyed INTERNALLY (template-id); classid custom-half keying deferred to W6/P4 | styles-as-lenses F2 lands here post-P4 |

Gate: replay equivalence green on every template change (template-smith rule).

## W4 — DTO ladder (parallel, any time)

| D-id | Deliverable | Notes |
|---|---|---|
| D-PERT-1 | Rename `dto.rs::ResonanceDto` → `PerturbationDto` (~9 files thinking-engine + cognitive-shader-driver engine_bridge), deprecated alias | adopts existing row; "cascade" key-tier vocabulary NOT renamed (canon) |
| D-V3-W4a | BusDto/cast pairing call sites in cognitive-shader-driver (consumes W1b) | BusDto never grows ownership fields — warden check 2 |
| D-V3-W4b | L4 learning loop doc-to-code check: converged residue → owner-stamped tenant lane → template reads row next cycle | probe over an end-to-end cycle |

## W5 — Consumer adoption (write-on-behalf fleet-wide)

| D-id | Deliverable | Notes |
|---|---|---|
| D-V3-W5a | q2: CI re-bakes (`osint-bake`, `fma`) + `body.soa` re-release; drop `FMA_V3_CLASSID_LEGACY` from BodyV3.tsx | handover continuation §1 |
| D-V3-W5b | q2 cpic contract pull with mereology (kinds → cascade positions under `0x0E01_1000`) — dissolves interim `0x0E01_000N` + `ISS-Q2-CPIC-MIRROR` | handover F3 |
| D-V3-W5c | Bake pipelines annotated bootstrap-owner; NEW online consumer writes route through the batch writer | per write-on-behalf.md interim rules |
| D-V3-W5d | Probes D-VCW-3 (P7 render, bitmask→askama) + D-VCW-5 (cascade3 nibble falsifier) — q2 gate already WAIVED | validate V3 keys end-to-end |
| D-V3-W5e | Stragglers: ladybug-rs (pre-V3/rustynum-era) contract pulls only | never bridges |
| D-V3-W5f | **smb-office-rs ORPHAN-WRITE migration**: `LanceConnector::upsert` (smb-bridge/lance.rs:176-201, live caller smb-woa/customer.rs:189) — the ONE online consumer write; stamp + batch-writer routing when W1 lands | consumer-map §2; until then explicitly flagged, never silently grandfathered |
| D-V3-W5g | OGAR `emit.rs` post-flip fix: 3× `facet_classid() as u16` doc-label → `concept_of(...)` (emit_rust/csharp/python) + regen check | 1 line × 3; propagates into every generated SDK until fixed |
| D-V3-W5h | MedCare-rs `medcare-soa` writer **born stamped** (design gate before first merge) | prevention half of M23 |
| D-V3-W5i | q2 dual-bake collapse: retire/re-bake stale pre-flip `data/osint-v3` codebook into the canon-high osint-bake (M22) | latent I-LEGACY shape |

## W6 — Monitor & retirement (proof-gated)

| D-id | Deliverable | Notes |
|---|---|---|
| D-V3-W6a | **Adoption/corpus scanner**: ONE two-metric range-count tool (canon-high adoption % + old-form row count) over Lance datasets | metrics are range counts over the DECODED classid u32 (or an order-preserving BE key) — never raw LE key-byte prefixes; old-form = ALL THREE legacy shapes incl. `0xAAAA_DDCC` render-prefix-high (codex #629 sharpenings; routing.md §1/§5) |
| D-CCF-4 | `0x1000` marker retirement (P4) — trigger DEFINED: adoption reads 100% | **operator checkpoint** |
| D-V3-W6b | Legacy alias retirement (`CLASSID_*_LEGACY`, compat reader narrowing) | corpus proof = zero old-form rows; never before |
| D-V3-W6c | Custom half opens: 64k ClassView render catalogue + template catalogue dispatch (completes W3d/F2) | post-P4 only |

---

## Standing gates (every wave)

1. **Probe-first** — mechanism lands after its failing probe (workspace rule).
2. **Warden + auditor green** on any write-path / layout diff (`/v3-audit` is the cheap pre-check).
3. **Board hygiene same-commit** (STATUS_BOARD row transitions, EPIPHANIES for findings).
4. **Model economy** — Sonnet 5 grindwork, Fable/Opus decisions (operator ruling 2026-07-02). Every Sonnet brief carries the §1 preamble of `knowledge/sonnet-worker-guardrails.md` verbatim; §5 escalation triggers are STOP+report, never worker-resolved.
5. **Wire, don't invent** — a new struct/trait/layer proposal must first fail the "existing machinery" search (COMPONENT-MAP is that search, precomputed).

## What this plan supersedes / adopts

- Adopts (does not duplicate): D-MBX-A6, D-PERT-1, D-CC-RUNTIME/EQUIV/COMPILER rows, D-VCW-3/5/7, D-CCF-4.
- Extends: `v3-convergence-wiring-v1` (its seam list is W1–W3's ancestry) and `soa-value-tenant-migration-v2` (Phase-2 tenant shaping proceeds under W2a's tenant discipline).
- Supersedes in prose only: any remaining CollapseGate-as-singleton framing in older docs (primer §6 table governs).

---

## Addendum 2026-07-02 — Fable-5 preflight epiphanies (pre-W1, operator-requested)

Ten-point pass over every layer before phase start (full text: EPIPHANIES
E-V3-PREFLIGHT-1 + session transcript). Plan deltas adopted:

1. **W1b/W1c collapse (WAL-shaped writer):** the cast IS the kanban move —
   the AHEAD update is the write-intent record, Lance ack confirms it. The
   board becomes the write-ahead log; crash recovery = replay unacked moves.
   New entropy row M24. Gate: kill-after-cast-before-ack replay test.
2. **M7 ruling recommendation:** re-scope `SoaEnvelope` as the spec/descriptor
   certification surface (`verify_layout` + field-isolation matrix are the
   value; trait polymorphism has zero production impls). Doc-line ruling,
   unblocks W1 without refactor.
3. **W6a scanner runs FIRST (baseline inversion):** build the two-metric
   range-count tool at W1 start; record t0 old-form counts in the M1 row.
   "Adoption 100%" is only falsifiable against a measured denominator.
4. **W3 oracle ratchet metric:** oracle-hit rate per cycle vs catalogue size
   must trend DOWN; flat = templates not generalizing = deterministic-first
   silently dead. One counter, plotted per replay run.
5. **W2 internal reorder:** W2e (dispatch probe) → W2d (budget) → W2a/b/c.
   Budget constants come from measured µs; probe measures batch 1/64/4096
   (sub-µs matters at batch 1); loser owns the slow/plan path (two-speed).
6. **Ractor batching by construction:** actor boundary takes `Vec<KanbanMove>`
   per message, never singles — helper-scope compliance enforced by API shape.
7. **D-PERT-1 rides the first W1 PR** (7 files, mechanical; waiting grows
   the blast radius).
8. **M21 pull-forward:** zero-dep `canon-node-bytes` extraction lands in W1
   (same LE work); byte-parity gate vs contract NodeGuid.
9. **Gate-run rule:** every wave PR's final commit runs `/v3-audit` + the
   touched M-row greps, results pasted into AGENT_LOG (self-updating ledger).
10. **Supervisor stays thin forever:** the product is the compile-time
    ownership attestation; restart policy is the only runtime duty. No
    routing/registry/pub-sub creep (the trap arrives dressed as convenience).

Nothing here invents machinery — every delta is a collapse or reorder of
what the plan already carries (the V3-shape test, passed).

### Addendum-2 2026-07-02 — operator direction: rs-graph-llm + rig parallel evaluation

Operator: "test rs-graph-llm + rig in parallel for speed and ergonomics —
under lance-graph it might need some kanban integration to become the
replayable langgraph handler." Folded as:

- **W3b sharpened → KanbanSessionStorage:** graph-flow's `SessionStorage`
  gets a mailbox-kanban-board-backed impl — task transitions persist as
  KanbanMoves through the W1b writer, so **replay = rebuild the Session
  from the board**. This unifies M24 (board = WAL) with orchestration
  persistence: one persistence surface, and every langgraph execution is
  replayable by construction (M25). If graph-flow's save() is
  whole-session-overwrite, a thin delta layer maps Session diffs to moves
  (bench worker reports the trait shape).
- **W2e gains a third measurement:** graph-flow per-step dispatch overhead
  at batch 1/64/4096 (bench worker in rs-graph-llm, release mode) sits
  beside the SurrealQL-on-kv-lance arm. rig = the oracle-node client (W3c)
  — ergonomics assessed for Task-wrapping, never on the hot path.

### Addendum-3 2026-07-02 — rig backend note (operator) + W1e landed red

- Operator note folded: **rs-graph-llm vendors rig**, and rig's storage/LLM
  backends span lancedb, SurrealDB-on-kv-lance (lance-graph-symbiont), and
  Claude/OpenAI/Grok(xAI)/Gemini APIs. Consequence for W3c: the oracle node
  is backend-plural behind ONE rig client surface — symbiont (arm #2) can be
  BOTH the kanban KV arm and rig's vector store, which would collapse the
  oracle's storage to the substrate itself (no second store). Verify when
  the bench worker reports rig's provider/store traits.
- W1e status: probes landed RED (probe-first honored); KanbanMove/KanbanColumn
  already shipped in contract kanban.rs — the skeleton consumes them, zero mints.

### Addendum-4 2026-07-02 — planner-SoA reality audit (operator question) → 3 wiring deltas

Verdict: **type-level reality, wiring-level dormant.** KanbanMove already
unifies mailbox/witness/libet/exec; supervisor + lance-graph core + symbiont
are WIRED; the planner crate is the gap (zero MailboxSoaView references,
zero classid awareness, style_strategy pass-through, mul-gate false friend).
Deltas:
1. **W2b += integration probe**: spawn KanbanActor over the REAL
   MailboxSoA<N> (today: TestBoard-only — proven mechanics, unproven
   integration).
2. **W2 += classid-awareness wiring**: planner reads class_id through
   MailboxSoaView (the getter already exists — wire, don't invent);
   supervisor likewise if move-routing needs read modes.
3. **M15 upgraded to BLOCKING-before-W2**: planner mul/gate.rs
   GateDecision{Proceed,Sandbox,Compass} vs contract
   mul::GateDecision{Flow,Hold,Block} — rename the planner-local one
   BEFORE any planner->kanban emission lands, or the wrong gate routes
   into advance_on_gate silently.
Full inventory with file:line cites: E-V3-PLANNER-SOA-AUDIT-1 + AGENT_LOG.

### Addendum-5 2026-07-02 — bench results land the W2e read + M25 design

- **Numbers (release, steady-state batch 4096):** graph-flow ~408-471
  ns/step (ContinueAndExecute) / ~512-538 ns/step (stepwise; delta = the
  storage round-trip). Two-speed CONFIRMED with data: graph-flow = the
  replayable orchestration layer; sub-us hot dispatch = ExecTarget.
- **M25 design finalized:** SessionStorage is overwrite-semantics (all 3
  impls upsert the whole Session) -> KanbanSessionStorage = Session
  snapshot upsert + KanbanMove cast through the W1b writer; the append-only
  move log ALREADY EXISTS as rs-graph-llm/graph-flow-kanban's
  KanbanPlanEnvelope (consumes contract kanban types + GateDecision) —
  wire it to the W1b writer, invent nothing. Replay = snapshot + move log.
- **rig = oracle-frequency only** (2 full history clones + tool-def fetch
  per call): W3c yes, per-transition no. Fork is upstream-faithful.
- **Build wall (ops):** rs-graph-llm/rig workspace-root cargo 403s on the
  AdaWorldAPI/burn git submodule via surreal-lance OPTIONAL deps (lock
  resolution pulls manifests even when features are off). Sandboxed builds
  use isolated path-dep crates until a lockfile/vendor lands. Bench file
  committed to rs-graph-llm @ claude/v3-substrate-migration-review-o0yoxv.

### Correction 2026-07-02 (codex #630 P2) — M7 premise was WRONG: SoaEnvelope HAS a production impl

`NodeRowPacket<'a>` (canonical_node.rs:1275) implements `SoaEnvelope` in
production — the Lance-facing zero-copy LE byte view over `&[NodeRow]`
(NODE_ROW_COLUMNS / NODE_ROW_STRIDE / as_le_bytes with the repr(C,64)
SAFETY argument). "Zero production impls" (preflight delta 2, inherited
from the fleet's M7 row) is retracted. **Revised M7 ruling:** the two
surfaces are COMPLEMENTARY, not duplicates — `SoaEnvelope` is the
storage-boundary surface (certification + the canonical Lance byte path,
with NodeRowPacket as its live impl); `MailboxSoaView/Owner` is the
runtime read/mutate surface. W1 implementers MUST route storage bytes
through the NodeRowPacket envelope path and preserve/test its
owner/byte-layout behavior — the trait is NOT descriptor-only. M7's gate
re-shapes accordingly (roles documented both ways + envelope path tested,
rather than "≥1 impl or re-scope").

### Addendum-6 2026-07-02 — operator ruling: zero-copy sink + mutual masking (W1b design closed)

Operator: "it was always zerocopy and the write masks the thinking and
vice versa so that the batch writer sinks the deltas asap." Pinned:

1. **The cast carries a DESCRIPTOR, never bytes:** (mailbox, dirty
   row-range, cycle) + intent moves. Deltas stay in the SoA backing
   store; the sink reads them through `NodeRowPacket::as_le_bytes` at
   flush time (the M7-corrected storage-boundary path). Zero-copy from
   creation to Lance tombstone INCLUDING through the writer — the
   payload-generic `P` in the skeleton is a descriptor type, never
   owned bytes.
2. **Mutual masking via the phase machine, not buffers:** while cycle
   N's dirty rows sink, the owner refuses phase re-entry on those rows
   (Rubicon arc = the mutation freeze); thinking proceeds on all other
   rows/mailboxes. Compute masks I/O and I/O masks compute — the kanban
   board IS the scheduler that makes the overlap race-free. No
   double-buffering, no copies.
3. **Eager drain:** the sink fires ASAP on cast (background), never
   batch-until-full — the unacked window IS the replay surface; keep it
   minimal. W2d's 550 ms budget may treat write latency as masked so
   long as sink throughput >= delta production rate (instrument both).

Gate added to W1b: a mutation-freeze test — a row in sink phase rejects
advance_phase until ack (lands with the real-owner wiring, W2b probe
extends it).

### Addendum-7 2026-07-02 — operator correction of Addendum-6: NO refusal — "melden macht frei"

The mutation-freeze point in Addendum-6 was over-design and contradicted
the standing rule ("updates reprioritize, never gate"). Corrected:

1. **Casting IS reporting, and reporting frees the thinker.** The writer
   NEVER refuses a cast because earlier casts on the same row/mailbox are
   unacked. Stacked writes (>=3) are stacked WAL entries: distinct ids,
   full ordered move history, independent acks.
2. **Coalescing is natural, not engineered:** the sink reads the LIVE
   backing store at flush, so one physical flush of a row satisfies every
   earlier stacked intent for it — last-state-wins is correct because the
   replay target is the row's latest state, while the move log preserves
   the full ordering history.
3. M24 gate updated: the mutation-freeze test is REPLACED by the
   stacked-casts test (probe 4, `probe_stacked_casts_never_refused` —
   landed ignored with the other three).

### Addendum-8 2026-07-02 — temporal.rs is the READ side of the WAL ruling (operator pointer)

Operator: "check temporal.rs for a deeper understanding." Verified against
`crates/lance-graph-planner/src/temporal.rs` (490 lines, read in full):

1. **No-refusal is PROVABLE, not just permitted.** No reader ever reads raw
   current state — `deinterlace()` classifies every row against the reader's
   `QueryReference` (Contemporary/Anachronistic/Spoiler/Unknowable, admitted
   per `EpistemicMode` rung policy Strict 0-4 / Aware 5-8 / Retro 9+). A
   Strict thinker at `V_ref` cannot see frames past its horizon, so writers
   stack freely: coherence is a READ-side property. "The write masks the
   thinking and vice versa" — the mask IS the epistemic classification.
2. **Replay = a read at a pinned reference.** `QueryReference::at(v, rung)`
   + `deinterlace` IS crash-replay (M24), session-replay (M25), and
   time-travel — ONE mechanism. KanbanSessionStorage's replay path
   implements nothing new: it pins a Strict reader in the past over the
   board + rows.
3. **W1b ack carries the assigned LanceVersion**: `ack(cast, LanceVersion)`
   is the CastId <-> LanceVersion join wiring the WAL into the temporal
   classifier. Unacked casts = rows on NO reader's timeline yet (exactly
   why unacked() is the replay surface). Probe file updates with the
   signature in the W1b commit.
4. **The ractor actor's payload is its V_ref** (temporal.rs frame table:
   "ractor (awareness) — each actor's own V_ref reading-horizon") — the
   helper carries the awareness horizon; one more reason it never needs
   to be on the hot path.
5. DATA-causal axis (DependsClosure/NoDeps) composes with the board:
   dispatchable = time-admitted AND data-ready — the standing rule
   "updates reprioritize, never gate" holds because a data-blocked row is
   dropped from the PROJECTION, not refused at the writer.

### Addendum-9 2026-07-02 — W3c oracle measured LIVE + the self-hosted test bench

- **Live numbers** (rig xai, grok-4-0709 oracle through FlowRunner):
  framework overhead **1-2 ms** vs **8.4-8.7 s** LLM round trip. W3c's
  budget IS the LLM call; orchestration is free. Full data:
  E-V3-ORACLE-LIVE-1.
- **Test-bench delta (operator):** wire rig's OpenAI-compatible provider
  at the planner's serve.rs (`/v1/chat/completions`, JITSON behind it)
  as the LOCAL oracle for CI — zero external spend, no key, graceful
  retry wrapping for a typical langchain-style bench. Lands with W3b's
  KanbanSessionStorage tests.
- **References:** rs-graph-llm's stated design (LangGraph-inspired,
  Rig-integrated, interruptible-by-design, step-by-step default) matches
  the W3b role exactly; the thread's queue-vs-stateful debate is resolved
  in V3 by the kanban board being both (M24 WAL + state). GraphRAG-rs
  noted as RAG prior art (native LanceDB/Arrow, Leiden, LightRAG, cAST).

### Addendum-10 2026-07-02 — GraphRAG-rs verdicts + cross-session convergence intake

- **GraphRAG-rs inventory closed** (full doc:
  `.claude/knowledge/graphrag-rs-inventory.md`; board:
  E-V3-GRAPHRAG-INV-1): every component REUSE-AS-REFERENCE or IGNORE;
  the Addendum-9 "native LanceDB/Arrow" note is hereby CORRECTED —
  their LanceDBStore is a 100% `NotImplemented` stub (Qdrant is the
  real default), "hierarchical" Leiden is single-level, cAST is
  example-only. Steal-worthy for W2a/W3: the TypedBuilder type-state
  pattern (`.build()` only exists on the fully-configured phantom
  instantiation — a mailbox/tenant/kanban builder that cannot compile
  until owner + layout version are set) and the sync/async trait-pair
  + adapter-module bridge. Anti-exhibit: `InferenceEngine` (config-only
  struct, graph as call parameter) is the inverse of Thinking-is-a-struct;
  `AsyncGraphRAG` is the organ-owning analog.
- **Cross-session intake executed** (three sibling wishlists + two
  synthesis passes; dispositions:
  `.claude/handovers/2026-07-02-cross-session-wishlist-intake.md`;
  board: E-V3-XSESSION-INTAKE-1): C6 `RouteBucketTyped` MERGED into
  `contract::codegen_spine` (nexgen retires its vendor diff);
  `contract::emission_scan` MINTED (classid_scan sibling); OGAR
  quick-win batch on the OGAR branch (flip fuse, COUNT_FUSE two-sided,
  Genetics 0x0E + OCR 0x08XX + 0x1000-reservation as ONE batched
  allocation-table arc per the convergence ruling "serialize the
  mints", prose sweep, truncation-doctrine DISCOVERY-MAP mirror).
- **The scan family is now a named contract pattern** (ratified by
  3-session convergence): `Form` enum + `classify_*` + fold-to-counts,
  zero-dep, in the contract — instances: `classid_scan` (V3 adoption),
  `emission_scan` (typed-DDL adoption). The third counter (soc-verdict
  counts, predicate-coverage, parity-fixture coverage) MIRRORS this
  shape; inventing a bespoke grep instead is the drift signal.
- **Interchange guard (L3/E5 fusion):** ONE Arrow schema family for
  extraction interchange (triples batch s/p/o/f/c + facets batch),
  provenance header carries `minter@sha`; ndjson STAYS as the
  committed-golden/diffable layer (Arrow batches are opaque to PR
  review); the ingestion side targets the W1b cast/descriptor shape —
  no second envelope. W5 design item.
- **Probe-corpus archival convention (unclaimed-gap fix):** every
  measurement that will be quoted (F17 triage ratio, count_adoption
  vs a real bake, medcare re-runs) archives input + generation recipe
  + hash WITH the run, or it recreates "last measured, can't
  re-verify".
- **RULING-NEEDED queue for the operator** (recorded in the intake
  handover §R): R-1 hi-u16 naming — evidence now strong for the
  concept reading (`0x0102` shared by OP `0x0102_0001` + Redmine
  `0x0102_0007` is inexpressible under an appid reading); deliver the
  ruling as ACCESSOR RENAMES not prose (4 prose-rot instances this
  arc). R-2 EdgeBlock canon → one const set + size asserts both repos
  pin. R-3 per-entry board files + per-repo coordination dir as ONE
  merged council proposal (X1 yielded). R-4 OGAR probe-ledger Wave A
  green-light.
- **CORRECTIONS (same day, operator rulings — canonical text in the
  intake handover appendices + E-V3-XSESSION-INTAKE-1-RULINGS):**
  R-1 was a PHANTOM (canon already exists: `domain:appid:classview`,
  le-contract.md §prefix + primer §5 — "concept" names the whole hi
  u16; the "app" homonym across the halves caused the thread; both
  ledgers now carry the line, OGAR D-CLASSID-HI-U16-SPELLING). R-2
  closed empirically (edges pull separately via the NODE_ROW_COLUMNS
  strided view; the 512-byte row — tested against kv-lance AND the
  batch writer, with time series wired into surrealdb from Lance
  versioning — is NOT touched; residual = a read-side edges-only
  test). L3 schema design KILLED ("defining arrow schemas is bullshit
  and hallucination — we already have a working SoA schema"):
  extraction interchange lands as SoA rows through the W1b cast path;
  survivors are minter@sha provenance + ndjson-as-golden only.

### Addendum-11 2026-07-02 — operator ruling: per-consumer classid ownership + tripwires; the five open workstreams

**Ruling:** "everyone should be responsible for his own OGAR classids
and trip the wire if not in sync." Distributed ownership + fuses IS the
coordination mechanism — each consumer owns its classid allocations and
wires a sync-tripping test (the flip fuse + two-sided COUNT_FUSE shipped
this arc are the pattern instances; the serialized-mint-batch was the
transitional vehicle, ownership+fuses is the steady state).

**The open workstreams after the fuses (operator-enumerated; "huge
tasks but manageable if done properly"):**

1. **Thinking ↔ substrate + V3 migration** — this plan's W-waves
   (W1 shipped #631; W2 kanban/board-as-tenant next).
2. **Orchestration rs-graph-llm** — W3b KanbanSessionStorage +
   graph-flow as the replayable handler (M25), oracle-frequency rig.
3. **OGAR as transpiler + compile-time API** — askama + ClassView +
   FieldMask bitmask (the Redmine ERB pattern), taking over from
   SurrealQL AST DLL parsing (SURREAL-AST-AS-ADAPTER completes into
   the compile-time render path).
4. **The complete cross-repo inventory** (the ~300-crate census asset:
   MODULE-TABLE + COMPONENT-MAP + consumer maps) — maintain and
   complete it; it is the substrate for every wave's preflight.
5. **Per-consumer accommodation** — ruff + OGAR adaptations,
   lance-graph UnifiedBridge ↔ OGAR, AST contract evolution (the W5
   consumer wave, now with the fuse doctrine as its safety rail).
