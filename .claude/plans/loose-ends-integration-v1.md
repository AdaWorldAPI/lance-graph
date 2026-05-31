# Loose-Ends Integration Plan (v1)

> **Scope:** wire the named loose ends into one running loop — from the D-MBX kanban
> contract down to a witness committed to cold store and back up through a SurrealDB
> LIVE subscription that flips the Rubicon kanban. Full-breadth spec (8 workstreams),
> no implementation in this doc. Every claim is grounded in a real symbol; every phase
> states its falsifiable check and the byte-freezes it touches.
>
> **Companion invariants:** `.claude/specs/cognitive-risc-core.md` (substrate), `…-classes.md`
> (class layer), `faiss-homology-cam-pq.md` (CAM), `wikidata-hhtl-load.md` (load). This plan
> may not violate an invariant in those; where it brushes one, it says so.

---

## 0. Reconciled ground-truth (2026-05-31)

Two recon passes + four merged PRs (#436/#438 aerial, #437/#439 kanban contract, #444 world-spine,
#445 lance-7) + direct diff reads. **The dev branch `claude/cognitive-risc-core-9PMW8` is ~stale
(pre-#437); Phase 0 rebases it onto current main before any code lands.**

| Piece | Symbol / path | State |
|---|---|---|
| Kanban lifecycle | `contract::kanban::{KanbanColumn, KanbanMove, ExecTarget, RubiconTransitionError}`; `next_phases/can_transition_to/is_absorbing/try_advance_phase` | **IMPL on main** (#437/#439); **zero consumers** outside contract |
| SoA view split | `contract::soa_view::{MailboxSoaView, MailboxSoaOwner}` (read/owner borrow split) | **IMPL on main** (#437); no `impl … for MailboxSoA` yet |
| Collapse gate | `contract::collapse_gate::{CollapseGateEmission, MergeMode(Xor/Bundle/**Superposition**/AlphaFrontToBack), GateDecision(Flow/Block/Hold)}` | **IMPL**; `Superposition` merge mode already exists → W4 reuses it |
| Witness → cold | `lance-graph::graph::witness_tombstone::{HotWitness, calcify, Tombstone::{from_hot,persist,verify}}` (**D-ATOM-5**) | **SCAFFOLD** — 4× `todo!()`, blocked on Lance write API |
| Witness table | `contract::witness_table::WitnessTable<N>` | **IMPL** |
| surreal_container | `surreal_container::{SurrealStore(PhantomData placeholder), write/read/fold/catalog/cache/epoch/ring/compaction}` | **ALL STUB**; `open()→Blocked`; **BLOCKED(C)** = surrealdb-fork dep (OQ-11.6) |
| SurrealDB LIVE | upstream `core::sql::statements::live::LiveStatement`, `core::doc::lives` notification routing | **IMPL upstream** — but fires at the **Document** layer, not the kv-lance flush |
| kv-lance backend | `surrealdb/.claude/lance-backend/*.patch.rs` (not applied); `core::kvs::lance::flusher` (`Notify` = LSM wake only) | **PATCH-ONLY**; **no LIVE-notify hook on flush** |
| Supervisor | `lance-graph-supervisor::consumer_msg::ConsumerEnvelope` (Dispatch/Ingest/Health/Qualia/Styles/…); **no Kanban arm** | **IMPL**; Kanban variant deferred (#437) |
| markov_soa | `lance-graph::graph::arigraph::markov_soa` (#444, vocab-agnostic, injected `Fn(u16,u16)->u8`) | **IMPL but `STATUS: provisional`** — never compiled (heavy deps) |
| deepnsm | `markov_bundle::{MarkovBundler, Kernel(Uniform/MexicanHat/Gaussian)}`, `trajectory::Trajectory`, `quantum_mode::{PhaseTag, HolographicMode(SinglePhase impl / PerRole stub)}`, `disambiguator_glue` | **IMPL** (quantum PerRole stub) |
| aerial | `lance-graph-arm-discovery::{AerialProposer, Proposer, CandidateRule, CodebookDistance/MatrixDistance/TopKDistance, FeedProjector, OntologyProjector, DolceCategory, translator::arm_to_nars}` | **IMPL on main** (#436/#438) |
| jc / splat | `jc::{jirak, pflug, ewa_sandwich(_3d), sigma_codebook_probe, probe_p1_gamma_phase(PASS)}`; `splat_louvain_modularity`; `examples/ontology_locality_probe` (PASS 98.6% on real ontologies, **not** Wikidata) | **IMPL** (Wikidata-scale unproven) |
| thinking styles | `contract::grammar::thinking_styles::{ThinkingStyle(36), GrammarStyleConfig(prior), GrammarStyleAwareness(NARS-revised), effective_config(composed), MarkovPolicy(TEKAMOLO), SpoCausalPolicy(Pearl 2³)}` | **IMPL, 3-layer, wired to planner** |
| planner strategies | `lance-graph-planner` 16 strategies incl. `JitCompile(#14)`→`cam_pq::jitson_kernel` (Cranelift), `WorkflowDAG(#15)`; `contract::plan::StrategySelector(Explicit/Resonance/Auto)`; `contract::jit::{JitCompiler, KernelHandle}` | **IMPL**; **Elixir backend = DOC-ONLY** |
| HHTL / CAM | `contract::high_heel`(LensProfile), `bgz-tensor::cascade`(HEEL→HIP→TWIG→LEAF 4/8/12/32b), `contract::cam` (CAM-PQ) | **IMPL**; `NiblePath`/delta-I-P-B-frame/radix = **DOC-ONLY** (`knowledge/delta-card-addressing-integration-map.md`, `agnostic-lazy-world-spine.md`) |
| BindSpace singleton | `cognitive-shader-driver::bindspace::FingerprintColumns` + `BindSpace::zeros(4096)`; threaded in `serve.rs`, `driver.rs`, `lance-graph-ontology/src/lib.rs` | **IMPL singleton**; per-mailbox columns (`MailboxSoA<N>.{edges,qualia,meta,entity_type}`) **shipped** (D-MBX-A1) but singleton **not yet deleted** (D-MBX-3/5) |

**The two hard blockers (everything reactive sits behind these):**
- **B1 — Lance write API for D-ATOM-5.** `calcify/from_hot/persist/verify` need the lance-7 append/checkout API. Unblocked by #445 (lance-7 in tree) — but the `todo!()`s still reference lance-4/6 patterns; they must be rewritten against lance-7.
- **B2 — surreal_container BLOCKED(C) / OQ-11.6.** The fork dep (git URL, branch, `kv-lance` feature flag) is unresolved, so `SurrealStore` is a `PhantomData` placeholder and the whole federation crate is inert. **No reactive loop exists until this is resolved.**

---

## 1. The dependency DAG

```
W0 hygiene ─┬─► W1 substrate spine (D-ATOM-5, MailboxSoaOwner impl) ──┬─► W2 reactive loop ─┐
            │        ▲ blocked by B1 (lance-7 write API)               │   ▲ blocked by B2   │
            │        │                                                 │   (surreal fork dep) │
            └─► W7 bindspace decommission (D-MBX-3/5) ─────────────────┘                      │
                                                                                              ▼
        W3 exec backends (ExecTarget dispatch) ◄──┐                                  W4 head2head
        W5 Hebbian prefetch (EW64 Markov) ────────┼── all consume the live loop ──►  superposition
        W6 language→SPO landing (D-LWS) ──────────┘                                  (+ arbiter)
                                                                                              │
                                                                              W8 (optional) ◄─┘
                                                                          bindspace→Markov wave/VSA
```

**Critical path:** B1 → W1 → B2 → W2. Until a witness can be calcified (W1) *and* a surreal LIVE
subscription can observe it (W2), W3/W4/W5/W6 have no live loop to attach to and can only be built
against fakes. **Recommended: drive the W0→W1→W2 vertical to one real thought first** (even though
the user chose full-breadth spec — the spec is full breadth, the *build order* should still be
vertical-first, see §4).

---

## 2. Guardrail spine (binds every workstream)

1. **Nothing semantic in the register file** (core #1). New columns carry bytes; meaning resolves above.
2. **Witness copied, not pointed, hot→cold** (core #5). `calcify` must *snapshot* — a cold fact may never point into an arena about to epoch-reset.
3. **Two-clock decoupling** (core #7). The LIVE subscription (W2) and prefetch (W5) *serve* the decoupling; they must never make the shader block on cold latency.
4. **Bounded mailboxes** (core #8). W4's extra head2head mailboxes are bounded; shed by ⟨f,c⟩ under pressure.
5. **CAM addresses; similarity only proposes** (classes-doc). VSA/ANN/Markov-prefetch/aerial **never** address or commit. W5's prefetch hint is **discardable** — a wrong prefetch is a cache miss, never a belief edit. W6's aerial output is a *candidate*, gated by D-ARM-7 (Jirak) before it becomes skeleton.
6. **Name every byte-freeze touched.** `KanbanMove` is at **15/16 B** (saturated — next field forces a layout decision); `discovery_origin` proposer-id is 3 bits/8 (already 6 named); `DolceCategory` 4 facets at nibbles 0x0–0x3; `class_id` (N1). Each phase below lists what it freezes.
7. **Don't freeze the column ISA until ≥2 domains run through** (classes N4): chess **and** Wikidata, not Odoo alone.

---

## 3. Workstreams

### W0 — Floor / hygiene  *(prereq, no new architecture)*
- **Goal:** a clean, current branch and a true floor before building on it.
- **Deliverables:**
  - **D-LE-0a** Rebase `claude/cognitive-risc-core-9PMW8` onto current main (#437/#439/#444/#445).
  - **D-LE-0b** Confirm `main` compiles under lance-7 with `arigraph::markov_soa` present (it merged `STATUS: provisional`, never compiled). If red, fix the 6→7 break; clear the provisional marker once it compiles+tests.
  - **D-LE-0c** Update the stale pin line in `cognitive-risc-core.md`: `lance 7.0.0 / lancedb 0.30 / datafusion 53` (+ note object_store 0.13 = the surrealdb-fork alignment point, F2).
- **Falsifiable check:** `cargo test` green on the rebased branch incl. the (de-provisionalised) `markov_soa`.
- **Freezes:** none.

### W1 — Substrate spine: a witness reaches cold store
- **Goal:** kill the D-ATOM-5 `todo!()`s and give `MailboxSoA` an owner impl, so a `Commit`
  transition actually materialises a witness to Lance.
- **Current state:** `witness_tombstone.rs` = 4 `todo!()`; `MailboxSoaOwner` has no impl; `WitnessTable<N>` done.
- **Deliverables:**
  - **D-ATOM-5** (existing) implement `calcify(HotWitness)->SpoRecord`, `Tombstone::{from_hot, persist, verify}` against the **lance-7** append/checkout API. `calcify` **snapshots** (guardrail #2).
  - **D-MBX-4** (existing) wire mailbox death → SPO-G quad + Lance versioned tombstone (append-only).
  - **D-LE-1** `impl MailboxSoaView + MailboxSoaOwner for MailboxSoA<N>` in `cognitive-shader-driver`; `advance_phase`/`try_advance_phase` mutate the lifecycle column only (R1: SoA never serialised through the view).
- **Falsifiable check:** the boot-doc smallest-slice — *write a SoA thought, `Commit`, read the materialised witness back after a simulated schema bump.* Witness survives, resolves, points at a snapshot (not an epoch-reset arena).
- **Freezes:** the tombstone wire layout (generation counter + witness back-pointer; minimal forwarding record, never payload — core #6); the SPO-G cold quad shape.

### W2 — Reactive loop: surreal LIVE flips the Rubicon kanban  *(the headline)*
- **Goal:** the kanban in SurrealDB **subscribes** to the witness commit and advances CI-style;
  the `Planning→CognitiveWork` edge is the Rubicon flip.
- **Design resolution (important):** SurrealDB LIVE fires at the **Document layer** (`doc::lives`), *not*
  on the kv-lance flush, and the flusher exposes **no** subscriber hook. Per F2's default lean
  (**federate, don't read Lance storage directly**), the witness commit is written as a **surreal
  document** (backed by kv-lance underneath); the kanban driver runs a **`LIVE SELECT`** over that
  table. "The Lance update *is* the witness pointer" holds — but the *subscription* rides the
  existing, implemented LIVE mechanism, not a new flush-notification. This avoids inventing a kv-layer
  callback and keeps us off the fragile read-Lance-directly path.
- **Current state:** `surreal_container` all-stub, `SurrealStore=PhantomData`, **BLOCKED(C)/OQ-11.6**; `ConsumerEnvelope` has no Kanban arm.
- **Deliverables:**
  - **D-LE-2a** Resolve **OQ-11.6**: pin the surrealdb fork dep (URL/branch/`kv-lance` feature), replace the `PhantomData` placeholder, make `surreal_container::write/read/fold/epoch` real. *(unblocks everything below)*
  - **D-LE-2b** `Commit` path writes the calcified witness as a surreal document (W1 output → surreal record).
  - **D-LE-2c** Kanban driver = a `LIVE SELECT` subscriber that maps a witness-row notification → `try_advance_phase` on the owning mailbox (CI-style: each commit ticks the board).
  - **D-LE-2d** Add `ConsumerEnvelope::Kanban(KanbanMove)` arm + handler in `lance-graph-supervisor` (the deferred #437 seam).
  - **D-MBX-9** (existing) "Rubicon kanban view in surrealkv-on-lance" is the integration target this realises.
- **Falsifiable check:** drive one mailbox `Planning→CognitiveWork→Evaluation→Commit`; assert a `LIVE SELECT` subscriber receives exactly one notification per legal transition and the board reflects it; an illegal transition yields `RubiconTransitionError` and **no** notification.
- **Freezes:** the surreal witness-record schema (table name, columns the LIVE query selects); the `ConsumerEnvelope` wire (additive arm). Guardrail #3 — the LIVE callback must not block the shader.

### W3 — Execution backends: ExecTarget dispatch + thinking-style layers
- **Goal:** make `ExecTarget {Native, Jit, SurrealQl, Elixir}` a real dispatch, tied to the 3-layer
  thinking styles and the planner's 16 strategies.
- **Current state:** `ExecTarget` is a field on `KanbanMove` but dispatched nowhere; `JitCompile(#14)`
  → `jitson_kernel` (Cranelift) **exists**; thinking_styles 3-layer **exists**; **Elixir backend is doc-only**.
- **Deliverables:**
  - **D-LE-3a** `ExecTarget` dispatch table: `Native`→planner engine, `Jit`→`cam_pq::jitson_kernel` (Cranelift), `SurrealQl`→lower to SurrealQL (reuses W2's surreal seam), `Elixir`→declarative template (new; smallest = a guarded-rewrite template interpreter, NOT a runtime ontology binder — stay compile-time per F1).
  - **D-LE-3b** Bridge `thinking_styles::effective_config` (prior⊗awareness⊗composed) → `StrategySelector` → the `ExecTarget` chosen on the emitted `KanbanMove`. "3+ layers" = the existing prior/awareness/effective composition; this wires its output onto the move.
- **Falsifiable check:** the same candidate, planned under two thinking styles, emits `KanbanMove`s with different `ExecTarget`s and both execute to the same SPO result (backend-agnostic correctness).
- **Freezes:** **`ExecTarget` is now load-bearing** — its variant set enters the `KanbanMove` byte (the 16th byte). Adding a 5th backend is a conscious widening. `ExecTarget` enum is append-only.

### W4 — Head2head superposition: two real mailboxes + arbiter  *(user-chosen)*
- **Goal:** spawn competing mailboxes as a superposition; resolve by Rubicon/EFE. The Go analogy:
  one mailbox plays **infight** (local tactical), one plays **Raumgewinn** (territory/space) — distinct
  proposers, full independent plans, arbiter picks at commit.
- **The meeting point (load-bearing):** the two heads are **two `MailboxSoaView`s of SoA that meet
  inside the `cognitive-shader-driver`.** This is the mechanism, not a metaphor: `MailboxSoaView` is
  the zero-copy read-borrow trait (#437); the shader driver holds **two** of them (head-A = infight,
  head-B = Raumgewinn — or intrinsic CE64 view vs temporal EW64 view of the *same* frozen SoA shape)
  and runs **one batch op that reads both grids at once** (the SIMD/record-batch execution mode,
  classes-doc §SIMD). The superposition is *where the two views converge under the shader*, not a
  copy: each view stays single-writer in its own mailbox (core #3), the driver only **reads** both and
  **writes neither** (R1). The collapse happens here and only here.
- **Current state:** `MergeMode::Superposition` and `GateDecision` already exist in `collapse_gate`;
  `cognitive-shader-driver` already batches one op over a grid; what's missing is the **two-view
  rendezvous** + the arbiter. No multi-mailbox spawn/arbitration driver.
- **Deliverables:**
  - **D-LE-4a** A `Head2Head` supervisor spawns the *N* (=2 first) bounded mailboxes (one per
    strategy/`discovery_origin`), each running W1–W3 independently as single-writer SoAs.
  - **D-LE-4b** **The rendezvous in the shader driver:** the driver takes `&dyn MailboxSoaView`×2,
    batch-superposes their columns via `MergeMode::Superposition` (energy/edges read-overlaid, never
    cross-written), producing the head2head comparison grid. Witnesses across the two stay **pointers**
    (R4) — nothing is copied until a winner commits (W1).
  - **D-LE-4c** Arbiter = cold/commit-tier EFE (core #10, real Friston at small N): score the two
    terminal `Evaluation` states by EFE×goal-alignment; gate via `GateDecision::{Flow,Block,Hold}`; the
    winner's `Commit` calcifies (W1), the loser `Prune`s (tombstone-only).
  - **D-LE-4d** Tie to `E-DUPLICATION-IS-INTRINSIC-VS-TEMPORAL`: when the two views are intrinsic
    (CE64 semantic, "what resonates now") vs temporal (EW64 episodic, "how the belief arose"), the
    rendezvous **names** the separation — it does not dedup it. SoA1:SoA2 superposition is well-defined
    precisely because the SoA byte-shape is frozen (same shape twice).
- **Falsifiable check:** the chess bring-up — infight-view and Raumgewinn-view of the same board SoA
  meet in the driver; each emits GM-flavoured legal candidates; the EFE arbiter's pick agrees with
  Stockfish's eval sign more often than either view alone. **And** assert the driver wrote to neither
  source mailbox (R1 held through the rendezvous).
- **Freezes:** none new (reuses `MailboxSoaView`/`MergeMode`/`GateDecision`); but the two-view batch op
  must stay within the 64k–512k shock-absorber (core #7) and shed low-c first (core #8). The rendezvous
  is **read-only on both inputs** — that invariant is the whole safety of the superposition.

### W5 — Hebbian prefetch / cognitively-relived edge hydration
- **Goal:** EW64-Markov transition statistics drive **predictive prefetch** of the next witness into
  the hot arena — *fire-together-wire-together*, BNN-grade. The **32k SPO-W chain** is the worldline
  ("string"): a 1-D ordered witness chain is the fundamental object the prefetcher walks.
- **Current state:** EW64 = queued design (not a code symbol yet); `markov_soa` (the wave) + deepnsm
  `MarkovBundler`/`Trajectory` (the bundling) exist; no prefetch driver; `D-MBX-A3` (witness_arc column) queued.
- **Deliverables:**
  - **D-MBX-A3** (existing) add `witness_arc: [u32; W]` per-row column (the belief-state arc handle) — the EW64 materialisation point. *(gated on OQ-11.2)*
  - **D-LE-5a** Maintain Markov transition counts over the witness_arc (P(next witness | current)) = the Hebbian weight = the prefetch prior. Co-occurrence in one observation strengthens the EW64 episodic edge (AriGraph Ee).
  - **D-LE-5b** Prefetch driver: on entering a witness, warm the top-k successors from cold→hot. **GUARDRAIL (load-bearing):** the prefetch hint is **discardable and never addresses or commits** — exactly the CAM-vs-ANN firewall (classes-doc). A prefetch miss is a latency cost, never a truth edit. Keep "predicted-next" out of the commit path; only the W1 commit gate writes a witness.
  - **D-LE-5c** "Relived edge hydration": lazily hydrate an edge from cold only when the chain walk reaches it (two-clock; the prefetcher hides the cold fetch).
- **Falsifiable check:** with prefetch on, hot-arena miss-rate on a replayed 32k SPO-W chain drops materially vs off; **and** ⟨f,c⟩/commit outcomes are *byte-identical* with prefetch on vs off (proves the hint never touched belief).
- **Freezes:** the `witness_arc` column width `W` (per-class, append-only, N3 discipline); the EW64 layout once it materialises.

### W6 — Language-parse mode → SPO landing (the discovery frontend)
- **Goal:** when in language mode, the kanban parses sentences via DeepNSM grammar heuristics
  (hybrid dark-horse `markov_soa` → aerial), and the **blasgraph splat fan-out is repurposed as the
  HHTL bucket router** over an SPO ontology; **Wikidata is the example dataset**; new SPO lands as a
  **CAM delta-frame (I/P/B keyframe) on a NiblePath radix trie**.
- **Current state:** deepnsm grammar/Markov **exist**; aerial `OntologyProjector`/`TopKDistance` **exist**
  (#438); jc splat/`ontology_locality_probe` **PASS on real ontologies** (not Wikidata); HHTL cascade
  **exists**; `NiblePath`/delta-I/P/B-frame/radix-trie/hydration-manager = **DOC-ONLY** (D-LWS).
- **Deliverables:**
  - **D-LE-6a** Language-mode gate on the kanban: in parse mode, route sentences through deepnsm
    (`MarkovBundler`/`Trajectory`/`disambiguator_glue`) → aerial `AerialProposer` candidates
    (the "hybrid dark-horse" Markov leg), `discovery_origin = ArmDiscovered`.
  - **D-LE-6b** Repurpose `jc::splat_louvain_modularity` / blasgraph top-k as the HHTL bucket router:
    feed aerial's `TopKDistance` from the splat neighbour lists (the #438 seam), bucket by 16ⁿ nibble.
  - **D-LE-6c** **D-LWS hydration manager** (from `wikidata-lazy-spine-hydration-v1.md`): land new SPO as a
    **CAM delta-frame** (deck = expectation / card = surprise; I-frame = full keyframe, P/B = deltas)
    on a **NiblePath radix trie**; Wikidata P279 skeleton as the first real corpus.
  - **D-LE-6d** Gate every landed edge on **D-ARM-7** (Jirak significance floor, `jc::jirak`) before it
    becomes live skeleton — aerial proposes, the floor ratifies (core #9: proposers dumb, arbiter smart).
- **Falsifiable check:** the **open** Wikidata-scale probe — run the locality probe on a real **Wikidata
  P279 subtree** (10⁸ regime, not the 10³ ontologies that already PASS). If fan-out re-balances onto 16ⁿ
  → FINDING; if it forces adaptive fan-out → the HHTL base must flex *before* the WAL freezes it.
- **Freezes:** the delta-frame header (I/P/B tag + NiblePath); the facet bit-budget (wikidata-load: closed→bitmask, open→ref; append-only); `DolceCategory` axis template. **N4 applies hardest here** — this is the second domain that must run through the column ISA before it freezes.

### W7 — BindSpace decommission
- **Goal:** delete the `BindSpace::zeros(4096)` singleton; everything per-mailbox.
- **Current state:** per-mailbox columns shipped (D-MBX-A1); singleton still threaded in `serve.rs`,
  `driver.rs`, `lance-graph-ontology/src/lib.rs`.
- **Deliverables:**
  - **D-MBX-2/3** (existing) move `engine_bridge` per-row surface onto mailbox rows; ShaderDriver holds the per-mailbox set; kill the singleton. *(gated on OQ-1 content-ref shape, OQ-2 temporal/expert fold)*
  - **D-MBX-5** (existing) delete `BindSpace` + the `Vsa16kF32` cycle plane.
  - **D-LE-7** grep-sweep every remaining `BindSpace`/`zeros(4096)` consumer to zero (the migration surface list in `EPIPHANIES.md`).
- **Falsifiable check:** `grep -r "BindSpace::zeros\|Arc<BindSpace>"` returns empty; all tests green per-mailbox.
- **Freezes:** removes a frozen singleton (net un-freeze); confirm no WAL artifact referenced the singleton layout.

### W8 — (optional / "naughty") BindSpace → Markov particle→wave fusion in VSA
- **Goal:** instead of deleting the retired bindspace, **re-materialise** its content as a DeepNSM-Markov
  **wave** (`markov_soa`) superposed with the EW64 **particle** — particle→wave fusion in VSA16k.
- **Current state:** `quantum_mode::HolographicMode::PerRole` is a **stub**; `markov_soa` provisional;
  VSA16kF32 exists as a *fuzzy proposer only* (never truth/identity — `I-VSA-IDENTITIES`).
- **Deliverables:**
  - **D-LE-8a** Express the old singleton's fingerprint planes as a `markov_soa` wave (the bundled
    expectation) + the per-row EW64 witness as the particle (the exact pointer); the shader superposes
    them (Hebbian), collapsing at Rubicon.
  - **D-LE-8b** Keep VSA **strictly fuzzy** — this fusion *primes*, it never addresses or commits
    (same firewall as W5). It is a discovery aid on the retired data, not a resurrection of the singleton
    as a source of truth.
- **Falsifiable check:** the fused wave reproduces the singleton's top-k priming neighbours within
  tolerance, while CAM identity/commit remain byte-identical (proves it stayed a proposer).
- **Freezes:** none — explicitly a discovery-layer experiment; must not enter addressing or the WAL.

---

## 4. Sequencing (PR ladder; vertical-first within full-breadth spec)

1. **W0** (rebase + main-green + pin) — one PR, mechanical.
2. **W1** (D-ATOM-5 + MailboxSoaOwner impl) — *the floor*. Gated on B1 (lance-7 write API, now in tree).
3. **W7** (bindspace decommission) — can run parallel to W1 (independent surface), unblocks a clean per-mailbox base.
4. **W2** (reactive loop) — gated on **B2/OQ-11.6** (surreal fork dep). This is the headline; do it as soon as B2 clears. First green here = the loop the user was shocked to find open is closed.
5. **W3 / W5 / W6** — fan out once the live loop exists (each consumes it).
6. **W4** (head2head) — after W3 (needs the strategy/ExecTarget seam) and the live loop.
7. **W8** — optional, last, clearly fenced as discovery-only.

**Recommendation despite "full-breadth spec":** spec is breadth (this doc); *build* is the W0→W1→W2
vertical to **one real thought through all five layers** before fanning out — that's the falsification
the whole architecture is waiting on (core "smallest possible first slice").

---

## 5. Open questions to ratify (gates, not optional)

| OQ | Subject | Gates |
|---|---|---|
| **OQ-11.6** | surreal fork dep (URL/branch/`kv-lance` feature) | **B2** → all of W2 |
| **OQ-11.2** | witness-arc chain handling (AriGraph episodic Markov) | D-MBX-A3 → W5 |
| **OQ-11.7** | planner DTO / KanbanMove cutover shape | D-MBX-A6 / W3 emit |
| **OQ-1 / OQ-2** | content-ref shape / temporal-expert fold | D-MBX-2/3 → W7 |
| **OQ-11.5** | SoA version byte strategy | D-MBX-10 (layout-root freeze) → all column adds |
| **new OQ-LE-1** | Elixir backend = template interpreter (compile-time, F1-NO) vs deferred | W3 `ExecTarget::Elixir` |
| **new OQ-LE-2** | W4 arbiter EFE objective (pure EFE vs EFE×goal-align proxy) | W4 |

## 6. The one end-to-end that proves it (closes the shock)

`Planning → CognitiveWork(Rubicon flip) → Evaluation → Commit` on one mailbox, where `Commit`
calcifies a witness (W1) → written as a surreal document (W2) → a `LIVE SELECT` subscriber flips the
kanban board (W2) → next cycle the prefetcher warms the successor from the EW64 chain (W5). Run it on
**chess** (W4 infight vs Raumgewinn, Stockfish ground-truth) *and* on a **Wikidata P279 subtree** (W6,
the 10⁸ locality probe). If both come out clean, the column ISA may freeze (N4 satisfied). If either
forces a layer to lie, we found it on one thought — before the WAL hardened.

---
*v1. Grounded against current main (#437/#439/#444/#445) + two recon passes, 2026-05-31. The dev
branch is stale; W0 rebases it. Change an invariant only with a stated reason.*
