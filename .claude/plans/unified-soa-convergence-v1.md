# unified-soa-convergence-v1 — THE single little-endian SoA, end-to-end across the workspace

> **ERRATA (2026-06-13, post-#490):** §1 "five layered rulings" remain authoritative (anchor for `E-SOA-IS-THE-ONLY`). §4.2 stack-pin table is stale — `lance` bumped to `=7.0.0` and `lancedb` to `=0.30.0` since this plan; the 2026-05-29 review addendum partially addressed but its own pins drifted too. The three-tier model in PR #477 (`docs/architecture/soa-three-tier-model.md`) ratifies "no emission, no inter-mailbox handoff" — supersedes any plan language implying a Baton carrier type. Full diff resolution: `soa-migration-diff-resolution-2026-06-13.md`.

> **Status:** PROPOSAL / integration plan. Design-spec only; **no code in this plan**.
> **Authored:** 2026-05-29 (session `017GFLBn`, branch `claude/splat3d-cpu-simd-renderer-MAOO0`).
> **Supersedes nothing; integrates / sequences:**
>   - `bindspace-singleton-to-mailbox-soa-v1.md` (§11.1–§11.6 — the layered rulings ratified in this session)
>   - `causaledge64-mailbox-rename-soa-v1.md` (the canonical 5-crate + 7-PR plan; this doc is its column-level + cross-component sequel)
>   - `cognitive-substrate-convergence-v1/v2/v3.md` (i4 mantissa, gapless baton, Σ10 Rubicon — already in flight / shipped)
>   - the `.claude/surreal/` POC (folded in per `RECONCILIATION_with_canonical_plan.md`)
>
> **Anchored to (FINDING-grade):** `E-BATON-1`, `E-CE64-MB-4`, `E-LADDER-SERVES-MAILBOX`, `E-MAILBOX-IS-BINDSPACE`, `E-RUBICON-RACTOR`, `E-SOA-IS-THE-ONLY`, `I-VSA-IDENTITIES`, `I-LEGACY-API-FEATURE-GATED`, `I-SUBSTRATE-MARKOV`, `I-NOISE-FLOOR-JIRAK`, `E-CONTRACT-NO-SERIALIZE(-2)`, `E-NORMALIZED-ENTITY-1`.
>
> **Owns the answer to:** *"all of [the nine half-baked components] have to consume the same SoA from A-Z; the SoA can be versioned so they stay readable after schema upgrade; for SurrealDB the versioning aligns with lance 6.0.1 / lancedb 0.29 / datafusion 53 to have one transparent container view; the kanban/ractor needs to be aligned with a new overhaul of lance-graph-planner DTO surface."*

---

## 0. Executive summary (one screen)

There is **ONE** little-endian SoA in this workspace — the per-mailbox `MailboxSoA<N>` byte layout (`E-SOA-IS-THE-ONLY`). Every component that touches per-mailbox state consumes *that* SoA, **never a translated DTO**. Three operations are allowed on it: cognitive-shader thinking (hot path), cold-path read/write to LanceDB (leading storage), and AriGraph Markov-chain context building. *Any change in any mailbox SoA = the only hot-path activity.*

Today **nine components** are *half-baked* — partially aligned to this contract or doing their own thing:

1. AriGraph · 2. Markov-grammar `Vsa16kF32` substrate · 3. `BindSpace` · 4. `crates/lance-graph` cold containers · 5. `lance-graph-planner` · 6. `cognitive-shader-driver` · 7. `lance-graph-callcenter` · 8. `lance-graph-ontology` (read-only, AS IS) · 9. thinking-styles/atoms.

This plan sequences each of them onto the same SoA, adds a **version byte at the layout root** (read-old-bytes after schema upgrade, governed by `I-LEGACY-API-FEATURE-GATED`), aligns the workspace to the **Lance 6.0.1 / LanceDB 0.29 / DataFusion 53** stack (one patch bump pending), wires the **4-phase Rubicon kanban (Planning → Cognitive work → Evaluation → Commit·Plan·Prune)** at **Libet −550 ms** into `SigmaTierRouter` + ractor outer-swarm + a `surrealkv`-on-lance VIEW, and overhauls the `lance-graph-planner` DTO surface to operate ON the SoA + emit kanban transitions.

Total new deliverables: **D-MBX-A2 / A3 / A4 / A5 / A6 / 7 / 8 / 9 / 10 / 11 / 12**. One deferred TD (`TD-RESONANCEDTO-DUP-1`, folded into D-MBX-2). Eight open questions (`OQ-11.1` … `OQ-11.8`).

**No code in this PR. No cargo invoked (per session-stability constraint).**

---

## 1. Architectural foundations — the five layered rulings

### R1. One SoA, never transformed (the carrier doctrine)

The mailbox SoA byte layout is *singular and canonical*. It is **never re-encoded** at any boundary. Only three operations are allowed on it:

1. **Cognitive-shader thinking** (the hot path) — `apply_edges` / `emit` over the per-row columns inside `MailboxSoA`.
2. **Cold-path read/write to LanceDB** — same bytes; `persisted_row: Option<u32>` is a pointer to the same row laid down in Lance, not a serialized copy in another shape.
3. **AriGraph Markov-chain context building** — read-only consumer of the SoA columns for context windows over the episodic chain (the chain *is* the index space, see R4).

Equivalence: **any change in any mailbox SoA = the only hot-path activity**. Anything else (Arrow / JSON / DTO translation) is a re-encode boundary that violates this rule and must be collapsed or made out-of-scope. *Today's `crates/lance-graph` containers are cold-path-adjacent thinking — only accidentally aligned; D-MBX-7 makes that alignment intentional.*

The 1.4–4.2× SIMD acceleration payoff comes from `ndarray::simd_soa.rs` operating on the SoA columns directly when `lance-graph` containers ≡ `MailboxSoA` layout ≡ `simd_soa.rs` aligned. Today: nice-to-have. **When SurrealDB needs a transparent view: hard prerequisite.**

### R2. Mailbox = full BindSpace, reinvented as LE; witness = belief-state arc

The mailbox SoA must carry *everything BindSpace had* — but as **LE-contract types** (`CausalEdge64`, `QualiaI4_16D`, `MetaWord`, i4/u8/u16/u32/u64 columns), never as the mushy `Vsa16kF32` resonance carrier (deprecated as a carrier — `E-BATON-1`).

The **witness IS the per-row arc of `CausalEdge64` emissions** (`CollapseGateEmission` arc). That arc *implicitly documents NARS revision*: every emission stamps `confidence_u8` + `inference_mantissa`, so reading the arc IS the `(frequency, confidence)` evolution trace. **No separate revision log column.** D-MBX-A1 columns landed on `mailbox_soa.rs` between #418 and #433 merges; D-MBX-A2 closes remaining BindSpace-expressivity gaps; D-MBX-A3 adds the witness-arc handle column.

### R3. Libet −550 ms anchors the Rubicon kanban

The Σ10 Rubicon commit (`SigmaTierRouter`, D-CSV-10 shipped #388) acquires a wall-clock anchor: **t = −550 ms** before the act, matching Libet's measured readiness-potential lead time. The 4-phase kanban (Planning → Cognitive work → Evaluation of goalstate → Commit·Plan·Prune) is the action-phase board over `surrealkv`-on-lance (a VIEW over leading LanceDB). Ractor lifecycle transitions = kanban moves.

Libet "free won't" has two expressions: (a) pre-commit veto at t < −550 ms (ghost-tier preempt — card never leaves Planning), (b) post-evaluation Prune (the action happened but is rejected for calcification).

### R4. SPO-W witness is a *pointer* into the AriGraph episodic Markov chain

The witness is NEVER stored data; it is an **arc-handle pointer** into the belief-state arc array. The pointer can live equivalently in (a) the mailbox SoA (per-row `[u32; W]` arc handle), (b) the kanban row, or (c) the mailbox index.

**The AriGraph episodic Markov chain IS the index space.** "Witness in other mailboxes" means a pointer *into the chain* — the temporal sequence of mailbox states that constitutes episodic memory. Mailboxes are the chain's nodes; a witness is a back-pointer into that chain. No parallel "episodic memory" structure exists (CLAUDE.md "The Click": *"AriGraph, episodic memory, SPO, CAM-PQ are thinking tissue — not storage"*).

The SoA itself decides commit modality:
- **(a) Pointer to other mailboxes in the chain** — inter-mailbox baton handoff carrying the arc-handle.
- **(b) Cold-path fact** — LanceDB SPO-G calcification with the witness pointer linking back to the AriGraph chain node.

### R5. Counterfactual Staunen × Wisdom = plasticity spreaders

When a mailbox is in Planning (pre-Rubicon counterfactual phase), high Staunen × Wisdom *spreads* `plasticity_counter` increments beyond the focal row (Hebbian spread). Hot-path-only — the spread IS a mailbox SoA mutation (R1), never a side channel. Radius / decay TBD (`OQ-11.1`).

---

## 2. The nine half-baked consumers — current state → target state

The "one SoA, never transformed" rule binds **nine components** to the same SoA carrier; they may differ in *what they do with it* (read / mutate / project) but never in *what shape it is*. Each gets a sub-deliverable under `D-MBX-12`.

### 2.1 AriGraph

**Today:** parallel `TripletGraph` / `OxigraphAriGraph` types; "episodic" is informal (no explicit Markov-chain structure exposed).

**Target:** the AriGraph episodic Markov chain *is* the chain-of-mailboxes (R4). SPO-G quads carry arc-handle pointers into mailbox SoA rows (the witness substrate). Reading episodic memory = traversing the mailbox chain via arc handles. The SoA columns are AriGraph's read surface; SPO-G quads are not parallel data but pointers into the SoA.

**Deliverables:** `D-MBX-12.1` (AriGraph sub-PR).

### 2.2 Markov-grammar `Vsa16kF32` substrate

**Today:** the `Vsa16kF32` carrier is *deprecated* (`E-BATON-1`, CLAUDE.md Baton-scoping). Local intra-Think bundle computation remains.

**Target:** local-bundle compute reads SoA columns to produce ephemeral `Vsa16kF32` bundles when needed (e.g. resonance peaks); bundles never become cross-boundary state, never persist, never appear in cold storage. The Markov property (`I-SUBSTRATE-MARKOV`) is preserved at the local-compute level, untouched.

**Deliverables:** `D-MBX-12.2` (audit-only sub-PR: confirm no `Vsa16kF32` survives as cross-boundary state).

### 2.3 `BindSpace`

**Today:** the shared singleton `Arc<BindSpace>` at `crates/cognitive-shader-driver/src/driver.rs:56` and `bin/serve.rs:29`.

**Target:** **dissolved onto mailboxes** (`E-MAILBOX-IS-BINDSPACE`, plan §2.5). The mailbox SoA *is* the BindSpace surrogate. D-MBX-A1 already landed `edges`/`qualia`/`meta`/`entity_type`; remaining work is D-MBX-2 (collapse `engine_bridge` re-encode), D-MBX-3 (driver holds sea-star of mailboxes), D-MBX-5 (delete singleton + `Vsa16kF32` plane).

**Deliverables:** D-MBX-2, D-MBX-3, D-MBX-5 (already on STATUS_BOARD).

### 2.4 `crates/lance-graph` cold containers

**Today:** "cold-path-adjacent thinking" — only *accidentally* aligned to the SoA shape, not by design.

**Target:** **`lance-graph` container layout ≡ `MailboxSoA<N>` layout ≡ `ndarray::simd_soa.rs` aligned**. Unlocks the 1.4–4.2× SIMD payoff *and* the SurrealDB transparent view. The container *is* the SoA layout; persisting is `persisted_row` pointing at the same bytes.

**Deliverables:** `D-MBX-7` (the alignment), `D-MBX-12.4` (lance-graph sub-PR).

### 2.5 `crates/lance-graph-planner`

**Today:** DTO surface (`PlanResult`, `QueryFeatures`, `StrategySelector`, the 16 strategies, cache DTOs) predates the SoA contract; plan/MUL/elevation paths translate.

**Target:** **DTO surface overhauled** to express operations *on* the SoA + 4-phase kanban state-transitions. Strategy selection reads SoA columns; planner emits kanban moves; the MUL gate / σ-tier router stamps the Σ10 commit at −550 ms wall-clock.

**Deliverables:** `D-MBX-A6` (planner DTO overhaul), `D-MBX-12.5` (lance-graph-planner sub-PR).

### 2.6 `crates/cognitive-shader-driver`

**Today:** `MailboxSoA<N>` has D-MBX-A1 columns. `engine_bridge` (`bind_busdto` / `unbind_busdto` / `busdto_to_binary16k`) is the re-encode seam.

**Target:** the shader operates *on* the SoA only. `engine_bridge`'s re-encode seam collapses; `BusDto` becomes a thin read-projection over a SoA row; `StreamDto` survives only as an ingress adapter at the sensor membrane; `ResonanceDto.energy` is unified into `MailboxSoA.energy` (the two `ResonanceDto` defs in `thinking-engine` resolved per `TD-RESONANCEDTO-DUP-1`, deferred to D-MBX-2).

**Deliverables:** D-MBX-2 (engine_bridge collapse), D-MBX-A2 (BindSpace expressivity gaps), D-MBX-A3 (witness-arc handle column), D-MBX-A4 (Staunen×Wisdom spreader), `D-MBX-12.6`.

### 2.7 `crates/lance-graph-callcenter`

**Today:** Zone-2 persistence path; partial alignment with SoA (PR #414 + #416 work shipped axioms + style wiring, but the row layout itself is not yet THE SoA).

**Target:** Zone-2 cold reader/writer consumes the SoA bytes directly. Callcenter rows ARE the SoA rows; no Arrow re-encode for the thoughtspace (ontology RecordBatch path remains legitimate; see §2.8). `Reasoner` impls (D-ODOO-2 / D-ODOO-SAV-4) consume `ReasoningContext.evidence: &[EvidenceRef]` that points into SoA rows.

**Deliverables:** `D-MBX-12.7`.

### 2.8 `crates/lance-graph-ontology` — AS IS

**Today:** `LazyLock<NamespaceRegistry>` seed at `registry.rs:39`; `ontology_dictionary` Lance cache (`lance_cache.rs`, feature `lance-cache`, TTL-sourced, drop-and-rebuild). Its own header already states: *"BindSpace (FingerprintColumns / QualiaColumn / MetaColumn / EdgeColumn) is the live runtime SoA and is unrelated — it never lands here."*

**Target:** **stays AS IS.** The ontology is NOT part of the SoA; it is a lazylock-via-cache shared resource. Mailboxes consult it through the `entity_type: u16` column (1-based index into the registry). No work in this plan for ontology *internals*; only confirm the boundary doesn't leak (audit-only).

**Deliverables:** `D-MBX-12.8` (audit-only sub-PR: ensure no SoA bytes leak into ontology cache and vice versa).

### 2.9 Thinking styles / atoms

**Today:** `ThinkingStyle(36)` lives partly in `lance-graph-contract::thinking`, partly in shader, partly in planner; atoms (D-ATOM, recipe catalogue from #416) live in contract.

**Target:** encoded into the SoA's `meta` column (`MetaWord`'s `thinking(6)` bits) + p64-bridge `layer_mask` for palette dispatch. One canonical home per kind; consumers read from the SoA, not from a parallel registry.

**Deliverables:** `D-MBX-12.9`.

---

## 3. THE shared SoA layout — column-by-column

### 3.1 Layout root header (versioning)

The SoA is prefixed by a small layout-root header that carries the **version gate** (D-MBX-10). Conceptual shape (binary layout TBD per OQ-11.5):

```rust
#[repr(C)]
pub struct MailboxSoAHeader {
    pub magic:    [u8; 4],   // "MBX0" — identifies a MailboxSoA blob
    pub version:  u16,       // schema version (governed by I-LEGACY-API-FEATURE-GATED)
    pub flags:    u16,       // reserved (endianness assertion, feature bits)
    pub n_rows:   u32,       // const-generic N at runtime
    pub layout_checksum: u32, // sanity of the column-offset table below
    // followed by:
    //   - column-offset table (one entry per per-row column)
    //   - per-column row arrays
}
```

A v(M) reader **MUST refuse** to decode v(N>M) bytes without an explicit version handshake. Field-isolation matrix tests (`I-LEGACY-API-FEATURE-GATED` discipline, Sprint-11 5-instance catalogue) are mandatory on every column addition/widening.

**OQ-11.5:** width of `version` field (u8 vs u16 vs u32?) and whether per-column version stamps are needed in addition to the layout-root one.

### 3.2 Per-row columns — owned by `MailboxSoA<N>` (the hot path)

| Column | Type | Bytes/row | Source | Status |
|---|---|---|---|---|
| `energy` | `f32` | 4 | accumulator (`D-CSV-7`) | **Shipped** |
| `plasticity_counter` | `u8` | 1 | accumulator | **Shipped** |
| `last_emission_cycle` | `u32` | 4 | accumulator | **Shipped** |
| `edges` | `CausalEdge64` | 8 | migrated from BindSpace (D-MBX-A1) | **Shipped** |
| `qualia` | `QualiaI4_16D` | 8 | migrated from BindSpace (D-MBX-A1) | **Shipped** |
| `meta` | `MetaWord` (`u32`) | 4 | migrated from BindSpace (D-MBX-A1) | **Shipped** |
| `entity_type` | `u16` | 2 | migrated from BindSpace (D-MBX-A1) | **Shipped** |
| `content_ref` | `ContentId` (CAM-PQ code, 6 B) | 6 | NEW; resolves to shared codebook (D-MBX-A2) | **Queued** |
| `witness_arc` | `[u32; W]` arc-handle | 4·W | NEW; pointer into the belief-state arc array (D-MBX-A3) | **Queued** |
| `temporal` / `expert` | folded into `CausalEdge64` (v2 layout) | 0 | OQ-2 — resolve fold vs separate column | **OQ-open** |

**Per-row hot total:** ~30 B (bare migrated SoA columns) + ~6 KB Hamming identity planes (kept hot for resonance — see §3.3) = **~6 KB/thought** ⇒ **64k–256k thoughts hot ceiling at ~300–600 MB / ~1.2–2.4 GB**.

### 3.3 Identity / reference columns (per-mailbox, hot)

| Column | Type | Bytes/row | Role |
|---|---|---|---|
| `content` plane | `[u64; 256]` | 2 048 | Hamming identity fp (the "what" — topic) |
| `topic` plane | `[u64; 256]` | 2 048 | Hamming identity fp (topic plane) |
| `angle` plane | `[u64; 256]` | 2 048 | Hamming identity fp (angle plane) |

**These STAY hot per thought** (resolves OQ-1, per §11.2.7 capacity math). They are *identity* fingerprints (`I-VSA-IDENTITIES`), not content; bundling/comparing them is a hot-path op. The 64 KB `Vsa16kF32` `cycle` plane is **DROPPED** (deprecated carrier; ephemeral local-compute only when a step needs it).

### 3.4 Mailbox identity / control (constant per mailbox)

| Field | Type | Role |
|---|---|---|
| `mailbox_id` | `MailboxId` (u32) | Corpus root handle |
| `w_slot` | `u8` (6-bit) | OGIT domain corpus selector (≤ 64) |
| `current_cycle` | `u32` | Monotonic cycle stamp |
| `threshold` | `f32` | Emission threshold |

### 3.5 What stays SHARED (NOT in the SoA)

| Resource | Owner | Why out of SoA |
|---|---|---|
| `OntologyRegistry` | `Arc<OntologyRegistry>` + `LazyLock<NamespaceRegistry>` + `ontology_dictionary` Lance cache | Read-only, calcified cold knowledge. Lazy-locked, drop-and-rebuild on schema change. **AS IS** (§2.8). |
| CAM-PQ codebooks | shared cold codebook | Reference resolution target for `content_ref`; never bundled into rows. |
| AriGraph SPO-G cold quads | LanceDB SPO-G dataset | Calcified facts; quads carry pointers INTO the SoA arc-handles (R4), not data. |

---

## 4. SoA versioning + Lance 6.0.1 / LanceDB 0.29 / DataFusion 53 stack alignment

### 4.1 SoA version gate (D-MBX-10)

The layout-root header from §3.1 carries the schema version. Discipline:

1. **A v(M) reader MUST refuse v(N>M) bytes without an explicit version handshake.** Same rule as the `CausalEdge64` v1↔v2 layout reclaim (`I-LEGACY-API-FEATURE-GATED`, the Sprint-11 5-instance catalogue).
2. **Every column addition / widening / reclaim ships with field-isolation matrix tests** (write each field, assert all other fields unchanged). Mandatory; the Sprint-11 catalogue caught 5 of these.
3. **Serialization paths gate on version**. The cold-path writer stamps the writer's version; the reader rejects newer versions cleanly.
4. **Migration tests are mandatory** for any version bump (round-trip new → old reader must produce a documented refusal; round-trip old → new reader must produce documented compatibility or refusal).

### 4.2 Workspace stack pins (D-MBX-11)

Current pins (verified 2026-05-29):

| Layer | Current | Target | Delta |
|---|---|---|---|
| arrow | `"58"` | `"58"` | ✓ |
| datafusion | `"53"` | `"53"` | ✓ |
| lance | `"=7.0.0"` | `"=7.0.0"` | ✓ **SHIPPED #445** (jumped past the planned `=6.0.1`) |
| lancedb | `"=0.30.0"` | `"=0.30.0"` | ✓ **SHIPPED #445** (was `=0.29.0` at author-time) |
| ndarray | path-dep | path-dep | ✓ (governed by `PR-NDARRAY-MIRI-COMPLETE`) |

**[2026-06-14 SUPERSEDED]** No bump pending — main shipped `lance =6.0.0 → =7.0.0` + `lancedb =0.29.0 → =0.30.0` (lockstep, 7 crates, PR #445), jumping past the planned `=6.0.1` (which never existed on the lancedb path; lancedb 0.30 is the first release pinning lance =7). D-MBX-11 is closed by #445; the only residual is the surrealdb-fork pin (`TD-SURREALDB-KVLANCE-LANCE7`). Files that carried the pins: `crates/lance-graph/Cargo.toml:38`, `crates/lance-graph-benches/Cargo.toml:10`, `crates/lance-graph-callcenter/Cargo.toml:30`, `crates/lance-graph-ontology/Cargo.toml:46`, `crates/holograph/Cargo.toml:38`.

### 4.3 SurrealDB transparent-view enablement

Once D-MBX-7 (lance-graph containers = MailboxSoA layout) + D-MBX-10 (version gate) + D-MBX-11 (Lance 6.0.1) land, the SurrealDB-on-kv-lance backend (the *view*) can read the LanceDB rows zero-copy. SurrealDB is **NOT** a store; it is a view (§2.7). The Rubicon kanban is one SurrealDB query over the SoA-shaped LanceDB rows.

**Blocker:** `surreal_container` (`crates/surreal_container/src/lib.rs`) is still BLOCKED(A/B/C/D) — fork dep + Lance 6 pin confirmation + ndarray patch alias. `D-MBX-11` removes BLOCKED(A); the others still need a fork-access human to provide the surrealdb fork URL + `kv-lance` feature flag (`OQ-11.6`).

---

## 5. The 4-phase Rubicon kanban — wiring spec

### 5.1 The columns

(Same as `bindspace-singleton-to-mailbox-soa-v1` §11.3; restated for self-containment.)

| # | Kanban column | Mailbox state | Ractor | Libet wall-clock |
|---|---|---|---|---|
| 1 | **Planning** | counterfactual deliberation; **the ractor mailbox OWNS the SoA**; no commit; energy integrates under `InferenceType::Counterfactual` | alive, accumulating | t < −550 ms |
| 2 | **Cognitive work** | actional phase; mailbox SoA mutates (R1); Σ10 commit ratchets the card into this column | actional | t ≥ −550 ms |
| 3 | **Evaluation of goalstate** | post-actional reflection: read back over the witness arc; compute residual F | evaluating | t > 0 |
| 4 | **Commit · Plan · Prune** | 3-way terminal decision: | terminating | terminal |
|   | → Commit | calcify (LanceDB SPO-G + AriGraph episodic chain pointer) | STOP + tombstone | — |
|   | → Plan | re-enter column 1 with witness folded into next deliberation | RESTART | — |
|   | → Prune | drop without persistence (Libet veto consummated post-hoc) | drop | — |

### 5.2 Wiring (D-MBX-8 + D-MBX-9)

- **`D-MBX-8`** (Σ10 timing anchor in `SigmaTierRouter`): the commit decision `ΔF < threshold ∧ resonance > Rubicon-bar` acquires a wall-clock stamp at t = −550 ms. The router's commit emission carries the stamp; downstream the ractor START fires.
- **`D-MBX-9`** (Rubicon kanban view in `surrealkv`-on-lance): one SurrealDB query projecting the 4 columns over the SoA-shaped LanceDB rows. Ractor lifecycle hooks (spawn / start / evaluate / commit·plan·prune) write to the SurrealDB view; reading the view = reading the kanban state.

### 5.3 Active-inference loop closure

The "Plan" branch in column 4 makes "the shader can't resist the thinking" literal: failed goalstate evaluation re-feeds the mailbox; the system rests only at terminal Commit (calcify) or Prune (drop). Free-energy floor (`MUL::homeostasis`) is the rest condition.

---

## 6. AriGraph episodic Markov chain — witness pointer model

### 6.1 The chain IS the index space

The AriGraph episodic Markov chain is *not* a separate data structure; it is the **temporal sequence of mailbox states**. Each mailbox is a chain node; the witness arc inside a mailbox carries the local emission sequence; the chain edges are baton handoffs between mailboxes (`CollapseGateEmission`).

Reading episodic memory = traversing the chain backward from a current mailbox via arc-handle pointers; resolving a witness = dereferencing a pointer into the chain.

### 6.2 The witness pointer (D-MBX-A5)

Per-row witness arc handle: `witness_arc: [u32; W]` (width `W` per `OQ-11.2`). Each `u32` is an index into the global arc array; reading W of them gives the row's emission history.

Commit modalities (the SoA decides — R4):

- **(a) Pointer to other mailboxes in the chain.** The mailbox emits a baton `(u16 target, CausalEdge64)` to another mailbox; the receiving mailbox stores the arc-handle as part of its witness. This is the inter-mailbox episodic link.
- **(b) Cold-path fact.** The SoA emits a SPO-G quad to LanceDB; the quad carries the arc-handle pointer back into the AriGraph chain node. The chain node is the witness.

### 6.3 Storage invariant

The witness lives where the arc lives (a mailbox row inside the chain). Everywhere else is *pointers, never copies*. No SPO-G quad carries witness payload — only the pointer. Resolvable without storage redundancy.

---

## 7. Counterfactual Staunen × Wisdom plasticity spreader (D-MBX-A4)

### 7.1 Today's behaviour

`mailbox_soa.rs::apply_edges` increments `plasticity_counter[row]` only on the receiving row (saturating u8). Single-row Hebbian.

### 7.2 Target behaviour

When the mailbox is in Planning (column 1, counterfactual phase, t < −550 ms) AND the row's Staunen × Wisdom qualia magnitude exceeds threshold, the plasticity bump *spreads* to a small radius of adjacent rows (Hebbian spread). The radius / decay / column-local-vs-baton-routed semantics are open (`OQ-11.1`).

The spread happens AS A MAILBOX SOA MUTATION (R1) — not a side channel, not an async event. The mutation is the hot path; no other write surface introduced.

### 7.3 Why "counterfactual" gates it

Counterfactual phase = exploration, where alternatives are weighed. Plasticity spread = "this is interesting / worth remembering more broadly". Outside Planning (i.e. once committed), plasticity stays focal — committed edges only bump their own row's counter.

### 7.4 Mechanics (sketch — final pinned in D-MBX-A4 review)

```rust
// In apply_edges (mailbox in Planning + Staunen×Wisdom high):
if mailbox.phase == Planning && row_qualia.staunen_wisdom_product() > THRESHOLD {
    for offset in -RADIUS..=RADIUS {
        if let Some(neighbor) = row.checked_add_signed(offset) {
            if neighbor < N {
                let decayed = primary_bump.saturating_div(1 + offset.abs() as u8);
                self.plasticity_counter[neighbor] =
                    self.plasticity_counter[neighbor].saturating_add(decayed);
            }
        }
    }
} else {
    // existing single-row bump
    self.plasticity_counter[row] = self.plasticity_counter[row].saturating_add(1);
}
```

Final parameters pinned per OQ-11.1.

---

## 8. `lance-graph-planner` DTO surface overhaul (D-MBX-A6)

### 8.1 What's wrong today

The planner DTOs (`PlanResult`, `QueryFeatures`, `StrategySelector`, the 16 strategies, the AutocompleteCache DTOs) predate the SoA contract. They carry their own payloads; the planner translates between MUL assessment, NARS type selection, semiring selection, etc., and the eventual shader dispatch happens *after* this translation chain.

This is exactly the re-encode boundary R1 forbids: the planner reads its own DTOs, makes decisions, then translates to shader inputs. Under the rule, the planner should operate *on* the SoA directly and emit *kanban moves*.

### 8.2 Overhaul shape

- **Replace** payload-carrying DTOs with view-projection types: each DTO becomes a typed lens into a SoA row (cf. `E-NORMALIZED-ENTITY-1`'s `NormalizedEntity<Stage>` pattern — typestate over a mailbox row).
- **Planner output ≡ kanban transition.** A planner step produces a `KanbanMove { from, to, witness_pointer }` instead of a `PlanResult`. The 4-phase kanban (§5.1) is the planner's output state space.
- **Strategy selection reads SoA columns directly** (`meta` for ThinkingStyle, `edges` for CausalEdge64, `qualia` for i4-16D); no parallel `QueryFeatures` struct.
- **The MUL gate stamps Σ10 commit at −550 ms** wall-clock (D-MBX-8). The commit decision is the Planning → Cognitive Work kanban transition.
- **Active-inference loop** is the Plan-branch in column 4: a failed goalstate evaluation emits a `KanbanMove { to: Planning, ... }` and the cycle re-enters.

### 8.3 Backward-compatibility discipline

`I-LEGACY-API-FEATURE-GATED`: the v1 planner DTOs survive behind a feature flag pointing to the v2 SoA-operation equivalents. Same function names MUST NOT silently change semantics; feature-gate to documented no-op + migration pointer, or route through the canonical mapping.

### 8.4 Sequence

- Phase 1: introduce `KanbanMove` type + `KanbanColumn` enum in `lance-graph-contract`.
- Phase 2: add SoA-row-lens variants of `PlanResult` etc. behind `planner-soa-v2` feature.
- Phase 3: rewrite each of the 16 strategies to emit `KanbanMove`s.
- Phase 4: cut over `SigmaTierRouter` to consume `KanbanMove` directly.
- Phase 5: delete v1 DTOs.

---

## 9. Migration phases — sequenced gating

### Phase P0 — design ratification (this plan)

- **Status:** SHIPPED in PR #434 (merged 2026-05-29; see post-merge review addendum at `.claude/plans/unified-soa-convergence-v1-addendum-2026-05-29-review.md`).
- **Output:** unified-soa-convergence-v1.md (this file) + handover doc + epiphany pointers.

### Phase P1 — prerequisites

- **D-CE64-MB-1-impl** (par-tile crate apex + `Mailbox<T>` + 3 backings + AttentionMask SoA + `BindSpaceView`). Already specced (Sprint-11 W1). Blocking gate.
- **`PR-NDARRAY-MIRI-COMPLETE`** — close `U16x32 / U32x16 / U64x8` SIMD method gaps. Cross-repo (ndarray PR). Blocking gate.
- **D-MBX-11** — ~~Lance 6.0.0 → 6.0.1 patch bump~~ **SUPERSEDED by #445** (main shipped `=6.0.0 → =7.0.0` / lancedb `=0.30.0`; no `=6.0.1` step).

### Phase P2 — SoA version gate + stack alignment

- **D-MBX-10** — SoA version byte at layout root + field-isolation matrix tests.
- **D-MBX-11** — Lance 6.0.1 (verifies stack pin for SurrealDB view).
- **OQ-11.5** ratified (version field width).

### Phase P3 — cognitive-shader-driver column completion

- **D-MBX-A2** — close BindSpace expressivity gaps: `content_ref` column, S/P/O role slices, temporal/expert fold per OQ-2.
- **D-MBX-A3** — `witness_arc: [u32; W]` column. OQ-11.2 ratified (W width).
- **D-MBX-A4** — Staunen × Wisdom plasticity spreader. OQ-11.1 ratified (radius/decay).
- **D-MBX-2** — collapse `engine_bridge` re-encode seam (folds in TD-RESONANCEDTO-DUP-1).

### Phase P4 — singleton dissolution

- **D-MBX-3** — `ShaderDriver` holds sea-star of mailboxes; kill the `BindSpace::zeros(4096)` singleton in `serve.rs`.
- **D-MBX-A5** — SPO-W witness pointer column (the dual residency: SoA / kanban / mailbox index).
- **D-MBX-5** — delete `BindSpace` singleton + `Vsa16kF32` `cycle` plane. **Blocked on OQ-11.4** (CLAUDE.md "The Click" doctrinal update — must precede the deletion).

### Phase P5 — kanban / planner / Rubicon wiring

- **D-MBX-8** — Σ10 commit stamps t = −550 ms wall-clock in `SigmaTierRouter`.
- **D-MBX-9** — Rubicon kanban view in `surrealkv`-on-lance. Blocked on `surreal_container` BLOCKED(B/C/D) — OQ-11.6.
- **D-MBX-A6** — `lance-graph-planner` DTO surface overhaul (5 internal phases, see §8.4).

### Phase P6 — cold-path alignment

- **D-MBX-7** — `lance-graph` container layout ≡ `MailboxSoA` ≡ `simd_soa.rs`-aligned.
- **D-MBX-4** — death → SPO-G quad + Lance tombstone-witness (link-integrity back-pointer to AriGraph chain node).
- **D-MBX-6** — `ThoughtStruct` transparent hot/cold view over LanceDB (the SurrealDB view).

### Phase P7 — workspace-wide consumer alignment

- **D-MBX-12** — multi-PR sequence (one PR per consumer): D-MBX-12.1 AriGraph · 12.2 Vsa16k substrate audit · 12.4 lance-graph · 12.5 lance-graph-planner · 12.6 cognitive-shader-driver · 12.7 lance-graph-callcenter · 12.8 lance-graph-ontology audit · 12.9 thinking-styles/atoms.

**Sequencing recommendation (OQ-11.8):** start with D-MBX-12.4 (lance-graph cold containers) since it unlocks D-MBX-7 + the SurrealDB view; then D-MBX-12.5 (planner); the rest in dependency order.

---

## 10. Per-deliverable specifications

> Each deliverable below is one PR (or a small PR series for D-MBX-12 + D-MBX-A6). All include: tests (per-deliverable spec), board hygiene (LATEST_STATE / PR_ARC / STATUS_BOARD / EPIPHANIES updates in the same commit), and the `I-LEGACY-API-FEATURE-GATED` discipline where applicable.

### D-MBX-A2 — close BindSpace expressivity gaps in `MailboxSoA<N>`

**Owner:** `cognitive-shader-driver` + `lance-graph-contract`. **~140 LOC + tests.** **Risk: MED.**

**Adds:** `content_ref: [ContentId; N]` column (CAM-PQ code, 6 B/row) for content-identity reference; resolves OQ-1. S/P/O role slice columns (3 × `[u8; N]`) for spine coordinates per `I-VSA-IDENTITIES`. Resolves OQ-2 (temporal/expert fold into `CausalEdge64.temporal_v2` vs separate column).

**Tests:** field-isolation matrix for new columns; round-trip read/write per row; SoA byte-layout golden test (must match `MailboxSoAHeader` v(N) layout).

**Gates on:** D-CE64-MB-1-impl (par-tile crate); OQ-1 + OQ-2 ratified.

### D-MBX-A3 — `witness_arc: [u32; W]` arc-handle column

**Owner:** `cognitive-shader-driver`. **~100 LOC + tests.** **Risk: MED.**

**Adds:** per-row `witness_arc` column; `apply_edges` writes the arc-handle on every accepted baton; `emit` reads the arc to populate the emission's witness pointer. The arc is a ring (rotation on overflow) — width `W` per OQ-11.2.

**Tests:** witness arc preservation across mailbox lifecycle; rotation correctness on overflow; pointer-resolution round-trip.

**Gates on:** D-MBX-A2; OQ-11.2 ratified.

### D-MBX-A4 — Staunen × Wisdom plasticity spreader

**Owner:** `cognitive-shader-driver`. **~80 LOC + tests.** **Risk: LOW.**

**Adds:** conditional spread in `apply_edges` when mailbox is in Planning phase AND row's Staunen × Wisdom product exceeds threshold. Radius / decay per OQ-11.1. **Hot-path-only** (no side channel).

**Tests:** spread radius correctness; decay; phase-gating (no spread outside Planning); committed-mailbox immunity.

**Gates on:** D-MBX-A3; OQ-11.1 ratified; the `phase: KanbanColumn` field on `MailboxSoA` must exist (introduce in D-MBX-A6 phase 1 or earlier).

### D-MBX-A5 — SPO-W witness pointer dual-residency

**Owner:** `cognitive-shader-driver` + AriGraph SPO-G. **~150 LOC + tests.** **Risk: HIGH.**

**Adds:** SPO-G quad commit path writes the arc-handle pointer (not data) back to LanceDB; inter-mailbox baton handoff carries the arc-handle in the `CollapseGateEmission` payload. The SoA decides commit modality (a) chain-pointer vs (b) cold fact based on rung / confidence threshold.

**Tests:** pointer resolution from cold quad back to chain node; chain traversal correctness; no witness data duplication (audit).

**Gates on:** D-MBX-A3; D-MBX-4; AriGraph SPO-G schema reservation for arc-handle columns.

### D-MBX-A6 — `lance-graph-planner` DTO surface overhaul

**Owner:** `lance-graph-planner` + `lance-graph-contract`. **~600 LOC + tests, 5 internal phases (§8.4).** **Risk: HIGH.**

**Adds:** `KanbanMove` + `KanbanColumn` types in contract; SoA-row-lens variants of planner DTOs; cutover of 16 strategies; SigmaTierRouter consumes `KanbanMove`; v1 DTOs deleted at the end. Feature-flagged behind `planner-soa-v2` during cutover.

**Tests:** strategy-by-strategy regression suite; kanban transition correctness; field-isolation matrix on `KanbanMove`.

**Gates on:** D-MBX-10 (version gate); D-MBX-8 (timing anchor).

### D-MBX-7 — `lance-graph` containers ≡ `MailboxSoA` ≡ `simd_soa.rs`-aligned

**Owner:** `lance-graph` + `ndarray::simd_soa`. **~300 LOC + benchmark suite.** **Risk: HIGH.**

**Adds:** lance-graph container layout matches `MailboxSoA<N>` byte-for-byte; `ndarray::simd_soa` operates on the SoA columns directly; benchmark suite establishes the 1.4–4.2× SIMD payoff baseline.

**Tests:** layout equivalence (byte-level); SIMD correctness (cosine, hamming, bundle); benchmark gates (no regression vs current).

**Gates on:** D-MBX-A2; D-MBX-10; D-MBX-11; `PR-NDARRAY-MIRI-COMPLETE`.

### D-MBX-8 — Σ10 commit stamp at t = −550 ms wall-clock

**Owner:** `sigma-tier-router` crate + `cognitive-shader-driver`. **~120 LOC + tests.** **Risk: MED.**

**Adds:** the `SigmaTierRouter` commit decision (`ΔF < threshold ∧ resonance > Rubicon-bar`) stamps a wall-clock instant; the stamp is `now() - 550ms` per Libet anchor; downstream the ractor START fires with this stamp. The kanban `Planning → Cognitive work` transition records the stamp.

**Tests:** stamp monotonicity; veto-window correctness (no commit if veto fires before −550 ms); regression vs D-CSV-10.

**Gates on:** D-MBX-A4 (`phase` field present); D-MBX-A6 Phase 1 (`KanbanMove` type).

### D-MBX-9 — Rubicon kanban view in `surrealkv`-on-lance

**Owner:** `surreal_container` + ractor outer-swarm. **~250 LOC + tests.** **Risk: HIGH.**

**Adds:** SurrealDB query projecting the 4 kanban columns over SoA-shaped LanceDB rows; ractor lifecycle hooks write into the view; reading the view returns kanban state.

**Tests:** kanban column membership invariants; lifecycle hook idempotence; view query correctness.

**Gates on:** D-MBX-7; D-MBX-8; surreal_container BLOCKED(B/C/D) resolved (OQ-11.6); `D-PERSONA-5` ractor outer-swarm runtime in flight.

### D-MBX-10 — SoA version byte + field-isolation discipline

**Owner:** `lance-graph-contract`. **~100 LOC + extensive test suite.** **Risk: HIGH.**

**Adds:** `MailboxSoAHeader` with version field; serialize / deserialize gating; refusal of v(N>M) bytes by v(M) reader; field-isolation matrix tests for every column addition/widening (template).

**Tests:** version refusal correctness; per-column isolation; serialize-deserialize round-trip; cross-version reader/writer compatibility matrix.

**Gates on:** none (foundation work); should land early in P2.

### D-MBX-11 — Lance 6.0.0 → 6.0.1 patch bump · **[2026-06-14 SUPERSEDED by PR #445 — shipped `=6.0.0 → =7.0.0` / lancedb `=0.30.0`, not `=6.0.1`]**

**Owner:** workspace Cargo.toml. **~10 LOC (mechanical).** **Risk: LOW.** **(Closed by #445; detail below is the original as-authored spec.)**

**Adds:** patch bump in `crates/lance-graph/Cargo.toml:38`, `crates/lance-graph-benches/Cargo.toml:10`, `crates/lance-graph-callcenter/Cargo.toml:30`, `crates/lance-graph-ontology/Cargo.toml:46`, `crates/holograph/Cargo.toml:38`.

**Tests:** workspace `cargo check` clean (run when cargo prohibition lifts).

**Gates on:** none; can land in parallel with par-tile prereq.

### D-MBX-12 — workspace-wide consumer alignment (multi-PR sequence)

**Owner:** per-consumer. **Total ~800 LOC across 8 sub-PRs.** **Risk: per-consumer.**

Sub-PRs:
- **D-MBX-12.1** — AriGraph chain integration (SPO-G quads carry arc-handle pointers).
- **D-MBX-12.2** — Vsa16k substrate audit (confirm no cross-boundary state; audit-only).
- **D-MBX-12.4** — lance-graph cold containers consume SoA (the unlock for SurrealDB view).
- **D-MBX-12.5** — lance-graph-planner consumes SoA (folds D-MBX-A6).
- **D-MBX-12.6** — cognitive-shader-driver fully aligned (after all A2-A5 land).
- **D-MBX-12.7** — lance-graph-callcenter Zone-2 consumes SoA.
- **D-MBX-12.8** — lance-graph-ontology boundary audit (no SoA leak into cache).
- **D-MBX-12.9** — thinking-styles/atoms unified into MetaWord + p64-bridge layer_mask.

**Sequencing (OQ-11.8):** 12.4 → 12.5 → 12.6 → 12.7 → 12.1 → 12.9 → 12.2 → 12.8.

---

## 11. Open questions catalogue

> Each OQ blocks at least one deliverable; ratification gate listed.

| OQ | Question | Blocks | Default proposal |
|---|---|---|---|
| **OQ-11.1** | Staunen × Wisdom plasticity spread: radius? decay function? column-local vs baton-routed? | D-MBX-A4 | radius = 3; decay = bump / (1 + |offset|); column-local in v1, baton-routed in v2. |
| **OQ-11.2** | Witness arc width `W` (per-row arc-handle count before rotation)? | D-MBX-A3 | W = 16 (~64 B/row at u32 handles; supports a typical short-horizon belief arc). |
| **OQ-11.3** | Kanban needs a "vetoed" or "ghosted" column distinct from "Prune"? | D-MBX-9 | No — Prune is the terminal-veto; ghost-tier preempt drops the card before column 2 (Cognitive work) entry and doesn't need its own column. |
| **OQ-11.4** | CLAUDE.md "The Click" / `Vsa16kF32` doctrinal update — when? | D-MBX-5 | Doctrinal update must land BEFORE D-MBX-5 ships (delete plane). Separate doc-PR. |
| **OQ-11.5** | SoA version field width and per-column version stamps? | D-MBX-10 | `version: u16` at layout root; no per-column stamps in v1 (rely on the layout-checksum field). Per-column stamps can be added in v2 if schema-bumps become frequent. |
| **OQ-11.6** | surrealdb fork URL + branch + `kv-lance` feature flag name? | D-MBX-9 | Needs a fork-access human. Long-standing BLOCKED(C). |
| **OQ-11.7** | `lance-graph-planner` DTO overhaul scope: clean break vs feature-gated v1/v2 coexistence? | D-MBX-A6 | Feature-gated per `I-LEGACY-API-FEATURE-GATED`; v1 lives behind `planner-soa-v1`, v2 default. Cut over per §8.4 phases. |
| **OQ-11.8** | D-MBX-12 sub-PR sequencing? | D-MBX-12 | 12.4 → 12.5 → 12.6 → 12.7 → 12.1 → 12.9 → 12.2 → 12.8. |

---

## 12. Risk matrix

| Risk | Severity | Mitigation |
|---|---|---|
| SoA version gate skew across consumers | HIGH | D-MBX-10 layout-checksum field; cross-consumer field-isolation matrix tests; refuse-on-mismatch + clean error path. |
| `lance-graph-planner` DTO overhaul regresses 16 strategies | HIGH | Per-strategy regression suite; feature-gated cutover (OQ-11.7); 5-phase staged migration (§8.4). |
| SurrealDB fork stays blocked (OQ-11.6) | HIGH | D-MBX-9 deferred; the rest of the plan ships without the SurrealDB view (it's a view, not the store). Kanban can still operate as in-memory state behind ractor; SurrealDB view materializes when surreal_container unblocks. |
| `Vsa16kF32` plane deletion (D-MBX-5) hits in-flight uses | MED | Doctrinal update (OQ-11.4) lands first; deletion behind feature gate first, then remove gate; field-isolation matrix on the column. |
| Lance 6.0.1 bump (D-MBX-11) breaks a transitive dep | LOW | Patch bump only; can be reverted; CI gates (when cargo prohibition lifts). |
| AriGraph chain back-pointer link integrity | HIGH | D-MBX-A5 chain-traversal tests; SPO-G quad → chain node round-trip; tombstone outlives mailbox invariant. |
| Witness arc rotation loses old-witness facts | MED | Rotation only after column 4 Commit (or Prune); committed witnesses persist via the SPO-G pointer (R4) before rotation. |
| Staunen × Wisdom spread amplifies noise | MED | Phase-gated (Planning only); plasticity counter saturates u8; in v1 column-local (no baton-routed amplification across mailboxes). |

---

## 13. Success criteria / acceptance tests

> Cargo currently prohibited (session-stability constraint). Tests below are the per-deliverable acceptance bar; **run when cargo prohibition lifts**.

### Per-phase acceptance

- **P1 (prereqs):** par-tile crate exists; ndarray SIMD primitives complete; `lance =7.0.0` workspace-wide (shipped #445; was specced as `=6.0.1`).
- **P2 (version gate):** `MailboxSoAHeader` carries version; round-trip read/write green; refusal-on-mismatch path tested.
- **P3 (shader columns):** all new columns present; field-isolation matrix green; existing tests still pass.
- **P4 (singleton dissolution):** `BindSpace::zeros(4096)` removed; `serve.rs` builds a sea-star of `MailboxSoA`s; integration test passes a full think-cycle through the new path.
- **P5 (kanban/planner):** SigmaTierRouter emits −550 ms stamp; `KanbanMove` flows through planner; SurrealDB view queryable.
- **P6 (cold-path):** `lance-graph` containers byte-equal `MailboxSoA`; SIMD benchmark shows ≥ 1.4× on representative workloads.
- **P7 (consumer alignment):** all nine D-MBX-12 sub-PRs merged; workspace `cargo check` clean; no Arrow re-encode boundary for the thoughtspace.

### Workspace-level acceptance

- **`cargo check`** clean across workspace.
- **`cargo test`** green for all lance-graph crates.
- **No `Vsa16kF32` carrier** in any cross-boundary type signature (`grep` audit).
- **No re-encode** between mailbox SoA and lance-graph storage row (verified by `D-MBX-12.4` golden tests).
- **SurrealDB transparent view** materializes the kanban over LanceDB rows without an Arrow round-trip (when surreal_container unblocks).

---

## 14. Dependencies graph (textual)

```
PR-NDARRAY-MIRI-COMPLETE ───────────┐
                                    ▼
D-CE64-MB-1-impl (par-tile) ───┬──► D-MBX-A2 ──► D-MBX-A3 ──► D-MBX-A4 ──► D-MBX-A5
                               │                                              ▼
                               │                                          D-MBX-2 ──► D-MBX-3 ──► D-MBX-5*
                               │                                              ▼
                               └─► D-MBX-10 (foundation) ─────────────────► D-MBX-A6 ──► D-MBX-12.5
                                                                                 ▲
D-MBX-11 (Lance 6.0.1) ────────────────────────────────────────────────────────┘
                                                ▼
                                            D-MBX-7 ──► D-MBX-12.4 ──► D-MBX-6 ──► D-MBX-9 (needs surreal_container)
                                                ▼                          ▲
                                            D-MBX-4 ───────────────────────┘
                                                ▼
                                            D-MBX-8 ──► D-MBX-9

(*) D-MBX-5 also gated on OQ-11.4 (CLAUDE.md doctrinal update).
```

---

## 15. Cross-references

- **Plans:**
  - `bindspace-singleton-to-mailbox-soa-v1.md` §11.1–§11.6 — the layered rulings ratified in this session.
  - `causaledge64-mailbox-rename-soa-v1.md` — canonical 5-crate + 7-PR plan; this doc sequences post-impl convergence.
  - `cognitive-substrate-convergence-v1/v2/v3.md` — i4 mantissa, gapless baton, Σ10 Rubicon (shipped + in flight).
  - `rung-persona-orchestration-v1.md` — D-PERSONA-5 ractor outer-swarm runtime (Queued).
  - `.claude/surreal/RECONCILIATION_with_canonical_plan.md` — surreal POC folded in.
  - `normalized-entity-holy-grail-v1` — `NormalizedEntity<Stage>` typestate pattern (E-NORMALIZED-ENTITY-1; informs D-MBX-A6 DTO-as-lens).

- **Epiphanies:**
  - `E-SOA-IS-THE-ONLY` (2026-05-29, this session) — the five rulings.
  - `E-MAILBOX-IS-BINDSPACE` (2026-05-27) — singleton dissolution.
  - `E-RUBICON-RACTOR` (2026-05-27) — Heckhausen + Libet grounding of Σ10 Rubicon.
  - `E-BATON-1` (2026-05-26) — LE Baton contract.
  - `E-CE64-MB-4` — mailbox-as-owner ⇒ compile-time UB impossibility.
  - `E-LADDER-SERVES-MAILBOX` — escalation ladder serves the mailbox; AriGraph hot→cold→tombstone.
  - `E-NORMALIZED-ENTITY-1` (2026-05-28) — typestate carrier informing planner DTO overhaul.
  - `linguistic-epiphanies-2026-04-19.md E21` — canonical Σ10 Rubicon tier doctrine.

- **Iron rules:**
  - `I-VSA-IDENTITIES` (bundle identities, not content).
  - `I-LEGACY-API-FEATURE-GATED` (governs the SoA version gate).
  - `I-SUBSTRATE-MARKOV` (Chapman-Kolmogorov; untouched at the local-bundle compute level).
  - `I-NOISE-FLOOR-JIRAK` (weak-dependence Berry-Esseen; relevant for any threshold derivation).

- **Code anchors:**
  - `crates/cognitive-shader-driver/src/mailbox_soa.rs` (D-MBX-A1 columns shipped; A2/A3/A4/A5 add to this file).
  - `crates/cognitive-shader-driver/src/bindspace.rs` (singleton to dissolve).
  - `crates/cognitive-shader-driver/src/driver.rs` (the `Arc<BindSpace>` holder at :56).
  - `crates/cognitive-shader-driver/src/bin/serve.rs:29` (the `BindSpace::zeros(4096)` to remove).
  - `crates/cognitive-shader-driver/src/engine_bridge.rs` (re-encode seam to collapse).
  - `crates/lance-graph-contract/src/cognitive_shader.rs:382` (`ShaderCrystal.persisted_row`).
  - `crates/lance-graph-ontology/src/registry.rs:39` (`LazyLock<NamespaceRegistry>` — AS IS).
  - `crates/lance-graph-ontology/src/lance_cache.rs` (the ontology cache — AS IS).
  - `crates/surreal_container/src/lib.rs` (the SurrealDB view layer — BLOCKED).
  - `crates/p64-bridge/src/lib.rs` (the conformance template — already LE).

- **PRs (recent context):**
  - PR #388 — `SigmaTierRouter` Rubicon-resonance dispatch (shipped).
  - PR #414 — Odoo families 0x64 / 0x90 + axioms + StyleCluster wiring (shipped).
  - PR #416 — recipes + atoms + savants + FIBU re-parent (shipped).
  - PR #417 — `E-CONTRACT-NO-SERIALIZE-2` correction (shipped).
  - PR #418 — bindspace-singleton-to-mailbox-soa migration spec (shipped; this plan extends).
  - PR #433 — `style_recipe` D-Atom + epiphany-brainstorm-council + 5 savant cards (shipped).
  - PR #434 — unified-soa-convergence-v1 + handover (shipped 2026-05-29; review addendum at `.claude/plans/unified-soa-convergence-v1-addendum-2026-05-29-review.md`).

---

## 16. Council bypass note

The `epiphany-brainstorm-council` pre-merge gate for `EPIPHANIES.md` additions (shipped in PR #433) is **bypassed** for this plan's epiphany pointers because the underlying rulings (`E-SOA-IS-THE-ONLY` plus refinements) are **author-stated** by the user, not derived. The council is for derived epiphanies where a wrong one would pollute downstream priors; user-stated rulings have higher authority and a different review surface (the user themselves).

This plan, however, IS open to the council for its *spec content* — the convergence sequencing, the D-MBX deliverable breakdown, the OQ catalogue, and the risk matrix. PR review is the appropriate channel.

---
