# Unified SoA — the Rubikon-model integration (v1)

> **One integrated SoA.** The golden image proved the stack composes; this plan
> is how the pieces become ONE struct-of-arrays that the **planner**, the
> **kanban** lifecycle, the **SurrealQL membrane**, and **thinking styles** all
> consume — landing in `lance-graph-planner` + `lance-graph-contract`, consuming
> **Ontology** + **OGAR inheritance**, with **ractor as the dummy ownership
> guarantee**.
>
> **Status legend:** ✅ SHIPPED (verified file:line this session) · ◐ PARTIAL ·
> ☐ PROPOSED (named target located; not yet wired) · 2026-06-20.
>
> Per CLAUDE.md "AGI IS the struct-of-arrays": new capability lands as a planner
> **operation over the one SoA**, never a new layer.

---

## 0. The thesis

The 16,384-board SoA — `NodeRow = key(16) | edges(16) | value(480)`, 8 MiB total
(CANON, `canonical_node.rs`) — is ONE struct-of-arrays. Everything below is a
**read or a write over that one SoA**: the planner queries it, the kanban
lifecycle advances it, SurrealQL projects it, thinking styles dispatch over it.
No copies, no per-subsystem mirror (R1 "one SoA never transformed").

---

## 1. What's SHIPPED — the golden-image foundation (verified this session)

- ✅ **Golden image** links the full stack in one binary, lockstep lance-7
  (`crates/symbiont`, D0).
- ✅ **Each node = one SoA `NodeRow`; external f64 → a typed `ValueTenant`**
  (`canonical_node.rs:394` `ValueTenant`; D1 = perturbation `node_field` → the
  `Energy` F32 tenant). 16k boards = 8 MiB zero-copy (E2).
- ✅ **BF16 4×4-Morton-tile Domino** via `ndarray::simd::bf16_tile_gemm_16x16`
  (AMX `TDPBF16PS` dispatch; AVX-512 fallback on AMX-denied guests) — D3.
- ✅ **NaN-detection projection surface = the demoted singleton BindSpace**
  (`lance_graph_contract::nan_projection`): a read-only fixed-offset/stride sweep,
  NaN/Inf by one integer exponent mask.
- ✅ **Kanban loop, synchronous writer-fires-kanban** (D2, `symbiont/kanban_loop.rs`):
  `version-tick → VersionScheduler::on_version → try_advance_phase`, Domino sweep
  as the `CognitiveWork` phase. `on_version` is a SYNC pure function
  (`contract/scheduler.rs`); `scheduler_seam.rs` drives the whole Rubicon arc with
  plain `#[test]`s; `mailbox_soa.rs:700` "no surreal / ractor message bus needed".
- ✅ **The live trigger** `LanceVersionScheduler::{drive_once,drive_at_latest}`
  over `VersionedGraph::versions()` (`lance-graph/src/graph/scheduler.rs`, 5
  `#[tokio::test]`s) — async ONLY because a *subscriber* reads a version it didn't
  write; the *writer* fires the kanban update synchronously.
- ✅ **ractor = ownership guarantee, not a message bus** (E-CE64-MB-4 / #477):
  `SymbiontBoard`'s single `&mut self` owner IS the mailbox-as-owner compile-time
  proof. No tokio, no messages — a structural/dummy wrapper.
- ✅ **Key-only neo4j render (superpower §2.4)** — `symbiont/key_render.rs` reads
  16384 boards touching ONLY the 32-byte head (128-bit GUID + 128-bit EdgeBlock):
  16384 nodes / 32768 edges from 512 KiB of heads, 7680 KiB of value slabs COLD;
  zero-value-decode proven by the `0xFF`-poison falsifiable probe. `SymbiontBoard`
  now materialises the contract's `edge_block_at`/`hhtl_path_at` key facets.

---

## 2. The four superpowers of the planner over the ONE SoA

The SoA is column-major (`MailboxSoaView`: `energy() -> &[f32]`,
`meta_raw() -> &[u32]`, `edges_raw() -> &[u64]`, `entity_type() -> &[u16]`,
`soa_view.rs:57-64`). That layout is what makes these O(1)-ish sweeps:

1. ☐ **Tenant → fingerprint META QUERY (meta-awareness over the standing wave).**
   Project ONE tenant column across all 16k boards into a SINGLE fingerprint —
   the planner reduces e.g. `energy()` (or any tenant) over the whole SoA into one
   `Fingerprint<256>`/`Vsa16kF32`, then queries THAT (cosine / CAM) as the
   mailbox-set's *standing wave*. "One tenant over 16k rows → one fingerprint →
   one meta-query" = self-awareness as a read, never a new struct.
2. ☐ **Temporal implicit Markov chain.** Chain the SoA via
   `lance-graph-planner` `temporal` (referenced `lib.rs` / `prediction/mod.rs`)
   as an IMPLICIT Markov chain — guaranteed Chapman-Kolmogorov by construction
   (I-SUBSTRATE-MARKOV: VSA bundle is the semigroup). No transition matrix.
3. ☐ **Project ANY tenant with the same trick.** `witness` (EpisodicWitness64),
   `CausalEdge64` (`edges_raw()` raw u64 → `CausalEdge64(raw)`), qualia, plasticity
   — superposed/reduced over the SoA the same way as (1). The tenant catalogue
   (`VALUE_TENANTS`) is the column set; the projection is generic.
4. ✅ **Key-only neo4j-grade render — ZERO value decode.** Read all 16k boards
   touching ONLY the 32-byte head: the 128-bit `NodeGuid` (node) + the 128-bit
   `EdgeBlock` (12 in-family + 4 inherited out-of-family edges). `key(16)+edges(16)`,
   never the 480-byte value slab — a Neo4j-like graph view at memory-scan speed
   (`hhtl_path_at`/`edge_block_at` accessors are declared on `MailboxSoaView` for
   exactly this, defaulting to `None` until the owner materialises the head).
   **SHIPPED** (`symbiont/key_render.rs` + `SymbiontBoard` overrides of
   `edge_block_at`/`hhtl_path_at`): `render_key_only(&[NodeRow])` reads only
   `row.key` + `row.edges`; the binary renders **16384 nodes / 32768 edges from
   512 KiB of heads, 7680 KiB of value slabs left COLD**. Zero-value-decode is a
   FALSIFIABLE probe (`render_ignores_value_slab`): poison every value slab with
   `0xFF` → render is byte-identical. `hhtl_path_of` lowers the 3×4 HHT cascade
   (HEEL·HIP·TWIG = 12 nibbles) to a `NiblePath`; classid/identity excluded
   (tested). 12 symbiont tests green.

---

## 3. The Rubikon / Heckhausen + Libet lifecycle

The kanban columns ARE the Heckhausen Rubicon action phases, Libet-anchored
(`kanban.rs:25-49`):

| Heckhausen phase | Kanban column | Libet anchor |
|---|---|---|
| Predecisional (weighing) | `Planning` | spawn |
| **Rubicon crossing** (Σ-commit) | `Planning → CognitiveWork` | **−550 000 µs** ✅ (`kanban.rs:124`) |
| Preactional + actional | `CognitiveWork` → `Evaluation` | 0 |
| Postactional (evaluation) | `Evaluation → {Commit \| Plan \| Prune}` | 0 |
| **Libet veto** ("free won't", last phase) | `Planning → Prune` (pre-Rubicon) | ☐ **−200 000 µs** (PROPOSED) |

- ✅ The −550 ms readiness-potential anchor is already stamped on the Σ-commit
  crossing.
- ☐ **PROPOSED contract enrichment:** the Libet veto window is **−550 ms .. −200 ms**;
  stamp **−200 000 µs** on the `Planning → Prune` veto edge (today it stamps 0).
  The veto IS the Rubikon model's last phase — the abort before the act.

---

## 4. The integration — where it lands, what consumes what

**Home:** `lance-graph-planner` + `lance-graph-contract` (the planner consumes the
SoA; the contract owns the trait airgap). The path is: *it entered the golden
image* (✅, symbiont links planner+surrealdb+OGAR+ractor) → now wire the planner
to drive the SoA.

- ☐ **Planner ↔ SoA.** `lance-graph-planner` plans OVER the `MailboxSoaView`
  columns (the four superpowers §2 are planner operations). JITson
  (`contract::jit::JitCompiler`) compiles the selected thinking-style kernel.
- ☐ **Thinking styles ↔ Rubikon.** OGAR **class DO/THINK** selects the
  thinking-style via an **i4-32D fingerprint CAM** with **implicit sparse
  adjacency** ("how other tasks do it" — i4-distance PROPOSES → `ClassView`
  ADDRESSES, per AGENT_LOG WD-1/WD-2). Best-practice styles come from **OGAR
  inheritance + the Ontology** (`lance-graph-ontology`), resolved `classid →
  ClassView` one layer up from the SoA (never in the columns).
- ☐ **SurrealQL DLL/AST adapter = the consumer/commit membrane.**
  `ogar-adapter-surrealql` (already a symbiont dep) lowers Cypher/Class DDL →
  SurrealQL; `surreal_container` projects the SoA columns read-only
  (`SurrealMailboxView`); the **SurrealQL re-read** (`read_via_kv_lance`) is the
  one remaining stub. Writes commit through it (ORM/SQL membrane).
- ☐ **Outer boundary = `lance-graph-callcenter`.** The outer SLA + the
  outer SQL/consumer commit membrane (ORM or whatever) + the version watcher
  (`LanceVersionWatcher`/`WatchReceiver`) for *subscriber* consumers.
- ✅ **ractor = the dummy ownership guarantee** threading through all of it —
  the mailbox owns its SoA exclusively (`&mut`), compile-time, no messages.

---

## 5. Sequence (queued increments, each a falsifiable probe)

1. ☐ Planner reads the symbiont SoA (a `MailboxSoaView`) and runs a real query.
2. ☐ Superpower §2.1 — tenant→fingerprint meta-query (one tenant, 16k rows → one
   fingerprint; cosine/CAM over the standing wave).
3. ✅ Superpower §2.4 — key-only 32-byte render (materialise `hhtl_path_at` /
   `edge_block_at`; assert zero value-slab reads). **SHIPPED** — see §2.4.
4. ☐ Superpower §2.2/§2.3 — `temporal` Markov chaining + project
   witness/CausalEdge64.
5. ☐ Rubikon §3 — the −200 ms Libet-veto anchor on `Planning → Prune`.
6. ☐ Thinking-style §4 — OGAR DO/THINK i4-32D CAM selects the style; JITson
   compiles it; the Rubikon lifecycle dispatches it.
7. ☐ Membrane §4 — `read_via_kv_lance` un-stubbed; callcenter SLA + commit.

---

## 6. Honest status (no overclaim)

- ✅ **SHIPPED + verified** (§1): the golden-image SoA foundation — every claim
  has a file:line read THIS session.
- ☐ **PROPOSED / named-target-located** (§2–§5): the planner integration and the
  four superpowers as planner operations, the −200 ms veto, OGAR DO/THINK
  thinking-style selection, the SurrealQL membrane, the callcenter boundary. The
  targets exist (`temporal` module, `jit.rs`, `ogar-adapter-surrealql`,
  `lance-graph-callcenter`, OGAR i4-32D / `ClassView`), but the wiring is the
  work — none of §2–§5 is claimed running.

---

## 7. Reconciliation with prior plans — this is NOT a new arc

This plan **complements, does not duplicate**, the existing Rubikon/kanban
convergence work. Cross-read this session (`unified-soa-convergence-v1.md` +
`D-MBX-COMPLETION-MAP.md` + the kanban EPIPHANIES). The mapping, so the two
plans cannot drift:

| This plan (§) | Canonical home (prior plan / D-id) | Status there | Relationship |
|---|---|---|---|
| §1 golden image SoA | — (new harness this session) | ✅ shipped | NEW — the proof-of-composition the prior arc lacked |
| §2.1–.4 four superpowers | `D-MBX-A6` (planner DTO overhaul) + §2 "one SoA, never transformed" | ☐ A6-P3 NEXT | the superpowers ARE planner ops over the SoA — land **under A6**, not a new layer |
| §3 Rubikon + Libet −550 ms | `unified-soa-convergence-v1` §5 R3 + `D-MBX-8` | ✅ −550 ms shipped | SAME anchor; no divergence |
| §3 Libet −200 ms veto | `unified-soa-convergence-v1` §5 R3**(a)** "pre-commit veto, card never leaves Planning" | ☐ unbuilt | **REFINES, not invents:** R3(a) named the *veto*; this plan gives it the *anchor* — RP at −550 ms, conscious intention ≈ −200 ms, "free won't" window between −200 ms and the act → stamp **−200 000 µs on `Planning → Prune`**. R3(b) post-eval Prune is the *other* veto expression and already exists. |
| §4 planner ↔ SoA | `D-MBX-A6` / `D-MBX-12.5` (planner sub-PR) | ☐ A6-P3 NEXT | identical target; **do NOT mint a new "PlannerDTO"** (E-2026-05-30 COUNCIL: that name is drift — canonical = `Candidate` + `KanbanMove`) |
| §4 thinking-style ↔ Rubikon | `style_strategy.rs` (#439 D-MBX-A6-P3a) + OGAR i4-32D CAM | ◐ #439 passthrough | the missing edge is `Outcome → KanbanMove` emit = **A6-P3**, gated by the planner-output overhaul |
| §4 SurrealQL membrane | `D-MBX-9` (Rubicon kanban VIEW) + `surreal_container::view::read_via_kv_lance` | ☐ stub, `OQ-11.6` | identical stub; gated on the surrealdb-fork dep being uncommented |
| §4 callcenter boundary | `callcenter-membrane-v1` DM-4 (`LanceVersionWatcher`) | ✅ shipped (std-sync) | the *subscriber* fan-out already exists — §4's "version watcher" = this |
| §2.4 key-only render | `E-GUID-IS-THE-GRAPH` + `E-CYPHER-IS-THE-KANBAN-AST` | ✅ FINDING | the zero-value-decode graph view is the doctrine; §2.4 is its planner op |

**Loose ends tied (the reconciliation deltas):**

1. **§4 is `D-MBX-A6-P3`, full stop.** The planner integration is not a parallel
   effort — it is the NEXT node of the shipped A6 chain (P1 #437 + P2 #439 land
   2 of 3 edges; the missing edge is `Outcome → KanbanMove`). This plan's §4
   should be read as the **capstone narrative** for A6-P3, not a competing spec.
2. **The −200 ms veto is the one genuinely-new contract enrichment** here, and it
   is small: a single `libet_offset_us = -200_000` stamp on the `Planning → Prune`
   edge in `advance_phase` (today it stamps 0), mirroring the −550 000 already on
   `Planning → CognitiveWork`. It refines `unified-soa-convergence-v1` R3(a); it
   does not contradict R3(b).
3. **No new kanban/SoA/scheduler type.** Everything routes through shipped
   `contract::{kanban, soa_view, scheduler}` + `surreal_container::view` +
   `callcenter::version_watcher`. Per CLAUDE.md "AGI IS the struct-of-arrays" and
   `E-CYPHER-IS-THE-KANBAN-AST`: board-ops + ontology-traversal + thinking-style
   dispatch + SurrealQL egress are **one AST seen from four sides**, not four
   subsystems to bridge.
4. **`D-MBX-7` is the hard prerequisite for the transparent view** (lance-graph
   containers ≡ `MailboxSoA` layout) and `D-MBX-11`/`OQ-11.6` (Lance-7 pin +
   surrealdb-fork URL) gate the kv-lance re-read. The golden image already pins
   lance-7 lockstep, so the `D-MBX-11` blocker is *retired in the harness* — the
   surrealdb fork's `claude/kvs-lance-timeline` branch bumped to `lance =7.0.0 /
   lancedb =0.30.0` to match (verified this session). What remains is `OQ-11.6`
   (uncomment the fork dep, fill `read_via_kv_lance`).

---

## 8. The SurrealQL superpowers (grounded in the AdaWorldAPI fork)

The operator's note — *"I think we didn't understand properly the superpowers
that SurrealQL grants us, time series etc"* — is correct: SurrealQL is not just
an egress dialect, it is a **second native runtime over the SAME Lance bytes**.
All four below are VERIFIED present in `/home/user/surrealdb` (the AdaWorldAPI
fork), not generic-SurrealDB hearsay:

1. ✅ **kv-lance engine = SurrealDB runs ON Lance.**
   `surrealdb/core/src/kvs/lance/{mod,schema,tx_buffer,background_optimizer,
   timeline}.rs` — a full `Transactable` KV backend on the Lance versioned
   columnar format. **Consequence:** the SAME `nodes.lance` dataset the SoA
   batch-writer commits IS the SurrealDB store. No ETL, no copy — the
   "transparent container view" (`D-MBX-7`) is literally SurrealDB pointed at the
   SoA's own Lance files. This is why `surreal_container::view` borrows zero-copy
   (§4): the bytes are already there.
2. ✅ **Time-series = `kvs/lance/timeline.rs` + record-id ranges.**
   The `timeline.rs` module in the kv-lance backend is the time-series surface;
   SurrealQL record-id ranges (`SELECT * FROM mailbox:[$lo]..[$hi]`) scan the
   monotonic version/cycle axis directly. **Maps to** superpower §2.2 (the
   `temporal` implicit Markov chain) and the witness arc (`CausalEdge64`
   emissions are an ordered series; the SurrealQL range IS the arc read). The
   Lance version log = the time axis = the kanban witness chain; no separate
   event store (cf. `E-2026-...` "the actor's state history IS the Lance version
   log").
3. ✅ **LIVE SELECT + CHANGEFEED = the version-subscription IN-direction.**
   `surrealdb/core/src/expr/statements/live.rs` + `key/table/lq.rs` (live-query
   keys) + `CHANGEFEED` on `DEFINE TABLE`/`DEFINE DATABASE`
   (`expr/statements/define/{table,database}.rs`). **This is CDC**, and it is the
   exact shape of `LanceVersionScheduler::drive_once` (async subscriber reads a
   version it didn't write) + `callcenter::LanceVersionWatcher` (std-sync
   always-latest fan-out). A `LIVE SELECT … FROM mailbox` is the SurrealQL
   spelling of "subscribe to the kanban version tick" → `on_version` →
   `KanbanMove`. The writer still fires the kanban update **synchronously** (§1);
   LIVE/CHANGEFEED is for *other* consumers (the subscriber half), matching the
   "async only because it reads a version it didn't write" ruling.
4. ✅ **RELATE = native graph edges ↔ `EdgeBlock`.**
   `surrealdb/core/src/expr/statements/relate.rs` — SurrealQL `RELATE a->edge->b`
   is a first-class graph edge. **Maps to** the `EdgeBlock` (12 in-family + 4
   out-of-family slots) and superpower §2.4 (key-only neo4j-grade render).
   A `RELATE`/`->edge->` traversal over the kv-lance store IS the prefix-route +
   slot-deref of `E-GUID-IS-THE-GRAPH`, expressed in SurrealQL instead of Cypher
   — the same AST, the egress side (`E-CYPHER-IS-THE-KANBAN-AST`).

**Net:** the four superpowers of §2 (meta-query / temporal Markov / project-any-
tenant / key-only render) each have a **SurrealQL spelling** over the identical
Lance bytes — meta-query via aggregate `SELECT`, temporal via record-range +
`timeline.rs`, tenant projection via column `SELECT`, key-only render via
`RELATE`/`->`. SurrealQL is not a sink we *export to*; it is a co-equal lens on
the one SoA, gated only by `OQ-11.6` (uncomment the fork dep + fill
`read_via_kv_lance`). The membrane is already typed and tested
(`surreal_container::view`, 5 tests green); only the kv-lance scan body is a stub.

---

**Cross-ref:** symbiont `INTEGRATION_PLAN.md` + `BATTLE_TEST_PLAN.md`; CANON
(`canonical_node.rs`); `kanban.rs` / `scheduler.rs` / `soa_view.rs`;
`scheduler_seam.rs`; `surreal_container/src/view.rs` (the `read_via_kv_lance`
stub); `lance-graph-callcenter/src/version_watcher.rs` (the subscriber fan-out);
`/home/user/surrealdb/surrealdb/core/src/kvs/lance/` (the kv-lance engine +
`timeline.rs`). **Prior plans reconciled:** `unified-soa-convergence-v1.md`
(D-MBX-7/8/9/A6/12, §5 R3 Libet, §7 plasticity); `cypher-kanban-ast-unification-v1.md`;
`callcenter-membrane-v1.md` (DM-4 watcher). **EPIPHANIES:**
`E-NODE-IS-SOA-IS-KANBAN-BOARD`, `E-BINDSPACE-IS-A-NAN-PROJECTION-SURFACE`,
`E-SCENT-IS-NOT-READING`, `E-CYPHER-IS-THE-KANBAN-AST`, `E-GUID-IS-THE-GRAPH`,
`E-SUBSTRATE-IS-THE-SCHEDULER`. STATUS_BOARD `symbiont-golden-image-harness`.
