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
4. ☐ **Key-only neo4j-grade render — ZERO value decode.** Read all 16k boards
   touching ONLY the 32-byte head: the 128-bit `NodeGuid` (node) + the 128-bit
   `EdgeBlock` (12 in-family + 4 inherited out-of-family edges). `key(16)+edges(16)`,
   never the 480-byte value slab — a Neo4j-like graph view at memory-scan speed
   (`hhtl_path_at`/`edge_block_at` accessors are declared on `MailboxSoaView` for
   exactly this, defaulting to `None` until the owner materialises the head).

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
3. ☐ Superpower §2.4 — key-only 32-byte render (materialise `hhtl_path_at` /
   `edge_block_at`; assert zero value-slab reads).
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

**Cross-ref:** symbiont `INTEGRATION_PLAN.md` + `BATTLE_TEST_PLAN.md`; CANON
(`canonical_node.rs`); `kanban.rs` / `scheduler.rs` / `soa_view.rs`;
`scheduler_seam.rs`; EPIPHANIES `E-NODE-IS-SOA-IS-KANBAN-BOARD`,
`E-BINDSPACE-IS-A-NAN-PROJECTION-SURFACE`, `E-SCENT-IS-NOT-READING`; STATUS_BOARD
`symbiont-golden-image-harness`.
