# Particle / SoA Envelope Audit — Does the Struct Geometry Measure What It Claims?

> **Branch:** `claude/stoic-turing-M0Eiq`
> **Date:** 2026-06-06
> **Scope:** Current code only. No standing-wave, no emergent-cognition, no
>   philosophy. No assumption of a global carrier, wave field, f32 carrier,
>   singleton thought, or distributed truth ownership.
> **Method:** Whole-file reads of the envelope, edge, reference, scheduler,
>   bridge, and ontology surfaces in `lance-graph` + `OGAR`. Every claim is
>   file-path grounded and tagged Confirmed / Inferred / Absent / Contradiction.

**Intended model under test:**
1. One mailbox-owned logical thought.
2. One SoA envelope represents that thought.
3. Identity from immutable semantic address space.
4. Schema inheritance via `OGAR::classes::from(address)`.
5. Lance versioning = self-through-time.
6. `CausalEdge64` = compact local causal/NARS claim **payload**.
7. References = explicit pointers, not copied truth.
8. Pragmatics = local state in the envelope.
9. AST adapter selection inherited from OGAR classes.
10. The question: does the geometry measure what it claims?

---

## 1. File-path grounded findings

### Phase 1 — Locate the actual thought envelope

**There are TWO competing SoA envelopes in the tree. This is the single
biggest structural finding.**

| Envelope | File | Role | Status |
|---|---|---|---|
| `MailboxSoA<const N>` | `crates/cognitive-shader-driver/src/mailbox_soa.rs` | The **intended** per-mailbox thought envelope (one owner, N rows) | **Confirmed** — matches intended model |
| `BindSpace` | `crates/cognitive-shader-driver/src/bindspace.rs` | The **older** global/process-wide SoA, still carrying the deprecated 65 KB/row `Vsa16kF32` `cycle` plane | **Contradiction** — co-exists with MailboxSoA, no migration seam |
| `MailboxSoaView` / `MailboxSoaOwner` (traits) | `crates/lance-graph-contract/src/soa_view.rs` | The **canonical zero-copy read/write contract** the envelope is supposed to satisfy | **Confirmed** — clean abstraction |

- `MailboxSoA<1024>` columns (`mailbox_soa.rs`): `energy [f32;N]`,
  `plasticity_counter [u8;N]`, `last_emission_cycle [u32;N]`,
  `edges [CausalEdge64;N]`, `qualia [QualiaI4_16D;N]`, `meta [MetaWord;N]`,
  `entity_type [u16;N]`; scalars `mailbox_id`, `current_cycle`, `w_slot`,
  `threshold`. **This is the particle envelope the intended model describes.**
  ⚠ The source contains `emit(MailboxId) -> CollapseGateEmission` — this method
  contradicts the zero-copy model (creation to Lance tombstone, no emission,
  no baton). It is a code artifact to be removed, not the intended design.
- `BindSpace` columns (`bindspace.rs`): `FingerprintColumns { content [u64×256],
  cycle [f32×16_384], topic, angle, sigma }`, `EdgeColumn`, `QualiaI4Column`
  (+ deprecated `QualiaColumn` f32×18), `MetaColumn`, `temporal`, `expert`,
  `entity_type`, `ontology: Option<Arc<OntologyRegistry>>`. Per-row footprint
  **71,713 bytes**, of which **65,536 bytes is the `cycle` `Vsa16kF32` plane**.
- The `ShaderDriver` (`driver.rs`) holds **both**: `bindspace: Arc<BindSpace>`
  **and** `mailboxes: HashMap<MailboxId, MailboxSoA<1024>>` (commented
  "transitional per-mailbox routing (slice A2)"). The driver is mid-migration
  from the global BindSpace to per-mailbox SoAs and **both surfaces are live**.

**Phase 1 output:**

| Tag | Finding |
|---|---|
| **Confirmed** | `MailboxSoA<N>` is the per-thought envelope; `soa_view.rs` traits are the canonical contract; identity scalar is `mailbox_id: MailboxId (u32)`. |
| **Confirmed** | State for one thought is *intended* to be one `MailboxSoA`. |
| **Inferred** | The driver's `mailboxes: HashMap` is the active routing path; BindSpace is legacy being drained. |
| **Absent** | No single canonical envelope is *declared* as THE envelope — both compile, both are reachable. |
| **Contradiction** | Two SoA representations co-exist (`BindSpace` global + `MailboxSoA` per-mailbox). The deprecated `Vsa16kF32` carrier is **not** absent — it is the live `cycle` column of `BindSpace` at 65 KB/row, contradicting "no f32 carrier." |

---

### Phase 2 — Identity audit

**Identity = `mailbox_id: u32` (envelope identity) + `entity_type: u16`
(class identity pointer). The intended `OGAR::classes::from(address)` reverse
resolver does NOT exist in OGAR — its equivalent lives in lance-graph as a
linear scan.**

- Envelope identity: `MailboxSoA.mailbox_id: MailboxId` (`= u32`), the corpus
  root handle. `WitnessEntry.mailbox_ref: u32` preserves *full* identity across
  cohort rotation (wider than the 6-bit W-slot index — good design).
- Class identity: `entity_type: u16` per row. `bindspace.rs:244` doc:
  *"0 = untyped. Non-zero = 1-based index into Ontology.schemas."*
- Resolution (`driver.rs:331-338`): `etid = entity_type[row]` → if 0 no context,
  else `OntologyRegistry::enumerate_first_with_entity_type_id(etid)` →
  `MappingRow.ontology_context_id()` → `MulThresholdProfile::for_context(ctx_id)`.
- **The resolver is a linear `.find()` scan** (`lance-graph-ontology/src/registry.rs:327`)
  over `SchemaPtr::entity_type_id()`, **not** an O(1) array index, despite the
  "1-based index" doc wording. `entity_type_id` is packed in `SchemaPtr` bits 23..8.
- OGAR side: only the **forward** builder exists —
  `ogar_ontology::class_identity(prefix, class_name) -> String` (e.g.
  `"ogit-op/WorkPackage"`). **No `classes::from(address)`, no `class_for`,
  no reverse resolver** anywhere in OGAR crates. The address string *is* the
  class key (Confirmed); the reverse map is delegated to lance-graph's
  `NiblePath` router + `OntologyRegistry`.
- Schema is stored **once per class** (`OntologyRegistry` `MappingRow`, idempotent
  on checksum). **No schema is duplicated into SoA rows** — rows carry only the
  `u16` pointer. (Confirmed — clean.)

**Identity diagram:**

```
                       envelope identity                 class identity
                       ────────────────                  ──────────────
   MailboxSoA.mailbox_id : u32  ◄── owner key       entity_type[row] : u16   (0 = untyped)
            │                                               │  1-based, dense in append order
            │                                               ▼
            │                              OntologyRegistry::enumerate_first_with_entity_type_id(u16)
            │                                               │  ⚠ LINEAR SCAN, not O(1) index
            │                                               ▼
            │                                         MappingRow  (schema stored ONCE per class)
            │                                          ├─ ontology_context_id()  → MulThresholdProfile
            │                                          ├─ schema_ptr (SchemaPtr: ns|entity_type_id|marking)
            │                                          └─ thinking_style: Option<ThinkingStyle>  ⚠ stored, UNUSED at dispatch
            │
            ▼
   OGAR forward only:  class_identity(prefix, name) -> "ogit-op/WorkPackage"
   OGAR reverse  (classes::from(address)) : ABSENT — lives in lance-graph as the scan above
```

| Tag | Finding |
|---|---|
| **Confirmed** | Schema meaning inherited (not duplicated); address string is the class key; schema stored once per class. |
| **Confirmed** | Identity is two-level: `mailbox_id` (envelope) + `entity_type` (class). |
| **Inferred** | "Immutable address space" holds as an *intended* downstream property (const prefixes, `@vN` append, VART append-only trie) — **not enforced in code**. |
| **Absent** | `OGAR::classes::from(address)` reverse resolver. The intended inheritance call site does not exist as named. |
| **Contradiction** | The realized resolver is a **linear scan keyed by `entity_type_id`**, while the doc claims "1-based index into schemas." Two different access models documented vs implemented. |

---

### Phase 3 — `CausalEdge64` audit

**Two incompatible `CausalEdge64` types exist. The canonical one is a payload,
but the v2 layout smuggles a *reference* (W-slot) into the payload word.**

Canonical: `crates/causal-edge/src/edge.rs` — `struct CausalEdge64(pub u64)`,
`Copy`, 8 bytes, no lifetime. v1 / v2 (feature `causal-edge-v2-layout`) layouts:

| Bits | v1 field | v2 field | Class | Authoritative? |
|---|---|---|---|---|
| 0..7 | S palette index | S palette index | payload (SPO) | **authoritative** |
| 8..15 | P palette index | P palette index | payload (SPO) | **authoritative** |
| 16..23 | O palette index | O palette index | payload (SPO) | **authoritative** |
| 24..31 | NARS frequency u8 | NARS frequency u8 | payload (truth) | **authoritative** |
| 32..39 | NARS confidence u8 | NARS confidence u8 | payload (truth) | **authoritative** |
| 40..42 | CausalMask (Pearl 2³) | CausalMask (Pearl 2³) | payload (causal) | **authoritative** |
| 43..45 | direction triad | direction triad | payload (causal) | **authoritative** |
| 46..48 / 46..49 | inference type (3-bit unsigned) | inference mantissa (4-bit **signed**) | payload (derived tag) | derived (enum) |
| 49..51 / 50..52 | plasticity | plasticity | pragmatic (local) | **authoritative (local)** |
| 52..63 | temporal (12-bit) | **RECLAIMED** | — | v1: payload; v2: gone |
| 53..58 | — | **W-slot (6-bit witness ref)** | **REFERENCE** | reference (→ WitnessTable) |
| 59..60 | — | truth-band TrustTexture | payload (derived) | derived |
| 61..63 | — | spare | — | — |

- **Thinking style is NOT a field of `CausalEdge64`.** Style lives in `MetaWord`
  (`MetaColumn`) and is selected by qualia auto-detection / `UnifiedStyle`, not
  carried on the edge. (Answers "which fields are thinking style": none.)
- Local/incompatible duplicate: `crates/thinking-engine/src/layered.rs` —
  a *different* `struct CausalEdge64(pub u64)` whose 8 bytes are **8 named
  channels** (BECOMES/CAUSES/SUPPORTS/REFINES/GROUNDS/ABSTRACTS/RELATES/
  CONTRADICTS), with `to_spo()/from_spo()` transcoders to the canonical type at
  the L3 commit boundary. Same name, different geometry. (Contradiction.)

**Field table verdict:**

| Question | Answer |
|---|---|
| Which fields authoritative? | S, P, O, frequency, confidence, CausalMask, direction, plasticity (local). |
| Which derived? | inference mantissa (enum tag), truth-band (from confidence). |
| Acts as payload? | **Yes** — Confirmed. It is a compact local causal/NARS claim. |
| Accidentally acts as identity? | **No** for v1. **Partially for v2** — the W-slot (bits 53-58) is a *reference index* into `WitnessTable`, mixing a pointer into the payload word. Not identity, but reference-in-payload. (Contradiction with "references should be explicit pointers, not packed into truth.") |

---

### Phase 4 — Reference audit

**References are explicit `Copy` handles (good). `EpisodicWitness64` as a SoA
*column* is absent (queued). Stale-copy risk is bounded; no circular ownership
in the Rust sense.**

- `EpisodicEdges64(pub u64)` (`lance-graph-contract/src/episodic_edges.rs`):
  4 × 16-bit slots, each an `EdgeRef { family: u8 (nibble), local: u16 (1-based) }`.
  MRU hot tier, slot 0 = strongest, slot 3 = eviction candidate. All `Copy`,
  functional updates (`promote`, `push` return new words). **Explicit pointers,
  not copied truth.** (Confirmed.)
- `DemotionSink::demote(&mut self, EdgeRef)` — the only mutating receiver, the
  seam to the cold connectome.
- `WitnessTable<64>` + `WitnessEntry { mailbox_ref: u32, spo_fact_ref: Option<u64> }`
  (`witness_table.rs`): resolves the v2 W-slot. `mailbox_ref` is the **full** u32
  identity (not the 6-bit slot) → rotation-safe. `spo_fact_ref` is `None`
  (ephemeral) or `Some` (crystallised AriGraph triple). `WitnessEntry` is `Copy`;
  the table is `Clone`-only (1.5 KiB).
- **`EpisodicWitness64` as a SoA column is ABSENT** — `soa_view.rs:75-95`
  explicitly comments the episodic/witness column as *queued, not yet landed*.
  The envelope does **not** yet carry the reference column the intended model
  names.

**Reference topology diagram:**

```
   MailboxSoA (owner of truth: edges[], energy[], qualia[], meta[])
        │ owns
        ▼
   CausalEdge64  ──(v2 W-slot, 6-bit)──►  WitnessTable<64>
        │                                      │ entry.mailbox_ref : u32   (REFERENCE, full id)
        │                                      └ entry.spo_fact_ref: Option<u64> (REFERENCE → AriGraph triple, or None)
        │
   EpisodicEdges64 (hot MRU word)
        └ slot[0..4] : EdgeRef{family,local}  (REFERENCE, explicit, Copy)
                 │ evict slot 3
                 ▼
           DemotionSink (→ cold connectome, the only &mut receiver)
```

| Classification | Members |
|---|---|
| **owned truth** | `MailboxSoA` columns (`edges`, `energy`, `qualia`, `meta`, `plasticity`). The mailbox is sole owner. |
| **reference** | `EdgeRef`, `EpisodicEdges64` slots, `CausalEdge64` v2 W-slot, `WitnessEntry.{mailbox_ref, spo_fact_ref}`. All explicit, all `Copy`. |
| **copied state** | `WitnessEntry.mailbox_ref` is a *copy of a u32 id* (not the row) — cheap, rotation-safe; not stale because it's the canonical wide id, not a slot index. |
| **derived state** | truth-band, inference enum from mantissa, `class_id` alias of `entity_type`. |

| Tag | Finding |
|---|---|
| **Confirmed** | References are explicit `Copy` pointers, not copied truth. No circular ownership (all handles, no `Rc`-cycles; mailbox is single owner). |
| **Inferred** | Stale-copy risk is bounded: the only copied identity is the *wide* `mailbox_ref`, which survives cohort rotation by design. The 6-bit W-slot *would* go stale on rotation, which is exactly why `WitnessTable` stores the full u32. |
| **Absent** | `EpisodicWitness64` SoA column (queued in `soa_view.rs`). |
| **Contradiction** | none for ownership; the only smell is the v2 W-slot reference packed into the `CausalEdge64` payload word (see Phase 3). |

---

### Phase 5 — Lance versioning audit

**Self-through-time is via `DatasetVersion(u64)` + `VersionScheduler` →
`KanbanMove`. History is NOT duplicated in rows. Clean, with one placeholder.**

- `scheduler.rs`: `DatasetVersion(pub u64)`; trait
  `VersionScheduler::on_version<V: MailboxSoaView>(&self, view: &V, at: DatasetVersion,
  exec: ExecTarget) -> Option<KanbanMove>`. The scheduler takes `&V` (shared,
  read-only) — **"propose, don't dispose": the scheduler never mutates; only
  `MailboxSoaOwner::advance_phase` mutates.** (Confirmed — clean ownership split.)
- `NextPhaseScheduler` advances the 6-phase Rubicon Kanban lifecycle on each
  Lance version tick. Planning→CognitiveWork stamps `libet_offset_us = -550_000`.
- Per-row time stamps in the envelope are `current_cycle: u32` and
  `last_emission_cycle [u32;N]` — these are **same-cycle idempotency guards**,
  not history. No previous-self snapshot is copied into rows.

**Version lineage diagram:**

```
   Lance dataset tick:  DatasetVersion(v)  ── increments ──►  DatasetVersion(v+1)
            │ on_version(&view, at, exec)
            ▼
   VersionScheduler (READ-ONLY &V)  ──proposes──►  KanbanMove { mailbox, from→to phase,
            │                                                    witness_chain_position, libet_offset_us }
            │ (caller applies)
            ▼
   MailboxSoaOwner::advance_phase(to)   ← SOLE mutator
            │
            ▼
   self-through-time = the sequence of Lance versions of the SAME mailbox dataset
   (previous-self is implicit in Lance history; NOT duplicated in rows)
```

| Classification | Verdict |
|---|---|
| **clean** | Previous-self is implicit via Lance versioning; no history duplicated in rows; read-only scheduler vs single mutating owner is a correct ownership split. |
| **redundant** | none material. `current_cycle` + `last_emission_cycle` are guards, not history. |
| **unclear** | `KanbanMove.witness_chain_position` is currently set to `view.current_cycle()` — a structural **placeholder** until the witness-arc column (A3) lands. |
| **contradictory** | none. |

---

### Phase 6 — SPO rung audit

**The 2³ rung is explicit as `CausalMask` (Pearl) bits 40-42 of `CausalEdge64`;
SPO decomposition is explicit as three palette indices + the
`markov_soa::SpoRanks` triple. Partial resolution is supported.**

- SPO indices: `CausalEdge64` bits 0-23 = S/P/O palette indices (each 8-bit).
- 2³ causal rung: `CausalMask` at bits 40-42 (Pearl's 2³ = 8 causal mask
  states). Explicit, not derived. (Confirmed.)
- Vocabulary-agnostic SPO: `markov_soa::SpoRanks { s: u16, p: u16, o: u16 }` —
  three opaque ranks; distance injected as a closure (`Fn(u16,u16)->u8`), so
  language (DeepNSM/COCA) stays upstream and never reaches into the rung.
- Partial resolution: `markov_soa::SoaWavePrimer::project` folds a ±radius
  window into a `WaveProjection` of `SpoRanks` with `best_guess_match` returning
  a fuzzy score — i.e. partial / probabilistic SPO resolution is supported.

**SPO rung table:**

| Property | Status | Evidence |
|---|---|---|
| 2³ rung explicit? | **Yes (Confirmed)** | `CausalMask` bits 40-42, Pearl 2³ |
| Derivable? | partly | direction triad (43-45) derivable from mask context |
| Redundant? | **No** | SPO palette indices and `SpoRanks` serve different layers (edge payload vs Markov projector); not duplication |
| Partial resolution? | **Yes (Confirmed)** | `SoaWavePrimer::project` → `best_guess_match` fuzzy score |

---

### Phase 7 — Little-endian contract audit

**The SoA envelope is zero-copy in-place (creation → Lance tombstone). There is
no baton, no emission, no serialization. The LE contract describes where columns
sit in the in-place backing store, not a transmitted packet. Two fragmentation
risks remain.**

- The SoA is owned in-place by the mailbox. A Lance version IS the backing store
  at cycle N — not a serialized snapshot, not a transmitted packet.
  `CausalEdge64` and `EpisodicEdges64` both expose `to_le_bytes / from_le_bytes`
  for Lance's columnar write path (Lance reads/writes LE bytes from/into the store).
  These are Lance I/O seams, not cross-mailbox serialization. (Confirmed — correct design.)
- p64-bridge (`p64-bridge/src/lib.rs`): `edges_to_layered_rows(&[CausalEdge64])
  -> [[u64;64];8]`, `edge_to_block(&CausalEdge64) -> (usize,usize)`. **This is a
  projection / derivation, not a layout-preserving transport.** It reads SPO +
  mask bits and *computes* palette addresses; it does not round-trip the edge.
  p64 derives a different geometry (palette planes) from the edge — one-way lens,
  not a serializer. (Inferred — acceptable by design.)
- SoA view reinterpret seam: `MailboxSoaView::edges_raw() -> &[u64]` (NOT
  `&[CausalEdge64]`) — the contract crate stays zero-dep on `causal-edge` by
  handing back raw `u64` that callers reconstruct via `CausalEdge64(raw)`. This
  is a **hidden reinterpret**: correctness depends on every caller agreeing on
  the v1-vs-v2 layout. Under the `causal-edge-v2-layout` feature, a caller that
  reconstructs with v1 semantics silently mis-reads bits 46-63
  (the exact `I-LEGACY-API-FEATURE-GATED` hazard).

**Contract map:**

```
   ENVELOPE LE CONTRACT (in-place backing store)
   ├─ CausalEdge64::{to,from}_le_bytes          (causal-edge crate, v1/v2 feature-gated — Lance I/O seam)
   ├─ EpisodicEdges64::{to,from}_le_bytes        (matches CE64 convention)
   │
   FRAGMENTATION RISKS
   ├─ ⚠ TWO CausalEdge64 layouts (causal-edge SPO-palette  vs  thinking-engine 8-channel)
   │      bridged only by layered.rs::to_spo()/from_spo()  — name collision, lossy
   └─ ⚠ soa_view::edges_raw() -> &[u64]  (reinterpret seam; v1/v2 layout agreement is implicit)

   NOTE: CollapseGateEmission / "baton" DO NOT exist in the correct design.
   MailboxSoA::emit() in source is a code artifact to be removed. Every SoA is
   zero-copy from creation to tombstone; there is no cross-mailbox handoff type.
```

| Question | Answer |
|---|---|
| Single canonical LE contract? | **Yes** — CE64/EpisodicEdges64 byte methods are the Lance I/O seam. (Confirmed.) |
| Hidden conversion? | **Yes** — `edges_raw() -> &[u64]` reinterpret; v1/v2 layout agreement is implicit (Contradiction risk). |
| Serialization tax? | **None** — the backing store is in-place. Lance writes LE columns directly. p64 derives palette geometry, does not serialize. |
| Contract fragmentation? | **Yes** — two `CausalEdge64` types is the principal fragmentation. |

#### Phase 7 follow-up — the envelope must know the LE contract, not just the columns (RESOLVED 2026-06-06)

The original Phase 7 finding ("single canonical LE contract: yes for the
column byte methods") was **incomplete**. Column-level LE knowledge is
necessary but not sufficient: `CausalEdge64`, `EpisodicEdges64`, and ndarray's
`MultiLaneColumn` each decode their own bytes correctly, but the **SoA envelope
as a whole** — what a Lance version snapshots and what `simd_soa` sweeps — had
no contract describing how those columns *assemble* into one row-strided packet
with a cycle stamp. The parts knew the LE contract; the envelope did not.

**Resolution (shipped this branch):**

- **Column LE contract = `ndarray::simd::MultiLaneColumn`** (`ndarray/src/simd_soa.rs`).
  Already exists, already LE-correct per column (`iter_f32x16` / `iter_u64x8`
  via `from_le_bytes`), already standalone — any pure-SIMD consumer uses it
  with zero lance coupling. **No change needed.**
- **Envelope LE contract = `lance_graph_contract::soa_envelope::SoaEnvelope`**
  (new module, this commit). Zero-dep, byte-geometry only: `columns()`
  (`ColumnDescriptor[]` — ordering + offset + LE `ColumnKind` + elems/row),
  `row_stride()`, `cycle(): u32`, `LAYOUT_VERSION`, `as_le_bytes()`, plus
  `row_le` / `column_le` zero-copy views and a `verify_layout()` gate (catches
  stride mismatch, column overlap, backing-store size mismatch, and version
  skew at the Lance read boundary — closing the `edges_raw() -> &[u64]`
  implicit-agreement hazard). The envelope describes the in-place backing
  store; nothing is packaged or transmitted.
- **Composition = `lance-graph`** (the one crate that always has both deps):
  carve each envelope column per its descriptor, wrap in `MultiLaneColumn`.

**Why NOT a shared `simd-soa-contract` crate, and why NOT pull ndarray into
the contract:** `lance-graph-contract` is consumed by OGAR classes and ractor
actors precisely because it is zero-dep. ndarray is the heavy HPC foundation
(BLAS L1/L2/L3, MKL/OpenBLAS FFI). Pulling ndarray into the contract would
force that build onto every contract consumer AND force a pure-SIMD ndarray
consumer to transitively pull a graph contract crate. The two-level split
above keeps **both** crates clean: ndarray standalone for SIMD-only consumers,
contract featherweight for class/actor consumers. The levels are complementary
(column = "how to sweep one typed column"; envelope = "where columns sit in the
backing store, what cycle"), never restated, and neither crate depends on the other.

**Iron rule that falls out:** *ndarray owns the column contract; lance-graph
owns the envelope contract; neither restates the other; lance-graph binds them.*

---

### Phase 8 — OGAR inheritance audit

**OGAR is a vocabulary/IR producer. It mints class-identity strings and emits
SPO triples; it owns NO runtime registry, NO reverse resolver, NO thinking
style, and NO class-driven adapter dispatch. The coupling is one trait.**

| Semantic | Source | Status |
|---|---|---|
| Class identity string (`prefix/Class`) | OGAR `ogar_ontology::class_identity` (forward only) | **Confirmed (from OGAR)** |
| Schema (fields/assoc/enums/attributes) | OGAR `Class` IR (once per class) | **Confirmed (from OGAR)** |
| Inheritance edges (`parent`, `mixins`, STI) | OGAR `Class.parent` / `Class.mixins` (representational; **resolution deferred to consumer**) | **Confirmed (representational only)** |
| Class → version (`knowable_from`) | trait `KnowableFromStore` (OGAR declares; **lance-graph implements**) | **Confirmed (seam)** |
| Reverse `classes::from(address)` | — | **Absent** |
| AST adapter selection from class | OGAR `Adapter` trait exists but **no class-driven dispatch**; emitter takes no adapter | **Absent (manual)** |
| Thinking style from class | docs-only refs; not on `Class`; stored on lance-graph `MappingRow` but **unused at dispatch** | **Absent (inheritance), stored-but-dead (lance side)** |
| Pragmatics (energy, plasticity, qualia, cycle) | **local** to `MailboxSoA` | **Confirmed (local)** |

**Inheritance map:**

```
   OGAR (producer, Lance-free)                         lance-graph (consumer/runtime)
   ───────────────────────────                         ──────────────────────────────
   class_identity(prefix,name) ──► "ogit-op/WorkPackage" ──► NiblePath router / OntologyRegistry key
   Class { parent, mixins, attributes, ... } ──emit──► SPO Triples ──► triple loader
   trait KnowableFromStore  ◄──implemented by──         OntologyRegistry / ClassRegistryWriter
        register(id, ddl_hint)->u64                     enumerate_first_with_entity_type_id (scan)
        knowable_from(id)->u64                           MappingRow.thinking_style (STORED, UNUSED)
                                                         MulThresholdProfile::for_context (the ONLY
                                                            thing entity_type actually drives today)
   AST adapter:  Adapter::map (static identity xlate)   ── NO class→adapter dispatch wired ──
   Pragmatics:                                          MailboxSoA.{energy,plasticity,qualia,meta}  (LOCAL)
```

---

## 8. Redundancy list

1. **Two SoA envelopes** — `BindSpace` (global, 65 KB/row `Vsa16kF32` cycle plane)
   vs `MailboxSoA` (per-mailbox). The driver holds both. (highest priority)
2. **Two `CausalEdge64` types** — `causal-edge` (SPO-palette) vs
   `thinking-engine::layered` (8-channel). Same name, different geometry.
3. **Deprecated `Vsa16kF32` cycle plane** still allocated in `BindSpace`
   (65 KB/row) despite being scoped out as a cross-mailbox carrier.
4. **`QualiaColumn` (f32×18, deprecated)** co-exists with `QualiaI4Column`
   (canonical, 8 B/row).
5. **`MappingRow.thinking_style`** is stored per class but never read on the
   `entity_type` dispatch path — dead field.
6. **`entity_type` resolution** is a linear scan despite a "1-based index" doc
   contract — the index model is documented but not implemented.

## 9. Circular ownership risks

- **None in the Rust ownership sense.** All cross-structure links are `Copy`
  handles (`EdgeRef`, W-slot index, `mailbox_ref`, `spo_fact_ref`), not `Rc`/`Arc`
  cycles. The mailbox is the single owner of its truth columns.
- **Dependency-cycle avoidance is correct:** planner ↔ shader-driver cycle is
  broken by the closure seam in `convergence.rs::run_convergence(..., impl FnOnce(...))`;
  AriGraph→planner cycle is broken via the p64 convergence point. (Confirmed.)
- **Latent hazard, not a cycle:** the v2 W-slot reference lives *inside* the
  `CausalEdge64` payload word, coupling payload and reference layers. If the
  witness table is rebuilt out of step with the edges, the 6-bit index resolves
  to a wrong (but valid) entry — silent, not a panic.

## 10. Recommended minimal corrections

Ordered by leverage, each is a *classification-respecting* minimal change — no
theory-driven rename, no refactor beyond making ownership/reference/version/
execution geometry explicit.

1. **Pick one envelope. Declare it.** Mark `BindSpace` `#[deprecated]` with a
   migration pointer to `MailboxSoA`, or invert: state in one doc-comment which
   is canonical and gate the other behind a `legacy-bindspace` feature. Today
   both compile and the driver holds both — the reader cannot tell which is the
   particle. (Resolves redundancy 1, 3.)
2. **Rename the local edge.** `thinking_engine::layered::CausalEdge64` →
   `ChannelEdge64` (or similar). Keep `to_spo()/from_spo()`. The name collision
   is the single biggest correctness trap in the LE contract. (Resolves
   redundancy 2, contract fragmentation.)
3. **Make `entity_type` resolution match its contract.** Either (a) change the
   `OntologyRegistry` to a real O(1) `Vec` index keyed by the 1-based
   `entity_type` (matching the doc), or (b) fix the doc to say "scan by
   `entity_type_id`." Pick one; don't ship both stories. (Resolves redundancy 6.)
4. **Decide the W-slot home.** If references must be explicit (intended model
   #7), the v2 W-slot is a reference packed into payload. Either document it as a
   deliberate exception with a version gate (it already has one), or move the
   witness reference out of `CausalEdge64` into the (queued) `EpisodicWitness64`
   SoA column. Land that column — `soa_view.rs` already reserves the slot.
   (Resolves Phase 3/4 contradiction + Phase 4 Absent.)
5. **Wire or delete the dead inheritance.** `MappingRow.thinking_style` is stored
   per class but unused at dispatch. Either consume it in `driver.rs:331-339`
   (true class→style inheritance, matching intended model #9/#4) or remove the
   field. (Resolves redundancy 5; closes the "inheritance from OGAR class"
   intended-vs-actual gap.)
6. **Type the `edges_raw()` seam.** The `&[u64]` reinterpret in `soa_view.rs`
   relies on implicit v1/v2 agreement. Add a `const EDGE_LAYOUT_VERSION: u8`
   to the view trait and assert it at reconstruction, per
   `I-LEGACY-API-FEATURE-GATED`. (Resolves Phase 7 hidden-conversion risk.)
7. **Retire the deprecated qualia/cycle columns** once (1) lands —
   `QualiaColumn` f32×18 and the `Vsa16kF32` cycle plane are pure footprint
   (65.5 KB/row) on the legacy envelope. (Resolves redundancy 3, 4.)
9. **Remove `MailboxSoA::emit()` and `CollapseGateEmission`.** The zero-copy
   model (creation → Lance tombstone, no emission, no baton) means `emit()` and
   the `CollapseGateEmission(u16 target, CausalEdge64)` type are code artifacts
   from a superseded design. The KanbanColumn/`VersionScheduler`/ractor
   orchestration is the only secondary path; there is no inter-mailbox handoff
   type. Remove `emit()` from `MailboxSoA`, remove or gate `CollapseGateEmission`
   behind `#[deprecated]`, and update `ShaderDriver` accordingly.
10. **Demote the `ndarray-hpc` fallback wording in CLAUDE.md.** In practice
   **no shipped consumer uses lance-graph without ndarray** — every consumer
   uses both. The `ndarray-hpc` feature / `blasgraph/ndarray_bridge.rs`
   fallback is a **CI-compile-check only** path, not a real deployment mode.
   CLAUDE.md currently presents it as a parallel architecture ("Fallback
   without ndarray ... for minimal builds / downstream consumers who don't need
   HPC"), which is exactly what makes sessions re-derive the wrong dependency
   boundary (e.g. "keep ndarray ignorant of the envelope to preserve the
   fallback"). Reword to: *ndarray is mandatory for every shipped consumer; the
   no-ndarray path is a CI compile gate, nothing ships on it.* This is the
   load-bearing correction behind recommendation-set context for the SIMD_SOA
   contract split (Phase 7 follow-up).

---

## Geometry verdict — per major field

| Field / structure | Verdict | Rationale |
|---|---|---|
| `MailboxSoA.mailbox_id` | **KEEP** | Envelope identity, correct. |
| `MailboxSoA.entity_type` | **KEEP (fix resolver)** | Correct as class pointer; resolver should match its O(1) doc. |
| `MailboxSoA.energy / plasticity / last_emission_cycle` | **KEEP (rename)** | Local pragmatics, correctly owned. `last_emission_cycle` is a same-cycle idempotency guard; rename to `last_active_cycle` to remove the emission framing. |
| `MailboxSoA.edges (CausalEdge64)` | **KEEP** | Payload, correctly owned by the mailbox. |
| `MailboxSoA.qualia (QualiaI4_16D)` | **KEEP** | Canonical local pragmatic vector. |
| `MailboxSoA.meta (MetaWord)` | **KEEP** | Thinking-style/awareness bits belong here, not on the edge. |
| `CausalEdge64` S/P/O/freq/conf/mask | **KEEP** | Authoritative payload. |
| `CausalEdge64` v2 W-slot | **REFERENCE ONLY** | Move to `EpisodicWitness64` column or gate as explicit exception. |
| `CausalEdge64` truth-band / inference enum | **DERIVE** | Derivable from confidence / mantissa. |
| `thinking_engine::layered::CausalEdge64` | **KEEP + RENAME** | Distinct concept; collision is the hazard. |
| `BindSpace` (whole) | **MOVE TO LANCE / DEPRECATE** | Legacy global SoA; persist via Lance, drain into `MailboxSoA`. |
| `BindSpace.cycle (Vsa16kF32)` | **REMOVE** | Deprecated carrier; 65 KB/row footprint. |
| `QualiaColumn (f32×18)` | **REMOVE** | Superseded by `QualiaI4Column`. |
| `entity_type → OntologyRegistry` schema | **MOVE TO OGAR** | Schema is OGAR's `Class` IR; lance should hold only the registry/version impl of `KnowableFromStore`. |
| `MappingRow.thinking_style` | **UNCLEAR** | Wire it (inheritance) or remove it (dead). Decide. |
| `DatasetVersion` + `VersionScheduler` | **KEEP** | Clean self-through-time; read-only propose vs single-owner dispose. |
| `KanbanMove.witness_chain_position` | **UNCLEAR** | Placeholder (= current_cycle) until A3 witness-arc lands. |
| `EpisodicEdges64 / EdgeRef` | **KEEP (REFERENCE ONLY)** | Correct explicit pointers. |
| `WitnessTable / WitnessEntry` | **KEEP (REFERENCE ONLY)** | Wide `mailbox_ref` is rotation-safe by design. |
| `EpisodicWitness64` SoA column | **LAND IT** | Currently Absent; reserved in `soa_view.rs`. |
| `p64-bridge` projections | **KEEP (DERIVE)** | One-way lens CE64→palette; not a transport, document as such. |

---

## Bottom line

**Does the struct geometry measure what it claims?**

- **Payload, references, versioning, zero-copy lifecycle: YES.** `CausalEdge64`
  is a clean payload, references are explicit `Copy` handles with no ownership
  cycles, Lance versioning gives self-through-time without row-level history
  duplication, the read-only-scheduler / single-owner-mutator split is correct,
  and the SoA envelope is zero-copy in-place from creation to Lance tombstone
  (no emission, no baton, no serialization).
- **Identity, inheritance, single-envelope: NOT YET.** The intended
  `OGAR::classes::from(address)` does not exist (forward-only in OGAR; a linear
  scan in lance-graph); two SoA envelopes co-exist with a live deprecated f32
  carrier; two `CausalEdge64` types share a name; and class→{thinking-style,
  AST adapter} inheritance is either stored-but-dead or absent.

The geometry is **load-bearing where it concerns one mailbox's owned payload and
its versioned lifecycle**, and **under-specified where it concerns identity
resolution and OGAR class inheritance**. The seven minimal corrections above
close the gap without renaming on theory — they make ownership, reference,
versioning, and execution geometry explicit, which is exactly the stated goal.
