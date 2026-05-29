# bindspace-singleton-to-mailbox-soa-v1 — dissolve the shared `Arc<BindSpace>` into per-mailbox `MailboxSoA<N>` thoughtspace

> **Status:** CONJECTURE / design (migration spec). NOT yet implemented.
> **Date:** 2026-05-27.
> **Owns the answer to:** *"make MailboxSoA the individual, mailbox-owned, ephemeral
> 'thoughtspace' carrier — the BindSpace surrogate — and document where + how the
> BindSpace singleton migrates onto it."*
> **Anchored to (FINDING-grade):** `E-BATON-1` (no persisted singleton BindSpace; Baton =
> LE `(u16, CausalEdge64)`), `E-CE64-MB-4` (mailbox-as-owner ⇒ Rust move/ownership proves
> no-alias / no-race / no-UAF at compile time; UB = compile error), `E-LADDER-SERVES-MAILBOX`
> (mailbox is the owned unit; hot ephemeral → cold SPO + Lance tombstone-witness),
> `I-VSA-IDENTITIES` (bundle identities, not content), the "restore the contract, never port
> the carrier" iron rule (deprecated `Vsa16kF32`).
> **Subsumed under:** `causaledge64-mailbox-rename-soa-v1` (§5 `MailboxSoA<N>`, §1 ownership,
> §0 zones). This doc is the *column-level migration map* that plan deferred to impl.

---

## 0. The correction this doc encodes

The request was first phrased as *"MailboxSoA has an individual **copy** of the BindSpace."*
That is **not** the model. A per-mailbox *copy* of a global singleton is still a singleton
(N copies of one shared shape, kept in sync = the aliasing problem `E-CE64-MB-4` exists to
kill). The correct model:

> **`MailboxSoA<N>` *becomes* the BindSpace.** There is no global address space that a
> mailbox copies from. Each mailbox **owns its own ephemeral "thoughtspace"** — the per-row
> SoA columns are *born in the mailbox, live only as long as the mailbox, and die with it*
> (sea-star spawn → think → emit → die → tombstone). The singleton `Arc<BindSpace>` is
> **dissolved**, not sharded.

`Vsa16kF32` is deprecated as a carrier (`E-BATON-1`); cumulative state lives in
**CausalEdge64 emissions + AriGraph SPO-G quads + the per-mailbox SoA columns**. The mailbox
*is* the BindSpace surrogate for the duration of one think-arc.

---

## 1. Current state — the singleton inventory (what migrates)

The address space exists today as **one shared `Arc<BindSpace>`** in
`crates/cognitive-shader-driver`:

| Site | What it is |
|---|---|
| `src/driver.rs:55` `pub struct ShaderDriver` | the holder |
| `src/driver.rs:56` `pub(crate) bindspace: Arc<BindSpace>` | the singleton handle |
| `src/driver.rs:80` (ctor arg) / `:594` (builder field) / `:612` (builder setter) | how it's injected |
| `src/driver.rs:116` `fn bindspace(&self) -> &BindSpace` | read accessor |
| `src/bin/serve.rs:29` `Arc::new(BindSpace::zeros(4096))` | **the one instance** — 4096 rows, shared across all thinking |
| `src/engine_bridge.rs` (`bind_busdto`/`unbind_busdto`/`read_qualia_*`/`write_qualia_*`, `&mut BindSpace`) | the per-row read/write surface |
| `src/bindspace.rs:234` `pub struct BindSpace` / `:280` `BindSpace::zeros` | the type + allocator |

`BindSpace` is "read-only universal address space; mutations go through `CollapseGate`"
(`bindspace.rs:225`). That is the singleton contract: **one space, all rows, shared.**

### `BindSpace` columns today (per row), from `bindspace.rs`

| Column | Type / width | Bytes/row | Role |
|---|---|---|---|
| `fingerprints.content` | `[u64; 256]` | 2 048 | identity fp — "what" (topic) |
| `fingerprints.topic` | `[u64; 256]` | 2 048 | identity fp — topic plane |
| `fingerprints.angle` | `[u64; 256]` | 2 048 | identity fp — angle plane |
| `fingerprints.cycle` | `f32 × 16 384` (`FLOATS_PER_VSA`) | **65 536** | **`Vsa16kF32` carrier — DEPRECATED** |
| `edges` | `EdgeColumn` (`CausalEdge64`) | 8 | the LE contract / baton edge |
| `qualia` | `QualiaI4Column` (i4-16D) | 8 | affective role (already LE i4) |
| `meta` | `MetaColumn` (`MetaWord` u32) | 4 | thinking·awareness·nars_f·nars_c·free_e |
| `temporal` | `u64` | 8 | temporal stamp |
| `expert` | `u16` | 2 | expert id |
| `entity_type` | `u16` (column H) | 2 | OGIT type binding (1-based into ontology) |
| `ontology` | `Option<Arc<OntologyRegistry>>` | (shared) | **read-only cold ontology** |

The `cycle` plane alone is **64 KB/row** — at 4096 rows that's 256 MB of `Vsa16kF32` the
singleton carries. This is exactly the "empty cathedral" carrier the deprecation kills.

---

## 2. Target — `MailboxSoA<N>` as the LE-contract thoughtspace

`MailboxSoA<N>` today (`src/mailbox_soa.rs:32`) is only a **spatial-temporal accumulator**:
`energy: [f32; N]`, `plasticity_counter: [u8; N]`, `last_emission_cycle: [u32; N]`,
`current_cycle`, `w_slot`, `threshold`, `mailbox_id`. It has no thoughtspace columns.

**Target shape** — the same per-row SoA, *owned by the mailbox*, in little-endian contract
types (i4 / u8 / u16 / u32 / u64 + `CausalEdge64`), **minus the `Vsa16kF32` plane**:

```rust
pub struct MailboxSoA<const N: usize> {
    // ── identity / ownership (unchanged) ──
    pub mailbox_id: MailboxId,
    pub w_slot: u8,
    pub current_cycle: u32,
    pub threshold: f32,

    // ── accumulator (already present, D-CSV-7) ──
    pub energy: [f32; N],
    pub plasticity_counter: [u8; N],
    pub last_emission_cycle: [u32; N],

    // ── NEW: the migrated thoughtspace columns (per-mailbox owned) ──
    pub edges: [CausalEdge64; N],   // ← BindSpace.edges  (8 B/row — the LE baton edge)
    pub qualia: [QualiaI4_16D; N],  // ← BindSpace.qualia (8 B/row — already i4)
    pub meta:   [MetaWord; N],      // ← BindSpace.meta   (u32/row)
    pub entity_type: [u16; N],      // ← BindSpace.entity_type (references shared ontology)
    // temporal/expert: fold into CausalEdge64 where the v2 layout already carries them,
    //   else keep [u64;N]/[u16;N] — see §3 note + OQ-2.

    // ── content identity: NOT the 64 KB plane — a reference, see §3 / OQ-1 ──
    // pub content_ref: [ContentId; N],   // CAM-PQ code / codebook id (≤ 6 B/row)
}
```

Shared-immutable state (the **ontology registry**) does **not** migrate into the mailbox —
it stays a shared `Arc<OntologyRegistry>` handed to mailboxes by reference (it is calcified
cold Zone-2, not ephemeral thinking state). See §4.

`DefaultMailboxSoA = MailboxSoA<1024>` (already 4× the old 4096-singleton's *effective* hot
working set per mailbox; many small mailboxes replace one giant space).

---

## 2.5 — THE little-endian contract: ONE SoA, end-to-end, no boundary re-encode

**Ruling (2026-05-27):** the per-mailbox `MailboxSoA<N>` layout *is* **THE little-endian
contract** — singular, canonical. The **same SoA byte layout runs the whole vertical**, from
the cognitive-shader-driver hot path down to lance-graph storage, with **no re-encode /
translation at any boundary**:

```
cognitive-shader-driver        lance-graph-contract              lance-graph storage
  MailboxSoA<N>  ───────────►  LE byte contract types  ────────►  Lance columns /
  (hot, owned, ephemeral)      CausalEdge64 · QualiaI4_16D ·       SPO-G + tombstone-witness
                               MetaWord · SoaColumns<N> ·          (cold, durable)
                               entity_type u16
        └──────────────── ONE little-endian SoA the whole way ──────────────┘
                  persisted_row: Option<u32>  is the link, not a re-encode
```

Consequences:

- **No Arrow/JSON translation membrane for the thoughtspace.** The columns the mailbox owns
  in RAM are the columns Lance stores. A persisted row (`ShaderCrystal.persisted_row`,
  `lance-graph-contract/src/cognitive_shader.rs:382`) is a **pointer to the same SoA row**
  laid down in Lance, not a serialized copy in a different shape. (Contrast the *ontology*
  path, which legitimately translates `MappingRow ↔ RecordBatch` in `lance_cache.rs` — see §4
  for why the ontology is a different animal.)
- **The LE byte contract types are the shared vocabulary** (`lance-graph-contract`):
  `CausalEdge64`, `QualiaI4_16D`, `MetaWord`, the `SoaColumns<N>` floor (ndarray), `entity_type`
  u16. Every layer compiles against these exact types — the contract is a *compile-time*
  handshake (cf. `E-CONTRACT-NO-SERIALIZE`), and the *bytes* are identical from compute to disk.
- **Lance is append-only/versioned**, so persisting the SoA row IS the tombstone-witness +
  GoBD audit trail by construction (`E-LADDER-SERVES-MAILBOX` §6) — no separate logging layer.
- This is the "restore the contract, never port the carrier" rule made vertical: one i4/u8/
  u64 SoA from shader to storage; never re-inflate to `Vsa16kF32` at any tier.

This widens the migration scope from "cognitive-shader-driver only" to **the full vertical**:
shader-driver SoA → contract LE types → lance-graph storage must all be THAT one SoA.

---

## 2.6 — DTO vertical audit: StreamDto / ResonanceDto / BusDto / p64 (what re-encodes vs what already conforms)

The flow DTOs live in `crates/thinking-engine/src/dto.rs` (a workspace member) and stage the
cycle Φ→Ψ→B→Γ. Audited against the "one SoA, no re-encode" ruling (§2.5), **two patterns
coexist** and the migration must collapse the legacy one onto the SoA:

### The GOOD pattern (already conforms — the reference) — `p64-bridge`

`crates/p64-bridge/src/lib.rs` maps the **canonical LE-contract types directly** to p64
palette storage with **no re-encode and no `p64` dependency** (compile-time bridge, joined at
the call site): `CausalEdge64 → edge_to_block` (palette row/col), `ThinkingStyle → layer_mask
+ combine + contra`, `HdrSemiring → combine/contra mode`. This **is** THE SoA reaching
storage — the i4/u8/u64 contract types address the palette directly. *Everything from the
shader SoA to lance-graph storage should look like p64-bridge.* Keep as-is; it is the
storage-ward end of the vertical.

### The LEGACY pattern (re-encodes — collapse it) — `thinking-engine` DTOs ↔ `engine_bridge`

These are heap `Vec`-based representations bridged into `BindSpace` rows by a **translation
membrane** in `crates/cognitive-shader-driver/src/engine_bridge.rs`
(`bind_busdto` / `unbind_busdto:310` / `busdto_to_binary16k:199`). That bind/unbind seam is
exactly the re-encode the ruling forbids for the thoughtspace. Fates:

| DTO (`thinking-engine`) | Shape today | Fate under THE SoA |
|---|---|---|
| `StreamDto` (Φ ingress) | `source`, `codebook_indices: Vec<u16>`, `timestamp` | **Thin ingress adapter only** — a sensor-membrane boundary (like the ontology); codebook indices (identity refs, `I-VSA-IDENTITIES`-clean) hand straight into the mailbox SoA. Does NOT carry a parallel `Vec` through the hot path. |
| `ResonanceDto` (Ψ, `dto.rs`) | `energy: Vec<f32>` (the 4096 ripple field) + `top_k`/`cycle_count`/`converged` | **Already IS `MailboxSoA.energy: [f32; N]`.** The ripple field is the mailbox's energy column — unify, don't keep a separate heap `Vec` DTO. `top_k`/`converged` are derived reads. |
| `ResonanceDto` (`awareness_dto.rs`) | richer multi-perspective: `hdr: HdrResonance`, `dominant_perspective`, `gate`, `dissonance`, `total_energy`, inferred user state | **NAME-COLLISION (two `ResonanceDto`) — dedup.** Scalars (`dissonance`/`total_energy`/`gate`) → SoA `meta`/`edge` columns; `HdrResonance` = the S/P/O 3-perspective read over the SoA. Rename or merge (tech-debt entry filed). |
| `BusDto` (B consequence) | `codebook_index: u16`, `energy: f32`, `top_k`, `cycle_count`, `converged` | **Becomes a view/projection over the SoA row** (read `edges`/`qualia`/`meta` + `persisted_row`), not a bound/unbound separate struct. `unbind_busdto` collapses to a column read; `bind_busdto`/`busdto_to_binary16k` collapse (the SoA row's content-ref IS the binary16k identity). |
| `ThoughtStruct` (Γ collapse) | `bus`, lazy `text`, `sensor_contributions`, `tension_history: Vec<Vec<f32>>`, `style_trajectory` | **Γ projection of the persisted SoA row** (text stays lazy). `tension_history`/`style_trajectory` become witness columns / tombstone fields (`E-LADDER-SERVES-MAILBOX` §6), not parallel `Vec`s. |

**Ruling:** `engine_bridge`'s `bind_busdto`/`unbind_busdto`/`busdto_to_binary16k` are the
re-encode boundary to dissolve (migration step S2). The thinking-engine DTOs survive only as
(a) the `StreamDto` ingress adapter at the sensor membrane and (b) thin *read projections*
(`BusDto`/`ThoughtStruct`) over the mailbox SoA — never as a parallel owned representation the
hot path translates to/from. p64-bridge is the conformance template for the storage-ward half.

---

## 2.7 — Hot/cold: `ThoughtStruct` is a transparent view over the SurrealDB container table(s)

The same "one SoA, no re-encode" rule (§2.5) extends **past RAM into persistence**. The hot
path can hold **only ~64k–256k thoughts**; beyond that, thoughts live in **LanceDB — the
*leading* cold storage** (append-only/versioned, source of truth). **SurrealDB is a *view*
over LanceDB, NOT a separate store** (`crates/surreal_container` = SurrealDB-on-`kv-lance` =
a query/kanban surface projecting the LanceDB rows; LanceDB stays leading). The ruling:

> `ThoughtStruct` (the Γ-collapse thought) is **later also a transparent view into the
> ThoughtStruct container table(s)** — reading a thought resolves over the **same SoA layout**
> whether it is hot (in a mailbox) or cold (in the SurrealDB container). No re-encode at the
> RAM↔storage boundary: the container columns are the mailbox columns. `persisted_row` is the
> seam; the view spans hot+cold transparently.

### Capacity (the working-set bound)

- **Hot ceiling: ~64k–256k thoughts.** At **64k ≈ 300–600 MB RAM** ⇒ **~4.5–9 KB/thought** hot.
  256k ⇒ ~1.2–2.4 GB.
- **Why ~6 KB/thought (not the ~24–30 B of bare SoA columns):** the dominant cost is the
  **content/topic/angle Hamming identity planes** = 3 × 256 × 8 B = **6 KB/thought**. The bare
  migrated columns (edge 8 + qualia 8 + meta 4 + entity_type 2 + accumulator ~9 B) add only
  ~30–50 B. **6 KB × 64k ≈ 400 MB** — squarely in the stated 300–600 MB range. (This is *with*
  the 64 KB `Vsa16kF32` `cycle` plane dropped; keeping it would make 64k thoughts ≈ 4.7 GB, so
  dropping it is what makes the 64k–256k hot working set fit at all.)
- **Resolves OQ-1 (content reference shape):** the 300–600 MB-for-64k budget implies the
  **Hamming identity planes stay hot per thought** (≈ 6 KB) — they are *not* reduced to a 6 B
  CAM-PQ ref in the hot path. The only thing dropped is the 64 KB f32 `cycle` carrier. A
  CAM-PQ/codebook ref is the *cold/storage* form (and the spill key), not the hot form.

### The spill / transparent-read model

```
        hot (mailbox SoA, ≤ 64k–256k thoughts, ~300–600 MB)
            │  energy decays / mailbox dies / working set full
            ▼  spill — SAME SoA columns, no re-encode (persisted_row)
        cold = LanceDB  (LEADING storage: append-only/versioned, source of truth)
            ├──────────────► SurrealDB  (a VIEW over LanceDB — the Rubicon kanban)
            ▲
            │  ThoughtStruct view reads hot mailbox OR cold LanceDB transparently (same layout)
```

- **LanceDB is the leading store.** The cold rows are the **same SoA columns** persisted
  (edge/qualia/meta/entity_type + content identity), append-only/versioned — so LanceDB *is*
  the tombstone-witness + GoBD trail (`E-LADDER-SERVES-MAILBOX` §6) by construction.
- **SurrealDB is a view, not a store.** `surreal_container` (SurrealDB-on-`kv-lance`) projects
  the LanceDB rows into query surfaces — notably the **Rubicon-model kanban** (action-phase
  board; see the `E-RUBICON-RACTOR` epiphany). It never owns the data; LanceDB does.
- `ThoughtStruct` (`thinking-engine`) is the **transparent view** over `(hot mailbox row |
  cold LanceDB row)`; text stays lazy. The §2.6 ruling holds at the storage tier:
  `ThoughtStruct` is a read-projection, identical whether the row is in RAM or LanceDB; the
  SurrealDB kanban is a *second* view over the same LanceDB rows, not a re-encode.
- **Gating:** the cold tier is LanceDB (already the workspace storage); the SurrealDB *view*
  needs `surreal_container` unblocked (BLOCKED(A/B/C/D) — fork dep + Lance 6 pin) but is
  **optional** (a kanban surface, not the source of truth). Transparent-view wiring is
  migration step **S4** (plan §6) and deliverable **D-MBX-6**.

---

## 3. Column-by-column migration map

| `BindSpace` column | → Destination | How |
|---|---|---|
| `fingerprints.cycle` (`Vsa16kF32`, 64 KB) | **DROP** | The bundle is *ephemeral local compute* inside one `Think`, never a stored column (`E-BATON-1`, "The Click" Baton-scoping note). Compute it transiently if a step needs it; never allocate it per row. |
| `fingerprints.content/topic/angle` (3×2 KB Hamming planes) | **Reference, not own** (OQ-1) | Per `I-VSA-IDENTITIES` (bundle *identities*, not content): the mailbox holds a small **content reference** (CAM-PQ code / codebook id, ≤ 6 B), resolving to the dense plane in the shared cold codebook only when resonance is actually needed. Dense per-row planes do **not** belong in the hot mailbox. |
| `edges` (`CausalEdge64`) | **Own** `[CausalEdge64; N]` | This *is* the LE contract / baton edge. It moves into the mailbox unchanged — it is the cumulative-state home (`E-BATON-1`). |
| `qualia` (`QualiaI4_16D`) | **Own** `[QualiaI4_16D; N]` | Already i4-16D LE; straight move. |
| `meta` (`MetaWord` u32) | **Own** `[MetaWord; N]` | Straight move (u32 packed). |
| `temporal` (u64) | **Fold or own** (OQ-2) | The v2 `CausalEdge64` layout reclaimed the old temporal bits (`I-LEGACY-API-FEATURE-GATED` items 1/3). Prefer folding temporal into the edge / the mailbox's `current_cycle`; keep a `[u64; N]` column only if a separate stamp is still required. |
| `expert` (u16) | **Subsume** | The mailbox already *is* an expert/corpus (`mailbox_id` + `w_slot`). Per-row `expert` collapses to the mailbox identity in most cases; keep a column only for multi-expert rows. |
| `entity_type` (u16) | **Own** `[u16; N]` | 1-based index into the **shared** ontology; the index is per-row mailbox state, the table it indexes is shared (next row). |
| `ontology` (`Arc<OntologyRegistry>`) | **Stays shared** | Read-only cold Zone-2; handed by `Arc` to the mailbox, never owned/copied. See §4. |

**Net footprint (two figures — don't conflate):**
- *Bare migrated SoA columns* ≈ **24–50 B/row** (edge 8 + qualia 8 + meta 4 + entity_type 2 +
  accumulator energy/plasticity/last_emit ~21).
- *Full hot thought* ≈ **~6 KB** — because the **content/topic/angle Hamming identity planes
  (3 × 2 KB) stay hot** (see §2.7; this is what sets the 64k ≈ 300–600 MB budget). The win is
  dropping the **64 KB `Vsa16kF32` `cycle` plane**: per-thought ~71.6 KB → ~6 KB, which is what
  makes the 64k–256k hot working set fit. The bare columns are L1/L2-resident; the Hamming
  planes are the bulk and bound the working-set ceiling.

---

## 4. What STAYS shared (does not migrate) — the Ontology is lazylock-via-cache, AS IS

The **Ontology is explicitly NOT part of THE SoA contract** and does **not** migrate. It stays
exactly as it is today — a lazily-initialised, cache-backed, read-only shared resource:

- **`LazyLock<NamespaceRegistry>`** — the seed registry is `static SEED_NAMESPACE_REGISTRY:
  LazyLock<NamespaceRegistry>` (`crates/lance-graph-ontology/src/registry.rs:39`), built once,
  read-only thereafter. Mailboxes consult it by reference; never own or copy it.
- **The `ontology_dictionary` Lance cache** (`lance-graph-ontology/src/lance_cache.rs`,
  feature `lance-cache`) — a **CACHE of hydrated TTL** keyed by `ttl_root_checksum`; the TTL
  files on disk are source-of-truth; on version mismatch it **drop-and-rebuilds** (no
  migration ladder). Its own header already states the boundary: *"BindSpace
  (FingerprintColumns / QualiaColumn / MetaColumn / EdgeColumn) is the live runtime SoA and is
  unrelated — it never lands here."* That invariant holds unchanged under this migration.
- **Why it's a different animal (and why it legitimately re-encodes):** the ontology cache
  *does* translate `MappingRow ↔ RecordBatch` — but it is calcified, immutable-at-read,
  shared cold knowledge, not ephemeral per-think thoughtspace. The "one SoA, no re-encode"
  rule (§2.5) governs the **thoughtspace** vertical; the ontology cache is an orthogonal
  TTL-projection cache and is left **as is**.

So: a mailbox holds the ontology as `Arc<OntologyRegistry>` / `&OntologyRegistry` (lazylock +
cache, read-only); the per-row `entity_type: u16` (1-based index into the shared ontology)
travels in THE SoA, but the ontology tables it indexes stay shared and cached. **No ontology
work in this migration — it is `as is`.**

Other shared-immutable state that likewise stays out of the SoA: CAM-PQ codebooks /
centroids (the cleanup codebook content references resolve against, `I-VSA-IDENTITIES`
Test 3); the AriGraph SPO-G cold store. Rule of thumb: **ephemeral per-think state → into the
mailbox SoA; calcified/immutable shared knowledge → stays shared (lazylock/cache), out of the
SoA.**

---

## 5. Lifecycle — the mailbox owns its thoughtspace cradle-to-grave

```
spawn      MailboxSoA<N>::new(mailbox_id, w_slot, threshold)   ← owns N empty rows
  │                                                              (no global space touched)
think      apply_edges(batons) → energy integrates              ← receives batons (moved in)
  │        rows read shared ontology/codebook by &ref
emit       emit(source) → CollapseGateEmission (batons out)     ← hands ownership of edges out
  │        (the ONLY cross-boundary state = the LE baton)
die        mailbox dropped                                       ← Rust drop = thoughtspace freed
  │                                                              (E-CE64-MB-4: no alias survives)
witness    calcify stable rows → SPO-G quad + Lance tombstone   ← witness outlives the mailbox
```

Because the mailbox *moves* batons in and *moves* emissions out (owned `(u16, CausalEdge64)`),
the borrow checker proves there is no shared mutable aliasing of the thoughtspace — the exact
guarantee a shared `Arc<BindSpace>` + `CollapseGate` had to enforce by *convention*
("read-only; mutate only through the gate"). Ownership makes the convention a **compile
error** when violated (`E-CE64-MB-4`).

---

## 6. Migration staging (gated order)

| Step | Change | Gate / blocker |
|---|---|---|
| **S0** | This doc + board entries (design ratified) | — (done here) |
| **S1** | Add the migrated columns to `MailboxSoA<N>` (`edges`/`qualia`/`meta`/`entity_type`) behind a `mailbox-thoughtspace` feature; keep `Arc<BindSpace>` alive in parallel | `D-CE64-MB-1-impl` (par-tile `Mailbox<T>` apex) lands first; needs `PR-NDARRAY-MIRI-COMPLETE` (U16x32/U32x16/U64x8 SIMD method gaps) |
| **S2** | Move the `engine_bridge` per-row read/write surface (`bind_busdto`/`unbind_busdto`/`read/write_qualia`) onto `MailboxSoA` rows; the `cycle` Vsa16kF32 plane becomes a transient local, not a column | S1; OQ-1 (content reference shape) ratified |
| **S3** | `ShaderDriver` holds a **set of mailboxes** (sea-star) instead of one `Arc<BindSpace>`; `serve.rs:29` stops allocating the 4096-row singleton | S2; OQ-2 (temporal/expert fold) ratified |
| **S4** | Wire death → SPO-G + Lance tombstone-witness (`E-LADDER-SERVES-MAILBOX` §6); link-integrity back-pointer | S3 + Zone-2 persistence (`surreal_container` unblock OR `lance-graph-callcenter` path) |
| **S5** | Delete `BindSpace` singleton + the `cycle` plane; remove the feature gate | S4 green; CLAUDE.md "The Click" updated off `Vsa16kF32` (see §8) |

**Backwards-compat discipline (`I-LEGACY-API-FEATURE-GATED`):** S1–S4 run with both paths
live behind a feature; the v1 `Arc<BindSpace>` accessors must route to the mailbox mapping or
be feature-gated to a documented no-op with a migration pointer — never silently change
semantics. Field-isolation matrix tests are mandatory when the `CausalEdge64` temporal/expert
fold (OQ-2) reclaims bits.

---

## 7. Invariants the migration must preserve

1. **`E-CE64-MB-4`** — ownership = compile-time safety. The mailbox must *own* (not borrow
   `&mut` from a shared space) its columns; batons move in/out. No `Arc<Mutex<…>>` over the
   thoughtspace.
2. **`E-BATON-1`** — no persisted singleton; the only cross-boundary state is the LE baton
   `(u16, CausalEdge64)` (`CollapseGateEmission`). Wire cost stays `13 + 10·baton_count`.
3. **No double-mailbox** (`E-LADDER-SERVES-MAILBOX` §1) — the inner think stays sync; the
   async fan-out/respawn is the ractor outer-swarm (`D-PERSONA-5`), not a second queue inside
   `MailboxSoA`.
4. **`I-VSA-IDENTITIES`** — bundle identities, not content; the dense planes do not get copied
   per-mailbox (that would be N× the carrier we are deleting).
5. **`I-SUBSTRATE-MARKOV`** — the bundle math is untouched; only the *carrier/ownership* is
   re-homed. The transient cycle bundle still obeys Chapman-Kolmogorov when a step computes it.

---

## 8. Open questions (ratify before the gated step that needs them)

- **OQ-1 (S2): RESOLVED by §2.7 capacity.** The 64k ≈ 300–600 MB budget implies the
  content/topic/angle **Hamming identity planes stay hot (~6 KB/thought)**; only the 64 KB
  `Vsa16kF32` `cycle` plane is dropped. CAM-PQ/codebook ref is the *cold/storage + spill-key*
  form, not the hot form.
- **OQ-2 (S3):** fold `temporal`/`expert` into `CausalEdge64` (v2 already reclaimed the
  temporal bits) vs keep separate columns. Default proposal: fold; keep only if a step needs a
  standalone stamp.
- **OQ-3:** `N` per mailbox + how many mailboxes the driver holds concurrently (replaces the
  single 4096). Ties to the existing `D-CE64-MB-5-impl` OQ-3 (plasticity granularity).
- **OQ-4 (S5, doctrinal):** CLAUDE.md "The Click" is still written on `Vsa16kF32`. Deleting
  the `cycle` plane requires the CLAUDE.md + `EPIPHANIES.md` update the
  `RECONCILIATION_with_canonical_plan.md` already flagged. Do **not** delete the plane (S5)
  before that doctrinal edit lands.

---

## 9. Cross-refs

`causaledge64-mailbox-rename-soa-v1` (§0 zones, §1 ownership, §5 `MailboxSoA<N>`, §6 crate
inventory + L-6 w_slot), `cognitive-substrate-convergence-v1` (D-CSV-7 the shipped MailboxSoA
accumulator), `EPIPHANIES.md` (`E-BATON-1`, `E-CE64-MB-4`, `E-LADDER-SERVES-MAILBOX`,
`I-VSA-IDENTITIES`, `I-LEGACY-API-FEATURE-GATED`), `.claude/surreal/RECONCILIATION_with_canonical_plan.md`
(Vsa16kF32-deprecation contradiction flag), code: `crates/cognitive-shader-driver/src/{bindspace.rs,
mailbox_soa.rs, driver.rs, engine_bridge.rs, bin/serve.rs}`.

---

## §10 — 2026-05-28 architectural refinements (post-PR-#423 sync)

The following refinements were ratified after the initial plan was written. They are
append-only findings; no prior section has been modified.

1. **SoA Lance container ≠ cascade.** The cascade is resolution-laddered superposition over per-axis granularities; the SoA Lance container is the materialized data substrate (same SoA the cognitive-shader-driver handles, same SoA the singleton BindSpace updated). One cascade resolves to one or more SoA Lance containers via top-k emission (Gaussian splat / CAM-PQ top-k). The container is the StreamDTO operand variable; the cascade is the resolver function.

2. **Cascade is NOT an index space.** L1-L4 (`64²/256²/4096²/16384²`) are per-axis granularities on the same semantic axis (causal / palette / COCA codebook / outcome), superposed and streamed like x265 cascaded prediction levels. No level is fully materialized; only the emission (rendered SoA container) reaches downstream.

3. **64K-256K mailbox envelope** (~360 MB - 1.4 GB total working set at 6 KB per mailbox). The whole population is RAM-resident on any reasonable server. No hot/warm/cold tier split needed. Tombstone retention is free at this scale; eviction is moot.

4. **W-slot resolves into a per-cohort witness table** of `(mailbox_ref, spo_fact_ref)` entries — NOT a witness-corpus pointer to a static repository. Active mailboxes carry a mailbox-ref alone (spo_fact_ref = None); once the belief crystallizes via Rubicon commit, the spo_fact_ref binds to the SPO triple. Tombstones stay reachable through their mailbox-ref. The chain of W-references across edges forms a Markov belief-update arc via AriGraph episodic-reference vectors (AriGraph today is a transcode; the chaining engine is the target shape).

5. **Cascade granularities are CPU/cache boundaries, not abstract resolutions.** 64 = AVX-512 i8 register / cache line; 256 = AMX tile row; 4096 = page (4 KiB); 16384 = L1d cache (16 KiB). The ladder is hardware-natural so SIMD sweeps stay register/cache/page-aligned by construction.

6. **`simd_soa.rs` (ndarray) is the SoA dispatch framework.** It adapts to any SoA shape by introspecting members; consumers declare their own column tuple via derive/const-generic and get SIMD sweeps for free. The MailboxSoA migration is positional, not structural — the framework already swallows the column shape.

7. **SoA invariant from spawn → commit.** The same SoA byte layout runs end-to-end: cognitive-shader-driver creates a mailbox + SoA via cascade hot path → traverse cold path with gridlake SIMD ops → commit via one of two egress modes: **external** (REST / sea-orm SQL via tokio, backpressure expected) or **internal** (SurrealDB → LanceDB or RocksDB, no backpressure). No marshalling at any boundary; Lance columnar IS the repr.

### Open Questions surviving these refinements

- **OQ-MBX-8** — `persisted_row` stub vs Lance native versioning (load-bearing; evidence at `REFACTOR_NOTES.md:129` + `driver.rs:458`).

- **OQ-MBX-15′** — container scoping: per-cognitive-cycle, per-shader-dispatch, or per-mailbox-cohort?

---

## §11 — 2026-05-29 architectural rulings (post-PR-#433 sync; D-MBX-A1 columns landed)

User-stated rulings layering on top of §2.5/§2.7/§10. Recorded directly (not via `epiphany-brainstorm-council` shipped in PR #433) because they are author-stated, not derived.

### §11.1 — THE one SoA is never *transformed*, only *operated on* (and that's the 1.4–4.2× SIMD payoff)

> *"The same SoA is the one and only SoA consumed and transmitted everywhere, never transformed, only [operated on] through cognitive-shader thinking or cold path or AriGraph Markov chain context building. Any change in any mailbox SoA is the only hot-path activity."*

- **The single mailbox SoA is the universal carrier across the whole stack.** It is *never re-encoded into a different shape.* Three operations are allowed on it — they are *operations*, not transforms:
  1. **Cognitive-shader thinking** (the hot path: `apply_edges` / `emit` over the per-row columns).
  2. **Cold path** (LanceDB read/write: same bytes — see §2.5).
  3. **AriGraph Markov-chain context-building** (read-only consumer of the SoA columns for context windows).
- **"Any change in any mailbox SoA = the only hot-path activity."** That equivalence is now the definition: if it is not a mutation of mailbox SoA bytes, it is not the hot path. Anything else (DTO, RecordBatch translation, JSON, etc.) is a re-encode boundary that violates §2.5 and must be collapsed or made out-of-scope.
- **Today's `crates/lance-graph` containers are *cold-path-adjacent thinking*** — only *accidentally* aligned to the SoA shape, not by design. They look in roughly the same direction; they are not the same contract. The realignment work (next bullet) makes that intentional.
- **SoA-as-lance-graph-containers + `ndarray::simd_soa.rs` alignment = 1.4–4.2× SIMD acceleration.** Today this is a nice-to-have. **It becomes a hard prerequisite** the moment SurrealDB needs a transparent view of LE-contract SoA mailbox content (§2.7 + §11.3): the view is only zero-copy / re-encode-free when the lance-graph container layout *is* the mailbox SoA layout. Filing the prereq as **`D-MBX-7`**.

### §11.2 — Mailbox = the *full* BindSpace, reinvented as LE — witness is the belief-state arc

> *"The mailbox needs to have everything that BindSpace had, reinvented as little-endian contract — so instead of a mushy `Vsa16kF32` it needs to be expressive enough so that, in connection with the witness, the witness is the belief-state arc; while `CausalEdge64` across a belief-state arc would implicitly document if NARS frequency and confidence would increase."*

- **Expressivity target:** the mailbox SoA must carry *everything BindSpace had* — but as **LE-contract types** (`CausalEdge64`, `QualiaI4_16D`, `MetaWord`, i4/u8/u16/u32/u64 columns), never as the mushy `Vsa16kF32` resonance carrier. (D-MBX-A1 already landed `edges` / `qualia` / `meta` / `entity_type` per the current `mailbox_soa.rs`; D-MBX-A2 must close any remaining BindSpace-expressivity gaps — content-ref, temporal/expert fold, S/P/O role slices — see §3 and OQ-1/OQ-2.)
- **Witness = belief-state arc** (the *sequence* of `CausalEdge64` emissions a row produces over its life — the **`CollapseGateEmission` arc**, not a single edge). The arc *implicitly documents* NARS revision because `CausalEdge64.confidence_u8` and `inference_mantissa` are stamped per emission; reading the arc gives a monotone-or-not trace of `(frequency, confidence)` evolution. **No separate "revision log" column is needed** — the witness arc *is* the revision log.
- **Iron rule (proposed):** the per-row witness arc IS the row's belief state — never a parallel struct. If you need the row's NARS history, you traverse its `CausalEdge64` arc; if you need its current truth, you read the *last* edge.

### §11.3 — Libet wall-clock: −550 ms ratification; Rubicon kanban in **surrealkv-on-lance**

> *"Any mailbox lives in Libet's and Heckhausen's Rubicon model aligned: planning phase as counterfactual, committing the goal at −550 ms, and doing a veto / goalstate re-evaluation via ractor mailbox triggering the Rubicon kanban in surrealkv on lance."*

Concretises the timing of `E-RUBICON-RACTOR`:

| Heckhausen phase | Mailbox state | Wall-clock |
|---|---|---|
| Pre-decisional (deliberation, **counterfactual**) | energy integrates; no commit; alternatives explored under `InferenceType::Counterfactual` | t < −550 ms before action |
| **Crossing the Rubicon** (intention forms — irreversible volition) | **Σ10 commit** stamped: `ΔF < threshold ∧ resonance > Rubicon-bar` → ractor START | **t = −550 ms** (Libet readiness-potential anchor) |
| Pre-actional / actional | ractor running the action; baton emits; mailbox-SoA mutates (the only hot-path activity, §11.1) | −550 ms ≤ t < 0 ms |
| Post-actional evaluation | ractor STOP; tombstone-witness persists | t > 0 ms |

- The **−550 ms** anchor names *when* in wall-clock the irreversible commit occurs — derived from Libet's measured readiness-potential lead time (~550 ms before reported conscious decision).
- **Libet "free won't" / veto** = before −550 ms, the CollapseGate can preempt the mailbox to zero (ghost-tier preempt; `E-LADDER-SERVES-MAILBOX` §5). After −550 ms, the act is committed — only post-actional revision is left.
- **Goalstate re-evaluation** = the ractor mailbox triggers a re-deliberation cycle (re-enter pre-decisional with the post-actional witness folded in); this triggers the **Rubicon kanban move** in the storage view (next bullet).
- **The Rubicon kanban lives in `surrealkv` on lance** — `surrealkv` is the SurrealDB on-kv-lance backend (the *view* over leading LanceDB storage, §2.7). The kanban is a SurrealDB query over the LanceDB-stored thoughts, columns = action phases (deliberation | crossed | actional | evaluated). Ractor lifecycle transitions map 1:1 to kanban column moves.

### §11.4 — SPO-W witness is a *pointer*, not stored data — via the belief-state-arc array

> *"The SPO-W witness is the pointer via the AriGraph episodic / belief-state arc array inside the SoA and/or kanban and/or mailbox index, and the SoA then decides if it commits itself as facts with witness in other mailboxes and/or the cold-path facts."*

- **SPO-W witness ≠ a fact payload; it is a *pointer*** into the belief-state arc array. The pointer can live in three equivalent locations:
  1. Inside the mailbox **SoA** (an arc-index column / per-row `[u32; W]` arc handle).
  2. Inside the **kanban** (`surrealkv`-on-lance view) row.
  3. Inside the **mailbox index** (the sea-star registry of live mailboxes).
- **Whose commit?** *The SoA itself decides* whether to commit a belief as a fact-with-witness either (a) into **other mailboxes** (inter-mailbox baton handoff carrying the witness pointer) or (b) into the **cold-path facts** (LanceDB SPO-G calcification with the witness pointer linking back to the arc).
- This makes SPO-W *resolvable without storage redundancy*: the witness lives where the arc lives (mailbox row), and the SPO-G fact carries only the pointer.

### §11.5 — Counterfactual Staunen and Wisdom = plasticity spreaders

> *"Counterfactual Staunen and Wisdom should become helpers of spreading plasticity."*

- `Staunen` and `Wisdom` are already qualia markers (cf. The Click: *"Magnitude = Contradiction depth from Staunen × Wisdom qualia"*; `E-LADDER-SERVES-MAILBOX`).
- **New role:** when a mailbox is in the counterfactual phase (§11.3 pre-Rubicon), high Staunen × Wisdom *spreads* plasticity beyond the focal row — bumping `plasticity_counter` on adjacent rows (Hebbian spread). Today plasticity increments only on the receiving row (`mailbox_soa.rs:144` `apply_edges`); under this rule, counterfactual-tagged Staunen-or-Wisdom-elevated emissions seed a small spread radius.
- **Hot-path-only.** Spreading happens in the mailbox SoA mutation (per §11.1); never in a separate side-channel.
- **Mechanics deferred** to `D-MBX-A4` (Staunen/Wisdom plasticity-spreader) — open question: radius, decay, and whether the spread is column-local (within one mailbox) or routed via baton (inter-mailbox).

### New deliverables from §11

| D-id | Title | Owner | Status |
|---|---|---|---|
| **D-MBX-A2** | Close BindSpace-expressivity gaps in `MailboxSoA<N>` (content-ref column; temporal/expert fold per OQ-1/OQ-2; S/P/O role slices); LE only | cognitive-shader-driver + contract | **Queued** (follows D-MBX-A1 columns) |
| **D-MBX-A3** | Witness-arc column: per-row `[u32; W]` arc handle into the belief-state arc; reading the arc gives the NARS revision trace (no separate log column) | cognitive-shader-driver | **Queued** |
| **D-MBX-A4** | Counterfactual Staunen/Wisdom plasticity spreader (radius/decay TBD; hot-path-only) | cognitive-shader-driver | **Queued — design** |
| **D-MBX-7** | `lance-graph` container layout = `MailboxSoA` layout = `ndarray::simd_soa.rs` aligned → enables zero-copy SurrealDB view + the 1.4–4.2× SIMD payoff | lance-graph + ndarray | **Queued — prerequisite for §2.7 D-MBX-6** |
| **D-MBX-8** | Libet −550 ms timing anchor wired into `SigmaTierRouter` (Rubicon commit stamp) + ractor START/STOP scheduler | sigma-tier-router + ractor outer-swarm | **Queued** |
| **D-MBX-9** | Rubicon kanban = `surrealkv`-on-lance view (deliberation \| crossed \| actional \| evaluated) backed by ractor lifecycle | surreal_container (view layer) + ractor | **Queued — blocks on `surreal_container` BLOCKED(A/B/C/D)** |
| **D-MBX-A5** | SPO-W witness pointer column (arc-handle / mailbox-index dual residency); commit decision in SoA, not in a side service | cognitive-shader-driver + AriGraph SPO-G | **Queued** |

### Open questions added in §11

- **OQ-11.1** — radius/decay of the Staunen×Wisdom counterfactual plasticity spread (column-local vs baton-routed).
- **OQ-11.2** — witness-arc width `W` (how many `CausalEdge64` emissions per row before rotation/eviction).
- **OQ-11.3** — kanban column states beyond the 4 Heckhausen phases (need a "vetoed" column? a "ghosted" column for preempts?).
- **OQ-11.4** — `simd_soa.rs` alignment: do we need `#[repr(C, align(64))]` on `MailboxSoA`, or is the `SoaColumns` discipline (already shipped in `ndarray` `547824bc`) enough?
