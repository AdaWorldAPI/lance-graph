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

**Net footprint:** per-row hot state drops from ~71.6 KB (dominated by the 64 KB `cycle`
plane) to **≈ 24–30 B/row** (edge 8 + qualia 8 + meta 4 + entity_type 2 + optional
content_ref ≤ 6). That is the whole point: the mailbox thoughtspace is L1/L2-resident
(canonical plan §5 ~1.2 KB/compartment), the singleton never was.

---

## 4. What STAYS shared (does not migrate)

- **`Arc<OntologyRegistry>`** — the OGIT/DOLCE/FIBO calcified ontology. Immutable at read
  time, hydrated/mutated on its own owner (`OntologyRegistry::append_mapping`), shared by
  `Arc` to every mailbox. This is cold Zone-2, not ephemeral thinking. Mailboxes hold
  `&OntologyRegistry` (or a cloned `Arc`), never a copy of its tables.
- **Codebooks / CAM-PQ centroids** — the cleanup codebook the content references resolve
  against (`I-VSA-IDENTITIES` Test 3). Shared, immutable, cold.
- **AriGraph SPO-G cold store + Lance dataset** — where the mailbox's witness calcifies on
  death (`E-LADDER-SERVES-MAILBOX` §6). Shared persistence, not per-mailbox.

Rule of thumb: **ephemeral per-think state → into the mailbox; calcified/immutable shared
knowledge → stays a shared `Arc`.** The singleton's sin was conflating the two in one
`BindSpace`.

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

- **OQ-1 (S2):** content identity in the mailbox — CAM-PQ code, codebook id, or a slim
  Hamming slice? (drives the per-row content_ref width). Default proposal: CAM-PQ code (6 B).
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
