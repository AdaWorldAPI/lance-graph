# D-RMAP-1A: RelationshipMap tables, borrowed projections and thought palettes

> **Status: PROPOSAL / ARCHITECTURE ONLY (2026-10-19).** Clarifies [D-RMAP-1](d-rmap-implicit-relationship-register-v1.md). This amendment **supersedes** earlier suggestions that CE64 P could identify a relationship map. No code, persisted data format, wire/ABI change, opcode or performance claim.
>
> **Versioning (2026-10-19):** §7 derives map, palette and binding generations from Thought Git (D-TGA) and #1605's `ThoughtCommitId` / `DurableRef`; D-RMAP adds no durable counter. PROPOSED.
>
> **Operator ruling for this proposal:** RelationshipMaps live in separate explicit tables, expose a zero-copy map of existing indexed values, and are not stored inside RelAlg. Each thought selects a *palette of global RelationshipMaps* using another independent table. Nothing is crammed into CausalEdge64.

> **Related physical-lifetime study (2026-10-25):** [D-DRAV-1](d-dr-address-first-value-demand-v1.md) treats the attested relationship/address map as independently reusable from optional gathered payload values, and compares map-only, borrowed, materialized and physical-storage execution. RelationshipMapStore, BIND, Quack and source ownership remain as defined here; this is **not** a new mapping authority or backend implementation.

## 1. Ownership contract

| Owner | Owns | Must not own |
| --- | --- | --- |
| OGAR Class / ClassView / DTO-IR | Types, fields, associations, actions, compiled class vocabulary, generated field masks and DTO projection descriptors | Persistent relationship index pages, RelAlg evaluator, runtime DTO copies |
| RelationshipMapStore | Separate relation-fact tables, sorted index permutations, global relationship-map identity, seal/version, eligible physical capabilities and optional relation tiles | Datalog syntax/AST, CE64 fields, per-thought graph copies |
| ThoughtMapPaletteBinding | Thought slot and incarnation -> MapPaletteId **and the palette generation it bound** under a witnessed frame; selected subset and order of maps | Relation-fact payload, evidence roots |
| MapPalette | Palette ID **and its own palette generation**, plus 0..255 local map selectors -> globally stable MapId + map generation | Global relationship truth or per-object row materialization |
| FocusBinding | Source/target global identities -> focus-local slots; lifetime, version and ACL attestation | Persistent semantic facts |
| BIND | Resolves metadata and borrows native spans, masks, indexed ranges or permutations with a mapping witness | RelAlg rewrite rules, storage ownership, separate runtime evaluator |
| RelAlg | Pure algebra and rule meaning: variables, set/bag semantics, joins, recursion contract, measures, normalization | Map tables, index build, cache eviction, DTO storage |
| Quack / R2IL / Mask-RISC | Existing numerical lowerings and one executor over admitted operands | Semantic maps, new graph VM, DTO parsing |
| CE64 | Unchanged opaque local registers and inference/truth fields | MapId, MapPaletteId, relationship topology or DTO routing |

This fits the [Quack storage-portability ruling](../board/entries/2026-10-05-quack-storage-portability.md): Quack owns stable numeric read semantics; the backend supplies capabilities; sealing owns durable writes. It also fits OGAR's [per-class transpile substrate](https://github.com/AdaWorldAPI/OGAR/blob/main/docs/OGAR-TRANSPILE-SUBSTRATE.md), [ClassView × field-mask projections](https://github.com/AdaWorldAPI/OGAR/blob/main/docs/CLASSVIEW-FIELDVIEW-ASKAMA-BITMASK.md), and [OGAR AST contract](https://github.com/AdaWorldAPI/OGAR/blob/main/docs/OGAR-AST-CONTRACT.md). OGAR explicitly distinguishes its compiled per-class transpiler from a mere DTO store; preserve that distinction.

## 2. Explicit table family, not a new shared monolith

The following are proposed **logical tables**, not physical record layouts or mandatory separate database files. Resolve to already existing Lance/SoA/OGAR types before creating storage or types.

| Logical table | Contents | Status |
| --- | --- | --- |
| RelationCatalog | Global predicate ID, arity, argument sorts, functional/bag/set constraints, permitted index projections | Semantic declaration |
| RelationFacts | Canonical fact/statement identities, typed arguments, evidence roots, time/version/ACL | Source of truth |
| RelationOrder | Sealed indexed permutations of original fact row references: e.g. predicate-subject-object, object-predicate-subject | Derived ordering; no source payload copy |
| **RelationshipMap** | Separate MapId, map generation, underlying relation-order view handles, constraints and capability metadata | **Global shared table** |
| **MapPalette** | MapPaletteId + **palette generation**, local map ordinal 0..255 to MapId and map generation. Reassigning any slot bumps the palette generation (or mints a new palette; palettes may also be immutable) | **Shared palette table** |
| **ThoughtMapPaletteBinding** | (thought slot, incarnation, frame witness) to MapPaletteId + the palette generation it bound; resolution refuses when the palette's current generation differs; optional sparse palette visibility override | **Separate thought map** |
| FocusBinding | Frame witness, source/target slot-to-global identity maps and authorization | Local map binding |
| RelationTile | Optional cached source-slot to local target mask, keyed by **global MapId + map generation (or index seal digest)** + FocusBinding generation + authorization scope. A frame id or local palette slot alone is never a sufficient key | Disposable derived bits |
| RelationDelta | Separately typed fact changes, binding changes, tile invalidations, speculative thought claims | Incremental witnesses |

**One map may be referenced by many palettes and many thoughts without copying its fact storage.** A local palette number is never a global map identity. Thought 0 is immutable givens; thoughts 1..63 select or overlay views without copying canonical relation tables. Any per-thought changing of map selection is a binding delta, not a mutation of relation truth.

## 3. Zero-copy binding and DTO routing

The proposed interface is a **borrowed view of existing native data**, not a new DTO object carrier:

~~~text
OGAR schema / ClassView / codebook
                |
       schema-resolved relation
                |
ThoughtMapPaletteBinding -> MapPalette -> RelationshipMapStore
                                                 |
                                      sealed map/index view
                                                 |
FocusBinding ------------------------------------ BIND
                                                 |
                       borrowed typed relation span / index / mask
                                                 |
                RelAlg / Quack logical demand -> R2IL / mask-risc
~~~

BIND may emit lightweight descriptors: borrowed SoA lanes, row-strided fields, native Arrow/Lance spans, sealed ordered runs or permutation cursors, typed local mask references, and scope/version witnesses. The canonical fact bytes stay in their owner. It does not hydrate each object, marshal per-record DTOs, allocate a graph adjacency object per edge or serialize to JSON in the hot substrate.

**Zero copy is specifically about not duplicating canonical fact payload.** Immutable sorted index permutations, derived tiles, authorization receipts, palette bindings and page-cache/decompression work all cost bytes/time and must be measured.

OGAR's DTO plans can provide an ergonomically tiny API over this binding, but OGAR must remain the schema/compiler layer; RelationshipMapStore is the data owner; RelAlg is the mathematical consumer. No storage-owner field belongs in the RelAlg AST. No schema-DTO record belongs in the execution register.

## 4. Thought chooses a relationship-map palette, not CE64

~~~text
ThoughtMapPaletteBinding
  (thought_slot=7, incarnation=41, frame=9) -> map_palette=23, palette generation 5

MapPalette[23], palette generation 5
  slot 0 -> global MapId(is_a), map generation 12
  slot 1 -> global MapId(part_of), map generation 8
  slot 2 -> global MapId(mentioned_in), map generation 3
  slot 3 -> global MapId(implements), map generation 2
  ...
  
BIND: (thought, palette, focus, scope, version) -> admitted bound relations
EXECUTE: RelAlg/Quack -> existing R2IL/Mask-RISC -> result
~~~

The names are documentation labels; their runtime identities resolve to typed numeric codes. A thought may switch to another palette, restrict particular maps, or modify speculative relationship claims with a sparse overlay. It does not need to copy a global map or materialize the entire thought × map × source × target tensor.

**No CE64 map-selection rule:** S/P/O remain opaque operands under their declared ClassView reading. **P does not denote MapId by default. W does not select the relationship palette.** W's declared thought reading remains subject to the existing October 17 witness-root compatibility/version-gate issue. Plain RelAlg queries can use RelationshipMaps without carrying CE64 at all.

The entire conceptual execution context is: (FrameWitness, ThoughtSlot+Incarnation, MapPaletteId+Generation, FocusBinding, AccessScope), with each field attested outside the 64-bit wire. Raw CE64 register equality does not imply semantic equality across frames.

## 5. RelAlg separation is not optional

The [RelAlg v1 implementation](relalg-v1.md) currently covers Boolean selection within **one row universe**, not general foreign joins or recursion. Any future multi-relation capability must introduce logical variable-binding semantics in the existing selected owner; it **must not** make RelAlg build or own maps. The binder separately supplies borrowed physical operands, already aligned or with an explicit legal mapping. The **October 16 no-gather firewall** and H14c/H14d direct-coordinate proof remain the present authority. Mapping distant objects into a focus is a proposed *pre-fold* operation that requires new correctness, legality and full-cost evidence.

Typed selection of method or ActionDef targets is eligible for relational evaluation; actually invoking an effectful method remains a distinct authorized action.

## 6. Specific falsifiers and proposed implementation order

1. **Map-table separation:** one semantic relation answered from HHTL, ordered TrieMap and borrowed tile, with identical results and no RelAlg-owned index state.
2. **Thought palette proof:** 1/8/64 thought incarnations each select overlapping subsets of 64/128/256 global MapIds, borrowing shared index buffers without copying canonical payload or whole map bank per thought.
3. **Wire blindness:** duplicate local S/P/O/W bytes under distinct frames and palettes produce different bound global facts; no CE64 changes, no accidentally shared MapId.
4. **Lifetime and ACL:** a changed map generation, source/target palette, authorization scope, resealed index, compacted row mapping or recycled thought slot refuses stale use.
5. **Evidence and algebra:** multi-parent ontologies, cyclic joins, out-of-focus frontier neighbors, bag counts, distinct counts, proof roots and unknown facts remain exact; Boolean tiles cannot silently replace evidence-bearing relations.
6. **Zero-copy and economics:** separately report borrowed buffer identities, bytes for indexing/permutations, palette and tile memory, cold/warm seeks, focus churn and reuse crossover. A 2 KiB participation plane or an 8 KiB adjacency tile is **not** total storage cost.

Sequence if later approved: (a) audit existing OGAR DTO/ClassView and Quack physical binding seams, (b) prototype standalone RelationMap borrow semantics, (c) implement/test the thought-to-map-palette lookup using typed tables, (d) establish the existing Quack/RelAlg execution parity without a second evaluator, (e) only then consider optional cached tiles and incremental spatial frontier overlays.

### 6a. Generation witnesses (2026-10-19 correction, #1617 review)

Two review findings, both correct. (1) A `RelationTile` keyed only by focus/frame keeps matching after its map's facts are appended, deleted, compacted or resealed, so it returns obsolete adjacency. Its key must name the global `MapId` and the map generation (or index seal digest). (2) `MapPalette` had no generation of its own, although §4 requires `MapPaletteId + Generation`; per-entry map generations cannot tell which slot-to-map assignment a thought bound. A palette carries its own generation, and reassigning a slot invalidates dependent bindings and tiles instead of silently reusing them.

Falsifiers, each a refusal (never a plausible stale answer):

| Change | Must refuse |
| --- | --- |
| Map regeneration (append / delete / compaction / reseal) | any tile or binding holding the old map generation or seal digest |
| Palette-slot reassignment | any binding or tile holding the old palette generation |
| Focus-slot reuse | a tile built for the previous FocusBinding generation |
| Thought incarnation reuse | a ThoughtMapPaletteBinding from the previous incarnation |
| Stale tile access | a tile whose key generations do not all match the current context |
| Cross-tenant / authorization change | a tile or binding minted under another access scope |

Status: PROPOSED. Nothing here is implemented; RelationshipMap storage stays a proposal until a separately approved implementation establishes its legality (co-addressed operands, not a gather) and its economics. Map identity never enters CausalEdge64, and RelAlg never owns map tables.

## 7. Versioning: one logical history, borrowed from Thought Git and #1605

> **Status: PROPOSED (2026-10-19).** Nothing in this section is implemented. It decides which of the identities named in §2, §4 and §6a are *new* and which are readings of identities that D-TGA ([thought-git-api-v1](thought-git-api-v1.md)) and D-CPG ([causal-pickaxe, #1605](causal-pickaxe-resident-git-lance13-temporal-v1.md)) already define. D-TGA-0..6 and CPG-00..06 are themselves unbuilt, so every "existing authority" below is a *specified* authority, not running code, unless marked otherwise.

### 7.1 Ownership

| Owner | Owns | Source |
| --- | --- | --- |
| Thought Git (D-TGA) | logical revision of speculative state: `branch`, `write`, `rebase`, `land`, `squash_into`, `blame` | thought-git-api-v1 §Surface, Rules 1–5 |
| `ThoughtCommitId` (D-CPG §1) | identity of one in-memory thought commit and its parents | #1605 §1, "speculative" row |
| Lance, through `DurableRef{dataset, branch, version}` | sealed snapshots; one `land` = one next version (D-TGA "versions as PR entries") | #1605 §1 "durable" row, §5.2 |
| `temporal.rs` | epistemic reading of a version: `QueryReference::at(v, rung)`; replay and blame use an explicit `EpistemicMode::Strict` | `crates/lance-graph-planner/src/temporal.rs`, `QueryReference` and `QueryReference::at` (code) |
| Physical index seal | validity of a sorted permutation against the row-address map it points into | this plan, §7.3 row 5 |
| Binding witness | the ephemeral tuple BIND checks before it borrows a span (§7.5) | this plan |

D-RMAP owns **relational binding**, and nothing else: which maps a thought selected, under which focus and scope, and the witness that the borrowed spans are still the ones it selected. It does not own a revision counter of its own.

### 7.2 What is code today (measured 2026-10-19, read in full)

- **Version identity is never dataset-scoped.** `temporal.rs` `pub type LanceVersion = u64`; `temporal_pov.rs` `pub type LanceVersion = u64`; `scheduler.rs` `pub struct DatasetVersion(pub u64)` (also `LanceCycleWriter::head()`); `versioned.rs` returns bare `u64`. `QueryReference` (`temporal.rs`) has `server_id, ref_version, hlc_tick, mode, rung` and no dataset or branch. No `DurableRef` or `ThoughtCommitId` exists in `crates/` (search: `DurableRef|ThoughtCommitId|dataset_id` over `crates/**/*.rs`, 0 hits; a negative over that named space only).
- **`versioned.rs` already shows the collision this section guards against.** `commit_encounter_round` (`versioned.rs`) writes three datasets (nodes, edges, fingerprints) in three separate commits and returns only the nodes version; the edges and fingerprints versions are dropped. `at_version`, `current_version`, `versions` and `tag` read the nodes dataset only. A caller holding that `u64` cannot name the edges snapshot it was written with. (Its doc says each write "appends"; `write_batch` uses `WriteMode::Overwrite`.)
- **`QueryReference::default()` is unbounded** (`ref_version = u64::MAX`), the dangerous default #1605 names. A witness must never be built from it.
- **The alpha overlay's `visits` is not a generation.** `AlphaStamp.visits: u16` counts revisits of one address and saturates (`alpha.rs` `claim`, `saturating_add`); `seq`/`rung`/`cycle` keep the first claim. A palette rewritten A→B→A at one address is two revisits, not a version, and stops counting at 65 535.
- **D-TGA has no thought incarnation.** Slots 1..63 are reused after `squash`/`discard` (D-TGA Rule 4: "the thought number does not survive"). §2's `(thought slot, incarnation)` therefore needs an identity D-TGA does not define; #1605's `ThoughtCommitId` is the only specified candidate.

### 7.3 N→1: each D-RMAP identity against the authority that already exists

| D-RMAP concept (§2/§6a) | Existing authority | Derived or retained | Falsifier (§7.4) |
| --- | --- | --- | --- |
| **Map generation** | `DurableRef` of the sealed RelationshipMap/RelationFacts version | **Derived.** Equality of `DurableRef`, never of a bare version. A new Lance version (append, delete, compaction, reseal) is a new generation, so nothing needs counting | X1 cross-dataset, X2 compaction, X9 deletion |
| **Palette generation** | sealed palette: `DurableRef` of the palette table version. Speculative palette edit inside a thought: the `ThoughtCommitId` of the commit that made it | **Derived.** Versions and commit ids are never reused, so A→B→A cannot reproduce an old generation | X3 palette ABA |
| **Thought-palette binding** | `(ThoughtCommitId, palette generation)` | **Derived.** `ThoughtCommitId` replaces `(slot, incarnation)`; the slot (CE64 W) is a register, not an identity. Only the binding *row* is new | X4 slot reuse, X6 rebase |
| **Focus generation** | none. A focus is a frame-local lens; changing it is a binding delta, which D-RMAP-1 §5 already separates from fact and thought deltas | **Retained**, ephemeral: dies with the frame, never persisted, never recovered | §6a focus-slot reuse |
| **RelationTile seal** (index seal) | `DurableRef` of the RelationOrder version **plus** the row-address map it was built against | **Retained** unless RelationFacts is written with stable row ids (opt-in, per dataset). With them it collapses to the RelationOrder `DurableRef` (H14c-5 measured 0 of 3 510 `_rowid`s moved by compaction); without them compaction moved 3 510 of 3 510 and a stale map omitted 160 of 160 live rows silently | X2 compaction |
| **Durable revision** | `DurableRef{dataset, branch, version}`, read through `QueryReference::at(v, rung)` with `Strict` for replay | **Existing** (specified by #1605; no type yet) | X1, X8 |
| **Authorization scope** | the existing RBAC decision at bind time | **Retained**, separate key component; no revision carries it | X7 |

Result: of the six D-RMAP generation concepts, four are readings of two existing identities (`ThoughtCommitId`, `DurableRef`), and two stay as witnesses (focus generation: ephemeral; index seal: unless stable row ids). D-RMAP adds **no durable counter**.

### 7.4 Falsifiers (specified, none run)

Each is a refusal with a named reason. A plausible stale answer is the failure. Each lists the disable that must turn it red.

| # | Case | Fixture | Must | Disable |
| --- | --- | --- | --- | --- |
| X1 | Cross-dataset collision | two datasets both at version 7; a tile keyed on map A at v7 | refuse a read against map B at v7 | key on `version` alone |
| X2 | Compaction | compact RelationFacts without stable row ids; tile and RelationOrder built before | refuse the stale permutation (new `DurableRef`, new seal); never return the H14c-5 160/160 omission | key the seal on RelationOrder's `DurableRef` only |
| X3 | Palette ABA | palette slot 3: MapId P → Q → P across three sealed versions; binding taken at the first | refuse at the third (version differs although the content repeats) | use a content hash or `visits` as the generation |
| X4 | Thought slot reuse | thought in slot 7 binds palette 23, is squashed; a new thought takes slot 7 | the new thought does not inherit the binding (different `ThoughtCommitId`) | key the binding on the slot |
| X5 | Uncommitted change | thought writes a speculative claim after its last commit; tile built from that commit | refuse the accelerated path while dirty; a reader holding only a `DurableRef` never sees the claim | drop the dirty check |
| X6 | Rebase | thought bound map M at `merge_base` v; main lands v+1 changing M; rebase | bindings to M at v revalidate or refuse; D-TGA Rule 2 retraction applies to tiles derived from M | keep the bound generation across rebase |
| X7 | Authorization change | same thought, same versions, scope revoked for one map | refuse; no payload span is borrowed before the check | omit scope from the key |
| X8 | Squash + historical replay | land, squash, reuse slot, then blame and replay the landed fact | blame returns `(landing DurableRef, source bitset)`; replay reads `Strict` at that version; a request for erased micro-steps returns `LineageUnavailable` | resolve blame through the slot; replay with `QueryReference::default()` |
| X9 | Deletion / DRed | thought derived B from A; main deletes A | the tile over A's map is refused (new `DurableRef`) and B is retracted before re-derivation (D-TGA Rule 2) | invalidate tiles without retracting derived cells |

### 7.5 Minimal sketches (not code)

```rust
// Reused, unchanged: temporal_pov::LanceVersion, planner::temporal::QueryReference.
pub struct DurableRef { dataset: DatasetId, branch: BranchId, version: LanceVersion } // #1605 §1; DatasetId/BranchId representation open until CPG-00
pub struct ThoughtCommitId(u64);                 // #1605 §1; must be never-reused (see 7.6)
pub enum PaletteRevision { Sealed(DurableRef), Speculative(ThoughtCommitId) }
pub struct IndexSeal { order: DurableRef, addresses: DurableRef } // addresses dropped when RelationFacts uses stable row ids
pub struct FocusGeneration(u32);                 // frame-local, never persisted

/// What BIND checks, every time, before borrowing a span. Ephemeral; never stored.
pub struct BindingWitness<'a> {
    thought: ThoughtCommitId,
    palette: PaletteRevision,
    maps: &'a [(MapId, DurableRef, IndexSeal)],
    focus: FocusGeneration,
    scope: AuthorizationReceipt,                 // existing RBAC owner; type not chosen here
    read: QueryReference,                        // concrete ref_version; Strict for replay; never Default
}
```

`MapId` and `MapPaletteId` remain the §2 identities. No field reaches CausalEdge64.

### 7.6 Gaps this section finds in D-TGA and #1605

1. **`ThoughtCommitId` reuse is undefined.** #1605 says "stable while retained". X3 and X4 need *never reused*, including after `discard` and `squash`. Until that is specified, a binding keyed on it cannot be trusted after a discard.
2. **`QueryReference` has no dataset or branch.** It is a correct epistemic coordinate and an incomplete durable one. The read coordinate is the pair (`DurableRef`, `QueryReference`); `QueryReference` alone must not be accepted as a map or palette generation.
3. **A multi-dataset write has no single version.** `versioned.rs` is the measured instance; D-TGA's "one land = one version" holds only if a landed thought writes one dataset, or if the PR record (D-TGA open item "where the PR record lives") names the `DurableRef` of every dataset it wrote.
4. **Dirty state has no witness.** No authority says whether a thought has writes after its last `ThoughtCommitId`; X5 needs one bit, and #1605 §5.8 forbids a commit per fold.

### 7.7 Cost and entropy: A (independent generations) vs B (derived from Thought Git and Lance)

No timings exist for either; none are given.

| | A: D-RMAP keeps its own counters | B: readings of `ThoughtCommitId` / `DurableRef` |
| --- | --- | --- |
| Identity spaces | map gen, palette gen, incarnation, focus gen, tile seal, plus `ThoughtCommitId` and `DurableRef`: 7 | `ThoughtCommitId`, `DurableRef`, plus focus gen and index seal: 4 |
| Durable counters D-RMAP must persist and recover | 3 (map gen, palette gen, incarnation) | 0 |
| Consistency checks between spaces | every counter against the version it shadows (a bumped map gen with an unchanged `DurableRef` has no meaning) | none: the derived ones are the authority |
| Wrap / reuse rules to write | one per counter | inherited: Lance versions never reuse; `ThoughtCommitId` needs gap 7.6.1 |
| Cost B pays that A does not | a `DurableRef` is wider than a counter in every key, and compaction invalidates every tile on the map even when content is unchanged (a refusal, then a rebuild) | |

A adds a second ledger beside Lance and Thought Git, the duplicated truth this plan's §1 rules out. B is chosen. Its known cost is the compaction rebuild; whether that matters is measured with the §6 economics, not assumed.

### 7.8 Schnellverschluss (#1620)

The MORK/PathMap probe ([#1620](https://github.com/AdaWorldAPI/lance-graph/pull/1620)) binds a `RelationView { rows, generation: u64 }` and refuses `Stale { bound, current }`. That `u64` is a stand-in: under §7 it becomes the `(DurableRef, IndexSeal, QueryReference)` part of the binding witness. The backend switch (MORK, PathMap, native) does not touch the witness: the witness is about the data, the backend is a physical capability BIND attests. No new executor, store, or per-fold commit; the no-gather ruling is unchanged.

## 8. Focus probe: a focus-bound RelationshipMap over existing indexes (measured 2026-10-19)

> **Status: MEASURED, probe only.** Code: `crates/mork-pathmap-probe/src/rmap.rs` (module `rmap`), `tests/rmap.rs`, `examples/rmap_cost.rs` (`cargo run --release -p mork-pathmap-probe --example rmap_cost`). Release build, one machine, warm caches, medians of 5 repetitions. Nothing here is adopted. No new executor, store, opcode, ABI, CE64 field or gather: every evaluation runs through `lance-graph-mask-risc` `execute`/`execute_into` with the existing `MaskOp` set, and address translation happens only in preparation.

### 8.1 What the probe builds

| Layer | Probe type | Owns | Copies of canonical facts |
| --- | --- | --- | --- |
| Canonical facts | `&[Fact]` (`[s,p,o]` u32) | the only payload | none |
| Ordered index | `SortedRun` (row permutation, `(p,s,o)`) or `SpogTrie` (PathMap, key `p‖s‖o`) | row references / trie keys | trie: keys copy p, s, o |
| Focus | `Focus{ids, generation}`, 256 slots | slot↔identity binding | none |
| Prepared map | `Csr` (per predicate: offsets, `u8` targets, canonical row, escapes) | slot-space adjacency + `Witness` | none (row refs kept) |
| Tile | `Tile`, one 65 536-bit plane per predicate, address `(target<<8)\|source` | derived from `Csr` | none |
| Thought delta | `ThoughtDelta{thought, p, add, del}` | per-thought edits over the shared tile | none |

`Witness{facts, seal, focus, scope}` stands in for §7.5's `BindingWitness` (`DurableRef`, `IndexSeal`, `FocusGeneration`, `AuthorizationReceipt`); `ThoughtDelta.thought` stands in for `ThoughtCommitId`. Out-of-focus targets are never dropped: a reach reports them as escapes (identities, or source slots with identities on demand), and `global()` refuses `IncompleteFocus` while any remain.

### 8.2 Correctness (12 tests, all green, independent scan oracle)

Every arm (direct over run, direct over trie, CSR, tile via mask-risc; identity and source-slot escape modes) equals a full-slice scan for 1, 2 and 3 hops on 18 cases. Disable runs confirmed all 11 guards load-bearing (red with the guard removed, green restored), including: per-dimension witness refusal, thought-delta refusal for another incarnation, the escape report, canonical-row retention, `Ordering` mismatch refusal, and incremental rebind ≡ full preparation (which needed an expansion fixture with incoming edges before its escape re-resolve guard could fail).

### 8.3 Cost per representation, 65 536 identities, focus 256 (µs unless noted)

| Fixture | facts | sorted run build | trie build | CSR prepare | tile from CSR | direct/run per eval | CSR per eval | tile per eval | break-even reuses (CSR / tile) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse, 3 preds, degree 2 | 392 238 | 29.4 ms | 126.4 ms | 341.7 | 1.5 | 8.49 | 3.33 | 5.08 | 76 / 116 |
| sparse, 3 preds, degree 8 | 1 561 508 | 236.0 ms | 1 350.6 ms | 1 187.4 | 3.0 | 114.73 | 5.58 | 5.74 | 11 / 11 |
| dense, 1 pred, degree 48 | 2 862 756 | 483.2 ms | 2 118.2 ms | 955.1 | 6.4 | 565.53 | 12.38 | 4.53 | 1.8 / 1.8 |

Per-eval figures are a 3-hop scoped reach in source-slot mode. Break-even counts are **arithmetic** from those medians (prepare ÷ per-eval saving), not separate measurements. Amortization was measured at N = 1, 8, 64, 1 024, 65 536 and 1 000 000; direct-arm runs above 20 s were capped and extrapolated, and are labelled CAPPED in the log.

- **Escape identities dominate when degree grows.** At degree 8, reporting escapes as identities costs CSR 50.4 µs per eval; as source slots, 5.6 µs. Materialize identities only when the next frontier needs them.
- **Tile vs CSR is a density question.** The tile wins only on the dense relation (4.5 vs 12.4); on sparse ones CSR wins (3.3 vs 5.1). Neither is a default.
- **Bytes per focus:** binding 3 072 B; CSR 18–66 KB; tile 8 KiB per predicate.

### 8.4 Focus churn: incremental rebind vs full re-preparation (µs)

| replaced slots k | sparse d2 | sparse d8 | dense |
| ---: | --- | --- | --- |
| 1 | 41.8 / 357.2 | 223.9 / 1 657.7 | 308.2 / 892.9 |
| 16 | 60.4 / 433.1 | 298.1 / 1 985.2 | 752.6 / 744.9 |
| 128 | 156.5 / 310.7 | 850.4 / 937.7 | 970.3 / 833.2 |
| 256 | 344.3 / 305.3 | 1 238.4 / 971.3 | 1 011.0 / 644.9 |

Incremental rebind (forward seek for changed slots, old escapes re-resolved against the new focus; no reverse index) wins for small k and loses at k = 256; on the dense relation it loses from k ≥ 16. A rebind policy needs the crossover, not a fixed choice.

### 8.5 Thoughts over one shared tile

Per-thought delta: 32 B as sparse edits, 16 KiB when lowered to executor planes. Per-eval cost over 1 / 8 / 64 concurrent thoughts: 2.61 / 2.66 / 2.79 µs (sparse d2), 8.04 / 8.14 / 8.28 (d8), 5.27 / 5.56 / 5.51 (dense). The shared tile is stored once; no thought copies it.

### 8.6 PathMap / MORK: what they do and do not buy here

- **Prefix seek is real.** Direct traversal over the trie beats the sorted run on sparse relations (7.0 vs 8.5 µs; 100.9 vs 114.7 µs), because a prefix seek reaches the objects without an indirect fact lookup. It loses on the dense relation (1 184 vs 566 µs).
- **Build and bytes cost more.** Trie build is 4–6× the sorted run, and its keys are at least 3× the run's bytes, because keys copy p, s and o. Node overhead is not measurable under MORK's allocator.
- **`restrict` yields the focus facts exactly, without copying shared nodes**, but takes 0.45–1.44 ms (slower than CSR preparation on the sparse fixtures) and its result still needs slot translation before execution.
- **Postings:** zipper merge 2.32 µs vs sorted merge 0.41 µs vs mask-risc And+Count 0.082 µs over prepared membership planes.
- **MORK pattern matching was not needed** for these fixtures and was not exercised; nothing here measures it.

### 8.7 Places: zero-copy focus mask vs register copy

On 65 536 places with a focus of 256, a bounding-box + class count through mask-risc over the canonical lanes takes 20–24 µs with 0 B copied, while loading the focus's lanes into a 3 072 B register and counting there takes 0.78 + 0.38–1.27 µs. The canonical-lane path is slow because the `under` gate is word-granular: 228–235 of 1 024 words are live for 256 rows. This is an open tension between zero-copy and a register copy, not a decision; bit-granular gating would remove it.

### 8.8a Joins over heterogeneous relations

The same preparation idea, applied to a shared-variable join between two datasets, is in the Schnellverschluss plan §7b: a versioned relationship table lays R and S out at the `y` slot, so the leapfrog intersection is one mask-risc `And` and each side reseals on its own. Measured there: prepare 2.7 ms vs PathMap 14.7 ms and MORK 22.9 ms at n = 50 000; bag count 36 µs per query; cyclic bodies not covered.

### 8.8 Not covered, and blockers

Lance-backed `DurableRef` and real stable-row-id seals (the probe's seals are integers); cold-cache and page-fault measurement; MORK matching; any gather-admission ruling. The probe answers §6's economics for in-memory, warm, single-machine fixtures only.

**Stop condition:** any design that requires copying canonical map facts per thought, using RelAlg as a table store, or allocating new CE64 bits to identify MapId/MapPaletteId has chosen the wrong boundary.

**Short form: RelationshipMapStore stores. OGAR types. ThoughtMapPalette selects. BIND borrows. RelAlg describes. Quack lowers. Mask-RISC executes. CE64 remains opaque.**
