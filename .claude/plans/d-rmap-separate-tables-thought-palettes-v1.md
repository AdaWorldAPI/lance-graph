# D-RMAP-1A: RelationshipMap tables, borrowed projections and thought palettes

> **Status: PROPOSAL / ARCHITECTURE ONLY (2026-10-19).** Clarifies [D-RMAP-1](d-rmap-implicit-relationship-register-v1.md). This amendment **supersedes** earlier suggestions that CE64 P could identify a relationship map. No code, persisted data format, wire/ABI change, opcode or performance claim.
>
> **Operator ruling for this proposal:** RelationshipMaps live in separate explicit tables, expose a zero-copy map of existing indexed values, and are not stored inside RelAlg. Each thought selects a *palette of global RelationshipMaps* using another independent table. Nothing is crammed into CausalEdge64.

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

**Stop condition:** any design that requires copying canonical map facts per thought, using RelAlg as a table store, or allocating new CE64 bits to identify MapId/MapPaletteId has chosen the wrong boundary.

**Short form: RelationshipMapStore stores. OGAR types. ThoughtMapPalette selects. BIND borrows. RelAlg describes. Quack lowers. Mask-RISC executes. CE64 remains opaque.**
