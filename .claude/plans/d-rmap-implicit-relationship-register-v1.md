# D-RMAP-1: implicit RelationshipMap and focus-relative relational registers

> **CORRECTION / PRECEDENCE (2026-10-19):** [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md) is the controlling ownership and binding contract for this proposal. **RelationshipMaps are independently stored explicit tables with borrowed zero-copy views; RelAlg does not own or store them; each thought selects a palette of maps in a separate ThoughtMapPaletteBinding table. No map ID, palette selector or map-routing semantics are added to CE64.** Any ambiguous P/W interpretation or combined map/RelAlg ownership below is superseded by D-RMAP-1A.

> **Status:** PROPOSAL / architecture review, 2026-10-19. This PR adds documentation only. It neither changes the wire/ABI nor licenses new runtimes, gathers, opcodes, persisted formats, or performance claims.
>
> **Operator thesis:** The wire format must not know whether it is comparing apples from China or Senegal. The awareness belongs to a relationship map.
>
> **Principle:** The world owns meaning. The RelationshipMap binds coordinates. The register executes.

Read first: [fold execution laws](../knowledge/fold-execution-laws.md), [epistemic frontier](../knowledge/epistemic-frontier-capstone.md), [H14c](d-holo-h14c-thought-first-v1.md), [H14d](d-holo-h14d-logic-mask-read-v1.md), [CausalEdge64 layout](../../crates/causal-edge/src/layout.rs), [CausalEdge64 ISA](../../crates/causal-edge/src/isa.rs), [D-SRAM](souffle-ram-64-thought-stack-v1.md), [RelAlg v1](relalg-v1.md), [WCOJ exploration](../board/entries/2026-10-16-coresearch-wcoj-leapfrog.md).

## 0. The question

Existing 64 thoughts cheaply vary epistemic state over a **stationary identity population**. H14c reads pre-co-addressed operands, refusing foreign coordinate hops. A thought is an epistemic overlay, not a local graph copy.

**New hypothesis:** 64..256 global relationship indexes can share a bounded 256-item focus mapping, then expose cross-corpus relationships as register-local operands to Quack / RelAlg / Datalog / NAL / MeTTa. Repeated local folds may amortize cold global seeks and traversal without requiring global CSR, forcing all data into HHTL16, or copying payloads.

The critical addition is a **coordinate binding delta**, independent of 64 speculative thought deltas. The global fact graph remains canonical; the identity-to-local-slot lens moves.

This proposition does **not** claim that all graph traversals, joins, cache misses, high-degree expansion, or materialized outputs disappear. Binding, indexing and cache maintenance are charged.

## 1. Contract: separate meaning, binding and wire

| Owner | Must know | Must not silently become |
| --- | --- | --- |
| Canonical semantic facts / OGAR | Globally stable identity, predicate semantics, arity, argument order, multiplicity, provenance, version, ACL, method/ActionDef contract | A shape-specific physical adjacency scheme |
| RelationshipMap / BindingFrame | Typed index routing, map slot -> global predicate, source/target palette slot -> global identity, ordering/seal witnesses, authorization, optional local relation tiles | A second semantic evaluator or new authoritative fact store |
| CE64 / R2IL / Mask-RISC | Opaque code bytes, active operands, inference/truth arithmetic, Boolean/population folds | Apple geography, global identities, predicate names or transport serialization |

**No wire-field reinterpretation without an explicit ClassView/rail/generation reading.** The 64-bit CE64 layout already carries S (0..7), P (8..15), O (16..23), epistemic/inference fields, W (53..58) and EpistemicState5 (59..63). The following are design readings, not a migration:

- S/P/O remain **opaque local palette codes under their declared CE64 reading**; **P is not a RelationshipMap ID or routing selector**. The separate thought-to-map-palette table determines which global map is bound. Local wire codes are neither global ontology IDs nor permanent foreign keys.
- W is **not** a map ID, map-palette selector or an evidence root. The 2026-10-17 epistemic decision calls W a thought slot (0 givens, 1..63 hypotheses); the current layout source still documents a witness-corpus handle and the capstone records the conflict. Existing version-gated typed refusal must resolve the ambiguity before any new reading is admitted.
- The local semantic address requires an **external frame witness**: (frame identity + generation, thought incarnation, S, P, O). Two byte-identical CE64 values under different frames are NOT semantically equal; the frame has to reconcile canonical identities first.
- For example, S=17/P=4/O=29 can denote two apples under one frame. Their origins China/Senegal are attributes resolved through the map's typed relations only when requested. CE64 need not encode geography or decide whether these apples are comparable.

No addition to CE64, NodeGuid, 512-byte row canon, ClassView ABI, or hot JSON/string representation is requested.

## 2. More than one legal physical data shape

HHTL16 must remain a high-value locality/prefix carrier where hierarchy is lawful. It need not become the obligatory physical encoding for multi-parent global ontologies or arbitrary foreign object relationships.

| Carrier | Legal specialization | Caveat |
| --- | --- | --- |
| HHTL16 / sealed prefix bounds | Co-addressed hierarchy and Morton locality | Inherited geometry does not prove ontology facts |
| Indexed TrieMap view | Ordered global (predicate, subject, object) / alternate permutations; seek, leapfrog, intersection | Trie levels can be **views** over sorted keys, not a separate allocated trie or a materialized CSR |
| SoA lanes / resident masks | Direct co-addressed fields, known functional key, H14c | No hidden gather inside a mask fold |
| Local FocusPalette / RelationTile | Repeated use of 256 local S/O slots across 64..256 relation maps | Membership map is not full adjacency; high fanout must escape to global index |
| Sparse relation/thought overlay | Changed facts, temporary local bindings or hypotheses | Do not confuse changed global facts with changed view coordinates |

The **same logical relation** may have multiple indexed physical projections. BIND attests usable ones; BUNDLE selects between direct seeks, prefix bounds, selective caching and local folds according to a fully charged cost model. Do not create a new per-carrier semantic AST or executor.

## 3. Explicit tables inside the existing data family

**Ownership correction:** The **RelationshipMap** indexes and tiles belong to a separate store/table family, and **MapPalette** plus **ThoughtMapPaletteBinding** are distinct typed lookup tables. They are **not** part of RelAlg or of CE64. See [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md) for the controlling table schema and borrowed zero-copy binding contract.

These names are a proposed **logical schema**, not new binary layouts or a second database:

| Table | Responsibility |
| --- | --- |
| RelationCatalog | Canonical predicate identity, arity, type constraints, multiplicity, direction, semantic permissions, supported physical views |
| RelationFacts | Canonical subject/predicate/object or n-ary argument identities, claim identity, source and evidence roots, temporal validity, ACL |
| RelationOrder | Sealed sorted/index permutations of global facts, referencing canonical data instead of copying value payloads |
| **RelationshipMap** | Independent typed table linking global map identities to sealed index views; may be borrowed by many thoughts |
| **MapPalette** | Independent map-slot-to-MapId codebook, up to 256 slots, reusing canonical map tables without a copy |
| **ThoughtMapPaletteBinding** | Independent thought-slot + incarnation + frame to map-palette mapping; selectors and generation held *outside CE64* |
| FocusBinding | Frame identity, map-codebook, source/target palettes, generation, slot incarnations, fact/index versions, auth scope and seal attestations |
| RelationTile | Optional cache: local source slot -> target presence bits, keyed by the full binding witness; disposable projection |
| RelationDelta | Fact add/retract/revise receipts, cache invalidations, palette changes and thought overlays, typed separately |

A schema/DTO router resolves **declared** relations and capabilities from the OGAR/ClassView/codebook boundary. It must not inspect arbitrary Java/Python object graphs to decide hot physical semantics, and must not transcode internal data to JSON. Consumer APIs may be simple because the hot planner does not need to see their DTOs.

Methods participate as **typed relations** such as implements, realizes, calls, depends_on and authorizes. Selecting an ActionDef is pure; executing an effectful action remains a separately authorized, ordered, idempotent/transaction-aware operation.

## 4. Focus-relative graph register

Let R[p] be a globally indexed, canonical Boolean relation for typed predicate p. Let Bs and Bo map chosen global source/target identities into 256-slot local palettes. The bounded local adjacency is denoted:

    A[p,F] = Bs[F] * R[p] * transpose(Bo[F])  // Boolean semiring

This is a **mathematical denotation only**. Do not build a dense global R[p]. A TrieMap cursor or witnessed ordered permutation can resolve selected row neighborhoods lazily; a dense tile is optional. Nonresident target identities **must remain visible as frontier-expansion candidates**, rather than being truncated to 256.

The source and target palettes need not be the same population. A valid ordinal in one palette cannot be dereferenced in the other without a typed, witnessed binding.

Memory economics for Boolean projections, excluding map/index and metadata:

| Operand | Bytes |
| --- | ---: |
| One 256-bit participation mask | 32 |
| 64 maps x 256 focus slots, one bit each | 2 KiB (NOT adjacency) |
| One complete 256 x 256 adjacency matrix | 8 KiB |
| 64 full adjacency tiles | 512 KiB |
| 256 full adjacency tiles | 2 MiB |
| One 256 x 16-byte GUID palette | 4 KiB |

A 16 x 16 adjacency tile is 32 B; it is a useful bitwise block but **cannot** answer exact bags, source provenance or arbitrary multi-variable joins from one Boolean bit alone. Demand semantics must be decided *before* lowering.

## 5. Thought x RelationshipMap x Focus, without a giant tensor

Existing proven H14c hot form:

    Thought(t)[T_a[i] : T_b[i]]  // same sealed coordinate i

Proposed possible input to that computation:

    bind(Frame F, Map p, local slots s/o, version, access) -> legal local operand(s)

The proposed tensor T[thought][map][source][target] describes **potential coordinates**, never an eagerly allocated state array. **The map axis is chosen via ThoughtMapPaletteBinding, external to CE64 and RelAlg, using borrowed RelationshipMap tables.** The thought's W annotation alone is never enough to identify the frame/map palette. Thought 0 is immutable givens. 1..63 borrow shared global facts and cached relation projections, then apply sparse epistemic changes.

Keep four independent delta classes:

1. **Fact delta:** globally asserted/withdrawn relation or attribute, with its own provenance.
2. **Binding delta:** local slot changes or map codebook changes, with no implied global fact mutation.
3. **Cache delta:** refreshed tile rows or invalidation, never authoritative truth.
4. **Thought delta:** speculation, revision or retraction under a thought incarnation, never overwriting givens.

For Boolean existence only, a thought view might be represented as:

    R[t,p,F] = restrict_F((R[0,p] union add[t,p]) minus retract[t,p])

That expression is **not** an adequate definition for bags, count-distinct, evidence independence, negation or source-authorization changes. Those require typed fact identities, provenance, DRed/replay or explicit refusal. Thought-slot reuse cannot alias an old world merely because W repeats.

## 6. Compiler and cache boundaries

Use [PARSE -> BIND -> V4 -> BUNDLE -> EXECUTE](../knowledge/fold-execution-laws.md). The new work, if warranted by tests, belongs at **BIND plus the independently owned RelationshipMap tables** (and not RelAlg storage): prove identity and predicate meanings, row-universe compatibility, mapping generation, source index order/seal, version and permission scope. BUNDLE may then choose direct co-addressed fold, ordered index seek, sparse local row cache, dense relationship tile or factorized evaluation.

The [2026-10-16 WCOJ exploration](../board/entries/2026-10-16-coresearch-wcoj-leapfrog.md) explicitly marks foreign gathers and preimage hops as a **VIOLATION**, preferring the proven H14c/H14d co-addressed path. **This proposal does not reverse that ban.** A pre-fold global->local binding is only a candidate new admissible *membrane* and must be independently reviewed. Until accepted, such operations are a typed refusal, not a hidden foreign-lane dereference. No new R2IL op without proving the existing execution vocabulary insufficient.

Cache keys must include frame identity and generation, source and target palette incarnations, predicate-to-map mapping, global fact version, sealed index generation/digest, auth/tenant scope and thought generation where relevant. Stale mapping => refuse accelerated path. ACL checks occur **before** indexed results are made visible and before any payload load.

Caching costs:

    Direct(k) = k * Cost(global indexed seek)
    Reuse(k)  = Cost(bind) + Cost(optional tile) + k * Cost(local fold) + Cost(maintenance)

Charge all index build/maintenance, cold seeks and faults, frame binding, palette remapping, tile invalidation, auth checks, and terminal materialization. Cached wins are not guaranteed. If 240/256 focus identities survive a transition, retained slots might preserve local tile work; if renumbering is required, charge the permutation. For high-degree hubs or cross-focus edges, represent the external frontier faithfully or refuse the local-plan shortcut.

## 7. Falsifier matrix before implementation

Independent oracles: reference facts with exact stable IDs, a separate Datalog/set/bag evaluator, existing co-addressed H14c/H14d when lawful, and ordered-index direct seeks. All paths must answer the SAME logical demand.

| Gate | Test | Failure |
| --- | --- | --- |
| F0, opacity | Two frames with same local CE64 codes but different global identities; apple origins | Cross-frame codes treated as the same global fact or origin |
| F1, bit/layout | Existing golden CE64, H14c, H14d, Quack outputs; thought-W ambiguity | Wire changes, old bytes silently reinterpreted, new hidden opcode |
| F2, graph truth | Multi-parent ontology, high-degree nodes, disconnected graph, 2+2+2 cyclic triangle | Lost out-of-focus target, false positive semijoin, missing binding |
| F3, multiplicity | SET/EXISTS, BAG/COUNT, COUNT DISTINCT, walks vs distinct vertices, provenance | Boolean bit returned as exact bag or evidence history |
| F4, security | Cross-tenant reused slots, ACL changes, field deny, auth scope rebind | Any unauthorized payload load or stale authorized cache hit |
| F5, version | Append/delete/reopen/compaction/reseal, frame and thought-slot reuse | Previous map yields plausible but wrong result |
| F6, amortization | Cold/warm/hot maps, N from 10^2 to 10^9 logical facts, varied reuse counts | No reproducible crossover including all binding and indexing costs |
| F7, spatial delta | Replace 1/8/16/128/256 focus identities, both clustered/random | Delta reuse identity-wrong or slower than rebuilding without an honest fallback |
| F8, thought sharing | 1/8/64 thoughts over same maps with sparse conflicting facts | Full relation data copied per thought, givens mutated, evidence roots conflated |
| F9, compiler neutrality | Equivalent Datalog, RelAlg, Quack and ActionDef target-selection queries | Consumer-specific graph evaluator necessary or side effect executed by pure fold |
| F10, memory accounting | Track global perms, palette, tile, metadata, invalidation, network/disk I/O | 2 KiB / 8 KiB claims hide actual identities, adjacency or index storage |

Report total end-to-end cost, page faults/cache misses if measurable, bytes touched, time per emitted result and preprocessing amortization. A tiny in-cache tile benchmark alone is insufficient. A valid final outcome is **FALSIFIED** or **BOUNDED**; do not rescue losing results by changing test populations after measurements.

## 8. Proposed sequence, subject to operator authorization

1. **Architecture and legality:** settle whether a witnessed pre-fold BindingFrame is admissible under the no-gather firewall; identify the *existing* contract/OGAR type owning capabilities, and settle CE64 W reading through ClassView/version gating separately.
2. **Small reference bind:** resolve one global indexed relation onto up to 256 palette slots; no new runtime index, no payload copy, oracle/ACL/version checks first.
3. **Selective local tile:** compare direct seek, per-row cache and dense Boolean tile, with all costs charged. Demand exact egress semantics.
4. **Incremental spatial frontier:** preserve surviving palette slots, explicitly account for rebinding and out-of-focus neighbors.
5. **Thought palette + overlay:** separate ThoughtMapPaletteBinding chooses a subset/order of shared 64..256 global maps for each thought incarnation, with no CE64 map ID and no per-thought complete map copy; speculative changes sparse, givens immutable, evidence independent.
6. **Schema-routed consumer API:** Rust, Java, Python, Datalog, MeTTa, NAL, document and IAM consumers see typed relations, not HHTL, CSR, TrieMap or wire layout.
7. **Only if earned:** expose a witnessed RelationMap as a first-class bound operand to existing folds. No second evaluator and no convenience opcode without an insufficiency proof.

### Non-goals

No universal forced HHTL migration; no change to the 512-byte SoA canon; no new CE64 field or reinterpretation to carry a map palette; no RelAlg-owned RelationshipMap tables; no global CSR or per-thought graph copies; no runtime JSON/string hot paths; no reinterpretation of W as a RelationMapId; no unproved cross-population mask AND; no promotion of cache presence to semantic or causal truth.

**Decision requested:** Explore **implicit global relationship addressing through explicit typed tables**, with a local reusable register projection and independently amortized spatial/epistemic deltas. Keep the wire ignorant of the world's ontology.
