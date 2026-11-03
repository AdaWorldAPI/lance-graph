# D-DRAM-0: DR address memoization and adaptive Abakus masking lens

> **Status: ARCHITECTURE / PRE-REGISTRATION ONLY, 2026-11-03.** This PR adds no production or lab executable code, dependency, register, storage format, public API, opcode, planner switch or measured speedup. **Not merge or implementation authorization.** It maps the operator's two ideas to existing owners, names required falsifiers and preserves all outstanding EXO decisions.
>
> **Operator thesis:** (a) remember **already resolved SoA addresses** as append-like *receipts within the current witnessed DR binding*, without materializing/copying their values; (b) use the existing binary Abakus / SIMD facade under dynamically scoped ClassView/Quack lenses **where the algebra benefits**, without building a second mask engine.
>
> Distinguish **address memoization** from payload gather, **bit-sliced arithmetic** from Boolean conjunction, and a **proposed dynamic lens** from an already shipped 5×24 kernel. Evidence labels are normative: VERIFIED-IN-CODE, MEASURED-LAB, PROPOSED, GATED, UNVERIFIED.

## 1. Why this plan exists

The D-GLOBAL-WHERE P0/P2 discussion confused several independent operations: mapping identities into DR tiles, selecting qualifying identities with WHERE, resolving demanded properties through ClassView, locating canonical SoA bytes, and executing an already-bound mask algebra. Expensive re-routing or cold source reads do **not** imply that values should be copied into a compact per-query cache.

The guiding invariant is **DR remembers where; ClassView names what; Quack describes which algebra; ndarray executes; canonical SoA keeps the values.** A read-only source locator is not relationship truth. A previously loaded source address is not a new identity, field value, or durable map revision.

### 1.1 One flow, different lifetimes

```text
ClassId -> ClassView + schema/permission witness -> WideFieldMask (fields demanded)
                     |                    |
          Quack WHERE/DT WHEN        source facet/layout contract
                     |                    |
         DR population/relation mask + sealed Frame/MapTable
                     |
       DR tile/local ordinal, already present in existing mapping
                     |
       [optional, leased] remembered physical/source locator receipt
                     |
          canonical SoA borrowed read + advisory lookahead
                     |
       Quack / mask-risc -> ndarray::simd existing mask words
                     |
           optional bit-sliced Abakus threshold lens
                     |
                COUNT / EXISTS / other typed terminal
```

Read stages are not always sequential: a MAP-only COUNT can terminate before source value resolution; first evaluation of a value predicate legitimately reads source values. Fields can be physically packed (including proposed 4×8-bit words), without implying that property ordinal, physical SoA lane, predicate count, DR tile width or source row address are interchangeable.

### 1.2 Semantics that must remain separate

| Name | Meaning / authority | Not |
| --- | --- | --- |
| Global ID | stable typed identity, outside a local tile | a Lance row address |
| DR Frame + MapTable | versioned global ID ↔ tile/local binding and replay; one route authority | a second source locator map or value copy |
| DR relation / population mask | bit-per-local-position support or WHERE survivor, under semiring and witness | field-presence mask; arbitrary SQL truth |
| ClassView / WideFieldMask | logical field basis, demanded fields, stable positions; >64 possible | row-address mask or physical column ordinal |
| Source locator receipt | validated route from a *bound* local position to canonical storage view, valid only under source/index/scope lease | new globally authoritative hash index; raw pointer persisted across remap |
| Lance RowAddrMask | physical source-row eligibility for Lance 13 scanner | DR mask, unless attested coordinate translation is proven |
| Abakus significance planes | independent bit-sliced integer counters/thresholds over *the same* selected positions | a way to mix row identity with carry position |

The observed WHERE criterion-count working distribution is 1:34%, 2–4:46%, 5–16:14%, 17+:6%. It is operator-supplied and **not a measured selectivity distribution**. A 64-field ClassView, 32 physical SoA lanes, up to 256 WideFieldMask positions, 256×256 relation tile and 65,536-position DR context are different dimensions. More than 64K identities requires lawful composition across multiple DR relations/tiles, not automatic global dataset paging.

## 2. Primary law: a memoized locator, never a gathered value

**Proposed meaning of “append the address in DR”.** When a canonical SoA address has been resolved for a DR local position, a focus-scoped, version-attested *receipt* for that existing position becomes reusable by subsequent reads. The receipt can be sparsely populated (miss → resolve → memoize; hit → borrow), and prefetched positions may be prepared ahead of the consumer.

**Do not append the source address to `globals` of `LocalMap`.** That vector is local → *global identity*, and `MapTable`'s append/prefix history is about changed coordinate membership. Mixing physical locators into it would change its semantics and corrupt old-frame replay. Before implementation, the DR/RelationshipMap owner must identify the existing focus/binding lease to host **disposable locator metadata**, with no second identity or coordinate authority. A receipt is derivative of a single already-attested canonical map.

### 2.1 Conceptual receipt (not a proposed Rust type or ABI)

```text
Key:   DR map/frame identity + tile/local ordinal
       + source dataset/branch/version + index/row-address seal
       + ClassView schema/facet layout revision + scope/authorization
Value: validated source location / borrowed view descriptor
       + its live source lease and optional prefetched-line hint
```

- **No bare pointer longevity:** a resident `&[T]` can be borrowed only while the canonical owner/Arrow-buffer lease lives. mmap, page-cache, decompression and storage reads are not presumed zero-copy.
- **No duplication:** subsequent methods sharing the *same* fully witnessed binding may reuse the resolved locator, not copy its payload. Changes in permission, projection or source location must not inherit a stale descriptor.
- **Versioning:** old DR frames remain replayable for coordinates. They can replay values only while the *corresponding* historical SoA snapshot and index seal are available; otherwise typed refusal. Epoch counters or frame-number equality alone cannot certify a foreign source row.
- **Mutation:** a value-only change can leave the DR relationship binding intact, but may invalidate a value-derived predicate and the affected value borrow. ADD/RETIRE of a relation changes support/delta, not global ID identity. Compaction, overwrite, remap, fragment split or stable-row-id policy changes require revalidation/rebind/refusal.
- **Eviction:** locator receipts may be forgotten independently of sealed relationship history. Losing a receipt causes a slower legal resolve, never lost relational truth or an empty answer.
- **Concurrency:** one admitted lease/witness for concurrent readers; no ABA from focus-slot reuse, speculative Thought-Git incarnation, scope changes or map-palette revisions.
- **Selective reads:** Lance 13 `Scanner::with_row_addr_prefilter`, `Scanner::project` and indexed/range reads are existing capabilities. Source ingress masks may be copied or reserialized; **do not label that seam zero-copy**. D4 (single DR ordinal→`_rowid` authority) and S-INGRESS remain gated.
- **Prefetch:** optional source-line or future-focus lookahead, never equivalent to materializing a new value array; test mispredictions and resident limits. No prefetch if the operation is MAP-only.

### 2.2 Where actual gathered payload is lawful

Default: MAP-only, or direct canonical borrow with optional attested source read and prefetch. A separately allocated facet cache is **not** a peer of borrowing in the general optimizer. D-DRF-1 `5 permits a disposable, witness-keyed *foreign-key/foreign-ontology* cache only for an explicitly admitted, repeated workload where measured whole-run gain exceeds build, allocation, eviction, invalidation and dirty-writeback costs. A numeric winner does not authorize bypassing the no-foreign-gather firewall (EXO-140 / EXO-46). The word `MaskOp::Gather` is an indexed mask operation, not automatically a copied SoA-value cache.

## 3. Abakus as an adaptive physical realization of Quack's algebra

**Verified library words:** ndarray PR #402 landed `ndarray::simd::half_add_mask` and `full_add_mask`, whose columns hold 64 independent bit-lanes and whose carries move to higher significance **within** a lane; `src/simd_masking_ops.rs` owns cross-ISA slice ergonomics. PR #390 landed `U64x8::sum8_u8_add_masked` for its specific masked byte-sum consumer. D-U9 PRs #389/#391/#393 provide research fixtures.

**Not verified as a public API:** `Abacus<BITS>` is a #402 lab fixture; its public type was deliberately deferred for lack of a consumer. The operator reports hand-rolled `ndarray::simd::5x24::*` kernels. **The exact 5×24 symbols/signatures, shipped status and intended bit-width are UNVERIFIED by this review.** Do not invent or import an API until the ndarray owner pins the actual file/symbol/commit. Do not infer that U9 numerical Abakus performance generalizes to Boolean masks.

### 3.1 Dynamic lens, not a new VM

- ClassView/DT64 resolves the *logical* field basis and read footprint using `WideFieldMask`, with no semantic bits hidden inside a presence mask.
- Quack/RelAlg owns typed predicate normalization, SQL TRUE/FALSE/UNKNOWN, Boolean / Bag choice and WHERE semantics. The execution consumer receives a **sealed, valid population mask** over one coordinate universe.
- The existing mask-risc/ndarray facade handles AND/OR/XOR/ANDNOT, mask compares, COUNT, and appropriate reductions. Do not reimplement those in a Quack loop or handwrite another vector backend.
- Only when the expression actually requires **arithmetic over many predicate masks**, such as “at least k of n conditions”, may bit-sliced `half_add_mask`/`full_add_mask`, vertical masked sums or a verified 5×24-family primitive be tested as a *physical realization*. Preserve identical lane order and tail bits.
- Scope width is chosen from the **actual demanded fields, admitted relation masks and selectivity**, not by materializing an entire 64-property or 256-predicate volume. A sparse predicate/threshold descriptor is an option, not a new data-storage contract.
- Possible 4×8-bit packed property/threshold routing is **hypothesis only** until codec, signedness, null/absence and overflow/refusal semantics are specified; do not silently reinterpret four bytes as four independent semantic fields.
- No new F### method ID, ClassId interpretation, DT64 algebra registry, mask-risc opcode or ndarray public type is authorized here. A lawful chosen realization would be admitted by current DT64/Quack ownership once supported by owner decisions.

### 3.2 Strong negative controls

- AND of 2–4 existing Boolean masks should usually lower directly to `mask_and`; proving Abakus correct does not justify extra carry operations here.
- Bag multiplicity, DISTINCT, SQL NULL/Kleene UNKNOWN, absent field versus present zero, negative U9/Palette512 sentinel and overflow must **not** be silently collapsed into Boolean thresholds.
- A “k-of-n” aggregate counts satisfied predicates **per identity**, not DR tiles, physical byte slots or join derivations. Bag counts must retain multiplicity if requested.
- A new kernel cannot bypass mask-risc/Quack ownership to get a better microbenchmark. First check existing facade and fixed-cost terminal options.
- Abakus is not a hash table, row-ID mapping, storage read mask, prefetch mechanism or reconstruction of source values.

## 4. Existing shipped evidence / reuse map

| Evidence | Grounded result | Implication |
| --- | --- | --- |
| `dt64::dr::{LocalMap,MapTable,Frame}`, lance-graph #1905/#1923 | versioned coordinate table and old-frame replay, identity-only `LocalMap` | keep authority; seek existing receipt lease, not a new global→local table |
| `contract::class_view::{FieldMask,WideFieldMask}`; `dt64::{ValueDemand,MethodBinding}` | demanded-field masks, borrowed mask words, map-only versus value demand | separate sparse logical projection from physical layout and WHERE |
| D-RMAP-1A; D-DRF-1; D-DRAV-1 | separate RelationshipMap owner, zero-copy default, admitted exceptional foreign cache, source leases | binding policy outranks microbenchmark winner |
| lance-graph #1573/#1575/#1633 | Lance 13 masked source reads already measured; selective range path with ordered-lane attestation | reuse, don't recreate selective reader |
| lance-graph #1788/#1789 | DR↔Lance wiring, S-INGRESS and D4 open; chunk-touched byte predictor failed | do not claim source zero-copy or infer physical bytes from logical rows |
| lance-graph #1943/#1946 | in-memory DR routing/fold, corrected DuckDB answer parity | no Lance I/O in those measurements; re-route per query is an avoidable cost hypothesis |
| lance-graph #1945 | top-down vector×tile beats reverse gather-shaped arm across 54 lab cells, with stated falsifier; no storage I/O | evaluation direction matters; C builds its own chunks over DR coordinates |
| ndarray #402 | half/full adder facade shipped; bit-sliced public counter deferred | choose a real Quack threshold consumer before new public type |
| ndarray #390 | vertical masked byte sums shipped, with independent per-ISA realizations | verify semantics and existing words first |

Sources:
- https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/dr.rs
- https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-contract/src/class_view.rs
- https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-dt64/src/lib.rs
- https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-rmap-separate-tables-thought-palettes-v1.md
- https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-dr64-focus-lease-foreign-cache-v1.md
- https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-dr-address-first-value-demand-v1.md
- https://github.com/AdaWorldAPI/ndarray/blob/master/src/simd_masking_ops.rs
- https://github.com/AdaWorldAPI/ndarray/pull/402
- https://github.com/AdaWorldAPI/ndarray/pull/390

## 5. Pre-registered, *separable* falsifiers

### L: locator memoization, no payload copy

| Gate | Fixture / independent oracle | Refusal or cost requirement |
| --- | --- | --- |
| L1 | Same frame/map, two distinct methods select overlapping properties at 64 / 256 local positions; oracle reads canonical SoA directly | second read reuses a validated locator, result equal, no copied values |
| L2 | Miss→hit, hit→evict→miss, scattered and clustered canonical 512-byte rows | correctness unaffected; count lookups, actual loads, allocations, cache lines |
| L3 | Changed values only, unchanged relationship, then value-dependent predicate change | DR identity remains; dependent derived mask refreshed/refused; other masks reusable |
| L4 | ADD, RETIRE, old frame replay, map pure extension versus replacement | no stale association, old frame requires correct historical source |
| L5 | Compaction/overwrite, stable-ID on/off, changed index seal and dataset version | old physical locator never silently reused |
| L6 | Foreign frame, scope change, palette reassignment, reused focus slot, revoked source lease | typed refusal; zero cross-scope bleed |
| L7 | Two concurrent readers with same witness and one invalidating writer | bounded locator construction; no ABA / dead borrow |
| L8 | No prefetch / lookahead 0/4/8/16/32/64, mispredicted lookahead | no reordering or payload copy; compare whole-run cost and memory |
| L9 | Full / Lance ALLOW / Lance BLOCK / attested range on 0.1%,1%,10%,50% scattered and clustered | account mask ingress and bytes; retain D4 refusal |
| L10 | 1M and 10M corpus, multiple DR256 relations across address domains | identity and Bag/Boolean oracles; no giant dense matrix, no new paging authority |

### A: adaptive Abakus lens

| Gate | Oracle and disable | Required verdict |
| --- | --- | --- |
| A1 | Native AND/OR/COUNT vs Abakus on k-of-n: n=1,2,4,5,16,24; k=0,1,half,n,n+1 | exact lane-by-lane threshold, no assumed win |
| A2 | Adversarial masks: zero, full, alternating, one lane, tail, sparse, dense, random | cross-backend parity and no carry into neighboring lane |
| A3 | FieldMask 64 vs WideFieldMask 256, 32-lane physical SoA mapping, noncontiguous fields | same logical projection with no field/row ordinal alias |
| A4 | Packed 4×8 threshold codec experiment only after encoding attestation | signedness, saturating vs exact, missing/present zero and overflow pinned |
| A5 | SQL NULL/UNKNOWN, absent property, Boolean set vs Bag multiplicity | never invent 0 from missing, nor equate witness count and support |
| A6 | Replace Abakus with naive per-row adds, then deliberately misplace one carry | independent oracle catches changed answers |
| A7 | Full Quack lowering vs isolated SIMD-only microkernel, cold/warm, all ISA tiers | calculate end-to-end benefit including dispatch; NO-EXISTING-KERNEL is allowed |

### C: combined recall + dynamic lens

C1: one ClassView lens, 3 of 64 demanded fields, 1–4 WHERE criteria, reused DR frame, first value read vs repeated hit vs changed source. Compare **MAP-only**, borrow, prefetch-borrow, and **authorized exception** only; no generic gather planner.

C2: fixed DR frame with multiple relation compositions beyond 64K. Vary predicate count, result cardinality and number of DR domains independently; oracle must not share lookup/index code.

C3: use DuckDB's published engineering mechanisms as **learning evidence** (selection lifetime, late materialization, join direction, index pruning) alongside semantic and performance parity. Do not invent feature parity because answers match.

C4: if any stage needs DR ordinal→Lance `_rowid` without an attested single authority, report `BLOCKED_D4`. Likewise report `S_INGRESS_COPY` on masked Lance ingress. Never force a full scan under a hidden failure, call it zero-copy or silently materialize.

## 6. Cost accounting (no free amortization)

Report **separately**, in absolute units and by query count Q:

```text
class/schema resolve | WHERE | DR coordinate map preparation | DR route reuse
locator miss/hit | mask construction | mask ingress copy
physical Lance rows/bytes/pages/ranges | SoA borrow loads | prefetch
optional exceptional gather build/eviction (separate admitted arm)
bit-sliced vs native Boolean fold | terminal | total wall | peak bytes
```

Do not hide a cold mapping in a warm number, count result rows as physical bytes, call a scan mask a copied payload, or assert CPU-cache residency from a source pointer. Stratify mask locality, field count, predicate count, selected cardinality and repeated-use Q. Statistical spread and runs required before any performance verdict.

For an optional receipt, the break-even hypothesis is saved repeated locator resolution and avoided latency **minus** receipt insertion, witness validation, indirection, cache footprint, invalidation and prefetch errors. For Abakus, compare actual typed expressions, not invented workloads that force carries to win. Keep the hot-path 1 ms failure gate.

## 7. Execution order, owners and stopping points

| Step | Deliverable | Authority / status |
| --- | --- | --- |
| D-DRAM-0 | this map + board entry + EXO-50 message | docs only, operator can review |
| D-DRAM-1 | symbol & caller census, exact 5×24 ndarray signature, existing focus lease source, one DR map owner | READ-ONLY; no implementation |
| D-DRAM-2 | isolated memoized locator receipt lab with L1–L8; only existing owned source/leases | GATED on EXO-46 + EXO-89/EXO-151 admission; no new identity table |
| D-DRAM-3 | Abakus Quack threshold bench A1–A7 with existing public words | lab proposal; SIMD/Quack owners; no public type without consumer |
| D-DRAM-4 | combined C1–C4 and Lance source-read cost | D4/S-INGRESS remain explicitly blocked for end-to-end binding |
| D-DRAM-5 | owner review, measured decision per ability: REUSE / WIRE / DEFER / REFUSE | no production action implied |

**Owner registers:** EXO-50 is an append-only coordination bus, **not** an approval. EXO-46 (binding admission), EXO-89 (RelationshipMap authority), EXO-151 (focus lease/foreign cache), EXO-121 (ClassView/WideFieldMask), EXO-140 (foreign gather classification), EXO-108 (Bag vs set), EXO-112 (contract/ndarray dependency shape), and D4/S-INGRESS (#1788). DT64/Quack own semantic admission; ndarray owns facade and architecture backend; Lance owns physical source reads. EXO-50 links are handoffs, not new runtime authority.

**STOP immediately** on: new global→row-id store, fresh SIMD backend in Quack, changed field-presence meaning, duplicate physical payload by default, source locator surviving a dead lease, unverified 5×24 API claimed as shipped, unauthorized Gather, new ClassId/F###, or production patch while this remains a docs-only proposal.

## 8. Board / literature links

- Board: [D-DRAM-0 entry](../board/entries/2026-11-03-d-dram-abacus-address-memoization.md), [STATUS_BOARD](../board/STATUS_BOARD.md), [INTEGRATION_PLANS](../board/INTEGRATION_PLANS.md).
- EXO-50 bus: https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id
- EXO-46: https://linear.app/exo-red/issue/EXO-46/a1-decide-counted-globalfocus-binding-membrane-for-production
- EXO-89: https://linear.app/exo-red/issue/EXO-89/d-rmap-1-1a-implicit-relationship-register-and-separate-thought
- EXO-151: https://linear.app/exo-red/issue/EXO-151/d-drf-immutable-dr64-focus-leased-foreign-cache-sparse-writeback
- DuckDB's public engineering references: https://duckdb.org/internals/vector , https://duckdb.org/2024/11/14/optimizers , https://duckdb.org/library/selective-late-materialization/ ; shipped features and research prototype must be distinguished.
