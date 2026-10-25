# D-DRAV-1 — address-first DR: independent map and value lifetimes, demand-driven hydration

**Status:** PROPOSAL / DOCUMENTATION-ONLY AMENDMENT (2026-10-25). No Rust, wire/ABI, storage format, optimizer rule, numeric method, benchmark, CI gate, or production implementation is introduced. This is a reviewable hypothesis and falsification plan, **not** an authorization to implement a new runtime.

**Scope and precedence.** Specializes [D-DRF-1](d-dr64-focus-lease-foreign-cache-v1.md) (focus residency, optional foreign gather, writeback), [D-RMAP-1A](d-rmap-separate-tables-thought-palettes-v1.md) (RelationshipMap ownership and attested binding), [D-NFC](d-native-first-dependency-correction-v1.md) / [D-NFR](d-native-repr-contract-v1.md) (native-first representation), and the [Quack storage portability ruling](../board/entries/2026-10-05-quack-storage-portability.md). None of their guards is superseded. D-DRF-1's present lab *implementation* packages an 8 KiB map and one gathered facet in a single Focus; that coupling is a probe choice, **not** the required production lifetime.

## 1. Thesis: an address map is a useful result before any value gather

DR64/DR256 should be considered an **independently reusable, sealed relational address structure**, not merely an intermediate buffer on the way to a local value register. Once global-to-local coordinates and typed relationships have been resolved, many subsequent operations can consume the map, masks, index permutations, or admitted relation slices without re-reading original values.

Four things must not be conflated:

1. **Topology / address semantics:** which typed source, target, predicate, and ordinal are related in a specified revision and scope. The DR64 u16 map is 4,096 × 2 B = **8 KiB**; the DR256 u16 map is 65,536 × 2 B = **128 KiB**. These are not 512-byte SoA row copies.
2. **Relationship support:** optional Boolean masks, slice views, adjacency/permutation-like descriptors or a typed DR64³ relation. Presence/support does not automatically prove truth, multiplicity, SQL NULL status, evidence independence, or numerical magnitude.
3. **Values:** canonical SoA rows and any disposable gathered facets / derived numerical artifacts. These may be needed once, repeatedly, or not at all for a particular operation.
4. **Physical location:** a storage-version-pinned translation from canonical identity/ordinal to addressable memory or storage extents. A 16-bit DR ordinal is **not** an NVMe logical block address or a globally stable physical row identifier.

The CSR/adjacency analogy is about **amortizing a sealed indexing structure**: resolve an irregular relationship once, then use it many times. It does not require converting every DR map into CSR, allocating a parallel graph, or replacing existing RelationOrder/index owners. Arithmetic coordinates (e.g. fixed-stride base + ordinal × row width) can be cheaper than following a stored permutation **only where a witnessed layout actually permits that formula**.

**Not a free-lunch claim:** inexpensive ordinal arithmetic does not remove cache misses, TLB pressure, page faults, decompression, network access, I/O amplification, or the cost of proving that a relationship/support view is complete.

## 2. Separate the lifetimes, not the semantic owners

| Representation or resource | Retain when | Drop / recompute when | Existing authority |
| --- | --- | --- | --- |
| Sealed map / sorted relation index | The versioned binding is reused | Its actual readers end, budget policy evicts it, or a different revision is requested | RelationshipMapStore / original owner |
| FocusBinding, palette, scope and index witness | Required to interpret local ordinals | The binding is retired; a new binding must be attested before reuse | D-RMAP-1A BIND, scope/Thought-Git witnesses |
| Optional Boolean support/DR slices | A complete, typed relation-only operation reuses them | Dependency changes or their derived-cache lease ends | Existing derived relation/view owner |
| Gathered i32/16-byte facets or other local payload | A costed, admitted repeated numerical path needs them | Last dependent reader ends; values are safely rehydratable | Canonical SoA owner + DRF derived cache |
| Canonical source snapshot / row-address seal | Any borrowed reader, descriptor, historical answer or pending non-detached writer needs it | **Only** after all source-dependent uses are detached and replay/retention rules permit | Existing source owner and persistence |
| Dirty local mask and sparse change set | Uncommitted typed changes still need reduction | Owned, reconciled payload is handed off under existing failure semantics | Existing transient work / CycleWriter |

**Map-only** means no *resident copied facet* is required for this operation; it does **not** mean a sealed map can be used after its source/binding witness expires, nor that historically required source snapshots can be deleted. A map may remain in memory while a previously materialized value cache is independently evicted. Re-fetching values must revalidate the source version, scope, index seal, projection and source lifetime.

Hardware caches are not explicit objects to forcibly flush. Releasing a disposable software buffer is different from invalidating CPU cache lines.

## 3. Address mapping versus value residency: optional execution shapes

The logical query, typed relation, exact source revision, contributor closure and demanded result remain constant. BIND/Quack may choose among already-admitted capabilities; a physical winner does **not** silently create a new legality rule.

| Shape | Reads | Candidate win | Cost / limitation |
| --- | --- | --- | --- |
| **MAP** | Map + typed masks/index support; no canonical value payload | Relationship-only operations; repeated joins/composition from complete support | Building/maintaining support; incomplete closure requires expansion, fallback or refusal |
| **BORROW** | Sealed map → live borrowed SoA facet | Few uses, resident or local rows; avoid derived value copy | Repeated irregular DRAM/cache/TLB access |
| **GATHER** | One admitted pre-fold gather → compact disposable facet → repeated kernels | Many value-dependent reuses | Cold gathering, allocation, memory budget, focus switching, changed values |
| **COMPILED** | Versioned weight/codebook/math artifact and necessary runtime inputs | Reuse a validated mathematical transformation without rehydrating original weight format | Build/calibration and identity gate; compiled value information still exists |
| **STORAGE** | Map → witnessed physical locator → async/batched pages/extents → local operand | Cold source data with reusable index, useful read ordering / batching | NVMe page/read amplification, queue depth, decompression, faults, reorder and restoration overhead |

These are *planning candidates*, not new public types, mandatory allocations, extra storage APIs or universal DR runtime paths. In particular, DR64³ is a native *relational* focus proposal, not a requirement to copy 64³ numerical cells.

**Prefetch has two independent jobs:** anticipate/bind the next focus map; and optionally fetch SoA value lines for an upcoming materialization or demand. A map-only operation does not owe value prefetch merely because its address map is resident. Conversely a numerical operation cannot skip required operands merely because the topology is known.

## 4. Dirty-driven updates, immutable roots and asynchronous handoff

**Case A: only payload values changed and the relationship/index does not depend on those values.** Retain the sealed relationship map; stage changed local positions, map them back through the **same attested original revision**, coalesce aliases by canonical destination and operation, and submit only the owned dirty-row images (or a separately admitted sparse payload). Do not rebuild unrelated address maps or transfer unchanged rows.

**Case B: value-derived relationship/support changed.** A support/map cache depends on its inputs. Track the relevant field/source/version dependency, invalidate affected derived views and compute a **successor sealed revision** (incremental delta only if closure and index-maintenance semantics prove it complete). Reusing yesterday's support because only "values" changed would be stale.

**Case C: the topology, physical index, source schema, palette/scope, or row addressing changed.** Rebind/reseal or refuse. Compaction can preserve stable logical identities while changing physical addresses; raw u16 equality is not sufficient. A change in access rights can invalidate authorization even with unchanged bytes.

For every case:

- Dirty-row reduction is typed (overwrite order, Boolean union, additive/bag, explicit conflict/refusal), not arbitrary last-writer-wins. Duplicate DR slots targeting one canonical row are reconciled once; unrelated fields survive.
- Keep **one authority for durable writes**, via existing BatchWriter/CycleWriter/WalSink. "Fire and forget" applies **only after successfully transferring an owned, detached payload** to the queue. Successful enqueue is not durability. Queue-full/error retains ownership; ambiguous acknowledgement replays the **same frozen batch**.
- A descriptor that still points to live SoA cannot release its source lease merely because it has been enqueued. Historical/replay retention is separate from transient cache residency.
- Proposed incremental map maintenance must account for deletion, inverse indexes, joins, negative dependencies and cross-focus contributor closure; "only dirty rows" is insufficient when a changed fact has nonlocal effects.

## 5. NVMe / persistent addressing is a separate, measured physical layer

The conceptual path is:

~~~text
sealed DR coordinate
  -> canonical SoA ordinal / stable ID (source + branch + version + index seal)
  -> layout-specific row/page/fragment/extent resolver
  -> resident memory, mmap/page cache or explicit read plan
  -> optional local DR facet, preserving original slot/result semantics
~~~

For an actually contiguous, fixed-width 512-byte SoA layout, an address may be computed from **base + ordinal × 512 + facet offset**. For Lance fragments, compressed columns, reordered rows, compaction, or another backend, the mapping may require a versioned index and page/fragment lookup. No raw-LBA shortcut is assumed. S3/object storage remains artifact hydration, **not** a per-pixel random-access replacement.

Benchmark cold-storage access using **actual I/O requested and completed**, not only nominal 512-byte field width: count physical bytes/pages/extents, reads, page faults, decompression, batching, queue depth, sorted-vs-original visitation, stable slot permutation restoration and throughput/latency including p95/p99. Sorting reads by physical address is an optional plan; if result order, bag multiplicity, floating-point reduction order or side effects cannot be preserved, it must not be applied.

## 6. ONNX/TFLite value-demand frontier

An external ONNX/TFLite node graph can be decoded to native operator identities, dependency order, shapes, quantization/calibration and a witness of inputs/weights. After admission and native lowering, operation classes have different *runtime value demands*:

| Work class | May run without original source payload? | Boundary |
| --- | --- | --- |
| Topology, known shape/coordinate transform, typed dependency scheduling, complete Boolean support/relational composition | **Yes**, for those operations | Complete typed contributors and scope are essential; Boolean support is not numeric truth |
| Relationally gated numerical operation | **Sometimes** avoid loading noncontributors | A zero/presence mask is not automatically safe under IEEE NaN, ±Inf, signed zero, SQL K3, bag or evidence semantics |
| Precompiled model weights / palette LUT / factored math | **Possibly** omit fetching original weight bytes during execution | Compiled artifact carries necessary information; full model/layer/weight/codebook/calibration/snapshot identity and independent numeric/quality gates |
| Runtime activations, generic MatMul/Gemm/Softmax, value-dependent predicates, model updates | **No generic value-free claim** | Actual numerical input/weight information must still be supplied or equivalently represented |
| Candidate plasticity/learning output | May update a separate *candidate* relation overlay | Not certified truth or automatic base-map mutation; commit only with admitted provenance and independent approval |

Thus "no longer need the values" means **either the operation is genuinely value-independent, or the required numerical information exists in a proven alternative representation**. Mathematical method compilation does not make arbitrary numeric inputs disappear. ONNX/TFLite remain interchange/conformance leaves; do not put a second graph interpreter or storage owner in them.

## 7. Cost hypotheses and falsification, not a cache rule

For a fixed logical workload, report separately:

\[
T_{\mathrm{total}} =
T_{\mathrm{bind\,+\,map}} +
T_{\mathrm{locate}} +
T_{\mathrm{hydrate}} +
T_{\mathrm{execute}} +
T_{\mathrm{reduce\,+\,handoff}}
\]

Report initial and rebind costs, total wall time, allocation/bytes resident, actual source reads, and amortized **per-map** reuse (N=1/8/64/256). This is an accounting decomposition, **not** a model asserting these terms are always independent or additive in overlapped pipelines. A compute-only warm number cannot decide a full-run storage plan.

[D-DRF-1's A1/A2/A3](d-dr64-focus-cache-falsifiers-v1.md) keep their original meanings: **A1** = distinct dirty canonical rows / distinct rows read; **A2** = facet reads served from resident cache / all facet reads after cold build; **A3** = relationship-only learning steps / all learning steps. **A1 and A3 are NOT RUN** without a representative write/learning stream. Map-only eligibility measured on a synthetic fixture is *not* an A3 observation. The cache-exception lapse rule remains undecided for the intended workload.

**Falsifier matrix; pre-register fixtures and independent oracles before measurement:**

| ID | Control / negative test | Required outcome |
| --- | --- | --- |
| AV1 | Build sealed map, release every gathered value cache, then run a complete map-only Boolean query | Same typed result and provenance as the independent scalar oracle; no value fetch on the map-only timed path |
| AV2 | Force value-dependent operation after cache eviction | Rehydration from the **exact** source revision or explicit refusal, never stale/invented values |
| AV3 | Change source numeric field that *does not* affect map/support, then change a field that *does* | First case preserves eligible topology; second invalidates derived support and seals/rebinds the successor |
| AV4 | Compaction, map/palette ABA, dataset/branch collision, ACL revocation | Stale physical/scope binding refuses even if ordinal, shape or map bytes match |
| AV5 | Aliased slots update different fields; then conflicting same-field updates | Typed deterministic coalesce preserving both fields or explicit conflict; one durable canonical destination image where admitted |
| AV6 | Queue-full, delayed WAL and lost acknowledgement | Ownership not lost; no early lease release; retry identical frozen batch; enqueue never reported as durable |
| AV7 | Same query via MAP/BORROW/GATHER, including out-of-focus contributors, NULL/UNKNOWN, Bag counts and evidence roots | Semantic parity where legal; otherwise explicit typed expansion/fallback/refusal |
| AV8 | Same map and values with sequential versus seeded shuffled visits and physical-order batched reads | Explain latency by access counts, address reuse and page distribution; identical ordered result under declared algebra |
| AV9 | Verified 4 KiB versus 2 MiB backing; distinct lines/offset histogram held fixed | Measure translation hypothesis independently of skewed line/cache-set distributions; no TLB causality claim from wall time alone |
| AV10 | ONNX/TFLite topology-only nodes, compiled weights and generic value-dependent nodes, including changed weight with same shape | Fetch only truly necessary values; stale compiled artifact refuses; compare numeric/semantic oracle and bounded approximations separately |
| AV11 | Deliberately omit one dependency contributor outside local DR64 focus | Expand by a legal witnessed mapping or refuse, never silently treat absence as FALSE/zero |
| AV12 | Sparse relation mutation that changes a transitive dependency or inverse index | Incremental map update matches full rebuild, or explicitly refuses and rebuilds; unchanged-index shortcut must fail |

The latest [#1774 TLB lab](https://github.com/AdaWorldAPI/lance-graph/pull/1774) provides **locality evidence only**: balanced within-page offsets remove an earlier artificial 512→1,024-page jump; warm gathers get slower above roughly 2,048 pages at 4,096 distinct lines. The original 1,024-versus-65,536-row P2 gap also changed the number of distinct lines. Huge pages, hardware counters and a causal split were not verified there. None of this proves that direct NVMe reads are cheap or that map-only learning is common.

## 8. Minimal staged next steps (not implementation authorization)

1. **Doc review and ownership admission:** have D-RMAP-1A / D-DRF-1 owners confirm map-versus-value lifetime separation, source snapshot obligations and legal borrowed/address-only bindings. Do not modify the no-foreign-gather firewall, H14c/H14d, CE64, or Quack semantic owners in this stage.
2. **Map-only falsifier:** extend the standalone DRF lab with AV1/AV2/AV3/AV7/AV11, comparing to independent scalar semantics. Account for allocations and value reads, not just CPU cycles.
3. **Address translation vs I/O:** physically pin source layouts and benchmark borrowed resident, gathered, mmap-backed and explicit batched-read paths on identical map ordinals, with original/result permutation integrity checks. Require a genuine NVMe-capable host before claims about NVMe.
4. **Workload-driven admission:** collect real per-thought value demand, per-map reuse, changed-row counts and learning-step eligibility; report A1/A2/A3 honestly and compare whole-run P2 A/B/C with map-only as a distinct, semantically eligible case.
5. **External numeric conformance:** separately test whether native ONNX/TFLite operations are topology-only, value-dependent, or compiled-artifact reusable; no auto-learning rule or new runtime from this document.

**Decision to postpone:** whether address-first/map-only becomes a runtime physical-plan capability, and whether a separate gather cache remains valuable for the motivating 64-thought foreign-key workload. The architecture can describe both possibilities now; only admitted workload evidence should choose.

## Source crosswalk

- [D-DRF-1 focus lease, cache exception and writeback](d-dr64-focus-lease-foreign-cache-v1.md) and [falsifier ledger](d-dr64-focus-cache-falsifiers-v1.md).
- [D-RMAP-1A RelationshipMap/RelationOrder and BIND owner](d-rmap-separate-tables-thought-palettes-v1.md) and [D-NFC native dependency direction](d-native-first-dependency-correction-v1.md).
- [D-DR3 holographic focused relation semantics](d-dr3-holo-onnx-s3a-s3c-v1.md) and [S3d physical-plan comparisons](d-onnx-relalg-s3d-physical-shootout-v1.md).
- [Quack storage portability ruling](../board/entries/2026-10-05-quack-storage-portability.md) and [#1774 diagnostic](https://github.com/AdaWorldAPI/lance-graph/pull/1774).
