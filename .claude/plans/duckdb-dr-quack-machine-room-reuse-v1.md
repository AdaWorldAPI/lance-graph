# DuckDB ↔ DR/Quack: machine-room reuse without a second executor

> **Status: PROPOSAL / DOCUMENTATION ONLY (2026-10-26).**
> No runtime code, new ABI, benchmark run, source copy, optimiser decision, or board admission in this PR.
> This is a focused continuation of the existing DuckDB harvest, **not** a new DuckDB translation matrix.
> **Owners:** Quack lowering → `lance-graph-quack`; mask execution → `lance-graph-mask-risc`; SIMD polyfills → AdaWorldAPI/ndarray; address-first focus and its scoped derived cache → existing D-DRF/D-DRAV owners.

## Question

Given that DR already supplies a canonical, sealed global-to-local ordinal mapping and resident membership/validity planes, **which of DuckDB's mature vector execution techniques can improve Quack without reintroducing a per-query SelectionVector, tuple materialization, duplicate storage ownership, or an interpreter?**

This is **not** "build DuckDB in Rust." The test is whether any DuckDB-inspired specialization improves the **existing DR mapping + mask-risc terminal + ndarray** path when compared on the same semantic result and the same accounted work.

## Source-first inventory: do not re-derive

### Existing local authority

| Source | What it already establishes | Consequence |
| --- | --- | --- |
| [DuckDB → V3 matrix](../plans/duckdb-to-v3-translation-matrix-v1.md) (§2 R1–R8, E*, A*) | September source-level KEEP/ADAPT/ELIMINATE decisions with falsifiers; deliberately no new executor | Extend only rows affected by DRF map/focus residency and *new* upstream evidence. Do **not** duplicate its 80-KiB harvest. |
| [Quack](../../crates/lance-graph-quack/src/lib.rs), [binder](../../crates/lance-graph-quack/src/bind.rs) | Bound numeric IR only; `lower` vs `lower_fused`; resident SQL K3 planes; no second evaluator | All implementation candidates must lower to existing mask-risc operations or explicitly refuse. |
| [mask-risc](../../crates/lance-graph-mask-risc/src/exec.rs), [IR](../../crates/lance-graph-mask-risc/src/ir.rs) | One execution owner, terminal folds, gated word skips | Don't add runtime compute to Quack. Don't claim fused ternlog skips under a gate. |
| [DuckDB differential](../../crates/lance-graph-quack/tests/duckdb_differential.rs) and [fixture/oracle](../../crates/lance-graph-quack/tests/duckdb/README.txt) | SQL semantic oracle + allocation and intermediate-byte accounting already exist | **Add timing, not a second correctness oracle.** Never hand-edit `cases.tsv` expected values. |
| [Lane-fold capstone](../../.grok/board/CLAUDE_LANE_FOLD_CAPSTONE.md), [integration](../../.grok/board/LANE_FOLD_INTEGRATION.md) | Timed DuckDB column is an **explicit open item**; 64k ordered lane, `u16` index, borrowed aperture, terminal collapse and skip gates | The DuckDB comparison must reuse this harness/cost model. Do not inflate a word-fold into a join/plane timing. |
| [DRF-P2 measured entry](../board/entries/2026-10-24-drf-p2-breakeven.md), [TLB correction](../board/entries/2026-10-25-drf-tlb-probe.md) | Gather-build and repeated cached fold are measured; the original 1,024-versus-65,536 comparison changed cache-line count | Reuse the actual P2 fixtures; report distinct lines/pages and randomized visit order before attributing a speedup. |
| [D-DRAV address-first plan](../plans/d-dr-address-first-value-demand-v1.md) and [DRF code](../../crates/drf-focus-probe/src/lib.rs) | Sealed map can be useful **without values**; optional gathered facet column is a derived disposable cache, not authority | Keep MAP / BORROW / GATHER distinct. Do not make materialization mandatory or call copied facet bytes zero-copy. |

### DuckDB source pointers (read-only, pinned snapshot)

Snapshot: [duckdb/duckdb@`1641475ef1db08ec063631ff6a8f08d9d6736ae7`](https://github.com/duckdb/duckdb/tree/1641475ef1db08ec063631ff6a8f08d9d6736ae7), captured 2026-10-26. These are **source entrypoints**, not evidence that any candidate already wins DR benchmarks.

- [SelectionVector](https://github.com/duckdb/duckdb/blob/1641475ef1db08ec063631ff6a8f08d9d6736ae7/src/include/duckdb/common/types/selection_vector.hpp): index-carrier shape, owned buffer, `Slice`.
- [Vector::Slice / SelCache / Dictionary](https://github.com/duckdb/duckdb/blob/1641475ef1db08ec063631ff6a8f08d9d6736ae7/src/common/types/vector.cpp#L209-L270): indirection composition and reuse.
- [Vector::ToUnifiedFormat](https://github.com/duckdb/duckdb/blob/1641475ef1db08ec063631ff6a8f08d9d6736ae7/src/common/types/vector.cpp#L461-L473) and [vector.hpp](https://github.com/duckdb/duckdb/blob/1641475ef1db08ec063631ff6a8f08d9d6736ae7/src/include/duckdb/common/types/vector.hpp#L117-L127): normalized access to data, selection and validity.
- [ScalarExecutor](https://github.com/duckdb/duckdb/blob/1641475ef1db08ec063631ff6a8f08d9d6736ae7/src/include/duckdb/common/vector_operations/scalar_executor.hpp#L129-L453): flat/constant/generic specialization and nullable/selection loops.
- [Conjunction execution](https://github.com/duckdb/duckdb/blob/1641475ef1db08ec063631ff6a8f08d9d6736ae7/src/execution/expression_executor/execute_conjunction.cpp#L61-L144): selection permutations, intermediate buffers and runtime-adaptive ordering.

**License boundary:** DuckDB is MIT licensed. A source-inspired Rust reimplementation still requires independent semantic tests and attribution where copying expression or code; this plan does not vendor or copy C++.

## Representation: map ≠ mask ≠ cached values

```text
source identity + version + scope + index seal       (checked at bind/lease)
                           │
                           ▼
                 sealed DR pointer map   ord[slot] : u16
                           │
             ┌─────────────┴────────────┐
             ▼                          ▼
    map-only membership            value demanded
    Boolean / relation mask        ├ BORROW: SoA[ord[slot]]
    no value read                  └ GATHER: derived focus column
             │                          │
             └─────────────┬────────────┘
                           ▼
              existing Quack lowering
                           ▼
           mask-risc → ndarray terminal
```

- **Selection role:** packed DR resident mask (`u64` words) already says which slots participate. No reified `Vec<row_id>` on a fold path.
- **Dictionary role:** the sealed `u16` global-to-local permutation is the **one admitted pointer index**. It is not an independently minted relational identity. Duplicate mapped slots remain duplicates when Bag semantics require them.
- **Cached value role:** DRF's optional `Focus::facet_col` stores derived values and is **not** the map. Account for its `FOCUS_BYTES`; require matching `SourceId`, map, scope and facet lease. Eviction must preserve map-only operations.
- **SQL role:** valid/present/unknown are **not interchangeable**. SQL K3, Bag/Distinct, `NULL`, order and multiplicities must match the independent oracle before any timing.
- **No unearned guarantees:** consistent ordinals follow only from attested source/bind invariants; stale seals, wrong scope and cross-version data must refuse before the hot path.

## Candidate harvest: inspect, map, falsify

| Upstream technique | Existing DR owner | Only plausible transfer | Kill/refusal criterion |
| --- | --- | --- | --- |
| SelectionVector + dictionary slicing | DR mask + sealed `u16` map | Compare cost of composing/using resident maps against directly borrowed SoA | Any proposal adding a second per-query row-id vector or losing duplicates/order is rejected |
| `SelCache` dictionary-on-dictionary reuse | D-DRF focus lease / D-DRAV address map | Separate **map reuse** from optional **facet cache** and measure each | Wrong generation/scope/facet cache hit, cache bytes hidden, or cache required for map-only work |
| Specialized flat/constant/generic loops | mask-risc and ndarray | Identify exactly one missing specialization; add *only* with native + AVX2/AVX-512/NEON/WASM/scalar polyfill discipline | Runtime ISA dispatch, Quack evaluator or a regression on existing paths |
| Adaptive conjunction ordering | Quack `Filter::and_by_skip`, gate input | Compare sparse-gated predicate order under **real dead-word/dead-tile fractions** | `lower_fused` claims under-skip, resident plane reordered, or gain disappears on shuffled data |
| Chunk and validity handling | borrowed SoA + resident planes | Borrow DuckDB's experiment design, not its arbitrary 2,048-element width | Extra allocation, copied tuples, K3/NULL discordance |
| Aggregation / terminal | `Agg` + mask-risc / ndarray fold | Verify terminal collapse opportunities already registered in lane-fold | Duplicated evaluator, widened `u16` address, hidden hash/row output |

A DuckDB component being similar is **not** evidence that a matching DR implementation is faster. Record `already exists / missing / different / refused` against named code paths, not narrative analogies.

## Follow-up PR ladder (none approved or executed by this document)

**M0: This PR — baseline and decision register (docs only).**
- [x] Locate the existing matrix, Quack differential, lane-fold open item, D-DRF-P2 and D-DRAV.
- [x] Identify pinned DuckDB headers/implementation seams.
- [x] Define the one performance question: `MAP` vs `BORROW` vs disposable `GATHER`, and amortized fold, under equivalent semantics.
- [ ] Verify existing matrix verdicts against *new* DRF behaviors before changing code. Every candidate gets a named existing owner.

**M1: Differential timing arm (small, separate PR).**
- Extend the **existing** `lance-graph-quack/tests/duckdb_differential.rs`/oracle workflow; do not fork cases or retype expected answers.
- Untimed checksum/parity first. Time native Quack after binding/scratch creation and DuckDB separately on the same logical fixture, projection, output cardinality and materialization boundary.
- Report DuckDB planning/binding and any Python/FFI conversion **outside** the fold separately; do not compare whole-API latency against an in-cache word operation. Do not put timing assertions in correctness tests.
- Preserve `alloc_bytes_exec`, `pair_relation_bytes`, `population_state_bytes`, `fixture_view_bytes`, `scratch_bytes` and add measured `t_compile_or_bind_ns`, `t_execute_ns` and oracle checksum.
- **Exit gate:** parity on all registered cases; no edits to committed expected values except re-generation by documented oracle; timings have identical measured boundaries and units. Otherwise STOP without a performance claim.

**M2: DRF landing-zone threshold crossover (small, separate PR).**
- Reuse `crates/drf-focus-probe/examples/p2_breakeven.rs`, the AV1 map-only path and `tlb_probe.rs`, not a new production API.
- Three arms: (A) MAP-only membership/filter; (B) direct BORROW through sealed map; (C) derived GATHER once + N cached folds. Include `N ∈ {1,2,4,8,64}`, mask density `{1, 1/4, 1/64}`, clustered/balanced/shuffled maps and repeated ordinals.
- Equal source/version/seal/scope, same values and mask, identical K3/Bag target; if map-only has no numerical value answer, compare only a common complete Boolean query, **not** a numerical fold.
- Measure first build and subsequent reuse separately, warm/cold as reported, count distinct cache lines and 4-KiB pages, visit-order histogram, memory bytes and eviction/churn. Do not relabel correlation as TLB causality or quietly include an extra cache copy.
- For each workload, infer crossover from **observed** costs: `N* = min {N >= 1 | C_build + N·C_cached < C_first_direct + (N-1)·C_warm_direct}`; report "no crossover" if observed costs do not meet it. Do not hardcode the prior 1–3 threshold as universal.
- Add invalid-map, stale generation, wrong scope, changed facet and duplicate-ordinal tests. A derived cache must remain disposable.
- **Exit gate:** exact parity and bytes accounted, threshold seen in measured end-to-end path with can-fire negative controls, not hand-picked `get()` microbenchmarks.
- **Input (2026-10-26): Lance 13 masks the READ too.** `Scanner::with_row_addr_prefilter(RowAddrMask)` (lance 13.0.0 `dataset/scanner.rs:1663`) takes an allow/block list in the dataset's `_rowid` space; on a plain scan the mask is the row source and any `filter` refines on top. `RowAddrTreeMap` (lance-select `mask.rs:350`) holds per fragment either `Full` or a `RoaringBitmap`. Consequences for the arms:
  - The membership mask an arm computes can bound what Lance reads, not only what Quack folds. Measured in D-HOLO H14c (`entries/2026-10-15-d-holo-h14c-thought-first.md`): a uniform 1 % mask reads the same bytes as no mask (rows scanned drop 65,536 → 692), a clustered 1 % mask reads 0.24×. So the clustered/balanced/shuffled map axis above decides I/O as well as cache lines; report bytes read per map shape.
  - The handoff must stay a fold. Today's handoff (`crepe-plan-probe/src/lance_store.rs:264`, `row_addr_mask`) enumerates set bits into a `Vec<u64>`. Where slot order equals the fragment's row offset, the mask words can go in whole: `RoaringBitmap::from_lsb0_bytes(offset, bytes)` (roaring 0.11.4) per fragment, `insert_fragment` for an all-ones fragment. Where a DR map sits between slot and row, translation goes through the sealed map; that is arm (A)/(B)'s cost and must be counted, not hidden in the scan.
  - `_rowid` is a row address unless stable row ids are on; H14c: stable ids survive update and compaction, not overwrite, and a stale map omits rows silently. The map's version check is required.

**M3: One targeted optimization only if M1/M2 expose a real gap (separate PR).**
- Candidate is one existing backend specialization or a plan-level reuse decision. Prioritize **no additional work** (reuse borrowed map, fuse mask terminal, avoid materialization).
- Keep Quack strictly as a lowerer; execute in mask-risc/ndarray; no new ubiquitous vector API, dynamic dispatch or hash/tuple fallback.
- Check every supported backend's peer implementation, scalar oracle, SQL K3, Bag/Distinct, 0 hot-path allocations, no pair relation, no silent `u32→u16` cast.
- **Kill:** improvement disappears when preparation, cache eviction and comparable source/target work are included; or any correctness/refusal gate fails. Revert candidate rather than widen architecture.

**M4: Decide / document (not a default implementation).**
- KEEP / ADAPT / ELIMINATE / REJECT for each audited DuckDB technique.
- If there is no independent benefit beyond DRF-P2 and lane-fold, **close with evidence; no new crate, planner layer, or migration.**

## Pre-registered reporting discipline

Rows to report: `case, semantics, source_id, map_id, scope, n_slots, n_distinct_ordinals, n_distinct_lines, n_distinct_pages, mask_density, visit_order, N, map_bytes, facet_cache_bytes, scratch_bytes, fixture_view_bytes, pair_relation_bytes, alloc_bytes_exec, checksum, t_bind_ns, t_build_ns, t_first_fold_ns, t_reuse_fold_ns, total_ns, crossover`.

- Compare the **same** physical work per arm; no "cached local fold" vs "uncached remote SQL" headline.
- Zero is a measured count for the defined boundary, not a claim that upstream preparation was free.
- Per-operation costs in ns; cold/preparation costs in µs where necessary and clearly marked **not fold gates**, following `CLAUDE.md`'s existing exception. Aggregate latencies cannot be passed off as a resident fold.
- Keep the already measured P2 figures with host/cache qualifications: at 65,536 rows/random/full mask, warm build 23,256 ns, cached fold 670 ns, direct warm 20,982 ns; warm crossover 2 folds. This is an **existing lab fixture**, not an expected universal ratio.
- The DRF-TLB page-locality probe isolated a page-count effect under controlled line counts, **not** the full cause of the original build gap. Huge-page causation remains unproven.

## Scope firewall / no drift

- Do **not** modify the DuckDB matrix's old verdicts by implication; amend them with source/date and an explicit reason only after a counterexample.
- Do **not** touch DataFusion grace-period planner, OGAR semantics, ONNX/TFLite, cognitive-shader-driver, P64, or a distributed KV backend in this arc.
- Do **not** treat `DuckDB POSITIONAL JOIN` as proof of a shared, sealed physical ordinal space; our admission contract is independently established in DR.
- Do **not** call row reconstruction, tuple lists, pointer-table copies, or a copied facet cache zero-copy. DR's legal global→local inversion materializes the `u16` pointer map; cache materialization is a distinct, audited concession of D-DRF.
- Do **not** substitute a timed `COUNT` for a requested Bag/Set relation with different semantics.
- No M1–M4 authorization, production implementation or benchmark result is implied by merging this docs PR.

## References / breadcrumbs

- [Lane-fold reference](../../.grok/board/LANE_FOLD.md) and [bound computation](../../.grok/board/BOUND_COMPUTATION.md).
- [Quack DuckDB source harvest](../plans/duckdb-to-v3-translation-matrix-v1.md), [DuckDB oracle files](../../crates/lance-graph-quack/tests/duckdb/).
- [DRF focus](../../crates/drf-focus-probe/src/lib.rs), [map-only membership](../../crates/drf-focus-probe/src/membership.rs), [DRAV plan](../plans/d-dr-address-first-value-demand-v1.md).
- [P2 measurement](../board/entries/2026-10-24-drf-p2-breakeven.md), [page-control correction](../board/entries/2026-10-25-drf-tlb-probe.md), PRs [#1774](https://github.com/AdaWorldAPI/lance-graph/pull/1774), [#1775](https://github.com/AdaWorldAPI/lance-graph/pull/1775).
