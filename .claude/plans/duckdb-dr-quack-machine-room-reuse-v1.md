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

---

## Appendix A (2026-10-26): DR64 ↔ DuckDB ↔ Lance 13 capability crosswalk

> **Status:** documentation only, follow-up to merged #1787. No runtime code, no new crate, no new executor,
> no DataFusion change. Every claim below names a file and symbol; anything not read first-hand is marked
> **UNVERIFIED**. Sources: lance-graph `main` at `841491c7e`; ndarray `master`; DuckDB @
> [`1641475`](https://github.com/duckdb/duckdb/tree/1641475ef1db08ec063631ff6a8f08d9d6736ae7) (MIT); Lance
> 13.0.0 crates (`lance`, `lance-select`, `lance-table`, `lance-encoding`, `lance-io`; Apache-2.0) as resolved by
> the workspace pin `lance = "13.*"` (`Cargo.toml:374`). Exploration record:
> [`entries/2026-10-26-coresearch-lance13-masked-read-dr64.md`](../board/entries/2026-10-26-coresearch-lance13-masked-read-dr64.md).

### A.0 Corrections to earlier text (explicit, not silent)

1. **The M2 "Input (2026-10-26)" bullet above says "The handoff must stay a fold."** Wrong word. Building a
   Lance `RowAddrMask` is an **ingress copy outside the fold**: Lance's scanner is upstream execution, the mask is
   re-encoded into Lance's `_rowid` space, and Lance then serialises it again into an Arrow batch
   (`lance-13.0.0/src/dataset/scanner.rs:3852-3866`, `mask_as_take_input`). Nothing on that path may be called
   zero-copy or a fold (see A.3, seam S-INGRESS).
2. **September matrix R6** ("absence is a zero-fallback, never a validity bit") is superseded by code: Quack
   carries SQL K3 validity planes (`lance-graph-quack/src/lib.rs:549-614` `Filter::sql_where`, test
   `tests/sql_null_3vl.rs`). That is R6's own falsifier ("a consumer that must serve SQL 3VL at a membrane").
3. **September matrix R5** cites the hex-tenant probe (D-GTM-0m); the hex tenant is superseded by the Moore tenant,
   so that evidence is historical. The range half of R5 is now shipped (G6 below).
4. **Gap label clash:** the September matrix's G8 ("cheap emptiness", `duckdb-to-v3-translation-matrix-v1.md:175`)
   is not ndarray `masking-ops-state.md`'s "G8" (`lzcnt_bswap_u64_to_u8`, line 211). Cite with the file.

### A.1 Coordinate domains: never reinterpret one as another

| domain | carrier | size | owner | witness |
|---|---|---|---|---|
| DR64 focus slot `s ∈ 0..4096` | resident slot mask `[u64; 64]` | 512 B | DRF / Quack | `FocusKey` (source+map+scope+facet), `drf-focus-probe/src/lib.rs:61-69` |
| canonical ordinal `ord[s] : u16` | sealed map `Box<[u16; 4096]>` | 8 KiB | DRF `SealedMap`, `lib.rs:48-52` | `index_seal` via `check_seal`, `lib.rs:132-140` |
| canonical domain `0..65536` | domain mask `[u64; 1024]` | 8 KiB | DRF `membership`, `membership.rs:103-125` | same |
| Lance `_rowid` | `RowAddrMask` / `RowAddrTreeMap` (fragment → `Full` \| `Partial(RoaringBitmap)`) | per fragment | Lance, `lance-select/src/mask.rs:26,350` | dataset version (stable row ids survive update+compaction, not overwrite: H14c) |
| physical `_rowaddr` (`fragment<<32 \| offset`) | Lance internal | — | Lance | equals `_rowid` only when stable row ids are off |
| output batch position | Arrow batch row | — | Lance | none: **never** used as a coordinate |

`DR ordinal ≠ _rowid ≠ _rowaddr ≠ output position.` No code today composes `ord` with any `_rowid` table
(closed search space `crates/**/*.rs`). There must be exactly one version-checked coordinate→`_rowid` authority;
today there are two lab ones with different witnesses (`holo-lance-masked-io-probe/src/main.rs:148-156`
`CoordMap{version}` + `ReadError::StaleMapping`; DRF `check_seal` on `index_seal`), and the crepe handoff has none
(`crepe-plan-probe/src/lance_store.rs:264`, keyed on `_rowaddr`).

### A.2 DuckDB capability crosswalk

Status vocabulary: EXISTS · PARTIAL · INTEGRATION GAP · MISSING · INTENTIONALLY ELIMINATED · UNVERIFIED.
DuckDB paths are relative to `src/` at `1641475`.

| # | DuckDB capability (source @1641475) | our owner (file : symbol) | status | what DR64 makes unnecessary / what technique stays useful | home if missing | falsifier · minimal test | duplication risk | decision |
|---|---|---|---|---|---|---|---|---|
| 1 | SelectionVector — `include/duckdb/common/types/selection_vector.hpp:31-180` | packed `u64` mask + sealed `u16` map; mask-risc `Pred`/`Terminal` (`lance-graph-mask-risc/src/ir.rs:129-321`) | INTENTIONALLY ELIMINATED | A per-query row-id list; R1 + D-GATED-GATHER-0 (ordinal list never wins outside noise). Useful: nothing beyond the sealed map as the one pointer table. | — | R1's falsifier (`masked_sum_i32` vs index-list sum at s ∈ {1/64,1/256,1/1024}) | HIGH: any `Vec<row_id>` on a fold path | KEEP ELIMINATED |
| 2 | DictionaryVector / `Vector::Slice` — `include/duckdb/common/vector/dictionary_vector.hpp:17-172`, `common/types/vector.cpp:209-250` | DRF `SealedMap` (`lib.rs:48-52`) + BORROW `direct_sum_where` (`lib.rs:373-402`) | EXISTS (lab) | Rewrapping a column: the column stays borrowed, the map rides beside it. Useful: composition reuse is the DRF facet cache, not a second map. | — | P2-breakeven parity assert (`examples/p2_breakeven.rs:147-150`) | MEDIUM: a second map type | KEEP |
| 3 | SelCache — `include/duckdb/common/types/sel_cache.hpp:17-19` | DRF `FocusManager` lease/evict (`lib.rs:241-357`) | PARTIAL | Index-on-index caching. Gap: no push invalidation on source-version change (a new `SourceId` is just a new key). | DRF | stale-version entry is never served: build focus at v1, bump version, assert a load returns a different key | MEDIUM: a row-index cache | KEEP (note gap) |
| 4 | FlatVector — `include/duckdb/common/vector/flat_vector.hpp:76` | `LaneRef::{I32,U32,U64,Strided}` (`ir.rs:16-51`), `Planes` borrowed | EXISTS | Identity row (R2). | — | none possible (identity) | LOW | KEEP |
| 5 | ConstantVector — `include/duckdb/common/vector/constant_vector.hpp:15-91` | scalar argument of every `Pred` (`ir.rs:129-200`) | INTENTIONALLY ELIMINATED | A constant never becomes a vector (R3). | — | — | LOW | KEEP ELIMINATED |
| 6 | UnifiedVectorFormat — `include/duckdb/common/vector/unified_vector_format.hpp:22-67` | `Planes{n_rows, masks, lanes}` + `Foreign` (`ir.rs:73-113`) | EXISTS | One access triple; `sel` eliminated, validity carried as planes (A.0-2). Missing as *functionality*: nothing measured. | — | R7 falsifier (express a multi-column equi-match in `Planes` alone) still unrun | LOW | KEEP |
| 7 | ValidityMask — `include/duckdb/common/types/validity_mask.hpp:60-366` | Quack K3 planes `Filter::sql_where` (`lib.rs:549-614`), `BoundField.validity` (`bind.rs:77-90`) | EXISTS | Same packed-u64 carrier. Keep apart: NULL (UNKNOWN) ≠ absent row ≠ zero selected ≠ not materialised. A NULL in a *foreign* value lane is not modelled (`lib.rs:537`). | Quack | `sql_null_3vl.rs` (Kleene oracle, gated + fused) | LOW | KEEP |
| 8 | DataChunk / `STANDARD_VECTOR_SIZE` — `include/duckdb/common/vector_size.hpp:16-23` | 64-row words, 256-word tiles (`exec.rs:110-116`); storage batch = Lance's | EXISTS | Chunk survives, 2048 does not (R8). | — | R8 sweep (chunk width ∈ {256..4096}) unrun | LOW | KEEP |
| 9 | Vectorized comparisons — `include/duckdb/common/vector_operations/binary_executor.hpp:106-248`, `scalar_executor.hpp:762-877` | ndarray `*_to_mask(_under)` (`simd_masking_ops.rs`); mask-risc `Pred` | PARTIAL + INTEGRATION GAP | Specialised loops per input shape are ndarray's backends. Gaps: u16 compare (G1b) and signed i64 ordered (G2b) missing in ndarray; u8 and unsigned u64 ordered exist (`simd_masking_ops.rs:3569-3724`, `3787-3936`) but have **no `Pred` variant** (`ir.rs:129-200`), so Quack cannot lower to them. | mask-risc `Pred` + Quack (wiring); ndarray only for u16/i64 | lower a u8 / u64 predicate; today it is refused or widened (expected to fire) | MEDIUM: a Quack-side compare | ADAPT (wiring) |
| 10 | AdaptiveFilter / conjunction — `execution/adaptive_filter.cpp:116-141`, `execution/expression_executor/execute_conjunction.cpp:61-146` | gated `lower` (`lib.rs:2235-2346`, gate at `:2280`), `lower_fused` (`assign_slots`, `:2369-2380`), `Filter::and_by_skip` | PARTIAL | Runtime reordering by timing is not needed for correctness. Facts: progressive survivor narrowing exists only in gated `lower` for plane-free ANDs (a plane wins the gate); `lower_fused` gates only on `under` planes and has **no** progressive narrowing (`MaskOp::Ternlog` has no `under`, `ir.rs:250-256`); `Range`, `EqU32Via` and strided preds have no `_under` twin (`exec.rs:569-750`); no dead-tile skip. | ndarray `_under` twins + Quack lowering | P4 (A.6) | MEDIUM: a second scheduler | ADAPT (measure first) |
| 11 | SequenceVector — `include/duckdb/common/vector/sequence_vector.hpp:15-42` | `Cmp::Range`→`Pred::Range`→`mask_set_range` (`exec.rs:658-667`, ndarray `:4069`); `prefix_facet` witnessed (`lib.rs:888`) | EXISTS | DuckDB flattens a sequence at the first predicate; we keep the range. Range order is not attested by `Planes` (`ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER`). | — | `mask_set_range_*` tests (`simd_masking_ops.rs:8404-8502`) | LOW | KEEP |
| 12 | Zone maps / pruning — `storage/table/row_group.cpp:734-804` `RowGroup::CheckZonemap` | storage: Lance (8,192-row zone granularity per `entries/2026-10-11-lance-source-probe.md`, **UNVERIFIED** in source); resident: `Pred::Range`, aperture lowering | EXISTS (different layers) | DR masks are not zone maps: they select rows, Lance zone maps prune storage. Do not build zone maps over Lance data (that patches storage). | — | — | MEDIUM: resident zone maps | KEEP |
| 13 | GROUP BY — `execution/aggregate_hashtable.cpp:836-1005`, `execution/operator/aggregate/physical_hash_aggregate.cpp` | `lower_group_by` (`lib.rs:1407`), `lower_group_by_semantic` (`:1466`), `lower_group_by_auto` (`:1922`); terminals `GroupSumI32`, `GroupReduce` (`ir.rs:399-465`) | PARTIAL | No hash table: groups are masks or one folded program. The "K knee / refusal" is **claimed only** (`.grok/board/LANE_FOLD_INTEGRATION.md` section "What this must not integrate"); no K cap in code. mask-risc has terminals Quack never emits (`MaskedStridedGroupSum`, `GroupPowerSumsI32`, `GroupCrossPowerSumsI32`). | Quack lowering | count K at which `lower_group_by_auto` changes route; assert a refusal past the knee (fires today: none exists) | MEDIUM | KEEP; knee = OPEN |
| 14 | JOIN / FK — `execution/join_hashtable.cpp:635-1206`, `execution/operator/join/perfect_hash_join_executor.cpp:73-309` | `Filter::EqU32Via`→`Pred::EqU32Via` (`exec.rs:569`), `Filter::Semijoin`→`MaskOp::Gather` over a resident plane; Bag/Set witness `tests/gremlin_parity.rs` | EXISTS (one functional hop) | No hash join by design: a bound FK lane is the perfect-hash case. Fan-out re-anchors; two-deep and second fan-out are refused (test-local `Refusal`, `gremlin_parity.rs:269`). `OutgoingHop` quarantined (`quarantined-gather-hop`). | — | `gremlin_parity.rs` (13 tests) | HIGH: a join executor | KEEP |
| 15 | Late materialization — `optimizer/late_materialization.cpp:159` `TryLateMaterialization` | Lance projection + `with_row_addr_prefilter`; DRF BORROW/GATHER | EXISTS (split across layers) | DuckDB re-joins on row ids (an owned row-id join, a tuple shape): rejected. The useful idea is "read only demanded columns of surviving rows", which Lance (storage) and DRF (resident) already do. Integration between them is A.4. | Lance seam | H14c/H14d (measured) | HIGH: a row-id re-join | KEEP (no port) |
| 16 | Output compaction — `common/types/vector.cpp:443-459` `Vector::Flatten`, `common/types/data_chunk.cpp:369-375` `DataChunk::Slice` | `exec.rs:343` `materialize_rows` (named egress only); G5 | INTENTIONALLY ELIMINATED (inside); egress only | Compaction is egress work. G5 stays DO-NOT-BUILD until an egress-point count exists. | ndarray (if ever) | — | MEDIUM | KEEP ELIMINATED |
| 17 | Executor specialisation — `include/duckdb/common/vector_operations/unary_executor.hpp:76-208`, `scalar_executor.hpp:762-877` | mask-risc fusion shapes (`ir.rs:1349-1372`), ndarray backends, generated `ternlog_dispatch` | EXISTS | Specialisation is compile-time (ndarray realisations) plus mask-risc fused shapes; no runtime ISA dispatch in consumers (`exec.rs:2471` `the_crate_names_no_isa`). | — | existing tests | HIGH: a second executor | KEEP |

**September gaps re-audited** (`duckdb-to-v3-translation-matrix-v1.md:168-175`):

| gap | status now | implementation · test |
|---|---|---|
| G1 u8/u16 compare-to-mask | u8 **CLOSED**; u16 OPEN (no consumer) | `simd_masking_ops.rs:3569` `eq_u8_to_mask` … `:3724` `ne_u8_to_mask`; test `:7277`. Not in mask-risc `Pred`. |
| G2 ordered 64-bit compare | unsigned **CLOSED**; signed i64 OPEN (no consumer) | `:3787-3936` `*_u64_to_mask`; test `:7422`. Not in `Pred`. |
| G3 masked argmin/argmax | OPEN (OUTLOOK) | none; `MaskedMinI32`/`MaxI32` return values, not indices |
| G4 Morton neighbour shift | **CLOSED** (its cost model stays [H]) | `:5402` `mask_shift_morton`; tests `:7917-8112` |
| G5 egress compaction | OPEN by standing order (DO NOT BUILD) | none |
| G6 range-mask primitive | **CLOSED** | `:4069` `mask_set_range`; lowered as `Pred::Range`; tests `:8404-8502` |
| G7 lane-vs-lane compare | DELIBERATELY ABSENT | ndarray `masking-ops-state.md:196-209`; the IR-doc note the matrix asked for is **UNVERIFIED** (not found in `mask-risc/src/*.rs`) |
| G8 cheap emptiness | OPEN | `mask_any` (`:4230`) ORs every word, no early exit; `Any` terminal exists |

### A.3 Lance 13 masked read: what already exists, and what it does to bytes

Already implemented and measured; do **not** rebuild:

- **API (Lance 13.0.0):** `Scanner::with_row_addr_prefilter(RowAddrMask)` (`dataset/scanner.rs:1663`);
  `RowAddrMask::{from_allowed, from_block}` (`lance-select/src/mask.rs:50,55`); projection, `filter` refine on
  top, `MaterializationStyle`, `scan_stats_callback`.
- **Inside Lance:** on a plain scan the mask is intersected and serialised into an Arrow batch for
  `FilteredReadExec` (`scanner.rs:3852-3866`); it becomes offset ranges per run, not per row
  (`lance-table/src/rowids.rs:609-646`, `io/exec/filtered_read.rs:1198-1209`); a `Full` fragment is one
  `0..physical_rows` range. Narrow columns are mini-block: a chunk is ≤ 8 KiB−6 B and ≤ 4,096 values by default
  (`lance-encoding/src/encodings/logical/primitive/miniblock.rs:19,21`) and is fetched and decoded whole.
  lance-io merges byte ranges closer than its block size (64 KiB cloud default, `lance-io` `scheduler.rs`,
  **section-read by a scout, not re-checked here**). The KNN/FTS flat branch filters per `_rowid` *after* reading
  (`io/exec/row_addr_mask.rs`): a mask there cuts rows, not bytes.
- **H14c** (#1573, `entries/2026-10-15-d-holo-h14c-thought-first.md:17-22`): uniform 1 % reads 1.00× the bytes
  (rows scanned 692 of 65,536); clustered 1 % 0.24×; `MaterializationStyle` changed bytes in 0 of 120 cells;
  independent per-lane masked reads cost 1.5–5.3× one unmasked projection, the combined mask 0.05–0.21×; a stale
  map omits rows silently (never wrong coordinates); stable row ids keep `_rowid` across update and compaction,
  not overwrite; without them compaction moved 3,510 of 3,510.
- **H14d** (#1575, `entries/2026-10-16-d-holo-h14d-logic-mask-read.md:6-28`): `logic → address mask → Lance
  masked read → payload terminal`, rows scanned == |R| in 96/96 cells; full tile uniform 1 % 1.00×, clustered 1 %
  0.19×, clustered 10 % 0.99× (64-address runs are below Lance's chunk size), 1,024/8,192-address apertures
  0.01–0.12×, empty R reads 0 B.

These are fixture measurements (65,536 rows, v4, Lance 13.0.0), not universal predictions. Why they come out this
way is now explained from source: a uniform mask at density d touches a 4,096-row chunk with probability
`1 − (1 − d)^4096` (≈ 1 at d = 1 %), so every chunk is read; a clustered or aperture mask touches few chunks.
**Logical DR64 density does not predict Lance I/O; chunk locality of the selected `_rowid`s does.**

**Copy status of the handoff (seam S-INGRESS, OPEN, operator decision).** The words → `RowAddrTreeMap` step
re-encodes an elected `Keep` set into a container Lance owns, and Lance copies it twice more. The waben ruling
(`waben-fold-execution-loop-v1.md:1967-1979`, "where a set IS elected … the container is a free choice") covers
our own carriers; whether it covers a foreign-owned ingress container is not ruled. Until the operator names the
seam (as D-DRF-1 did for its cache), nothing on this path is called zero-copy or a fold.

**Timing rule for storage reads.** The CLAUDE.md exception covers cold compile and oracle arms only. A Lance
read is neither, so storage-side results are reported as **bytes read, ranges, requests and rows scanned**
(as H14c/H14d do), not wall time; any wall time stays in µs or is omitted.

### A.4 End-to-end execution map

```text
canonical identity (dataset, branch, version, index_seal, scope)            witness: SourceId / FocusKey (DRF)
   │  C_bind: admit or refuse (StaleMap / StaleMapping)                      owner: DRF bind + coordinate map
   ▼
DR focus: slot mask [u64;64] (512 B) + sealed ord[slot]:u16 (8 KiB)          lifetime: DRF lease (FocusManager)
   │
   ├── resident source (canonical SoA borrowed):                             no storage read
   │      Quack lower / lower_fused → mask-risc → ndarray terminal           C_fold
   │      MAP-only: membership(map, set) — no value reachable                C_map
   │      BORROW: direct reads SoA[ord[s]]                                   C_fold (scattered)
   │      GATHER once (Focus::build, FOCUS_BYTES 16 KiB) + N cached folds    C_gather + N·C_reuse; eviction C_eviction
   │
   └── nonresident source (storage read required):
          C_map:  ord[s] → canonical ordinal → _rowid through ONE version-checked map   [INTEGRATION GAP]
          C_mask: words → RowAddrTreeMap (Full | from_lsb0_bytes | insert_range | per-bit residual)
                  — S-INGRESS copy, outside the fold                                    [OPEN seam]
          C_io:   Lance with_row_addr_prefilter + projection of demanded columns only   (Lance owns)
          C_decode: chunks touched × decode                                             (Lance owns)
          re-attach: values attributed by a stored coordinate column, re-expanded through ord[s]
                  as a template fold (Bag multiplicity, slot position, requested order restored here;
                  Lance batch order is never a coordinate)
          optional: DRF GATHER over the fetched values only if N reuses pay (no second gather)
   ▼
terminal fold (Count/Sum/Group/Keep) in mask-risc/ndarray
```

Distinguish at the result boundary: **zero selected** (empty mask: skip the read entirely, H14d measured 0 B),
**absent/deleted** (not in the mask's universe), **SQL NULL** (validity plane, UNKNOWN), **not materialised**
(outside the captured version or extent). A block-list (complement) mask is only legal if the complement is taken
in K3 before collapse to `Keep` and over the version-pinned full extent; an allow list fails safe (omission).

**Cost ledger per path** (all terms accounted; storage terms in bytes/ranges, fold terms in ns):

| path | C_bind | C_map | C_mask | C_io | C_decode | C_gather | C_fold | C_reuse | C_eviction |
|---|---|---|---|---|---|---|---|---|---|
| 1 MAP-only | ✓ | membership | — | — | — | — | Boolean terminal | — | map lease |
| 2 resident BORROW | ✓ | `ord` | — | — | — | — | scattered SoA reads | per fold | — |
| 3 GATHER once + N folds | ✓ | `ord` | — | — | — | 16 KiB facet | dense fold | N × cached | facet lease |
| 4 Lance masked fetch + native terminal | ✓ | ord→`_rowid` | S-INGRESS | chunks touched | chunks touched | — | fold over fetched | — | — |
| 5 Lance fetch, then DRF cache if N pays | ✓ | ord→`_rowid` | S-INGRESS | once | once | 16 KiB | dense fold | N × cached | facet lease |

Measured so far: path 3 vs 2 (DRF-P2: 65,536 rows random full mask, warm build 23,256 ns, cached fold 670 ns,
direct warm 20,982 ns, warm crossover 2 folds; page-count component isolated by DRF-TLB). Paths 4–5: storage half
measured (H14c/H14d); **the integrated DR64 → Lance → Quack query is unmeasured**. A full-domain mask scan is not an
I/O optimisation unless bytes drop; a storage read is never timed as if it were a 512-byte resident operation.

### A.5 Missing-feature register

| gap | why missing | existing partial implementation | correct owner | minimal action | test | duplication risk | decision |
|---|---|---|---|---|---|---|---|
| ONE coordinate→`_rowid` authority | two lab maps with different witnesses; crepe handoff unchecked | `CoordMap`+`StaleMapping` (H14c); `check_seal` (DRF) | Lance integration seam | the DR map records the Lance dataset version it was built against; one refusal type | P3 stale cases | HIGH (a third map) | ADAPT (wiring) |
| DR ordinal → `_rowid` composition | nothing composes `ord` with a `_rowid` table; H14d's `out[coord]` is keyed by H14d's own coordinate, not a DR ordinal | H14d `row_ids` (`holo-h14d-probe/src/main.rs:448-470`) | Lance integration seam | look up through both maps (never store `row_ids∘ord` as a third table) | P3 | HIGH | ADAPT, gated on S-INGRESS |
| bulk words → `RowAddrTreeMap` | every handoff enumerates bits (`lance_store.rs:264`, H14c `masked_read`, H14d `row_mask`) | `insert_range` per run (`addressed.rs:265-271`, order-witness gap) | Lance integration seam | `insert_fragment` for all-ones fragments (only when the slot span equals the physical extent at a checked version); `from_lsb0_bytes` where slot == fragment offset (dense containers only avoid per-bit work) | P2 | LOW | ADAPT, gated on S-INGRESS |
| chunk-touched I/O predictor | never built; `full_permille` heuristic in `addressed.rs` is rows covered, not bytes | mask-risc `Any`/`Count` | mask-risc (express with existing ops first) | **estimator** (not exact): non-zero 64-word groups × bytes/chunk; true boundaries need Lance page metadata, and lance-io range merging breaks "chunks × bytes" | P1 | LOW | PROBE (A.6) |
| u8 / u64-ordered `Pred` | ndarray kernels exist, IR never exposed them | ndarray G1/G2 | mask-risc `Pred` + Quack lowering | add `Pred` variants pointing at existing kernels; no new SIMD | P5 | MEDIUM | ADAPT when a consumer asks |
| `_under` twins / fused narrowing | Range, EqU32Via, strided full-read then AND; fused has no accumulator gate | gated `lower` for plane-free ANDs | ndarray + Quack | measure first | P4 | MEDIUM | PROBE later |
| cheap emptiness (G8) | `mask_any` has no early exit | `Any` terminal | ndarray | early-exit `mask_any`; skip `with_row_addr_prefilter` on an empty mask | can-fire/silent on empty vs one-bit masks | LOW | KEEP OPEN (value small) |
| K knee / group refusal | claimed in `LANE_FOLD_INTEGRATION.md` section "What this must not integrate", not in code | `lower_group_by_auto` route switch | Quack | write the knee down from a measurement or drop the claim | route-switch test | LOW | OPEN |
| block-list (complement) masks | K3 and universe hazards (A.4) | `Not` | Lance seam | none until a K3-safe complement and version-pinned universe exist | — | — | REJECT for now |
| DuckDB late-materialisation re-join, selection vectors, hash join, resident zone maps | tuple shape / second executor / storage patch | — | — | — | — | HIGH | REJECT (intentional) |

**Genuine missing code:** u16 and signed-i64 compares (no consumer), argmin/argmax (G3), early-exit emptiness (G8).
**Missing wiring:** one coordinate authority, ord→`_rowid` composition, bulk mask construction, IR exposure of u8/u64.
**Intentional exclusions:** selection vectors, row-id re-join, hash join, egress compaction before an egress count,
resident zone maps, complement masks.

### A.6 Probes (pre-registered, none authorised) and the one recommended next PR

Each probe reuses H14c/H14d/DRF-P2 fixtures; none rediscovers H14c/H14d.

- **P1 (recommended) — chunk-touched estimator vs measured bytes, offline.** Input: the committed raw CSVs in the
  H14c and H14d entries (`bytes_read` per cell; H14d `lance_bytes_masked`/`lance_bytes_unmasked`) and the masks
  regenerated from the probes' deterministic seeded draws. Measure the predicted ratio (non-zero 64-word groups /
  total groups, assuming 4,096-value chunks) against the measured masked/unmasked byte ratio per cell. Kill:
  |predicted − measured| > 0.10 in more than 10 % of cells, or rank correlation < 0.9; then test the 64 KiB-merge
  hypothesis once, no third tuning. Can fire: uniform 1 % ≈ 1.00, clustered 1 % ≈ 0.2. Can stay silent: masks
  offset by 2,048 rows (every run crosses a chunk boundary) must raise the clustered prediction. Cost: small; no
  Lance run if the CSVs suffice; no production code.
- **P2 — bulk handoff equivalence and cost.** Build the same `RowAddrTreeMap` by per-bit loop, by
  `insert_fragment`+`from_lsb0_bytes`, and by `insert_range`; assert equal maps and equal `rows_scanned` /
  `bytes_read` / payload sum; time the build per 64-row word (any ms is a failed gate). Can fire: a fragment-base
  ≠ slot fixture must make the bulk path differ. Can stay silent: empty mask → 0 fragments, 0 bytes.
- **P3 — DR→Lance semantic equivalence.** DRF-P2 maps with duplicate ordinals composed with a `_rowid` table;
  compare SUM, COUNT and per-slot attribution against `direct_sum_where`; stale cases (compaction without stable
  ids, overwrite) must refuse, update+compaction with stable ids must not. Dedup check: removing Bag re-expansion
  must change the answer. **Must pass before any DR→Lance code lands.**
- **P4 — gate skipping.** Count words of the second operand loaded under gated `lower`, `lower_fused`, and with a
  `Range`/`EqU32Via` child. Fused must load every word (documents no skip).
- **P5 — IR exposure of u8/u64.** Expected to fire today (refused or widened).

**Recommendation: one probe PR, P1 — not an implementation.** It is the cheapest (offline over committed data), it
is the only probe that changes a decision (whether a masked read pays, decidable on the fold side in ns before any
handoff), and a failure kills the estimator before any op is proposed. Everything else waits: the DR→Lance seam is
gated on the operator naming S-INGRESS and on P3.

### A.7 What a maintainer can answer from this appendix

1. **DuckDB does and DR64/Quack also does:** flat/constant/unified access, validity (K3 planes), comparisons,
   sequence ranges, specialised executors, one-hop FK joins, aggregation by masks (A.2 rows 2, 4–9, 11, 13–14, 17).
2. **DR64 eliminates by representation:** selection vectors, dictionary rewraps, sel caches as row-index caches,
   constant vectors, the hash join for a bound FK, output compaction inside the fold (rows 1, 3, 5, 14, 16).
3. **Lance 13 handles before data reaches the resident engine:** row selection by mask, run-to-range planning,
   whole-fragment reads, projection, chunk-granular decode, range merging (A.3).
4. **Actually missing:** u16/i64 compares, G3, G8 (code); one coordinate authority, ord→`_rowid` composition,
   bulk mask construction, IR exposure of u8/u64 (wiring) (A.5).
5. **Smallest defensible next action:** P1, offline (A.6).
6. **Must not be duplicated:** H14c/H14d pipelines and measurements, the two coordinate maps (unify, don't add a
   third), DRF MAP/BORROW/GATHER, mask-risc terminals, ndarray kernels, the DuckDB differential oracle, Lance's
   scanner and range planner.

### A.8 Amendment (2026-10-26, later the same day): SQL K3 → Keep → Lance ALLOW/BLOCK

Append-only; A.0–A.7 stand. Scope: whether a SQL three-valued `WHERE` can lower to one final Keep that Lance 13
consumes as an allow or block list.

**SQL K3 evidence, re-run today against a live PostgreSQL 16.15** (the examples print `SKIP` without a server, so
none of this gates CI; these runs are the evidence):

| run | result |
|---|---|
| `cargo run -p lance-graph-quack --example k3_where_pg_parity` | 2,000 expressions × 200 rows, 55,916 UNKNOWN verdicts: `sql_where` → `lower` / `lower_fused` → mask-risc keeps PostgreSQL's TRUE **and** FALSE rows exactly |
| `… --example qsp_null_operand_pg_parity` | 2,000 expressions incl. 250 NOT-IN-with-NULL: exact, also with every NULL payload scrambled |
| `… --example qsp_agg_pg_parity` | 3,600 queries, COUNT(*) / COUNT(x) / SUM / MIN / MAX / AVG: 0 mismatches (naive SUM wrong on 1,962) |
| `cargo run -p lance-graph-contract --example k3_pg_parity` | `normalize_kleene` = Kleene oracle = PostgreSQL on 2,000/2,000; two-valued `normalize` wrong on 474 |
| `cargo test -p lance-graph-quack --test sql_null_3vl --test sql_aggregate_null` | 10 + 3 passed |

How the TRUE-only Keep is built (`lance-graph-quack/src/lib.rs` `Filter::sql_where` / `rewrite`): each subformula
lowers to one polarity, `T(φ)` or `F(φ)`; a nullable leaf becomes `valid ∧ leaf` (or `valid ∧ ¬leaf`), `NOT` flips
polarity, `AND`/`OR` take their De Morgan duals; the result is `T(self)`, a two-valued filter. A second program is
needed to observe `F`. Payload at invalid slots is loaded inside a live 64-row word and then masked by the validity
gate (ndarray `pack_under`); it never reaches the answer (`tests/sql_null_operand.rs` `invalid_payload_changes_nothing`).
No rewrite on this path applies a two-valued law: Quack does not call `lance-graph-relalg` or contract
`set_algebra::normalize`; the fuser evaluates truth tables over the leaves it is given.

**Remaining SQL gaps for the scan use case** (owner in brackets; nothing here blocks a filter-only scan):

- COUNT(expr), MIN, MAX are correct only when the caller conjoins `is_not_null(valid)`; no lowering helper enforces
  it [Quack, a `lower_count_expr`-style wrapper like `lower_sum`].
- COUNT DISTINCT (`CountDistinctOrderedU32`) requires a key lane ordered over every row; a NULL key's stale payload
  can trip `LaneNotOrdered`, and no NULL test exists [Quack].
- GROUP BY / HAVING take no validity for key or value; NULL group keys are not modelled; tested on NULL-free data
  only (`duckdb_differential.rs` group cases) [Quack].
- MISSING: column-to-column comparison (G7, deliberately absent in the IR), COALESCE/NULLIF, nullable arithmetic,
  LEFT join (null extension), subquery IN / NOT IN. A NULL in a FOREIGN value lane is not modelled.
- Test gap: no named `x = c OR NOT (x = c)` / `x = c AND NOT (x = c)` case on a nullable `Cmp` in
  `sql_null_3vl.rs` (only the fk leaves and the random trees). The contract rewriter pins it
  (`set_rules.rs` `experiment_e_null_normalizes_in_the_kleene_form`).
- Unguarded hazard (no production caller today): contract `Sym` and Quack `Filter::Plane` are two-valued by
  contract; binding either to a collapsed K3 TRUE mask makes `NOT` the complement of Keep.

**The ALLOW/BLOCK boundary.** `U` = the live rows of the scanned version, `K = {r ∈ U | P(r) = TRUE}`,
`B = U \ K`. `ALLOW(K) == BLOCK(B)` is the required equivalence. `BLOCK(K(NOT P))` is **not** equivalent: it
admits every row where P is UNKNOWN. Live PostgreSQL counterexample, rows `x ∈ {5, 3, NULL}`, `P = (x = 5)`:
`WHERE P` → {1}; `U \ K(NOT P)` → {1, 3} (the NULL row leaks); `U \ (U \ K(P))` → {1}; `WHERE P OR NOT P` → {1, 2}.

What Lance 13 does with each (source-read, `lance-13.0.0`):

- The block universe is the scanned dataset version's fragment set (`scanner.rs` `mask_as_take_input`:
  `fragments_covered = self.dataset.fragment_bitmap`), per fragment the physical rows minus the deletion vector
  (`io/exec/filtered_read.rs` `full_frag_range` intersected with the mask ranges). Deleted rows drop out under both
  lists unless `with_deleted_rows`.
- **Fail-open asymmetry:** a fragment added after the mask was built is fully visible under BLOCK and excluded under
  ALLOW. An allow list built from an older version fails closed (omission); a block list fails open (extra rows).
  BLOCK therefore needs the version check to be exact, not best-effort.
- BLOCK loads metadata for every fragment ("A block-list may select rows from every fragment",
  `filtered_read.rs`); ALLOW skips fragments with no candidates; an empty ALLOW skips every fragment (no data I/O).
  A block-everything mask is not short-circuited.
- Both become offset ranges per roaring run; scattered blocked rows mean many small ranges. Stable-id segments of
  type `RangeWithHoles` / `Ranges` / `Array` take a per-id path.
- With an external mask the scalar index is not used and limit/offset are not pushed below the mask
  (`effective_filter_plan` → refine-only). Lance's own `filter(...)` runs as a refine on top of the mask; a NULL
  filter result is dropped (Arrow `prep_null_mask_filter`, NULL → false), i.e. WHERE-TRUE-only, but over Lance's
  expression language, not Quack's.
- One scanner ≠ one I/O: reads are scheduled per fragment and data file, per column page, merged per scheduling step
  into one sorted request. With a mask and no `filter`, all projected columns are read in one `FilteredReadExec`;
  with a `filter`, filter columns join the projection and late columns are fetched by a second `take`.

**BLOCK admission gates (all unmet; BLOCK stays PARKED):** (1) Keep built from `sql_where` T-polarity, never from
`NOT P`; (2) `B` computed against the exact live-row universe of the version the scan will open (not the 4,096-slot
focus); (3) one version check that refuses if the dataset moved between mask build and scan (BLOCK fails open);
(4) `_rowid` space correct in stable-id mode; (5) executable negative fixture: `BLOCK(K(NOT P))` must differ from
`ALLOW(K)` on a NULL row, and `BLOCK(U \ K)` must equal it; (6) COUNT, SUM, attribution and Bag duplicates equal
across ALLOW and BLOCK. Even with all six, BLOCK's expected advantage is mask size and construction, not rows read,
and it costs metadata loads for every fragment.

**Verdict on the full chain** `SQL K3 → collapsed Keep → DR64 identities → version-checked Lance ALLOW/BLOCK →
projected read → Quack/ndarray terminal`:

- semantics: **LEGAL** for the SQL subset Quack admits (PostgreSQL-verified above);
- ALLOW route: **BLOCKED BY WIRING** (one coordinate authority, ord → `_rowid` composition, S-INGRESS);
- BLOCK route: **BLOCKED BY GATES** (above), not semantically impossible;
- outside the admitted subset (column-to-column, COALESCE, nullable arithmetic, LEFT join, subquery IN): refused or
  missing upstream of the mask, never silently lowered.

Recommended next step is unchanged: **P1** (A.6). No K3 counterexample found a correctness blocker on the
production path; the BLOCK negative fixture becomes required only when BLOCK is pursued.

### A.9 Consolidation amendment (2026-10-26): delta only

Reads A.0–A.8 as authority and adds only what they do not already say. No experiment was rerun.

**1. SQL capability status.** VERIFIED against PostgreSQL 16.15 (A.8): WHERE over nullable leaves with
NOT/AND/OR, exact on TRUE and FALSE rows; NULL operands; the aggregate subset in `qsp_agg_pg_parity`; contract
`normalize_kleene`. MISSING or caller-dependent: COUNT(expr)/MIN/MAX (caller adds `is_not_null`), COUNT DISTINCT
NULL keys, GROUP BY/HAVING with nullable key or value, column-to-column compare, COALESCE/NULLIF, nullable
arithmetic, LEFT join, subquery IN/NOT IN, and a named nullable `P OR NOT P` fixture. Unchanged from A.8.

**2. What blocks ALLOW.** Wiring, not semantics: one version-checked coordinate authority (A.1, A.5); the
DR ordinal → `_rowid` composition, which no code performs and which must look up through both maps rather than
store a third table; S-INGRESS (operator); P3 equivalence; and measured construction cost against bytes saved.

**3. Why BLOCK stays gated.** `BLOCK(U \ K(P))` equals `ALLOW(K(P))`, so it is expressible. It stays parked
because the six gates in A.8 are unmet, because it fails open on fragments added after the mask was built, and
because its gain is mask size, not rows read, while it loads metadata for every fragment. One outside
confirmation: DuckDB folds `a != b` into `!(a == b)` and `a >= b` into `!(b > a)` only on its no-NULL path
(`binary_executor.hpp:22-38,112-118`, `OperationNoNull`). Complementing a predicate is legal only where NULL
is absent, which is the same rule as gate (1).

**4. PgLance** (`lance-format/pglance`, Apache-2.0, `lance = "1.0"`, `arrow = "56.2"`; not Lance 13, so nothing
about Lance 13 is inferred from it). What it provides: a measured-shape reference for the cost of leaving the
fold, since it converts one row at a time and one cell at a time (`scan.rs:286-412` calls
`arrow_value_to_datum` per row per attribute, `convert.rs:208`, which re-dispatches on the Arrow type for every
cell and decodes dictionaries per cell, `:658`); a type map Arrow → PostgreSQL (`type_mapping.rs`), including the
refusal of `u64 > i64::MAX` (`convert.rs:354-368`) and NULL → `isnull` (`:213`). What it does not provide: no
filter pushdown (`get_foreign_plan` keeps every qual in PostgreSQL, `scan.rs:90`); no projection pushdown
(`create_stream` scans all columns, `:574`); no row mask, no ALLOW or BLOCK; no version pin (`Dataset::open`
at the latest version on each scan, `:146`); a fixed estimate of 1,000 rows (`:48`). Usable only as a
reference arm for tuple-conversion overhead. Not imported, and no FDW is proposed.

**5. DuckDB physical specialization → native polyfill.** At @1641475 DuckDB picks the physical type once per
vector (`execute_comparison.cpp:40`, one `switch` over the type, then a monomorphised loop), specialises on
vector form (flat, constant, dictionary) and on validity (a separate no-NULL path), and leaves ternary
expressions generic (`ternary_executor.hpp:17-35`, routed through `VariadicExecutor`). D-NAM-4 already moves the
choice earlier, to preparation, with no lookup in the loop, so the type dispatch adds nothing new. Two rules
are new and worth carrying:

- *A code table pays only when it is amortised.* DuckDB evaluates a function over a dictionary instead of the
  rows only when the dictionary has at most half as many entries as the rows and the function cannot error
  (`unary_executor.hpp:124-127`). Its cross-chunk cache is limited to storage dictionaries under 20,000 entries,
  and above 2,048 entries only when the chunk is at least half full, "to protect against selective filters"
  (`execute_function.cpp:51-84`). DuckDB's dictionary-execution thresholds demonstrate workload-dependent
  amortization, but do not establish the crossover between a 65,536-entry LUT and bit-exact SIMD arithmetic. That
  remains an independent D-TFLP-2 measurement (open question at `d-tflp-tflite-palette-polyfill-capstone-v1.md:277`).
- *Complement only without NULL* (item 3).

| shape | ONNX | TFLite | native route (prepared once) |
|---|---|---|---|
| unary float | Tanh, Exp | float activation | ndarray VML (`vstanh`, `vsexp`, `hpc/vml.rs`) |
| unary int8 | quantized activation | TANH, LOGISTIC | 256-entry table built at preparation (`d-tflp-tflite-palette-polyfill-capstone-v1.md:53`; PZL-1 in `d-nam-palette-native-lut-capstone-v1.md:69`) |
| binary float | Add, Mul | ADD, MUL | ndarray direct (`vsadd`, `vsmul`) |
| binary int8 | quantized op, where admitted | ADD, MUL | 65,536-entry table or fixed-point SIMD (`simd_fixedpoint.rs`); measure both and keep the faster (open question at `d-tflp-tflite-palette-polyfill-capstone-v1.md:277`), apply the amortisation rule |
| predicate | Greater | GREATER | ndarray compare to mask, then R2IL / mask-risc |
| reduction | ReduceSum | SUM, Softmax stages | ndarray fold; Softmax is not a unary table (`d-nam-palette-native-lut-capstone-v1.md:58`) |

RelAlg is not extended with Tanh, MatMul or Conv (D-NAM-3: no float math in `Rel`). Nothing in the hot path
matches an ONNX or TFLite operator.

**Evidence roles stay separate.** PostgreSQL = SQL semantics; DuckDB = reference performance and
specialisation; PgLance = tuple-conversion overhead; Quack / R2IL / mask-risc = resident folds; ndarray =
kernels; Lance 13 = I/O in bytes, ranges and rows; DRF = reuse. They are never combined into one speed-up.

**6. Does anything displace P1?** No. PgLance has no masked read to compare against, and DuckDB's rules apply to
the native polyfill, not to the Lance handoff. P1 (A.6) remains the smallest next step; it has no dependency on M1 (it reads no DuckDB output);
S-INGRESS and the coordinate authority remain operator decisions.

### A.10 Scope clarification (2026-10-26): sealed-read masks only; writer and resident engine unchanged

Documentation only. Nothing here is an implementation requirement. Squash, release and their dependency handling,
`MailboxSoA`, `CycleWriter`, the 512-byte node layout, DR64 execution and model handling are out of scope and
unchanged. A.9 item 5 was corrected in place (one sentence: the DuckDB thresholds show amortisation, not the
LUT/SIMD crossover, which stays a separate D-TFLP-2 measurement).

**Where A.3–A.9 apply.** Everything about Lance masks in this appendix concerns **reading committed, sealed Lance
versions**. It is not a statement about how changes are written. The write path is:
resident SoA (DR64 focus = 4,096 of 65,536 addresses, i.e. 1/16 of the address space; up to 64 thought layers, a
separate axis from the 32 × 16-byte slots of the 512-byte node) → repeated native folds → existing squash/release →
coalesced dirty payload → existing asynchronous `CycleWriter` handoff → background freeze/WAL/Lance commit. The
focus reduction is the optimisation; it never licenses dropping a changed field, and a read projection must never
be reused as a write projection. Field-activity share and amortisation depth (operator targets: roughly 40–70 % of
lanes in focused nodes, up to about a million operations per handoff) are execution economics, not persistence
rules.

**Writer ownership, verified in source.**
- Exclusivity comes from holding `MailboxSoA` by move, with column writes through the owner's `&mut self` ops
  (`mailbox_soa.rs:747-750`, `:790-800`). `MailboxSoaOwner`'s contract-level mutation surface is the phase
  transition only (`soa_view.rs:295-301`).
- `MailboxSoA::cast_on_behalf(&self, …)` pairs the cast with `self.mailbox_id()` (`mailbox_soa.rs:765-772`). The
  `&self` borrow does not hand the SoA to anyone.
- `BatchWriter` is "ephemeral staging, not a durable WAL" (`batch_writer.rs:23`). Its payload is a descriptor; deltas
  stay in the SoA backing store until flush (Addendum-6, `:29-32`). Its own doc notes `cast()` has no production call
  site yet (`:16-19`).
- `CycleWriterHandle::try_submit` moves the `Vec<SweepSlot>` into the writer and returns a `SealTicket` without
  waiting (`cycle_writer.rs:166-179`). The writer task owns the sink and calls the existing `seal_cycle`
  (`cycle_writer.rs:314-315`; `seal_cycle` is defined in `cycle_driver.rs:434`, not in `persist_sink.rs`). It frames
  each cycle on the head the previous one published, so the caller need not know it (`:29-34`). Transitions apply only
  after the ticket resolves `Ok`; a failed seal poisons the queue and hands every cast back (`:36-50`).
- These are two ownership boundaries: a staged `BatchWriter` descriptor still refers to live SoA bytes, while a
  submitted `Vec<SweepSlot>` is owned by the writer (`SweepSlot.payload: Vec<u8>`, `persist_sink.rs:193`). Neither is
  a general zero-copy guarantee.

**Lance mask ownership is separate.** `Scanner::with_row_addr_prefilter(&mut self, mask: RowAddrMask)` takes the mask
by value and keeps it in the scanner (`lance-13.0.0/src/dataset/scanner.rs:1663-1665`). The resident DR mask can stay
borrowed while the Lance mask is built; that conversion is a representation copy at the sealed-read boundary (the
crepe probe's bit enumeration, `crepe-plan-probe/src/lance_store.rs:264-268`). It does not touch `MailboxSoA`'s
write ownership and belongs outside the resident fold.

**MemWAL is outside this mask capability.** From the Lance 13.0.0 and lancedb 0.40.0 sources:
- The MemWAL scanner's block list is internal, built from primary-key membership between generations
  (`dataset/mem_wal/scanner/block_list.rs:4-17`) and applied as `PkBlockFilterExec` to drop superseded vector
  candidates.
- `LsmScanner` exposes `project`, `filter`, `limit`, `nearest`, full-text search, `with_row_address`,
  `with_memtable_gen`, and no `RowAddrMask` or external ALLOW/BLOCK input (`mem_wal/scanner/builder.rs:429-563`).
- The fresh tier has no stable `_rowid`; lancedb rejects `with_row_id` there and joins hybrid search on a
  primary-key surrogate (`lancedb-0.40.0/src/query/wal_fusion.rs:4-16`). A MemWAL `_rowaddr` only orders rows within
  one generation (`builder.rs:551-555`).
- `use_lsm(false)` reads the sealed base only and misses fresh records (`lancedb-0.40.0/src/table/query/lsm.rs:6-21`).
  Filtering fresh rows by primary key is a different operation in a different coordinate space.
- lancedb's `materialized_view/vector_dedup.rs:414-434` is a sealed-dataset precedent for coordinate checking
  (offsets → `_rowid` in stable-id mode, out-of-fragment refusal, one fragment per scan), not a MemWAL interface.
- Our WAL components (`persist_sink.rs`, `cycle_writer.rs`, `batch_writer.rs`) have no read-side mask.

Consequently there is no MemWAL mask adapter, no use of a block list to skip clean rows during writes, and no
primary-key substitute for DR ordinals. Add **MemWAL primary key** as a sixth space to the A.1 list: DR64 focus slot
≠ canonical SoA ordinal ≠ stable `_rowid` ≠ physical `_rowaddr` ≠ Arrow output position ≠ MemWAL primary key.

**Status, unchanged:** ALLOW is a possible sealed-read integration behind the coordinate and S-INGRESS gates; BLOCK
stays parked behind the six A.8 gates. Recommendations awaiting the operator (not authorised): S-INGRESS as a named,
measured DR-mask → Lance-mask conversion at the sealed-read boundary, never claimed zero-copy; one checked binding
carrying both the dataset-version witness (H14c) and the index-seal witness (DRF), built from the existing maps, with
no third table and no Lance types in the contract. **P1** (A.6) remains next, offline over committed sealed
H14c/H14d data. P1 does not depend on M1. M1's status is unchanged: answer parity passed;
the Quack-vs-DuckDB timing comparison is inconclusive because the measured boundaries differ.
