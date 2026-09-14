# duckdb-to-v3-translation-matrix-v1 — Phase 1 of the mask execution engine arc

> **Status:** DRAFT v1 — Phase 1 deliverable. This document **decides nothing and
> builds nothing.** It is a per-concept translation matrix: for every DuckDB
> vector-execution concept, what the V3 mask substrate does instead, whether that
> is better/worse/equal, and — non-negotiably — **what measurement would flip the
> verdict**.
>
> **Register discipline.** Every "exists" claim carries `file:line` at read time
> (2026-09-14). Every claim not verified against code in this session is marked
> **`[claimed, unverified]`**. Verdicts are *proposals with falsifiers*, never
> findings; a verdict with no falsifier that could fire is exactly the vacuous
> assertion `CLAUDE.md` § "The falsifiability rule" forbids, and three rows below
> were re-written when their first falsifier turned out to be unfireable.
>
> **Sources read.** DuckDB checkout at
> `<scratchpad>/duckdb/src` (22 TUs + the headers they dispatch into);
> `ndarray/src/simd_masking_ops.rs` (2730 lines, 31 `pub fn` at the time of this read; 33 after `mask_shift_morton` + `MortonDir` landed the same day) and
> `ndarray/src/bitwise.rs`; `ndarray/.claude/blackboard.md` entries 2026-09-13
> and 2026-09-14; the `ruff_cpp_spo` harvest under `<scratchpad>/harvest/duckdb/`;
> the consumer side (`crates/lance-graph-mask-risc/src/{ir.rs,lib.rs}`,
> `lance-graph-java/.claude/plans/mask-risc-lowering-v1.md` §0–§3,
> `.claude/v3/soa_layout/le-contract.md` §3).

---

## §0 — The three laws this matrix is written under

Restated, not claimed — each is already ruled elsewhere and this document has no
authority to widen any of them.

1. **The three-layer contract** (`blackboard.md` 2026-09-13, operator-ruled).
   Consumers speak semantic ops; `simd_masking_ops.rs` owns slice/chunk/tail
   ergonomics and **never names an ISA**; `simd.rs` owns architecture-agnostic
   lane types; `simd_{avx512,avx2,neon,wasm,scalar}.rs` each own a realization
   **as peers**. Nothing in this matrix may propose a consumer-side compute path.

2. **POLYFILL LAW** (same entry). Every public mask primitive has compile-time
   implementations for all five backends. **Scalar is a peer backend, not a
   fallback.** No runtime ISA dispatch, no fallback chains. A translation that
   requires the consumer to branch on ISA is rejected at the matrix level, before
   it reaches design.

3. **BACKEND LAW** (same entry). No shared generic implementation body the
   backends delegate into; shared *tests* and shared *generated* truth-table logic
   are fine, a shared *runtime* body is not. The route to remove repeated source
   is code generation emitting backend-LOCAL bodies
   (`tools/gen_ternlog_bodies.py`).

**Planning register.** Per the operator ruling of 2026-09-05
(`CLAUDE.md:1017-1025`, board
`E-PLANNING-MIGRATES-TO-LOCO-R2IL-DATAFUSION-IS-GRACE-PERIOD-1`): *"Every
planning is in migration to ogar-loco and ogar-r2il, especially datafusion is out
of the picture, what exists gets a grace period, nothing new will migrate to
it."* **Consequence binding on this matrix: no row may propose landing behaviour
in `datafusion_planner`, `sql_query`, `rls.rs`, or any `datafusion-*` surface.**
A KEEP or ADAPT verdict here lands in a loco/r2il program or directly on the mask
substrate, never in the grace-period planner. Where a DuckDB concept has an
obvious DataFusion-shaped home, that is a reason to route it away, not toward.

---

## §1 — The two representations, dimensioned

Every verdict below is downstream of this one comparison, so it is stated once
with its arithmetic rather than repeated per row.

| quantity | DuckDB | V3 mask |
|---|---|---|
| selection carrier | `SelectionVector` = `sel_t*` index list, `sel_t = uint32_t` (`typedefs.hpp:30`) | packed `u64` bit-plane, bit `i%64` of word `i/64`, LSB-first, tail zeroed (`simd_masking_ops.rs:55-70`, NORMATIVE) |
| chunk width | `STANDARD_VECTOR_SIZE = 2048` (`vector_size.hpp:16,20`) | the population of the version; no chunk boundary in the representation |
| bytes for one full 2048-row selection | 2048 × 4 = **8192 B** | 2048 / 8 = **256 B** |
| ratio | — | **32×** |
| cost of AND of two selections | re-scan + re-materialize an index list | one pass of `u64 &`, `mask_and` (`simd_masking_ops.rs:377`) |
| cost of a 3-term Boolean | two passes | **one** pass, `mask_ternlog<IMM>` (`:588`) |
| hot-path allocation | `make_uniq<SelectionVector>(STANDARD_VECTOR_SIZE)` per conjunction Select (`execute_conjunction.cpp:76,79`) | **0 B/step measured** with a counting allocator at every density and depth (`blackboard.md` 2026-09-05, D-GTM-0k; probe `ndarray/examples/hex_trie_vs_gemm_probe.rs`) |

The 32× is memory, not speed, and it is the *least* interesting number here. The
two that carry the matrix are the ternlog fusion (3 masks, 1 pass) and the
measured zero-allocation hot path.

---

## §2 — THE TRANSLATION MATRIX

Verdict vocabulary, used strictly:

- **KEEP** — the mechanism transfers essentially unchanged; a V3 counterpart
  already exists and does the same thing.
- **ADAPT** — the *intent* transfers, the *shape* does not; a named V3 primitive
  (or composition) replaces it.
- **ELIMINATE** — the mechanism exists only to service a representation V3 does
  not have. There is nothing to port.
- **V3 BETTER** — V3 answers the same question structurally cheaper, with a
  measured or mechanically-derivable margin.
- **NEEDS FALSIFIER** — the verdict genuinely is not determined by anything read
  this session; the row's falsifier is the *next probe*, not a tie-breaker.

`[T1 gap: <name>]` in the V3 column means the counterpart does not exist in
`ndarray::simd` today; every one is collected in §3.

### 2.1 Representation

| # | DuckDB concept | verdict | DuckDB file:line | V3 / ndarray counterpart | mechanism (one sentence) | FALSIFIER |
|---|---|---|---|---|---|---|
| R1 | `SelectionVector` (index list, `sel_t*` + `SelectionData` owned buffer) | **ELIMINATE** | `selection_vector.hpp:31-180`; `set_index` `:124-127`; `get_index` `:136-138` | packed `u64` mask; algebra at `simd_masking_ops.rs:377-680` | A selection vector IS the index-list materialization the mask-native invariant forbids: it stores one 4-byte row id per *surviving* row, so every composition re-derives an ordering the bit-plane never lost. `mask-risc/src/lib.rs:31-33` states the consequence structurally — *"a `SelectionVector` cannot be expressed in this vocabulary at all."* | A workload where the surviving fraction is low enough that an index list is cheaper to *consume* than a sparse mask is to *scan*. Concretely: at selectivity s over N rows, the list costs 4sN bytes read, the mask N/8 bytes read; the list wins for s < 1/32 **only if the consumer cannot skip zero words**. Measure `masked_sum_i32` (`:692`) against an index-list sum at s ∈ {1/64, 1/256, 1/1024}; if the mask loses at any of them, R1 downgrades to ADAPT with a documented sparse threshold. |
| R2 | `FlatVector` | **KEEP** | `vector_type.hpp:16` | the SoA value lane itself; `LaneRef::{I32,U32,U64}` (`mask-risc/src/ir.rs:16-23`) | A flat vector is just "the column, contiguous" — which is what a V3 SoA lane already is. Nothing to translate; it is the *baseline* both systems share. | None available. This row cannot fail — it asserts an identity. Kept in the matrix only so a reader does not infer it was overlooked; it carries **no evidentiary weight**. |
| R3 | `ConstantVector` (one value, logical count N) | **ELIMINATE** | `vector_type.hpp:17`; `execute_constant.cpp:14-18` (`result.Reference(value, count)`) | the scalar argument: `gt_i32_to_mask(values, threshold, out)` (`simd_masking_ops.rs:340`) | A constant becomes a *vector* in DuckDB because every kernel signature takes two Vectors; V3's predicate signature takes a lane and a **scalar**, so a constant never acquires a representation at all. | A predicate whose constant side is genuinely per-row (a column-vs-column compare). Today `simd_masking_ops` has **no** lane-vs-lane compare — every `*_to_mask` takes a scalar. If column-vs-column is needed, R3 does not change but a new T1 primitive is required (§3, G7). |
| R4 | `DictionaryVector` (a `SelectionVector` on top of another vector) | **ELIMINATE** | `vector_type.hpp:18`; `Vector::Slice(sel,count)` `vector.cpp:221-234`; `Vector::Dictionary` `:250-270`; `SelCache` merge `:236-248` | none — the mask is the *second argument* to the consumer, never a rewrap of the data | A dictionary vector exists so a filtered column can be passed onward without copying. V3 never rewraps: the column stays `&[i32]`, the surviving set travels alongside as `&[u64]`. The tell is `SelCache`: DuckDB needs a cache to compose dictionary-on-dictionary, because composed index lists are expensive to re-derive; composed masks are one `mask_and` pass with nothing to cache. | A consumer that must hand a *narrowed column* to an API it does not control (an Arrow export, an FFI boundary that takes only values). There, compaction is real work, not an artifact — see C2, and §3 G5 (`vpcompress`-style masked gather). If such a consumer is on the critical path, R4's "nothing to translate" is false for it. |
| R5 | `SequenceVector` (start, increment, count) | **ADAPT — and this is the closest DuckDB comes to the V3 address** | `vector_type.hpp:19`; `Vector::Sequence` `vector.cpp:498-500` | a **range write** on the mask; `[T1 gap: mask_set_range]` (§3 G6) | A sequence vector compresses a contiguous range to 3 scalars — exactly the V3 insight that a minted prefix is a contiguous row range. But DuckDB **throws it away at the first predicate**: `Vector::ToUnifiedFormat` (`vector.cpp:458-467`) flattens anything that is not FLAT/CONSTANT/DICTIONARY, so a sequence is materialized to N values before any kernel sees it. V3 keeps the range as a range: D-GTM-0m measured a trie-node reveal (`2^(16-4L)` contiguous rows) at **49–99 ns** against **22.4–22.8 µs** for the general TCAM sweep — **228–462×** (`blackboard.md` 2026-09-14 (2)). | The 228–462× is measured on ONE fixture (65,536 rows = 256×256 axial hex, 62 % permeable, one tile size, timing floor 50 ms, no `perf` — the probe's own stated limits). Re-run `hex_tenant_mq_probe.rs` at a second tile size and a second density. If the range/sweep ratio collapses below ~10× the "top-down is a range, not a compare" claim needs re-scoping to the specific geometry. **Separately:** `mask_set_range` does not exist, so the 49 ns figure is the probe's inline range write, not a shipped primitive. |
| R6 | `ValidityMask` (bit-per-row NULL bitmap, `validity_t = uint64_t`) | **KEEP the representation, ELIMINATE the role** | `validity_mask.hpp:50` (`validity_t`), `:64-65` (`BITS_PER_VALUE`, `STANDARD_ENTRY_COUNT`), `Combine` `validity_mask.cpp:47-74` | the *same* packed-u64 shape — `ValidityMask::Combine`'s word loop at `validity_mask.cpp:72` (`result[i] = data[i] & other[i]`) **is** `mask_and` (`simd_masking_ops.rs:377`) | DuckDB independently arrived at V3's exact carrier for its NULL mask and then used it for one purpose only. The *shape* is confirmation, not a translation task. The *role* is what V3 eliminates: per `CLAUDE.md:1657` § CANON, "zero = fall through to the broader default", so absence is already encoded in the value rail and needs no parallel bitmap. `mask-risc/src/lib.rs:44-48` states it: *"there is no NULL: absence in the V3 substrate is a zero-fallback, never a validity bit."* | A consumer that must serve **SQL three-valued semantics at a membrane** — where `NULL` and `0` are distinguishable and an external contract says so. Then the second plane returns (see L2/L3 below) and "ELIMINATE the role" is false for that membrane. Falsifiable now: point at any V3 rail whose zero is *meaningfully different* from "not consulted". If one exists, this row is wrong. |
| R7 | `UnifiedVectorFormat` (`{sel, data, validity, owned_sel, physical_type}`) | **ADAPT** | `unified_vector_format.hpp:22-66`; `Vector::ToUnifiedFormat` `vector.cpp:458-467`; `DataChunk::ToUnifiedFormat` `data_chunk.cpp:401-408` | `Planes { n_rows, masks: &[&[u64]], lanes: &[LaneRef] }` (`mask-risc/src/ir.rs:41-50`) | Both are the same move: collapse N carrier shapes into ONE access triple so kernels have one path. DuckDB needs it because it has six `VectorType`s; V3's `Planes` needs it because it has several *lane widths*. Note DuckDB already concedes half the collapse — FSST/SEQUENCE/SHREDDED are **flattened** before unification (`vector.cpp:461-465`), i.e. three of six types do not survive their own abstraction. | `Planes` currently carries `masks` and `lanes` but **no per-lane validity and no `sel`** — because R1 and R6 eliminate both. If a real consumer needs a third member, the collapse is incomplete and `Planes` is under-specified. Concretely: attempt to express a `RowMatcher`-shaped multi-column equi-match (M1) purely in `Planes` + `Pred::MatchU64`; if it needs a field `Planes` lacks, this row moves to NEEDS FALSIFIER. |
| R8 | `DataChunk` + `STANDARD_VECTOR_SIZE = 2048` | **ADAPT (the chunk survives; the *value* 2048 does not transfer)** | `data_chunk.cpp:32-60` (Initialize), `:107` (Reset), `:356-400` (Slice family); `vector_size.hpp:16` | the version's SoA: 512 × 32 × (4+12) (`le-contract.md` §1/§3; `mask-risc-lowering-v1.md` §0) | 2048 is a **cache-residency** choice (2048 × 4 B = 8 KiB per column, L1-sized) and that reasoning survives; the *number* is DuckDB's, tied to its 4-byte sel. The V3 analog is the 1024-row survivor-word chunk already named in `MaskOp::Pred { under }` (`ir.rs:84-86`). **`DataChunk::Slice(offset, count)` at `data_chunk.cpp:389-399` is the single sharpest citation in this document**: it builds a whole `SelectionVector` in a per-row loop (`sel.set_index(i, offset + i)`, `:395-397`) **to express a contiguous range**. That is R5's thrown-away sequence, re-manufactured. | The 1024-row `under` chunk in `ir.rs:84` is an *asserted* granularity with no measurement behind it. Sweep the survivor-skip chunk width ∈ {256, 512, 1024, 4096} on the D-GTM-0m fixture; if the optimum is not near 1024, the IR's constant is wrong and this row's "the chunk survives" needs the real number. |

### 2.2 Expression execution

| # | DuckDB concept | verdict | DuckDB file:line | V3 / ndarray counterpart | mechanism | FALSIFIER |
|---|---|---|---|---|---|---|
| E1 | `ExpressionExecutor` state tree (`ExpressionState` with `child_states`, `intermediate_chunk`, `types`) | **ADAPT** | `expression_executor_state.hpp:23-45`; `ExpressionExecutorState` `:89-96`; `ExpressionExecutor::Execute` dispatch `expression_executor.cpp:256` | `Program { ops: Vec<MaskOp>, terminal, scratch_slots }` (`mask-risc/src/ir.rs:131-140`) + caller-owned `Scratch` | DuckDB carries a *tree* with per-node intermediate storage, walked recursively per chunk. V3 flattens to a **straight-line program** over numbered scratch slots computed once (`Program::new`, `ir.rs:143-163`). The tree's `intermediate_chunk` becomes `scratch_slots`, sized at assembly, owned by the caller — which is what makes the 0 B/step hot path structural rather than careful. | A Boolean expression that is **not** expressible as straight-line code over a fixed slot count. Candidate: an expression whose scratch requirement depends on data (none in the current `Pred`/`MaskOp` vocabulary — verify by construction). If one is found, `scratch_slots` cannot be computed at assembly and E1 is wrong. |
| E2 | `ExpressionExecutor::Select` vs `Execute` (two return shapes: index list vs value vector) | **ADAPT — collapse the two into one** | `Select` `expression_executor.cpp:315-331`; `Execute` `:256`; `SelectExpression` `:111-130` | one `Program` whose `Terminal` names what is wanted (`ir.rs:108-128`): `Count`/`Any`/`All`/`MaskedSum`/`Min`/`Max`/`Blend`/`Keep` | The Select/Execute split exists because a *predicate* wants a selection and an *expression* wants values, and the two carriers differ. With a mask, both are the same object: the program computes a mask, and the `Terminal` decides whether to popcount it, reduce a lane under it, blend with it, or hand it back. **N result shapes collapse to 1 carrier + 1 terminal tag.** | `Terminal` has 8 variants and `Keep` is the escape hatch. If a real consumer's need is served only by `Keep` + consumer-side per-row work, the collapse is cosmetic. Test: express the three `lgj-abi` plan-eval consumers (`eq_u32`, `gt_i32`, `eq_classid` — cited `mask-risc-lowering-v1.md` §1) end-to-end without touching `Keep`. |
| E3 | `ExpressionExecutor::DefaultSelect` (materialize `bool[2048]`, then scalar-loop to an index list) | **ELIMINATE** | `expression_executor.cpp:373-394`; the loop `DefaultSelectLoop` `:333-355` | `Pred` → mask directly (`ir.rs:84-86`; `gt_i32_to_mask` `simd_masking_ops.rs:340`) | `bool intermediate_bools[STANDARD_VECTOR_SIZE]` at `:379` is **2048 bytes of one-byte booleans** that exist purely to be re-read by `DefaultSelectLoop` and converted to 8192 bytes of indices. V3's predicate writes the 256-byte mask in one pass and there is no intermediate. The byte-per-row bool vector is the purest instance of "a representation that exists only to be converted". | None that changes the verdict — but the *scope* is falsifiable: this is DuckDB's **generic fallback** path, taken when no specialized `Select` exists for the expression. If the specialized path dominates real workloads, eliminating the fallback is worth less than the 2048-vs-256 arithmetic suggests. Measure the fraction of `Select` calls reaching `DefaultSelect` on a representative query set before quoting E3 as a win. |
| E4 | `Execute*` — comparison (`TryPrimitiveSelectOperation`, 14-way physical-type switch) | **ADAPT (narrow)** | `execute_comparison.cpp:18-100`; the 14 arms `:40-96` | 6 `*_i32_to_mask` + 2 `*_u32_to_mask` (`simd_masking_ops.rs:340,881,920,947,975,1017,164,1044`) | The 14-way switch is one `BinaryExecutor::Select<T,T,OP>` instantiation per physical type. V3 covers **i32 and u32 only** today. That is not a defect of the design, it is the current T1 surface: `[T1 gap: u8/u16/u64 compare-to-mask]` (§3 G1, G2). For the widths it covers the correspondence is exact, including operand-swap derivation — DuckDB's `LessThan` is literally `GreaterThan(right, left)` (`execute_comparison.cpp:194-200`, the swap at `:198`), and ndarray's `lt_i32_to_mask` is `t.gt_bitmask(value)` (`simd_masking_ops.rs:891`), the same trick. | **The width gap is already measured as load-bearing**: D-GTM-0m widened a u8 permeability column 4× to use `gt_i32_to_mask`, and reported its `n_gen` and coal numbers as **upper bounds** because of it (`blackboard.md` 2026-09-14 (2), "Stated limits"). Build `gt_u8_to_mask` and re-run; if the 8.9 µs re-chain does not improve, the widening was not the cost and G1's priority drops. |
| E5 | `Execute*` — conjunction AND (short-circuit over a narrowing selection) | **ADAPT** | `execute_conjunction.cpp:60-104`; break at `:94-96`; per-call alloc `:76,79` | `mask_and`/`mask_ternlog` + `mask_any` early-out (`simd_masking_ops.rs:377,588,1215`) | Term k evaluates only on survivors of terms 1..k-1 — DuckDB does this by shrinking the index list, V3 by ANDing masks and (per `MaskOp::Pred { under }`, `ir.rs:84-86`) skipping chunks with no survivor. **Two asymmetries, opposite signs.** In V3's favour: three terms fuse into one `mask_ternlog` pass. Against: DuckDB's `current_count == 0` short-circuit (`execute_conjunction.cpp:94`) is **O(1)** (it carries a count), while `mask_any` at `simd_masking_ops.rs:1215-1221` ORs **every word with no early exit** — O(n/64). | Two, and they must both fire. (a) Emptiness: measure `mask_any` against a count carried alongside the mask, at n_rows ∈ {64K, 1M} with the mask empty. If the O(n/64) scan is a measurable fraction of a step, `[T1 gap: cheap emptiness]` (§3 G8) is real. (b) Fusion: measure a 3-term conjunction as `and+and` vs one `ternlog`. D-GTM-0m fitted `ternlogq = 291 ns/pass` (8 KiB masks, 0.285 ns/word); if `and+and` is not ~2× that, the fusion claim is wrong. |
| E6 | `Execute*` — conjunction OR (accumulate, then **sort** to restore row order) | **V3 BETTER** | `execute_conjunction.cpp:106-144`; the sort `:139` | `mask_or` / `mask_or_assign` (`simd_masking_ops.rs:410,465`) | DuckDB's OR walks the *false* set each round, so passing rows arrive out of order and `true_sel->Sort(result_count)` at `:139` must restore it. **A packed mask has no order to restore**: OR is bitwise, commutative, and the row's position IS its bit position. The sort is not an optimization DuckDB failed to make — it is a cost the index-list representation creates. | The sort's real cost. `SelectionVector::Sort` is called once per OR-Select over at most 2048 `uint32_t`. If that is a negligible fraction of an OR-heavy query, E6 is *correct but immaterial* and should not be quoted as a win. Measure before citing. |
| E7 | `Execute*` — CASE (WHEN/THEN cascade over a narrowing false-set) | **ADAPT** | `execute_case.cpp:33-90`; state-held sel vectors `:12-19`; scatter `TemplatedFillLoop` `:92-112` | `blend_i32` (`simd_masking_ops.rs:1582`) + `mask_andnot_assign` (`:537`), i.e. `Terminal::BlendI32` (`ir.rs:124`) | `remaining &= !when_k` is the cascade; `blend` is THEN/ELSE. The scatter-by-index (`res[sel.get_index(i)] = ...`, `execute_case.cpp:108`) disappears: every row is written, selected by mask, no index indirection. **Worth copying from DuckDB, not eliminating:** `CaseExpressionState` (`:12-19`) pre-allocates its two sel vectors **in the state**, per-call-allocation-free — the pattern E5's conjunction executor violates at `execute_conjunction.cpp:76`. DuckDB is inconsistent with itself here and CASE is the half that is right. | **Blend is eager: it evaluates BOTH arms for ALL rows.** DuckDB's cascade evaluates each arm only on rows that reached it. Falsifier: at what per-row arm cost does eager blend lose? Sweep arm cost (a trivial lane read → a multi-op sub-program) at WHEN-selectivity ∈ {0.01, 0.5, 0.99}. Any crossover means E7 needs a cost condition, not a flat ADAPT. Second, harder falsifier: DuckDB's `AdaptiveFilter` refuses to permute a term that `CanThrow()` (`adaptive_filter.cpp:22`) — an arm with an **error condition** cannot be eagerly evaluated at all. If V3 ever admits a faulting predicate, blend is unsound, not merely slower. |
| E8 | `Execute*` — constant | **ELIMINATE** | `execute_constant.cpp:14-18` | see R3 | Same mechanism as R3: the constant is a scalar argument, never a vector. | Same as R3. |
| E9 | `Execute*` — reference (`result.Slice(chunk->data[i], *sel, count)`) | **ELIMINATE** | `execute_reference.cpp:13-23` | `LaneRef` borrow (`ir.rs:16-23`) | With a sel present, a reference *wraps* the column into a dictionary vector (R4); without one it is a plain reference. V3 has only the second case: a lane reference is `&[i32]`, and the mask travels separately. The `if (sel)` branch at `:19` has no V3 counterpart at all. | Same as R4 — if a consumer genuinely needs a compacted column handed onward, the branch's work reappears as `[T1 gap: masked compaction]` (§3 G5). |
| E10 | `Execute*` — operator: `IN` (OR-fold of per-constant equality) | **ADAPT** | `execute_operator.cpp:24-63`; per-child `Vector comp_res` + `Vector new_result` `:43,53` | k × `eq_i32_to_mask` + `⌈(k-1)/2⌉ × mask_ternlog<OR3>` (`simd.rs:604`, `simd_masking_ops.rs:588`) | For an IN list of k constants DuckDB does k compares and k−1 ORs over byte-bools, constructing **two `Vector`s per element** (`:43,53`). V3 does k mask generations and folds three at a time with `OR3`, halving the fold. | The fold is not the cost — the k predicate sweeps are. At k=10 over a 64K column, the folds are ~10 × 291 ns ≈ 3 µs against 10 × 22.4 µs of sweeps (D-GTM-0m rates). **So the ternlog halving of the fold is ~1.4 % of the operation.** State it that way or not at all. The real question, and the actual falsifier: can k equality sweeps become ONE `ternary_match` pass when the k constants share a prefix? Measure against `ternary_match_u32_to_mask` (`:1285`) with a care mask covering the common prefix. |
| E11 | `Execute*` — operator: `COALESCE` (4 sel vectors, ping-pong, per-row validity branch) | **ADAPT** | `execute_operator.cpp:64-109`; the four allocations `:66-69` | the same cascade as E7: `mask_andnot_assign` + `blend_i32` | COALESCE is CASE with the WHEN being "is valid". Four `SelectionVector(count)` per call become zero allocations and two mask passes per arm. | Inherits E7's eager-blend falsifier exactly. Additionally: COALESCE's WHEN is *validity*, so under R6 (no NULL bitmap) the "is valid" predicate must come from the rail's zero-fallback. If a V3 lane exists where zero is a legal value, COALESCE cannot be expressed and this row is wrong for it. |

### 2.3 Comparison kernels, NULL semantics, distinctness

| # | DuckDB concept | verdict | DuckDB file:line | V3 / ndarray counterpart | mechanism | FALSIFIER |
|---|---|---|---|---|---|---|
| C1 | `TemplatedComparisonOperation` / `ScalarExecutor::SelectFlatLoop` (the innermost select loop) | **ADAPT — and DuckDB half-discovered the answer here** | `scalar_executor.hpp:580-610`; the three-way word branch `:588-607`; `binary_executor.hpp:241-246` | `gt_i32_to_mask` et al. (`simd_masking_ops.rs:340-1070`) | `SelectFlatLoop` walks **validity in u64 words** and branches three ways per word: `AllValid` → skip the per-row null test; `NoneValid` (`scalar_executor.hpp:596`) → `AppendInvalidRange` (`:597`), **64 rows dispatched at once**; mixed → per-row. DuckDB therefore already has the word-level skip — **for the NULL mask only.** It cannot apply the same skip to the *selection*, because the selection is an index list with no word structure. V3 gets both skips from one representation. | The skip only pays if words are actually uniform. Measure the fraction of `AllValid`/`NoneValid` words on a realistic V3 mask after 2–3 conjunctive narrowings. If survivors are uniformly scattered, every word is mixed and the skip is worth nothing — which would also invalidate `MaskOp::Pred { under }`'s premise (`ir.rs:84-86`). This is the same measurement E5(a) needs; run it once. |
| C2 | `ScalarExecutor::Select*` sink (`local_sink.Append(result, idx)`) | **ELIMINATE** | `scalar_executor.hpp:594,606` (Append), `:597` (AppendInvalidRange) | the mask word is the sink; `out_words[g/4] \|= (bits as u64) << ((g%4)*16)` (`simd_masking_ops.rs:894-895`) | The sink abstraction exists to hide "which of true_sel/false_sel/both am I filling", a question that only arises because the output is an index list with a length. A mask has a fixed size and both polarities are one `mask_not` (`:1072`) apart. | If a consumer needs true_sel AND false_sel **as compacted lists** simultaneously, the mask gives one object and two compactions, not two lists in one pass. That is §3 G5 territory. Measure only if a consumer appears. |
| C3 | Three-valued AND/OR (`TernaryAnd`, `TernaryOr`) | **ADAPT — a 2-plane, 3-ternlog composition (derivation NOT yet verified)** | `boolean_operators.cpp:80-107` (`TernaryAnd`); the loops `:44-60` | 2 planes (`v` = value, `k` = known) + `mask_ternlog` (`simd_masking_ops.rs:588`); cf. Belnap 2-plane precedent `mask-risc-lowering-v1.md` §1 (`E-THE-STATE-LAYER-IS-A-BELNAP-BILATTICE-…-1`) | Encode a SQL boolean as two masks: TRUE=(v=1,k=1), FALSE=(v=0,k=1), NULL=(k=0). Then `v_out = v_a & v_b` (one `mask_and`), and `k_out = (k_a & k_b) \| (k_a & ~v_a) \| (k_b & ~v_b)` — a **4-input** function, so it does **not** fit one ternlog; it decomposes into two: `t = f(k_a, v_a, k_b)` then `k_out = g(t, k_b, v_b)`. **Three word-passes total** against DuckDB's branchy per-row loop at `boolean_operators.cpp:44-56`. ⚠ **The two immediates are NOT derived in this document** and must be computed against `tools/gen_ternlog_bodies.py`'s 256-table self-check before anyone writes them down. | Two, both required. (a) **The decomposition itself**: verify the 4-input truth table against `boolean_operators.cpp:80-107` row by row, and verify the two immediates reproduce it — if `k_out` needs three ternlogs, not two, the "three passes" claim is wrong. (b) **Whether it is ever needed**: under R6 the whole `k` plane is absent in V3. If no membrane demands SQL three-valued semantics, C3 is a translation with no consumer — which is a *stronger* verdict than ADAPT and should be recorded as such rather than left implied. |
| C4 | NULL propagation in comparison (`null_mask->SetInvalid` scalar loop) | **ELIMINATE (per R6), or inherit C3** — counted at its primary verdict in §8 (a verdict conditioned on another row's is not a split) | `execute_comparison.cpp:26-38` | zero-fallback ladder (`CLAUDE.md:1657-1665`) | DuckDB runs a **separate scalar loop** over `count` rows to populate `null_mask` before the compare even starts (`:31-37`, the `SetInvalid` at `:35`). Under the V3 zero-fallback ladder there is no second mask to populate: zero means "not consulted", monotonically, and `RESERVE, DON'T RECLAIM` (`CLAUDE.md:1663`) keeps that stable across mints. | Identical to R6's: exhibit one V3 rail where zero is a legal value distinguishable from absence. One counter-example moves C4 from ELIMINATE to C3's two-plane cost. |
| C5 | `IS [NOT] DISTINCT FROM` (NULL-safe equality) | **ADAPT — exactly one `mask_ternlog`** | `is_distinct_from.cpp:33-58` (`DistinctComparatorSelect`); `:7-31` (the value forms); dispatch `comparison_operators.cpp:952,1009` | `eq_*_to_mask` + one `mask_ternlog<IMM>` (`simd_masking_ops.rs:588`) | `NOT DISTINCT FROM` = `(eq & k_a & k_b) \| (~k_a & ~k_b)`. That is a **3-input** function of `(eq, k_a, k_b)` — unlike C3's AND, it fits a **single ternlog**. DuckDB instead fills a whole `Vector comparator_result(TINYINT, count)` (2048 bytes, `is_distinct_from.cpp:37`) and scalar-loops it (`:42-56`). | Same shape as C3(a): the immediate is **not derived here**. Derive it, self-check it against the generator's 256-table sweep, and verify against `is_distinct_from.cpp`'s own semantics table. If the single-ternlog claim fails, C5 collapses into C3. And as with C3, it is moot wherever R6 holds. |
| C6 | `IS NULL` / `IS NOT NULL` (bit-mask → byte-per-row bool vector) | **V3 BETTER** | `null_operations.cpp:13-30` (`IsNullLoop`) | `mask_not` (`simd_masking_ops.rs:1072`) — or nothing at all under R6 | `IsNullLoop` reads a **bit**-per-row validity mask and writes a **byte**-per-row bool vector: an 8× expansion whose only purpose is to be re-compressed by the next `Select`. In V3 the answer is already a mask; `IS NOT NULL` is the mask itself and `IS NULL` is one `mask_not` pass with the tail re-cleared. | Nothing can flip the 8×-expansion observation. What *can* flip the row's relevance: R6. If V3 has no validity plane, `IS NULL` is not expressible at all rather than cheap — a different statement, and the honest one for most V3 lanes. |
| C7 | `HasNull` / `HasNotNull` (per-row early-exit scan) | **V3 BETTER** | `null_operations.cpp:39-58`, `:60-79` | `mask_any` (`:1215`), `mask_all` (`:1242`) | These are literally `mask_any(validity)` and `!mask_all(validity, n)` written as per-row loops. V3 reads 64 rows per word. | `mask_any` has **no early exit** (`:1215-1221` ORs every word) while DuckDB's loop returns on the first hit (`null_operations.cpp:53`). On a mask whose first word is set, DuckDB wins. Measure at n_rows = 1M with bit 0 set; if `mask_any` is materially slower there, C7's verdict is width-dependent and §3 G8 applies. |
| C8 | Integer boundary exactness at `INT_MIN`/`INT_MAX` | **KEEP (both correct, by different routes)** | `execute_comparison.cpp:194-200` (`LessThan` = `GreaterThan(right,left)`; the swap at `:198`) | `lt_i32_to_mask` = `t.gt_bitmask(v)` (`simd_masking_ops.rs:891`), `ge` = `!lt` + tail clear (`:920-925`), `le` = `!gt` + tail clear (`:947-953`) | Both systems derive the four ordered compares from ONE primitive rather than shifting a threshold — DuckDB by operand swap, ndarray by operand swap plus complement. The blackboard states the reason (2026-09-13): *"Ordered compares derive from `gt` by complement, so they are exact at `i32::MIN`/`MAX` (threshold shifting underflows)."* | A boundary test: `ge_i32_to_mask(&[i32::MIN], i32::MIN, …)` must set the bit, and `le_i32_to_mask(&[i32::MAX], i32::MAX, …)` likewise, with no wraparound. **`[claimed, unverified]`** — the derivation was read at `:891,920,947`, the boundary behaviour was NOT executed this session (no cargo run permitted). Verify before quoting exactness. |

### 2.4 Adaptivity, hashing, joins, matching

| # | DuckDB concept | verdict | DuckDB file:line | V3 / ndarray counterpart | mechanism | FALSIFIER |
|---|---|---|---|---|---|---|
| A1 | `AdaptiveFilter` — runtime permutation of conjunction terms by measured selectivity | **NEEDS FALSIFIER — the one row where DuckDB has something V3 does not** | `adaptive_filter.cpp:17-29` (ctor), `:113-186` (`AdaptRuntimeStatistics`); the swap `:127,163`; likeliness decay `:132-134`; intervals `observe=10 / execute=20 / warmup=5` (`:17,31,180`) | **absent** — `Program.ops` is a fixed `Vec<MaskOp>` (`ir.rs:133`) executed in order, with no reordering and no measurement | An adjacent-transposition hill-climb: swap two neighbouring terms, measure 10 iterations, keep if mean runtime dropped else revert and **halve** that position's swap likeliness (floor 1, so exploration never dies). `GetInitialOrder` seeds from the optimizer's static heuristic. V3 has no analog at any layer. | **The falsifier is prior to the port, and it may kill the whole idea.** DuckDB's reordering pays because term k runs only on survivors of 1..k−1, so a selective term first *shrinks the input*. In V3 a predicate sweep costs the **full column** regardless of position (`gt_i32_to_mask` writes every word; D-GTM-0m: 22.4–22.8 µs, survivor-independent) — so **reordering saves nothing on generation**. It can only save by *avoidance*: `mask_any` says empty, skip the rest (E5), or `MaskOp::Pred { under }` skips chunks (`ir.rs:84-86`). **Measure:** on a representative predicate stream, what fraction of `Pred` ops are skippable by `under`, and does term order change that fraction? If order does not move the skip fraction, A1 is ELIMINATE and the machinery must not be ported. If it does, the port is a reordering of *skip opportunities*, not of costs — a different algorithm than DuckDB's, and its swap-likeliness decay is not obviously the right control law for it. |
| A2 | `VectorHash` / `CombineHash` (`TightLoopHash`, `CombineHashScalar`) | **NEEDS FALSIFIER** | `vector_hash.cpp:50-67` (`TightLoopHash`); `:43-47` (`CombineHashScalar`); NULL_HASH `:24` | **absent** from `simd_masking_ops` by design (a hash is a value, not a mask) | Hashing is not a mask operation and does not belong in T1. The open question is whether V3 needs hashing **at all** on the addressed path: a minted classid prefix is already a *semantic* bucket, so the hash's job (map a key to a bucket) is done by the address. | Where does V3 need a hash that the address does not already give? Candidate: joining on a **non-address column**. Enumerate real consumers; if every join key is a minted address, A2 is ELIMINATE for this substrate and hashing stays outside T1 entirely. If a non-address join key exists, A2 is a real gap but belongs to a value-kernel family, **not** to the mask vocabulary — do not let it widen `simd_masking_ops`. |
| A3 | `JoinHashTable` — probe, salt prefilter, linear-probe chains | **V3 BETTER for the addressed case; NEEDS FALSIFIER otherwise** | `join_hashtable.cpp:248-296` (`ProbeForPointersInternal`); `ht_entry.hpp:34-37` (salt/pointer split), `:49-51` (`IsOccupied`) | prefix range on the minted address; `ternary_match_u32_to_mask` with a care mask over the prefix (`simd_masking_ops.rs:1285`) | `ht_entry_t` packs **16 bits of salt + 48 bits of pointer** in one u64 and prefilters on the salt to avoid a full key compare (`:271-277`). That is a *probabilistic* prefix derived from a hash. **The V3 classid prefix is the real thing**: matching it is not a filter that may be wrong, it is a **contiguous row range** (R5, D-GTM-0m: 49–99 ns vs 22.4 µs). Also note `IsOccupied() == (value != 0)` (`ht_entry.hpp:50`) — DuckDB independently arrived at zero-is-absence, the same convention as `CLAUDE.md:1657`'s ladder. | The comparison only holds when **both sides are minted into the same address space**. A join between a minted V3 population and an external, unminted key set has no shared prefix and falls back to A2's hash question. Falsifier: exhibit the intended join workload. If either side is unminted, "V3 BETTER" is false for it and the honest verdict is NEEDS FALSIFIER. Second: `join_hashtable.cpp` is 6,986 harvested events, by far the largest TU read — this row is a **reading of two functions, not of the join**; spilling, radix partitioning, and chain building are not assessed. |
| A4 | `RowMatcher` (multi-column match by sequential in-place sel compaction) | **ADAPT — the single cleanest mapping in the matrix** | `row_matcher.cpp:19-62` (`TemplatedMatchLoop`); the in-place compaction `:56`; the 4-way validity specialization `:64-89`; `row_matcher.hpp:47` (`Match`) | `ternary_match_strided_to_mask(bytes, first_offset, stride_bytes, count, &[u8;12] pattern, &[u8;12] care, out)` (`simd_masking_ops.rs:1415-1424`) | DuckDB matches **one key column at a time**, narrowing `sel` in place each round (`sel.set_index(match_count++, idx)`, `:56`) — k columns, k passes, k compactions. The V3 12-byte register **is** the multi-column key, and the `care` mask names which rails participate, so k columns become **one pass** with no compaction. The strided form is shaped for exactly the V3 facet: `stride_bytes = 16`, `first_offset = 4`, pattern/care = `[u8;12]` — the 4+12 atom of `le-contract.md` §1. | **Scope, and it is narrow.** `ternary_match` is *equality with don't-cares only*. `RowMatcher` dispatches a per-column `ExpressionType` predicate (`row_matcher.hpp:52-56`) and NULL-semantics variants (DISTINCT FROM). So A4 covers the **equi-match** case and nothing else. Falsifier: take a real multi-column match from a V3 consumer; if any column needs an ordered or distinct-from predicate, the one-pass claim fails for it and it degrades to per-column masks + `mask_and` — still allocation-free, but k passes, not one. |
| A5 | `VectorOperations::Copy` / `DataChunk::Slice` (materialize a selection) | **ELIMINATE for composition, KEEP for egress** | `vector_copy.cpp:12-29`; `data_chunk.cpp:356-399` | none for composition; `[T1 gap: masked compaction]` (§3 G5) for egress | Inside a plan, copying a narrowed column is pure waste — the mask travels instead. At the **egress membrane** (Arrow, an FFI call, a consumer that takes only values) compaction is genuine work that must happen once, at the end, instead of at every operator. DuckDB pays it per operator; V3 should pay it once or never. | Count the egress points in the intended consumer set. If compaction is needed at more than one place per query, "once at the end" is not the shape and G5's priority rises sharply. |

---

## §3 — T1 gaps this matrix creates

Primitives `ndarray::simd` does **not** have today that a KEEP/ADAPT row above
needs. Verified absent this session by grep over `src/simd_masking_ops.rs` and
`src/simd.rs` (the facade is the only legal consumer path — POLYFILL LAW).

Each entry names the row that needs it, what is absent, and — because a gap
without one is a wish — the measurement that establishes it is worth building.
**Priority is argued, not assigned**; nothing here is scheduled.

| # | gap | needed by | status of absence | why it is a gap, and the measurement that justifies building it |
|---|---|---|---|---|
| **G1** | **`gt/ge/lt/le/eq/ne_u8_to_mask` and `_u16_to_mask`** — compare-to-mask at byte and half-word width | **E4**, R5/D-GTM-0m | **verified absent.** Every `*_to_mask` in `simd_masking_ops.rs` is i32 or u32 (lines 164, 340, 881, 920, 947, 975, 1017, 1044). No u8/u16 form at any width. | This is the gap the substrate's own probe already hit and **named as a limit**: *"The T1 compare is i32-wide, so the u8 permeability column is widened 4× for `gt_i32_to_mask` — `n_gen` and coal are UPPER bounds; a u8/u16 compare-to-mask is a T1 addition"* (`blackboard.md` 2026-09-14 (2)). The V3 12-byte register is carved into **bytes** (`le-contract.md` §3: `6×(8:8)`, `4×(8:8:8)`, `3×(8:8:8:8)`), so byte-width compare is the *native* width of the substrate and i32 is the foreign one. **Measurement:** build `gt_u8_to_mask`, re-run `hex_tenant_mq_probe.rs`, compare the 8.9 µs re-chain and 75 µs M1b generation. If they do not move, the 4× widening was not the cost and G1 drops in priority — a real possibility, since the widening cost is memory traffic and the probe is 8 KiB-mask-bound. |
| **G2** | **`gt/lt/ge/le_i64_to_mask`** — ordered compare at 64-bit width | E4 | **verified absent.** `ternary_match_u64_to_mask` (`:1338`) exists — equality with care — but no *ordered* u64/i64 compare. | `LaneRef::U64` is in the IR (`ir.rs:22`: *"edge targets, ids, the low 8 payload bytes"*) but `Pred` offers only `MatchU64` for it (`ir.rs:76`) — no `GtU64`. So a 64-bit lane is currently equality-only by construction. **Measurement:** count how many intended predicates over a U64 lane are ordered rather than equality. Report B's census flagged this family as absent pre-#306 and #306 did not close it. If the answer is zero, G2 is not a gap, it is a correctly-scoped surface. |
| **G3** | **`masked_argmin_i32` / `masked_argmax_i32`** — the *index* of the extremum under a mask | E2 (`Terminal` family) | **verified absent.** `masked_min_i32` / `masked_max_i32` return `Option<i32>` — the **value** only (`simd_masking_ops.rs:1502-1530`, both delegating to `masked_fold_i32`). No index-returning form; grep for `argmax`/`argmin` over `simd_masking_ops.rs` and `simd.rs` → 0 hits. | `MIN(x)` is answerable; `the row where x is minimal` is not, and the second is the common one for any "pick the best survivor" step. A caller today must re-scan to find which row held the value — a second full pass, and ambiguous on ties. **Measurement:** a two-pass find-value-then-find-row against a fused single-pass argmin at selectivity ∈ {0.01, 0.5}. Under 2× the gap is real but low priority; the tie-break rule (first index wins) must be pinned in the API either way, since a fold has no natural one. |
| **G4** | **`mask_shift_hex` / mask-level neighbour shift on the Morton lattice** | R5, D-GTM-0m's cost model | **⊘ REGRADED 2026-09-14 (PR2 council): absent at the time of this read; SHIPPED the same day as `mask_shift_morton` (`ndarray/src/simd_masking_ops.rs`, ndarray `255c36d`) — and its own pre-registered falsifier below FIRED.** Over the full field the word op recovers **−14 %** (14.5 vs 17.0 µs), not the modelled ~98 %, and loses to the NNUE delta arm (9.3 µs); the **−66 %** (`n` = 5.7 µs) came from restricting the op to the trie node's own word span — a mechanism this row never named. Verdict now `[H]`: the fitted model was right about the COST of `n` and wrong about the REMEDY. Original cell follows unedited. | **The single highest-value gap on this list, and the only one with a fitted cost model behind it.** D-GTM-0m fitted `step = x·ternlogq + n` with `ternlogq = 291 ns/pass` and **`n = 17.3 µs`**, max residual 2.8 % over x ∈ {0,1,2,4,8,16,32}; at x=1 the whole mask chain is **1.7 %** of the step. `n` is *"the ONE non-mask op on the path — the per-active-bit hex shift (dilated-integer add per direction)"*, and the blackboard names the fix: *"a mask-level neighbour shift on the Morton lattice (`mask_shift_hex(state, d, dst)` — within a nibble a 4×4 block shift, carries across blocks), which would fold `n` into a handful of word passes."* **Filed, not built.** **Measurement:** already banked — the NNUE delta-frontier reading reaches 8.8 µs (−48 %) *without* the primitive; a word-pass implementation should approach the ternlog rate. If a built `mask_shift_hex` does not beat 8.8 µs, the per-active-bit form was not the bottleneck and the fitted model needs re-reading. |
| **G5** | **`vpcompress`-style masked gather / compaction** — mask + lane → dense lane + count | R4, E9, A5, C2 (the pack/egress seam) | **verified absent** (grep `fn compress`/`masked_gather` → 0 hits). Report B's census independently found no gather/scatter surface. | Every ELIMINATE verdict above rests on "the mask travels instead of the data" — which is right *inside* a plan and false *at the membrane*. Egress needs the index list exactly once, and building it by hand is the per-row loop the whole design avoids. AVX-512 has `vpcompressd`; AVX2/NEON/WASM/scalar need generated bodies (BACKEND LAW), which makes this the **most expensive gap to build** on the list. **Measurement:** count egress points per query in the intended consumers (A5's falsifier). One per query ⇒ a scalar compaction is fine and G5 is not needed. More than one ⇒ build it. **Do not build it before that count exists.** |
| **G6** | **`mask_set_range(dst, lo, hi)`** — set a contiguous row range in one call | R5, R8, A3 | **verified absent** (grep `fn mask_range`/`mask_set_range` → 0 hits) | The prefix-is-a-range dividend (228–462×, D-GTM-0m) is currently spent through the probe's own inline range write, not a shipped primitive. A trie-node reveal at level L is `2^(16−4L)` contiguous rows: full words in the interior, two partial words at the ends — a handful of stores. Every consumer that reveals a prefix will otherwise re-derive the same edge arithmetic, and the partial-word ends are exactly where a tail bug lands (the NORMATIVE tail rule, `simd_masking_ops.rs:61-70`). **Measurement:** none needed to justify *correctness* consolidation; the perf case is already the probe's 49–99 ns. The thing to measure is whether a general `[lo,hi)` beats a nibble-aligned `reveal(prefix, level)` — the latter is alignment-guaranteed and may be strictly simpler. |
| **G7** | **lane-vs-lane compare-to-mask** (`gt_i32_lanes_to_mask(a, b, out)`) | R3's falsifier, E4 | **verified absent.** Every predicate takes a **scalar** threshold (`simd_masking_ops.rs:340`, `:881`, …); `Pred` likewise (`ir.rs:57-77` — every variant carries `t`/`v`/`pattern`, never a second lane). | DuckDB's comparison surface is inherently binary (`BinaryExecutor::Select<LEFT,RIGHT,OP>`, `binary_executor.hpp:241`); V3's is unary-with-constant. That is a **deliberate narrowing**, not an oversight — and it is exactly the register in which R3's ELIMINATE holds. If a consumer needs column-vs-column, the narrowing is wrong. **Measurement:** enumerate intended predicates; if all compare against a constant, record G7 as *deliberately absent* rather than a gap, and say so in the IR's docs so a future session does not "fix" it. |
| **G8** | **cheap emptiness** — a running popcount alongside a mask, or an early-out `mask_any` | E5(a), C7 | **present but unconditional.** `mask_any` (`:1215-1221`) ORs **every** word with no early exit; `mask_all` (`:1242`) takes `n_rows`; `popcount_batch_u64` (`bitwise.rs:274-277`) is `.iter().map(count_ones).sum()` — **scalar, no SIMD, no dispatch**, contrary to what its placement beside the mask family suggests. | DuckDB's short-circuit is O(1) because it carries `current_count` (`execute_conjunction.cpp:94`); V3's is O(n/64). **Both candidate fixes have a cost**: an early-out `mask_any` adds a data-dependent branch that may be slower on dense masks, and a carried count must be maintained by every writer (a real API burden, and a correctness hazard if one writer forgets). **Measurement:** time `mask_any` at n_rows ∈ {64K, 1M}, mask empty and mask-with-bit-0-set, against both alternatives. Only build if the scan is a measurable fraction of a step — at 64K rows the mask is 8 KiB and the OR is ~291 ns by the ternlogq rate, against a 17.3 µs step: **1.7 %**, which argues G8 is NOT worth building for that workload. Record the number before deciding. |

**Gaps deliberately NOT filed**, with reasons, so a later session does not file
them by reflex:

- **`array_windows`-style sliding compare** — the windowed statistics this would
  serve are integral-image shaped elsewhere in the fleet; a large-window box
  filter has an O(1) formulation and a sliding window is a ~270× *increase* in
  work at whsize 16. Not a mask gap.
- **SIMD `popcount_batch_u64`** — real (it is scalar today, `bitwise.rs:274-277`)
  but it is a *bitwise* gap, not a *masking* gap, and it belongs to whoever owns
  `bitwise.rs`. Named here only so the observation is not lost.
- **a value-kernel hash family** — see A2. If it is needed it is a different
  family; letting it into `simd_masking_ops` would violate the file's own charter
  (`simd_masking_ops.rs:1-40`: masking is the family).

---

## §4 — What DuckDB has that V3 must NOT copy

Five shapes. Each is load-bearing *in DuckDB* for a reason that does not hold
here; each has a citation on both sides.

### 4.1 Index-list materialization

`SelectionVector` is an index list, and **every** operator that narrows must
rewrite it. The costs are structural, not incidental, and they compound:

- **8192 B vs 256 B** for one 2048-row selection (§1).
- **A sort to restore row order** after OR (`execute_conjunction.cpp:139`) — a
  cost that exists *only* because the list has an order the mask never lost
  (E6).
- **A cache to compose** dictionary-on-dictionary (`SelCache`,
  `vector.cpp:236-248`) — masks compose in one pass with nothing to cache (R4).
- **A range re-manufactured as a list**: `DataChunk::Slice(offset, count)`
  builds a whole `SelectionVector` in a per-row loop to express a contiguous
  range (`data_chunk.cpp:394-397`) — while DuckDB's own `SequenceVector`
  (`vector.cpp:498-500`) already had the compressed form and
  `ToUnifiedFormat` threw it away (`vector.cpp:461-465`).

The mask-native invariant already forbids this, and `mask-risc` makes it
structural rather than a rule to remember: *"a `SelectionVector` cannot be
expressed in this vocabulary at all"* (`lib.rs:33`). **The correct reading of
that sentence is that it is STATED as a type-system property; nothing enforces it today (the repo-wide `&[u32]` row-id sweep is unrun, §9), so the failure mode is a discipline one** — if a
future op ever takes or returns `&[u32]` of row ids, the invariant has been lost
silently.

### 4.2 Per-call allocation on the hot path

`execute_conjunction.cpp:73,76` does `make_uniq<SelectionVector>(STANDARD_VECTOR_SIZE)`
— up to **two 8 KiB heap allocations per conjunction Select**, per chunk.
`execute_operator.cpp:66-69` allocates **four** for COALESCE. `is_distinct_from.cpp:37`
and `execute_comparison.cpp:109` each construct a full `Vector` of `count`
TINYINTs as a comparator scratch.

Against this, D-GTM-0k is measured with a counting global allocator, not
asserted: **mask hot path = 0 bytes/step at every density, every depth, both
relation shapes** (`blackboard.md` 2026-09-05; `ndarray/examples/hex_trie_vs_gemm_probe.rs`),
re-confirmed in D-GTM-0m as **0 B/step everywhere** across 32/32 cells. The
structural reason is the `Scratch` contract: `Program::new` computes
`scratch_slots` at assembly time (`ir.rs:143-163`) and the **caller owns every
byte** (`lib.rs:22-28`).

**DuckDB is inconsistent with itself here, and the good half is worth copying:**
`CaseExpressionState` holds its two selection vectors **in the state**
(`execute_case.cpp:12-19`), allocated once at InitializeState, not per call. That
is the same shape as caller-owned `Scratch`. Copy CASE's discipline; refuse
conjunction's.

### 4.3 NULL as a second mask, where the rail already carries absence

DuckDB maintains a parallel `ValidityMask` per vector and pays for it three
separate ways: a scalar pre-loop to populate `null_mask` before a comparison even
begins (`execute_comparison.cpp:26-38`); a merge that **allocates a new buffer**
per combine (`validity_mask.cpp:63-73`); and a per-row branchy three-valued
truth-table evaluation (`boolean_operators.cpp:44-56,80-107`).

The V3 zero-fallback ladder makes the second plane unnecessary for the addressed
path: *"zero = fall through to the broader default"*, monotonic, with
**RESERVE, DON'T RECLAIM** so a zero tier means *not consulted*, never *compacted
away* (`CLAUDE.md:1657-1665`). `mask-risc` states the consequence for its own
vocabulary (`lib.rs:44-46`).

**The honest boundary, stated because it is easy to over-read this section:**
eliminating the NULL plane is correct *where the zero-fallback holds*. It is a
claim about the V3 address space, not a claim that three-valued logic is wrong.
At a membrane that must serve SQL semantics, C3/C5 give the two-plane form, and
its cost is 2–3 ternlog passes rather than a per-row loop — still better than
DuckDB, but no longer free. **A row that needs the second plane is not a
violation; a row that needs it and does not say so is.**

### 4.4 Runtime ISA dispatch and per-backend branching in a consumer

DuckDB selects kernels through template instantiation at compile time (the
14-way switch at `execute_comparison.cpp:40-96`), which is the correct half; what
must not be copied into V3 is the *consumer-side* branch. The POLYFILL LAW
(`blackboard.md` 2026-09-13) is absolute: five compile-time realizations, **no
runtime ISA dispatch, no fallback chains, scalar is a peer backend**. The BACKEND
LAW adds that there is no shared implementation body to delegate into — repeated
source is removed by **generating backend-local bodies**
(`tools/gen_ternlog_bodies.py`), never by a generic.

`simd_masking_ops.rs` holds this line structurally: it is `#![forbid(unsafe_code)]`
(`:39`), and its own header states *"no backend semantics live here… it never
branches on an ISA, never names an intrinsic, and never carries a per-architecture
cost model"* (`:19-27`). `mask-risc` repeats it one layer up: *"This crate
contains no `cfg(target_feature)`, no ISA cost model, no fallback chain"*
(`lib.rs:36-37`).

**The failure mode this prevents is recent and measured.** The AVX2 mask-family
audit (`blackboard.md` 2026-09-14) planned a `[__m256i; 2]` rewrite of six shapes
the codegen oracle then showed were **already packed** from scalar source — the
rewrite would have re-implemented six working lowerings and broken every `.0[i]`
site for nothing. *A consumer that believes it knows what a backend does is the
bug.* The instrument (the oracle) said so before code was written, and that is
the pattern: measure the lowering, do not assume it.

### 4.5 A second row-index universe

Beneath 4.1 is the deeper thing: `SelectionVector` introduces a **second
coordinate system** for rows, and `get_index` (`selection_vector.hpp:136-138`)
is the translation between them — with the subtle `sel_vector ? … : idx`
identity-fallback that makes "no selection" and "identity selection" the same
call site. Every operator then has to know which universe it is in, which is why
`SelectFlatLoop` carries *both* a `sel` and a `bsel` and indexes through both
(`scalar_executor.hpp:591-593`).

In V3 the row index is the **address**, and it is the only one. `mask-risc`
names it: *"no per-row object, no hidden rowset, no second row-index universe"*
(`lib.rs:26-28`). This is the invariant that makes §5's prefix-is-a-range claim
possible at all: a range is only meaningful if row position is stable and global.
Any construct that reintroduces a local row numbering — a compacted chunk, a
per-morsel renumbering — costs that property, and should be paid for
deliberately, at an egress membrane (G5), never inside a plan.

---

## §5 — What V3 has that DuckDB lacks

Four. The first is the one that matters.

### 5.1 The address IS the trie — a prefix predicate is a range, not a sweep

In DuckDB a predicate on any column is a **sweep**: touch every value, compare,
pack. There is no exception, because DuckDB's row ids carry no semantics — they
are arrival order. The nearest approach is the zone map, which *skips* blocks but
still sweeps the ones it keeps.

In V3 a minted address is Morton-keyed and prefix-routable
(`CLAUDE.md` § P0: `classid | HEEL | HIP | TWIG | …`, 3 tiers × 4 nibbles,
tier-of-level = `level >> 2`), so **a trie node at nibble level L is
`2^(16−4L)` CONTIGUOUS rows** and revealing it is a range write.

Measured, D-GTM-0m (`blackboard.md` 2026-09-14 (2); probe
`ndarray/examples/hex_tenant_mq_probe.rs`, committed, `--release`; substrate
65,536 rows = 256×256 axial hex, row = Morton(q,r)):

| operation | cost |
|---|---|
| range reveal of a trie node | **49–99 ns** |
| general `ternary_match_u32_to_mask` sweep over the address column | **22.4–22.8 µs** |
| **ratio** | **228–462×** |

Three gates green, 32/32 cells: range reveal == TCAM reveal at every level and
prefix; Morton-arm spread == an independent row-major axial BFS at every step
(plasticity bytes compared too); hot-path heap 0 B/step everywhere.

**The scope leg, which is the important half.** The TCAM sweep **stays** — for
addresses that are *not* laid out (the D-GTM-0l linker case, where prefix
locality was real at ~11× enrichment but the tract codebook did **not**
compress and the verdict was `[G]` fail on physical addresses). The dividend is
not "prefixes are fast"; it is *"a **minted**, Morton-keyed tenant never pays the
sweep."* An unminted address space gets DuckDB's cost, correctly. And the
measurement's own limits are stated in its entry: one fixture, one density
(62 % permeable), one tile size, timing floor 50 ms, no `perf`, **no production
caller** — this is W0.

### 5.2 `6×(u8:u8)` rails — a content-blind register the ClassView reads many ways

The V3 12-byte payload is an **axis-grouped byte register** carved
`6×(8:8)` / `4×(8:8:8)` / `3×(8:8:8:8)` — `6·2 = 4·3 = 3·4 = 12`
(`le-contract.md` §3, operator-locked; the L1–L8 catalogue). `u8:u8` is **two
separate bytes, never widened to u16 or u24**. The ClassView selects the reading
per class; the register itself is content-blind.

DuckDB has no counterpart: its columns are **typed**, and a bitwise predicate
over a packed payload has to go through the expression engine and its physical-
type switch (`execute_comparison.cpp:40-96`). The V3 form matches on raw bits
with **no decode** — `ternary_match_strided_to_mask` takes `[u8;12]` pattern and
`[u8;12]` care directly over a 16-byte stride (`simd_masking_ops.rs:1415-1424`),
which is the 4+12 atom read as-is.

Two consequences worth naming: (a) the substrate's native compare width is the
**byte**, which is why §3 G1 is the byte-width compare gap and not a nicety; and
(b) D-GTM-0m resolved the standing question of whether hex adjacency and rail
carving are "the same six" **by construction** — once the rail index *is* the
direction, adjacency and carving are one object.

### 5.3 `ternlog` — three masks, one pass, semantics above the ISA

`mask_ternlog<IMM>(a, b, c, dst)` (`simd_masking_ops.rs:588`) computes **any**
3-input Boolean function in one pass over the mask words. It is *semantics*: how
a given immediate is realized is entirely the backend's business —
`_mm512_ternarylogic_epi64` on AVX-512, and on the other four a **generated**
Shannon ladder (`f = (!c & T0) | (c & T1)`), at **7 ops** where the vocabulary
has a native and-not (NEON `vbic`, WASM `v128.andnot`) and **8** where it is
spelled `x & !y` (AVX2, scalar) — bounds the generator *asserts* on the emitted
text (`blackboard.md` 2026-09-13; the earlier "≤ 7 for any table" was wrong for
two of four backends and was corrected by the C2 council pass).

DuckDB's conjunction is strictly 2-input (`VectorOperations::And` /
`Or`, `execute_conjunction.cpp:48,51`), so three terms are always two passes.

**Fitted cost, from D-GTM-0m:** `ternlogq = 291 ns/pass` on 8 KiB masks
(**0.285 ns/word**), inside `step = x·ternlogq + n` with max residual **2.8 %**
over x ∈ {0,1,2,4,8,16,32}. Acceptance is exhaustive where it can be: for every
IMM in 0..=255 the bit-serial reference equals the compiled realization, on both
lane types on x86 (113/113 at v3 and at v4), WASM run for real under node, NEON
verified by cross-compiled assembly selection (424 vector ops vs 41 scalar in
scaffolding).

**The discipline that makes this a V3 advantage rather than a hardware bet:** the
first generated NEON body **scalarized** (536 scalar vs 4 vector ops) with the
same truth tables and the same tests passing — rung 3 of the AArch64 acceptance
ladder caught it. "One pass" is a claim about the *emitted code*, and it is only
true because an instrument checks it.

### 5.4 No NULL bitmap — zero is fall-through

Already stated at 4.3 from the "don't copy" side; from this side it is a
capability: V3 spends **zero** bytes and **zero** passes on validity for the
addressed path, because absence is already the monotonic zero of the ladder
(`CLAUDE.md:1657-1665`). DuckDB spends a `validity_t` plane per vector, a
populate loop, an allocating combine, and a per-row three-valued evaluation.

Note the convergence, which is the interesting part rather than the win:
DuckDB's own `ht_entry_t::IsOccupied()` is `value != 0` (`ht_entry.hpp:49-51`)
— when DuckDB designs a *dense address space* it reaches for zero-is-absence
too. The V3 ladder is that instinct made monotonic and made canon.

---

## §6 — Findings about the harvest (not papered over)

The `ruff_cpp_spo` harvest at `<scratchpad>/harvest/duckdb/` ran over 22 TUs and
produced `events.tsv` / `methods.tsv` / `scopes.tsv` / `symbols.tsv` per TU
(`run.sh`, `run.log`, `args.txt` all present; `run.log` ends `HARVEST_DONE`).

**No output file is empty or zero-length.** The smallest are
`numeric_inplace_operators/methods.tsv` and `scalar_executor/methods.tsv` at one
row each — and one row is **correct**: each of those TUs defines exactly one
out-of-line function (`VectorOperations::AddInPlace`,
`ScalarExecutor::PrepareGenericResultValidity`). `methods.tsv` has **no header
row**, so a line count of 1 is one method, not an empty file. *A first reading of
this session called them header-only; that was wrong and is corrected here rather
than left in the record.*

**The real emptiness is semantic, and it is a genuine finding about the
instrument.** `methods.tsv` column 6 is a per-method verdict. Across all 373
harvested methods:

| verdict | count |
|---|---|
| **Empty** | **182 (48.8 %)** |
| Observe | 95 |
| Compute | 32 |
| Guard | 28 |
| Normalize | 17 |
| WriteRaise | 11 |
| Cascade | 7 |
| Default | 1 |

**Seven TUs harvest to 100 % `Empty`** — the harvester extracted no behavioural
facts from any method in them:

| TU | Empty / total |
|---|---|
| `execution/expression_executor/execute_comparison` | **6 / 6** |
| `common/vector_operations/is_distinct_from` | **10 / 10** |
| `common/vector_operations/null_operations` | **5 / 5** |
| `common/vector_operations/vector_copy` | **3 / 3** |
| `execution/expression_executor/execute_constant` | **2 / 2** |
| `common/vector_operations/numeric_inplace_operators` | **1 / 1** |
| `common/vector_operations/scalar_executor` | **1 / 1** |

with three more close behind: `comparison_operators` 32/39 (82 %),
`execute_case` 4/5, `boolean_operators` 5/7.

**Why this matters rather than being trivia: the blind spots are exactly the
concepts this matrix most needed.** `execute_comparison` is the TU behind row E4
and C1 — the comparison→mask translation, the centre of the arc — and it
harvested 6/6 Empty. `is_distinct_from` is row C5, 10/10 Empty. `null_operations`
is C6/C7, 5/5 Empty. **Every row in §2.3 sits on a TU the harvest could not
see.**

**Mechanism (mine, stated as a hypothesis with its evidence, not as the
harvester's design):** DuckDB's execution is template-dispatched. The `.cpp`
files are thin dispatch shims and the work lives in headers —
`execute_comparison.cpp:40-96` is a 14-arm switch whose arms are all
`BinaryExecutor::Select<T,T,OP>`, and the actual loop is
`ScalarExecutor::SelectFlatLoop` in `scalar_executor.hpp:580-610`. A harvester
walking a TU's own out-of-line definitions sees a switch that calls a template
and records no facts. The corroborating signal is in the counts: `execute_operator`
has 2 methods against 569 events, `execute_case` 5 against 567 — event-rich,
method-poor, which is the shape of a file whose bodies are elsewhere.

**Consequence for this document, stated plainly: the harvest contributed
essentially nothing to §2.3, and little to §2.2.** Those rows were written from
reading the C++ directly, including the headers the harvest does not cover. **No
row in this matrix cites a harvest TSV as evidence**, and that is not an
oversight — it is what the 48.8 % Empty rate forces. A future session that wants
the SPO harvest to cover DuckDB-shaped code must point it at the **headers**
(`scalar_executor.hpp`, `binary_executor.hpp`, `comparison_operators.hpp`) or at
instantiated template bodies, not at the dispatch TUs.

---

## §7 — State of the consumer (read, not assumed)

**⊘ as of PR2's later commits (`c095dcc`): the crate IS a workspace member, declares only `ir`, and builds/lints clean — the state below is what this read saw.** `crates/lance-graph-mask-risc` is **WIP and not a workspace member** (grep
`mask-risc` in the root `Cargo.toml` → no hit). On disk:

- `src/ir.rs` (204 lines) — the op vocabulary: `Operand`, `LaneRef`, `Planes`,
  `Pred` (10 variants), `MaskOp` (8 variants), `Terminal` (8 variants),
  `Program`, `OpHistogram`. Complete and coherent.
- `src/lib.rs` (71 lines) — declares `pub mod exec; fuse; hop; reference;
  ternlog_table;` and re-exports from all of them.
- **None of those five modules exist.** `src/` contains `ir.rs` and `lib.rs`
  only; `examples/` and `tests/` are **empty directories**.

So the crate **does not compile today**, and the four laws its `lib.rs` states
(`:20-48`) are a *declared* contract, not an enforced one. This matters for how
§2 should be read: every row whose V3 column names a `mask-risc` symbol is
describing a **vocabulary**, not a working executor. The `ndarray::simd`
primitives those rows lower onto *are* shipped and tested (#306); the layer
between them is not.

Nothing in this document depends on the missing modules — the citations are to
`ir.rs` and `lib.rs`, both present — but a Phase-2 reader should not infer an
executor exists.

---

## §8 — Verdict tally and what Phase 2 must answer first

**32 rows** across §2.1–§2.4 (R1–R8, E1–E11, C1–C8, A1–A5). Three of them —
**R6, A3, A5** — carry **split verdicts**, because their two halves genuinely
land in different columns; each is counted once in each half. So the columns sum
to 35 entries over 32 rows, and that is the honest arithmetic, not a rounding
error. *A row forced to a single verdict when its two halves differ would be a
rounding error dressed as a finding.*

| verdict | whole rows | split halves | rows |
|---|---|---|---|
| **ADAPT** | **14** | — | R5, R7, R8, E1, E2, E4, E5, E7, E10, E11, C1, C3, C5, A4 |
| **ELIMINATE** | **8** | **+2** | R1, R3, R4, E3, E8, E9, C2, C4 · *plus* R6(the NULL **role**), A5(composition) |
| **V3 BETTER** | **3** | **+1** | E6, C6, C7 · *plus* A3(addressed case) |
| **KEEP** | **2** | **+2** | R2, C8 · *plus* R6(the mask **representation**), A5(egress) |
| **NEEDS FALSIFIER** | **2** | **+1** | A1, A2 · *plus* A3(unaddressed case) |
| | **29** | **+6** | = 35 entries / 32 rows |

Read the shape rather than the totals: **ADAPT dominates (14/32)**, which is the
non-obvious result. The tempting summary — "DuckDB's machinery is an artifact of
index lists, delete it" — is contradicted by the matrix's own count: only 8 rows
eliminate cleanly, and they are concentrated in *representation* (R1, R3, R4) and
in the two places representation leaks into execution (E3, E8, E9). The
**intent** of most DuckDB execution transfers intact; it is the **carrier** that
does not. And exactly **one** DuckDB mechanism has no V3 counterpart at all
(A1, `AdaptiveFilter`) — which is why its falsifier is one of the three that
gate Phase 2.

**Phase 2's ordering falls out of the falsifiers, not from the verdict counts.**
Three measurements gate the largest number of downstream rows and should run
before any design:

1. **The word-uniformity measurement** (C1 / E5(a) falsifier). After 2–3
   conjunctive narrowings on a realistic V3 population, what fraction of mask
   words are all-zero / all-one / mixed? This single number decides whether
   `MaskOp::Pred { under }`'s survivor-skip premise (`ir.rs:84-86`) is real,
   whether C1's "V3 gets both skips" is worth stating, and whether A1's adaptive
   reordering has anything to reorder. **If survivors are uniformly scattered,
   three rows change at once.**
2. **The byte-width compare** (G1 / E4 falsifier). Build `gt_u8_to_mask`, re-run
   `hex_tenant_mq_probe.rs`, compare against the banked 8.9 µs re-chain and
   75 µs M1b generation. The substrate's native width is the byte
   (`le-contract.md` §3); i32 is the foreign one, and D-GTM-0m's numbers are
   explicitly **upper bounds** because of it.
3. **A1's prior question** — does term order change the *skip fraction*? If not,
   `AdaptiveFilter` is ELIMINATE and the only DuckDB mechanism V3 lacks turns out
   not to be needed. That is a cheap answer with a large consequence, and it is
   answered by measurement (1) plus one sweep.

**What Phase 2 must NOT do**: build G5 (masked compaction) before A5's egress
count exists — it is the most expensive gap on the list (five backends, BACKEND
LAW, generated bodies) and the whole case for it rests on a number nobody has
counted.

---

## §9 — Claims in this document that are NOT verified in code

Collected so they are not quoted as findings. Everything else in §2–§5 carries a
`file:line` read this session.

- **`[claimed, unverified]`** C8's boundary exactness at `i32::MIN`/`i32::MAX`.
  The *derivation* was read (`simd_masking_ops.rs:891,920-925,947-953`); the
  *behaviour* was not executed — no cargo command was run in either repo this
  session (disk budget).
- **`[claimed, unverified]`** The two ternlog immediates for C3's three-valued
  AND and the one for C5's DISTINCT FROM. The **decompositions** are derived in
  this document; the **immediates are not computed** and must be checked against
  `tools/gen_ternlog_bodies.py`'s 256-table self-check before use. C3's claim
  that `k_out` needs exactly two ternlogs (not three) is part of what must be
  checked.
- **`[claimed, unverified]`** That `execute_comparison.cpp`'s 6/6-Empty harvest
  result is caused by template dispatch. The correlation is documented in §6
  (event-rich / method-poor TUs) and the mechanism is plausible from the source
  shape, but the harvester's own extraction rule was not read.
- **`[claimed, unverified]`** That no consumer of `simd_masking_ops` constructs
  a `SelectionVector`-shaped `&[u32]` row-id list. §4.1's "type-system claim"
  reading is asserted from `mask-risc/src/ir.rs`'s vocabulary; a repo-wide sweep
  for row-id-list surfaces in `lgj-abi` and the planner was **not** run.
- All D-GTM-0m and D-GTM-0k figures are quoted from `ndarray/.claude/blackboard.md`
  (2026-09-14 (2) and 2026-09-05). The probes are committed
  (`examples/hex_tenant_mq_probe.rs`, `examples/hex_trie_vs_gemm_probe.rs` —
  both present on disk, verified) but were **not re-run** here. The blackboard
  states their limits; §5.1 repeats them rather than dropping them.

---

## §10 — What this document is not

It is not a plan: no D-ids, no waves, no schedule, no mint. It is not a
ratification — no council has read it (`.claude/agents/5plus3-council.md`'s
sequencing applies to whatever Phase 2 becomes: the 5 streamline first, the 3
attack the hardened draft only). It proposes **no** code, and per §0's planning
register it proposes nothing in a DataFusion-hosted surface.

Its one job is to make Phase 2's design decisions falsifiable before they are
made.
