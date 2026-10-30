# D-XLOG-NATIVE — XLOG plan vocabulary over the ndarray::simd polyfill (the algebra under lance-graph-onnx)

**Status:** PROPOSED / DOCS ONLY. This adds no crate, code, ABI or test. It is the crosswalk that decides whether a `lance-graph-xlog` crate, wired like `lance-graph-onnx`, is worth creating.

**Date:** 2026-10-30. **Parent work:** D-XLOG-DR64-0 (#1857), D-LITHO-1 (#1861, #1864; D-LITHO is now folded into D-DT64-6 Amendment A). **Precedent:** [D-ONNX-NATIVE](d-onnx-native-v1.md).

**Rule.** XLOG is a vocabulary over the same algebra `lance-graph-onnx` sits on: the `ndarray::simd` polyfill. The path would be `XLOG plan node → existing ndarray::simd function` and nothing else, exactly as `ONNX operator → existing native function`. Where both vocabularies name the same operation (`And`, `Or`, `Not`, `Greater`, `Equal`, masked sums), both facades alias the same native symbol. There is no plan type, interpreter, fixpoint engine, mask-risc program or graph database. The lowering from XLOG's compiled plan stays in `crates/xlog-relcomp-probe`, outside the workspace, and becomes the crate's caller and oracle harness. mask-risc and relalg are cross-checks, as they are for `lance-graph-onnx` (dev-dependencies only).

**Scope boundaries:**
- The crate depends on `ndarray` only, on the coordinate `lance-graph-onnx` uses. It takes no `xlog-*` dependency: `xlog-core` pins arrow 60, the lance stack arrow 58 (D-XLOG-DR64-1). Only lane slices, constants and masks cross, never XLOG or arrow values.
- No `RirNode` mirror inside lance-graph. Mirroring it would be the second plan representation the original task ruled out. The crate exposes functions.
- Coordinates and dimensions are DR's; meaning and method identity are DT64's (D-DRDT, D-DT64-6). The crate does not route and does not bind.
- ndarray owns SIMD and the kernels and does not depend on this crate.

## 1. Categories

The same four as D-ONNX-NATIVE §1, with the same cost contract: an ALIAS is the native symbol itself (pinned by `fn_addr_eq`), an INLINE adds only the named check.

| category | meaning |
|---|---|
| **ALIAS** | an `ndarray::simd` function has the XLOG node's contract and is re-exported under the XLOG name |
| **INLINE** | a small adapter does only the semantic work the native function lacks (a column-to-lane choice, a range or universe check), then calls it |
| **NATIVE** | the mathematics is not available as one native function yet; a composition or a decision is missing |
| **REFUSE** | the XLOG semantics differ from what we have, or would need an execution system |

**Data model.** An XLOG relation is columnar: one `u32` (or `i32`) slice per column, one row per tuple, and a packed row mask in the native convention (bit `i % 64` of word `i / 64`, tail bits zero) that `lance-graph-onnx` uses for `tensor(bool)`. A column's value domain (for example the Y ids) is mapped by DR, never derived here.

## 2. Crosswalk

Status of every row: **name-checked, not tested.** Each native target is a function `lance-graph-mask-risc/src/exec.rs` imports from `ndarray::simd` (read on 2026-10-30); its exact signature and edge cases (empty input, tails, out-of-range keys) are checked when the row gets its test (§5), as D-ONNX-NATIVE checked each alias. No ndarray checkout was available to read the bodies. The mask-risc column is the cross-check, not the call path.

### Selections and set operations

| XLOG | `ndarray::simd` | category | mask-risc cross-check | note |
|---|---|---|---|---|
| `Scan { rel }` | the column slices | ALIAS | — | naming the resident columns |
| `Project` keeping a subset of columns (no `Computed`, no dedup) | none: a `WideFieldMask` selection over the relation lens's `ClassView` fields | ALIAS | — | the ERB / Redmine field-view pattern (`ogar-consumer-preflight.md`, operator 2026-10-10: "Classview can be used for any compute masking"; `cross-glove-business-parity-v1.md` §C.9.3–§C.9.4). The classid picks the view, the mask picks the fields; no column is copied and no native function runs. Already the shape of `xlog-relcomp-probe/tests/dto_binding.rs` |
| `Filter`, `Column(c) == U32(v)` | `eq_u32_to_mask` | INLINE | `Pred::EqU32` | column index → slice |
| `Filter`, `Column(c) != U32(v)` | `ne_u32_to_mask` | INLINE | `Pred::NeU32` | |
| `Filter`, `Lt/Le/Gt/Ge/Eq/Ne` on an `I32` column | `{lt,le,gt,ge,eq,ne}_i32_to_mask` | INLINE | `Pred::*I32` | the same symbols `lance-graph-onnx` aliases for `Less` … `Equal` |
| a filter under an earlier selection | the `*_to_mask_under` variants | INLINE | `MaskOp::Pred { under }` | no separate And pass |
| `Filter`, `Lt/Le/Gt/Ge` on a `U32` column | none | NATIVE | none | the polyfill orders only `i32`; `u32` has `eq`/`ne`/`ternary_match` |
| `Expr::And` / `Or` / `Not` over filters | `mask_and` / `mask_or` / `mask_not` | ALIAS | `MaskOp::And/Or/Not` | shared with ONNX `And`, `Or`, `Not` |
| `Union` (same table) | `mask_or` | ALIAS | `MaskOp::Or` | |
| `Diff` (same table) | `mask_andnot` | ALIAS | `MaskOp::AndNot` | |
| non-emptiness / all-rows checks | `mask_any` / `mask_all` | ALIAS | `Terminal::Any/All` | |
| `Distinct` over all columns | none needed | ALIAS | — | a row mask is already a set |
| `Distinct` over a column subset | `mask_scatter_or_u32` | INLINE, result only | `Terminal::ScatterOrU32` | legal only as the demanded result (its survival condition), never an intermediate |
| `Unit` | not needed | REFUSE | — | only seeds negation-only rules, which are refused below |

### Joins

| XLOG | `ndarray::simd` | category | mask-risc cross-check | note |
|---|---|---|---|---|
| `Join { Semi }` on a key that names a row of the other table (many-to-one) | `mask_gather_u32` (`_under`) | INLINE | `MaskOp::Gather` | the fk semijoin; out-of-range keys do not match |
| `Join { Anti }`, same shape | `mask_gather_u32`, then `mask_andnot` | INLINE | `Gather` + `AndNot` | |
| `Join { Inner }`, many-to-one, filtering on the other side's column | `eq_u32_via_to_mask` | INLINE | `Pred::EqU32Via` | the factored join filter |
| `Join { Inner }` / `ChainJoin`, many-to-many (Gate B's `r(X,Y), s(Y,Z)`) | none | NATIVE | none | the answer needs both sides projected onto the Y domain; `mask_scatter_or_u32` may only produce a final result, and `mask_gather_u32` needs a key naming a row of the other side. Today it runs only on `dr-math-binding-probe::relcomp::compose`, a lab crate |
| `Join { LeftOuter }` | none | REFUSE | — | needs nulls |
| `MultiWayJoin`, `TensorMaskedJoin` | via their `fallback` | INLINE via fallback | — | admitted only when the fallback is, as `ChainJoin` is in the probe |

### Aggregates (`GroupBy`)

| XLOG `AggOp` | `ndarray::simd` | category | mask-risc cross-check | note |
|---|---|---|---|---|
| `Count`, keyed by a column of one table | `masked_group_count_u32` | INLINE | `GroupReduce { Lane, Count }` | keys past the group universe are dropped natively; the adapter refuses them instead |
| `Count` keyed through a many-to-one key | `masked_group_count_u32_via` | INLINE | `GroupReduce { Via, Count }` | |
| `Count` with no key | `popcount_batch_u64` | ALIAS | `Terminal::Count` | |
| `Count` over a many-to-many join (the D-LITHO-1 multiplicity fixture) | none | NATIVE | none | deg_R(y) × deg_S(y): both factors are `masked_group_count_u32` results, but their per-key product is no native function. The probe's `BagNeedsMultiplicity` refusal stands |
| `Sum` on an `I32` column | `masked_group_sum_i32` / `masked_sum_i32` | INLINE | `GroupSumI32` / `MaskedSumI32` | same row bound as the mask-risc terminal; beyond it, refuse. `masked_sum_i32` is the symbol ONNX's `ReduceSum(Where(..))` composition uses |
| `Min` / `Max` on an `I32` column | `masked_group_min_i32` / `masked_group_max_i32` | INLINE | `GroupReduce { MinI32/MaxI32 }` | an empty group must be reported as absent, not as the seed value |
| `LogSumExp` | none | REFUSE | — | a float, probabilistic aggregate |

### Expressions, constants, recursion, probability

| XLOG | category | note |
|---|---|---|
| `ConstValue::U32`, `I32` | ALIAS | as slice values |
| `ConstValue::U64`, `I64` | NATIVE | `ternary_match_u64_to_mask` covers equality only; no 64-bit ordered compare |
| `ConstValue::Symbol` | NATIVE | needs a label-codebook id first |
| `ConstValue::F32`, `F64`, `Bool` | REFUSE | not column types here |
| `Expr` arithmetic in a `Computed` projection | REFUSE | `blend_i32` is a select, not arithmetic |
| `Fixpoint` (recursion) | NATIVE | semi-naive deltas are D-XLOG-DR64-2 |
| stratified negation | REFUSE | `mask_not` is complement within one table, not stratified negation over a derived program |
| probabilistic programs | REFUSE | Gate G of D-XLOG-DR64-0: weights are not counts |

## 3. What the crosswalk says

- Column projection is not an operation: it is a `ClassView × WideFieldMask` selection, the field-view pattern the classid switches over otherwise agnostic addresses.
- Most of XLOG's single-table vocabulary is a name over `ndarray::simd` functions that already exist, and several are the same symbols `lance-graph-onnx` already aliases (`mask_and/or/not`, the `*_i32_to_mask` comparisons, `masked_sum_i32`). Two facades, one algebra.
- The two things D-XLOG-DR64-0 and D-LITHO-1 rely on most have no native function: the many-to-many join, and the bag count over it (a per-key product of two counts).
- Ordered `u32` comparison is missing from the polyfill. The probe's fixtures use `u32` columns throughout.

## 4. Crate decision

**Not yet.** The crate is created only when both hold:

1. Several ALIAS or INLINE rows in §2 have a test against the probe's oracles (scalar enumeration and the RIR walk) and a native-signature check, as D-ONNX-NATIVE §2 did.
2. The many-to-many join has a native home, decided with the ndarray and mask-risc owners. It is never added by this crate.

**Shape when created.** `crates/lance-graph-xlog`, a workspace member like `lance-graph-onnx`: `ndarray` as its one dependency, on the same coordinate; mask-risc, relalg and r2il only as dev-dependency cross-checks; no `xlog-*` dependency and no plan type; no native crate depends on it. Shared operations are aliases of the same native symbol in both facades, pinned by `fn_addr_eq`. The probe calls the crate and keeps the oracles.

## 5. Work items

- [x] Crosswalk onto `ndarray::simd`, name-checked (§2).
- [ ] D-XLOG-NATIVE-1: in the probe, test the selection rows (`eq_u32_to_mask`, `ne_u32_to_mask`, the `i32` comparisons, `mask_and/or/andnot`) against both oracles, with each native signature checked.
- [ ] D-XLOG-NATIVE-2: in the probe, test `masked_group_count_u32` keyed by one table's column against the RIR walk, including the out-of-universe refusal.
- [ ] D-XLOG-NATIVE-3: ask the ndarray and mask-risc owners for a native home for the many-to-many join; the probe's Gate B is the fixture.
- [ ] D-XLOG-NATIVE-4: create the crate once §4 holds.

## 6. Gates

- **N1, alias identity.** As `aliases_are_the_native_symbols` in `lance-graph-onnx`: an alias is the native operation, never a wrapper.
- **N2, oracle agreement.** Every ALIAS and INLINE row agrees with scalar enumeration and the RIR walk on the probe's fixtures, with the D-XLOG-DR64-0 differential.
- **N3, typed refusal.** Every REFUSE and NATIVE row is a typed refusal in the adapter, never a silent fallback.
- **N4, no dependency leak.** `cargo tree -p lance-graph-xlog` shows `ndarray` as the only normal dependency, and no `xlog-*` or arrow 60.
- **N5, one algebra.** Every operation both facades name is the same native symbol in `lance-graph-onnx` and `lance-graph-xlog`.
