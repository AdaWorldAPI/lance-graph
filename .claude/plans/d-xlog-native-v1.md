# D-XLOG-NATIVE — XLOG plan vocabulary over the ndarray::simd polyfill (the algebra under lance-graph-onnx)

**Status:** PROPOSED. Amendment C supersedes the direct-ndarray rule. The crosswalk (§2) is docs only; it decides whether a `lance-graph-xlog` crate, wired like `lance-graph-onnx`, is worth creating. Amendment A (2026-10-31) records what moved since: the binding now names mask-risc's op vocabulary, the probe binds through the production bind store, and XLOG results have a stated Epi5 reading.

**Date:** 2026-10-30. **Parent work:** D-XLOG-DR64-0 (#1857), D-LITHO-1 (#1861, #1864; D-LITHO is now folded into D-DT64-6 Amendment A). **Precedent:** [D-ONNX-NATIVE](d-onnx-native-v1.md).

**Rule.** ⊘ *Superseded 2026-10-31 by the D-SOC-FENCE ownership rule (Amendment C).* The original text read: "XLOG is a vocabulary over the same algebra `lance-graph-onnx` sits on: the `ndarray::simd` polyfill. The path would be `XLOG plan node → existing ndarray::simd function` and nothing else … mask-risc and relalg are cross-checks." That made `ndarray::simd` the call path for folds, which bypasses the one evaluator. **Current rule:** XLOG plan nodes are described in Quack/relalg, evaluated by `lance-graph-mask-risc`, and computed by `ndarray::simd` only under mask-risc. Elementwise (non-fold) aliases shared with `lance-graph-onnx` stay `ndarray::simd` symbols, as D-ONNX-NATIVE sanctions.

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
| `Count` over a many-to-many join (the D-LITHO-1 multiplicity fixture) | `masked_group_count_u32` twice, then `checked_binary_i64(I64Op::Mul)` | NATIVE | `GroupReduce { Lane, Count }` twice | deg_R(y) × deg_S(y). The binding exists (Amendment A); no executor runs it, and the probe's `BagNeedsMultiplicity` refusal stands. The multiply word is the ndarray #407 re-export, not checked from here |
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
- [x] D-XLOG-NATIVE-4: create the crate once §4 holds (in PR, 2026-11-01; shape per Amendment D).
- [x] D-XLOG-NATIVE-5: bind the bag count through the production bind store (`lance-graph-dt64` and its `dr` module), and state the Epi5 reading of XLOG results (Amendment A).

## 6. Gates

- **N1, alias identity.** As `aliases_are_the_native_symbols` in `lance-graph-onnx`: an alias is the native operation, never a wrapper.
- **N2, oracle agreement.** Every ALIAS and INLINE row agrees with scalar enumeration and the RIR walk on the probe's fixtures, with the D-XLOG-DR64-0 differential.
- **N3, typed refusal.** Every REFUSE and NATIVE row is a typed refusal in the adapter, never a silent fallback.
- **N4, no dependency leak.** `cargo tree -p lance-graph-xlog` shows `ndarray` as the only normal dependency, and no `xlog-*` or arrow 60.
- **N5, one algebra.** Every operation both facades name is the same native symbol in `lance-graph-onnx` and `lance-graph-xlog`.

## Amendment A (2026-10-31): the binding's vocabulary, the production bind store, Epi5

**A.1 What a binding names.** D-DT64-6 A.5 addenda 8 and 9 (operator) settle the catalogue-id question: there are no ids. The polyfill is one lens classid, its words are that lens's fields labelled through its `ClassView`, and a word is a field position (`AlgebraRef { lens, word }`). For folds the word is mask-risc's op vocabulary (`Pred`, `MaskOp`, `Terminal`); `ndarray::simd` is named only where mask-risc has no word, such as `checked_binary_i64`. So in §2 the **mask-risc column is what a binding names**, and the `ndarray::simd` column is what an executor would call. The ALIAS / INLINE categories and gates N1–N5 are about the crate's execution symbols and are unchanged.

**A.2 The probe binds through production.** `crates/xlog-relcomp-probe/tests/dto_binding.rs` now routes the answer positions with `BindStore::route`, seals frames with `lance_graph_dt64::dr::FrameStore`, labels fields through `Resolved::field_view`, and admits the count by `MethodRef` (`Population::Bag`). Its steps name two polyfill words: mask-risc's keyed count twice, then the checked i64 multiply. Execution stays in the lab: `bind_dr64` and `relcomp` are not in `dt64::dr` (addendum 12), so the execution test carries the admitted population into the lab register, which refuses it (`BagNeedsMultiplicity`). `tests/litho.rs` keeps the lab frame store because `relcomp::Chain` only accepts that one.

**A.3 Epi5 (CE64 bits 59..63).** If an XLOG result is ever written as a `CausalEdge64`:

| result | `EpistemicState5` | raw5 |
|---|---|---|
| base fact (`r(1, 2)`) | `Direct × Open` | 0 |
| composed edge (`r ; s`, X to Z through a retained Y) | `IndirectKnown × Open` | 1 |

- A join certifies nothing (`Open`).
- If an answer keeps no witness of Y, its topology is `IndirectUnknown`. That case is stated, not tested.
- A bag count is a value field of the binding, never a certification and never frequency/confidence.
- `raw5 == 0` is `Direct × Open`, so an unstamped composed edge silently claims a direct relation.
- The reading is opaque until the XLOG lens has an `Epi5Declarations` entry. That is an OGAR declaration, including which rail it sits on; the probe uses a lab placeholder.

`tests/epi5_reading.rs` pins these rules.

**A.4 Still open.** Method identity: `MethodSignature` vs `MethodRef` vs Tract's `MethodKey` (EXO-50). Whether a frame records the store's catalogue extent (addendum 7, D8 owner). The XLOG and polyfill lenses' production classids and the XLOG Epi5 declaration (OGAR). A native home for the many-to-many join (D-XLOG-NATIVE-3).

## Amendment B (2026-10-31): execution on the `ndarray::simd` polyfill

> ⊘ **Superseded by Amendment C** on ownership: the table below records what `native.rs` does today, not the architecture. Its fold calls are a D-SOC-FENCE `fold-evaluator` violation, fixed by D-DYNBAG-0.

Operator: refactor XLOG onto the `ndarray::simd` polyfill (CubeCL as an intermediary only in the worst case). It was not needed: every function below is on the `ndarray::simd` facade, including `checked_binary_i64` (ndarray #407, merged and read in the checkout).

`crates/xlog-relcomp-probe/src/native.rs` is now the probe's execution path for the admitted two-atom chain. Each step is one function:

| step | `ndarray::simd` |
|---|---|
| a bound end (`X == x`, `Z == z`) | `eq_u32_to_mask` |
| each side projected onto Y | `mask_scatter_or_u32` (the demanded result, never an intermediate) |
| support = forward ∧ backward | `mask_and` |
| rows per Y on each side | `masked_group_count_u32` |
| derivations per Y | `checked_binary_i64(I64Op::Mul)`; an overflow is refused |

- The Y domain is DR's `lance_graph_dt64::dr::LocalMap`, at 64 or 256 wide. The native path is therefore not limited to one 64-slot tile; a 200-Y fixture is answered that the register refuses.
- `admit_count` admits XLOG's compiled `count` rule (`Count` keyed by the answer column, over a join `admit` accepts) and refuses any other aggregate (`UnsupportedAggregate`).
- The DR64 register path (`propagate`) stays as the differential. Every Gate B answer asserts that the native forward, backward and answer masks equal the register's.
- The overflow refusal is not exercised by a test: per-side counts are bounded by the row count, so no fixture reaches `i64`.

Effect on §2:
- The two-atom many-to-many chain projected onto Y (support and bag count) now has a native home. Its rows move from NATIVE to INLINE: they are tested against both oracles.
- Still NATIVE: a many-to-many join whose answer keeps both ends (X, Z), and chains of more than two atoms.
- §4 condition 2 holds for the projected chain only.

## Amendment C (2026-10-31): the D-SOC-FENCE ownership rule

- **Folds** (group count, scatter, masked sums, popcount terminals) are described in Quack (`Query`, `GroupBy`, `Agg::GroupReduce`) or relalg and evaluated by mask-risc `execute_into`. `ndarray::simd` is never called for them from XLOG or its probe.
- **Elementwise aliases** (`mask_and/or/not`, `*_to_mask`) shared with `lance-graph-onnx` remain `ndarray::simd` symbols, the D-ONNX-NATIVE sanction.
- **No mask-risc word yet:** `checked_binary_i64` is the one direct call, TRACKED in D-SOC-FENCE until mask-risc gains an i64 combine (D-DYNBAG §17.5 MISSING-3).
- In §2, the `ndarray::simd` column is what mask-risc calls, not what XLOG calls. The NATIVE / INLINE categories now mean "has a mask-risc program", not "has an ndarray function".
- The work is D-DYNBAG-0 (`d-dynbag-holographic-bags-v1.md` §17).

## Amendment D (2026-11-01): the crate as created

**DECISION (operator, 2026-11-01: "lance-graph-xlog has priority, and dynbag").** `crates/lance-graph-xlog` is created after D-DYNBAG-0, on the shape Amendment C implies, which supersedes §4's "ndarray as its one dependency":

- **Dependencies:** `lance-graph-quack`, `lance-graph-mask-risc`, `lance-graph-dt64`, `lance-graph-contract`, and `ndarray` only for the elementwise alias `and` and the TRACKED `checked_binary_i64`. No `xlog-*` dependency and no arrow (N4, checked with `cargo tree -e normal`).
- **Contents:** the computation the probe ran (`Intake`, `support`, `bag_count`, `bag_count_admitted`, `Word`, `Support`, the refusals, `ids`, `nonzero`). Bounds arrive as plain `Bounds { left, right }`, so no XLOG type crosses.
- **Left in the probe:** XLOG admission (`admit`, `admit_count`) and the `ChainQuery` → `Bounds` reading. The probe calls the crate and keeps both oracles.
- **Gates:**
  - N1 and N5: `and` is `ndarray::simd::mask_and` and `lance_graph_onnx::and`, by `fn_addr_eq`.
  - N2: an in-crate scalar oracle under four bound combinations, plus the probe's scalar oracle and RIR walk.
  - N3: typed refusals (`NativeRefusal`, `BoundRefusal`).
- **§4 condition 2 holds for the projected two-atom chain only.** A join keeping both ends (`p(X, Z)`) and chains of more than two atoms remain NATIVE and are refused at admission.

## Amendment E (2026-11-01): assimilate XLOG into our algebra; polyfill back to the caller

**DECISION (operator, 2026-11-01):** "perspectively we want to assimilate it into our own algebra IR substrate and create polyfill back to the caller."

- **Inward (assimilation).** XLOG's RIR node kinds become descriptions in our algebra (Quack / relalg / R2IL, evaluated by mask-risc, addressed by DT64/DR), never a backend inside XLOG's runtime.
  - XLOG's names are an alias view over the same words, as SQL and Quack names already are over the polyfill lens (D-DT64-7 §1b).
  - An XLOG-owned hardcoded lens uses `0xFFFF::F1__` (O7), and only after a deliberate promotion. There is no automatic XLOG promotion (D-DT64-7 §1d).
- **Outward (polyfill back).** The surface kept stable is XLOG's caller API: what `pyxlog`, `xlog-cli` and `xlog-integration` call to run a program and read its results. It is answered by our substrate. XLOG's internal call sites and its `CudaBuffer` plumbing are not ported.
- **Census (2026-11-01, `xlog` fork at `bff6da6`; Haiku count, three counts re-run by hand):**
  - All execution goes through one dispatcher, `xlog-runtime` `executor/node_dispatch.rs` `execute_node_uncached`, over 13 RIR kinds and about 11 `execute_*` methods.
  - Operator call sites: 26, plus 40 for `execute_node`.
  - `CudaBuffer` appears on 278 lines in `xlog-runtime` and 1,924 in the workspace. This is the reason the runtime is not the seam.
- **Coverage today** (`lance-graph-xlog` and the probe):
  - Admitted: `Scan`, `Filter` (equality with a constant), `Project`, `Join` (inner, two atoms), `ChainJoin`, `GroupBy` (`count`), `Fixpoint` (bounded, linear; #1896).
  - Refused: `Union`, `Distinct` (as its own node), `Diff` (negation), `MultiWayJoin`, `TensorMaskedJoin`, `Unit`.
  - Each refused kind needs one admission rule and one native description.
- **Target (DECISION, operator, 2026-11-01): `crates/lance-graph-xlog`.** This settles where the polyfill lives. Dependency direction:
  - **`lance-graph-xlog` owns the semantics.** It holds the algebra descriptions and the caller-facing API in its own types (`Bounds`, `u32` tuples, DR maps, results). There is no `xlog-*` dependency and no arrow (N4), and no `RirNode` mirror (§0).
  - **The XLOG fork depends on `lance-graph-xlog`, never the reverse.** Its executor arms translate an RIR node into a `lance-graph-xlog` call: that is the polyfill back to `pyxlog` / `xlog-cli` / `xlog-integration`. `lance-graph-xlog` carries no arrow, so the arrow 60 / 58 split never meets.
  - **`xlog-relcomp-probe` stays the oracle harness:** XLOG compile, admission, scalar enumeration, the RIR walk, and the DR64 register differential.
- **OPEN:**
  - The exact caller API. Next step is a census of what `pyxlog` / `xlog-cli` / `xlog-integration` call on the executor, and how each call maps onto a `lance-graph-xlog` call.
  - Which translation the XLOG fork carries: a translator per RIR kind, or a thin adapter crate inside the fork.
