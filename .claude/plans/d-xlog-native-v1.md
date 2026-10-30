# D-XLOG-NATIVE — XLOG plan vocabulary over the existing relational algebra

**Status:** PROPOSED / DOCS ONLY. This adds no crate, code, ABI or test. It is the crosswalk that decides whether a `lance-graph-xlog` crate, wired like `lance-graph-onnx`, is worth creating.

**Date:** 2026-10-30. **Parent work:** D-XLOG-DR64-0 (#1857), D-LITHO-1 (#1861, #1864; D-LITHO is now folded into D-DT64-6 Amendment A). **Precedent:** [D-ONNX-NATIVE](d-onnx-native-v1.md).

**Rule.** XLOG is a vocabulary over our existing algebra, not another Datalog runtime. The path would be `XLOG plan node → existing mask-risc / relalg operation` and nothing else. There is no plan type, interpreter, fixpoint engine or graph database. The lowering from XLOG's compiled plan stays in `crates/xlog-relcomp-probe`, outside the workspace, and becomes the crate's caller and oracle harness.

**Scope boundaries:**
- The crate takes no `xlog-*` dependency. `xlog-core` pins arrow 60, and the lance stack pins arrow 58 (D-XLOG-DR64-1). Only names, lane indexes, constants and masks cross, never XLOG or arrow values.
- No `RirNode` mirror inside lance-graph. Mirroring it would be the second plan representation the original task ruled out. The crate exposes functions, as `lance-graph-onnx` does.
- Coordinates and dimensions are DR's; meaning and method identity are DT64's (D-DRDT, D-DT64-6). The crate does not route and does not bind.

## 1. Categories

The same four as D-ONNX-NATIVE §1.

| category | meaning |
|---|---|
| **ALIAS** | an existing operation has the XLOG node's contract and is named for it |
| **INLINE** | a small adapter does only the semantic work the existing operation lacks, then calls it |
| **NATIVE** | the mathematics is not available as a lawful fold yet; a composition or a decision is missing |
| **REFUSE** | the XLOG semantics differ from what we have, or would need an execution system |

**Data model.** An XLOG relation is a mask-risc table: one row per tuple, one `U32` lane per column, and a mask over the rows. A selection is a row mask. A column's value domain (for example the Y ids) is a second table whose rows are those ids. The domain is mapped by DR. It is never derived here.

## 2. Crosswalk

Status of every row: **source-read, not tested.** Each native target was re-read in `crates/lance-graph-mask-risc/src/ir.rs` and `crates/lance-graph-relalg/src/ir.rs` on 2026-10-30. No row has a test yet; §5 gates that.

### Selections and set operations

| XLOG | target | category | note |
|---|---|---|---|
| `Scan { rel }` | the relation's table and lanes | ALIAS | the table is resident; scanning is naming it |
| `Filter`, `Column(c) == U32(v)` | `Pred::EqU32 { lane: c, v }`; relalg `Atom::U32 { eq: true }` | INLINE | the column index becomes the lane index |
| `Filter`, `Column(c) != U32(v)` | `Pred::NeU32`; relalg `Atom::U32 { eq: false }` | INLINE | as above |
| `Filter`, `Lt/Le/Gt/Ge` on an `I32` column | `Pred::{Lt,Le,Gt,Ge}I32`; relalg `Cmp` | INLINE | signed, exact |
| `Filter`, `Lt/Le/Gt/Ge` on a `U32` column | none | NATIVE | mask-risc and relalg order only `I32` lanes; `U32` has only `Eq`/`Ne`/`MatchU32` |
| `Expr::And` / `Or` / `Not` over filters | `MaskOp::And` / `Or` / `Not` (tail cleared) | ALIAS | word-wise, exact |
| `Union` (same table) | `MaskOp::Or` | ALIAS | set union of row masks |
| `Diff` (same table) | `MaskOp::AndNot` | ALIAS | set difference |
| `Distinct` over all columns | none needed | ALIAS | a row mask is already a set |
| `Distinct` over a column subset | `Terminal::ScatterOrU32` | INLINE, terminal only | legal only as the demanded result (its survival condition); never an intermediate |
| `Unit` | not needed | REFUSE | only seeds negation-only rules, which are refused below |

### Joins

| XLOG | target | category | note |
|---|---|---|---|
| `Join { Semi }` on a key that names a row of the other table (many-to-one) | `MaskOp::Gather` / `Pred::EqU32Via` | INLINE | the fk semijoin; out-of-range keys do not match |
| `Join { Anti }`, same shape | `Gather`, then `AndNot` | INLINE | |
| `Join { Inner }`, many-to-one, projecting the other side's column | `Pred::EqU32Via` / `GroupKey::Via` | INLINE | the factored join filter |
| `Join { Inner }` / `ChainJoin`, many-to-many (Gate B's `r(X,Y), s(Y,Z)`) | none in production | NATIVE | the forward/backward answer needs both sides projected onto the Y domain. `ScatterOrU32` may not feed a follow-on program, and `Gather` needs a key that names a row of the other side. Today this runs only on `dr-math-binding-probe::relcomp::compose`, a lab crate |
| `Join { LeftOuter }` | none | REFUSE | needs nulls |
| `MultiWayJoin`, `TensorMaskedJoin` | their `fallback` | INLINE via fallback | admitted only when the fallback is admitted, as `ChainJoin` is in the probe |
| join-body acyclicity | relalg `join_body::is_alpha_acyclic` | ALIAS | the router test; it decides nothing about execution here |

### Aggregates (`GroupBy`)

| XLOG `AggOp` | target | category | note |
|---|---|---|---|
| `Count`, keyed by a column of ONE table | `Terminal::GroupReduce { GroupKey::Lane, GroupFold::Count }` | INLINE | the key must be `< K`, the group universe; keys past it are dropped, so the adapter refuses them instead |
| `Count` over a many-to-many join (the D-LITHO-1 multiplicity fixture) | none | NATIVE | the count is deg_R(y) × deg_S(y). Both factors are `GroupReduce Count` results, but their per-key product is outside any terminal. The probe's `BagNeedsMultiplicity` refusal stands |
| `Sum` on an `I32` column | `Terminal::GroupSumI32` | INLINE | same row bound as `MaskedSumI32`; beyond it, refuse |
| `Min` / `Max` on an `I32` column | `GroupReduce { GroupFold::MinI32 / MaxI32 }` | INLINE | an empty group keeps the seed (`i64::MAX` / `i64::MIN`); the adapter must report it as absent, not as a value |
| `LogSumExp` | none | REFUSE | a float, probabilistic aggregate |

### Expressions, constants, recursion, probability

| XLOG | category | note |
|---|---|---|
| `ConstValue::U32`, `I32` | ALIAS | as lane values |
| `ConstValue::U64`, `I64` | NATIVE | no 64-bit ordered predicate; `MatchU64` covers equality only |
| `ConstValue::Symbol` | NATIVE | needs a label-codebook id first (relalg keeps no strings) |
| `ConstValue::F32`, `F64`, `Bool` | REFUSE | not lane types here |
| `Expr` arithmetic (`Add`, `Mul`, `Pow`, `Cast`, …) in a `Computed` projection | REFUSE | no projection arithmetic in a fold; `BlendI32` is a select, not arithmetic |
| `Fixpoint` (recursion) | NATIVE | semi-naive deltas are D-XLOG-DR64-2 |
| stratified negation | REFUSE | XLOG's negation is stratified over the derived program; `MaskOp::Not` is complement within one table |
| probabilistic programs | REFUSE | Gate G of D-XLOG-DR64-0: weights are not counts |

## 3. What the crosswalk says

- About half of XLOG's single-table vocabulary maps onto mask-risc at once: equality filters, `I32` comparisons, set operations, per-key counts and sums on one table, and many-to-one semijoins.
- The two things D-XLOG-DR64-0 and D-LITHO-1 rely on most have no production home:
  - the many-to-many join, which runs today on `relcomp::compose` in a lab crate;
  - the bag count over a join, a per-key product of two `GroupReduce` results.
- Ordered `U32` comparison is missing in both mask-risc and relalg. The probe's fixtures use `u32` columns throughout.

## 4. Crate decision

**Not yet.** The crate is created only when all three hold:

1. Several ALIAS or INLINE rows in §2 have a test against the probe's oracles (scalar enumeration and the RIR walk).
2. The many-to-many join has a production home in mask-risc or relalg, decided by their owners, not added by this crate.
3. relalg's workspace membership is settled. It has its own `[workspace]`; `lance-graph-onnx` uses it only as a dev-dependency.

**Shape when created.** `crates/lance-graph-xlog`, a workspace member that depends only on `lance-graph-mask-risc` (and relalg once it is a member). It has no `xlog-*` dependency and no plan type, and no native crate depends on it. The probe calls it and keeps the oracles.

## 5. Work items

- [x] Crosswalk, source-read (§2).
- [ ] D-XLOG-NATIVE-1: in the probe, test the selection rows (`EqU32`, `NeU32`, `I32` comparisons, `And`/`Or`/`AndNot`) on mask-risc against both oracles.
- [ ] D-XLOG-NATIVE-2: in the probe, test `GroupReduce Count` keyed by one table's column against the RIR walk, including the out-of-universe refusal.
- [ ] D-XLOG-NATIVE-3: ask the mask-risc and relalg owners for a home for the many-to-many join; the probe's Gate B is the fixture.
- [ ] D-XLOG-NATIVE-4: create the crate once §4 holds.

## 6. Gates

- **N1, alias identity.** As `aliases_are_the_native_symbols` in `lance-graph-onnx`: an alias is the native operation, never a wrapper.
- **N2, oracle agreement.** Every ALIAS and INLINE row agrees with scalar enumeration and the RIR walk on the probe's fixtures, with the D-XLOG-DR64-0 differential.
- **N3, typed refusal.** Every REFUSE and NATIVE row is a typed refusal in the adapter, never a silent fallback.
- **N4, no dependency leak.** `cargo tree -p lance-graph-xlog` contains no `xlog-*` and no arrow 60.
