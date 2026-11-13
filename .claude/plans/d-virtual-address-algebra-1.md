# Virtual address algebra: one address law shared by order and geometry (lab plan)

**Status:** PLAN (clean recovery). Pass conditions fixed 2026-11-13, before any candidate of THIS branch ran. Lab only.
**D-ids:** D-VIRTUAL-ADDRESS-ALGEBRA-1 (`STATUS_BOARD.md`).
**Crate:** `crates/virtual-address-lab` (new workspace member). No production crate changes.
**Follows:** D-DR-SELECT-K (#2033, #2035, #2039, #2040), D-QUACK-BOUNDARY-LAB (#2045), D-WX-DR256-TERNLOG (#2055), D-QUACK-DERIVED-PLANE-PROBE (#2058).

> **Correction of the first attempt.** Branch `ccr-8062bcc7-l7lkz9` (kept as
> evidence, not merged) carried a plan that said no indexed reorder exists.
> That was a name search, and it is wrong: the path is
> `OrderIndex` / `SealedOrder` → `SelectionSource` → `LaneBinding::selected`
> → `fold_bound_selection` → `execute_sel` → `ndarray::simd::sum_i32_at`.
> That attempt also built its own `Permutation`, `MaskView` and a lane
> `Vec`-holding table beside that path. None of the three is ported. Its
> recipe algebra, materialising oracle and 256^3 module are, after review.

## 1. Two transformations, never exchanged

| | A. ordered row selection | B. geometric address recipe |
|---|---|---|
| answers | WHICH source-local rows are visited, in what order | HOW a requested view cell `(row, property)` resolves to `(field, position)` |
| carrier | a version-bound position list issued by the source | three bits; nothing is stored |
| existing code | see the seam map | none; this lab adds the recipe only |
| can change the field axis | no | yes (`swap`) |

`Rotate90(r, p) = (p, N-1-r)`: view cell `(r, p)` reads lane `p` at position
`N-1-r`. For lanes `[1 2 3] [4 5 6] [7 8 9]` the view is
`3 6 9 / 2 5 8 / 1 4 7`. A reversal of the row index alone is not this map.

## 2. Seam map (read at `main` `630ad6a34`)

| edge | state | where |
|---|---|---|
| Lance produces an order or an exact selection | OPEN | `.claude/probes/2026-11-12-dr-select-k-1/REPORT.md` "Scope limits"; `lance-graph-dt64-lance` persists the `MapTable` only |
| order as a version-bound map, lanes not moved | LAB | `OrderIndex` (`crates/dr256-ternlog-lab/src/lib.rs`), `SealedOrder` (`src/order_store.rs`); borrowed positions through `SealedOrder::rows` |
| frame, revisions, global to local | SHIPPED | `FrameStore`, `Frame`, `MapTable` in `crates/lance-graph-dt64/src/dr.rs` |
| `NodeGuid` to row | OPEN | `dr-select-k-v1.md` §2 |
| positions and lane under one check | SHIPPED | `SelectionSource`, `SelAddr`, `LaneBinding::selected` in `crates/lance-graph-dt64/src/lane.rs` |
| a production `SelectionSource` | OPEN | the only implementors are fixtures under `crates/lance-graph-report/{tests,examples}` |
| the boundary before the fold | SHIPPED | `fold_bound_selection` in `crates/lance-graph-report/src/bound_sel.rs` |
| Quack | SHIPPED, two `i32` fold shapes | `lower_sel` in `crates/lance-graph-quack/src/lib.rs` |
| mask-RISC | SHIPPED | `execute_sel`, `SelFold` in `crates/lance-graph-mask-risc/src/sel.rs` |
| ndarray | SHIPPED, a plain loop | `sum_i32_at`, `count_gt_i32_at` in `ndarray/src/simd_masking_ops.rs` |
| `Scope` as a grant | BLOCKED | `LaneBinding::selected` doc: a consistency check |
| an order witness outside the lab | BLOCKED | `dr-select-k-v1.md` §15, item 2 |

`fold_bound_selection` builds its `SelFold` itself; it does not call
`lower_sel`. Both edges exist and no call path joins them.

## 3. What the lab adds, and what it does not

Adds: `Recipe` (three bits, composition, inverse, normalisation), a
materialising oracle, the 256^3 module, and a fixture source that implements
the EXISTING `LaneSource` and `SelectionSource`. The fixture's ordered
selection is a `SealedOrder` built by `OrderIndex::by_i32` and lent through
`SealedOrder::rows`; nothing is copied out of it.

Does not add: a position or permutation owner, a mask view, a map table, a
frame store, an executor, an operand form, a ClassId, an opcode.

The smallest integration: a recipe chain is normalised to one recipe BEFORE
`LaneBinding::selected` is called, and only chooses the `field` argument of
that call. If a view property is not one field, the lab reports the missing
operand and runs nothing.

## 4. Pass conditions (fixed before running)

| id | condition | FAILS IF |
|---|---|---|
| P0 | the eight recipes compose as their addresses do | a cell of the 8 x 8 table differs from pointwise composition for N = 1..8; an inverse is missing; `Rotate90 then Rotate270`, `Rotate180` twice, either mirror twice is not `Identity` |
| P1 | a chain that normalises to `Identity` reaches the fold with the same `field` and `sel` as no chain, allocates nothing and moves nothing | another field, another selection, an allocation, or a different answer. The materialising oracle must show the same chain moving data |
| P2 | a row object holds a recipe and a row number; a demanded property is one resolution and one value read | a value read at construction; more reads than demands; another lane's value |
| P3 | a source-issued selection (an order from `OrderIndex` / `SealedOrder`, a filter, a bag) reaches `execute_sel` through `LaneBinding::selected` and `fold_bound_selection`, and equals a scalar oracle | the positions are copied on the way (pointer inequality), or the answer differs |
| P4 | a row-order change cannot stand in for a field-axis rotation | some permutation of the rows of the lane matrix equals its `Rotate90` view on a fixture with distinct cells; or a position list over one field yields a `Rotate90` view row |
| P5 | the coupled relation `x + y + z ≡ 0 (mod 256)` is counted exactly, unrestricted (65,536) and under axis masks, with no allocation | unequal to the enumeration of 16,777,216 addresses, or any allocation |

Negative controls, each a test that must be able to fail:

1. `Rotate90` differs from row reversal alone.
2. Reordering one of two aligned lanes changes a two-lane answer.
3. A selection with a duplicate is a bag: its sum differs from the set's.
4. A selection in another coordinate space is refused (`SelectionSpace`).
5. A frame moved after binding is refused.
6. RECORDED, not required to refuse: a lane rewritten under an unmoved frame
   is read under an unchanged recipe digest. The digest names a recipe, not
   the producer of bytes.
7. RECORDED: `Scope` equality is equality of a plain value; nothing in the
   path observes a grant.

Kill condition: if P3 needs a new owner of positions, or P1 needs a new
executor or operand, stop and report the missing seam.

## 5. Ledger (measured per route)

Allocated address descriptors; copied value bytes; constructed position-list
bytes; constructed mask bytes; value reads; gathers (positions folded);
physical transpositions; warm allocations; admission latency; fold latency.
Cold terms are reported in µs and are not gates.

## 6. Scope

No Lance path is claimed: the source is a fixture. `lower_sel` admits two
fold shapes and is not called a virtual-property executor.

## 7. Architecture correction, 2026-11-13: the 512-byte unit (implementation STOPPED)

The lab committed so far (`82adffdc6`, `d60d89d22`) models 32 separately
allocated `i32` lanes. That is superseded: the unit is one source-owned
512-byte record. Nothing below was run; it is what the code says.

Read in full: `ndarray/src/simd_stride.rs`; `ndarray/src/simd_masking_ops.rs`
lines 185-274, 995-1125, 5040-5340; `lance-graph-contract/src/canonical_node.rs`
lines 730-1454 and 1820-1951; `lance-graph-mask-risc/src/ir.rs` lines 1-125 and
170-329; `exec.rs` lines 505-554 and 660-752; `sel.rs`;
`lance-graph-report/src/bound_sel.rs`. Not read: PR #283 and #382 bodies, the
per-backend ternlog realization, `facet.rs`, DT64 `lib.rs` (searched for
"presence" only: no per-instance presence type found, which is not proof of
absence).

### 7.1 Physical geometry, as coded

| claim | state | where |
|---|---|---|
| A record is 512 bytes, contiguous, `repr(C, align(64))` | VERIFIED-IN-CODE | `NodeRow`, const asserts, `canonical_node.rs:837-848` |
| `&[NodeRow]` and `&[u8]` are the same bytes both ways | VERIFIED-IN-CODE | `NodeRowPacket::as_le_bytes`, `node_rows_from_le_bytes` (needs a multiple of 512 and 64-byte alignment) |
| A Lance `FixedSizeBinary(512)` buffer satisfies that | STATED in a doc comment, not traced | `canonical_node.rs:1914-1923` |
| The record is 32 facets of 16 bytes | **NOT the coded layout** | `GUIDS_PER_NODE = 32` is a capacity ("up to 32 GUID-sized entries") |
| `addr = base + 512 * row + 16 * slot` | holds for `key` (0) and `edges` (16) only | below |

The coded carve is `key(16) | edges(16) | value(480)`, and the value slab is 21
tenants of mixed width (`VALUE_TENANTS`, contiguous, compile-asserted):
8, 8, 32, 32, 6, 16, 4, 4, 2, 8, 12, 12, 12, 16, 16, 32, 16, 16, 16, 16, 16
bytes at row offsets 32 to 332; 332 to 512 is reserved. Eight tenants are 16
bytes wide, at row offsets 118, 188, 204, 252, 268, 284, 300 and 316. None of
those is a multiple of 16. The address law the code supports is therefore

```
addr(row, tenant) = base + 512 * row + VALUE_TENANTS[tenant].row_offset
```

with the offset read from the table, never computed as `16 * slot`. The strided
kernels load with `from_le_bytes` over byte slices, so an unaligned 16-byte
field is legal.

Presence, as coded, is per CLASS: `ValueSchema` is a `FieldMask` over tenants,
chosen through `ClassView::value_schema`. An all-zero tenant reads as
unasserted (zero-fallback). A per-instance presence plane was not found.

### 7.2 Execution seams for a 16-byte field

| step | state |
|---|---|
| A field view over unchanged record bytes: `LaneRef::Strided(StridedRef { bytes, first_offset, stride, records })` | SHIPPED; this is the address law above, with `stride = 512` |
| 16-byte care-masked match to a mask: `Pred::MatchFacet16Strided`, `ternary_match_strided16_to_mask` | SHIPPED; field loads are scalar by design, the compare is `U64x8` ternlog |
| Sum over a strided register: `Terminal::MaskedStridedGroupSum` | SHIPPED; unsigned groups of 1 to 4 bytes, scalar, takes a MASK |
| A strided view over a subset of rows | NOT EXPRESSIBLE: `records` must equal `n_rows`, and the stride is regular |
| A position-list fold over a strided field: `execute_sel` | **MISSING**: `SelFold` is `SumI32` and `CountGtI32` over an `i32` lane |
| A source returning `LaneRef::Strided` through `LaneBinding::selected` | type-admissible (`Lane<'s> = LaneRef<'s>`); the fold then refuses the lane kind |
| A general 512-byte, 32-facet fold in `simd_stride.rs` | ABSENT: the module holds `lut_u8` and `lut9` |
| "1 to 4 cycles" for the register-local operation | UNVERIFIED: no measurement read; both strided kernels document themselves as memory-bound at this stride |

So a source-issued selection reaches a 16-byte facet today only as MEMBERSHIP
(a mask over all rows of the record store), which drops order and
multiplicity. The exact missing operand seam is a strided lane kind in
`execute_sel`. It is recorded here and not built.

### 7.3 Open decision before any rebuild

The brief's unit (32 uniform 16-byte facets, `16 * slot`) and the coded
`NodeRow` (mixed tenants, table offsets, ten 16-byte fields including key and
edges) are different layouts. The fixture is not rebuilt until it is decided
which one the experiment measures.
