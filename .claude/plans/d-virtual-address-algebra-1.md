# Virtual address algebra: a transformation as an address law (lab plan)

**Status:** PLAN. Hypotheses, expected answers and kill conditions fixed 2026-11-13, before any candidate ran. Lab only.
**D-ids:** D-VIRTUAL-ADDRESS-ALGEBRA-1 (`STATUS_BOARD.md`).
**Crate:** `crates/virtual-address-lab` (new workspace member). No production crate changes.
**Follows:** D-QUACK-BOUNDARY-LAB (#2045), D-WX-DR256-TERNLOG (#2055), D-WX-BIT-PLANES (#2057), D-QUACK-DERIVED-PLANE-PROBE (#2058).

## 1. Question

Can a transformation of a source-owned SoA tile be held as an immutable,
composable address recipe, so that recipes are simplified before anything is
read, a chain that reduces to the identity executes no transformation, and a
fold reads the original lanes only where its terminal demands values?

## 2. What exists (read before designing)

| primitive | where | what it gives |
|---|---|---|
| `address`, `coordinates`, `rotate_yz` | `lance-graph-dt64/src/dr.rs` | a reversible `Morton8x8` cell address and one axis swap that "moves no stored bits" and is its own inverse |
| `LaneSource`, `LaneBinding`, `CoordSpace`, `borrow_all` | `lance-graph-dt64/src/lane.rs` | per-FIELD lanes with frame, row-order and mapping checks |
| `LaneRef::Strided` (`StridedRef`) | `lance-graph-mask-risc/src/ir.rs` | a `(bytes, first_offset, stride, records)` view; read only by `EqU32Strided`, `NeU32Strided`, the two facet matches and `MaskedStridedGroupSum` |
| `MaskOp::Ternlog`, `Terminal::{Count, MaskedSumI32, Keep}` | same | Boolean algebra and folds over planes in ONE row order |
| `MaskOp::Gather` | same | an fk semijoin over a foreign plane; not a value gather |
| `execute_sel` (`SelFold::{CountGtI32, SumI32}`) | `lance-graph-mask-risc/src/sel.rs` | two folds over a borrowed position bag of an `i32` lane |
| `lower`, `lower_sel` | `lance-graph-quack/src/lib.rs` | the lowerings of the two forms above |

Not found: any "indexed reorder" operand or lowering. A program names lanes
by index and has no notion of a row permutation; the only position-list
operand is `execute_sel`'s, for two folds. No primitive reads a lane
backwards (a stride is a `usize`).

## 3. Lab objects

- `Recipe`: an element of the symmetry group of the square (eight elements:
  identity, three rotations, two mirrors, transpose, anti-transpose), stored
  as three bits. View cell `(r, c)` of a tile with lanes `M[lane][pos]` reads
  `M[recipe(r, c)]`. Convention, fixed here: `Rotate90` is
  `view(r, c) = M[c][N-1-r]`, the counter-clockwise quarter turn of the lane
  matrix drawn with one lane per row. The entity view (row `r` is an object,
  property `c` is lane `c`) is `Transpose`; `Rotate90` is the entity view
  with the rows reversed.
- `Grid`: the source owner, lanes of `i32` in one buffer, behind `LaneSource`.
- `Trace`: counters for address resolutions, value reads, gathers, physical
  transforms, copied values.

## 4. Hypotheses, expected answers, kill conditions

| id | hypothesis | expected | FAILS / KILL |
|---|---|---|---|
| H1 | the eight recipes are closed under composition; the bit composition equals pointwise composition | an 8 x 8 table; `Rotate90∘Rotate270`, `Rotate180∘Rotate180`, `MirrorX∘MirrorX`, `MirrorY∘MirrorY` are `Identity` | any cell of the table differs from pointwise composition on N = 1..8 |
| H2 | a resolved recipe reads the same values as a scalar reference that physically builds the view | equal for all eight recipes, the 3 x 3 fixture giving `3 6 9 / 2 5 8 / 1 4 7` for `Rotate90` | any cell differs |
| H3 | a chain that normalises to `Identity` executes no physical transform, no gather, copies no value and allocates nothing | counters `transforms = gathers = copied = 0`, 0 allocations, answer equal to the oracle | **primary falsifier:** any of them non-zero. Disable: without normalisation the same chain must show transforms > 0 |
| H4 | a row object of 32 properties is representable without reading any; one demanded property is one read | 0 reads at construction, exactly 1 read and 1 resolution per demanded property | more reads than demands |
| H5 | `TERNLOG(P A, P B, P C) = P(TERNLOG(A, B, C))` for one shared bijection `P`, all 256 immediates; `COUNT(P M) = COUNT(M)` | equal | any immediate differs |
| H6 | the rewrite is refused where it is not lawful | permuting only `A` changes the result on the fixture; an untransformed selection mask changes the result; a position list with a duplicate or a gap is not accepted as a bijection; an order-sensitive demand keeps the recipe | a negative control that does not differ, or a non-bijection accepted |
| H7 | admission is unchanged by the recipe: a lane of another `CoordSpace` and a frame moved after binding are refused | `LaneRefusal` | accepted |
| H8 | a fold over a view column whose recipe only reverses the row order runs the SAME program as the unreversed view when the terminal is permutation-invariant | identical `Program`, identical answer | a reversed lane, a reversed mask or a gather in the route |
| H9 | family A (`A(x) ∧ B(y) ∧ C(z)`) is counted from three 256-bit masks | product of three popcounts, equal to an enumeration of 16,777,216 addresses | unequal |
| H10 | family B (`x + y + z ≡ 0 mod 256`, optionally under axis masks) is counted by solving for `z`: a reflected and rotated axis mask, an AND and a popcount per residue | 65,536 unrestricted; equal to the enumeration under masks; no plane over `X x Y` and no cube allocated | unequal, or an allocation of 8 KiB or more |
| H11 | family C (pseudo-random membership) has no such form | the product of axis projections over-counts, and the exact count needs one membership test per candidate | (control) if the projection product equals the exact count the control is void |
| H12 | `address = (z << 16) \| Morton8x8(x, y)` through `dr::address` is reversible | round trip for every `(x, y)` and sampled `z` | a collision or a failed round trip |
| H13 | an immutable DT64 method can NAME a recipe independently of the physical layout | a method whose guard carries the recipe's digest is admitted; another recipe's description is refused; renumbered lane ids give the same answer | (limit expected) DT64 compares the digest and interprets nothing |

Recorded, not required to pass:

- **Where materialisation is first forced.** Expected: (a) a view column of a
  non-swapping recipe crosses lanes; the lane socket issues one lane per
  field, so through a binding it is one read per lane, and as a single
  strided operand it needs the source's slab, which no binding issues;
  (b) an order-sensitive output of a row-reversing recipe; (c) any
  non-bijective or arbitrary reorder, which is a position list and a gather.
- **Routes (section 4 of the brief).** Contiguous lane, strided cross-lane
  column, indexed reorder, sparse position list, dense mask, materialised
  view. Each timed warm with its cold preparation reported separately;
  population sizes stated with every number.

## 5. Measurements to report

Source-owned bytes; resident intermediate bytes; copied values; address
resolutions; value reads; gathers; physical transforms; warm allocations;
cold preparation; warm latency per route; oracle parity; disable runs.

## 6. Scope

No ClassId, no F word, no production opcode, no Scope issuer, no change to
DT64, Quack or mask-RISC. The field-to-operand rule of #2045 is kept:
operand index is the rank among bound fields.
