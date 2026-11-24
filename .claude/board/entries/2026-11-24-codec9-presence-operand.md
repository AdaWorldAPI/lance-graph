# 2026-11-24 — A presence operand over Codec9 lanes counted negatives as NULL

**Status:** fixed for declared layouts by refusal; the correct reading is a
binding gap, not missing arithmetic. OPEN for undeclared layouts.

## What Codec9 is, physically

`ndarray::hpc::abakus::pm` and `hpc::codec9`: one `U8x64` magnitude and one
`u64` polarity word per 64 lanes. 512 logical codewords, no 9-bit SIMD type,
no second validity plane. `LaneWidth::Bits9` in `lane_codec` names this
logical layout; it does not ask for a 9-bit lane.

| P | M | meaning |
|---|---|---|
| 0 | 0 | untouched (−0): no established value; hydrate it or do not count it |
| 1 | 0 | established zero (+0) |
| 0 | >0 | negative |
| 1 | >0 | positive |

Setting P and M both to 0 deletes the value and keeps the coordinate. Clearing
P alone is not deletion: it turns `+M` into `−M`.

The validity law is already `abakus::pm::touched = P ∨ (M ≠ 0)`.

## Finding (MEASURED)

`Arg::Present` (EXO-296) reads absent from the plane alone, and its doc gives
`IS NULL` as `IntNot` of it. Through the public binding API (`FieldPresence`,
`prepare_native_present`, `run_native`), four rows `+7, −7, +0, −0` with
P = `1, 0, 1, 0` gave `parent IS NULL` = **2**. The contract says 1: only `−0`
is absent. The existing fixtures could not see this: they are i32/u32 lanes and
force the value to 0 wherever P = 0.

## Change

`lance-graph-r2il::binding::prepare` refuses `Arg::Present` with
`ExecRefusal::PresenceIsSign { effect, field }` when `ClassView::epi_lanes`
declares a Codec9 layout. Bits8 twin counts from the plane. Disable run: guard
off, the refusal test fails.

## The gap is binding/wiring (VERIFIED-IN-CODE)

1. **Both operands can already be lent under one frame.**
   `LaneBinding::borrow_law_bound` (`lance-graph-dt64/src/lane.rs`) returns the
   lane and its presence plane after one `check_bound`, keyed by the same
   `LaneAddr` (same frame, row order and scope). `LaneSource::Lane` is the
   source's own type, so a `&[u8]` magnitude lane fits the socket.
2. **The reader cannot take a byte lane.** mask-risc `LaneRef` is
   `I32`/`U32`/`U64`/`Strided`, and no predicate tests a byte lane against
   zero. That is the missing piece.
3. **The arithmetic exists.** `U8x64::cmpeq_mask` gives `Z = (M = 0)` as a
   `u64`; mask-risc already has `MaskOp::Ternlog`; Abakus has `touched`,
   `add_pm`, `neg_pm`, `sub_pm`.

## The readings a method selects

With S the selection and Z = `M = 0`, fused at the demanded terminal; nothing
is persisted:

| demand | expression |
|---|---|
| count established values | `popcnt(S ∧ (P ∨ ¬Z))` |
| hydration demand | `S ∧ ¬P ∧ Z` |
| raw coordinates | `S` alone |
| arithmetic | `add_pm` / `sub_pm` directly; untouched is the identity, so no validity prefilter |

Relational `PairPlane` presence is a different plane and stays independent of
numerical polarity.

ndarray #438: PM9, PM17 and PM25 are logical representations, not SIMD widths.
The method decides how the physical operands are read.

## OPEN

- An undeclared layout (`epi_lanes` = `None`) is not refused. It reads the plane
  as 8-bit, so the misread remains reachable there; a test pins it. Refusing
  undeclared layouts would refuse every current presence operand; that is a
  decision, not a fix.
- The correction: a `u8` lane form plus a zero test in mask-risc, bound through
  the existing `PresenceSource` pairing. The refusal stays until that contract
  exists.
