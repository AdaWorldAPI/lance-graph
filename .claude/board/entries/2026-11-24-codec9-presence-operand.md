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

## Reconciliation with D-DRISA, D-LANESOCK and `borrow_all_at` (#2160)

The register contract is the socket; `LaneRef` is only mask-risc's operand
view. Read at `lance-graph-dt64/src/lane.rs` on `29f5cff`:

- **Codec9 is already co-addressed by the socket.** One `borrow_all*` call,
  `borrow_all_at` included, yields `PresentLane = (field, S::Lane, &[u64])`
  (`lane.rs:893`): the lane and its presence plane, both from ONE `LaneAddr`
  (`:904-919`), under the frame `check_bound` and `borrow_all_at`'s position
  witness admit. A Codec9 field is one field whose lane is the magnitude and
  whose presence plane is the polarity. It must not be modelled as two
  fields: two `LaneAddr`s could resolve under different mappings, and then the
  magnitude and the sign stop describing one value.
- **The reading is chosen by the class, not by new grammar.** `Arg::Present`
  already means "present". `ClassView::epi_lanes` already declares the width.
  Under Bits9, present is `P ∨ M ≠ 0`, absent (`IS NULL`, hydration demand) is
  its complement, and a value count is `popcnt(S ∧ present)`. The refusal in
  `prepare` sits exactly where that reading would be selected.

### Remaining gap, classified

| Kind | Gap | Evidence |
|---|---|---|
| Mathematics | none | `abakus::pm::{touched, add_pm, neg_pm, sub_pm}`, `U8x64::cmpeq_mask`, mask-risc `MaskOp::Ternlog` |
| Grammar | none for present / count / hydration (the width selects the reading) | `Arg::Present`; `EpiLaneSpec.width` |
| Grammar, OPEN | where an adaptive width lives (below) | `LaneWidth` is a per-class compile-time constant (`lane_codec.rs:71`) |
| Wiring | mask-risc has no byte operand and no byte-zero test | `LaneRef` = `I32`/`U32`/`U64`/`Strided` (`mask-risc/src/ir.rs:16-36`); strided predicates are u32 only |
| Wiring | no source lends a Codec9 field | `type Lane<'a>` implementers lend `LaneRef`, `&[i32]`, `&[u32]` or cells (21 sites); none a byte lane |
| Wiring | the in-row layout has no row-order polarity plane | see below |

**Orientation.** `EpiLaneSpec` puts up to 64 Codec9 lanes inside one 512-byte
row: `values..values+64` is one `U8x64`, the presence bytes are one `u64`. A
row is therefore one Abakus `Pm` word, and arithmetic across a row's lanes
needs no gather. Reading one field across rows is different:

- its magnitude is a byte at `values + f`, stride 512, which a strided byte view
  could lend;
- its polarity is bit `f` of each row's presence bytes, a strided bit. The
  presence socket lends a contiguous LSB-first row-order plane, so a source
  could only lend it by gathering, which is forbidden.

The column-major realization (a `&[u8]` magnitude lane plus a `&[u64]` plane
per field) fits `PresentLane` directly. The method decides which orientation is
cheap: a fold of one field over rows, or arithmetic across a row's lanes. This
is an open wiring question, not a reason for a second socket.

### ndarray #438 (adaptive exact width)

The seam is this socket. The register contract (`PresentLane` under a sealed
frame) and the method (Abakus `add_pm` semantics, `touched`) stay the same;
only the slice width changes: PM9 is a u8 magnitude, PM17 or PM25 wider. The
carry mask (`PmSum::carry`) is the promotion trigger. A promotion re-lays the
lane, so it re-seals the frame, and `FrameStore::check` already refuses stale
readers. A frame-qualified width needs no per-value tag.

OPEN, and it is a grammar decision: `LaneWidth` is fixed per class at compile
time, while an adaptive width is a property of a frame or tile. Either the
width moves into what the frame seals, or `LaneWidth` grows variants that
ClassView declares per revision. No second type system either way.
