# 2026-11-24 — A presence operand over 9-bit lanes counted negatives as NULL

**Status:** fixed for declared layouts; OPEN for undeclared ones.

## Finding (MEASURED)

`lane_codec` and `ndarray::hpc::codec9` define absent as presence 0 **and**
value 0. At 9 bits the presence bit is also the sign, so `P = 0` with a nonzero
magnitude is a negative value.

`Arg::Present` (EXO-296) reads absent from the plane alone, and its doc gives
`IS NULL` as `IntNot` of it. Through the public binding API (`FieldPresence`,
`prepare_native_present`, `run_native`), four rows `+7, -7, +0, -0` bound as a
field's presence plane P = `1, 0, 1, 0` gave `parent IS NULL` = **2**. The
contract says 1: only `-0` is absent.

The existing fixtures could not see this. They are i32/u32 lanes and force the
value to 0 wherever P = 0 (`binding_presence.rs`, the `value` closure).

## Change

`lance-graph-r2il::binding::prepare` refuses `Arg::Present` with
`ExecRefusal::PresenceIsSign { effect, field }` when the class's
`ClassView::epi_lanes` declares `LaneWidth::Bits9`. It refuses rather than
corrects: the absent rule needs `P ∨ M ≠ 0`, and
`ndarray::hpc::abakus::pm::touched` already states it as
`pos | !mag.cmpeq_mask(0)`. But this dialect binds no 8-bit magnitude lane
(`LaneRef` is `I32`/`U32`/`U64`/`Strided`), so the magnitude mask is not
expressible here.

The tests have a Bits8 twin that counts from the plane. Disable run: guard off,
the refusal test fails.

`borrow_law_bound` has the same P-alone reading in its doc. It is unreachable
for Codec9 today: no `LaneSource` in `crates/**/*.rs` lends a lane narrower than
32 bits. The change is doc only.

OPEN:
- An undeclared layout (`epi_lanes` = `None`) is not refused. It reads the plane
  as 8-bit, so the misread remains reachable there. A test pins it.
  Refusing undeclared layouts would refuse every current presence operand;
  that is a decision, not a fix.
- A correction path would need a magnitude-nonzero operand bound from a 9-bit
  lane and fused as `S ∧ (P ∨ M ≠ 0)` (ternlog `0xE0`).
