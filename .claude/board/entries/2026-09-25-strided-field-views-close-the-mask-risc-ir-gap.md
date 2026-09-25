# 2026-09-25 — strided field views close the mask-risc IR gap

**Status:** TEST-PINNED (`crates/lance-graph-mask-risc/tests/strided.rs`, 9 tests, disable-verified) · OPEN (see below)
**D-ids:** none new — closes the gap named in `lance-graph-mask-risc/src/lib.rs` ("a strided `Operand` … nothing here can name a `(base, stride, group)` source") and the hot-path item in `2026-09-24-three-clocks-never-meet-implicitly.md`.

## What landed

- `LaneRef::Strided(StridedRef { bytes, first_offset, stride, records })` — a field VIEW over unchanged
  record bytes: record `i`'s field at `bytes[first_offset + i*stride]`. A `NodeRow` column is read where it
  sits; the classid view (`+0`) and a facet view (`+4`) borrow the same buffer in one program.
- Readers: `Pred::EqU32Strided`, `Pred::NeU32Strided`, `Pred::MatchFacetStrided` (12-byte ternary match),
  `Terminal::MaskedStridedGroupSum` (`groups × group_bytes` LE unsigned fields, `Value::StridedSum`, `None`
  instead of a wrapped sum; admitted under a partial extent — a sum merges by addition).
- Realised through the already-shipped `ndarray::simd::{eq_u32_strided_to_mask, ternary_match_strided_to_mask,
  masked_strided_group_sum}`; the scalar oracle reads bytes directly. Shared validator refuses
  `StridedOutOfBounds { lane, need, have }` and `StridedGroupWidth { group_bytes }`.

## Evidence

- Executor == oracle == hand computation over 0 / 70 / 1573 rows (3 tiles + ragged tail): eq/ne partition,
  cross-path equality with an extracted `U32` column, facet match, gating, two views over one buffer + an
  I32 lane, group sum as 6×u16 and 3×u32 (whole and partial extent), in-place mutation moves the count by
  exactly one, refusals identical on both paths, ternlog (even `0xE8` and odd `0x17`) over three strided masks.
- Disable runs: tile offset ignoring `r0` → 8/9 red; `Ne` without its complement → 2 red; bounds width
  dropped → the refusal test red.

## OPEN

- Strided predicates reach ternlog only through tile-local scratch; `FusedTernlog` takes resident planes
  only, so predicate → ternlog → Count is not one fused pass. Whether a strided-input fused shape beats the
  scratch path is unmeasured (the collapse probe says control cost can dominate — measure before building).
- `StridedSum(None)` (i64 overflow) is not driven by a test: it needs ~8.4M records of max-valued u32 fields.
