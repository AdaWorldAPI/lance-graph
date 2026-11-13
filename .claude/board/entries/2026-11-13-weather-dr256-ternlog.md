# 2026-11-13 — D-WX-DR256-TERNLOG: weather × DR256 slabs × one ternlog × an integer fold (lab probe)

**Kind:** lab probe in `crates/dr256-ternlog-lab`. No production crate changes.
**Grade:** HYPOTHESIS until the results section below is filled; the
pass conditions in this section were written before any candidate ran.

## Hypothesis

A weather or billion-row style query `WHERE region AND predicate AND valid`
with `COUNT / SUM / MIN / MAX / MEAN` is expressible as source-owned DR256
slabs (65,536 positions each), three aligned Boolean planes, one
`Ternlog(AND3 = 0x80)` and an existing mask-RISC integer terminal. No GEMM,
no BF16, no cube allocated, no population-sized intermediate.

## Six things kept apart

| concern | what it is | carrier |
|---|---|---|
| Cartesian addressing | row ordinal split as slab : u8 : u8 | `Table::space`, slab index |
| three-input Boolean logic | one truth table over three aligned planes | `MaskOp::Ternlog` |
| three-axis relation | a subset of X × Y × Z | one plane over the product, or nothing |
| duplicated observations | a bag of rows | one row per observation |
| numeric aggregation | sum, min, max over selected rows | `Terminal::Masked{Sum,Min,Max}I32` |
| weather quantization | f32 K to a u8 code under a versioned floor | `weather_poc::floor::CalibratedFloor` |

## Pass conditions, fixed before running

- **H1 Boolean.** The slab fold's AND3 count equals a row-at-a-time oracle on
  empty, full, sparse, dense and ragged populations (last slab short, last
  word short). A plane with a set bit past the slab's rows is REFUSED. FAIL on
  any inequality.
- **H2 relation.** For `R = {(x,y,z) : (x+y+z) mod 256 = 0}` the oracle count
  is 65,536. If the AND3 of the three axis projections, broadcast over the
  product, counts anything else, "three axis masks stand for a three-axis
  relation" is REFUTED. An explicit relation plane over 256 slabs must count
  65,536 or the slab stack FAILS. Positive control: a factorized selection
  must count the product of its popcounts.
- **H3 exact statistics.** Count, sum, min and max over an `i32`
  tenths-of-a-degree lane equal an `i128` scalar oracle, with duplicate rows
  counted once each and an empty selection answering `None` for min and max.
  The mean rounds as the 1BRC reference does (to tenths, ties toward positive
  infinity), checked on negative ties.
- **H4 quantized.** Folds over u8 codes equal a scalar oracle over the same
  codes (exact in code space). Against the full-precision f32 field: the mean
  must be within one bucket width, min and max within one bucket width, and
  the count within the selected rows whose value lies in the two buckets that
  hold the predicate bounds. Anything outside is reported as FAIL, not
  rescaled. A decode under another floor version is REFUSED.
- **H5 data movement.** A warm query makes zero allocations and holds no
  temporary of population size (the scratch is per slab). The slab arm must
  be within 2x of the contiguous mask-RISC arm on the same program; if not,
  binding and address translation dominate. Each candidate must beat the
  scalar row loop, or the simplification buys nothing.
- **H6 admission.** A stale source frame and a plane from another coordinate
  space are REFUSED for the statistics folds as they are for counts.
- **u8 lanes.** If mask-RISC has no range predicate or min/max over a
  one-byte lane, that is recorded as the missing contract; the probe then
  widens codes to `i32` and reports the 4x source bytes.

## Results

Not run yet.
