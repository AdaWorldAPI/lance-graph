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

## Results (run 2026-11-13, after the conditions above were committed in `4a036c1`, which was `8ef03a3` before the rebase onto #2054)

**Grade:** MEASURED on one machine (Intel Xeon 2.1 GHz, 4 cores, AVX-512 with
VPOPCNTDQ; `rustc` 1.99, release). Field: ARCO-ERA5 `2m_temperature`, time
index 96000, 721 x 1440 `f32`, sha256 `bc2024bf…6c3e`, fetched for the run and
not committed. Query: 35N-70N, 10W-40E, `0.0 C < t < 10.0 C`.
Reproduce: `ERA5_T2M_F32=<file> cargo run --release -p dr256-ternlog-lab --example weather_ternlog`;
tests: `cargo test --release -p dr256-ternlog-lab`.

| claim | verdict | evidence |
|---|---|---|
| H1 AND3 count equals the row loop; dirty tail refused | PROVEN (tests) | `tests/three_axis.rs`, 5 tests |
| H2 three axis masks stand for a three-axis relation | REFUTED | axis masks count 16,777,216; the relation holds 65,536 |
| H2 an explicit relation plane over 256 slabs counts it | PROVEN (tests) | 65,536 from a 2 MiB plane; three planes are 6 MiB |
| H3 exact count, sum, min, max, bag semantics, 1BRC mean rounding | PROVEN (tests) | `tests/weather_fold.rs`; four disable runs went red |
| H4 code-space folds exact; version mismatch refused | PROVEN (tests) | same file |
| H4 count against f32 | MEASURED, PASS | 10,323 vs 10,556; diff 233, bound 580 |
| H4 mean against f32 | MEASURED, PASS | 277.9578 K vs 277.9695 K, 0.04 bucket |
| H4 selected min, max against f32 | MEASURED, min PASS, max FAIL | min +0.204 K; max -0.328 K against a 0.263 K bucket |
| H4 global min, max through the floor | MEASURED, FAIL | min +14.07 K, max -5.31 K: 8,308 cells saturate |
| H5 zero allocations, no population-sized temporary | MEASURED, PASS | 0 allocations in every arm; scratch per program 0 B (fused three-plane count), 2,056 B (one slot) or 6,152 B (three slots), summed per arm: at most 24,608 B for four lane-predicate folds |
| H5 slab arm within 2x of contiguous | MEASURED, PASS | 1.00 to 1.07 with lane predicates; 1.13 to 1.52 on the pure three-plane count |
| H5 candidate beats the scalar loop | MEASURED, mixed | see timings |
| H6 stale frame, foreign plane refused for statistics | PROVEN (tests) | `a_stale_frame_and_a_foreign_plane_refuse_the_statistics_folds` |
| one-byte lane | REFUSED by mask-RISC | sum in place works (`MaskedStridedGroupSum`); ordered compare and min/max answer `ExecError::LaneKind` |
| billion-row throughput | NOT TESTED | largest run is 16.6 M rows, 254 slabs |
| per-station statistics inside the slab fold | NOT TESTED | run through mask-RISC directly, contiguous |

Timings, median of 21 samples, one timestep (1,038,240 rows, 16 slabs):

| arm | AVX-512 | generic x86-64 |
|---|---|---|
| scalar loop, one pass, four statistics | 416 us | 552 us |
| three resident planes, count (slab) | 5.0 us | 17.3 us |
| three resident planes, count (contiguous) | 3.8 us | 16.3 us |
| three resident planes, four folds (slab) | 118 us | 167 us |
| bounds from the lane, region-gated, count (slab) | 53 us | 130 us |
| bounds from the lane, region-gated, four folds (slab) | 303 us | 631 us |
| bounds from the lane, not gated, count (contiguous) | 290 us | 2,257 us |
| borrowed 256-position selections, count | 45 us | 33 us |

At 16.6 M rows (AVX-512): scalar 14.9 ms, resident four folds 3.2 ms, gated
four folds 6.3 ms, resident count 0.26 ms. At one region-dense slab the gated
four folds take 32 us against 20 us scalar: the candidate loses there.
Cold cost of the three resident planes for one timestep: 6.5 ms, 389 KiB.

What the numbers say:

- The three-plane ternlog and popcount is the fast part: about 100x the scalar
  loop when the planes are resident. The compare that makes a plane from an
  `i32` lane costs as much as the scalar loop unless the region gates it.
- Four statistics are four programs, so the selection is recomputed four
  times. With resident planes that still wins 3.5x to 4.7x; with lane
  predicates it wins only when the region is sparse (1.4x at one timestep,
  0.6x on a dense slab, 0.9x on generic x86-64).
- Slab stacking costs nothing measurable beyond the binding check
  (about 80 ns per slab).
- Rounding kelvin to tenths moves 92 of 10,556 rows across the two bounds.
  That is the conversion, not the fold; the fold is exact over the integers.
- A quantized code predicate with exclusive bounds drops both bound buckets,
  so the selected maximum is off by more than one bucket by construction.

**Decision:** the existing ternlog and masked integer terminals are sufficient
for the Boolean selection and for exact global statistics over `i32` lanes.
No GEMM and no cube is needed. Two things are not covered and are the next
measurements, not promotions:

1. a one-byte lane has no ordered compare and no extrema, so u8 weather codes
   must be widened to `i32` (4x source bytes) or carried as bit-sliced planes;
2. count, sum, min and max in one pass over a selection does not exist as a
   terminal.

**Open:** whether resident predicate planes are affordable when the bounds
change per query (they are 6.5 ms to rebuild per timestep here); per-station
statistics on a slab stack (grouped sinks need a merge across slabs);
throughput at 10^9 rows.

**Correction, same day (Codex review on PR 2055):** the first run reported
scratch as slots x 8 KiB for the slab arms and 0 for the contiguous arms.
Both were wrong. The executor carves one 256-word tile per slot plus a
bitmap word, whatever the population, so the figures are the ones in the
table above, now read from `scratch_words_for(tile_words_for(n), slots)`.
The H5 verdict does not change: no arm holds a population-sized temporary.
