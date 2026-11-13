# 2026-11-13 — D-WX-BIT-PLANES: a one-byte code as resident bit-planes (lab probe)

**Kind:** lab probe in `crates/dr256-ternlog-lab`. No production crate changes.
**Grade:** HYPOTHESIS until the results section is filled; the conditions
below were written before any candidate ran.
**Follows:** D-WX-DR256-TERNLOG (`2026-11-13-weather-dr256-ternlog.md`), which
left "a one-byte lane has no ordered compare" open, and section 8.4 of
`docs/research/2026-11-13-ce64-logic-neutrino-grammar-spine.md`, which
measured that a byte read as a subset of eight 3-bit codes is the ternlog
immediate over the three bit-planes.

## Hypothesis

A u8 weather code does not need widening to `i32` to be selected on. Held as
resident bit-planes (plane `k` = bit `k` of every row's code):

- **B1, bands.** The top three bits of the code are eight bands. Any subset
  of bands is one byte, and `ternlog(byte, bit7, bit6, bit5)` selects it
  (`a` = bit 7, `b` = bit 6, `c` = bit 5; band index = `(a<<2)|(b<<1)|c`).
- **B2, full range.** `lo < code < hi` over all eight planes is a Boolean
  expression the existing fuser (`lance_graph_mask_risc::fuse`) lowers to a
  chain of ternlogs.
- **B3, sum.** `Σ code` over a selection is `Σ_k 2^k · popcount(sel & bit_k)`:
  eight counts, no value lane read.

## Pass conditions, fixed before running

- **B1.** For all 256 subset bytes, the count of `ternlog & region` equals a
  row loop over `code >> 5`. FAIL on any inequality. A bound that is not a
  band edge is not expressible by B1; that is recorded, not worked around.
- **B2.** For every threshold 0..=255, the fused `code > t` and `code < t`
  counts equal a row loop; so does the range used in the first probe.
- **B3.** The eight-count sum equals the row-loop sum for the same selection.
- **Cost.** Zero allocations per warm query. On one ERA5 timestep:
  the band count must be no slower than the region-gated widened-`i32` count
  of the first probe (53 us there; re-measured in the same run), or B1 buys
  memory only. The eight-plane range count must be no slower than the
  ungated `i32` count (290 us there), or bit-slicing is not a usable
  replacement for the compare; its ratio to the gated `i32` count is
  reported either way.
- **Bytes.** Reported: planes read per query against `4n` for the widened
  lane.
- **Not claimed.** Min and max over bit-planes; a slab-stack form (the lab's
  `Prepared` holds three planes per slab and this needs four or nine); any
  change to mask-RISC.

## Results (run 2026-11-13, after the conditions above were committed in `840f454`)

**Grade:** MEASURED on one machine (Intel Xeon 2.1 GHz, 4 cores, AVX-512 with
VPOPCNTDQ; release). Field and query as in D-WX-DR256-TERNLOG: one ARCO-ERA5
`2m_temperature` timestep, 1,038,240 rows, codes `145 < code < 183`.
Reproduce: `ERA5_T2M_F32=<file> cargo run --release -p dr256-ternlog-lab --example bit_planes`;
tests: `cargo test --release -p dr256-ternlog-lab --test bit_planes`.

| condition | verdict | evidence |
|---|---|---|
| B1 every band subset is one ternlog immediate | PROVEN (test) | all 256 subsets equal the row loop |
| B1 a bound off a band edge | NOT EXPRESSIBLE | no subset equals `145 < code < 183`; a band is 8.4 K wide here |
| B2 fused `>` and `<` at every threshold, and the range | PROVEN (test) | 256 + 256 thresholds and the range equal the row loop |
| B3 sum as eight popcounts | PROVEN (test) | equals the row-loop sum |
| zero allocations | MEASURED, PASS | 0 in every arm |
| band count no slower than gated `i32` count | MEASURED, PASS | 5.4 us against 53 us |
| bit-plane range count no slower than ungated `i32` count | MEASURED, PASS | 48 us against 278 us |

Four disable runs (band planes swapped, `Or` to `And` in `>`, region dropped,
`Not` dropped in `<`) each turned two tests red.

Timings, median of 21 samples; the three runs agree except one noisy gated
`i32` sample (84 us):

| arm | time | source bytes read, at most | scratch |
|---|---|---|---|
| scalar loop over u8 codes, range count | 238 to 330 us | 2.1 MB | 0 |
| widened `i32`, region-gated, range count | 51 to 53 us | 8.6 MB | 6 KiB |
| widened `i32`, not gated, range count | 278 to 365 us | 8.6 MB | 8 KiB |
| eight bit-planes, fused range count (13 ops) | 48 to 53 us | 1.2 MB | 26 KiB |
| three bit-planes, band subset count | 5.3 to 5.4 us | 0.5 MB | 0 |
| eight bit-planes, sum as eight counts | 430 to 451 us | 9.3 MB | 224 KiB |
| widened `i32`, region-gated, masked sum | 82 to 86 us | 12.7 MB | 6 KiB |

Held: eight planes and the region are 1.17 MB against 4.15 MB for the widened
lane (0.28x). Building the planes once costs 11 ms.

What the numbers say:

- A range compare over eight resident bit-planes costs the same as the
  region-gated compare over a widened `i32` lane, without the region gate and
  on a quarter of the memory. It is 5x to 6x faster than the ungated compare.
- A band subset is the cheap case: one ternlog, no scratch, about 10x faster
  than either range form. Its limit is resolution, 8.4 K per band on this
  floor; the band query here selects 20,720 rows where the range selects
  10,323.
- The sum as eight counts is exact and about 5x slower than the masked `i32`
  sum, because each of the eight programs recomputes the 13-op selection.
  With the selection computed once it would be eight AND-and-count passes;
  that needs a program with more than one terminal, the same gap the first
  probe recorded.

**Decision:** for selection, bit-planes are a usable replacement for widening
a u8 code to `i32`. For sums they are not, as mask-RISC stands. Nothing is
promoted; no mask-RISC change is proposed here.

**Open:** min and max over bit-planes (a bit-descent of `Any` tests, not
written); the slab-stack form (the lab's `Prepared` holds three planes per
slab); whether a 13-op chain can be shortened by a dedicated comparator.
