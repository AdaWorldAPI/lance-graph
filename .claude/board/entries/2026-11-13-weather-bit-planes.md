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

## Results

Not run yet.
