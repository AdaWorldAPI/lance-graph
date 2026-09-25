# 2026-09-25 — Aperture masks are Self-derived, not random; the ternlog kernel is not the gap

**Status:** MEASURED · OPEN (the 5-plane scalar-loop lead; the aperture→Range lowering in the planner)

## Correction to the probe method
- Every mask-risc probe so far fed uniformly random bit planes. That measures kernel throughput. It models nothing about where a membership mask comes from.
- A real mask is the Self of the row: its 12-byte V3 facet (6×`u8:u8` = 96 bits) seen through an APERTURE, the `care` bytes of `Pred::MatchFacetStrided`.
- An aperture is a deterministic cut over the address/content bytes. It is not a similarity neighbourhood around a cluster (no Mexican hat), and popcount is not the operation that performs it: VPOPCNTDQ measures cardinality and adjacency, it does not select.

## Aperture probe (`crates/lance-graph-mask-risc/examples/aperture_probe.rs`)
One 64k-row tile, 16-byte records (classid 4 B + facet 12 B), with rail 0 = the row address. Median of 201, 3 runs.

| aperture | survivors | runs | mask cost | as `Range` |
|---|---|---|---|---|
| rail0.hi (address prefix) | 256 | 1 | 112 µs (1.72 ns/row) | 57 ns |
| rail0 (exact address) | 1 | 1 | 112 µs | 57 ns |
| rail1.hi (coarse bucket, address-correlated) | 4096 | 1 | 112 µs | 57 ns |
| rail1.lo (palette, content) | 4095 | 4095 | 112 µs | — |
| rail0.hi + rail1.lo | 17 | 17 | 112 µs | — |

- The match cost does not depend on the aperture: every row pays the 12-byte ternary compare.
- An aperture over address bytes selects one contiguous run on an address-ordered tile. As a `Range` the fold touches no word: about 2000× cheaper. The IR already records that deciding address order is the planner's job (`Pred::Range` doc).
- Hot masks amortize: `bucket & palette & !other` over two cached aperture masks costs 330 ns; recomputing both apertures in the same program costs 230 µs, about 700× more.

## ndarray kernel gap (`ndarray/examples/ternlog_popcnt_gap_probe.rs`)
- Three planes: production `mask_ternlog_popcount` equals a Mula-popcount lab arm at every size. The scalar-lane popcount fallback (no VPOPCNTDQ on this host) is not the cost. It sits within about 1.15× of a no-popcount floor, and at 1024 words (one tile) it is 1.3× faster than the scalar loop.
- Five planes, 4K–16K words: the register-resident two-ternlog loop buys only 1.1× over the production two-pass shape. The scalar loop is still 1.2–1.5× faster than every `U64x8` form. Unrolling does not explain it; a vector-width rebuild was inside noise. OPEN, and small next to the aperture numbers above.
