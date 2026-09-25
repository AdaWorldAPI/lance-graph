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

## Cached-mask hits and HHTL partial masks (`examples/mask_cache_hit_probe.rs`)
This measures `D-WFL-CACHE`'s `C_lookup`, which the plan recorded only as the premise "cached mask → zero-copy peek → ~11 ns". Before this, the fold figure (1.7 ns hot, 1.7–4.2 ns cold; #1245/#1250) was the only measured side of the ratio.

| query (throughput, random nodes) | hot | cold |
|---|---|---|
| cached mask by slot + bit/word peek | 1.8–2.3 ns | 18.6–22.4 ns (4096 masks, 32 MB) |
| cached mask by `HashMap` key + peek | 13.6 ns | 85–104 ns |
| HHTL partial mask, vertical, 6-byte half path | 9.9–10.3 ns (1 tile) | 11.6–19.3 ns (4 tiles) · 48.6 ns (64 tiles) |

- The recalled 12 / 25 ns for HHTL partial masks (vertical, half length) fits: the hot figure is ~10 ns, and 25 ns sits between the L2-edge and DRAM working sets.
- Measured with each query waiting on the previous one (latency), the same operations cost 30 / 55 / 160 ns. A cached mask read by slot costs 5.5–6 ns hot.
- The premise "~11 ns per cached-mask hit" holds only for hot throughput with a keyed lookup (a `HashMap` costs 13.6 ns). A direct slot is about 2 ns. A cold dependent hit costs DRAM latency (~150 ns), not 11 ns.
- Caveat: queries hit uniformly random nodes. HHTL-ordered access is the realistic case and would sit closer to the hot column.

