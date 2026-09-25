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

| cached-mask query (throughput, random) | hot | cold (4096 masks, 32 MB) |
|---|---|---|
| by slot + bit/word peek | 1.8–2.3 ns | 18.6–22.4 ns |
| by `HashMap` key + peek | 13.6 ns | 85–104 ns |

HHTL partial mask applied vertically to one node's 6-byte tier path (half the facet), by working set of 16-byte records:

| working set | latency (dependent) | throughput |
|---|---|---|
| ≤ 2k rows (≤ 32 KB, L1d) | 6.8 ns | 1.6 ns |
| 4k–8k rows (64–128 KB) | 9.0–10.4 ns | 1.9–2.0 ns |
| **32k rows (512 KB)** | **12.5–13.4 ns** | 2.2 ns |
| 64k rows (1 MB, one tile) | 16.5–18.8 ns | 3.0–3.2 ns |
| **256k rows (4 MB)** | **27–34 ns** | 3.9–4.2 ns |
| 4M rows (64 MB) | 134–145 ns | 23 ns |

- **The recalled 12 / 25 ns is the latency of a vertical HHTL partial mask.** About 12 ns is a working set of ≤ 32k rows; about 25 ns is one that has spilled past a tile.
- **This is why positive vs negative selection pays off.** Store whichever of the selection or its complement is smaller, and a 64k tile never holds more than 32k members (half length), which keeps a partial mask in the ~13 ns regime.
- **Precision on "32k":** with 16-byte records L1d holds 2k rows, so at 32k rows the records are already L2-resident. The 32k cap is a member count. As a bitplane, 32k rows is 4 KB and fits L1.
- **The "~11 ns per cached-mask hit" premise:** it holds for a keyed (`HashMap`) hot hit. A direct slot hit is about 2 ns (throughput) and 5.5–6 ns (latency). A cold dependent hit costs DRAM latency (~150 ns).
- **Probe correction:** a first run used `%` by the row count, and the integer division inflated latency (17 ns in L1). The row index is now a power-of-two bit-mask.
- **Caveat:** queries hit uniformly random nodes. HHTL-ordered access is the realistic case.

### At the real `NodeRow` stride: in place (#1284) vs a packed copy
#1284's strided views read a `NodeRow` field where it sits. A MemWAL memtable's rows can therefore be filtered with no second SoA copy for reading. The cost is cache footprint: every row touched pulls its own 64-byte line, versus four 16-byte records per line when packed.

| rows touched | packed 16 B records (latency) | in place, 512 B `NodeRow` (latency) |
|---|---|---|
| 512 | 6.8 ns | 11.4 ns (32 KB of lines, L1) |
| 4k–8k | 9.0–10.4 ns | 22–27 ns |
| **32k** | **12.5–13.4 ns** | **50–55 ns** |
| 64k (one tile) | 16.5–18.8 ns | 95–100 ns |

- **In place, the ~12 / ~25 ns regimes shift down by about 64×.** They sit at roughly 512 and 4k–8k rows touched, not 32k and 256k. Positive/negative selection still halves the member count, but at 32k members an in-place random probe costs about 50 ns.
- **Traversal takes the packed column, not the rows.** Traversal selects only the NodeGuid lane over the 64k tile: 64k × 16 B = 1 MB of keys. That is the packed 16-byte column above (about 13 ns at 32k members, 17 ns for a full tile). In a columnar SoA the key lane is a column, not a second copy. The 512-byte in-place column applies only when a probe must reach past the key into the row's value.
- **Measured under random access only.** HHTL-ordered access touches rows in address order, which the hardware prefetcher streams. Measuring that before choosing is the open item.

