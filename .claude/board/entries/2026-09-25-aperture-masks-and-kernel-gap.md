# 2026-09-25 — Aperture masks are Self-derived, not random; the ternlog kernel is not the gap

**Status:** MEASURED · OPEN (the 5-plane scalar-loop lead) · the aperture→Range lowering landed (`2026-09-25-aperture-prefix-lowers-to-range.md`)

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
| by slot + bit/word peek | 1.9–2.3 ns | 17–21 ns |
| by `HashMap` key + peek | 15.3–15.8 ns | 99–109 ns |

The dependent (latency) form of the slot hit is 5.6–5.8 ns hot, but its chain runs through a ~3.1 ns hash (`mix()`, measured alone), so the lookup's own latency is about 2.5 ns.

HHTL partial mask applied vertically to one node's 6-byte tier path (half the facet), 16-byte records. Latency is a pointer chase: each member's record holds the next row of one random cycle, so nothing but the load is on the chain. Throughput queries are drawn before the clock starts. Four runs:

| members | latency (pointer chase) | throughput |
|---|---|---|
| ≤ 2k rows (≤ 32 KB, L1d) | 3.4–3.5 ns | 1.1–1.2 ns |
| 4k–8k rows (64–128 KB) | 5.2–6.1 ns | 1.3–1.4 ns |
| 32k rows, packed (512 KB) | 7.9–10.0 ns | 1.6–1.7 ns |
| 32k members scattered over a 64k tile | 12.2–12.7 ns | 2.7 ns |
| **64k rows (1 MB, one tile)** | **12.3–13.6 ns** | 3.1–3.7 ns |
| **256k rows (4 MB)** | **25–33 ns** (one run 98) | 3.7–5.8 ns |
| 4M rows (64 MB) | 167–206 ns | 21–23 ns |

- **The recalled 12 / 25 ns is the latency of a vertical HHTL partial mask**, at a working set of one tile (1 MB, ~13 ns) and of a few tiles (4 MB, ~28 ns).
- **Positive/negative selection halves the member count, not the footprint.** 32k members scattered over a 64k tile still span its 1 MB and cost 12.2–12.7 ns, close to the full tile, not the 8–10 ns of 32k packed rows. Staying under 32k helps the cache only if the members are also compacted, or if the probe reads the selection bitplane (32k rows = 4 KB, L1) rather than the records.
- **The "~11 ns per cached-mask hit" premise** is between a hot slot hit (~2 ns throughput, ~2.5 ns own latency) and a hot keyed `HashMap` hit (~15 ns throughput). A cold dependent hit costs DRAM latency (~150 ns).
- **Probe corrections (the last two caught in review):** a first version used `%` by the row count, and the integer division inflated latency. A second one drove the dependent chain through `mix()`, adding ~3.1 ns to every latency figure and making "32k rows" look like the 12 ns regime (12.5–13.4 ns reported; 7.9–10.0 ns measured). A third conflated member count with footprint (above). A fourth timed the throughput loop's own query generation, roughly doubling those figures.
- **Caveat:** queries hit uniformly random members. HHTL-ordered access is the realistic case.

### At the real `NodeRow` stride: in place (#1284) vs a packed copy
#1284's strided views read a `NodeRow` field where it sits. A MemWAL memtable's rows can therefore be filtered with no second SoA copy for reading. The cost is cache footprint: every row touched pulls its own 64-byte line, versus four 16-byte records per line when packed.

| rows touched | packed 16 B records (latency) | in place, 512 B `NodeRow` (latency) |
|---|---|---|
| 512 | ≈ 3.5 ns | 6.5–6.7 ns (32 KB of lines over a 256 KB span) |
| 4k–8k | 5.2–6.1 ns | 25–30 ns |
| **32k** | **7.9–10.0 ns** | **74–116 ns** (near L3 capacity, noisy) |
| 64k (one tile) | 12.3–13.6 ns | 143–147 ns |

- **In place, the regimes arrive far sooner in rows.** ~26–28 ns arrives at 4k rows touched instead of 256k (64×); ~13 ns falls between 512 and 4k rows instead of 64k. At 32k members an in-place random probe costs 74–116 ns.
- **Traversal takes the packed column, not the rows.** Traversal selects only the NodeGuid lane over the 64k tile: 64k × 16 B = 1 MB of keys. That is the packed 16-byte column above (about 13 ns for a full tile). In a columnar SoA the key lane is a column, not a second copy. The 512-byte in-place column applies only when a probe must reach past the key into the row's value.
- **Measured under random access only.** HHTL-ordered access touches rows in address order, which the hardware prefetcher streams. Measuring that before choosing is the open item.

