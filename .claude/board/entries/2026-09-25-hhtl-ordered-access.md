# 2026-09-25 — HHTL-ordered access keeps a partial mask at ~3–4 ns per node at any working set

**Status:** MEASURED (`crates/lance-graph-mask-risc/examples/hhtl_order_probe.rs`; 3 runs, median of 5) · OPEN (see below)
**D-ids:** none new. Answers the "HHTL-ordered access has not been measured" item of `2026-09-25-aperture-masks-and-kernel-gap.md`.

## What was measured
The query is the one from `mask_cache_hit_probe`: one node's 6-byte tier path compared under a care mask. Only the visit order changes:
- **random:** a random permutation of the members;
- **ordered:** ascending address order, which is what a prefix walk over a sealed lane yields;
- **clustered:** 256-row blocks in random order, with rows in address order inside each block.

Latency is a pointer chase. Each member's record stores the next row in the chosen order, so only the load sits on the chain.

## Result (latency, ns per node)

| members, 16-byte records | random | ordered | clustered |
|---|---|---|---|
| 32k rows, packed (512 KB) | 7.8–8.7 | 3.2–3.3 | 3.2–3.3 |
| 32k scattered over a 64k tile | 12.3–16.5 | 4.2–4.6 | 4.3–4.5 |
| one tile (64k, 1 MB) | 13.1–14.6 | 3.2–3.3 | 3.3–3.5 |
| four tiles (256k, 4 MB) | 28.5–30.5 (one run 84) | 3.2–3.4 | 3.4 |
| 64 tiles (4M, 64 MB), walked end to end | 157–163 | 3.8–3.9 | 4.4–4.6 |
| 256k scattered over 4M (1 in 16) | 138–151 | 77–91 | 99–105 |

| members, 512-byte `NodeRow` in place | random | ordered | clustered |
|---|---|---|---|
| 4k rows (2 MB span) | 26–28 | 14.0–14.6 | 14.1–16.7 |
| 32k rows (16 MB span) | 65–125 | 22–24 | 18–27 |
| 64k rows (32 MB span) | 125–152 | 57–71 | 54–70 |
| 4k scattered over 64k | 40–42 | 36–40 | 36–41 |

## What it says
- **Ordered access over the packed key lane is flat, at ~3–4 ns per node, from L1 up to 64 MB.** The hardware prefetcher streams it. That is ~40× below random at 64 MB, and 4× below random within one tile. The ~12 / ~25 ns regimes are the cost of random order, not of the working set.
- **Only order within a subtree matters.** Jumping between 256-row blocks in random order costs almost nothing extra, so a frontier that finishes one subtree before moving on gets the ordered cost.
- **Sparsity defeats the prefetcher.** At 1 member in 16 rows (members about 4 cache lines apart), the ordered walk is still ~80–90 ns over 64 MB. At 32k members scattered over one tile (one in two), ordered holds at ~4 ns.
- **The in-place 512-byte stride gains less.** Consecutive rows are 8 cache lines apart, so ordering roughly halves the cost (2 MB span: 27 → 14 ns; 32 MB span: ~140 → ~60 ns) rather than flattening it. This favours the packed key lane for traversal even more strongly than the random-access numbers did.
- **Consequence for the positive/negative selection argument:** capping members at 32k matters much less than visiting them in address order. A mask is address-ordered by construction, so a walk that follows its set bits in word order is already the ordered case.

## Probe correction (review)
A first version capped every case at 2^20 visits, so the 4M-member cases walked only their first 16 MB, which could sit partly in L3. Every case now visits at least all of its members. The rows over 4M are the full-walk figures from 3 runs. The ordered 64 MB result did not change (3.8–3.9 ns).

## OPEN
- The sparse case (1 in 16 and sparser) is where neither order helps. The alternatives (compacting survivors, software prefetch a few members ahead, or reading the bitplane instead of the records) are unmeasured.
- The measurement uses one core and no competing traffic. Prefetcher behaviour under multi-core load is untested.
