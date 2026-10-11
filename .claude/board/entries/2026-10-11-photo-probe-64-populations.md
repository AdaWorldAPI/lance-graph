# 2026-10-11 — 64 resident populations: bit-sliced planes vs signatures vs classes (D-PHOTO-0)

**Status:** MEASURED. Probe only, no production change.
`crates/lance-graph-quack/examples/photo_probe.rs`.

```text
CARGO_PROFILE_RELEASE_DEBUG=0 cargo run --release -p lance-graph-quack --example photo_probe
```

The question: with 64 already-derived populations over 1,048,576 addresses
resident, which representation answers boolean questions over them fastest,
and what does keeping them cost? Every arm answers COUNT, Any, SUM and Rows
for the canonical `(P0|P1|P2|P3) & P17 & !P29` plus 20 random expressions,
asserted equal to an oracle. Population creation is a generator stand-in;
no real reasoning baseline exists yet.

## Arms

- **B** quack `lower_fused` / **Bi** `lower` (in-place) over the 64 mask-risc planes.
- **D** `execute_extent` over each expression's support range, from one occupied range per population (AND intersects, OR takes the hull, NOT gives up).
- **C** probe-local transpose to one u64 signature per address; the expression is tested per address.
- **E** HashMap signature → (count, sum); the expression is tested per class.

## Results (per query, COUNT, steady state; 1,000 queries for C/E)

| regime | B | Bi | D | C | E | classes |
|---|---|---|---|---|---|---|
| sparse (1%) | 0.11 ms | 0.046 ms | 0.037 ms | 12.2 ms | 0.39 ms | 24,180 |
| dense (50%) | 0.11 ms | 0.048 ms | 0.036 ms | 22.8 ms | 29.9 ms | 1,048,576 |
| clustered | 0.11 ms | 0.047 ms | 0.028 ms | 12.3 ms | 0.006 ms | 274 |
| archetypes | 0.13 ms | 0.048 ms | 0.036 ms | 22.5 ms | 0.009 ms | 256 |

Costs: transpose 4–51 ms; class build 14–176 ms; updating 1,000 addresses
of one population: planes 0.005–0.012 ms, signatures + classes 0.30–0.95 ms.
Spill of 8 MiB of planes 19–56 ms, reload 1.7–7.3 ms (page cache, not
cold). Nonzero words: sparse 47.5%, dense 100%, clustered 1.6%.

## Readings

1. **Bit-sliced planes win on every regime but one.** They touch only the
   planes the expression names (~5 of 64, ~82k words) and cost the same
   whatever the data looks like. Signatures (C) always test all 1M
   addresses and lose by 250–600×.
2. **Classes win only when there are few of them.** At 256–274 classes E is
   ~5× faster than planes; at 24k classes it is ~10× slower; at 1M classes
   (dense) it is slower than C. Class count is data-dependent and unknown
   in advance, and every update rebuilds them; planes update 30–100× cheaper.
3. **One occupied range per population is a weak descriptor.** On clustered
   data the populations occupy 1.6% of the words, yet D cuts per-query
   words only from ~82k to ~56k, because OR takes the hull of four runs.
   Multiple ranges per population would be needed to use the clustering.
4. **Spill/replay is cheap at this size** (8 MiB in tens of ms), and a
   sparse spill is worth it only for clustered populations.

## Against the real alpha channel

`AlphaMask` must not hold these: `alpha.rs` keeps motion and reference
frames as separate types that "may share a bitmap layout; they must not
share a type, a name or a conversion" (`ISS-LXA-ALPHA-FIT`). A cached
population therefore needs its own type over the same word layout, scoped
to (universe, snapshot version, expression identity, principal and policy
version for authorization-derived populations). Alpha persistence exists
only in MedCare-rs test writers, so there is no shipped spill path to reuse.

## Smallest missing seam

None for evaluation: quack + mask-risc already answer every query shape
here, fused, at ~0.03–0.05 ms. What is missing is a **population-cache
descriptor** (the scope tuple above plus exact/approximate and dependency
state) so a cached plane can be reused safely; and, if clustered data
matters, multi-range extents. Both are OPEN until a real reasoning
workload produces populations to measure.
