# 1/40 σ Lens — Research Note (additive, no code changes)

> **Status:** research only. No production code touched.
>
> **Companion:** `INVARIANT_MATRIX_RESEARCH.md` — this note fills in the
> `1/40 σ ring lens` row of the invariant matrix.
>
> **Why this note exists:** the user's question — *"i hope the 1/40 sigma lens
> adds insights we can cover with"* — deserves an honest accounting of what
> that lens actually contributes, before any code is written.

---

## What 1/40 σ actually is (unambiguously)

Given a query anchor (a single centroid), look at the distance distribution from
that anchor to every other centroid in the codebook. Compute σ (standard deviation
of those distances). Slice the distribution into 40 concentric bands, each
1/40 σ = 0.025 σ wide.

```
band 0:   0.000 σ to 0.025 σ   (closest neighbors, "inner ring")
band 1:   0.025 σ to 0.050 σ
band 2:   0.050 σ to 0.075 σ
...
band 39:  0.975 σ to 1.000 σ   ("outermost ring within 1 σ")
band 40+: > 1.000 σ            (out-of-focus, ignored or down-weighted)
```

Each centroid falls into exactly one band per query. The lens output for a
query is a 40-element vector: `count[band_i]` or `activation[band_i]` depending
on variant.

**Critical:** σ is computed *per query*. Each query gets its own ring
normalization, so the lens is query-adaptive. A query in a dense neighborhood
has a small σ; a query in a sparse neighborhood has a large σ. The 40 bands
always span the same number of standard deviations regardless.

---

## Why 1/40 σ is not "just rank"

This is the subtle point the invariant matrix clarifies. At first glance,
slicing a distance distribution into bands looks like a quantile operation,
which we already know (from γ+φ) is rank-preserving and nothing more.

It is not rank-only, for three reasons:

### 1. σ carries magnitude

The band width `0.025 σ` is measured in the units of the actual distance
distribution. A query with widely dispersed neighbors has physically wider
bands than a query with tightly clustered neighbors. When two queries are
compared, their ring profiles encode *how spread out* their neighborhoods are,
not just the ordering. That's magnitude information that survives the band
assignment.

### 2. The band *count* per ring is not monotone in any global quantity

Under a rank-only lens, the number of items in each decile is fixed by
construction (10 % per decile). Under 1/40 σ, the number of items in each
band varies dramatically with the *shape* of the neighborhood distribution.

- A query at the center of a tight cluster: band 0 is densely populated, band 39
  is sparse.
- A query at the edge of a cluster: band 0 is sparse (nothing right next to it),
  bands 10–20 are dense (the cluster body), band 39 sparse again.
- A query in a uniform neighborhood: all bands roughly equal.

The 40-element ring profile is a **shape descriptor of the local density
function**, and different neighborhoods produce measurably different profiles.
That's not rank — that's manifold structure.

### 3. With "hole" (block self-reference) the center band changes meaning

If we zero out the diagonal (query does not count itself as its own nearest
neighbor), band 0 becomes "nearest actual neighbor, not self." For a query at
the center of a tight cluster this is still dense; for a query with no close
neighbors, band 0 becomes a gap. The gap itself is information. A rank lens
cannot distinguish "densely surrounded" from "isolated but forced to rank
something first" because rank always produces a well-ordered list. 1/40 σ
distinguishes them via band occupancy.

---

## What invariants 1/40 σ actually carries

Mapping to the invariant matrix:

| Axis | 1/40 σ contribution |
|---|---|
| 1. Magnitude | ✓ (via per-query σ normalization; band width is physical) |
| 2. Sign | – (distances are non-negative by construction) |
| 3. Rank | ✓ (band ordering is monotone in distance) |
| 4. Pair identity | – (individual pair (i,j) is not preserved; bands aggregate) |
| 5. Manifold curvature | ✓ (band occupancy = local density function shape) |
| 6. Trajectory | – (no sequence) |
| 7. Sparse structure | ✓ (empty bands are first-class signal, especially with hole) |
| 8. Phase | ? (only if combined with ring-to-ring perturbation, see companion note) |

Four `✓`s and one `?`. That's more than any other single candidate in the
matrix carries for the mid-range axes (4–8). **This is why the lens is
interesting — it's one of the few candidates that covers axis 5 (manifold)
and axis 7 (sparse) in the same encoding.**

CLAM covers axis 5 from a different angle (hierarchical clustering + knee
detection). 1/40 σ covers axis 5 locally, per-query, without needing a tree
build. They're complementary, not substitutes:

- **CLAM:** global structure, amortized tree build, knee-delta as scalar per cluster.
- **1/40 σ:** per-query local density profile, no index build, 40-element vector
  per query.

If they agree, the manifold axis is robust. If they disagree, one of them is
seeing structure the other isn't, and the disagreement itself is diagnostic.

---

## What 1/40 σ does NOT give us (honesty)

Three claims to refuse until measured:

### 1. It is not phase reconstruction on its own

The ring profile is radial. Radial does not mean angular. Phase = relative angle
between two centroids around a query anchor. 1/40 σ does not encode angle; all
centroids at distance 0.3 σ go into the same band regardless of direction.

Phase enters only when 1/40 σ is *combined* with another lens that carries
angular information — which is where the ring perturbation mechanism comes in
(see `RING_PERTURBATION_PROPAGATION.md`). The ring lens is the radial half;
perturbation between rings is the angular half. Neither alone is phase
reconstruction.

### 2. It is not "meta-awareness"

The lens gives a 40-element vector per query. That vector is a local descriptor,
nothing more. Calling it meta-awareness is aesthetic framing that will not
survive peer review. What the lens produces is testable: does the profile
correlate with downstream task quality (SPO extraction accuracy, NARS truth
stability, contradiction detection)? That's the bar.

### 3. It does not replace attention

The ring profile is a compressed distance histogram. Attention in a live
transformer is a weighted mixture with learned query/key projections per head.
They are not the same operation. 1/40 σ may *approximate* parts of what
attention computes, specifically the radial weighting around a query, but it
cannot substitute for the learned projections.

The fair claim is:

> 1/40 σ is a static, compressible approximation of the radial component of
> attention around a query. Whether that approximation is accurate enough to
> support downstream tasks is an empirical question, not a theoretical guarantee.

---

## How to measure what 1/40 σ actually adds

Three probes, all additive, all cheap, all runnable without touching library code:

### Probe A: Profile distinctness

For a fixed set of query centroids, compute the 40-element ring profile under
1/40 σ for each query. Then compute pairwise distances between profiles
(L1 or χ²). Expected signal: semantically similar queries should have similar
profiles; semantically distant queries should have distinct profiles.

**Pass:** Spearman correlation between profile distance and semantic distance
(from the Jina v5 anchor) > 0.5.
**Fail:** Correlation < 0.2 — profile is noise relative to semantics.
**Uncertain:** 0.2 < ρ < 0.5 — profile carries some signal but not decisively.

**Cost:** one example file, ~100 LOC, runs on the existing 256-centroid codebook
in under a minute.

### Probe B: Hole vs no-hole

Run Probe A twice: once with self-reference allowed (query counts itself at
distance 0), once with the diagonal blocked. Compare.

**If hole matters:** profiles diverge, especially in band 0. That means
self-reference was contaminating the measurement and blocking the diagonal
is a legitimate structural fix.

**If hole does not matter:** profiles barely change. That tells us the "hole"
framing is aesthetic, and the diagonal self-reference was not the bottleneck
for K=256 collapse. Different bottleneck to hunt.

**Cost:** ~30 additional LOC in the same probe file.

### Probe C: σ normalization sensitivity

Run Probe A with three band-width normalizations:
- Per-query σ (the default)
- Global σ (computed once across all query–centroid distances)
- Fixed band width (e.g., 0.025 × max_distance)

**Expected:** per-query σ wins on tasks where neighborhoods have very different
densities (e.g., dense scientific vocabulary vs. sparse named entities). Global
σ wins when the codebook is roughly uniform. Fixed width loses in both cases.

**If per-query σ does not win:** the "query-adaptive" framing is oversold and
we can use the cheaper global normalization.

**Cost:** ~50 additional LOC, three runs of Probe A.

---

## What the probes would let us write in the matrix

After Probe A runs, the row in `INVARIANT_MATRIX_RESEARCH.md` updates from:

```
1/40 σ ring lens    ? – ? – ? – – ?
```

to one of:

```
PASS:  ✓ – ✓ – ✓ – ✓ ?    (strong manifold + sparse contribution)
FAIL:  ✗ – ✓ – ✗ – ✗ ✗    (degenerate rank lens in disguise)
MIXED: ✓ – ✓ – ? – ✓ ?    (magnitude and sparse preserved, manifold unclear)
```

And Phase 5.12.0 gets a data point: either 1/40 σ earns its column or it
becomes a documented negative result. Either outcome is progress.

---

## Interaction with the 128-step forward cache

One place the 1/40 σ lens could matter unexpectedly: the 128-step cache
computes its trajectories in u8 integer space at 372K tok/s. If we apply
1/40 σ *as the cache's scoring function* instead of plain cosine, the cache
would prioritize trajectories that stay within a known ring profile
(i.e., semantically coherent trajectories) over trajectories that drift
across bands (i.e., topic jumps).

This is speculative and would need its own probe. Note it for later — do
**not** wire it into the cache yet. The Python/Rust contradiction has to be
resolved first; adding a new scoring function to a pipeline that already has
inconsistent output is how we'd create a third contradiction.

---

## Summary

**What 1/40 σ adds to the invariant matrix:**

1. A candidate for axis 5 (manifold) that is cheap, per-query, and does not
   require a tree build.
2. A candidate for axis 7 (sparse) via the hole variant — empty bands become
   first-class signal.
3. A candidate for axis 1 (magnitude) via per-query σ normalization — physical
   band width, not pure rank.

**What 1/40 σ does not add:**

1. Phase information (that requires the ring-perturbation companion).
2. Pair identity (bands aggregate, individual pairs are lost).
3. Any guarantee at all until the probes run.

**What to do next (additive, no library edits):**

1. Write `1_40_sigma_probe.rs` as a new example file. Run Probes A, B, C.
2. Update the invariant matrix row with the measured values.
3. Decide: keep 1/40 σ as an active lens, or document it as a negative result.

**The honest version of the user's hope:**

> The user hopes 1/40 σ adds coverage we can't get elsewhere. It *could* —
> it's one of the few candidates that plausibly carries four invariant axes in
> one encoding. But until Probes A–C run, "could" is the honest word. Not
> "does." Not "will."
