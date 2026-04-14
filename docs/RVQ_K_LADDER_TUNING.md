# RVQ k-Ladder Tuning

How to pick `k_levels` per tensor shape, why the current defaults collapse on
large-vocab embeddings, and a concrete remediation using hierarchical CLAM
(256x256) for large-row tensors.

## 1. The shape-vs-k problem (TL;DR)

Progressive residual RVQ converges to `cos = 1.0` only when the terminal
codebook is dense enough relative to the row count. Empirically:

```
k_final >= n_rows / 4   =>  cos -> 1.0
k_final <  n_rows / 4   =>  cos collapses
```

At `n_rows = 151936` with `k_final = 4096`, the coverage ratio is
`4096 / 151936 = 2.7%`. There are too few terminal centroids to cover the row
manifold; residuals on the last stage are still large, so the reconstruction
cosine is far from 1.

Runtime scales as `O(n_rows * k * n_cols)` per stage. The same under-sized
ladder that fails on cosine also dominates wall time, because assignment scans
all `k` centroids against all `n_rows` rows at each level.

### What actually happened (Qwen3-TTS-12Hz-0.6B-Base, 1.8 GB, 478 tensors)

First-ever end-to-end RVQ completion on this checkpoint produced:

| Metric | Value |
|---|---|
| Tensors at `cos = 1.0000` | 477 / 478 |
| Failing tensor | `model.text_embedding.weight [151936, 2048]` at `cos = 0.0544` |
| Codec token match | 181 / 225 = 80.4% (below 90% success threshold) |
| Codebook storage | 4523.7 MB vs 3657.2 MB original = 1:1.24 (codebook LARGER than weights) |
| Wall time for the single bad tensor | 891 s of 1417 s pass-2 total |

The single embedding tensor is responsible for every failure mode at once:
bad cosine, bad token match, net-negative storage, and ~63% of runtime.

### Current defaults in code

| Tensor class | `k_levels` |
|---|---|
| `k_proj`, `v_proj`, `down_proj` | `[256, 512, 1024]` |
| Everything else with `n_rows >= 128` and `n_cols >= 128` | `[256, 512, 1024, 4096]` |
| `norm`, `bias`, sub-128 tensors | skip RVQ, keep BF16 |

These work on 477 of 478 tensors. They do not work for tensors with
`n_rows > 8192`.

## 2. Per-shape k-ladder recommendation

| `n_rows` | `n_cols` | `k_levels` | Rationale |
|---|---|---|---|
| `< 128` | any | skip RVQ, keep BF16 | norms, biases; too small to benefit |
| `128 - 2048` | any | `[256, 512, 1024]` | default, cos = 1 in practice |
| `2049 - 8192` | any | `[256, 512, 1024, 4096]` | current default, works |
| `> 8192` | any | hierarchical CLAM 256x256 (not progressive residual) | see Section 3 |

The `> 8192` threshold is chosen so that `k_final = 4096` satisfies
`k_final >= n_rows / 4` at the boundary (8192 / 4 = 2048, with 2x margin up to
8192). Above it, progressive residual cannot be rescued by larger `k` without
blowing up storage (see Section 5).

## 3. Hierarchical CLAM (256x256) for large-row tensors -- DESIGN

Structurally different from progressive residual RVQ. Each row lands in
exactly ONE L2 leaf and is reconstructed as a single centroid -- there is no
residual sum across stages.

### Algorithm

```
L1_centroids = clam_sample(rows, 256)            # coarse 256 clusters
L1_assign    = assign_nearest(rows, L1_centroids)

L2_codebooks = {}                                 # one per L1 cluster
L2_assign    = vec![0usize; n_rows]

for cluster in 0..256:
    sub_rows              = rows.filter(|i| L1_assign[i] == cluster)
    L2_codebooks[cluster] = clam_sample(sub_rows, 256)
    for i in sub_rows:
        L2_assign[i] = nearest in L2_codebooks[cluster]

reconstruct(i) = L2_codebooks[L1_assign[i]][L2_assign[i]]
```

### Storage at 151936 x 2048 (bf16, 2 B/elem)

| Component | Size |
|---|---|
| L1 centroids: `256 x 2048 x 2 B` | 1 MB |
| L2 codebooks: `256 clusters x 256 centroids x 2048 x 2 B` = `65536 x 2048 x 2 B` | 256 MB |
| Indices: `151936 x 2 B` (packed 8 + 8 bits) | 297 KB |
| Total | 257 MB |
| Original tensor | 620 MB |
| Ratio | 2.4 : 1 at `cos ~= 1.0` |

Average 151936 / 65536 = 2.32 rows per L2 leaf -- essentially one centroid per
two or three rows, which is why fidelity is near 1:1.

### Contrast with progressive residual at same row-coverage budget

To hit `k_final >= n_rows / 4 = 37984` with progressive residual, you would
need a ladder like `[256, 4096, 16384]` or larger. The intermediate codebooks
dominate storage without improving cos past 1, and you still may not clear the
coverage threshold. See Section 5.

## 4. Decision rule in code

Shape-time dispatch at the caller (`load_weights` picks the path based on
tensor dimensions). Roughly twenty lines at the dispatch site:

```rust
if n_rows > 8192 {
    build_hierarchical_clam_256x256(rows)           // new path
} else {
    build_progressive_residual_rvq(rows, k_levels)  // existing path
}
```

Mirror the same branch in `reconstruct_rvq` so readers see a matching
`if n_rows > 8192 { reconstruct_hierarchical_clam(...) }` path. No changes to
the existing progressive-residual code for the 477 tensors that already hit
`cos = 1.0`.

This doc is tuning guidance. The implementation beyond this dispatch sketch is
out of scope here.

## 5. Why not just add `k = 16384` to progressive residual?

It would likely fix cosine, but it makes storage worse, not better:

- A 16384-centroid final codebook at `n_cols = 2048`, bf16, costs
  `16384 x 2048 x 2 B = 64 MB` on top of the existing levels.
- Every intermediate level (`256`, `512`, `1024`, `4096`) still materializes
  as a full-width codebook.
- The codebook-larger-than-the-weights failure mode (1:1.24) gets worse,
  not better.

Hierarchical CLAM avoids materializing the product codebook. The effective
codebook size is `256 * 256 = 65536` centroids, but only `65536 x 2048 x 2 B =
256 MB` of L2 storage is needed because the L2 lookup is conditional on L1
assignment -- you never store `L1_k * L2_k` independent full-width entries.

## 6. Which tensors in Qwen3-TTS-0.6B trigger the new path?

| Tensor | Shape | Path |
|---|---|---|
| `model.text_embedding.weight` | `[151936, 2048]` | hierarchical CLAM 256x256 |
| `lm_head.weight` (if present in a future variant) | `[151936, hidden]` | hierarchical CLAM 256x256 |
| All other 477 tensors | various, `n_rows <= 8192` | current progressive residual (already `cos = 1`) |

## 7. Expected outcome after remediation

| Metric | Current | Target |
|---|---|---|
| Tensors with `cos >= 0.99` | 477 / 478 | 478 / 478 |
| Codec token match | 80.4% | >= 95% |
| Model-wide storage ratio | 1 : 1.24 (regression) | >= 2 : 1 |

The codec token match recovery comes from cleaning up the first-token
embedding lookup -- once that tensor reconstructs correctly, every downstream
layer receives clean inputs and the token-match degradation at 181/225 is
resolved.

## 8. Cross-reference

- `RVQ_ENCODER_REPLICATION.md` -- how to run the encoder end-to-end.
- `RVQ_ALTERNATIVES.md` -- cases where RVQ is not the right codec at all.
- Original failure analysis: PR comment
  `AdaWorldAPI/lance-graph#176#issuecomment-4245767939`.
- Example entry point: `crates/thinking-engine/examples/tts_rvq_e2e.rs`.
