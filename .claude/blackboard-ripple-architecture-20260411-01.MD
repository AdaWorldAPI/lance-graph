# Blackboard: Ripple Architecture 2026-04-11 / 01

## Purpose

This blackboard records three P0 falsifications discovered during a deep
Agent-driven investigation of the BF16-HHTL correction chain, plus the
reframed Probe M1 that replaces the architecturally-impossible original,
plus the structural implications for HEEL/HIP/BRANCH/TWIG/LEAF.

Written in response to:
- "P0 never assume"
- "Use your agents for depth, not bookkeeping"
- "Use the blackboard"

All claims here are labeled CONFIRMED (measured via Agent investigation of
actual source code) or CONJECTURE (next probe target).

---

## Three P0 Falsifications (CONFIRMED via source-level investigation)

### F1. CLAM is binary-only — "3 levels × 16-way" is architecturally impossible

**Evidence**: `/home/user/ndarray/src/hpc/clam.rs:341-379`. `ClamTree::build()`
always produces exactly 2 children per node via `partition_left_right()`.
No `branching_factor` parameter. The lance-graph wrapper at
`/home/user/lance-graph/crates/lance-graph/src/graph/neighborhood/clam.rs:74`
is a Pareto-radius validation harness — it only produces `RadiusObservation`
records, no bucket IDs, no hierarchy traversal. The bgz17 version at
`crates/bgz17/src/clam_bridge.rs:250` is a full wrapper but still binary.

**Implication**: The canonical `bf16-hhtl-terrain.md § 4` Slot D layout

```
bits 15..12 = CLAM L0: 16 coarse clusters → HEEL
bits 11..8  = CLAM L1: 256 mid-level → HIP (1:1 Jina centroids)
bits  7..4  = CLAM L2: 4096 terminal → TWIG (COCA vocabulary)
```

is structurally incompatible with existing CLAM. On 256 input points, binary
CLAM produces ≤ 256 leaves at depth ~8 (2^8 = 256), not 4096 leaves at depth 12.
Reaching 4096 leaves would require 4096 input points, not 256.

This does NOT mean the bit layout is wrong. It means the layout's **alignment
claim** (16→256→4096 as CLAM levels 0/1/2) is wrong. The layout could still
be used with a different clustering algorithm, or with 4 binary CLAM levels
per 4-bit chunk (16 → 256 → 4096 via binary tree depths 4 → 8 → 12), but that
requires 4096 input points, not 256.

### F2. The 256 Jina codebook on disk is a PAIRWISE COSINE MATRIX, not embeddings

**Evidence**: `/tmp/codebooks/jina-v5-semantic-256/` contains:
- `cosine_matrix_256x256.f32` (262 KB, 256×256 f32 cosines, diagonal=1.0)
- `distance_table_256x256.i16` (128 KB, quantized version)
- `codebook_index.u16` (297 KB, 151,936 tokens → 256 centroid indices)

But NOT the 256 × 1024 original embeddings. `semantic_codebook.rs` at
`crates/thinking-engine/examples/semantic_codebook.rs:170-175` writes only
the cosine matrix, not the vectors.

**Implication**: CLAM's API is `build(data: &[u8], vec_len: usize, ...)` —
it takes flattened vectors, not a distance matrix. So Probe M1 as originally
specified cannot even feed CLAM its input without first regenerating the
256 Jina v5 forward passes (~7 minutes of compute, once). Or reframing the
probe to use a clustering method that accepts distance matrices directly
(k-medoids / PAM / hierarchical agglomerative).

### F3. Four pairwise storage systems already exist in production — Slot P is novel

**Evidence**: Agent-verified with file:line references:
1. `crates/thinking-engine/src/bf16_engine.rs:21-100` — 256² i16 cosine table (128 KB)
2. `crates/thinking-engine/src/f32_engine.rs:260-356` — 4096×16 sparse top-K (512 KB)
3. `crates/lance-graph-planner/src/cache/kv_bundle.rs:39-102` — AttentionMatrix 64²/256² Base17 (139 KB / 2.23 MB)
4. `crates/bgz-tensor/src/attention.rs:34-170` — 256² AttentionTable + 256² ComposeTable (128 KB + 64 KB)

All four store **centroid-to-centroid** (aggregated), NOT per-weight-pair.
A per-weight Slot P would be novel but intractable at N=1M weights
(N² = 1 trillion pairs). Pairwise at the **centroid level** is well-supported;
pairwise at the **weight level** is neither precedented nor feasible.

**Implication**: The canonical pairwise rule from `two-basin-routing.md` is
about preserving centroid-level pairwise (which exists and is cheap), not
about creating per-weight pairwise (which is intractable and does not exist
in production).

---

## What got clarified

1. **The Slot D "16→256→4096 alignment" was aesthetic, not structural.**
   The alignment comes from `4^3 = 64` for 3 digits of 4-state or `16^3 = 4096`
   for 3 digits of 16-state — pure arithmetic, not from CLAM's natural output
   shape. CLAM gives you a binary tree; forcing 16-way through it requires a
   different clustering algorithm (k-medoids, agglomerative, spectral).

2. **"Bucketing > resolution" is the arxiv consensus, but NOT specifically
   about CLAM.** The claim (from PQ, LSH, RVQ, SqueezeLLM, GPTQ) is that any
   clean bucketing scheme dominates within-bucket precision for retrieval.
   CLAM is ONE possible bucketing. K-medoids, k-means, hierarchical, spectral
   are others. The probe should test whether 16-way bucketing at all works,
   not whether CLAM specifically works.

3. **The 256 Jina centroids may not cleanly partition into 16 sub-groups of 16.**
   This is the empirical question. It may turn out that 256 forward-pass
   semantic centroids partition naturally into a different shape (e.g., 8
   coarse clusters of varying sizes, or 32 tight clusters with long tails).
   The probe must measure, not assume.

---

## What remains uncertain

1. **Whether ANY clustering method produces a natural 16-way top level on
   the 256-point cosine matrix.** This is Probe M1's reframed question.
2. **Whether the embeddings (not just the cosine matrix) need to be
   regenerated to run the probe with higher-fidelity clustering methods.**
   k-medoids works on the distance matrix directly; CLAM needs vectors.
3. **Whether the Slot D layout survives the probe result.** If clustering
   naturally gives 8-way or 32-way, the 4-bit-per-level chunk should change.
4. **Whether the original "pairwise is sacred" rule (from `two-basin-routing.md`)
   is about centroid-level or weight-level pairwise.** The four existing
   systems are centroid-level, so the rule is automatically satisfied;
   weight-level pairwise is a separate (intractable) question.

---

## What should be tested next

### Reframed Probe M1: Hierarchical Clustering Fit on 256 Jina Centroids

**Written as**: `crates/thinking-engine/examples/probe_m1_bucket_fit.rs`

**Inputs**: `/tmp/codebooks/jina-v5-semantic-256/cosine_matrix_256x256.f32`
(already on disk, 262 KB)

**Methods compared (variants, per the user's "compare all variants" rule)**:
1. K-medoids (PAM) with k=16 on the full 256-point distance matrix
2. Agglomerative hierarchical clustering (average linkage) on the same matrix
3. Random k=16 partition (baseline — any real method must beat it)

**Per-method measurements**:
- Silhouette coefficient (>0.2 = meaningful, >0.5 = strong)
- Cluster size balance (max/min ratio; <3 = balanced)
- Within/between cluster distance ratio
- Natural depth of hierarchy (how many recursive splits before collapse)

**Pass**: Best method has silhouette > 0.2, balance < 3x, decisively beats
the random baseline, AND natural depth supports a 2–3 level recursion.

**Fail**: Best method has silhouette < 0.1 OR balance > 5x OR no natural
hierarchy beyond 1 level. In that case, the 16-way Slot D alignment is
unjustified and must be reframed.

**If Probe M1 PASSES**: promote the reframed Slot D layout (whatever
clustering algorithm wins) from CONJECTURE to FINDING in
`bf16-hhtl-terrain.md`. The original CLAM-based wording should be
corrected to name the actual clustering method.

**If Probe M1 FAILS**: Slot D layout must be rethought from scratch. The
fallback is "direct 256-state centroid tag = 1:1 Jina centroid index, no
hierarchy," which aligns with what the 4 existing pairwise systems already
do. This would be a simplification, not a regression.

### Depends on Probe M1 (do not run first)

- Probe I: 4 discrete γ-phase offsets producing different ranked outputs
- Probe M3: Bucket-only retrieval quality vs full BF16
- Probe V1: Per-task variant comparison
- Probe R1/R2: 1/40 σ lens + ring perturbation pipeline

---

## Implications for HEEL/HIP/BRANCH/TWIG/LEAF

The ripple architecture's hierarchy (from blackboard 01):
- HEEL = basin
- HIP = family
- BRANCH = contradiction / polarity split
- TWIG = local neighborhood
- LEAF = exact member

With the three P0 falsifications, these bind to actual storage as follows:

| Level | Original BF16-HHTL-D claim | After F1/F2/F3 correction |
|---|---|---|
| HEEL | 4 CLAM top-level clusters (bits 15..12) | 16 top-level clusters from WHATEVER clustering method wins Probe M1 |
| HIP | 256 Jina centroids (bits 11..8) | 256 Jina centroids — only if the clustering produces them as mid-level; else direct indexing |
| BRANCH | implicit in γ-phase flag | unchanged (polarity bit + γ-phase bucket) |
| TWIG | 4096 COCA terminal buckets (bits 7..4) | 4096 from clustering OR drop if 256 points can't produce 4096 leaves naturally |
| LEAF | BF16 value (Slot V) | unchanged |

The critical architectural question: **can HEEL and HIP be computed from a
256-point codebook at all, or do they need a larger underlying space
(e.g., 4096 Jina forward passes, which would take ~90 minutes)?**

Probe M1 answers the 256-point version. If it passes, HEEL/HIP are
structurally supported with the existing codebook. If it fails, we either
regenerate the codebook at 4096 or change the bit layout to match what
256 naturally produces.

---

## Open questions for blackboard readers

1. Is the Slot D 16→256→4096 alignment worth preserving if it requires
   generating 4096 Jina centroids (90-minute build cost)? Or is 8-way or
   direct 256-state better?
2. Should the clustering method be selected by Probe M1's empirical result,
   or pinned to CLAM for consistency with other codec-research work?
3. Does the ripple architecture want HIP to be 256-way (1:1 with Jina
   centroids, direct lookup) or to have a smaller HIP (e.g., 16-way) that
   clusters the 256 centroids into family groups?
4. Is `two-basin-routing.md § Pairwise Rule` about centroid-level pairwise
   (which is safe and well-supported) or weight-level pairwise (which is
   intractable)? This needs clarification in the doctrine.

---

## One-line doctrine

Three P0 assumptions were structurally wrong (CLAM binary-only, codebook
format is pairwise-not-vectors, weight-level pairwise is intractable);
Probe M1 must be reframed to test whether ANY hierarchical clustering method
produces a natural 16-way partition on the existing 256-point cosine matrix,
and the reframed probe is written as `probe_m1_bucket_fit.rs` per truth-architect
mandate before any further synthesis.
