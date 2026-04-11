# Probe M1 Result: 2026-04-11

## Status: **FAIL with k=4 surprise** (not a clean fail, needs math savant vetting)

## Command

```bash
cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
    --example probe_m1_bucket_fit
```

## Input

- `/tmp/codebooks/jina-v5-semantic-256/cosine_matrix_256x256.f32` (262 KB)
- Angular distance primary: `acos(clamp(cos, -1, 1)) / π ∈ [0, 0.5143]`, mean 0.2339

## Four-method panel at k=16

| Method | Silhouette | Balance (max/min) | W/B ratio | Cluster sizes |
|---|---|---|---|---|
| k-medoids PAM (median of 5 seeds) | 0.0758 | 7.60 | 0.4043 | [18,38,10,12,17,23,13,13,20,13,7,15,16,5,24,12] |
| Agglomerative (average linkage) | 0.5958 | **191.00** | 0.2356 | [2,**191**,26,19,1,1,2,1,4,1,3,1,1,1,1,1] |
| Binary CLAM at depth 4 | 0.2558 | 99.00 | 0.1911 | [7,4,7,4,60,99,17,10,5,3,2,1,2,4,24,7] |
| Random baseline (seed 0xC0FFEE) | -0.2388 | 2.62 | 0.9865 | [19,18,16,17,21,16,11,17,14,21,14,8,19,11,18,16] |

**Reading**: Agglomerative's silhouette 0.5958 is an **artifact of the 191/256 dominant cluster**, not real clustering. Binary CLAM also lands 99/256 in one leaf. PAM is the only method that distributes points across 16 clusters at all, but its silhouette is 0.0758 (fail).

## K-sweep on PAM (5-seed median protocol)

| k | silhouette | balance | within/between |
|---|---|---|---|
| **4** | **0.2049** | **1.86** | 0.4765 |
| 8 | 0.1274 | 2.08 | 0.5765 |
| **16** | **0.0758** | 7.20 | 0.3110 |
| 32 | 0.0318 | 15.00 | 0.3990 |
| 64 | 0.0105 | 12.00 | 0.3733 |

**Peak: k=4, monotonically decreasing for k ≥ 8.** The Jina v5 256-centroid
semantic codebook has a **natural 4-basin shape**, not a natural 16-family
shape. This is the surprise finding.

## Validity / reliability (Cronbach α + pairwise ARI)

### Cronbach's α over 4 methods × 256 per-point silhouettes

- α over all 4 methods (incl. random): **-0.2081**
- α over 3 real methods (excluding random): **0.4109**

Interpretation: α = 0.41 is **below 0.5 "questionable" threshold**. Methods do
not reliably agree on which points are well-clustered. Inter-method reliability
is not established at k=16.

### Pairwise Adjusted Rand Index (convergent validity)

| method A | method B | ARI |
|---|---|---|
| PAM | Agglomerative | 0.0725 |
| PAM | binary-CLAM-d4 | 0.2237 |
| PAM | random | 0.0014 |
| Agglomerative | binary-CLAM-d4 | **0.3290** |
| Agglomerative | random | 0.0005 |
| binary-CLAM-d4 | random | 0.0031 |

- Mean ARI (real pairs): **0.2084** (at lower edge of ripple-architect's [0.3, 0.7] partial-agreement band)
- Variance: 0.0166

**Pattern**: The two hierarchical methods (Agg + binCLAM) agree most (0.33);
PAM and Agg disagree (0.07). This is not random — it's **method-family
dependent structure**. Medoid-based and link-based methods see the manifold
differently.

### Creative agent verification (ripple-architect's multi-sieve claim)

**NOT SUPPORTED at k=16** (α 0.41 < 0.5, ARI 0.21 just below [0.3, 0.7] band).

But not cleanly rejected either — the pattern of disagreement is informative.
The call should come from a higher-altitude test (Probe M1-multi with 5 sieves
including the 40-ring 1/40σ lens), not from this bucket-fit probe alone.

## Cross-lens consensus matrix

| Lens | Value | Pass? |
|---|---|---|
| silhouette > 0.2 | 0.5958 (Agg) | ✓ (artifact of 191/256 imbalance) |
| balance < 3.0 | 7.60 (best real) | ✗ |
| absolute gap > 0.15 | 0.8347 (Agg) | ✓ (artifact of same) |
| k=16 in k-sweep peak band (Δ<0.05) | peak=4, k=16 sil=0.076 | ✗ |
| Cronbach α (real methods) > 0.5 | 0.4109 | ✗ |
| Mean ARI in [0.2, 0.85] | 0.2084 | ✓ (at lower edge) |
| All real methods agree on direction | true | ✓ (all FAIL) |

**Raw count: 4/7 pass.**
**Honest count (discounting Agg artifacts): 2/7 pass.**

## Per-method verdicts

- **PAM (median)**: silhouette=0.0758 fail, balance=7.60 fail, gap=0.3147 pass → FAIL
- **Agglomerative**: silhouette=0.5958 pass (artifact), balance=191 fail, gap=0.8347 pass (artifact) → FAIL
- **Binary CLAM d=4**: silhouette=0.2558 pass, balance=99 fail, gap=0.4946 pass → FAIL

**All three real methods fail, but for DIFFERENT reasons.** This is the
multi-lens tension the user warned about.

## Verdict

### RESULT: FAIL on 16-way (at k=16), with k=4 natural peak as informative surprise

**What is empirically rejected:**
1. Slot D's 16-way top-level nibble allocation (no clustering structure at k=16)
2. Uniform 16-way family partition on the 256 Jina centroids
3. Ripple-architect's multi-sieve partial-agreement band at k=16

**What is empirically suggested (k=4 finding):**
1. The Jina semantic space has 4 natural basins (PAM k=4: silhouette 0.2049, balance 1.86)
2. family-codec-smith's **HEEL = basin (2-4 states)** claim is ACCIDENTALLY VALIDATED — k=4 is exactly the HEEL layer count
3. The manifold is highly unbalanced: a dense core (191 points in agglomerative's big cluster) + ~65 outliers + a small number of mid-tier clusters

**What remains uncertain and needs math-savant review:**
1. Whether PAM's k=4 peak is real or an artifact of medoid initialization
2. Whether agglomerative's 191/256 is correct or a bug in the average-linkage implementation
3. Whether Cronbach's α on per-point silhouette vectors is the right reliability metric here (vs. Fleiss' κ, Krippendorff's α, or Normalized Mutual Information)
4. Whether the ripple-architect band [0.3, 0.7] is principled or ad-hoc
5. Whether silhouette on unit-sphere angular distance has the correct mathematical behavior

## Canonical fallback (corrected per findings)

Family-codec-smith's original fallback was "direct 256-state Jina tag, no hierarchy."
**This needs correction:** the k=4 finding says there IS a hierarchy, just not 16-way.

Corrected Slot D layout:
```
bits 15..14 = HEEL basin (2 bits, 4 states — validated by PAM k=4)
bits 13..6  = HIP Jina centroid (8 bits, 256 states, 1:1 direct)
bit     5   = BRANCH polarity
bits  4..0  = reserved (5 bits: γ-phase bucket, flags)
```

This is a **different fallback** from both the original 16→256→4096 claim AND
from the "direct 256 tag, no hierarchy" option. It threads the k=4 surprise
into the architecture.

## Next action

1. Do NOT promote Slot D conjecture to FINDING on this result — it's a FAIL
   with an informative surprise, not a clean pass/fail.
2. Dispatch math savant (MIT-style) to vet the algorithms (PAM, agglomerative,
   binary CLAM, silhouette, Cronbach α, ARI) before updating canonical knowledge.
3. Update `.claude/knowledge/bf16-hhtl-terrain.md § Probe Queue` with status
   **M1: FAIL (k=16 rejected); k=4 basin finding emerged — awaits math savant vet**.
4. If math savant confirms the algorithms are correct, plan Probe M1-multi
   (ripple-architect's 5-sieve version with the 40-ring 1/40σ lens as sieve #5)
   to test whether the multi-resolution manifold claim holds at the right altitude.
5. If math savant finds bugs, fix and re-run BEFORE touching canonical knowledge.

---

## Follow-up: Isotropy correction probe (item 13, 2026-04-11)

Probe `probe_isotropy_correction.rs` in `crates/deepnsm/examples/` applied
per-row mean centering (the simplest γ+φ Regime A operation) to the 4096²
DeepNSM distance matrix and re-measured the eigenspectrum.

Implementation uses `ndarray::simd::F32x16` via explicit `mul_add` (the
fused multiply-add primitive preferred across the workspace over plain
Sub). The identity `v - m = v * 1 + (-m)` maps directly to VFMADD213PS
on AVX-FMA or VDPBF16PS equivalent via the LazyLock CPU dispatch in
`ndarray/src/simd.rs` (routes to `simd_avx512.rs` / `simd_avx2.rs` /
`simd_amx.rs` based on hardware). No SIMD optimization code touched —
only the exported `F32x16::mul_add` primitive used.

### Results

| metric | baseline (raw u8 matrix) | corrected (per-row mean-centered) | delta |
|---|---|---|---|
| Participation ratio | 1.53 / 4096 | **9.56 / 4096** | **+8.03 (6.2×)** |
| Top-1 eigenvalue share | 80.66% | **28.12%** | **−52.54 pp** |
| Row-sum CV | 0.3087 | ≈0 (genuine centering) | — |
| Top-5 eigenvalues (raw)  | [3348.6, 229.9, 75.4, 46.2, 21.4] | — | — |
| Top-5 eigenvalues (ctr)  | — | [65001.7, 29759.5, 19122.2, 5675.0, 4389.2] | — |

### Verdict (per probe's own criteria): split — FAIL on PR, PASS on top-1 share

- PR 9.56 < 10 (probe's MARGINAL floor) → FAIL on the PR axis
- Top-1 share 28.12% < 30% (probe's PASS threshold) → PASS on the dominance axis
- **Dominant-axis problem is substantially fixed, but the underlying manifold
  has a long-tail spectrum** that keeps effective rank low even after the
  dominant axis is removed.

### What this means for the architecture

**γ+φ Regime A direction is validated empirically** (first positive signal
this session on the architectural question):
- Removing the dominant axis with the simplest possible Regime A op
  (per-row mean centering) drops top-1 share by 52 percentage points.
- This is the HDR-TV-style normalization that `bf16-hhtl-terrain.md § C3`
  (corrected to three regimes in commit f0429e5) describes, now measured
  as genuinely effective on real data.

**BUT the DeepNSM codebook is long-tail, not clustered**:
- Even after the dominant axis is removed, the eigenspectrum decays slowly
- Top-2 is 12.87% of the new variance — no clean break into "16 clusters"
- The codebook is a continuous distribution with a long tail of weak
  signal modes, not a collection of discrete clusters that could be
  labeled HEEL/HIP/TWIG

**Implication for Slot D**: the earlier probe `probe_m1_bucket_fit` was
measuring the wrong question. With isotropy correction applied, the right
question is NOT "does 16-way clustering fit?" but "is a **rank-based**
encoding (e.g., direct 4096-state Jina centroid ID with no hierarchy)
the natural match for this long-tail spectrum?" A flat 12-bit or 13-bit
direct centroid ID without hierarchical structure may be the correct
Slot D shape, not 4-bit nibbles.

### What to do next (pending user go)

1. **Run the full γ+φ Regime A pipeline** (`gamma_encode` + `phi_encode`
   in `bgz-tensor/src/gamma_phi.rs`) instead of just mean centering.
   Follow-up probe should live in `bgz-tensor/examples/` where gamma_phi
   is native. Measure whether PR improves from 9.56 further (e.g., to
   the 20-50 range).

2. **Update `bf16-hhtl-terrain.md § C2`** (bucketing > resolution) with a
   caveat: if the source codebook is long-tail after isotropy correction,
   direct rank encoding may beat bucket encoding. Canonical knowledge
   update pending user authorization.

3. **Revisit Slot D layout in `bf16-hhtl-terrain.md`** with the long-tail
   finding: 4-bit HEEL / 4-bit HIP / 4-bit TWIG may be the wrong structure
   for this data. A flat direct centroid ID might be the natural match.
   Also pending user authorization.

4. **DeepNSM 4096 codebook generation pipeline** (item 10): once Jina v5
   / Reader-LM v3 is added to the release manifest, the bake pipeline
   should apply γ+φ Regime A calibration BEFORE palette construction.
   The long-tail nature means we should NOT expect 16-way clustering to
   emerge from better isotropy correction alone — the palette should be
   sized and shaped to match long-tail structure.

### Files / runs

- Probe file: `crates/deepnsm/examples/probe_isotropy_correction.rs` (new, uses F32x16::mul_add)
- Cargo.toml: `crates/deepnsm/Cargo.toml` — ndarray moved to `[dependencies]` as mandatory per workspace rule ("zero external deps" preserved — ndarray is a path dep to the AdaWorldAPI fork, same-binary compile)
- Output dump: `/tmp/deepnsm_distance_4096x4096_centered.f32` (67 MB, transient)
- Python follow-up: eigenspectrum compared via numpy.linalg.eigvalsh — numbers already verified (see table above)

No canonical knowledge files modified in this commit. The C2/C3 proposals
above need explicit authorization before they ship.
