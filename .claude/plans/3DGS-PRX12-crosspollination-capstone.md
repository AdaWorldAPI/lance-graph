# 3DGS / PR-X12 Cross-Pollination Capstone — lance-graph

## Purpose

Capture the wider expansion discovered by connecting the Cesium / maps / 3DGS exploration with the older PR-X12 codec canon.

The key realization:

```text
Cesium-style 3D tile traversal
PR-X12 Skip/Merge/Delta/Escape block grammar
GGUF/safetensors tensor compression
HHTL datalake traversal
SplatShaderBlas / BLASGraph execution
```

are not separate tracks. They are different faces of the same substrate:

```text
hierarchical local fields
  -> approximate block reconstruction
  -> certified skip / refine / hydrate / render / decode decisions
```

## Capstone thesis

```text
The maps path proves the traversal.
The codec path proves the block grammar.
The GGUF/safetensors path proves tensor deployment value.
The datalake path proves generality.
```

Together they define a universal execution pattern:

```text
structured source
  -> hierarchical block graph
  -> compressed/certified local kernels
  -> query/render/decode/hydrate schedule
  -> exact work only where the certificate demands it
```

## Cross-pollination map

| Source line | Core idea | What it contributes |
|---|---|---|
| Cesium / 3D Tiles | HLOD, tile hierarchy, implicit tiling, screen-space error | traversal grammar |
| ArcGIS | enterprise geospatial service model, FeatureServer/MapServer/Scene layers | data-source compatibility |
| 3DGS | anisotropic kernels, EWA projection, scene anchors | local-field reconstruction |
| PR-X12 x265 | Skip/Merge/Delta/Escape, BLAS/GEMM codec loops | compressed block grammar |
| PR-X12 x266 | EWA splat as `Basis<T>` | basis-swappable reconstruction |
| PR-X12 GGUF | tensor CTUs, activation-aware RDO, decode-during-GEMM | model deployment path |
| PR-X12 anti-neural | frozen lookup tables instead of NN hot loops | deterministic low-latency runtime |
| SplatShaderBlas | tiered bitpacked / palette / 3DGS execution | unified execution naming |
| Datalake HHTL | certified block pruning and hydration | non-visual generalization |
| JVET standards track (H.266/VVC → ECM → NNVC → H.267) | public beyond-VVC trajectory: ECM tool accretion, NN-in-loop, ≥40%-over-VVC requirement | external benchmark axis + the complexity/determinism counter-examples |

## Standards watch (added 2026-07-16)

The public trajectory the codec line must track — grounded with sources in
ndarray `.claude/knowledge/pr-x12-h266-h267-standards-landscape.md`:

```text
H.266/VVC  finalized 2020; ~40-50% over HEVC; decoder 1.5-2x, encoder ~10x
ECM-16.1   ~27% BD-rate over VTM-11 (RA), ~40% screen content;
           complexity industry-flagged impractical -> cautionary anchor
           for the codec-body LoC envelope (R-3)
NNVC v7    NN in-loop filters ~9% RA each -> the public ANTITHESIS of the
           anti-neural rule; determinism is our moat (evidentiary /
           medical / scientific video); mobile decoder-complexity
           pushback (Samsung, ITU 2025) is evidence for the inversion
H.267      requirement >=40% over VVC Main 10 at 4K+;
           CfP Jul 2026 -> submissions Nov 2026 -> evaluation Jan 2027
           -> finalize ~2028 -> deploy ~2034+
```

Naming rule: "x266" in workspace docs = the PR-X12 3DGS scene codec, never
H.266/VVC. **Internal codename since 2026-07-16: H.268** (INTERNAL ONLY,
never an ITU designation — H.267 itself is still prospective). blasgraph
remains the bit-exact canonical kernel home; bgz17 and siblings are lossy
adapters (ndarray audit, corrections applied 2026-07-16).
Watch dates: November 2026 (CfP submissions), January 2027 (evaluation).

Graded feasibility matrix (2026-07-16, adversarially verified with
receipts): ndarray `.claude/knowledge/pr-x12-h268-morton-wgpu-synergies.md`
— Morton-cascade / perturbation-pyramid / wgpu-wasm synergies, each claim
FEASIBLE-NOW / NEEDS-PROBE / OVERCLAIM-CORRECTED. Load-bearing for this
repo: bgz17's 256×256 distance table is texture-isomorphic (dense u16,
R16Uint-ready — PROBE-GPU-LUT is the parity gate); the shipped Morton 2bit
primitives (`FacetTier::morton`, symbiont `morton4`) prove the address
algebra but the ndarray CTU codec does not use it yet; D-PHASE/D-WHP stay
[H] probe-gated (J2 kill: dither-only).

**PROBE-GPU-LUT oracle spec (pinned 2026-07-16, codex P2 on #696 —
supersedes the looser "parity vs `batch_palette_distance`" wording here
and in the ndarray matrix doc's probe table):** bgz17 ships THREE distinct
conventions and a naive cross-comparison tests layout drift, not parity —
`PaletteDistanceTable` is **fixed-stride-256, raw `l1 as u16`**
(`palette.rs:77-88`); `batch_palette_distance` indexes a **compact k×k**
buffer (`row_offset = query*k`, `simd.rs:47-79`); `DistanceMatrix::build`
produces **compact k×k, SCALED** values (`d·65535/(17·65535)`,
`distance_matrix.rs:24-40`). The probe must pin ONE buffer + stride +
scale on BOTH sides: **arm A (256-stride)** — the upload buffer is built
through the PUBLIC accessor (`PaletteDistanceTable.table` is a private
field with no slice export — codex P2 on #697): the probe materializes
`buf[a*256 + b] = table.distance(a, b)` for all `(a, b)` in `0..256²`,
which is bit-identical to the private buffer by construction
(`distance` is the direct indexed read `table[a*256 + b]`,
`palette.rs:275-277`); upload `buf` as the R16Uint texture; CPU oracle =
the same `table.distance(q, c)` calls (equivalently
`batch_palette_distance(&buf, 256, …)`, valid because stride==k==256 with
zero-padding). If the probe PR prefers zero-copy, it adds a one-line
`pub fn as_slice(&self) -> &[u16]` to bgz17 in the SAME PR — never
assumes it exists. **arm B (compact)** — upload `DistanceMatrix.data` as
a k×k texture; CPU oracle = `batch_palette_distance(&dm.data, k, …)`.
Never mix arms: for k<256 the strides differ and raw-vs-scaled values
differ, so a mixed comparison is meaningless. Probe results that don't
name their arm are not trusted.

Addendum §7-§10 (2026-07-16): the ndarray matrix doc now carries the
comma-closure/constants correction, the 96-bit facet carving (48 CAM-PQ
+ 24 helix + 24 turbovec = the V3 12-byte payload), the kernel-shape
rule (turbovec 11.4× LUT-vs-GEMM receipt), and the replayable-tile
synergy set (H.268 × cognitive shaders) — all probe-gated; see
E-H268-REPLAYABLE-TILE-1.

## General substrate shape

```text
Source domain
  -> adapter
  -> block hierarchy
  -> local kernel summary
  -> certificate
  -> action
```

Actions should stay common:

```text
skip
keep approximate
refine
hydrate metadata
hydrate exact payload
render
decode
fallback
reject
```

## Maps as the first visible body

The maps path starts with:

```text
3D Tiles / Cesium / ArcGIS source
  -> tiles / contents / features / splat blocks
  -> Lance/Arrow sidecar
  -> ndarray 3DGS EWA/SYRK kernels
  -> certified tile decision
```

What it enables:

```text
offline or local-first 3D worlds
queryable digital twins
certified tile skipping/refinement
3DGS scene overlays
feature-aware rendering
change detection with error envelopes
```

But the same structure extends beyond maps.

## Datalake as the first non-visual body

The datalake path maps tiles to fragments:

```text
tile                 -> fragment / row group / page
bounding volume      -> stats / bloom / centroid / schema domain
screen-space error   -> query relevance / confidence error
refine               -> hydrate deeper metadata or exact rows
render               -> execute query / scan rows / return vectors
```

What it enables:

```text
metadata-only query planning
certified shard skipping
vector + SQL + graph hybrid traversal
EXPLAIN output with skip/refine reasons
partial hydration of exact rows only when needed
```

## GGUF / safetensors as the tensor body

The tensor-container path maps codec CTUs to model-weight blocks:

```text
video CTU
  -> tensor CTU
  -> activation-aware RDO
  -> basin/codebook pointer
  -> residual tail
  -> decode-during-GEMM
```

What it enables:

```text
GGUF Q_PRX12 experimental quant type
safetensors sidecar compression
Lance tensor chunk storage
partial model loading
cross-layer / cross-head Merge
model-weight lakehouse search
smaller distribution artifacts
```

## Anti-neural rule

The runtime discipline should be explicit:

```text
NNs may train tables.
NNs do not run in the hot loop.
```

Preferred inner-loop primitives:

```text
lookup table
popcount
palette distance
BLAS / GEMM / SYRK
EWA sandwich
rANS
HHTL traversal
```

This is what keeps the substrate deterministic, explainable, portable, and cheap.

## Unifying graph model

Represent all domains with the same graph skeleton:

```text
Source
  -> Frame / Tensor / Dataset / Tileset
  -> Block
  -> KernelSummary
  -> Certificate
  -> Decision
  -> Output / Payload
```

Examples:

```text
Tileset -> Tile -> SplatBlock -> Certificate -> RenderDecision
Dataset -> Fragment -> RowGroup -> Certificate -> HydrationDecision
GGUF -> Tensor -> TensorBlock -> Certificate -> DecodeDecision
Safetensors -> TensorShard -> TensorBlock -> Certificate -> LanceChunk
Ultrasound -> Frame -> PSFBlock -> Certificate -> FusionDecision
```

## What this enables by domain

```text
geospatial:
  certified 3D maps, digital twins, ArcGIS/Cesium interop

datalake:
  HHTL query planning, explainable skipping, vector/SQL/graph fusion

LLM deployment:
  GGUF/safetensors compression, decode-during-GEMM, partial loading

RAG:
  chunk-family traversal with provenance and retrieval confidence

observability:
  incident-root traversal over logs/traces/metrics/deploy events

robotics:
  sensor-field updates with map-change certificates

ultrasound:
  RF/IQ/Doppler frame fusion into certified splat volumes

genetics:
  motif/gene/pathway block traversal with uncertainty envelopes

neuronal/cognitive fields:
  activation/edge/time block comparison with 4x4 carrier summaries
```

## Implementation priority

Do not implement every domain. Stabilize the substrate through two hard bodies:

```text
1. 3DGS geospatial tile runtime
2. HHTL datalake traversal
```

Then attach tensor-container work:

```text
3. PR-X12 GGUF/safetensors tensor block prototype
4. Lance/Arrow tensor chunk storage
5. decode-during-GEMM microbench
```

Exploratory domains remain adapters until the core proves itself.

## Required references across plan set

This capstone should be kept linked with:

```text
3DGS-Cesium-feature-mapping-plan.md
3DGS-HHTL-datalake-traversal-plan.md
3DGS-SplatShaderBlas-BLASGraph-crosspollination-plan.md
3DGS-blast-radius-application-map.md
3DGS-domain-adapter-strategy-plan.md
3DGS-certified-query-render-plan.md
```

The ndarray sibling capstone is:

```text
PR-X12-tensor-container-expansion-capstone.md
```

## Acceptance criteria

- The plan set no longer frames 3DGS as only geospatial.
- PR-X12, GGUF/safetensors, and datalake traversal are explicitly connected.
- SplatShaderBlas / BLASGraph is treated as the execution bridge.
- Domain adapters remain separate from the core substrate.
- The first implementation path remains small and testable.

## Wall sentence

```text
Cesium gave us the world-tree, PR-X12 gave us the block grammar, GGUF/safetensors give us tensor deployment, and HHTL turns all of it into certified traversal.
```
