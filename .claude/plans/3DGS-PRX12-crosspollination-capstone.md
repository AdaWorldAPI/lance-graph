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
