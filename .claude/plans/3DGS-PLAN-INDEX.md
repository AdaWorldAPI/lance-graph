# 3DGS Implementation Plan Index — lance-graph

This directory contains the lance-graph-side implementation plans for the 3DGS geospatial rebuild, the PR-X12 cross-pollination expansion, PhiSpiral256 SoA integration, the Cesium/BindSpace4 headstone exploration, the BGZ17/Poincare SparseEdgeField tectonic-map line, and the Palette256/3DSB/PhiSpiral attention-integration line.

## lance-graph responsibility

`lance-graph` owns the orchestration, geospatial contracts, metadata, query layer, and service/runtime wiring:

- Cesium / 3D Tiles feature mapping.
- 3D Tiles reader/writer/server trajectory.
- ArcGIS and Cesium source ingestion.
- Blender scene graph / mesh / material / camera transcode corridor.
- Cesium/BindSpace4 L1-L4 stream bridge as a headstone/capstone exploration.
- BGZ17/Poincare SparseEdgeField tectonic-map reconstruction as a sparse sidecar field.
- Palette256/3DSB/PhiSpiral attention replacement as an economic semantic-spatial context packet.
- Lance/Arrow schemas for tiles, splats, features, metadata, and certificates.
- Graph/DataFusion/Cypher query interfaces.
- Integration wiring to `ndarray::hpc::splat3d`, `ndarray::hpc::pillar`, and HHTL kernels.
- Runtime selection policy, traversal, scheduling, and certified tile decisions.
- SplatShaderBlas / BLASGraph orchestration across bitpacked, palette, and 3DGS tiers.
- Datalake HHTL traversal and domain adapter strategy.
- PhiSpiral256 SoA schema placement, route-key/candidate tables, and cross-lane calibration.
- Cross-domain raw-field fanout into ultrasound, genetics, neuronal networks, and 4x4 cognitive-shader blocks.
- PR-X12 cross-pollination across 3DGS, GGUF/safetensors, HHTL datalakes, and SplatShaderBlas.

## Markdown convention

Program-related material should use fenced Markdown blocks so Claude Code, GitHub review, and future handovers can parse it cleanly.

Use fences for:

```text
crate/module layouts
commands
Cargo feature sets
Rust DTO sketches
schema sketches
endpoint lists
call-flow diagrams
file paths when shown as groups
```

Use inline code only for short identifiers such as `lance-graph`, `TileId`, or `3DGS-certified-query-render-plan.md`.

## Geospatial / scene 3DGS plans

```text
3DGS-Cesium-feature-mapping-plan.md
3DGS-3D-Tiles-runtime-plan.md
3DGS-Lance-Arrow-storage-plan.md
3DGS-integration-wiring-plan.md
3DGS-ArcGIS-Cesium-ingestion-plan.md
3DGS-Blender-transcode-crosspollination-plan.md
3DGS-certified-query-render-plan.md
3DGS-epiphany-roadmap-plan.md
3DGS-SplatShaderBlas-BLASGraph-crosspollination-plan.md
3DGS-Cesium-BindSpace4-headstone-exploration.md
```

## Palette256 / 3DSB / PhiSpiral attention plans

```text
Palette256-3DSB-PhiSpiral-attention-integration-plan.md
```

This plan captures:

```text
Palette256 cosine one-dimensional attention replacement
3DSB / spatial-splat-block 4D local carrier
original Poincare leaf location
PhiSpiral256 orthogonal local residue
OGIT / OWL / DOLCE semantic inheritance
BGZ17 recoverable sparse traces
GGUF / PR-X12 implicit-context headers
CausalEdge64 mailbox trajectory candidates
```

## BGZ17 / Poincare sparse field plans

```text
BGZ17-Poincare-SparseEdgeField-tectonic-map.md
```

This plan captures:

```text
BGZ17 recoverable sampling
Poincare / PhiSpiral256 local direction
SparseEdgeField atoms
Gaussian / EWA field reconstruction
OGIT / OWL / DOLCE semantic inheritance
GGUF / PR-X12 implicit-context headers
CausalEdge64 trajectory candidates
```

## Blast-radius and datalake plans

```text
3DGS-HHTL-datalake-traversal-plan.md
3DGS-blast-radius-application-map.md
3DGS-domain-adapter-strategy-plan.md
3DGS-PRX12-crosspollination-capstone.md
```

## PhiSpiral256 SoA plans

```text
PhiSpiral256-SoA-cross-system-integration-plan.md
```

This plan keeps the lanes distinct:

```text
CAM_PQ       -> meaning / semantic basin lane
PolarQuant   -> magnitude / similarity lane
PhiSpiral256 -> orthogonal local residual location lane
BGZ17        -> golden offset/stride recoverable sampling skeleton
Fisher-z     -> optional statistical angular scorer/gate after candidate ranking
```

## Cross-domain fanout plans

```text
3DGS-cross-pollination-raw-field-plan.md
3DGS-ultrasound-SaMD-plan.md
3DGS-genetics-4x4-fanout-plan.md
3DGS-neuronal-network-4x4-plan.md
3DGS-4x4-cognitive-shader-integration-plan.md
```

## Cross-repo boundary

`lance-graph` should not own CPU SIMD renderer internals. It should call stable kernels from `ndarray`.

The intended flow is:

```text
3D Tiles / ArcGIS / Cesium / Blender source
        ->
lance-graph ingest + Lance/Arrow metadata
        ->
traversal / query / tile decision planning
        ->
ndarray 3DGS SIMD + certification kernels
        ->
certified render/query decision report
```

The Cesium/BindSpace4 internal stream flow is:

```text
Cesium / 3D Tiles external envelope
        ->
lance-graph tile/content/feature graph
        ->
BindSpace4 L1-L4 sidecar stream
        ->
ndarray 3DGS / depth / PhiSpiral / certificate kernels
        ->
certified tile, content, residual, or render decision
```

The Palette256/3DSB/PhiSpiral attention flow is:

```text
thought / tensor block / scene block / sparse edge field
        ->
OGIT / OWL / DOLCE inherited domain lookup
        ->
CAM_PQ meaning anchor
        ->
Palette256 cosine attention candidate
        ->
3DSB local field/block carrier
        ->
Poincare local chart and PhiSpiral256 residual location
        ->
BGZ17 offset/stride recovery family
        ->
CausalEdge64 trajectory candidate or certified hydrate decision
```

The BGZ17/Poincare SparseEdgeField flow is:

```text
field / tensor / scene / semantic map
        ->
BGZ17 sparse pressure traces
        ->
Poincare-local direction and PhiSpiral256 atom placement
        ->
SparseEdgeField sidecar atoms
        ->
Gaussian / EWA reconstruction or exact hydration only when certified
```

The BLAS-backed 3DGS orchestration flow is:

```text
SplatShaderBlas / BLASGraph tier selection
        ->
lance-graph tile/block schedule
        ->
ndarray EWA/SYRK/BLAS backend kernels
        ->
certified covariance / render decision report
```

The datalake HHTL flow is:

```text
SQL / graph / vector / semantic query
        ->
lance-graph metadata traversal over datasets/fragments/blocks
        ->
ndarray optional field-kernel scoring/certification kernels
        ->
certified skip / refine / hydrate decision report
```

The PhiSpiral256 SoA flow is:

```text
leaf / residual / route-key request
        ->
lance-graph CAM_PQ + PolarQuant + PhiSpiral256 + BGZ17 lane composition
        ->
ndarray PhiSpiral256 encode / neighbor / distance / calibration kernels
        ->
Lance/Arrow atom table + candidate table + calibration report
```

The PR-X12 tensor-container expansion flow is:

```text
GGUF / safetensors / Lance tensor source
        ->
lance-graph adapter + block graph + provenance
        ->
PR-X12 block schedule / SplatShaderBlas-style decision
        ->
ndarray decode-during-GEMM / codebook / BLAS kernels
        ->
certified tensor decode or hydration report
```

The 4x4 raw-field fanout flow is:

```text
raw field source
        ->
lance-graph provenance + graph/schema registration
        ->
4x4 cognitive-shader block request
        ->
ndarray Mat4 / Sym4 / Block4 SoA kernels
        ->
certified block decision report
```

Central principle: keep orchestration and numerical hot paths separate, but make their DTO boundary explicit and testable.
