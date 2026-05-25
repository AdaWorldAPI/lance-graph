# 3DGS Implementation Plan Index — lance-graph

This directory contains the lance-graph-side implementation plans for the 3DGS geospatial rebuild.

## lance-graph responsibility

`lance-graph` owns the orchestration, geospatial contracts, metadata, query layer, and service/runtime wiring:

- Cesium / 3D Tiles feature mapping.
- 3D Tiles reader/writer/server trajectory.
- ArcGIS and Cesium source ingestion.
- Lance/Arrow schemas for tiles, splats, features, metadata, and certificates.
- Graph/DataFusion/Cypher query interfaces.
- Integration wiring to `ndarray::hpc::splat3d`, `ndarray::hpc::pillar`, and HHTL kernels.
- Runtime selection policy, traversal, scheduling, and certified tile decisions.
- SplatShaderBlas / BLASGraph orchestration across bitpacked, palette, and 3DGS tiers.
- Cross-domain raw-field fanout into ultrasound, genetics, neuronal networks, and 4x4 cognitive-shader blocks.

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

## Geospatial 3DGS plans

```text
3DGS-Cesium-feature-mapping-plan.md
3DGS-3D-Tiles-runtime-plan.md
3DGS-Lance-Arrow-storage-plan.md
3DGS-integration-wiring-plan.md
3DGS-ArcGIS-Cesium-ingestion-plan.md
3DGS-certified-query-render-plan.md
3DGS-epiphany-roadmap-plan.md
3DGS-SplatShaderBlas-BLASGraph-crosspollination-plan.md
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
3D Tiles / ArcGIS / Cesium source
        ->
lance-graph ingest + Lance/Arrow metadata
        ->
traversal / query / tile decision planning
        ->
ndarray 3DGS SIMD + certification kernels
        ->
certified render/query decision report
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
