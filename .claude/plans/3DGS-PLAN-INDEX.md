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

## Plans

1. `3DGS-Cesium-feature-mapping-plan.md`
   - Maps Cesium/3D Tiles concepts to Rust/Lance/Arrow/3DGS responsibilities.

2. `3DGS-3D-Tiles-runtime-plan.md`
   - Defines the 3D Tiles reader/writer/traversal/server crate trajectory.

3. `3DGS-Lance-Arrow-storage-plan.md`
   - Defines durable columnar schemas for tiles, contents, splats, features, and certificates.

4. `3DGS-integration-wiring-plan.md`
   - Defines cross-repo DTOs and call flow between `lance-graph` and `ndarray`.

5. `3DGS-ArcGIS-Cesium-ingestion-plan.md`
   - Defines ingestion from ArcGIS REST, 3D Tiles, glTF, SPZ, GeoJSON, and vector tiles.

6. `3DGS-certified-query-render-plan.md`
   - Defines the certified render/query decision layer.

7. `3DGS-epiphany-roadmap-plan.md`
   - Collects additional high-leverage directions discovered during planning.

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

Central principle: keep geospatial orchestration and numerical hot paths separate, but make their DTO boundary explicit and testable.
