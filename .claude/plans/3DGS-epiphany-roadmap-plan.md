# 3DGS Epiphany Roadmap Plan — lance-graph

## Goal

Collect high-leverage follow-up ideas discovered during the 3DGS planning pass.

This file is intentionally exploratory. Promote items into separate plans when they become implementation-ready.

## 1. CHLOD: Certified Hierarchical Level of Detail

Replace heuristic-only LOD with certified decisions.

```text
HLOD + error certificate + query relevance = CHLOD
```

Key idea:

- parent tile can be accepted only if error certificate passes
- child refinement is triggered by certified error, not just geometric error
- stored certificates allow repeatable render decisions

Potential crate:

```text
crates/geo-chlod
```

## 2. 3DGS change detection with error envelopes

Use certificates to distinguish real scene change from approximation artifact.

Pipeline:

```text
scan A
scan B
  -> align / normalize
  -> compare splat blocks and features
  -> subtract known error envelopes
  -> emit changed regions above confidence threshold
```

Useful for:

- construction progress
- infrastructure inspection
- forestry
- archaeology
- digital twins

## 3. Query-aware foveated traversal

Classic foveated rendering cares about camera gaze.

Query-aware traversal also cares about semantic focus:

```text
camera focus + query focus + certificate budget -> tile priority
```

Example:

- user selects pipe network
- renderer refines splats near pipe assets
- unrelated tiles stay coarse even if visually nearby

## 4. ArcGIS-compatible output mode

Goal: export enough 3D Tiles / FeatureServer-like resources to interoperate with GIS tools.

Do not reimplement full ArcGIS Server first.

Start with:

- static 3D Tiles export
- GeoJSON / vector overlay export
- simple FeatureServer-like query endpoint for local data

## 5. Lance-backed implicit subtree server

Use Lance tables as the source of truth for implicit tiling.

Endpoint generates subtree files from table rows and availability bitstreams.

Advantages:

- no giant JSON tile trees
- fast updates
- compact change sets
- queryable availability

## 6. 3DGS tile compiler

Build a compiler:

```text
raw splats / glTF / SPZ / photogrammetry output
        ->
local frames
        ->
HHTL partitioning
        ->
3D Tiles + Lance sidecar
        ->
certificates
```

This is likely the first impressive demo path.

## 7. Scientific/medical sibling path

The same certified 3DGS machinery may work for volumetric ultrasound / medical field rendering.

Shared concepts:

- splat-like anisotropic kernels
- covariance projection
- certified uncertainty
- queryable metadata
- CPU-first previews

Keep this as a sibling idea, not in the core geospatial path.

## 8. Tile intelligence as graph

Represent the world as:

```text
Tile -> Content -> SplatBlock -> Feature -> Asset -> Observation -> Certificate
```

This makes visual data queryable and auditable.

The viewer becomes a spatial intelligence engine, not just a renderer.

## 9. External compatibility envelope, internal columnar core

Recommended strategy:

```text
external envelope: 3D Tiles / glTF / ArcGIS-compatible outputs
internal core: Lance / Arrow / ndarray SIMD / certificates
```

This avoids fighting standards while still gaining performance and auditability.

## 10. First demo target

Do not start with the whole planet.

Start with:

```text
one small 3DGS scene
one tileset.json
one Lance sidecar
one camera traversal request
one certified decision report
one CPU projection preview
```

Then grow from there.

## Promotion criteria

Promote an epiphany into its own plan when:

- it has a clear crate owner
- it has a testable input/output boundary
- it can be built without blocking the base 3D Tiles runtime
- it improves either compatibility, performance, or auditability
