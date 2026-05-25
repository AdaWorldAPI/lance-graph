# 3DGS Cesium Feature Mapping Plan — lance-graph

## Goal

Map the useful Cesium / 3D Tiles feature set into a Rust-native 3DGS geospatial runtime architecture.

This plan is intentionally separate from implementation wiring. It answers:

```text
What should be borrowed from Cesium?
What should be translated into Lance/Arrow/Rust?
What should be outgrown?
```

## Borrow directly

### 1. Tileset hierarchy

Cesium / 3D Tiles concept:

```text
tileset.json
  root tile
    boundingVolume
    geometricError
    refine: ADD | REPLACE
    content / contents
    children / implicit tiling
```

Rust mapping:

```rust
pub struct TileNode {
    pub tile_id: TileId,
    pub bounding_volume: BoundingVolume,
    pub geometric_error_m: f32,
    pub refine_mode: RefineMode,
    pub content_refs: Vec<ContentRef>,
    pub metadata_ref: Option<MetadataRef>,
}
```

### 2. Screen-space error

Cesium concept:

```text
geometricError + camera distance + viewport -> screen-space error
```

Rust mapping:

```rust
pub struct ScreenSpaceErrorInput {
    pub geometric_error_m: f32,
    pub distance_m: f32,
    pub viewport_height_px: u32,
    pub fovy_rad: f32,
}
```

### 3. Refinement modes

Borrow `ADD` and `REPLACE` semantics.

```rust
pub enum RefineMode {
    Add,
    Replace,
}
```

### 4. Implicit tiling

Borrow:

- quadtree / octree addressing
- subtree availability
- tile availability bitstreams
- content availability bitstreams
- child-subtree availability
- Morton/Z-order locality

Rust mapping:

```text
crates/geo-tile-index
  tile_id.rs
  morton.rs
  implicit_subtree.rs
  availability.rs
```

### 5. Multiple contents per tile

Borrow the idea that one tile can carry multiple content payloads.

Runtime mapping:

```text
one tile
  -> mesh glTF
  -> 3DGS splat block
  -> metadata batch table
  -> raster/vector overlay references
```

## Translate, do not copy

### 1. JavaScript object graph -> Arrow/Lance tables

CesiumJS uses runtime object graphs. `lance-graph` should use columnar metadata tables:

```text
tiles
contents
features
subtrees
certificates
```

### 2. Styling language -> query predicates

Cesium styling should translate into:

- DataFusion SQL predicates
- Cypher predicates
- ontology-aware filters
- certified render masks

Example:

```sql
SELECT tile_id FROM features
WHERE class = 'building'
  AND confidence > 0.95
```

### 3. Runtime cache -> durable tile graph

Cesium cache decisions are runtime-only. `lance-graph` should persist useful summaries:

- last computed error certificate
- tile byte sizes
- feature density
- splat density
- covariance stability
- change-detection fingerprints

### 4. Renderer-driven selection -> certified decision planning

Cesium selects by visual error. `lance-graph` should plan with:

- screen-space error
- statistical certificate
- weak-dependence inflation
- quantization error
- covariance validity
- query relevance

## Outgrow

### 1. Browser-first rendering assumptions

Do not make WebGL/CesiumJS constraints the architecture ceiling.

Target:

```text
server/compiler mode
WGPU optional viewer mode
CPU preview/headless mode via ndarray
```

### 2. Pure visual tiles

Tiles should also be queryable objects:

```text
Tile -> Content -> Feature -> Asset -> Observation -> Certificate
```

### 3. Heuristic-only LOD

Every approximation decision should be able to carry a report:

```rust
pub struct TileDecisionReport {
    pub tile_id: TileId,
    pub action: TileAction,
    pub screen_space_error_px: f32,
    pub certified_error_px: Option<f32>,
    pub confidence: Option<f32>,
    pub reasons: Vec<TileDecisionReason>,
}
```

## Feature mapping table

| Cesium / 3D Tiles feature | lance-graph mapping | ndarray mapping |
|---|---|---|
| tileset.json | `ada-3dtiles` DTOs | none |
| bounding volumes | tile metadata + traversal | SIMD culling kernels |
| geometric error | tile column + SSE policy | error estimate kernel |
| implicit tiling | bitstream + Morton index | SIMD availability scan |
| multiple contents | content table | content-specific kernels |
| glTF payload | content ref / importer | optional splat conversion only |
| Gaussian splats | splat content schema | `hpc::splat3d` hot path |
| metadata | Lance/Arrow tables | certificate inputs |
| styling | SQL/Cypher/ontology filters | masks / block decisions |
| tile cache | scheduler + durable stats | no ownership |
| screen-space error | traversal policy | fast estimate kernels |
| foveated SSE | policy | approximate scoring kernel |
| skip LOD | policy | HHTL cascade support |

## Acceptance criteria

- Every borrowed Cesium concept has a Rust owner.
- No feature is assigned to both repos without a clear boundary.
- The plan supports 3D Tiles compatibility without requiring CesiumJS at runtime.
- The plan supports 3DGS as first-class content, not an afterthought.
