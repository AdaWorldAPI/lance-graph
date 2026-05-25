# 3DGS Lance/Arrow Storage Plan — lance-graph

## Goal

Define durable Lance/Arrow schemas for 3DGS geospatial content.

The storage model must support:

- 3D Tiles hierarchy
- implicit tiling metadata
- 3DGS splat block metadata
- feature and asset overlays
- error certificates
- query and traversal planning
- future ArcGIS/Cesium compatibility exports

## Core tables

### `tiles`

One row per tile node.

```text
tile_id: Utf8
parent_tile_id: Utf8?
level: UInt32
x: UInt64?
y: UInt64?
z: UInt64?
morton: UInt64?
refine_mode: Utf8
geometric_error_m: Float32
bounding_volume_type: Utf8
bounding_volume_values: FixedSizeList<Float64>
content_count: UInt32
metadata_id: Utf8?
```

### `contents`

One row per tile content payload.

```text
content_id: Utf8
tile_id: Utf8
content_kind: Utf8
uri: Utf8
media_type: Utf8?
byte_size: UInt64?
feature_count: UInt64?
splat_count: UInt64?
codec: Utf8?
certificate_id: Utf8?
```

### `subtrees`

For implicit tiling.

```text
subtree_id: Utf8
root_tile_id: Utf8
subtree_levels: UInt32
tile_availability_kind: Utf8
content_availability_kind: Utf8
child_subtree_availability_kind: Utf8
tile_availability_bytes: Binary?
content_availability_bytes: Binary?
child_subtree_availability_bytes: Binary?
```

### `splat_blocks`

One row per 3DGS block, not per splat.

```text
block_id: Utf8
content_id: Utf8
tile_id: Utf8
splat_count: UInt64
local_origin_x: Float64
local_origin_y: Float64
local_origin_z: Float64
bounds_min_x: Float32
bounds_min_y: Float32
bounds_min_z: Float32
bounds_max_x: Float32
bounds_max_y: Float32
bounds_max_z: Float32
codec: Utf8
position_error_max: Float32
position_error_rms: Float32
scale_error_max: Float32
opacity_error_max: Float32
covariance_min_eigen: Float32
covariance_max_eigen: Float32
```

### `features`

Queryable semantic/geospatial objects.

```text
feature_id: Utf8
tile_id: Utf8
content_id: Utf8?
block_id: Utf8?
class: Utf8?
name: Utf8?
asset_id: Utf8?
geometry_kind: Utf8?
bbox_min_x: Float64?
bbox_min_y: Float64?
bbox_min_z: Float64?
bbox_max_x: Float64?
bbox_max_y: Float64?
bbox_max_z: Float64?
attrs_json: Utf8?
```

### `certificates`

One row per certificate.

```text
certificate_id: Utf8
scope_kind: Utf8        # tile | content | splat_block | traversal_decision
scope_id: Utf8
geometric_error_px: Float32?
sampling_error: Float32?
covariance_error: Float32?
quantization_error: Float32?
dependence_inflation: Float32?
total_error_px: Float32?
confidence: Float32?
passed: Boolean
failure_reasons: List<Utf8>
source_pillars: List<Utf8>
created_by: Utf8
```

### `tile_decisions`

Optional runtime/debug table.

```text
decision_id: Utf8
camera_id: Utf8
tile_id: Utf8
action: Utf8
priority: Float32
screen_space_error_px: Float32
certified_error_px: Float32?
confidence: Float32?
reason_codes: List<Utf8>
```

## Design rules

- Store per-block splat metadata in Lance; do not store every splat as one row unless needed for debug.
- Store dense splat columns as payloads or nested column chunks, not as millions of tiny rows.
- Keep tile hierarchy queryable without loading payloads.
- Keep certificates separate from raw tile metadata so they can be recomputed/versioned.
- Use stable IDs for tiles, contents, blocks, and certificates.

## Query examples

```sql
SELECT tile_id, uri
FROM contents
WHERE content_kind = 'GaussianSplat3d'
  AND splat_count > 0;
```

```sql
SELECT tile_id
FROM certificates
WHERE scope_kind = 'tile'
  AND passed = true
  AND confidence >= 0.995
  AND total_error_px < 2.0;
```

```cypher
MATCH (t:Tile)-[:HAS_CONTENT]->(c:Content)
WHERE c.content_kind = 'GaussianSplat3d'
RETURN t.tile_id, c.uri
```

## Integration with existing lance-graph

Use DataFusion/Arrow for SQL scans and `lance-graph` graph abstractions for relationships:

```text
Tile - HAS_CONTENT -> Content
Tile - CHILD_OF -> Tile
Content - HAS_SPLAT_BLOCK -> SplatBlock
Tile - HAS_CERTIFICATE -> Certificate
Feature - LOCATED_IN -> Tile
```

## Acceptance criteria

- Schemas can be created from Rust code and written to Lance.
- Queries can select tiles without loading content payloads.
- Certificates can be versioned or recomputed without rewriting tile rows.
- The schema supports both explicit and implicit 3D Tiles.
- The schema supports 3DGS blocks as first-class content.
