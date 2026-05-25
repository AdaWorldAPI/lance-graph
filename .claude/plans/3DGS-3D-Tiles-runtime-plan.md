# 3DGS 3D Tiles Runtime Plan — lance-graph

## Goal

Build the Rust-side 3D Tiles runtime trajectory for 3DGS geospatial content.

This includes:

- reading 3D Tiles tilesets
- representing tile hierarchy and implicit tiling
- planning traversal and refinement
- serving compatible tile resources
- writing/exporting generated tilesets later

## Proposed crate split

```text
crates/ada-3dtiles
  DTOs for tileset.json, tile nodes, bounding volumes, content refs, metadata

crates/geo-tile-index
  TileId, Morton/Hilbert helpers, implicit tiling, availability bitstreams

crates/ada-3dtiles-selection
  camera model, screen-space error, traversal, skip-LOD, foveated policy

crates/ada-scene-server
  HTTP service for tileset.json, content payloads, metadata, and debug reports
```

Names are placeholders. Keep them workspace-local until the API stabilizes.

## Phase 1: explicit tileset reader

Support:

- `asset`
- `geometricError`
- `root`
- `children`
- `boundingVolume.region`
- `boundingVolume.box`
- `boundingVolume.sphere`
- `refine`
- `content.uri`
- `contents[]`
- extension passthrough
- extras passthrough

Output:

```rust
pub struct TilesetDocument {
    pub asset_version: String,
    pub root: TileNode,
    pub extensions_used: Vec<String>,
    pub extensions_required: Vec<String>,
}
```

## Phase 2: traversal

Implement deterministic traversal:

```text
camera
  -> frustum culling
  -> screen-space error
  -> refine / render decision
  -> request priority
  -> TileDecisionReport
```

Traversal policy knobs:

```rust
pub struct TraversalBudget {
    pub maximum_screen_space_error_px: f32,
    pub maximum_tiles_selected: usize,
    pub allow_skip_lod: bool,
    pub foveated_relaxation: Option<FoveatedRelaxation>,
    pub include_certificate: bool,
}
```

## Phase 3: implicit tiling

Support:

- subtree URI templates
- tile availability
- content availability
- child subtree availability
- constant availability
- bitstream availability
- Morton order mapping

Important: this phase should produce a small, well-tested availability API before integrating with traversal.

```rust
pub trait TileAvailability {
    fn tile_available(&self, tile: TileId) -> bool;
    fn content_available(&self, tile: TileId, content_set: u32) -> bool;
    fn child_subtree_available(&self, tile: TileId) -> bool;
}
```

## Phase 4: 3DGS content type

Represent 3DGS payloads as first-class content:

```rust
pub enum ContentKind {
    Gltf,
    B3dm,
    I3dm,
    Pnts,
    Composite,
    Subtree,
    GaussianSplat3d,
    Unknown(String),
}
```

Initial payload handling:

- pass-through glTF/SPZ/GLB resources
- local Lance/Arrow splat block resources
- metadata link to `contents` table

## Phase 5: scene server

Expose development endpoints:

```text
GET /3dgs/tileset.json
GET /3dgs/content/{content_id}
GET /3dgs/subtree/{level}/{x}/{y}/{z}.subtree
POST /3dgs/traverse
GET /3dgs/tile/{tile_id}/report
```

Do not make this production-authenticated initially. Keep it local/dev first.

## Phase 6: writer/exporter

Once reading and traversal are stable, add export:

```text
Lance/Arrow tile graph
        ->
tileset.json
        ->
implicit subtrees
        ->
content payload URIs
```

This enables compatibility with external viewers while keeping the internal runtime stronger.

## Acceptance criteria

- Read official/sample explicit 3D Tiles tilesets.
- Traverse a tile tree deterministically from camera input.
- Produce stable `TileDecisionReport` output.
- Support multiple contents per tile.
- Support unknown extensions without data loss.
- Do not require renderer availability to parse or traverse.

## Cross-repo hooks

`ada-3dtiles-selection` may call ndarray kernels for:

- SIMD frustum tests
- HHTL block scoring
- 3DGS splat block certification

But the traversal policy remains in `lance-graph`.
