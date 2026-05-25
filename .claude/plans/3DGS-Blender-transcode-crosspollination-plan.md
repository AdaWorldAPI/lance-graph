# 3DGS Blender Transcode Cross-Pollination Plan — lance-graph

## Goal

Add Blender as a third transcode corridor beside ArcGIS and Cesium.

```text
ArcGIS
  -> service semantics, GIS layers, enterprise spatial data

Cesium
  -> 3D Tiles, HLOD, implicit tiling, world streaming

Blender
  -> scene graph, mesh/material/light/camera assets, artist pipeline, asset conversion
```

The Blender path should not replace CAD/BIM or GIS. It should act as a bridge for scene assets, materials, meshes, cameras, animations, and test fixtures.

## Why Blender matters

Blender gives access to a mature authoring ecosystem:

```text
mesh objects
materials
textures
lights
cameras
animations
collections
modifiers
geometry nodes
imports/exports
```

For the AdaWorld stack, Blender can provide:

```text
asset conversion
scene normalization
material extraction
mesh-to-splat experiments
3DGS fixture authoring
CAD/BIM visual QA
artist-friendly debug loop
```

## Candidate Rust bridge layers

```text
blr
  Rust interface over Blender Python API through PyO3
  useful for automation and scripting inside Blender

blender-rs / custom readers
  possible native parsing or Blender-source-adjacent experimentation
  treat as exploratory until a concrete reader is stable

Python bridge
  acceptable as a conversion worker boundary
  avoid putting Python into the core runtime
```

## Non-goals

Do not make Blender the core runtime.

Do not make the geospatial or datalake substrate depend on Blender.

Do not treat Blender as AutoCAD-compatible CAD kernel.

Do not claim DWG/BRep/parametric CAD support from Blender import alone.

## Transcode path

```text
.blend / imported scene
        ->
Blender adapter
        ->
normalized scene graph
        ->
mesh/material/camera/light tables
        ->
optional 3DGS splat fitting or texture/radiance blocks
        ->
Lance/Arrow sidecar
        ->
Graph relationships
        ->
3D Tiles / glTF / native Ada scene export
```

## Scene graph mapping

```text
Blender Scene       -> Source
Collection          -> Group / Layer / SemanticCluster
Object              -> Feature / Asset / Renderable
Mesh                -> GeometryPayload
Material            -> MaterialNode
Image Texture       -> TexturePayload
Camera              -> Viewpoint / CapturePose
Light               -> IlluminationNode
Animation           -> TemporalTrack
Custom Property     -> Attribute / MetadataEdge
```

## Tables

Possible durable tables:

```text
blender_sources
blender_objects
blender_meshes
blender_materials
blender_textures
blender_cameras
blender_lights
blender_animation_tracks
blender_custom_properties
```

Common cross-domain tables should still be used for:

```text
features
contents
splat_blocks
certificates
decisions
provenance
```

## Cross-pollination with ArcGIS / Cesium

Blender contributes authoring and material detail.

Cesium contributes streaming hierarchy.

ArcGIS contributes geospatial semantics.

Combined path:

```text
Blender object/material scene
        +
Cesium 3D Tiles hierarchy
        +
ArcGIS feature/layer semantics
        ->
graph-native spatial scene
```

Examples:

```text
Blender material becomes tile/content style metadata.
Blender object hierarchy becomes feature graph.
Blender cameras become test traversal viewpoints.
Blender mesh bounds become tile bounding volumes.
Blender textures become 3DGS/radiance-field source evidence.
```

## CAD/BIM bridge

Blender is useful for visualization, conversion, and inspection, but CAD/BIM semantics should come from domain formats/adapters:

```text
IFC
STEP
glTF / USD
DWG via licensed/export bridge only
Blender scene assets
```

Graph model:

```text
BIMObject / CADObject
  -> GeometryPayload
  -> BlenderRenderable optional
  -> SplatRealitySkin optional
  -> Certificate
```

## 3DGS texture/radiance layer

The Blender transcode path can emit or consume 3DGS layers:

```text
mesh/material object
        ->
texture/radiance sampling
        ->
Gaussian splat block
        ->
material/feature link
```

This enables:

```text
reality-skinned CAD/BIM objects
material-aware splat rendering
scan-vs-model comparison
progressive viewport refinement
Blender-authored 3DGS fixtures
```

## Acceptance criteria

- Blender is documented as adapter/corridor, not core runtime.
- Scene objects, materials, cameras, and textures have clear graph mappings.
- Export path can target 3D Tiles / glTF / native Ada scene records.
- Blender-derived assets can link to 3DGS splat blocks and certificates.
- ArcGIS/Cesium/Blender responsibilities remain separate but composable.

## First demo

```text
one small Blender scene
  -> extract object hierarchy and materials
  -> write Lance/Arrow sidecar
  -> create feature graph
  -> attach one synthetic 3DGS texture/radiance block
  -> export glTF or tiny 3D Tiles tileset
  -> query object/material/splat relationship
```

## Wall sentence

```text
Blender is the workshop, Cesium is the world-tree, ArcGIS is the map bureaucracy, and LanceGraph is the memory of what every object means.
```
