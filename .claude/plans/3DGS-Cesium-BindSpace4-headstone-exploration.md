# 3DGS Cesium / BindSpace4 Headstone Exploration — lance-graph

## Purpose

This document is a headstone exploration for the full line of thought connecting:

```text
Cesium / 3D Tiles
3DGS / Gaussian splats / EWA
SplatShaderBlas / BLASGraph
BindSpace4 / cognitive-shader-driver / (4x4)^4
PhiSpiral256 / Leaf Planetarium
BGZ17 recoverable sampling
CAM_PQ / PolarQuant lanes
Bevy runtime viewport
Blender transcode corridor
PR-X12 tensor-container expansion
```

The goal is to preserve the architectural synthesis before implementation details scatter it into separate plans.

## Capstone thesis

```text
Cesium streams the world-tree.
BindSpace4 streams the meaning, state, signal, time, and proof inside each branch.
PhiSpiral256 tells where the unexplained residual lives.
BGZ17 remembers how to recover it.
SplatShaderBlas / BLASGraph routes it at cache speed.
ndarray certifies the math.
Bevy makes it touchable.
```

## The two-layer architecture

### External envelope

Cesium / 3D Tiles supplies the external spatial contract:

```text
tileset.json
  root tile
    boundingVolume
    geometricError
    refine: ADD | REPLACE
    content / contents
    children / implicit tiling
```

This layer answers:

```text
what exists
where it is
how coarse or fine it is
whether content and children are available
how the world should stream progressively
```

### Internal stream

BindSpace4 supplies the internal SoA stream grammar:

```text
bind_source[]
bind_state[]
bind_signal[]
bind_time[]
bind_cert[]
```

This layer answers:

```text
what each tile/content/block means
what state or covariance it carries
what signal / opacity / confidence it emits
what time/provenance/phase it belongs to
what certificate or proof governs it
```

## Why Cesium alone is not enough

Cesium gives excellent world streaming, but the default runtime is still primarily visual and heuristic:

```text
screen-space error
bounding volumes
refinement mode
tile cache
```

The AdaWorld expansion adds:

```text
statistical certificates
query relevance
depth and occlusion uncertainty
3DGS covariance validity
PhiSpiral residual location
BGZ17 recoverability
CAM_PQ meaning
PolarQuant magnitude
BLASGraph batch routing
```

So the tile is no longer just a renderable unit. It becomes a certified spatial intelligence packet.

## The (4x4)^4 L1-L4 stream

Interpret `(4x4)^4` as a four-level block stream, not one dense monster matrix.

```text
L1: Mat4 local carrier
  local splat / residual atom / mesh proxy / feature-local state

L2: 4x4 block of carriers
  splat block / material block / object-local field block

L3: 4x4 block of blocks
  tile content group / subtree-local block family

L4: 4x4 super-block
  region / tileset / domain graph block
```

Cesium mapping:

```text
Cesium root / tileset
  -> L4 region or domain block

Cesium subtree / implicit tiling group
  -> L3 block-of-blocks

Cesium tile content
  -> L2 block of carriers

Cesium glTF / 3DGS / metadata payload
  -> L1 local carrier
```

## The 3x3 vs 4x4 rule

Do not destroy the 3DGS covariance spine.

```text
3x3 SPD covariance
  remains authoritative for spatial Gaussian covariance

4x4 carrier
  wraps transform, time, semantic role, provenance, signal, and certificate lanes
```

The 4x4 stream is an envelope and execution grammar. It does not replace the mathematically necessary 3x3 covariance unless the fourth lane has a defined meaning and passes certification.

## BindSpace4 lane mapping

Recommended lane semantics for geospatial/3DGS:

```text
bind_source[]
  tile_id / content_id / feature_id / local coordinate / splat id

bind_state[]
  covariance / transform / depth interval / material state / residual state

bind_signal[]
  opacity / color / confidence / magnitude / semantic score

bind_time[]
  capture time / LOD phase / animation frame / provenance epoch

bind_cert[]
  certificate id / error bound / pass-fail / reason code
```

Cross-domain reuse remains possible, but geospatial must not depend on experimental domains.

## How PhiSpiral256 fits

PhiSpiral256 is a separate orthogonal location lane.

It must not be mixed with CAM_PQ, PolarQuant, BGZ17, Palette256, or Fisher-z.

```text
CAM_PQ
  meaning / semantic basin

PolarQuant
  magnitude / similarity

PhiSpiral256
  orthogonal local residual location

BGZ17
  golden offset/stride recoverable sampling skeleton

Palette256
  candidate ranking / codebook layer where applicable

Fisher-z cosine
  optional statistical scorer/gate after candidate ranking
```

PhiSpiral256 atom:

```text
bits  0..=7   phi_spiral_id
bits  8..=11  mag4
bits 12..=13  BGZ offset family
bits 14..=15  BGZ stride family
```

This gives:

```text
256 spiral locations
× 16 magnitudes
× 4 offset families
× 4 stride families
= 65,536 recoverable local residual states
```

The important claim is not the count. The important claim is that each atom encodes where the unexplained residual lives.

## How BGZ17 fits

BGZ17 is the recoverable sampling skeleton:

```text
golden-ratio offset/stride sampling
Base17 compression
sparse recovery schedule
SIMD-friendly comparison kernels
```

In this architecture:

```text
BGZ17 supplies how to sample/recover.
PhiSpiral256 supplies where the residual lives.
CAM_PQ supplies what it means.
PolarQuant supplies how much.
```

## How SplatShaderBlas / BLASGraph fits

SplatShaderBlas gives the execution discipline:

```text
hot path must become one of:
  table lookup
  popcount
  palette/distance lookup
  sparse vector op
  BLAS / GEMM / SYRK
  EWA sandwich
```

For PhiSpiral256 and BindSpace4:

```text
single query
  route key -> O(1) candidate lookup

batch query
  frontier_vector[route_key]
    × transition_matrix[route_key -> candidate]
    -> candidate scores
```

The same cache-shaped style should govern 3DGS tile/block scheduling, residual routing, and datalake traversal.

## How 3DGS / EWA fits

3DGS supplies local field packet thinking:

```text
center
covariance
opacity
color
```

PhiSpiral atoms apply the same idea to unresolved residuals:

```text
local direction
magnitude
recovery schedule
confidence
```

A multi-atom leaf becomes a residual constellation:

```text
atom 0 -> geometry or center residual
atom 1 -> covariance / shape residual
atom 2 -> opacity / texture / signal residual
atom 3 -> contradiction / provenance / uncertainty residual
```

This is splat thinking without forcing every residual to become a visual splat.

## How render-depth certification fits

Cesium screen-space error should be extended with depth and certificate terms:

```text
screen-space error
+ depth uncertainty
+ ordering uncertainty
+ occlusion confidence
+ covariance validity
+ query relevance
+ residual-location pressure
```

A tile/block decision should be able to report:

```text
why it was skipped
why it was refined
why exact payload hydration was required
which certificate passed or failed
which lane triggered the decision
```

## How Blender fits

Blender is the scene/asset workshop:

```text
scene graph
mesh
material
camera
light
animation
texture
custom properties
```

It contributes authoring and conversion, not the durable core runtime.

Blender-derived objects can become:

```text
Feature / Asset / GeometryPayload / MaterialNode / TexturePayload
```

and can attach to:

```text
3DGS splat blocks
BindSpace4 stream blocks
certificates
PhiSpiral residual atoms
```

## How Bevy fits

Bevy is the Rust-native interactive viewport:

```text
lance-graph
  durable graph, tiles, features, queries, certificates
        ->
bevy
  ECS mirror, camera, picking, UI, runtime viewport
        ->
ndarray
  CPU-SIMD 3DGS / HHTL / render-depth / PhiSpiral kernels
        ->
bevy
  texture upload, overlays, interaction feedback
```

Bevy should show:

```text
certificate overlays
HHTL / tile refinement decisions
PhiSpiral residual constellations
depth uncertainty
query focus
feature provenance
```

## Unified data model

The durable graph should preserve this skeleton:

```text
Tileset
  -> Tile
  -> Content
  -> Feature / Asset
  -> FieldBlock / SplatBlock / BindSpace4Block
  -> PhiSpiralAtom / CAM_PQ / PolarQuant / BGZ17Schedule
  -> Certificate
  -> Decision
```

The hot path should use SoA tables and sidecars, not node-per-atom graph overhead.

## Suggested sidecar tables

```text
tiles
contents
features
field_blocks
bindspace4_lanes
splat_blocks
leaf_planetarium_atoms
leaf_planetarium_candidates
certificates
decisions
provenance
```

## First proof fixture

Start tiny:

```text
one tileset
one tile
one content payload
one synthetic 3DGS block
one BindSpace4 L1-L4 stream
one CAM_PQ meaning code
one PolarQuant magnitude code
one PhiSpiral atom packet
one BGZ17 offset/stride recovery schedule
one render-depth certificate
one tile decision report
```

Then show it in Bevy:

```text
image/preview
bounding box
depth uncertainty
PhiSpiral residual constellation
certificate overlay
reason codes
```

## Calibration gates

The holy-grail condition is not conceptual elegance. It is measurable:

```text
same or better correctness
fewer exact replays
smaller leaf payload
lower candidate fanout
cache-shaped routing
stable spiral occupancy
low wrong-high-confidence rate
```

Required comparisons:

```text
Mag4 only
PolarQuant only
BGZ17 L1 / weighted L1
BGZ17 sign agreement
PhiSpiral256 Euclidean
PhiSpiral256 Poincare/Mobius
PhiSpiral256 + Fisher-z gate
Hybrid: CAM_PQ + PolarQuant + PhiSpiral256 + BGZ17
```

## What this enables

```text
certified 3D Tiles with internal proof lanes
query-aware geospatial streaming
reality-skinned CAD/BIM
Blender-authored 3DGS fixtures
Bevy debug viewport
PhiSpiral leaf-location compression
datalake HHTL analogues
PR-X12 scene/tensor block expansion
cross-domain raw-field adapters
```

## Anti-patterns

Do not make Cesium understand every BindSpace4 lane.

Do not make PhiSpiral256 mean magnitude or semantics.

Do not replace 3x3 SPD covariance with 4x4 unless the fourth lane has a certified meaning.

Do not make Blender or Bevy durable sources of truth.

Do not implement every domain before the first tiny fixture proves the substrate.

Do not let certificates become decorative telemetry. Certificates must change skip/refine/hydrate/render/decode behavior.

## Implementation priority

```text
1. Keep Cesium/3D Tiles compatibility as external envelope.
2. Add BindSpace4 sidecar stream for tile/content internals.
3. Add PhiSpiral256 atom table beside CAM_PQ and PolarQuant lanes.
4. Use ndarray kernels for 3DGS, depth, PhiSpiral, and certificate math.
5. Use SplatShaderBlas / BLASGraph style tables for cache-shaped routing.
6. Show the first tiny fixture in Bevy.
7. Only then expand to Blender, datalake, PR-X12, and cross-domain adapters.
```

## Wall sentence

```text
Cesium streams the world-tree; BindSpace4 streams the living proof inside each branch; PhiSpiral256 marks where the unexplained difference lives.
```
