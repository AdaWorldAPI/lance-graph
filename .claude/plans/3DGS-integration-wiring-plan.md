# 3DGS Integration Wiring Plan — lance-graph

## Goal

Define the boundary between `lance-graph` and `ndarray` for the 3DGS geospatial rebuild.

This plan covers wiring and trajectory. The Cesium feature map is documented separately.

## Ownership

`lance-graph` owns geospatial orchestration:

- 3D Tiles and ArcGIS ingestion
- tile hierarchy and metadata
- Lance/Arrow storage
- query planning
- traversal policy
- service/runtime endpoints
- tile decision aggregation

`ndarray` owns numerical hot paths:

- CPU-SIMD 3DGS projection
- HHTL block scoring
- covariance math
- splat/block-level certification
- codec kernels

## Main flow

```text
camera request
  -> candidate tiles from Lance/Arrow
  -> base screen-space error
  -> candidate 3DGS contents
  -> ndarray block view
  -> ndarray HHTL / splat3d / certificate kernels
  -> merged tile decision report
  -> render/query schedule
```

## Boundary objects

Keep the boundary stable:

- camera DTO
- tile candidate DTO
- splat block metadata DTO
- render budget DTO
- decision report DTO
- certificate summary DTO

Do not leak renderer internals into graph query code.

## Integration options

### Prototype path

Use direct local dependency on `ndarray` with the needed features:

```text
std, linalg, splat3d, pillar
```

This is the fastest path for experiments and integration tests.

### Stabilized path

Extract a small DTO-only contract crate once the boundary is stable:

```text
crates/geo-3dgs-contract
```

This crate should contain only serializable request/response shapes and reason codes.

## Decision actions

The traversal layer should produce one of these actions:

- reject
- keep coarse
- refine
- load content
- project exact
- render exact

Every action must include reason codes. Example reasons:

- outside frustum
- below error budget
- above error budget
- certificate passed
- certificate failed
- covariance invalid
- weak-dependence inflation high
- query relevant
- query irrelevant
- cache budget exceeded

## Development endpoints

Initial local-only surface:

```text
POST traverse
POST certify block
POST project block
GET tile decision
GET certificate
```

These endpoints are for development and debugging, not the final public API.

## Tests

Unit tests:

- DTO conversion
- camera transforms
- tile candidate selection
- reason-code stability

Integration tests:

- tiny tile tree to traversal to ndarray HHTL to decision report
- tiny splat block to projection report to certificate row
- query predicate to selected tile set to traversal schedule

## Acceptance criteria

- `lance-graph` can call ndarray kernels without coupling query code to renderer internals.
- Traversal can run metadata-only when renderer kernels are unavailable.
- Traversal can include ndarray certificates when enabled.
- Decisions are deterministic for the same camera and tile metadata.
- Every decision has machine-readable reasons.

## Migration path

1. Prototype direct dependency.
2. Stabilize DTOs.
3. Add integration tests.
4. Extract contract crate if the boundary becomes useful outside the workspace.
5. Add local service endpoints.
6. Persist certificates and decisions.
