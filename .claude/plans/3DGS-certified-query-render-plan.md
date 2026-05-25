# 3DGS Certified Query and Render Plan — lance-graph

## Goal

Make 3DGS rendering and querying decisions auditable.

A traversal decision should not only say what to render. It should also say why the approximation is acceptable or why refinement is required.

## Core idea

```text
visual error
+ sampling error
+ covariance error
+ quantization error
+ weak-dependence inflation
+ query relevance
= certified decision
```

## Decision scopes

Certificates can apply to:

- tile
- content payload
- splat block
- feature overlay
- traversal decision
- query result

## Decision report

```text
tile_id
content_id
block_id optional
action
priority
screen_space_error_px
certified_error_px optional
confidence optional
reason_codes
source_certificate_ids
```

## Runtime stages

### Stage 1: metadata-only decision

Uses only tile metadata:

- bounding volume
- geometric error
- camera distance
- content type
- content size
- existing stored certificates

### Stage 2: block preflight decision

Loads block stats but not full splat payload:

- splat count
- block bounds
- covariance eigenvalue range
- quantization error stats
- opacity/density estimates

### Stage 3: exact ndarray decision

Calls ndarray kernels:

- HHTL block scoring
- 3DGS projection report
- splat/block certificate

### Stage 4: persisted decision

Writes optional `tile_decisions` row for debug or repeatability.

## Query integration

A query should be able to influence traversal.

Examples:

- render only tiles containing buildings
- refine tiles containing assets changed after a date
- reject splat blocks below confidence threshold
- prioritize tiles near selected feature IDs

SQL shape:

```sql
SELECT tile_id
FROM features
WHERE class = 'building'
```

Graph shape:

```cypher
MATCH (t:Tile)-[:HAS_CONTENT]->(c:Content)
WHERE c.content_kind = 'GaussianSplat3d'
RETURN t.tile_id
```

## Certificate aggregation

Block-level certificates come from ndarray.

Tile-level certificates aggregate:

- max block error
- weighted mean block error
- minimum confidence
- failure reason union
- query relevance score

Aggregation should be deterministic.

## Policy knobs

```text
maximum_screen_space_error_px
maximum_certified_error_px
minimum_confidence
allow_uncertified_tiles
allow_metadata_only_decisions
query_relevance_weight
motion_relaxation_factor
foveated_relaxation_factor
```

## Failure policy

When certification fails, choose one:

- reject tile
- refine tile
- render coarse fallback
- render with warning report
- require exact projection

The policy should be explicit in the traversal budget.

## Acceptance criteria

- Every traversal decision has reason codes.
- Metadata-only traversal works without loading splat payloads.
- Exact mode can call ndarray certification kernels.
- Stored certificates can be reused.
- Query predicates can influence tile priority.
- Decision reports can be persisted and replayed.

## Product implication

This creates a runtime where visual approximation is not a hidden heuristic. It is inspectable, repeatable, and query-aware.
