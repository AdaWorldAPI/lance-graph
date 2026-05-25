# 3DGS HHTL Datalake Traversal Plan — lance-graph

## Goal

Generalize the 3DGS / Cesium tile traversal pattern into a datalake traversal engine.

The core insight:

```text
3D Tiles traversal
  selects visual tiles by camera, bounds, and error

Datalake HHTL traversal
  selects data blocks by query, metadata, distribution, and certified error
```

This is likely the most immediate non-geospatial application of the 3DGS plan family.

## Mapping

```text
3DGS / Cesium concept       Datalake / query concept

Tile                        Lance fragment / Parquet row group / Iceberg file
Bounding volume             min/max stats / bloom / centroid / schema domain
Geometric error             approximation error / sketch error / stale-summary error
Screen-space error           query relevance / confidence error / hydration priority
Camera frustum              SQL predicate / Cypher pattern / semantic query focus
Splat covariance            distribution spread / embedding uncertainty / local dependency
Refine                      hydrate deeper metadata or exact rows
Skip                        certified prune
Render                      return rows / aggregates / graph edges / vectors
Tile decision report         query planning decision report
```

## HHTL traversal stages

```text
HEEL
  global dataset / table / partition pruning
  cheap filters: schema, partition key, time range, tenant, file stats

HIP
  row-group / fragment / family selection
  cheap semantic/vector/codebook basin checks

TWIG
  local sketches, centroids, histograms, covariance, bloom filters
  query-aware error estimates

LEAF
  exact rows, exact graph edges, exact vectors, exact payload hydration
```

## Proposed crate/module shape

```text
crates/lance-hhtl-traversal/
  mod.rs
  request.rs
  budget.rs
  candidate.rs
  metadata_scan.rs
  hhtl_decision.rs
  certificate.rs
  datafusion_bridge.rs
  graph_bridge.rs
  vector_bridge.rs
```

Names are placeholders. Keep workspace-local until stable.

## Request shape

```rust
pub struct DatalakeHhtlRequest {
    pub query_text: Option<String>,
    pub sql_predicate: Option<String>,
    pub graph_pattern: Option<String>,
    pub vector_query_ref: Option<String>,
    pub dataset_refs: Vec<String>,
    pub budget: DatalakeHhtlBudget,
}

pub struct DatalakeHhtlBudget {
    pub max_blocks_to_hydrate: usize,
    pub max_estimated_error: f32,
    pub min_confidence: f32,
    pub allow_approximate: bool,
    pub require_certificate: bool,
}
```

## Candidate metadata

Each candidate block should expose a common metadata surface:

```rust
pub struct DataBlockCandidate {
    pub block_id: String,
    pub dataset_id: String,
    pub physical_ref: String,
    pub row_count: u64,
    pub byte_size: u64,
    pub stats_ref: Option<String>,
    pub vector_centroid_ref: Option<String>,
    pub schema_fingerprint: Option<u64>,
    pub freshness_epoch: Option<u64>,
}
```

## Decision report

```rust
pub enum DatalakeHhtlAction {
    Skip,
    KeepApproximate,
    HydrateMetadata,
    HydrateExactRows,
    HydrateVectors,
    HydrateGraphEdges,
}

pub struct DatalakeHhtlDecision {
    pub block_id: String,
    pub action: DatalakeHhtlAction,
    pub priority: f32,
    pub estimated_error: f32,
    pub confidence: f32,
    pub certificate_id: Option<String>,
    pub reason_codes: Vec<String>,
}
```

## Certificates

A datalake traversal certificate should answer:

```text
Why was this block skipped?
Why was this block hydrated?
Which approximation was used?
What error/confidence envelope justified the decision?
Which metadata version was used?
```

Potential certificate components:

```text
partition compatibility
min/max predicate proof
bloom filter proof
centroid distance margin
covariance/sketch uncertainty
freshness/version proof
query relevance score
```

## Integration with DataFusion

Potential integration points:

```text
logical plan analysis
physical plan pruning
custom table provider
custom execution node
EXPLAIN output extension
```

The first useful path is probably an offline planner wrapper:

```text
SQL/query request
  -> HHTL traversal preplanner
  -> selected fragments
  -> DataFusion scan over selected fragments only
```

## Integration with lance-graph

Graph relationship model:

```text
Dataset -> Fragment -> RowGroup -> Page -> Row
Dataset -> BlockCertificate
Fragment -> Feature / VectorCentroid / SchemaFingerprint
Query -> DecisionReport -> Fragment
```

## First demo

Keep the first demo small:

```text
one Lance/Parquet dataset
one query predicate
one metadata table
one HHTL decision report
one exact DataFusion scan over selected blocks
one explanation: why skipped / hydrated
```

## Acceptance criteria

- Metadata-only traversal can select or skip blocks without reading payload rows.
- Every skip/refine decision has machine-readable reason codes.
- Approximate decisions include an error/confidence envelope.
- Exact hydration is still available as a fallback.
- The traversal can run before DataFusion execution.
- The system can explain why a block was skipped.

## Product framing

This turns lakehouse access into hierarchical certified traversal:

```text
Do not scan the lake.
Traverse it.
```
