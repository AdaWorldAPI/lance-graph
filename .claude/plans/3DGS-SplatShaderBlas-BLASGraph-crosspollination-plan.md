# 3DGS SplatShaderBlas / BLASGraph Cross-Pollination Plan — lance-graph

## Goal

Cross-pollinate the new geospatial 3DGS plans with the existing SplatShaderBlas, EWA-Sandwich, BLASGraph, and PR-X12 codec canon.

The older line already established:

```text
splat deposition
  -> EWA sandwich propagation
  -> L1-L4 spatial BLAS framing
  -> SplatShaderBlas naming
  -> BLASGraph / tropical-GEMM / bgz17 scalar sparse substrate
```

The new 3DGS plans should reuse this spine instead of inventing an unrelated renderer-only path.

## Existing canon to preserve

Prior work distinguishes two tiers:

```text
SplatShaderBlas-Bitpacked
  storage: AwarenessPlane16K = [u64; 256]
  operation: popcount, AND/OR-popcount
  workloads: set membership, Jaccard, Adamic-Adar, triangle, LPA, Louvain

SplatShaderBlas-Palette
  storage: BGZ17 palette + distance table
  operation: D[palette_a[i]][palette_b[j]] lookup
  workloads: continuous metric similarity, CAM-PQ-like fields
```

The geospatial 3DGS path adds a third sibling tier:

```text
SplatShaderBlas-3DGS
  storage: Lance/Arrow 3DGS blocks + ndarray SoA views
  operation: EWA sandwich, tile/block traversal, certificate aggregation
  workloads: maps, ultrasound, raw-field kernels, certified visualization
```

## BLASGraph correction (re-corrected 2026-07-16 per the ndarray audit)

The earlier version of this section had the canon/adapter relationship
**inverted** — it presented bgz17 as the current kernel home and blasgraph
as "the future abstraction name." Per the ndarray `PR-X12-docs-audit.md`
(ground truths #3/#5, corrections applied 2026-07-16):

- **`lance-graph::blasgraph` is the canonical, bit-exact kernel home.**
  The symbol `blasgraph::tropical_gemm` does not exist; the numerical f32
  min-plus (tropical-GEMM) kernel is **unwritten** and lands in blasgraph
  when the codec's A6 RDO wires it.
- The free-function path `lance-graph::bgz17::scalar_sparse::tropical_spmv`
  **never existed**. The only shipped min-plus is the method
  `bgz17::ScalarCsr::spmv_min_plus`
  (`crates/bgz17/src/scalar_sparse.rs:98`, `fn(&self, x: &[f32]) -> Vec<f32>`).
- bgz17 is a **lossy sibling encoding stack** — usable as a prototype
  adapter, never a substitute for the bit-exact blasgraph canon.

## Responsibility split

```text
ndarray:
  numerical kernels
  EWA/SYRK/MKL/OpenBLAS/AMX backend dispatch
  3DGS projection and certificates
  4x4 Block4 carrier math

lance-graph:
  graph/tile/block orchestration
  SplatShaderBlas naming and tier selection
  BLASGraph / bgz17 sparse substrate routing
  Lance/Arrow persistence
  query planning
  certified decision reports
```

## Unified SplatShaderBlas model

Represent all three tiers under one conceptual interface:

```rust
pub enum SplatShaderBlasTier {
    BitpackedPlane,
    PaletteDistance,
    Gaussian3d,
}

pub struct SplatShaderBlasRequest {
    pub tier: SplatShaderBlasTier,
    pub source_ref: String,
    pub query_ref: Option<String>,
    pub budget_ref: Option<String>,
}

pub struct SplatShaderBlasDecision {
    pub tier: SplatShaderBlasTier,
    pub action: String,
    pub score: f32,
    pub confidence: Option<f32>,
    pub certificate_id: Option<String>,
    pub reason_codes: Vec<String>,
}
```

This should start as a plan-level DTO, not immediate public API.

## Cross-pollination targets

### 1. EWA sandwich propagation

Use the same conceptual kernel across:

```text
2D cognitive planes
OSINT path covariance
3DGS spatial covariance projection
ultrasound PSF covariance
raw-field Block4 carrier certification
```

### 2. Tropical / sparse BLAS planning

Use tropical/spatial BLAS for decision planning:

```text
tile refinement graph
CTU / HHTL partition graph
certificate propagation graph
query relevance graph
```

### 3. Palette substrate

Reuse palette/codebook machinery for:

```text
3DGS Gaussian palette compression
raw-field kernel block clustering
attention/KV palette experiments
genetics/neuron motif blocks
```

### 4. Certified decision report

All tiers should return reports with:

```text
source id
operation tier
error / confidence / certificate id
reason codes
fallback action
```

## Integration into new 3DGS plans

Patch or reference these plans:

```text
3DGS-Lance-Arrow-storage-plan.md
  add SplatShaderBlas tier metadata and certificate links

3DGS-certified-query-render-plan.md
  include SplatShaderBlas decision tiers

3DGS-4x4-cognitive-shader-integration-plan.md
  map Block4 carrier to SplatShaderBlas-3DGS

3DGS-epiphany-roadmap-plan.md
  add CHLOD + SplatShaderBlas convergence path
```

## Acceptance criteria

- New 3DGS plans clearly reference the older SplatShaderBlas / PR-X12 line.
- blasgraph is treated as the canonical bit-exact kernel home (the tropical-GEMM kernel itself is unwritten until A6 wires it).
- The method `bgz17::ScalarCsr::spmv_min_plus` is recorded as the only shipped min-plus — a lossy-sibling prototype anchor, never the canon (re-corrected 2026-07-16).
- The three tiers are named separately to avoid bitpacked/palette/3DGS conflation.
- Tile/render/query decisions can consume SplatShaderBlas-style reports.

## Demo trajectory

First integrated demo should be deliberately small:

```text
one 3DGS tile block
one Lance sidecar
one ndarray EWA projection report
one SplatShaderBlas-3DGS decision
one persisted certificate
one query asking why the tile was rendered/refined/rejected
```

This proves the seam without requiring the whole ArcGIS/Cesium stack.
