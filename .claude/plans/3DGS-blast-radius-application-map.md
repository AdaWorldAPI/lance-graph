# 3DGS Blast Radius Application Map — lance-graph

## Goal

Map the wider blast radius of the 3DGS / HHTL / certified-field architecture.

This document prevents the architecture from being mistaken for only:

```text
ArcGIS integration
Cesium clone
3D map viewer
Gaussian splat renderer
```

The real substrate is broader:

```text
hierarchical local fields
  -> approximate kernels
  -> certified skip/refine/hydrate decisions
  -> query / render / compress / register / simulate / learn
```

## Core unifying pattern

```text
raw field / table / graph / signal
        ->
hierarchical blocks
        ->
approximate local kernels
        ->
certified skip / refine / hydrate decisions
        ->
executor
```

The executor differs by domain:

```text
geospatial: render/query tiles
lakehouse: scan/hydrate rows
codec: encode/decode frames
RAG: retrieve grounded chunks
ultrasound: fuse/register frames
robotics: update maps
science: refine simulation regions
genetics: scan motifs/pathways
neuronal: compare activation fields
observability: find root cause
finance: refine scenario trees
```

## Ring 1 — immediate / buildable

```text
datalake HHTL traversal
Lance/Arrow/DataFusion query acceleration
3DGS tile runtime
SplatShaderBlas-3DGS decision reports
PR-X12 codec acceleration
```

These should stay closest to the implementation path.

## Ring 2 — adjacent / high-value

```text
RAG memory and semantic retrieval
observability and incident response
digital twins and inspection
robotics / SLAM / drones
video upscaling via scene anchors
```

These should be prototyped after the core traversal and certificate DTOs stabilize.

## Ring 3 — specialized / plausible

```text
ultrasound RF/IQ/Doppler splat volumes
scientific simulation fields
genomics block traversal
finance scenario trees
cyber / OSINT propagation
```

These require domain fixtures and careful validation.

## Ring 4 — research-grade fanout

```text
neuronal network field summaries
agent cognition substrates
cross-domain 4x4 field algebra
certified approximate reasoning engines
```

These should stay fenced behind raw-field / 4x4 plans until lower rings prove the substrate.

## Application table

| Application | Block | Kernel | Certificate | Executor |
|---|---|---|---|---|
| Datalake traversal | fragment / row group | stats / centroid / covariance | skip / hydrate proof | DataFusion / graph scan |
| RAG memory | chunk / document | embedding centroid + provenance | retrieval confidence | answer grounding |
| 3D maps | tile / splat block | 3D Gaussian / EWA | render/refine error | renderer / map query |
| Video codec | CTU / scene anchor | DCT / EWA basis | rate-distortion bound | encoder / decoder |
| Ultrasound | RF/IQ frame block | PSF / Gaussian kernel | registration residual | volume fusion / AR |
| Robotics | sensor frame / map patch | occupancy / splat / covariance | map update confidence | SLAM / navigation |
| Scientific sim | grid patch / particle block | local PDE surrogate | residual / conservation error | refine / simulate |
| Genomics | motif / gene block | sequence transition kernel | motif confidence | motif/pathway query |
| Neuronal fields | microcircuit block | activation/edge kernel | temporal stability | graph/state comparison |
| OSINT/cyber | evidence plane / path | pressure splat / EWA | propagation bound | threat graph traversal |
| Observability | trace/log/service block | anomaly / dependency kernel | root-cause confidence | incident search |
| Finance/risk | scenario subtree | nested-distance kernel | branch risk bound | stress/pricing scan |
| Schema intelligence | table/column block | distribution + semantic role | drift/trust score | migration / lineage |

## Anti-patterns

Do not build one giant domain-aware crate.

Do not let geospatial code depend on ultrasound/genetics/neural semantics.

Do not let exploratory domains block Ring-1 implementation.

Do not make certificates decorative. If a certificate does not influence skip/refine/hydrate behavior, it is not production signal yet.

## Correct boundary

```text
core substrate:
  hierarchy
  kernels
  certificates
  HHTL traversal
  BLAS/SYRK/GEMM/tropical/popcount backends
  Lance/Arrow block storage

domain adapters:
  geospatial
  datalake
  video
  RAG
  ultrasound
  robotics
  science
  genomics
  finance
  observability
```

## First non-ArcGIS priority

Prioritize:

```text
3DGS-HHTL-datalake-traversal-plan.md
```

Reason:

```text
If datalake HHTL works, the architecture stops being a renderer project and becomes a general certified traversal substrate.
```

## Wall sentence

```text
Anything that can be represented as hierarchical local fields can be skipped, refined, rendered, queried, compressed, or learned with the same certified HHTL substrate.
```
