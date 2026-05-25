# 3DGS 4x4 Cognitive-Shader Integration Plan — lance-graph

## Goal

Wire the ndarray-side 4x4 cognitive-shader SoA carrier into `lance-graph` orchestration and `cognitive-shader-driver` style runtime flows.

This plan connects the numeric substrate to graph/query/runtime scheduling.

## Core idea

```text
3DGS / raw-field kernel block
        ->
4x4 carrier
        ->
BindSpace-style SoA columns
        ->
query / traversal / certification / render decision
```

The 4x4 carrier is not only for geospatial rendering. It is the shared shape for cross-domain field blocks.

## Boundary with ndarray

`ndarray` owns:

```text
Mat4 / Sym4 / Block4 carriers
3x3-to-4x4 lifts
4-lane SoA kernels
HHTL contraction/scoring
pillar tests for lift invariants
```

`lance-graph` owns:

```text
schema registration
field source provenance
graph relationships
query planning
runtime decision reports
service endpoints
cross-domain dispatch
```

## BindSpace mapping

Recommended logical columns:

```text
bind_source[]
bind_state[]
bind_signal[]
bind_time[]
bind_cert[]
```

Domain interpretation examples:

```text
geospatial:
  source = tile / feature / local coordinate
  state  = covariance / transform
  signal = opacity / color / confidence
  time   = capture time / LOD phase / provenance

ultrasound:
  source = probe frame / local coordinate
  state  = PSF covariance / registration state
  signal = amplitude / Doppler / frequency
  time   = frame time / IMU pose / confidence

genetics:
  source = locus / motif coordinate
  state  = transition / edit neighborhood
  signal = expression / abundance / confidence
  time   = sample / lineage / provenance

neuronal:
  source = neuron / edge source
  state  = synaptic weight / uncertainty
  signal = activation / firing proxy
  time   = phase / layer / provenance
```

## Schema additions

Create cross-domain tables only after a fixture exists. Candidate tables:

```text
field_sources
field_frames
field_blocks
field_block_lanes
field_certificates
field_relationships
```

Do not force every geospatial table to use the generic field model. Use bridges.

## Runtime dispatch

```text
query or traversal request
  -> identify field kind
  -> load block metadata
  -> construct ndarray Block4 view
  -> call scoring/certification kernel
  -> convert result to decision report
```

## Certification aggregation

A 4x4 block certificate should support:

```text
domain kind
block id
input lane summary
kernel version
error terms
confidence
reason codes
```

`lance-graph` aggregates these into domain-specific reports.

## Acceptance criteria

- `lance-graph` can represent a 4x4 field block without knowing renderer internals.
- `cognitive-shader-driver` style SoA concepts map to durable schema names.
- Geospatial 3DGS remains compatible with 3D Tiles and does not become dependent on experimental domains.
- Cross-domain blocks can share reason codes and certificate summaries.
- The first implementation can be tested with a tiny synthetic fixture.
