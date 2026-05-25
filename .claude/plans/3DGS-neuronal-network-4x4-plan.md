# 3DGS Neuronal Network 4x4 Plan — lance-graph

## Goal

Explore how the 3DGS raw-field and 4x4 cognitive-shader representation can model neuronal networks, activation fields, and graph dynamics.

This is an exploratory plan for cross-pollination. It should not block the geospatial 3DGS path.

## Core analogy

```text
3DGS scene
  anisotropic Gaussian kernels over visual/spatial evidence

neuronal network
  anisotropic kernels over activation, connectivity, timing, and uncertainty
```

The purpose is not to claim biological equivalence. The purpose is to reuse the same compact certified field representation.

## 4x4 carrier interpretation

```text
lane0: neuron id / source coordinate / local embedding
lane1: edge weight / covariance / synaptic uncertainty
lane2: activation / firing proxy / confidence
lane3: time / phase / layer / provenance
```

## (4x4)^4 hierarchy

```text
level 0: neuron or synapse carrier
level 1: local microcircuit block
level 2: region / layer block
level 3: network state / cognitive field block
```

This mirrors:

```text
splat -> block -> tile -> region
synapse -> microcircuit -> area -> network state
```

## Graph model

```text
Neuron -> Synapse -> MicroCircuit -> Region -> NetworkState
ActivationFrame -> ActivationBlock -> Certificate
```

## Tables

Potential tables:

```text
neuronal_sources
activation_frames
neuron_features
synapse_edges
microcircuit_blocks
network_state_blocks
neuronal_certificates
```

## HHTL cascade

Use cascade selection for graph/activation scans:

```text
HEEL: structural adjacency candidate
HIP: activation/weight threshold
TWIG: covariance/temporal refinement
LEAF: exact graph traversal or simulation step
```

## Certificates

Potential certificates:

```text
activation stability
edge uncertainty
temporal drift
microcircuit contraction
graph signature uniqueness
sampling confidence
```

## Integration with cognitive-shader-driver

The 4x4 carrier should map into BindSpace-style SoA columns:

```text
source_lane[]
edge_lane[]
activation_lane[]
time_phase_lane[]
certificate_lane[]
```

The driver should not care whether the block came from 3DGS geospatial data, ultrasound, genetics, or neuronal traces.

## Candidate research uses

```text
activation field compression
attention graph visualization
neural simulation summaries
connectome subgraph search
temporal state comparison
```

## Acceptance criteria for promotion

Promote this into implementation only when:

- a small fixture is selected
- the 4x4 lane semantics are precise
- the certificate has measurable meaning
- the graph schema is separable from geospatial schemas
- no biological or medical overclaim is made

## Boundary

This is a computational representation plan, not a neuroscience claim.
