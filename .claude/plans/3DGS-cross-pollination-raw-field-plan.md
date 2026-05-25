# 3DGS Cross-Pollination Raw-Field Plan — lance-graph

## Goal

Generalize the 3DGS geospatial rebuild into a reusable raw-field pipeline.

The common pattern is:

```text
raw sensor / raw signal / raw graph field
        ->
structured anisotropic kernels
        ->
4x4 cognitive-shader SoA carrier
        ->
certified query / render / learn decisions
```

ArcGIS and Cesium are one use case. Ultrasound, genetics, neuronal networks, and scientific fields can use the same backbone.

## Core abstraction

```text
RawFieldSource
  -> FieldFrame
  -> KernelFit
  -> Gaussian3D / Gaussian4 carrier
  -> Lance/Arrow block store
  -> graph/query/certificate layer
```

Candidate DTOs:

```rust
pub enum RawFieldKind {
    Geospatial3d,
    UltrasoundRfIq,
    GeneticSequence,
    NeuronalGraph,
    ScientificVolume,
    Unknown,
}

pub struct RawFieldFrame {
    pub source_id: String,
    pub frame_id: String,
    pub kind: RawFieldKind,
    pub coordinate_frame: String,
    pub payload_ref: String,
    pub timestamp: Option<i64>,
}

pub struct KernelFitReport {
    pub frame_id: String,
    pub kernel_count: u64,
    pub residual_error: f32,
    pub confidence: f32,
    pub certificate_id: Option<String>,
}
```

## Cross-domain mapping

```text
geospatial scan
  splats represent surface/radiance/geometry kernels

ultrasound RF/IQ
  splats represent echo amplitude, PSF covariance, Doppler/frequency lanes

genetics
  splats/blocks represent local sequence motifs, transition kernels, expression fields

neuronal networks
  splats/blocks represent activation fields, adjacency motifs, synaptic uncertainty
```

## 4x4 carrier interpretation

```text
lane0: spatial/source coordinate
lane1: covariance/transition/edge state
lane2: intensity/expression/activation/confidence
lane3: time/phase/provenance/semantic role
```

The same 4-lane grammar should be usable across domains.

## Lance/Arrow role

Store domain-specific raw payloads separately, but normalize summaries into common tables:

```text
field_sources
field_frames
kernel_blocks
kernel_certificates
field_features
field_relationships
```

## Graph role

Represent the transformation chain:

```text
Source -> Frame -> KernelBlock -> Feature -> Certificate
```

For domain fanout:

```text
Feature -> Gene
Feature -> Neuron
Feature -> AnatomicalRegion
Feature -> Tile
Feature -> Asset
```

## Acceptance criteria

- A field source can be registered without knowing its domain-specific decoder.
- Kernel blocks can be stored and queried by common metadata.
- Certificates can attach to frames, blocks, and features.
- Geospatial 3DGS remains the first production path, not blocked by other domains.
- Domain-specific experiments do not pollute the core 3D Tiles runtime.

## Implementation note

Keep this as a meta-plan. Promote concrete domain work into separate plans only when there is a testable fixture and a crate owner.
