# 3DGS Ultrasound SaMD Plan — lance-graph

## Goal

Capture the ultrasound-specific branch of the 3DGS raw-field architecture.

The idea is to avoid collapsing raw ultrasound data into lossy 2D B-mode pixels too early.

Instead:

```text
RF / IQ / channel-RF / Doppler / IMU
        ->
beamform or delay-and-sum stage
        ->
splat-fit stage
        ->
Gaussian3D / Gaussian4 batch
        ->
4D patient-aligned splat volume
        ->
AR / atlas / registration / certified rendering
```

## Why this matters

Classic ultrasound pipeline:

```text
RF -> beamform -> envelope -> log-compress -> scan-convert -> 2D B-mode pixels
```

Splat-native pipeline:

```text
RF/IQ -> beamform -> splat-fit -> Gaussian3D batch
```

The second path preserves more structure:

- amplitude
- phase/frequency content
- point spread function shape
- Doppler/color flow information
- IMU pose
- frame-to-frame registration

## Source hardware class

Initial hardware categories:

```text
wireless ultrasound probes
research ultrasound SDKs
RF/IQ-export-capable devices
Doppler/IMU-capable devices
```

Do not hard-code one vendor. Represent capabilities.

```rust
pub struct UltrasoundSourceCapabilities {
    pub rf: bool,
    pub iq: bool,
    pub channel_rf: bool,
    pub doppler: bool,
    pub imu: bool,
    pub frame_pose: bool,
    pub sdk_streaming: bool,
}
```

## Splat mapping

```text
echo amplitude        -> opacity / density
PSF estimate          -> anisotropic covariance
Doppler/frequency     -> color / spherical-harmonic-like lane
IMU pose              -> frame transform
multi-frame fusion    -> splat-to-splat registration
```

## Tables

Add optional ultrasound-specific tables on top of raw-field tables:

```text
ultrasound_sources
ultrasound_frames
ultrasound_probe_pose
ultrasound_splat_batches
ultrasound_registration_edges
ultrasound_atlas_alignment
```

## SaMD trajectory

Treat this as a research/development plan first.

Milestones:

```text
research tool
  -> offline viewer
  -> reproducibility study
  -> clinical study support
  -> regulated SaMD path if justified
```

Never imply diagnostic use before validation and regulatory review.

## 4x4 carrier role

Use the 4x4 cognitive-shader carrier as:

```text
lane0: local 3D position + homogeneous coordinate
lane1: PSF/covariance state
lane2: amplitude / Doppler / frequency feature
lane3: time / frame pose / registration confidence
```

## Certification hooks

Possible certificates:

```text
PSF covariance validity
frame registration residual
multi-frame fusion confidence
Doppler lane consistency
atlas alignment residual
quantization error
```

## AR output

Potential output modes:

```text
CPU preview
WGPU viewer
AR headset stream
atlas-aligned overlay
patient-aligned splat volume
```

## Acceptance criteria

- Ingest path can represent RF/IQ/Doppler/IMU capabilities without vendor lock.
- Splat batch metadata can attach to frame pose and registration confidence.
- Atlas and live-patient volumes can share the same splat representation.
- Certification records can attach to registration and fusion steps.
- This remains separate from the geospatial 3D Tiles core.
