# NDARRAY_WIRING_PLAN.md -- Standalone Crate Type Duplication Audit

> Generated: 2026-03-30
> Branch: claude/unified-query-planner-aW8ax
> Scope: 5 standalone crates vs ndarray hpc/ canonical types

---

## Type Duplication Table

| # | Crate | File:Line | Duplicated Type | ndarray Equivalent |
|---|-------|-----------|-----------------|-------------------|
| 1 | causal-edge | `edge.rs:11` | `InferenceType` (enum, 8 variants) | `ndarray::hpc::nars` (implicit in `nars_deduction`/`nars_revision` dispatch) |
| 2 | causal-edge | `tables.rs:11` | `PackedTruth` (type alias u16) | `ndarray::hpc::nars::NarsTruth` (f32 pair, different repr) |
| 3 | causal-edge | `tables.rs:41` | `NarsTables` (precomputed 256x256 lookup) | `ndarray::hpc::nars::nars_revision`/`nars_deduction` (computed, not tabled) |
| 4 | causal-edge | `pearl.rs:28` | `CausalMask` (enum, 8 variants) | `ndarray::hpc::causality` (causal projection, different encoding) |
| 5 | causal-edge | `plasticity.rs:13` | `PlasticityState` (3-bit newtype) | No direct equivalent -- novel type |
| 6 | causal-edge | `edge.rs:60` | `CausalEdge64` (packed u64) | No direct equivalent -- novel type |
| 7 | deepnsm | `vocabulary.rs:23` | `WordEntry` | `ndarray::hpc::deepnsm::NsmEntry` (similar semantic word record) |
| 8 | deepnsm | `vocabulary.rs:36` | `Token` (word_idx + PoS) | No direct equivalent (ndarray deepnsm uses string-based API) |
| 9 | deepnsm | `vocabulary.rs:67` | `Vocabulary` (4096-word lookup) | `ndarray::hpc::deepnsm::nsm_vocabulary()` (returns `Vec<NsmEntry>`) |
| 10 | deepnsm | `similarity.rs:20` | `SimilarityTable` (CDF calibration) | No direct equivalent in ndarray |
| 11 | deepnsm | `encoder.rs:23` | `VsaVec` (512-bit, 8 u64 words) | `ndarray::hpc::vsa::VsaVector` (10000-bit, 157 u64 words) |
| 12 | deepnsm | `encoder.rs:17` | `VSA_BITS = 512` | `ndarray::hpc::vsa::VSA_DIMS = 10_000` (different dimensionality) |
| 13 | deepnsm | `spo.rs:24` | `SpoTriple` (packed u64, 3x12-bit) | `ndarray::hpc::deepnsm::SpoTriple` (packed u64, 3x12-bit -- identical layout) |
| 14 | deepnsm | `spo.rs:178` | `WordDistanceMatrix` (4096x4096 u8) | No direct equivalent (ndarray uses `CamCodebook` + ADC for distances) |
| 15 | deepnsm | `codebook.rs:24` | `Codebook` (6 subspaces x 256 centroids) | `ndarray::hpc::cam_pq::CamCodebook` (same PQ structure) |
| 16 | deepnsm | `codebook.rs:34` | `CamCodes` (per-word PQ codes) | `ndarray::hpc::deepnsm::load_cam_codes()` -> `Vec<CamFingerprint>` |
| 17 | deepnsm | `pos.rs:9` | `PoS` (enum, 8 parts-of-speech) | No direct equivalent |
| 18 | bgz-tensor | `projection.rs:39` | `Base17` (i16[17]) | `ndarray::hpc::bgz17_bridge::Base17` (identical struct) |
| 19 | bgz-tensor | `projection.rs:15` | `BASE_DIM = 17` | `ndarray::hpc::bgz17_bridge` (const `BASE_DIM = 17`) |
| 20 | bgz-tensor | `projection.rs:18` | `GOLDEN_STEP = 11` | `ndarray::hpc::bgz17_bridge` (const `GOLDEN_STEP = 11`) |
| 21 | bgz-tensor | `projection.rs:21` | `FP_SCALE = 256.0` | `ndarray::hpc::bgz17_bridge` (const `FP_SCALE = 256.0`) |
| 22 | bgz-tensor | `palette.rs:20` | `MAX_PALETTE = 256` | `ndarray::hpc::palette_distance` (const `MAX_PALETTE_SIZE = 256`) |
| 23 | bgz-tensor | `palette.rs:27` | `WeightPalette` (Vec<Base17> codebook) | `ndarray::hpc::palette_distance::Palette` (same shape) |
| 24 | bgz-tensor | `palette.rs:39` | `PaletteAssignment` (indices + residuals) | No direct equivalent (ndarray palette has no residual tracking) |
| 25 | bgz-tensor | `attention.rs:34` | `AttentionTable` (k x k u16 distance) | `ndarray::hpc::palette_distance::DistanceMatrix` (same structure) |
| 26 | bgz-tensor | `attention.rs:49` | `ComposeTable` (k x k u8 compose) | No direct equivalent (novel algebraic type) |
| 27 | bgz-tensor | `attention.rs:58` | `AttentionSemiring` (distance + compose) | Partial: `ndarray::hpc::palette_distance::DistanceMatrix` (distance half only) |
| 28 | bgz-tensor | `cascade.rs:133` | `ScentByte(u8)` | No direct equivalent (used in bgz17 as raw `u8` scent, not newtype) |
| 29 | bgz-tensor | `cascade.rs:35` | `CascadeLevel` (enum: Scent/Palette/Base17/Full) | `bgz17::Precision` (same 4-level enum) |
| 30 | bgz17 | `base17.rs:25` | `Base17` (i16[17]) | `ndarray::hpc::bgz17_bridge::Base17` (identical struct) |
| 31 | bgz17 | `base17.rs:177` | `SpoBase17` (3x Base17) | `ndarray::hpc::bgz17_bridge::SpoBase17` (identical struct) |
| 32 | bgz17 | `palette.rs:14` | `Palette` (Vec<Base17>) | `ndarray::hpc::palette_distance::Palette` (identical struct) |
| 33 | bgz17 | `palette.rs:21` | `PaletteEdge` (3x u8) | `ndarray::hpc::bgz17_bridge::PaletteEdge` (identical struct) |
| 34 | bgz17 | `distance_matrix.rs:15` | `DistanceMatrix` (k x k u16) | `ndarray::hpc::palette_distance::DistanceMatrix` (identical struct) |
| 35 | bgz17 | `distance_matrix.rs:80` | `SpoDistanceMatrices` (3x DistanceMatrix) | `ndarray::hpc::palette_distance::SpoDistanceMatrices` (identical struct) |
| 36 | bgz17 | `similarity.rs:12` | `SimilarityTable` (256-entry CDF) | No direct equivalent in ndarray |
| 37 | bgz17 | `clam_bridge.rs:85` | `Bgz17Metric` (layered distance fn) | `ndarray::hpc::clam::HammingDistance` (different metric, same role) |
| 38 | bgz17 | `clam_bridge.rs:197` | `Lfd` (local fractal dimension) | `ndarray::hpc::clam::Lfd` (identical struct) |
| 39 | bgz17 | `clam_bridge.rs:214` | `Bgz17Cluster` (cluster node) | `ndarray::hpc::clam::ClamTree` internal nodes (same concept) |
| 40 | bgz17 | `clam_bridge.rs:241` | `Bgz17ClamTree` (binary tree) | `ndarray::hpc::clam::ClamTree` (same algorithm, different metric) |
| 41 | bgz17 | `simd.rs:17` | `SimdLevel` (enum: Scalar/SSE2/AVX2/AVX512) | `ndarray::hpc::simd_dispatch::SimdTier` (identical concept) |
| 42 | bgz17 | `palette_semiring.rs:17` | `PaletteSemiring` (distance + compose) | Partial: `ndarray::hpc::palette_distance::DistanceMatrix` + no compose |
| 43 | bgz17 | `scalar_sparse.rs:9` | `ScalarCsr` (CSR sparse matrix) | No direct equivalent in ndarray hpc/ |

**Total duplicated types: 43**
**Exact structural duplicates (identical layout): 12** (#13, #18-22, #30-35, #38)
**Near-duplicates (same concept, different repr): 16** (#1-3, #7, #9, #11, #15-16, #23, #25, #29, #37, #39-42)
**Novel types (no ndarray equivalent): 15** (#4-6, #8, #10, #12, #14, #17, #24, #26-28, #36, #43)

---

## Recommended Wiring Order (least dependencies first)

### Phase A: bgz17 (0 external deps, 121 tests, canonical palette codec)

bgz17 is the **source of truth** for Base17, Palette, DistanceMatrix, PaletteSemiring.
ndarray::hpc::bgz17_bridge and palette_distance are copies OF bgz17, not the other way around.

1. Add `bgz17` as an optional dependency of ndarray behind feature `bgz17-codec`
2. Replace `ndarray::hpc::bgz17_bridge::{Base17, SpoBase17, PaletteEdge}` with re-exports from bgz17
3. Replace `ndarray::hpc::palette_distance::{Palette, DistanceMatrix, SpoDistanceMatrices}` with re-exports
4. Wire `ndarray::hpc::clam::Lfd` to use `bgz17::clam_bridge::Lfd` (or extract to shared micro-crate)
5. Wire `SimdLevel` -> `SimdTier` mapping (4-variant enum, trivial From impl)

**Risk**: Low. bgz17 is standalone with 121 tests. No transitive deps.

### Phase B: deepnsm (0 external deps, depends on vocabulary data)

1. Add `deepnsm` as optional dependency of ndarray behind feature `deepnsm-engine`
2. Replace `ndarray::hpc::deepnsm::SpoTriple` with re-export from `deepnsm::spo::SpoTriple`
3. Wire `deepnsm::codebook::Codebook` to share structure with `ndarray::hpc::cam_pq::CamCodebook`
   (or provide `From<Codebook> for CamCodebook` conversion)
4. VsaVec (512-bit) and VsaVector (10000-bit) are intentionally different sizes -- do NOT merge.
   Instead, provide a `From<VsaVec> for VsaVector` that zero-extends.

**Risk**: Low. deepnsm is standalone. The only structural duplicate is SpoTriple.

### Phase C: bgz-tensor (0 external deps, depends on Base17/Palette concepts)

1. Add `bgz17` as a dependency of bgz-tensor (replaces local Base17, Palette, constants)
2. Replace `bgz-tensor::projection::Base17` with `bgz17::base17::Base17`
3. Replace `bgz-tensor::palette::WeightPalette` internals with `bgz17::palette::Palette`
4. Replace `bgz-tensor::attention::AttentionTable` distance storage with `bgz17::distance_matrix::DistanceMatrix`
5. Keep `ComposeTable` and `AttentionSemiring` as bgz-tensor-specific (novel types)
6. Align constants: `MAX_PALETTE`, `BASE_DIM`, `GOLDEN_STEP`, `FP_SCALE` -> import from bgz17

**Risk**: Medium. bgz-tensor's `Base17` has a `from_f32_row` projection method not in bgz17.
Need to add that as an extension trait or method on bgz17::Base17.

### Phase D: causal-edge (depends on NARS concepts)

1. Add `ndarray` as optional dependency behind feature `ndarray-nars`
2. Provide `From<NarsTruth> for (f32, f32)` and `From<(u8, u8)> for PackedTruth` bridges
3. NarsTables is a performance optimization not present in ndarray -- keep in causal-edge,
   but add a builder that consumes `ndarray::hpc::nars::nars_*` functions to fill tables
4. InferenceType (causal-edge) maps to ndarray's nars_* function dispatch -- provide a
   `dispatch(InferenceType, NarsTruth, NarsTruth) -> NarsTruth` bridge function
5. CausalEdge64, CausalMask, PlasticityState are novel -- keep as-is

**Risk**: Medium. causal-edge's u8-quantized NARS differs from ndarray's f32 NARS.
The bridge must handle quantization/dequantization at the boundary.

### Phase E: p64-bridge (depends on causal-edge + bgz17, bridges to external p64)

p64-bridge is already a bridge crate. It has no ndarray type duplication -- its types
(`StyleParams`, `combine::*`, `contra::*`, predicate layer constants) mirror the external
p64 crate's API, not ndarray. No wiring action needed beyond ensuring its bgz17 dependency
uses the same bgz17 version as ndarray.

**Risk**: None for ndarray wiring. Coordinate p64 crate version only.

---

## Feature Gate Pattern Example

```toml
# In bgz-tensor/Cargo.toml:
[dependencies]
bgz17 = { path = "../bgz17", optional = true }

[features]
default = []
bgz17-native = ["dep:bgz17"]   # Use bgz17 canonical types
```

```rust
// In bgz-tensor/src/projection.rs:

#[cfg(feature = "bgz17-native")]
pub use bgz17::base17::Base17;

#[cfg(not(feature = "bgz17-native"))]
mod standalone_base17 {
    /// Standalone Base17 for zero-dep builds.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Base17 {
        pub dims: [i16; 17],
    }
    // ... standalone impls ...
}
#[cfg(not(feature = "bgz17-native"))]
pub use standalone_base17::Base17;
```

```rust
// In ndarray/src/hpc/bgz17_bridge.rs (after Phase A):

#[cfg(feature = "bgz17-codec")]
pub use bgz17::base17::{Base17, SpoBase17};
#[cfg(feature = "bgz17-codec")]
pub use bgz17::palette::{Palette, PaletteEdge};

#[cfg(not(feature = "bgz17-codec"))]
mod standalone { /* existing self-contained impls */ }
#[cfg(not(feature = "bgz17-codec"))]
pub use standalone::*;
```

---

## Summary

- **43 types** audited across 5 standalone crates
- **12 exact duplicates** (identical struct layout, should be re-exports)
- **16 near-duplicates** (same concept, bridgeable via From/Into)
- **15 novel types** (no action needed)
- **Wiring order**: bgz17 -> deepnsm -> bgz-tensor -> causal-edge -> p64-bridge
- **Zero breaking changes** if feature-gated: standalone fallback always available
