# highheelbgz

crate: highheelbgz
path: crates/highheelbgz
workspace: excluded
digest: adc780f445752101
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep bgz-tensor normal optional — crates/highheelbgz/Cargo.toml:19 CONFIRMED

## Depended on by

- rdep thinking-engine normal — crates/thinking-engine/Cargo.toml:20 CONFIRMED

## Public items

- mod rehydrate [pub] — crates/highheelbgz/src/lib.rs:14 CONFIRMED
- mod simd_hardened [pub] — crates/highheelbgz/src/lib.rs:15 CONFIRMED
- mod source [pub] — crates/highheelbgz/src/lib.rs:16 CONFIRMED
- mod tensor_bridge [pub] cfg=feature="tensor" — crates/highheelbgz/src/lib.rs:19 CONFIRMED
- const BASE_DIM [pub] — crates/highheelbgz/src/lib.rs:23 CONFIRMED
- struct SpiralAddress [pub] — crates/highheelbgz/src/lib.rs:37 CONFIRMED
- const SpiralAddress::BYTE_SIZE_U32 [pub] — crates/highheelbgz/src/lib.rs:44 CONFIRMED
- const SpiralAddress::BYTE_SIZE_U16 [pub] — crates/highheelbgz/src/lib.rs:45 CONFIRMED
- fn SpiralAddress::new [pub] — crates/highheelbgz/src/lib.rs:47 CONFIRMED
- fn SpiralAddress::total_samples [pub] — crates/highheelbgz/src/lib.rs:56 CONFIRMED
- fn SpiralAddress::octave_positions [pub] — crates/highheelbgz/src/lib.rs:61 CONFIRMED
- fn SpiralAddress::end [pub] — crates/highheelbgz/src/lib.rs:76 CONFIRMED
- enum CoarseBand [pub] — crates/highheelbgz/src/lib.rs:87 CONFIRMED
- fn SpiralAddress::coarse_band [pub] — crates/highheelbgz/src/lib.rs:106 CONFIRMED
- fn SpiralAddress::overlap_fraction [pub] — crates/highheelbgz/src/lib.rs:127 CONFIRMED
- enum TensorRole [pub] — crates/highheelbgz/src/lib.rs:152 CONFIRMED
- fn SpiralAddress::role [pub] — crates/highheelbgz/src/lib.rs:163 CONFIRMED
- enum ThinkingScale [pub] — crates/highheelbgz/src/lib.rs:177 CONFIRMED
- struct NeuronPrint [pub] — crates/highheelbgz/src/lib.rs:190 CONFIRMED
- const NeuronPrint::BYTE_SIZE [pub] — crates/highheelbgz/src/lib.rs:200 CONFIRMED
- fn NeuronPrint::thinking_scale [pub] — crates/highheelbgz/src/lib.rs:203 CONFIRMED
- fn NeuronPrint::effective_rank [pub] — crates/highheelbgz/src/lib.rs:213 CONFIRMED
- fn NeuronPrint::coarse_band [pub] — crates/highheelbgz/src/lib.rs:219 CONFIRMED
- struct SpiralWalk [pub] — crates/highheelbgz/src/lib.rs:240 CONFIRMED
- fn SpiralWalk::execute [pub] — crates/highheelbgz/src/lib.rs:251 CONFIRMED
- fn SpiralWalk::cosine [pub] — crates/highheelbgz/src/lib.rs:277 CONFIRMED
- fn SpiralWalk::l1 [pub] — crates/highheelbgz/src/lib.rs:301 CONFIRMED
- fn SpiralWalk::to_f32 [pub] — crates/highheelbgz/src/lib.rs:314 CONFIRMED
- fn address_distance [pub] — crates/highheelbgz/src/lib.rs:331 CONFIRMED
- fn address_overlap [pub] — crates/highheelbgz/src/lib.rs:340 CONFIRMED
- struct SpiralPalette [pub] — crates/highheelbgz/src/lib.rs:365 CONFIRMED
- fn SpiralPalette::default_256 [pub] — crates/highheelbgz/src/lib.rs:372 CONFIRMED
- fn SpiralPalette::assign [pub] — crates/highheelbgz/src/lib.rs:386 CONFIRMED
- fn SpiralPalette::byte_size [pub] — crates/highheelbgz/src/lib.rs:407 CONFIRMED
- fn calibrate [pub] — crates/highheelbgz/src/lib.rs:418 CONFIRMED
- struct SpiralEncoding [pub] — crates/highheelbgz/src/rehydrate.rs:17 CONFIRMED
- struct GammaProfile [pub] — crates/highheelbgz/src/rehydrate.rs:29 CONFIRMED
- const GammaProfile::BYTE_SIZE [pub] — crates/highheelbgz/src/rehydrate.rs:37 CONFIRMED
- fn GammaProfile::calibrate [pub] — crates/highheelbgz/src/rehydrate.rs:40 CONFIRMED
- fn SpiralEncoding::byte_size [pub] — crates/highheelbgz/src/rehydrate.rs:65 CONFIRMED
- fn SpiralEncoding::encode [pub] — crates/highheelbgz/src/rehydrate.rs:70 CONFIRMED
- fn SpiralEncoding::rehydrate [pub] — crates/highheelbgz/src/rehydrate.rs:87 CONFIRMED
- fn SpiralEncoding::rehydrate_interpolated [pub] — crates/highheelbgz/src/rehydrate.rs:102 CONFIRMED
- fn SpiralEncoding::cosine [pub] — crates/highheelbgz/src/rehydrate.rs:144 CONFIRMED
- fn SpiralEncoding::cosine_interpolated [pub] — crates/highheelbgz/src/rehydrate.rs:151 CONFIRMED
- struct SpiralTokenizer [pub] — crates/highheelbgz/src/rehydrate.rs:198 CONFIRMED
- fn SpiralTokenizer::build [pub] — crates/highheelbgz/src/rehydrate.rs:215 CONFIRMED
- fn SpiralTokenizer::nearest [pub] — crates/highheelbgz/src/rehydrate.rs:239 CONFIRMED
- fn SpiralTokenizer::nearest_k [pub] — crates/highheelbgz/src/rehydrate.rs:257 CONFIRMED
- fn SpiralTokenizer::byte_size [pub] — crates/highheelbgz/src/rehydrate.rs:277 CONFIRMED
- struct SpiralAddr [pub] — crates/highheelbgz/src/simd_hardened.rs:17 CONFIRMED
- fn SpiralAddr::new [pub] — crates/highheelbgz/src/simd_hardened.rs:27 CONFIRMED
- fn SpiralAddr::new_unchecked [pub] — crates/highheelbgz/src/simd_hardened.rs:36 CONFIRMED
- fn SpiralAddr::n_samples [pub] — crates/highheelbgz/src/simd_hardened.rs:43 CONFIRMED
- fn SpiralAddr::role [pub] — crates/highheelbgz/src/simd_hardened.rs:49 CONFIRMED
- enum TensorRole [pub] — crates/highheelbgz/src/simd_hardened.rs:63 CONFIRMED
- enum ThinkingScale [pub] — crates/highheelbgz/src/simd_hardened.rs:74 CONFIRMED
- enum CoarseBand [pub] — crates/highheelbgz/src/simd_hardened.rs:87 CONFIRMED
- fn coarse_band [pub] — crates/highheelbgz/src/simd_hardened.rs:97 CONFIRMED
- struct NeuronPrint [pub] — crates/highheelbgz/src/simd_hardened.rs:142 CONFIRMED
- fn NeuronPrint::thinking_scale [pub] — crates/highheelbgz/src/simd_hardened.rs:155 CONFIRMED
- fn NeuronPrint::effective_rank [pub] — crates/highheelbgz/src/simd_hardened.rs:166 CONFIRMED
- fn NeuronPrint::coarse_band [pub] — crates/highheelbgz/src/simd_hardened.rs:172 CONFIRMED
- struct SpiralIndex [pub] — crates/highheelbgz/src/simd_hardened.rs:200 CONFIRMED
- fn SpiralIndex::new [pub] — crates/highheelbgz/src/simd_hardened.rs:207 CONFIRMED
- fn SpiralIndex::push [pub] — crates/highheelbgz/src/simd_hardened.rs:215 CONFIRMED
- fn SpiralIndex::len [pub] — crates/highheelbgz/src/simd_hardened.rs:221 CONFIRMED
- fn SpiralIndex::is_empty [pub] — crates/highheelbgz/src/simd_hardened.rs:224 CONFIRMED
- fn SpiralIndex::addr [pub] — crates/highheelbgz/src/simd_hardened.rs:228 CONFIRMED
- fn SpiralIndex::heel_filter [pub] — crates/highheelbgz/src/simd_hardened.rs:238 CONFIRMED
- struct NeuronIndex [pub] — crates/highheelbgz/src/simd_hardened.rs:256 CONFIRMED
- fn NeuronIndex::new [pub] — crates/highheelbgz/src/simd_hardened.rs:266 CONFIRMED
- fn NeuronIndex::push [pub] — crates/highheelbgz/src/simd_hardened.rs:277 CONFIRMED
- fn NeuronIndex::len [pub] — crates/highheelbgz/src/simd_hardened.rs:286 CONFIRMED
- fn NeuronIndex::heel_filter [pub] — crates/highheelbgz/src/simd_hardened.rs:292 CONFIRMED
- const PHI4 [pub] — crates/highheelbgz/src/simd_hardened.rs:314 CONFIRMED
- const PHI8 [pub] — crates/highheelbgz/src/simd_hardened.rs:315 CONFIRMED
- const PHI16 [pub] — crates/highheelbgz/src/simd_hardened.rs:316 CONFIRMED
- fn hip_sample [pub] — crates/highheelbgz/src/simd_hardened.rs:331 CONFIRMED
- fn hydrate_4 [pub] — crates/highheelbgz/src/simd_hardened.rs:342 CONFIRMED
- fn hydrate_8 [pub] — crates/highheelbgz/src/simd_hardened.rs:359 CONFIRMED
- fn hydrate_16 [pub] — crates/highheelbgz/src/simd_hardened.rs:376 CONFIRMED
- fn leaf_hydrate [pub] — crates/highheelbgz/src/simd_hardened.rs:392 CONFIRMED
- fn cosine_f32_8 [pub] — crates/highheelbgz/src/simd_hardened.rs:401 CONFIRMED
- fn cosine_f32_16 [pub] — crates/highheelbgz/src/simd_hardened.rs:421 CONFIRMED
- struct TensorLocation [pub] — crates/highheelbgz/src/source.rs:15 CONFIRMED
- fn TensorLocation::bytes_per_element [pub] — crates/highheelbgz/src/source.rs:31 CONFIRMED
- fn TensorLocation::n_rows [pub] — crates/highheelbgz/src/source.rs:41 CONFIRMED
- struct GgufIndex [pub] — crates/highheelbgz/src/source.rs:54 CONFIRMED
- fn GgufIndex::open [pub] — crates/highheelbgz/src/source.rs:61 CONFIRMED
- fn GgufIndex::find [pub] — crates/highheelbgz/src/source.rs:93 CONFIRMED
- fn GgufIndex::find_matching [pub] — crates/highheelbgz/src/source.rs:98 CONFIRMED
- fn GgufIndex::hydrate_row [pub] — crates/highheelbgz/src/source.rs:109 CONFIRMED
- fn GgufIndex::hydrate_walk [pub] — crates/highheelbgz/src/source.rs:117 CONFIRMED
- fn GgufIndex::index_bytes [pub] — crates/highheelbgz/src/source.rs:128 CONFIRMED
- fn GgufIndex::summary [pub] — crates/highheelbgz/src/source.rs:133 CONFIRMED
- fn walk_to_stacked [pub] cfg=feature="tensor" — crates/highheelbgz/src/tensor_bridge.rs:29 CONFIRMED
- fn addr_to_stacked [pub] cfg=feature="tensor" — crates/highheelbgz/src/tensor_bridge.rs:42 CONFIRMED
- fn cascade_search [pub] cfg=feature="tensor" — crates/highheelbgz/src/tensor_bridge.rs:64 CONFIRMED
- fn cascade_search_with_query_source [pub] cfg=feature="tensor" — crates/highheelbgz/src/tensor_bridge.rs:149 CONFIRMED

## Imports from in-scope crates

### bgz-tensor
- use bgz_tensor::stacked_n::StackedN cfg=feature="tensor" — crates/highheelbgz/src/tensor_bridge.rs:18 CONFIRMED

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n1["bgz-tensor"]
  n22["highheelbgz"]
  n86["thinking-engine"]
  n22 --> n1
  n86 --> n22
```
