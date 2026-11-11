# helix

crate: helix
path: crates/helix
workspace: excluded
digest: 7730532a31173773
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep ndarray normal — crates/helix/Cargo.toml:22 CONFIRMED

## Depended on by


## Public items

- const GOLDEN_RATIO [pub] — crates/helix/src/constants.rs:26 CONFIRMED
- const GOLDEN_RATIO_INV [pub] — crates/helix/src/constants.rs:30 CONFIRMED
- const GOLDEN_ANGLE [pub] — crates/helix/src/constants.rs:35 CONFIRMED
- const EULER_GAMMA [pub] — crates/helix/src/constants.rs:38 CONFIRMED
- const E [pub] — crates/helix/src/constants.rs:43 CONFIRMED
- const MODULUS [pub] — crates/helix/src/constants.rs:47 CONFIRMED
- const STRIDE [pub] — crates/helix/src/constants.rs:52 CONFIRMED
- const LN_17 [pub] — crates/helix/src/constants.rs:56 CONFIRMED
- const TRANSIENT_SKIP [pub] — crates/helix/src/constants.rs:61 CONFIRMED
- const PALETTE_SIZE [pub] — crates/helix/src/constants.rs:67 CONFIRMED
- struct CurveRuler [pub] — crates/helix/src/curve_ruler.rs:20 CONFIRMED
- const CurveRuler::MODULUS [pub] — crates/helix/src/curve_ruler.rs:26 CONFIRMED
- const CurveRuler::STRIDE [pub] — crates/helix/src/curve_ruler.rs:28 CONFIRMED
- fn CurveRuler::from_place [pub] — crates/helix/src/curve_ruler.rs:31 CONFIRMED
- fn CurveRuler::from_hhtl [pub] — crates/helix/src/curve_ruler.rs:41 CONFIRMED
- fn CurveRuler::start_offset [pub] — crates/helix/src/curve_ruler.rs:46 CONFIRMED
- fn CurveRuler::index [pub] — crates/helix/src/curve_ruler.rs:53 CONFIRMED
- fn CurveRuler::arc [pub] — crates/helix/src/curve_ruler.rs:59 CONFIRMED
- struct DistanceLut [pub] — crates/helix/src/distance.rs:18 CONFIRMED
- fn DistanceLut::linear [pub] — crates/helix/src/distance.rs:25 CONFIRMED
- fn DistanceLut::from_floor [pub] — crates/helix/src/distance.rs:39 CONFIRMED
- fn DistanceLut::circular [pub] — crates/helix/src/distance.rs:70 CONFIRMED
- fn DistanceLut::distance [pub] — crates/helix/src/distance.rs:84 CONFIRMED
- struct Similarity [pub] — crates/helix/src/fisher_z.rs:28 CONFIRMED
- const Similarity::CLAMP_EPS [pub] — crates/helix/src/fisher_z.rs:36 CONFIRMED
- fn Similarity::fisher_z [pub] — crates/helix/src/fisher_z.rs:55 CONFIRMED
- fn Similarity::hyperbolic_depth [pub] — crates/helix/src/fisher_z.rs:76 CONFIRMED
- mod constants [pub] — crates/helix/src/lib.rs:60 CONFIRMED
- mod curve_ruler [pub] — crates/helix/src/lib.rs:63 CONFIRMED
- mod distance [pub] — crates/helix/src/lib.rs:64 CONFIRMED
- mod fisher_z [pub] — crates/helix/src/lib.rs:65 CONFIRMED
- mod placement [pub] — crates/helix/src/lib.rs:66 CONFIRMED
- mod prove [pub] — crates/helix/src/lib.rs:67 CONFIRMED
- mod quantize [pub] — crates/helix/src/lib.rs:68 CONFIRMED
- mod residue [pub] — crates/helix/src/lib.rs:69 CONFIRMED
- mod simd [pub] — crates/helix/src/lib.rs:70 CONFIRMED
- enum Sign [pub] — crates/helix/src/placement.rs:31 CONFIRMED
- fn Sign::as_f64 [pub] — crates/helix/src/placement.rs:42 CONFIRMED
- fn Sign::of [pub] — crates/helix/src/placement.rs:52 CONFIRMED
- struct HemispherePoint [pub] — crates/helix/src/placement.rs:66 CONFIRMED
- fn HemispherePoint::lift [pub] — crates/helix/src/placement.rs:112 CONFIRMED
- fn HemispherePoint::cartesian [pub] — crates/helix/src/placement.rs:141 CONFIRMED
- fn HemispherePoint::rim [pub] — crates/helix/src/placement.rs:161 CONFIRMED
- fn HemispherePoint::signed_lift [pub] — crates/helix/src/placement.rs:185 CONFIRMED
- struct ProofResult [pub] — crates/helix/src/prove.rs:106 CONFIRMED
- fn ProofResult::report [pub] — crates/helix/src/prove.rs:122 CONFIRMED
- fn prove [pub] — crates/helix/src/prove.rs:149 CONFIRMED
- struct RollingFloor [pub] — crates/helix/src/quantize.rs:50 CONFIRMED
- fn RollingFloor::uniform [pub] — crates/helix/src/quantize.rs:72 CONFIRMED
- fn RollingFloor::with_params [pub] — crates/helix/src/quantize.rs:80 CONFIRMED
- fn RollingFloor::quantize [pub] — crates/helix/src/quantize.rs:99 CONFIRMED
- fn RollingFloor::observe [pub] — crates/helix/src/quantize.rs:113 CONFIRMED
- fn RollingFloor::drift_score [pub] — crates/helix/src/quantize.rs:131 CONFIRMED
- fn RollingFloor::roll [pub] — crates/helix/src/quantize.rs:165 CONFIRMED
- fn RollingFloor::version [pub] — crates/helix/src/quantize.rs:221 CONFIRMED
- fn RollingFloor::occupancy [pub] — crates/helix/src/quantize.rs:226 CONFIRMED
- fn RollingFloor::samples [pub] — crates/helix/src/quantize.rs:232 CONFIRMED
- fn RollingFloor::bounds [pub] — crates/helix/src/quantize.rs:237 CONFIRMED
- fn RollingFloor::bucket_center [pub] — crates/helix/src/quantize.rs:248 CONFIRMED
- struct ResidueEdge [pub] — crates/helix/src/residue.rs:20 CONFIRMED
- fn ResidueEdge::to_bytes [pub] — crates/helix/src/residue.rs:31 CONFIRMED
- fn ResidueEdge::from_bytes [pub] — crates/helix/src/residue.rs:36 CONFIRMED
- fn ResidueEdge::distance_adaptive [pub] — crates/helix/src/residue.rs:47 CONFIRMED
- fn ResidueEdge::distance_heuristic [pub] — crates/helix/src/residue.rs:56 CONFIRMED
- struct Signed360 [pub] — crates/helix/src/residue.rs:77 CONFIRMED
- fn Signed360::to_bytes [pub] — crates/helix/src/residue.rs:92 CONFIRMED
- fn Signed360::from_bytes [pub] — crates/helix/src/residue.rs:99 CONFIRMED
- fn Signed360::sign [pub] — crates/helix/src/residue.rs:109 CONFIRMED
- struct ResidueEncoder [pub] — crates/helix/src/residue.rs:121 CONFIRMED
- fn ResidueEncoder::new [pub] — crates/helix/src/residue.rs:130 CONFIRMED
- fn ResidueEncoder::encode [pub] — crates/helix/src/residue.rs:164 CONFIRMED
- fn ResidueEncoder::encode_signed [pub] — crates/helix/src/residue.rs:182 CONFIRMED
- fn ResidueEncoder::observe [pub] — crates/helix/src/residue.rs:207 CONFIRMED
- fn ResidueEncoder::roll [pub] — crates/helix/src/residue.rs:214 CONFIRMED
- fn ResidueEncoder::floor [pub] — crates/helix/src/residue.rs:219 CONFIRMED
- fn ResidueEncoder::total [pub] — crates/helix/src/residue.rs:224 CONFIRMED
- fn batch_fisher_z [pub] — crates/helix/src/simd.rs:24 CONFIRMED
- fn batch_l1_u8 [pub] — crates/helix/src/simd.rs:56 CONFIRMED

## Imports from in-scope crates

- none

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n22["helix"]
```
