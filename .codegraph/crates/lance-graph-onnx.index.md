# lance-graph-onnx

crate: lance-graph-onnx
path: crates/lance-graph-onnx
workspace: member
digest: 9dee13814165c2ea
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep lance-graph-mask-risc dev — crates/lance-graph-onnx/Cargo.toml:17 CONFIRMED
- dep lance-graph-r2il dev — crates/lance-graph-onnx/Cargo.toml:18 CONFIRMED
- dep lance-graph-relalg dev — crates/lance-graph-onnx/Cargo.toml:19 CONFIRMED
- dep ndarray normal — crates/lance-graph-onnx/Cargo.toml:12 CONFIRMED
- dep ogar-loco dev — crates/lance-graph-onnx/Cargo.toml:21 CONFIRMED

## Depended on by

- rdep lance-graph-xlog dev — crates/lance-graph-xlog/Cargo.toml:21 CONFIRMED
- rdep tract-harvest-probe normal — crates/tract-harvest-probe/Cargo.toml:22 CONFIRMED

## Public items

- mod bounded [pub] — crates/lance-graph-onnx/src/lib.rs:75 CONFIRMED
- fn mask_words [pub] — crates/lance-graph-onnx/src/lib.rs:110 CONFIRMED
- enum Refusal [pub] — crates/lance-graph-onnx/src/lib.rs:119 CONFIRMED
- trait Ieee [pub] — crates/lance-graph-onnx/src/lib.rs:139 CONFIRMED
- trait Sealed [pub] — crates/lance-graph-onnx/src/lib.rs:152 CONFIRMED
- fn broadcastable [pub] — crates/lance-graph-onnx/src/lib.rs:162 CONFIRMED
- fn neg [pub] — crates/lance-graph-onnx/src/lib.rs:222 CONFIRMED
- struct ReduceSum [pub] — crates/lance-graph-onnx/src/lib.rs:232 CONFIRMED
- struct Reduced [pub] — crates/lance-graph-onnx/src/lib.rs:239 CONFIRMED
- fn ReduceSum::new [pub] — crates/lance-graph-onnx/src/lib.rs:247 CONFIRMED
- fn ReduceSum::sum_i32 [pub] — crates/lance-graph-onnx/src/lib.rs:279 CONFIRMED
- fn ReduceSum::sum_where_greater_i32 [pub] — crates/lance-graph-onnx/src/lib.rs:288 CONFIRMED

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
  n56["lance-graph-onnx"]
  n93["tract-harvest-probe"]
  n93 --> n56
```
