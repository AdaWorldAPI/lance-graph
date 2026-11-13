# dr-select-k-lab

crate: dr-select-k-lab
path: crates/dr-select-k-lab
workspace: excluded
digest: c76b4e75d87f8356
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep lance-graph-mask-risc normal — crates/dr-select-k-lab/Cargo.toml:11 CONFIRMED
- dep ndarray dev — crates/dr-select-k-lab/Cargo.toml:21 CONFIRMED

## Depended on by


## Public items

- enum Order [pub] — crates/dr-select-k-lab/src/bind.rs:9 CONFIRMED
- enum BindRefusal [pub] — crates/dr-select-k-lab/src/bind.rs:23 CONFIRMED
- struct Bound [pub] — crates/dr-select-k-lab/src/bind.rs:30 CONFIRMED
- fn Bound::bind [pub] — crates/dr-select-k-lab/src/bind.rs:40 CONFIRMED
- fn Bound::k [pub] — crates/dr-select-k-lab/src/bind.rs:61 CONFIRMED
- fn Bound::version [pub] — crates/dr-select-k-lab/src/bind.rs:65 CONFIRMED
- fn Bound::order [pub] — crates/dr-select-k-lab/src/bind.rs:69 CONFIRMED
- fn Bound::rows [pub] — crates/dr-select-k-lab/src/bind.rs:74 CONFIRMED
- fn Bound::staged [pub] — crates/dr-select-k-lab/src/bind.rs:79 CONFIRMED
- enum Residual [pub] — crates/dr-select-k-lab/src/bind.rs:86 CONFIRMED
- enum View [pub] — crates/dr-select-k-lab/src/bind.rs:97 CONFIRMED
- fn Residual::view [pub] — crates/dr-select-k-lab/src/bind.rs:104 CONFIRMED
- fn Residual::retained_bytes [pub] — crates/dr-select-k-lab/src/bind.rs:113 CONFIRMED
- struct Counters [pub] — crates/dr-select-k-lab/src/counters.rs:8 CONFIRMED
- fn Counters::new [pub] — crates/dr-select-k-lab/src/counters.rs:22 CONFIRMED
- fn Counters::bump [pub(crate)] — crates/dr-select-k-lab/src/counters.rs:26 CONFIRMED
- fn Counters::snapshot [pub] — crates/dr-select-k-lab/src/counters.rs:32 CONFIRMED
- enum Pattern [pub] — crates/dr-select-k-lab/src/fixture.rs:7 CONFIRMED
- fn source [pub] — crates/dr-select-k-lab/src/fixture.rs:32 CONFIRMED
- mod bind [pub] — crates/dr-select-k-lab/src/lib.rs:27 CONFIRMED
- mod counters [pub] — crates/dr-select-k-lab/src/lib.rs:28 CONFIRMED
- mod fixture [pub] — crates/dr-select-k-lab/src/lib.rs:29 CONFIRMED
- mod ops [pub] — crates/dr-select-k-lab/src/lib.rs:30 CONFIRMED
- mod oracle [pub] — crates/dr-select-k-lab/src/lib.rs:31 CONFIRMED
- mod source [pub] — crates/dr-select-k-lab/src/lib.rs:32 CONFIRMED
- enum Op [pub] — crates/dr-select-k-lab/src/ops.rs:76 CONFIRMED
- enum Inadmissible [pub] — crates/dr-select-k-lab/src/ops.rs:88 CONFIRMED
- fn admissible [pub] — crates/dr-select-k-lab/src/ops.rs:99 CONFIRMED
- fn residual_a_ge [pub] — crates/dr-select-k-lab/src/ops.rs:121 CONFIRMED
- struct PreparedFolds [pub] — crates/dr-select-k-lab/src/ops.rs:151 CONFIRMED
- fn PreparedFolds::new [pub] — crates/dr-select-k-lab/src/ops.rs:169 CONFIRMED
- fn PreparedFolds::count_b_gt [pub] — crates/dr-select-k-lab/src/ops.rs:195 CONFIRMED
- fn PreparedFolds::sum_a [pub] — crates/dr-select-k-lab/src/ops.rs:248 CONFIRMED
- fn PreparedFolds::top_by_key [pub] — crates/dr-select-k-lab/src/ops.rs:282 CONFIRMED
- mod baseline [pub] — crates/dr-select-k-lab/src/ops.rs:326 CONFIRMED
- fn count_b_gt [pub] — crates/dr-select-k-lab/src/ops.rs:333 CONFIRMED
- fn sum_a [pub] — crates/dr-select-k-lab/src/ops.rs:351 CONFIRMED
- struct Query [pub] — crates/dr-select-k-lab/src/oracle.rs:11 CONFIRMED
- fn count_b_gt [pub] — crates/dr-select-k-lab/src/oracle.rs:18 CONFIRMED
- fn sum_a [pub] — crates/dr-select-k-lab/src/oracle.rs:30 CONFIRMED
- fn top_by_key [pub] — crates/dr-select-k-lab/src/oracle.rs:48 CONFIRMED
- struct Version [pub] — crates/dr-select-k-lab/src/source.rs:11 CONFIRMED
- struct KeyRange [pub] — crates/dr-select-k-lab/src/source.rs:15 CONFIRMED
- fn KeyRange::has [pub] — crates/dr-select-k-lab/src/source.rs:21 CONFIRMED
- struct Selection [pub] — crates/dr-select-k-lab/src/source.rs:29 CONFIRMED
- struct Staged [pub] — crates/dr-select-k-lab/src/source.rs:37 CONFIRMED
- struct Source [pub] — crates/dr-select-k-lab/src/source.rs:45 CONFIRMED
- fn Source::new [pub] — crates/dr-select-k-lab/src/source.rs:60 CONFIRMED
- fn Source::version [pub] — crates/dr-select-k-lab/src/source.rs:84 CONFIRMED
- fn Source::n_rows [pub] — crates/dr-select-k-lab/src/source.rs:88 CONFIRMED
- fn Source::advance [pub] — crates/dr-select-k-lab/src/source.rs:93 CONFIRMED
- fn Source::select_filter [pub] — crates/dr-select-k-lab/src/source.rs:99 CONFIRMED
- fn Source::select_ordered [pub] — crates/dr-select-k-lab/src/source.rs:119 CONFIRMED
- fn Source::take [pub(crate)] — crates/dr-select-k-lab/src/source.rs:138 CONFIRMED
- fn Source::lanes [pub] — crates/dr-select-k-lab/src/source.rs:156 CONFIRMED

## Imports from in-scope crates

### lance-graph-mask-risc
- use lance_graph_mask_risc::Foreign — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::LaneRef — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::MaskOp — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::Operand — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::Out — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::Planes — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::Pred — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::Program — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::execute_into — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::words_for — crates/dr-select-k-lab/src/ops.rs:9 CONFIRMED
- use lance_graph_mask_risc::Scratch — crates/dr-select-k-lab/src/ops.rs:10 CONFIRMED
- use lance_graph_mask_risc::Terminal — crates/dr-select-k-lab/src/ops.rs:10 CONFIRMED
- use lance_graph_mask_risc::Value — crates/dr-select-k-lab/src/ops.rs:10 CONFIRMED
- use lance_graph_mask_risc::Scratch — crates/dr-select-k-lab/src/ops.rs:329 CONFIRMED
- use lance_graph_mask_risc::Value — crates/dr-select-k-lab/src/ops.rs:329 CONFIRMED

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n14["dr-select-k-lab"]
  n53["lance-graph-mask-risc"]
  n14 --> n53
```
