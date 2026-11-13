# lance-graph-mailbox-lanes

crate: lance-graph-mailbox-lanes
path: crates/lance-graph-mailbox-lanes
workspace: member
digest: 18583aadb0b0f0c3
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep causal-edge dev — crates/lance-graph-mailbox-lanes/Cargo.toml:15 CONFIRMED
- dep cognitive-shader-driver normal — crates/lance-graph-mailbox-lanes/Cargo.toml:9 CONFIRMED
- dep lance-graph-contract normal — crates/lance-graph-mailbox-lanes/Cargo.toml:10 CONFIRMED
- dep lance-graph-dt64 normal — crates/lance-graph-mailbox-lanes/Cargo.toml:11 CONFIRMED
- dep lance-graph-mask-risc normal — crates/lance-graph-mailbox-lanes/Cargo.toml:12 CONFIRMED

## Depended on by


## Public items

- enum Column [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:40 CONFIRMED
- enum ColumnRefusal [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:55 CONFIRMED
- const Column::ALL [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:64 CONFIRMED
- fn Column::iri [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:74 CONFIRMED
- fn Column::from_iri [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:85 CONFIRMED
- struct MailboxLanes [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:109 CONFIRMED
- fn MailboxLanes<'a,N>::new [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:119 CONFIRMED
- fn MailboxLanes<'a,N>::frame_store [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:135 CONFIRMED
- fn MailboxLanes<'a,N>::seal [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:141 CONFIRMED
- fn MailboxLanes<'a,N>::space [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:147 CONFIRMED
- fn MailboxLanes<'a,N>::rows [pub] — crates/lance-graph-mailbox-lanes/src/lib.rs:156 CONFIRMED

## Imports from in-scope crates

### cognitive-shader-driver
- use cognitive_shader_driver::mailbox_soa::Published — crates/lance-graph-mailbox-lanes/src/lib.rs:30 CONFIRMED
### lance-graph-contract
- use lance_graph_contract::ontology::FieldRef — crates/lance-graph-mailbox-lanes/src/lib.rs:31 CONFIRMED
- use lance_graph_contract::soa_view::MailboxSoaView — crates/lance-graph-mailbox-lanes/src/lib.rs:32 CONFIRMED
### lance-graph-dt64
- use lance_graph_dt64::dr::Frame — crates/lance-graph-mailbox-lanes/src/lib.rs:33 CONFIRMED
- use lance_graph_dt64::dr::FrameRevisions — crates/lance-graph-mailbox-lanes/src/lib.rs:33 CONFIRMED
- use lance_graph_dt64::dr::FrameStore — crates/lance-graph-mailbox-lanes/src/lib.rs:33 CONFIRMED
- use lance_graph_dt64::dr::Scope — crates/lance-graph-mailbox-lanes/src/lib.rs:33 CONFIRMED
- use lance_graph_dt64::CoordSpace — crates/lance-graph-mailbox-lanes/src/lib.rs:34 CONFIRMED
- use lance_graph_dt64::LaneAddr — crates/lance-graph-mailbox-lanes/src/lib.rs:34 CONFIRMED
- use lance_graph_dt64::LaneSource — crates/lance-graph-mailbox-lanes/src/lib.rs:34 CONFIRMED
- use lance_graph_dt64::SchemaRev — crates/lance-graph-mailbox-lanes/src/lib.rs:34 CONFIRMED
### lance-graph-mask-risc
- use lance_graph_mask_risc::LaneRef — crates/lance-graph-mailbox-lanes/src/lib.rs:35 CONFIRMED

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n7["cognitive-shader-driver"]
  n45["lance-graph-contract"]
  n48["lance-graph-dt64"]
  n53["lance-graph-mailbox-lanes"]
  n54["lance-graph-mask-risc"]
  n53 --> n7
  n53 --> n45
  n53 --> n48
  n53 --> n54
```
