# dr256-ternlog-lab

crate: dr256-ternlog-lab
path: crates/dr256-ternlog-lab
workspace: member
digest: 5d01f97ec6a64b83
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep lance-graph-contract normal — crates/dr256-ternlog-lab/Cargo.toml:9 CONFIRMED
- dep lance-graph-dt64 normal — crates/dr256-ternlog-lab/Cargo.toml:10 CONFIRMED
- dep lance-graph-mask-risc normal — crates/dr256-ternlog-lab/Cargo.toml:11 CONFIRMED
- dep weather-poc dev — crates/dr256-ternlog-lab/Cargo.toml:16 CONFIRMED

## Depended on by

- rdep lance-graph-benches normal — crates/lance-graph-benches/Cargo.toml:29 CONFIRMED

## Public items

- const SLAB_ROWS [pub] — crates/dr256-ternlog-lab/src/lib.rs:47 CONFIRMED
- const SLAB_WORDS [pub] — crates/dr256-ternlog-lab/src/lib.rs:49 CONFIRMED
- const MAX_SLABS [pub] — crates/dr256-ternlog-lab/src/lib.rs:51 CONFIRMED
- const AND3 [pub] — crates/dr256-ternlog-lab/src/lib.rs:53 CONFIRMED
- const MASK_BITS [pub] — crates/dr256-ternlog-lab/src/lib.rs:55 CONFIRMED
- const MASK_WORDS [pub] — crates/dr256-ternlog-lab/src/lib.rs:57 CONFIRMED
- const SLAB_MASKS [pub] — crates/dr256-ternlog-lab/src/lib.rs:59 CONFIRMED
- const PERSON [pub] — crates/dr256-ternlog-lab/src/lib.rs:64 CONFIRMED
- const FRIEND_OF [pub] — crates/dr256-ternlog-lab/src/lib.rs:66 CONFIRMED
- mod order_store [pub] — crates/dr256-ternlog-lab/src/lib.rs:70 CONFIRMED
- mod person [pub] — crates/dr256-ternlog-lab/src/lib.rs:73 CONFIRMED
- const ID [pub] — crates/dr256-ternlog-lab/src/lib.rs:74 CONFIRMED
- const NAME [pub] — crates/dr256-ternlog-lab/src/lib.rs:75 CONFIRMED
- const AGE [pub] — crates/dr256-ternlog-lab/src/lib.rs:76 CONFIRMED
- mod friend [pub] — crates/dr256-ternlog-lab/src/lib.rs:79 CONFIRMED
- const SRC [pub] — crates/dr256-ternlog-lab/src/lib.rs:80 CONFIRMED
- const DST [pub] — crates/dr256-ternlog-lab/src/lib.rs:81 CONFIRMED
- struct View [pub] — crates/dr256-ternlog-lab/src/lib.rs:85 CONFIRMED
- fn select [pub] — crates/dr256-ternlog-lab/src/lib.rs:130 CONFIRMED
- fn columns [pub] — crates/dr256-ternlog-lab/src/lib.rs:142 CONFIRMED
- fn lane_index [pub] — crates/dr256-ternlog-lab/src/lib.rs:155 CONFIRMED
- struct World [pub] — crates/dr256-ternlog-lab/src/lib.rs:163 CONFIRMED
- fn World::new [pub] — crates/dr256-ternlog-lab/src/lib.rs:177 CONFIRMED
- fn admits_reservation [pub] — crates/dr256-ternlog-lab/src/lib.rs:219 CONFIRMED
- struct Table [pub] — crates/dr256-ternlog-lab/src/lib.rs:242 CONFIRMED
- fn Table<'a>::new [pub] — crates/dr256-ternlog-lab/src/lib.rs:255 CONFIRMED
- fn Table<'a>::n_rows [pub] — crates/dr256-ternlog-lab/src/lib.rs:282 CONFIRMED
- fn Table<'a>::slabs [pub] — crates/dr256-ternlog-lab/src/lib.rs:286 CONFIRMED
- fn Table<'a>::slab_rows [pub] — crates/dr256-ternlog-lab/src/lib.rs:291 CONFIRMED
- fn Table<'a>::space [pub] — crates/dr256-ternlog-lab/src/lib.rs:296 CONFIRMED
- fn Table<'a>::frames [pub] — crates/dr256-ternlog-lab/src/lib.rs:303 CONFIRMED
- fn Table<'a>::advance_source [pub] — crates/dr256-ternlog-lab/src/lib.rs:309 CONFIRMED
- struct SlabSource [pub] — crates/dr256-ternlog-lab/src/lib.rs:321 CONFIRMED
- struct Plane [pub] — crates/dr256-ternlog-lab/src/lib.rs:371 CONFIRMED
- fn Plane::from_fn [pub] — crates/dr256-ternlog-lab/src/lib.rs:378 CONFIRMED
- fn Plane::count [pub] — crates/dr256-ternlog-lab/src/lib.rs:388 CONFIRMED
- fn Plane::masks [pub] — crates/dr256-ternlog-lab/src/lib.rs:394 CONFIRMED
- fn Plane::mask [pub] — crates/dr256-ternlog-lab/src/lib.rs:400 CONFIRMED
- fn Plane::selections [pub] — crates/dr256-ternlog-lab/src/lib.rs:415 CONFIRMED
- struct Selection [pub] — crates/dr256-ternlog-lab/src/lib.rs:428 CONFIRMED
- fn Selection<'p>::space [pub] — crates/dr256-ternlog-lab/src/lib.rs:436 CONFIRMED
- fn Selection<'p>::row [pub] — crates/dr256-ternlog-lab/src/lib.rs:441 CONFIRMED
- fn Selection<'p>::words [pub] — crates/dr256-ternlog-lab/src/lib.rs:446 CONFIRMED
- fn Selection<'p>::has [pub] — crates/dr256-ternlog-lab/src/lib.rs:451 CONFIRMED
- fn Selection<'p>::count [pub] — crates/dr256-ternlog-lab/src/lib.rs:458 CONFIRMED
- fn Selection<'p>::is_empty [pub] — crates/dr256-ternlog-lab/src/lib.rs:462 CONFIRMED
- fn Selection<'p>::same_bits [pub] — crates/dr256-ternlog-lab/src/lib.rs:468 CONFIRMED
- struct RankMap [pub] — crates/dr256-ternlog-lab/src/lib.rs:481 CONFIRMED
- fn RankMap::new [pub] — crates/dr256-ternlog-lab/src/lib.rs:491 CONFIRMED
- fn RankMap::len [pub] — crates/dr256-ternlog-lab/src/lib.rs:515 CONFIRMED
- fn RankMap::is_empty [pub] — crates/dr256-ternlog-lab/src/lib.rs:519 CONFIRMED
- fn RankMap::contains [pub] — crates/dr256-ternlog-lab/src/lib.rs:523 CONFIRMED
- fn RankMap::local [pub] — crates/dr256-ternlog-lab/src/lib.rs:531 CONFIRMED
- fn RankMap::words [pub] — crates/dr256-ternlog-lab/src/lib.rs:542 CONFIRMED
- enum Refused [pub] — crates/dr256-ternlog-lab/src/lib.rs:551 CONFIRMED
- struct Prepared [pub] — crates/dr256-ternlog-lab/src/lib.rs:596 CONFIRMED
- fn Prepared::slabs [pub] — crates/dr256-ternlog-lab/src/lib.rs:605 CONFIRMED
- fn Prepared::plane_bytes [pub] — crates/dr256-ternlog-lab/src/lib.rs:610 CONFIRMED
- fn Prepared::set_plane [pub] — crates/dr256-ternlog-lab/src/lib.rs:619 CONFIRMED
- fn Prepared::plane [pub] — crates/dr256-ternlog-lab/src/lib.rs:623 CONFIRMED
- fn prepare_nodes [pub] — crates/dr256-ternlog-lab/src/lib.rs:661 CONFIRMED
- fn prepare_edges [pub] — crates/dr256-ternlog-lab/src/lib.rs:700 CONFIRMED
- enum Agg [pub] — crates/dr256-ternlog-lab/src/lib.rs:758 CONFIRMED
- enum Answer [pub] — crates/dr256-ternlog-lab/src/lib.rs:767 CONFIRMED
- struct Fold [pub] — crates/dr256-ternlog-lab/src/lib.rs:777 CONFIRMED
- fn scratch_bytes_of [pub] — crates/dr256-ternlog-lab/src/lib.rs:788 CONFIRMED
- fn Fold::count_gt [pub] — crates/dr256-ternlog-lab/src/lib.rs:818 CONFIRMED
- fn Fold::and3 [pub] — crates/dr256-ternlog-lab/src/lib.rs:845 CONFIRMED
- fn Fold::between [pub] — crates/dr256-ternlog-lab/src/lib.rs:863 CONFIRMED
- fn Fold::count_and3 [pub] — crates/dr256-ternlog-lab/src/lib.rs:898 CONFIRMED
- fn Fold::scratch_bytes [pub] — crates/dr256-ternlog-lab/src/lib.rs:926 CONFIRMED
- fn Fold::count [pub] — crates/dr256-ternlog-lab/src/lib.rs:931 CONFIRMED
- fn Fold::answer [pub] — crates/dr256-ternlog-lab/src/lib.rs:946 CONFIRMED
- fn mean_round_half_up [pub] — crates/dr256-ternlog-lab/src/lib.rs:1028 CONFIRMED
- fn display_row [pub] — crates/dr256-ternlog-lab/src/lib.rs:1037 CONFIRMED
- struct OrderIndex [pub] — crates/dr256-ternlog-lab/src/lib.rs:1070 CONFIRMED
- fn OrderIndex::by_i32 [pub] — crates/dr256-ternlog-lab/src/lib.rs:1103 CONFIRMED
- fn OrderIndex::by_key [pub] — crates/dr256-ternlog-lab/src/lib.rs:1129 CONFIRMED
- fn OrderIndex::positions [pub] — crates/dr256-ternlog-lab/src/lib.rs:1151 CONFIRMED
- fn OrderIndex::len [pub] — crates/dr256-ternlog-lab/src/lib.rs:1159 CONFIRMED
- fn OrderIndex::is_empty [pub] — crates/dr256-ternlog-lab/src/lib.rs:1163 CONFIRMED
- fn OrderIndex::row [pub] — crates/dr256-ternlog-lab/src/lib.rs:1168 CONFIRMED
- fn OrderIndex::gt [pub] — crates/dr256-ternlog-lab/src/lib.rs:1173 CONFIRMED
- fn OrderIndex::between [pub] — crates/dr256-ternlog-lab/src/lib.rs:1178 CONFIRMED
- struct PreparedOrder [pub] — crates/dr256-ternlog-lab/src/lib.rs:1201 CONFIRMED
- fn prepare_order [pub] — crates/dr256-ternlog-lab/src/lib.rs:1210 CONFIRMED
- fn PreparedOrder::plane [pub] — crates/dr256-ternlog-lab/src/lib.rs:1259 CONFIRMED
- fn PreparedOrder::set_plane [pub] — crates/dr256-ternlog-lab/src/lib.rs:1263 CONFIRMED
- fn PreparedOrder::count [pub] — crates/dr256-ternlog-lab/src/lib.rs:1269 CONFIRMED
- const TIE_KEY_THEN_FRAME_POSITION [pub] — crates/dr256-ternlog-lab/src/order_store.rs:44 CONFIRMED
- mod layout [pub] — crates/dr256-ternlog-lab/src/order_store.rs:47 CONFIRMED
- const MAGIC [pub] — crates/dr256-ternlog-lab/src/order_store.rs:48 CONFIRMED
- const VERSION [pub] — crates/dr256-ternlog-lab/src/order_store.rs:49 CONFIRMED
- const OFF_MAGIC [pub] — crates/dr256-ternlog-lab/src/order_store.rs:50 CONFIRMED
- const OFF_VERSION [pub] — crates/dr256-ternlog-lab/src/order_store.rs:51 CONFIRMED
- const OFF_TIE_BREAK [pub] — crates/dr256-ternlog-lab/src/order_store.rs:52 CONFIRMED
- const OFF_CLASS [pub] — crates/dr256-ternlog-lab/src/order_store.rs:53 CONFIRMED
- const OFF_KEY_KIND [pub] — crates/dr256-ternlog-lab/src/order_store.rs:54 CONFIRMED
- const OFF_KEY_ID [pub] — crates/dr256-ternlog-lab/src/order_store.rs:55 CONFIRMED
- const OFF_TABLE_POPULATION [pub] — crates/dr256-ternlog-lab/src/order_store.rs:56 CONFIRMED
- const OFF_INDEX_POPULATION [pub] — crates/dr256-ternlog-lab/src/order_store.rs:57 CONFIRMED
- const OFF_N_ROWS [pub] — crates/dr256-ternlog-lab/src/order_store.rs:58 CONFIRMED
- const OFF_SOURCE [pub] — crates/dr256-ternlog-lab/src/order_store.rs:59 CONFIRMED
- const OFF_RELATION_MAP [pub] — crates/dr256-ternlog-lab/src/order_store.rs:60 CONFIRMED
- const OFF_COORDINATES [pub] — crates/dr256-ternlog-lab/src/order_store.rs:61 CONFIRMED
- const OFF_PRESENCE [pub] — crates/dr256-ternlog-lab/src/order_store.rs:62 CONFIRMED
- const HEADER_BYTES [pub] — crates/dr256-ternlog-lab/src/order_store.rs:63 CONFIRMED
- enum KeyLane [pub] — crates/dr256-ternlog-lab/src/order_store.rs:68 CONFIRMED
- enum StoreRefused [pub] — crates/dr256-ternlog-lab/src/order_store.rs:77 CONFIRMED
- struct SealedOrder [pub] — crates/dr256-ternlog-lab/src/order_store.rs:135 CONFIRMED
- fn SealedOrder::seal [pub] — crates/dr256-ternlog-lab/src/order_store.rs:177 CONFIRMED
- fn SealedOrder::to_bytes [pub] — crates/dr256-ternlog-lab/src/order_store.rs:216 CONFIRMED
- fn SealedOrder::from_bytes [pub] — crates/dr256-ternlog-lab/src/order_store.rs:249 CONFIRMED
- fn SealedOrder::write_to [pub] — crates/dr256-ternlog-lab/src/order_store.rs:346 CONFIRMED
- fn SealedOrder::read_from [pub] — crates/dr256-ternlog-lab/src/order_store.rs:350 CONFIRMED
- fn SealedOrder::attest [pub] — crates/dr256-ternlog-lab/src/order_store.rs:357 CONFIRMED
- fn SealedOrder::into_index [pub] — crates/dr256-ternlog-lab/src/order_store.rs:395 CONFIRMED
- fn SealedOrder::matches_index [pub] — crates/dr256-ternlog-lab/src/order_store.rs:412 CONFIRMED
- fn SealedOrder::rows [pub] — crates/dr256-ternlog-lab/src/order_store.rs:420 CONFIRMED
- fn SealedOrder::keys [pub] — crates/dr256-ternlog-lab/src/order_store.rs:425 CONFIRMED
- fn SealedOrder::len [pub] — crates/dr256-ternlog-lab/src/order_store.rs:429 CONFIRMED
- fn SealedOrder::is_empty [pub] — crates/dr256-ternlog-lab/src/order_store.rs:433 CONFIRMED
- fn SealedOrder::key [pub] — crates/dr256-ternlog-lab/src/order_store.rs:437 CONFIRMED
- fn SealedOrder::presence [pub] — crates/dr256-ternlog-lab/src/order_store.rs:441 CONFIRMED
- fn SealedOrder::source_revision [pub] — crates/dr256-ternlog-lab/src/order_store.rs:445 CONFIRMED

## Imports from in-scope crates

### lance-graph-contract
- use lance_graph_contract::class_view::ClassId — crates/dr256-ternlog-lab/src/lib.rs:33 CONFIRMED
- use lance_graph_contract::class_view::ClassView — crates/dr256-ternlog-lab/src/lib.rs:33 CONFIRMED
- use lance_graph_contract::class_view::WideFieldMask — crates/dr256-ternlog-lab/src/lib.rs:33 CONFIRMED
- use lance_graph_contract::ontology::DisplayTemplate — crates/dr256-ternlog-lab/src/lib.rs:34 CONFIRMED
- use lance_graph_contract::ontology::FieldRef — crates/dr256-ternlog-lab/src/lib.rs:34 CONFIRMED
### lance-graph-dt64
- use lance_graph_dt64::dr::Extent — crates/dr256-ternlog-lab/src/lib.rs:35 CONFIRMED
- use lance_graph_dt64::dr::Frame — crates/dr256-ternlog-lab/src/lib.rs:35 CONFIRMED
- use lance_graph_dt64::dr::FrameRevisions — crates/dr256-ternlog-lab/src/lib.rs:35 CONFIRMED
- use lance_graph_dt64::dr::FrameStore — crates/dr256-ternlog-lab/src/lib.rs:35 CONFIRMED
- use lance_graph_dt64::dr::Scope — crates/dr256-ternlog-lab/src/lib.rs:35 CONFIRMED
- use lance_graph_dt64::BindStore — crates/dr256-ternlog-lab/src/lib.rs:37 CONFIRMED
- use lance_graph_dt64::ClassSchema — crates/dr256-ternlog-lab/src/lib.rs:37 CONFIRMED
- use lance_graph_dt64::CoordSpace — crates/dr256-ternlog-lab/src/lib.rs:37 CONFIRMED
- use lance_graph_dt64::LaneAddr — crates/dr256-ternlog-lab/src/lib.rs:37 CONFIRMED
- use lance_graph_dt64::LaneBinding — crates/dr256-ternlog-lab/src/lib.rs:37 CONFIRMED
- use lance_graph_dt64::LaneRefusal — crates/dr256-ternlog-lab/src/lib.rs:37 CONFIRMED
- use lance_graph_dt64::LaneRequest — crates/dr256-ternlog-lab/src/lib.rs:37 CONFIRMED
- use lance_graph_dt64::LaneSource — crates/dr256-ternlog-lab/src/lib.rs:38 CONFIRMED
- use lance_graph_dt64::Refusal — crates/dr256-ternlog-lab/src/lib.rs:38 CONFIRMED
- use lance_graph_dt64::RouteAt — crates/dr256-ternlog-lab/src/lib.rs:38 CONFIRMED
- use lance_graph_dt64::SchemaRev — crates/dr256-ternlog-lab/src/lib.rs:38 CONFIRMED
- use lance_graph_dt64::dr::FrameRevisions — crates/dr256-ternlog-lab/src/order_store.rs:38 CONFIRMED
### lance-graph-mask-risc
- use lance_graph_mask_risc::exec::scratch_words_for — crates/dr256-ternlog-lab/src/lib.rs:40 CONFIRMED
- use lance_graph_mask_risc::exec::tile_words_for — crates/dr256-ternlog-lab/src/lib.rs:40 CONFIRMED
- use lance_graph_mask_risc::ExecError — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::Foreign — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::LaneRef — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::MaskOp — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::Operand — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::Out — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::Planes — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::Pred — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::execute_into — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::words_for — crates/dr256-ternlog-lab/src/lib.rs:42 CONFIRMED
- use lance_graph_mask_risc::Program — crates/dr256-ternlog-lab/src/lib.rs:43 CONFIRMED
- use lance_graph_mask_risc::Scratch — crates/dr256-ternlog-lab/src/lib.rs:43 CONFIRMED
- use lance_graph_mask_risc::Terminal — crates/dr256-ternlog-lab/src/lib.rs:43 CONFIRMED
- use lance_graph_mask_risc::Value — crates/dr256-ternlog-lab/src/lib.rs:43 CONFIRMED
- use lance_graph_mask_risc::LaneRef — crates/dr256-ternlog-lab/src/order_store.rs:39 CONFIRMED

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n15["dr256-ternlog-lab"]
  n38["lance-graph-benches"]
  n45["lance-graph-contract"]
  n48["lance-graph-dt64"]
  n53["lance-graph-mask-risc"]
  n15 --> n45
  n15 --> n48
  n15 --> n53
  n38 --> n15
```
