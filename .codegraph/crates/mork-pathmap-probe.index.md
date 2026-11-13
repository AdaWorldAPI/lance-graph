# mork-pathmap-probe

crate: mork-pathmap-probe
path: crates/mork-pathmap-probe
workspace: excluded
digest: 131fa9e0639e7027
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep lance-graph-mask-risc normal — crates/mork-pathmap-probe/Cargo.toml:37 CONFIRMED
- dep arrow-buffer normal — crates/mork-pathmap-probe/Cargo.toml:44 CONFIRMED
- dep mork normal — crates/mork-pathmap-probe/Cargo.toml:32 CONFIRMED
- dep mork-expr normal — crates/mork-pathmap-probe/Cargo.toml:33 CONFIRMED
- dep ndarray normal — crates/mork-pathmap-probe/Cargo.toml:41 CONFIRMED
- dep pathmap normal — crates/mork-pathmap-probe/Cargo.toml:34 CONFIRMED

## Depended on by

- rdep crepe-plan-probe normal — crates/crepe-plan-probe/Cargo.toml:24 CONFIRMED

## Public items

- enum Backend [pub] — crates/mork-pathmap-probe/src/boundary.rs:27 CONFIRMED
- enum Refusal [pub] — crates/mork-pathmap-probe/src/boundary.rs:43 CONFIRMED
- fn mork_join [pub] — crates/mork-pathmap-probe/src/boundary.rs:68 CONFIRMED
- fn capability [pub] — crates/mork-pathmap-probe/src/boundary.rs:84 CONFIRMED
- struct BoundIndex [pub] — crates/mork-pathmap-probe/src/boundary.rs:100 CONFIRMED
- fn bind [pub] — crates/mork-pathmap-probe/src/boundary.rs:112 CONFIRMED
- fn BoundIndex::backend [pub] — crates/mork-pathmap-probe/src/boundary.rs:138 CONFIRMED
- fn BoundIndex::answers [pub] — crates/mork-pathmap-probe/src/boundary.rs:144 CONFIRMED
- fn BoundIndex::native [pub] — crates/mork-pathmap-probe/src/boundary.rs:155 CONFIRMED
- struct SpogTrie [pub] — crates/mork-pathmap-probe/src/boundary.rs:289 CONFIRMED
- fn SpogTrie::build [pub] — crates/mork-pathmap-probe/src/boundary.rs:301 CONFIRMED
- fn SpogTrie::len [pub] — crates/mork-pathmap-probe/src/boundary.rs:310 CONFIRMED
- fn SpogTrie::is_empty [pub] — crates/mork-pathmap-probe/src/boundary.rs:315 CONFIRMED
- fn SpogTrie::key_bytes [pub] — crates/mork-pathmap-probe/src/boundary.rs:322 CONFIRMED
- fn SpogTrie::objects [pub] — crates/mork-pathmap-probe/src/boundary.rs:327 CONFIRMED
- fn SpogTrie::restrict_to [pub] — crates/mork-pathmap-probe/src/boundary.rs:342 CONFIRMED
- struct PostingTrie [pub] — crates/mork-pathmap-probe/src/boundary.rs:355 CONFIRMED
- fn PostingTrie::build [pub] — crates/mork-pathmap-probe/src/boundary.rs:359 CONFIRMED
- fn PostingTrie::docs [pub] — crates/mork-pathmap-probe/src/boundary.rs:371 CONFIRMED
- fn PostingTrie::intersect [pub] — crates/mork-pathmap-probe/src/boundary.rs:379 CONFIRMED
- enum OracleTerm [pub] — crates/mork-pathmap-probe/src/boundary.rs:415 CONFIRMED
- struct OracleAtom [pub] — crates/mork-pathmap-probe/src/boundary.rs:424 CONFIRMED
- struct OracleRule [pub] — crates/mork-pathmap-probe/src/boundary.rs:432 CONFIRMED
- struct OracleFacts [pub] — crates/mork-pathmap-probe/src/boundary.rs:440 CONFIRMED
- struct TupleSet [pub] — crates/mork-pathmap-probe/src/boundary.rs:448 CONFIRMED
- struct OracleCost [pub] — crates/mork-pathmap-probe/src/boundary.rs:454 CONFIRMED
- const MORK_MAX_CHILDREN [pub] — crates/mork-pathmap-probe/src/boundary.rs:523 CONFIRMED
- fn mork_rules [pub] — crates/mork-pathmap-probe/src/boundary.rs:580 CONFIRMED
- struct CountingIndex [pub] — crates/mork-pathmap-probe/src/chain.rs:51 CONFIRMED
- fn CountingIndex<'a,'f>::new [pub] — crates/mork-pathmap-probe/src/chain.rs:58 CONFIRMED
- fn fold_program_is_co_addressed [pub] — crates/mork-pathmap-probe/src/chain.rs:82 CONFIRMED
- fn is_co_addressed [pub] — crates/mork-pathmap-probe/src/chain.rs:89 CONFIRMED
- enum ChainRefusal [pub] — crates/mork-pathmap-probe/src/chain.rs:103 CONFIRMED
- enum Rebuild [pub] — crates/mork-pathmap-probe/src/chain.rs:112 CONFIRMED
- struct Frame [pub] — crates/mork-pathmap-probe/src/chain.rs:120 CONFIRMED
- struct Admission [pub] — crates/mork-pathmap-probe/src/chain.rs:132 CONFIRMED
- fn Frame::prepare [pub] — crates/mork-pathmap-probe/src/chain.rs:147 CONFIRMED
- fn Frame::slots [pub] — crates/mork-pathmap-probe/src/chain.rs:161 CONFIRMED
- fn Frame::admit [pub] — crates/mork-pathmap-probe/src/chain.rs:171 CONFIRMED
- struct Answer [pub] — crates/mork-pathmap-probe/src/chain.rs:236 CONFIRMED
- fn ask [pub] — crates/mork-pathmap-probe/src/chain.rs:250 CONFIRMED
- enum JoinRefusal [pub] — crates/mork-pathmap-probe/src/join.rs:55 CONFIRMED
- struct JoinDomain [pub] — crates/mork-pathmap-probe/src/join.rs:78 CONFIRMED
- const DENSE_SPAN [pub] — crates/mork-pathmap-probe/src/join.rs:85 CONFIRMED
- fn JoinDomain::build [pub] — crates/mork-pathmap-probe/src/join.rs:113 CONFIRMED
- fn JoinDomain::len [pub] — crates/mork-pathmap-probe/src/join.rs:164 CONFIRMED
- fn JoinDomain::is_empty [pub] — crates/mork-pathmap-probe/src/join.rs:168 CONFIRMED
- fn JoinDomain::slot_of [pub] — crates/mork-pathmap-probe/src/join.rs:173 CONFIRMED
- fn JoinDomain::is_dense [pub] — crates/mork-pathmap-probe/src/join.rs:181 CONFIRMED
- fn JoinDomain::contains [pub] — crates/mork-pathmap-probe/src/join.rs:186 CONFIRMED
- fn JoinDomain::id [pub] — crates/mork-pathmap-probe/src/join.rs:191 CONFIRMED
- fn JoinDomain::bytes [pub] — crates/mork-pathmap-probe/src/join.rs:195 CONFIRMED
- struct Side [pub] — crates/mork-pathmap-probe/src/join.rs:206 CONFIRMED
- struct Run [pub] — crates/mork-pathmap-probe/src/join.rs:219 CONFIRMED
- fn Run<'a>::len [pub] — crates/mork-pathmap-probe/src/join.rs:225 CONFIRMED
- fn Run<'a>::is_empty [pub] — crates/mork-pathmap-probe/src/join.rs:229 CONFIRMED
- fn Run<'a>::get [pub] — crates/mork-pathmap-probe/src/join.rs:234 CONFIRMED
- fn Run<'a>::iter [pub] — crates/mork-pathmap-probe/src/join.rs:238 CONFIRMED
- fn Run<'a>::count_eq [pub] — crates/mork-pathmap-probe/src/join.rs:244 CONFIRMED
- fn Run<'a>::prefetch [pub] — crates/mork-pathmap-probe/src/join.rs:250 CONFIRMED
- fn Side::prepare [pub] — crates/mork-pathmap-probe/src/join.rs:261 CONFIRMED
- fn Side::prepare_where [pub] — crates/mork-pathmap-probe/src/join.rs:274 CONFIRMED
- fn Side::at [pub] — crates/mork-pathmap-probe/src/join.rs:320 CONFIRMED
- fn Side::rows [pub] — crates/mork-pathmap-probe/src/join.rs:328 CONFIRMED
- fn Side::count [pub] — crates/mork-pathmap-probe/src/join.rs:333 CONFIRMED
- fn Side::present [pub] — crates/mork-pathmap-probe/src/join.rs:337 CONFIRMED
- fn Side::offsets [pub] — crates/mork-pathmap-probe/src/join.rs:343 CONFIRMED
- fn Side::positions [pub] — crates/mork-pathmap-probe/src/join.rs:348 CONFIRMED
- fn Side::lane [pub] — crates/mork-pathmap-probe/src/join.rs:354 CONFIRMED
- fn Side::bytes [pub] — crates/mork-pathmap-probe/src/join.rs:360 CONFIRMED
- struct JoinTable [pub] — crates/mork-pathmap-probe/src/join.rs:368 CONFIRMED
- type SelectedSide [pub] — crates/mork-pathmap-probe/src/join.rs:376 CONFIRMED
- enum Rows [pub] — crates/mork-pathmap-probe/src/join.rs:380 CONFIRMED
- fn rows [pub] — crates/mork-pathmap-probe/src/join.rs:391 CONFIRMED
- const R_KEY [pub] — crates/mork-pathmap-probe/src/join.rs:426 CONFIRMED
- const S_KEY [pub] — crates/mork-pathmap-probe/src/join.rs:427 CONFIRMED
- fn JoinTable::prepare [pub] — crates/mork-pathmap-probe/src/join.rs:431 CONFIRMED
- fn JoinTable::prepare_star [pub] — crates/mork-pathmap-probe/src/join.rs:437 CONFIRMED
- fn JoinTable::prepare_star_where [pub] — crates/mork-pathmap-probe/src/join.rs:444 CONFIRMED
- fn JoinTable::replace_side [pub] — crates/mork-pathmap-probe/src/join.rs:461 CONFIRMED
- fn JoinTable::check [pub] — crates/mork-pathmap-probe/src/join.rs:472 CONFIRMED
- fn JoinTable::bytes [pub] — crates/mork-pathmap-probe/src/join.rs:485 CONFIRMED
- fn JoinTable::words [pub] — crates/mork-pathmap-probe/src/join.rs:489 CONFIRMED
- struct JoinExec [pub] — crates/mork-pathmap-probe/src/join.rs:496 CONFIRMED
- fn JoinExec::new [pub] — crates/mork-pathmap-probe/src/join.rs:502 CONFIRMED
- fn JoinExec::joint [pub] — crates/mork-pathmap-probe/src/join.rs:512 CONFIRMED
- fn JoinExec::matching [pub] — crates/mork-pathmap-probe/src/join.rs:567 CONFIRMED
- fn JoinExec::for_each_answer [pub] — crates/mork-pathmap-probe/src/join.rs:606 CONFIRMED
- fn JoinExec::bag_count [pub] — crates/mork-pathmap-probe/src/join.rs:644 CONFIRMED
- struct JoinRow [pub] — crates/mork-pathmap-probe/src/join.rs:669 CONFIRMED
- fn JoinRow<'_>::x [pub] — crates/mork-pathmap-probe/src/join.rs:678 CONFIRMED
- fn JoinRow<'_>::y [pub] — crates/mork-pathmap-probe/src/join.rs:682 CONFIRMED
- fn JoinRow<'_>::z [pub] — crates/mork-pathmap-probe/src/join.rs:686 CONFIRMED
- fn for_each_set_ahead [pub] — crates/mork-pathmap-probe/src/join.rs:693 CONFIRMED
- fn for_each_set [pub] — crates/mork-pathmap-probe/src/join.rs:712 CONFIRMED
- fn filter_plane [pub] — crates/mork-pathmap-probe/src/join.rs:728 CONFIRMED
- mod boundary [pub] — crates/mork-pathmap-probe/src/lib.rs:29 CONFIRMED
- mod chain [pub] — crates/mork-pathmap-probe/src/lib.rs:30 CONFIRMED
- mod join [pub] — crates/mork-pathmap-probe/src/lib.rs:31 CONFIRMED
- mod reference [pub] — crates/mork-pathmap-probe/src/lib.rs:32 CONFIRMED
- mod rmap [pub] — crates/mork-pathmap-probe/src/lib.rs:33 CONFIRMED
- mod tile_gemm [pub] — crates/mork-pathmap-probe/src/lib.rs:34 CONFIRMED
- type Lane [pub] — crates/mork-pathmap-probe/src/lib.rs:40 CONFIRMED
- struct RelationView [pub] — crates/mork-pathmap-probe/src/lib.rs:47 CONFIRMED
- fn RelationView::new [pub] — crates/mork-pathmap-probe/src/lib.rs:53 CONFIRMED
- fn RelationView::from_rows [pub] — crates/mork-pathmap-probe/src/lib.rs:66 CONFIRMED
- fn RelationView::len [pub] — crates/mork-pathmap-probe/src/lib.rs:72 CONFIRMED
- fn RelationView::is_empty [pub] — crates/mork-pathmap-probe/src/lib.rs:76 CONFIRMED
- struct JoinInput [pub] — crates/mork-pathmap-probe/src/lib.rs:83 CONFIRMED
- fn JoinInput::generations [pub] — crates/mork-pathmap-probe/src/lib.rs:89 CONFIRMED
- type Row [pub] — crates/mork-pathmap-probe/src/lib.rs:95 CONFIRMED
- enum Semantics [pub] — crates/mork-pathmap-probe/src/lib.rs:99 CONFIRMED
- struct Answers [pub] — crates/mork-pathmap-probe/src/lib.rs:111 CONFIRMED
- fn Answers::to_set [pub] — crates/mork-pathmap-probe/src/lib.rs:116 CONFIRMED
- fn Answers::total [pub] — crates/mork-pathmap-probe/src/lib.rs:122 CONFIRMED
- fn matching [pub] — crates/mork-pathmap-probe/src/reference.rs:9 CONFIRMED
- fn bag_count [pub] — crates/mork-pathmap-probe/src/reference.rs:28 CONFIRMED
- type Fact [pub] — crates/mork-pathmap-probe/src/rmap.rs:56 CONFIRMED
- const SLOTS [pub] — crates/mork-pathmap-probe/src/rmap.rs:59 CONFIRMED
- const SLOT_WORDS [pub] — crates/mork-pathmap-probe/src/rmap.rs:61 CONFIRMED
- const TILE_ROWS [pub] — crates/mork-pathmap-probe/src/rmap.rs:63 CONFIRMED
- const TILE_WORDS [pub] — crates/mork-pathmap-probe/src/rmap.rs:65 CONFIRMED
- struct Witness [pub] — crates/mork-pathmap-probe/src/rmap.rs:75 CONFIRMED
- enum Refusal [pub] — crates/mork-pathmap-probe/src/rmap.rs:84 CONFIRMED
- fn Witness::check [pub] — crates/mork-pathmap-probe/src/rmap.rs:103 CONFIRMED
- struct SortedRun [pub] — crates/mork-pathmap-probe/src/rmap.rs:125 CONFIRMED
- fn SortedRun<'f>::build [pub] — crates/mork-pathmap-probe/src/rmap.rs:143 CONFIRMED
- fn SortedRun<'f>::range [pub] — crates/mork-pathmap-probe/src/rmap.rs:155 CONFIRMED
- fn SortedRun<'f>::neighbours [pub] — crates/mork-pathmap-probe/src/rmap.rs:166 CONFIRMED
- fn SortedRun<'f>::bytes [pub] — crates/mork-pathmap-probe/src/rmap.rs:175 CONFIRMED
- trait Navigate [pub] — crates/mork-pathmap-probe/src/rmap.rs:181 CONFIRMED
- struct Focus [pub] — crates/mork-pathmap-probe/src/rmap.rs:203 CONFIRMED
- fn Focus::bind [pub] — crates/mork-pathmap-probe/src/rmap.rs:212 CONFIRMED
- fn Focus::slot_of [pub] — crates/mork-pathmap-probe/src/rmap.rs:229 CONFIRMED
- fn Focus::id [pub] — crates/mork-pathmap-probe/src/rmap.rs:237 CONFIRMED
- fn Focus::len [pub] — crates/mork-pathmap-probe/src/rmap.rs:241 CONFIRMED
- fn Focus::is_empty [pub] — crates/mork-pathmap-probe/src/rmap.rs:245 CONFIRMED
- fn Focus::ids [pub] — crates/mork-pathmap-probe/src/rmap.rs:249 CONFIRMED
- fn Focus::replace [pub] — crates/mork-pathmap-probe/src/rmap.rs:254 CONFIRMED
- fn Focus::bytes [pub] — crates/mork-pathmap-probe/src/rmap.rs:263 CONFIRMED
- fn discover [pub] — crates/mork-pathmap-probe/src/rmap.rs:271 CONFIRMED
- type SlotSet [pub] — crates/mork-pathmap-probe/src/rmap.rs:301 CONFIRMED
- fn slot_set [pub] — crates/mork-pathmap-probe/src/rmap.rs:303 CONFIRMED
- fn slots_of [pub] — crates/mork-pathmap-probe/src/rmap.rs:311 CONFIRMED
- struct ScopedReach [pub] — crates/mork-pathmap-probe/src/rmap.rs:321 CONFIRMED
- fn ScopedReach::global [pub] — crates/mork-pathmap-probe/src/rmap.rs:332 CONFIRMED
- struct ScopedMask [pub] — crates/mork-pathmap-probe/src/rmap.rs:343 CONFIRMED
- fn ScopedMask::global [pub] — crates/mork-pathmap-probe/src/rmap.rs:350 CONFIRMED
- fn reach_direct [pub] — crates/mork-pathmap-probe/src/rmap.rs:361 CONFIRMED
- fn reach_direct_sources [pub] — crates/mork-pathmap-probe/src/rmap.rs:398 CONFIRMED
- struct CsrPred [pub] — crates/mork-pathmap-probe/src/rmap.rs:437 CONFIRMED
- struct Csr [pub] — crates/mork-pathmap-probe/src/rmap.rs:450 CONFIRMED
- fn Csr::prepare [pub] — crates/mork-pathmap-probe/src/rmap.rs:459 CONFIRMED
- fn Csr::edges [pub] — crates/mork-pathmap-probe/src/rmap.rs:495 CONFIRMED
- fn Csr::escapes [pub] — crates/mork-pathmap-probe/src/rmap.rs:500 CONFIRMED
- fn Csr::bytes [pub] — crates/mork-pathmap-probe/src/rmap.rs:505 CONFIRMED
- fn Csr::rows [pub] — crates/mork-pathmap-probe/src/rmap.rs:513 CONFIRMED
- fn Csr::escape_plane [pub] — crates/mork-pathmap-probe/src/rmap.rs:524 CONFIRMED
- fn Csr::escape_ids [pub] — crates/mork-pathmap-probe/src/rmap.rs:536 CONFIRMED
- fn Csr::reach_sources [pub] — crates/mork-pathmap-probe/src/rmap.rs:545 CONFIRMED
- fn Csr::reach [pub] — crates/mork-pathmap-probe/src/rmap.rs:595 CONFIRMED
- fn tile_addr [pub] — crates/mork-pathmap-probe/src/rmap.rs:632 CONFIRMED
- struct Tile [pub] — crates/mork-pathmap-probe/src/rmap.rs:638 CONFIRMED
- fn Tile::from_csr [pub] — crates/mork-pathmap-probe/src/rmap.rs:653 CONFIRMED
- fn Tile::bytes [pub] — crates/mork-pathmap-probe/src/rmap.rs:672 CONFIRMED
- fn Tile::plane [pub] — crates/mork-pathmap-probe/src/rmap.rs:676 CONFIRMED
- struct ThoughtDelta [pub] — crates/mork-pathmap-probe/src/rmap.rs:688 CONFIRMED
- fn ThoughtDelta::new [pub] — crates/mork-pathmap-probe/src/rmap.rs:699 CONFIRMED
- fn ThoughtDelta::bytes_sparse [pub] — crates/mork-pathmap-probe/src/rmap.rs:718 CONFIRMED
- fn ThoughtDelta::bytes_planes [pub] — crates/mork-pathmap-probe/src/rmap.rs:722 CONFIRMED
- struct TileExec [pub] — crates/mork-pathmap-probe/src/rmap.rs:729 CONFIRMED
- fn TileExec::new [pub] — crates/mork-pathmap-probe/src/rmap.rs:742 CONFIRMED
- fn TileExec::program [pub] — crates/mork-pathmap-probe/src/rmap.rs:755 CONFIRMED
- fn TileExec::reach [pub] — crates/mork-pathmap-probe/src/rmap.rs:838 CONFIRMED
- fn TileExec::reach_sources [pub] — crates/mork-pathmap-probe/src/rmap.rs:879 CONFIRMED
- struct FocusLanes [pub] — crates/mork-pathmap-probe/src/rmap.rs:926 CONFIRMED
- fn FocusLanes::load [pub] — crates/mork-pathmap-probe/src/rmap.rs:935 CONFIRMED
- fn FocusLanes::bytes [pub] — crates/mork-pathmap-probe/src/rmap.rs:945 CONFIRMED
- fn FocusLanes::count [pub] — crates/mork-pathmap-probe/src/rmap.rs:951 CONFIRMED
- fn rebind [pub] — crates/mork-pathmap-probe/src/rmap.rs:994 CONFIRMED
- fn Csr::edge_set [pub] — crates/mork-pathmap-probe/src/rmap.rs:1092 CONFIRMED
- fn Csr::escape_set [pub] — crates/mork-pathmap-probe/src/rmap.rs:1104 CONFIRMED
- enum Ordering [pub] — crates/mork-pathmap-probe/src/rmap.rs:1118 CONFIRMED
- struct FocusMask [pub] — crates/mork-pathmap-probe/src/rmap.rs:1127 CONFIRMED
- fn FocusMask::over [pub] — crates/mork-pathmap-probe/src/rmap.rs:1134 CONFIRMED
- struct Places [pub] — crates/mork-pathmap-probe/src/rmap.rs:1148 CONFIRMED
- struct BoxQuery [pub] — crates/mork-pathmap-probe/src/rmap.rs:1156 CONFIRMED
- fn box_program [pub] — crates/mork-pathmap-probe/src/rmap.rs:1163 CONFIRMED
- fn box_count [pub] — crates/mork-pathmap-probe/src/rmap.rs:1216 CONFIRMED
- fn box_count_by_id [pub] — crates/mork-pathmap-probe/src/rmap.rs:1243 CONFIRMED
- fn morton_order [pub] — crates/mork-pathmap-probe/src/rmap.rs:1256 CONFIRMED
- fn membership_planes [pub] — crates/mork-pathmap-probe/src/rmap.rs:1277 CONFIRMED
- fn intersect_planes [pub] — crates/mork-pathmap-probe/src/rmap.rs:1286 CONFIRMED
- fn intersect_count [pub] — crates/mork-pathmap-probe/src/rmap.rs:1316 CONFIRMED
- fn intersect_sorted [pub] — crates/mork-pathmap-probe/src/rmap.rs:1340 CONFIRMED
- struct Rng [pub] — crates/mork-pathmap-probe/src/rmap.rs:1359 CONFIRMED
- fn Rng::next_u64 [pub] — crates/mork-pathmap-probe/src/rmap.rs:1362 CONFIRMED
- fn Rng::below [pub] — crates/mork-pathmap-probe/src/rmap.rs:1370 CONFIRMED
- fn graph [pub] — crates/mork-pathmap-probe/src/rmap.rs:1379 CONFIRMED
- fn places [pub] — crates/mork-pathmap-probe/src/rmap.rs:1419 CONFIRMED
- truncated (223 total), see symbols.tsv

## Imports from in-scope crates

### lance-graph-mask-risc
- use lance_graph_mask_risc::Foreign — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::MaskOp — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::Operand — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::Out — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::Planes — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::Program — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::Scratch — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::Terminal — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::execute_into — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::words_for — crates/mork-pathmap-probe/src/join.rs:48 CONFIRMED
- use lance_graph_mask_risc::Foreign — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::LaneRef — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::MaskOp — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::Operand — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::Out — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::Planes — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::Pred — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::Program — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::Scratch — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::Terminal — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::Value — crates/mork-pathmap-probe/src/rmap.rs:49 CONFIRMED
- use lance_graph_mask_risc::execute — crates/mork-pathmap-probe/src/rmap.rs:50 CONFIRMED
- use lance_graph_mask_risc::execute_into — crates/mork-pathmap-probe/src/rmap.rs:50 CONFIRMED
- use lance_graph_mask_risc::words_for — crates/mork-pathmap-probe/src/rmap.rs:50 CONFIRMED

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n9["crepe-plan-probe"]
  n54["lance-graph-mask-risc"]
  n75["mork-pathmap-probe"]
  n9 --> n75
  n75 --> n54
```
