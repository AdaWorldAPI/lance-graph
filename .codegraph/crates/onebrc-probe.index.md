# onebrc-probe

crate: onebrc-probe
path: crates/onebrc-probe
workspace: excluded
digest: f1c0ac8eb010404b
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep lance-graph-contract normal optional — crates/onebrc-probe/Cargo.toml:71 CONFIRMED
- dep ndarray normal optional — crates/onebrc-probe/Cargo.toml:62 CONFIRMED
- dep ractor normal optional — crates/onebrc-probe/Cargo.toml:65 CONFIRMED
- dep tokio normal optional — crates/onebrc-probe/Cargo.toml:66 CONFIRMED

## Depended on by


## Public items

- const STATION_COUNT [pub] — crates/onebrc-probe/src/gen.rs:17 CONFIRMED
- struct GenResult [pub] — crates/onebrc-probe/src/gen.rs:22 CONFIRMED
- fn station_names [pub] — crates/onebrc-probe/src/gen.rs:61 CONFIRMED
- fn gen [pub] — crates/onebrc-probe/src/gen.rs:106 CONFIRMED
- const SIMD_LANES [pub(crate)] cfg=all(feature="lane-b", test) — crates/onebrc-probe/src/lane_b.rs:73 CONFIRMED
- fn lane_b_simd [pub] cfg=feature="lane-b" — crates/onebrc-probe/src/lane_b.rs:80 CONFIRMED
- enum ChunkMsg [pub] cfg=feature="lane-d" — crates/onebrc-probe/src/lane_d.rs:42 CONFIRMED
- struct ChunkWorker [pub] cfg=feature="lane-d" — crates/onebrc-probe/src/lane_d.rs:56 CONFIRMED
- fn lane_d_ractor [pub] cfg=feature="lane-d" — crates/onebrc-probe/src/lane_d.rs:100 CONFIRMED
- fn lane_e_kanban [pub] cfg=feature="lane-e" — crates/onebrc-probe/src/lane_e.rs:136 CONFIRMED
- const SLOTS [pub(crate)] — crates/onebrc-probe/src/lane_f.rs:65 CONFIRMED
- fn fnv1a64 [pub(crate)] — crates/onebrc-probe/src/lane_f.rs:69 CONFIRMED
- fn morton_slot [pub(crate)] — crates/onebrc-probe/src/lane_f.rs:83 CONFIRMED
- struct SoaTable [pub(crate)] — crates/onebrc-probe/src/lane_f.rs:101 CONFIRMED
- fn SoaTable::new [pub(crate)] — crates/onebrc-probe/src/lane_f.rs:118 CONFIRMED
- fn SoaTable::observe [pub(crate)] — crates/onebrc-probe/src/lane_f.rs:134 CONFIRMED
- fn table_to_map [pub(crate)] — crates/onebrc-probe/src/lane_f.rs:196 CONFIRMED
- fn lane_f_morton [pub] — crates/onebrc-probe/src/lane_f.rs:244 CONFIRMED
- fn lane_r_radix [pub] — crates/onebrc-probe/src/lane_f.rs:250 CONFIRMED
- const DEFAULT_MORSEL_ROWS [pub] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:81 CONFIRMED
- struct MorselEntry [pub(crate)] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:84 CONFIRMED
- enum ShardMsg [pub(crate)] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:91 CONFIRMED
- struct OwnerSoa [pub(crate)] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:114 CONFIRMED
- fn OwnerSoa::for_span [pub(crate)] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:125 CONFIRMED
- struct ShardState [pub(crate)] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:191 CONFIRMED
- struct ShardOwner [pub(crate)] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:201 CONFIRMED
- fn worker_scan_grouped [pub(crate)] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:369 CONFIRMED
- fn lane_g_kanban_soa_with_morsel [pub] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:445 CONFIRMED
- fn lane_g_kanban_soa [pub] cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:527 CONFIRMED
- const DEFAULT_BATCH_K [pub] cfg=feature="lane-h" — crates/onebrc-probe/src/lane_h.rs:74 CONFIRMED
- fn lane_h_orchestrated_with [pub] cfg=feature="lane-h" — crates/onebrc-probe/src/lane_h.rs:223 CONFIRMED
- fn lane_h_orchestrated [pub] cfg=feature="lane-h" — crates/onebrc-probe/src/lane_h.rs:344 CONFIRMED
- const BATCH_ROWS [pub] cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:68 CONFIRMED
- struct BatchTable [pub(crate)] cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:160 CONFIRMED
- struct RowOwner [pub(crate)] cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:218 CONFIRMED
- fn lane_i_batch_pipeline_with [pub] cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:547 CONFIRMED
- fn lane_i_batch_pipeline [pub] cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:651 CONFIRMED
- struct GridBatch [pub] cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:137 CONFIRMED
- struct GridlakeColumns [pub] cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:202 CONFIRMED
- enum GridlakeCarrierError [pub] cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:217 CONFIRMED
- fn GridBatch::as_gridlake_columns [pub] cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:246 CONFIRMED
- fn lane_j_grid_pipeline_with [pub] cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:501 CONFIRMED
- fn lane_j_grid_pipeline [pub] cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:683 CONFIRMED
- fn lane_s_swar [pub] — crates/onebrc-probe/src/lane_s.rs:107 CONFIRMED
- fn lane_t_trie [pub] — crates/onebrc-probe/src/lane_t.rs:196 CONFIRMED
- fn lane_t_byte [pub] — crates/onebrc-probe/src/lane_t.rs:202 CONFIRMED
- mod gen [pub] — crates/onebrc-probe/src/lib.rs:52 CONFIRMED
- mod lane_b [pub] cfg=feature="lane-b" — crates/onebrc-probe/src/lib.rs:54 CONFIRMED
- mod lane_d [pub] cfg=feature="lane-d" — crates/onebrc-probe/src/lib.rs:56 CONFIRMED
- mod lane_e [pub] cfg=feature="lane-e" — crates/onebrc-probe/src/lib.rs:58 CONFIRMED
- mod lane_f [pub] — crates/onebrc-probe/src/lib.rs:59 CONFIRMED
- mod lane_g [pub] cfg=feature="lane-g" — crates/onebrc-probe/src/lib.rs:61 CONFIRMED
- mod lane_h [pub] cfg=feature="lane-h" — crates/onebrc-probe/src/lib.rs:63 CONFIRMED
- mod lane_i [pub] cfg=feature="lane-i" — crates/onebrc-probe/src/lib.rs:65 CONFIRMED
- mod lane_j [pub] cfg=feature="lane-j" — crates/onebrc-probe/src/lib.rs:67 CONFIRMED
- mod lane_s [pub] — crates/onebrc-probe/src/lib.rs:68 CONFIRMED
- mod lane_t [pub] — crates/onebrc-probe/src/lib.rs:69 CONFIRMED
- mod presets [pub] cfg=feature="presets" — crates/onebrc-probe/src/lib.rs:71 CONFIRMED
- mod sha256 [pub] — crates/onebrc-probe/src/lib.rs:72 CONFIRMED
- struct Stats [pub] — crates/onebrc-probe/src/lib.rs:104 CONFIRMED
- fn Stats::single [pub] — crates/onebrc-probe/src/lib.rs:113 CONFIRMED
- fn Stats::observe [pub] — crates/onebrc-probe/src/lib.rs:123 CONFIRMED
- fn Stats::merge [pub] — crates/onebrc-probe/src/lib.rs:137 CONFIRMED
- fn Stats::mean_tenths [pub] — crates/onebrc-probe/src/lib.rs:149 CONFIRMED
- fn lane_a_scalar [pub] — crates/onebrc-probe/src/lib.rs:186 CONFIRMED
- fn chunk_bounds [pub] — crates/onebrc-probe/src/lib.rs:219 CONFIRMED
- fn lane_c_threads [pub] — crates/onebrc-probe/src/lib.rs:269 CONFIRMED
- struct Preset [pub] cfg=feature="presets" — crates/onebrc-probe/src/presets.rs:18 CONFIRMED
- const PRESETS [pub] cfg=feature="presets" — crates/onebrc-probe/src/presets.rs:29 CONFIRMED
- fn run_preset [pub] cfg=feature="presets" — crates/onebrc-probe/src/presets.rs:101 CONFIRMED
- struct Sha256 [pub] — crates/onebrc-probe/src/sha256.rs:26 CONFIRMED
- fn Sha256::new [pub] — crates/onebrc-probe/src/sha256.rs:40 CONFIRMED
- fn Sha256::update [pub] — crates/onebrc-probe/src/sha256.rs:50 CONFIRMED
- fn Sha256::finalize [pub] — crates/onebrc-probe/src/sha256.rs:85 CONFIRMED
- fn Sha256::hex [pub] — crates/onebrc-probe/src/sha256.rs:102 CONFIRMED
- fn hash_hex [pub] — crates/onebrc-probe/src/sha256.rs:165 CONFIRMED

## Imports from in-scope crates

### lance-graph-contract
- use lance_graph_contract::collapse_gate::MailboxId cfg=feature="lane-e" — crates/onebrc-probe/src/lane_e.rs:40 CONFIRMED
- use lance_graph_contract::kanban::ExecTarget cfg=feature="lane-e" — crates/onebrc-probe/src/lane_e.rs:41 CONFIRMED
- use lance_graph_contract::kanban::KanbanColumn cfg=feature="lane-e" — crates/onebrc-probe/src/lane_e.rs:41 CONFIRMED
- use lance_graph_contract::kanban::KanbanMove cfg=feature="lane-e" — crates/onebrc-probe/src/lane_e.rs:41 CONFIRMED
- use lance_graph_contract::soa_view::MailboxSoaOwner cfg=feature="lane-e" — crates/onebrc-probe/src/lane_e.rs:42 CONFIRMED
- use lance_graph_contract::soa_view::MailboxSoaView cfg=feature="lane-e" — crates/onebrc-probe/src/lane_e.rs:42 CONFIRMED
- use lance_graph_contract::collapse_gate::MailboxId cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:72 CONFIRMED
- use lance_graph_contract::kanban::ExecTarget cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:73 CONFIRMED
- use lance_graph_contract::kanban::KanbanColumn cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:73 CONFIRMED
- use lance_graph_contract::kanban::KanbanMove cfg=feature="lane-g" — crates/onebrc-probe/src/lane_g.rs:73 CONFIRMED
- use lance_graph_contract::collapse_gate::MailboxId cfg=feature="lane-h" — crates/onebrc-probe/src/lane_h.rs:66 CONFIRMED
- use lance_graph_contract::collapse_gate::MailboxId cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:58 CONFIRMED
- use lance_graph_contract::kanban::ExecTarget cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:59 CONFIRMED
- use lance_graph_contract::kanban::KanbanColumn cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:59 CONFIRMED
- use lance_graph_contract::kanban::KanbanMove cfg=feature="lane-i" — crates/onebrc-probe/src/lane_i.rs:59 CONFIRMED
- use lance_graph_contract::kanban::ExecTarget cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:45 CONFIRMED
- use lance_graph_contract::kanban::KanbanColumn cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:45 CONFIRMED
- use lance_graph_contract::kanban::KanbanMove cfg=feature="lane-j" — crates/onebrc-probe/src/lane_j.rs:45 CONFIRMED

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n39["lance-graph-contract"]
  n71["onebrc-probe"]
  n71 --> n39
```
