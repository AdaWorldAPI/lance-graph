# view: LaneSource

Generated; the code is the truth. Same-name symbols stay separate per file.

## Definitions

### lance-graph-dt64 crates/lance-graph-dt64/src/lane.rs
- trait LaneSource [pub] — crates/lance-graph-dt64/src/lane.rs:51 CONFIRMED

## Impls mentioning it

- impl LaneSource for U8Socket<'_> — crates/dr-isa-stream-lab/src/int8.rs:159 POSSIBLE
- impl LaneSource for Socket<'_> — crates/dr-isa-stream-lab/src/source.rs:225 POSSIBLE
- impl LaneSource for Network — crates/foundry-parity-lab/src/source.rs:178 POSSIBLE
- impl LaneSource for AbiLanes<'_> — crates/lance-graph-report/src/lane_source.rs:143 POSSIBLE
- impl LaneSource for U64Source<'s> — crates/soa64-u64-lab/src/socket.rs:136 POSSIBLE

## Imports (not resolved to a definition)

- use lance_graph_dt64::LaneSource — crates/dr-isa-stream-lab/src/bind.rs:7 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/dr-isa-stream-lab/src/int8.rs:26 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/dr-isa-stream-lab/src/shape.rs:49 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/dr-isa-stream-lab/src/source.rs:12 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/foundry-parity-lab/src/bin/bench.rs:15 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/foundry-parity-lab/src/bind.rs:8 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/foundry-parity-lab/src/method.rs:36 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/foundry-parity-lab/src/source.rs:9 POSSIBLE
- pub-use lane::LaneSource — crates/lance-graph-dt64/src/lib.rs:108 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/lance-graph-report/src/lane_source.rs:39 POSSIBLE
- use lance_graph_dt64::LaneSource — crates/soa64-u64-lab/src/socket.rs:15 POSSIBLE

## Textual occurrences (37 total)

- crates/dr-isa-stream-lab/src/bind.rs:7 POSSIBLE
- crates/dr-isa-stream-lab/src/bind.rs:103 POSSIBLE
- crates/dr-isa-stream-lab/src/int8.rs:26 POSSIBLE
- crates/dr-isa-stream-lab/src/int8.rs:159 POSSIBLE
- crates/dr-isa-stream-lab/src/lib.rs:8 POSSIBLE
- crates/dr-isa-stream-lab/src/lib.rs:20 POSSIBLE
- crates/dr-isa-stream-lab/src/shape.rs:49 POSSIBLE
- crates/dr-isa-stream-lab/src/shape.rs:223 POSSIBLE
- crates/dr-isa-stream-lab/src/source.rs:12 POSSIBLE
- crates/dr-isa-stream-lab/src/source.rs:225 POSSIBLE
- crates/foundry-parity-lab/src/arm_b.rs:149 POSSIBLE
- crates/foundry-parity-lab/src/bin/bench.rs:15 POSSIBLE
- crates/foundry-parity-lab/src/bind.rs:8 POSSIBLE
- crates/foundry-parity-lab/src/lib.rs:11 POSSIBLE
- crates/foundry-parity-lab/src/method.rs:36 POSSIBLE
- crates/foundry-parity-lab/src/source.rs:3 POSSIBLE
- crates/foundry-parity-lab/src/source.rs:9 POSSIBLE
- crates/foundry-parity-lab/src/source.rs:178 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:6 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:51 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:171 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:220 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:307 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:352 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:379 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:396 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:401 POSSIBLE
- crates/lance-graph-dt64/src/lane.rs:420 POSSIBLE
- crates/lance-graph-dt64/src/lib.rs:108 POSSIBLE
- crates/lance-graph-report/src/lane_source.rs:3 POSSIBLE
- crates/lance-graph-report/src/lane_source.rs:39 POSSIBLE
- crates/lance-graph-report/src/lane_source.rs:87 POSSIBLE
- crates/lance-graph-report/src/lane_source.rs:143 POSSIBLE
- crates/lance-graph-report/src/lane_source/tests.rs:543 POSSIBLE
- crates/soa64-u64-lab/src/arms.rs:169 POSSIBLE
- crates/soa64-u64-lab/src/socket.rs:15 POSSIBLE
- crates/soa64-u64-lab/src/socket.rs:136 POSSIBLE

## Architecture

- arch LaneSource implemented-by lance-graph-report::AbiLanes [CONFIRMED] validation=OK — crates/lance-graph-report/src/lane_source.rs:143 CONFIRMED
