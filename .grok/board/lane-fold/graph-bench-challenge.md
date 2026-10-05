# Graph Bench Challenge (GBC)

Rust-only, single machine, wall clock. Modeled on the 1BRC: one input, one checksummed output, no partial credit for a microbenchmark that answers a different question.

Rules for every event:

- Time is wall clock from process start to checksum on stdout, unless the event says "kernel only".
- Hot run: one warmup of the same binary, then the timed run. Cold run: drop caches, timed run is the first execution.
- Output is a single `u64` checksum (xxh3 of canonical rows) plus the row count. Wrong checksum is a DQ.
- Report median of 5 runs, and the spill point where the register path stops applying.
- Hardware line required: CPU model, cores used, RAM, AVX-512 on or off.
- Reference engines, same machine, same files: DuckDB (latest stable), Ladybug, upstream `lance-format/lance-graph` at the tagged release. A number with no reference column is not a result.

Scale ladder: 10^6, 10^7, 10^8, 10^9 edges. Register events also run at 2^4 and 2^8 keys.

## Inputs

Generate once, publish the seed.

- `edges.bin`: packed `src:u32, dst:u32, label:u16, pad:u16`. Sorted by `src` for the CSR events. Unsorted twin `edges_unsorted.bin` for the sort events.
- `nodes.bin`: `id:u32, attr:u64`.
- Station-style text twin `edges.csv` only for the parse event, so parse cost is not hidden inside a binary load.
- Generator: RMAT or LDBC SNB SF1/SF10, plus a uniform 2^k key file for the register events. Seed `0x6BC0_0001`.

## Events

### R0 — register equality

The 1.7 ns claim, isolated.

Two `u64` keys already in registers. Operation: compare, mask, select. Kernel only, after the values are loaded. Report cycles/op from `perf stat`, plus `objdump -d` of the basic block.

Pass: hot ≤ 2 ns, cold (L1) ≤ 5 ns, L1-miss count ≈ 0 on the hot run. This event cannot be cited as a join result.

### R1 — register ABI format

PowerShell `"{0}{1}" -f` analogue. Two values in registers, produce a packed `u128` or a 16-byte ABI word. Kernel only.

Pass: hot ≤ 3 ns. Publish the instruction count. If it is above 12 instructions, the ns number is not believed.

### R2 — mask width ladder

Masks over 4, 8, 16, 32, 64 bytes, data in L1.

Pass: report ns at each width. The 4–40 ns band is the expected shape. A flat line means the timer is wrong.

### R3 — spill curve

The event that makes R0 a database claim.

N keys, N from 4 to 2^20. Implementation may use registers, then a stack slot, then an L1 buffer, then a real index. Report ns/key and the N where you leave the register file (architectural GPRs, or ZMM if you declare AVX-512).

Pass: a plot with a knee. No knee, no claim that the join "never leaves the registers".

### J0 — two-table equijoin, keys resident

Build side fits in L1 (≤ 256 keys). Probe 10^6 keys. Output matching pairs, checksummed.

Reference: DuckDB `SELECT count(*) FROM a JOIN b USING (k)` on Arrow inputs, and a handwritten hash join.

Pass: checksum match. Latency reported separately for build and probe. Probe may be in the ns/key range. Build may not.

### J1 — two-table equijoin, keys not resident

Build side 10^7, 10^8, 10^9 rows, keys `u64`, payloads 16 bytes. This is the DuckDB event.

Reference: DuckDB, DataFusion, a SwissTable probe.

Pass: checksum match at every scale. Quote end-to-end and probe-only. Citing R0 here is a DQ.

### J2 — self-join / self-pointing pattern

The upstream hang. Pattern equivalent to LDBC SNB Q30: comments that reply to posts by the same person. High-cardinality intermediate.

Reference: upstream lance-graph issue #111 query, Ladybug, DuckDB SQL rewrite.

Pass: returns, checksum matches Ladybug, does not hang at SF1. Time is secondary.

### G0 — one-hop expand

`MATCH (a)-[:R]->(b) RETURN count(*)` over the packed edge file. 10^6 to 10^9 edges.

Reference: upstream lance-graph `single_hop_expand`, DuckDB join of the edge list to itself is not the reference — Ladybug one-hop is.

Pass: checksum match, allocations logged (`dhat` or `stats_alloc`). Zero-copy means zero heap allocs on the probe path after the file is mapped. Mapping the file counts as I/O, not as a copy.

### G1 — two-hop expand

`MATCH (a)-[:R]->(b)-[:R]->(c) RETURN count(*)`. Same scales.

Reference: upstream Criterion `two_hop_expand` (4–6 ms on small N is the published baseline), Ladybug.

Pass: checksum match. Report intermediates materialized vs streamed. A number that materializes the full 2-hop bag and then counts it is a different event; label it G1-mat.

### G2 — n-hop path count

Queries shaped like graph-benchmark q8 and q9 (paths through an attribute predicate). SF1 and a generated RMAT of similar density.

Reference: Ladybug 0.15 numbers on that suite (q8 ≈ 7.3 ms, q9 ≈ 95 ms on their published run — remeasure locally; do not cite the README as your result).

Pass: checksum match. Winning this is a planner result (WCOJ or factorization), not an ABI result. Say which plan you picked.

### G3 — filtered expand

`MATCH (a)-[:R]->(b) WHERE a.attr > X AND b.attr < Y RETURN count(*)`. Selectivity 0.1, 0.01, 0.001.

Reference: DuckDB on the same Arrow columns, upstream lance-graph.

Pass: checksum match. This is where predicate pushdown shows up. Register ABI does not.

### P0 — parse tax

1BRC shape. `edges.csv`, 10^9 rows, `station,src,dst` text. Compute per-label count, min src, max dst. Rust only.

Reference: a hand-written 1BRC-style parser, not the graph engine. This event exists so the engine cannot hide cost by starting from a pre-digested mmap.

Pass: checksum of the aggregate. Report GB/s.

### P1 — mmap, no parse

Same aggregate over `edges.bin`, `mmap` + scan. The delta P0 − P1 is the parse tax. Quote both. Never quote P1 as the cost of a query that started from CSV.

### Z0 — copy audit

Run G0 and J1 under `dhat` or a counting allocator.

Pass: bytes allocated on the hot path, named. "Zero-copy" is allowed only if the hot path allocates 0 after setup. Setup (index build, CSR construct) is reported on its own line and is not zero.

### Z1 — CSR build

Build a CSR from `edges_unsorted.bin` at 10^8 and 10^9. Time the sort and the index build separately from G0 probe.

Pass: probe of G0 on the built CSR, plus build seconds. A probe number without build is R0 again.

### C0 — checksum wall

Every event prints `rows=<u64> checksum=<u64> secs=<f64>`. A script diffs checksums against the reference engine. Mismatch fails the run even if you were faster.

## What you may claim after which event

| Claim | Requires |
| --- | --- |
| 1.7 ns hot, 4.3 ns cold | R0 + R2, with `perf stat` and `objdump` |
| Never leaves the registers | R3, with the knee published |
| Faster join than DuckDB | J1 at 10^8 or 10^9, checksum match, same machine |
| Faster graph engine than Ladybug | G2 at SF1, checksum match |
| Zero-copy | Z0 showing 0 hot-path bytes |
| Faster than upstream lance-graph | G0 or G1 at the same N as their Criterion table, checksum match |

R0 alone supports none of the last four rows.

## Out of scope

Cognitive crates, ontology bridges, agent logs, session notes. They are not in the timed path. A binary that links them is fine; a result that depends on them is not an entry.
