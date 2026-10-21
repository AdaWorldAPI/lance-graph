# 2026-10-21 — D-RU-2 (U2): let-bound planes over the contract set algebra

**Status:** MEASURED (lab; `crates/lance-graph-contract/src/set_program.rs`, `crates/lance-graph-relalg/src/program.rs`). Plan: `.claude/plans/d-relalg-unified-v1.md`, stage table (U2). Builds on D-CMR-1 (`lance_graph_contract::set_algebra`, #1644).

## What landed

- `set_program::SetProgram { version, resident, lets, queries }`.
  - `Sym(i)` with `i < resident` is a resident plane.
  - `Sym(resident + k)` is `lets[k]`.
  - A let may read only earlier lets; queries may read any let.
  - `validate()` refuses forward and unknown references.
  - A program describes one table version.
- `set_program::share(version, resident, queries)` lifts a junction subset into a let when two or more junctions of the same kind hold it. Rules:
  - largest subset first;
  - judged after `normalize`, because `normalize` flattens nested junctions, so identical subtrees rarely survive whole;
  - lets are ordered by dependency and renumbered densely.
- `relalg::run_program`:
  - evaluates each let once into a scratch plane of `n_rows / 8` bytes;
  - runs every query over the resident planes plus those planes;
  - a let's plane is one more resident plane, so every fold stays a tree.
- `relalg::run_recompute` is the baseline: each query with its lets inlined.
- Both refuse:
  - a program for another version (`VersionMismatch`);
  - a malformed program (`Shape`);
  - the wrong number of resident planes;
  - a measure count that does not match the queries.
- `ProgramCost` counts, from the folds that ran: programs, ops, lane reads, plane reads, planes materialized and their bytes, and the most scratch slots one fold used.

## Measured

`examples/program_bench`, release build: 16 queries that share one three-comparison conjunction; one of every three also reads a resident plane. Medians of 9 runs on 4 cores, load below 1.

| rows | program ms | recompute ms | time | lane reads | ops | materialized |
|---|---|---|---|---|---|---|
| 100,000 | 0.35 | 1.15 | 3.3× | 14 vs 59 | 15 vs 59 | 2 planes, 25 kB |
| 1,000,000 | 4.77 | 14.87 | 3.1× | 14 vs 59 | 15 vs 59 | 2 planes, 250 kB |
| 4,000,000 | 20.98 | 69.35 | 3.3× | 14 vs 59 | 15 vs 59 | 2 planes, 1 MB |

- The lane-read count is the row-traversal count: each comparison reads one whole lane of `n_rows` values.
- The program runs 18 mask-risc programs against 16 for recompute: one per let more, each a whole pass.
- With 4 queries, the program reads 6 lanes against 15 and is 2.1× faster at 1M rows.
- Both paths use at most 2 scratch slots per fold, inside the ≤ 3 target.

**Correctness.**
- Three paths agree on every answer (count, sum, min, max) and on every kept mask: the program, recompute, and a row-at-a-time oracle over the inlined descriptions.
- Fixtures: hand-built sets with one shared conjunction; with nested and distinct lets; with nothing shared; and 60 generated sets.
- With nothing shared, the program's cost equals recompute's exactly.
- Versions: a program built for version 7 is refused against version 8.

**Disables.** Eight, all red-then-green:
- wrong let evaluated;
- lets cannot read earlier lets;
- version check removed;
- `validate` removed;
- resident-count check removed;
- lane reads not counted;
- bytes not counted;
- `share` lifts nothing.

The first and second were green at first: no fixture held a program with more than one let. `nested_and_distinct_lets_agree_with_recompute_and_the_oracle` closes that.

## Open

- `share` is greedy (largest subset first). It does not weigh a let's materialization cost against the reads it saves. A subset held by two junctions is lifted even when its comparisons are cheaper than the extra pass. Measured here only where sharing wins.
- The ops count includes the let passes. Total cost is bounded by time, which is measured; there is no byte-level traffic counter.
- U3 (cross-rule CSE in crepe-plan-probe) and U4 (join over `JoinDomain`) are unblocked.
