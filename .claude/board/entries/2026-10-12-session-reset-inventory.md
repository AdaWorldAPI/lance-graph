# 2026-10-12 — Session reset: method layer moved to backup (D-CMR-RESET)

**Status:** DONE (removal); the rebuild is OPEN.

**Finding:** the class-method layer built on 2026-10-12 was a second algebra
beside the existing folds, not a binding over them, and it was slower than a
scalar row loop: `billable_hours` measured 430–480 µs per call against 283 µs
for the row loop (1,048,576 rows, AVX-512, release). Its hops were
foreign-key reads; the graph has none. Everything here is moved to backup and
rebuilt.

## Where the old state lives

| repo | branch | what |
|---|---|---|
| lance-graph | `backup/session-011BMJ-2026-10-12` | `main` at `8ae5bce4`, with the whole method layer |
| lance-graph | `backup/session-011BMJ-2026-10-12-chain2` | #1522 (`GatherChain`, `Filter::Chain`, two-hop glove) |
| lance-graph | `backup/session-011BMJ-2026-10-12-sagfix` | #1519 (SAG reprojection fixes) |
| lance-graph | `backup/session-011BMJ-2026-10-12-capstone` | #1523 (SAG docs capstone) |
| ndarray | `backup/session-011BMJ-2026-10-12` | `master` at `e726f98`, with `mask_gather_chain_u32` (#358) |

## Removed

| item | where | why |
|---|---|---|
| `method.rs`, `call.rs`, `decl.rs`, `gremlin.rs`, `awareness.rs` | `crates/lance-graph-glove-parity/src` | own `MethodBody` algebra, lowered per call through `Filter::cmp` over value lanes with fresh scratch (`run_quack`), no `Compiled`, no `lower_fused`, no resident planes |
| `Relation { fk }`, `ForeignLane` hops, `Cond::Reaches/Via`, edge-table anchor, `Fold::Distinct` | `method.rs` | foreign-key model; a hop is a coordinate range, not a key read |
| SAG graph (`Vec<Edge>`, `retain`, `HashSet`) | `awareness.rs` | O(edges) per operation; built on the removed method layer |
| 10 test files, 2 method fixtures, 4 examples | `crates/lance-graph-glove-parity/{tests,examples}` | test the removed code |
| mask-risc `GatherChain`, Quack `Filter::Chain` | #1522 (not merged) | per-row chained key reads |
| `mask_gather_chain_u32[_under]`, `ChainHop`, `ChainTest` | ndarray (#359 removes) | same |
| `lance-graph-r2il` dependency, OGAR checkout in `glove-parity.yml` | crate `Cargo.toml`, workflow | only the method layer needed them; both restored to their state before this session |

## Kept, and why

| item | why |
|---|---|
| `basis.rs`, `tests/authority.rs`, `tests/canonical_front.rs` | another session's work (D-XGP-1/2); untouched, tests green |
| r2il edits: every fold reads a resident plane as its mask (`8d5a36fb`); MIN/MAX and ordered-i32 procedures (`964a95c5`); declarative bodies (`0dbb6b52`, `b61f62fa`) | woven into r2il's own procedure surface; reviewed in the rebuild, not reverted blind |
| contract: compute DAG reads all 256 field positions (`6a83f99f`) | a correctness fix in `class_view.rs`, independent of the method layer |
| build: native target-cpu, debug 0, x86_64 pinned to v3 for wheels (`5cfc8453`, `12cdfde9`) | build configuration, not a primitive |
| NAL recipe audit, CE64 plateau, nal-polyfill plan, planner revision/receipt probes | research and oracle tests, no query path |

## The rebuild (D-CMR)

A class method is a binding `classid × WideFieldMask → an existing fold`: one
`Compiled` mask-risc program over resident planes and coordinate ranges
(`Lowering::Range`, ternlog fusion, the fold terminals, `CountKeyRunsU32`,
Quack `lower_fused`, `fold_ranges`, `prefix_*`). A call resolves the binding
and runs the compiled program; it does not lower, allocate or interpret. Chained
hops are coordinate-range intersections inside the same program. A fold that
does not exist is reported as a gap in the fold catalogue, not built in the
method layer.

OPEN: #1523 anchors `ProposedMethod` on `awareness::Candidate`, which this
reset removes; the anchor moves to the rebuilt binding.
