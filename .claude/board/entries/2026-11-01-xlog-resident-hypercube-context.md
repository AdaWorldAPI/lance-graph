# D-XLOG-NATIVE-7: one resident hypercube as XLOG's execution context (probe)

**Date 2026-11-01 · ID D-XLOG-NATIVE-7 · Status PROBE · Plan `.claude/plans/d-xlog-native-v1.md` (Amendment F, pending beside E in #1910) · Crate `crates/lance-graph-xlog` (`cube`)**

DECISION (direction, from the session): the experiment replaces the machine
under XLOG's CUDA-facing API with one resident context, rather than speeding up
operators one relation at a time. Observable results of admitted classical
XLOG queries must match; the internal model is free.

What `lance_graph_xlog::cube` does (VERIFIED-IN-CODE, `src/cube.rs`):

- `Cube::new` looks up every id of every relation, both columns, once in one
  tile's `LocalMap` under a `MapTable` frame. Nothing is looked up again.
- `scan` / `bind` / `join` return `View`s: `Copy` handles into the cube's node
  list. Making a view runs no program and builds no row.
- Terminals: `support` (two `ScatterOrU32` + `mask_and`), `bag_count` (two
  `GroupReduce Count` + checked product), `materialize_rows` (the one named
  place tuples are built). Each re-checks the frame.
- Any relation joins any other on either column from the same lanes: lanes
  are stored per relation, not per join orientation.

TEST-PINNED (`tests/cube.rs`, 6 tests):

- parity with the per-call `support` / `bag_count` under five bound shapes;
- all 36 (relation pair × column pair) joins over three relations match an
  enumeration oracle, and `LocalMap::lookups()` is unchanged after intake;
- 50 composed joins run 0 programs; `support` then runs exactly 2, builds 0 rows;
- a bound outside the map short-circuits its side (1 program, not 2);
- 300-case randomised differential (ids 0..12, either orientation, optional
  bounds): bag count, support and rows; 150+ non-empty;
- stale frame, foreign view, bound on the join column, non-join terminal refused.

Disable runs, all red: bound ignored in programs; support on the wrong column;
support skips the frame check; `materialize_rows` ignores the bound.

XLOG side (MEASURED by reading `xlog-runtime` / `xlog-cuda`):

- `CudaBuffer` and `CudaKernelProvider` are concrete structs, not traits.
  `CudaBuffer` fields are `pub(crate)`; callers use 16 public methods.
- `xlog-runtime` calls 79 of the provider's 397 public functions; ~25 are
  telemetry/resets, ~30 are WCOJ/clique variants.
- 12 `provider.device()` sites: 1 in the core dispatcher (join-index cache
  key), 2 `synchronize`, 9 in `epistemic_workspace` / `resident` (outside the
  13 RIR kinds).
- `xlog-cuda/build.rs` panics without `nvcc` unless `XLOG_RUSTDOC_NO_CUDA=1`
  (empty kernel data). With it, `cargo check -p xlog-runtime` finishes on a
  host with no CUDA toolkit (MEASURED 2026-11-01, 3 warnings, 0 errors), so a
  device-less build of the XLOG fork is reachable.

Comparison with Amendment E (caller-level polyfill):

- E keeps XLOG's API and moves its callers onto `lance-graph-xlog` one
  operator at a time; every call still owns its inputs.
- F gives the CUDA-facing handles a shared context: a `CudaBuffer` would be a
  `View` plus the cube it belongs to, so a join between two XLOG buffers needs
  no new lanes, no index (XLOG's `JoinIndexKey` cache becomes redundant) and no
  device.
- Neither is promoted. F is better for the 13 RIR kinds only if the dispatcher
  can keep buffers as views between nodes; a node that downloads rows ends the
  advantage at that node.

**OPEN:** recursion (semi-naive delta over versioned masks), union and
difference as views; more than one tile; n-ary relations; a `CudaBuffer::View` variant and a
device-less provider in the XLOG fork; per-node cost vs the per-call path.
Reasoning surfaces (NARS revision, thought layers, CausalEdge64) are not in this
probe and need their own contracts.
