# Plan: Cycle-Coherent SoA Snapshot — No-Cross-Cycle-Lag Guarantee

**Version:** v1
**Date:** 2026-06-06
**Status:** Queued
**D-ids:** D-SOA-SNAP-1 through D-SOA-SNAP-6

---

## The problem

`temporal.rs` (PR #468) closes the row-scale deinterlace gap: HLC tick →
`classify/deinterlace` → causally-coherent row sequence. But there is a
parallel byte-scale gap: nothing prevents a reader from holding a mix of
column data from cycle N and cycle N+1 within the same SIMD sweep. This is
the **cross-cycle lag problem** — a SIMD sweep that is not internally
single-cycle is not coherent.

The deinterlace operation is one operation at two scales:

```text
Row/query scale  →  HLC tick + DependsClosure  →  temporal.rs (SHIPPED, PR #468)
Byte/column scale → SoaEnvelope::cycle() stamp → MailboxSoA Arc-swap (THIS PLAN)
```

---

## The mechanism: Arc-swap COW at column granularity

The SoA mailbox carries its columns as `Arc<[u8]>` slices (via
`MultiLaneColumn` in ndarray). The invariant is:

> **A reader that snapshots all column Arcs at the same `cycle()` stamp sees
> a single coherent cycle. No column can be from a prior cycle.**

### Write path (in `lance-graph`, `MailboxSoa::advance_phase`)

On every `advance_phase(to: KanbanPhase)`:

1. Increment `cycle` counter on the envelope.
2. For each mutated column: swap the `Arc` pointer — `Arc::make_mut` on the
   backing `Arc<[u8]>` of the `MultiLaneColumn`, write the new data, then
   publish the new Arc via an `ArcSwap` (or `RwLock<Arc<MultiLaneColumn>>`).
3. The cycle increment is a `SeqCst` store (fence) BEFORE the column Arc
   swaps. Readers who observe the new cycle will see the new column data.

### Read path (in `lance-graph`, `MailboxSoaView`)

On `snapshot()`:

1. Load cycle stamp.
2. Clone all column Arcs under the same cycle stamp (atomic snapshot loop:
   re-read cycle after loading all Arcs; retry if it changed — lock-free
   single-retry is sufficient because writers are serialized through
   `advance_phase`).
3. Return `MailboxSoaSnapshot { cycle, cols: [...] }`.

The snapshot guarantees all column data is from the same cycle.

### Boundary: ndarray stays layout-only

`MultiLaneColumn` in ndarray is `Arc<[u8]>` with typed lane iterators —
**layout-only**. The Arc-swap policy (when to swap, how to snapshot, the
cycle fence) belongs in `lance-graph`'s `MailboxSoa`. ndarray never learns
that cycles or snapshots exist. The boundary is:

```text
ndarray::simd::MultiLaneColumn  — Arc<[u8]>, lane iters, Send + Sync, zero-copy reads
lance-graph::MailboxSoa         — Arc-swap on advance_phase, cycle fence, snapshot()
```

### Connection to temporal.rs

`SoaEnvelope::cycle()` is the byte-scale clock. `QueryReference::ref_version`
is the row-scale clock (a Lance version). They are the same monotonic clock
at different granularities — Lance version N corresponds to SoA cycle C(N).
When `temporal.rs::deinterlace` runs at query time, the `V_ref` it uses should
align with the `cycle()` of the snapshot being queried.

Wiring: `VersionScheduler::on_version(&view, at, exec)` provides the Lance
version; the `MailboxSoaSnapshot` that went into that version carries its
`cycle`. Threading `snapshot.cycle` into `QueryReference` closes the loop so
row-scale and byte-scale deinterlace use the same clock.

---

## Deliverables

### D-SOA-SNAP-1 — `MailboxSoaSnapshot<C>` type in lance-graph-contract

> **REVISED 2026-06-11 (PR #477 CodeRabbit Critical):** the original sketch
> (`cols: Vec<Arc<MultiLaneColumn>>` directly in contract) would create a
> `lance-graph-contract → ndarray` dependency — `MultiLaneColumn` is an
> ndarray type, and the contract crate is zero-dep by iron invariant. The
> snapshot type is therefore **generic over the column type**; the concrete
> binding happens in lance-graph, never in contract.

A `MailboxSoaSnapshot<C>` struct: `cycle: u32`, `cols: Vec<Arc<C>>`.
Snapshot is `Send + Sync` (when `C: Send + Sync`). No reference to the
originating `MailboxSoa`. This is a point-in-time read — immutable after
creation. Contract never names `MultiLaneColumn`; it only carries the
generic parameter.

### D-SOA-SNAP-2 — `SnapshotProvider` trait in lance-graph-contract

```rust
// In lance-graph-contract (zero-dep — no ndarray import):
pub struct MailboxSoaSnapshot<C> {
    pub cycle: u32,
    pub cols: Vec<Arc<C>>,
}

pub trait SnapshotProvider {
    type Column;
    fn snapshot(&self) -> MailboxSoaSnapshot<Self::Column>;
}

// In lance-graph (the binding side):
//   impl SnapshotProvider for MailboxSoa {
//       type Column = ndarray::simd::MultiLaneColumn;
//       fn snapshot(&self) -> MailboxSoaSnapshot<MultiLaneColumn> { … }
//   }
```

Zero deps in contract — the associated type defers the column choice to the
implementor. `MailboxSoa` in lance-graph binds `Column = MultiLaneColumn`;
ndarray never learns the snapshot exists, contract never learns ndarray
exists, lance-graph binds them (same triangulation as `SoaEnvelope`).

### D-SOA-SNAP-3 — Arc-swap write path in `MailboxSoa::advance_phase`

In lance-graph (not contract, not ndarray): implement the cycle fence +
column Arc-swap on every `advance_phase`. Use `std::sync::RwLock<Arc<MultiLaneColumn>>`
per column (no external arc-swap crate needed unless benchmarks show
contention; add as a feature flag if needed).

### D-SOA-SNAP-4 — `snapshot()` implementation on `MailboxSoa`

Lock-free snapshot: load cycle, clone all column Arcs, re-read cycle, retry
once if changed. Return `MailboxSoaSnapshot`.

### D-SOA-SNAP-5 — No-cross-cycle-lag falsification test

```rust
// Spawn a writer thread: advance_phase in a loop (100 cycles).
// Spawn 8 reader threads: each calls snapshot() in a loop.
// Assert: every snapshot has all columns reporting the same cycle.
// Assert: no snapshot mixes data from two different cycles.
```

The test is the formal statement of the guarantee. If it passes, the
invariant is mechanically enforced, not just documented.

### D-SOA-SNAP-6 — Wire `snapshot.cycle` into `QueryReference`

In the planner: when a query resolves a `MailboxSoaSnapshot`, thread
`snapshot.cycle` through `QueryReference::hlc_tick` (or a new
`QueryReference::soa_cycle: Option<u32>` field) so `deinterlace` at
row scale uses the same cycle boundary as the snapshot at byte scale.

---

## Prerequisite gap fixes (order matters)

These mechanical fixes should land before or alongside D-SOA-SNAP-1
(they settle the column shape):

1. Remove `MailboxSoA::emit()` + `CollapseGateEmission` from source.
2. Rename `last_emission_cycle` → `last_active_cycle` in MailboxSoA.
3. Drop `entity_type: u16` from SoA row — MailboxId IS NiblePath.
4. Fix `OntologyRegistry::enumerate_first_with_entity_type_id` linear scan.
5. Remove `MappingRow.thinking_style` — Kanban owns thinking styles.
6. Fix `unbundle_from` in `kv_bundle.rs:29` — `wrapping_sub` is not the
   inverse of weighted-average `bundle_into`.

Items 1-5 settle the column shape before the Arc-swap schema is frozen.
Item 6 is independent but should not be deferred (correctness bug).

---

## Non-goals

- No recurrence / standing wave implementation. The standing wave is the
  deinterlaced Lance version projection, provided by Lance versioning
  (O(1) 90° lookup). Do not implement it in compute.
- No baton. No emission. No inter-mailbox handoff type. The snapshot is
  consumed in-place; nothing is transmitted.
- ndarray does not learn about cycles, snapshots, or advance_phase.

---

## Cross-references

- `temporal.rs` (PR #468) — row-scale deinterlace (SHIPPED)
- `soa_envelope.rs` (PR #477) — envelope LE contract (IN REVIEW)
- `soa-three-tier-model.md` — three-tier lifecycle model
- `q3-standing-wave-falsification.md` — falsification: standing wave = Lance
  versioning, not compute recurrence
- `.claude/board/EPIPHANIES.md` E-DEINTERLACE-TWO-SCALES — the synthesis
- `ndarray/src/simd_soa.rs` — `MultiLaneColumn` (layout-only; Arc-swap lives
  in lance-graph, not here)
