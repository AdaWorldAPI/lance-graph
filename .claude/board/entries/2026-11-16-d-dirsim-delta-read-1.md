# D-DIRSIM-DELTA-READ-1: reading what changed between observations, from the stored deltas (2026-11-16)

**Status:** MEASURED + TEST-PINNED, lab. Builds on #2099 (`reobserve`).

## A. Reuse matrix

| Need | Existing | Verdict |
|---|---|---|
| What changed between two observations, as recorded | `Version.delta` per version; no range reader | **WIRE**: `VersionStore::history(from, to)` (this PR) |
| Net change between two versions (made-and-undone pairs cancel) | `VersionStore::diff` (same root: `diff_shared`) | REUSE; but it builds both views, so its cost is the whole lineage (C) |
| State at an observed version (selected user / group) | `VersionStore::view(v)` + `View::is_member` / `node_state` / `attr` | REUSE; folds the lineage from the root into one overlay. No per-version snapshot is built |
| Before / after values | carried in `Change` (`SetAttribute { from, to }`, `SetActive`, node state on create / delete) | REUSE |
| Observed vs simulated | `Version.origin` (`Observed { source, observed_at_ms }` / `Simulated { rule, evidence }`) | REUSE |
| Durable replay | `log::replay`, `LanceLog::restore`, `VersionStore::restore_stored` | REUSE; restore cost is quadratic in lineage length (C) |

## B. The bridge

`VersionStore::history(from, to) -> Result<VersionRange<'_>, HistoryRefusal>`:
the stored versions after `from` up to and including `to`, newest first,
borrowed. It folds nothing, builds no view and allocates nothing. It refuses
`UnknownVersion(id)` and `NotAncestor { from, to }` (another root, such as
another directory scope, a reversed range, or another branch). It is the
event reading: a node deleted and created again shows both events, and an
unchanged re-read shows a version with an empty delta. The net reading is
`diff`. Order is the lineage (logical version) order; observation timestamps
are metadata and order nothing.

## C. Cost (`examples/history_cost.rs`, 100 users, one membership changed per re-read, release, warm medians)

| re-reads | versions | delta records | log bytes | bytes / re-read | `history` last step | `view` + `is_member` | `diff` last step | replay + restore (cold) | roots |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 1 | 13 873 | 125 | 34 ns, 0 alloc | 0.23 µs, 2 alloc | 0.57 µs, 6 alloc | 183 µs | 1 |
| 16 | 17 | 16 | 15 748 | 125 | 34 ns, 0 alloc | 2.1 µs, 7 alloc | 5.4 µs, 18 alloc | 220 µs | 1 |
| 128 | 129 | 128 | 29 748 | 125 | 34 ns, 0 alloc | 16.5 µs, 24 alloc | 38 µs, 58 alloc | 1 498 µs | 1 |
| 1 024 | 1 025 | 1 024 | 141 748 | 125 | 47 ns, 0 alloc | 92.8 µs, 94 alloc | 186 µs, 193 alloc | 50 190 µs | 1 |

- Writes stay sparse: 125 bytes per re-read at every depth, one root.
- `history` is flat: one step per version in the range, no allocation.
- `view` and `diff` grow linearly with the lineage (~90 ns per version).
- **Replay grows quadratically**: 1 024 re-reads take 50 ms cold. Cause:
  `VersionStore::restore` checks each version with `st.view(parent)`, which
  folds the lineage from the root again for every version. #2099 made long
  lineages routine, so this threshold is now reachable. Not fixed here.

## Proposed next PR (not done here)

1. **Linear restore**: in `restore`, keep the view of the last applied version
   and apply the next version's delta to it when its parent is that version
   (the re-observation chain), falling back to `view(parent)` otherwise.
   Falsifier: a counter of deltas applied during restore equals the total
   delta count; replay of 1 024 re-reads stays within a constant factor of
   128. Version identity and replay output are unchanged.
2. **Bounded reads**: a checkpoint (a root written with `observe` every K
   re-reads) is already available to callers and needs no change; an
   automatic policy is a separate decision.

## D. Falsifiers

`tests/history_read.rs` (9), against a scalar oracle that compares two
`Observation`s by set logic and shares no code with dir-sim's diff:
one membership change = one change; unchanged re-read = an event without
changes; four reads pairwise and across the interval; a simulated branch is
not observed history; delete + re-create = two events while the net reading is
empty; another scope, an unknown id and a reversed range refuse; attribute
before/after with absent, explicit empty and unknown kept distinct; an old
observation reads as it was after ten newer ones; a refused read changes
nothing. `dir-sim-lance/tests/lance_log.rs`: the re-observed history read back
from Lance and from an in-memory replay gives the same `history` and `diff`
for every pair; a refused re-read writes no batch.

Disable runs, each red before restore: ancestor check off (2 failed);
empty-delta versions skipped (1 failed); range lower bound ignored (5 failed).

## E. report-rs handoff

- Entry points: `VersionStore::{reobserve, history, diff, view, version, tag}`,
  `log::{LogState, replay}`, `LanceLog::{append, restore}`.
- Borrowed: every `&Version` from `history` (id, parent, origin, delta).
  Allocated: a `View` per state read (one overlay), `diff`'s result vector, and
  the strings at presentation (`VersionStore::value`).
- Witnesses available: the version id and lineage, the observed tag, source
  and read time per observation, the Lance dataset version at restore (a
  physical commit counter, never a logical version).
- Refusals: `HistoryRefusal::{UnknownVersion, NotAncestor}`,
  `ObserveError::{Build, Sim}`, `SimError`, `StoreError` (Lance).
- Missing: a source authorization witness. Nothing here says who may read
  which directory's history; report-rs must not treat a readable version as an
  authorized one. report-rs's `RunReceipt` (license evaluation, #120) is a
  separate record and is not this history.
