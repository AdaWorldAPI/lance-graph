# 2026-09-23 — Stable row ids hold; tombstones are exact; the seal still owns cohort and fold

**Status:** MEASURED (lance 11.0.0 in-repo; lance 12.0.0 identical in a scratch crate) · OPEN (newest-first append-only read, see below)
**D-ids:** D-LNC-5b (new, In PR). Extends D-LNC-5a; unblocks D-LNC-5's delete arm *with the switch on*.

## Question

`enable_stable_row_ids` is still documented as experimental in lance 11 **and** 12 — the
same comment in both (`write.rs`): *"Experimental … stable after compaction operations, but
not after updates"*. Tombstone records need two things from it: the row keeps its `_rowid`,
and `delta().get_deleted_row_ids()` reports exactly the real deletes. D-LNC-5a measured only
the switch-OFF half (the version columns are empty without it).

## Measured — `crates/lance-graph/tests/stable_row_id_probe.rs`, switch ON throughout

| operation | `_rowid` | tombstone read |
|---|---|---|
| compaction (2 → 1 fragments; control: fragments were actually removed) | none moved | — |
| update via `merge_insert` | kept | `[]` — no false tombstone |
| update via `UpdateBuilder` (`UPDATE … SET`) | kept | `[]` |
| delete | no other row moved | exactly the deleted row |
| compaction materializing the delete | none moved | `[]` |
| 4 writers × 6 rounds concurrently, BTree index on the key | 64 seed rows: none moved; 84 live rows, 84 distinct ids | exactly the 4 deleted keys over the whole run |
| compaction + `optimize_indices` after that | none moved | index lookup == plain scan for 88 keys; the plan control shows `ScalarIndexQuery … id_idx(BTree)` with the index and none without |

The doc comment's *"not after updates"* is contradicted on both update paths at this scale.

**Disable run (two-sided):** with the switch OFF the same file fails both tests — compaction
moves all 8 ids (`[1..8]`), and concurrent upserts move the contended row's id (`[0]`). So the
probe measures the switch, and switch-off `_rowid` is a physical address.

## What the switch does NOT give — the seal's properties stay the seal's

Per `.claude/knowledge/seal-vs-temporal-ordering-information.md` the seal supplies a total
order, a per-row fold decided by it, and one cohort per version. The concurrent arm pins the
consequence of committing directly instead:

- **cohort:** 24 upserts + 4 deletes minted **28** versions. Nothing groups a cycle into one
  version, so a version range cuts through cycles and cannot name "this cycle's tombstones".
- **fold:** the contended row goes to the highest committed version — a commit race. Across
  runs the winner changed (`w2r5`, `w1r5`, `w2r5`). The seal fixes the order before the append.

**So:** stable row ids make a tombstone *exact*; the seal makes it *attributable to a cycle*.
Under the sole sealed writer (one cycle = one version) a cycle's tombstones are
`deleted(base_version, sealed_version)`. Complement, not replacement.

## Scope

8–88 rows, one process, local filesystem, BTree only (no vector index), 4 writers.
`cleanup_old_versions` untested — deliberately out of scope: all versions are kept, and the
switch deletes nothing (versions are removed only by cleanup, rows only by compaction).

## OPEN — newest-first over append-only (WORKING-MODEL, not measured)

With everything append-only and every version kept, a tombstone can be an appended record
read newest-first: `QueryReference::at(v, rung)` → `deinterlace` → reverse → first hit per
subject. The order that needs is already durable — version number across versions, and within
a version `scan_sealed`'s stored order (it never sorts, `persist_sink.rs:523`), which
`deinterlace`'s *stable* sort preserves (`temporal.rs:367`). That needs no experimental switch;
stable row ids would only make the tombstone read O(1) per range instead of a scan. Relation to
the alpha split tunnel (`alpha_tunnel.rs`): it could supply the persistence half the tunnel
lacks (overlay writes → rung-stamped appends at the seal), but not the in-cycle simultaneous-rung
visibility, which only the overlay has. Proposed third probe arm, not yet built: two sealed
cycles of rung-stamped appends + tombstone records, newest-first read at each `(v, rung)`,
equality with the stable-row-id read, scan cost vs history depth.
