# 05 — deletion policy and tombstones: can we still go back?

The operator's concern: *"the deletion policy might need to be investigated
since in many if not most cases we want the tombstones to be preserved for
temporal to be able to go back in versions."*

Audited 2026-09-06. Three verdicts, each marked MEASURED or UNVERIFIED.

## (a) Can this stack go back to a version after a delete? **MEASURED: YES**

`crates/lance-graph/tests/lance_row_identity_probe.rs:280` deletes a row;
`:328` asserts node 3 is present at version C; `:314-320` re-reads every prior
version byte-exact.

The mechanism, from the extracted lance 11 source
(`/tmp/lance-src/lance-11.0.0/src/dataset/write/delete.rs:49-73, 469-514`):
**a delete is a per-fragment deletion FILE.** The prior manifest still points
at untouched data, so time travel is intact by construction.

## (b) What would destroy it? **MEASURED: `cleanup_old_versions`**

Probe `:436-440` runs it with zero retention, then `:459-462`:

```rust
assert!(graph.at_version(versions[0]).await.is_err(), "FIRE HALF")
```

`at_version` **ERRORS** — the history is gone. Only **TAGGED** versions survive
(`:448-453`, tag set at `:288`).

**The risk is prospective, not current.** There is **ZERO production caller** of
`cleanup_old_versions` / `compact_files` / `CleanupPolicy` in lance-graph.
MedCare-rs has none either: its `retention_days` sweep
(`audit_rotation.rs:100-152`) deletes rotated FILES and defaults to 0 /
disabled.

⚠ **The concrete way this gets broken.** surrealdb's kv-lance carries exactly
the retention loop one would be tempted to port: `background_optimizer.rs`, a
5-minute loop with a **7-day default** (`cnf.rs:41-42`), **enabled by
default**. Porting that shape without tagging would silently destroy history
older than a week. If a retention policy is ever adopted here, tagging must
come first and the default must be OFF.

## (c) Is a durable tombstone row needed? **SPLIT**

- **Byte-level time travel: NO tombstone needed** (MEASURED). The version chain
  plus deletion files already preserve readability.
- **Semantic delete-awareness: YES, and it is missing** (MEASURED).
  - `GraphDiff` has no removed-nodes field (`versioned.rs:70-86`), pinned as a
    blind spot at `probe:381-386` — *a removal is an empty `GraphDiff`*.
  - `graph_seal_check` only reports `Staunen` (`versioned.rs:632-636`,
    `probe:366-370`): you learn that something vanished, never **what**.
  - Durable audit tombstones are DESIGNED AND NOT BUILT: `witness_tombstone.rs`
    is all `todo!()` (`:6`, `:273-277`) behind a stale lance-4.0.0 BLOCKED list.

So the stack can REPLAY history but cannot NAME a removal. That is exactly the
gap a tombstone closes, and it is a semantic gap rather than a durability one.

## `WriteMode::Overwrite` — confirmed, and NOT the destroyer

CONFIRMED on every non-first write (`versioned.rs:286`). It is **not
destructive**: it mints a new version. Its real hazards are different:
- **fragment-id reuse** (`probe:342-378`, lance#8206) — the two-sided arm the
  row-identity probe gates with `LNC2_FRAGMENT_REUSE=forbidden|expected`;
- **the nodes/edges lockstep break** (`probe:403-411`) — `diff()` checks out
  the EDGES dataset at the NODES version number, so a direct `Dataset::delete`
  on nodes advances nodes to a version edges never had and `diff(C, D)` returns
  `Err(DatasetNotFound …/edges.lance/_versions/4.manifest)`, not a diff. The
  lockstep holds only while every write goes through `commit_encounter_round`.
  Ledger: `TD-VERSIONED-GRAPH-DIFF-LOCKSTEP-AND-NO-REMOVALS-1`.

## The prior art shows the right shape

surrealdb kv-lance: `tombstone: Boolean` (`schema.rs:62`), a delete recorded as
a **positive row**, reads filtered by predicate `tombstone = false`
(`schema.rs:162-165`), historical reads via `lance_version_as_of` +
`checkout_version` (`mod.rs:678-687`).

The property that buys: it **distinguishes "never existed" from "deleted at
V₃"**. That distinction is precisely what `graph_seal_check`'s `Staunen` cannot
make, and it is available for the price of one boolean column.

## Consequence for the alpha work

Alpha never deletes — a claim is only ever added or its `visits` incremented —
so the alpha overlay needs no tombstone of its own. But the **delete arm** of
lance's delta (`get_deleted_row_ids`) is the one arm that requires stable row
ids, and it is also the one arm `GraphDiff` structurally cannot represent. Both
facts point the same way: leave deletes to `graph_seal_check` as the one truth
for removals, and treat a durable tombstone as its own deliberate deliverable
rather than a side effect of the delta work.
