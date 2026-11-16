# D-DIRSIM-REOBSERVE: a directory re-read is stored as its verified changes (2026-11-16)

**Status:** TEST-PINNED, lab. Closes the first missing seam named in
`2026-11-16-g2-dr-virtual-projection-reconcile.md` (C): every re-read was a
new full root.

## What changed (`crates/lance-graph-dir-sim`)

- `VersionStore::reobserve(source, observed_at_ms, obs) -> Result<VersionId, ObserveError>`.
  It diffs the read against the version tagged `observed` (`diff_full`),
  applies the diff to that version as a check, and records a version with
  `parent = previous observation`, `origin = Observed { source, observed_at_ms }`
  and only the diff as its delta. A first read or another directory scope is
  a root, as `observe` makes. An unchanged read is recorded with an empty
  delta.
- `observe` is unchanged (its body moved into a private `insert_root`).
- `plan` takes intent from the nearest observed ancestor instead of the
  lineage root. Every lineage made before this change starts at an observed
  root, so their plans are the same; with re-observations, changes reality
  made between reads are never counted as intent.
- A refused re-observation rolls back what it interned
  (`Dicts::mark` / `rollback`, crate-private): no version, no tag move, no
  value left behind.
- `explain_membership` doc: a chain ending at an observed version means
  observed.

## Measured

- 100-user directory, one membership added: the re-observation's delta is
  1 change; its log batch is **125 bytes**, against **11 094 bytes** for the
  same read recorded with `observe` (a second root).
- Every re-read reads as a fresh root of the same observation
  (`diff(reobserved, root) == []` both ways): nodes created and deleted, a
  property changed, memberships added and removed, four reads in a row.
- The log of a re-observed history replays to a store with the same diffs
  and the same observed tag.

## Refusals and invariants pinned

- Duplicate node: `ObserveError::Build(DuplicateNode)`; the store image is
  byte-identical before and after.
- A simulated branch between reads is not the re-observation's parent and
  reads as before.
- Logical versions only move forward; each re-observation's parent is the
  previous observation; source and read time are kept.

## Disable runs (each red, then restored)

| Disabled | Failed |
|---|---|
| sparse route (re-read always a root) | 5 tests: one change, fresh-root parity, repeated reads, branch, bytes |
| nearest observed ancestor (back to lineage root) | `the_plan_takes_intent_from_the_nearest_observation` |
| dictionary rollback | `rollback_restores_the_dictionary_at_the_mark` |

## Not done

- `lance-graph-dir-sim-lance` was not re-run (its records are the existing
  `Version` encoding; the parent and origin fields already exist). The
  in-memory log round trip is tested here.
- Each re-read lengthens the lineage a read folds; when to start a fresh
  root (compaction) is the caller's choice via `observe`.
- The `TooManyProxies` refusal, which fires after interning, is covered only
  by the rollback unit test; no fixture reaches 2^32 proxy rows.
