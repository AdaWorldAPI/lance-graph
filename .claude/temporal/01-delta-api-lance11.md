# 01 — lance 11's delta API, measured against the banked plan

Source: real `lance-11.0.0` and `lancedb-0.38.0` sources. Not on disk in this
container; pulled from the crates.io static CDN (crates.io's own API 403'd
through the proxy) and extracted to `/tmp/{lance,lancedb}-src/`.

## Delta is READ-ONLY

`Dataset::delta()` returns a builder with `compared_against_version`,
`with_begin_version`, `with_end_version`, `with_begin_date`, `with_end_date`,
`build()`. That builds a `DatasetDelta` with four async methods, every one of
them returning a `DatasetRecordBatchStream`:

```
list_transactions()      -> Result<Vec<Transaction>>
get_deleted_row_ids()    -> Result<DatasetRecordBatchStream>
get_inserted_rows()      -> Result<DatasetRecordBatchStream>
get_updated_rows()       -> Result<DatasetRecordBatchStream>
```

There is no write method anywhere on it. **It is purely a read over
already-committed versions.**

There is also **no combined change-set type**: no `RowDelta`, `ChangeSet`, or
generic `Diff` bundling insert+update+delete. They are three separate methods,
each with its own stream. A private `fragment_delta` helper exists, used
internally by the delete path only.

## The conditionality SPLITS — and this corrects an earlier statement

⊘ **Correction.** In session I first said all three arms are keyed on stable
row ids. That is right ONLY for the delete arm.

| arm | mechanism | needs stable row ids |
|---|---|---|
| `get_deleted_row_ids` | row-id set difference | **yes**, documented: *"Requires stable row ids at both endpoints and an ordered range"* |
| `get_inserted_rows` | filter `_row_created_at_version > begin AND <= end` | **no gate in its own source** |
| `get_updated_rows` | filter `_row_created_at_version <= begin AND _row_last_updated_at_version > begin AND <= end` | **no gate in its own source** |

`_row_created_at_version` / `_row_last_updated_at_version` are **ordinary
schema columns** whose constants (`ROW_ID`, `ROW_CREATED_AT_VERSION`,
`ROW_LAST_UPDATED_AT_VERSION`) come from `lance_core`. Both methods are
unconditional in the shipped 11.0.0 API surface — no `#[cfg]`, no feature gate.

### The open probe, stated as unverified

Nothing in `get_inserted_rows` / `get_updated_rows`'s own source gates on
`uses_stable_row_ids()`. **But whether the two version columns are POPULATED
and MEANINGFUL under physical-row-address mode is not determined by anything
in that file.** That is a probe, not a conclusion, and it is the single gate
on whether the insert/update arms are reachable without the stable-row-id
decision.

## `enable_stable_row_ids`

- Field on `WriteParams` (`src/dataset/write.rs`, struct ~:270, field ~:325).
- Default `false`.
- Settable at CREATE time, no migration needed if set from the start.
- `migrate_to_stable_row_ids` exists for existing datasets.
- Documented **"Experimental: ... stable after compaction operations, but not
  after updates."** That caveat is load-bearing and must not be dropped.

## `_rowid` vs `_rowaddr` — the non-breaking fact

Quoted doc confirms: `_rowid` **equals** `_rowaddr` when stable row ids are
off, and becomes the stable id when on. The two regimes COINCIDE in the
default configuration, so introducing the column is not itself a break.

## MemWAL is real, and it is NOT an addressing regime

Verified against source: a ~13.5K-line LSM subsystem present in both crates,
with a full `ShardWriter` / `LsmWriteSpec` API.

Decisive properties:
- MemWAL is a **pre-version staging area**. Flushed "generations" live under
  `_mem_wal/`; the base table is untouched (confirmed by test comments).
- **Its row ids are generation-local offsets assigned at flush time**, NOT
  lance's stable-row-id space.
- Unrelated to `enable_stable_row_ids` until compaction merges data into the
  base table via ordinary `merge_insert`.

**Consequence:** MemWAL cannot be used to ADDRESS an ephemeral overlay. It is
staging, not identity. An overlay parked in MemWAL has no stable name until it
lands.

Also flagged: a stale doc-comment contradiction between `table.rs` ("dispatch
is a follow-up") and the actually-implemented `merge/lsm.rs`.

## `memory://`

A real, test-exercised URI convention, but the lance/lancedb crates do not
themselves parse the scheme — it lives in a lower dependency that was not
fetched. UNVERIFIED beyond that.

## What this corrects in the banked plan

`.claude/plans/lance-convergence-staged-migration-v1.md` §4E and §7.7–7.9
discuss ONLY the delete side as "lance's native delta", and D-LNC-5's row is
phrased as *"row ids DELETED between two versions"*. That verdict stands for
deletes. **It was never a verdict about the insert side, which nothing here
had looked at until 2026-09-06.**
