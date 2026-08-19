# Orchestrator pre-pass — lance 9.0.0 capability audit (PRIVATE until consolidation)

> Purpose: cross-check against RP-SEAL Domain-A independent reports at
> consolidation. NOT shown to researchers (independence rule). All findings
> from /tmp/sources/lance-9 (upstream tag v9.0.0; workspace Cargo.lock
> checksum 23d04bed056e254bc6e31264b031c8492507ca57939586f016924081dcf221a9).

## (a) Fragment-level write-without-commit: EXISTS in 9.0.0

- `InsertBuilder::execute_uncommitted(data: Vec<RecordBatch>) -> Result<Transaction>`
  — rust/lance/src/dataset/write/insert.rs:132; doc example shows the
  two-phase pattern explicitly (execute_uncommitted → CommitBuilder::execute).
- `InsertBuilder::execute_uncommitted_stream(source) -> Result<Transaction>`
  — insert.rs:181; doc: "Write data files, but don't commit the transaction
  yet. Use CommitBuilder to commit."
- `FragmentCreateBuilder` — rust/lance/src/dataset/fragment/write.rs:71;
  fragment-level `write_fragments` at :120.
- Top-level `write_fragments` (dataset/write.rs:586) is DEPRECATED since
  0.20.0 in favor of execute_uncommitted_stream; its doc: fragments "have
  not yet been assigned an ID... so this function can be called in
  parallel, and the IDs can be assigned after writing is complete."
- `do_write_fragments` (write.rs:597+) is the internal parallel writer.

## (b) Two-phase prepared commit: EXISTS

- `Operation` enum — dataset/transaction.rs:320 (Append among variants).
- `CommitBuilder::execute(Transaction)` — dataset/write/commit.rs.
- **`CommitBuilder::execute_batch(Vec<Transaction>) -> BatchCommitResult`**
  — commit.rs:475 — MANY prepared transactions in ONE commit; the
  many-petals→one-root shape natively. (BatchCommitResult at :512; a
  commented-out `rejected: Vec<Transaction>` field at :518 suggests
  partial-acceptance semantics are not final — verify behavior.)
- `.with_skip_auto_cleanup(...)` exists on CommitBuilder (seen in
  insert.rs do_commit).

## (c) Orphan/uncommitted cleanup: EXISTS, with an in-flight guard

- `cleanup_old_versions` — dataset/cleanup.rs:1350: removes "files that are
  not referenced by any valid manifest" → abandoned prepared fragments are
  collectable garbage BY DEFINITION.
- Unverified-file retention: cleanup.rs:32 "Otherwise we will leave the file
  unless delete_unverified is set to true"; RemovalStats tracks `unverified`
  (:122-124) — files younger than a retention threshold are NOT deleted
  unless explicitly forced. This is the guard that protects IN-FLIGHT
  prepared fragments from GC mid-preparation → F-GC's answer candidate.
- `auto_cleanup_hook` (:1356+) runs per lance.auto_cleanup config on commit.

## (d) Blob storage: EXISTS

- `Dataset::take_blobs` / `take_blobs_by_addresses` / `take_blobs_by_indices`
  — dataset.rs:1737/1752/1761; blob.rs has ReadBlob/ReadBlobRange builders;
  write-side ExternalBlobMode + `with_blob_pack_file_size_threshold`
  (write.rs:560-575, blob v2 .blob pack sidecar files).

## Reader-invisibility consequence (the epistemic point)

Uncommitted data files exist on storage but are invisible to every reader
because readers resolve exclusively through manifests. So "durable but
unpublished" exists NATIVELY in Lance without breaking snapshot atomicity —
the F-VISIBILITY guarantee moves from "structural absence" to "guarded by
manifest-reference", which is the git-object pattern precisely.

## Open questions the researchers should hit independently

- Transaction recoverability across restart (is a prepared Transaction
  re-derivable from its files, or in-memory only?).
- execute_batch conflict/rebase semantics under concurrent commits.
- Whether any of this surface changed between 9.0.0 and current upstream
  (two-column discipline).
- Row-address stability under compaction (stable row IDs? remapping?).

## Also banked: deltalake ceiling measurement

deltalake-core newest = 0.32.4, datafusion req ^53.1.0 (registry sparse
index, 2026-08-18) — no DF-54 deltalake exists.
