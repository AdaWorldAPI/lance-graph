# dir-sim persistence (W-3) — v1

**Status:** ACTIVE (2026-10-11). Consumer ask: HubSPO-rs handover
`.claude/handovers/2026-10-10-1600-hubspo-rs-to-lance-graph.md` W-3; meaning
half OGAR `.claude/handovers/2026-10-10-1600-hubspo-rs-to-ogar-dir-sim.md`
W-3 (`VersionId` maps 1:1 onto a Lance dataset version, `provenance.rs:16`).

## Overview

`VersionStore` is in memory only. HubSPO-rs needs a history that survives a
restart without re-observing AD, keeps provenance and tags, and can restore
an earlier version to undo a directory change.

## What is persisted, and why not the lanes

The persisted form is the store's **inputs**, not its derived planes:

| part | persisted as | why |
|---|---|---|
| each root | the `Observation` it was built from | `Snapshot` lanes are derived and lossy (`alias_key` keeps only the comparison form; `owner` / `mail_owner` are computed); their layout changes as planes are added. The observation is the source's own schema. |
| the dictionary | raw values in `ValueId` order | comparison keys and routing aliases are recomputed by `Dicts::intern`; interning the values in order reproduces every id, so persisted `Change`s stay valid. |
| each version | `Version` (id, parent, origin, delta) verbatim | the delta is what a simulation produced; rules are not re-run. |
| tags, verdicts | as held | a verdict is evidence about a version, kept with it. |

Restore = intern the dictionary in order, rebuild each root with
`Snapshot::build`, push the versions in id order. A restored store answers
every `view` exactly as the original.

Restoring an earlier version to undo a change is a new version, never a
rewrite: `diff(current, earlier)` is the change list that reverts, recorded
as a simulated version whose origin names the restored version.

## Work items

- [x] **P-W3-0 (probe first):** `StoreImage` (owned, in `lance-graph-dir-sim`,
      no I/O) + `VersionStore::image()` / `VersionStore::restore(image)`;
      test: a store with two roots, simulated versions, tags and verdicts
      round-trips, and every version's view, diff and plan are equal before
      and after. Done (`tests/persist.rs`); five disable
      runs red: dictionary order, verdicts, tags, delta check, dictionary
      completeness.
- [x] `VersionStore` keeps each root's `Observation` (needed for the image).
- [x] `VersionStore::revert(current, earlier, evidence)` — the undo as a version, recorded under `REVERT`.
- [x] **Decision 2(a), 2026-10-11:** persisted rules resolve by exact
      `(name, version)`. `StoredImage` holds every rule as text
      (`StoredRule`); `RuleRegistry` resolves only to the registered rule
      with the same name and version (`REVERT` always registered);
      `VersionStore::restore_stored` refuses an unknown rule
      (`RestoreError::UnknownRules`, every version listed); `History` reads
      such an image without ever resolving or replacing the rule, and
      exposes no write. Tests in `tests/persist.rs`; disable runs red
      (name-only match, `REVERT` not registered).
- [ ] **D-DSP-2, Lance writer/reader** in a separate crate
      (`lance-graph-dir-sim-lance`), so `lance-graph-dir-sim` keeps its
      no-I/O rule. Shape, decided 2026-10-11 (3(b)):
  - **Logical versions are not Lance versions.** `VersionId` is the
    history's own clock. A save appends one **batch** holding every logical
    version written since the previous save, in order; one batch = one
    Lance commit, so a Lance version counts saves, never dir-sim versions.
  - **Sparse append, resident fold** (`knowledge/split-tunnel-sparse-append-resident-fold.md`):
    a batch carries only what is new since the last save (values interned
    since, roots observed since, versions since, tag and verdict changes).
    Nothing is updated in place or compacted; Lance is read once, at
    restore, into a `StoredImage`.
  - **Verifiable batches.** Each batch carries a header record: batch
    sequence number, first and last logical version, record count, and a
    checksum over the batch's records, written in the same commit. Restore
    checks, batch by batch in sequence order: the sequence is dense, the
    first version follows the previous batch's last, the record count and
    checksum match.
  - **Torn saves are refused, never repaired.** A batch failing any check
    (missing records, a gap, a checksum mismatch, a header without its
    records) refuses the restore with the batch named. No prefix of the
    history is restored silently; reading up to the last good batch is a
    separate, explicit call, not a fallback.
  - **One writer.** The writer checks the dataset's latest version before
    and after its append (stale writer refused, race detected); a race
    leaves the log refused until reconciled. In-band idempotency stays
    OPEN, as in the reference implementation.
  - Records: `Observation`, `Change`, `Origin` and `Violation` need a
    stable encoding. **Amended 2026-10-11:** it lives in
    `lance-graph-dir-sim::log` (pure, no I/O), next to the types, so the
    storage crate only maps a `Record` to a row and back.
  - [x] **D-DSP-2a** `log`: `LogState::batch` / `committed`, `replay`,
        `replay_prefix`; per-variant round trips with every strict prefix
        refused; two-batch restore; torn saves (lost tail, lost header,
        flipped byte, missing batch) refused. Disable runs red: checksum,
        record count, batch gap, the sparse skip, stale commit.
  - [x] **D-DSP-2b** the Lance crate `lance-graph-dir-sim-lance` (own
        workspace, lance 13 without cloud stores; dir-sim itself stays
        free of lance): one batch = one Lance append (`Create`, then
        `Append`); the dataset version checked before (stale writer
        refused, nothing written) and after (`Race` when the commit is not
        the next version); read once at restore. Logical versions stay in
        the records; the Lance version is only the commit counter.
- [x] **Decision 4(b), 2026-10-11:** group-to-role grants are the
      authority's versioned policy, not directory state: OGAR
      `ogar_rbac::group_policy::GroupRolePolicy` (append-only, own
      `PolicyVersion`, `GrantProvenance` per change). dir-sim's
      `GroupRoles` is filled from a policy state; `None` (never named) and
      `Some([])` (all revoked) stay distinct (5(a)).
- [ ] Board: STATUS_BOARD row; HubSPO handover reply.

## Open

- ~~Whether a Lance version per dir-sim version, or per write batch, is the
  right mapping.~~ Decided 2026-10-11 (3(b)): per batch; the logical clock
  is independent (OGAR #354 corrects `provenance.rs`).
- Encoding of `Change` / `Origin` rows: they are OGAR types; their columnar
  form belongs next to them in OGAR or here.
- ~~`RuleId::name` is `&'static str`.~~ Decided 2026-10-11 (2(a)): a rule
  registry, exact `(name, version)`, unknown = refusal; `History` for
  read-only inspection.
