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
- [ ] Lance writer/reader in a separate crate (`lance-graph-dir-sim-lance`),
      so `lance-graph-dir-sim` keeps its no-I/O rule: one Lance dataset per
      store, tables `values`, `nodes`, `members`, `proxies`, `versions`,
      `changes`, `tags`; one Lance version per dir-sim version written.
- [ ] Board: STATUS_BOARD row; HubSPO handover reply.

## Open

- Whether a Lance version per dir-sim version, or per write batch, is the
  right mapping (OGAR says 1:1 "by design"; a batch import would write many
  versions at once).
- Encoding of `Change` / `Origin` rows: they are OGAR types; their columnar
  form belongs next to them in OGAR or here.
- `RuleId::name` is `&'static str` (OGAR `provenance.rs:29`). An image held in
  memory restores fine, but one read back from bytes has no `'static` name to
  put there. D-DSP-2 needs either a rule registry the reader maps names
  through (unknown name = refusal), or an owned name in OGAR's `RuleId`.
