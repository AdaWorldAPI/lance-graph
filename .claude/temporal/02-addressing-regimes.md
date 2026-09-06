# 02 — the four addressing regimes, and the survey that cleared a change

The operator's warning, verbatim in substance: *"lance 11 now uses row
addressing while some use versions or old physical addressing."* Confirmed,
and it is worse than three regimes.

## The census (measured 2026-09-06, whole lance-graph tree)

| regime | where | production |
|---|---|---|
| Lance VERSION + HLC tick | `temporal.rs`, `QueryReference`, `DatasetVersion`, the whole planner/supervisor path | **yes** |
| content address: `NodeGuid` 16B + base-slice ORDINAL | `alpha.rs` and every overlay consumer | **yes** |
| Lance physical row address `_rowaddr` | `crates/lance-graph/tests/lance_row_identity_probe.rs` ONLY | no — test only |
| Lance stable row id `_rowid` | nowhere on disk | no — absent entirely |

`grep -E '_rowaddr|_rowid|with_row_address|with_row_id|stable_row_id|enable_stable'`
over all of `crates/*/src` returns NOTHING. Production addresses by version and
by content GUID; it does not address by Lance row in any form.

## temporal.rs is entirely version-addressed

Its deinterlace key is `(server_id, lance_version, hlc_tick)`; it merge-sorts
four producers each on its own clock (lance storage versions, surrealql
`knowable_from`, ractor `V_ref` horizons, the thinking trajectory) onto one
timeline. A grep for row-addressing terms over `temporal.rs` returns ONE hit,
and it is the word "stable" in a comment about SORT stability.

**temporal.rs never opens a dataset.** It operates on in-memory
`DeinterlaceRow` / `LocalCausalRow` slices already fetched by a caller, and
every clock it reads (`lance_version()`, `hlc_tick()`, `cast_seq()`) is an
app-level field. **Nothing in it needs to change for stable row ids.**

## The mapping chain a delta-backed alpha would need

```
AlphaAddr (NodeGuid, 16-byte content key)
  -> ordinal (a position in the base SLICE, not a Lance address of any kind)
  -> Lance stable row id
  -> delta row
```

Only the first arrow is guaranteed today, by `AlphaAllocation`'s `OnceLock`
ordinal index derived from the base slice itself. Each later arrow is a place
the mapping can drift silently — the `I-LEGACY-API-FEATURE-GATED` class: one
name, two semantics, discovered at runtime.

## The four-worker survey — ALL GREEN

| worker | scope | verdict |
|---|---|---|
| s1 | Lance write paths | 9 production sites, NONE constructs a row-id field. No breakage mechanism found. Safest default-flip candidates: `audit_sink/lance_sink.rs` (new partition per `(super_domain, date)`, isolated blast radius) and `lance_cache.rs` (explicitly disposable/rebuildable). Two risks flagged rather than guessed: `versioned.rs`'s Overwrite-on-every-write vs stable row ids, and whether `InsertBuilder::with_params` honours row-id params identically. |
| s2 | Lance read paths | Reads address by version or by column value, never by row position. **temporal.rs needs no change at all.** `VersionedGraph::diff` materializes BOTH versions then compares by `node_id` HashSet + seal bytes; edges by `(src_id,dst_id)` — purely content-addressed. Named exposure: fragment-id reuse under `WriteMode::Overwrite`. |
| s3 | ordinal / mask contract | **DECISIVE GREEN: ordinals and `AlphaMask` are NEVER persisted, serialized, or sent across a boundary.** Only the base-independent 16-byte `AlphaAddr` reaches Arrow, via `overlay_to_batch` / `merged_rows`; the ordinal position never enters a row. Masks are safe as a resident-only form. |
| s4 | resident-form break analysis | **5 true BREAK sites, ~15 trivial, only TWO real code fixes** — MedCare's `overlay_to_batch` and `write_alpha_overlay`, which genuinely need materialized rows and should call a named materializer. |

## The non-breaking fact

`_rowid` EQUALS `_rowaddr` while stable row ids are OFF, and becomes the stable
id when ON. The regimes COINCIDE in the default configuration, so introducing
the column is not itself a break.

## Latent defect found on the way (fix on its own merits)

`AlphaMask`'s combinators route through the private `zip()`, whose only length
guard is a `debug_assert_eq!` (`alpha.rs:273`) — **compiled OUT in release**.
Two masks of different `len` combined in release silently truncate to the
shorter word array; debug panics on the assert; release risks a later
out-of-bounds panic in `contains()`/iteration rather than a silently-wrong
boolean. Nothing does this today, and nothing prevents it.

## One drift found

`crates/lance-graph/src/graph/cycle_sink.rs:46` cites
`lance-9.0.0/src/io/commit.rs` as the measured source for rebase behaviour,
while the repo is on lance 11 — a full major stale. Re-measure, do not just
renumber. (`witness_tombstone.rs:273` cites lance 4.0.0 but is a dead/blocked
path, not a live write.)
