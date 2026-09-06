# 09 — PLAN. Staged, non-breaking, probe-gated

Status: DRAFT, 2026-09-06. Nothing here is built. Every stage names its gate
and its kill condition.

## The operator question answered first

> *"can temporal stay as is to avoid pollution and wire the storage in a
> non-breakable way at first"*

**YES — temporal stays exactly as it is.** Measured (s2): temporal.rs never
opens a dataset; it operates on in-memory `DeinterlaceRow` / `LocalCausalRow`
slices a caller already fetched, and every clock it reads is an app-level
field. **Nothing in it needs to change for stable row ids, for the delta, or
for the alpha work.**

The delta belongs in the **FETCH** that feeds temporal, not in temporal. That
fetch is `VersionedGraph::diff`. Keeping the change there is what makes it
non-polluting.

## Stage 0 — PROBES ONLY, no code change

| id | question | pass condition | status |
|---|---|---|---|
| **P1** | Are `_row_created_at_version` / `_row_last_updated_at_version` populated WITHOUT stable row ids? | A2 control (stable=true) reports the 2 appended rows; A1 (physical) is then interpretable either way | **RUN, RED.** Control held (2); physical returned **0**. Insert delta needs stable row ids. Update arm INCONCLUSIVE — its own control also read 0. See `01-delta-api-lance11.md` |
| **P2** | Can `ShardWriter::put` seal N landing rows + the frame row ATOMICALLY? | kill mid-flush on a 5,000-row cycle: 0 or 5,000 ⇒ FOLD stands; anything between ⇒ KEEP | not started |
| **P3** | Does ternlog chaining pay on THIS workload? | `ndarray/examples/ternlog_amortization_probe.rs` per-constraint cost flat in K, with the bandwidth column as the residency evidence | instrument exists, not re-run |

**P1 is the gate on Stage 3.** P2 is the gate on Stage 4. P3 gates any SPEED
claim about masks; the SHAPE claim does not need it.

## Stage 1 — the resident-form change. No addressing, no lance, no new dep

Make the alpha overlay's payload compact and give the 512-byte form a NAMED
materializer, exactly as `AlphaMask::materialize_ordinals()` already is for
ordinals — the module's own no-unnamed-materializer law, applied consistently.

- **Cost removed:** 464 zero bytes per claim.
- **Blast radius, measured (s4):** 5 true BREAK sites, ~15 trivial, **two real
  code edits** (MedCare's `overlay_to_batch`, `write_alpha_overlay`).
- **Safe because (s3):** ordinals and masks are NEVER persisted or serialized;
  only the base-independent 16-byte `AlphaAddr` reaches Arrow.
- **Must preserve:** claim order (the `(rung, seq)` monotonic traversal with NO
  sort), refusal-is-a-true-no-op, the 512-byte V3 stride on `merged_rows()`,
  and the parallel≡sequential byte-identity falsifier.
- **Ordinal-sorted storage is REJECTED** — it breaks the merge traversal. Sorted
  order can only ever be a named projection computed once at persist time.

**Kill condition:** if the parallel≡sequential falsifier cannot be kept green,
stop and report; that guarantee outranks the byte saving.

## Stage 1b — fix the release-mode mask guard (independent, do it anyway)

`AlphaMask::zip`'s only length guard is a `debug_assert_eq!` (`alpha.rs:273`),
compiled out in release. Make mismatched lengths fail closed. Small, safe, and
a precondition for any mask stacking.

## Stage 2 — the rung × tenant mask cross (the meta-awareness layer)

The object: a 10×N mask matrix over ONE allocation, each cell one `AlphaMask`
AND. Needs **no new stored state** — both operands are recomputed projections.
It is the first thing that would make the rung dimension observable at all.

Useful reads it unlocks: `tenant_mask AND NOT any_rung_mask` = "this domain was
addressable and no rung ever looked".

**Placement is forced:** if it is to be SIMD it CANNOT live in
`lance-graph-contract` (zero-dep by design). It goes one crate out, consuming
the contract's mask words. `mask_ternlog_assign` takes `&[u64]`, which is
exactly `AlphaMask`'s backing store.

**Trap:** never carry a rung ordinal in the residue band — the 3-bit band and
the rung ladder are unrelated enums sharing four variant names.

**Kill condition:** if P3 shows no amortization on this shape, build it scalar
and say so. The shape win stands without the speed win.

## Stage 3 — delta-backed `VersionedGraph::diff`. ⊘ P1 IS RED — GATED ON D-LNC-5

**Measured 2026-09-06: the insert arm returns nothing on physical row addresses**
(the control held, so the zero is real). All three delta arms therefore sit
behind the stable-row-id decision. This stage does NOT proceed as an
independent change; it becomes part of D-LNC-5. The one-to-one field mapping
below stays correct and is what D-LNC-5 should implement once row ids are
decided.

One-to-one, no new field, no new type:

| `GraphDiff` field | today | native |
|---|---|---|
| `new_nodes` | HashSet difference over two full materializations | `get_inserted_rows(from, to)` |
| `modified_nodes` | seal compare over the intersection | `get_updated_rows(from, to)` |
| `new_edges` | same on edges | `get_inserted_rows` on edges |
| *(absent)* | — | `get_deleted_row_ids` has nowhere to land |

This corrects D-LNC-5's scope: it was written for the DELETE arm, which is the
one arm `GraphDiff` structurally cannot represent, which is why its own probe
found it unreachable.

**Leave deletes alone.** `graph_seal_check` stays the one truth for removals.

**Kill condition:** P1 red ⇒ the whole stage is gated behind the stable-row-id
decision, which is D-LNC-5's to make, not this work's.

## Stage 4 — FOLD onto MemWAL. GATED ON P2

Keep the cast/descriptor layer; put durability on MemWAL through `WalSink`.

- **Retires:** hand-rolled `(cycle, batch_hash)` reconciliation and fencing —
  MemWAL's `writer_epoch` fencing gives the same guarantee WITH replay.
- **Gains free:** backpressure (`StoreFull`), which staging currently has none
  of and which the persistence plan already names as a required unbuilt
  property.
- **The thing that would be silently LOST:** the sealed read horizon is the
  INVERSE of MemWAL's memtable scanner. Nothing would fail; thoughts would
  simply begin seeing in-flight siblings as history. Any fold must re-establish
  it explicitly.
- **Must keep:** `on_behalf_of` pairing, `intent_moves`, ≤1-move-per-owner, the
  artifact gate, the delegation cache.

## Explicit NON-GOALS

1. **Do not build a new delta mechanism.** The sparse-delta rule is RATIFIED
   and IMPLEMENTED in `LanceCycleWriter`; the alpha overlay is its inverse and
   should conform, not compete.
2. **Do not wire `revision` into the kanban `try_advance`.** It stops before
   mutation by design. The pothole → rung degradation → revision handoff is a
   deliberate deliverable, not a side effect of this work.
3. **Do not enable `cleanup_old_versions` / any retention.** It is the ONE
   measured way to destroy time travel, there are currently ZERO production
   callers, and tagging must come first with the default OFF. Do not port
   surrealdb's `background_optimizer` defaults (5-min loop, 7-day, enabled).
4. **Do not touch temporal.rs.**
5. **Do not enable stable row ids repo-wide.** If P1 forces it, the safest
   first sites are `audit_sink/lance_sink.rs` (new partition per
   `(super_domain, date)`, isolated blast radius) and `lance_cache.rs`
   (explicitly disposable).

## Two debts to record rather than fix here

- `cycle_sink.rs:46` cites `lance-9.0.0/src/io/commit.rs`; the repo is on
  lance 11. Re-measure, do not renumber.
- `BatchWriter`'s `board` is never cleared — unbounded growth, currently
  harmless only because nothing production-side drives `run_cycle`.

## Board hygiene owed

PR #1112 and `d7bb9a98` have no `PR_ARC_INVENTORY` entry and nothing in
`AGENT_LOG`. The migration that made rung/alpha the substrate default is
unrecorded against this workspace's own mandatory rule.
