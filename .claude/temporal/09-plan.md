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
| **P1a** insert | Is `_row_created_at_version` populated WITHOUT stable row ids? | control (stable=true) reports the 2 appended rows | **RUN, RED.** Control held (2); physical **0** |
| **P1b** update | Is `_row_last_updated_at_version` populated WITHOUT stable row ids? | control: version advanced AND the row's value changed | **RUN, RED.** ⊘ first run was INCONCLUSIVE (its control did not hold — `when_matched` defaults to `DoNothing`, so no update committed). With the control: stable=true → **1**, physical → **0** |
| **P2** | Can `ShardWriter::put` seal N landing rows + the frame row ATOMICALLY? | kill mid-flush on a 5,000-row cycle: 0 or 5,000 ⇒ FOLD stands; anything between ⇒ KEEP | **RUN, GREEN.** SIGKILL sweep 5–500 ms: `0,0,0,5000,5000,5000,5000`. Boundary bracketed, no partial batch. FOLD stands. Scope: one put, local store, and NOT the landing-rows-plus-frame-row composite. See `08-…` |
| **P3** | Does ternlog chaining pay on THIS workload? | `ndarray/examples/ternlog_amortization_probe.rs` per-constraint cost flat in K, with the bandwidth column as the residency evidence | **RUN 2026-09-06 — GREEN, bounded** (see below) |

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

## Stage 1b — fix the release-mode mask guard — **DONE 2026-09-06**

`AlphaMask::zip`'s only length guard WAS a `debug_assert_eq!`, compiled out in
release. Now `assert_eq!`, so it holds in the build where the damage is silent.

**The defect was measured before it was fixed, in release**, and it is worse
than a truncation: iterator `zip` truncates to the shorter operand while `len`
is copied from `self`, so `wide.and(&narrow)` returned a mask claiming
`self.len` addresses over `other.words.len()` words. That is an INVALID mask,
not a smaller one, and it fails two ways at a distance — `count`/`is_empty`
under-report **silently**, while `contains`/`materialize_ordinals` index past
the slice and panic far from the call that caused it.

Two-sided falsifiers, both disable-verified: the mismatch case panicked only
after the fix (`should_panic`, and pre-fix the runner reported *"test did not
panic as expected"* — the defect, reproduced); the equal-length case proves
the guard stays silent on ordinary input, including `len % 64 != 0`, so a
guard that rejected everything could not pass both.

A length mismatch is a caller mixing two allocations — a programming error,
not a data condition — so it fails closed at the operation that made it. Cost
is one `u32` compare against a loop over every word. No API change; the ops
have no external callers yet, so this had no ripple.

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

### P3 result — 2026-09-06, `--release`, host L1d 48 KiB/core, L2 2 MiB/core

Amortization **holds**, and the sweep also fixes the two boundaries that bound it.

```
A. DEPTH SWEEP — mask 4 KiB, working set = (K+1)x4 KiB
  K  | T1 and | T3 ternlog | T3/T1 | T3 GB/s
   1 |  147.8 |      152.6 |  1.03 |    53.7
   2 |  149.2 |      103.1 |  0.69 |    79.5
   4 |   85.5 |       57.6 |  0.67 |   142.3
   8 |  106.3 |       58.3 |  0.55 |   140.6
  32 |  107.8 |       54.1 |  0.50 |   151.3
  64 |  117.1 |       64.8 |  0.55 |   126.4
```

**Pass criterion (pre-registered): per-constraint cost flat in K.** Held — T3
*total* is 57.6 ns at K=4 and 54.1 ns at K=32, no growth across a 16x increase
in constraint count, so per-constraint cost falls with depth. `T3/T1` bottoms at
**0.50**.

**The K=1 row is the control and it reads 1.03** — with one constraint there is
nothing to chain and the win is exactly zero. A probe that showed a win at K=1
would have been measuring something other than chaining.

**Boundary 1 — residency (sweep B).** The ratio holds 0.61-0.86 through L1 and
L2, then bandwidth collapses `138 -> 15 GB/s` from L2 to the largest L3 case and
the ratio drifts back to `1.03` at a 512 KiB mask. Past L2 the operation is
bandwidth-bound and the instruction-count win is masked. **The win is a
residency result, not a throughput constant** — it is contingent on the mask set
fitting L2, which is a design constraint on rung population size, not a free
speedup.

**Boundary 2 — density (sweep D).** Mask beats sparse down to 0.8% active; at
0.1% and below sparse wins (`4055` vs `1161` ns). The crossover is real and sits
far sparser than any rung population this workload has, so mask is the right
default here — but the crossover exists and a future sparse rung would cross it.

**What this does NOT change:** the standing caveat in `04-reusable-patterns.md`
- *"the ternlog wire is one of three mask passes and cannot account for 5x"*.
The 2x is real and applies to one pass of three, only while L2-resident.
Ternlog chaining stays a correctness/shape win first; the speed claim is
narrowly scoped by both boundaries above.

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

## Stage 4 — FOLD onto MemWAL. ✅ P2 GREEN — the gate is passed, the risk moves

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
