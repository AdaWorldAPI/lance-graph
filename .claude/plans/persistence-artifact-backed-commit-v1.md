# persistence-artifact-backed-commit-v1 — the canonical persistence contract

> **Status:** RATIFIED (operator ruling 2026-08-09). Phase A **implemented** on
> branch `claude/phase-a-owned-writer`; Phases B–F planned (§7).
> **Supersedes:** the cycle-persistence contract as shipped in PR #911 and the
> "one version per cycle, empty cycles included" reading of
> `persistence-cycle-wal-bootstrap-v1.md` §1 guarantee 5. Those documents are
> NOT deleted — they are read through this one, which is authoritative wherever
> they disagree.
> **Owns:** what may become durable, who may write it, what a commit returns,
> and what the normal path may read.
> **Does NOT own:** the OGAR-loco representation (Phase B), granular kanban
> visibility (Phase C), the clinical conclusion boundary (Phase D), MedCare
> wiring (Phase E), the A2UI renderer (Phase F).

---

## 1. The governing storage rule

**No artifact-backed semantic change → no write → no new `DatasetVersion`.**

Thinking is cheaper than persisting its intermediate control flow. A thought
runs its complete Rubicon/Heckhausen ladder **transiently** — often dozens of
cheap steps — and only what survived the ladder becomes durable:

```text
sealed inputs
  → complete transient thought ladder
  → artifact-backed semantic delta
  → canonical ABI row + compact metadata
  → non-empty batch
  → ONE official Lance MVCC commit
```

NOT `Planning → persist → CognitiveWork → persist → Continue → persist → …`.

A commit is allowed only for a real semantic delta: a new or changed grounding
result, a genuinely reusable ontology/RO walk product, an adjudication, a
terminal conclusion revision, a non-repeatable external-effect receipt, or
another explicitly registered semantic artifact.

**A timer tick, an empty cycle, `Continue`, `Hold`, a changed live priority, a
census observation, scheduler movement or a granular phase pulse is NOT a
durable artifact.**

Kanban never decides when to persist. If a semantic artifact is being written
anyway, the kanban progress accumulated since the preceding artifact rides
along as compact metadata in that same commit, under the same idempotency
identity.

### The mechanical form (Phase A, implemented)

The artifact gate is the cast's payload:

| cast shape | meaning | fate |
|---|---|---|
| NON-EMPTY payload | artifact cast — a real semantic delta | persisted |
| EMPTY payload | intent-only cast (held-intent re-stage, pure kanban step) | EPHEMERAL — never reaches the store |

`persist_cycle` partitions intent-only casts out **before** the freeze, so:

```text
zero artifact casts
  → CommitOutcome::NoChange { head }
  → zero sink calls · zero rows · zero frames · unchanged DatasetVersion
```

`restage_held`'s empty payload is therefore not a defect to be padded around —
it is precisely what makes a re-staged intent ephemeral. No payload/ABI gate
can ever trip on it, because such a cast never reaches the writer.

**#911's deliberate empty-cycle versioning is REMOVED, not repaired.** Its
falsifier `empty_cycle_advances_timeline_only` is inverted into
`no_artifact_delta_writes_nothing_and_creates_no_version`.

---

## 2. One logical writer — capability split, not merely method mutability

There is exactly **one logical application writer** per cycle store. The 64k
thoughts / SoA owners are parallel **producers**, not Lance writers:

```text
independent SoA owners/thoughts
  → cast_on_behalf(owner)            (fire-and-forget: no acknowledgement)
  → ephemeral BatchWriter staging
  → deterministic collect/freeze
  → ONE detached cycle batch
  → the SOLE writer's &mut commit
  → returned DatasetVersion becomes the new head
```

The split is by **capability**, not by uniform mutability (operator
correction, 2026-08-09):

| surface | shape | why |
|---|---|---|
| producer submission + read-only projections | `Clone + &self` | fire-and-forget; many producers |
| the concrete Lance writer | **non-`Clone`**, owns the `Dataset` handle + head, commits through **`&mut self`** | two application commits cannot interleave through the type boundary |

**Fire-and-forget means producers receive no acknowledgement — it does NOT mean
the sole writer ignores the commit result.** The writer fully honors every
outcome (§3).

Lance's own transaction/manifest machinery (the backend durability path) is
**internal to that one writer**; `Dataset::write` / `Dataset::append` are
official atomic Lance MVCC commits. Deliberately absent: per-plan writer
leases, multi-writer consensus, a second WAL, a bespoke confirmation ledger,
64k actor acknowledgements, a foreign-writer recovery protocol.

An unexpected head can therefore mean only: an earlier commit became durable
but its response was lost; a restart reopened from a stale cached head;
unauthorized maintenance or another writer violated the topology; or
corruption. **It is a fence/reconciliation condition, never normal
competition.**

---

## 3. No rollback, no compensating delete — reconciliation is authoritative

**Measured, not assumed:** Lance 9 has **no atomic expected-version fence for
`Append`**. The conflict rebase runs even on a single-attempt commit; strict
no-rebase mode exists only for `Overwrite`
(`lance-9.0.0/src/io/commit.rs:914-950`). This is stated honestly rather than
papered over with a read-check pretending to be compare-and-swap.

Once Lance publishes a manifest, that commit is **history**. `Dataset::delete`
creates *another* version and is not rollback — **#911's compensating delete is
removed entirely, not repaired with a nonce** (its `(cycle, base_version)`
predicate could also destroy a concurrent same-cycle winner's rows).

Instead, idempotency is **durable and in-band**: every committed batch carries
its `(cycle, batch_hash)` in the same commit, and the writer **reconciles
first**.

```rust
CommitOutcome::NoChange   { head }                          // nothing to write
CommitOutcome::Committed  { version, cycle, batch_hash }    // durable now
CommitOutcome::Reconciled { version, cycle, batch_hash }    // was already durable
CommitError::Fenced       { current_head }                  // nothing written
CommitError::HashConflict { cycle, stored_hash, offered_hash }  // fail closed
CommitError::Io           (WriteFailed)                     // nothing published
CommitError::Ambiguous    { cycle, batch_hash, cause }      // genuinely unknown
```

The rules that bind them:

- Failure **proven** before publication → discard transient staging, regenerate
  from the unchanged `Vn`.
- Success → accept the **actual returned version**; never "correct" it.
- Timeout / lost response → **re-submit the SAME frozen batch**; the writer's
  reconciliation-first lookup returns `Reconciled`, so the retry cannot
  double-append.
- Same identity, different hash → **fail closed**; never promote, never
  overwrite.
- Never delete history to restore an expected version number.
- Never regenerate assuming nothing landed until reconciliation proves it.

**No generic error may promise "nothing landed" when failure could have
occurred after manifest publication.** `Ambiguous` is the honest "I don't
know", and it is only reachable when reconciliation itself could not answer.

`batch_hash` is FNV-1a 64 over the **canonical** (already deinterlaced +
coalesced) content, so randomized worker completion order yields an identical
hash: determinism comes from canonical content, never from arrival order or a
process-local counter.

---

## 4. Reference the new version; never reload normal state

After a successful commit the caller **already knows** the submitted batch, the
affected owners and rows, the sparse transitions, the batch hash, and the
returned version. The hot path is therefore:

```text
outcome = commit(batch)
current_head = outcome.version
apply_sparse_effects_from_the_submitted_batch()
continue
```

It must NOT reopen the dataset, call the timeline, scan sealed landings, scan
the image, recover the fleet, rehydrate SoAs, replay the timeline, or run
read-time deinterlace to rediscover its own write.

**Instrumented, not asserted:** `LanceCycleWriter::opens()` counts every
`Dataset::open` the writer has ever performed. The falsifier
`successful_commit_reopens_nothing` drives three commits and asserts the count
stays at its post-startup value. The writer holds ONE long-lived `Dataset`
handle plus an in-memory head token; it re-opens only at startup and to resolve
an ambiguous outcome (both counted).

Reads are bounded and projected:

| read | bound | projection |
|---|---|---|
| `timeline()` | frame rows only (`kind = 0`) | `cycle`, `base_version`, `batch_hash` — payload column never scanned |
| `scan_sealed(after_cycle)` | `cycle > bound`, pushed into the Lance scan | transition metadata only; landing rows carry NO payload |
| `scan_image(cycle)` | `kind = 2 AND cycle = …` | `row` + payload, on request only |

Recovery takes an explicit `after_cycle` bound — the unbounded
`scan_sealed(None)` full-history read is no longer the recovery path.

---

## 5. Physical layout (Phase A)

One dataset, three row kinds, payload physically `FixedSizeBinary(512)`
(nullable — only image rows carry it):

| kind | rows per cycle | carries |
|---|---|---|
| 0 frame | 1 | cycle · base_version · batch_hash |
| 1 landing | one per artifact cast | stream_position · owner · row · move_* (nullable) · payload NULL |
| 2 image | one per DIRTY ROW | row · the FINAL 512-byte payload after the fold |

**The measured consequence (falsifier
`sixty_four_breaths_on_one_row_cost_one_image_row`):** 64 successive artifact
updates to one row produce **512 durable payload bytes**, not 64 × 512. The
coalesced image is the durable end-form; the per-cast landing rows are compact
transition metadata that recovery needs and carry no payload at all.

The 512-byte witness ABI is enforced **physically** (the Arrow column type) and
**at build time** (an artifact payload of any other length refuses the whole
commit before anything durable happens) — and it can only ever apply to
artifact casts, since intent-only casts never reach the writer.

### The honest copy boundary

Phase A materializes the frozen batch's payloads into Arrow builders (one copy
each) and copies bytes back out on read (`to_vec`). True zero-copy (Arc-backed
Arrow buffers pinned over SoA ranges) does **not** fit this focused repair. The
copy boundary is exactly those two seams and nothing else; it is isolated for a
later measured PR rather than claimed away. `BatchWriter<P>`'s doc calls `P` a
descriptor while `cycle_driver` instantiates `BatchWriter<Vec<u8>>` — that
contradiction is recorded here, not silently corrected.

---

## 6. `temporal.rs` — oracle, not ordering service

`temporal.rs` remains the temporal **admissibility and replay oracle**: pick a
pinned horizon, prove no future knowledge entered a result, reproduce
deterministic thought, validate latecomers and cohort composition, support
audit and restart recovery.

It is **not** a global write-order weaver, not the scheduler, not kanban
authority, not a normal post-commit operation, and never a reason to reload the
current fleet. The workspace's measured finding stands unchanged
(`.claude/knowledge/seal-vs-temporal-ordering-information.md`): the seal
computes a cross-owner total order, arrival as a durable input, the per-row
destructive fold, and the cohort boundary — none of which the per-owner
temporal projection encodes.

`DatasetVersion` remains a **physical publication position**. Combined with
dataset identity and an anchor identity it is also the simplest durable audit
address; the version alone carries no semantic identity.

---

## 7. Phase sequence

| phase | scope | state |
|---|---|---|
| **A** | this contract + the owned `LanceCycleWriter` | **implemented** (branch `claude/phase-a-owned-writer`) |
| B | shared representation + projection ABI (`VersionRef`, `PhasePulse`, `KanbanRollup`, artifact-backed `DurableAnchor`, OGAR-loco phase mapping, pure KanbanView/GothamProjection + golden rebuild) | planned |
| C | live granular visibility + wavefront (`EphemeralProgress`, `Continue`, pulse coalescing, plateau prioritization, latecomer semantics, rs-graph-llm/Blockly seams — zero durable pulse writes) | planned |
| D | conclusion boundary (`ConclusionRevision` as terminal `DurableAnchor`, immutable `(patient_episode, plan_id, revision)` identity, predecessor `VersionRef`, witness root, cohort manifest) | planned |
| E | MedCare proof + Gotham wiring (production caller, one real persisted medical thought, restart/reopen, bounded sealed read, no request-time reconstruction presented as sealed) | planned |
| F | A2UI/ClassView renderer (ABI projection, foveated ontology hydration, PII-free overlays, widefield masks) | planned |

Each phase restarts from merged `main`; public lance-graph stays generic and
clinical mappings stay private to MedCare-rs.

---

## 8. Authority table

```text
OGAR-loco Representation      plan truth (Phase B)
PhasePulse / EphemeralProgress granular live, crash-lossable visibility (Phase C)
KanbanRollup + semantic artifact  compact durable progress/result truth
VersionRef                     Lance-native audit address
KanbanView / GothamProjection  derived views over the same source
WitnessArcFacet                evidence / ontology / crosswalk detail
temporal.rs                    horizon / replay / admissibility oracle
Lance DatasetVersion           physical publication position
Lance commit machinery         backend durability of the SOLE writer
```

---

## 9. Phase-A falsifiers (implemented, green)

`crates/lance-graph/src/graph/cycle_sink.rs` (11, every one against a REOPENED
store — a fresh `LanceCycleWriter::open` over the same path):

1. `no_artifact_delta_writes_nothing_and_creates_no_version` — 2,000 intent-only casts ⇒ `NoChange`, no dataset created, restart still empty, then a real cycle commits at V1.
2. `sixty_four_breaths_on_one_row_cost_one_image_row` — the measured bytes-written falsifier (512 B, not 64 × 512).
3. `successful_commit_reopens_nothing` — `opens()` flat across three commits.
4. `bounded_tail_recovery_reads_no_payloads` — `after_cycle` bound + payload column never projected, transition metadata intact.
5. `timeline_is_frame_metadata_only`.
6. `resubmitting_the_same_batch_reconciles_to_one` — no duplicate rows, no second version, no delete.
7. `a_conflicting_batch_for_a_durable_cycle_fails_closed`.
8. `a_stale_horizon_is_fenced_and_writes_nothing`.
9. `intent_only_casts_leave_no_durable_trace` — mixed cycle persists only its artifact cast.
10. `randomized_completion_order_yields_the_same_durable_set` — identical batch hash + image.
11. `a_malformed_artifact_payload_is_refused` — 511 bytes refused, nothing written.

`crates/lance-graph-planner/src/persist_sink.rs` adds the contract-level pairs
(`intent_only_cycle_is_nochange_zero_store_calls`,
`retrying_the_same_batch_reconciles_never_duplicates`,
`a_different_batch_for_a_committed_cycle_fails_closed`,
`randomized_completion_order_yields_the_same_batch_hash`,
`after_cycle_bound_limits_the_sealed_scan`).

### Deferred with reasons (not silently skipped)

- **Real S3/object-store commit + ambiguous-response paths** — no credentials in
  this environment. **The capability IS compiled in**: S3 requires lance's
  `aws` feature, and `lance = "=9.0.0"` is taken with default features (which
  include `aws`, `azure`, `gcp`), verified in the dependency graph
  (`aws-config`/`aws-credential-types` reach `lance-io` → `lance` →
  `lance-graph`). `LanceCycleWriter::open` takes any URI `Dataset::open`
  accepts, so an `s3://` store compiles and routes today. What is unproven is
  the credentialed RUN — commit, reconciliation after a lost response, and the
  bounded tail read against a real bucket. **No object-store durability claim
  is made** until that falsifier executes; a deployment enabling S3 must not
  read "the aws feature is on" as "the path is measured".
- **`Continue`-from-pulse with zero Lance reads** — Phase C (the pulse
  machinery does not exist yet).
- **64k-scale gather/commit measurement** — the contract is proven at contract
  scale here; the 64k arm belongs with the measurement harness.
