# 2026-10-05 — Storage portability: Quack owns query semantics, a durable commit boundary owns writes, storage supplies capabilities

Follow-up to `2026-10-05-quack-two-world-frontend.md` (#1328, merged). It applies the same resolve-once lifecycle to the storage boundary. **Contract and architecture only: no backend, no storage trait, no new type. Alpha is unchanged and is not redefined.**

## DECISION — the boundaries

```
QUACK                  makes reads portable:  stable numeric query semantics across backends
CYCLE SEAL             makes writes amortizable: many transient operations -> one durable transition
BACKEND                supplies capabilities: physical read and write capabilities, at both boundaries
```

- **SCOPE:** architecture for lance-graph's query and persistence boundaries.
- **BASIS:** the two-world contract (describe once, resolve once, canonicalize once, execute numeric), the #1326 lifecycle, and the source audit below.
- **REVISIT WHEN:** a second backend adapter is actually built. Measure the capability list against it then; do not extend it speculatively.

```
                 DEVELOPER WORLD
         SQL / Rust / Java / SAP / IAM
                      |
                      v
             RESOLUTION MEMBRANE
        names -> ids, source -> numeric
        canonical LE at byte boundaries
                      |
                      v
                    QUACK
             stable numeric semantics
                      |
              physical binding (once)
             /        |         \
      Lance/MOCA   Iceberg     DuckDB / RocksDB
            |
            v
   Quack numeric execution
            |
            v
   Planning
            |  Rubicon: intent becomes action (Planning -> CognitiveWork)
            v
   CognitiveWork / Action
            |
            v
   transient execution                     never a storage requirement
     folds + Alpha overlay (transient, discardable, row-level attention)
            |
            v
   Evaluation / revision.rs                epistemic judgment; performs no write
     |-- NoIncrease, or IncreaseEligible with docket incomplete
     |      -> Plan -> Planning (re-deliberate, carrying the witness)
     |-- Suspend -> held in Evaluation (tension open, pending grounding)
     `-- IncreaseEligible + counterfactual Necessary -> Commit (accepted)
            |
            | 0..many casts (BatchWriter::cast)
            v
   cycle seal: DetachedCycleBatch::freeze -> one WAL write per cycle
            |   performs no epistemic evaluation
            |
            v
   durable net change        (current: full 512-byte image per dirty row;
                              future candidate: SparseDelta, field-level)
            |
      backend write mapping  (current seam: WalSink::commit_cycle)
       /      |       |      \
    MOCA   RocksDB  Iceberg   S3
```

Not every backend takes part in every layer, and not in the same way.

**Rubicon commits intent. Revision judges the result. The cycle seal commits state.** These are three distinct boundaries; none absorbs another.

## MEASURED — the lifecycle as wired today (audit 2026-10-05)

| step | code | status |
|---|---|---|
| Rubicon: intent becomes action | `KanbanColumn` Planning → CognitiveWork (`contract/src/kanban.rs`), read by `rubicon_witness` | implemented; not redefined here |
| revision: judge the result | `revision.rs` `GadamerRevision::revise` → `RevisionDelta { kind, evidential_effect, … }`; `RevisionVerdict { effect, counterfactual }`; `is_acceptable()` = `IncreaseEligible ∧ Necessary` | pure policy. Its output types "deliberately stop before actual-world mutation" (`revision.rs:450`). **No production caller**: only tests and `examples/probe_revision_attention_view.rs` |
| route on the verdict | `KanbanColumn::advance_on_revision` (`kanban.rs:262`) | contract only; **no production caller** |
| the seal | `persist_cycle` / `DetachedCycleBatch::freeze` → `WalSink::commit_cycle` | implemented and driven by `supervisor/cycle_driver.rs` |

How the verdict feeds back, per `advance_on_revision`:
- **`IncreaseEligible` and counterfactual `Necessary`** → `advance()` → `Commit`. The cycle settles.
- **`IncreaseEligible` with the docket incomplete** → `revise()` → `Plan`. Eligible is not accepted.
- **`NoIncrease`** (including `Echo` and `ClosedCycle`) → `Plan`. Understanding rose and evidence did not, so the cycle re-deliberates carrying the witness.
- **`Suspend`** → `None`. The mailbox stays in Evaluation; the tension stays open pending grounding.
- **Revision never prunes.** `Prune` is the MUL gate's `Block`, a different act on a different arm.

Feedback therefore goes `Plan → Planning` and re-crosses the Rubicon. It does not jump straight back into CognitiveWork (`next_phases`: `Evaluation → {Commit, Plan, Prune}`, `Plan → {Planning}`).

**Gap between this chain and today's wiring (recorded, not fixed):**
- The seal is **not gated on revision acceptance.** A cycle seals whatever artifact casts it holds. Kanban moves, including `Evaluation → Commit`, ride in `SweepSlot::paired_move` and are applied after the seal (`recover_and_apply` → `try_advance_phase`). A pure Kanban step writes nothing (`cycle_sink.rs:34-40`).
- The `Commit` column's "calcify" step is itself **declared, not implemented** (`kanban.rs:47-54`).

So "accepted → `BatchWriter::cast`" is the intended ordering, not a wired one. Making the seal wait for acceptance would be a deliberate change to the caster/driver, not something this entry does.

## DECISION — Quack ↔ backend: the read boundary

- **Storage does not own Quack semantics.** `quack::Query` is the stable logical and numeric query contract: no strings, no catalog lookup, canonical little-endian only where raw bytes are read. A backend may execute some or all of it physically.
- **A backend never approximates Quack semantics.** Each part of a query either runs exactly on the backend or runs beside it over numeric lanes the backend supplies. Otherwise binding fails before execution.
- **Binding happens once, before execution** (the #1326 law): `Query + schema + backend capabilities → resolved physical plan → repeated execution`. There is no capability check per row and no per-fold dispatch where the choice can be made beforehand.
- **Storage format is replaceable; query semantics are not.** The same `Query` may run over Lance lanes, Arrow batches, DuckDB vectors, Iceberg/Parquet columns or decoded RocksDB values, provided the numeric result is reproduced exactly.
- **No universal `Storage` trait.** Capabilities run in two independent, asymmetric directions:
  - **read:** projection, `EqU32`, ordered range, count, group reduce, semijoin, strided reads, mask-native execution, predicate pushdown;
  - **write:** immutable object append, atomic batch, compare-and-swap, snapshot/generation commit, manifest publication, compaction/rewrite.

  These are recorded as vocabulary, not as an enum.

## MEASURED — source audit (2026-10-05)

### Alpha: authoritative and unchanged

Files: `crates/lance-graph-contract/src/alpha.rs` and `alpha_tunnel.rs` (the split-tunnel). Types: `AlphaAddr`, `AlphaStamp`, `AlphaMask`, `AlphaAllocation`, `AlphaOverlay`, `AlphaClaim`, `AlphaError`, `AlphaTunnel`.

Current semantics:
- **Discardable transient overlay** (`alpha.rs:11-16`, "ephemer daneben, verwerfbar": no bake, no digest; `discard(self)`).
- **Whole-row claims.** `claim` materializes one 512-byte `NodeRow`: key copied verbatim from the base, edges zeroed, and an `AlphaStamp { cycle, seq, rung, visits }` in value slot 0. **The rest of the value slab stays zero.**
- **Population tracking.** `AlphaMask` is a bitset over the overlay's addresses (`alpha.rs:60-61`).
- An unclaimed address reads `None` ("not attended") and never falls back to the base (`alpha.rs:25-28`).

**Alpha holds no field-level information and no payload.** There is no API that writes data into a claimed row's value. `claim` writes only the stamp, `get` and `rows` are read-only, and `alpha_tunnel` only merges stamps by rung.

Alpha may *inform* a future durable change: it records where attention went in a cycle. It is not, and does not become, that change. Durable storage mappings are never called "Alpha storage".

### Rubicon: implemented code, but as the pre-execution phase crossing, not as the durable write

Rubicon exists in code (category A), but **not as the durable commit boundary**:
- `contract/src/kanban.rs` defines `RubiconTransitionError` and the Rubicon lifecycle transitions over `KanbanColumn`.
- `contract/src/rubicon_witness.rs` (`RubiconVerdict`, `RubiconReading`) reads which side of the Planning → CognitiveWork crossing a thought is on, from its focus mask. It reads; it never moves anything.
- `planner/src/owner_adapter.rs` preserves "the Rubicon crossing itself (`Planning → CognitiveWork`)" bit for bit.
- `contract/src/action.rs`: an `ActionState` stays `Pending` until the cycle decides the result sound, then commits out.
- `cognitive-compiler::RubiconPhase` is a separate phase enum.

So in current code Rubicon is the Heckhausen commitment point **before** execution (deliberation → implementation). It is not the point where transient work becomes durable.

### The actual durable write boundary: the cycle seal, not Rubicon

The amortizing write membrane exists, under different names:

| file / symbol | role |
|---|---|
| `planner/src/batch_writer.rs` `BatchWriter::cast` | ephemeral intent records; payload is a descriptor `(mailbox, dirty row-range, cycle)`, never delta bytes |
| `planner/src/persist_sink.rs` `persist_cycle`, `DetachedCycleBatch::freeze`, `SweepSlot`, `CycleFrame` | cast ⊂ chunk ⊂ cycle. The casts are stable-ordered, then folded per row (last state wins). **One WAL write per cycle, one `DatasetVersion`.** Intent-only casts produce `NoChange` and zero writes. |
| `planner/src/persist_sink.rs` `trait WalSink { commit_cycle, scan_sealed, … }` | the existing backend **write-capability seam**: one amortized durable commit per cycle, reconciliation by `(cycle, batch_hash)` |
| `lance-graph/src/graph/cycle_sink.rs` `LanceCycleWriter: WalSink` | the real Lance implementation: the sole, non-`Clone` writer |
| `lance-graph-supervisor/src/cycle_driver.rs` | drives it: planner casts → one `persist_cycle` |

This already realizes the properties the brief asks of a write membrane:
- the backend sees one commit per cycle, never one per thought, and does not know how many thoughts contributed;
- 64k parallel producers are fire-and-forget casters, never writers;
- recovery reads sealed landings through `recover_and_apply`; the intent records are not a WAL.

### What changed inside a row: tracked nowhere at field level

- **Granularity is the row.** `SweepSlot.row: u64` names the dirty SoA row, and `LanceCycleWriter` stores one `FixedSizeBinary(512)` **final image per dirty row per cycle** (`cycle_sink.rs:105-123`). The batch-writer descriptor is a dirty *row range*.
- **No field, coordinate or lane dirtiness exists in the substrate write path:**
  - no mask beside Alpha;
  - no split-tunnel metadata beyond the stamp;
  - no per-lane tracking;
  - no comparison against the base row.
- `lance-graph-hydrate::dirty::is_dirty` is per **dataset**: it compares `version_id()` against the version the caller obtained when it hydrated.
- The only attribute-level change representation is domain-local and in memory: `lance-graph-dir-sim`'s `Overlay` (per-attribute override maps) and `Change::SetAttribute { from, to }`. That is a directory-simulation model, not the substrate write path.

### NodeGuid × Version: not the write path's identity today

| layer | where it lives today |
|---|---|
| semantic object identity | `NodeGuid`: the first 16 bytes of the 512-byte payload; the write path keys by SoA `row: u64`, not by `NodeGuid` |
| semantic generation | `(cycle, batch_hash)` per committed batch: `FrameMeta`, `LandedSlot` ("semantic identity lives in `(cycle, batch_hash)`") |
| physical history | Lance `DatasetVersion`, deliberately not re-derivable per row |

There is no per-node `Version`. The separation the brief wants (semantic lineage ≠ backend history) already holds as `(cycle, batch_hash)` ≠ `DatasetVersion`. A future `NodeGuid × Version` lineage would extend that, never replace it with Lance, Iceberg, RocksDB or S3 identifiers.

## OPEN — SparseDelta (architecture vocabulary only, not a type)

A future durable sparse-change contract can represent changed coordinates over immutable semantic identity and support merge-on-read:

```
SparseDelta { semantic_identity: NodeGuid, version, changed_coordinates, payload }   -- representation unresolved
effective = base ⊕ Δ(v1) ⊕ … ⊕ Δ(vN), resolving only the coordinates a query needs
```

`changed_coordinates` could be a field bitmap, a lane bitmap, byte ranges, a typed coordinate list or a backend-native delta. **No choice is made, and nothing is implemented.**

**The smallest real seam is `DetachedCycleBatch::freeze` → `WalSink::commit_cycle`.** At freeze time the batch already holds the coalesced final 512-byte image per dirty row, and `CycleFrame.base_version` names the sealed predecessor. A SparseDelta is therefore derivable at exactly one place, by comparing each final image against the same row at `base_version`. That comparison is the missing field-dirtiness signal; today it does not exist. It could later become a refinement of `WalSink`'s image rows, not a new write path.

Alpha does not supply that signal (it carries no payload), and the split-tunnel writer is not touched.

## WORKING-MODEL — backend mappings (vision, not implementation)

None of these is built, promised equivalent, or measured. None is "Alpha storage".

- **Lance / MOCA.**
  - Read: `Quack → lower → mask-risc → ndarray` over native lanes (exists).
  - Write: `WalSink` / `LanceCycleWriter` (exists, full-row images). A SparseDelta refinement would be a native delta/chunk beside the base, merged on read and compacted natively, with an optional push to S3.
- **RocksDB.**
  - Read: exact key/range lookups, with the rest computed locally.
  - Write: a `WalSink` whose `commit_cycle` is one versioned `WriteBatch`, read newest-visible, with RocksDB compaction.
- **Iceberg.**
  - Read: pushdown only where exact, with the rest computed locally.
  - Write: one cycle → immutable data/delta files + one snapshot/manifest publication.
  - **An adapter mapping to investigate, not a one-to-one claim.** Candidates are a base table plus a delta relation, or native row-level delete/update where its semantics match exactly. This needs a dedicated, measured experiment.
- **DuckDB.**
  - `Quack → relational subset translation`, with a transactional mapping for commits.
  - Also a differential semantic oracle for Quack; `quack/tests/duckdb_differential.rs` already does this.
- **S3 / object storage.**
  - Write: one cycle → one immutable commit object, plus an optional manifest/head/generation pointer. "Append" means appending immutable objects to the logical history; S3 needs no in-place append.
  - With local MOCA objects as the hot tier, ~10⁶ transient folds → one or a few durable objects instead of 10⁶ remote writes. **Not measured; no performance claim.**

## Replay / fire-and-forget

Intermediate fold states are not persisted. This is already the code's rule: `BatchWriter` intent records are explicitly not a WAL, and recovery is a pinned-reference read of sealed landings. A cheaper future replay (checkpoint `C0` + input batch + deterministic recipe → `C1`) remains conceptual; R2IL stays out of scope.

## Scope guards held

- No RocksDB, Iceberg, DuckDB or S3 backend, and no `Storage` trait or capability enum.
- No `SparseDelta` type; Alpha is unchanged and not redefined; the split-tunnel and cycle writers are not redesigned.
- Kanban and Rubicon stay out of Quack, and no write-concurrency requirement is placed on storage.
- No change to `NodeGuid`, version semantics, #1326 or IAM.
