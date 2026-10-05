# 2026-10-05 — Storage portability: Quack owns query semantics, Rubicon owns durable commit, storage supplies capabilities

Follow-up to `2026-10-05-quack-two-world-frontend.md` (#1328, merged). It applies the same resolve-once lifecycle to the storage boundary. **Contract and architecture only: no backend, no storage trait, no new type.**

## DECISION — the three boundaries

```
QUACK      makes reads portable:   stable numeric query semantics across backends
RUBICON    makes writes amortizable: stable durable-transition semantics across backends
BACKEND    supplies capabilities:  physical read and write capabilities, at both boundaries
```

- **SCOPE:** architecture for lance-graph's query and persistence boundaries.
- **BASIS:** the two-world contract (describe once, resolve once, canonicalize once, execute numeric) and the #1326 lifecycle, applied to physical execution.
- **REVISIT WHEN:** a second backend adapter is actually built. At that point the capability list below must be measured against it, not extended speculatively.

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
       transient execution
       Kanban / folds / masks          <- never a storage requirement
            |
            | 0..many operations
            v
          RUBICON
      durable transition
            |
            v
      sparse immutable commit
   NodeGuid × Version × field mask × payload
            |
      backend write mapping
       /      |       |      \
    MOCA   RocksDB  Iceberg   S3
```

Not every backend takes part in every layer, and not in the same way.

## DECISION — Quack ↔ backend: the read boundary

- **Storage does not own Quack semantics.** `quack::Query` is the stable logical and numeric query contract: no strings, no catalog lookup, canonical little-endian only where raw bytes are read. A backend may execute some or all of it physically.
- **A backend never approximates Quack semantics.**
  - Each part of a query either runs exactly on the backend, or runs beside the backend over numeric lanes the backend supplies.
  - Otherwise binding fails before execution.
  - "Close enough" pushdown is a correctness bug, not an optimization.
- **Binding happens once, before execution** (the #1326 law): `Query + schema + backend capabilities → resolved physical plan → repeated execution`. There is no capability check per row, and no dynamic dispatch per fold where the choice can be made beforehand.
- **Storage format is replaceable; query semantics are not.** The same `Query` may run over Lance native lanes, Arrow batches, DuckDB vectors, Iceberg/Parquet columns or decoded RocksDB values, provided the adapter reproduces the same numeric result. The canonical LE rule fixes byte interpretation where raw bytes cross a boundary. It does not require identical files.
- **No universal `Storage` trait.** Capabilities run in two independent, asymmetric directions:
  - **read:** projection, `EqU32`, ordered range, count, group reduce, semijoin, strided reads, mask-native execution, predicate pushdown;
  - **write:** immutable object append, atomic batch, compare-and-swap, snapshot/generation commit, manifest publication, compaction/rewrite.

  These are recorded as vocabulary, not as an enum. No type is added until a second backend gives one something real to describe.

## MEASURED — the capability seams that already exist in code

The binding-before-execution shape is already the code's shape. No new abstraction is needed to state the rule.

| seam | what it already does |
|---|---|
| `lance_graph_quack::lower(&Query) -> Result<Program, LowerError>` | The current physical binding (Quack → mask-risc). It refuses unsupported shapes before execution (`GroupedBlend`, `EmptyJunction`, `HavingAggOutOfRange`, …) rather than answering approximately. |
| mask-risc `validate` (`check_lane` / `LaneKind`, run by `execute_into` and `reference_execute_into` before any row) | Lane-kind and bounds checks are total and happen before the first row. An `Err` leaves scratch untouched. This is the "no capability check per row" property. |
| `lance_graph_contract::hotplug::Activation::resolve_for_context` → `ResolvedReading` | Storage reading resolved once per population, with named `ActivationDrift` failures. |

A future backend adapter is a sibling of `lower`: `Query → (backend-native part, local numeric remainder)` or `Err`, decided once.

## DECISION — Rubicon: the write boundary

```
transient state: Kanban / thoughts / folds
        |  potentially ~1M folds, in memory
        v
     RUBICON            the commit decision
        |
        v
 durable sparse commit  one amortized transition
        |
        v
     backend            "commit this generation"
```

- **The backend receives an already-amortized durable transition.** The contract is "commit this durable generation/change set", not "persist every logical operation".
- **None of the following is a backend requirement:** 64k concurrent writers, thought scheduling, Kanban state, Rubicon phases, one write per fold. The backend cannot tell whether a commit came from 1 fold or 10⁶.
- **The name is anchored in existing code.** In the contract, Rubicon is the commit decision point (`rubicon_witness`, `action.rs`: an action is `Pending` until the Rubicon commit boundary). **No durable-commit type exists yet.** This entry names where one belongs; it does not build one.
- **Fire-and-forget, with replay.** Intermediate fold states need not persist. Crash recovery needs one of:
  - a stable base plus deterministic input;
  - a compact replay/event input;
  - a periodic checkpoint;
  - an equivalent durable recipe (`C0 + input batch + deterministic recipe → C1`).

  None of these needs one write per fold. R2IL and reasoning machinery stay out of scope.

## DECISION — identity layers stay separate

| layer | meaning | owner |
|---|---|---|
| `NodeGuid` | semantic object identity | ours |
| Version / generation | logical object generation | ours |
| backend snapshot, sequence, file or fragment id | physical history | the backend |

`NodeGuid × Version` is the semantic lineage. It is never aliased to an Iceberg snapshot id, an S3 object name, a RocksDB sequence number or a Lance fragment id.

## OPEN — "Alpha" as the durable sparse write: a conflict with the existing contract

The proposed durable record is a sparse write over an immutable address:
- `NodeGuid × Version × mask × ChangedPayload`;
- merge on read: `effective = base ⊕ Δ(v1) ⊕ … ⊕ Δ(vN)`, with the mask deciding which coordinates each Δ overrides;
- only the coordinates a query needs are resolved, so projection pruning and sparse merge cooperate.

That semantics is sound. **It is not, however, what `lance_graph_contract::alpha` / `alpha_tunnel` (the split-tunnel) implement today.** The two disagree on four points:

| | proposed durable sparse write | existing `contract::alpha` |
|---|---|---|
| granularity | per coordinate/field (mask over a row's fields) | per row: `claim` materializes one whole 512-byte `NodeRow` (`alpha.rs:20-23`) |
| what the mask ranges over | fields of one address | `AlphaMask` is a population bitset over the overlay's addresses (`alpha.rs:60-62`) |
| read of an unwritten coordinate | falls back to the base (merge on read) | `None` = "not attended", explicitly **never** the base row (`alpha.rs:25-28`) |
| durability | the durable commit form, versioned | "ephemer daneben, verwerfbar": discardable whole, no bake row, no digest (`alpha.rs:11-17`) |

`.claude/knowledge/reference-frame-vs-motion.md` (`ISS-LXA-ALPHA-FIT`) also forbids a value that a bake measured from living in alpha, and allows motion to be "discardable whole, or versioned as runtime state". So a *versioned* motion overlay is permitted. A base-falling-back field merge is a different read contract.

**Not resolved here, and not renamed.** Calling the durable sparse commit "Alpha" would give one name two incompatible read semantics: "absent = not attended" versus "absent = inherit the base". The decision belongs to the operator:
- (a) Alpha gains a second, explicitly separate *durable* reading, with merge-on-read living beside the existing overlay read;
- (b) the durable sparse write gets its own name, and Alpha stays the discardable attention overlay.

Until then, this entry calls it the **sparse commit**. Neither `alpha.rs` nor the split-tunnel writer is touched.

## WORKING-MODEL — backend mappings (vision, not implementation)

None of these is built, promised equivalent, or measured. Each is a capability profile to verify.

- **Lance / MOCA.**
  - Read: `Quack → lower → mask-risc → ndarray` over native lanes (exists).
  - Write: sparse commit → sparse delta beside the spine → merge on read → native compaction → optional push to S3.
- **RocksDB.**
  - Read: exact key/range lookups, with the remaining numeric work local.
  - Write: sparse commit → versioned keys in one `WriteBatch` → newest-visible read or merge operator → RocksDB compaction.
- **Iceberg.**
  - Read: predicate and projection pushdown only where exact; the remainder runs locally.
  - Write: sparse commit → immutable delta/data files → snapshot/manifest publication → merge on read → later rewrite/compaction.
  - **This is an adapter mapping to investigate, not a claim that Iceberg implements the sparse-commit contract one to one.** Candidates are a base table plus a sparse delta relation, or native row-level delete/update where its semantics match exactly. Choosing needs a dedicated, measured experiment.
- **DuckDB.**
  - `Quack → relational subset translation → DuckDB`.
  - It is useful both as an execution backend and as a differential semantic oracle for Quack. `crates/lance-graph-quack/tests/duckdb_differential.rs` already plays that role.
- **S3 / object storage.**
  - Requires no in-place append.
  - Minimal mapping: a chain of immutable commit objects, plus an optional manifest/head/generation pointer. "Append" means appending immutable objects to the logical history, not appending bytes to one object.
  - With local MOCA objects as the hot tier, `~10⁶ transient folds → Rubicon → one or a few durable objects → S3` replaces 10⁶ remote writes with a handful. **Not measured; no performance claim.**

## Scope guards held

No RocksDB, Iceberg, DuckDB or S3 backend, and no `Storage`/`Backend` trait or capability enum. Kanban and Rubicon stay out of Quack, and no write-concurrency requirement is placed on storage. `NodeGuid`, version semantics, #1326, the IAM contracts, `alpha.rs` and the split-tunnel writer are unchanged. Alpha is not exposed to developer queries.
