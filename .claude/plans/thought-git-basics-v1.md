# Thought git basics (D-TGB) — v1

**Status:** PROPOSED (2026-10-20), test-first; nothing built. Lab first. Every "exists" claim below names its file; every Lance claim was read in the `lance-13.0.0` / `lancedb-0.40.0` crate sources.

**Parent:** [`thought-git-api-v1.md`](thought-git-api-v1.md) (D-TGA), its section "Correction: the slot axis and the witness pointer" (#1628). Durable side: [`wal-shootout-v1.md`](wal-shootout-v1.md) (D-WAL) and [`causal-pickaxe-resident-git-lance13-temporal-v1.md`](causal-pickaxe-resident-git-lance13-temporal-v1.md) (#1605). Persistence pattern: [`split-tunnel-sparse-append-resident-fold.md`](../knowledge/split-tunnel-sparse-append-resident-fold.md).

In short: thinking with a git API, at the cost of a sparse write and a merge on read.

## Overview

This plan starts from the ideal state and plans only the basics that lead to it. Datalog awareness over thoughts comes later and is out of scope.

**The ideal state.**

1. **No stale branches.** A thought exists only while its kanban card is live. Its branch is opened by the Rubicon crossing and closed by the card's outcome. Nothing outlives its card, so there is nothing to clean up afterwards.
2. **Squash, prune and garbage collection are kanban outcomes, not a background job.** `Commit` lands, `Plan` re-plans (folding into a new thought), `Prune` discards.
3. **Every thought is a deterministic, replayable operator.** Replaying it from its receipt reproduces its writes. That is what makes discarding safe. Keeping old thoughts matters less than knowing what they were (awareness), which is the later Datalog layer.

**The Rubicon lifecycle already exists in code** (`crates/lance-graph-contract/src/kanban.rs`, `KanbanColumn` and `next_phases`):

| Rubicon phase | `KanbanColumn` | thought |
|---|---|---|
| predecisional | `Planning` | deliberating; no slot yet |
| the Rubicon crossing | `Planning → CognitiveWork` | the conscious branch decision: `branch()` takes a slot |
| actional | `CognitiveWork` | `write` / `diff` in the slot |
| postactional | `Evaluation` | `status` / `rebase` against main |
| outcome | `Commit` (absorbing) | `land`: one sparse batch, pointers rewritten to 0, slot freed |
| outcome | `Plan` (re-enters `Planning`, carrying the witness) | re-plan: the work folds into a new thought (`squash_into`), slot freed |
| outcome | `Prune` (absorbing; also `Planning → Prune`, the pre-Rubicon veto) | `discard`, slot freed |

`Commit`'s calcify step is DECLARED only: `kanban.rs` states no Lance-version-producing code exists for it. D-TGB-3 is the step that gives it one.

## What Lance 13 / LanceDB 0.40 already provide

Read in the crate sources (`lance-13.0.0/src/dataset.rs`, `dataset/refs.rs`, `dataset/mem_wal/`; `lancedb-0.40.0/src/table.rs`):

- **A global version identifier.** `refs::Ref::{VersionNumber(u64), Version(Option<branch>, Option<version>), Tag(name)}`; the source calls `Version(branch, version)` "a global version identifier". `DurableRef` is therefore the dataset plus a `Ref::Version`.
- **Branches.** `create_branch`, `checkout_branch`, `list_branches`, `delete_branch`, `force_delete_branch`. `create_branch` is two-phase and can leave a zombie branch dataset; `"main"` is reserved. Too heavy for one branch per thought: Lance branches are for long-lived experiments only.
- **Tags, checkout, restore.** `tags()`, `checkout_version(Ref)`, `restore()`. LanceDB's `Table` mirrors branch, tag, checkout and restore.
- **Delta between versions.** `delta().compared_against_version(v)` with `get_inserted_rows()`: an independent oracle for `diff` across durable versions.
- **The PR record.** `read_version_transaction(v)` returns a version's transaction and commit timestamp.
- **Retention.** `cleanup_old_versions(older_than, delete_unverified, error_if_tagged_old_versions)`.
- **MemWAL**, measured in D-WAL (`2026-10-15-d-wal-shootout.md`):
  - an LSM per shard with epoch fencing (one writer per shard);
  - crash-atomic only with `durable_write = true`;
  - acknowledged rows visible to the writing handle only until a flush, and in the base table only after compaction;
  - commit flat at 2.0–2.8 ms over 1,649 versions, where `LanceCycleWriter` grew from 5.8 to about 10.5 ms; slower than `LanceCycleWriter` at 64 casts per cycle and faster from 1,024;
  - 1.57–2× the disk, because WAL files are not removed;
  - the read scan grows without a separate compaction job;
  - decision D-LNC-7: `LanceCycleWriter` stays the default and MemWAL is an option; the user decides.

## Work items (test-first; each lands with its falsifier first)

- [ ] **D-TGB-0 Pre-registration.** Fixtures: one thought per Rubicon outcome (`Commit`, `Plan`, `Prune`, pre-Rubicon veto), three thoughts that reference each other through `w_slot`, one mutual pair. Oracle: a from-scratch run on main at the next version. Disables written before code.
- [ ] **D-TGB-1 Slot pool on the Rubicon crossing.** A slot is taken only at `Planning → CognitiveWork` and freed only at `Commit`, `Plan` or `Prune`.
  - Falsifier: no slot exists without a live card, so a stale branch is unrepresentable.
  - Edges: with 63 slots taken the crossing is refused, never given a 65th slot.
  - Disables: free on `Evaluation`; allocate at `Planning`.
- [ ] **D-TGB-2 Witness pointers.** `w_slot` (CE64 bits 53..58) qualified by the frame's base version (0) or the slot's generation (1..=63). The rewrite rules are #1628's.
  - Blocked on the reading decision in D-TGA's correction, item 2: `MailboxSoA::apply_edges` reads the same bits as the mailbox's corpus and drops edges that point elsewhere. D-TGB-2 does not change `apply_edges` before that decision.
  - Falsifiers: after `Commit`, no live edge carries the landed slot; an edge from a recycled slot's old generation is refused.
- [ ] **D-TGB-3 `Commit` lands through `LanceCycleWriter`.** The writer's commit boundary is the supervisor cycle, not the thought: it accepts one frozen batch per cycle and refuses a different batch for the same cycle (`HashConflict`). So every `Commit` reached in one cycle is coalesced into that cycle's single sparse batch (split-tunnel rules 1–3) and one Lance version, reconciled by the existing `(cycle, batch_hash)` path. A landed thought's PR entry is (version, thought id within the batch). `DurableRef` = dataset + `Ref::Version(branch, version)`. This gives `KanbanColumn::Commit` its missing calcify step.
  - Falsifiers: an empty cycle writes nothing; two thoughts landing in one cycle produce one version holding both; a re-submitted batch reconciles; two Lance branches both at version 7 are told apart.
  - Disable: one batch per thought; the second `Commit` in a cycle must then fail with `HashConflict`.
- [ ] **D-TGB-4 Receipt instead of history.** A thought's receipt is (base `DurableRef`, template or program id, seeds, slot generation).
  - Falsifier: replaying a discarded or landed thought from its receipt reproduces its writes cell for cell. That makes `Plan` and `Prune` safe to free.
  - Disable: drop the seeds from the receipt.
- [ ] **D-TGB-5 Reads are resident folds.** `checkout`, `log`, `diff` and `blame` fold over the resident sparse records.
  - Falsifier: each equals the from-scratch oracle; `diff` across two durable versions also equals Lance's `delta()`.
  - No Lance read on the hot path (rule 4).
- [ ] **D-TGB-6 MemWAL arm, behind a switch.** The same `Commit` path into a MemWAL shard per mailbox (`identity_sharding` on the owner), with compaction and garbage collection as a separate job.
  - Enabled only from the measured crossover (about 1,024 casts per cycle).
  - Reads stay resident folds, never MemWAL's DataFusion scanner (DataFusion is in its grace period).
  - Waits on D-LNC-7.

## Open

- **Card granularity.** Is the kanban card that owns a slot the mailbox's card, or one card per move inside a mailbox? `KanbanColumn` is per mailbox today.
- **Which reading owns `w_slot`.** Witness pointer (`deepnsm-v2` wire) or mailbox corpus (`MailboxSoA::apply_edges`); D-TGA correction, item 2. Blocks D-TGB-2.
- **Squash or land under slot pressure.** WORKING-MODEL in D-TGA: the `w_slot` pointer graph decides (topological land order, cycles squash first), under the witness reading.
- **The PR entry under MemWAL.** A land there makes no Lance version until compaction, so the PR entry would be a WAL position, or `Commit` forces a flush.
- **Later:** Datalog awareness over thoughts and receipts.

## Links

- STATUS_BOARD rows D-TGB-0..6.
- Entry `2026-10-20-d-tgb-thought-git-basics.md`.
- Parent correction: #1628, `thought-git-api-v1.md`.
