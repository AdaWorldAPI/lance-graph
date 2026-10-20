# 2026-10-20 — D-TGB — thought git basics planned from the ideal state

**Status:** PROPOSED, lab. **Basis:** VERIFIED-IN-CODE for the lifecycle (`lance-graph-contract/src/kanban.rs`, `KanbanColumn::next_phases`) and for the Lance surface (`lance-13.0.0` sources: `refs::Ref`, `create_branch`, `delta`, `read_version_transaction`, `cleanup_old_versions`, `mem_wal`); MEASURED for MemWAL (`2026-10-15-d-wal-shootout.md`).

- Thinking with a git API, at the cost of a sparse write and a merge on read.
- A slot is taken only at the Rubicon crossing and freed at `Commit` / `Plan` / `Prune`: no stale branch can exist, and squash, prune and GC are kanban outcomes, not a background job.
- `Commit` lands one sparse batch through `LanceCycleWriter`; `KanbanColumn::Commit`'s calcify step is declared only today.
- `DurableRef` = dataset + Lance's own global `Ref::Version(branch, version)`. Lance branches are two-phase and can leave zombies, so they are for long-lived experiments, not one per thought.
- Receipts make discarding safe: a replay from (base ref, program id, seeds, slot generation) must reproduce the writes.
- MemWAL is an optional `Commit` arm past the D-WAL crossover; reads stay resident folds.

OPEN: card granularity (mailbox card or per-move card); the PR entry under MemWAL; Datalog awareness later.

Correction (2026-10-20, review #1628): `LanceCycleWriter` commits one batch per supervisor cycle, so every `Commit` in a cycle coalesces into one batch and one version; a landed thought is (version, thought id within the batch). D-TGB-2 is blocked on the `w_slot` reading decision.

Correction (2026-10-20, operator): D-TGB-2 is not blocked. W = thought layer 0..63 is the 2026-10-17 decision (#1597); D-TGB-2 carries the migration of the competing readers (`MailboxSoA::apply_edges`, `deepnsm-v2`'s `WitnessTable` index).

## Status note (2026-10-20, later)

The full thought git API (`thought-git-api-v1.md`, D-TGA) is still a PROPOSAL. `crates/thought-git-probe` implements the slot pool (D-TGB-1), the from-scratch oracle and the tests (D-TGB-0). It does not implement the production land, rebase or persistence pipeline (D-TGB-2..5 are Queued).
