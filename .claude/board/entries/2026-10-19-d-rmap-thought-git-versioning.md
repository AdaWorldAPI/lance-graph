# 2026-10-19 — D-RMAP-1A §7: RelationshipMap versioning reads Thought Git and #1605

**Status:** PROPOSED. Documentation only; no code, no type, no falsifier run.

**Plan:** [d-rmap-separate-tables-thought-palettes-v1.md §7](../../plans/d-rmap-separate-tables-thought-palettes-v1.md)

**Decision (PROPOSED):** D-RMAP keeps no durable revision counter. Map generation and palette generation are readings of `DurableRef{dataset, branch, version}` (sealed) or `ThoughtCommitId` (speculative); the thought-palette binding is `(ThoughtCommitId, palette generation)`, replacing `(slot, incarnation)`. Focus generation stays an ephemeral frame-local witness. The index seal stays separate unless RelationFacts uses stable row ids (H14c-5: compaction moved 3 510 of 3 510 `_rowid`s without them, 0 with them). Authorization scope stays its own key component. Replay reads `QueryReference::at(v, rung)` with `Strict`, never `QueryReference::default()`.

**Measured in code (read 2026-10-19):** no version type carries a dataset or branch (`temporal.rs` and `temporal_pov.rs` `LanceVersion = u64`, `scheduler.rs` `DatasetVersion(u64)`); `versioned.rs::commit_encounter_round` writes three datasets and returns only the nodes version; `AlphaStamp.visits` saturates and is not a generation; D-TGA has no thought incarnation.

**Gaps filed into the source plans:** `ThoughtCommitId` must be never-reused (thought-git-api-v1, causal-pickaxe); a dirty bit for writes after the last commit (thought-git-api-v1); multi-dataset writes need one `DurableRef` per dataset or a refusal (both).

**Falsifiers specified:** X1–X9 (§7.4), each a named refusal with the disable that must turn it red.

**Entropy:** ENTROPY-MILESTONES M28 (version spellings → `DurableRef` + `ThoughtCommitId`).

**Unchanged:** no CE64 change, no new executor or store, no per-fold commit, the no-gather ruling stands.
