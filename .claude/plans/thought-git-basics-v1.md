# Thought git basics (D-TGB) — v1

**Status:** PROPOSED (2026-10-20), test-first; nothing built. Lab first. Every "exists" claim below names its file; every Lance claim was read in the `lance-13.0.0` / `lancedb-0.40.0` crate sources.

**Parent:** [`thought-git-api-v1.md`](thought-git-api-v1.md) (D-TGA), its section "Correction: the slot axis and the witness pointer" (#1628). Durable side: [`wal-shootout-v1.md`](wal-shootout-v1.md) (D-WAL) and [`causal-pickaxe-resident-git-lance13-temporal-v1.md`](causal-pickaxe-resident-git-lance13-temporal-v1.md) (#1605). Persistence pattern: [`split-tunnel-sparse-append-resident-fold.md`](../knowledge/split-tunnel-sparse-append-resident-fold.md).

In short: thinking with a git API, at the cost of a sparse write and a merge on read.

## Overview

This plan starts from the ideal state and plans only the basics that lead to it. Datalog awareness over thoughts comes later and is out of scope.

**The ideal state.**

1. **No stale branches.** A branch is squashed, discarded or landed as soon as its result is read; nothing waits to be cleaned up. ⊘ Struck 2026-10-20 (operator): "A thought exists only while its kanban card is live. Its branch is opened by the Rubicon crossing". Branching is autonomous and costs folds; the Rubicon decision is the write (see "Correction: replay is the API, the write is the Rubicon decision").
2. **Squash, prune and garbage collection are kanban outcomes, not a background job.** `Commit` lands, `Plan` re-plans (folding into a new thought), `Prune` discards.
3. **Every thought is a deterministic, replayable operator.** Replaying it from its receipt reproduces its writes. That is what makes discarding safe. Keeping old thoughts matters less than knowing what they were (awareness), which is the later Datalog layer.

**The Rubicon lifecycle already exists in code** (`crates/lance-graph-contract/src/kanban.rs`, `KanbanColumn` and `next_phases`):

| Rubicon phase | `KanbanColumn` | thought |
|---|---|---|
| predecisional | `Planning` | deliberating; no slot yet |
| the Rubicon crossing | `Planning → CognitiveWork` | the conscious branch decision: `branch()` takes a slot |
| actional | `CognitiveWork` | `write` / `diff` in the slot |
| postactional | `Evaluation` | `status` / `rebase` against main |
| outcome | `Commit` (absorbing) | `land`: one sparse batch, landed cells relabelled to 0, slot freed |
| outcome | `Plan` (re-enters `Planning`, carrying the witness) | re-plan: the work folds into a new thought (`squash_into`), slot freed |
| outcome | `Prune` (absorbing; also `Planning → Prune`, the pre-Rubicon veto) | `discard`, slot freed |

`Commit`'s calcify step is DECLARED only: `kanban.rs` states no Lance-version-producing code exists for it. D-TGB-3 is the step that gives it one.

## Correction: replay is the API, the write is the Rubicon decision (2026-10-20)

**Status:** DECISION (operator, 2026-10-20) for the principle; WORKING-MODEL for the mapping onto `KanbanColumn` and for the anomaly predicate. **Basis:** the operator's words ("1.7ns x 1mio..1bio folds sind effizienter als ein Schreibzugriff … 99.9% sind autoattended autonomous git heuristics … Squash für stacking ohne Auffälligkeiten continue new branch … in memory replay als API und write als bewusste Rubikon modell Heckhausen Entscheidung wenn das Ergebnis zufrieden stellt"), and the implementation read on 2026-10-20: `alpha.rs`, `alpha_tunnel.rs`, `examples/givens_layer_probe.rs`, `examples/thought_layers_probe.rs`, `split-tunnel-sparse-append-resident-fold.md`.

1. **Folds against a write.** One fold costs about 1.7 ns (MEASURED, `2026-09-24-cost-scales-with-dirty-rows…`). The operator puts one write at 233–350 ms; the figures measured in this workspace are a 1.7 ms write floor (same entry), a `LanceCycleWriter` commit of 5.8–10.5 ms (D-WAL), and 336 ms / 532 ms for a Lance scan / a replay from Lance (split-tunnel doc). Either way one write buys 10⁶ to 10⁸ folds, so replaying a thought in memory is cheaper than writing it.
2. **99.9 % of the git operations are autonomous.** `branch`, `write`, `diff`, `status`, `squash`, `discard`, `replay` and continuing on a new branch run in memory as heuristics, without a decision and without a Lance call. They are the API.
3. **Squash stacks the unremarkable.** A thought whose result shows no anomaly is squashed onto the stack, and the work continues on a new branch. The anomaly signals already exist in code:
   - a contradiction record: a hypothesis offering a different value on a given lane (`givens_layer_probe`, `Tunnel::write`);
   - a regression: `visits > 1` (`AlphaOverlay::regressions`);
   - a cold/hot surprise: `expected.xor(&attended)` not empty (`AlphaMask`);
   - two thoughts reading one address differently (`ThoughtTables::differs`).
   Which of these stop a squash, and at what threshold, is OPEN and pre-registered in D-TGB-0.
4. **The write is the Rubicon decision.** A result is landed into Lance only when it is satisfactory: the conscious decision of Heckhausen's Rubicon model, taken once per result, not once per branch. The kanban card therefore tracks the intention to write, not the existence of a branch.
5. **Presence decides what is written (read in code).** Absence is not zero.
   - `givens_layer_probe`: per row a lane-presence mask plus the payload at the lane's popcount rank. A lane without its bit is unknown (the NaN case); a read returns the mask of lanes it filled, never a value for the others.
   - The same probe: a hypothesis offering the given's own value is agreement and is not stored; a different value on a given lane is a contradiction record and is never stored.
   - `thought_layers_probe`: `RowDelta` keeps only the 16-byte slots that differ from the base (XOR); an equal write is `Written::NoChange`; a look records only the saccade.
   - `AlphaOverlay`: an unclaimed address reads `None`, never the base row.
   So a sparse write is a presence bit plus a payload at its rank, and only for cells that differ.
6. **Rebase is replay.** A thought that fell behind main replays its receipt on the new base (D-TGB-4) instead of re-deriving incrementally. That costs folds and needs no recursion driver, so v1 does not wait on the banned D-RA-4 frontier driver.
7. **What "satisfactory" is measured against** (DECISION, operator, 2026-10-20: "JC crate wird dafür R als API für Statistik erhalten … Die unterschiedlichen Strategien auszuprobieren für 'Sudoku' und dann die 115 MeTTa Strategien zu testen. Branches als automatisiertes Hypothesen testen und synthesis").
   - Each strategy runs as one branch on the same Sudoku fixture against the same read-only base. The branches are automated hypothesis tests; their comparison and synthesis decide which result is worth a write.
   - `jc` provides the statistics for that comparison through an R-style API. Its interface is not specified here (OPEN).
   - The strategy engines run in parallel, one per crate. In the workspace today: `lance-graph-act-r`, `lance-graph-metta`, `lance-graph-nal`, `lance-graph-relalg`, `lance-graph-mask-risc`. Not yet in the workspace: `lance-graph-cog` (Hyperon) and `lance-graph-ona`.
   - Order: first the different strategies on Sudoku, then the 115 strategies. The 115 are the rows of `.claude/knowledge/nal-recipe-audit/nal_inventory.tsv` (NAL-1..9), run as MeTTa programs.
   - **How a MeTTa strategy runs** (DECISION, operator, 2026-10-20: "MeTTa = PeTTa, statt Prolog > datafrog; Ruff hat extractor; hyperon-experimental ggf"):
     - PeTTa's approach: MeTTa is compiled to relational clauses, not interpreted. The clauses run on `datafrog` instead of SWI-Prolog. `datafrog` is the AdaWorldAPI fork already used as the exact Datalog oracle (dev dependency of `lance-graph-metta`).
     - `ruff_metta_spo` is the extractor. It parses MeTTa with the Hyperon parser from the fork and marks every rule head `Lowerable` or interpreter-only. The natives come from PeTTa's builtin registry (`Natives::petta`), so the verdict is the PeTTa one.
     - `hyperon-experimental` (fork) is optional: a reference for the rules that are interpreter-only under PeTTa as well, and a cross-check of answers.
     - Measured in `metta-polyfill-v1.md` §3 (2026-10-14, single runs, ratios not primitives): join_10000 takes 1.47 s on PeTTa, 0.51–0.62 ms on datafrog, and does not finish on Hyperon (trie panic). The seeded ancestor query at n = 800 takes 0.42 s on PeTTa, 0.14 ms on datafrog and 416.9 s on Hyperon.
   - Same frame as the ce64 pre-registration "lane = treatment": one thought per strategy, the same fixtures, the same read-only base (`ce64-hypothesis-layers-v1.md` § Pre-registration).
8. **Reasoning state, git metadata and replay live in a separate table** (DECISION, operator, 2026-10-20: "effizienter relalg Datalog datafrog und Git Metadaten und thought replay in einer separaten Tabelle zu speichern damit es keine Abhängigkeiten gibt").
   - One table, separate from the data table, holds: the relations and derived facts of the strategy engines (relalg, Datalog, the datafrog oracle), the git metadata (the PR entry of each landed thought), and the receipts needed to replay a thought.
   - The data table carries only the sparse cells (item 5). It does not depend on the metadata schema, and the metadata table does not depend on the row layout.
   - A metadata row is keyed by (data dataset, data version, thought id), so it links to the kanban lane's version without being stored in it. This settles where a landed receipt lives (D-TGB-4) and the PR-record question in `thought-git-api-v1.md` § "A sealed version is a PR entry".
   - Like the data table it follows the split-tunnel rules: sparse append, read once at replay, folded in memory.
   - **Relations are stored as tries, behind the Schnellverschluss** (operator, 2026-10-20, recalling `mork-pathmap-schnellverschluss-v1.md`). The relation part of this table takes the layout of D-MPS-8, the versioned relationship table: one domain per shared variable, a presence plane and a generation per side, and a reseal of one side without the others. The thought branches share it: 64 thoughts over one 403 KB table, a 1,568 B filter plane each, a bag count in 5.2 µs per thought (MEASURED, probe only, n = 50,000). MORK and PathMap stay disposable indexes behind `boundary.rs`, used as oracles; they hold sets and refuse bag questions (`SetOnly`). Not admitted until the D-RMAP-1 BIND membrane review; cyclic bodies and chain joins over large domains stay open (D-MPS §7b).
   - OPEN: a Lance commit covers one dataset, so landing writes two versions, which are not atomic together (not checked against the Lance 13 sources in this session). Working rule until measured: write the data version first, then its metadata row naming that version. A data version without a metadata row is detected at replay and reconciled, as the split-tunnel `(cycle, batch_hash)` rule does for a single table.

**WORKING-MODEL mapping onto `KanbanColumn`** (to be pinned by D-TGB-0): `Planning` holds the autonomous in-memory work (branch, squash, replay); `Planning → CognitiveWork` is the decision to write a satisfactory result; `CognitiveWork` prepares the batch; `Evaluation` runs the up-to-date check and the revision gate; `Commit` writes; `Plan` returns to the in-memory work; `Prune` drops the intention. The table above (slot taken at the crossing) is superseded by this mapping.

## The reusable git API and its drivers (2026-10-20)

**Status:** DECISION (operator, 2026-10-20: "Fokussier dich auf git aber die Struktur sollte so sein daß es auf git als reusable API landet. Du kannst gerne Act-R mit einbeziehen für kanban. OGAR-LOCO und lance-graph-flow <> rs-graph-llm langgraph-ish sind die alternativen") for the split; PROPOSAL for the placement.

The thought git is one reusable API. Whoever decides when to branch, squash or write is a driver outside it, and the API must not depend on any driver.

| layer | owns | must not |
|---|---|---|
| **the git API** | the verbs (`branch`, `write`, `diff`, `status`, `squash`, `discard`, `replay`, `land`), the slot pool, presence and sparse cells, the anomaly signals as readings, receipts and the metadata table | choose what to do next; know a kanban column, an ACT-R buffer or a flow node |
| **a driver** | the policy: which branch to continue, squash or discard, and when a result is satisfactory enough to land (the Rubicon decision) | touch cells, slots or tables except through the API's verbs |

Drivers, all outside the API:

- **ACT-R** (`lance-graph-act-r`, included for kanban): `arbitrate` picks one witnessed intent from candidates evaluated against one snapshot, and refuses a batch with a stale candidate. The git API supplies the candidates (the open branches with their `status` readings), and ACT-R picks the intent: squash, continue, discard or land. ACT-R cannot execute; the intent comes back to the API as a verb call. Its `Snapshot` and witness are crate-local types today and are meant to move to canonical contract types.
- **`graph-flow-kanban` / rs-graph-llm** (langgraph-style): carries one mailbox's cycle through `KanbanColumn` and advances on a `mul::GateDecision` (the live kanban gate, REUSE in `.claude/v3/COMPONENT-MAP.md`; not the REPURPOSE `collapse_gate::GateDecision`). As a driver it maps the write intention onto the board (correction item 4).
- **`ogar-loco`**: the low-code program surface; a flow written there calls the same verbs.
- `lance-graph-flow` is named in `.claude/prompts/flow-cog-ona-rubicon-work.md` but is not a crate in the workspace.
- **`lance-graph-supervisor`, in parallel** (operator, 2026-10-20: "Lance-graph-supervisor > kanban_actor.rs readonly"). Read 2026-10-20:
  - `kanban_actor.rs` is read-only: `PhaseCensus` (one `&self` pass, one count per `KanbanColumn`), `mul_target` (the MUL gate lowered to the next legal phase) and `parse_kanban_step`. The actor and its advance / tick RPCs are deleted (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`). The git API reads the census; it never sends a message to advance a phase.
  - Phases advance only through `cycle_driver` (#879). It drains the staged casts into one WAL write and one `DatasetVersion` per cycle, then applies only the sealed sparse transitions, at most one per owner. Further intents of the same owner are held for a later cycle.
  - That is where `land` rides: a write intention is cast as the owner's paired move, and the landed thoughts of one cycle go into that cycle's single version, as D-TGB-3 coalesces them.
  - Pre-commit failure is deterministic regeneration from the sealed version ("sealed Vn + unchanged Kanban task + deterministic computation = the same provisional intent on the next sweep"). That is the same rule as rebase-as-replay (correction item 6), so the git API needs no recovery of its own before a commit. After a commit, `recover_fleet` replays sealed landings.
  - Related plans: `alpha-channel-rung-overlay-v1.md` §1 row C (`PhaseCensus` as the mask over activities) and §3l (a count is not a hydration precondition); `cycle-loop-closure-driver-v1.md`.

**Orchestration in phases, and the stack under the API** (DECISION, operator, 2026-10-20: "orchestration in Phasen durch loco oder actiondef und später abstrakt über Act-R, datafrog > darunter RISC arithmetic polyfill relalg r2il. Eventuell noch petgraph API als usability"):

| phase | who drives the verbs |
|---|---|
| 1 (first) | an `ogar-loco` program or an OGAR `ActionDef` (`ogar-vocab`): the order of verbs is written down, concrete and replayable |
| 2 (later) | ACT-R (`lance-graph-act-r`), choosing the next intent abstractly from the API's candidates |

Below the API, from top to bottom:

1. `datafrog` runs the strategy clauses (the PeTTa way, correction item 7);
2. the RISC arithmetic polyfill (`lance-graph-mask-risc`, which delegates to `ndarray::simd`);
3. `lance-graph-relalg` describes the non-recursive set expressions;
4. `lance-graph-r2il` lowers them onto mask-risc.

A petgraph API is an optional usability layer on top, not a storage format or an executor, as in `relstore-quack-petgraph-traversal-v1.md` (point 1 there). The existing petgraph experiment rests on the quarantined `OutgoingHop` gather and is not a precedent (same plan, point 2).

**Placement (PROPOSAL).** Following the convention, types live in the contract and execution in mask-risc / quack:

- the verb trait, the slot, receipt, `DurableRef`, `ThoughtCommitId` and the readings (`status`, anomaly signals) go into `lance-graph-contract` (zero-dep), so every driver depends only on the contract;
- the implementation over `AlphaAllocation` / `AlphaOverlay`, the presence planes and the metadata table goes into one implementation crate (name OPEN);
- each driver adapter lives with its driver, not in the API.

**Falsifier (driver independence, D-TGB-7):** the same Sudoku fixture driven by two drivers that make the same decisions (phase 1: an `ogar-loco` program and the same sequence as an `ActionDef`; phase 2: ACT-R and a script replaying its recorded intents) produces byte-identical cells, receipts and metadata rows. A run where the API reads a driver type, or a driver writes a cell without a verb, fails to compile or fails this test.

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

- [ ] **D-TGB-0 Pre-registration.** Fixtures: one write intention per Rubicon outcome (`Commit`, `Plan`, `Prune`, pre-Rubicon veto); a stack of unremarkable thoughts that squash and continue on a new branch; one thought per anomaly signal that must stop the squash; an unwritten lane that must read as absent, not zero. Oracle: a from-scratch run on main at the next version. Disables written before code. ⊘ Struck 2026-10-20: "three thoughts that reference each other through `w_slot`, one mutual pair" (W names a lane and points nowhere).
- [ ] **D-TGB-1 Slot pool, autonomous.** A slot is taken by `branch` and freed by `squash`, `discard` or `land`, without a decision.
  - Falsifier: after a run, every slot is either live work or free; no slot is held by a thought whose result was already read.
  - Edges: with 63 slots taken, unremarkable thoughts are squashed first; if none is, `branch` is refused, never given a 65th slot.
  - Disables: never free on squash; squash a thought that carries an anomaly.
  - ⊘ Struck 2026-10-20: "slot pool on the Rubicon crossing … taken only at `Planning → CognitiveWork`" and its falsifier "no slot exists without a live card".
- [ ] **D-TGB-2 Lane labels.** `w_slot` (CE64 bits 53..58) names the thought a state belongs to, qualified by the frame's base version (0) or the slot's generation (1..=63); it is not a pointer between thoughts. The relabel rules are #1628's.
  - W = thought layer is already decided (DECISION 2026-10-17, #1597). D-TGB-2 includes migrating the competing readers named in D-TGA's correction, item 2: `MailboxSoA::apply_edges` (its corpus filter moves to `SoaEnvelope::mailbox_owner`) and `deepnsm-v2`'s `WitnessTable` index.
  - Falsifiers: after `Commit`, no live edge carries the landed slot; an edge from a recycled slot's old generation is refused.
- [ ] **D-TGB-3 `Commit` lands through `LanceCycleWriter`.** The writer's commit boundary is the supervisor cycle, not the thought: it accepts one frozen batch per cycle and refuses a different batch for the same cycle (`HashConflict`). So every `Commit` reached in one cycle is coalesced into that cycle's single sparse batch (split-tunnel rules 1–3) and one Lance version, reconciled by the existing `(cycle, batch_hash)` path. A landed thought's PR entry is (version, thought id within the batch). `DurableRef` = dataset + `Ref::Version(branch, version)`. This gives `KanbanColumn::Commit` its missing calcify step. The sealed batch carries no thought meaning in CE64: `w_slot` is arbitrary once sealed (D-TGA correction item 7), so the batch is addressed by version and thought id, never by slot.
  - Falsifiers: an empty cycle writes nothing; two thoughts landing in one cycle produce one version holding both; a re-submitted batch reconciles; two Lance branches both at version 7 are told apart; a data version whose metadata row is missing (crash between the two writes, correction item 8) is detected at replay, never read as complete.
  - Disable: one batch per thought; the second `Commit` in a cycle must then fail with `HashConflict`.
- [ ] **D-TGB-4 Receipt instead of history** (stored in the separate metadata table, correction item 8). A thought's receipt is (base `DurableRef`, template or program id, seeds, slot generation). The template or program id records the thinking styles used. While the thought is live the receipt lives with it; when it lands, the receipt is linked to the kanban lane's Lance version, because the slot generation means nothing after the seal (DECISION 2026-10-20, D-TGA correction item 7).
  - Falsifier: replaying a discarded or landed thought from its receipt reproduces its writes cell for cell. That makes `Plan` and `Prune` safe to free.
  - Disable: drop the seeds from the receipt.
- [ ] **D-TGB-5 Reads are resident folds.** `checkout`, `log`, `diff` and `blame` fold over the resident sparse records.
  - Falsifier: each equals the from-scratch oracle; `diff` across two durable versions also equals Lance's `delta()`.
  - No Lance read on the hot path (rule 4).
- [ ] **D-TGB-7 The reusable API and the first driver.** The verb trait and its types in the contract, one implementation crate, and a phase-1 driver: an `ogar-loco` program or an `ActionDef` (section "The reusable git API and its drivers"). The ACT-R adapter is phase 2.
  - Falsifier: driver independence, as stated in that section.
  - Disable: let the implementation read a `KanbanColumn` to decide a squash; the scripted driver must then diverge.
- [ ] **D-TGB-6 MemWAL arm, behind a switch.** The same `Commit` path into a MemWAL shard per mailbox (`identity_sharding` on the owner), with compaction and garbage collection as a separate job.
  - Enabled only from the measured crossover (about 1,024 casts per cycle).
  - Reads stay resident folds, never MemWAL's DataFusion scanner (DataFusion is in its grace period).
  - Waits on D-LNC-7.

## Open

- **Card granularity.** A card now tracks a write intention, not a slot. Is it the mailbox's card, or one card per intention inside a mailbox? `KanbanColumn` is per mailbox today.
- **The anomaly predicate.** Which signals (contradiction, regression, cold/hot surprise, divergence between thoughts) stop a squash, and at which thresholds.
- **What "satisfactory" means for a write.** Measured with `jc` statistics across strategy branches on the Sudoku fixture (correction item 7); the statistic, its threshold and the `jc` R-style interface are OPEN.
- **Squash or land under slot pressure.** Squash the unremarkable first (correction item 3); landing stays a decision. The remaining policy is OPEN. It reads presence bits (the deltas' source bitsets, D-HL-3, and the written-by words), never `w_slot` as a pointer. ⊘ Struck 2026-10-20: "the `w_slot` pointer graph decides"; W names a lane and points nowhere.
- **The PR entry under MemWAL.** A land there makes no Lance version until compaction, so the PR entry would be a WAL position, or `Commit` forces a flush.
- **Later:** Datalog awareness over thoughts and receipts.

## Links

- STATUS_BOARD rows D-TGB-0..6.
- Entry `2026-10-20-d-tgb-thought-git-basics.md`.
- Parent correction: #1628, `thought-git-api-v1.md`.
