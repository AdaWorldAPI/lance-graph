# lance-graph-flow: bounded rs-graph-llm assimilation as a ClassView method driver (v1)

**Status:** PARALLEL DOCS-FIRST INVESTIGATION / PROPOSAL, 2026-10-21. 'lance-graph-flow' does not yet exist in lance-graph. Do not copy the entire framework, open a runtime fork, create a new session database, or take ownership of #1666, relalg, thought-git or the supervisor.

**Decision proposal:** YES to parallel **interface census + small contract probe** while relalg and thought-git remain the implementation priority. Runtime assimilation waits until a validated replay/semantics seam exists. This updates the question already recognized in 'thought-git-basics-v1.md' / EXO-98, without replacing that plan or introducing a second executor.

## Verified source capabilities

- 'AdaWorldAPI/rs-graph-llm/graph-flow/src/task.rs' has Task, TaskResult, NextAction (Continue, ContinueAndExecute, WaitForInput, GoTo, End, GoBack). 'graph-flow/src/graph.rs' has Graph, conditional edges and GraphBuilder. These are orchestration primitives, not relational execution.
- 'AdaWorldAPI/rs-graph-llm/crates/graph-flow-lance/src/lib.rs' already specifies a **lance-abi handle with source generation** between tasks, a z8run-lance registry/envelope shared by the graph and z8run, and explicit materialize-only data export.
- Its 'tasks.rs' rewrites immutable plans under handles; 'z8run.rs' registers graph-flow nodes, pause/session handles and resume, never passes row/mask data between layers. This is the smallest assimilable API seam.
- 'lance-graph-report' already lowers ReportPlan through Quack and one mask-RISC evaluator and includes a z8run-shaped connection; reuse it, never another runtime evaluator.
- 'lance-graph-contract/src/kanban.rs' already declares KanbanColumn/next_phases, phase transitions and MUL gate handling. 'lance-graph-supervisor' sealed cycle_driver owns state progression, WAL and versioned sparse apply. 'kanban_actor.rs' is read-only. No graph-flow task may autonomously move a mailbox state or commit a Lance version.
- 'lance-graph-act-r/src/lib.rs' has deterministic witnessed proposal arbitration; neither persistent cognitive buffers nor learning. The ACT-R caller evaluates eligibility elsewhere; action selection is not authority to commit.

## Desired user surface (ergonomics, NOT executor)

The LangGraph-ish appearance can be GraphBuilder -> Task -> NextAction / Pause -> Resume, with a **bound method handle** (ClassId, MethodOrdinal, law identifier, source generation, focus and thought version), not ORM objects. 'think', 'counterfactual' and 'do' are capabilities exposed as *method routes*, not three engines. ClassView declares legal method and action metadata; Quack Binding resolves operands; optional DR maps physical shape; relalg/R2IL/mask-RISC evaluates.

- THINK task: returns immutable derived result/Thought delta, not a new sealed version.
- COUNTERFACTUAL task: executes same method law over a witnessed overlay/focus; records delta/provenance; prevents hypothetical evidence becoming certified production state.
- DO task: evaluates method and guards, produces a witnessed ActionDef proposal; separate MUL/Heckhausen Rubicon phase and caller authorization determine whether to stage the effect. Actual write only through existing supervisor/cycle_driver + explicit Rust LanceCycleWriter terminal.
- A task's Context/session may contain IDs, generation-bound handles, small control state and external human reply; NEVER resident SoA row/plane, cached fold buffers or reconstructed node object graph.
- Live LangGraph-style pause/resume must fail closed on generation or codebook change; a stale session may deterministically rebind/replay but cannot reuse invalid effects or silently auto-commit.
- Do not equate psychological Heckhausen Rubicon (predecisional -> preactional -> actional -> postactional) with storage transaction commit or with Kanban columns by identity. Document a *declared mapping* to existing Planning, CognitiveWork, Evaluation, Commit/Plan/Prune and test every permitted transition.

## What to assimilate / what not to assimilate

| Adopt as thin behavior surface | Keep out of the execution/storage core |
| --- | --- |
| Task/GraphBuilder/NextAction ergonomic API, conditional branch, interrupt/resume | full graph-flow executor rewrite or independent ORM |
| Generation-bound lance-abi handle and z8run-lance envelope interop | PostgreSQL/in-memory session storage as cognitive truth source |
| Explicit terminal, operator-visible proposal/approval | side-effectful Task::run bypassing ActionDef/MUL/revision |
| Rig completion trait only at explicit LLM boundary if requested | LLM as default source of truth, learned epistemic cert from text |
| Graph flow as optional caller of thought-git verbs | duplicate scheduler/state-machine to replace supervisor's sealed cycle |

The z8run adapter is a **host** of flow graphs; Quack remains the evaluator. No requirement to install or fork LangGraph Python or instantiate graph-flow's original session backends in production. Names/labels resolve to numeric IDs on cold construction only.

## Staged parallel work and independent falsifiers

**F0 (parallel now, documentation only):** census which Task/NextAction/context/session/registry interfaces can be carried intact versus need a version witness or a typed refusal; attach concrete source paths. Identify and count forbidden stateful copies, duplicate evaluators, second write owners, default string dispatch, and all coercions between Rubicon and Kanban.

**F1 (after relalg T0 / thought-git T2 seam):** *one* representative graph with two method calls, a counterfactual branch, and WaitForInput. On pause, the saved session contains ONLY handle/control metadata. Simulated source update makes resume refuse. No node data materialization except terminal.

**F2:** independent phase-1 OGAR ActionDef driver and graph-flow driver run exactly the same scripted verb sequence against thought-git's from-scratch oracle. Verify byte-identical thought cells, receipts, mask outcomes and metadata; no action without explicit authorization. ACT-R selects a proposal in separate test with stale candidate refusal.

**F3 (optional after verified fences):** scaffold 'lance-graph-flow' crate as a small consumer-side surface; use existing code and dependencies only if legally/technically compatible. Preserve 'rs-graph-llm' as oracle/reference until equivalence proven; refuse dependencies/copying that violate the one-engine, one-truth-store contract.

**Stop conditions:** new execution loop or SQL evaluator, checkpointer becoming authoritative storage, conflicting Kanban transition owner, workflow graph identity becoming row identity, missing generation refusal, non-equivalent replay, or requiring pre-existing data materialization in Context. Halt runtime assimilation rather than papering over with adapters.

## Coordination

Main PR: [ClassView/Quack PostgreSQL parity plan](classview-quack-postgres-semantic-parity-v1.md). Priority companion: [relalg × thought-git frontier](relalg-thought-git-frontier-polyfill-v1.md). Existing source: '.claude/plans/thought-git-basics-v1.md' and EXO-98; '.claude/plans/relalg-v1.md' and EXO-86; EXO-50 messages. Any code handoff needs an explicit owner/claim and separate implementation PR. This PR changes plans only.
