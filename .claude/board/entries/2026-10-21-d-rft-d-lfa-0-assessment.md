# 2026-10-21 — D-RFT-1..5 sequence, D-LFA-0 census, physical-shape proof plan

**Status:** PROPOSAL, documentation only. Nothing here is measured or admitted. Plans: #1667 (`d-rft-…`, `d-lfa-…`, `d-qsp-…`; not on main). The IDs below are #1667's; no new D-ids are minted. Sources: four read-only scout maps (relalg / Quack / R2IL / `mork-pathmap-probe::join` / `dir-rmap-probe::pop`; thought-git-probe / ACT-R / kanban / `cycle_driver`; rs-graph-llm `graph-flow` + `graph-flow-lance`). Each claim names the file it rests on.

## D-RFT-1..5: the smallest falsifiable sequence

The kickoff's six asks map onto #1667's five IDs as follows (asks in brackets).

| # | ask | existing parts (verified) | missing | falsifier |
|---|---|---|---|---|
| D-RFT-1 | [1] typed composition without foreign gather | Quack `Filter::Semijoin` (EXISTS, distinct, `Gather` over a resident foreign plane; NULL fk → FALSE); `mork-pathmap-probe` `JoinDomain` + `JoinExec::bag_count` (`Σ_y Π |side_i at y|`, no enumeration); `dir-rmap-probe::Pop` (distinct, witness-bound, 256-slot register) | one composition `R ∘ S` over a shared domain that declares its law (Distinct / Bag) and NULL key handling, and answers both with the same coordinates. Today the three live in three crates with three conventions | IAM diamond (u ∈ g1, g2; both grant r): Distinct = 1, Bag = 2, XOR = 0 must be refused as a law; a NULL key matches nothing. PostgreSQL arm: `count(*)` vs `count(DISTINCT …)` over the same join (reuse `k3_pg_parity`'s pattern) |
| D-RFT-2 | [2] substitution through numeric method coordinates | `set_rules::Arith` guards (Exact / Wrapping / Checked / IEEE) already refuse rewrites that change overflow or rounding | a rule `a = k·b, b = m·c ⇒ a = k·m·c` in the same catalogue, guarded by `Arith` | substitution fixture `a=2b, b=2c, z=2a ⇒ z=8c` holds under Exact; refused under Checked (overflow at the boundary) and IEEE; an e-graph is not needed for this step |
| D-RFT-3 | [6] datafrog parity, bounded recursion | `dir-rmap-probe` Warshall closure in a 256 tile (refuses > 256); D-TGB-10 keeps recursion on `datafrog` | a bounded-recursion fixture run by both, same answers and the same stop | nested groups with a cycle and a diamond: closure equal to `datafrog`; the 257th group refused, never truncated |
| D-RFT-4 | [3] a 64-thought frontier sharing versioned bindings | `thought-git-probe::SlotPool` (63 live slots + givens; generation-checked `Slot`, `Stale`); `Pop` witness (`Refusal::Stale`) | slot generation and binding witness are two separate staleness checks with no link | 63 thoughts each hold a population over ONE register version. A rebind invalidates all 63 at once; a freed slot's handle is `Stale`. Disable: drop the witness check → a stale population answers |
| D-RFT-5 | [4] replay after changed premises, [5] multi-support deletion | `oracle::next_version` (from-scratch), `Pop::repair` (re-derives, never clears; second path kept, test-pinned in #1666) | repair as a thought-git verb checked against `next_version` | add / delete / cycle / diamond / ABA: incremental = from-scratch on every fixture. CausalEdge64 59..63 (EpistemicState5) is not touched and not used as scratch |

Order: 1 → 2 and 3 in parallel → 4 → 5. D-RFT-0 (K3) is the prerequisite and is the implementation PR this entry ships with.

Not assumed: NARS revision as a semiring. Epistemic certification (59..63), causal provenance (receipt, D-TGB-4) and hypothetical results (live slots) stay three separate things.

## D-LFA-0: copy-versus-reuse census (rs-graph-llm → `lance-graph-flow`)

| part | where | verdict | why |
|---|---|---|---|
| `Task`, `TaskResult`, `NextAction{Continue, ContinueAndExecute, GoTo, GoBack, End, WaitForInput}` | `graph-flow/src/task.rs` | REUSE | the step vocabulary; no state of its own |
| `GraphBuilder`, `Graph::execute_session`, `ExecutionStatus{Paused, WaitingForInput, Completed, Error}` | `graph.rs` | REUSE | it does not call storage itself (`graph.rs:121`) |
| `FlowRunner::run` (load → execute → save) | `runner.rs:378-393` | REUSE as a driver | it is a driver, not the write owner |
| `Context` = `DashMap<String, serde_json::Value>` + chat history | `context.rs:390-394` | REUSE, handles only | `graph-flow-lance` already enforces that the only data key is `MATERIALIZED_KEY` |
| `SessionStorage{save, get, delete}` (InMemory, Postgres, Kanban, Surreal) | `storage*.rs` | REUSE the trait; DO NOT reuse `KanbanSessionStorage::{moves, replay}` as truth | a second replay store would compete with the thought-git receipt (D-TGB-4) |
| `LanceTask{source, filter, measure, rotate, execute, materialize}` | `graph-flow-lance/src/tasks.rs` | COPY the pattern | the crate path-depends on z8run checkouts that lance-graph does not have |
| `Envelope{handle, generation, source}`, fail closed on a newer source generation | `z8run-lance` via `graph-flow-lance/src/lib.rs:12-29` | REUSE the rule | the same witness discipline as `Pop` / `Qr` |

**Rubicon mapping** (`kanban.rs:102-110`: Planning → {CognitiveWork, Prune}; CognitiveWork → Evaluation; Evaluation → {Commit, Plan, Prune}; Plan → Planning):
- *think*: `CognitiveWork`;
- *counterfactual*: an `Evaluation → Plan → Planning` loop over a live slot;
- *do*: `Evaluation → Commit`.

Four things stay distinct:
- *commitment*: crossing into `CognitiveWork`;
- *proposal*: a `KanbanMove` cast into the supervisor's `BatchWriter`;
- *authorization*: the MUL gate in `cycle_driver::shade_owner` / `run_cognitive_work_gated`;
- *persistence*: `seal_cycle` → one `DatasetVersion`.

A flow's `End` never writes.

**API sketch** (not a frozen ABI; no third executor):

```text
lance-graph-flow
  task per ClassView method ordinal → LanceTask-shaped plan handle in Context
  NextAction::WaitForInput          = deliberation before the Rubicon
  ProposeTask (terminal)            → KanbanMove intent to BatchWriter; returns a handle
  resume                            → Envelope generation checked; stale → Error, no answer
```

**Falsifier proposal:** a flow that ends with a `Commit` proposal leaves the Lance dataset version unchanged until `cycle_driver` seals it. Resuming after the source generation moved fails closed. Disable: let the task write directly → the version moves before the seal.

## Physical-shape proof plan (unmeasured)

Fixture: #1666's `dir-rmap-probe` (65k users, 254 security groups). Shapes:
- direct ordinal (`KEEP_DIRECT`);
- sparse runs (`Runs`);
- DR32×128 (512 B);
- DR64×64 (512 B);
- DR256×256 (8 KiB).

Equal answers are required, including:
- snapshot invalidation (`Stale`);
- the 257th coordinate (`TooLarge`);
- nesting and cycles;
- independent support paths (repair keeps the second path).

Costs are measured and reported separately: cold bind, first touch, repeated fold, cache, prepare, prefetch, invalidation and materialization. A millisecond stage is reported as FAILED GATE, never hidden in an amortized per-user number. A faster shape that changes one answer is rejected. Known from #1666: the prepare stage is millisecond-scale (FAILED GATE); with GUID-ordered positions a team focus touches 69 of 254 tiles, so tile locality needs a Dn128 ordering first.

## Open

- **Owners:**
  - D-RFT-1..5 and D-LFA-1..3 are unclaimed;
  - EXO-119 (positional lens) and EXO-114 (order witness) are unassigned;
  - ADMISSION B still forbids value-directed foreign gather at PREPARE. Whether Quack's `Semijoin` `Gather` over a resident plane counts is not decided here.
- **egglog:** only an offline proof candidate (#1667). Not needed for D-RFT-2's first step.
