# Co-research: D-TUPLE-POLYFILL-0, one addressed algebra, many execution systems

**Date 2026-10-31 · Status OPEN (exploration map; nothing adopted) · Plan refs: D-DYNBAG §17-§20, D-SOC-FENCE, d-xlog-native Amendment C**

**⊘ OPERATOR CORRECTION 2026-10-31 (read first; the map below predates it).** The frameworks are AdaWorldAPI forks, made so their internal representations can change. So:
- Premise (a) is reversed: internal tuple substitution inside a fork is in scope. Every "INTERNAL-NATIVE-KEEP" and "UPSTREAM-OWNER" verdict in §1, §4 and the map below is re-read under it. Keep the algorithm (union-find, congruence, fixpoint, scheduler, effects); replace the redundant physical representation.
- (b) stands: address reuse and value reuse are separate.
- (c) is softened: a small typed adapter or identity witness is allowed if documented and it stores nothing twice. A new carrier still needs an owner decision.
- Rubikon is an authorization boundary, not the substitution boundary. Redundant tuples below it may go; effect identity, order and idempotency stay.
- Reference pattern: lance-graph-java (R5 projection vs hydration, R11 layout is a schema).
- First target: one internal XLOG materialisation and one internal egglog projection, shown actually disappearing, measured against the unchanged upstream.
- New classes: DIRECT / CONDITIONAL / ALGORITHM-COUPLED / ABI-EFFECT / UNKNOWN.

**Premise gate (premise-auditor), as first run: PREMISE-SPLIT.** A tuple inside an external runtime belongs to that runtime; substitution is only possible where a framework reads from or writes to our substrate. Amortisation splits into shared addressing (one DR map revision) and shared values (one fold pass over lanes at one version). "Tuple polyfill" may not become a carrier type: a finding that needs one counts against the hypothesis.

Scouts: six (substrate, internal prior art, relational frameworks, e-graph frameworks, workflow/agent/enterprise, literature). Reports in the session scratchpad; the synthesis below spot-checked eleven citations at source.

Status: exploration only. Ratifies nothing. Cells: A logical, B physical, C identity/lifecycle, D semantics, E boundary.

## 0. Citations I checked myself (search finds, reading decides)

Read at source this session, all confirmed unless noted:
- `lance-graph-dt64/src/dr.rs:424-437` LocalMap holds `globals`, `sorted`, `sorted_local`, `lookups: Cell<u64>`; `:493-497` digest is "a digest, not an identity"; `:265-272` FrameRevisions {source, relation_map, coordinates}.
- `lance-graph-contract/src/kanban.rs:94-95` Planning -> CognitiveWork is "the -550 ms Sigma-commit / Rubicon crossing"; `:190-205` `advance_on_gate`.
- `lance-graph-contract/src/action.rs:25-45` commit only on RBAC + MUL Flow; "Maps onto the Rubicon commit boundary".
- `rs-graph-llm/graph-flow-action/src/lib.rs:101-124` `dispatch`: comment at the commit call reads "commit() IS { def-match · RBAC · Libet · Rubikon@SLA }"; `handle()` only on `Committed`.
- `lance-graph-relalg/src/program.rs:21-37` ProgramCost fields as scout 1 lists.
- `xlog-relcomp-probe/src/native.rs:29-33` imports `ndarray::simd` fold words directly; Cargo.toml:20-22 has ndarray, no mask-risc or Quack dependency (only a comment naming mask-risc). `:160-177` `ids()`/`nonzero()` decode masks into `Vec<u64>` / `Vec<(u64,i64)>`.
- `thought-git-probe/src/receipt.rs:1-12` "Replay runs the program again from the receipt."
- `egglog/core-relations/src/table/mod.rs:262` SortedWritesTable; `uf/mod.rs:77-83` DisplacedTable; `egglog-bridge/src/lib.rs:441,736,1720` add_term, rebuild_now, union.
- `scallop/core/src/runtime/dynamic/relation.rs:13-26` stable/recent/to_add/waitlist.
- `lance-graph-planner/src/nars/belief.rs:129-136` BeliefArena entries + `HashMap<CStmt,u32>` index.
- `dt64-admission-probe/src/lib.rs:176-186` (worktree agent-a2da49302a33bb1f3) decide calls `store.admit` then classifies.

Corrections:
- Scout 4 lists DisplacedTable fields as `{uf, displaced, lookup_table}`. The struct also has `changed` and `buffered_writes` (uf/mod.rs:78-82). Substance unchanged.
- Two code sites use "Rubicon" for DIFFERENT transitions: kanban.rs labels Planning -> CognitiveWork; graph-flow-action labels the commit step that includes RBAC. Scouts 1, 2 and 5 each picked one. Both are in code. See §6.
- Kanban.rs:96 calls CognitiveWork -> Evaluation "post-actional read-back". Scout 2, from the OGAR RFC, calls Evaluation pre-acceptance validation. The two texts disagree; I did not resolve it.

## 1. Tuple inventory

| Framework | file:line | A / B / C / D / E (condensed) | Class |
|---|---|---|---|
| XLOG probe | xlog-relcomp-probe/src/lib.rs:263 `facts()` | r,s base rels / `Vec<(u32,u32)>`, O(n²) `contains` dedup / per call / set / ingress from parsed XLOG Program | OUR-SIDE-FIXABLE |
| XLOG probe | native.rs:60-80 `Side` | same rel columnised / two `Vec<u32>`, one LocalMap lookup per row / per call / bag / feeds ndarray directly | OUR-SIDE-FIXABLE |
| XLOG probe | native.rs:84-131 rows/support | selection / `Vec<u64>` mask words / per call / set / internal | INTERNAL-NATIVE-KEEP (already target shape) |
| XLOG probe | native.rs:142-158 `bag_count` | per-Y derivation count / three `Vec<i64>` / per call / bag / to caller | OUR-SIDE-FIXABLE (should be GroupReduce Count x2 + checked mul) |
| XLOG probe | native.rs:160-177 `ids`/`nonzero` | answer / decode to `Vec<u64>`, `Vec<(u64,i64)>` / per call / set, bag / egress | OUR-SIDE-FIXABLE (unnamed materialiser; legal only as oracle) |
| XLOG | xlog-runtime/src/relation.rs:44-75 | named rel / `HashMap<String, VersionedCudaBuffer>` / name + version + mutation_epoch / per plan / GPU internal | INTERNAL-NATIVE-KEEP |
| XLOG | executor/join_cache.rs:6-84, executor/mod.rs:57 | join index LRU; `RelationDelta{insert,delete}` / semi-naive | INTERNAL-NATIVE-KEEP |
| XLOG | pyxlog/src/lib.rs:362-449 | DLPack column in/out | UPSTREAM-OWNER (only declared tuple boundary) |
| Scallop | runtime/dynamic/relation.rs:13-26; common/tuple.rs:5,18 | stratum rels; `Tagged<Tuple,Prov>`, boxed tuple trees / Rc<RefCell> per iteration / set keyed by tuple with provenance tag | INTERNAL-NATIVE-KEEP |
| Scallop | database/extensional/relation.rs:19; dynamic/output_collection.rs:5 | facts in `Vec<(Tag,Tuple)>`; output collection | BOUNDARY-LENS-CANDIDATE (I/O only); tag semantics UPSTREAM-OWNER |
| Soufflé | not local | per-index B-tree/brie, delta_/new_ rels | INTERNAL-NATIVE-KEEP (literature only) |
| egglog | core-relations table/mod.rs:262; uf/mod.rs:77 | function rows row-major + hash; union-find + displaced log / RowId per generation / keyed function table; e-class equivalence | INTERNAL-NATIVE-KEEP |
| egglog | egglog-bridge lib.rs:441,1720; parse_and_run_program | `&[Value]` of egglog-interned ids; program text | UPSTREAM-OWNER (no columnar import) |
| egglog | src/extract.rs:600-640, termdag.rs | owned TermDag on extraction | UPSTREAM-OWNER |
| egglog-r2il-probe | encode.rs:40; session.rs:75-93; lower.rs:46-66 | premises -> text; text -> Term -> `Vec<Step>` / cold, tens of rows | BOUNDARY-LENS-CANDIDATE, value ~0 (cold, scoped exception) |
| mlir-egglog, eggcc | egglog_optimizer.py:17-52; eggcc src/util.rs:770-790 | Python Term objects; egglog text program | UPSTREAM-OWNER. DialEgg not local |
| graph-flow | context.rs:391; storage.rs:10 | keyed blackboard DashMap<String,Value>; Session cloned on save / session id, no version / keyed + ordered messages | INTERNAL-NATIVE-KEEP; graph-flow-lance already carries handles |
| graph-flow-action | lib.rs:101-124 | ActionInvocation + HandlerOutcome / Pending -> terminal, sticky / transaction-like | INTERNAL-NATIVE-KEEP (boundary record) |
| graph-flow-kanban | lib.rs:1-25 | KanbanMove log / ordered events | INTERNAL-NATIVE-KEEP |
| NARS | planner nars/belief.rs:89,129 | Belief {stmt,truth,stamp,…}, statement-keyed unique / epistemic, stamp bitset anti-bag | INTERNAL-NATIVE-KEEP (arena); row read BOUNDARY-LENS-CANDIDATE |
| SAP | lance-graph-sap edge.rs:157-186 | BAPI ordinals over canonical fields; `BapiCatsInsert` payload | Ordinals already a lens; payload is a boundary record. LUW: not in repo |
| Rig, LangGraph, A2A, ABAP, BEAM, z8run | no local code (Rig not in cargo registry) | messages, checkpoints, Task/Artifact, LUW, mailbox | UPSTREAM-OWNER, evidence from docs or none |

Net: no external framework reads our substrate. Every tuple we can change is one our probe builds.

## 2. Substrate mapping and evidence

| Candidate | Existing words that serve it | Bag/set legal | Grade |
|---|---|---|---|
| probe `facts()`/`Side` | intake once into lanes; Binder (names -> codes) + LocalMap (ids -> positions); Quack `Filter::EqU32Via` | bag kept for one table (EqU32Via does not fan out) | VERIFIED-IN-CODE (words exist); composition not run |
| `bag_count` | mask-risc `GroupReduce{GroupKey::Lane, GroupFold::Count}` x2 + `checked_binary_i64` Mul (d-xlog Amendment C, D-DYNBAG MISSING-3 for the combine) | bag; combine of two group sinks is MISSING-3 | plan-level; combine word absent |
| `rows`/`support` | mask-risc `Pred`/`And` + `Count`/`Keep` | set | VERIFIED-IN-CODE |
| `ids`/`nonzero` | `Agg::Rows` returns a mask; ids only at a named `materialize*` egress | set (ids), bag (counts) | VERIFIED-IN-CODE |
| Scallop/egglog I/O | Quack describe + mask-risc Keep produce the selection; the framework still allocates its own row per fact | set only via Keep; tags and e-ids are not ours | API-limited (scout 4: egglog accepts text, AST, or `&[Value]` of its own ids) |
| NARS belief rows | read (stmt, f, c, stamp) as lanes | neither: truth is not multiplicity; stamp is a set | do not map to Count |

Bag gaps: no word expands one-to-many multiplicity; `Semijoin`, `ScatterOrU32`, `CountDistinctOrderedU32`, `Any/All`, RelAlg `Rel` are set-only. RelAlg has no join operator; join_body.rs classifies (GYO) but does not execute. `Semijoin` needs a resident foreign plane, so a fold cannot feed another fold at the adapter.

## 3. Safe substitution candidates

Only on our side. Falsifier for each: the change uses no new carrier type, no new ClassId/MethodId, no row decode except a named `materialize*`.

S1. xlog-relcomp-probe `facts()` + `Side`: one named intake to lanes, then Binder + LocalMap + Quack Filter, lowered to mask-risc. Removes the O(n²) dedup and the direct ndarray calls (the D-SOC-FENCE violation the d-xlog plan already names).
S2. `bag_count`: GroupReduce Count on each side; the per-Y product is MISSING-3. Until MISSING-3 lands, the one TRACKED direct `checked_binary_i64` call stays and is named as such.
S3. `ids`/`nonzero`: rename to `materialize_*` or confine to `#[cfg(test)]` oracles.
S4. egglog-r2il-probe encode/decode: technically lens-able, but cold, tens of rows, under the CLAUDE.md cold-cost exception. Gain is purity, not cost. Not worth a change.

Everything else in §1 is native or upstream.

## 4. Native structures that must remain

- egglog union-find, displaced log, rebuild index and congruence by re-insertion. Equivalence plus congruence is not a selection; masks cannot express find/union.
- Semi-naive deltas: XLOG `RelationDelta`, Scallop stable/recent/to_add, Soufflé delta_/new_. They carry iteration identity. DBSP shows delta needs the integrated base and a weight carrier, which WideFieldMask is ruled not to be.
- Provenance semirings (Scallop tags, Soufflé provenance columns): tag merge on duplicate is neither bag count nor set.
- NARS BeliefArena: unique statement index is the termination argument (belief.rs:122-126).
- XLOG JoinIndexCache, Soufflé indexes: framework-owned prepared bases.
- ActionInvocation, HandlerOutcome, KanbanMove, BAPI payload, A2A task, LUW, BEAM send: effect identity. Boundary records, never projections.
- graph-flow Context/Session: framework state; graph-flow-lance already passes handles.

## 5. Codegraph reconstruction opportunities

A lazy reading can rebuild, cold and on demand, from what already exists:
- the static body frame of a method from OGAR IR plus D-SAG-6 evidence rows. Caveat measured by ogar-behavior-probe P9: no ActionDef names an ogar-loco Program (0 of 50 across four producers), and Program has no frontend producer, so today the readable body is ActionDef facts and `body_source` text only; dt64-admission-probe already reconstructs rows (from board entries, not measured; no TSV committed);
- the composition of an R2IL procedure from procs.rs call sequences;
- the proof/term of an egglog rewrite from the cold egglog session.
The hot path emits only a per-run receipt (entry 2026-10-31-coresearch-mandatory-codegraph ADOPT-NOW). No store, no persisted e-graph, no graph lookup on the hot path (D-DYNBAG §20, Tract precedent).

## 6. Version / identity / effect boundaries

Diagram mapped to code:

| Diagram | Code today |
|---|---|
| ON | A2A task / BEAM message / NARS activation / z8run: no local code for A2A, BEAM, z8run. NARS activation is in planner nars/*. |
| WHEN | XLOG/egglog/Scallop native; graph-flow tasks; NARS revision; D-SAG-6 evidence classification inside decide. |
| Rubikon gate | Three gates, uncomposed: (1) kanban Planning -> CognitiveWork, `advance_on_gate` on MUL GateDecision (kanban.rs:94-95, 190-205); (2) `ActionInvocation::commit` = def-match + RBAC + guard + MUL, called by graph-flow-action `dispatch` (action.rs:25-45; lib.rs:117-123); (3) `ExecutionPolicy::decide` Pure / UnderDoGrant / refusals (lab branch, PR #1891, lib.rs:176-215). |
| DO | handler.handle (graph-flow-action), BAPI sink, mask-risc execution. |
| Observe / Receipt | D-TGB-4 Receipt (re-runs program), ActionInvocation state, audit receipts (Mediator, Fusion, Support). |

Plain statements:
- Three gates exist in code. None calls or composes with another.
- The word "Rubicon" names gate (1) in kanban.rs and gate (2) in graph-flow-action. Whether DO authorization sits inside the Rubikon gate (the diagram, and graph-flow-action's comment) or after it (the §2 anchor and the OGAR RFC §3 three-boundary split) is an owner decision. The code supports both readings because it uses one name twice.
- decide is a consumer of a DO grant, not the grant's issuer. `DoGrant` has no issuer.
- No receipt is replay-safe for external effects. D-TGB-4 re-executes the program, safe for pure compute only. ActionInvocation is a state record with an idempotency key, not a run record. SupportReceipt's own doc says the admission-event receipt does not exist (causal_audit.rs:259-262, not re-read by me).
- Tuples above the gate (WHEN) may be recomputed and are lens candidates. Tuples at or below carry effect identity and stay boundary records.
- Identity across processes: a Frame is process-local (Weak ptr); FrameRevisions are "for reporting, never identity" (dr.rs:354-358, not re-read by me). No serializable witness exists.

## 7. Amortisation mechanisms

- b1 addressing: one `LocalMap` prepared, shared by `&` within a process; identity is `digest()` carried as `FrameRevisions.coordinates`; guard with Frames from the same FrameStore. Evidence: gate_b.rs:672-700 runs four queries over one map (scout 1; not re-read). LocalMap is not Sync, not Clone, no registry. Cross-process sharing = prepare again and compare digests. No prepare counter exists (MISSING-5 is the reuse cache).
- b2 values: RelAlg `SetProgram` lets computed once per `table_version`, `VersionMismatch` guard, `ProgramCost` counters, `run_recompute` as the no-share baseline. Set semantics only. FoldDialect `facade_passes()`/`programs_run()` for R2IL.
- DR never caches values (D-DYNBAG §17, zero-copy law).
- Literature: Soufflé minimum chain cover is the one measured shared-basis result (memory up to 6x, time up to 2x; sharing only across prefix-compatible searches; hash bases are not shareable). Leapfrog Triejoin and NPRR share a sorted per-relation index under a global variable order; no benchmark numbers in the paper. Factorised DBs win 10²-10⁶ in size only for many-to-many; key/foreign-key data closes the gap. Selective late materialisation in a modern vectorised engine gains about 9-15% on average. Expect modest gains outside many-to-many workloads.
- Folding break-even still governs: Stage B must quote U and C_compose (folding-doctrine §3); the program-collapse probe shows the recogniser re-running per execute can make sharing lose at 1% population.

## 8. Open owner decisions

1. Compose the three gates, or keep them separate; which one owns the word "Rubicon". Owner: operator + OGAR (ActionDef/GatedOgarHandler) + contract maintainers.
2. DO authorization inside or after the Rubikon gate; who issues `DoGrant` (RBAC, GatedOgarHandler, other). Owner: operator + OGAR.
3. Whether FrameRevisions may become a cross-process identity (a serializable witness). Owner: DT64/DR owner.
4. Whether a bag join with multiplicity expansion belongs in Quack/mask-risc at all, given the no-tuple P0. Owner: operator; backend-first if yes.
5. MISSING-3 (combine two group sinks) and MISSING-5 (LocalMap reuse keyed by frame): land or not. Owner: mask-risc / DR owners.
6. Columnar import into egglog, Scallop, XLOG (DLPack/Arrow). Owner: the upstream projects; operator decides whether to ask.
7. Effect receipt shape for A2A task id / BAPI LUW / BEAM send, and its idempotency contract (OGAR falsification doc T4-T5 crash window). Owner: OGAR + operator.

## Exploration map

| Idea | Verdict |
|---|---|
| S1 probe intake + Quack/mask-risc routing | ADOPT-NOW (already planned as D-DYNBAG-0; this run adds no new reason) |
| S3 rename/confine `ids`/`nonzero` | ADOPT-NOW |
| S2 bag_count via GroupReduce | PROBE (needs MISSING-3) |
| Shared LocalMap across consumers (b1) | PROBE (Stage C) |
| SetProgram shared lets across frontends (b2) | PROBE (Stage B) |
| Lens over Scallop/egglog I/O | PARK (framework allocates rows regardless) |
| Columnar import into upstream frameworks | PARK (owner) |
| egglog-r2il-probe encode/decode lens | SKIP |
| Masks replacing UF, semi-naive deltas, provenance tags, NARS truth | SKIP |
| Single composed Rubikon gate | PARK (owner decision 1-2) |
| Factorised representation inside lance-graph | SKIP for now (nested value store; conflicts with one owner of bytes) |
| Generic tuple carrier | SKIP (evidence against the hypothesis by definition) |

## Pre-registered probes

All release build, median of 101 runs, x86 AVX-512 host named in the result, N = 65,536 and 1,048,576, selectivity ~1% and ~50%. Units ns per row or µs per query. Any ms figure = failed gate. Comparison by folds only, never by decoded rows.

Stage A (falsifier: no new type). H: the XLOG Gate B query (q(Y) :- r(1,Y), s(Y,4)) runs as intake + Binder + LocalMap + Quack + mask-risc with zero new types and zero ndarray calls outside mask-risc except the tracked `checked_binary_i64`. Count: new pub types (must be 0), direct ndarray fold call sites (must be <= 1), row decodes outside `materialize*` (must be 0), bytes allocated after intake. Kill: any new carrier type, or bag count differs from the XLOG oracle (support 1 vs count 4 fixture).

Stage B (b2 shared values). H: k = 2, 4, 8 queries sharing one predicate via SetProgram lets read fewer lanes than `run_recompute`. Measure: `ProgramCost.lane_reads`, `plane_reads`, `planes_materialized`, `plane_bytes`, µs per query batch, plus U and C_compose. Kill: shared run not faster than recompute by >= 10% at k = 4 and 50% selectivity, or scratch bytes exceed the lane bytes saved.

Stage C (b1 shared addressing). H: one prepared LocalMap serves four consumers with one `prepare` and lookups independent of consumer count. Measure: prepare calls (harness counter, since dr.rs has none), `lookups()`, ns per lookup, µs per prepare at 64/128/256 ids. Kill: lookups grow with consumer count, or a second `prepare` is needed within one process for an identical digest.

Stage D (codegraph lazy). H: the hot path of a method admitted by decide performs zero codegraph lookups; reconstruction runs only when a receipt is inspected. Measure: graph lookups on the hot path (must be 0), µs per cold reconstruction (reported, not a gate), receipt bytes per run. Kill: any hot-path lookup, or reconstruction needs a persisted graph.

Stage E (effect boundary). H: no current receipt can be read back to answer "did this effect happen" without re-execution. Measure, on a fixture: for D-TGB-4 Receipt, ActionInvocation, Mediator/Fusion/Support receipts, count re-executions on read (must be 0 for any claim of replay safety) and whether an effect identity (task id, LUW, idempotency key) is present. Kill (of the hypothesis that an existing shape suffices): every shape either re-executes or lacks effect identity. Expected outcome per this run: kill fires; that hands decision 7 to the owner, it does not license a new record.

## Operator correction (2026-10-31): `ogar-loco::Program` is not the method body

DECISION (operator). The October 30/31 codegraph architecture stands: **Kythe semantic facts + R2IL executable semantics + DT64/DR binding.** `ogar-loco::Program` is not the canonical method body, and the ogar-loco program model is rejected as that architecture. Struck, not deleted, in this entry: every statement that an `ActionDef` points to an `ogar-loco::Program`, that `body_rev` is a digest of such a Program, or that executable `PROC_DEF` in ogar-loco is a prerequisite.

Consequences:
- No `ogar-loco::Program` body, no new codegraph store, no second interpreter or method registry, no mandatory eager graph materialisation.
- P9's "no `ActionDef` links a Program" (category 1 = 0 of 50) is a fact about the data, not an admission gate. It blocks nothing.
- R2IL execution is kept without the ogar-loco program model: `FoldDialect::call` is inherent and the `ogar_loco::Dialect` impl is an optional adapter (#1895). The Kythe reading from witnessed facts exists (#1894).

OPEN: where the method's body revision is anchored once the Program is not the body.
