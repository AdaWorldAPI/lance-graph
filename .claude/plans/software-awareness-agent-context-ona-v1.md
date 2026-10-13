# D-SAG-4 — Addressable software context and ONA-inspired execution feedback

> **Aligned by [`sag-capstone-alignment-v1.md`](sag-capstone-alignment-v1.md)** — read it first: canonical `ProposedMethod`, the shared status vocabulary, D-id map, and the claims it marks stale (incl. "second hop refused").

**Status:** RESEARCH / unimplemented follow-up. **Base:** [merged D-SAG-1 #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509), merge commit `42fa2a347cda22eaaa327bba514f2029cc9daa82`. **Code examined:** `crates/lance-graph-glove-parity/src/awareness.rs`, `method.rs`, `gremlin.rs`; [D-SAG-1 plan](software-awareness-graph-v1.md). **Date:** 2026-10-12.
**Companions:** [execution prompt](../prompts/d-sag-4-ona-agent-context-implementation.md), [primary research excerpts](../../.arxiv/2026-10_sag_ona_internal_experience_source_excerpts.md), [repository-graph papers](../../.arxiv/2026-10_sag_repo_graph_context_source_excerpts.md).

## 0. Problem and result that would change our mind

Can a local, source-backed graph answer the *structural* part of Claude Code's repository-context questions without having to repeatedly send prose descriptions and file scans through the language-model context window? Can the same graph record versioned execution receipts and make falsifiable predictions about legal optimizations?

**Testable claim, not a result:** precomputed symbol/semantic edges, expanded by existing Quack queries, can reduce **model-visible input tokens and tool invocations** while maintaining or improving the quality of dependency discovery. Local queries themselves require zero LLM inference tokens, but the compact results sent to Claude consume tokens. A claim of zero total tokens would be false.

**Single architecture:** canonical software nodes -> declarative edges/evidence -> Quack `Draft` -> existing mask-risc executor -> bounded typed context slice or read-only optimization proposal. No Cypher/Gremlin/ONA interpreter in the hot path; no second semantic method registry; no autonomous edits.

### Shipped baseline (must not regress)
- #1509 projects one concept's prepared methods/fields/conditions/folds/laws/programs, refusals and compiled primitives into typed rows; its `SoftwareGraph` implements Quack's `Binder`. One-hop, one-program `ScatterOrU32` questions have been tested.
- #1509's code/plan describe in-memory rows, caller-provided source revision and current multi-hop blockage. No persisted Lance index or automatic full-source AST harvesting.
- #1509's H1 is an **optimization hypothesis**. A materializing program was slower than a fused **scalar oracle** on particular workloads/tiers; no fused SIMD backend was proved.
- #1508/G1 and #1504 establish relevant method composition/preparation, but source symbol navigation is a distinct identity and provenance problem.
- The supposed seven-language Bible graph is **not shipped**. #1509 audited an English KJV triple experiment, a small six-lane lexeme fixture and German labels. Reuse only the abstract rule: concept identity is not text spelling.
- Here **SoC means separation of concerns**, not a fabricated system-on-chip.

## 1. Separate three graphs, unify through qualified evidence edges

1. **Source-symbol graph** (candidate ingestion): repository, file, module, definition, function, trait, impl, field, callsite, test, build target and source span. Use compiler-backed definitions/references where possible; Rust Analyzer/LSP or an appropriate indexer is a *source adapter*, not a second reasoning engine. Source symbols are tied to `git commit / worktree digest / build cfg / macro expansion`; never mint an OGAR concept from a matching Rust name.
2. **Semantic-method graph** (already partially implemented): OGAR-qualified concept, `MethodKey`, sealed fields and readings, typed `MethodBody`, `Calls`, `LowersTo`, `UsesPrimitive`, semantic laws and compiler refusal. Bind source symbols to these **only with an explicit witness** such as implementation registration or generated reference, not lexical similarity.
3. **Execution-evidence graph** (proposal): prepared program instance, experiment, receipt, outcome, observed cost and hypothesis. Observation is not proof. A validated equivalence must cite a pinned independent oracle and the exact source/body/version.

Physical rows may be shared; identities, relation kinds, source-of-truth owners and validity periods may not be conflated. Keep `Tier::{Declared,Derived,Compiled,Verified,Hypothesis}` distinctions. Edge provenance: `{subject,rel,object,source_ref,source_sha,generation,provenance,tier,valid_at}`.

**Semantic hygiene:** `IAM::active` and `CRM::active` remain distinct even if label/ordinal/lane kind matches. Different population, NULL, bag/set, time window, authority or reading laws require typed refusals.

## 2. Cold incremental graph construction; immutable snapshots

**Inputs:** committed repository index, current working-tree overlay, prepared method table, compiler decisions, tests, benchmark receipts and explicitly marked documents/PR assertions. Treat documents and LLM proposals as *claims*, never compiled facts.

**Invalidation:** SHA of git tree + uncommitted file content + cargo features/target + macro expansion/codegen context; method `BodyVersion`, `SchemaSeal`, `ReadingId`, `FullClassId`, source generation, CPU tier, independent test signature. Stale edges must be omitted or returned explicitly as stale. Changed files reindex only impacted records, but any ambiguous dynamic callsite widens the uncertainty, not a fabricated dependency.

**Two-hop blocker:** #1509's single-program path cannot pass arbitrary mask-to-mask traverser populations between hops. First test the legal `EqU32Via` / factored relation carrier and G1's referenced-call edges; if neither suffices, return `UnsupportedMultiHop` with proof of where it fails. Do **not** implement a standalone BFS, traversal interpreter or new query engine to make demo screenshots work. The existing one-hop relationship may still be presented as a small useful slice.

**No hot-path symbol lookup:** indexing/context preparation is cold or incremental. Runtime method calls keep the already bound Quack/R2IL/mask-risc execution path.

## 3. Local read-only context-provider contract (MCP/CLI adapter is optional)

Three **candidate** tool operations, not claims of installed integrations:

```text
context(SymbolRef, Intent, Budget, Revision) -> ContextSlice | TypedRefusal
impact(SymbolRef, ProposedMethod, Budget, Revision) -> ImpactSlice | TypedRefusal
propose(SymbolRef, Objective, Budget, Revision) -> ProposedMethod[] | TypedRefusal
```

- `SymbolRef` is a canonical location/ID plus optional exact source location, not an unqualified string.
- `Intent`: explain, implement, test, optimize, review, navigate. No arbitrary tool writes.
- `Budget`: max nodes/edges/hops, returned bytes/tokens (estimated), wall time, privacy/redaction policy; show `truncated` and a continuation handle.
- `ContextSlice`: exact anchor IDs, source links and line ranges, evidence tiers, graph revision, discovered dependencies, relevant laws, tests/refusals, uncertainties, `not_checked`, and explicit omissions.
- `ImpactSlice`: affected functions/methods/tests/build targets by typed reachability, plus separate potential or unresolved impacts. Missing an edge is NOT a negative proof.
- `ProposedMethod` (canonical; wraps main's `awareness::Candidate`): deterministic graph-pattern recommendation, obligations, confidence provenance and a suggested *read-only* verification query. No execute/edit capability.

Example payload, strictly illustrative:

```text
context(GroupSum, intent=modify, revision=HEAD+worktree, budget=4096B)
  known: Quack lowering; R2IL parity path; population/mask laws
  tests: method parity; independent fold oracle
  measured: only version- and CPU-matched receipts
  unknown: unindexed macros / unresolved dynamic callsite
  next: inspect source span / run named tests
```

Prefer a tiny typed JSON/text envelope rendered for Claude rather than broad Markdown dumps. No repeated full-file prose. Stable ordered answers and provenance permit local diffs between sessions. The provider must not claim to be VS Code IntelliSense or a Claude Code plugin until one is actually wired.

## 4. Cheap deterministic proposer before LLM reasoning

Rules over *validated* edges can recommend:
- reuse `is_billable` rather than duplicate predicate bodies;
- run tests covering the changed primitive/reading/population;
- flag a body that references a class-incompatible field or cycle;
- detect a shared materialization + fold pattern and propose H1;
- notify that a `FirstOfRun`/retraction obligation is missing without pretending independence;
- mark source+body+seal drift and demand re-preparation.

Proposals must be `Fact | Inference | Hypothesis` tagged, with reasons and independent falsifiers. Query answers cannot silently authorize OGAR `ActionInvocation` or provider effects.

**LLM boundary:** Claude receives only a relevant slice, then uses its own judgment/editing tools; the graph's data is untrusted context, never behavioral instructions. Source comments, Markdown, PR bodies and model-generated proposals can include prompt injections. Filter tool output to typed data and citation pointers.

## 5. Observable execution and ONA-inspired feedback

**Pinned upstream reference:** ONA `opennars/OpenNARS-for-Applications` at `dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a`; paper by Hammer & Lofthouse (2020). ONA's event/temporal implication/anticipation/operation/revision loop is a reference, **not** an imported runtime. Wang, Li & Hammer (2018) is broader theory, not proof every mental operation runs in ONA.

Minimal experimental entities:
- `Observation`: execution of `PreparedProgram` against `PopulationSnapshot` under `ReadingId`, a concrete outcome/refusal, and source-backed provenance.
- `ExecutionReceipt`: receipt ID, experiment/run ID, compiled program and body digests, `FullClassId`, method key, schema/reading/source version, CPU feature/tier and dataset/window; objective timing, allocations and materialization *when actually measured*, sampling/method and measurement error; oracle/test identity and status.
- `OptimizationHypothesis`: precondition predicate, permitted candidate, predicted effect **registered before result**, predicted metric/range, observation deadline/window, evidence ancestry, applicable method versions, falsifier and later observed outcome.
- `Revision`: positive/negative/unknown observation with no double counting from repeats or shared evidence base. Negative evidence only after a real opportunity to observe a promised consequence.

Do not equate `Truth(f,c)` with a calibrated probability/confidence interval; do not deduce causal improvement from temporal association. A benchmark is not an evidence-independence certificate. Keep immutable ISA bytes, ClassView readings and policy distinct.

**Receipt gap:** #1495 showed NAL recipe 10/MCP cannot use prior execution history because dispatch discards `run_recipes` output. Use that research when testing a receipt feed; avoid duplicating its existing falsifier.

**Bounded instrumentation:** emit outside hot SIMD loops through an optional bounded buffer, flush out of band, carry an origin flag / recursion guard, suppress sampling of the monitor's own context reads by default. Collect dropped count and quantify overhead. Do not call graph query/write recursively from `mask-risc` inner loops.

## 6. Experimental program: separate three claims

### E1. Context replacement vs repository searches
Hold the same issue/tasks, repository SHA, worktree state, model, instructions and token budget. Randomize order or use repeated counterbalanced tasks. Compare (A) `CLAUDE.md`+grep/read, (B) deterministic symbol/LSP index, (C) SAG/Quack context slices, (D) hybrid graph+source snippets.

Measure prompt+tool-response input tokens, local index CPU/latency/storage, tool calls, end-to-end time, dependency recall/precision against an independent source-derived gold set, correctness of patch/test outcomes, citations, hallucinated edges, stale answers and human correction. Count graph/index build cost and first-run cold-start separately. `No LLM tokens for local queries` is not a claim that Claude receives free context.

**Falsifier:** graph arm saves tokens but omits a critical dynamic dependency or produces a wrong patch. It fails even if latency improves.

### E2. ONA-inspired hypothesis revision vs simple cost models
Use #1509 H1 `Filter -> materialized mask -> MaskedSumI32`. Measure current materializing program against a legal independently checked scalar reference. A selectable fused SIMD path is **not implemented** by #1509; only benchmark a new alternative if actually shipped and parity-verified. Register predicted materialization and speed effects before trials; distinguish the two.

Compare static rule, historical empirical cost model, and an ONA-inspired precondition→anticipation→feedback selector on matched and held-out workloads. Include n, selectivity, NULL/validity, CPU tier, warm/cold, interleaved reps and variance. A policy must beat its simple baseline on calibration/regret/quality at an acceptable overhead to justify new machinery.

**Falsifiers:** predicted gain fails on another selectivity/tier; replayed receipt inflates evidence; side effects differ; nullable rows change answers; cycle or stale body isn't refused.

### E3. Entropy/structural candidate ranking
Keep three signals separate: syntax-level MDL redundancy, epistemic uncertainty and measured physical cost. Use #1509's `J = L(P)+L(M|P)+lambda*C` strictly where costs are measured; leave `lambda*C` unscored where fusion not built. Test same-label cross-class negative control. Compare candidate ranking against a simple frequency/impact baseline.

## 7. Prioritized implementation waves and gates

**W0 (source audit and independent oracles):** pin exact SHA and confirm public API/Quack one-hop behavior. Compile a ledger `shipped | demonstrated | documented | proposed | blocked`. Verify current G1 state and source-index options.

**W1 (minimal read-only context):** project one real `MethodTable`/symbol mapping with verified source spans; implement only `context` for one hop, deterministic ordered output and stale-revision refusal. No new opcodes. Compare direct Quack answer and source-derived ground truth.

**W2 (impact and validated proposals):** add `impact`/ `propose` only for supported reachability; return typed insufficiency for unsupported multi-hop, unresolved Rust macro/dynamic calls and unknown law. Independently validate dependency/test recall.

**W3 (receipts):** demonstrate method receipt -> bounded event ingestion -> Quack query -> ONA-style registered prediction -> observed outcome -> revision, without recursion or cost laundering.

**W4 (agent-facing experiment):** optionally expose tools through a documented local adapter, verify actual Claude Code invocation and measure context tokens, correctness and cold indexing cost. No claim of connected Claude/VS Code without live invocation.

**W5 (learned proposer, only if measured):** test ONA-style selection against static/historical baselines. Do not optimize deployed code or rewrite source.

### Required non-vacuous guards
1. Cross-ClassId equal-name/ordinal refuses; 2. method cycle; 3. stale `SchemaSeal` or working-tree edit; 4. wrong population or bag/set semantics; 5. partial two-hop query reports insufficiency; 6. forged `Verified` edge from Markdown rejected; 7. no data outside explicit read scope; 8. prompt-injection text stored as inert graph value; 9. receipt replay has no new evidence; 10. expected-but-unobserved not marked failure prematurely; 11. clock/cycle instrumentation overhead reported; 12. selected optimizations preserve typed answer, NULL and authority; 13. context truncation is visible; 14. gold dependency missing produces a red test.

**Scope of this PR:** research plan/prompt/source excerpts only. No runtime capability, plugin, LSP bridge, benchmark, token savings or ONA parity is implemented or measured by these documents.

## 8. Decision to revisit

Proceed beyond W1 only if the graph returns a **smaller and at least equally correct**, source-backed context slice than the strongest conventional baseline, with bounded indexing/invalidations. Proceed beyond W3 only if the receipt+revision loop improves prediction or decision over a simple history-based selector. Otherwise retain the read-only graph context service and stop.

References: [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509), [#1504](https://github.com/AdaWorldAPI/lance-graph/pull/1504), [#1508](https://github.com/AdaWorldAPI/lance-graph/pull/1508), [#1495](https://github.com/AdaWorldAPI/lance-graph/pull/1495), [#1359](https://github.com/AdaWorldAPI/lance-graph/pull/1359), [#1311](https://github.com/AdaWorldAPI/lance-graph/pull/1311), and companion `.arxiv` notes.

## 9. Deferred contracts now expanded (research-only)

The first documentation pass intentionally left implementation detail unresolved. The explicit contracts and negative fixtures are now written out in [D-SAG-4 deferred contracts](software-awareness-agent-context-deferred-contracts-v1.md), including:

- source-symbol versus OGAR/MethodKey identity, version/worktree freshness and atomic graph generations;
- `resolve/context/impact/propose/expand/explain` proposals as bounded, read-only, typed calls;
- one-hop Quack mechanics and refusal on unsupported multi-hop;
- mandatory correctness facets, query budgeting, delta slices and explicit coverage omissions;
- deterministic local propositions with witnessed edges, prompt-injection treatment and authority separation;
- receipts, prediction registration, observer-recursion protection, CPU-tier validity and baseline ablations;
- independent retrieval, execution and learning economics and a stop/go ladder.

The [attention and active-experiment excerpts](../../.arxiv/2026-10_sag_agent_attention_active_selection_excerpts.md) add paper-grounded controls from Lost in the Middle, RepoHyper/RepoGraph, costly active learning and verification-first code uncertainty.

**Scope remains unchanged:** these additions are specifications and source reviews, *not* executable features or measured gains. Any follow-up CC run should start with W1's one-method one-hop proof, not implement every proposal simultaneously.
