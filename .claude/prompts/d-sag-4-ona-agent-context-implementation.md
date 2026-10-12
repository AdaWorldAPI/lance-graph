# Claude Code prompt — D-SAG-4: graph-backed agent context, ONA feedback, and falsifiers

> **Hand this to a fresh Claude Code session after #1509.** The associated plan and research excerpts are already part of the branch. This document asks for a small executable follow-up, not a claim that the functionality shipped in the documentation PR.

## Mission

Build from merged [lance-graph #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509). Explore whether its `SoftwareGraph` can give Claude Code a local, addressable, **source-grounded** context provider and whether versioned execution receipts can support ONA-inspired anticipation and revision.

Goal: substitute *deterministic local graph queries* for repeated prose scans where safe, and send Claude only bounded context slices when model reasoning is needed. Local operations themselves use no LLM tokens; response content supplied to Claude **does** consume tokens.

**Do not create a second graph query engine, semantic method registry, runtime interpreter or autonomous source editor.**

Read in this order:
1. `.claude/plans/software-awareness-graph-v1.md`, `.claude/plans/software-awareness-agent-context-ona-v1.md`;
2. `crates/lance-graph-glove-parity/src/{awareness,method,gremlin}.rs` and tests/examples;
3. #1504, #1508/G1, #1495 receipt gap, #1359 observe/revise/learn;
4. `.arxiv/2026-10_sag_ona_internal_experience_source_excerpts.md`;
5. `.arxiv/2026-10_sag_repo_graph_context_source_excerpts.md`;
6. pinned ONA `dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a`, especially `src/{NAR,Cycle,Memory,Decision,Inference}.c`.

### Explicit status corrections

- #1509 demonstrates Quack-backed one-hop introspection over **projected prepared methods**, not repository-wide source indexing, persistence or arbitrary k-hop reachability.
- The seven-language Bible graph is not a shipped corpus graph; #1509's source audit explains the English KJV, small lexeme fixture and plan-only seven-language claim.
- H1's scalar-vs-materializing measurements do not establish a fused SIMD backend or guaranteed speed gain.
- Method keys are currently caller-issued, not magically minted by OGAR.
- Do not pretend `CLAUDE.md` can be removed without comparing its task instructions and human-written guidance; keep safety/instructions outside the replaceable factual context.
- 'SoC' is separation of concerns in this repository, not a chip design.

## Deliver one minimal executable vertical slice first

Choose **one existing, real method** from #1509 (e.g. `billable_hours`).

Implement/verify, only if existing APIs allow it:

```text
real MethodTable + source provenance
             ↓
SoftwareGraph projection
             ↓
Quack Draft + mask-risc
             ↓
ContextSlice {anchor, related nodes, typed edges, source pointers,
              evidence tiers, revision, omissions, uncertainties}
             ↓
small CLI output that Claude Code could consume
```

The *proof* is independent: the returned definitions/dependencies agree with the actual prepared `Program` and a source-derived test oracle.

If one-hop works but multi-hop doesn't, **stop at one hop** and emit `UnsupportedMultiHop`; the user should see a meaningful limitation, not a fake chain. Test a legally expressed `EqU32Via` or existing relation carrier only as a separately gated follow-up.

### Proposed API (refine against code, do not proclaim it shipped)

```rust
context(symbol_ref, intent, budget, revision)
impact(symbol_ref, change_spec, budget, revision)
propose(symbol_ref, objective, budget, revision)
```

Symbol identity must be revision-qualified, including `HEAD` and dirty worktree. Return typed `NotIndexed`, `AmbiguousSymbol`, `StaleRevision`, `InsufficientEvidence`, `UnsupportedMultiHop` or similar exact current error variants. Do not silently fall back to matching a label.

The first prototype can be an example binary. A read-only local MCP/Claude Code tool wrapper is **optional** and must only be documented as connected after it is demonstrably callable from Claude Code. No VS Code extension required.

### Source-symbol layer

Investigate rust-analyzer/LSP or other compiler-backed definition/reference exports without introducing a second reasoning engine. Do not assume AST scraping gives full macro/dynamic-call coverage. Bind source symbols to ClassId/MethodKey only via demonstrated registration or code-based witness. Treat imports, callsites, trait implementations, tests and external source pointers as versioned graph facts with source-line provenance and a completeness label.

Generate a `context` result with explicit `checked`, `not_checked`, `truncated`, and `uncertain` categories. Missing edge != proven absence.

### Cheap local proposer

Start with at most three deterministic patterns:
1. After a method references a primitive, propose its compiled-dependency and oracle tests.
2. When a repeated condition is already a named method, propose typed method reuse, preserving semantic scope.
3. For materialized-filter→fold, flag H1 as a **candidate** and list legality/performance obligations, not promised acceleration.

Do not generate/edit Rust based on these heuristics without separate agent review.

### One receipt and anticipation falsifier

Use existing benchmark and #1495 execution-receipt research. Produce a candidate receipt linked to the program and source revision. Include only measurements you can **actually collect**. Keep wall time, cycles, allocations, materialized bytes, source snapshot, CPU tier and oracle verdict distinct.

Register a performance prediction **before** collecting the result. Compare materialized-vs-scalar references under pinned data and typed oracle. Demonstrate a positive, negative or inconclusive observation and an explicit revised hypothesis record. If the output cannot be consumed by existing NAL machinery, record that gap rather than adding an ONA executor.

Disable instrumentation for the graph's own context queries or provide bounded origin-gated tracing. A graph read must not recursively trigger unbounded graph event creation.

### Baselines required for agent-context claims

Compare:
- A: current `CLAUDE.md` + grep/read workflow;
- B: source symbol index / deterministic IDE-like lookup;
- C: Quack/SAG bounded context slices;
- D: optional graph + only essential source snippets.

For fixed tasks/SHAs and comparable model budgets measure:
- local indexing/setup CPU, elapsed time and storage;
- model-visible input tokens, tool roundtrips and total task latency;
- independent-gold dependency precision and recall;
- correct change/test outcomes and missed invariants;
- freshness with dirty worktree and altered `SchemaSeal`;
- effects of truncated graph slices.

No 'token-free Claude' claim. If correctness drops, token savings alone cannot pass.

### ONA reference checks

Trace and distinguish in pinned ONA source:
- `NAR_AddInput` / `NAR_Cycles`;
- `Cycle_Perform`, `Cycle_ProcessBeliefEvents`, `Cycle_ReinforceLink`;
- `Memory_Conceptualize`, `Memory_ProcessNewBeliefEvent`;
- `Decision_BestCandidate`, `Decision_Anticipate`, `Decision_Execute`, `Decision_AddNegativeConfirmation`;
- `Inference_RevisionAndChoice`, `Stamp_checkOverlap`.

Report whether ONA's positive/negative temporal implication evidence is appropriate for this cost-prediction domain. Compare it with a static heuristic and empirical cost history. Preserve ONA's temporal/procedural semantics; do not brand a generic log or a simple average 'ONA-compatible'.

The Wang/Li/Hammer 2018 mental-operation/self paper has a different theoretical scope from shipped ONA. Keep that separation explicit.

### Validation/negative controls (must turn red)

1. Same field name and ordinal under two ClassIds; 2. method dependency cycle; 3. body or seal mutation invalidates context/receipt; 4. source ref resolves to wrong commit/line; 5. macro/trait/dynamic call not indexed and marked uncertain; 6. multi-hop request refused if unsupported; 7. unordered/limited slice explicitly marked incomplete; 8. forged `Verified` edge from prose rejected; 9. source comment containing 'ignore instructions' treated as inert data; 10. local read-only context cannot call an external mutation; 11. receipt replay preserves evidential weight; 12. missing anticipated observation not counted as a failure before its observation window closes; 13. CPU-tier/dataset mismatch refuses benchmarking conclusion; 14. no timestamp/cycle claim without actual measurement; 15. alternative has identical typed answers over null/unknown and bag/set fixtures; 16. observer does not observe itself indefinitely; 17. token savings cannot conceal missed dependencies.

A purported guard that remains green when intentionally disabled needs a stronger fixture.

## Evidence ledger and PR discipline

Every claim carries `SHIPPED / EXECUTED / MEASURED / PARTIAL / BLOCKED / HYPOTHESIS`, source SHA and concrete evidence. Distinguish cold construction, lookup latency, warm method execution and model context cost. Keep exact source pointers, no invented test results.

Order the work:
- W0 source audit + method oracle;
- W1 one-hop `context`, stale-ref and uncertainty guards;
- W2 `impact` and `propose` only if existing algebra is sufficient;
- W3 receipts and bounded ingestion;
- W4 deliberate prediction, controlled cost comparison and revision;
- W5 token/accuracy benchmark against simple baselines;
- W6 optional local agent adapter **only after** the prior gates.

Do not broaden into general NAL-9, new ISA bytes, source rewriting or a UI until experiments justify it.

**Output:** (a) design summary, (b) small executable tests with disable-red reports, (c) code pointers and measurements, (d) gap ledger with no false positives, (e) a draft PR referencing #1509. If only a conceptual integration is possible, stop after tests documenting the specific blocker.

### Success condition

One method, one revision-scoped context query and one independent oracle; one explicit pre-registered performance prediction and measured revision; and a quantified comparison showing whether this approach beats simpler context retrieval without losing correctness.

The ambition is *local architectural memory with addressable evidence*, not an invisible second agent.

## Deferred research addendum: read before broadening the implementation

This initial prompt was intentionally a minimal falsifier. For the deeper alternatives and rigorous edge cases, read:

- [Addressable context, identity/freshness, foveated selection, receipts, experiment economics and 15 negative fixtures](../plans/software-awareness-agent-context-deferred-contracts-v1.md).
- [High-signal arXiv excerpts: context ordering, graph retrieval, costly experiment selection, code uncertainty and verification](../../.arxiv/2026-10_sag_agent_attention_active_selection_excerpts.md).

**Do not silently ship those deferred proposals in W1.** Select the smallest experiment that can break the hypothesis. Implement `context` for one real #1509 `MethodTable` method and one Quack hop; add a source provenance witness; test negative controls. Escalate to incremental symbol indexing, ONA receipt selection, extra graph hops or Claude adapter only after independent correctness and efficiency gates pass.

Further research probes to retain in the ledger:
- intent-specific **evidence coverage**, token ordering and a mandatory-law budget overflow refusal;
- `HEAD` vs dirty-worktree cache invalidation and macro/trait-call incompleteness;
- a bounded `DeltaSlice` protocol which cannot invent context absent from the model's prior accessible history;
- experiment cost versus expected information gain, separate from materialization/cycle benefit;
- source-verified low-level dependency facts versus source prose/LLM claims;
- ONA-style prediction ablation, receipt replay on another layer, CPU-tier and selectivity shift, and static/history-model controls.

When reporting results, distinguish `TokenEstimate` from tokenizer-counted `ModelInputTokens`; `LogicalMaskOps` from `MaterializedBytes`; `CpuTier` from observed code generation; and `ObservedOutcome` from `VerifiedCausalEffect`.
