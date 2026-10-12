# D-SAG-4 appendix: deferred concrete contracts for an agent-facing software graph

**Status:** PROPOSAL / NOT SHIPPED. **Added after #1513's first documentation pass.** Built on merged [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509), not an alternative to it. Parent: [D-SAG-4 plan](software-awareness-agent-context-ona-v1.md). Research: [source excerpts](../../.arxiv/2026-10_sag_agent_attention_active_selection_excerpts.md). This appendix unpacks design and falsification details deliberately left small in the first draft. It does **not** certify measurements, APIs, an IDE plug-in or runtime integration.

## A. What the existing executor can actually answer

The current `lance-graph-glove-parity/src/awareness.rs` provides `SoftwareGraph { nodes,edges,provs,digests,measurements }`. A `Node` has `Kind`, identity and optional label. An `Edge` has `{src,rel,dst,tier,ctx,prov}`. `Provenance` has `{source,rev,seal,reading}`. The known `Tier` values are `Declared,Derived,Compiled,Verified,Hypothesis`. `SoftwareGraph::hop_to_ctx` binds one `Draft`, lowers a `ScatterOrU32` over the edge table and maps selected rows to their method/context node. `methods_using` and `refused_for` use that exact path.

**Careful interpretation:** this is NOT a generic `GraphTraversal`, not a source-level call graph, not arbitrary reachability and not a durable evidence database. `SoftwareGraph::out` is a convenience in-memory method; do not accidentally use it as a hidden traversal engine while claiming Quack executed a multi-hop query. The existing `Measurement` stores program, rows, tier, median ns, not a full event/receipt with calibrated cycles/allocations. `CpuTier` is target and `avx512f`, not evidence of the actual selected instruction kernel.

The next step is to **extend a typed projection**, not to substitute an ordinary graph library for Quack. Label any eventual `SourceSymbol`, `TraceReceipt` or `AttentionChoice` kinds as additions until code proves them.

## B. Identity, ownership and freshness protocol

Three non-interchangeable identities:

| Entity | Proposed canonical key | Owner / witness | Invalidation |
|---|---|---|---|
| SourceSymbol | `repo_id / commit_or_worktree_generation / crate / cfg / DefPathHash or verified source def / expansion context` | compiler/LSP source provider | source tree, features, compiler/toolchain, macros |
| SemanticMethod | `FullClassId / MethodKey / SchemaSeal / ReadingId / BodyVersion / SourceGeneration` | OGAR + existing compiler binding | class/schema/body/read/generation change |
| ExecutionRecord | `ExperimentId / ReceiptId / PreparedProgramDigest / WorkloadSnapshot / CpuTier` | measurement producer and independent oracle | data/window/tier/instrumentation/config changes |

Identity rows are immutable within a generation; cross-generation equivalence is a **verified relation**, not a raw ID alias. `MethodKey` currently caller-minted; do not claim OGAR owns its minting. Rust source symbols and semantic method keys are not the same ontology.

Needed extra edge metadata (proposed, not in #1509 `Edge`): source span, source digest, compiler or source-index version, relation witness, confidence/coverage grade, observation validity interval, snapshot/worktree revision. **Do not** make identity or edge validity depend on human-readable labels.

Proposed cold incremental update:
1. Hash repository commit plus dirty worktree blobs and `Cargo.lock`, selected cfg/features/toolchain.
2. Index changed source definitions/references; explicitly record unexpanded/unresolved macros, trait calls, generated code, FFI and dynamically selected methods as `Unresolved`.
3. Project or re-prepare only affected semantic methods with existing seals and readings.
4. Keep old immutable graph generation visible to pinned sessions, publish a new coherent generation atomically.
5. Invalidate affected graph certificates, cached context slices and receipt applicability. Never reuse older source spans across changed revisions without a verified relocation.

**Causal ordering:** source index revision, prepared-method generation and benchmark run must not silently straddle incompatible snapshots. A caller asking for `HEAD+worktree` gets that generation or an explicit refusal; no pretending `HEAD` includes uncommitted edits.

## C. Addressable context is a protocol, not a giant memory blob

First useful interface is a *local* read-only CLI or protocol adapter:

```text
resolve(symbol_query, revision)       -> SymbolRef | Ambiguous | NotIndexed
context(SymbolRef, intent, budget)    -> Slice | Stale | Unsupported
impact(SymbolRef, change_kind, budget)-> Slice | Uncertain | Unsupported
propose(SymbolRef, objective, budget) -> Candidate[] | InsufficientEvidence
expand(SliceHandle, relation, budget) -> SliceDelta | UnsupportedMultiHop
explain(EdgeRef)                      -> SourceSpan + Witness + Tier + Revision
```

`expand` **does not authorize** a new traversal opcode: it is a user-facing request for a separately certified Quack query, or a typed refusal if the existing one-hop carrier cannot express it. A continuation handle only references a frozen indexed snapshot; it is not a secret store that Claude can remember without requesting a tool again.

Minimal response envelope (illustrative; no wire format shipped):

```json
{
  "snapshot": {"commit": "sha", "worktree": "digest", "cfg": ["features..."]},
  "query": {"anchor": "method:<qualified-key>", "intent": "modify"},
  "limits": {"max_bytes": 4096, "max_hops": 1, "timeout_ms": 250},
  "result": {
    "anchors": ["method:<qualified-key>"],
    "edges": [{"ref": "edge-id", "rel": "lowers_to", "tier": "compiled",
               "source": {"path": "file.rs", "start": 123, "end": 132, "sha": "blob"}}],
    "obligations": ["preserve-population-law", "run-independent-oracle"],
    "unresolved": ["trait-dispatch not indexed"],
    "truncated": true,
    "continuation": "opaque-snapshot-bound-handle"
  }
}
```

Source span and `Verified` grades must come from producers that earned them, never be copied from a README statement or free-form user text. Strings from source/comments/Markdown/PRs are **data**, not tool instructions. Give Claude short cited evidence, not executable control text. Never allow reading outside granted repository/workspace scope.

### What should stay in CLAUDE.md

Human/team instructions, coding policy, architectural intent, on-call/ownership conventions and safety rules are still valuable. Only replace reconstructable *factual repository context*. A compiler graph cannot establish intent, human authority or policy from a file name. Do not market removal of Markdown as a feature.

## D. A foveated context algorithm without pretending model inference is free

Proposed pipeline:

```text
request → exact symbol anchor resolution → intent-specific allowed relation set
        → one-hop Quack retrieval → local ranking/coverage packing
        → typed bounded context slice → optional expand
        → Claude context (actual tokens are now spent)
```

Examples of allowed relation priorities:
- `modify`: caller/callee *if witnessed*, law, schema, compiler refusal, changed-file impact, tests.
- `debug`: current failure/refusal, producer source span, consumers, oracle mismatch, negative control.
- `optimize`: compiled physical primitives, measurements on **matching** CPU/data versions, candidates and unfulfilled law obligations.
- `research`: hypothesis/source distinction, provenance, falsifiers, unmeasured gaps.

Use an explicit **hard filter** before scoring: scope, source version, provenance validity, allowed edges, privacy, requested intent and semantic law. Discard stale/forged items; never 'score' an illegal fact back into relevance.

Within valid candidates, experiment with a greedy budgeted coverage score, e.g.:

```text
F(S) = sum over required facets f of weight(f) * covered(f,S)
       + novelty_bonus(S) - repetition_penalty(S)
       - token_cost_penalty(S) - unresolved-risk-penalty(S)
subject to: bytes(S) <= B; all mandatory correctness obligations included
```

This is **not** Shannon entropy and is not claimed submodular until diminishing returns are established for the chosen coverage function. Simple sorted relevance, degree or chronological slicing remain controls. If any mandatory obligation exceeds budget, return `InsufficientBudget` with a short pointer to expansion; never silently drop the guard to fit.

**Visible omissions matter:** returned `checked`, `unknown`, `unsupported_multi_hop`, `truncated`, `stale`, `source_only` and `graph_derived` signals should be separate. A tiny slice that falsely claims completeness is worse than no graph.

### Session context deltas

Two consecutive requests against the **same snapshot and exact slice handle** may return a deterministic `DeltaSlice` (added/removed edge IDs + source digests). This can reduce repetition in tool output. A language model without accessible prior slice text cannot reconstruct it from an opaque handle; support explicit full-snapshot fallback. A `DeltaSlice` is not durable memory and never crosses incompatible source generations without invalidation.

## E. IntelliSense-style proposals are not autonomous code generation

Local proposer owns only safe patterns with exact witnesses:

| Pattern | Required witness | Suggested action | Negative control |
|---|---|---|---|
| impacted primitive | compiled `Program -> Primitive` | list named oracle/tests | documentation-only `USES` assertion |
| repeated predicate | G1 typed `CALLS` and same sealed reading | suggest method reuse | same spelling from another ClassId |
| repeated selection | two compiled folds consuming same typed population | propose reuse candidate | folds have different NULL/authorization law |
| potential fusion | measured materializing plan + independent oracle | propose *benchmark* of fused candidate | no fused executable backend exists |
| uncertain Rust callgraph | trait/dynamic call has no proof | request LSP/source inspection | falsely omit a dependent function |
| stale evidence | body digest or source generation changed | demand revalidation | certificate from old body reported valid |

The proposer does not supply a Rust implementation, alter `MethodTable`, mutate an OGAR action, or write to AD/Entra/SAP. It returns proposed next steps and evidence. Claude Code uses its separate authorized editor/test tools for changes. A stable edge is not an action capability.

## F. Receipt, anticipation and counterfactual boundary

Define `ExecutionReceipt` as a measurement record, not a NARS truth value. Key at least by `{MethodIdentity, PreparedProgramDigest, SnapshotId, ExperimentId, TrialId, CPU-tier, InstrumentationVersion}`. Include wall-clock time and, if actually measured, reliable cycles, allocations, scratch peak, temporary bytes, program op classes and output oracle verdict. If unknown, encode `NotMeasured` instead of numerical zero.

Separate:
- **Structural facts**: Quack program contains a mask/fold; sourced from prepared program.
- **Predictions**: materialization and runtime hypotheses registered before a trial.
- **Observations**: independently measured outcomes with their measurement method.
- **Causal claims**: permitted only by controlled intervention/ablation over fixed input and source laws.

An ONA-style `Precondition + Operation -> Outcome` is a *candidate* reading over versioned experiment events. ONA's temporal implication frequency/confidence is not a calibrated physical cost distribution. Do not feed benchmark timings into truth bytes without an explicit mathematical law.

### Cost experiment E2b: anti-self-deception arms

At minimum run:
A. No history: static deterministic choice.
B. Historical median + robust confidence/uncertainty estimates: simple empirical chooser.
C. ONA-inspired evidence-based anticipated consequences.
D. C but shuffle/remove prior receipts (ablation).
E. C but replay the same receipt under a new 'layer' identifier (must not gain confidence).
F. C under domain shift: different selectivity/CPU tier or population size.

Score prediction calibration/error, regret against hindsight oracle, total wall time including exploration/index/instrumentation, correctness and coverage of refusal. Hypothesis-support changes must point to a fresh observed receipt. Separate *counterfactual unseen performance* from real execution, and never call a missing trial negative evidence.

Define observation deadline explicitly; premature failure due to silence is illegal. If the simple historical baseline wins, do **not** adopt ONA machinery just because a concept graph can express it.

### Loop containment

Sampling events from Quack/mask-risc may be emitted *outside* the measured kernel boundary into a bounded outbox. A `Source::AgentContextGraph` origin bit or scoped suppression prevents recursive self-observation; still record explicit drop counts and instrumentation overhead. An asynchronous consumer stores into a **new** graph generation rather than changing the pinned snapshot of the running query.

## G. Three independent economics, one decision

Track separately:
1. **Retrieval economics:** local CPU/IO+index build vs model input tokens, retrieval correctness and total task time.
2. **Compiler/runtime economics:** warm execution cycles, memory traffic, allocations; semantic oracle parity.
3. **Learning economics:** cost of exploration/receipts vs future improvement over static and cost-history baselines.

Proposed net benefit per repeated task, for bookkeeping only:

```text
Net = value(correct_tasks_graph) - value(correct_tasks_baseline)
      - index_cost - graph_query_cost - extra_model_tokens
      - instrumentation_cost - experiment_cost - added_maintenance_cost
```

This is *not* a measured equation or monetary conversion. Report raw objective metrics and amortization for 1, 10, 100 repeated tasks. Test cold-start, warm cache, no-cache, and source mutation. Keep safety and semantic correctness hard gates before utility scoring.

## H. Hard negative fixtures to implement, not paper over

1. **Hidden cross-class alias:** equal `active` field names under two sealed views.
2. **Poisoned source documentation:** README asserts 'verified' but compiler never prepared method.
3. **Source drift:** changing macro/cfg or editing worktree changes actual definition without updating source index.
4. **Second hop demand:** exact one-hop correct, requested two-hop must refuse or use verified legal relation carrier.
5. **Graph-cycle acceptance:** computed-field cycle crossing >64 ordinal is not silently passed by an incomplete validator.
6. **Hot-path logging:** tracing enabled vs disabled is isolated; observation cannot improve benchmark by omitting its overhead.
7. **Materialization confusion:** a logical mask operation that remains resident is not reported as temporary byte allocation.
8. **No free context:** model sees bounded returned text, count its input tokens.
9. **Wrong program tier:** replayed AVX2 data as AVX-512 performance must refuse.
10. **Novel but invalid candidate:** H1 reduces memory in one dataset but changes NULL semantics in another; must not select.
11. **Attention trap:** high-degree symbol buries the only required oracle; budgeted selection must surface guard or visibly abstain.
12. **Receipt overlap:** same run repackaged as two independent observations with different layer IDs.
13. **Recursive observer:** context tool calls must not generate infinite query/receipt cascades.
14. **Gold-dependency miss:** LSP independently resolves dynamic/trait target absent from graph; return incomplete, not safe-to-edit.
15. **Forged source instruction:** malicious comment attempts to tell Claude to ignore tests; parser preserves it as inert text and never elevates its authority.

Every guard needs a non-vacuous red disable or an independently observed failure; a green-only test does not prove its relevance.

## I. Incremental delivery order and explicit exclusions

1. **D-SAG-4A: read-only one-hop context slice** using existing Quack `Draft`, with source refs and stale checks; compare independent source lookup.
2. **D-SAG-4B: source-symbol adapter** only if it improves task correctness against LSP baseline. No broad tree-sitter rewrite before benchmarking.
3. **D-SAG-4C: bounded deterministic proposer**, three witnessed rules; supported one-hop only.
4. **D-SAG-4D: receipt schema + nonrecursive offline graph projection**, with oracle and version guarantees.
5. **D-SAG-4E: registered anticipation/feedback and baseline comparison**, initially benchmark-only.
6. **D-SAG-4F: optional Claude Code adapter and token study**, only with a live measured call.
7. **D-SAG-4G: multi-hop projection**, only after current Quack carrier laws/semantics permit it.

Explicitly deferred: full Rust call-graph completeness; all-repository persistent index; causal proof from mere temporal data; unbounded graph walks; automatic code edits; automatic git merges; NAL-9 parity; production optimizer switching; personalized tool policy not backed by receipts; speculative ISA/CE64 reinterpretation; any claim about deployed self-modification.

## J. References

- Local code: `crates/lance-graph-glove-parity/src/awareness.rs` `SoftwareGraph`, `Rel`, `Tier`, `hop_to_ctx`, `methods_using`, `fusion_candidates`, `Measurement`; [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509).
- [RepoGraph, arXiv:2410.14684](https://arxiv.org/abs/2410.14684): repository-wide navigation; not a demonstrated D-SAG result.
- [RepoHyper, arXiv:2403.06095](https://arxiv.org/abs/2403.06095): search-expand-refine and retrieval experiments; does not lift current one-hop restrictions.
- [Lost in the Middle, arXiv:2307.03172](https://arxiv.org/abs/2307.03172): context ordering/length matters, not just token count.
- [Song & Xu, arXiv:1701.06725](https://arxiv.org/abs/1701.06725): costly information acquisition in active learning, not an ONA implementation.
- [Shi et al., arXiv:2606.09577](https://arxiv.org/abs/2606.09577): code uncertainty has correctness dimensions beyond verbal confidence.
- [Rakasi et al., arXiv:2608.14659](https://arxiv.org/abs/2608.14659): comparison of uncertainty-based self-correction with verification-based approaches; see source-specific limits in companion research note.
- [ONA pinned source](https://github.com/opennars/OpenNARS-for-Applications/tree/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a), [2018 self-model paper](https://doi.org/10.3389/frobt.2018.00020).

**Stop/go:** if graph retrieval cannot beat a compiler-backed symbol index on accurate critical dependency retrieval at tolerable costs, restrict graph to uniquely semantic method/law/physical-program questions. If ONA prediction doesn't beat historical cost-based ranking, keep receipts and drop the learned selector. A reliable partial system is better than an impressive unsound one.
