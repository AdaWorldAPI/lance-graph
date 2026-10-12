# D-SAG-2 source excerpts: repository code graphs as low-token agent context

**Date:** 2026-10-12. **Kind:** RESEARCH / SHORT ATTRIBUTED PARAPHRASES. **Status:** no Claude Code connector, token measurement, source-code index or graph-to-editor implementation demonstrated by this note. **Base:** [merged #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509). **Companion:** [D-SAG-2 plan](../.claude/plans/software-awareness-agent-context-ona-v1.md).

Aim: compare the idea of *addressable, typed, locally queried repository context* against existing research rather than treating it as original or already measured. These are paraphrased excerpts keyed to the papers' abstracts and conceptual methods. Source-provided paper benchmark conclusions **do not transfer automatically** to lance-graph/Claude Code.

## R1. RepoGraph: repository-wide structural navigation

**Reference:** Siru Ouyang et al., *RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph*, ICLR 2025, [arXiv:2410.14684v2](https://arxiv.org/abs/2410.14684), [PDF](https://arxiv.org/pdf/2410.14684), [code](https://github.com/ozyyshr/RepoGraph).

**Excerpt (abstract, paraphrase).** File-local language-model context omits repository-scale structure needed for software engineering. The paper contributes a plug-in repository graph used for navigation and presents improved benchmark outcomes when added to several SWE-bench approaches, plus CrossCodeEval evaluation.

**Transfer:** establish a *source symbol* layer linked to D-SAG-1's prepared-method graph via witnessed relations. Queries should return graph neighbors, calls, tests and exact source pointers, not only text similarity. Do not assume D-SAG-1 already crawls a source repository.

**Limit and falsifier:** paper evaluations are on their own framework and tasks, not our C/Rust Quack/mask-risc substrate. Compare against a strong deterministic source index, not only no retrieval; demand dependency recall and working patch correctness. Do not copy benchmark numbers as predicted gains.

## R2. RepoHyper: search, expand, refine

**Reference:** Huy N. Phan et al., *RepoHyper: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion*, [arXiv:2403.06095v4](https://arxiv.org/abs/2403.06095), [PDF](https://arxiv.org/pdf/2403.06095), [code](https://github.com/FSoft-AI4Code/RepoHyper).

**Excerpt (abstract, paraphrase).** A repository semantic graph helps retrieval beyond a single file. The method first locates likely context, expands over semantic relations and refines/prioritizes retrieved candidates, with code-completion evaluation.

**Transfer:** `context(symbol,intent,budget)` could first resolve a **source-grounded anchor**, then expand one supported Quack hop, score relevant laws/tests/dependencies by intent, and return a deterministic bounded context slice. If a second hop is unavailable in #1509, report its insufficiency explicitly. Do not implement a separate traversal engine just to claim parity with RepoHyper.

**Falsifier:** long graph expansions return irrelevant high-degree dependencies and overwhelm the context budget; compare fixed one-hop, selected-hop and grep-based retrieval under identical tasks.

## R3. RANGER: symbol queries differ from natural-language queries

**Reference:** Pratik Shah et al., *RANGER: Repository-Level Agent for Graph-Enhanced Retrieval*, [arXiv:2509.25257v1](https://arxiv.org/abs/2509.25257), [PDF](https://arxiv.org/pdf/2509.25257).

**Excerpt (abstract, paraphrase).** Repository retrieval must handle two distinct needs: identifying concrete code entities and answering broad natural-language questions. The proposed approach combines cross-file code relationships, descriptions and embeddings; graph lookup addresses entity queries and search supports natural-language exploration. It reports multiple benchmark comparisons.

**Transfer:** first implement precise `SymbolRef` queries and typed failures for ambiguity. Natural-language retrieval should be an optional agent layer that proposes candidate anchors, never a new authority to mint `MethodKey`, `ClassId` or field equivalences. A local graph is not magically a natural-language interpreter.

**Falsifier:** an ambiguous label maps to a plausible but wrong ClassId; require a typed ambiguity report and explicit disambiguating source witness.

## R4. Knowledge-graph repository-level generation

**Reference:** Mihir Athale and Vishal Vaddina, *Knowledge Graph Based Repository-Level Code Generation*, [arXiv:2505.14394](https://arxiv.org/abs/2505.14394), [PDF](https://arxiv.org/pdf/2505.14394).

**Excerpt (abstract, paraphrase).** A graph with structural and relational repository information can improve retrieval context for repository-level code generation. The paper reports its own evaluation on EvoCodeBench.

**Transfer:** maintain separate `SourceDerived`, `Compiled`, `Verified` and `Hypothesis` edges, because repository relation retrieval by itself does not prove semantic correctness. Add source and method-version stamps to guard against changing codebases.

**Falsifier:** old or inaccurate graph connections make generation worse than baseline; index freshness and uncertainty are part of quality, not optional telemetry.

## R5. Established tooling to compare against

Non-arXiv technical reference: [Language Server Protocol specification](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/) for definitions, references and semantic navigation. An existing compiler-backed source index may answer some questions better and more cheaply than building a new graph layer.

**Architecture transfer:** an LSP/rust-analyzer index is a source adapter, NOT a replacement of Quack's executor and NOT OGAR semantic authority. Edges representing compiler-backed definitions should carry tool version, source revision, target features and macro-expansion limitations.

**Falsifier:** a trivial symbol rename causes stale or ambiguous graph results while an IDE index gets it right. Reject false certainty; reindex or return `StaleRevision`/`NotIndexed`.

## R6. What an addressable graph changes relative to Markdown

`CLAUDE.md` is useful for policy, build commands, preferred workflows, design intent and developer-authored conventions. It should **not** be removed solely because a graph can project facts.

Differences to validate:

| Concern | Markdown / grep | Source symbol index | D-SAG graph (proposal) |
|---|---|---|---|
| Build/repo instructions | Strong, human-maintained | Not primary purpose | Links to authority; never replace policy |
| Definitions/references | Re-read/search text | Compiler-backed, strong | Import qualified, versioned source edges |
| Method→physical program | Prose can drift | Usually not represented | #1509 prepared-method projection |
| Semantic reading/population law | Hard to keep addressable | Rarely authoritative | OGAR/ClassView law, compiler verdict |
| Test and benchmark evidence | Documentation / CI | Source/test symbols | Receipts with provenance if implemented |
| Local retrieval without invoking LLM | grep possible | yes | Quack query: yes for supported one-hop |
| Returned text tokens sent to Claude | Consumes model tokens | Consumes model tokens | Consumes model tokens |
| False absence on unindexed dynamic code | Common pitfall | Must report incomplete coverage | Must report incomplete coverage |

The claimed novelty is **not** 'using a knowledge graph for coding agents'. It is the convergence of an existing semantic method graph, compiled Quack programs, typed compiler refusals, independently validated physical costs and local context queries in **one substrate**, with versioned bridge to source symbols.

## R7. High-signal experiment that can falsify the benefit

Source task: change a prepared `GroupSum` or referenced predicate, with known compiler tests and independent oracle.

1. Pin `main` SHA, worktree hash, crate features, target and source-index revisions.
2. Derive a gold dependency/test set by reading actual source/compiler output independently of the context provider.
3. Compare task performance:
   - A: `CLAUDE.md` + grep/read;
   - B: language-server/source index;
   - C: Quack graph context with one-hop verified evidence;
   - D: C + narrowly cited source snippets.
4. Measure: answer/patch correctness, dependency recall/precision, missed guards, cold indexing cost, local query latency, number of tool calls, tokens **returned to the model**, and end-to-end task time.
5. Mutate a source file without committing, alter a `SchemaSeal`, inject misleading prose and ask for unsupported second-hop dependencies. Each must fail closed or visibly report uncertainty.
6. Pre-register success criteria. A graph result that is small and fast but misses a critical dependency loses to the baseline.

**Token accounting:** LLM-free local graph queries still consume CPU/IO. Model input tokens include tool results, tool descriptions and any echoed source material. Claims such as 'zero-token repository awareness' or 'graph automatically improves code quality' remain unverified.

## Connections to prediction, entropy and structural optimization

Repository-graph research primarily concerns **retrieval and grounding**, not introspective anticipation or safe automatic self-editing. Treat this as a separate thread from ONA's temporal implication/feedback research (see [companion ONA excerpts](2026-10_sag_ona_internal_experience_source_excerpts.md)).

Three independent propositions must be tested:
- Context: typed local graph slices improve correct retrieval at lower model token/context cost.
- Reasoning: versioned receipts plus learned hypotheses improve prediction over a simple historical cost model.
- Optimization: a proposed physical rewrite reduces materialization **and**, if measured, CPU cost under invariant semantics.

Do not let a positive result for one proposition serve as evidence for the other two.

### References
- [Ouyang et al. 2025, RepoGraph, arXiv:2410.14684](https://arxiv.org/abs/2410.14684)
- [Phan et al. 2024, RepoHyper, arXiv:2403.06095](https://arxiv.org/abs/2403.06095)
- [Shah et al. 2025, RANGER, arXiv:2509.25257](https://arxiv.org/abs/2509.25257)
- [Athale and Vaddina 2025, Knowledge Graph Based Repository-Level Code Generation, arXiv:2505.14394](https://arxiv.org/abs/2505.14394)
- [LSP 3.17 specification](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)
- [Merged D-SAG-1 #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509)
