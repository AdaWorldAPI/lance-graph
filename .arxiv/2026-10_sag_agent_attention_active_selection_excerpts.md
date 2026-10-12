# D-SAG-4: bounded context attention, costly exploration and correctness-first uncertainty

**Date:** 2026-10-12. **Mode:** primary-paper **abstract-level paraphrases**, not verbatim quotations, no code implementations or D-SAG performance claims. Some papers have detailed methods/results not reproduced here; CC must read and pin an exact PDF version before claiming detailed parity or method reproduction. Companion: [deferred contracts](../.claude/plans/software-awareness-agent-context-deferred-contracts-v1.md).

## A1. Liu et al. (2023): *Lost in the Middle: How Language Models Use Long Contexts*

**Paper:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni and Percy Liang, [arXiv:2307.03172v3](https://arxiv.org/abs/2307.03172), TACL 2023. **Source:** abstract, main experimental conclusion.

**Excerpt (paraphrase).** A longer input does not guarantee useful access to relevant evidence. In multi-document QA and key-value retrieval, the authors find that changing the **position** of relevant context can substantially change model performance; models often do best when the critical item is near the beginning or end and degrade when it is buried centrally.

**Proposed D-SAG transfer:** provide a brief intent-specific context slice placing *critical semantic guards and source evidence* clearly, not a high-degree neighbor dump. Measure context ordering as a factorial dimension separately from length, and do not presume an extra-long context window fixes omissions.

**Falsifier:** identical source facts returned in different order cause different correct-patch outcomes, or an important population/NULL law is buried by low-value nodes. A smaller graph context that drops a necessary law still fails.

**Boundary:** findings are for models/tasks tested in the paper, not a measured law for Claude Code 2026.

## A2. Phan et al. (2024): *RepoHyper: Search-Expand-Refine on Semantic Graphs for Repository-Level Code Completion*

**Paper:** Huy N. Phan, Hoang N. Phan, Tien N. Nguyen, Nghi D. Q. Bui, [arXiv:2403.06095](https://arxiv.org/abs/2403.06095), current abstract describes an RSG, graph expansion and link-prediction refinement. **Source:** abstract.

**Excerpt (paraphrase).** Repository context can be discovered by first locating relevant code, then expanding through semantic relations, then refining the candidate set; the paper reports stronger code-completion performance than its tested baselines.

**Proposed D-SAG transfer:** `resolve → one Quack hop → rank/budget → optionally expand`. Critically, unlike RepoHyper's own engine, D-SAG #1509 currently only supports its proven one-hop `ScatterOrU32` query. Document `UnsupportedMultiHop` rather than imitating an expanded walk with a new interpreter.

**Falsifier:** link prediction provides plausible but compiler-unverified method dependencies; returning them as `Compiled` or `Verified` is illegal. Keep link prediction (if used) in `Hypothesis`.

## A3. Ouyang et al. (2024/2025): *RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph*

**Paper:** Siru Ouyang et al., [arXiv:2410.14684](https://arxiv.org/abs/2410.14684), originally submitted 2024, revised 2025. **Source:** abstract.

**Excerpt (paraphrase).** Repository-level relationships support code agents on tasks where single-file/function representations omit relevant structure. The authors report improvements when integrating a repository graph with several tested agent frameworks and on another repository benchmark.

**Transfer:** **source-symbol** navigation is worth comparing with `CLAUDE.md`/grep, but a source dependency graph is neither a semantic ClassId ontology nor a Quack method compiler. Link the two with explicit source-to-method witnesses.

**Falsifier:** a trait impl or code-generated source reference is missing, and the graph incorrectly claims 'no callers'. Compare against compiler-backed navigation and independent source oracle.

## A4. Song & Xu (2017): *A Contextual Bandit Approach for Stream-Based Active Learning*

**Paper:** Linqi Song and Jie Xu, [arXiv:1701.06725](https://arxiv.org/abs/1701.06725). **Source:** abstract and stated motivation.

**Excerpt (paraphrase).** Many bandit formulations assume reward observation is free; active learning under costly labels must trade off information received against the cost of asking. This study explores contextual bandits that selectively request costly feedback and communicates prior information alongside queries.

**Proposed D-SAG transfer:** an optimization experiment is an **action with cost**, not free knowledge. Select whether to benchmark a candidate by considering current uncertainty, expected utility, impact and time/CPU budget. Compare information-gain choices against a simple random/impact-priority scheduler.

**Falsifier:** an ONA-like policy spends more compute acquiring performance receipts than it saves in later plan selections. A model that only optimizes per-execution cycles while ignoring exploration cost is incomplete.

**Boundary:** theorem/regret results for that paper's formal problem are not an automatic guarantee for method-selection or NARS evidence in lance-graph. Do not transplant reward or confidence formulas.

## A5. Shi et al. (2026): *Code Is More Than Text: Uncertainty Estimation for Code Generation*

**Paper:** Yuling Shi, Caiqi Zhang, Yuexian Li, Haopeng Wang, Yeheng Chen, Nigel Collier and Xiaodong Gu, [arXiv:2606.09577](https://arxiv.org/abs/2606.09577). **Source:** abstract.

**Excerpt (paraphrase).** Uncertainty about code has multiple dimensions, including lexical fragility, algorithmic intent/implementation divergence and functional behavior that can be executed and tested. The authors evaluate complementary signals rather than identifying one entropy score as a substitute for functional verification.

**Proposed D-SAG transfer:** distinguish (1) uncertainty about retrieval/identity, (2) semantic legality of a proposed transform, and (3) performance prediction dispersion. Source-backed compiler/test evidence outranks surface language confidence. Use uncertainty to **route to validation**, not to certify a patch.

**Falsifier:** a low-entropy proposed edit violates a sealed ClassView or gives wrong independent oracle output. The guard rejects it regardless of its model confidence.

## A6. Rakasi et al. (2026): *When Uncertainty Isn't Enough: An Empirical Study of Self-Correction in Code Generation*

**Paper:** Pranav Rakasi et al., [arXiv:2608.14659](https://arxiv.org/abs/2608.14659). **Source:** accessible abstract-level search result; **paper PDF not independently inspected for this note**.

**Excerpt (abstract paraphrase).** The authors report that simple uncertainty-driven code self-correction does not consistently increase correctness in their tested model/benchmark settings, while correction guided by executable verification is more consistently useful.

**Transfer:** the ONA-inspired selector should direct attention toward *checks and experiments*, not self-certify a generated patch from its own assessed confidence. Compare prediction-only to verification-gated policy.

**Falsifier:** 'confidence revised upward' but executable tests fail. Code correctness remains false even if the hypothesis scorer likes the patch.

**Boundary:** do not copy paper-specific numerical scores or claim universally superior verification across all models. Their evaluation is not a benchmark of our architecture.

## Research synthesis: two different attention problems

The analogy to foveation/saccades is useful only when made operational:

**Context attention (model-facing):** select the smallest sufficient facts, source references, tests and proof obligations under a model-visible token budget. Local graph queries spend CPU/IO, not model inference. Expansion later costs additional returned tokens. Measure correct dependency recall and patch correctness, not merely textual compression.

**Experiment attention (system-facing):** select the next *safe and informative* experiment under a CPU/energy/time budget. A new receipt may reduce hypothesis uncertainty, or reveal a semantic incompatibility that forces refusal. Measure regret/calibration/benefit net of experiment cost; no observed outcome is not the same as negative evidence.

```text
                      Same qualified software graph
                                |
                 +--------------+--------------+
                 |                             |
          Local context lens             Experiment lens
        which facts to show?         which legal probe to run?
                 |                             |
          bounded evidence             predicted outcome
                 |                             |
          Claude + citations           independent execution
                 |                             |
        source-aware change             measured receipt
                 |                             |
        tests and review                 revised hypothesis
```

The two lenses may share typed graph infrastructure and provenance but **must not** share unchecked authority: a high-interest concept is not a true concept; a frequently cited source is not a verified source; a predicted improvement is not a demonstrated optimization.

## Proposed experiment ladder

- **F1:** graph vs greedy symbol/LSP retrieval with identical 4K input-token budget. Preserve mandatory semantic guards; compare patch correctness and critical dependency recall.
- **F2:** vary context ordering, same total tokens and facts. Does order change performance?
- **F3:** compare random, highest-degree and evidence-coverage-based one-hop selection. Record graph query costs.
- **F4:** add source edit between requests, verify stale/dirty worktree invalidation.
- **F5:** select optimization experiments via random, impact heuristic, historical cost and ONA-inspired anticipation; register predictions before receipts and use held-out selectivity/CPU tiers.
- **F6:** ablate execution receipts, perturb evidence ancestry, keep exact final method answer. Does any claimed self-modeling benefit survive?

**Required log fields:** repo/branch SHA, worktree digest, graph generation, source-index version, task identifier, compiler version/features, model/runtime budget, prompt/tool token accounting, missed gold dependencies, oracle pass/fail, CPU/data versions, cold index time and confidence/uncertainty methodology.

## Sources

[2307.03172](https://arxiv.org/abs/2307.03172) · [2403.06095](https://arxiv.org/abs/2403.06095) · [2410.14684](https://arxiv.org/abs/2410.14684) · [1701.06725](https://arxiv.org/abs/1701.06725) · [2606.09577](https://arxiv.org/abs/2606.09577) · [2608.14659](https://arxiv.org/abs/2608.14659).

**Negative finding is allowed:** if an LSP-backed source index handles normal coding tasks better, reserve Quack/SAG for what conventional tools lack: ClassId readings, law/refusal/physical-program provenance and bounded optimization hypothesis queries.
