# 34 reasoning tactics: source and Lance-graph wiring

> **Plan and board (2026-10-20):** owned by [`alpha-channel-rung-overlay-v1`](../../.claude/plans/alpha-channel-rung-overlay-v1.md), [`thought-git-api-v1`](../../.claude/plans/thought-git-api-v1.md). Board: [2026-10-20-prs-1601-1617-plan-links.md](../../.claude/board/entries/2026-10-20-prs-1601-1617-plan-links.md). Merged in #1609.

**Owner:** AdaWorldAPI/lance-graph  
**Status:** research and implementation plan, not a claim of complete execution  
**Source:** James Lee Stakelum, [Beyond Chain of Thought: 34 Next-Gen LLM Tactics Nobody's Talking About](https://medium.com/@JamesStakelum/beyond-chain-of-thought-34-next-gen-llm-tactics-nobodys-talking-about-43b644e1d951)

The full user-supplied article extraction is held as a separate Markdown reference artifact. This repository note links to the original rather than reproducing its full copyrighted prose.

## Architectural ownership

The 34 recipes belong to lance-graph, not MedCare-rs. MedCare-rs is a downstream clinical benchmark consumer. Existing lance-graph catalogue, kernel, dispatcher and substrate surfaces are `crates/lance-graph-contract/src/{recipes,recipe_kernels,recipe_dispatch,recipe_substrate}.rs`; inspect them at implementation time rather than assuming all named recipes have full semantics.

## Typed execution

```text
34 recipe IDs + typed tactic intent
  -> Narsese / NAL / ONA and Hyperon / MeTTa / Atomese adapters
  -> RelAlg / Datalog AST (entailment and rewriting remain distinct)
  -> R2IL / mask-risc / masked Lance reads
  -> 64 concurrent active thought slots, sparse overlays
  -> semantic Git versioned derivations, branch, diff, replay, rebase
  -> evidence-bearing SPO / CE64 conclusions
```

64 thoughts are active slots, **not** 64 total steps. Stable thought identity and version witnesses survive slot recycling. The thought address is intentional; the version is the witness. Distinguish a derivation dependency graph from Git ancestry. CE64 W bits 53..58 encode the active thought slot; bits 50..52 encode eight Staunen-to-wisdom maturation stages, not a generic confidence score. Keep epistemic state distinct from NAL truth arithmetic and causal certification.

## Phased implementation plan

1. **Contract audit:** reconcile all 34 numeric source IDs with `recipes.rs`, dispatcher reachability, kernel effects, and actual supported semantic operations. Record discrepancies, including duplicated abbreviation ICR in the article (recipes 11 and 31). No claim that a mutable scalar kernel implements an entire reasoning method.
2. **Typed recipe ABI:** define `RecipeIntent`, `ThoughtRevision`, `EvidenceRoot`, `DerivationEdge`, `SemanticDiff`, `Intervention` and `MaturationDecision` with compact register-oriented IDs. No JSON in production reasoning paths.
3. **Execution vertical slice:** HTD #2, SMAD #3, TCP #5, MCP #10, and counterfactual ICR #31. Prove each recipe reads evidence and can change a typed derivation or decision under controlled inputs.
4. **Reasoner adapters:** translate Narsese/NAL/ONA truth operations, Hyperon/MeTTa/Atomese patterns and Datalog/RelAlg rules into canonical contracts without silently conflating their inference semantics. Preserve original expressions for provenance and loss reporting.
5. **64-thought frontier:** schedule independent, competing and dependent hypotheses with presence masks, bounded resource budgets, sparse overlays, and deterministic replay where promised.
6. **Semantic Git:** record immutable source-root witnesses, thought revision identity, premise dependency, semantic diff, retraction, branch and replay. Implement blame/pickaxe by provenance, not text search. Preserve contradictory branches until explicit reconciliation.
7. **Evaluation:** use source-root deduplication, overdetermination, contradiction preservation, temporal leakage guards, necessary-premise ablation, placebo reasoning steps, NAL revision tests, and counterfactuals with explicit causal models. Measure score changes, proof faithfulness, mask bytes read, latency and replay stability.
8. **Downstream consumers:** MedCare-rs/MedR-Bench validates clinical reasoning externally; IAM and directory simulations can exercise alternative domains. No domain-specific benchmark code inside the 34-recipe engine.

## Non-negotiable boundaries

- Distinguish source prompts from formally defined reasoning algorithms.
- Do not equate Datalog entailment with Pearl interventions.
- Do not count duplicate evidence as independent NAL support.
- Do not substitute decorative chain-of-thought text for operative computation.
- Do not overwrite or delete existing code, docs, or PRs without explicit approval.
- All proposed integrations must be independently testable, replayable and auditable.

## First PR exit criteria

This documentation PR is additive only. The next implementation PR must prove one complete thought-to-evidence-to-derivation-to-version cycle with an ablation test; no benchmark or clinical safety claim follows merely from passing that test.
