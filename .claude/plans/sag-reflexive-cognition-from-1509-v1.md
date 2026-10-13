# D-SAG-5: reflexive cognition and safe optimization from #1509

> **Aligned by [`sag-capstone-alignment-v1.md`](sag-capstone-alignment-v1.md)** — read it first: canonical `ProposedMethod`, the shared status vocabulary, D-id map, and the claims it marks stale (incl. "second hop refused").

**Date:** 2026-10-12. **Status:** PROPOSAL. Documentation only. **Parent:** merged [#1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509); follow-up research to merged [#1496](https://github.com/AdaWorldAPI/lance-graph/pull/1496), [#1497](https://github.com/AdaWorldAPI/lance-graph/pull/1497), [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498) and [#1499](https://github.com/AdaWorldAPI/lance-graph/pull/1499).

**Research sources:** [NARS, ONA, Hyperon](../../.arxiv/2026-10_sag1509_nars_hyperon_reflexive_appraisal_excerpts.md), [information theory and metagraphs](../../.arxiv/2026-10_sag1509_information_metagraph_optimization_excerpts.md), [pinned source and falsifier ledger](../../.arxiv/2026-10_sag1509_source_oracles_falsifier_matrix.md). **Runnable research brief:** [Claude Code prompt](../prompts/sag-reflexive-cognition-from-1509.md).

## Research question

Can the existing SoftwareGraph (SAG) become a **local, addressable, source-backed representation of its own prepared software**, and use that representation to prioritize useful investigations or *propose* optimizations, without either inflating epistemic certainty or silently building a second reasoning engine?

Treat this as **software introspection and cognitive control research**, not proof of a mind, qualia, AGI, transcendence or truly zero-token LLM context. Maslow is an analogy for epistemic prerequisites, not a theorem: reliable identity, evidence origin, memory continuity, limited resources, respect for conflicting hypotheses and time to reconsider are conditions for investigation. They do not guarantee a subjective experience or any high-level outcome.

## What #1509 actually shipped

The SoftwareGraph in crates/lance-graph-glove-parity/src/awareness.rs projects prepared MethodTable items into typed rows for class, method, field, condition, fold, law, program, primitive, refusal, claim, capability, test and compiled program lowering. Every edge includes a tier (declared, derived, compiled, verified, hypothesis), context and provenance. SoftwareGraph binds as an ordinary Quack table. Single-hop queries run through Draft -> lower -> mask-risc. Compiler verdicts and certificates are reflected as facts, not re-decided by the graph. Tests use the actual prepared programs and red-disable falsifiers.

The #1509 H1 candidate would fuse a predicate with a MaskedSumI32 terminal. The **measured current cost is worse**: roughly 603.5 microseconds tiled vs 379.1 microseconds scalar for one 1,048,576-row run, with other recorded ratios 1.6–1.8x. It is NOT a demonstrated fused SIMD speedup. A real Pred fusion and sum FusedFold, plus NULL and overflow obligations, remain open.

**Blocked:** multi-hop dependency queries beyond the one-hop context trick need D-SAG-1-4 / D-CMH-G1-4 relation-carrier or permitted EqU32Via; there is no free mask handoff between programs. SAG is currently in memory, not persistently read from Lance. MethodKey is caller-issued; body digest is debug-FNV build-local; SAG_REV is caller-supplied. No automatic source rewrite or actor self-optimization is implemented. Do not rewrite these facts when creating a prettier vision.

## Seven independent axes; do not encode them as one confidence score

| Axis | Proposed item | Source of authority | Failure mode to refuse |
|---|---|---|---|
| Identity | ClassId/MethodKey/NodeGuid, seal, ClassView reading, program/test key | typed OGAR, MethodTable, source compiler | same display label treated as same item |
| Observation | source code snapshot, compiled Program, test result, actual error | pinned Git revision, compiler/test | handwritten plan promoted to compiled tier |
| Evidence | observation/source IDs and independence/dependency bases | stamp or independent test | reread or copied layer counted twice |
| Derivation | executed rule/procedure and parents, signed witness | reproducible run receipt | inferred narrative presented as executed path |
| Attention | chosen target, policy, selection cost and observed outcome | run log, not factual belief | high priority mistaken for high truth |
| Interpretation | new later association or changed relevance of an episode | versioned relational significance | reinterpretation silently rewrites old event |
| Decision | proposed, independently proved, measured, authorized, applied/reverted | operator approval + compiler proof | unapproved source mutation or self-issued VERIFIED edge |

Proposed ContextSlice: bounded typed IDs, source file/span, repo SHA, MethodTable/Quack evidence paths, truth tier, omitted scope, and explicit unsupported operations. It aims to reduce repeated whole-document context transmission for Claude Code. An exact local graph query needs no LLM. But if an LLM consumes a ContextSlice, the slice still uses **input tokens**. Compare measured tokens, latency and answer fidelity against direct source reading, rg, the whole CLAUDE.md and ordinary text retrieval. Never promise literal token-free model context.

## Hypothesis policies, not emotions or CE64 bit assignments

- **Emberglow**: prioritize a candidate whose *independently evidenced learning progress* remains positive. Control: novelty churn and repeated old evidence.
- **Steelwind**: protect an approved objective from distractors, but stop upon disconfirmation or an expired budget. Control: inflexible perseverance.
- **Woodwarm**: choose familiar, historically reliable source paths to reduce lookup cost while testing freshness. Control: stale confirmation bias and source monopoly.
- **Velvetpause**: hold incompatible propositions open and schedule one discriminating observation before committing, with an explicit deadline. Control: endless postponement.

These four user-coined terms are proposed **functional policy labels**. Neither Wang nor Goertzel established them scientifically. Do not write them into CE64 bits 50..52 (which presently gate plasticity in one ClassView reading) or claim they represent subjective qualia. Zero-policy and fixed-policy baselines are essential.

**Respectful disagreement:** claim truth, independent evidence, source reliability and collaborative relation are distinct. Two researchers can disagree about proposition P without either source being automatically untrustworthy. An argument may be refuted without punishing other valid statements by the same person. Do not conflate source agreement with epistemic quality.

**Reinterpretation vs observation:** thought revisiting an episode at time k+1 may create a new relevance/relationship edge without becoming a second observation of the episode at k. Compare a fact-only history with a relational-history arm; a positive improvement must concern prediction/retrieval/repair, not a manufactured rise in NARS confidence.

## Crosswalk into the existing one-engine spine

| Source proposal | Which SAG operation could be tested | Important insufficiency |
|---|---|---|
| Pei Wang NARS internal experience, appraisal, mental operations | source-query, provenance chain, policy-only queue | no demonstrated mental operations or phenomenal feeling in SAG |
| ONA event priority and usefulness | schedule which compiler/source event to inspect | separate implementation from OpenNARS and PLN |
| PLN weighted inference and priority | compare truth-law outputs and queue policy on identical evidence | PLN strength/confidence != NARS frequency/confidence by definition |
| OpenCog ECAN | historical attention-control arm | ECAN != Shannon information gain |
| Shannon H and BALD | candidate observation ranking | H(f) is unsafe if f not calibrated probability; need predictive model |
| Miner empirical vs modeled surprise | detect surprising typed method patterns | repeated names across schema seals are not reusable code |
| Hyperon MeTTa | optional bounded source grammar and parity oracle | multiple results, grounded functions, dependent types, mutation not implied by Quack |
| Typed metagraph folds | compare constructor-preserving finite fold vs R2IL procedure | SIMD SUM != catamorphism; static snapshot assumption |
| Equality saturation (egg) | external equivalence reference for pure candidate rewrite | SAG is not an e-graph; no automatic compiler replacement |
| Heckhausen/Gollwitzer Rubicon | decision/commit authorization stages | governance analogy, not an NARS rung |

The engine remains: prepared MethodTable -> SoftwareGraph/Binder -> Quack -> R2IL procedures when actually implemented -> mask-risc -> ndarray. A proposed controller may rank **which existing legal query to execute next**, but may not mint a new OGAR byte, new VM, implicit carrier, optimizer backend or hidden global knowledge service.

## D-SAG-5 falsification ladder

| D-id | Probe | Independent oracle / red test |
|---|---|---|
| D-SAG-5-0 | literature plan, source map, prompt and board (this PR) | docs only, status explicitly proposed |
| D-SAG-5-1 | re-run #1509 one-hop queries and compiler tiers | actual MethodTable/compiled Programs; changed source or stale certificate refuses |
| D-SAG-5-2 | ContextSlice vs full source/rg/Markdown and tokens | exact answers, source SHA, false/omitted results, real model input token count; unsupported hop refuses |
| D-SAG-5-3 | evidence vs derivation vs attention | same observation copied into 2 layers counts once; 2 independent in same layer count twice; final truth unchanged by receipt replay |
| D-SAG-5-4 | episodic continuity and later appraisal | pinned version k replay identical while k+1 appends; changed significance does not rewrite old source event |
| D-SAG-5-5 | scheduler factorial | FIFO, ONA/PLN source-accurate queues, raw H(f), BALD, Miner/JSD, Moore/Staunen; identical truth laws and candidate budgets |
| D-SAG-5-6 | four functional-appraisal arms | policy changes attention only; measured utility vs no-appraisal and frozen/no-meta controls |
| D-SAG-5-7 | MeTTa and fold-law subset | 4 nondeterministic results not collapsed to 2 survivors; typed port ordering, error/unreduced form, constructor/snapshot parity |
| D-SAG-5-8 | safe H1, MDL and rewrite proposals | actual compiler + independent scalar oracle; NULL, bag/set, overflow, provenance, stale seal and mixed workloads |
| D-SAG-5-9 | conflicting arguments, relational significance | claim update not source-wide trust penalty; facts stable while context-aware prediction improves or policy rejected |
| D-SAG-5-10 | permission and rollback | proposal != authorization; no self-issued compiled/verified edge, no source or external write until separately authorized |

**Suggested order:** D-SAG-5-1 -> 2 -> 3/4 -> 5/6 -> 7/8/9 -> 10. An early failed claim may stop the downstream work.

## Required controls and metrics

Freeze one axis at a time: inference truth law, scheduler, source graph, evidence independence, appraisal, storage or context serialization. Score exact answer sets and evidence paths before optimizing performance. Include source SHA, class/seal/reading, event/provenance IDs, hypothesis-layer IDs, selected task, budget, candidate expansions, rows decoded, bytes, allocations, wall time, measured actual LLM tokens and reviewer decision. Pre-register useful-discovery and cost thresholds before running; do not retroactively fit.

**Adversarial fixtures:** same label/different ClassId, same fold/different reading, wrong-tier verified claim, expired digest, two-hop CALLS chain, copied evidence, irrelevant 50/50 noise, half-disagreeing deterministic ensemble vs all-uncertain ensemble, deadline missed under Velvetpause, stale "trusted" file, changed relationship without new observation, MeTTa 4-result branching, whole-edge vs port target, conflicting sources without personal blame, H1 fused candidate failing null/overflow, unapproved self-rewrite attempt.

The 2023 Hyperon paper describes the metagraph as substrate for reflexive cognition but explicitly says this does not itself produce deep self-understanding. That warning applies here: a **queryable graph** is necessary for some proposed introspection, not sufficient to establish self-awareness.

## Stop rules

- D-SAG-1-4 or D-CMH-G1-4 unresolved? Refuse multi-hop, do not add second evaluator.
- No independent evidence-base? Refuse cumulative confidence claim.
- No calibrated probability and predictive distribution? Label entropy heuristic, not Bayesian IG.
- No pinned static version? Refuse metagraph fold-equivalence claim.
- MeTTa produces new terms or multiple bindings that a mask cannot represent? Typed refusal.
- A proposed optimization lacks independent semantic oracle or actual fused primitive? Keep candidate unresolved.
- Rubicon-like commitment stage without operator authorization? Never execute.
- No independently observed performance/accuracy benefit? Report null or failure, do not claim adaptive self-optimization.
- No evidence of subjective states? Never claim machine qualia, feeling, respect or transcendence.

**No runtime code, CE64 meaning, new field, opcode, tenant, schema or ABI is authorized by D-SAG-5-0.**
