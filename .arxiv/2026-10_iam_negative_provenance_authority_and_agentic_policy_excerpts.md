# IAM negative provenance, robust policy and action gates: second research distillation

> **2026-10-12 / source-aware design notebook, no implementation.** Companion to [D-IAM-SAG-1](../.claude/plans/iam-substrate-frontier-gaps-v1.md), [D-IAM-SAG-0](../.claude/plans/iam-substrate-sag-dir-sim-v1.md) and [the earlier primary excerpts](2026-10_iam_access_paths_temporal_provenance_primary_excerpts.md).
>
> **Method:** short *paraphrased* abstracts or named-section extracts; the publication may support a claim about its own formalism but does not prove a cross-system IAM implementation. Each entry explicitly labels **Source result [R]**, **LG transfer hypothesis [H]**, **test/falsifier**, and **non-transfer**. No long verbatim excerpts. The references were checked against the primary arXiv/OASIS/IETF landing pages. Newer arXiv material is a research proposal unless a cited evaluation establishes a narrower claim.

## N1. Why *not* access? Provenance with negation

**Primary:** Seokki Lee, Sven Koehler, Bertram Ludäscher, Boris Glavic (2017), *Efficiently Computing Provenance Graphs for Queries with Negation*, [arXiv:1701.05699](https://arxiv.org/abs/1701.05699), abstract, firing-rule technique and query-focused provenance computation.

**Exzerpt [R], abstract (sinngemäß):** An explanation for a missing query result needs to account for derivations that fail, not only positive output tuples. The work rewrites Datalog rules to expose successful and unsuccessful derivations, handles negation, and prunes irrelevant portions of the provenance graph. Its evaluated implementation executes relationally in SQL.

**LG hypothesis [H]:** Positive `RoleEvidence::Held` already has a concrete SID-chain witness. Negative `why_not_access` should require an explicit domain-closure certificate (which sources/groups/policies have been read) and a *failed derivation*, not just an empty bitmap. An authorized negative decision and an indeterminate read are different outputs.

**Falsifier:** remove an unread source page that holds an alternative grant; `why_not_access` must stop claiming complete absence. Two equal negative results under different coverage must carry different provenance. **Non-transfer:** the paper does not claim that a directory oracle is closed-world or that arbitrary Cypher/Quack multi-hop negation already lowers.

## N2. Robustness under future policy extension

**Primary:** Alexander V. Gheorghiu (2026), *Verification of Robust Properties for Access Control Policies*, [arXiv:2603.13181](https://arxiv.org/abs/2603.13181), abstract, robust-support judgment and compositional proof theorem.

**Exzerpt [R], abstract (sinngemäß):** Policies are often incomplete or assembled from components. The paper defines properties that remain guaranteed regardless of unspecified pending choices and later policy extensions, proves a monotonicity property for *that support judgment*, and reduces verification to a finite proof search in second-order logic programming.

**LG hypothesis [H]:** An SoD rule proven at policy version p7 may or may not remain safe after external grants at p8. Store the assumptions and version constraints under which a certificate is robust; if the native policy language does not support this proof, invalidate its use after policy changes.

**Falsifier:** add a second role grant from another provider after the approved snapshot. A point-in-time `no SoD conflict` proof must not remain valid unless its stated robustness assumptions are met. **Non-transfer:** the theorem does not assert that all enterprise policy combinations are monotone or that OGAR policy has implemented the paper's logic.

## N3. External enforcement for agentic proposals

**Primary:** Nils Palumbo et al. (2026), *Formal Policy Enforcement for Real-World Agentic Systems*, [arXiv:2602.16708](https://arxiv.org/abs/2602.16708), v3, abstract, policy-independent reference monitor and observability assumptions.

**Exzerpt [R], abstract (sinngemäß):** Encoding permission rules solely in a language model prompt provides no formal guarantee. The authors describe a policy monitor over program execution context with an observability service, an assume/guarantee contract and Datalog rules, and evaluate several agentic use cases. Any formal guarantee is conditioned on the environment contract.

**LG hypothesis [H]:** ONA/Hyperon-inspired introspection can rank read-only identity questions and generate proposals; *permission to mutate* must instead come from an external actor/scope/capability gate with a fresh source/policy precondition. Prompt text, SAG tier and a derived truth score never mint authority.

**Falsifier:** feed the observer a forged administrator note, an apparent `Tier::Verified` dependency and a stale approval; run identical query data. No provider-capable action should be reachable. **Non-transfer:** citing FORGE does not mean LG already has a reference monitor or Datalog interpreter.

## N4. Policy conflict resolution is specific, typed and often indeterminate

**Primary standard:** OASIS (2013), *eXtensible Access Control Markup Language (XACML) Version 3.0*, [OASIS Standard](https://docs.oasis-open.org/xacml/3.0/xacml-3.0-core-spec-os-en.html); policy/rule-combining algorithms, decision types and obligations.

**Exzerpt [R], combining algorithms (sinngemäß):** Deny-overrides prioritizes Deny under defined precedence rules; permit-overrides, first-applicable and only-one-applicable differ. NotApplicable and different kinds of Indeterminate cannot be replaced by a Boolean “no grant”.

**LG hypothesis [H]:** An IAM graph may *project* `PolicyEvaluation{Permit,Deny,NotApplicable,Indeterminate}` with the versioned declared combining algorithm and obligation status. It must not “invent” a universal deny-overrides rule. This is a read-only semantic adapter, not production XACML enforcement.

**Falsifier:** one positive group role path + explicit deny, then switch the *declared* combining algorithm and test the expected (possibly indeterminate) outcomes against a reference evaluator. Missing condition must not turn into `Permit`. **Non-transfer:** an enterprise IdP need not use XACML semantics even if graph policy looks similar.

## N5. SCIM is an interoperability protocol, not a distributed transaction

**Primary standards:** IETF [RFC 7643](https://www.rfc-editor.org/info/rfc7643) (SCIM core resource schema); [RFC 7644](https://www.rfc-editor.org/info/rfc7644) (SCIM protocol), specifically §§3.5.2, 3.14, 4 and the Bulk section.

**Exzerpt [R], RFC 7644 (sinngemäß):** SCIM supports conditional updates using ETags where offered, PATCH with atomic semantics for the *single resource request*, and service-provider configuration discovery. The specification does not make writes across independent directories and SaaS products one ACID transaction.

**LG hypothesis [H]:** a future action connector needs capability discovery, source version/ETag, a purpose-scoped approval, provider idempotency semantics, and a receipt. If a provider returns timeout with uncertain effect, do not immediately repeat a destructive request or claim failure; re-observe.

**Falsifier:** issue a synthetic duplicate request after a simulated post-success timeout, verify only one intent is reconciled; stale If-Match refuses. **Non-transfer:** SCIM read/write features may be unsupported by an individual provider. Not every AD/Graph/PowerShell endpoint is SCIM.

## N6. Pre-commit policy conflicts and emergency access are domain-specific

**Primary:** Nasif Muslim, Jean-Charles Grégoire (2026), *An Extended Consent-Based Access Control Framework: Pre-Commit Validation and Emergency Access*, [arXiv:2603.07004](https://arxiv.org/abs/2603.07004), abstract, pre-commit conflict checking and emergency disclosure constraints.

**Exzerpt [R], abstract (sinngemäß):** In a healthcare consent setting, the authors propose checking contradictory permissions *before activation* and constrain emergency access using context-sensitive evidence. Reported experiments use controlled synthetic workloads. This is a narrow domain design, not a universal break-glass algorithm.

**LG hypothesis [H]:** simulate privileged activation and SoD-policy conflicts before a change is made; preserve separate emergency *reason*, human approval, valid interval and session evidence. Emergency access must never arise simply because a reasoning model estimates high urgency.

**Falsifier:** proposed emergency role with no declared break-glass authority is refused; a valid but expired emergency grant remains non-executable. **Non-transfer:** no health-data or emergency semantics are part of standard OGAR IAM policy unless explicitly imported.

## N7. Prior provenance complexity remains a release gate

**Primary:** Marco Calautti et al. (2023), *The Complexity of Why-Provenance for Datalog Queries*, [arXiv:2303.12773](https://arxiv.org/abs/2303.12773), abstract and complexity results; also detailed in the first IAM excerpts.

**Exzerpt [R], abstract (sinngemäß):** Computing witnesses for recursive Datalog why-provenance is intractable in some studied cases, whereas nonrecursive cases are substantially more tractable. Practical SAT-based experiments are reported.

**LG hypothesis [H]:** witness search must expose a budget, whether a result is one-path/exhaustive, and the size of the claimed source closure. Negative access explanations and minimal interventions require stronger claims than a single positive witness.

**Falsifier:** thousands of overlapping grant paths plus a group cycle, with a fixed query budget: refusal/incomplete result is allowed, silent false “no access” is not. **Non-transfer:** no automatic performance result for Quack, nor a proof that each IAM graph instance requires SAT.

## N8. Existing reference on evidence independence

**Primary:** Audun Jøsang (2007), *Cumulative and Averaging Fission of Beliefs*, [arXiv:0712.1182](https://arxiv.org/abs/0712.1182), fusion/fission sections. Prior extract: [NARS evidence notes](2026-10_nars_self_epistemic_needs_evidence_primary_excerpts.md).

**Exzerpt [R] (sinngemäß):** dependent observations require different fusion from independent observations; removal of a known contributor can require original evidence and preconditions. The formalism is Subjective Logic, not automatically NARS.

**LG hypothesis [H]:** one HR event relayed through AD and Entra is one upstream *event*, not three independent confirmations. Separate the observation receipt from its derivations; when simulating ablation, remove the upstream contribution and recompute downstream effects at one pinned version.

**Falsifier:** replicate the same HR event through two adapters, replay both, and verify that license evidence does not gain an artificial second source. A fully independent event may contribute separately. **Non-transfer:** the current F/C register does not encode all dependency information needed for exact fission.

## Synthesis / decision map

| Research result | Immediate safe reuse | Need to build | Never infer |
|---|---|---|---|
| Negation/why-not provenance | `dir_sim::RoleEvidence`'s explicit Undetermined | source-scope closure and failed-derivation receipt | missing edge = denial |
| Robust properties | versioned policy tests | certificate assumptions and policy-extension invalidation | all policies monotone |
| Agentic reference monitor | independent `do()` boundary | provider authorization and effect receipts (later PR) | SAG confidence authorizes action |
| XACML decision algebra | explicit evaluation outcome types | binding of declared policy-combining semantics | universal deny-overrides |
| SCIM ETag and PATCH | provider contract checklist | read-only capability discovery before any future write | cross-system atomicity |
| Pre-commit conflict checking | `dir_sim` what-if simulation | SoD/JIT policy rule projection | emergency inference is permission |
| Recursive provenance | bounded witnesses and explicit completeness | legal Quack relation carrier/parity | one path is full explanation |
| Dependent evidence | source receipts and delta lineage | evidence ancestor equivalence semantics | two adapters = two independent facts |

**Priority:** start with negative-proof completeness and multi-source identity/temporal provenance in a *read-only* follow-on, not a new engine or a DO integration. This literature supports sharper falsifiers; it does not certify implementation.
