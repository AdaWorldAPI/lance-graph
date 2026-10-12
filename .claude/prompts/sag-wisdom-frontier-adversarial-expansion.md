# Claude Code extension prompt: D-SAG-3 red-team / deferred falsifiers A16..A33

**Scope:** Follow-up on existing PR #1511, NOT a new engine or production implementation. Begin from merged #1509 and the three initial D-SAG-3 documents. Read:
- .claude/plans/sag-wisdom-frontier-ona-hyperon-v1.md
- .claude/plans/sag-wisdom-frontier-hard-problems-v1.md
- .arxiv/2026-10_wisdom_frontier_active_inference_delta_and_misspecification.md
- .claude/prompts/sag-wisdom-frontier-claude-code.md
- .claude/knowledge/maslov-pyramid-of-cognition.md

**Primary objective:** Prove that incremental wisdom-frontier expansion handles **withdrawal of support and epistemic inaccessibility** before optimizing scheduling. The first PR described positive discovery and CARE-style proposals. This extension deliberately targets the damaging negative cases: what happens when the brain must *unlearn*, abstain, change plans or give up.

## 0. Re-establish ground truth

Inspect actual current API/types, main revision and PR head; compare to #1509. List the source file and invocation for each proposed reuse (hydrate, ancestry/root masks, provenance, Quack Binder/Draft, one-hop limitation, CE64 plasticity and test-only Maturation3, action authorization, certificate invalidation). If an API is only proposed in a document, mark it NOT SHIPPED.

No production CE64 encoding change. No new opcode, graph engine, actor framework, universal field mask, or autonomous software mutation. Keep any new data types test-local and explicitly versioned.

## 1. Implement only A16-A19 first, if permitted to code later

Build a tiny independent **scalar replay oracle** over an append-only event fixture:

    Event = InsertOriginal(root_id, fact, source_epoch)
          | RetractOriginal(root_id, source_epoch)
          | Refute(fact, root_id)
          | MarkUnread(fact, scope)
          | RestoreReadable(fact, scope)
          | ChangeSchema(seal)
          | ChangeReading(reading)
    Outcome = Known | Missing | Refuted | Unread | Conflict | Stale | Incomplete

These are suggested test-local types, not production signatures. Document every mapping to real source types; do not assume enums exist.

Scenarios:
1. r0 grounds A, A -> B -> C, independent r1 also grounds B. Retract r0, then r1, and compare full scalar oracle after each step.
2. A->B->C->A without independent ground: never resolves. Add ground, resolve where eligible; retract it; ensure no stale child can ground its own ancestor.
3. Alternative prerequisite sets: one fully established route is enough if rules allow; union-popcount of all routes is *not* the number of required observations.
4. Required/known/refuted/unread/conflict masks under declared schemas, permuted input order and across visibility boundaries. An unread IAM edge is not false or missing-to-be-guessed.
5. One newly inserted independent observation changes exactly the relations that depend on it; duplicate insertion does not advance maturation.

For each step assert:
- resolved/open/refuted/unread/conflict state;
- exact independent original roots and derivation chain;
- optional stage 7..0 under an explicit experimental provenance only;
- byte-equivalence under repeated replay;
- no effect permission inferred from evidence.
- fixed point only when finite declared closure exhausted, not merely when budget expires.

Require a positive fixture, an adversarial negative/silence twin, and deliberate guard-disable RED test and restoration for every implemented invariant. Do not claim all A16-A33 ran after implementing A16 only.

## 2. Only after correctness, measure incremental vs full recomputation

Establish a full oracle from the immutable event history. Compare a candidate delta updater to oracle at *each event*, not only at end. Control source revision, root identity, schema, reading, candidate set and budgets. Test escalating fanout 1/4/16/128, duplicate paths, alternate supports and high churn.

Measure candidate enumeration, popcount, mask scatter, provenance unions, queue, memory, compilation, cache invalidation, refused work, and IO. Report wall, CPU, allocations, bytes decoded and complete end-to-end work. An O(1) bit count cannot by itself justify a whole-frontier speedup.

## 3. Attention policies need a counterexample suite

Compare FIFO, confidence, popcount, raw entropy, calibrated EIG, EIG with misspecification check, OpenPsi demand, and optional CARE challenger. Freeze reasoning semantics, observation roots and candidate pool while changing scheduler. Use separate engine axis for native ONA, the ONA MeTTa-NARS bridge, Hyperon/MeTTa and modern PLN.

Generate:
- noisy-TV irreducible randomness (high raw entropy, zero useful learning);
- one expensive missing prerequisite vs three local inexpensive ones;
- two observationally indistinguishable causal models with different interventions;
- all nominated models wrong;
- correlated support masquerading as independent evidence;
- stage-0 anchor with very large but irrelevant fanout;
- zero new evidence after fixed point;
- genuinely useful late-discovery target obscured by popular early anchors.

EIG is valid only for an explicit calibrated predictive model. If no calibrated model exists, return Uncalibrated and measure the baseline without pretending NARS f/c is p. Report both epistemic progress and actual task outcome. Do not convert speculative "causal usefulness" into a gate that suppresses all science.

## 4. Intention hysteresis, fairness and rollback

Pure simulation only:
- equal OpenPsi demand scores in reversed order;
- dominant demand changes every tick; does policy oscillate;
- high-entropy distractor continuously competes with lower urgency necessary work;
- committed goal is delayed but reachable vs impossible;
- goal exists but current subject lacks authorization;
- cue arrives just after revocation or schema seal change;
- planner chooses action but dispatcher refuses, with immutable receipt;
- new evidence contradicts the active *goal's* premise vs merely a descendant hypothesis.

Maintain separate timestamps for PRE selection, goal-commit, plan-ready, live-permit, effect and POST outcome. An immutable PRE snapshot must replay irrespective of later observations. A program hash/receipt is not cryptographic security absent proper canonicalization and threat model.

## 5. Causal disagreement triage

Run three *separate* oracles for the same example:
A. symbolic graph ablation (which derivations reopen),
B. interventional simulator (which world outcome changes),
C. misspecified-model critic (which candidate causal models lack adequate predictive coverage).

Create one case where A fires but B does not, one where B is not identifiable, and one where C detects that every candidate model is false. Report CANNOT_INFER_CAUSALITY instead of auto-promoting CE64 epistemic certification.

CARE from arXiv:2606.14581 is a separate challenger-versus-reference programmable decision gate, not a synonym for these causal tests.

## 6. External oracle interoperability

Pin source SHA, flags and dependencies before any run, and write a fieldwise trace matrix preserving raw f/c, temporal vs eternal events, root IDs, alternative bindings, rule identifiers, action consequences and refusal reasons. Native ONA vs MeTTa-NARS needs to be checked **before** comparing to native Hyperon/PLN. Unknown or unrepresentable meanings must be marked Unsupported rather than coerced.

## 7. Delivery protocol

Keep #1511 intact; add measured experiment code only in a *separate* explicitly scoped implementation PR after the source audit and approval. Update the research PR with evidence links rather than claiming work completed.

Return a table with each A16..A33 row and a status from:
SOURCE-VERIFIED / PROPOSED / BLOCKED / IMPLEMENTED / TESTED / MEASURED / REFUTED.

For every implemented row give exact path, command, exit status, source SHA, reference oracle, disabled guard/red result, restored green result, seed and all-in cost. For proposals give the smallest next experiment and its expected failure mode. If a condition cannot be tested using current interfaces, record a typed blocker and STOP that axis.

**Non-negotiable kill gates:** No CE64[50..52] reinterpretation in existing provenance; no use of [59..63] as utility; no self-generated independent observation roots; no authorization bypass, fabricated official CARE suite, multi-hop claim, or unimplemented H1 performance claim. If the evidence does not support the research idea, say so plainly.
