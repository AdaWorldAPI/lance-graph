# D-SAG-3.1: deferred hard problems in the wisdom frontier

> **Aligned by [`sag-capstone-alignment-v1.md`](sag-capstone-alignment-v1.md)** — read it first: canonical `ProposedMethod`, the shared status vocabulary, D-id map, and the claims it marks stale (incl. "second hop refused").

**Status:** RESEARCH ONLY. Extension of [D-SAG-3](sag-wisdom-frontier-ona-hyperon-v1.md) on PR #1511, based on merged #1509. Nothing here is a shipped feature, performed experiment, authorization, or production ABI decision. [Source excerpts](../../.arxiv/2026-10_wisdom_frontier_active_inference_delta_and_misspecification.md).

## H1. Four distinct epistemic quantities

Prerequisite debt, probabilistic uncertainty, causal identifiability and intrinsic motivational urgency are NOT the same numerical object.

For a finite, declared prerequisite universe U, define masks R (required), K (established), F (refuted), and X (unread/inaccessible). Then:

    conflict = (K & F) | (K & X) | (F & X)
    missing = R & ~(K | F | X)
    blocked = R & F
    unavailable = R & X
    debt = popcount(missing)

Reject or surface conflict rather than resolving its overlap by bitwise precedence. A zero debt means ready to independently validate, not true. R outside the declared U is a typed refusal. X is NOT negative evidence, especially for IAM security.

Unknown prerequisites can have alternative sufficient sets. Popcount of their union overestimates how many questions must be answered; compare each eligible alternative, accounting for shared roots without exponential unbounded DNF expansion.

A one-bit missing source that requires expensive, unauthorized remote work can cost more than a three-bit set answered from a local index. Popcount is a cheap structural feature, never a proxy for time, expected information gain or causal effect.

## H2. Fractal-like recursion is not mathematical self-similarity

A stage-0 anchor generates predictions, whose contradicted relations can begin their own seven-to-zero maturation cycle. A real graph is not necessarily a tree: it has joins, cycles, repeated ancestry, high-degree hubs, alternative derivations and versioned scopes. Do not claim a fractal dimension or infinite recursion from a recursive diagram.

Seed the frontier with verified stage-0 anchors only under explicit experimental provenance. Carry typed identifiers (ClassId, schema seal, reading, source revision, candidate identity, evidence epoch, scope) and a causal record for why a candidate was enumerated.

Each bounded exploration epoch has a stable snapshot, seed list, budgets for work and source access, a candidate dedup key, hop/row limits, cancellation token, per-anchor cap, and a declared deterministic ordering. Budget-exhausted is INCOMPLETE, not a fixed point. A real fixed point requires a finite, fully explored closure.

## H3. Incremental retraction and truth maintenance

Merged #1494 measured small test-local hydration and anchor ablation. That is not yet a general incremental truth-maintenance algorithm.

Fixture: independent root r0 grounds A; A supports B; B supports C; independent root r1 also supports B. Remove r0. A must reopen, but B and C may still be grounded via r1. Remove r1 next. B and C reopen. Never blindly erase every descendant and never retain a stale descendant after its only support disappears.

Cycle: A -> B -> C -> A with no external root must stay unresolved. With one external root it can ground according to declared rules. Removing that root must not re-ground the component using cached children. Newly adding a *different* independent root later may ground it again, but cannot inherit a phantom root from the first cycle.

**Independent oracle:** after EVERY delta, replay the entire immutable evidence-event history into a fresh scalar full-recomputation oracle. Require equality of resolved/open sets, independent root sets, maturation, refusal reasons, and eligible proposed actions. Only then evaluate delta update work. Differential Dataflow and seminaive evaluation provide research analogies, not a demand for another engine.

Events include insert, retract, independent-source correction, schema/readings change, rule revision, permit revision and explicit source inaccessible. An unread source must never silently become empty. Test every event ordering that should commute and declare where real-time observations cannot commute.

## H4. Multiple layers of identity and freshness

Test-local proposal record:

    (source_revision, schema_seal, class_id, reading_id,
     relation_identity, evidence_epoch, original_root_ids,
     derivation_ids, computation_version, authorization_scope)

Test-local auditable decision record:

    (snapshot, program_identity, compiler_version, candidate_set_hash,
     evidence_cutoff, reference_rank, challenger_rank, gate_features,
     tie_break, budget, intention_id, permit_id, outcome_receipt_id)

This is NOT an ABI and is NOT a cryptographically signed receipt. #1509 has a within-build FNV/Debug-derived digest that cannot be assumed stable or collision-resistant across builds. A cryptographic audit claim would require canonical serialization, keys, signature scheme and a threat model. Do not retrofit one by assertion.

A result cannot remain verified after its schema, class reading, provenance source or compiled program changes without a new check. Use source and data snapshot sealing to prevent temporal look-ahead.

## H5. Motivation, Rubikon and controller stability

OpenPsi-like demands select *what to investigate*, not permission to act. Keep four distinct records: epistemic statement, goal desirability, committed intention, and permitted effect.

Proposed cognitive work states:

    Candidate -> Deliberating -> GoalCommitted -> PlanReady
              -> PermitPending -> Executing -> Evaluating
              -> Done | Replan | Hold | Disengaged | Revoked

Checks:
- Equal demands plus reversed iteration order produce the same choice, and different winning demands map to meaningfully different goals (OpenPsi PR #117 is an unmerged upstream proposal at time of initial audit).
- Demand fluctuation every tick must not discard each previously committed plan. Measure goal-switching hysteresis against fast response to genuine contradictions.
- High-entropy curiosity may not starve a low-interest but necessary task forever. Report fairness/age quota as an explicit policy, not a hidden CE64 meaning.
- Impossible goal vs delayed-but-feasible goal must be distinguished. Authorization denial cancels dispatch but need not refute the goal itself.
- Every conditional action must re-check current cue, schema, capability, actor/subject and permit. A previous goal commitment is not authorization.

## H6. Three kinds of causal validation

1. **Graph support ablation** deletes an anchor in an isolated copy and measures which *derived statements* reopen. That establishes dependence on current proof/rule machinery.
2. **Causal intervention** changes a variable under specified structural/experimental assumptions and measures its outcome. This may require authorized simulated effects.
3. **Model criticism** checks whether all offered causal hypotheses might be wrong; high posterior confidence inside a misspecified model does not establish external truth.

Do not call (1) proof of (2). Do not ascribe (1) or (2) to CARE arXiv:2606.14581, whose transferable mechanism is executable challenger/reference selection with a pre-outcome commit gate.

Falsifiers: correlated evidence counted twice, ablation-sensitive derivation with false real-world cause, absent true mechanism from all rival models, observationally unidentifiable causal relation, rehydration masquerading as new independent evidence.

## H7. Evaluate information seeking against misspecification

A valid EIG requires a declared calibrated probability model over rival hypotheses and their observation likelihoods. NARS f/c and CE64 readings are not probabilities by default.

Always compare FIFO, confidence, typed popcount, raw Shannon, calibrated EIG, EIG with model-discrepancy checks, OpenPsi-like demand, and a separate CARE challenger gate. Freeze the inference engine when changing selection policy. Freeze selection when changing native ONA/Hyperon/PLN inference semantics.

Two traps: irreducible noisy-TV randomness (high entropy, no learnable structure), and confident-but-wrong finite hypothesis sets (internally large EIG for the wrong model). Include calibration and held-out model criticism, EIG vs out-of-distribution interventions, and negative/silence controls.

Treat pragmatic task value, epistemic learning value, causal testability, work and authorization as independent dimensions. Use Pareto and matched-budget curves before multiplying quantities into an arbitrary scalar objective.

## H8. Physical cost and limits

One popcount instruction is fast; generating a high-degree multigraph frontier is not automatically fast. Explore path explosions, duplicate nodes, alternate proof roots, memory contention, source fetch, Quack compiler, mask cardinality and receipt storage.

Measurements must include graph enumeration, root validation, queue management, decoding, IO, skipped/refused work, compile/validation, replay and cancellation. Evaluate chains and branching (1,4,16,128), shared descendants, cycles, stale-codebook churn and duplication storms. Claim a speedup only after equal correctness and all-in costs are measured.

Do not add a 64-field schema limit merely because a prototype uses a u64 mask. Make mask width a declared domain or use a typed multiword bitset. Multi-hop operations remain blocked in #1509 unless a separately verified legal relation-carrier path exists. No shadow traversal VM.

## H9. Cross-engine oracle matrix instead of parity marketing

For ONA native, ONA-through-MeTTa-NARS, Hyperon MeTTa, modern PLN/PeTTa and OpenPsi, capture raw input, temporal vs eternal event, raw f/c or truth object, derivation roots, rule IDs, answer alternatives and multiplicity, order contracts, selected goals, actual operations, runtime flags, source SHA, and stderr.

Only compare fields represented under both semantics. Unsupported is a legitimate outcome. Never normalize away disagreement between ONA time handling and MeTTa binding multiplicity, invent an official CARE benchmark, or call a fixture a cross-engine law without executing pinned oracles.

## H10. New adversarial ledger: proposals only

| ID | Fixture | Red condition |
| --- | --- | --- |
| A16 | Required / known / refuted / unread masks | Unread becomes false or conflict is hidden |
| A17 | Alternative support r0/r1 after retraction | Useful derivation lost or ghost support retained |
| A18 | Grounded and ungrounded reopened 3-cycle | Reopens and falsely regrounds itself |
| A19 | Delta permutations vs full recomputation | Inconsistent final proof/root sets |
| A20 | One expensive debt vs three cheap debts | Popcount declared optimal cost |
| A21 | High fanout with bounded queue | Budget exhaustion reported as fixed point |
| A22 | All candidate models wrong | High EIG called causal discovery |
| A23 | Pre-outcome cutoff under challenger scoring | Any future outcome leaks |
| A24 | OpenPsi tied demands reversed | Different goal selected by row order |
| A25 | Novelty storm and blocked task | Starvation or oscillating commits |
| A26 | Committed intention with canceled permit | Actual effect dispatched |
| A27 | Graph ablation vs causal intervention | Derivation dependence labeled causal truth |
| A28 | Mutated method body/seal/compiler/source | Stale proof accepted |
| A29 | Stage-0 anchor adjacent to private IAM edge | Cross-scope exploration |
| A30 | ONA eternal vs temporal, MeTTa alternatives | Non-equivalent rules coerced |
| A31 | Same evidence after fixed point | Reward/stage grows without new root |
| A32 | Cancel during execution race | Stale operation after cancellation |
| A33 | Alternative prerequisite sets overlap | Union-popcount misranks |

Each implemented row must have independently derived expected result, positive fixture, negative/silence twin, guard-disable -> red, restore -> green, seed, provenance and complete cost. All rows remain unexecuted research specifications.

## H11. Research release sequence and stop conditions

A0 audit current source and exact code API -> A1 four-state scalar masks -> A2 retraction vs full recomputation -> A3 existing Quack one-hop parity -> A4 calibrated attention -> A5 pure Rubikon/OpenPsi simulation -> A6 reference-conditioned CARE receipts -> A7 pinned ONA/Hyperon differential oracles -> A8 self-introspection proposal that is NOT auto-applied.

Stop if a run changes production CE64[50..52] plasticity, uses [59..63] as desire, invents independent observation roots, bypasses a live authorization check, mistakes budget exhaustion for proof closure, silently implements a new traversal engine, or reports a fused H1 speedup before the actual terminal exists.

**Open operator questions:** Is there a legal relation carrier for multihop? What provenance format can pin support sets under deletions? Which calibration data can be released? Would promoting maturation require its own declared semantic encoding or a versioned sidecar rather than stealing plasticity bits? Record decisions; do not silently choose them in this PR.
