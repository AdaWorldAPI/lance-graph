# D-CFA-3 — S3 behavioral discovery: ARM association plus counterfactual evidence

**Status:** DESIGN / preregistered negative controls; **NOT RUN**. Depends on [CFA-2](cfa-2-ablation-replay.md). This stage proposes a representation for *analysis results*, not a new learning crate, source store, engine or production dependency.

## Hypothesis to falsify
Given trusted S1 static candidate edges and S2 replayable ablation observations, can ARM candidate discovery and conditional/recursive relational reasoning recover **dependency, prevention, redundant sufficiency and conjunction** with fewer false positives than co-activation alone?

## Observations and shape boundaries
- `OBS`: evidence-bearing episodes on a pinned source revision: observed firing, guard outcomes, method identity, result/effect proposals, read/write sets, intervention and output difference. Missing/unauthorized coverage is explicitly UNKNOWN.
- `ARM`: integer counts `|X|`, `|X & Y|`, population `n`; confidence/support and proposed `{s,p,o,f,c}` from existing `lance-graph-arm-discovery` `extract_rules` / `arm_to_truth_u8`. An ARM rule is statistical association, not a discovered control-flow edge or semantic law.
- `CF`: intervention evidence `B[v] ↔ B[v] under disable(A)`; model-conditioned influence is directional, context-qualified and outcome-projection-qualified. Preserve differences between single-method and cell-wide ablation.
- `LOGIC`: relation closure on **explicitly admitted premises** in RelAlg/Datafrog/Ascent; MORK/MeTTa is an optional independent structural matching arm, not a second source authority. A derivation from hypothetical edges inherits candidate status.
- `HYPEREDGE`: `{A,B}|guard G -> C` for conjunction/redundancy/synergy; two binary links cannot automatically replace a conjunction.
- `EVIDENCE`: source roots, count windows, permissions/completeness, derivation chain, independent roots and counterfactual receipt kept distinct from learned score.

Canonical discovery source is existing `lance-graph-arm-discovery` (standalone zero-dep by default); on resident data, Quack/mask-risc Count supplies integer population evidence. This convergence is already documented by #1313 and measured on a limited SPOFC cycle in #1359. **Do not duplicate counts in a new hot executor.**

## Proposed relation labels (not ABI, not facts by declaration)
`co_activates`, `may_affect`, `observed_interference`, `conditionally_prevents`, `enables`, `redundant_under`, `conjunctively_requires`, `order_sensitive`, `equivalence_candidate`.

Each label must carry its epistemic grade and declared observable boundary. `supersedes` stays an explicitly declared versioning/ownership rule, **not** something learned from correlation.

## Preregistered Falsifiers
| ID | Fixture | Required observation |
|---|---|---|
| L01 | A,B always co-activate, neither affects the other's outcome | ARM association may hold; NO counterfactual dependency |
| L02 | A writes X read by B, but guard makes B unreachable | static may-depend edge, no active causal edge in that context |
| L03 | A prevents B only under guard G | conditional `prevents`, never unconditional |
| L04 | both A and B can produce C, each single ablation silent | redundancy recognized from joint ablation |
| L05 | C requires both A and B | preserve conjunctive hyperedge |
| L06 | one evidence root appears in three thought branches | no triple independent confidence |
| L07 | incomplete negative population or unauthorized rows | no false absence or universal prohibition |
| L08 | relation closure traverses proposed A→B and B→C | derived A→C remains proposal-grade |
| L09 | recursive negative cycle with no valid stratification | refuse, not lucky FIFO evaluation |
| L10 | association mined from train split, held-out class differs | report false positive and generalization gap |
| L11 | bag/multi-emitter counts used as Distinct set relation | refuse lossy multiplicity unless declared |
| L12 | link label reused with different source schema frame | reject cross-frame join absent witness |

## Fair comparisons
- Static Ruff/OGAR (possible-read/write graph).
- ARM counts alone (association baseline).
- ARM plus Thought-Git ablation evidence.
- ARM plus counterfactual evidence + independent relational oracle.
Keep training/held-out source roots disjoint. Measure precision, recall, false-positive inhibition edges, missed higher-order edges, abstention, quality of explanation, fold/ablation count and peak memory separately. Use no claimed scores until executed.

## Next stage
[CFA-4](cfa-4-nsl-onnx.md) may *rank* untested candidates from these witnesses. It cannot upgrade a prediction to admitted source facts or dependency laws.

Anchors: [#1313](https://github.com/AdaWorldAPI/lance-graph/pull/1313), [#1359](https://github.com/AdaWorldAPI/lance-graph/pull/1359), [#1584](https://github.com/AdaWorldAPI/lance-graph/pull/1584), [#1706](https://github.com/AdaWorldAPI/lance-graph/pull/1706).
