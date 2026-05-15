# Sprint-log-LL1 — PR-LL-1 (NARS Intervention/Counterfactual verbs)

**Branch:** `claude/pr-ll-1-nars-intervene-counterfactual`
**Goal:** add 2 new `NarsInferenceType` variants (Intervention, Counterfactual), thread through Pearl 2³, add `AriGraph::intervene_on()`. ~200 LOC across 6 files.
**Spec:** `.claude/knowledge/neurosymbolic-rlvr-causal-curriculum-v1.md` §6.1.

## Worker manifest

| # | Agent | Owned file | Wave |
|---|---|---|---|
| W1 | nars-variants | crates/lance-graph-planner/src/thinking/nars_dispatch.rs | 1 |
| W2 | nars-engine-dispatch | crates/lance-graph-planner/src/cache/nars_engine.rs | 2 (after W1) |
| W3 | causal-edge-g | crates/causal-edge/src/edge.rs | 1 |
| W4 | arigraph-intervene | crates/lance-graph/src/graph/arigraph/triplet_graph.rs | 1 |
| W5 | tests-e2e | crates/lance-graph/tests/intervene_counterfactual.rs (NEW) | 3 (after W1-W4) |
| W6 | doc-update | .claude/knowledge/causal-edge-64-spo-variant.md + EPIPHANIES PREPEND | 1 |
| M | meta-r1 | synthesis + code review + commit | after all |

Wave 1 = parallel (no dep). Wave 2/3 depend on W1.

## OQ ratification (autoresolved)
- OQ-LL-1: graded NARS confidence (not binary)
- OQ-LL-5: clear ICM bit on counterfactual contradiction
