# Wisdom frontier: active inference, robust curiosity and incremental reasoning

**Date:** 2026-10-12. **Status:** Research-only annotated literature shelf (second wave for [D-SAG-2 / PR #1511](https://github.com/AdaWorldAPI/lance-graph/pull/1511)). These are **original paraphrased research excerpts** and engineering transfer hypotheses, not verbatim copyrighted extracts, reproduction of proofs, run logs or benchmark results. This directory includes publisher and project research references as well as arXiv sources; the directory name does not assert that every source is an arXiv preprint.

Read with [D-SAG-2 architecture](../.claude/plans/sag-wisdom-frontier-ona-hyperon-v1.md), [deferred hard problems](../.claude/plans/sag-wisdom-frontier-hard-problems-v1.md), and [first-wave literature](2026-10_openpsi_rubicon_care_wisdom_frontier_source_excerpts.md).

## E01. Friston et al. (2026): active inference and artificial reasoning

- **Reference:** Karl Friston, Lancelot Da Costa, Alexander Tschantz, Conor Heins, Christopher L. Buckley, Tim Verbelen, Thomas Parr, *Active inference and artificial reasoning*, [arXiv:2512.21129](https://arxiv.org/abs/2512.21129), initially December 2025; [Nature Communications 17, 10125 (published 8 September 2026)](https://doi.org/10.1038/s41467-026-77209-5). Journal version takes precedence for publication details.
- **Paraphrased excerpt:** The authors frame a policy as a sequence of actions selected partly for its expected effect on beliefs over *alternative underlying world models*. Information gain can be formulated through differences between predictive and updated posteriors, while model reduction supports selecting informative tests. They illustrate structure learning in a partially observed three-ball problem associated with synthetic insight.
- **Interesting transfer:** This is a more precise analogue to our stage-0 wisdom anchor than undirected novelty bonuses: an established explanation is useful when it generates **discriminating predictions** that lead to better identification of hidden structure.
- **Crucial nonclaim:** The publication does not validate CE64's eight-stage model, OpenPsi goalSelector, the Rubikon controller or a popcount frontier. Nor does it establish machine awareness in lance-graph.
- **Probe:** Compare model-targeted EIG against raw entropy and popcount on a three-ball-style independently specified hidden-rule fixture. A correct outcome must predict unseen trials, not simply assign every node a smaller debt.

## E02. Wei (2024): value of information is not arbitrary intrinsic reward

- **Reference:** Ran Wei, *Value of Information and Reward Specification in Active Inference and POMDPs*, [arXiv:2408.06542](https://arxiv.org/abs/2408.06542) (2024).
- **Paraphrased excerpt:** Examines a connection between expected free energy (EFE), belief-state planning and reward-oriented decisions, including how an epistemic value term relates to information value in partially observable decision processes.
- **Transfer:** Compare EIG-only curiosity with an explicit pragmatic objective and costs, not one unexplained "cognitive pressure" scalar. Stage 0 is maturation; a demand is preference; an authorized action is still a separate event.
- **Falsifier:** A curiosity agent collects maximally informative but operationally irrelevant samples while a cost-aware policy finishes the task. Document assumptions before importing claims of equivalence.

## E03. Catanach and Das (2023): information gain under a wrong model

- **Reference:** Tommie A. Catanach and Niladri Das, *Metrics for Bayesian Optimal Experiment Design under Model Misspecification*, [arXiv:2304.07949](https://arxiv.org/abs/2304.07949) (2023).
- **Paraphrased excerpt:** Standard Bayesian experiment design optimizes information gain using the selected inference model's predictions. The paper motivates and examines additional criteria, including expected generalized information gain and expected discriminatory information, to account for model discrepancy.
- **Transfer:** Make the original D-SAG-2 probability-calibration requirement stricter: holding a normalized posterior over a *wrong set of explanations* is insufficient. Maintain model criticism or a rival hypothesis family outside the trusted candidate set.
- **Falsifier:** Every candidate causal graph is false, yet a closed-world EIG controller becomes more certain; an independent discrepancy test must challenge this self-certainty. Do not import source formulas without their assumptions and quantities.

## E04. Pathak et al. (2017): action-relevant curiosity

- **Reference:** Deepak Pathak, Pulkit Agrawal, Alexei A. Efros, Trevor Darrell, *Curiosity-driven Exploration by Self-supervised Prediction*, [arXiv:1705.05363](https://arxiv.org/abs/1705.05363), [ICML proceedings](https://proceedings.mlr.press/v70/pathak17a.html) (2017).
- **Paraphrased excerpt:** The intrinsic signal is prediction error in a learned representation connected to the consequences of the agent's actions, rather than raw pixel surprise. The paper emphasizes filtering aspects of the environment the agent cannot influence.
- **Transfer:** Compare whether a frontier question is actually answerable/learnable under available observations and permissions before spending budgets on surprise alone.
- **Falsifier:** A random, unlearnable high-entropy stream dominates selection forever despite yielding no prediction improvement and no verified discovery. **Do not** claim a model-free method automatically identifies true causal structure.

## E05. Burda et al. (2018): exploratory bonuses have their own biases

- **Reference:** Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov, *Exploration by Random Network Distillation*, [arXiv:1810.12894](https://arxiv.org/abs/1810.12894) (2018), [original research code](https://github.com/openai/random-network-distillation) (historical, archived).
- **Paraphrased excerpt:** A prediction-error bonus relative to a fixed random target network can supply intrinsic reward; the authors also study combining exploration and external reward, reporting Atari exploration results.
- **Transfer:** Add RND-like novelty to the **comparison ledger only if a real implementation is pinned**; do not silently conflate a stage transition, residual graph debt, and novelty bonus.
- **Falsifier:** Exploration score remains high on repeated observations without improved task outcome, or dominates task utility because a coefficient changed post hoc.

## E06. McSherry et al. (2013): differential dataflow, iterative deltas

- **Reference:** Frank McSherry, Derek Murray, Rebecca Isaacs, Michael Isard, *Differential Dataflow*, [Microsoft Research publication / CIDR 2013](https://www.microsoft.com/en-us/research/publication/differential-dataflow/) and [Naiad technical report (2012)](https://www.microsoft.com/en-us/research/publication/composable-incremental-and-iterative-data-parallel-computation-with-naiad/).
- **Paraphrased excerpt:** Maintains changing results through differences and supports nested iteration; it motivates incremental graph computations that do not necessarily recompute all derived state after small input changes.
- **Transfer:** Our stage-0 seeded frontier needs insertion/retraction semantics and a full-recomputation oracle. Extract *correctness questions* about deltas and fixed points, not a demand to adopt Differential Dataflow or a new runtime.
- **Falsifier:** Removing a support root leaves a cached descendant "known", reintroduces a mutually supporting cycle, or wrongly discards a conclusion that has an independent alternate support. Compare full scalar recomputation after each event.

## E07. Beckers (2021): actual cause differs from derivation dependence

- **Reference:** Sander Beckers, *Causal Sufficiency and Actual Causation*, [Journal of Philosophical Logic 50 (2021)](https://doi.org/10.1007/s10992-021-09601-z); [published correction](https://doi.org/10.1007/s10992-021-09632-6).
- **Paraphrased excerpt:** The literature offers multiple formally different criteria for actual causation; simple necessary/sufficient reasoning and counterfactual intuition do not uniquely settle causal judgments.
- **Transfer:** Our isolated stage-0 graph ablation establishes **dependence under current inference rules**, not real-world causal sufficiency. An interventional oracle needs explicit assumptions, outcome controls and identifiability handling.
- **Falsifier:** A parent relation whose deletion makes a derived conclusion disappear is mistakenly certified as the cause of the external observed event. Respect inability to identify a cause.

## E08. Existing internal research that constrains the transfer

- [#1494 / Maslov Pyramid of Cognition](https://github.com/AdaWorldAPI/lance-graph/pull/1494): stage-0 wisdom is *demonstrated dependence* in test-local scaffolding; production CE64 bits 50..52 **still mean plasticity**. It is not posterior certainty.
- [#1497 / PLN, Shannon and Staunen](https://github.com/AdaWorldAPI/lance-graph/pull/1497): inference semantics, scheduler and physical execution are independent axes; NARS frequency/confidence are not calibrated probabilities.
- [#1500 / Rubikon volition](https://github.com/AdaWorldAPI/lance-graph/pull/1500): goal commitment is separate from conditional plan and permission to perform an effect.
- [#1509 / SAG](https://github.com/AdaWorldAPI/lance-graph/pull/1509): one-concept, one-hop projection is demonstrated using real Quack; multi-hop and persistent dataset are open, H1's SIMD terminal is not built.
- [CARE arXiv:2606.14581](https://arxiv.org/abs/2606.14581): reference-vs-challenger executable program validation and **pre-outcome** gate are the proposed transfer, not proof of causal intervention or symbolic truth maintenance.

## E09. Three competing research hypotheses (explicitly unresolved)

**H_A: Popcount suffices as an attention scheduler.** Likely cheap on suitable masks, but can be defeated by high-cost single missing links, false prerequisites and important deep investigations.

**H_B: EIG suffices as intrinsic motivation.** More principled on calibrated distributions, but may become confidently wrong under model misspecification, or overexplore when pragmatic utility/authorization is neglected.

**H_C: Causally gated learning progress + Rubikon commitments improves useful work.** Plausible engineering proposal, but its own overhead may exceed any discovery advantage; causal-gating might block safe observational learning, and commitment may make the system slow to revise when evidence changes.

**Required disambiguation:** exact same candidate pool, evidence roots, external oracle, budgets and replay under A/B/C plus FIFO/reference; full costs and negative controls; report cases where H_A beats H_C as honestly as reverse cases.

## Notes on verification

Publication metadata and high-level abstracts were checked on 2026-10-12 against arXiv/publisher pages; source-specific evaluation code, downstream reproductions, licenses and current external commit SHAs were **not** executed or pinned in this document. Never convert these literature comparisons into "implemented" or "benchmarked" rows merely because the sources exist. A future experimental PR must pin artifacts and execute the named oracles.
