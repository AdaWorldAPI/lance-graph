# D-SAG-2 source excerpts: OpenPsi, Rubikon, Shannon/EIG, CARE, ONA and Hyperon

**Date:** 2026-10-12. **Status:** ANNOTATED RESEARCH, not a replication, formal proof, or benchmark.  
**Parent work:** merged [#1509 software-awareness graph](https://github.com/AdaWorldAPI/lance-graph/pull/1509); [D-SAG-2 plan](../.claude/plans/sag-wisdom-frontier-ona-hyperon-v1.md).

**Editorial rule:** These are **short paraphrased excerpts / takeaways**, not verbatim reproductions. Every entry states (a) what is supported by its linked source, (b) the proposed transfer to lance-graph, and (c) a disconfirming experiment. This directory is a literature shelf; OpenPsi READMEs, GitHub PRs and peer-reviewed articles are **not** being misrepresented as arXiv papers. The full text of each work must be consulted before claiming algorithmic equivalence. No source is evidence that a D-SAG-2 probe has already run.

## S01 — CARE reference-conditioned executable programs (2026)

**Reference:** Guanyu Liu, Weiyi Kong, Chao Tang, Zeyu Wang, Boer Zhang, Baiqing Li, Peiyu Zhang, Tianyu Shi, *CARE: Context-Aware Ranking Evolution with Executable Scoring Programs for Budgeted Reaction Optimization*, [arXiv:2606.14581v5](https://arxiv.org/abs/2606.14581), revised 2026-08-31. [Local detailed annotation](2606.14581_synthesize_programs_gate_commit.md).

**Paraphrased excerpt:** A language model synthesizes reusable candidate-scoring logic rather than selecting the next experiment directly. A non-LLM reference policy supplies a competing selection. An explicit gate validates the challenger and chooses before the experiment outcome is known; subsequent results can retain, revise or regenerate the program. Its reported empirical successes concern budgeted **reaction optimization**, not ONA, Hyperon or a general causal reasoner.

**Transfer hypothesis:** Use a persistent, hash-identified Quack/R2IL-compatible ranking recipe to propose the next SAG introspection or frontier investigation; score **all candidates** and compare with a deterministic reference at a separate pre-outcome gate. Commit intent only after that, and authorize effects independently.

**Disconfirming fixture:** Candidate permutation changes the selected result; challenger only scores a subset; an invalid/nonfinite challenger still wins; or the scorer reads a post-selection receipt. Each must cause refusal/reversion to reference.

**Crucial separation:** CARE-as-*paper* is a policy-control pattern. **CARE-as-causal-testing** (anchor deletion, Pearl intervention, falsification, and Gadamer-style reinterpretation) is our independently proposed evaluation layer; do not attribute these mechanisms to the CARE experiment paper.

## S02 — OpenCog Hyperon metagraph architecture (2023)

**Reference:** Ben Goertzel et al., *OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond*, [arXiv:2310.18318](https://arxiv.org/abs/2310.18318), 2023. [Reference interpreter](https://github.com/trueagi-io/hyperon-experimental).

**Paraphrased excerpt:** Hyperon redesigns OpenCog around diverse interconnected representational and reasoning components, including graph-like representations, programmatic symbolic operations and aspirations toward reflective change. It is an architectural overview, **not** an experimentally established human-level AGI result.

**Transfer hypothesis:** D-SAG-1 already projects software methods, dependencies, compiler refusals and measured costs into ordinary graph rows. Explore typed fold/rewrite parity **on that existing substrate**, without a duplicate AtomSpace, interpreter or query engine.

**Falsifier:** Nested structural terms or variable-binding multiplicity cannot be round-tripped as ordinary rows; expose the mismatch and stop instead of claiming general MeTTa parity. Refer to [#1498](https://github.com/AdaWorldAPI/lance-graph/pull/1498).

## S03 — Folding / unfolding typed metagraphs (2020)

**Reference:** Ben Goertzel, *Folding and Unfolding on Metagraphs*, [arXiv:2012.01759](https://arxiv.org/abs/2012.01759), 2020.

**Paraphrased excerpt:** Typed metagraph operations can be formulated through recursive morphisms, including catamorphisms and anamorphisms, with treatment of higher-order link targets and recursive structures. This is a mathematical proposal, not proof that mask-risc implements these morphisms.

**Transfer hypothesis:** Compare a finite scalar fold/unfold reference with restricted ClassView/Quack projections. The fractal-like wisdom frontier is a **repeated discovery process**, not an assertion of exact metagraph morphism equivalence.

**Falsifier:** Same sum but different nested structure, link-to-link reference or alternative bindings. Numerical equality alone is not semantic parity.

## S04 — Shannon's uncertainty measure (1948)

**Reference:** Claude E. Shannon, *A Mathematical Theory of Communication*, *Bell System Technical Journal* 27 (1948), pp. 379–423 and 623–656. [Part I DOI](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x), [Part II DOI](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x).

**Paraphrased excerpt:** Entropy describes uncertainty for an explicitly defined source probability distribution. It does **not** define agent interest, explanatory causality or trust in an edge.

**Transfer hypothesis:** EIG must specify the random variable Theta (rival explanations), calibrated posterior p(Theta|D), observation model and cost horizon. No automatic reinterpretation of NARS f/c, PLN strength or CE64 code as p.

**Falsifier:** An entropy ranker is attracted indefinitely to an irreducibly noisy observation while a lower-entropy discriminating experiment is ignored.

## S05 — Intrinsic motivation and learning progress (2007)

**Reference:** Pierre-Yves Oudeyer and Frédéric Kaplan, *What is intrinsic motivation? A typology of computational approaches*, *Frontiers in Neurorobotics* 1 (2007). [DOI 10.3389/neuro.12.006.2007](https://doi.org/10.3389/neuro.12.006.2007).

**Paraphrased excerpt:** Curiosity and exploration can be made computational through distinguishable intrinsic-motivation mechanisms; psychological accounts do not trivially map to one reinforcement-learning reward law. Novelty, uncertainty, surprise and progress are not interchangeable.

**Transfer hypothesis:** Stage-7 astonishment is not itself an unlimited reward. OpenPsi-like demands should prioritize learning progress under finite budget and separate that from goal commitment.

**Falsifier:** High surprise with no predictable learning consumes all exploration; also compare an EIG estimator against an independently defined learning-progress baseline.

## S06 — Bayesian information-gain intervention choice (2021)

**Reference:** Michele Zemplenyi and Jeffrey W. Miller, *Bayesian Optimal Experimental Design for Inferring Causal Structure*, [arXiv:2103.15229](https://arxiv.org/abs/2103.15229), 2021.

**Paraphrased excerpt:** Interventions can be selected to rapidly reduce expected posterior uncertainty about causal structures; the paper proposes tractable summaries rather than repeated full hypothetical posterior fitting. Its evaluations include simulated and protein-signaling networks.

**Transfer hypothesis:** Use an action/observation model for rival SAG or IAM causal explanations and compare expected posterior entropy before acting. The result is a proposed **investigation**, not a permit to change state.

**Falsifier:** Estimated EIG derived from future outcomes; uncertainty reduction with no accurate causal identification; or gains disappear on held-out interventions.

## S07 — Interventions with target and value selection (2022)

**Reference:** Panagiotis Tigas et al., *Interventions, Where and How? Experimental Design for Causal Models at Scale*, [arXiv:2203.02016](https://arxiv.org/abs/2203.02016), 2022.

**Paraphrased excerpt:** Bayesian experimental design for causal structure may need to choose both *which variable* and *what intervention value* under nonlinear models, not merely select a high-entropy observable. Results are reported on synthetic and in-silico networks.

**Transfer hypothesis:** A frontier question must distinguish observational lookup from intervention, include target/value and expected cost, and pass an independent permission gate if the intervention changes the world.

**Falsifier:** A policy maximizing observational surprises chooses a confounded edge that cannot distinguish rival causal mechanisms.

## S08 — NARS self and internal operations (2018)

**Reference:** Pei Wang et al., *Self in NARS, an AGI System*, *Frontiers in Robotics and AI* (2018). [DOI 10.3389/frobt.2018.00020](https://doi.org/10.3389/frobt.2018.00020/full).

**Paraphrased excerpt:** Self-related information and internal operations have an explicit place in a non-axiomatic reasoner under insufficient knowledge and resources. A proposal to represent software self-knowledge as edges is compatible as a *research analogy*, not automatically the NARS theory's identical mechanism.

**Transfer hypothesis:** D-SAG-1's compiler evidence can be queried as data, while executing a resulting proposal still needs verified methods and a distinct authority path.

**Falsifier:** Internal derived facts are recycled as independent evidence or a claim about a method is treated as proof that it ran.

## S09 — ONA reference evaluation and bridge (implementation source)

**References:** [OpenNARS-for-Applications](https://github.com/opennars/OpenNARS-for-Applications), [evaluation.py](https://github.com/opennars/OpenNARS-for-Applications/blob/master/evaluation.py), [ONA v0.9.3 release](https://github.com/opennars/OpenNARS-for-Applications/releases/tag/v0.9.3) (2025-04-22), and its [MeTTa-NARS declarative example](https://github.com/patham9/nartech_ros/blob/master/plugins/narsplugintests/declarative.metta), [procedural example](https://github.com/patham9/nartech_ros/blob/master/plugins/narsplugintests/procedural.metta).

**Observed from upstream descriptions:** ONA's evaluation.py runs C tests and Narsese/English integration examples. The maintainer documents a MeTTa interface using commands AddBeliefEvent, AddBeliefEternal and AddGoalEvent with a frequency/confidence pair; ONA, bridge and native MeTTa are not thereby shown to implement the same evidence algebra. **No official CARE evaluation suite has been identified in the cited material.**

**Transfer hypothesis:** First capture native ONA and bridge observable traces, then normalize identity, source roots, operation, time, output alternatives and costs. Compare *observable cases* before attempting a cross-engine improvement claim.

**Falsifier:** Lost event time, modified NARS truth values, missing alternative bindings, silent duplicate evidence, or differing input budget.

## S10 — Hyperon OpenPsi (implementation source)

**Reference:** [iCog-Labs-Dev/hyperon-openpsi](https://github.com/iCog-Labs-Dev/hyperon-openpsi), [test suite](https://github.com/iCog-Labs-Dev/hyperon-openpsi/tree/main/test), [curious-agent use case](https://github.com/iCog-Labs-Dev/hyperon-openpsi/tree/main/use-cases/curious-agent).

**Observed from upstream README:** Experimental MeTTa modules include demand updater, feeling updater, modulator updater, perception monitor and hill-climbing action planner. It is an experiment, not proof of a final general motivation theory.

**Transfer hypothesis:** Run demand arbitration externally as a typed input to a pure candidate ranking policy. Do not put motivations into evidence or imply action permits.

**Falsifier:** Same ranked demand maps to an unrelated goal, iteration-order tie changes the outcome, or the chosen goal never affects real action-selection traces.

## S11 — OpenPsi demand arbitration regression (2026 upstream PR, not merged evidence)

**Reference:** [hyperon-openpsi PR #117](https://github.com/iCog-Labs-Dev/hyperon-openpsi/pull/117), opened 2026-09-04, **OPEN when checked 2026-10-12**.

**Paraphrased excerpt from PR description, not established merge:** Multiple demands in the normal interval share satisfaction score 0.84, making implicit tie decisions order-sensitive. The current selector previously mapped affiliation, competence and energy to the same mission goal. The PR proposes explicit deterministic demand order and separate goal destinations.

**Transfer hypothesis:** Include ties, reversed row order and meaningful goal-to-action mapping in the D-SAG-2 evaluation even if the upstream fix remains unmerged.

**Falsifier:** Same semantics, different iteration order produces a different winner; or changing dominant demand leaves goal/action unchanged.

## S12 — Rubikon and implementation intentions (psychology, not an engine)

**References:** [Prior literature review](2026-10_rubicon_gollwitzer_volition_source_excerpts.md), [D-RV mapping](../.claude/plans/rubicon-volition-v1.md). Consult the original Heckhausen/Gollwitzer works linked there before attributing quantitative results.

**Paraphrased excerpt:** Goal desirability and feasibility evaluation precede goal commitment; later planning can specify cue-triggered behavior, while evidence of impossibility can justify reconsideration or disengagement. These are psychological constructs, not bits in a NARS belief state.

**Transfer hypothesis:** Distinguish predecisional investigation, revocable goal commitment, plan-ready cue matching, and a second *execution authorization* gate. Kanban records the lifecycle rather than replacing those distinctions.

**Falsifier:** A committed-but-unauthorized goal executes; a revoked cue fires; genuine impossibility never triggers disengagement; harmless contradictory evidence destroys every commitment.

## S13 — Modern PLN vs ONA (separate oracles)

**References:** [trueagi-io/PLN](https://github.com/trueagi-io/PLN), [executable library](https://github.com/trueagi-io/PLN/blob/main/lib_pln.metta), [test harness](https://github.com/trueagi-io/PLN/blob/main/test.sh), [PeTTa](https://github.com/trueagi-io/PeTTa), [#1497 source map](2026-10_opencog_pln_shannon_attention_research_map.md).

**Observed:** Modern PLN ships MeTTa truth-rule and test surfaces; ONA is an independently designed C NARS implementation. Neither one's results are a substitute for the other one's inference semantics.

**Transfer hypothesis:** First vary scheduler with **one fixed inference engine**, then freeze scheduler and vary engines. If a rule is inequivalent, record a mismatch instead of silently coercing it.

**Falsifier:** Scores change because inference rules or candidate population changed rather than attention policy.

## Source-to-experiment crosswalk

| Evidence class | How to use it | How *not* to use it |
| --- | --- | --- |
| arXiv/publisher paper | Formulate conjecture, baseline and explicit limitations | Claim implemented/hot-path parity or arbitrary speedups |
| GitHub README/release | Select actual oracle and CLI/test entrypoints | Claim two runtimes share truth laws |
| GitHub open PR | Construct an adversarial fixture | Treat unmerged fixes as shipped |
| Our merged #1494/#1509 | Ground existing test-local mechanisms and one-hop SAG | Assume a mature production motivational engine exists |
| Our proposed D-SAG-2 | Run pre-registered falsifiers | Claim discoveries merely because graph patterns are visually fractal |

### Citation/replication rules

Capture exact commit SHAs for each external runtime and interpreter/submodule; install only after license/compatibility review. Verify tests on pinned snapshots with stdout/exit code and provenance before any publication-quality comparison. Follow source-specific licenses and quote limits. Preserve contradictory results. Every measurement must identify an input fixture, version, budget and independent oracle.
