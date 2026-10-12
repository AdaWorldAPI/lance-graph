# Research excerpts: software awareness, ONA, evidence revision, attention, metagraph folds and Rubicon

**2026-10-12. Status: annotated primary-source excerpts and engineering transfer, not a proof of self-awareness.**
The repository's .arxiv directory is a literature shelf: some referenced papers are journal articles, not arXiv submissions. "Excerpt" here is a section-faithful concise **paraphrase**, not a reproduced text passage. Where chapter/section or abstract is named, verify against the linked edition before treating it as a mathematical implementation contract. This file does not authorize any CE64, NARS, ISA, OGAR, or runtime change.

## 1. Wang, Li & Hammer (2018): *Self in NARS, an AGI System*

Source: https://doi.org/10.3389/frobt.2018.00020 (open-access full text: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00020/full).
**Read:** §§2.1, 3.1–3.4, 4.2, 5.

**Source-faithful extract:** The system reasons under insufficient knowledge and resources. Built-in operations anchor the notion of SELF; internal experience is treated as sensory input with a different origin. Deliberate *mental operations* supplement the fixed automatic inference/attention routines without freely rewriting them. The authors describe this as a preliminary implementation whose self-knowledge is incomplete and fallible.

**Engineering transfer (hypothesis):** Treat compiler judgments, code harvest, execution traces, budget appraisals and independent test receipts as separate internal observations over existing graph data. Model executable capabilities as methods and evidence of preconditions/outcomes. Preserve the automatic executor and OGAR action authority rather than allowing introspection to modify either.

**Falsifier:** With the same source and outcomes, does removal of observation-to-revision cause a different or worse subsequent decision? An instrumented graph with no causal feedback fails the stronger adaptation claim.

**Do not claim:** the article proves machine consciousness, ONA's implementation has complete NAL-9 introspection, or a graph of code alone constitutes subjective awareness.

## 2. ONA: OpenNARS for Applications (software reference, not a paper)

Sources: https://github.com/opennars/OpenNARS-for-Applications and https://github.com/opennars/OpenNARS-for-Applications/wiki ; contrast https://github.com/opennars/opennars.

**Engineering extract:** ONA is a compact NARS implementation focusing on application reasoning, temporal implications, anticipation, operations and feedback. Its implementation may differ materially from the older Java/OpenNARS research code and the paper's introspective mental operations.

**Probe, not assumed fact:** Pin a concrete ONA commit. Locate source and tests for operation registration, anticipation, success/failure feedback, decision priorities and any internal observation hooks. Produce a table: "exists in ONA", "exists in research OpenNARS", "paper-only", "unverified". Run an executable fixture against pinned ONA before mapping any operator.

**Falsifier:** A proposed ONA feature with no source entry point is excluded from the implementation mapping. External operation feedback is not automatically internal introspection.

## 3. Jøsang (2007): *Cumulative and Averaging Fission of Beliefs*

Source: https://arxiv.org/abs/0712.1182 (Audun Jøsang, 2007, abstract and operator definitions).

**Source-faithful extract:** Cumulative and averaging belief fusion model different evidence assumptions. Fission attempts to recover the residual opinion after removing one contributor from a fused result. This provides a separate mathematical control for provenance-aware ablation.

**Engineering transfer:** Use explicit evidence-contributor keys. Compute cumulative versus averaging controls only under their own assumptions; use an independent source witness when removing a contribution. Compare removal with rerunning only the remaining observations.

**Falsifier:** If a specific contributor's identity is lost, exact inverse fission must return insufficiency rather than a fabricated truth value. Never substitute subjective-logic fission for NARS revision silently.

## 4. Gollwitzer (1999): *Implementation Intentions: Strong Effects of Simple Plans*

Source: https://doi.org/10.1037/0003-066X.54.7.493 ; author/publication page: https://www.socmot.uni-konstanz.de/publications/implementation-intentions-strong-effects-simple-plans (American Psychologist 54, 493–503).

**Source-faithful extract:** Implementation intentions link a specific anticipated situational cue to a goal-directed response. Their empirical subject is human goal execution, not a software permission model.

**Engineering transfer:** A Rubicon commitment can be represented as a versioned, revocable if-cue-then-propose policy. Deliberation and action selection remain distinct, and an action cannot skip the independent OGAR permission check.

**Falsifier:** Present a near-miss cue, expired/revoked intent, stale source revision, and Undetermined role. A commitment that executes anyway violates the software contract.

## 5. Goertzel (2020): *Folding and Unfolding on Metagraphs*

Source: https://arxiv.org/abs/2012.01759 (abstract; full mathematical definitions require checking PDF/text).

**Source-faithful extract:** Directed typed metagraphs admit multiple recursion/fold/unfold formulations, including catamorphisms, anamorphisms, histomorphisms and chronomorphisms. A metagraph edge may point to another edge and carry ordered typed ports; not every row-set reduction preserves these structures.

**Engineering transfer:** Compare finite typed metagraph fixtures against independent scalar reference, preserving edge-to-edge references, binding multiplicity, nested results and a pinned read snapshot. Quack SUM equivalence is not evidence for general categorical fold equivalence.

**Falsifier:** A hyperedge or nondeterministic rewrite whose multiplicity or nested order is lost must be marked unsupported, not called translated.

## 6. Goertzel et al. (2023): *OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond*

Source: https://arxiv.org/abs/2310.18318 (architectural overview; not a benchmark of AGI).

**Source-faithful extract:** Hyperon redesigns prior OpenCog ideas around metagraph representation, type systems, patterns and reflective future aspirations. The text discusses possibilities for reflective self-modification and improvement, not a validated production equivalent of our execution substrate.

**Engineering transfer:** Separate structural fold/unfold semantics from runnable MeTTa pattern matching, atom binding/rewrite, side effects, and probabilistic reasoning. #1498's explicit four-result nondeterminism fixture is a cardinality gate, not an optional nuance.

**Falsifier:** Run a pinned Hyperon nondeterministic fixture; if four possibilities collapse to two row-mask matches, fail parity and record the missing operation.

## 7. Shannon/EIG and Bayesian control: a measurement discipline

Sources: Kullback & Leibler (1951), *On Information and Sufficiency*, https://doi.org/10.1214/aoms/1177729694 ; companion internal study [#1497 attention research](https://github.com/AdaWorldAPI/lance-graph/pull/1497).

**Source-faithful extract:** KL divergence compares explicitly defined probability distributions. Shannon entropy and expected information gain require a named random variable, posterior/likelihood model and target outcomes. NARS frequency/confidence, PLN strength and software code frequency are not automatically calibrated probabilities.

**Engineering transfer:** Specify the target distribution before computing entropy. Compare FIFO, confidence priority, high entropy, EIG and Moore/Staunen at equal rule/candidate budgets with correlated evidence and irrelevant high-entropy distractors.

**Falsifier:** A scheduler that spends more work on irrelevant entropy but yields fewer independent oracle-supported discoveries loses, even if its raw entropy score is higher.

## 8. Ruff as an independent *software observation* source

Primary implementation references (source of current code behavior, not peer-reviewed research):
- https://github.com/AdaWorldAPI/ruff/blob/main/crates/ruff_spo_triplet/src/ir.rs — ModelGraph, common per-language IR.
- https://github.com/AdaWorldAPI/ruff/blob/main/crates/ruff_spo_triplet/src/reassemble.rs — explicitly restricted to C++ machine-plane projected form.
- https://github.com/AdaWorldAPI/ruff/pull/117 — ordered Ruby MethodOre; reported 4,459 events vs 850 set entries in tested OpenProject corpus, not an independent rerun in this study.
- https://github.com/AdaWorldAPI/ruff/pull/118 — free-function signatures; C linkage stays frontend-local; free function dispatch not yet projected.
- https://github.com/AdaWorldAPI/ruff/pull/120 — Odoo parent/extension inherit_id resolution; unresolved XPath remains unresolved.
- https://github.com/AdaWorldAPI/ruff/blob/main/.claude/OGAR-POLYGLOT-AST-INTEGRATION.md — design RFC, not proof of complete bidirectional AST translation.

**Transfer:** Ruff is an independent static observation oracle for supported constructs; runtime tests remain a separate behavioral oracle. Use exactly matching source revisions, trace missing facts as residuals and record partial extraction as unknown.

**Falsifier:** If identical unordered reads/writes/calls mask different order-dependent outcomes, static equivalence fails. If a source call graph predicate was never emitted, SAG cannot use its absence as proof there are no calls.

## 9. Experimental synthesis and non-claims

Proposed experiment: harvest software revision A with Ruff; issue SAG forecast about a controlled revision B; commit forecast and its evidence; change source; blind reharvest; score agreement; update evidence-based priority; run an identical no-feedback twin. Distinguish **instrumentation**, **operational introspection**, **anticipation**, **control**, and **validated adaptive improvement**. These are proposed engineering test tiers, not official ONA awareness levels or evidence of phenomenal experience.

Related merged sources: [#1496 CE64/NAL](https://github.com/AdaWorldAPI/lance-graph/pull/1496), [#1497 PLN/Shannon](https://github.com/AdaWorldAPI/lance-graph/pull/1497), [#1498 Hyperon](https://github.com/AdaWorldAPI/lance-graph/pull/1498), [#1499 NARS self](https://github.com/AdaWorldAPI/lance-graph/pull/1499), [#1500 Rubicon](https://github.com/AdaWorldAPI/lance-graph/pull/1500), and [#1509 SAG](https://github.com/AdaWorldAPI/lance-graph/pull/1509).

## 10. Reproduction and source restrictions

Pin runtime commits and package/tool versions before claiming parity. Cite a versioned test artifact, fixture, input/output and exact operator mapping. Treat peer-reviewed empirical effects, mathematical proofs, and project-specific engineering hypotheses as **different epistemic tiers**. No paper here licenses adding a new byte interpretation, operator, or action privilege without an independently verified insufficiency and explicit owner decision.
