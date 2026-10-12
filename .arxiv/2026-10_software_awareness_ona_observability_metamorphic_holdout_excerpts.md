# Supplemental research excerpts: ONA observability, adaptive holdouts and metamorphic software awareness

**Date:** 2026-10-12. **Scope:** source-grounded *paraphrased excerpts*, implementation constraints and red tests expanding D-SAG-2. **Not implementation evidence.**
**Companion:** [operational falsification addendum](../.claude/knowledge/sag-v2-observability-causality-and-evaluation-addendum.md).
The `.arxiv` folder is a curated literature shelf, not a claim that every item is published on arXiv. Links to journal/conference PDFs are separately identified. No long quotations are reproduced.

## 1. Hammer & Lofthouse (2020), *'OpenNARS for Applications': Architecture and Control*

Primary conference PDF: https://agi-conf.org/2020/wp-content/uploads/2020/06/AGI-20_paper_52.pdf (AGI 2020, 11 pages, printed pp. 1–11).
**Location:** §2 Architecture (printed pp. 2–4); §3 Data Structures (printed pp. 5–6).

**Research excerpt, paraphrase:** ONA's application-oriented architecture replaces aspects of older OpenNARS control with an event-selected, bounded priority process. A cycling event selects relevant concepts, which are checked for evidential overlap before inference. Semantic inference and sensorimotor/temporal procedural inference are separated. Every event has a term, NAL truth value, evidence stamp, occurrence time, and priority. In §3 an implication of the form `(a, operation) ⇒ b` can predict an outcome, and its evidence changes when the outcome is or is not observed. The authors describe *Assumption of Failure*, where initial negative evidence can be outweighed by later positive evidence, rather than using a strict anticipation deadline.

**Specific transfer:** SAG can define a counterfactual prediction receipt and a later observation receipt, but *must not* transfer assumption-of-failure to an open-world/static harvester wholesale. Absence is meaningful only for a complete observable predicate on the expected snapshot and interval. The narrower SAG model permits pending, incomplete and unobservable predictions.

**Probes:** (a) same event stamp encountered via multiple derivations must not become independent evidence; (b) deliberately disabled Ruff predicate does not count as failed prediction; (c) positive result arriving late must not irreversibly corrupt another snapshot; (d) event priorities alter which facts are inspected, never authorization or the truth of compiler verdicts.

**Limit:** event-handling, capacity and evidential overlap are described by the paper; this does not verify a specific API, NAL-9 self-observation implementation, or CE64 byte layout in ONA.

## 2. ONA issue #286 (2024), *Do-nothing operator* as an explicit research ambiguity

Primary implementation discussion: https://github.com/opennars/OpenNARS-for-Applications/issues/286
**Location:** issue description discussing `Concept::precondition_beliefs`, `Decision_Anticipate` and the special operation index 0.

**Excerpt, paraphrase:** An issue contrasts including or excluding index 0 when traversing operation precondition beliefs. Its author asks whether do-nothing and ordinary operation slots should be distinguished. This is a question on the issue page, **not** proof that the proposed interpretation was accepted or implemented.

**Transfer:** distinguish SAG's explicit `NoOp` intervention from a read-only `Probe` and an OGAR-gated `ExternalMutate`. A no-op provides a genuine causal baseline for spontaneous data drift, instrumentation overhead and false-positive prediction of change. Do not infer that ONA's operation code 0 is equivalent to SAG's action or source mutation.

**Red test:** a no-op A→B reharvest incorrectly creates a semantic dependency delta or a verified action effect.

## 3. Dwork et al. (2015), *Generalization in Adaptive Data Analysis and Holdout Reuse*

Primary arXiv: https://arxiv.org/abs/1506.02629; related https://arxiv.org/abs/1411.2664.
**Location:** abstracts and holdout-reuse framing; mathematical guarantees are specific to the proposed algorithms, not automatically available to SAG.

**Research excerpt, paraphrase:** When an analyst repeatedly adapts hypotheses after inspecting evaluation outcomes, even a held-out dataset can be overfit through reuse. The work studies how to answer adaptively selected questions while preserving generalization; simply calling a test "held-out" is insufficient once its feedback has guided development.

**Transfer:** split the D-SAG-2 corpus by repository and edit family into development, selection-validation and truly protected final evaluation. Log how many times an oracle is consulted. Freeze the policy and hypothesis set before the protected test and treat an already disclosed fixture as development material.

**Falsifier:** measured adaptation disappears on unseen edit families; the method looks strong only after being tuned to a repeatedly queried mutation corpus.

**Non-claim:** SAG is not implementing the paper's formally private or statistically valid holdout mechanisms; it adopts the leakage hazard and conservative evaluation discipline.

## 4. Li et al. (2024), *Metamorphic Relation Generation: State of the Art and Visions for Future Research*

Primary arXiv: https://arxiv.org/abs/2406.05397.
**Location:** abstract and systematic-review framing.

**Research excerpt, paraphrase:** Metamorphic testing addresses situations where a fully specified output oracle is hard to construct, by asserting an expected relationship between inputs and outputs after a known transformation. Metamorphic relations themselves need generation, validation and applicability checks.

**Transfer:** record the predicate and known applicability conditions of source transformations (whitespace, declaration order, safe renaming, extension-view locator edits, full vs incremental harvest, A→B→A' revert). Neither a successful identical checksum nor a single matching SUM is proof of broad semantic equivalence.

**Falsifier:** SAG considers reordered Ruby mutation/raise events equivalent merely because an unordered field-fact set matches.

## 5. Duque-Torres et al. (2023), *Bug or not Bug? Analysing the Reasons Behind Metamorphic Relation Violations*

Primary arXiv: https://arxiv.org/abs/2305.09640; related pipeline proposal https://arxiv.org/abs/2310.00338.
**Location:** abstracts and proof-of-concept description.

**Research excerpt, paraphrase:** A violated metamorphic relation is suspicious, but may reflect violated applicability preconditions rather than a defect in the program. Their work investigates how to refine constraints and explain these violations; it does not claim all MR violations are bugs.

**Transfer:** when Ruff reassembly drops unsupported core-7 fields or when C free function calls are deliberately absent from SPO, parity failure may indicate an unsupported projection, not a miscompilation. Label as `Unsupported`, `Incomplete`, or `Refuted` with separately justified semantics.

**Falsifier:** the validation runner files a "fixed" runtime bug by inventing a triple that the source frontend never promised to emit.

## 6. Operational synthesis for lance-graph SAG

A deliberately small sequence, inspired but not equivalent to ONA:

1. Pin immutable source A and metadata describing what Ruff can observe.
2. Select one source-graph proposition and an intervention from a declared candidate pool.
3. Commit a falsifiable forecast *before* observing B, including expected result, scoring universe and deadline.
4. Use an independent run to observe B and attach the collector completeness receipt.
5. Count independent evidence sources, not duplicated derivation paths or repeated hypothesis layers.
6. Revise only hypotheses when observations are admissible; keep compiler authority and OGAR policy untouched.
7. Compare against no-feedback, shuffled-feedback and no-op controls on protected, unseen edits.

A positive adaptation claim needs a **difference in next decision attributable to receipt feedback**, plus a predeclared improvement measure on held-out tests. A negative result or explicit `NotObservable` is a correct output.

This is an *engineering experiment*, not an official consciousness ladder, a port of ONA, a proof of AGI, or an authorization to rewrite source code.

## 7. Source/version ledger still to pin before execution

| Component | Source | Status |
|---|---|---|
| SAG base | https://github.com/AdaWorldAPI/lance-graph/pull/1509 | merge `42fa2a347cda22eaaa327bba514f2029cc9daa82` pinned |
| Current documentation | https://github.com/AdaWorldAPI/lance-graph/pull/1510 | branch head subject to PR updates |
| Ruff fork | https://github.com/AdaWorldAPI/ruff | exact test/run commit **not pinned yet** |
| ONA | https://github.com/opennars/OpenNARS-for-Applications | executable commit **not pinned yet** |
| Research OpenNARS | https://github.com/opennars/opennars | separate implementation; **not pinned** |
| Hyperon | https://github.com/trueagi-io/hyperon-experimental | unrelated parity track, **not pinned** |

No tests, benchmarks or complete ONA/Hyperon parity runs were performed to produce these research excerpts. Preserve that state when reporting this document.
