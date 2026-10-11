# NARS evidence, NAL-1…NAL-9, and metacognition — literature index

**Date:** 2026-10-11  
**Status:** Source distillations plus explicitly labeled lance-graph research hypotheses. No benchmark execution or implementation changes.

## Punchline

> **An exact ALU/ISA register contract enables deterministic execution, but it does not itself certify NARS truth semantics, independent evidential roots, nine NAL capability levels, or metacognitive self-control.**

## Literature and recommended reading order

| Source | Distillation | Focus |
|---|---|---|
| Pei Wang, *A Logical Model of Intelligence* (author-hosted 2026 introduction; **not a new arXiv paper**) | [NAL-1…9 capability ladder](./2026-08_nal1_to_nal9_is_a_capability_ladder_not_nine_cycles.md) | **Extended** analysis of each level, independently falsifiable tests, explicit difference from ISA rungs |
| Robin Nolte et al., arXiv:2503.13467 (2025 preprint; 2026 peer-reviewed version) | [Metacognition review](./2503.13467_metacognition_requires_replayable_self_models_not_just_logs.md) | **Extended** 34-section study: architectures, introspective traces, diagnosis, repair, source vs inference |
| Wang, 2001, *Confidence as Higher-Order Uncertainty* | [Evidence stability](./2001_wang_confidence_is_evidence_stability_not_a_95pct_interval.md) | frequency/confidence mathematics, interval interpretation, dependence and quantization |
| Wang, 2009, *Formalization of Evidence* | [Two-number evidence](./2009_wang_two_numbers_preserve_direction_and_stability_of_evidence.md) | direction vs weight; comparison of uncertainty formalisms |
| Wang, arXiv:1303.1517, *Belief Revision in Probability Theory* | [Revision vs updating](./1303.1517_revision_is_not_bayesian_conditioning.md) | difference between inference propagation and new supporting evidence |
| Gabriel & Wang, arXiv:2604.18873 | [Executable Narsese benchmark](./2604.18873_executable_narsese_needs_runtime_validation_not_label_guessing.md) | 1000 cases, 200 held-out, ONA protocol, avoid unsupported-case score inflation |
| Wang, Li & Hammer, 2018, *Self in NARS, an AGI System* | [Internal experience](./2018_self_in_nars_requires_internal_experience_and_mental_operations.md) | self-models, mental operators, feelings and introspective control |
| Pei Wang, *NAL-9 Implementation* (author-hosted slides; undated) | [NAL-9 executable events](./undated_nal9_mental_operations_are_executable_events_not_a_ninth_op.md) | IEB/GEB, cycles, channel operations, deliberate reflection, open issues |

## Cross-paper synthesis: five independent contracts

1. **ABI/ISA:** bits, lanes, encodings, opcodes, deterministic numeric transforms and versioned register readings.
2. **NARS truth:** the meaning and valid algebra of independent positive/negative evidence, including f/c, revision, deduction and choice.
3. **Provenance:** EventId distinguishes executed events; EvidenceRootId distinguishes independent original evidence; derivation lineage connects the two.
4. **Knowledge and causality:** epistemic status and Pearl interventions are neither implied by F/C nor by execution receipts; intent does not imply Causes.
5. **Metacognition:** retrieve actual process trace -> diagnose discrepancy -> plan lawful correction -> rerun -> evaluate against a no-meta control.

## Highest-priority falsifiers

- **Echo inflation:** two independently executed deductions from one observation must not count as two independent observations.
- **Unknown epistemic state:** missing evidence is not negative evidence; a "not run" verdict differs from an underidentified one.
- **True counterfactual:** removing a cause under overdetermination cannot automatically establish no causal contribution.
- **Trace fidelity:** a generated explanation cannot cite an unexecuted step.
- **NAL-9:** logging one's own operation is not yet evaluating and changing future operations.
- **Benchmark:** 6–9 *reasoning steps* in Gabriel/Wang 2026 do not signify NAL-6 through NAL-9.
- **Register semantics:** a BF16-like exact byte contract does not force a Wang f/c interpretation; the operator must declare the semantic reading.

## Editorial and evidence boundaries

This index covers the **seven textual papers/author treatments plus NAL-9 presentation** that arose in this conversation. Some items are not arXiv submissions; they are in `.arxiv/` to follow the existing repository literature convention, not to imply an arXiv ID. The 2025 Wang monograph has not been fully distilled because the complete book was not verified as available; the author-hosted 2026 introduction is used to ground the nine-level ladder. All experimental actions are proposals; no completed project measurements are claimed.

**Recommended next step:** conduct the NAL-1 truth/provenance probe and the metacognitive logging-vs-diagnosis-vs-repair ablation before extending any register or ISA primitive.
