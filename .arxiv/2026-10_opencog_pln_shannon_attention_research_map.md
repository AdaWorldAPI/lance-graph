# OpenCog / PLN x Shannon attention: research map and falsification questions

**Date:** 2026-10-12
**Status:** RESEARCH PROPOSAL / source inventory, NOT a result or authorization to implement.
**Read by:** work on D-HL, D-NALP, D-RPF, CE64/Staunen, Moore activation, Alpha overlays, inference-scheduler comparisons.
**Local anchor:** [CE64 plateau](../.claude/knowledge/ce64-reasoning-substrate-plateau.md), [D-HL plan](../.claude/plans/ce64-hypothesis-layers-v1.md), [D-NALP plan](../.claude/plans/nal-polyfill-v1.md), [CE64 research map](2026-10_ce64_hypothesis_layers_research_map.md).
**Not the same "entropy":** [V3 ENTROPY-MILESTONES](../.claude/v3/ENTROPY-MILESTONES.md) concerns reducing *duplicate representations* N to 1. This note concerns Shannon uncertainty and expected information gain. Do not conflate them.

## Research question

> At a fixed evidence corpus, graph, inference rule set and compute budget, does an information-gain scheduler discover more *useful, oracle-supported* relationships per unit of work than confidence-priority, FIFO, or the existing Moore/Staunen attention strategy? Under which adversarial conditions is entropy actively worse?

Separate three experimental axes. **Inference semantics** (PLN vs pinned NARS vs CE64), **selection policy** (FIFO / PLN confidence / raw Shannon / expected information gain / Staunen), and **physical execution** (foreign interpreter / scalar oracle / masked SoA folds). Never attribute an improvement in one axis to another. The existing byte-level LE/ALU/ISA register contract remains fixed; ClassView determines semantic readings. There is no mandate to give CE64 bytes PLN semantics.

## External source inventory (verified repo existence; source revision snapshot, 2026-10-12)

All SHAs below are *retrieval snapshots*, not proof of having executed upstream tests. Pin any fixture and toolchain before a future experiment; do not benchmark floating main/master. Repository licenses must be checked per copied file, not inferred from a project name.

| Source | Snapshot SHA | What to inspect | Scope / caveat |
|---|---|---|---|
| [trueagi-io/PLN](https://github.com/trueagi-io/PLN) ([README](https://github.com/trueagi-io/PLN/blob/main/README.md), [lib_pln.metta](https://github.com/trueagi-io/PLN/blob/main/lib_pln.metta), [examples](https://github.com/trueagi-io/PLN/tree/main/examples), [test.sh](https://github.com/trueagi-io/PLN/blob/main/test.sh), [LICENSE](https://github.com/trueagi-io/PLN/blob/main/LICENSE)) | `4405956947c4b53c7ff01bd565aa3b114bc970a1` | Executable MeTTa rules; `Truth_Revision`, `Truth_Deduction`, `Truth_Induction`, `Truth_Abduction`; EvidenceID, horizon/queue limits, confidence priority in README | Primary contemporary **PLN oracle candidate**; MIT per repository LICENSE. Runtime relies on PeTTa; rule behavior must be captured from execution, not comments. |
| [trueagi-io/PeTTa](https://github.com/trueagi-io/PeTTa) | **UNPINNED** | Interpreter/runtime for PLN's provided run instructions | Dependency and reproducibility question; pin/verify before first oracle run. |
| [trueagi-io/pln-experimental](https://github.com/trueagi-io/pln-experimental) ([docs](https://github.com/trueagi-io/pln-experimental/tree/main/docs), [experimental](https://github.com/trueagi-io/pln-experimental/tree/main/experimental)) | `50f9525ff734fde95a3d91e528e1101f1a61ca75` | nuPLN foundations; chaining and truth-formula experiments | Research material, not interchangeable with modern PLN's executable behavior. |
| [opencog/attention](https://github.com/opencog/attention) ([README](https://github.com/opencog/attention/blob/master/README.md), [docs](https://github.com/opencog/attention/tree/master/doc), [LICENSE](https://github.com/opencog/attention/blob/master/LICENSE)) | `96107652e1cf72747e69fcd2a20ffb6edba562d2` | Classic **ECAN** economic attention allocation, salience diffusion, attention currency and decay | Historical attention comparator, **not** a documented Shannon-EIG implementation. AGPL + extra linking permissions per LICENSE; do not copy into repo without license review. |
| [opencog/atomspace](https://github.com/opencog/atomspace) ([README](https://github.com/opencog/atomspace/blob/master/README.md), [AtomSpace spec](https://wiki.opencog.org/w/AtomSpace), [LICENSE](https://github.com/opencog/atomspace/blob/master/LICENSE)) | `c8d633bf272b838c2132c21b7c8ddf169a18dd22` | Hypergraph/query model; Atoms vs mutable Values; Frames/ChangeSets and branch/merge | Frames are an **analogy**, not proven equivalent to Alpha sparse per-field deltas. AGPL with stated linking exception; license review for code reuse. |
| [opencog/pln](https://github.com/opencog/pln) | `a2e1354aa37f67952218b42e15686f8dc1629418` | Classic Scheme/URE rules; historical comparison | AtomSpace README calls this line dead/abandoned. Do not substitute for trueagi-io/PLN without flagging version differences. |
| [trueagi-io/hyperon-experimental](https://github.com/trueagi-io/hyperon-experimental) ([MeTTa spec](https://wiki.opencog.org/w/File:MeTTa_Specification.pdf), [docs](https://trueagi-io.github.io/hyperon-experimental/)) | `3f76dc460da6961f57f69f6c3e550c59c74ada83` | Modern Rust/MeTTa atoms, spaces, pattern matching, and possible second interpreter | Optional projection/representation comparison. README describes experimental development; not needed to get the first entropy falsifier. |
| [PLN book](https://link.springer.com/book/10.1007/978-0-387-76872-4) | publication | Formula definitions, especially Appendix A comparing PLN and NARS | Specification/background; distinguish formulas in a book from an implementation actually run. |

**Crosscheck, not equivalence:** modern `lib_pln.metta` computes a heuristic weighted revision through `w = c / (1-c)` and `f = sum(w*f)/sum(w)`; it additionally clamps and floors the confidence relative to inputs. Its deduction includes conditional-probability consistency guards. These are **not** automatically byte-equivalent to NARS revision, nor is PLN `strength` automatically NARS `frequency`. Collect runnable outputs and edge-case behavior before asserting a crosswalk. D-NALP's `GroupCrossPowerSumsI32` is a possible *execution primitive*, not a license to unify the truth laws.

## Questions to close (OPEN, not decided)

| ID | Question to answer from source/probe | Evidence required / closure gate |
|---|---|---|
| OC-Q01 | **Entropy of what?** Is the score defined on predicted answer, binary proposition, distribution of hypotheses, or next observation? Can a NARS frequency or PLN strength be used as calibrated `p` at all? | Declare the random variable and predictive distribution; reliability curve and held-out calibration; refuse an unsupported mapping from u8 F/C into probability. |
| OC-Q02 | **Raw entropy versus information gain:** How is `E[H(before)-H(after)]` estimated without assuming the answer? | Deterministic oracle-backed or explicit predictive-model estimate; leakage test: hidden answer cannot be read by scheduler; include zero-information high-entropy distractors. |
| OC-Q03 | **PLN vs NARS truth law:** Which revision, deduction, induction and abduction operations match and which differ? | Pinned PLN + OpenNARS/ONA outputs on same premise ledger; per-rule numerical diffs and documented refusal/edge cases (c=0/1, unsupported conditionals). |
| OC-Q04 | **Evidence overlap:** What is an independent observation in each engine? How do PLN EvidenceID, NARS Stamp/EvidenceMask, and D-HL-3's per-delta base translate? | Duplicate across layers counts once, independent events in same layer count twice; derivation overlap, truncation and cycle probes; never use layer ID as source ID. |
| OC-Q05 | **What is the PLN scheduler actually prioritizing?** Does documented confidence-priority also consider horizon, recency, belief strength and query relevance? | Read executable queue code; pin comparator, tie-break, truncation semantics and same-budget replay; do not label ECAN as PLN's built-in queue. |
| OC-Q06 | **ECAN control:** Are STI/LTI, attention spreading, forgetting and resource currency reproducible independently of classic AtomSpace? Where is the specific algorithm and its parameterization? | Extract mathematical transition + source line references, a tiny hand-checkable fixture, and per-atom versus batched cost. No presumed entropy behavior. |
| OC-Q07 | **Graph projection:** How do typed atoms and arbitrary-arity hyperedges map to immutable `classid x NodeGuid` plus edge/overlay without losing joins, scopes, or revisable Values? | Typed translation table; reverse-roundtrip and query-answer parity; mark unsupported operations instead of silently approximating. |
| OC-Q08 | **Frames and Alpha:** Do Frame branch/rewrite semantics correspond to copy-on-write hypotheses and `SparseDelta` merged on read, or only resemble them? | Branch/revert/merge fixture, conflict and tombstone semantics, storage amplification, provenance visibility; compare D-HL-7 before claiming parity. |
| OC-Q09 | **Fair compute:** Is the budget counted as reasoning steps, visited edges, mask decode bytes, folds, wall time, or energy? | Report all; cap candidate expansions and oracle calls; fixed seed and equal answer opportunity. A 0.6 ns word-fold is not a complete inference step. |
| OC-Q10 | **Noise/adversaries:** Does high entropy chase unrelated noise, repeated evidence, contradictory clues, or cyclic derivations? | Reuse Staunen habituation/learning-progress controls and D-RPF-P5 duplicate fixtures; include positive signal, zero-information distractors and delayed payoff. |
| OC-Q11 | **Sparse execution:** Can read mask AND projected thinking mask reduce physical decoded rows, and does Moore adjacency help beyond scheduling alone? | D-HL-6 address-coordinate projection and separate scheduler/storage ablations; report decoded bytes and answer parity. |
| OC-Q12 | **Portability/legal:** Which upstream code, datasets, MeTTa runtime and exact licenses are needed for replication? | License/source audit, pinned SHAs + test recipes, minimal independent fixture format; avoid importing AGPL code under an assumed permissive license. |
| OC-Q13 | **Metacognition:** Does a learned scheduler beat its own frozen/no-meta control, rather than merely improve by adding work? | D-HL-8 no-meta control; identical evidence and compute, pre-registered useful-answer metrics. |

## Minimal falsifiable experiment, without new substrate code

1. **Pin provenance.** Snapshot trueagi-io/PLN, its PeTTa runtime and pinned OpenNARS/ONA in D-HL-4; import a *small* common fixture corpus as immutable term IDs, premises, evidence-source IDs and expected answer bindings. Keep original oracle syntax only at adapter boundaries. Document licenses.
2. **Rule-only baseline.** Fix scheduler to deterministic FIFO, fix source budget and premise stream. Compare PLN vs NARS vs CE64 outputs for revision and one chained deduction first. Preserve native truth laws; use calibrated mappings only for aggregate scores, never substitute one engine's F/C for another's in the oracle.
3. **Scheduler-only factorial.** Within **each unchanged rule engine**, compare FIFO, PLN documented confidence priority, naive binary Shannon `H(p)=-p log2 p-(1-p) log2(1-p)`, model-based expected information gain, and existing Staunen/Moore if implementable without ABI changes. A naive `H(f)` arm is explicitly an intentionally potentially-wrong baseline, not "PLN's entropy".
4. **Adversarial tasks.** (a) graph chain with one missing intermediate; (b) duplicate/independent evidence swapped across hypothesis layers; (c) irrelevant high-entropy noise; (d) contradiction later resolved by a new observation; (e) two equal-looking branches with different delayed payoff; (f) path whose needed edges are initially unhydrated.
5. **Execution-only ablation.** For the winning *behaviorally identical* selection policy, compare scalar row iteration with existing resident masks, folds and projected sparse reads; report row decoding and CPU cost separately from changes in the policy.
6. **Pre-register falsifiers and report raw rows** before any runtime integration. No NAL byte, tenant, CE64 bit, ABI, parser, or new IR is authorized by this doc.

### Measurements and required controls

- Primary: unique **oracle-supported** target discoveries at equal expansion/fold budget; time-to-first useful answer and area under useful-answer-versus-budget curve.
- Quality: exact answer set, F/C error **within each engine's semantics**, rank correlation where a meaningful common rank exists, calibration/reliability of any predicted probabilities, duplicate amplification, revisits of settled facts.
- Costs: source rows read, row bytes decoded, candidates generated, inference-rule firings, resident word folds, allocations, peak memory, wall time; do not substitute a kernel nanosecond for end-to-end throughput.
- Audit: `run_id, engine, engine_sha, runtime_sha, rule, scheduler, entropy_variable, seed, fixture_sha, evidence_ids, hypothesis_layer, budget_kind, budget_spent, answers, oracle_agreement, bytes_decoded, duration`. Serialize independent of CE64 layout.
- Controls: fixed source/term identities, independent provenance, frozen inference truth laws inside scheduler trials, deterministic tie-break, no oracle leakage, plain FIFO, confidence queue, no-meta baseline, identical starting knowledge.

### Stop / falsify

- If `p` is not empirically calibrated, **do not interpret H(f) as Shannon uncertainty about reality**; retain only as a scoring heuristic.
- If high-entropy irrelevant noise wins the scheduler priority repeatedly and reduces useful discoveries at the same budget, raw entropy is falsified as a standalone scheduler.
- If gain disappears under fixed truth laws, it was inference-law drift rather than attention.
- If gain disappears after candidate-generation or decode bytes are counted, it was not an execution-efficiency win.
- If source identity overlaps, don't count duplicated evidence as additional confirmation (D-HL-3).
- If a projection does not preserve answer semantics, classify it as a lossy approximation and refuse an equivalence claim.
- If gains vanish against a frozen/no-meta control, do not claim metacognitive improvement.
- Unknown licensing or runtime pin blocks a copied-code or oracle-reproducibility claim; it does not block independent mathematical investigation.

## Loose ends / prerequisites before an implementation PR

- **Source audit:** locate the exact trueagi-io/PLN queue comparator, evidence-overlap algorithm and inference examples; locate ECAN's spreading/decay equations and runnable fixture, not just README. Record source file+commit references.
- **Ontology/hypergraph audit:** atom/link/Value/Frame roundtrip, open arity and hidden assumptions; note which forms the current ClassView can and cannot project.
- **Semantics audit:** same-format premise/event ledger, source dedup and independent confidence calibration; compare PLN's actual clamped revision with NARS's parameterized k, D-NALP, and D-RPF-P5.
- **Estimator choice:** no definition of EIG without a predictive model. Decide predictive target, estimation method, cost of lookahead, and how uncertainty of the estimator is tracked.
- **Experiment sizing:** choose corpus, exact budgets, effect-size threshold, statistical intervals and oracle fixture coverage *before* observing which policy wins. Include a zero-change null.
- **Scheduling fairness:** ECAN, PLN confidence queue and Staunen need faithful source-grounded baselines or a prominent "inspired-by approximation" label.
- **Governance:** conclusions belong in a later board entry with measured/verified/speculative labels. Implementation requires a separate authorization/plan; do not silently repurpose CE64 fields 40..63 (plateau O1..O8), or replace the frozen byte-level ISA contract.

**This map does not claim:** Shannon is implemented in OpenCog/ECAN; OpenCog attention equals entropy; PLN and NARS share truth semantics; AtomSpace Frames implement our sparse deltas; 64 layers have shipped; a performance or AGI result has been measured.
