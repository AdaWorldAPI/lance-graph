# Inverted temporal ablation × ONA × OpenNARS × Hyperon: primary-source research map

**Date:** 2026-10-12. **Status:** annotated reference map; *paraphrased excerpts*, not a claim that any upstream engine has passed our benchmark. Companion RFC: [inverted temporal ablation](../.claude/plans/inverted-temporal-ablation-reasoner-oracles-v1.md). Builds on [existing primary excerpts for NARS/fission](2026-10_nars_self_epistemic_needs_evidence_primary_excerpts.md), [actual causality and contingencies](1608.07879_actual_causality_needs_contingencies_not_just_deletion.md), and [self in NARS](2018_self_in_nars_requires_internal_experience_and_mental_operations.md).

## Why this collection

A nine-dot *epistemic* probe differs from ordinary ablation by asking what happens when one **rule, interpretation, or shared epistemic assumption**, not only one observation, is suppressed. Each source below either motivates a testable distinction or acts as an independent oracle. Papers do not establish performance or novelty of the proposed lance-graph composition.

### R1. Jøsang (2007), Cumulative and Averaging Fission of Beliefs

- Primary: [arXiv:0712.1182](https://arxiv.org/abs/0712.1182), abstract, fusion and fission definitions; [PDF](https://arxiv.org/pdf/0712.1182).
- **Paraphrased excerpt:** Cumulative fusion models independent evidence; averaging fusion addresses dependent opinions; fission attempts to remove a known contributor from a fused opinion under that particular fusion model.
- **Do not conflate:** Subjective Logic opinions and NARS truth values have different semantics. Compressed u8 CE64 F/C alone cannot reconstruct arbitrary removed support; require contributor IDs, dependence relation, weights and original revision order, or refuse exact inverse.
- **Testable translation:** Independent A+B -> remove A -> only B remains (within stated rounding); duplicated A under aliases -> delete both references to root A; unknown overlap -> `Coverage=Partial/Unsupported`. Compare true recompute to putative algebraic inverse.

### R2. Wang, Li, Hammer (2018), Self in NARS, an AGI System

- Primary: [Frontiers Robotics and AI DOI:10.3389/frobt.2018.00020](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00020/full), especially §§2–3 on truth, evidence, internal experience, control.
- **Paraphrased excerpt:** Reasoning relies on finite resources and revisable experiential evidence; the system can treat some of its internal processing as experience. Distinguish reasoning about derivation from a claim of phenomenal consciousness.
- **Testable translation:** Track three IDs separately: independent external observation root, derivation/procedure trace, and attention-selection trace. Changing attention may change emitted results without changing logical entailment. A derivation trace alone cannot make a new independent NARS observation.
- **Falsifier:** Give two conclusions with equal final F/C but distinct provenance. Explanation must distinguish them while confidence stays unchanged.

### R3. Hammer & Lofthouse (2020), OpenNARS for Applications: Architecture and Control

- Primary artifact: [official ONA repository README and references](https://github.com/opennars/OpenNARS-for-Applications); upstream paper linked there (AGI 2020 conference contribution); [official ONA wiki](https://github.com/opennars/OpenNARS-for-Applications/wiki).
- **Paraphrased excerpt from maintainer README:** ONA is a new C implementation, not an OpenNARS branch, designed for application-oriented event handling, procedure learning and resource-based control. The maintainers explicitly acknowledge conceptual differences from research OpenNARS.
- **Consequences:** Do not silently assume identical revision rules, stamp representation, command syntax or derivation export between the two. Begin with upstream CLI documented `./NAR shell`, example `*.nal` inputs, `python3 evaluation.py`; pin SHA and config and inspect actual output before normalizing.
- **Testable translation:** Same temporal-event fixture, with and without a root observation, measure subsequent temporal implications, anticipation, learned operation and chosen action at equal time/budget. Separately vary priority or memory budget to diagnose resource-path sensitivity.
- **Unverified:** Upstream exposure of exhaustive proof ancestry, a reversible belief deletion API, and deterministic output under all scheduling modes.

### R4. Hammer, Lofthouse & Wang (2016), The OpenNARS Implementation of the Non-Axiomatic Reasoning System

- Primary repository: [opennars/opennars](https://github.com/opennars/opennars); maintainer-linked publication is in the ONA README; [OpenNARS stamp discussion](https://github.com/opennars/opennars/wiki/Stamp).
- **Paraphrased excerpt / technical idea:** A stamp records creation/occurrence context and evidential-base identifiers; this informs revision/choice and overlapping-evidence checks. Resource-bounded bases can truncate.
- **Testable translation:** Compare (i) independent evidence, (ii) duplicated evidence with a shared root, (iii) two paths with a hidden common ancestor, (iv) temporal vs eternal conclusions. Stamps may catch overlap, but *do not presume* they form a lossless provenance DAG.
- **Refusal:** When a stamp drops ancestry under capacity limits, report partial provenance; do not claim complete causal shadow or exact fission.

### R5. Goertzel et al. (2023), OpenCog Hyperon: A Framework for AGI at the Human Level and Beyond

- Primary: [arXiv:2310.18318](https://arxiv.org/abs/2310.18318), architecture §§3.1–3.2; [PDF](https://arxiv.org/pdf/2310.18318); [maintained experimental MeTTa implementation](https://github.com/trueagi-io/hyperon-experimental).
- **Paraphrased excerpt:** Hyperon represents cognitive processes and rewrite programs in a metagraph/Atomspace; making rules addressable data offers a route toward reflective modification. This is architectural motivation, *not evidence of achieved AGI or an ablation performance result*.
- **Testable translation:** Use two named MeTTa rewrite rules R1/R2 deriving C through distinct branches. Remove R1 from a cloned world, recompute and retain evidence atoms. Then disable a shared method/assumption upstream of both. Compare derived atoms and explanations, not NARS confidence fields.
- **Open question:** Which MeTTa runtime semantics, rewrite ordering and proof export support reproducible source-to-result attribution? Pin the version and distinguish pattern matching, nondeterminism and selection from entailment.

### R6. Chockler (2016), Causality and Responsibility for Formal Verification and Beyond

- Primary: [arXiv:1608.07879](https://arxiv.org/abs/1608.07879); [local detailed harvest](1608.07879_actual_causality_needs_contingencies_not_just_deletion.md).
- **Paraphrased excerpt:** Actual-cause analyses under Halpern–Pearl conditions admit contingencies. A contributor can be causally relevant even if a one-at-a-time deletion does not flip the outcome because of alternative support. Responsibility additionally considers how much context must change to make the contributor decisive.
- **Testable translation:** With Y=A OR B, test deleting A alone, then deleting A under the explicit contingency B=off. Record the context and resulting verdict; never equate a deletion's no-flip result with 'not causal'.
- **Boundary:** Conditional logical pivotality is not a claim of causal identification until a valid SCM/intervention model and assumptions are declared. Moral responsibility/intent are not inferred.

### R7. Differential Dataflow (McSherry et al.), incremental signed-update computation

- Implementation/doc: [TimelyDataflow/differential-dataflow](https://github.com/TimelyDataflow/differential-dataflow), [crate documentation](https://docs.rs/differential-dataflow/latest/differential_dataflow/), especially Collections and iteration.
- **Paraphrased implementation note:** Collection updates add and remove weighted records. Existing operators maintain derived results incrementally over updates; recursive relations can be iterated to fixed points.
- **Testable translation:** Create materialized support tuples and recursive reachability. Delete evidence and rule tuples, feed signed updates, compare stabilized closure and multiplicity to a *fresh recomputation*. Include cycles, diamond-shaped shared ancestors, and fan-out.
- **Boundary:** Differential Dataflow is a computation oracle, not NARS, OpenCog, or a causal model. Bench costs include arrangement/index build and resident memory.

### R8. Soufflé provenance and explanation

- Primary documentation: [Soufflé Provenance](https://souffle-lang.github.io/provenance), including the `-t explain` command and explanation trees.
- **Paraphrased excerpt:** Soufflé can explain derived tuples using reconstructed proof trees of minimal height, and support interactive explanations for failures. Its provenance is lazy, while stored annotations facilitate explanation.
- **Testable translation:** Query support for C and inspect branches, rule IDs and premises. Multiple derivations need complete enumeration or an additional support-count oracle; a *single shortest proof* is insufficient to establish that deleting one branch erases C.

### R9. Green, Karvounarakis & Tannen (2007), Provenance Semirings

- Primary: [DBLP bibliographic entry](https://dblp.org/rec/conf/pods/GreenKT07.html), [author PDF](https://www.cis.upenn.edu/~green/papers/pods07.pdf).
- **Paraphrased excerpt:** Annotating database tuples with provenance polynomials distinguishes alternative derivations (sum) from jointly required inputs (product), retaining more information than Boolean membership.
- **Testable translation:** For two proof paths, C may have polynomial terms (A*R1)+(B*R2); setting A=0 removes only its term, whereas shared upstream U in *both* terms factors as U*(A*R1+B*R2). This is a conceptual oracle for hidden shared causes, **not** a direct arithmetic prescription for NARS truth revision.

## Minimal synthetic world and comparators

    U -> A -> R1 -> C
      -> B -> R2 -> C
    independent D -> R3 -> C

World W0 contains C with three derivations; suppress A, C survives via B and D; suppress U, A and B disappear but D survives; suppress U AND D, C disappears. Compare:
- **Boolean conclusion membership**;
- **complete derivation multiplicity** and provenance root sets;
- **numerical truth** only within a specific declared reasoner's semantics;
- **temporal prediction** and selected action;
- **certification** (Observed/Related/Explains/Causes separately, no new earned 'Causes' from ablation);
- **horizon integrity** (future events after pin inaccessible).

The first hard falsifier is deliberately not 'did C disappear?'. It is **'did the correct support disappear while independent support survived?'** This distinguishes exact source-aware retraction from a deceptively successful but causally blind deletion.

## A cost-aware cognitive probe, not a truth generator

Candidate X can be selected from provenance branches/rule assumptions to maximize measured discriminatory value under a time/memory budget. A suggested hypothesis (not a result) is expected information gain per estimated probe cost. Benchmark it against random cuts, largest-subgraph cuts, highest-confidence cuts, and full replay. All candidate scheduling, epistemic certification, and Rubicon commitment remain distinct; an entropy reduction must never be counted as additional observational support.

## Research stop conditions

- No truth-law crosswalk: do **not** equate NARS F/C with PLN, subjective-logic opinions or MeTTa rewrite membership.
- No root-level ancestry: exact fission/retraction **Unsupported**.
- Unpinned source horizon: strict historical claim **Unsupported**.
- Incomplete or evicted derivation trace: affected cone **Partial**.
- No SCM and admissible contingency policy: causal identification **Unsupported**.
- No parity against fresh replay: sparse speedup **Not demonstrated**.
- No comparison with code and history: novelty **Not established**.

Related repo material: [existing 2026-10 NARS primary excerpts](2026-10_nars_self_epistemic_needs_evidence_primary_excerpts.md), [Gadamer ablation audit](../.claude/research/2026-10-recovery/GADAMER-ABLATION-RESEARCH.md), [SAG #1509](https://github.com/AdaWorldAPI/lance-graph/pull/1509).
