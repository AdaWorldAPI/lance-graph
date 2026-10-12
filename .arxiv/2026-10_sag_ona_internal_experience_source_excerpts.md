# D-SAG-2 source excerpts: ONA's event–anticipation–feedback loop and internal experience

**Read:** 2026-10-12. **Kind:** SOURCE-GROUNDED PARAPHRASES / RESEARCH; no ONA port, executable parity test, or runtime integration in this note. **Upstream pin:** [OpenNARS for Applications @ dc4efd0](https://github.com/opennars/OpenNARS-for-Applications/tree/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a). **Companions:** [D-SAG-2 plan](../.claude/plans/software-awareness-agent-context-ona-v1.md), [repository-code-graph papers](2026-10_sag_repo_graph_context_source_excerpts.md).

The term *excerpt* here means a short, attributed **paraphrase** of a source passage or an implementation observation, not a copied passage. Any project transfer is labeled a hypothesis. Existing [2018 self-model note](2018_self_in_nars_requires_internal_experience_and_mental_operations.md) and [NAL audit #1495](https://github.com/AdaWorldAPI/lance-graph/pull/1495) remain separate authorities.

## Primary S1. Hammer & Lofthouse (2020): ONA architecture and control

**Source:** Patrick Hammer and Tony Lofthouse, *'OpenNARS for Applications': Architecture and Control*, International Conference on Artificial General Intelligence (2020), pp. 193–204. [Author-hosted conference PDF](https://agi-conf.org/2020/wp-content/uploads/2020/06/AGI-20_paper_52.pdf). Compare with [ONA README](https://github.com/opennars/OpenNARS-for-Applications/blob/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a/README.md).

**Excerpt S1a (abstract / architecture, paraphrase).** The authors present a pragmatic NARS implementation with event-driven control, separated sensorimotor and semantic inference, and explicit attention under resource constraints. They identify architectural differences from OpenNARS for Research, rather than asserting that ONA is merely its latest release.

**For lance-graph:** use ONA as a *reference design* for an observation and experiment-selection policy. It is not a replacement for Quack's algebra, the class-method binder, or mask-risc physical execution.

**Falsifier:** the proposed 'ONA-inspired' path introduces an event interpreter in the warm query execution loop. That is an architectural failure, not necessary support for anticipation.

**Excerpt S1b (procedure learning / temporal consequences, paraphrase).** Observed sequences and actions can be associated with later consequences to guide future choices under incomplete knowledge. This supports a research analogy between performance hypotheses and subsequent benchmark feedback, **not** automatic authorization to change software.

**For lance-graph:** distinguish precondition and operation events, predicted consequence, measured outcome, revision and next selection. Demand predicted outcome registration before the benchmark run.

**Falsifier:** a 'prediction' is manufactured only after the result is visible, or a performance association is called causation without controlled comparison.

## Primary S2. ONA implementation: pin actual functions

**Pinned source:** [NAR.c](https://github.com/opennars/OpenNARS-for-Applications/blob/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a/src/NAR.c), [Cycle.c](https://github.com/opennars/OpenNARS-for-Applications/blob/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a/src/Cycle.c), [Memory.c](https://github.com/opennars/OpenNARS-for-Applications/blob/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a/src/Memory.c), [Decision.c](https://github.com/opennars/OpenNARS-for-Applications/blob/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a/src/Decision.c), [Inference.c](https://github.com/opennars/OpenNARS-for-Applications/blob/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a/src/Inference.c).

| Step | Source anchor (verified names; line numbering tied to the upstream pin) | What source actually does | Proposed analogy |
|---|---|---|---|
| Input | `NAR.c: NAR_AddInput` near L55 | Makes a stamped event, adds it to memory and advances a cycle. | Source-backed method/benchmark observation. |
| Scheduling | `NAR.c: NAR_Cycles`; `Cycle.c: Cycle_Perform` near L1102 | Selects belief/goal events, processes beliefs and goals, performs inference and relative forgetting. | Bounded attention policy outside executor. |
| Concepts | `Memory.c: Memory_Conceptualize` near L109; `Concept.h` near L35 | Reuses or creates a concept with event and implication tables, including predicted belief. | Canonical graph identity plus observation history (not the same physical data model). |
| Induction | `Cycle.c: Cycle_ReinforceLink` near L647 | Temporal belief induction after event checks including evidential stamp overlap. | Learn candidate precondition→consequence links. |
| Implication store | `Memory.c: Memory_ProcessNewBeliefEvent` near L276 | Extracts implication and optional operation, links source/target concepts and revises table entry. | Persist hypotheses with exact method/version/provenance. |
| Decision | `Decision.c: Decision_BestCandidate` near L414, `Decision_Suggest` near L814 | Considers applicable implications and sometimes exploration/babbling. | Compare candidate legal plans under uncertain costs. |
| Anticipation | `Decision.c: Decision_Anticipate` near L556 | Applies predictions and negative-confirmation machinery to implications. | Register predicted cost/outcome *before* experiment. |
| Operation | `Decision.c: Decision_Execute` near L51 | Calls a registered procedure and injects feedback belief events. | Run **authorized** pure benchmark, record receipt. |
| Revision | `Inference.c: Inference_RevisionAndChoice` near L152; `Memory.c: Table_AddAndRevise` call near L328 | Choice or revision depends on stamp overlap, term/time rules and evidence. | Deduplicate source ancestry; revise a hypothesis without replay inflation. |

**Source-specific subtlety.** `Decision_AddNegativeConfirmation` (near Decision.c L34) constructs a negative confirmation with a low configured confidence; `Decision_Anticipate` uses this machinery on active predictions. It should **not** be described simplistically as 'a timer always waits for a promised outcome and then records failure'. Our proposed benchmark system must define the opportunity/window for any non-observation claim explicitly. ONA's precise update rules require pinned oracle tests before parity is asserted.

**Source-specific subtlety.** `Decision_Execute` invokes the registered `Action`, checks its feedback and can re-inject the resulting event with `NAR_AddInputBelief`. This is not an execution receipt carrying CPU cycles, allocation counts, physical plan identity or source SHA. These must be designed and measured separately for lance-graph.

**Source-specific subtlety.** `Inference_RevisionAndChoice` calls `Stamp_checkOverlap`. Two receipts for the same experiment are not independent simply because they entered on different hypothesis layers. Stamp equality alone does not establish absence of shared underlying data or systematic bias.

**Falsifiers:** (i) replay of one receipt increases evidence; (ii) same operation with different arguments falsely shares a prediction; (iii) no actual decision/operation feedback path executes; (iv) concept/resource limits are ignored; (v) an unknown or delayed outcome is automatically marked negative; (vi) matching a method name is treated as authority.

## Primary S3. Wang, Li & Hammer (2018): *Self in NARS, an AGI System*

**Source:** Pei Wang, Xiang Li and Patrick Hammer, *Frontiers in Robotics and AI* 5:20, DOI [10.3389/frobt.2018.00020](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00020/full), §§2–3.5. Existing local note: [2018_self_in_nars_requires_internal_experience_and_mental_operations.md](2018_self_in_nars_requires_internal_experience_and_mental_operations.md).

**Excerpt S3a (§§2–3, paraphrase).** The paper treats an intelligent system as operating with insufficient knowledge and resources, with expectations and concept meanings revised by experience rather than perfected once and held fixed.

**Transfer:** a source-backed software graph is incomplete by construction. Index coverage and confidence must be visible; missing dependency edge means *unknown/not found by this index*, never proven absent.

**Excerpt S3b (§3.4, paraphrase).** Internal selections, reasoning steps and derivation relationships can be reintroduced as internal experiences for subsequent inquiries about the system's own activity. Internal self-models are partial, selected and revisable.

**Transfer:** `MethodExecution`, `QuerySelection`, `ProgramLowering` and `ExecutionReceipt` may be graph events; they must retain exact source links and limits. A graph that merely logs events does not automatically possess reflective control.

**Excerpt S3c (§§3.3–3.5, paraphrase).** The authors distinguish built-in automatic inference control from actions that deliberately redirect cognitive processing; appraisal-like mechanisms help allocate scarce attention. The text is a theory of self-related cognition with preliminary implementation, not empirical verification of artificial subjective experience.

**Transfer:** keep (i) structural introspection, (ii) runtime introspection, (iii) prediction, (iv) deliberate experiment choice and (v) independently validated method edits as separately tested capabilities.

**Negative control:** give a model the same static trace dump without active source graph queries or action/attention policy. If the experiment performs equally, the extra 'self' mechanism has not shown added utility.

## Project synthesis: observable self-model, not a second reasoner

```text
MethodTable + compiler verdicts → SoftwareGraph rows (#1509, measured)
execution receipt (proposed)    → versioned observation
ONA-like implication (proposed)  → precondition + possible outcome
registered anticipation         → predicted cost interval / resource effect
isolated benchmark              → observation with independent typed oracle
revision + provenance           → updated hypothesis, not rewritten truth bytes
bounded proposer                → select next **legal** experiment
```

The physical byte-level register/ISA stays invariant; method meanings and NARS readings are contextual. Cache keys and confidence values are **not** proofs. A receipt describes what ran, not what the compiler was legally allowed to run absent the existing gates.

### Related existing evidence (do not duplicate)

- [#1495 NAL audit](https://github.com/AdaWorldAPI/lance-graph/pull/1495): recipe 10/MCP lacks a read of executed-history receipts; a test-local receipt reader demonstrates a missing contract, not a shipped receipt feed.
- [#1359 observe–revise–learn](https://github.com/AdaWorldAPI/lance-graph/pull/1359): compiled witness for a fixed SPOFC/DO/learning cycle over a read-only population and separate owner-written `NEW` window, not a generic adaptive optimizer.
- [#1509 software-awareness graph](https://github.com/AdaWorldAPI/lance-graph/pull/1509): one-hop executable graph projection and H1 with unfulfilled fused-SIMD obligations.

### Acceptance experiment

Pin method/schema/source/CPU and independent answers. Record a legal H1-like plan choice, pre-register measurable predicted consequences, run the isolated measurement, then revise a hypothesis according to source-distinct outcomes. Compare next-choice correctness against (a) a fixed heuristic and (b) a simple history-based cost table. If simpler baselines win, record that result and stop adding cognitive machinery.

**References:** [Hammer & Lofthouse 2020](https://agi-conf.org/2020/wp-content/uploads/2020/06/AGI-20_paper_52.pdf); [ONA pinned code](https://github.com/opennars/OpenNARS-for-Applications/tree/dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a); [Wang, Li & Hammer 2018](https://doi.org/10.3389/frobt.2018.00020).
