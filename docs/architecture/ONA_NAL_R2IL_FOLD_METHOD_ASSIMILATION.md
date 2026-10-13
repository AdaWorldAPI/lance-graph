# ONA assimilation into LanceGraph NAL / R2IL: executable fold-method program

**Status:** IMPLEMENTATION TARGET / FALSIFIABLE DESIGN. **Not yet shipped by this document.**  
**Date:** 2026-10-13  
**Companion:** [Outer membrane graph lab](OUTER_MEMBRANE_GRAPH_LAB_VERTICAL_SLICE.md) and [epoch-indexed temporal graph research](EPOCH_TEMPORAL_GRAPH_RESEARCH_AGENDA.md)  
**Operator goal:** Assimilate **OpenNARS for Applications (ONA)** reasoning capabilities into the **existing** `lance-graph-nal` and `lance-graph-r2il` as global *folding IR methods*, not merely use ONA to populate a comparison slide.

## 0. Decisive ownership and non-goals

This extends, rather than reverses, the middle-ground ruling:

- **LanceGraph NAL:** one typed method API for global, consumer-ClassId-agnostic belief operations, NARS truth laws, inference, temporal projection, choice and candidate generation. The classid is an opaque addressing/binding input, not a hard-coded ontology.
- **LanceGraph R2IL:** compile methods to existing R2IL call sequences / `FoldDialect` / mask-risc programs, with explicit result shapes and fold boundaries. This is the *existing* globally reusable compute bridge.
- **OGAR `ogar-loco` / `ogar-r2il`:** remain as the authoritative Loco/ABI and r2sleigh-facing ecosystem. Their existing opcode vocabulary can be consumed by LanceGraph; there is **no blanket relocation, split-brain opcode table or second interpreter** in this program.
- **Quack / mask-risc / ndarray:** continue to own the bulk physical execution, without NAL-shaped knowledge in the byte/ALU register layout. Quack remains a relational frontend, NAL a reasoning frontend, and both use the same physical substrate.
- **Planner / Rubicon / callcenter + OGAR action gate:** own scheduling, commitments, authorization, effects and receipts. A fold may **rank or propose** an action but may not execute it, mint permission or advance Kanban on its own.
- **ONA:** pinned external *semantic and behavioral oracle*, and a source of pure methods to reimplement. Its C memory store, imperative shell, mutable concept tables, scheduler and motor-execution loop **do not** become new Rust runtimes.
- **No new permanent side ledger:** retain canonical NodeGuid, sealed SoA and existing evidence/provenance structures. Any richer view/receipt is a proposed typed reading or existing canonical storage projection, not a parallel truth store.

**Success criterion:** a real population of witnessed, provenance-bearing beliefs is processed end-to-end by typed NAL methods lowered through R2IL/mask-risc; ONA fixtures are independently re-run and compared; temporal and multi-premise inference are correct; action selection remains inert until authorized. A documentation-only method list is **not** completion.

## 1. Baseline: what is already real, and what is unfinished

| Surface | Verified evidence as of this research | Must not be overclaimed |
|---|---|---|
| [`lance-graph-nal/src/lib.rs`](../../crates/lance-graph-nal/src/lib.rs) | `NodeGuid` belief identity, separate `f` and `c` value lanes, ordered facet witness, `Aperture` interval selection, prepared execution through mask-risc | Does **not** currently provide row-wise truth laws, syllogisms, Narsese parser, executable goal operations. Revision: W3 in #1538 (below) |
| [`lance-graph-nal/src/nal.rs`](../../crates/lance-graph-nal/src/nal.rs) | `NalOp` trait, `Question` count, `TruthAtLeast` count, `Choice` highest-confidence value; `Prepared::of` demands a *scalar* at one fold | `Choice` is **max value**, not argmax identity; intermediate mask/range result shapes are not exposed as kept NAL outputs |
| [`lance-graph-r2il/src/lib.rs`](../../crates/lance-graph-r2il/src/lib.rs) | `FoldDialect` with `Val::Scalar / Slot / Address`, typed error/refusal semantics, fused/in-place masks, fold-boundary compilation to mask-risc; `SUM`, `MIN`, `MAX`, `GROUP_SUM`, `KEEP` and other calls | Not a whole-chain normalizer; `KEY_RUNS`, `ANY`, `ALL`, `SCATTER_OR`, `BLEND` are explicitly refused in the current dialect. No row-sized data on Loco stack |
| [`lance-graph-r2il/src/procs.rs`](../../crates/lance-graph-r2il/src/procs.rs) | Named procedures, no duplicated opcode allocation, calls lower through the existing dialect | `PROC_DEF`/`PROC_CALL` with real stack frames remain an OGAR ABI work item |
| [`nal-polyfill-v1`](../../.claude/plans/nal-polyfill-v1.md) | W3 weighted revision via group moments, W4 lane-producing truth laws, W5 syllogisms and witnesses, later Narsese UI already described as **proposals** | Existing plan prose is not proof those operations are shipped |
| [`nal-recipe-audit`](../../.claude/knowledge/nal-recipe-audit/README.md) | External oracles and 34-recipe audit, observed revision mismatch in two local laws, seven ONA Narsese fixtures; precise gaps for self-reflection | 34 recipes are not 34 NAL methods; an executable `{SELF}` fixture is not proof of introspection |

The existing [`NalOp`](../../crates/lance-graph-nal/src/nal.rs) and R2IL procedure style are the **starting implementation contract**. Improve them rather than minting a competing "ONA operator registry". NAL remains independently buildable without DataFusion.

## 2. ONA source mapping (what to assimilate)

**Primary upstream:** [opennars/OpenNARS-for-Applications](https://github.com/opennars/OpenNARS-for-Applications), in C under MIT license. Review and independently reimplement with source attribution; do not copy C generated dispatch tables into Rust by default.

| ONA authority | Relevant C functions / behaviors | Target NAL method and fold shape | Hard semantic gate |
|---|---|---|---|
| [`src/Truth.c`](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Truth.c) | `Truth_w2c`, `c2w`, `Revision`, `Expectation` | `TruthWeight`, `Revision`: grouped evidence moment fold / typed truth result | Independent original observation roots; right evidence horizon and rounding |
| Same | `Deduction`, `Induction`, `Abduction`, `Analogy`, `Comparison`, `Resemblance`, `Conversion`, `Negation`, intersections/unions | Pure typed truth-map methods over aligned lanes or scalar oracle | Exactly named law variant and input premise orientation |
| Same | `Truth_Projection`, `Eternalize`, time decay | `TemporalProjection`: typed clock-aware value-map, not a new clock | Compare source occurrence time / target time / eternal sentinel / decay parameter |
| [`src/Stamp.c`](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Stamp.c) | `Stamp_make`, `Stamp_checkOverlap`, `Stamp_hasDuplicate` | `IndependentEvidence` mask / `UnionRoots` witness constructor | Never revise as independent if roots overlap or cannot be proven disjoint |
| [`src/Inference.c`](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Inference.c) | Belief induction/intersection, event/implication revision, implication/goal deduction, revision-and-choice | `Inference::InduceTemporal` / `Derive` / `Revise`; pairing + mask + truth law + witnessed conclusion | Source and target node identities, temporality, separate witness roots, no recursive self-support |
| [`src/NAL.c`](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/NAL.c) | `NAL_GenerateRuleTable`, premise unification, conclusion substitution/reduction | Compile declarative rule templates into **typed bound NAL methods** over existing R2IL/Quack join primitives | Grammar coverage, variable capture, term identity, multiplicity, rule applicability; refuse unknown shape |
| [`src/Cycle.c`](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Cycle.c) | Temporal belief event selection and inference, goal-event decomposition | Pure event selection / ranking / temporal candidate predicates; produce bounded candidate handles | No second event loop, no speculative inference promoted to observation |
| [`src/Decision.c`](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Decision.c) | Goal/implication desirability, best candidate, anticipation, negative confirmation, `^op` execution | `GoalCandidateScore`, `BestCandidate`, `AnticipationProbe` → **inert proposed action** | Actual effects only via existing Rubicon, RBAC, capability, freshness, receipt and feedback checks |
| [`src/Memory.c`](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Memory.c) and priority queue | Concept activation, forgetting, selection pressure | Prioritization/budget as planner method over canonical SoA and existing masks | No second memory database, hidden non-deterministic reordering or copied retention mechanism |

**Known oracle caveats, not corner-case footnotes:**

- The [2026-10-11 oracle audit](../../.claude/knowledge/nal-recipe-audit/oracles.md) pinned ONA at `dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a` and OpenNARS for Research at `4d1bdb0f599ccf6ed44edfc398f02f930713d30f`. It ran seven ONA fixtures; it did **not** exhaust NAL or test every ONA method.
- ONA `evaluation.py` silently omits `reasoningbench.nal`'s comment expectation syntax, uses success `exit(0)` for some failures, and primarily threshold-checks confidence rather than exact frequency and confidence. Build an **independent** parser/oracle; do not use its "Passed" output alone.
- One observed reasoningbench result is confidence `0.815686` where its comment says `0.729`, with a repeated stamp source. Treat that exact output as an observation, not an unquestionable NARS axiom.
- ONA and OpenNARS for Research and Wang's published laws differ on some compounds and confidence rules. Every parity test names its oracle and revision. Explicitly record intended divergences with counterexamples; do not silently make all references "agree".
- ONA's `{SELF}` fixtures prove selection/execution of operations on itself as an operand, **not** the ability to inspect prior inference steps. Our actual self-correction requires addressable executed-step/outcome receipts (a separate admission gate).

## 3. How ONA becomes a fold method without becoming another engine

A NAL method is a **typed, pure procedure** that:
(1) names a statement / relation aperture;
(2) resolves premise populations and verifies ordered-lane witness, source generation, authority and root independence;
(3) binds named terms and rule variables to opaque identity/ordinal lanes *once*;
(4) lowers supported tests/joins/masks/value laws through existing R2IL procedures to mask-risc;
(5) emits a typed result (mask, range, scalar, truth pair, selected candidate, or witnessed *proposed* conclusion);
(6) materializes the approved conclusion only at the existing canonical write/seal boundary, never as a side effect of a fold.

```text
ONA rule / NARS reference test
           |
  typed NAL method definition (global law + oracle version)
           |
  bind terms / premise witness / ClassView reading / time + policy
           |
  normalize pure law + required masks + join constraints
           |
       lance-graph-r2il FoldDialect
           |
       mask-risc Program / compiled fold
           |
       ndarray scalar/SIMD backend
           |
  typed result + provenance + no-effect proposal
           |
  authorized seal / planner / action gate (if explicitly needed)
```

**Do not force `Prepared::of` to conclude only `[Val::Scalar(_)]`.** It currently has that exact check and one `CompiledProgram`. Introducing value-producing inference requires a **design audit of retained result shapes**: e.g., a typed result view containing a stable mask/range/argument handle and/or externally owned destination lane. This is not permission to add a second byte-register ABI or persist new context-free blobs. Existing `Val::Slot` generations and `FoldError::StaleSlot` must remain valid: a temporary slot cannot be promoted to a durable cross-fold handle by accident.

**No forced "all-at-once" R2IL lowering yet.** `FoldDialect` currently compiles/executes at **each scalar-producing fold boundary** because mask-risc `Program` has one terminal. The future Concept-Chain / D-CMR algebra should normalize compatible chains symbolically *before* execution; do not claim that current R2IL code performs that normalization or require Quack for every non-SQL chain. Preserve truthful boundary metrics.

**Store evidence as evidence, not just a rounded f/c pair.** The already proposed [W3 plan](../../.claude/plans/nal-polyfill-v1.md) uses `GroupCrossPowerSumsI32` over fixed-point evidence weights and frequency; verify its assumptions including independence, overflow, confidence clipping, group key and round-once. Current `Beliefs` only has separate `f,c` vectors. If no weight/witness roots are available, **refuse exact bulk revision** or materialize an explicitly labelled approximate view. Do not invent independence from identical stamps.

**W4 truth maps need a real lane-producing path.** Mask combination alone cannot output per-pair new `f,c` lanes. Compare existing `BlendI32` / appropriate shared lane terminals, one generic audited `MapPair`-like operation only if necessary, versus fixed-point arithmetic. Do not add an ONA-specific physical opcode if an existing instruction or procedure suffices. Check exact float/quantized parity, clipping and NaN/refusal policy at boundaries.

**W5 multi-premise inference is join + truth-map + witness.** Matching a middle term is not permission to combine duplicate derivations or conceal multiplicity. Use existing semijoin/gather only where it has the correct 1:1 / 1:many semantics, and prove correct handling of duplicate statements, identity aliases, cycles and cross-class binding. Unsupported compound syntax, independent/dependent variables and higher-order relations must be refused by name until their typed grammar and binder exist.

**Decision/goal methods must stay inert.** ONA's `Decision_Execute` is real imperative effect logic. We assimilate its precondition/expected-outcome/best-choice **calculation**, then hand a proposal to existing planner/Rubicon/action authorization. No candidate is allowed to cross the authorization boundary merely because confidence rose, a goal was selected, or an anticipated outcome looks likely.

## 4. Implementation waves (new target beyond the existing W1/W2)

Each row is a **separate coding PR** with raw engine fixtures pinned and independent Rust oracle/tests; this docs PR implements none of them.

| Wave | First executable feature | Required dependencies | Completion / negative controls |
|---|---|---|---|
| **W3** | `Revision` over grouped independent beliefs, evidence weight + `f,c` | `lance-graph-nal` trait + `r2il::procs` + existing grouped moments; no new global opcode | ONA 2-premise and N-premise cases; overlapping stamps rejected/choice-only; zero/one confidence and overflow; exact independent row oracle <br>**Status 2026-10-13: in PR #1538.** `Beliefs::seal_evidence` + `nal::Revision`: one `GroupCrossPowerSumsI32` keyed by an evidence class from the stamps (independent / exact duplicate counted once / overlap / no stamp). Overlap and missing provenance are refused, not choice; ONA's choice-on-overlap is not built. Within one code of ONA `Truth_Revision` (transcribed at `dc4efd0`) and of the three OpenNARS points, up to ONA's mid-fold saturation: past it ONA's pairwise cap makes the answer depend on arrival order, and the fold keeps the exact `Σw` instead (documented divergence, pinned by a test); `Σw`, `Σw·f` exact; c = 1 refused at write; seven disable runs red. The result carries sums, counts and duplicates, not the stamps themselves (§6 item 4 partly open) |
| **W4** | Per-row `Deduction`, `Induction`, `Abduction`, `Analogy` truth producing output lanes | Generic value-producing terminal (reuse/extend one shared terminal if measurements demand), portable SIMD/scalar parity | Exact oracle per rule, quantization boundaries, empty/masked rows, no wrong-sense premise swap |
| **W5** | `Inference::derive` and `Revise` over matched premise pairs | Typed term unifier, bind-time ClassView/ReadingId, existing Quack/mask-risc joins + witnessed output | ONA NAL fixtures, independent premises, duplicate/cyclic/self-reference, multi-edge, source-generation mismatch |
| **W6** | `Narsese` text cold parser and declared/method frontends onto the **same** typed ops | Existing parser precedent; no text in hot loop | One query in typed/Narsese form yields byte-identical prepared result or explicit unsupported-form refusal |
| **W7** | `TemporalProjection` / sequence induction with **epoch-typed** occurrence, valid and knowledge horizons | Existing `temporal_pov`, planner `QueryReference`, sealed-cycle and timestamp mapping | Time reversal/late knowledge falsifier, no as-known leakage, no duplicated temporal engine; ONA decay parity where declared |
| **W8** | `GoalCandidateScore`, `BestCandidate`, `Anticipation`; optional choice argmax carrying real identity | Existing planner / Rubicon, action gate; stable tie breaks, provenance | Wrong/stale/unauthorized goal remains inert; independent outcome receipt; no external action from fold |
| **W9** | First-class executed step / evidence root trace usable as NAL operand; rules about prior inference | Existing `ExecutedStep`/receipt work; no side ledger; provenance and causal ablation | Calibrated vs overconfident histories yield different verdicts, duplicates do not inflate support, self-referential evidence rejected |

**Definition of “finalize”:** Do not describe `lance-graph-nal` or `lance-graph-r2il` as finalized merely because W3 compiles. Require W1/W2 regression, W3–W5 parity with pinned ONA fixture or a documented mismatch, W7 causal temporal safeguards, both scalar and SIMD parity on exercised operations, stable witness+identity, no class-dependent hardcodes, and at least one **end-to-end** method called through the existing graph/planner frontends. W8/W9 may be exposed as explicitly scoped capability flags with refused unsupported paths if the operator agrees to that narrower “NAL core complete” milestone.

## 5. Required evidence and falsifiers

**External execution:** For each candidate law, run ONA `./NAR shell` against a pinned test file and capture *actual answers, both f and c, derivation stamp, occurrence time and chosen operation*. Do not compare to a comment (`expected:`) alone. Pin compiler/platform and ONA config values including evidential horizon and projection decay. Compare separately with published NARS laws and OpenNARS for Research, preserving disagreements.

**Execution parity:**
- Same source rows: scalar reference ⇔ typed procedure ⇔ compiled R2IL ⇔ mask-risc scalar vs available SIMD tiers. Verify exact integers/ids, bounded decimal tolerance/rounding policy for float pairs, and provenance/temporal result equality.
- `Count` and `Max` do not prove an **argmax** or a new **truth lane**. Output shape is part of the test.
- Lowering is deterministic for the same bound schema, oracle-law version, program body, reading, source version and policy; no mode may silently select a different law.
- No executable NAL byte is minted absent a proof that a procedure on existing operations cannot express the method. No second memory/event/graph executor.
- An independent negative control (deliberately wrong confidence, swapped premise orientation, shared root, future knowledge, stale generation, policy denied) **must fail the parity gate**.

**Causal/meta and action safety:**
- A proposal ranked by goal desire is not `ActionInvocation::Committed` and is not an executed operator. Gate with current RBAC, idempotency, freshness and receipt, including "unknown external effect" outcomes.
- ONA `{SELF}` operational examples are **not a self-inspection oracle**. Before claiming metacognition, show that the method can read and evaluate its actual earlier executed method/witness and produce a prediction that changes a subsequent independently measured choice.
- Evidence source roots cannot be minted from derived conclusions; prevent self-support, double-counting and false independence. Define response to truncated ONA stamps explicitly rather than treating absence as independent evidence.

**Performance:** end-to-end benchmark batch size, join fan-out, changed-edge churn, compile/bind cost, L1/L2 residency, allocations, full register movement, result materialization, epoch playback; compare natural Rust, intrinsics and optional assembly on same hardware. Never claim `0.6 ns` or a SIMD improvement without measured workloads and verified emitted instructions.

## 6. First practical implementation PR: smallest vertical falsifier

1. Lock ONA oracle revision/config and import **only test fixture inputs/expected observed outputs with attribution**. Avoid vendoring `src/RuleTable.c` or ONA runtime. Re-run one 2-premise NAL case, one overlap stamp, one time-reversed inference and one NAL rule with a conclusion requiring more than a scalar.
2. Keep `NalOp::FN = None` for W3 if revision is a procedure over existing grouped moments. Extend the *binding/read* interface only if a measured necessity exists for weight/witness inputs, and preserve current W1/W2 tests.
3. Lower the revision through `lance-graph-r2il::procs` to `mask-risc`. If `GroupCrossPowerSumsI32` is the right primitive, prove its weight and widening arithmetic against both an exact Rust row oracle and ONA floating truth. **No scalar for-loop over all rows inside the R2IL dialect.**
4. Return a *typed revision result* that retains source and stamp information, not merely `count` or highest confidence. Refuse overlapping roots or choose a documented alternative; never silently sum them.
5. Write a disable falsifier: deliberately remove the root-overlap gate or swap evidence weights; tests must go red for known counterexamples. Then restore, show tests green and report measured bound/query cost.
6. Only after W3 passes start W4, then W5. W7 temporal/epoch work should share the research agenda rather than introduce a second clock model.

## 7. References and decisions to preserve

- [ONA repository](https://github.com/opennars/OpenNARS-for-Applications) and [ONA truth laws](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Truth.c), [inference](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Inference.c), [rule generation](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/NAL.c), [decision machinery](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Decision.c), [evidence stamps](https://github.com/opennars/OpenNARS-for-Applications/blob/master/src/Stamp.c).
- [Existing D-NALP plan](../../.claude/plans/nal-polyfill-v1.md), [NAL implementation](../../crates/lance-graph-nal/src/nal.rs), [R2IL implementation](../../crates/lance-graph-r2il/src/lib.rs), [R2IL procedures](../../crates/lance-graph-r2il/src/procs.rs).
- [Prior independent ONA/OpenNARS oracle outputs](../../.claude/knowledge/nal-recipe-audit/oracles.md); [NAL capability audit](../../.claude/knowledge/nal-recipe-audit/README.md).
- [SAG method alignment](../../.claude/plans/sag-capstone-alignment-v1.md), [ONA/Hyperon wisdom frontier](../../.claude/plans/sag-wisdom-frontier-ona-hyperon-v1.md): `ProposedMethod` is inert and must not bypass canonical authorization.
- [Temporal architecture and epoch research](EPOCH_TEMPORAL_GRAPH_RESEARCH_AGENDA.md) and [seal/temporal ordering distinction](../../.claude/knowledge/seal-vs-temporal-ordering-information.md).

**Next review decision:** ratify W3's exact truth law/oracle/weight model and source independence gate. Do not approve "full ONA assimilation" until the W3 falsifiers and W4 value-lane feasibility pass. The completion claim comes from parity and live call paths, not prose.
