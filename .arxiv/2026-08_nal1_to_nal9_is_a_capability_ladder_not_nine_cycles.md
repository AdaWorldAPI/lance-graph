# NAL-1…NAL-9 — A capability ladder, not nine mandatory execution cycles

**Primary source:** Pei Wang, *A Logical Model of Intelligence — An introduction to NARS*, updated August 2026.  
**Source:** https://cis.temple.edu/~pwang/NARS-Intro.html  
**Corroboration:** Wang, *Non-Axiomatic Logic: A Model of Intelligent Reasoning*, 2nd ed., World Scientific, 2025 (book metadata/overview; full book **not** verified here).  
**Supplement:** Wang, *NAL-9 Implementation*, presentation, https://cis.temple.edu/tagit/presentations/NAL%209.pdf  
**Read date:** 2026-10-11. **Nature:** author-hosted web exposition plus presentation, **not itself a new arXiv paper**.

## Punchline

> **NAL-1 through NAL-9 progressively enlarge the grammar, semantics, and inference capability of one non-axiomatic reasoning system. They are not nine clock cycles, nine CPU instructions, nine Pearl rungs, or a promise of metacognition merely because nine labels are present.**

The productive connection to lance-graph is a **capability/falsifier matrix** evaluated against existing ALU/ISA operations, declared semantic readings, provenance, procedural control, and self-models. Keep those layers separate.

## Source contract and caution

- Wang's text (§1.2, §2, §3) deliberately introduces NAL as a **layered logic**: each layer extends grammar and inference rules and enriches control/memory.
- A system has **insufficient knowledge and resources** (AIKR); beliefs are experience-grounded and revisable, not eternal axioms.
- The architecture joins a *logic part* (language, semantics, inference rules) with a *control part* (memory and adaptive resource allocation).
- Layers are cumulative capabilities. Wang does **not** say every problem must traverse nine levels sequentially.
- The Gabriel/Wang 2026 dataset's "hard = 6–9 steps" is **reasoning-chain depth**, not certification of NAL-6 through NAL-9. Conflation would invalidate any reported score.
- Do not interpret CE64's five epistemic bits, the 10 AlphaTunnel rungs, a SIMD lane, a Moore neighborhood or a Pearl rung as a NAL level.

## Exact conceptual ladder, with examples and falsifiers

### NAL-1: atomic inheritance and uncertain syllogistic reasoning

**Source:** atomic terms, inheritance statements, positive and negative evidence; deduction, induction, abduction, revision, choice.

An inheritance statement expresses substitutability of terms in experience-grounded use. It is not the same as a typed graph edge with an automatically globally true taxonomic meaning.

- *Deduction*: combine supporting premises under a compatible syllogistic pattern.
- *Induction*: generalize a relation from shared cases without pretending that instances prove universality.
- *Abduction*: infer plausible explanations without turning the hypothesis into observation.
- *Revision*: combine new genuinely independent support with existing support.
- *Choice*: choose between competing beliefs where naive addition would duplicate support.
- Truth value has frequency **and** confidence in Wang's specific semantics; it is not an ordinary confidence interval.

**Minimal falsifiers:**
1. From A→B and B→C derive candidate A→C under the declared rule; preserve non-axiomatic confidence.
2. Replace one independent premise with a duplicate derivation from the same root; confidence must not inflate.
3. Give contradictory evidence; maintain a revisable state, do not explode into arbitrary claims.
4. A choice/revision example must expose evidence lineage; distinct log EventIds alone do not imply independent roots.

**lance-graph mapping hypothesis:** NARS mask-fold tables and replay can provide pieces of NAL-1, but must be judged with actual truth-function parity and provenance tests. The register ISA does not magically imply inheritance semantics.

### NAL-2: similarity, instance, property, analogy

**Source:** adds variants of inheritance copula: similarity, instance, property; comparison, analogy, resemblance, and singleton extension/intension constructions.

- Distinguish directional inheritance from symmetric similarity.
- An individual-instance assertion differs from class inheritance.
- A property assertion differs from shared identity.
- Analogical transfer is evidence-generating inference with bounded support, not an equivalence assertion.

**Falsifiers:** swap source and destination; similarity may be symmetric but inheritance should not. Give same surface embedding with distinct instance/property roles; the system must retain role distinctions. Reject unjustified transitive similarity.

**Transfer hypothesis:** typed edge readings and joins could host the operation; **do not** implement similarity by equating byte identity or Palette256 indices.

### NAL-3: compound terms by extensional/intensional set operators

**Source:** intersection, union and difference over extensions (instances) and intensions (properties); term composition derived from experience.

This is a new kind of *term construction*, not only more rows of atomic edges. The semantic relation between an extensional intersection and an intensional intersection is not automatically the same.

**Falsifiers:** known membership sets under intersection/union/difference; same individuals, different properties; empty and contradictory compounds; compound identity stable across input ordering where operation is commutative.

**Transfer hypothesis:** bitset intersections and SoA folds are useful execution primitives. Semantic term construction, typing, and provenance are separate obligations.

### NAL-4: arbitrary relations via product and image

**Source:** product and image term operators, allowing relations that cannot be expressed directly as inheritance copulas.

Think relational argument slots: an n-ary relation needs role-preserving projection/inversion rather than opaque concatenation of participants.

**Falsifiers:** apply product-image conversion and round-trip; distinguish agent, patient, instrument; keep argument order where relevant; compare an explicit relation with its projections without accidental loss of semantic arity.

**Transfer hypothesis:** SPO/graph joins and Quack group masks are excellent *candidate carriers*, but a join is not automatically a NAL-4 product/image operator. Proof requires equivalence on a declared grammar subset.

### NAL-5: higher-order statements, implication and equivalence

**Source:** statements can be terms in statements; implication/equivalence express inferential relationships among propositions.

This is the first clear step from reasoning about entities toward reasoning about **statements**. An implication is not simply a causal arrow; logical implication, experiential prediction and causal certification differ.

**Falsifiers:** nest a proposition as subject; reason over an implication as data; demonstrate distinction between P⇒Q and Q⇒P; keep implication separate from temporal prediction and Pearl do().

**Transfer hypothesis:** a claim↔chain-step mapping and declared relation reading are needed before trace-based higher-order inference can be called sound.

### NAL-6: variable terms, unification, hypothetical inference

**Source:** variables can be introduced, unified, eliminated/instantiated; hypothetical inference can operate at abstract-symbol level, serving as a metalogic of arbitrary reasoning systems.

This is not just bulk filter matching: variable scope, binding identity and substitution consistency matter.

**Falsifiers:** alpha-renaming invariance; capture-avoiding substitution; two clauses with shared variable agree on binding; reject accidental Cartesian joins; preserve variable distinctions under generalized rules.

**Transfer hypothesis:** typed Quack folds and mask-risc joins may handle large candidate sets, while binding/unification semantics remain a separate compiler or procedure obligation.

### NAL-7: events and temporal inference

**Source:** events are statements with temporal attributes relative to other events; jointly use logical and temporal information for prediction or explanation.

Do not confuse chronological order, statistical association, inferred temporal dependency and certified causality.

**Falsifiers:** reorder events; preserve interval/duration; distinguish before/after/simultaneous; predict then falsify; ensure revising a time relation doesn't silently certify Causes.

**Transfer hypothesis:** timestamps in sparse append logs provide event order, but not automatically a NAL temporal semantics. A log position is also not necessarily a stable EvidenceRootId after compaction.

### NAL-8: executable operations, goals, procedural inference

**Source:** an operation is an event directly realizable by the system through host programs; a goal is a desired event. The system selects operations to pursue goals.

An operation is not identical to an ordinary inference step: it can change the environment and has authority and effects. Deliberation must not bypass write ownership.

**Falsifiers:** goal unsatisfied without available action; available operation denied by authorization; action executed but goal not achieved; distinction between observed cause and intended purpose; record action receipt separately from forecast.

**Transfer hypothesis:** Thought/Recipe can describe operations; the ISA contract executes declared primitives; split-tunnel write-back and one-subject authorization govern side effects. None proves a planner can achieve goals.

### NAL-9: internal mental operations and self-control

**Source:** procedural operations act on NARS's own internal processing, producing a self-referential control loop; Wang mentions self-awareness, emotion and consciousness as associated phenomena. These are the author's theoretical claims, **not independently verified consciousness criteria**.

Distinguish:
1. recording that a reasoning step occurred;
2. retrieving its trace;
3. creating a self-model of that step;
4. comparing predicted vs actual reasoning;
5. choosing and executing an *authorized* change to future processing;
6. measuring whether that change improved the outcome.

Only 1–2 = logging/replay. NAL-9 concerns 3–5 (plus falsifiable 6).

**Falsifiers:** provenance-resolved self-query ("why this conclusion?"); diagnosis of a deliberately corrupted inference; revised plan changes only authorized owned state; reverted/ablation path shows when the revision mattered; bounded attention can skip unimportant self-events; no retroactive story invented in place of an execution trace.

**Implementation nuance from Wang's NAL-9 slides:** memory/inference engine, task buffers IEB/GEB, and I/O channels operate in cycles; local/global feelings and mental operations enter internal experience; memory, task-buffer and channel routines populate buffers. This describes a **work-cycle architecture**, not the claim that nine NAL layers are executed in order.

## Cross-layer dependency matrix (research hypothesis, not source theorem)

| Capability | Minimum conceptual dependency | Key risk |
|---|---|---|
| NAL-1 | evidence representation and inference | double-counted roots |
| NAL-2 | role-aware similarity/analogy | treating similarity as equivalence |
| NAL-3 | term algebra + extension/intension | Boolean set math mistaken for term meaning |
| NAL-4 | relation product/image + bindings | argument role loss |
| NAL-5 | proposition-as-term | implication mistakenly becomes Causes |
| NAL-6 | variable scope + substitution | accidental joins/capture |
| NAL-7 | explicit temporal interpretation | correlation mistaken for time/causality |
| NAL-8 | action/goal/permission/effect | intent mistaken for result or authority |
| NAL-9 | inspectable internal processes + self-control | replay mistaken for self-regulation |

These are *increasing language capabilities*, not nine instructions of an ALU.

## Critical distinction: BF16-like ISA contract vs NAL semantics

Our layout contract is analogous to a BF16 register format in **being byte-exact and executable**; it is not equivalent to BF16 floating semantics. A declared reading decides how bytes represent NARS F/C, CE64 evidence/state, Moore activation, a Palette carrier, etc. The instruction/recipe is external to the stored result. Consequences:

- Byte layout and lane order are *ABI facts*.
- Deterministic operators are *ISA facts*.
- Narsese inheritance, confidence revision and goal reasoning are *semantic facts* that must be specified separately.
- The same underlying fold algebra may implement multiple logical levels; that is only reuse of arithmetic, not proof of NAL parity.
- Never widen the AlphaTunnel's rung count just to match NAL-9. Never hardcode NAL into generic SlabReading.

## Recommended experiment: the NAL capability census

Read current main first. Build a table, one **concrete falsifier per sub-capability**, with columns:
`layer | feature | original source § | existing code path | actual execution | oracle | observed outcome | missing contract`.

1. Run NAL-1 truth/revision with *same root via two paths*; compare with ONA on overlapping grammar, and enforce no echo.
2. Run NAL-2/3/4 in a typed, minimal abstract language: directionality, compounds, product/image round trip.
3. Run NAL-5/6 with explicit proposition identity and scoped variables.
4. Run NAL-7 with temporal permutations and observed/not-observed distinction.
5. Run NAL-8 with denied/failed/achieved operation and intent-vs-causes guard.
6. Run NAL-9 with a self-diagnostic event, a stored receipt, a deliberate policy choice, and before/after ablation.
7. For all: count primitives used, masks/folds, allocations, output byte identity, snapshot and version, provenance roots, mutations, and reasons for abstention.

**Scoring rule:** capability ∈ {not represented, representable but not executable, executable unverified, verified on a declared subset, contradicted}. Never turn "possible with folds" into "already implemented."

## Crucial negative controls

- An 8-step chain from the 1,000 NARS benchmark is not automatically NAL-8.
- A recorded trace is not NAL-9 self-awareness.
- A causal-looking implication is not Pearl intervention evidence.
- Confidence byte 255 does not guarantee an infinite independent evidential basis.
- An active focus cannot write on behalf of several owners without separate authorization.
- A clever bit-layout match cannot substitute for proof of semantic equivalence.

## Source vs interpretation

**SOURCE:** Nine level descriptions and NARS AIKR architecture from Wang 2026; NAL-9 implementation components from his presentation.  
**INFERENCE:** Proposed mapping to lance-graph, tests, dependency table and implementation strategy.  
**NOT VERIFIED:** Complete contents of the 2025 copyrighted book, all NAL rules, or full NAL-9 behavior in current lance-graph.

## Further reading

- Wang 2026 introductory exposition: https://cis.temple.edu/~pwang/NARS-Intro.html
- Wang 2025 monograph information: https://cis.temple.edu/~pwang/papers.html
- Wang NAL-9 slides: https://cis.temple.edu/tagit/presentations/NAL%209.pdf
- Wang, Li & Hammer (2018), *Self in NARS*: https://doi.org/10.3389/frobt.2018.00020
- Gabriel & Wang (2026), benchmark: https://arxiv.org/abs/2604.18873
