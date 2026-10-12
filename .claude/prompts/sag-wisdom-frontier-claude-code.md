# Claude Code handoff — D-SAG-2: wisdom-frontier / OpenPsi / Rubikon / CARE / ONA / Hyperon

**Objective:** Turn the merged D-SAG-1 software-awareness graph (#1509) into a *testable research path* toward evidence-grounded, intrinsically motivated self-inspection. Implement only verified, phase-gated, reversible probes. **Do not build a second graph engine.**

**Branch:** research/sag-openpsi-rubicon-care-wisdom-frontier  
**Upstream base:** merged #1509, commit 42fa2a347cda22eaaa327bba514f2029cc9daa82.  
**Required reading before edits:**
- .claude/plans/software-awareness-graph-v1.md
- .claude/plans/sag-wisdom-frontier-ona-hyperon-v1.md
- .arxiv/2026-10_openpsi_rubicon_care_wisdom_frontier_source_excerpts.md
- .claude/knowledge/maslov-pyramid-of-cognition.md
- .claude/plans/rubicon-volition-v1.md
- .arxiv/2606.14581_synthesize_programs_gate_commit.md
- .arxiv/2026-10_opencog_pln_shannon_attention_research_map.md
- .arxiv/2026-10_hyperon_metta_to_lance_graph_parity_map.md

## Your task

First give a **source-grounded code census** of the actual merged #1509 API: SoftwareGraph, Binder, MethodTable, Quack Draft, mask-risc program/terminal, class_view seal, provenance/root algebra, pearl::hydrate, experimental maturation, any current stage reading and the legal one-hop limit. Also inspect relevant OGAR and ONA/Hyperon source interfaces and **mark unverified capabilities as unverified**. Do not treat document plans as production APIs.

Then choose the **smallest independent probe** which can be implemented in the current test tree and compared to a scalar oracle. Preferred first target: **D-SAG-2-P1, stage-0 frontier popcount and evidence-root invariants**. Do not expand to live adapters, volition, EIG and CARE all in one code change.

### Substrate invariants (non-negotiable)

1. CE64[50..52] **currently means plasticity** in production. #1494 uses maturation 7..0 only under its own test-local explicit reading. Do not change this layout or conflate provenance. Do not put entropy/demand into that field.
2. CE64[59..63] has declared epistemic meaning. Never use it for preference, intention or action authority.
3. Register bytes are ISA/ALU carriers. ClassView, provenance and method signatures determine readings. No hidden semantic assumptions in raw register tests.
4. NARS frequency/confidence are not automatically calibrated Bayesian probabilities; Shannon entropy/EIG require an explicitly calibrated and versioned hypothesis distribution.
5. Derived edges inherit original observation roots, never mint a new independent root. No self/descendant support, including reopened grounded cycles.
6. Mask debt must operate on a separately typed prerequisite universe: missing = required & ~(known | refuted); blocked = required & refuted; debt = popcount(missing). A debt of 0 is not proof of truth.
7. SAG #1509 is **one hop** using existing Quack Draft/ScatterOrU32. A second hop must be represented by a *proven legal* relation carrier or explicitly blocked. No second traversal VM, custom graph executor, shadow methods registry or invented Quack operator.
8. An LLM-synthesized ranking **program** is not a decision, a decision is not a committed goal, and a committed goal is not permission to effect a real-world operation. Version/seal/time, capability and live authorization gates remain separate.
9. Compiled, verified and hypothesized SAG tiers remain distinct. No automatic application of H1 fusion. Do not claim a SIMD improvement before a fused terminal exists and runs.
10. No commits to main, no merge, no external effects; keep PR as draft while hypotheses and runtime parity remain open.

### P0: inspect, do not hallucinate

Report:
- exact paths/symbols and tests for graph projection and Quack lowering;
- whether root-set masks and ancestor/descendant guards are reusable as-is;
- whether the current source already has a safe bitset/popcount primitive, or whether a test-local scalar is needed;
- whether there is a legal population carrier for a second hop;
- status of ONA MeTTa-NARS bridge, OpenPsi PR #117, PLN/PeTTa and CARE reference;
- runtime, licensing and test commands needed to reproduce each foreign oracle;
- any difference between docs and production, with actual code evidence.

Do not modify this research plan to claim the census ran until you show paths, commit hashes and test output.

### P1: first runnable scalar falsifier (preferred)

Use tiny typed fixtures (not raw CE64 reinterpretation):
- A confirmed stage-0 anchor with B,C consequences; a frontier with 0,1,3 unknown prerequisite bits; one explicit refutation;
- duplicate evidence through two paths with one physical root;
- an ungrounded 3-cycle A->B->C->A;
- grounded 3-cycle then remove external anchor, preventing resurrection;
- a relation contradicted by new observation: only that relation enters stage 7 unless anchor itself is contradicted;
- repeated replay after fixed point;
- stale class/schema seal and row permutation.

Write an **independent scalar oracle**, then optionally reuse existing mask/popcount primitives; assert exact equality of candidate IDs, missing/refuted, root sets, maturity, outcome and refusal reasons. Assertions should check that zero new evidence yields no epistemic progress. A test-local custom stage marker is allowed only under explicit provenance, never on production-minted CE64.

At least one guard-disable run must fail before restoring the guard; document the mutation and actual RED test output.

### P2: Quack masking parity

Only after P1: use the genuine #1509 Binder/Draft path to project candidate rows. Demonstrate one-hop query parity with scalar oracle. If two-hop graph traversal needs a forbidden population intermediate, assert typed refusal or write a self-contained pre-approved carrier experiment without modifying hot-path contracts. Preserve row permutation and tier invariance, exact NULL / bag / set semantics and reading IDs. Account for full work including decoding, candidate construction, mask operations and refused lowerings.

### P3: cross-engine ONA <-> Hyperon / OpenPsi

Use **pinned upstream snapshots**. First compare native ONA vs ONA-through-MeTTa-NARS on shared fixtures; record Narsese event-time, f/c and output bindings *without normalization that changes meaning*. Then compare Hyperon/MeTTa and modern PLN only on carefully delimited equivalent observables; record explicit divergences and interpreter versions.

Official upstream entrypoints to inspect:
- https://github.com/opennars/OpenNARS-for-Applications (build.sh, evaluation.py)
- https://github.com/opennars/OpenNARS-for-Applications/releases/tag/v0.9.3
- https://github.com/patham9/nartech_ros/tree/master/plugins/narsplugintests
- https://github.com/trueagi-io/hyperon-experimental
- https://github.com/trueagi-io/PLN
- https://github.com/trueagi-io/PeTTa
- https://github.com/iCog-Labs-Dev/hyperon-openpsi (test/run-tests.py)
- https://github.com/iCog-Labs-Dev/hyperon-openpsi/pull/117 (open upstream change; do not presume merged)

No claim that CARE is officially implemented by ONA or Hyperon. It is a **new comparative harness design**.

### P4: distinguish the learning and choice mechanisms

Use a frozen inference engine, candidate set, calibration split, environment oracle and budget. Swap only scheduler:
- FIFO;
- confidence-only;
- minimum popcount debt;
- naïve entropy (negative control);
- properly calibrated expected information gain;
- EIG + OpenPsi demand and test-local maturation;
- optional CARE program challenger vs reference.

Then freeze scheduler and vary native reasoning engines in a separate experiment. Report irreducible noisy-TV decoy, rare useful lower-entropy query, delay vs impossibility, tie/iteration permutations, duplicate roots, candidate-order invariance, and full cost. Independent out-of-sample validation is mandatory for claimed improvement.

Explicit EIG target:
    p(Theta | D) over declared rival models;
    EIG(a) = H(Theta|D) - E_{o|D,a}[H(Theta|D,o)].
No hidden outcome look-ahead, no NARS-f/c-as-probability shortcut.

### P5: CARE gate + Rubikon + Kanban in simulation only

Reference policy independently ranks available candidates. A challenger program may propose an alternative; validate boundedness, finite scores on **every** candidate, order independence, schema/reading version, source-of-evidence cutoff and total costs. The reference-conditioned gate records immutable pre-outcome decision. A goal may then be committed, with a **separate** revocable if-then action rule; dispatch only after live permit/capability/cue checks. Outcome/receipt updates demand, evidence and Kanban state **after** the commit record is sealed.

Compare repeated deliberation, goal commit alone, + implementation cue, + disengagement/CARE auditing. Include authorized and unauthorized twins, stale policy, cancellation race and empty new-evidence cycle.

### P6: software self-inspection and useful proposals

After the above, ask the existing SoftwareGraph whether a method's repeated condition or compiler refusal reveals an actionable *investigation* (e.g., #1509's H1 selection fusion or unresolved capability). Let the graph **nominate**, never enact, an optimization; permit a separate run only once semantic equivalence, reproducible cost tier, NULL/bag/set obligations and human/contract authorization pass.

## Reporting format

For each probe, produce:
- question and falsifier ID (D-SAG-2-x);
- source revision, actual call path, declared semantic reading;
- exact input/original observation roots, candidate count and budget;
- independent scalar oracle expectation;
- passing **and negative/silence twin** tests;
- disable/RED/restore evidence;
- actual command, exit status and timing (if measured);
- measured complete cost, not microbenchmark only;
- remaining unresolved contract decisions and next smallest discriminating probe.

The final status table must distinguish **IMPLEMENTED / MEASURED / SOURCE-VERIFIED / RESEARCH / BLOCKED / REFUTED**. If foreign runtime cannot be obtained or compiled, say so rather than inventing results.

## Hard acceptance gates

- Zero production CE64 bit migration, unauthorized side effects, rule coercion, new engine or silent multi-hop.
- Zero false-independent observation roots and zero self-resolving cycles.
- All pre-outcome decisions replay identically under row permutation and after restart on the same snapshots.
- Demonstrated scheduler gain only under equal oracle, equal evidence and all-in cost.
- Literature notes state exactly what upstream papers demonstrate and what remains our hypothesis.
- No unproven full cognitive architecture claim merely because individual probes pass.

**Deliverable of the current research PR:** documentation, annotated .arxiv source shelf and this prompt. Future code P1-P6 is *proposed work*, not claimed delivered.
