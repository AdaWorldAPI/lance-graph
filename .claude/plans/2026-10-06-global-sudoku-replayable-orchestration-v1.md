# Global Sudoku Replayable Orchestration — Wiring Plan v1

**Status:** architecture map and falsification plan.  
**Date:** 2026-10-06.  
**D-ids:** D-GSO-0..8, one per §18 proof step P0..P8 (board: `STATUS_BOARD.md` § D-GSO).  
**Scope:** cross-layer wiring only. This document does **not** allocate new bits, canonize a 0..63 encoding, create a new VM, or authorize a new write path.

## North star

The intelligence of the substrate is not a giant object, graph, weight matrix, or opaque reasoner.

It is the deterministic policy that chooses **when, where, why, and in which order** a small finite set of epistemic operations is applied to a hydrated 64k knowledge field.

The target is:

~~~text
64k hydrated knowledge field
        ↓
detect a structural / evidential tension
        ↓
choose recipe 0..63
        ↓
project / compare / propagate / test
        ↓
earn, preserve, downgrade, or refuse a stronger claim
        ↓
seal only at replay-significant boundaries
~~~

The substrate should be self-learning because observation changes evidence and local affinity, yet deterministic because the same sealed world, recipe policy, codebook/LUT generations, and external evidence must replay to the same result.

**Determinism is not truth.** The stronger requirement is:

> No claim may be stronger than the evidence and causal proof obligations that earned its reasoning band.

---

## 1. Keep the layers separate

The working decomposition is:

~~~text
PHYSICAL        resident bytes / columns / 64k ordinals
SEMANTIC        views over those bytes
ALGEBRAIC       closed laws / folds
KINEMATIC       schedules: who interacts with whom and when
EPISTEMIC       evidence, topology, reasoning permission
ORCHESTRATION   recipe 0..63 selects the bounded experiment
OBSERVATIONAL   terminal / revision / counterfactual verdict
DURABLE         Rubikon/seal boundary
~~~

Only the physical/durable layers are necessarily material.

A view must not become a second owner of the same bytes.
A closed schedule must not become a stored graph.
A finite product must not become a matrix.
A recipe must not become a 'Vec<Instruction>' unless a falsifier proves that an ordinal cannot regenerate the program.

---

## 2. Landed exemplars: the first three proofs

These are not the whole architecture. They are the first demonstrations of one recurring law:

> delete the intermediate; keep the rule that regenerates it.

### #1336 — closed value law

'palette_perturbation' proves a compact calibrated value law can stay closed:

~~~text
8:8 address
   ↓
Palette256 relation
   ↓
u8 state
~~~

No runtime cosine/Fisher-Z reconstruction and no giant cross product.

### #1337 / #1342 — implicit finite product

'Quad8' / 'ProductAddress12' proves the finite product can stay implicit:

~~~text
[a][b][c][d]
    ↓
8 × 8 × 8 × 8 possible ordinal product
    ↓
generate only occupied ProductAddress12 values
    ↓
consume immediately
~~~

Naming guardrail:

- **Cartesian** is reserved for the HHTL16 spatial coordinate model.
- Morton is spatial ordering/interleaving.
- Quad8 is a finite bit-lane / ordinal product.

Do not reintroduce "Cartesian" as a generic synonym here.

### #1343 — implicit reversible interaction schedule

'Switch16' over existing 'Register128' proves a closed causal topology does not need an edge population:

~~~text
stage ordinal
   ↓
closed-form stride
   ↓
pair ordinal
   ↓
(left lane, right lane)
   ↓
bijective local step
~~~

The 56 interactions exist mathematically, not as '[SwitchEdge; 8]'.

Critical vocabulary:

~~~text
(stage, pair) = schedule ordinals
not spatial coordinates
~~~

The graph is a view/schedule when topology is regenerable. A real graph earns materialization only when topology itself carries information that a closed schedule cannot regenerate.

---

## 3. The 64k field

Treat 0..65535 primarily as a finite hydrated epistemic address space, not as a mandate to allocate 65,536 node objects plus explicit edges.

Conceptually:

~~~text
ordinal universe 0..65535
        │
        ├─ ontology / schema says what is structurally admissible
        ├─ evidence says what has actually been observed
        ├─ local affinity says what co-activates / agrees nearby
        ├─ causal topology says how a relation is grounded
        └─ temporal/seal says what world generation was visible
~~~

A useful division of labour:

> Schema wants to become geometry; evidence stays population.

The ontology can hydrate a large amount of known structure once. Reasoning then measures local and global consistency against that structure rather than repeatedly rebuilding triples as objects.

---

## 4. Tenant16 / Register128 is a resident state, not a family of owner types

The 16 resident bytes may admit several zero-copy readings:

~~~text
same 16 bytes

├─ 16 × u8 lanes
├─ 8 × 8:8 pair addresses
├─ 4 × Quad8 product views
└─ Switch16 schedule over 16 lanes
~~~

Future local-plasticity experiments may add a Moore-style reading, but it must remain a **view/schedule over resident state**, not a second 16-byte storage owner.

Do not resurrect the old '6 × (2 × 8-bit)' shape as if Tenant16 were merely "two more rails". The power-of-two 16-lane resident state is a different and richer substrate.

---

## 5. Local learning: Moore × 8:8 Palette

The missing lateral axis is local plasticity.

Hierarchy gives reduction / composition:

~~~text
16 → 8 → 4 → 2 → 1
~~~

Moore-like neighbourhood interaction breaks pure hierarchy and provides local adaptation:

~~~text
      NW N NE
       \ | /
     W - ● - E
       / | \
      SW S SE
~~~

The intended shape is **not** eight edge objects:

~~~text
neighbor mask
    ↓
neighbor ordinal
    ↓
8:8 local relation address
    ↓
Palette256 / local law
    ↓
immediate fold / permitted plastic update
~~~

A single local hop may be implicit and closed.
"Repeat until stable" is a recurrent/cyclic residual and must remain semantically explicit.

### Separation of roles

~~~text
hierarchy / ontology   = global admissible structure
Moore locality         = local neighbourhood interaction
Palette256             = calibrated local relation law
plasticity gate        = whether local affinity may change
~~~

Do not turn Palette256 into a global weight matrix.

---

## 6. Observation learning: SPOFC

SPOFC is the bridge from observation to learning.

The intended conceptual flow is:

~~~text
observation
   ↓
S / P / O identity
   ↓
F / C evidence accumulation
   ↓
local Hebbian-style co-activation
   ↓
Bayesian / NARS-style calibration
   ↓
agreement / disagreement
   ↓
only then: causal proof obligations
~~~

Important guardrail:

> Repetition may strengthen association. Repetition alone must never promote correlation into causality.

Hebbian learning is local affinity accumulation.
Bayesian/NARS learning calibrates support against alternatives/evidence volume.
Neither one is permission to skip Pearl/counterfactual requirements.

---

## 7. Pearl 2³ projection is the experiment selector, not a predicate taxonomy

CausalEdge64 bits 40..42 are the 3-bit S/P/O projection surface: Pearl's 2³ decomposition.

The eight masks let one relation be interrogated under different projections without minting eight relation types.

Conceptually:

~~~text
000 .. 111
   ↓
which S/P/O planes participate in this test?
   ↓
association / intervention / confounder / counterfactual-shaped checks
~~~

The exact semantic mapping must follow the current 'causal-edge' implementation and ratified specs. Do not infer a new mapping from this document alone.

The important law is:

> The same evidence can be replayed under a different causal projection, and that difference is part of the experiment.

---

## 8. Bits 59..60 and 61..63: topology versus epistemic permission

The current board doctrine already separates these jobs:

~~~text
ENTROPY      WHERE the field is suspicious
59..60       WHAT kind of causal/topological hole exists
61..63       HOW strong a candidate is permitted to be read
~~~

This document preserves that separation.

### 59..60

These bits are topology / grounding shape, not confidence.

They distinguish known/direct structure from projected/unknown-intermediate structure and unresolved holes under the declared reading.

Do not turn them into another score.

### 61..63

The reasoning band is a **permission band**, not "confidence = 0.83".

The intended progression is from weak observational/correlational entitlement toward stronger causal/counterfactual entitlement, but the exact eight labels must be earned from existing tests and current layout specs before being frozen.

Critical invariant:

~~~text
reasoning_band may rise
ONLY IF its proof obligation was actually executed and passed
~~~

It may also fall under contradictory/new evidence.

The monotone quantity is not certainty. It is **epistemic accountability**: every stronger reading carries more replayable justification.

---

## 9. Agreement / disagreement is measured at multiple scales

The global Sudoku is not one local score.

For a concept such as 'mammal', evidence can accumulate from:

- descendants / ancestors,
- is_a / part_of structure,
- local semantic neighbours,
- SPOFC observations,
- calibrated relation tables,
- causal projections,
- temporal compatibility,
- independent roots versus inherited echoes.

The implementation direction is:

~~~text
address
  ↓
read contribution
  ↓
classify support / opposition / unknown
  ↓
fold immediately
~~~

Do not materialize a giant "Contribution" population merely because the conceptual model has contributions.

### Local versus global

~~~text
H_local  = tension in local neighbourhood
H_global = tension in accumulated ontology / evidence basin
~~~

Their disagreement is often more useful than either scalar alone.

Examples:

~~~text
low H_global + high H_local
→ local exception / missing relation / boundary

high H_global + low H_local
→ locally plausible but globally inconsistent ontology basin

high H_global + high H_local
→ high-value epistemic hotspot
~~~

---

## 10. Shannon entropy is search pressure, not a truth oracle

Entropy should primarily answer:

> Where is another bounded experiment worth spending computation?

Never:

> This node has low entropy, therefore it is known.

This is already falsified by the "Glass" condition: dense closure on thin evidence.

Therefore:

~~~text
entropy
   ↓
candidate pressure / scheduling priority

entropy + topology + evidence competence
   ↓
epistemic interpretation
~~~

Two equal-entropy basins with different 59..60 topology must be allowed to route differently.

This is a required can-fire falsifier for any recipe selector.

---

## 11. The global Sudoku loop

The architecture should treat "Sudoku" as an exact operational metaphor:

~~~text
GIVEN
    observed evidence

RULES
    ontology / schema / causal / temporal constraints

FIELD
    hydrated 64k ordinal space

HOLES
    legal but unoccupied or insufficiently grounded states

TEST
    replayable projection / propagation / counterfactual attack

OUTCOME
    accept stronger reading
    preserve contradiction
    downgrade
    keep unknown
    ask for missing means
~~~

The canonical loop is:

~~~text
DETECT
  entropy / agreement / topology mismatch
    ↓
BOUND
  59..60 + ontology + temporal + truth constraints
    ↓
PROPOSE
  finite candidate frontier
    ↓
FILTER
  evidence competence / ontology / provenance
    ↓
GATE
  reasoning-band permission
    ↓
TEST
  Pearl / intervention / counterfactual / removal
    ↓
REVISE
  new independent root? echo? contradiction? expansion?
    ↓
ACCEPT / DOWNGRADE / KEEP UNKNOWN / ASK
    ↓
optional SEAL
~~~

Narrowing 60,000 candidates to 7 is itself useful epistemic structure.
A hole never forces a guess.

---

## 12. Recipe 0..63: where the intelligence lives

0..63 should not become 64 poetic "thinking styles".

It is a finite, versioned orchestration surface for bounded epistemic experiments.

The intelligence lies in:

~~~text
WHEN which recipe?
WHERE on the field?
WHY this recipe?
AFTER which previous outcome?
UNTIL which falsifier or terminal?
~~~

A recipe may select/combine:

- view,
- Pearl projection,
- local versus global neighbourhood,
- mask,
- propagation schedule,
- closed law,
- agreement/disagreement fold,
- entropy measurement,
- frontier test,
- counterfactual/revision obligation,
- terminal / seal decision.

But **do not freeze the 6-bit encoding from aesthetics**.

Derive the 64 recipes from a falsification matrix of Sudoku classes and required operations.

### Non-materialized program law

The target is:

~~~text
recipe ordinal + context
        ↓
deterministic execution path
~~~

not:

~~~text
recipe ordinal
        ↓
allocate Vec<Op>
        ↓
interpret Vec<Op>
~~~

If a recipe can regenerate its operation schedule, the instruction population is an unnecessary intermediate.

---

## 13. Replay contract

A thought cycle becomes scientifically useful only when it can be replayed as the same bounded experiment.

At minimum, replay identity must account for the semantically relevant generations:

~~~text
sealed/base world version
recipe ordinal
recipe-policy version
codebook / Palette LUT generation
ontology/schema generation
external independent evidence identity/digest
~~~

Potentially additional generation IDs may be required by actual consumers. Do not invent them pre-emptively; measure which mutable surfaces affect the result.

Required property:

> Same replay identity and same external evidence produce the same execution path and result.

Again, that proves determinism, not truth.

Truth discipline comes from never allowing a band/gate to claim more than the replayed proof docket supports.

---

## 14. Revision and counterfactual must be causally necessary, not decorative

A wiring probe is not end-to-end merely because it calls 'revision.rs'.

The output of revision must influence the next replay-visible state.

Required falsifier:

> Remove/bypass the revision leg and the claimed post-revision replay result must fail or differ.

This applies directly to the current #1344 DAV × EWA × revision probe.

Desired vertical slice:

~~~text
epistemic frontier
   ↓
EWA/locality bound
   ↓
deterministic disagreement
   ↓
observation / new independent root
   ↓
GadamerRevision
   ↓
revision output becomes the only input to replay-visible state
   ↓
replay
   ↓
disagreement collapse / changed epistemic status
~~~

DAV/ranking may select where to look.
It must never mint evidence.

'revision.rs' makes an encounter eligible for stronger interpretation; existing counterfactual/adjudication contracts still govern whether a stronger reality claim may be accepted.

---

## 15. Rubikon / seal: materialization governor

Most internal deterministic operations should remain immaterial.

A semantic boundary earns durability.

Examples of candidate boundaries:

- independent external observation,
- accepted causal intervention result,
- human-visible decision requiring audit,
- non-associative residual that cannot be regenerated from bounded state,
- generation transition,
- replay checkpoint / published cohort.

Conceptually:

~~~text
many deterministic operations
          ↓
     no durable writes
          ↓
semantic / replay boundary
          ↓
         SEAL
~~~

Do not collapse 'temporal.rs' and seal ordering into one concept.

Current knowledge explicitly distinguishes:

- temporal = what a reader may see / per-owner temporal model,
- seal = total durable ordering, row fold, cohort, base-version boundary.

---

## 16. Guardrails

1. **NO NEW OBJECT** until an ordinal/view cannot express the state.
2. **NO MATERIALIZATION** until a terminal or irreducible residual requires it.
3. **NO NEW PRIMITIVE** until existing Local/Via/mask/fold laws are proven insufficient.
4. **NO STORED GRAPH** when topology has a closed schedule.
5. **NO PROGRAM VECTOR** when a recipe ordinal can regenerate the program.
6. **NO CAUSAL PROMOTION FROM REPETITION ALONE.**
7. **NO ENTROPY-AS-TRUTH.**
8. **59..60 ARE TOPOLOGY, NOT CONFIDENCE.**
9. **61..63 ARE PERMISSION, NOT A FLOAT SCORE.**
10. **NO REVISION THEATRE:** if bypassing revision leaves the same claimed result, it was not wired.
11. **NO FALSE REPLAY CLAIM:** if hidden arrival/order/generation affects the result, it belongs in the replay identity or must be eliminated.
12. **NO NEW CAUSALEDGE64 LAYOUT FROM THIS PLAN:** verify current shipped layout/spec before touching bits.
13. **CARTESIAN REMAINS HHTL16 VOCABULARY.**

---

## 17. Falsifiers that should kill an implementation

An implementation is wrong if any of these hold:

- a Moore/local test allocates a pair list merely to visit eight neighbours;
- a closed Switch16-like topology materializes edges;
- a Quad8-like finite product materializes the full product before folding;
- recipe execution first builds a 'Vec<Instruction>' without proving it is irreducible;
- repeated SPOFC co-occurrence raises causal permission without a causal proof obligation;
- entropy alone raises the reasoning band;
- two identical-entropy basins with Direct versus Unknown topology are routed identically when the policy claims grounding awareness;
- an inherited/echo root is counted as a new independent root;
- a counterfactual "test" is reported without actually running the removal/attack;
- the revision leg can be deleted and replay still claims the same revision-dependent result;
- the same replay identity yields a different recipe path or terminal;
- local plasticity silently rewrites ontology/schema structure;
- the system fills a legal unknown because it lacks evidence to leave it unknown.

---

## 18. Recommended proof sequence

Do not build the grand abstraction first.

### P0 (D-GSO-0) — document the wiring

This file. No ABI changes.

### P1 (D-GSO-1) — make #1344 causally end-to-end

Fix only:

- fmt/clippy,
- revision output must be causally necessary for replay collapse.

No new primitive or write path.

### P2 (D-GSO-2) — one-hop Moore local-plasticity probe

Use existing resident bytes / existing Palette law where possible.

Prove:

~~~text
neighbor mask → ordinal → 8:8 relation → immediate fold/update
~~~

No edge list. No recurrent "until stable".

### P3 (D-GSO-3) — global ontology agreement/disagreement probe

Use a small but structurally representative hierarchy (the mammal family is an appropriate synthetic/ontology fixture).

Measure support/opposition/unknown upward and downward without creating a population of path objects.

The test should distinguish local agreement from accumulated global agreement.

### P4 (D-GSO-4) — entropy × topology routing probe

Construct equal-entropy basins with different causal topology/grounding.

The selector must route them differently.

This is the Glass-versus-earned-closure anti-vacuity test.

### P5 (D-GSO-5) — first recipe quartet, not all 64

Prove four qualitatively different recipes can orchestrate existing primitives without creating an instruction vector.

For example, only as a test scaffold:

~~~text
R0 observe/fold
R1 finite-product interrogation
R2 local Palette/Moore interrogation
R3 causal projection + revision test
~~~

Do **not** canonize these ordinal meanings yet.

### P6 (D-GSO-6) — deterministic recipe selector

Make recipe selection a pure/versioned function of declared epistemic state.

Replay must choose the same recipe for the same input state.

### P7 (D-GSO-7) — reasoning-band earning/downgrade

Prove at least:

- association evidence cannot skip directly to causal permission,
- a required Pearl/counterfactual test can raise permission,
- contradictory independent evidence can lower/suspend it,
- band changes are replayable.

### P8 (D-GSO-8) — seal boundary

Run many internal operations without durable materialization, then seal only at one declared semantic boundary.

Replay from the prior seal must reproduce the next seal.

Only after these probes survive should common 'View/Law/Schedule/Recipe' abstractions be considered.

---

## 19. What not to build yet

- no universal cognitive VM;
- no generic 'View' mega-trait;
- no generic 'Schedule' framework because three examples happen to rhyme;
- no 'ThoughtRecipe' object with dozens of fields;
- no 64-entry hand-authored mythology;
- no new graph storage;
- no new float confidence for reasoning band;
- no global Moore relaxation engine;
- no new CausalEdge64 bit allocation;
- no "entropy controller" that bypasses topology/evidence competence;
- no production write path from a diagnostic probe.

First make the fourth, fifth, and sixth independent examples force the abstraction.

---

## 20. Architectural map

~~~text
                         64k HYDRATED FIELD
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
      GLOBAL / ONTOLOGY                    LOCAL / MOORE
 hierarchy, is_a, part_of               8-neighbour locality
 accumulated agreement                  Palette 8:8 relation
              │                                   │
              └─────────────────┬─────────────────┘
                                │
                              SPOFC
                    observation → evidence
                                │
                    Hebbian local learning
                                │
                    Bayesian/NARS calibration
                                │
                   agreement / disagreement
                                │
                         Shannon entropy
                       WHERE to inspect
                                │
                      CE64 topology 59..60
                       WHAT kind of hole
                                │
                        RECIPE POLICY
                             0..63
                WHEN / WHERE / HOW to test
                                │
                       Pearl S/P/O 2³
                                │
                 bounded Sudoku experiment
                                │
                 counterfactual / revision
                                │
                      reasoning band 61..63
                    WHAT may now be claimed
                                │
                ┌───────────────┴───────────────┐
                │                               │
          continue puzzle                  semantic boundary
                │                               │
                └───────────────┬───────────────┘
                                │
                           RUBIKON / SEAL
                                │
                              REPLAY
~~~

The desired end state is not "an AGI graph database".

It is:

> **A finite deterministic knowledge substrate that learns from observation, measures its own local and global disagreement, identifies legal epistemic holes, spends computation where uncertainty and grounding warrant it, and only earns stronger causal readings through replayable tests.**

The cleverness belongs in the **quality of the 0..63 orchestration policy**, not in hidden mutable intermediates.

---

## 21. Current anchors / reading order

Before implementing from this plan, read current source and these documents; they are more authoritative than old session prose:

- 'crates/cognitive-shader-driver/src/palette_perturbation.rs' — #1336 value-law exemplar.
- 'crates/cognitive-shader-driver/src/quad8.rs' — #1337/#1342 finite-product exemplar.
- 'crates/cognitive-shader-driver/src/switch16.rs' — #1343 closed schedule exemplar.
- 'crates/lance-graph-contract/src/register128.rs' — resident 16-byte register.
- 'crates/causal-edge/src/edge.rs', 'pearl.rs', 'plasticity.rs' — current shipped causal-edge semantics.
- 'crates/lance-graph-contract/src/revision.rs' and 'counterfactual.rs'.
- '.claude/board/entries/2026-08-26-e-entropy-measures-closure-bits-59-60-tell-whether-the-closure-has-causal-footing-1.md'.
- '.claude/board/entries/2026-08-20-e-the-recipe-surface-is-causally-blind-1.md'.
- '.claude/knowledge/seal-vs-temporal-ordering-information.md'.
- '.grok/board/SUBSTRATE_SYNERGIES.md'.
- '.grok/board/IMMATERIAL_HANDOVER.md'.
- '.grok/board/HANDOVER_FALSIFICATION.md'.
- '.grok/board/LANE_FOLD_SEVENTEEN_ROOMS.md'.

When these disagree, measure current source and record the divergence. Do not silently harmonize history.
