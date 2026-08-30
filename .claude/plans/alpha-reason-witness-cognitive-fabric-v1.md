# alpha-reason-witness-cognitive-fabric-v1

> **Status:** PROPOSED / PLAN ONLY. No production implementation in this PR.
> **Date:** 2026-08-30.
> **Base audited from:** `main` at `de1d0c2fe54f36bf0d4d3f1393c55d0cea0c3ae9` plus explicitly named sibling-repo evidence below.
> **Relationship to #1077:** #1077 remains the source-audit / first-composition bridge. This plan owns the larger cross-repo question surfaced by that audit: **who owns reusable cognition once Alpha, graph-native derivation, witness identity, loco/R2IL, Revision, and domain observations are composed?**
> **Lineage addendum:** `.claude/plans/alpha-reason-witness-shader-field-lineage-addendum-v1.md` extends D-ARW-0/1/4/5/6/8 with source-fenced archaeology of the historical `StreamDto → ResonanceDto/PerturbationDto → P64/cognitive-shader-driver → CE64 → BusDto` field path, variable Morton aperture, stockfish-rs 64×64 reference ergonomics, and the fingerprint/OCR × EWA 2×2 covariance research arm. It mints no parallel D-id family.

## 0. Thesis

The next integration step is **not** another rung carrier, another planner controller, another provenance field, or another MedCare-specific reasoning path.

The hypothesis to test is narrower and stronger:

> **Domain repositories should supply grounded observations and domain projections. `lance-graph` should own reusable reasoning mechanics over canonical graph addresses. Ephemeral cognition should ride the existing Alpha/attention substrate; derivation should remain premise-addressable; source/witness identity should be recoverable through existing witness machinery; reasoning policy should be composable through the existing ogar-loco / R2IL membrane; persistent epistemic change must pass Revision and replay.**

This is a **discovery-first convergence plan**, not a representation proposal. Every attractive shape below remains a hypothesis until the source path proves it.

The desired end-to-end form is:

```text
MedCare/domain observation
        │
        ▼
canonical graph address
        │
        ▼
Alpha cognitive plane
        │
        ▼
reusable reasoning atom / loco-R2IL program
        │
        ├── premise pointers / Tarski derivation geometry
        ├── witness identity / provenance chain
        ├── NARS f,c
        ├── Shannon H
        ├── EWA Σ
        ├── CausalTopology
        ├── ReasoningBand
        └── cognitive rung / horizon
        │
        ▼
typed receipt
        │
        ▼
Revision
        │
        ▼
Rubicon / persistence / temporal replay
        │
        ▼
domain projection
```

No single scalar is allowed to stand in for this coordinate set.

---

## 1. Why this plan exists

The #1077 audit exposed a category error: a rung-level Alpha design was interpreted as ten value tenants / ten stored rung rows before the sibling implementation was checked. The sibling MedCare implementation already provides a concrete precedent for a **thin-provisioned, same-address Alpha overlay** over canonical `NodeRow` geometry, including a `rung: u8` stamp and mask-native readings. Whether that implementation is correctly wired into live cognition is a separate question.

The same audit also surfaced a likely ownership inversion in MedCare:

- ontology/domain facts are legitimately MedCare inputs;
- but parts of patient-specific horizon construction, observation landing, evidence-to-NARS reduction, next-cognition policy, and orchestration may be handrolled generic cognition;
- `deepnsm-v2/src/reason.rs` already provides a graph-native precedent for reusable derivation: premise pointers, bounded closure, deduplication, and a mechanically derived Tarski-style depth;
- the existing W-slot / `WitnessTable` history indicates that witness identity is richer than the current shorthand documentation suggests;
- #1077 already established ogar-loco as an existing byte-addressed call membrane and treats a second DSL as a STOP.

The integration question is therefore not “where do we store ten rungs?” It is:

> **Can one real domain observation enter an existing Alpha surface, be processed by reusable lance-graph reasoning, carry its derivational and witness coordinates without inventing new authority, pass Revision, replay, and return to the domain as a projection?**

If yes, that is the reusable pattern. If no, the failure must name the smallest missing primitive or contract.

---

## 2. Constitutional separations

The following axes are **orthogonal until a source-defined relation proves otherwise**:

| Axis | Question it answers | Must not be collapsed into |
|---|---|---|
| **Cognitive rung / horizon** | From what abstraction / temporal / perspective position is cognition reading? | Tarski depth, ReasoningBand |
| **Tarski derivation depth** | How far is this derivation from its admitted leaves / premises? | cognitive rung, confidence |
| **ReasoningBand** | What coarse reasoning reading is warranted? | provenance, source type |
| **CausalTopology** | What causal path shape is represented? | ReasoningBand, evidence |
| **NARS `(f,c)`** | What evidential support state is carried? | Shannon entropy, source identity |
| **Shannon `H`** | How uncertain is the current alternative distribution? | evidence gain, confidence |
| **EWA `Σ`** | What covariance / tension geometry propagates? | trust authority, gate verdict |
| **Alpha / residual** | What is temporarily salient / attended / unresolved? | persistent truth |
| **Witness / provenance** | Which event/source/receipt grounds or witnessed the claim? | Tarski depth, dataset version |
| **Temporal/version** | Which sealed state / reading horizon is being replayed? | evidence-event identity |

### 2.1 Tarski is a measurement, not the rung ladder

The existing `reason.rs` `rung` semantics are treated as **derivational depth**:

```text
base / no premises        depth = 0
derived                    depth = 1 + max(depth(premises))
```

This is valuable and should survive. The field name may eventually deserve clarification (`tarski_depth` / `derivation_height`), but this plan does **not** rename it as a prerequisite.

**STOP:** `reason.rs::rung` must never be equated with `cognitive_shader::RungLevel` merely because both use the word “rung.”

### 2.2 “Mechanical vs observed” is two questions

This plan distinguishes:

1. **Leaf vs derivation:** a Tarski-0 / no-premise assertion versus a premise-derived assertion. This may already be structurally recoverable and must not receive another redundant bit without proof.
2. **Nature of the leaf:** observed by device, imported ontology/reference, literature-attested, human-entered, oracle/reference fixture, etc. This belongs to witness/provenance/receipt semantics if existing machinery can express it.

**NO-BUY rule:** if Tarski + premise pointers + existing W-linked receipt identity can answer the required query, do not mint another “origin” field.

---

## 3. Alpha is a cognitive plane, not ten drawers

### 3.1 Known precedent to verify directly

The MedCare sibling implementation is the primary precedent to read, not a summary:

- `medcare-nodesoa::alpha`
- canonical `NodeRow` address reuse;
- sparse/thin-provisioned Alpha allocation;
- `AlphaStamp { cycle, seq, rung, visits }`;
- mask-native `AlphaMask` / `AlphaOverlay::attended_mask`;
- optional Lance persistence of claimed Alpha rows.

The architectural reading to test is Photoshop-like:

```text
canonical graph address A
        │
        └── ephemeral Alpha state at A
                attention / visits / rung / residual / perspective...
```

Not:

```text
A × rung1 tenant
A × rung2 tenant
...
A × rung10 tenant
```

### 3.2 What remains unknown

Do **not** assume before measurement whether multiple rung observations at one graph address coexist through:

- multiple Alpha rows,
- multiple overlays/sessions,
- masks/views,
- temporal versions,
- facets/classviews,
- an existing rail,
- or another shipped carrier.

The source audit must answer what `rung` actually indexes and what coexistence semantics exist today.

**STOP:** no `(classid, rail)` mint for rungs 1–10 until an actual producer/consumer path proves that existing Alpha/view/session machinery cannot represent the requirement.

---

## 4. `reason.rs` is a precedent, not automatically the target API

`deepnsm-v2/src/reason.rs` establishes a useful ownership precedent:

- reasoning is graph-native structure;
- derived facts retain premise pointers;
- derivation is bounded and deduplicated;
- depth is mechanically derived from premise geometry;
- soundness/acyclicity can be checked from the structure.

This plan must determine which aspects are reusable substrate law and which are local to its current transitive-derivation implementation.

**Do not** wire MedCare directly to `DerivationArena` simply because it exists.

Instead classify the pattern:

```text
DOMAIN INPUT
    facts / observations / typed clinical grounding

GENERIC REASONING
    land in horizon
    claim Alpha
    attach premise/backreference
    deduplicate
    derive / counterfactually attack / revise
    emit typed receipt

ORCHESTRATION
    choose what cognition executes next
    without deciding truth itself

PROJECTION
    render / expose domain-specific neighborhood or result
```

Only operations proven generic should move into or be exposed by lance-graph.

---

## 5. MedCare ownership audit

Audit the live MedCare path around these surfaces before proposing code:

```text
PatientSession / patient graph opening
    ↓
observe_patient
    ↓
walk_resident / walk_resident_for_patient
    ↓
Alpha claim / ReasoningNeighborhood population
    ↓
patient_evidence_truth
    ↓
patient_outcome / next cognition
    ↓
drive_cohort_thoughts
    ↓
#879-style lifecycle / persistence seam
```

For **every operation**, record exactly one owner class:

- `DOMAIN INPUT`
- `GENERIC REASONING`
- `ORCHESTRATION`
- `PROJECTION`

Then search lance-graph / ogar-loco / ogar-r2il / r2sleigh precedents before inventing any equivalent.

### 5.1 Expected domain-owned material

Likely MedCare-owned, pending source verification:

- patient identity/session scoping;
- clinical observation acquisition;
- ICD/LOINC/HPO/ATC/other domain grounding;
- clinical evidence-role mapping;
- domain-specific rendering/projection.

### 5.2 Suspected handrolls to test, not assert

Specifically test whether MedCare has independently implemented any generic form of:

- observation → Alpha claim;
- observation → reasoning edge;
- premise/backreference bookkeeping;
- deduplication;
- evidence → NARS reduction;
- NARS / uncertainty → next cognition;
- graph reasoning → Kanban transition;
- replay/history bookkeeping.

If an existing substrate operation is equivalent, the BUY is **reuse/convergence**, not another MedCare API.

---

## 6. W-slot / witness archaeology

The current repository carries contradictory-looking descriptions that must be reconciled before provenance design:

- CE64 bits 53..58 are the W-slot;
- `witness_table.rs` describes W as a **per-cohort index** resolving to `WitnessEntry { mailbox_ref, spo_fact_ref }`, with a backward-walkable belief-update arc;
- older architecture text explicitly says this is **NOT a witness-corpus pointer**;
- some current CE64 layout wording describes W as a “witness corpus root handle.”

The plan must recover the intended semantics from primary source and history.

Questions:

1. What is the W-slot’s canonical identity domain today?
2. Is `mailbox_ref` an event/witness identity, an owner identity, or only an indirection step?
3. When `spo_fact_ref` becomes `Some`, what exactly does crystallisation certify?
4. Where is source-kind/attribution stored today, if anywhere?
5. Can a Tarski-0 leaf be followed through W to a receipt that distinguishes observed/imported/reference/etc.?
6. Which part is event identity, which part is source attribution, which part is evidential-base membership, and which part is dependence?

Preserve the standing separation:

```text
event identity
≠ evidential-base membership
≠ source dependence
≠ object/view identity
≠ dataset version
```

**STOP:** do not allocate CE64/V3 bits for `OBSERVED`, `MECHANICAL`, or provenance until this audit is complete.

---

## 7. Atom / program convergence

The reusable reasoning mechanics discovered in this audit should first be evaluated against the existing execution membrane:

```text
ogar-loco FnIndex / vocabulary
        ↓
ogar-r2il / R2IL composition where applicable
        ↓
hardened native atom
        ↓
typed result / receipt
```

Candidate mechanisms include, but are not limited to:

- Shannon `H(p)`;
- mask intersection/difference where semantically lawful;
- Alpha claim/read/composition;
- Tarski depth / premise walk;
- NARS deduction/induction/abduction/revision primitives;
- counterfactual application;
- EWA covariance/tension operations;
- CausalTopology read;
- ReasoningBand read;
- Revision admissibility tests.

The exact atom census belongs to #1077 and this plan consumes it. This plan should **not** create a competing atom registry.

### 7.1 Frozen remains lifecycle/authority, not “Rust club”

A primitive may have a hardened native implementation. A Frozen program may also be a versioned, validated composition over primitives. The plan must not equate “Frozen” with “implemented directly in Rust forever.”

### 7.2 Constitutional law remains unskippable

Even if mechanisms are callable atoms, the following must not become optional bytecode conventions:

- no empirical evidence minting from internal reinterpretation alone;
- typed provenance/identity separation;
- Revision before persistent epistemic authority where the lifecycle requires it;
- replay/version binding;
- Rubicon as the durable action/commit boundary.

Rule of thumb:

> **Make the mechanism callable; make the law unskippable.**

---

## 8. Rungs, Alpha, and metacognition

The plan carries forward the #1077 separation:

```text
rung / horizon     = where cognition reads from
program / atom     = what cognition executes
Alpha              = temporary working plane
orchestration      = what becomes ready / executes next
receipt            = what happened and under what warrant
Revision           = what may change persistent epistemic state
```

No rung owns the right to think. Any rung may invoke an admissible atom/program if its inputs and warrant permit it.

A future scheduler, if any, may:

- prefetch;
- queue;
- wake;
- co-locate;
- prioritize.

It must not decide:

- truth;
- causality;
- independence;
- ReasoningBand promotion;
- Revision acceptance.

**RUNG PROPAGATION ≠ RUNG PROMOTION.**
**SCHEDULER ≠ REASONER.**
**VISIBILITY ≠ MUTABILITY.**

---

## 9. Revision and replay are the hinge

The target integration only counts if the result survives the complete epistemic lifecycle:

```text
observation / prior
    ↓
Alpha / reasoning
    ↓
typed receipt
    ↓
Evaluation
    ↓
Revision
    ↓
Rubicon / seal
    ↓
temporal replay
    ↓
next cycle prior
```

An Alpha row existing is not a belief. A beautiful derivation is not an observation. A lower Shannon entropy is not evidence gain. A high ReasoningBand is not source authority.

The first BUY must prove that the next cycle can reconstruct **what was reasoned, from which premises/witnesses, under which program/version, and what Revision did with it**.

---

## 10. Waves

### W0 — primary-source ownership census

Read the actual live files, not prior plan prose, for:

- lance-graph `reason.rs` and consumers;
- CE64/V3 W-slot layout and `WitnessTable`;
- Alpha / attention / mask primitives in lance-graph;
- Revision / fusion / Rubicon / #879 sealed-cycle path;
- ogar-loco / ogar-r2il / R2IL entry points;
- MedCare `medcare-nodesoa::alpha` and its real producers/consumers;
- MedCare `PatientSession`, patient observation, patient reasoning, NARS/policy, and orchestration path;
- the shader-field lineage addendum's `StreamDto → ResonanceDto/PerturbationDto → P64/cognitive-shader-driver → CE64 → BusDto` historical/current path, including whether CE64 truly occupied the operator-recalled “vertical mantissa” role and whether `top_k` ever selected the Morton cascade aperture.

**Output:** one ownership table with `SOURCE FACT`, `PLAN FACT`, `MEASURED ABSENCE`, or `HYPOTHESIS` for every claim.

**F-ARW-0:** any absence claim based on a single repo or grep-only census fails the wave.

### W1 — Alpha semantic convergence

Determine the actual same-address Alpha geometry and what `rung` indexes.

Measure:

- address identity;
- overlay/session identity;
- sparse occupancy;
- multiple-rung coexistence semantics;
- mask/view semantics;
- actual producer → Alpha → consumer paths;
- persistence/version semantics where present;
- whether Alpha is historically continuous with Resonance/Perturbation field semantics or only a later, composable sibling mechanism.

**NO-BUY:** if the shipped Alpha/view/session machinery already expresses the desired multi-rung state, no new rung carrier/rail/tenant is minted.

### W2 — reasoning ownership cut

Classify the MedCare path operation by operation. For every `GENERIC REASONING` operation implemented in MedCare, search for an existing lance-graph/loco/R2IL equivalent.

**F-ARW-2:** if a proposed move into lance-graph requires importing clinical vocabulary or patient-specific policy into the generic layer, the ownership cut is wrong.

### W3 — witness/source semantics recovery

Reconcile W-slot documentation and trace one leaf through witness identity to any available receipt/source attribution.

**F-ARW-3A:** if `witness_table.rs` and CE64 layout docs disagree, documentation is not treated as settled contract until history/consumer behavior resolves it.

**F-ARW-3B:** if existing witness/receipt identity can answer the source query, new provenance bits/types are a NO-BUY.

### W4 — orthogonal-coordinate falsifiers

Construct paired cases proving the axes do not collapse:

- same Tarski depth, different source provenance;
- same provenance class, different Tarski depth;
- same Shannon H, different evidential status;
- same NARS `(f,c)`, different derivational depth;
- same ReasoningBand, different source/provenance;
- same cognitive rung, different Tarski depth;
- high Tarski depth with a contradictory low-depth observation;
- counterfactual elimination reducing H without minting empirical evidence;
- same semantic relation / CE64 reading under different field address permutations;
- same field energy/covariance with different epistemic warrant;
- same top-k cardinality under lawful versus randomized locality.

The purpose is not a new struct. It is to pin **non-equivalence**.

### W5 — atom/program reuse census

For each generic reasoning operation needed by the first end-to-end chain, classify:

- existing callable atom;
- existing operation not loco-addressable;
- existing R2IL composition;
- missing primitive;
- constitutional rule, therefore not optional program logic.

Include the shader-field addendum candidates before minting anything new:

- mask-native gather/scatter over a bounded active field;
- 2×2 / 4×4 local neighbourhood operations;
- structure/covariance tensor primitive where useful;
- EWA `Σ' = MΣMᵀ` propagation;
- top-k / cascade-aperture selection if historical/current ownership is proven;
- incremental local update versus full recomputation.

**F-ARW-5:** inconvenience of composition is not evidence for a new controller.

### W6 — ONE end-to-end BUY

Prove exactly one real path:

```text
one MedCare observation
    → canonical graph address
    → existing Alpha mechanism
    → one reusable lance-graph reasoning operation
    → premise / Tarski geometry
    → W / witness trace
    → typed receipt
    → Revision
    → canonical seal/persistence path
    → temporal replay
    → MedCare projection
```

If D-ARW-0 proves the historical shader-field identity, the path may additionally demonstrate the smallest lawful field pass. If not, W6 stays on the current Alpha path and the historical seam remains an independent restoration question.

No scheduler. No bulk migration. No universal `ReasoningContext`. No new DTO family.

The test must demonstrate both:

- **can-fire:** the observation changes the lawful cognitive/revision result;
- **can-stay-silent:** an irrelevant/missing observation does not fabricate a result.

### W7 — extract the reusable pattern

Only after W6 succeeds, name the smallest reusable API/pattern that allowed the chain to work.

Prefer:

- existing carrier + new free function/atom;
- existing type + new method;
- existing loco/R2IL composition;

before:

- new trait;
- new DTO;
- new tenant;
- new controller.

The domain adapter remains thin.

### W8 — optional second-domain falsifier

A reusable cognitive pattern earns its name only if a second non-MedCare caller can exercise it without importing MedCare semantics. This may be a small in-repo fixture or another sibling consumer.

The fingerprint/OCR and stockfish-rs arms in the lineage addendum may be used as **algorithmic reference/falsifier arms**, but they count as a second cognitive buyer only if they exercise the same cognitive contract rather than merely reusing the same local math.

If no second buyer exists yet, mark the pattern **MedCare-proven / generality-unproven**, not universal.

---

## 11. STOP gates

The plan stops and asks for an operator decision if any implementation proposes:

1. a new rung tenant/rail/classid before W1 proves existing Alpha/view geometry insufficient;
2. a new provenance/origin bit before W3 exhausts W-slot + receipt semantics;
3. a new reasoning controller because loco/R2IL composition is inconvenient;
4. a second scheduler semantics;
5. collapsing Tarski depth into cognitive rung or ReasoningBand;
6. collapsing Shannon, NARS, EWA, source provenance, and ReasoningBand into a “trust” scalar;
7. treating Alpha presence as evidence;
8. treating counterfactual elimination as empirical observation;
9. moving clinical vocabulary or patient policy into the generic graph layer;
10. bypassing Revision/Rubicon constraints because a derivation is internally consistent;
11. inferring a workspace-wide absence after checking only one repository;
12. adding a new DTO/trait/tenant before a real producer and consumer are named;
13. promoting “CE64 vertical mantissa,” “top-k = Morton aperture,” or “Alpha = Resonance/Perturbation continuation” from reconstruction to contract without a source/measurement trace;
14. treating Stockfish as evidence that Morton/inverse-pyramid addressing exists there rather than as an operational reference;
15. treating fingerprint/EWA field geometry as epistemic authority.

---

## 12. Deliverables

| D-id | Deliverable | Initial status |
|---|---|---|
| **D-ARW-0** | cross-repo primary-source ownership map (`SOURCE FACT` / `PLAN FACT` / `MEASURED ABSENCE` / `HYPOTHESIS`), including shader-field/DTO ancestry trace | Queued |
| **D-ARW-1** | Alpha same-address/rung semantics audit across lance-graph + MedCare; no representation assumption; test historical field-plane continuity separately | Queued |
| **D-ARW-2** | MedCare reasoning-handroll census and ownership cut | Queued |
| **D-ARW-3** | W-slot / witness / source-semantics archaeology + documentation reconciliation | Queued |
| **D-ARW-4** | orthogonal-coordinate falsifier matrix (rung/Tarski/Band/H/NARS/Σ/Alpha/W/version + address/aperture/field geometry) | Queued |
| **D-ARW-5** | atom/loco/R2IL reuse census for the first real chain, including bounded-field/local-operator candidates | Queued |
| **D-ARW-6** | one MedCare observation → Alpha → reusable reasoning → witness/premises → Revision → replay → projection proof | Queued |
| **D-ARW-7** | smallest reusable lance-graph cognitive pattern extracted from D-ARW-6 | Held on D-ARW-6 |
| **D-ARW-8** | optional second-domain falsifier of claimed reuse; imaging/chess math alone does not satisfy cognitive-buyer criterion | Held on D-ARW-7 |

---

## 13. Success criterion

This plan succeeds when a domain repository can change **what is observed** without owning a bespoke implementation of **how the graph reasons**, and when a reasoning program can change/version/replay without recompiling bespoke domain/planner policy unless a genuinely new primitive or constitutional rule is required.

The smallest winning sentence is:

> **The domain supplies facts; Alpha holds reversible cognition; lance-graph owns reusable reasoning over those facts; witness/premise geometry explains the derivation; Revision decides what survives; replay proves it happened.**

The lineage addendum adds a second, deliberately conditional sentence:

> **If source archaeology confirms the old field seam, the cognitive field supplies bounded horizontal activity, CE64 supplies the vertical semantic carrier, and the shader/program transforms the selected field without acquiring authority to rewrite truth.**

If the first real chain needs less machinery than this plan expects, delete the excess. The goal is not to complete the diagram. The goal is to discover the smallest substrate law that makes the diagram inevitable.
