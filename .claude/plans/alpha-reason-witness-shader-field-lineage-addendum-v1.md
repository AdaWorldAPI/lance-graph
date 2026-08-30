# alpha-reason-witness cognitive fabric — shader-field lineage addendum v1

> **Status:** ADDENDUM TO #1078 / PLAN ONLY. No production implementation.
> **Date:** 2026-08-30.
> **Owner plan:** `.claude/plans/alpha-reason-witness-cognitive-fabric-v1.md`.
> **Scope:** Recover, falsify, and if supported reconnect the historical cognitive-field / shader ancestry before inventing another cognition carrier.
>
> **Evidence discipline:** `SOURCE FACT` means current or historical source/PR text directly establishes the statement. `OPERATOR-RECALLED HYPOTHESIS` means the design intent was recalled in-session but has not yet been recovered verbatim from committed source. `TRANSFER HYPOTHESIS` means an algorithmic analogy that must win a probe before it can influence architecture.

---

## A0. Why this addendum exists

The #1078 discussion exposed a plausible older architecture hiding underneath several currently-disconnected surfaces:

```text
large graph / token / observation stream
        │
        ▼
StreamDto
        │
        ▼
ResonanceDto → PerturbationDto
  field energy / salience / top-k aperture
        │
        ▼
P64 / cognitive-shader-driver
  mask-native bounded working field
        │
        ▼
CausalEdge64 at active loci
  semantic / causal / epistemic carrier
        │
        ▼
BusDto
  projection across the execution membrane
```

This shape is **not canon yet**. The purpose of this addendum is to recover whether it was the actual intended architecture, identify exactly where the wire broke, and reuse only what survives source-first falsification.

The critical reconstruction question is:

> **Was the cognitive field intended to be a horizontal address/salience plane, with `CausalEdge64` as the vertical semantic/causal carrier at each active locus, while P64/cognitive-shader-driver applied bounded field operations and `BusDto` projected the result?**

---

## A1. SOURCE FACTS already recovered

### A1.1 The DTO seam is currently broken, not an ALU chain

Merged lance-graph PR #1051 audited `StreamDto` / `PerturbationDto` / `BusDto` and found the current path is transport/projection rather than a field-ALU chain:

```text
StreamDto.codebook_indices → ingest → BindSpace
PerturbationDto.energy: Vec<f32>       → dropped / unconsumed
PerturbationDto.top_k
    → threshold filter
    → min(index)..=max(index)
    → ColumnWindow                     // bounding window, not field mask
ShaderBus → BusDto → dispatch_busdto
    → Binary16K + qualia
```

The same audit established that P64 is the mask surface (`[u64;64]` / `[[u64;64];8]`, style masks/combine/contra) while remaining DTO-blind.

**Interpretation allowed:** the field producer and the mask ALU both exist but are not connected today.

**Interpretation forbidden:** claiming the exact historical intended connection before the archaeology below runs.

### A1.2 The 4096 relation is explicitly recorded as a broken wire

PR #1051 records:

- 64×64 field;
- COCA 4096 codebook LUT;
- 12-bit `6+6` address reading;
- SPO `2³` amortization at the same address;
- CE64 v2 bits 0..23 as the surviving `3×8` SPO fossil;
- no surviving `codebook_id ↔ (row,col)` mapping today;
- P64's modern role as a **≤4096 active-relation working-set ALU**, not the global lexicon.

This is evidence that `4096` is not merely a recent cache-size coincidence.

### A1.3 `ResonanceDto → PerturbationDto` is a real supersession

The generated supersession audit in PR #1043 records `ResonanceDto` as **REPURPOSE → `PerturbationDto`**. The older resonance vocabulary therefore belongs to the lineage and should be searched when reconstructing the field semantics.

### A1.4 The Morton cascade is a variable aperture, not one fixed table

The committed OGAR discovery-map lineage records:

```text
64 → 256 → 1024 → 4096 → 16k → 64k → 256k
```

as an **immaterialized Morton enumeration**, one additional nibble per level.

This addendum treats the cascade as **address/workload geometry**, not ontology and not cognitive rung.

### A1.5 Stockfish-rs supplies a reference operational grammar, not Morton evidence

`stockfish-rs/.claude/plans/stockfish-harvest-64x64-v1.md` establishes 64×64 = 4096 as chess's intrinsic `from × to` move-address surface and explicitly builds around incremental make/unmake rather than whole-state recomputation.

`stockfish-rs/.claude/knowledge/stockfish-pext-morton-adjacency.md` establishes a transferable low-level pattern:

```text
gather (PEXT or fallback)
    → small dense lookup / local compute
    → scatter (PDEP or fallback)
```

while explicitly fencing that Stockfish itself has **no Morton/Z-order hierarchy, no inverse pyramid, no comma quorum**. The transfer is operational discipline, not historical proof of the lance-graph cascade.

---

## A2. OPERATOR-RECALLED HYPOTHESIS — CE64 as the vertical mantissa

The operator recalls the intended geometry as:

> **`CausalEdge64` was the vertical mantissa in the StreamDto / ResonanceDto→PerturbationDto / BusDto cognitive-field path.**

No current source hit recovered in this session literally says “vertical mantissa.” Therefore this remains a **named hypothesis to recover or refute**, not a SOURCE FACT.

The candidate geometry is:

```text
                    cognitive field

             horizontal address / activity
             Morton / active-set coordinates
                       x,y
                        │
                        │ active locus
                        ▼
                  CausalEdge64
                vertical mantissa
        ┌────────────────────────────┐
        │ SPO relation reading       │
        │ inference / truth          │
        │ witness                    │
        │ causal topology            │
        │ ReasoningBand              │
        └────────────────────────────┘
```

Under this hypothesis the axes answer different questions:

```text
ADDRESS / Morton position   where is the active relation?
RESONANCE / PERTURBATION    how salient / energetic is it now?
MASK / ClassView            which loci / reading participate?
TOP-K / CASCADE APERTURE    how wide is the active working set?
CE64                        what semantic/causal relation is carried there?
SHADER / ATOM               what transformation is applied to the selected field?
BUS                         what projection crosses the execution membrane?
```

### A2.1 Falsifier

**F-ARW-SHADER-1:** search historical/current source, plans, comments, and actual DTO producers/consumers for the relationship between field address, perturbation/resonance value, P64 masks, CE64, and Bus projection.

The hypothesis is rejected or narrowed if:

- CE64 was only a downstream result unrelated to each active field locus;
- the field carried another semantic payload with no CE64 correspondence;
- `BusDto` owned reasoning semantics rather than projection;
- the DTO lineage cannot establish a common field identity at all.

Do not “repair” history to make the diagram true.

---

## A3. The cascade hypothesis — 4096 as tactical aperture, not ceiling

SOURCE FACT: the cascade lineage contains 64/256/1024/4096/16k/64k/256k and P64 is described today as a ≤4096 active-relation ALU.

TRANSFER HYPOTHESIS:

> **4096 was / should be treated as the tactical sweet spot of a variable-aperture cognitive renderer rather than the size of cognition itself.**

Candidate operational reading, deliberately non-canonical:

```text
64       very sparse aperture
256      widened local aperture
1k       neighbourhood aperture
4k       hot tactical field / P64 sweet spot
16k      wider ambiguity/search
64k      broad exploration
256k     exceptional/global aperture
```

The labels above are illustrative only. The measurable claim is simply that **aperture can widen/narrow without changing semantic identity or requiring a new representation**.

### A3.1 Top-k archaeology

Trace whether any historical `top_k`, resonance threshold, perturbation selector, or P64 path actually selected among these cascade levels.

**F-ARW-SHADER-2:** if the ladder and `top_k` never shared a producer/consumer path, do not call the cascade a historical top-k aperture. Record only that they are compatible geometries.

---

## A4. `cognitive-shader-driver` ancestry hypothesis

The literal-name hypothesis to test is:

```text
lance-graph / world state
        ↓
ClassView / horizon / stream projection
        ↓
resonance / perturbation / Alpha field
        ↓
bounded active mask plane (P64)
        ↓
local shader / atom / program
        ↓
Bus projection
        ↓
receipt / Revision / persistence
```

This would make `cognitive-shader-driver` analogous to a renderer driver:

| Graphics concept | Candidate cognition analogue |
|---|---|
| scene | lance-graph epistemic graph |
| view/camera | ClassView / horizon / perspective |
| framebuffer / working plane | resonance / perturbation / Alpha / P64 surface |
| stencil/mask | attention / focus masks |
| shader | Frozen atom / loco-R2IL composition |
| local tile | 2×2 / 4×4 gathered neighbourhood |
| render pass | one bounded cognitive pass |
| history/depth | temporal + premise/Tarski/witness ancestry |
| commit | Revision / Rubicon |

This table is a **reconstruction aid**, not proof.

**F-ARW-SHADER-3:** historical source must establish at least two independent load-bearing correspondences beyond naming before this lineage is promoted from metaphor to architecture finding.

---

## A5. Stockfish 64×64 as reference design

Use stockfish-rs as a **control/reference implementation philosophy** for the 4096 field:

```text
exact compact address space
+ bounded active set
+ mask-native selection
+ incremental perturbation/update
+ gather → local compute → scatter
+ cheap route / exact verify
+ CPU-family-specific fast path with measured fallback
```

Do not import chess semantics into cognition.

### A5.1 Reference questions

For a candidate cognitive-field implementation ask:

1. Are both 64 axes meaningful, or is 64×64 merely a packing trick?
2. Can one change update only the affected cells/columns?
3. Can masks select an irregular neighbourhood without expanding to a bounding window?
4. Can the local operation run densely after gather and scatter back exactly?
5. Is coarse routing separated from exact semantic verification?
6. Does a random/permuted address sabotage locality while preserving cardinality?

If the answer to (6) is “no change,” the spatial/cascade hypothesis is likely decorative.

---

## A6. Fingerprint/OCR × EWA research arm

This is a **TRANSFER HYPOTHESIS / algorithm-search arm**, not architecture yet.

Fingerprints are attractive because they expose simultaneously:

- local directional structure;
- smooth field continuation;
- noise/gaps;
- curvature;
- sparse topological events such as endings and bifurcations.

Candidate atomic measurement:

```text
J = [ Σ Ix²    Σ IxIy
      Σ IxIy   Σ Iy² ]
```

A 2×2 structure/orientation tensor yields local orientation, anisotropy/coherence and energy. EWA uses the same class of covariance geometry:

```text
Σ' = M Σ Mᵀ
```

The research question is therefore not “use fingerprint recognition for cognition.” It is:

> **Can one hardened 2×2 covariance/local-field primitive buy useful work in both degraded-text/fingerprint structure and EWA/Alpha field propagation without changing its mathematics?**

### A6.1 Morton/inverse-pyramid test geometry

Candidate decomposition over a 64×64 field:

```text
64×64 field
    ↓ gather a Morton-local 4×4 tile
4×4 dense neighbourhood
    ↓ decompose/refine
2×2 atomic neighbourhood
    ↓ tensor / covariance / mask / LUT
updated local state
    ↓ scatter
same 64×64 field
```

The inverse pyramid is an **addressing/decomposition hierarchy over one field**, not a requirement to materialize every resolution.

### A6.2 Three-domain falsifier

Use the same physical primitive across:

```text
A. Stockfish-style 64×64 control/reference
   intrinsic compact field + incremental updates

B. fingerprint / OCR field
   ridge/stroke orientation, curvature, gaps, bifurcations

C. Alpha / EWA cognitive field
   attention, residual, contradiction/tension geometry
```

Compare:

- row-major address;
- Morton address;
- random permutation sabotage;
- full recomputation;
- incremental local update;
- 2×2 vs 4×4 neighbourhoods.

Required outcomes:

- **can-fire:** lawful locality/operator choice improves a named metric;
- **can-stay-silent:** irrelevant/local-noise change does not provoke global state;
- **sabotage:** random permutation must damage locality-sensitive performance if locality is actually doing work.

If only the imaging arm wins, the result is an imaging atom. If cognition alone wins, the fingerprint analogy was decorative. If the same primitive wins both, it becomes a strong Frozen-atom candidate.

---

## A7. Alpha is the modern candidate field plane, but this must be proven

The parent #1078 plan already treats Alpha as thin-provisioned, same-address, ephemeral cognition and forbids confusing it with persistent truth.

The lineage hypothesis is:

```text
ResonanceDto
    ↓ historical semantic evolution
PerturbationDto
    ↓ field mechanics
Alpha / attention plane
```

This is **not SOURCE FACT** merely because the concepts rhyme.

Trace whether Alpha actually inherited any of:

- address identity;
- sparse activation semantics;
- energy/residual semantics;
- mask composition;
- top-k / aperture semantics;
- producer/consumer ownership.

**F-ARW-SHADER-4:** if no source lineage exists, keep Resonance/Perturbation and Alpha as separate mechanisms even if a future integration can lawfully compose them.

---

## A8. Critical separation: field geometry must not mint epistemic authority

Even if the shader lineage is recovered, the following remain constitutional:

```text
2×2 tensor / EWA Σ       = geometry
resonance / perturbation = salience / field activity
Alpha                    = reversible working state
Morton / top-k           = address / workload aperture
CE64                     = semantic/causal carrier
Witness / Tarski         = grounding / derivational ancestry
Revision                 = epistemic writeback decision
Rubicon                  = durable action / commit boundary
```

Therefore:

- high field energy is not evidence;
- low Shannon H is not evidence;
- high covariance coherence is not truth;
- a local residual is not a contradiction receipt by itself;
- widening top-k is not promotion;
- a shader transformation cannot manufacture an observation;
- CE64 band/topology changes require their own lawful warrant.

The field may **steer cognition**. Revision decides what survives.

---

## A9. Integration into #1078 deliverables

This addendum does **not** mint a parallel D-id family. It sharpens existing #1078 work:

- **D-ARW-0:** include `StreamDto → ResonanceDto/PerturbationDto → P64/cognitive-shader-driver → CE64 → BusDto` historical/current producer-consumer trace, graded SOURCE FACT / PLAN FACT / HYPOTHESIS.
- **D-ARW-1:** test whether Alpha is the modern field-plane continuation or only a composable sibling mechanism.
- **D-ARW-4:** add address/aperture/field-geometry non-equivalence tests to the orthogonal-coordinate matrix.
- **D-ARW-5:** include local mask/gather/scatter, tensor/covariance, and top-k/cascade operations in the existing atom/loco/R2IL reuse census before adding primitives.
- **D-ARW-6:** the first end-to-end proof may use the recovered field path only if D-ARW-0 establishes its identity; otherwise use the smallest current Alpha path and leave historical restoration separate.
- **D-ARW-8:** fingerprint/OCR and stockfish-rs may serve as algorithmic falsifier/reference arms, but they do **not** count as second cognitive-domain buyers unless they exercise the same cognitive contract rather than merely the same math.

---

## A10. STOP rules added by this addendum

1. Do not claim “CE64 is the vertical mantissa” as current contract until source archaeology recovers the relation or a new measured integration explicitly establishes it.
2. Do not equate `top_k` with the Morton cascade merely because both expose powers-of-four scales.
3. Do not call Stockfish a Morton/inverse-pyramid ancestor; it is an operational reference for compact fields, incremental updates, masks, gather/lookup/scatter and exact verification.
4. Do not turn fingerprint orientation tensors into a cognitive truth model.
5. Do not store EWA covariance in CE64 merely because CE64 is the candidate vertical semantic carrier.
6. Do not materialize every cascade level. The hypothesis is an immaterialized address/aperture grammar.
7. Do not widen the field when a sparse mask or local incremental update can answer the question.
8. Do not invent a new DTO to reconnect the seam until the surviving `StreamDto`, `PerturbationDto`, P64, Alpha, CE64, and `BusDto` surfaces are shown insufficient.

---

## A11. Smallest useful verdict

The archaeology is a BUY if it can establish a real path equivalent to:

```text
stream / observation
      ↓
field activation at stable address
      ↓
mask-native bounded aperture
      ↓
CE64-backed semantic relation at the active locus
      ↓
local reusable shader atom/program
      ↓
typed result / receipt
      ↓
Revision / replay
```

It is a NO-BUY if the old names describe unrelated mechanisms and reconnecting them requires more invention than deleting the historical analogy.

The aim is not nostalgia. The aim is to determine whether several “new” integration problems are actually one old wire that was cut.
