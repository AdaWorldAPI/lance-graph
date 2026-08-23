# PROBE-REVISION-ATTENTION-VIEW-1

## Thesis

This probe tests one narrow substrate claim with larger architectural consequences:

> Cognitive work does not need to move through a single serial rung or materialize a new context in order to change focus. Keep the underlying cognitive population stationary, preserve concurrent rung-tagged activity, change the semantic view through typed selectors, and retain the transformation that changed the view.

This is a probe, not a production attention architecture.

It deliberately connects two already-established architectural disciplines:

1. **Mask-native execution** from the lance-graph-java/Panama/Valhalla work: populations stay resident, semantic descriptors compose, materialization is terminal, and execution may lower to predicate/mask/SIMD as appropriate.
2. **Typed behavioral IR** from the r2sleigh → R2IL → ruff work: behavior remains typed, transformations retain provenance, and reconstruction is a first-class falsifier rather than prose after the fact.

The working split is:

```text
masking       = what state becomes visible / conductive
behavioral IR = what transformation occurred
```

No scheduler is introduced.

## Architectural target

The hinge remains:

```text
CognitiveWork
    ↓
Kanban Evaluation
    ↓
Revision
    ↓
another cognitive pass
    OR
Rubicon Commit
```

The probe must not reinterpret the rung field as a global program counter. Multiple rung-associated contributions may coexist over one cognitive state:

```text
R0  ─────── activity
R1  ── activity
R2  ───────────── activity
R3  ───── activity
R4  ────────── activity
R5  ─ activity
R6  ─────────────── activity
R7  ─── activity
R8  ───────── activity
R9  ───── activity
```

The question is whether focus can change while those independently meaningful contributions remain intact.

Useful mental model: cognitive contributions are Photoshop-like layers; attention/selectors are visibility masks; the effective reasoning view is the rendered composite; a Revision edit changes the view; Rubicon Commit is decision lock-in. Rendering must not require destroying the layer identities.

## Grounded starting facts

The source audit preceding this probe established:

- `RungElevator` is local to one `ShaderDriver`; it is not proof of one global cognitive rung.
- Multiple independent rung-tagged representations already coexist in shipped code.
- `ReadOut.lifts` is plural.
- No representation found so far forces one-hot rung occupancy.
- Revision currently exposes no focus/mask/scope/branch editing surface.
- Kanban phase transitions currently record no attention provenance.
- Existing mask union/intersection operations generally collapse contribution provenance.
- `RowFocusMask` is prefix/antichain algebra and is intentionally not interchangeable with `FieldMask`.
- `FieldMask`, `WideFieldMask`, `RowFocusMask`, `Locus`, `GapKind`, and `CausalTopology` are heterogeneous semantic selector families; no universal `CommonMask` contract has been established.
- `WitnessLens<'a>` provides an existing zero-copy borrowed view seam.
- Existing witness/lens functions already accept `impl Fn(usize) -> bool` selectors.
- `GapKind`/`ReasoningGap` and `Locus` already provide production bulk-selection surfaces.
- `CausalTopology` has the desired semantic variants but currently has no production bulk selector.
- `ScenarioBranch` is not wired.
- `↑n` denotes remaining 256-ary address-space breadth, not rung movement.
- `6×2×8-bit` is a real regular `FacetCascade` geometry over twelve tier bytes.

## Key design rule

Do **not** build:

```text
CommonMask
    ↓
opaque universal bitmap
```

Test instead:

```text
heterogeneous typed selectors
        ↓
provenance-preserving composition description
        ↓
terminal lowering
    ┌───────────┴───────────┐
    ▼                       ▼
predicate                packed mask
    │                       │
    └────── zero-copy view ─┘
```

The semantic composition and the execution artifact are distinct things. The execution artifact may be fused and opaque. The semantic composition must retain who contributed what.

A probe-local representation may conceptually resemble a `ViewPlan` with typed selectors, but this document does not prescribe a production type or enum. Existing receipt/view/plan containers must be audited first.

## Falsifiers

### F-PARALLEL-RUNG-1

Construct one positive executable witness using existing production types only. Prefer one problem / one sealed cognitive state. Produce at least two, preferably three, meaningful rung-tagged contributions in that same state.

Assert that:

- all contributions remain observable;
- creating a higher-rung contribution does not overwrite, invalidate, or demote lower-rung contributions;
- no global `current_rung` is required to describe the resulting state.

Do not claim wall-clock thread parallelism unless measured. This falsifier is about concurrent cognitive occupancy, not CPU scheduling.

### F-VIEW-PROVENANCE-1

Compose at least two semantically distinct selectors while preserving their individual identities in the composition description.

The lowered effective predicate/mask may be opaque. The semantic descriptor stack may not be.

Do not modify existing `union()` implementations merely to make this pass.

### F-ZERO-COPY-VIEW-1

Use an existing zero-copy execution seam where possible, first checking `WitnessLens<'a> + impl Fn(usize) -> bool`.

Prove that changing the selected view does not require materializing or copying the underlying `NodeRow` population. If a small selector descriptor allocates, report that separately from population materialization. The invariant is zero population copy, not zero allocation.

### F-TYPED-EDIT-ROUNDTRIP-1

Represent one view/attention change as a typed transformation and require exact reconstruction:

```text
BEFORE + EDIT == observed AFTER
```

The edit vocabulary must be the smallest lawful shape derived from existing semantics. Do not invent a universal cognitive instruction set.

This is behavioral IR at probe level. It is not behavioral learning.

### F-REVISION-FOCUS-1

If current Revision APIs cannot carry a view edit, record `ABSENT`.

Then prove the intended behavior only through a clearly probe-local adapter:

```text
same underlying cognitive state
    ↓
initial typed view
    ↓
Evaluation
    ↓
probe-local Revision view edit
    ↓
same underlying state, different effective reasoning view
```

Do not substitute `RungElevator` for Revision.

### F-NON-DESTRUCTIVE-1

Changing the effective view must not erase underlying concurrent cognitive contributions. Only visibility/selection may change.

### F-HETEROGENEOUS-SELECTOR-1

Demonstrate that semantically different selector families can participate in one composition description without pretending to have identical physical representation or algebra.

Success means heterogeneous semantics are preserved and a common lowering path exists where lawful. Failure means reporting the concrete incompatibility. Do not fix failure by introducing `CommonMask`.

### F-UNKNOWN-TEXTURE

Do not force this through `CausalTopology` in this probe. Its variants remain semantically important, but no production bulk selector currently exists.

Use production-wired `GapKind` / `ReasoningGap` / `Frontier` and `Locus` first if they can express the distinction honestly. Record CausalTopology bulk selection as later work only if a real consumer warrants it.

### F-RUBICON-BOUNDARY-1

Do not claim Commit provenance is solved. Current findings indicate `KanbanMove` carries no attention/mask provenance and calcification remains incomplete.

This probe stops at a typed view/edit existing before Commit. Persistence across the Rubicon remains open.

## Behavioral-BPE consequence

This probe must **not** implement behavioral BPE.

Its responsibility is only to make future learning possible by preserving a typed, reconstructible cognitive transformation.

Future direction, explicitly outside scope:

```text
repeated grounded episodes
        ↓
typed attention/reasoning transformations
        ↓
recurrence detection
        ↓
behavioral compression
        ↓
reusable behavioral IR/template
```

The intended future learning unit is not prose such as “try a counterfactual.” It is a recurring transformation of attention, scope, reasoning geometry, warrants, and resulting state that repeatedly survives grounding.

## AGI relevance, deliberately bounded

The architectural wager is:

- **masking** prevents movement/materialization of cognitive state;
- **typed behavioral IR** prevents useful cognitive transformations from evaporating into opaque outcomes;
- **Revision** provides pre-commit plasticity;
- **Rubicon** provides commitment discipline;
- **grounding** is required before recurrent behavior may become learned grammar.

Together these may eventually support a system that improves its reasoning repertoire without serializing cognition or retraining an entire model for each learned maneuver.

This probe proves none of that. It proves one prerequisite:

> A cognitive view can change non-destructively over concurrent state, and the transformation can remain typed and reconstructible.

## Explicit non-goals

This probe does not:

- create a central scheduler;
- define a global current rung;
- serialize R0..R9;
- create `CommonMask`;
- add universal NOT/XOR merely for algebraic symmetry;
- change CE64 layout;
- wire CausalTopology bulk queries merely to satisfy the diagram;
- resurrect `ScenarioBranch`;
- couple `temporal.rs` to attention;
- define Frozen/Learned/Explore thresholds;
- implement behavioral BPE or reinforcement learning;
- implement Rubicon persistence;
- tune the #1000 heuristic thresholds;
- make `RungElevator` the metacognitive controller;
- merge #1000 by implication.

## Success criterion

The strongest acceptable claim after this probe is:

> At probe level, one cognitive state can retain multiple rung-associated contributions while a typed, provenance-preserving selector composition is changed and lowered into a different zero-copy reasoning view. The change is reconstructible as a typed behavioral transformation. No global rung transition or central scheduler is required.

Nothing stronger.

## Failure is useful

Fail closed. If concurrent rung occupancy, heterogeneous selector composition, zero-copy lowering, edit reconstruction, or preservation of underlying contributions is impossible, report the exact representation or ownership boundary that prevents it.

Do not widen the probe to repair every discovered gap.

The purpose of this slice is to find the smallest real substrate seam where:

```text
attention conductivity
        meets
typed behavioral transformation
```
