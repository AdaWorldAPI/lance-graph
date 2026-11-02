# D-DRDT: DR = HHTL16 Cartesian direct routing, DT = ClassId binding store (amendment)

**Date:** 2026-10-28. **Status:** PROPOSED / DOCS ONLY. **Amends:** [D-DT64-6](d-dt64-6-routing-bindstore-behavior-v1.md) §1 (the DR#3 and per-cell WHEN×ON×DO decisions) and the DT64 definition in Bardioc-rs #9 (`.claude/harvest/DR64-DT64-QUACK-HIRO-SCHEMA-DISTILLATION-2026-10-28.md`). **Proof:** [#1834](https://github.com/AdaWorldAPI/lance-graph/pull/1834) (D-DRCA-1). **Not:** ABI, runtime, schema store, identity authority or merge authorization.

The merged plans are not edited. This file records the new contract, says which earlier wording it supersedes, maps it onto what exists (§3), and hands off the parts that belong to other owners (§4, §5).

## 1. Decisions recorded

**DECISION (2026-10-28):** DR is the reusable Direct Routing adapter plate. Its concern is coordinate mapping, projection, rotation of Cartesian axes and fold-based joins. Its addressing is HHTL16 Cartesian addressing. Global-to-local preparation is optional and amortized; it is not part of every hot-path operation. DR is agnostic to ontology, class meaning and application behaviour.
**SCOPE:** what DR owns and what it must not own.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement).
**REVISIT WHEN:** a DR operation needs class meaning to compute its result.

**DECISION (2026-10-28):** Identity : Property is a fold join through coordinate mapping. The adapter plate maps the required coordinates and the algebra operates on the mapped positions. A Cartesian cross-product is one available algebraic operation, not the mandatory physical form of every join. Four projections: Identity × Identity (optional), Identity × Property (primary), Property × Property (higher-order, derived; correlation is an analytical result, not a causal claim), cross-table Identity × Property (a foreign key as a local Cartesian projection, ordinals aligned through a witnessed relationship, never by equal numbers).
**SCOPE:** the DR projection set. An ontology may be one table with a 64K address domain, subject to its declared capacity; that is no reason for a global shared namespace or a 64K × 64K product.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement).
**REVISIT WHEN:** a projection cannot be expressed as a fold over mapped positions.

**DECISION (2026-10-28):** DT is the existing DTO binding store, addressed by ClassId: `ClassId → class schema + method schema` (structural class definition, field semantics, method identities, registered behaviour descriptions, their versioned bindings). DT owns meaning and behaviour; DR does not. `ClassView × WideFieldMask` is domain-neutral and belongs to the shared contract, not exclusively to OGAR; OGAR, ERP, IAM, biomedical-ontology and future SAP adapters provide class and method definitions through that one contract. Quack's `Binder` stays the operational query-binding membrane; its prepared numeric bindings are not DT's semantic bindings.
**SCOPE:** DT ownership and the boundary with Quack.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement).
**REVISIT WHEN:** a consumer needs a schema that `ClassId → class + method schema` cannot carry.

**DECISION (2026-10-28):** DR #4 routes method operations over the same mapped coordinates. Cross-product is one method among others (relational composition, mask projection, Boolean and K3 operations, admitted numerical folds). Method semantics belong to DT's ClassId-bound method schema. Source MethodIds, parameter signatures, guards, multiple applicable methods and ordered effects are preserved. DR invents no method semantics; DT needs no execution engine.
**SCOPE:** the DR #4 / DT interface.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement).
**REVISIT WHEN:** a method's semantics cannot be referenced without DR storing them.

**DECISION (2026-10-28):** Thought-Git (0+63 perspectives, hypothesis testing, Gestalt, epistemic rungs, recipes, time-series reasoning) sits above DR and DT and reuses their routed structures and semantic bindings without copying worlds. Not in this session's scope.
**SCOPE:** layer order.
**BASIS:** the user's statement in session, 2026-10-28 (decision, not measurement).
**REVISIT WHEN:** the Thought-Git owner needs a DR or DT surface that does not exist.

**Canonical invariant.** DR = HHTL16 Cartesian addressing, optional amortized global-to-local mapping, agnostic fold-based projection. DT = ClassId-addressed binding store for class schema, meaning and methods. Thought-Git = metacognition over their versioned projections. The same coordinate algebra serves one table, several tables, an ontology or a domain behaviour model.

### 1.1 What this supersedes

| Earlier wording | Where | Now |
|---|---|---|
| "WHEN × ON × DO attaches per DR#3 cell, as the registry over methods" | D-DT64-6 §1, third decision | Method semantics (incl. WHEN/ON/DO) are DT's, keyed by ClassId and MethodId. A DR#3 cell reaches them through its class (D-DT64-6 §3 already stores the registry per `(class, property)`); DR does not hold them. |
| "DR64 is direct routing: a mapping table" | D-DT64-6 §1, fourth decision | Kept and made precise: the mapping is HHTL16 Cartesian addressing plus fold-based projection. |
| "DT64 is the DTO bind store" | D-DT64-6 §1, fourth decision | Kept: DT is the existing binding store. |
| "DT64 = DTO Tensor 64 folding hypercube register, not merely a bind store" | Bardioc-rs #9, distillation, "Correct nouns" | Superseded. Folding is DR's (with Quack / mask-risc / ndarray computing it); DT is the binding store. |
| DT64 three levels, all 64³ | D-DT64-6 §1, first decision | A logical shape only; no physical cube follows (consistent with its own "not goals"). Level 3 (WHEN × ON × DO) is DT method schema, not a DR coordinate axis. |

### 1.2 The DR64 / DR256 overview, read against this contract

The user's 2026-10-28 overview diagram maps onto existing pieces:

| Diagram box | Meaning | Existing piece |
|---|---|---|
| 1. `DR256::WHERE(criteria)`, global selection | one population mask over the 65,536 addresses of a 256² space (the diagram prints 65,336; 256² = 65,536), 8 KiB, popcount; execution choice: stay in DR64, rebind DR64, or masked DR256 read | [D-DR256-WHERE](dr256-where-capstone-v1.md); a Quack `Query`, `Agg::Rows → Terminal::Keep`; labs #1790, #1791 |
| 2. DR#1 Identity | selected identities, versioned focus binding, global ↔ local witness | the identity `LocalMap` + `Frame` in D-DRCA-1 (lab); production is EXO-46 |
| 2. DR#2 Properties | selected properties, typed field bindings, presence / validity / schema | the property `LocalMap`, typed `Axis`, `Operand.validity` in D-DRCA-1; field semantics are DT (`ClassView::fields`) |
| 3. DR#3 Identity × Properties | native algebra over aligned coordinates, Quack → mask-risc → ndarray, SoA edges remain the relationship authority | the bound register and its projections (D-DRCA-1); the register is a view, the SoA edges stay the authority |
| 4. optional Identity × Identity, Properties × Properties | causes / caused-by, direction flip, typed inverse only when declared; co-occurrence, covariance, derived on demand | D-DRCA-1 Identity × Identity (the rotation reads the inverse without asserting symmetry) and `cooccurrence` |
| not goals; persistence separate | no 65K × 65K matrix, no eager 64-property cube, no new writer; resident → squash/release → coalesced dirty rows → async writer | unchanged (EXO-55) |

## 2. Grounding

| Item | State | Citation |
|---|---|---|
| HHTL16 Cartesian address = `Morton8x8` (x on even bits, y on odd; each nibble a 4 × 4 refinement, fan-out 16) | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/morton8x8.rs`, module doc; same code as `facet::FacetTier::morton` |
| The 16ⁿ nibble router (one tree axis) | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/hhtl.rs`, `NiblePath` |
| A proposed `HHTL16ISA[16]` / `HHTL16PartOf[16]` zipper view (8-bit steps) | PROPOSED, not ABI | `d-qr-abi-classview-rails-capstone-v1.md`, "Vertical HHTL16 zipper candidate" |
| Rotation is a coordinate system, not a mask op; `MaskOp::Rotate` must not exist | CURRENT-CONTRACT | `crates/lance-graph-mask-risc/tests/rotation.rs`, module doc |
| DR64 binding, frame / focus / scope guards, Boolean compose, typed `Chain` | VERIFIED-IN-CODE (lab) | `crates/dr-math-binding-probe/src/{lib,relcomp}.rs` |
| Quack `Binder` (`table`, `live`, `field`, `code`) | VERIFIED-IN-CODE | `crates/lance-graph-quack/src/bind.rs`, trait `Binder` |
| `ClassView::fields` (ordered field basis), `compute_dag` | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/class_view.rs` |
| Method signatures and action declarations per classid | VERIFIED-IN-CODE | `codegen_manifest::{MethodSig, methods_for}`, `action::{ActionDef, StateGuard, actions_for, effective_actions}` |
| Versioned activation per consumer | VERIFIED-IN-CODE | `crates/lance-graph-contract/src/hotplug.rs`, `HotPlug`, `Activation` |
| D-DRCA-1 projections, rotation, inverse map, refusals, costs | MEASURED (lab, one host) | #1834, crate README section D-DRCA-1 |

## 3. Reuse and gap analysis: Cartesian and relational fold operators

| Operator | Exists in | Used by D-DRCA-1 | Gap |
|---|---|---|---|
| Cartesian address and inverse | `Morton8x8::{from_xy, x, y}` | yes (`address`, `coordinates`) | none for 8:8. A 64 × 64 register uses 6 bits per axis; a larger focus is an open item. |
| Neighbour / ascent on the code | `Morton8x8::{checked_offset, nibble_climb}` | no | not needed for projection. |
| YZ rotation | rule in `mask-risc/tests/rotation.rs`; no operator | `hhtl16::rotate_yz` (address map, lab) | if production needs it, it belongs next to `Morton8x8` as an address map, never as a mask op. |
| Identity row | the borrowed register word | `hhtl16::row` | none. |
| Property column (read through the map) | `ndarray::simd::eq_u32_strided_to_mask` for value lanes; nothing for a bit column of a 64 × 64 register | `hhtl16::column` (scalar, 10 ns per column) | a bit-matrix column read or 64 × 64 bit transpose in `ndarray::simd` (`U64x8::transpose8` transposes words, not bits). Largest warm cost after the folds. |
| Boolean relational composition | `relcomp::compose`, `Chain`; `mork-pathmap-probe` `BitMat::compose`; `relalg` `Rel` | `Chain` for the cross-table FK | `Chain::eval` copies each leaf (512 B) and allocates (4 per call, measured). A borrowed-leaf evaluation would remove both. |
| Property × Property Boolean (`Rᵀ ∘ R`) | none as a named op | `hhtl16::cooccurrence` (outer product over stored rows, 818 ns) | none functionally; a SIMD form is unmeasured. |
| Co-occurrence count, bag measure | `relalg` `Measure`; `dir-rmap-probe::compose::add_bag` | `cooccurrence_count` (popcount of two columns) | a full count matrix would need a counted semiring (D-DT64-3 §3); not built. |
| K3 / validity | `Operand.validity`; Quack `BoundField.validity`, `Filter::sql_where` | `column_validity` | none at this size. |
| Population selection | Quack `Query` → `Terminal::Keep`; DR256 labs #1790/#1791 | no | DR256::WHERE × DR#3 composition and the stay / rebind / masked-read choice are untested here. |
| Global → local | lab: `dir-rmap-probe` prep, D-DRCA-1 `LocalMap`; production: none | `LocalMap` (1,050 ns cold for 64 ids, 7.7 ns per lookup, 0 lookups in warm reads) | EXO-46 owns production admission; EXO-140 classifies gather call sites. |
| Frame / source witness | lab `FrameStore` (store-issued seal) | yes | production snapshot attestation is open (EXO-148, D-DT64-Q0). |

No new higher-order graph engine is needed: every projection above is an address map plus an existing Boolean or count fold.

## 4. Handoff: DT behaviour binding

What DT already has, by ClassId:

- **Class schema:** `ClassView::fields(class)` (ordered `FieldRef`s, the `WideFieldMask` basis), `compute_dag`.
- **Method schema:** `codegen_manifest::methods_for(registry, classid)` → `MethodSig { name, params, ret, overrides }`; `action::actions_for` / `effective_actions` → `ActionDef { predicate, object_class, exec, guard: Option<StateGuard>, required_role, overrides }`.
- **Versioned binding:** `hotplug::{HotPlug, Activation}`; a changed behaviour mints a derived classid (D-DT64-6 §1).

Gaps for the DT owner (none is built here):

1. **No method identity independent of its name.** `MethodSig.name` and `ActionDef.predicate` are names; a stable MethodId with a signature revision is not in the contract. Multiple methods applicable to one cell need distinct identities.
2. **Guards are a single `field == value`.** A K3, conjunctive WHEN (and HIRO KI's ON scope) does not fit `StateGuard`.
3. **No effect order.** `ActionDef` has no ordered effect list; two methods with the same read/write sets but different order cannot be told apart.
4. **No single lookup** joining class schema, method schema and activation for one ClassId at one version.

Bounded interface to propose (not implemented): a read-only resolution `(ClassId, schema revision) → { fields, methods: [(MethodId, signature revision, guard, ordered effects)] }`, served by the existing stores above, with no persistence of its own.

## 5. Handoff: DR #4

DR #4 is the route from a mapped coordinate set to a method operation:

- **Input:** bound DR registers (`BoundDr64` or a successor) under one frame, focus and scope; a method reference `(ClassId, MethodId, signature revision)` resolved by DT; the algebra operation the method declares (composition, mask projection, Boolean/K3, an admitted numerical fold).
- **Output:** a mask, register or terminal, with the frame and the method reference it was computed under. DR stores neither the method's meaning nor its effects.
- **Must preserve:** every applicable MethodId (no collapse), guard witnesses, effect order as an opaque reference, source identity.
- **Must refuse:** an operation whose declared algebra DR does not have; a method reference resolved at another schema revision than the frame's; operands of different frames or domains (as `Chain` does).

The first falsifier for DR #4: two methods on one cell with the same read/write sets and different effect order must stay distinct through routing, and a dropped guard must refuse.

## 6. Ownership and coordination

- **DR (this lab):** `dr-math-binding-probe` (`hhtl16`, `relcomp`). Production placement of `rotate_yz` and a column read: next to `Morton8x8` and in `ndarray::simd`, by their owners.
- **DT:** contract `class_view`, `codegen_manifest`, `action`, `hotplug`; gaps §4 to the contract owner.
- **Quack / DR256:** `lance-graph-quack`, D-DR256-WHERE.
- **EXO-46** (global → focus production), **EXO-140** (gather call sites), **EXO-148** (durable refs), **NFR-OP-2** (physical DR³) stay with their owners. Coordination through EXO-50.

## 7. Not done

DR256::WHERE × DR#3; focus larger than 64; SIMD column read; borrowed-leaf `Chain::eval`; DT MethodId / guard / effect-order contract; DR #4 code; Thought-Git; a real source owner, ACL or registry.
