# D-LITHO: the photolithography contract — one coordinate geometry, many exposures

**Date:** 2026-10-30. **Status:** PROPOSED / DOCS ONLY. Nothing is authorized: no ABI, kernel, mask-risc
op, terminal, runtime or contract type.
**Board:** [entry](../board/entries/2026-10-30-d-litho-photolithography-contract.md),
[STATUS_BOARD](../board/STATUS_BOARD.md) D-LITHO.
**Builds on, read and not restated:**
- [D-DRDT](d-drdt-dr-dt-ownership-amendment-v1.md) (#1835): DR is the HHTL16 Cartesian adapter plate, DT the ClassId binding store.
- [D-DT64-6](d-dt64-6-routing-bindstore-behavior-v1.md): DR64 routing, DT64 bind store.
- [D-DR256-WHERE](dr256-where-capstone-v1.md): the population-masked read.
- [D-DT64 capstone](d-dt64-holographic-predicate-capstone-v1.md): 64 typed Boolean predicate slots.
- `lance-graph-mask-risc` (`ir.rs`, `exec.rs`): straight-line programs over slots, tiled execution.
- `lance-graph-contract`: `morton8x8.rs` (D-MORTON-0), `register128.rs` (D-LXC-29), `hotplug.rs`.

**Coordination:** Linear EXO-50, messages 10 and 11 (2026-10-30). Message 11 supersedes message 10's counter
framing.

## Overview

Photolithography is the north star for the relational shader. The wafer (the substrate) never moves. A
reticle (a predicate or address law) is written once and stepped across demanded dies (tiles). Each
exposure is one field-parallel instruction. Dose levels give significance, and develop is the late,
demanded readout. Persistence (etch and deposit) is decoupled from exposure.

This plan writes that down as a contract: the vocabulary, the address grammar, the design rules, and who owns
what. The **Abacus** is the address grammar and the shader mental model (frame, rod, bead position). It
makes no claim about materialization, execution or cost.

## Checklist

- [x] Record the user decisions of 2026-10-30 (§1).
- [x] Vocabulary table: lithography ↔ stack (§2).
- [x] The address grammar (Abacus), with the Morton fact it rests on (§3).
- [x] Design rules (§4) and non-goals (§5).
- [x] Ownership (§6); clarify the D-DRDT word "rotation" without editing that plan (§7).
- [x] Consumers who would express operations in the grammar (§8).
- [ ] Owner replies on EXO-50: DR/DT owner on §9 Q1–Q2; mask-risc owner on Q3.
- [ ] First consumer writes one operation in the grammar, with a failing fixture (§10, G1).
- [ ] Pre-registered gates G1–G5 (§10). None run.

## 1. Decisions recorded

**DECISION (2026-10-30):** The north star of the relational shader is photolithography.
**SCOPE:** the shader contract and every design that projects, exposes or folds over DR coordinates.
**BASIS:** the user's statement in session (decision, not measurement).
**REVISIT WHEN:** an operation is needed that has no counterpart in the exposure model.

**DECISION (2026-10-30):** The Abacus is the addressing contract and the mental model of a shader. Its
concern is not materialization; execution and cost belong to mask-risc, ndarray and the membrane below them.
**SCOPE:** the Abacus. It does not cover the `domino.rs` counter fixture in ndarray, which stays a lab
fixture (H13 DEFER).
**BASIS:** the user's statement in session (decision, not measurement).
**REVISIT WHEN:** a consumer cannot express an operation in frame/rod/position terms.

**DECISION (2026-10-30):** You do not rotate the cube. A different view is a different deterministic
address law over the same words, as simple as defining which axis of a cube is X, Y and Z. No bit is moved
to change a view.
**SCOPE:** every view, projection and axis choice over DR coordinates.
**BASIS:** the user's statement in session (decision, not measurement).
**REVISIT WHEN:** a demanded projection cannot be read through any law without moving bits.

**DECISION (2026-10-30, carried from the shared diagram):** No mandatory dense 64K³ cube, no duplicate
relationship authority, no eager global materialization, no bypass of version witnesses. Identity stays
canonical ("Identity ↓"); the edges are the relational addressing space ("Edges ↑"), and the views add no
edges.
**SCOPE:** the coordinate geometry Identity (X) × Edge (Y) × Property (Z). It is a geometry, not a
storage layout.
**BASIS:** the user's diagrams, 2026-10-30 (decision, not measurement).

## 2. Vocabulary

| Lithography | This stack |
|---|---|
| Wafer | the substrate; never copied, never moved |
| Die / exposure field | one demanded tile: a 64-row word or a 256-row block |
| Stepper stepping die to die | streaming demanded tiles: one shader, many tiles |
| Reticle (mask) | a predicate or a DR address law, reused on every die |
| Exposure | one shader instruction, field-parallel (64-way per word) |
| Dose levels (2^k exposure times) | Abacus rods, significance k |
| Overlay alignment marks | version witnesses and address laws; layers line up without moving anything |
| Develop | a fold or terminal; only what is demanded is read out |
| Layer stack | a program: a sequence of exposures |
| Etch / deposit after inspection | witnessed deltas → coalesced dirty rows → asynchronous writer |

The dose stack is a mechanism, not decoration. Grayscale exposure from binary masks with doubling exposure
times develops to Σ_k 2^k · mask_k. That is the bit-plane fold the ndarray Abacus fixtures compute exactly
(`examples/u9_domino/domino.rs`, `Abacus::total`). The identity is cited here as a fact about the grammar,
not as an execution choice (§6).

## 3. The address grammar (the Abacus)

| Term | Meaning |
|---|---|
| **Frame** | one demanded tile |
| **Rod** | significance k of a value field exposed on that frame |
| **Bead position** | an identity coordinate mapped by a DR address law to a position in the frame; edges and properties are selected against it (§3a), never addressed as fixed-width axes |
| **Instruction** | "expose these rods at these positions" with a verb: mask, combine, compare, carry, fold |

**Law.** A law is a deterministic map from coordinates to a frame and a position. The rod is part of the
address, not part of the value. Meaning (width, signedness, what a count stands for) is not in the grammar.
It comes through DT and hotplug (`Activation::register_law_for`, `RegisterLaw`), the same way a
`Register128` read gets its meaning today.

**The Morton fact the grammar rests on** (`morton8x8.rs`, D-MORTON-0). `Morton8x8` interleaves an 8:8
`(x, y)` pair into a `u16` Z-order code: x on the even bits, y on the odd bits. It addresses all 65,536
positions, which is the DR256::WHERE domain (8 KiB mask = 2^16 bits).
- In code order, the 64 rows of one mask word share every bit above bit 5, so each word is an aligned 8 × 8
  square in (x, y).
- A 256-row block is an aligned 16 × 16 square.
- The "active 64-row words / 256-row blocks" of the DR64/DR256 overview are therefore spatial tiles under
  this law, with no remapping.

**Projections are reticles.** Under a law, fixing one coordinate inside a frame selects a constant bit
pattern of the word. For example, fixing the low three bits of x selects a fixed subset of the even bits. A
projection is therefore an And with a law mask, and a fold over it is a popcount. Choosing another law
changes the mask, never the words.

**What the grammar does not decide.** Whether an instruction runs as a mask-risc `Ternlog`, a fixed-depth
ripple, a carry-save tree or something else is execution, and execution belongs to §6. Reading one cell's
value across all rods (a per-position decode) is a develop step: it is named as a terminal, never assumed
mid-program.

## 3a. Address-space shape (user nudge, 2026-10-30)

**DECISION (2026-10-30):** The address space is fixed-width in one direction only. Edges and properties
are not further fixed-width axes. They are projected against that fixed width in one of two ways:

- **as a cross-product coordinate-range reference:** an (identity range × edge/property range) pair
  that names a region and is not necessarily materialized;
- **PSCustomObject-style:** a named selection of properties and edges, the way `Select-Object -Property`
  picks fields from an object. The existing carrier is `ClassView × WideFieldMask`
  (`lance-graph-contract/src/class_view.rs`; D-DRDT §1 names it the shared contract).

The selection is **versioned via a table in lance-graph**, so a projection is replayable "as of v" like a
route.
**SCOPE:** §3's bead position, and every reading of the Identity × Edge × Property geometry in §1.
**BASIS:** the user's statement in session (decision, not measurement).
**REVISIT WHEN:** a projection needs a second fixed-width axis.

**DECISION (2026-10-30, second nudge):** What is always there, and what is not material:

| Always present | Not material (resolved on demand, never stored per cell) |
|---|---|
| **identity = `NodeGuid`** (`lance-graph-contract/src/canonical_node.rs`), the fixed-width direction | **`ClassView` → methods** (DT, via hotplug) |
| **property mappings**: the selection of properties and edges against the identity width (§3a above) | **64 thoughts → Thought-Git** ([D-DT64-4](d-dt64-4-edge-thoughtgit-hypothesis-time-v1.md); D-DRDT: "Thought-Git sits above both") |

**SCOPE:** what a frame may assume exists, and what an exposure may only reference.
**BASIS:** the user's statement in session (decision, not measurement).
**REVISIT WHEN:** a method or a thought has to be stored per cell to be correct.

Consequences:
- **The wafer has two layers: `NodeGuid` identities and property mappings.** Exposures read these.
- **Methods come through `ClassView` resolution.** A shader names a method; it never carries a copy of one.
- **The 64 thoughts are references into Thought-Git**, not lanes on the frame. A thought is versioned history, not exposed state.
- **This supplies the identity coordinate that §3's bead position needs:** it is a `NodeGuid`, mapped to a frame position by the DR law.

Consequences:
- §1's "Identity (X) × Edge (Y) × Property (Z)" is a selection geometry, not three fixed-width address
  axes. Only the identity direction has fixed width (the 64-row word, the 65,536-address DR256 domain).
- The rods (§3) are a property of a selected field, not a fourth address axis. This narrows Q1 to: does
  the rod width belong to the selected field's DT law?
- A cross-product reference is two ranges and need not be materialized. Whether it is, and how much, is
  an execution choice made on demand, not part of the address. What stays forbidden is a *mandatory* dense
  product. That is design rule 6 and the §5
  non-goals restated as an addressing rule.
- **Open (Q5):** which lance-graph table carries the selection version. Candidates are
  `FrameRevisions` (`dr-math-binding-probe/src/lib.rs`: `relation_map` or `coordinates`) or a dedicated
  selection table. This is the DR/DT owner's decision.

## 4. Design rules

1. **The wafer never moves.** A view is a different reticle (law), never a copy, a transpose or a rotation.
2. **Pay in the reticle, not per die.** Law and predicate preparation are amortized across tiles, never
   repeated per tile.
3. **Exposure is field-parallel.** An instruction acts on a whole frame word at once. Walking positions one
   at a time is the wrong tool (FACET STRIDE R2, one level up).
4. **Alignment is mandatory.** An exposure without a version witness, or under a law other than the one the
   frame was bound with, is a defect even if it runs.
5. **Develop only what is demanded.** Readout and writeback are late and separate from exposure.
6. **Identity is not multiplied.** Relations are addressed as edges. A view never adds an edge and never
   materializes an Identity × Identity matrix.

## 5. Not the goal

- A mandatory dense 64K³ cube, or a 65,536 × 65,536 matrix.
- A duplicate relationship authority: SoA edges remain the relationship authority.
- Eager global materialization, including an eager 64-property cube.
- Any bypass of version witnesses.
- A new writer or persistence path.
- An Abacus runtime type. H13 stays deferred in ndarray, and this plan does not reopen #400–#404.

## 6. Ownership

| Concern | Owner |
|---|---|
| Coordinate geometry, address laws, Morton / HHTL16 traversal, global → local routing | DR (D-DRDT) |
| Class meaning, field laws, method identity, guards, effect order | DT, through hotplug / ogar-vocab (D-DRDT §1, D-CMH) |
| The address grammar and the shader mental model | this contract (the Abacus) |
| Straight-line programs over slots, tiled execution, terminals | `lance-graph-mask-risc` |
| Kernels (mask ops, `half_add_mask` / `full_add_mask`, popcount) | ndarray, through `ndarray::simd` |
| Prepared numeric binding of a query | Quack `Binder` |
| Persistence | resident computation → witnessed dirty rows → asynchronous writer (unchanged) |

## 7. Clarification of D-DRDT's word "rotation"

D-DRDT §1 lists "rotation of Cartesian axes" among DR's concerns. Under §1's third decision, a rotation is
the choice of a different address law over the same words, never a data movement. No bit transpose kernel
is implied or requested. D-DRDT is merged and is not edited; this section is the record of the reading.

## 8. Consumers who would write in the grammar

Each would express its operations as frame/rod/position instructions first. None has sent a signature or a
fixture yet.

| Consumer | Likely first operation |
|---|---|
| ONNX methods (Tract registry, D-TRACT-HARVEST-0) | count-shaped reductions over a predicate (bool → `ReduceSum`, counts behind `Where` / compare) |
| TFLite int8 leaf (D-TFLP) | the same count-shaped reductions |
| egglog-mlir / DialEgg | offline proof that two instruction sequences under two laws develop the same result |
| Identity × Edge evidence → SPO / SPOFC | evidence counts (w⁺, w) on edge rows, folded into frequency and confidence |
| NNUE (later) | binary feature-occurrence counts |
| cognitive-shader-driver (later) | Hamming and popcount folds over frames |

Known misfit, measured earlier and not reopened here: bit-plane *multiplication*. The ndarray probe
`u9-abacus-probe.md` found it 5–120× slower than the byte-dot and refused it. An int8 dot is not a dose
stack.

## 9. Open questions

- **Q1 (DR/DT).** Does the rod (significance k) belong to DR's HHTL16 address, or to DT's per-class field
  law?
- **Q2 (DR/DT).** Is DialEgg the place to prove that a projection under one law develops the same result as
  the equivalent projection under another?
- **Q3 (mask-risc).** Once a consumer writes an operation down, which of its verbs have no straight-line
  form today? Candidates seen in the fixtures: a fixed-depth carry, overflow returned as a plane, a weighted
  develop Σ 2^k · popcount. These are execution questions and stay unasked until G1.
- **Q4 (naming).** The diagram's "DR64K" and "Membrane64" have no definition in code or plans. Before either
  is used as a term, its owner defines it.

## 10. Pre-registered gates (none run)

- **G1, grammar sufficiency.** One named consumer writes one operation as frame/rod/position instructions,
  with a failing fixture and expected values. **Fails if** the operation needs a term outside §3.
- **G2, law, not rotation.** Two different laws over the same frame, the same projection: both develop the
  same result, and the frame bytes before and after are byte-identical. **Fails if** any bit is written to
  change a view.
- **G3, Morton tile.** For every word w of a 65,536-row Morton-ordered population, the 64 codes it covers
  form one aligned 8 × 8 square. **Fails if** any word straddles two squares.
- **G4, dose identity.** A dose stack of k binary masks develops to Σ 2^k · popcount(mask_k), equal to the
  per-position oracle. Disable run: drop one rod; the test turns red.
- **G5, alignment refusal.** An exposure under a law, or a version witness, other than the frame's binding
  is refused, never executed. Disable run: skip the witness check; the test turns red.

## Errata to the shared diagram

The DR64/DR256 overview says "65,336 global addresses". It should say 65,536 (8 KiB × 8 bits = 2^16, the
`Morton8x8` domain).
