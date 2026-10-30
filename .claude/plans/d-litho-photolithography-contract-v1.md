# D-LITHO: the photolithography contract — the DTO layer over DR's 64ⁿ / 256ⁿ hypercube

**Date:** 2026-10-30. **Status:** PROPOSED / DOCS ONLY. Nothing is authorized: no ABI, kernel, mask-risc
op, terminal, runtime or contract type.
**Board:** [entry](../board/entries/2026-10-30-d-litho-photolithography-contract.md),
[STATUS_BOARD](../board/STATUS_BOARD.md) D-LITHO.
**Builds on, read and not restated:**
- [D-DRF-1](d-dr64-focus-lease-foreign-cache-v1.md) §0–§3: DR64 focus geometry, the u16 pointer map, the
  64³ Boolean relation volume, axis semantics declared by the caller, focus lease.
- [D-DRAV-1](d-dr-address-first-value-demand-v1.md): sealed DR map kept apart from value demand.
- [D-DRDT](d-drdt-dr-dt-ownership-amendment-v1.md) (#1835): DR is the Cartesian adapter plate; DT is the
  ClassId binding store.
- [D-DT64 capstone](d-dt64-holographic-predicate-capstone-v1.md): DT64 = DTO Tensor 64, a semantic adapter
  plate that "does not itself decide DR64³ axis semantics" (NFR-OP-2 belongs to the DR physical ABI).
- [D-DT64-6](d-dt64-6-routing-bindstore-behavior-v1.md): DR64 routing, DT64 bind store.
- `lance-graph-mask-risc` (`ir.rs`, `exec.rs`), `lance-graph-contract` (`hotplug.rs`, `class_view.rs`,
  `canonical_node.rs`).

**Coordination:** Linear EXO-50, messages 10–18 (2026-10-30).

> **⊘ Scope correction (2026-10-30, user).** "You're the DTO layer of a holographic hypercube 64ⁿ or 256ⁿ.
> Everything in regards to dimensions is not your concern." Earlier revisions of this plan (#1859 and the
> first commits of #1860) defined coordinate geometry: a Morton tile fact, axis widths, presence-bit
> underflow, coordinate freezing, an address law and a "rotation" reading. All of that is DR's, is already
> written in D-DRF-1, and was written here without reading D-DRF-1 first. It is removed. §4 lists what this
> contract assumes DR provides, by reference only.
>
> **⊘ Correction to the correction (2026-10-30, user).** "Of course you need to know the dimensions in order
> to apply the law effectively." Removing the dimensions overcorrected. The error was stating them as this
> contract's decisions, not stating them at all. §4 now lists the DR facts the DTO layer must know, each
> cited to its DR source, with what the DTO layer does with it. They are DR's; this contract decides none of
> them. Gates G2 and G3 are restored in §11.

## Overview

Photolithography is the north star for the relational shader. The wafer (the substrate) never moves. A
reticle is written once and stepped across demanded dies (tiles). Each exposure is one field-parallel
instruction. Dose levels give significance, and develop is the late, demanded readout. Persistence (etch
and deposit) is decoupled from exposure.

This contract is the **DTO layer** of that picture. DR owns the hypercube: its dimensions, coordinates,
routing and the sealed global → local map. Over a coordinate that DR has already routed, the DTO layer binds
meaning: which identity, which property mappings, which methods and thoughts it may reference. The
**Abacus** is the mental model a shader author uses to read that binding (frame, rod, bead). It names DR
coordinates; it does not define them.

## Checklist

- [x] Record the user decisions of 2026-10-30 (§1).
- [x] Vocabulary, with the owner of each row (§2).
- [x] What the DTO layer binds (§3); the DR facts it must know and apply (§4); the order (§5).
- [x] Design rules (§6), non-goals (§7), ownership (§8), consumers (§9).
- [ ] Owner replies on EXO-50 (§10).
- [ ] First consumer writes one binding with a failing fixture (§11, G1).
- [ ] Gates G1–G5 (§11). G3, G4 PASS; G1, G2, G5 not run.

## 1. Decisions recorded

**DECISION (2026-10-30):** The north star of the relational shader is photolithography.
**SCOPE:** the shader contract and every design that exposes or folds over DR coordinates.
**BASIS:** the user's statement in session (decision, not measurement).
**REVISIT WHEN:** an operation is needed that has no counterpart in the exposure model.

**DECISION (2026-10-30):** This contract is the DTO layer of a holographic hypercube 64ⁿ or 256ⁿ. It owns
and decides no dimension, but it must know DR's dimensions to apply the law.
**SCOPE:** everything below. Axes, widths, coordinates and routing are DR's; §4 lists the ones the DTO layer
applies, cited.
**BASIS:** the user's statements in session (decision, not measurement).
**REVISIT WHEN:** a DTO binding cannot be stated without choosing a dimension.

**DECISION (2026-10-30):** The Abacus is the mental model of a shader. It concerns naming, not
materialization; execution and cost belong to mask-risc, ndarray and the layers below them.
**SCOPE:** the Abacus as notation. It does not cover the `domino.rs` counter fixture in ndarray, which
stays a lab fixture (H13 DEFER).
**BASIS:** the user's statement in session (decision, not measurement).

**DECISION (2026-10-30):** What is always present and what is not material.

| Always present | Not material (resolved on demand, never stored per cell) |
|---|---|
| **identity = `NodeGuid`** (`canonical_node.rs`) | **`ClassView` → methods** (DT64, via hotplug) |
| **property mappings**, i.e. which properties and edges are selected | **64 thoughts → Thought-Git** ([D-DT64-4](d-dt64-4-edge-thoughtgit-hypothesis-time-v1.md)) |

**BASIS:** the user's statements in session (decision, not measurement).

**DECISION (2026-10-30):** The rule for thoughts is "reserve, don't claim". The mental model is thin
provisioning, not thick: a thought binding allocates on its first write (a Thought-Git `Commit`), never in
advance. Where the thought positions live is DR's and Thought-Git's (§4); the DTO layer only refuses to
preallocate.
**BASIS:** the user's statements in session (decision, not measurement).

## 2. Vocabulary

| Lithography | This stack | Owner |
|---|---|---|
| Wafer | the substrate; never copied, never moved | SoA owner |
| Die / exposure field | one demanded tile | DR |
| Stepper stepping die to die | streaming demanded tiles: one shader, many tiles | DR / execution |
| Reticle (mask) | a predicate, or a DR projection | DT64 (predicate) / DR (projection) |
| Exposure | one field-parallel shader instruction | mask-risc / ndarray |
| Dose levels (2^k exposure times) | Abacus rods: significance k of a bound value | DTO notation over DR coordinates |
| Overlay alignment marks | version witnesses: DR's sealed map, the DTO binding's version | DR / DT64 |
| Develop | a fold or terminal; only what is demanded is read out | mask-risc |
| Layer stack | a program: a sequence of exposures | mask-risc |
| Etch / deposit after inspection | witnessed deltas → coalesced dirty rows → asynchronous writer | persistence (D-DRF-1 §6) |

The dose stack is a mechanism, not decoration. Grayscale exposure from binary masks with doubling exposure
times develops to Σ_k 2^k · mask_k, the bit-plane fold the ndarray Abacus fixtures compute exactly
(`examples/u9_domino/domino.rs`, `Abacus::total`).

## 3. What the DTO layer binds

Over a coordinate DR has routed:

- **Identity:** a `NodeGuid`. The DTO layer never invents or compares local ordinals as identity
  (D-DRF-1 §2: "one ordinal shared numerically across datasets is not evidence of identical identity").
- **Property mappings:** which properties and edges are selected, PSCustomObject-style, the way
  `Select-Object -Property` picks fields. The carrier is `ClassView × WideFieldMask` (`class_view.rs`). The
  selection is versioned via a table in lance-graph (Q5).
- **Methods:** named by reference and resolved through `ClassView` / hotplug. A shader never carries a copy.
- **Thoughts:** references into Thought-Git, thin-provisioned (§1).
- **Meaning of a rod:** width, signedness and what a count stands for come through DT64 and hotplug
  (`Activation::register_law_for`, `RegisterLaw`), never from the coordinate.

## 3a. Co-architecture readings (WORKING-MODEL)

- **The Abacus rods are DR's missing counted semiring.** D-DRDT §3: a count matrix "would need a counted semiring (D-DT64-3 §3); not built"; D-DRF-1 §1: a Boolean cell carries no multiplicity. A dose stack of k Boolean planes develops to Σ 2^k · popcount(mask_k) without changing the Boolean cell. XLOG bag multiplicity and SPO evidence counts (w⁺, w) are rods over Boolean DR planes; G4 is that semiring's soundness gate.
- **The die map is DR256's occupancy statistic.** `dr256-where-capstone-v1.md` B.7(2) measured that mask ops and masked folds do not skip dead 256-row blocks (NB/QB 0.02–0.39 at sparse selections), and P3 waits for "a consumer decision that reads active blocks". By G3 a 256-row block is a 16 × 16 die; a stepper exposes only patterned dies. Candidate consumer decision for P3: build the die map once from `Keep`, step only active dies. Unmeasured through Quack.
- **A DTO binding is D-DRF-1's "caller".** Frame witness + declared axes (`AxisKind`, `dr-math-binding-probe`) + declared semiring (Boolean or rods) + law (an address map over `Morton8x8`) + method references. Nothing else reaches a shader.

## 4. What the DTO layer must know (DR's facts, applied here)

Every fact below is DR's. This contract decides none of them; the authoritative text is the cited source.
The right-hand column is what the DTO layer does with the fact. Sources were read in full on 2026-10-30.

| DR fact | Source | What the DTO layer does with it |
|---|---|---|
| DR64 focus: 64 × 64 = 4,096 positions; Boolean tile 512 B; u16 position → SoA-row map 8 KiB; at most 2 MiB of distinct 512 B rows. DR256: 65,536 positions; 8 KiB; 128 KiB; 32 MiB. The 32 × 128-bit and 64 × 64-bit views are the same 512 B tile. | D-DRF-1 §1; falsifier G1 (PASS, lab, `drf-focus-probe`) | Every binding names DR64 or DR256 and states its cost in these units, never as one row per position. |
| Dense 64³ Boolean volume 32 KiB; 256³ 2 MiB. Map plus cube is 40 KiB, not 32 KiB. A 64³ cell is not a pointer, a row or a thought. | D-DRF-1 §1 | A binding never assumes a cube exists. A cube is a separate admitted view. |
| **Axis semantics are declared by the caller.** What the third axis physically indexes is NFR-OP-2, owned by the DR physical ABI. | D-DRF-1 §1; DT64 capstone (DT64-OP-2) | **The DTO layer is that caller.** Each binding declares its typed axes (Identity, Property, relation selector) and never decides the physical third axis. |
| The HHTL16 address is `Morton8x8`: `x` on even bits, `y` on odd bits; each nibble is a 4 × 4 refinement, fan-out 16; a power-of-two subgrid is a code prefix (pinned for 4 × 4). | `crates/lance-graph-contract/src/morton8x8.rs` (`a_power_of_two_subgrid_is_a_code_prefix`); D-DRDT §1–§2 | A law is a deterministic map over this address. Rotation changes the coordinate law and is never a mask op (`MaskOp::Rotate` must not exist, `crates/lance-graph-mask-risc/tests/rotation.rs`). |
| In code order a 64-row mask word is an aligned 8 × 8 square and a 256-row block a 16 × 16 square. | `morton8x8.rs`, `every_aligned_code_run_is_an_aligned_square` (G3, PASS, disable-verified) | A binding may read DR256's active 256-row blocks as 16 × 16 dies: the stepper's die map (§3a). |
| Fixed width 64 or 256. Underflow is carried by the presence bit, written sparse and merged on read; absent is never FALSE or zero. | user ruling 2026-10-30 (EXO-50 messages 17–18); matches D-DRF-1 G7, D-DRAV-1 AV11, `dr256-where-capstone-v1.md` A.11 | An out-of-focus contributor is expanded through a witnessed binding or refused. |
| The sealed global → local map is immutable and versioned; a change is a successor revision. The focus lease begins at load. | D-DRF-1 §0, §2, §3; D-DRDT §1 | This is why coordinates freeze: a binding pins one map revision. The same ordinal under another frame is refused (D-DRF-1 G3; D-SAG-6 F07). |
| A Boolean cell is support under a declared semiring; it carries no Bag multiplicity, SQL NULL or evidence. Presence is not validity. `ClassView::FieldMask` is presence, never predicate truth. | D-DRF-1 §1; `dr256-where-capstone-v1.md` A.5, A.11; DT64 capstone | Every binding declares its semiring. A count-shaped reading needs a counted semiring, not a Boolean cell. |
| `Keep` → `Plane` is reusable under the same frame; `ScatterOrU32` is only ever a demanded result. A detached `Keep` has no witness and is recomputed or refused (1,032 vs 31). | `lance-graph-mask-risc/src/ir.rs:329-337`; `dr256-where-capstone-v1.md` B.6, E.5 | Named intermediates follow the four lifetimes of `dr256-where-capstone-v1.md` A.2; no escaped reuse. |
| Map and value lifetimes are separate; a map-only operation is legal without fetching values. | D-DRAV-1 §2, §6 | A method binding declares whether it is value-free or value-demanding. |
| The 64-thought occupancy plane is separate from the relation selector and the mapping index. | D-DRF-1 §1; falsifier R2 | Thoughts are reserved, not claimed (§1), and are never an axis of a DR cube. |

Projection and views: a view is a different DR projection over the same words, never a data move.

## 5. Order

**DR routes first** (global → local, address → classid). **DT64 then assigns** meaning and methods to the
routed position. Execution (mask-risc, ndarray) comes after both. DT64 does not route, and DR assigns no
meaning.

## 6. Design rules

1. **The wafer never moves.** A DTO binding never copies the substrate.
2. **Pay in the reticle, not per die.** Bindings and predicates are prepared once and stepped across tiles.
3. **Exposure is field-parallel.** A binding that has to be walked position by position is the wrong tool.
4. **Alignment is mandatory.** A binding used under a DR seal or a binding version other than its own is a
   defect even if it runs.
5. **Develop only what is demanded.** Readout and writeback are late and separate from exposure.
6. **Identity is not multiplied.** Relations are addressed as edges. A binding never adds an edge.

## 7. Not the goal

- A mandatory dense 64K³ cube, or a 65,536 × 65,536 matrix.
- A duplicate relationship authority: SoA edges remain the relationship authority.
- Eager global materialization.
- Any bypass of version witnesses.
- A new writer or persistence path.
- An Abacus runtime type. H13 stays deferred in ndarray.
- Any decision on dimensions, axes, widths or coordinates. They are DR's; §4 cites the ones applied here.

## 8. Ownership

| Concern | Owner |
|---|---|
| The hypercube: dimensions, axes, coordinates, Morton / HHTL16, global → local routing, focus lease | DR (D-DRF-1, D-DRDT) |
| Class meaning, field laws, method identity, guards, effect order, assigned after DR routes | DT64, through `ClassView` / hotplug / ogar-vocab |
| The shader mental model and the DTO binding vocabulary | this contract |
| Thought positions and commits | Thought-Git (D-TGB, EXO-148) |
| Straight-line programs, tiled execution, terminals | `lance-graph-mask-risc` |
| Kernels | ndarray, through `ndarray::simd` |
| Persistence | resident computation → witnessed dirty rows → asynchronous writer |

## 9. Consumers

Each would state its operation as a DTO binding over DR coordinates first. None has sent a signature or a
fixture yet.

| Consumer | Likely first binding |
|---|---|
| ONNX methods (Tract registry, D-TRACT-HARVEST-0) | count-shaped reductions over a predicate |
| TFLite int8 leaf (D-TFLP) | the same count-shaped reductions |
| XLOG (D-XLOG-DR64-0) | bag multiplicity per answer tuple (EXO-50 message 12) |
| egglog-mlir / DialEgg | offline proof that two exposure sequences develop the same result |
| Identity × Edge evidence → SPO / SPOFC | evidence counts (w⁺, w) folded into frequency and confidence |
| NNUE, cognitive-shader-driver (later) | feature-occurrence counts; Hamming and popcount folds |

Known misfit, measured earlier: bit-plane *multiplication* (ndarray `u9-abacus-probe.md`, 5–120× slower
than the byte-dot, refused).

## 10. Open questions

- **Q2 (DR/DT).** Is DialEgg the place to prove that two exposure sequences develop the same result?
- **Q3 (mask-risc).** Once a consumer writes a binding down, which verbs have no straight-line form today?
  Unasked until G1.
- **Q4 (naming).** "DR64K" and "Membrane64" have no definition in code or plans; their owner defines them.
- **Q5 (DR/DT).** Which lance-graph table versions the property-mapping selection.
- Q1 (where the rod width lives) is withdrawn: it is a dimension question, so it belongs to DR (NFR-OP-2).

## 11. Pre-registered gates (none run)

- **G1, binding sufficiency.** One named consumer states one operation as a DTO binding over DR
  coordinates, with a failing fixture and expected values. **Fails if** the statement needs a dimension
  decision.
- **G4, dose identity. PASS** (`crates/lance-graph-mask-risc/tests/dose.rs`: K = 6 rods over a 4,096-row focus, one `And` + `Count` program per rod, densities 0/3/50/97/100 %; every rod load-bearing; disable run: swap adjacent rod weights → red). A dose stack of k binary masks develops to Σ 2^k · popcount(mask_k), equal to the
  per-position oracle. Disable run: drop one rod; the test turns red.
- **G5, alignment refusal.** A binding used under a DR seal or binding version other than its own is
  refused, never executed. Disable run: skip the witness check; the test turns red.
- **G2, law, not rotation.** A DTO binding that reads a rotated or transposed view does so through an
  address map over the canonical words, and its answer equals the canonical answer read through the map.
  DR's existing proof is `crates/lance-graph-mask-risc/tests/rotation.rs`; G2 is the same check for the
  first DTO binding that uses a non-identity view. **Fails if** the binding needs a rotated copy or a mask op.
- **G3, Morton tile. PASS** (`every_aligned_code_run_is_an_aligned_square`, j = 0..=8; disable run: runs starting one code late → red). Over all 65,536 codes, every aligned 64-code run is one aligned 8 × 8 (`x`, `y`)
  square and every aligned 256-code run one 16 × 16 square, checked against decoded `Morton8x8::{x, y}`.
  This extends DR's pinned 4 × 4 prefix test; the test belongs beside it in `morton8x8.rs`, owned by DR.
  Disable run: shift the block boundary by one code; the test turns red.
- G5 covers frame refusal on reuse: D-DRF-1 G3 / G4 / G6 applied to a DTO binding, not a second copy.
- The earlier withdrawal of G2 and G3 (2026-10-30) is superseded by the correction at the top.

## Errata to the shared diagram

The DR64/DR256 overview says "65,336 global addresses". It should say 65,536 (D-DRF-1 §1: 256 × 256).
