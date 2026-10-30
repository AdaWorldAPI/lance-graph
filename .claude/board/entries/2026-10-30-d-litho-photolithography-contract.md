# D-LITHO: the photolithography contract

**Status:** PROPOSED, docs only. Plan `.claude/plans/d-litho-photolithography-contract-v1.md`. Coordination: Linear EXO-50, messages 10–11 (2026-10-30).

## Result

- Records three user decisions (2026-10-30):
  - photolithography is the north star of the relational shader;
  - the Abacus is the addressing contract and shader mental model (frame, rod, bead position), not a materialization or cost claim;
  - views are deterministic address laws over the same words, never rotations.
- Vocabulary table: wafer = substrate, die = demanded tile, reticle = predicate or address law, exposure = field-parallel instruction, dose = rod k, alignment = version witness, develop = demanded fold/terminal, etch = witnessed deltas → async writer.
- Morton fact (from `morton8x8.rs`): in code order each 64-row mask word is an aligned 8 × 8 (x, y) square and each 256-row block is a 16 × 16 square. So the DR64/DR256 "active words / blocks" are spatial tiles with no remapping. A projection is an And with a law mask, and its fold is a popcount.
- Clarifies D-DRDT's "rotation of Cartesian axes" as a change of law; no transpose kernel is implied. D-DRDT is not edited.
- Gates G1–G5 are pre-registered; none run.

## Open

- Q1: does the rod belong to DR's address or to DT's field law?
- Q2: is DialEgg the place for law-equivalence proofs?
- Q3: which mask-risc verbs a consumer's first operation lacks; unasked until G1.
- Q4: "DR64K" and "Membrane64" are undefined in code and plans.
- Errata: the shared DR64/DR256 diagram says 65,336 addresses; the correct number is 65,536.

## Amendment (2026-10-30, plan §3a)

- ~~The address space is fixed-width in the identity direction only.~~ Corrected the same day: every axis assumes fixed width 64 or 256 for now. Underflow is carried by the presence bit (sparse write, merge on read).
- Edges and properties are selected against it. Either a cross-product coordinate-range reference (two ranges, not necessarily materialized) or a PSCustomObject-style named selection (`ClassView × WideFieldMask`).
- The selection is versioned via a lance-graph table. Q5, which table, is open for the DR/DT owner.
- Always present: identity = `NodeGuid`, and property mappings. Not material: `ClassView` → methods, and 64 thoughts → Thought-Git (references, resolved on demand).
- DR does the amortized global (`NodeGuid`) → local (frame position) mapping: prepared once per focus, sealed and versioned. Exposures work on local positions only.
- Order: DR routes (global → local, address → classid), then DT64 assigns meaning and methods to the mapped position. DT64 does not route; DR assigns no meaning.
- Thoughts: reserve, don't claim; the model is thin provisioning. The 64 positions are reserved as logical address space and absent (presence bit). Storage is allocated only when a thought lands via `Commit`. A live branch's pool slot is occupancy, not allocation. Thick provisioning (all 64, or a dense block, up front) is forbidden.

## Scope correction (2026-10-30, user)

"You're the DTO layer of a holographic hypercube 64ⁿ or 256ⁿ. Everything in regards to dimensions is not your concern."

- The plan is rewritten as the DTO layer over DR's hypercube.
- Removed as DR's concern, already written in D-DRF-1 §1–§3:
  - the Morton fact above;
  - the "rotation" reading;
  - the address law;
  - axis widths, presence-bit underflow and coordinate freezing (now listed under "assumed from DR", by reference);
  - Q1, and gates G2 and G3.
- Kept as DTO-layer:
  - identity = `NodeGuid`;
  - property mappings (`ClassView × WideFieldMask`, versioned, Q5);
  - methods by reference;
  - thoughts reserve-don't-claim (thin provisioning);
  - the order DR routes → DT64 assigns;
  - gates G1, G4, G5.
- Lesson: D-DRF-1 was not read before geometry was written into this plan.

## Correction to the scope correction (2026-10-30, user)

"Of course you need to know the dimensions in order to apply the law effectively."

- The removal above overcorrected. The error was stating DR's dimensions as this contract's decisions, not stating them.
- Plan §4 is now "What the DTO layer must know": DR's facts, each cited (D-DRF-1 §1–§3, `morton8x8.rs`, D-DRAV-1, `dr256-where-capstone-v1.md`, `mask-risc` `ir.rs:329-337` and `tests/rotation.rs`), with what the DTO layer does with each. The key one: D-DRF-1 says axis semantics are declared by the caller, and the DTO layer is that caller.
- The 8 × 8 / 16 × 16 Morton tile reading is derived from the prefix property and pinned only for 4 × 4; it is gated as G3 before use.
- Gates G2 (law, not rotation) and G3 (Morton tile) are restored. Q1 stays withdrawn: deciding rod width is still DR's.
- Strikes the "Removed as DR's concern" list above; it stays in place as history.

MIRROR
- LESSON: geometry was first written without reading DR's plans, then removed instead of attributed; both times the sources were not read in full.

## Correction (2026-10-30, user): metaphor, and the bind store

- Photolithography was the user's teaching metaphor, not a production name. The plan is now a reading aid that maps the metaphor onto existing seams (banner at the top of the plan).
- "XLOG is the first consumer of the bind store" is withdrawn. D-NFR-0 S5 says the binding store is MISSING. D-DT64-Q0 says the binding is an interface seam, not a store. D-NFR-1 (the S5 × S6 adapter) is the smallest seam and is not built.
- XLOG's `BagNeedsMultiplicity` is S8's existing refusal in `bind_dr64`. D-QSP already says a tile carries no bag multiplicity without a further lawful carrier. The dose test (G4) stays as a measured fact, not a proposed carrier.

MIRROR
- BLIND SPOT: the binding plans (Q0, D-NFR-0, D-QSP, D-QR-CAP) were not read before the binding was described.

## Correction to the correction (2026-10-30, user): the bind store plans are authoritative

- DT is the bind store (D-DT64-6 §1 D4; D-DRDT §1): the existing ClassId-addressed DTO binding store. It holds bindings and their version log, never business data.
- D-NFR-0 S5's "MISSING" is about Quack, which is the query-binding membrane, not about DT.
- XLOG (#1864) is a consumer of the DT bind store. "There is no bind store yet" is withdrawn.

## Folded (2026-10-30, user)

- The plan is folded into the bind store plan as D-DT64-6 Amendment A. The D-LITHO file is now only a pointer.
