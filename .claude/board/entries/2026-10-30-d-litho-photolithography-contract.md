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
- Thoughts: reserve, don't claim. The 64 positions are reserved at fixed width and marked absent by the presence bit until a thought is committed through Thought-Git.
