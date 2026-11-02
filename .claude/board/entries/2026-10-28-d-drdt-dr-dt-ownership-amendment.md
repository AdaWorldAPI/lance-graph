# D-DRDT: DR = HHTL16 Cartesian direct routing, DT = ClassId binding store

**Status:** PROPOSED / DOCS ONLY. Plan: [d-drdt-dr-dt-ownership-amendment-v1.md](../../plans/d-drdt-dr-dt-ownership-amendment-v1.md). Proof: [#1834](https://github.com/AdaWorldAPI/lance-graph/pull/1834) (D-DRCA-1). Read at lance-graph `main` `ea709af`.

## Decisions recorded (user, 2026-10-28)

DR is the agnostic adapter plate: HHTL16 Cartesian addressing, optional amortized global→local mapping, fold-based projection (Identity × Identity optional, Identity × Property primary, Property × Property derived, cross-table Identity × Property through a witnessed FK). DT is the existing ClassId-addressed binding store for class schema, meaning and methods; `ClassView × WideFieldMask` is shared contract. DR #4 routes method operations over mapped coordinates; semantics stay in DT. Thought-Git sits above both.

## Supersedes

D-DT64-6 §1 "WHEN × ON × DO attaches per DR#3 cell" (method semantics are DT's, reached through the cell's class); Bardioc-rs #9 "DT64 = DTO Tensor 64 folding register" (folding is DR's, DT is the binding store). D-DT64-6's "DR64 is direct routing" and "DT64 is the DTO bind store" stand.

## Findings (VERIFIED-IN-CODE / MEASURED)

- The HHTL16 Cartesian address already exists: `Morton8x8`. Rotation already has a rule: an address map, never a mask op (`mask-risc/tests/rotation.rs`).
- Every DR projection is an address map plus an existing Boolean or count fold; no new graph engine.
- Gaps: no bit-column read / bit transpose in `ndarray::simd`; `Chain::eval` copies leaves and allocates; no DR256::WHERE × DR#3 test; DT has no MethodId independent of names, single-field guards only, no effect order.

## Open

DT MethodId / guard / effect-order contract (contract owner); DR #4 code; DR256 composition; EXO-46, EXO-140, EXO-148, NFR-OP-2 with their owners.
