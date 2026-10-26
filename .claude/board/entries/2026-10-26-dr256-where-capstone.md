# 2026-10-26 — D-DR256-WHERE capstone: P0 census + P1 occupancy / shared-schedule lab

**Status:** P0 DONE, P1 DONE (lab, reported not gated). P2–P5 PROPOSED, NOT AUTHORIZED. Amendment A (DR64³ gap inventory): `2026-10-26-dr64-cubed-amendment.md`. Amendment B (P2 Quack-path lab, DONE): see the plan § Amendment B.
Plan: `.claude/plans/dr256-where-capstone-v1.md`. Lab: `crates/drf-focus-probe/examples/where_occupancy.rs`.

- WHERE, COUNT, ANY, gated predicates, witnessed range reads, exact fk semijoin and single-property
  masked reads all exist in Quack / mask-risc today (file:line in the plan, §2).
- Correction to an earlier note: `Terminal::Any` does not exit early on the generic tile path
  (`exec.rs:1793`); only the fused ternlog path does (`exec.rs:1049`, `:1204`).
- Smallest real gap: exact occupancy (active 64-row words, active 256-row blocks). Scalar lab cost is
  about 240 ns above `popcount_batch_u64` (≈100 ns) per 8 KiB mask.
- Multi-property reads: a shared block schedule gives B/S from 0.017 (one active block) to ≈1.0 (all 256
  blocks), roughly flat in P. The gain is dead-block skipping, not sharing R's walk. The two 4,096
  fixtures (dense 0.24–0.33, scattered 0.63–0.95) show cardinality alone does not predict it.
- P=64 over dense selections exceeds 1,000 µs on both arms: failed gate.
- ndarray-level only; the Quack-path measurement is P2.

OPEN: P2 (real Quack path); whether any consumer reads occupancy to choose a strategy; the #1791
contract; S-INGRESS and the joint witness (#1788).

## P2 (2026-10-26, Amendment B)

- Reusing a kept R (`Keep` → `Plane`) costs 0.05–0.60 of recomputing R per property. It is available today.
- mask-risc does not skip dead 256-row blocks in mask ops or folds. A block schedule is 0.02–0.39 of the
  Quack reuse path at sparse selections and 0.63–1.08 at full occupancy.
- Building R costs ~20 µs regardless of size; COUNT without a mask costs the same.
- Stale codebook, scope change and selection change all run silently stale through Quack. relalg refuses
  by version value.
- `BindStore` occurs in no file in any branch's history; the binding store is proposal only (D-QSP, D-NFR S5).

## Compiler compatibility (2026-10-26)

Plan § "Future mathematical compiler compatibility". It references D-DRMC-0, D-NFR / D-NFC, D-ONNX-RELALG-CAP,
D-DR3 and D-RPR rather than restating them.

Existing lab precedents:
- #1701 / #1708: typed DR64 binding.
- #1703: law ≠ recipe ≠ binding.
- #1740: a certified recipe reused on a fresh frame, 0 egglog calls.
- #1666: `Pop` refuses stale populations by witness.
- #1659: zero-copy relation lanes.

Correction to P2: a witnessed kept population exists in the lab (#1666); a Quack `Keep` plane still has no
witness. P2 scope is unchanged.

## Evidence reconciliation (2026-10-26, plan § D)

- Binding store: the generalized Quack Binding Store is PLANNED / UNBUILT. Quack numeric binding is
  IMPLEMENTED. RelAlg let reuse is IMPLEMENTED (type in the contract, executor lab). Witnessed DR
  populations, typed DR method binding and compiled recipes are IMPLEMENTED IN LAB. Source-attested
  production integration is MISSING.
- Spine linked with source: #1649, #1653, #1659, #1666, #1681, #1685, #1693, #1694 (via #1696 / #1698),
  #1701 / #1703 (via #1708), #1707, #1740.
- New falsifier: a predicate-dependent property mutation leaves a reused R2 at 1,032 rows vs 31 fresh,
  silently.
- Smallest seams: D-NFR-1 frame (needs NFR-OP-1); an input-generation dependency witness for kept planes
  (binding store; NFR-OP-3 for the type); D-NFR-2 codebook generation.
