# 2026-10-06 — DAV probe: resident bit-plane carrier, population boundary (Round 2)

## MEASURED

Carrier C in `crates/jc/examples/dav_active_observation_probe.rs`
(`resident_differential`) reads the horizon's `independent_roots` (validity)
and `projected_claims` (value) as two borrowed `&[u64]` planes. It builds no
value lane. A == B == C over 5 apertures × 4 poisons × 2 orders, covering
candidate ids, disagreement, EWA weight, score and ranking. The DAV and
ordinal cycles also match: delta, resulting horizon, replay, residual and
final state. Poison for C sets `projected_claims` at unobserved sites.

C has no SQL NULL semantics (a plane is never nullable to Quack). B stays the
NULL test.

k = 1 is a streaming fold over the kept mask (`top1_fold`, state one
`Option<Candidate>`). It matches A's `ranked[0]` everywhere. The ordinal
pick is the lowest set bit.

Heap @ a=0.55, 2 candidates, plans lowered outside the window:
- C candidate selection + visitation: 0 B
- C streaming top-1: 8 B = 2 × 4 B, from `jc::quorum::pairwise_agreement_u8`'s
  result `Vec`, which is route semantics and unchanged
- C full rank: 586 B; B full rank: 762 B

Derived population bytes on C: the kept mask, 8 B, a caller-owned stack
array. `stay-on-the-lane.md` §6 counts that as materialization. The candidate
program's lowering is `Tiled`; no fused lowering keeps membership without
writing it.

Disable runs, each red: leaky gate (projected bit at an unobserved site read
as observed-true); C selecting on the value plane; one value bit flipped; the
fold's tie-break dropped; the validity gate removed; a `Vec` row-id list in
the selection window; the tie-break can-fire made vacuous.

Inverting EVERY value bit leaves ranking unchanged. The route disagreement is
symmetric under a global flip, so only the final-state check catches it.

## FINDING

No shipped terminal returns a position. The list ends at Count/Any/All/
Masked{Sum,Min,Max}I32/Keep/Scatter*/CountKeyRuns/Group*.
`MaskedMaxI32` returns a value, not its row. The DAV score is not a lane
either: it is per-site route semantics in Rust. So top-1 in mask-risc would
need two things. One is a masked argmax with an ordinal tie-break, over a lane
whose order matches the score order exactly. The other is the score as a
lane, which would be a population-sized derived lane. Neither is built.

## OPEN

- The kept mask is the first population object. Visiting it needs
  membership out of mask-risc. No fused lowering hands a word to a consumer
  visitor without a `Keep` write.
