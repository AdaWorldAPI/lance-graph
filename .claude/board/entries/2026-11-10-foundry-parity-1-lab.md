# 2026-11-10 — D-FOUNDRY-PARITY-1: 64-store operating network over DT64 + mask-RISC

**Status:** MEASURED (lab, read-only Level R; not production admission). Crate:
`crates/foundry-parity-lab`. Plan + evidence matrix:
`.claude/plans/foundry-parity-v1.md`.

**Question.** Can source-owned planes, one DT64 `LaneBinding` and mask-RISC
programs answer `CanCook` / `CanFulfill` / `NetworkCapacity` / network cost
over 64 stores × 64 categories with exact oracle parity and correct refusals?

**Answer.** Yes for WHERE-shaped feasibility, fulfilment, network capacity and
network cost: 16 tests agree with an independent scalar oracle and a
materialized-object null model; 0 source bytes copied; stale frame, foreign
scope, other row order, unmapped field, unbound field and out-of-range
quantity are refused; unknown stock is never read as zero; the denied store
is reported NotAuthorized. Five disable runs turn the suite red.

**No** for the per-store limiting resource and per-store cost: the admitted
algebra has no element-wise cross-lane arithmetic (register G1). The
per-store maximum is reachable only by a layer-cake sweep of monotone masks,
measured ~15 µs against 1.8 µs for walking already-built objects (which cost
21 µs and 51 968 B to build). One WHERE question: 470 ns, ~4× cheaper than
arm A. One VM (Xeon 2.30 GHz, 4 vCPU), median of 41, three runs.

**Not established:** any Foundry parity (no Foundry comparator); production
IAM (lab stand-in predicate); scenarios through DR (oracle only; no overlay
frame); actions/writeback (no contract); bound relations (ADMISSION B).

**OPEN:** G1 element-wise arithmetic decision; G2 IAM in the read path; G3
overlay frame; G4 action gate; G5 bound relations; G6 method lowering from
`MethodRef`. Scaling past one mask word not measured.
