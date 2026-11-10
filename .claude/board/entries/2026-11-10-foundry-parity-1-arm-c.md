# 2026-11-10 — D-FOUNDRY-PARITY-1-P1A: per-store cost as one INT8 GEMM; LGJ knowledge transfer

**Status:** MEASURED (lab, read-only Level R). Crate `crates/foundry-parity-lab`
(arm C, `src/arm_c.rs`), merged in #1995 (`15d4b4c`). Plan:
`.claude/plans/foundry-parity-v1.md` §5a. Linear: EXO-223.

**Question.** Can the per-store ingredient cost that mask-RISC cannot express
(register G1) be answered over the same DT64 binding with exact parity?

**Answer.** Yes. `C[store][item] = Σ_k cost[store][k]·demand[k][item]` is one
`u8 × i8 → i32` GEMM through `ndarray::simd::int8_gemm_packed_strided`
(ndarray#418, AMX `TDPBUSD` tier on the lab host). 24 tests agree with the
scalar oracle and the materialized null model (64 stores × 8 items ×
q ∈ {1, 3, 128}); arm C summed over arm B's `CanCook` mask equals arm B's
`network_cost`. Refused: a binding missing a demanded cost lane (`NotBound`,
Codex P2 on #1995), a cost that does not fit `u8` (never truncated), a stale
frame after a price change, a demand band from another menu revision, q
outside `1..=128`. Per-store cost 0.65–1.17 µs against 1.19–1.66 µs for arm A
over hoisted recipes; A is staged once per binding (4 096 B, counted).

**Not established:** the limiting resource (element-wise min/argmin, the G1
remainder, P1b — a decision); the kernel's scalar tier (the host has AMX);
scaling past one mask word.

**LGJ knowledge transfer (read-only; EXO-223 comment).** lance-graph-java is
read as a consumer, never a dependency. Already solved in DT64: lazy
filtering with one crossing per terminal, projection without hydration,
named materialisation, lookup counting. Transferable: generated vocabularies
must dispatch through `ClassView`; a mechanical consumer-surface guard; a
source declaring a second stride reading of the same bytes (would remove arm
C's staging). R11: identity is already layout-free (`LaneAddr` has no stride),
but nothing witnesses layout — `FrameRevisions` has no layout field, so a
re-layout is invisible to `check`. That is D-DRISA-M1. LGJ `rbac.rs` unread.

**OPEN:** G1 remainder (P1b); D-DRISA-M1 layout witness; generator from
`Resolved` to a typed vocabulary; G2–G6 unchanged.
