# 2026-10-12 — D-CMH-G1-7: chained hops are folded hops (mask-risc + Quack)

**Status:** In PR (needs AdaWorldAPI/ndarray#358). **Plan:** `.claude/plans/gremlin-composable-algebra-v1.md`.

`MaskOp::GatherChain { lane, chain, under, dst }` with `Program::chains`:
a key leads through every hop of the chain, each hop's reached row live in
its RESIDENT plane and passing its tests read in place, each hop's `next`
key naming the following hop's row. One op, one terminal, no mask over any
hop table, no program feeding another. Realised by
`ndarray::simd::mask_gather_chain_u32[_under]`. Quack's `Filter::Chain`
lowers to exactly one such op in both lowerings.

WHY a new op: `Gather` tests a target only through a plane over the
target's table, so a depth-2 predicate needed a depth-1 mask built first
and a `Keep` fed into another program's `Gather`, the shape the
`ScatterOrU32` survival condition forbids. (#1521, which built that
per-hop shape, was closed.)

MEASURED: `mask-risc/tests/chain.rs` (executor, oracle and a fixture row
loop agree gated and ungated across tile boundaries; one bare hop equals
`Gather`; eight refusals identical on both paths); `no_alloc.rs` (1000
gated two-hop executes, 0 bytes); Quack
`chain_folds_to_one_gather_chain_and_matches_a_per_row_reading`. Seven
disable runs red (gate dropped, tests dropped, oracle negate, chain shape
check, lowering negate, lowering to `Gather`, oracle stopping at depth 1).

DECISION: chains are stack-bounded (`MAX_CHAIN_HOPS = 8`,
`MAX_HOP_TESTS = 8`) so `execute` keeps L1 (no allocation). REVISIT WHEN: a
real query needs a deeper chain.

Glove (plan §11): `Path` + `Cond::ReachesPath` / `ViaPath`; every
condition on a path and its prefixes binds into one chain; Gremlin `out`
after `out` extends the path. `tests/two_hop.rs` against a row loop over the
text; six more disable runs red.

OPEN: the M:N two-edge-table hop (run-level OR) awaits a decision.
