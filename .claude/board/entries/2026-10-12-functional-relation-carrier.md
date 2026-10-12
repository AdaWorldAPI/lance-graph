# 2026-10-12 — D-CMH-G1-4a: functional relation carrier in the method glove

**Status:** In PR. **Plan:** `.claude/plans/gremlin-composable-algebra-v1.md` §8.

A `Glove` carries `Relation`s (key field, target glove, foreign plane and lane
indices). `out(rel)` composes to `Cond::Reaches` → `Filter::Semijoin` over the
target's live plane; `out(rel).has(f, eq/neq)` adds `Filter::EqU32Via`. The
anchor stays, so `count()` after the hop counts anchor rows. No new op in
Quack, mask-risc or R2IL.

MEASURED (tests/relation.rs, row loop over field text, n ∈ {1,64,65,4097}):
all answers equal; six disable runs red on the test that names each guard.
Removing the liveness test fails the row oracle, so `EqU32Via` alone is not
`out(rel)`: it keeps rows whose key names a dead target.

OPEN: R2IL's fold band has `VIA` but no `GATHER`, so it cannot test that a key
names a live row; relation methods return `CallError::FrontendGap { R2il,
"GATHER" }`. Fan-out (`in`, M:N, `E()`) and a second hop stay refused;
D-SAG-1-4's multi-hop dependency is still blocked on the second hop.
