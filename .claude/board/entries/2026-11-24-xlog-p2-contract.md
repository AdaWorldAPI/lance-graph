# 2026-11-24 — XLOG P2 contract: virtual correlated n-ary relations

**Status:** CONTRACT, docs only. EXO-395 under EXO-377.

Plan: `.claude/plans/xlog-p2-virtual-correlated-relations-v1.md`.

- A derived n-ary relation is a recipe, not a population. Readers unfold it; a
  recursive SCC through it, a negated or aggregated read, a diagonal, a
  constant, more than 8 classes or a non-binary base atom is a typed refusal.
- Rebuilding a head from per-variable supports is wrong: the 2-triangle twin
  has 2 tuples, the product of its supports 8; `p(X,Z)` has 2 pairs, the
  product 4.
- MISSING: a count with an existential variable, and a binary head projection.
  `body_pair` is a hypothesis with unmeasured cost; over the no-millisecond
  gate it stays a refusal.
- All six corpus falsifiers named for P2 are recursive SCCs (P4). P2's own
  gates use their non-recursive base steps.

OPEN: how the 68 candidate rules split into P2 / P3 / P4 / out of scope (needs
an SCC column in the census); `body_pair` cost; EXO-108 scope.
