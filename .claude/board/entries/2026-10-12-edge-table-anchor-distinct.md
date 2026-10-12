# 2026-10-12 — D-CMH-G1-4b: an edge table as the anchor, and distinct targets

**Status:** In PR. **Plan:** `.claude/plans/gremlin-composable-algebra-v1.md` §10.

An M:N relation is its own concept with a functional relation to each end, so
each edge row is one traverser: `out(end).count()` is the bag and the new
`Fold::Distinct` (`out(end).dedup().count()`) the set. Both ends are filtered at
the source through the key; no program produces a mask another reads.
`Fold::Distinct` lowers to the liveness semijoin plus
`CountDistinctOrderedU32` (two words of state) and needs the key lane in key
order; an unordered lane is refused (`LaneNotOrdered`), never miscounted.

MEASURED (tests/distinct.rs, row loop over field text, n ∈ {1,64,65,4097}):
bag and set equal the oracle and differ on this data; seven disable runs red on
the test that names each guard.

OPEN: a second hop stays refused. It needs the chained-hop decision
(`2026-10-03-coresearch-chained-hop.md`): every multi-hop question in
#1510/#1512/#1515 waits on it.
