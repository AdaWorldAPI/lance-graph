# 2026-10-07 — TinkerPop + JanusGraph harvest: no new V4 op; four transferable laws

## FINDING (primary source: TinkerPop `82242e7f`, JanusGraph `dddfdcd1`)

- **TinkerPop merge key = live distinctions.** Traversers merge only when
  object, next step, tags, loops and (labelled) path are equal; bulk sums.
  `withBulk(false)` still merges but discards multiplicity: that is support
  (S), and bulk is per-node walk count (K).
- **WeightedSupport = K = Quack's `GroupReduce{Local(dst), Count}`.** It is
  sufficient for `count(*)` and group-by-terminal counts, and insufficient for
  earlier-variable questions and TRAIL (`carrier_sufficiency.py --min`,
  exhaustive). No new carrier type is needed.
- **GValue pinning is the prepared-program invalidation rule.** A parameter
  whose value a rewrite read is pinned and invalidates the plan; an unread one
  is execute-only.
- **JanusGraph's slice prefix rule** (eq…eq + one range) is our
  `OrderedLaneWitness` → `Cmp::Range` lowering. Measured locally at 30–38× at
  0.1–1 % selectivity (`bounded_slice_probe.rs`).

## REJECTED

- LazyBarrier as a mechanism: it compensates for object traversers; a mask op
  is already a barrier (F6).
- multiQuery/prefetch: remote-latency compensation (F7).
- Full-copy edge duplication.

## OPEN

- No ordering witness exists for scalar property lanes (only facet keys).
- cypher-quack's `dst_ordered: bool` should become a witness.
- K cannot yet be chained into a next hop (foreign-value sum).

Report: `.claude/research/D-TINKER-JANUS-FOLD-0.md`.
