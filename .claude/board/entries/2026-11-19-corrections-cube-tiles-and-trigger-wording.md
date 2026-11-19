# Corrections to two of this session's own documents (2026-11-19)

**Status:** VERIFIED-IN-CODE for the cube facts; wording only for the plan.

- `.claude/knowledge/classid-on-do-call-shape.md` §4 said the cube's
  "row-returning and counting terminals work on one tile (at most 256 ids)".
  Wrong for the counting terminals. `Cube::new_tiled` spans tiles; four entry
  points refuse a tiled cube (`add_relation`, `materialize_rows`,
  `body_answer`, `materialize_witnesses`); `bag_count`, `column_counts` and
  `join_size` have no tiled check. Found by the OpenObserve x sentinel-rs
  session (EXO-338), which listed three of the four refusals.
- `.claude/plans/ontology-method-registry-v1.md` §2 said "ON: which
  predicates trigger". Reworded: nothing is triggered, a method is called.

Open: the header at `cube.rs:28` and the dt64 doc comments still carry the
old wording (EXO-338, EXO-336). Not changed here.
