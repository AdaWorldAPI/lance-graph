# 2026-09-30 — Cypher on masks is a replacement engine beside upstream, not a phase inside it

**Status:** VERIFIED-IN-CODE · OPEN. Plan `.claude/plans/cypher-mask-lowering-v2.md` (D-CML-0..9).

## Read (2026-09-30, `main` `0d31c54f`)
- `crates/lance-graph/Cargo.toml` `[dependencies]`: `datafusion` is a **non-optional** dependency of `lance-graph`.
- `error.rs` `GraphError`: `DataFusionError` is inside it (a variant source and a `From` impl).
- Consequence: any crate that reuses the parser has DataFusion in its build graph, even if it never calls it. Removing DataFusion from our surface is possible; removing it from the build is not possible without an upstream edit (OQ-CML-1).
- `parser.rs`, `ast.rs`, `semantic.rs` and `logical_plan.rs` do not import DataFusion themselves.
- `examples/w0b_corpus_census.rs` already drives parse → plan from outside the planner, using only the public API.
- v1's seam (inside `query.rs`) and the modelgraph plan's `NodeMapping` field (inside `config.rs`) are both edits to upstream files. v2 moves both into the new crate.
- W0-b: 113 of the 303 classified queries lower fully (37.3 %). This counts before path multiplicity is classified, so it is an **upper bound**. #1305 reports **70 of 313 (≈ 22 %)** once multiplicity is classified; that has not been re-run here (plan §5, §12 H-5). Under v2 the rest are refusals.

- #1305 closed unmerged; its six must-have facts are in plan §12 (branch `ccr-2fcc2bd3-8o7m2l` @ `67abd29` keeps the rest). Its `consumer_semantics()` is not ported: per-path carriers are refused, not carried.

## Open
- Whether the assumed upstream/fork split of files holds (D-CML-0).
- Which in-row carriers hold absolute vs relative targets, and whether mask-risc needs a `Shift` op (OQ-CML-2).
- Whether `lance-graph-python` is fork-owned (OQ-CML-3).
