# Query stage measurement — never cite a combined number for one stage

> **READ BY:** `query-stage-profiler`, and any session reporting Cypher / SQL /
> Quack latency, a parser speed-up, or a prepared-query win.
> **Status:** CURRENT-CONTRACT; instruments are committed.

## Rules (each from a measured trap)

1. **Split the stages**: parse · bind · logical plan · engine plan ·
   registration · physical plan · execute · materialise. A total is never
   evidence for one stage. (Parse is < 0.1 % of a cold DataFusion query; a
   parser rewrite judged on totals would look like a win or a loss for reasons
   unrelated to the parser.)
2. **Count allocations** with a counting global allocator, and run an
   allocation-only control, so allocator cost is separable from parser work.
3. **Measure on a quiet machine.** Runs taken while cargo compiles are
   discarded, not averaged.
4. **Cold vs prepared are different questions.** DataFusion physical plans
   cannot be re-executed (`partition not used yet`); "prepared" for DataFusion
   means cache the logical plan and rebuild the physical plan.
5. **Name what dominates before optimising it.** In the bundle probe the
   semijoin alone was 95 % of every mask route; reordering conjuncts could not
   help because the gate never reached it. Run a per-conjunct control.
6. **Assert the answer before printing the time.** Every route, every case,
   against an oracle that touches no engine.
7. **A disable that does not apply proves nothing.** Assert the patch anchor
   matched (`assert s.count(old) == 1`); rustfmt reflows lines and silently
   breaks string-anchored disables.

## Instruments

- `crates/lance-graph-benches/examples/cypher_stage_probe/` — `df`, `parser`,
  `quack`, `prepared`, `reexec` modes; `common.rs` builds against upstream too.
- `crates/lance-graph-benches/examples/bundle_probe.rs` — one bound query,
  five physical routes, density ladder.
- `.claude/tools/carrier_sufficiency.py --min` — cheapest sufficient carrier
  per demand.
