# 2026-10-21 — D-RU-3 (U3): cross-rule CSE in crepe-plan-probe

**Status:** MEASURED (lab; `crates/crepe-plan-probe/src/cse.rs`, `examples/rule_cse.rs`). Plan: `.claude/plans/d-relalg-unified-v1.md` §7, §12, stage table (U3). Builds on U2 (`contract::set_program`, `relalg::run_program`, #1649).

## What landed

- `cse::evaluate_shared(rules, demand, kind, opts, store, layout, sharing)`. It makes the same rewrites as `pipeline::evaluate` and gives the same answer. Per group of rules it does three things:
  - lays out each distinct `SideKey` once, over one domain;
  - writes every rule as `And(Sym(side))`;
  - with `sharing` on, lifts shared parts into lets (`set_program::share`) and runs the group through `relalg::run_program`.
- `Layout::Components`: rules that share a side, directly or through other rules, share a table. A rule that shares nothing keeps a table of its own. `Layout::OneTable` puts every rule into one table.
- Tuples are read per rule by `native::tuples_over_mask`. This is the per-rule path's own read, moved into a helper. The 41 existing tests are unchanged.
- Refusals: a stale side (`JoinRefusal::Stale`), a generation vector of the wrong shape, and a program for another version.
- The shared path ignores `Options::semijoin`. A semijoin's reduced domain is per rule.

## Measured

Workload: 256 rules `Out(k,x,z) <- R_g(x,c), S_g(c,z), T(c,k)`. The first `p · 256` rules use core 0; every other rule has a core of its own. Release build, 4 cores, medians.
- Arms: A is the per-rule pipeline; U is one table with lets; D0 is components without lets; D is components with lets.
- `exists` runs only the folds. `bindings` reads the full bag.

200,000 centers, 20,000 rows per core relation (3 reps):

| core shared | arm | prepare | rows prepared | folds / ops | fold time | share (compile) | tuples |
|---|---|---|---|---|---|---|---|
| 0 % | A | 1851 ms | 15.3 M | 256 / 512 | — | — | 132 ms |
| 0 % | U | 2721 ms | 15.3 M | 256 / 512 | 2.38 ms | 1.1 ms | 259 ms |
| 0 % | D0 | 1794 ms | 15.3 M | 256 / 512 | 0.86 ms | — | 130 ms |
| 90 % | A | 1777 ms | 15.3 M | 256 / 512 | — | — | 142 ms |
| 90 % | D0 | 1284 ms | 6.2 M | 256 / 512 | 1.39 ms | — | 175 ms |
| 90 % | D | 1218 ms | 6.2 M | 257 / 283 | 1.18 ms | 7.1 ms | 169 ms |

At 4,000 centers (5 reps) the pattern is the same:
- 90 %: prepare A 197 ms, D0 35 ms, D 36 ms;
- `exists` folds: D0 0.44 ms, D 0.41 ms;
- `share` 7.7 ms.

**Findings.**
- **The win is the layout, not the fold.** At 90 % the shared core is laid out once instead of 230 times: 15.3 M rows become 6.2 M, and prepare falls 1.78 s → 1.22 s. That is −31 % end to end for `exists` and −27 % for `bindings`.
- **Lets cut ops 512 → 283, but fold time is negligible here** (≤ 2.4 ms against ≥ 1.2 s per run). `share` costs 7–8 ms of compile time at 90 %, more than the folds it saves. The pairwise subset search is quadratic in the number of rules.
- **One table for everything is wrong.** Every side is laid out densely over its table's domain. With nothing shared, U is slower than A: prepare 2.72 s vs 1.85 s, tuples 259 ms vs 132 ms. That is why `Layout::Components` exists. With it, D0 at 0 % lays out exactly A's rows, one table per rule.
- **A shared group pays width.** At 90 % the core-0 group spans almost all centers, so tuple reads are 175 ms vs 142 ms for A. The layout saving outweighs that.

**Correctness.**
- Every arm, under both layouts, equals:
  - the reference evaluator (bags);
  - `crepe!` and `ascent!` (sets; the 256 rules written as one parametrized rule);
  - arm A.
- Checked at 0/50/90/100 % shared in the tests, and at 0/50/90 % at 4,000 centers in the benchmark.
- Bound demands go through the post filter, with and without propagation.
- **Disables:** seven, all red:
  - sharing ignored;
  - wrong table side read;
  - generation check removed;
  - keys not deduplicated;
  - post filter dropped;
  - components never joined;
  - components always one table.
- The post-filter disable first did not compile. The suite had no bound demand, so a test was added first.

## Open

- `share`'s subset search costs more than it saves when planes are this small. Two possible fixes, neither measured: a cheaper share for star rules (group by identical core keys), or skipping lets when one plane is cheaper than the search.
- Factorization (plan arm E) and a shared BIND (arm F) are not built.
- The JC batch through the same lets (arm H) is not built.
