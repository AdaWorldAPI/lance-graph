# 2026-10-21 — D-RU-4 (U4): the co-addressed join as a named operator

**Status:** MEASURED (lab; `crates/crepe-plan-probe/src/coaddressed.rs`, `examples/coaddressed.rs`). Plan: `.claude/plans/d-relalg-unified-v1.md` §5.3, §6, stage table (U4). Builds on U2 (`relalg::run_with_program`) and the D-MPS-8 table (`mork_pathmap_probe::join`).

## What landed

- `coaddressed::eval(table, plan, rels, current)`: a conjunction of per-table `Rel`s over one `JoinDomain`. For each side `i`:
  - `rel_i` runs as a relalg fold over side `i`'s own rows (lane 0 = the side's other column);
  - one mask-risc `GroupReduce Count` keyed by the side's slot lane turns the selected rows into per-slot counts;
  - a side whose `Rel` normalizes to `All` reads its presence plane and offsets as prepared, with no row read.
- The joint is relalg `And(Plane(i))` over the per-side slot planes. Answers:
  - `distinct_centers`, the popcount of the joint;
  - `bag_count`, a co-addressed dot product of the per-side counts (`Σ_slot Π_i count_i[slot]`) that enumerates no row;
  - `bindings`, the tuples in `native::bindings`' shape, reading each side at the joint's slot and testing the row bits in that slot's run.
- `CoTable::prepare` lays out every side without its restriction. `plan_rels` turns a `StarPlan`'s restrictions into `value == k` (or `Empty` for a constant above `u32::MAX`). A restriction is therefore a query-time `Rel` and no longer part of the layout. The `Rel` can be anything relalg lowers on a `U32` lane (`==`, `!=`, `Or`, `Not`), which the star path cannot express.
- The slot lane is built from the side's own offsets, so it is non-decreasing. A row is counted in the run it already sits in, and no operand is read at a position taken from another operand's values. `CoEval::programs` keeps every program run so a caller can check that with `native::has_foreign_hop`.
- Refusals: a table holding other sides than the plan reads, the wrong number of `Rel`s or generations, a stale side.
- `mork_pathmap_probe::join::Side` gains two read-only accessors, `values()` and `offsets()`.

## Measured

Workload: 64 queries `Out(x, c, z) <- R(x, c), S(c, z), T(c, q)` for `q` in `0..64`. 200,000 rows per relation, 20,000 centers. Release build, AVX-512, best of 3.
- Arms: C is `native::prepare` per query, which lays out all three sides again with T restricted; CS adds the semijoin; O is one `CoTable` plus `eval` per query.

| arm | time, 64 queries | rows touched |
|---|---|---|
| C | 1423 ms | 25.8 M laid out |
| CS | 776 ms | 3.9 M laid out |
| O | 12.4 ms (+ 30.4 ms one layout) | 12.8 M read (T's value and slot lanes, 64 × 200k) |

**Findings.**
- With the restriction moved out of the layout, O answers all 64 queries in 42.8 ms including its one layout: 33× faster than C and 18× faster than CS.
- CS lays out fewer rows than O reads, but is still 63× slower per query. Its row count does not include the scan that filters 600k rows per query before layout, nor the sort. So "rows touched" here does not cover the same work across arms, and only the times are comparable.
- This workload restricts one side. O reads only that side's rows per query; R and S are read as prepared. A query restricting all three sides would read all three per query.

**Correctness.**
- `parity_with_the_star_path`: for five restriction combinations (none; one; two; three; a constant above `u32::MAX`), bindings (bag and set), the bag count and distinct centers equal `native::prepare` + `native::bindings` / `distinct_centers`. At least four of the five cases have answers.
- `disjunctive_restrictions_match_the_reference`: `Or`, `Not` and `!=` restrictions equal a nested-loop reference, bag and set.
- The benchmark asserts that the answers of C, CS and O are equal on every rep, and `cargo test --example coaddressed` runs the same comparison on a small fixture in CI.
- **Disables:** five, each red on the test it targets:
  - a projection that scatters by the slot lane (`ScatterOrU32`, answer-correct) fails only `no_gather`, at the gather assertion;
  - row bits ignored in tuple enumeration;
  - per-slot counts replaced by run lengths;
  - witness check removed;
  - stale-generation check removed.
- The first scatter disable used `ScatterCountU32` with an `Out::I64`. It failed on an `ExecError` before the gather assertion ran, so it proved nothing about that assertion. It was replaced with the answer-correct `ScatterOrU32` version above.

## Open

- The projection writes per-slot counts as `i64` for every filtered side (`8 × domain` bytes), and builds the slot plane from them in a Rust loop over the domain. A fused "count into plane" terminal would remove both; not built, not measured.
- Each filtered side runs two programs (fold, then projection) and keeps the row mask for tuple enumeration. When only counts are asked, the fold and the projection could be one program; not measured.
- No restriction on the center itself (a `Rel` over the domain) yet.
- §6 factorization beyond the star (a chain through two domains) is not built.
