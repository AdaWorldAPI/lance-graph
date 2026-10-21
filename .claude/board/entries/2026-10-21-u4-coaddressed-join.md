# 2026-10-21 — D-RU-4 (U4): the co-addressed join as a named operator

**Status:** FAILED GATE (no-ms rule, `CLAUDE.md` P0, 2026-10-21) — correctness MEASURED, speed not (lab; `crates/crepe-plan-probe/src/coaddressed.rs`, `examples/coaddressed.rs`). Plan: `.claude/plans/d-relalg-unified-v1.md` §5.3, §6, stage table (U4). Builds on U2 (`relalg::run_with_program`) and the D-MPS-8 table (`mork_pathmap_probe::join`).

## What landed

- `coaddressed::eval(table, plan, rels, current)`: a conjunction of per-table `Rel`s over one `JoinDomain`. For each side `i`:
  - `rel_i` runs as a relalg fold over side `i`'s own rows (lane 0 = the side's other column);
  - one mask-risc `GroupReduce Count` keyed by the side's slot lane turns the selected rows into per-slot counts;
  - a side whose `Rel` normalizes to `All` reads its presence plane and offsets as prepared, with no row read.
- The joint is relalg `And(Plane(i))` over the per-side slot planes. Answers:
  - `distinct_centers`, the popcount of the joint;
  - `bag_count`, a co-addressed dot product of the per-side counts (`Σ_slot Π_i count_i[slot]`) that enumerates no row;
  - `matching(pattern)`, the derivations whose head reads as a pattern: a fold over the joint, no row built. (This replaced the first version's `bindings`, a tuple builder, in `1d1c0b77` after #1657 removed `native::bindings`.)
- `CoTable::prepare` lays out every side without its restriction. `plan_rels` turns a `StarPlan`'s restrictions into `value == k` (or `Empty` for a constant above `u32::MAX`). A restriction is therefore a query-time `Rel` and no longer part of the layout. The `Rel` can be anything relalg lowers on a `U32` lane (`==`, `!=`, `Or`, `Not`), which the star path cannot express.
- The slot lane is built from the side's own offsets, so it is non-decreasing. A row is counted in the run it already sits in, and no operand is read at a position taken from another operand's values. `CoEval::programs` keeps every program run so a caller can check that with `native::has_foreign_hop`.
- Refusals: a table holding other sides than the plan reads, the wrong number of `Rel`s or generations, a stale side.
- `mork_pathmap_probe::join::Side` gains two read-only accessors, `values()` and `offsets()`.

## Measured — failed gate

The benchmark (`examples/coaddressed.rs`: 64 queries `Out(x, c, z) <- R(x, c), S(c, z), T(c, q)`, 200,000 rows per relation, 20,000 centers) produced results in milliseconds for every arm: the per-query layout of the star path (`native::prepare`, with and without semijoin) and this operator alike. Under the `CLAUDE.md` P0 rule (2026-10-21) a millisecond result is a failed gate, not a measurement, so no timing is reported here. The first version of this entry and of #1658's body reported them; they are withdrawn.

What stays true and is not a timing: the operator lays the table out once and reads only the rows of sides whose `Rel` is not `All` (here T's value and slot lanes, 200k rows per query); the star path lays out all sides again per query.

**Where the operator leaves the mask-risc terminals** (from the code; not attributed by measurement):
- each filtered side keeps a full row mask (`Measure::Keep`) instead of folding into the terminal;
- the projection's `GroupReduce Count` sinks into an `i64` buffer of domain size (`8 × domain` bytes);
- the slot plane is built from those counts by a scalar Rust loop over the domain;
- `bag_count` / `matching` walk the joint bit by bit and multiply side counts in scalar code per slot;
- each filtered side runs two programs (fold, then projection).

**Correctness.**
- `parity_with_the_star_path`: for five restriction combinations (none; one; two; three; a constant above `u32::MAX`), the bag count, distinct centers and pinned per-derivation counts equal the star path (`native::matching` / `distinct_centers`). At least four of the five cases have answers.
- `disjunctive_restrictions_match_the_reference`: `Or`, `Not` and `!=` restrictions equal a nested-loop reference, bag and set.
- The benchmark asserts that the answers of every arm are equal on every rep, and `cargo test --example coaddressed` runs the same comparison on a small fixture in CI.
- **Disables:** five, each red on the test it targets:
  - a projection that scatters by the slot lane (`ScatterOrU32`, answer-correct) fails only `no_gather`, at the gather assertion;
  - row bits ignored in tuple enumeration;
  - per-slot counts replaced by run lengths;
  - witness check removed;
  - stale-generation check removed.
- The first scatter disable used `ScatterCountU32` with an `Out::I64`. It failed on an `ExecError` before the gather assertion ran, so it proved nothing about that assertion. It was replaced with the answer-correct `ScatterOrU32` version above.

## Open

- **The speed gate.** Bring the per-query path to the mask-risc terminals: fold each side's `Rel` straight into the slot projection, sink into a plane rather than an `i64` buffer, and answer counts through a terminal over the kept joint. Until that runs in nanoseconds per operation (microseconds per batch), D-RU-4 is not shipped.

- The projection writes per-slot counts as `i64` for every filtered side (`8 × domain` bytes), and builds the slot plane from them in a Rust loop over the domain. A fused "count into plane" terminal would remove both; not built, not measured.
- Each filtered side runs two programs (fold, then projection) and keeps the row mask for tuple enumeration. When only counts are asked, the fold and the projection could be one program; not measured.
- No restriction on the center itself (a `Rel` over the domain) yet.
- §6 factorization beyond the star (a chain through two domains) is not built.
