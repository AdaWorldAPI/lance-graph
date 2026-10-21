# 2026-10-21 — D-CMR-2: the contract set algebra lowered onto mask-risc

**Status:** TEST-PINNED (in PR, branch `claude/eager-albattani-iufugq`). **Basis:** plan `d-cmr-concept-chain-normalization-v1.md` §3.4 and the D-CMR-2 row; ADMISSION B (`2026-10-20-six-decisions.md` §1) for the seeks.

- **Where:** `crates/lance-graph/src/fold/set_lower.rs` (`lance_graph::fold::set_lower`). Core already depends on the contract and mask-risc unconditionally; no new dependency.
- **Split:** `SetPlan::new(&SetExpr, n_rows, Option<&SortedBasis>)` normalizes, then takes `Runs` from comparisons on the attested sorted lane (an exact basis-only expression, or the basis conjuncts of a top-level `And`) and leaves the rest as the residual. Mixed `Or`s stay residual and spend no seeks.
- **Costs apart:** `SortedBasis::attest_reads` (one scan, n keys), `SetPlan::seeks` (one binary search per boundary), `Measured::{extents, extent_rows}` (query side).
- **Levels:** geometry answers from the runs alone; optics runs the residual program (`Count`) per run via `execute_extent`; etch runs it with `Keep` into `Out::Mask`, bits outside the runs cleared.
- **Tests (6):** 400 random nested expressions over a sorted basis lane, an unsorted lane, a code lane and two planes (N = 1000, not a word multiple): count and set equal the row oracle at every level and for `unsplit`, both levels exercised, runs narrow the extents (anti-vacuity). Geometry runs against `Planes` with no lanes and no masks and reports 0 extents (≤ 6 seeks). Experiment C (basis range + unsorted predicate → one run + residual, extent under half the rows). Non-basis range stays residual (silence twin). Unsorted basis and row-count mismatches refused.
- **Disable runs** (`cargo test -p lance-graph --lib set_lower`):

  | probe | red |
  |---|---|
  | geometry off (no residual still runs a program) | `geometry_reads_no_plane_lane_or_extent` |
  | split off (`new` = `unsplit`) | geometry, Experiment C, oracle anti-vacuity |
  | basis-lane check off (any lane's comparison becomes runs) | Experiment C, silence twin, oracle |
  | `keep` does not clear the destination | oracle (`keep`) |

- **Not done:** D-CMR-2b, a scratch-free `Pred` fold in mask-risc. `fused_ternlog` declines `Pred`, so a residual comparison runs tiled with per-tile scratch (bounded by `TILE_WORDS`, never population-sized). No rows/words-read counter exists in mask-risc; the geometry claim is proven by running against empty planes instead.

OPEN: D-CMR-2b (mask-risc fused predicate fold; changes `ir.rs` / `exec.rs`, coordinate on EXO-50 first); a `u32` / facet sorted basis (`SealedFacetLane`) instead of an `i32` lane.
