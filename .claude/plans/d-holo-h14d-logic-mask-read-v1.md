# D-HOLO-H14d — logic → address mask → masked read → terminal (C only)

**Status:** MEASURED 2026-10-16 (v4, `tier: v4 (avx512f)`, lance 13.0.0). Results in § Results; entry `2026-10-16-d-holo-h14d-logic-mask-read.md`. Returned to the operator.
**Follows:** D-HOLO-H14c (#1573). H14, H14b and part of H14c measured C by
rebuilding rows or against row-first layouts (B, M); that framing is withdrawn
(#1569). This probe measures C's own operation and nothing else.

## Overview

A thought `t` selects lanes `a, b, …`. Its operation is

```
Thought(t)[T_a[i] : T_b[i]]   for every i the logic admits
```

and it runs in four stages, in this order, each only shrinking what the next
one touches:

| stage | what runs | touches | existing code |
|---|---|---|---|
| 1. logic | the set expression is reduced symbolically (Prolog reduces the set theory): intersection, union, difference, contradiction, absorption | no data | `lance-graph-relalg::normalize` + `symbolic` (#1546) |
| 2. address materialization | the reduced expression becomes the concrete address set `R` over the thought's presence planes `P_l` and the aperture `A`, e.g. `R = P_a ∧ P_b ∧ A` or `(P_a ∨ P_b) ∧ A` | presence planes only, 64 addresses per word | `relalg::run` → R2IL → one mask-risc program (`Measure::Keep` writes `R`, `Measure::Count` is the mask terminal) |
| 3. masked read | `R` prunes the read at the source | payload of survivors only | Lance 13 `with_row_addr_prefilter(RowAddrMask)` (as D-TELEPORT-2) |
| 4. terminal | COUNT / EXISTS from `R`; or per survivor `i`, read `T_a[i], T_b[i]` positionally and run the operation | nothing more | positional lanes (H14c read path) |

The only thing materialized before a terminal is addresses. A per-address
read is the payload terminal's correct primitive: it runs only at addresses
whose presence bits the mask has already proved. No arm B, S or M.

## Checklist

- [x] Pre-registration committed before any timing (this file)
- [x] Probe crate `crates/holo-h14d-probe` (excluded from the workspace; path deps relalg, mask-risc, r2il; registry lance 13, arrow 58, tokio)
- [x] Oracles (§ Checks) written before the timing code ran. Not run against a stub: red evidence comes from the five disable runs and from check 8 going red on the old clustered fixture (A2).
- [x] Disable runs (§ Disable runs) red, files restored; repeated after A2
- [x] v4 run, tier printed by the program
- [x] Results section, board entry with raw CSV, STATUS row (no agent run, so no AGENT_LOG entry)
- [x] PR #1575 (marked ready by the operator); no merge

## Sub-ids

| id | stage | measures |
|---|---|---|
| D-HOLO-H14d-1 | 1 | fraction of a generated program corpus decided symbolically (`Empty`/`All`, zero programs run); mask-risc passes rewritten vs as written |
| D-HOLO-H14d-2 | 2 | ns per 64-address word to materialize `R`; ns per word for the COUNT terminal |
| D-HOLO-H14d-3 | 3 | Lance rows scanned and `bytes_read` with `R` vs without it, same projection |
| D-HOLO-H14d-4 | 4 | ns per survivor for the resident payload terminal; ns per word for the fused Σ terminal (`Measure::Sum`) |
| D-HOLO-H14d-5 | 4 | ns for a single-address query: presence bits, then the loads |

## Fixture

- One tile, `N = 65,536` addresses. One thought; up to 8 lanes.
- Presence planes `P_l`: patterns uniform (independent per address), clustered
  (runs of 64), at densities 0.1 %, 1 %, 10 %, 50 %.
- Aperture `A`: a contiguous window of 1,024, 8,192 or 65,536 addresses.
- Value lanes `T_l`: positional `i32`, `T_l[i] = f(l, i)` where present and
  `POISON = i32::MIN` where absent. A read at an absent address therefore
  breaks every sum and every comparison, so the reference tests catch it.
- Program forms: `∧` over k presence planes and `A` (k = 1, 2, 4, 8); `∨` over
  k planes, then `∧ A`.
- Stage-1 corpus: seeded generator over the same planes adding duplicates,
  `x ∧ ¬x`, `x ∨ ¬x`, nested absorption (`x ∧ (x ∨ y)`), `∧ All`, `∨ Empty`;
  2,000 programs.

## Predictions (stated before the run)

1. Stage 1 decides every program containing `x ∧ ¬x` (or reduced to it) as
   `Empty` with 0 programs run, and every `x ∨ ¬x` COUNT as `All`.
2. Stage 2 cost per word is flat in density (within 1.25× between 0.1 % and
   50 %) and grows with the number of operands, not with `|R|`.
3. Stage 3: `rows_scanned == |R|` in every cell. `bytes_read` with `R` is below
   the unmasked read only where survivors are page-local (clustered, or a
   narrow aperture); at uniform 1 % it equals the unmasked bytes (H14c).
4. Stage 4: payload reads == `|R| × k` exactly; ns per survivor flat in density
   within 1.5×.
5. A single-address query costs one presence test per requested lane plus k
   loads, with no search.

No layout decision depends on these numbers. The result returns to the
operator.

## Checks (written first)

1. `R` from stage 2 equals a per-address reference for every fixture cell.
2. Stage 1 (`run`) and the unrewritten program (`run_as_written`) give the
   same answer on all 2,000 programs; the reference agrees with both.
3. COUNT terminal == `popcount(R)` == reference.
4. Resident payload terminal: Σ and the comparison count equal the reference;
   payload reads == `Σ_l popcount(R ∧ P_l)` (counted), which is `|R| × k` for
   the `∧` form.
5. Fused Σ (`Measure::Sum`) equals the explicit payload terminal.
6. Lance payload terminal: returned rows == `|R|`, values equal the reference.
7. Single-address query equals the reference at every address of a sample,
   absent included.
8. Fixture shape: the poison value really sits at every absent address, and
   each density cell has the stated `|P_l|` within 10 %.

## Disable runs (each must turn its check red; files restored after)

1. Skip stage 1 (`run_as_written` everywhere): check 2's "0 programs run" for
   contradictions must fail.
2. Read payload outside `R` (iterate the aperture, not `R`): checks 4 and 6
   must fail on the read count and on the poison.
3. Treat absent as a value (drop the presence test in the single-address
   query): check 7 must fail.
4. Hand Lance a mask that is not `R` (the aperture alone): check 6 must fail.
5. Drop `A` from the expression: check 1 must fail.

## Amendment A1 (2026-10-16, before any code or timing)

In the `∨` form a survivor of `R` need not hold every requested lane, so the
payload terminal tests lane `l`'s presence bit at `i` before loading `T_l[i]`.
The exact read count is `Σ_l popcount(R ∧ P_l)`; for the `∧` form every
survivor holds every lane and the count is `|R| × k`, as first stated.
Prediction 4 and check 4 read with this count.

## Amendment A2 (2026-10-16, after the first test run, before any timing)

Two corrections, both found by the checks before a single timing ran.

1. **Check 2, refusals.** `FoldDialect` refuses a fold wider than
   `lance_graph_r2il::MAX_SLOTS` (8) scratch slots, by design and never by
   truncation. Deep corpus programs hit it as written. Check 2 now compares
   answers only where a run answers, and stage 1 additionally reports how many
   programs are refused rewritten, refused as written, answered only after the
   rewrite, and **widened by the rewrite** (answered as written, refused
   rewritten). Only the scratch-slot refusal is accepted; any other error
   fails the check. Anti-vacuity: more than half the corpus must be answered
   both ways.
2. **Check 8, clustered fixture.** An independent draw per 64-address word
   gives only 1,024 draws per lane, so the clustered count missed the declared
   density (5,568 vs 6,554 present addresses at 10 %, lane 4): cells with the
   same label would compare different densities. Clustered presence now
   selects exactly `round(d × 1,024)` words per lane (the lowest seeded draws),
   so every clustered cell holds exactly its declared density; check 8 keeps
   the pre-registered 10 % bound and additionally asserts the exact clustered
   count. (A first correction widened the tolerance instead; review rejected
   it because it hid the density mismatch rather than removing it.)

First observation, from the corrected check (not a timing): of 2,000 programs,
1,583 are answered both ways, 396 only after the rewrite, and 2 are widened by
it. The widened case is `¬(x ∧ (x ∨ y))`: `normalize` applies De Morgan before
absorption, flattens, and leaves `¬x ∨ (¬x ∧ ¬y)`, so absorption no longer
matches. A relalg defect, recorded for its own test-first fix; this probe
counts it and does not change relalg.

## Timing protocol

v4 only, tier printed by the program; release, debug 0. Medians of 7 batches;
hot (repeated) and cold (cache flushed between batches) reported apart. ns per
word, per survivor, per query; never whole-call milliseconds. About 40 %
run-to-run noise was measured on this VM in H14c; differences below that are
not read as results.

## Results (v4, 2026-10-16)

`tier: v4 (avx512f)`, lance 13.0.0, one tile of 65,536 addresses. All eight
checks pass on the measured commit (`802db7df`); the five disable runs each turn
their check red (disable 2 turns checks 4 and 5 red, not check 6: the Lance
read uses `R` independently of the payload terminal, so the plan's "and 6" was
wrong for that disable). Raw CSV in the board entry. Timings are medians of 7
batches; about 40 % run-to-run noise applies, so single-cell differences below
that are not read.

**Stage 1 (logic, D-HOLO-H14d-1).** Of 2,000 generated programs:

| outcome | programs |
|---|---|
| decided symbolically (`Empty`/`All`, 0 programs run) | 675 (34 %) |
| answered both rewritten and as written | 1,583 |
| answered only after the rewrite (as written exceeds the 8 scratch slots) | 396 |
| refused both ways | 19 |
| widened by the rewrite (answered as written, refused rewritten) | 2 |

On the 1,583 answered both ways the rewrite runs 1,065 mask-risc passes against
5,898 as written (5.5× fewer). The reduction costs 1.9 µs per program
(`normalize` + `symbolic`, no data). Prediction 1 holds: every program that
normalizes to `Empty` or `All` ran 0 programs (check 2). The 2 widened programs
are the absorption miss in A2.

**Stage 2 (address mask, D-HOLO-H14d-2).** Hot, full aperture, ns per 64-address word:

| form | k = 1 | k = 2 | k = 4 | k = 8 |
|---|---|---|---|---|
| `∧` materialize R | 2.0–3.8 | 2.3–3.4 | 3.0–6.6 | 6.5–7.6 |
| `∨ … ∧ A` materialize R | — | 3.6–4.3 | 4.8–5.6 | 9.1–9.8 |
| COUNT (`∧`) | 1.8–2.8 | 2.2–3.4 | 3.0–5.7 | 6.2–8.5 |

That is 0.03–0.15 ns per address. Cost grows with the number of operands, not
with `|R|`. Prediction 2 (flat in density within 1.25×) holds in 11 of 14
pattern × form × k rows; the three misses are single-cell spikes (1.39×, 2.0×,
1.81×), not trends in density, and sit inside the measured noise.

**Stage 3 (Lance masked read, D-HOLO-H14d-3).** `rows_scanned == |R|` and
returned rows == `|R|` in all 96 `∧` cells (prediction 3, first half, holds).
Bytes read with `R`, as a fraction of the same projection without it, k = 1:

| aperture | uniform 0.1 % | uniform 1 % | uniform 10 % | clustered 1 % | clustered 10 % |
|---|---|---|---|---|---|
| 1,024 | 0.01 | 0.01 | 0.03 | 0.00 (R = 0) | 0.00 (R = 0) |
| 8,192 | 0.09 | 0.10 | 0.11 | 0.02 | 0.09 |
| 65,536 | 0.65 | 1.00 | 1.00 | 0.19 | 0.99 |

The aperture decides the bytes: a narrow `A` keeps survivors on a few pages.
Over the full tile, uniform 1 % reads all bytes (H14c reproduced), clustered 1 %
reads 0.19, but clustered 10 % reads 0.99: runs of 64 addresses are much
smaller than a Lance page, so 10 % of words still touch nearly every page.
Clustering helps only at page scale. An empty `R` reads 0 bytes. Prediction 3,
second half, holds with that refinement.

**Stage 4 (terminals, D-HOLO-H14d-4/-5).** Payload reads == `|R| × k` in every
`∧` cell (check 4). The explicit payload terminal costs about 6 ns per lane
load per survivor once `R` is large (hot, 50 %: 6.3 / 11.5 / 24.4 / 48.3 ns per
survivor at k = 1 / 2 / 4 / 8), plus a fixed scan of `R`'s 1,024 words of about
1.5 ns per word. Prediction 4 (flat per survivor within 1.5×) is **falsified**
as stated: at small `|R|` the fixed word scan dominates (k = 2, `|R|` = 1:
1.6 µs per survivor). Per lane load it is flat. The fused Σ terminal
(`Measure::Sum`, one mask-risc program) costs 3.5–41 ns per word, about
1.3 ns per summed value at 50 %, roughly 5× cheaper than the explicit terminal
on dense `R`. Single-address query (prediction 5): hot 8–45 ns, cold 17–184 ns;
cost grows with k and with the number of present lanes (loads), with no search.

**What this answers.** C's own operation runs in four stages whose costs are
separable: the logic stage is microseconds per program and removes a third of
the corpus outright; the address mask costs nanoseconds per 64 addresses; the
mask makes Lance scan exactly the survivors, and saves bytes when the survivors
sit on few pages (a narrow aperture, or clustering at page scale); the payload
terminal costs about 6 ns per lane load, and a fused terminal several times
less. No layout decision follows from these numbers.

**Open.** Page-scale clustering (which address order puts a thought's
survivors on few Lance pages); the payload terminal's fixed word scan (iterating
only non-zero words of `R`, or a survivor list from the mask program); the
relalg absorption miss (2 of 2,000).

## Not measured

Hardware counters (none on this VM). Lance decoded bytes, pages and
allocations (not exposed). More than one tile. Quack's `lower_fused` path
(another front end over the same mask-risc programs; relalg is used here
because it carries the stage-1 reduction).

## Connected sources from #1601–#1617 (2026-10-20)

Docs and data merged in #1601–#1617 that this plan owns. They are reading notes or reference data, not implementation; each now links back here. Board: `.claude/board/entries/2026-10-20-prs-1601-1617-plan-links.md`.

- [`.arxiv/2606.24948_superposition_capacity_breaks_two_hop_retrieval.md`](../../.arxiv/2606.24948_superposition_capacity_breaks_two_hop_retrieval.md) (#1601): arXiv:2606.24948 — Global superposition breaks two-hop retrieval before composition
- [`.arxiv/2502.20233_selective_yannakakis_smash_mask_first_thought_coordinates.md`](../../.arxiv/2502.20233_selective_yannakakis_smash_mask_first_thought_coordinates.md) (#1611): Selective Yannakakis / SMASH: choose when to reduce before joining
