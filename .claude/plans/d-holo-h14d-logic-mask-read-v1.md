# D-HOLO-H14d — logic → address mask → masked read → terminal (C only)

**Status:** PRE-REGISTERED 2026-10-16. Nothing below has been run.
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

- [ ] Pre-registration committed before any timing (this file)
- [ ] Probe crate `crates/holo-h14d-probe` (excluded from the workspace, like the relalg crate; path deps: relalg, mask-risc; registry: lance 13, arrow 58, tokio)
- [ ] Oracles (§ Checks) written first and red against a stub
- [ ] Disable runs (§ Disable runs) red, files restored
- [ ] v4 run, tier printed by the program
- [ ] Results section, board entry with raw CSV, STATUS row, AGENT_LOG
- [ ] Draft PR; no merge

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
   payload reads == `|R| × k` (counted).
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

## Timing protocol

v4 only, tier printed by the program; release, debug 0. Medians of 7 batches;
hot (repeated) and cold (cache flushed between batches) reported apart. ns per
word, per survivor, per query; never whole-call milliseconds. About 40 %
run-to-run noise was measured on this VM in H14c; differences below that are
not read as results.

## Not measured

Hardware counters (none on this VM). Lance decoded bytes, pages and
allocations (not exposed). More than one tile. Quack's `lower_fused` path
(another front end over the same mask-risc programs; relalg is used here
because it carries the stage-1 reduction).
