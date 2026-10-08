# 2026-10-08 — Mexican hat: per-query D/F choice and early exit on the D path (D-MHB-2)

**Status:** MEASURED. Probe only. Closes the two OPEN items of D-MHB-1.
- **Probe:** `crates/lance-graph-mask-risc/examples/mexhat_bucket_probe.rs`, new arms Q, HD, HQ and a cut sweep.
- **Revision:** lance-graph `origin/main` 7f63ae9d.
- **Host:** Intel Xeon @ 2.8 GHz, 4 vCPU, `avx2` build, release, `debug = 0`, rustc 1.98.1. Medians of 5 runs; the VM is noisy, about ±15 % between sections.

## Arms

- **Q:** fetch the window's 37 words once, count present disk cells exactly (37 popcounts), then run D if the count is ≤ the cut and F otherwise. Cut pinned at 100 cells (D-MHB-1 finding 3).
- **HD / HQ:** arm H's exact suffix-bound early exit with D's per-point row evaluator (HD), or with a per-query D/F choice (HQ). The bounds are the template's, so they hold for any exact row evaluator.

## Correctness

- Q equals D on every checked centre for cuts 0, 100 and `u32::MAX`, on both grids and every density.
- HD and HQ return the same decision as H, after the same number of rows, on every query.
- Disable run: a D row that drops one window cell makes `Q (cut 100) differs from D` fire on the first grid.

## Results

### Q against min(D, F): 1M grid, 2,048 queries, ns/query

| ρ | cut 25 | cut 50 | cut 100 | cut 200 | cut 400 | D | F |
|---|---|---|---|---|---|---|---|
| 0.01 | 1.19 | 1.16 | 1.16 | 1.22 | 1.21 | 234 | 593 |
| 0.05 | 1.30 | 1.10 | 0.93 | 0.93 | 0.93 | 530 | 579 |
| 0.1 | 1.18 | 1.18 | 1.10 | 1.04 | 1.03 | 602 | 578 |
| 0.2 | 1.40 | 1.42 | 1.33 | 0.97 | 1.03 | 727 | 850 |
| 0.5 | 1.13 | 1.14 | 1.09 | 1.10 | 1.11 | 1,057 | 614 |
| 0.9 | 1.16 | 1.16 | 1.16 | 1.15 | 1.18 | 1,483 | 584 |

Ratios are Q / min(D, F). Below 1 is noise or the shared word fetch.

### Early exit at the median threshold, 1M grid, ns/query

| fixture | H (F rows) | HD (D rows) | HQ | full F | full D | full Q |
|---|---|---|---|---|---|---|
| ρ 0.01 | 515 | 267 | 364 | 596 | 242 | 248 |
| ρ 0.1 | 520 | 559 | 606 | 602 | 642 | 618 |
| ρ 0.5 | 548 | 977 | 1,454 | 601 | 1,773 | 1,497 |
| ρ 0.9 | 481 | 1,258 | 608 | 597 | 1,479 | 687 |
| wavefront + 1 % noise | 487 | 272 | 357 | 589 | 262 | 247 |

## Findings

1. **A per-query choice reaches min(D, F) to within 0.93–1.22× at cut 200, and 0.93–1.33× at cut 100.**
   - It removes the worst case of either fixed path: D is 2.5× slower than F at ρ 0.9; F is 2.5× slower than D at ρ 0.01.
   - Its whole overhead is the exact 37-popcount count, about 40–90 ns.
   - On this host cut 200 is slightly better than the pinned 100 (the ρ 0.2 row). The cut stays a host pin, not a constant.
2. **Early exit does not pay on the D path.**
   - At sparse density a D row costs as many lookups as it has present cells, which is already few. The early exit's bound bookkeeping costs more than the rows it skips: HD 267 against full D 242.
   - HQ pays the full window count up front, which the early exit cannot then save. It is worse than H or HD everywhere.
3. **Early exit stays a modest win on F only:** −10 to −20 % against full F, as in D-MHB-1.

## Recommendations

| item | verdict | why |
|---|---|---|
| Per-query D/F choice from the exact window count (Q) | **ADOPT** together with D and F (R-MHB-1's production PR) | within 1.0–1.3× of the better path at every density; removes a 2.5× worst case |
| Early exit over D rows (HD) | **REJECT** | no gain over full D where D is chosen |
| Early exit after a per-query choice (HQ) | **REJECT** | the count it needs cannot be skipped |
| Early exit over F rows (H) | stays **PROBE** | −10 to −20 % |

## OPEN

- One host. The cut must be re-measured per host class.
- A density estimate from the grid's global popcount would cost nothing per query, but it fails on local structure (the wavefront fixture). Not measured.
