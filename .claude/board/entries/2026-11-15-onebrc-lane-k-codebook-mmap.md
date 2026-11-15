# 2026-11-15 — onebrc-probe lane K: codebook indices over mmap

**Kind:** lab lane in `crates/onebrc-probe` (workspace-excluded). No production
code changed.
**Grade:** MEASURED; predictions pre-registered in `FINDINGS.md` §6 before the
first run, results in §7. Disable controls: 3 of 4 red (sum offset correction,
alphabetical sort, table fallback); the full-name check on a 64-bit hash
collision is untested and recorded as such.

## Result (4 workers, 100M rows, warm page cache, median of 5)

| | lane S (`fs::read`) | lane K, parse | lane K, temp codebook |
|---|---|---|---|
| compute only, Mrows/s | 134.2 | 143.5 | 117.8 |
| open to result, ns/row | 14.64 | 7.02 | 8.02 |

- The large effect is mmap: not copying the input halves the end-to-end
  time. `fs::read` costs about as long as the whole computation.
- Codebook grouping (dense ids, names as input slices, order-preserving
  temperature codes, one sort of distinct names) beats lane S's hash table
  by 6.9 % compute-only.
- A temperature codebook lookup is slower than the parse (−17.9 %): the
  pre-registered K2 fails.

OPEN: cold page cache, the 10,000-station key set, more than 4 workers.

## Update: merykitty parse and word-wise name hash (FINDINGS §8-9)

| 100M, 4 workers | `run` Mrows/s | `wall` ns/row |
|---|---|---|
| k | 136.5 | 7.56 |
| km (merykitty parse) | 188.2 | 5.42 |
| kw (word hash) | 150.5 | 5.40 |
| kmw (both) | 245.0 | 4.00 |

The two compound. Pre-registered K4 holds, K5 holds weakly (overlapping
ranges), K6 holds on 100M and fails on the noisy 10M `wall`. Disable
controls 3 of 3 red (tail mask, sign, next-line offset).

## Update: per-station histogram (FINDINGS §11-12)

1,999 counters per station, one increment per row: 16 fewer instructions
per row (`kmwh` 178.48 vs `kmw` 194.48), no separable time difference at
400 stations (`run` 251.4 vs 248.3 Mrows/s, overlapping ranges).
Pre-registered H1 and H2 hold. Disable controls 3 of 3 red.

## Update: median and percentiles (FINDINGS §13)

The per-station histogram gives exact nearest-rank percentiles and the
median, merged across workers by adding counts; tested against a sorted
reference, 3 of 3 disable controls red. About 3.7-4.9 ns/row open to result
on 100M rows with percentiles for all stations.
