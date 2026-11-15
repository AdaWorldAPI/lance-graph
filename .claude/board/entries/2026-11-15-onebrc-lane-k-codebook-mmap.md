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

## Update: 10,000 stations (FINDINGS §14-15)

100M rows, 4 workers: `kmw` 149.1 Mrows/s and 6.15 ns/row (lane S 79.4,
20.89); the histogram `kmwh` drops to 59.9 Mrows/s and 13.77 ns/row while
executing the fewest instructions (185.29 vs 201.30). At 80 MB of counters
per worker the cost is memory, not instructions. All four pre-registered
predictions hold.

## Update: Codec9 counter window (FINDINGS §16-17)

512 counters per station indexed by Codec9 word (±25.5), exact overflow
row beyond. At 10,000 stations it is not faster than the full histogram
(64.0 vs 69.8 Mrows/s, overlapping ranges; Y1 fails): about 38 MB per
worker is still out of cache, overflow stations touch two tables, and the
encoder adds about 21-24 instructions per row. Disable controls 3 of 3 red.
