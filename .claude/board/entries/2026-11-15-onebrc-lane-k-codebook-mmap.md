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
