# D-DRF-TLB — gather cost against distinct pages touched (lab, reported not gated)

**Status:** MEASURED (lab), PARTLY EXPLAINED. **Probe:** `crates/drf-focus-probe/examples/tlb_probe.rs`
(`cargo run --release --example tlb_probe`). Narrows, does not close, the OPEN point in
`2026-10-24-drf-p2-breakeven.md` (warm 65 536-row build 23 µs vs 3.8 µs at 1 024 rows): that
comparison was not at constant cache lines (see Correction), so this probe shows a locality
effect at constant lines, not the full cause of the original gap.
Host: Xeon 2.1 GHz, L2 8 MiB (4 instances), L3 260 MiB, THP `madvise` (so the 32 MiB row
buffer is on 4 KiB pages; no 2 MiB control was run). Page, line and within-page offset counts
come from the real row addresses.

Pre-registered in the file header before the first run: H_TLB (build rises with distinct pages
past the second-level TLB reach) against H_other (flat in pages; cache lines set the cost).
Slots fixed at 4 096. For P >= 512 the distinct cache-line count is constant at 4 096.

## Two generators (added after review)

The first generator, `skewed`, read row `(i % P) * 8 + (i / P) % 8`. That makes the number of
distinct within-page offsets fall with P (8 at 512 pages, 4 at 1 024, 2 at 2 048, 1 at 4 096),
so cache-set coverage changed with P along with the page count. The `balanced` generator
rotates the offset by page (`(visit + page % 8) % 8`), so all 8 offsets are used at every P.

Warm build ns (median of 401; `balanced` over three native runs and one `x86-64-v3` run):

| pages | skewed (1 run) | balanced |
|---|---|---|
| 512 | 12 200 | 8 500-11 100 |
| 1024 | 17 700 | 8 800-10 400 |
| 1536 | 39 800 (outlier) | 9 000-9 700 |
| 2048 | 18 100 | 9 900-13 900 |
| 3072 | 29 700 | 22 400-25 200 |
| 4096 | 27 600 | 26 700-32 100 |

Cold builds (one run per row, `balanced`) rise 63 000 -> 150 000 ns over 512 -> 4 096 pages.

- **H_other rejected.** At constant 4 096 lines and balanced offsets the warm build is about
  9-10 µs up to 2 048 pages and 22-32 µs at 3 072-4 096: 2.5-3x. Pages matter on their own.
- **The 512 -> 1 024 step was a generator artifact.** It is gone under `balanced` (9 µs ->
  9-10 µs). With `skewed` the 1 024-page build was 17.7 µs, almost double.
- **Shape under `balanced`:** flat to 2 048 pages, rise between 2 048 and 3 072. That is
  consistent with second-level TLB reach, but no counter or huge-page control confirms it.
- **Not shown:** that huge pages remove the effect; any hardware-counter evidence (perf is
  not usable here, `perf_event_paranoid` = 2 and no `perf`); randomized visit order (all
  runs visit slots in map order); the cause of the cold-build rise.

## Correction to `2026-10-24-drf-p2-breakeven.md`

That entry says a focus touches "one cache line per distinct ordinal: 248-256 KiB here,
whatever the row count" and compares 1 024 and 65 536 rows as "same cache-line footprint".
Distinct rows drawn by 4 096 random slots from N rows is about N(1 - e^(-4096/N)): about 1 005
at N = 1 024 (about 63 KiB of lines) and about 3 970 at N = 65 536 (about 248 KiB). So the
original 3.8 µs vs 23 µs comparison also changed the line count roughly fourfold, and the
duplicates at 1 024 rows were served from lines already loaded. The gap combines more distinct
lines, fewer repeated addresses and more page dispersion; this probe isolates only the last.

Consequence for the cost model: the focus build is not a function of slot count alone, and
a clustered map is cheaper than a scattered one with the same slots. How much is pages and how
much is lines in the original gap is not separated.

OPEN: the 2 MiB-page control; hardware counters; randomized visit order; the original
1 024 vs 65 536 gap decomposed into lines vs pages.
