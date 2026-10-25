# D-DRF-TLB — gather cost against distinct pages touched (lab, reported not gated)

**Status:** MEASURED (lab). **Probe:** `crates/drf-focus-probe/examples/tlb_probe.rs`
(`cargo run --release --example tlb_probe`). Closes the OPEN point in
`2026-10-24-drf-p2-breakeven.md` (warm 65 536-row build 23 µs vs 3.8 µs at 1 024 rows).
Host: Xeon 2.1 GHz, L2 8 MiB (4 instances), L3 260 MiB, THP `madvise` (so the 32 MiB row
buffer is on 4 KiB pages). Page and line counts come from the real row addresses.

Pre-registered in the file header before the first run: H_TLB (build rises with distinct pages
past the second-level TLB reach) against H_other (flat in pages; cache lines set the cost).
Slots fixed at 4 096. For P >= 512 the distinct cache-line count is constant at 4 096, so that
range isolates the page effect. Below 512 the line count also falls and does not decide.

| pages | lines | build warm (ns) | build cold (ns) |
|---|---|---|---|
| 512 | 4096 | 8 800-10 200 | 60 200 |
| 1024 | 4096 | 17 200-18 000 | 65 700 |
| 2048 | 4096 | 18 300-18 900 | 89 500 |
| 3072 | 4096 | 21 900-25 200 | 108 400 |
| 4096 | 4096 | 26 000-29 500 | 110 300 |

Warm ranges are over three runs (two native, one `x86-64-v3`); the cold column is one run.

- **H_other rejected.** At constant 4 096 lines the warm build rises about 3x from 512 to 4 096
  pages, and the cold build about 1.8x. Pages matter on their own, warm and cold.
- **Shape is not a single knee.** A step at 512 -> 1 024 pages, a plateau to 2 048, then a rise
  to 4 096. The rise past 2 048 matches the second-level TLB reach and supports H_TLB. The step
  at 512 -> 1 024 is not explained by it and is left unexplained here.
- **Not shown:** that huge pages remove the effect. Doing so needs the same gather on 2 MiB
  pages (`madvise`), which this probe does not do. Until then "TLB" is the best-fitting
  reading, not a measured cause.

Consequence for the cost model: the focus build is not a function of slot count alone. A map
whose slots land on few pages (clustered) builds about 2-3x cheaper than a scattered one with
the same slots, which is why `clustered` was cheaper than `random` in the P2 table.

OPEN: the 512 -> 1 024 step; the 2 MiB-page control.
