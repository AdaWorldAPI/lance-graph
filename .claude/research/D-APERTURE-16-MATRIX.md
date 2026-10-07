# D-APERTURE-16 — technique matrix

Companion to `D-APERTURE-16-0.md`, measured 2026-10-07 by `aperture16_probe`. "Extra bytes" means bytes beyond the 8 KiB support mask that the technique allocates or writes. "Exact support" asks whether the technique answers which selves are live; "multiplicity" asks whether it carries counts. "Geomean" is relative to `u64` bit visit over the 464-case payload ladder.

| Technique | Extra bytes | Exact support | Multiplicity | Measured | Verdict |
|---|---|---|---|---|---|
| u64 bit visit (skip zero word, walk set bits) | 0 | yes | via the fold it drives | baseline; production shape (`group_walk`) | **KEEP** — the schedule |
| u64 bit visit + full-word dense path | 0 | yes | via the fold | geomean 0.80; ~3× on runs; worst layout +6 % | **KEEP / BUILD** into the fold kernels |
| u16 aperture bit visit | 0 (shift view) | yes | via the fold | geomean 2.79; empty-mask floor 7× slower | **KILL** as a scheduling unit |
| adaptive u16 aperture (per cell: skip / dense / set-bit) | 0 | yes | via the fold | geomean 2.07; 2–5× slower on scattered data | **KILL** |
| u64 → full u16 quarter dense (`adapt64>16`) | 0 | yes | via the fold | geomean 0.73; 0.41 on 16-islands, 1.2–1.8× slower on scattered data | **OPT-IN only**, behind a measured workload |
| branch-free full-quarter detect (`adapt64q`) | 0 | yes | via the fold | geomean 0.85; does not remove the scattered-data cost | **KILL** |
| popcount-threshold mode switch | 0 | yes | — | crossover is payload-dependent (dense wins only at k ≥ 13 of 16, never for 32-byte records) | **KILL** |
| ordinal list (`Vec<u16>`) | 2 · live | yes | via the fold | geomean 1.31; no reproducible win | **KILL** (no earned exception) |
| bounded extent (`partition_point → Cmp::Range`) composed with u64 words | 0 | yes | — | multiplies with the mask: 3.0 µs vs 29.1 (extent only) and 15.7 (mask only) | **KEEP**; needs an ordering witness for scalar lanes |
| coarse target bitmap (`target >> 4`, written during the exact pass) | 512 | no (cells only) | no | up to ~17× over clearing and scanning all of `K_next`; beats the exact mask only for few-cell concentration | **PARK**; superseded by the exact target mask except for concentrated targets |
| exact target mask written during the exact pass | 0 (it is the next frontier) | yes | no; it schedules clearing `K_next` | wins or ties every case except few-cell concentration; up to ~19× | **KEEP / BUILD** for multi-hop K propagation |
| u16 target histogram (`[u16; 4096]` pre-pass) | 8 192 | no | coarse counts, saturating | slower than the cell bitmap in 29 of 30 cases; the exception (uniform, 100 %) lost to the exact target mask | **KILL** (`MultiplicityAperture`) |
| full scan, branch-free | 0 | yes | yes | only for 100 %-density support into concentrated targets (15–20 %) | **KEEP** as the dense-end lowering |
| per-row branch on the mask bit | 0 | yes | yes | geomean 14.6; up to 181× slower | **KILL** |

## The rule that survived

```text
extent  → [lo, hi)                (ordered lane; binary search)
words   → skip m[w] == 0          (one test per 64 selves)
dense   → m[w] == u64::MAX        (contiguous fold)
bits    → walk set bits           (trailing_zeros)
bytes   → touch payload last      (cost tracks live rows, or live lines from 32 B up)
target  → write the next frontier mask in the same pass; it schedules the next round
```

No new carrier and no new V4 opcode. The u16 cell is a free shift of the same bytes, useful only for detecting full 16-runs.
