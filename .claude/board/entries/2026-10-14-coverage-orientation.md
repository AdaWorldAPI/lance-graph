# 2026-10-14 — Coverage orientation for the count queries

**Status:** MEASURED. Both orientations kept, by the pre-registered rule.

**Setup (DECISION, operator 2026-10-14):** counting is popcount under an
explicit equivalence key (NodeGuid, ThoughtId, the pair), intersected with
the aperture; EvidenceRoot is not a coverage count (covering is not
independence) and goes through the evidential base and the gate. The
thought x identity coverage of one class tile is a 65,536 x 64 bit matrix
(512 KiB), stored per thought (64 masks, the M_t of the formula) or per row
(one written-by word, what the identity and dense-tile probes store).
Pre-registered in `.claude/plans/ce64-hypothesis-layers-v1.md` before the
probe was written.

**Probe:** `crates/cognitive-shader-driver/examples/coverage_orientation_probe.rs`
(`cargo run --release --example coverage_orientation_probe`). Built with
`rustc -O -C target-cpu=native`; tier printed `v4 (avx512f)`. 6 tests against
a brute-force oracle; 9 disable runs red. Two full runs on a shared 4-core VM.

d = 10 %, full tile, ns per query (row words / thought masks):

| query | 8 thoughts | 64 thoughts |
|---|---|---|
| distinct NodeGuid | 9,536 / **1,644** | **9,808** / 12,831 |
| distinct ThoughtId | 6,233 / **52** | 5,624 / **154** |
| (NodeGuid, ThoughtId) | 19,207 / **2,126** | 17,770 / **15,722** |
| perspectives on one identity | **1.10** / 7.61 | **1.11** / 76.7 |

**Outcomes:**
- Neither orientation is within 2 times the best everywhere: both are kept,
  each query routes to its faster one, the second derived at seal (transpose
  6 to 120 ns per row, under 8 ms per class tile).
- One identity's perspectives: always the row word. Distinct thoughts:
  always the masks. Distinct NodeGuid and pair counts: the masks for few
  thoughts, the row words near all 64 (the masks read |T| x 8 KiB, the row
  words always 512 KiB).
- Nanosecond target met: 1.1 ns per perspective count; at most 0.40 ns per
  aperture row for the aperture counts.

**Corrected 2026-10-15 (review):** with the row arm's early exit taken,
distinct ThoughtId at d = 10 %, full tile, is R 60 / C 54 at 8 thoughts and
**R 86** / C 176 at 64; at d = 50 % the row words win every cell, at d = 1 %
the masks still win. So "distinct thoughts: always the masks" no longer
holds: route it by density. Transposes re-timed with the whole result
observed: 6.3 to 128 ns per row, up to 8.4 ms per class tile.

**OPEN:** correlated (non-independent) coverage is not tested; EvidenceRoot counting
needs the evidence stamps per thought.
