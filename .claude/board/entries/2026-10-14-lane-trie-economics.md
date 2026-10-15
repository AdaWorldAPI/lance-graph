# 2026-10-14 — Lane-tables on the NodeGuid trie: per-lane tries vs one trie per thought

**Status:** MEASURED. Arm chosen by the pre-registered rule; nanosecond target
not met; next arm OPEN.

**Setup (DECISION, operator 2026-10-14):** a thought owns up to 32
lane-tables (0 identity, 1 edge, 2..31 value), up to 64 thoughts; addressing
is the NodeGuid trie; a read is a merge on read over prefix x thought mask x
lane mask, keyed by the NodeGuid (no join, no fold). Pre-registered in
`.claude/plans/ce64-hypothesis-layers-v1.md` § "Lane-tables addressed by the
NodeGuid trie" before the probe was written.

**Probe:** `crates/cognitive-shader-driver/examples/lane_trie_probe.rs`.
Level-ordered trie, fan-out 16, 16-bit child masks, popcount child index,
16 levels (classid 8 + HEEL 4 + HIP 4). Built with `rustc -O -C
target-cpu=native`; tier printed `v4 (avx512f)`. One run; it reached the
30-minute limit before its last two rows, after the reference cell. 5 tests
against a brute-force oracle; 5 disable runs red.

| reference cell T = 64, d = 10 %, w = 4, all lanes | P (per-lane tries) | S (one trie per thought) |
|---|---|---|
| bytes | 79.3 MB | 41.5 MB |
| point read | 569 µs | 6.36 µs |
| per trie-level step | 26.4 ns | 8.1 ns |
| range, per emitted NodeGuid | 11.1 µs | 1.08 µs |

**Rule outcome:** S. **Nanosecond target:** failed for both arms (pre-registered
bound 1 µs per point read at the reference cell).

**Why:** point cost = tries descended x levels. At T = 64 that is up to
1024 steps at 5 to 10 ns; the 8 classid levels are identical in every trie
and carry no information. P with one lane selected matches S on points and
beats it on ranges.

**OPEN (needs a new pre-registration):** one trie over all thoughts with a
64-bit thought mask per node (one descent instead of 64); classid taken out
of the trie (one trie per class).
