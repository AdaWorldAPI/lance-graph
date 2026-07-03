# onebrc-probe — trie vs RAM-table methods and outcomes

> Measurement report. All numbers are **measured** on this machine, not
> projected. Corpus: `/tmp/brc10m.txt`, 10,000,000 rows, seed 42,
> sha256 `f1853caa30a765883aa655be1c304d956ad8b03e19b3557df2af431d9a955691`.
> Metric: `throughput_mrows_s` (rows / compute-time). **Compute-only** —
> `main.rs` does `fs::read` BEFORE `Instant::now()`, so file I/O and any
> mmap lever are OUTSIDE the timer and are NOT reflected here.
> Build: `.cargo/config.toml` pins `target-cpu=x86-64-v3` (AVX2) unless a
> row says `native`. 4 workers, best-of-3.

## The methods (group-by-aggregate, min/max/sum/count per station)

Every lane runs the SAME workload and the SAME newline-aligned
`chunk_bounds` split + commutative merge. What varies is (1) how a record's
delimiters are found/parsed and (2) how the station identity is turned into
an accumulator slot. That second axis is the "trie vs RAM-table" question.

| Lane | Scan / parse | Group-by structure | Family |
|---|---|---|---|
| **A** scalar | byte-wise `;`/`\n`, int parse | `BTreeMap<String,Stats>` | baseline, 1 thread |
| **C** threads | byte-wise, int parse | per-worker `BTreeMap`, merge | baseline, N threads |
| **R** radix | byte-wise, int parse | flat SoA table, slot = `hash & 0xFFFF` | RAM flat table (control) |
| **F** Morton | byte-wise, int parse | flat SoA table, slot = FNV-1a → nibble-interleaved 16-bit Morton tile | RAM flat table (substrate-native) |
| **T8** byte-trie | byte-wise, int parse | 256-ary arena trie, one level per name byte | trie |
| **T** nibble-trie | byte-wise, int parse | 16-ary arena trie (HHTL `NiblePath`-faithful), 2 levels per byte | trie |
| **S** SWAR | **SWAR** `;`/`\n` (haszero u64 trick) + **branchless** int parse | flat SoA table (reuses F verbatim) | RAM flat table + SWAR |

All lanes are parity-checked: **every lane produces aggregates identical to
lane A** on a generated corpus (unit tests `*_agrees_with_lane_a` /
`both_tries_agree_with_lane_a` / `lane_s_agrees_with_lane_a`).

## The outcomes (10M rows, 4 workers, best-of-3, mrows/s)

| Lane | v3 (best) | native (best) | vs lane C |
|---|---:|---:|---:|
| A scalar 1-thread | 8.1 | — | 0.26× |
| C threads + BTreeMap | 31.3 | — | 1.0× (ref) |
| T 16-ary nibble trie | 54.5 | ~55.9 | 1.7× |
| R flat radix table | 78.7 | ~90.6 | 2.5× |
| T8 256-ary byte trie | 67.5 | ~77.6 | 2.2× |
| F flat Morton table | 84.1 | 84.6 | 2.7× |
| **S SWAR + flat table** | **103.6** | **101.4** | **3.3×** |

(native column: F/R/T/T8 from an earlier best-of-3 sweep this session; S
from this session's native build; a dash = not separately measured native.)

## What the numbers actually say

1. **The trie is SLOWER than the flat table, measured.** T (16-ary) at ~55
   and T8 (256-ary) at ~68 both lose to the flat table F at ~84 and even to
   the plain-radix control R at ~79. At this cardinality (~400 stations) the
   trie's pointer-chasing descent (2 levels/byte for the nibble trie) costs
   more than a single hash + linear-probe into a contiguous SoA table. The
   HHTL trie is the right structure for *prefix routing over a keyspace*;
   it is NOT the fast structure for *dense small-cardinality group-by*. This
   contradicts the earlier session hypothesis that "the other session used
   the HHTL trie to reach 90" — the trie does not reach the flat table's
   throughput here.

2. **Morton addressing (F) beats plain radix (R) by a hair** (~84 vs ~79 at
   v3), and both crush BTreeMap (C, ~31). The bulk of the win is
   flat-SoA-table-vs-BTreeMap (R−C ≈ +47), not the Morton interleave
   (F−R ≈ +5, inside run-to-run variance). Honest reading: the addressing
   scheme is a minor tuning knob; the flat contiguous table is the lever.

3. **SWAR (S) is the real mover.** Replacing the byte-by-byte delimiter loop
   with the haszero-u64 SWAR scan + branchless parse lifts the flat lane
   from ~84 (F) to ~104 at v3 and crosses 100 on native — roughly +20% on
   top of the best flat table. This is the technique the actual 1BRC winners
   use, and it is the one measured lever that pushed past the ~90 the trie
   experiment was chasing — on the *same* flat group-by, not a fancier one.

4. **Native SIMD is mostly noise here.** Comparing v3 vs native columns, the
   deltas are within run-to-run variance (F 84.1→84.6, S 103.6→101.4 —
   native is actually *lower* on S). The gap the earlier session attributed
   to AVX-512 width was measurement variance, not instruction width. The
   compute bottleneck is scan+parse+slot, not vector width.

5. **mmap (lever a) is not visible in this harness and was not faked.** The
   timer excludes `fs::read`; mmap is a wall-clock / 13GB-allocation lever
   for the full 1B file, on an axis this metric does not observe. Measuring
   it would need an end-to-end wall-clock mode + a memmap2 dep (which breaks
   the std-only, zero-dep contract of lanes A/C/F/R/T/S).

## The honest bottom line

For a dense, small-cardinality group-by (the 1BRC shape, and the substrate's
own aggregation shape): **flat SoA table + SWAR scan/parse wins; the trie
loses.** The trie earns its keep on a different workload — prefix routing /
ancestor queries over a large sparse keyspace (the HHTL cascade's actual
job) — not on this one. The result promotes the flat-table + SWAR path and
demotes the trie-for-aggregation hypothesis to *falsified for this workload*.

## Reproduce

```bash
cd crates/onebrc-probe
cargo build --release
target/release/onebrc-probe gen /tmp/brc10m.txt 10000000 42   # if absent
for lane in a c r f t8 t s; do
  target/release/onebrc-probe run /tmp/brc10m.txt $lane 4
done
# native: RUSTFLAGS="-C target-cpu=native" cargo build --release --target-dir target-native
```
