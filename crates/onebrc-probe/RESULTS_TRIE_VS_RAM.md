# onebrc-probe — trie vs RAM-table methods and outcomes

> Measurement report. All numbers are **measured** on this machine, not
> projected. Corpus: `/tmp/brc10m.txt`, 10,000,000 rows, seed 42,
> sha256 `f1853caa30a765883aa655be1c304d956ad8b03e19b3557df2af431d9a955691`.
> Metric: `throughput_mrows_s` (rows / compute-time). **Compute-only** —
> `main.rs` reads the file (`fs::read`, line 91) BEFORE `Instant::now()`
> (line 94), so file I/O and any mmap lever are OUTSIDE the timer.
> Build: `.cargo/config.toml` pins `target-cpu=x86-64-v3` (AVX2) unless a
> row says `native`. 4 workers.
>
> **Statistical note (corrected after adversarial review).** An earlier
> draft of this report headlined single **best-of-3** numbers. Three review
> agents (truth-architect, overclaim-auditor, brutally-honest-tester)
> correctly flagged that best-of-N reports the luckiest run and hides the
> spread — at run-to-run variance of ~8–13%, that is a real reporting sin.
> This version reports **median / min / max / sd over n=11** (v3) and n=7
> (native) per lane. Cardinality is ~400 stations (`gen.rs STATION_COUNT`);
> results are for THIS workload, THIS machine, ONE corpus — see "Scope" at
> the end.
>
> **Lane S provenance.** Lane S (SWAR) shipped separately (PR #637, merged to
> `main`); this branch is rebased on top of it, so `lane_s` and its
> `lane_s_agrees_with_lane_a` parity test are present here and all lanes in the
> ladder — `a c r f t8 t s` — are runnable. S is kept in the ladder because the
> report's whole point is the full RAM-table-vs-trie comparison, and S is the
> fastest RAM-table method.

## The methods (group-by-aggregate, min/max/sum/count per station)

Every lane runs the SAME workload and the SAME newline-aligned
`chunk_bounds` split + commutative merge. What varies is (1) how a record's
delimiters are found/parsed and (2) how the station identity becomes an
accumulator slot — the "trie vs RAM-table" axis.

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
lane A** on a generated corpus (unit tests `lane_a_and_lane_c_agree…`,
`lane_f_and_lane_r_agree_with_lane_a…`, `both_tries_agree_with_lane_a…`,
`lane_s_agrees_with_lane_a`, plus a forced-collision probe on the shared
table). Verified in-code, not asserted.

## The outcomes (10M rows, 4 workers, mrows/s)

**v3 (x86-64-v3), n=11 per lane:**

| Lane | median | min | max | sd | vs C (median) |
|---|---:|---:|---:|---:|---:|
| C threads + BTreeMap | 31.2 | 29.4 | 32.0 | 0.7 | 1.0× (ref) |
| T 16-ary nibble trie | 54.2 | 50.4 | 54.7 | 1.4 | 1.7× |
| T8 256-ary byte trie | 58.3 | 54.5 | 66.5 | 3.8 | 1.9× |
| F flat Morton table | 84.6 | 75.1 | 86.3 | 3.6 | 2.7× |
| R flat radix table | 87.7 | 61.2 | 89.1 | 8.6 | 2.8× |
| **S SWAR + flat table** | **103.9** | 76.6 | 105.5 | 7.9 | **3.3×** |

**native (target-cpu=native), n=7, controlled same-session (F, S only):**

| Lane | median | min | max | sd |
|---|---:|---:|---:|---:|
| F flat Morton table | 74.0 | 65.8 | 84.0 | 6.3 |
| **S SWAR + flat table** | **96.9** | 90.6 | 106.2 | 5.3 |

## What the numbers actually say (each claim scoped to its evidence)

1. **SWAR (S) is the one real, robust win. [supported]** At the median, S
   beats F by **+23% on v3** (103.9 vs 84.6) and **+31% on native** (96.9 vs
   74.0). The gap (≈19 mrows/s) is ~2.4× S's own sd and clears F's max
   (86.3) at the median. Caveat kept honest: S is the noisier lane — its
   *worst* run (76.6) dips below F's median, so the guarantee is "typically
   +~25%, occasionally ties F," not a hard floor. The earlier best-of-3
   draft happened to draw an unlucky S run (77.4) that made the number look
   cherry-picked; the n=11 median vindicates the SWAR win but only with the
   spread disclosed.

2. **The trie is slower than the flat table here — but this is the arena-trie
   IMPLEMENTATION, not "the trie idea," and the distinction matters.
   [supported, confounded]** T (54.2) and T8 (58.3) medians both sit far
   below F (84.6) and R (87.7); the gap is large and robust across n=11. But
   two confounds are uncontrolled and inflate the trie's cost: (a) `Trie`
   carries `fanout` as a **runtime struct field** (`descend` computes
   `node*self.fanout+sym`), losing the strength-reduction/monomorphization
   the flat table gets from its `const SLOTS`; (b) `descend` does an
   **in-loop arena realloc** (`children.extend(...)` per new node, 256×u32 =
   1 KB/node for T8) *inside the timed scan*, while the flat table allocates
   once up front. So the honest claim is: **this arena-trie is not
   competitive with the flat table on dense small-cardinality group-by** —
   NOT "the trie is falsified." The direction (a trie chases ~10–20
   dependent loads/record vs the table's ~1 hash + 1 near-L1 slot) is
   plausible, but a const-fanout, pre-sized-arena trie was not built, so the
   idea itself is untested. This does contradict the earlier-session
   hypothesis that the HHTL trie is what reached ~90 — no trie variant here
   reaches the flat table's throughput.

3. **Morton (F) vs plain radix (R): no measurable difference. [supported —
   corrected from the prior draft]** R actually medians *slightly above* F
   (87.7 vs 84.6), and that 3.1-mrows/s difference is well inside R's sd
   (8.6). The nibble-interleave is a **no-op on throughput** (possibly a
   marginal negative). The prior draft's "F beats R by a hair" was wrong and
   is retracted. The big structural win is flat-SoA-table-vs-BTreeMap
   (R−C ≈ +56 median), not the addressing scheme.

4. **`target-cpu=native` gives no benefit for these lanes — and this table
   contains NO SIMD lane, so it says nothing about AVX-512. [narrow claim
   supported; the broad one retracted]** Controlled same-session, native F
   (74.0) and S (96.9) medians are *below* their v3 counterparts (84.6,
   103.9) — native did not help and if anything ran slightly slower (likely
   codegen/thermal, within the noise band). The defensible statement is
   "the compiler's `native` flag does not speed up these SCALAR/SWAR lanes."
   The prior draft's "native SIMD is noise" overreached: lane S is *SWAR*
   (scalar u64 tricks), not vector SIMD, and the actual SIMD lane (B) is
   feature-gated and absent from this table. This probe cannot adjudicate
   any AVX-512 claim — it runs no AVX-512.

5. **mmap (lever a) is not measurable in this harness and was not faked.
   [verified in code]** The timer starts after `fs::read` (main.rs:91 → :94);
   mmap is a wall-clock / 13 GB-allocation lever for the full 1B file, on an
   axis this metric does not observe. Measuring it needs an end-to-end
   wall-clock mode + a memmap2 dep, which breaks the std-only, zero-dep
   contract of lanes A/C/F/R/T/S.

## The honest bottom line

For a dense, ~400-cardinality group-by at 10M rows on this machine: **flat
SoA table + SWAR scan/parse is the fastest method measured (~104 mrows/s
median, +~25% over the plain-scalar flat table); the arena-trie lanes are
the slowest of the non-baseline group.** The Morton interleave buys nothing
over plain radix. Native codegen buys nothing over v3.

**What this does NOT establish (explicit conjecture, unmeasured here):**
- That a trie is the wrong structure *in general* — only that THIS arena
  trie loses on THIS workload; a const-fanout/pre-sized variant is untested.
- That the trie "wins at prefix routing" — no prefix-routing / ancestor-query
  benchmark exists in this crate. That is the HHTL cascade's claimed job,
  but it is a CONJECTURE here, not a result.
- That ~400-cardinality dense group-by is "the substrate's own aggregation
  shape" — unmeasured; `lane_f.rs` itself flags high-cardinality as a
  different regime.

## Scope / how to reproduce

Single machine, single corpus, one cardinality (~400 stations), one row
count (10M). Not a claim about other CPUs, other cardinalities, or the full
1B-row file.

```bash
cd crates/onebrc-probe
cargo build --release
target/release/onebrc-probe gen /tmp/brc10m.txt 10000000 42   # if absent
for lane in a c r f t8 t s; do
  for i in $(seq 1 11); do target/release/onebrc-probe run /tmp/brc10m.txt $lane 4; done
done   # take median + min/max/sd per lane, not best-of-N
# native: RUSTFLAGS="-C target-cpu=native" cargo build --release --target-dir target-native
```
