# FINDINGS — 1BRC substrate probe (agnostic record)

> This file records WHAT was measured: environment, methods, numbers,
> and the asserted invariants with their code. It deliberately contains
> NO interpretation, rulings, or architecture doctrine — that lives in
> `COMMENTARY.md` so a later session can read these facts from its own
> angle without inheriting this session's prime.

## 1. Environment

- Container: 4 cores (`nproc`=4), Linux, AVX-512-capable CPU
  (`avx2`+`avx512f`+`avx512bw` in `/proc/cpuinfo`).
- Probe pinned to `target-cpu=x86-64-v3` (`.cargo/config.toml`) —
  `U8x32` ops compile to real AVX2 intrinsics; matches lance-graph CI.
- Release builds. Rust stable.
- Corpus recipe (reproducible; see §2 of `README.md`):
  `rows=10000000 seed=42
  sha256=f1853caa30a765883aa655be1c304d956ad8b03e19b3557df2af431d9a955691`
  — 142 MB, 400 distinct group keys ("stations"), hash re-verified
  byte-identical at every regeneration.
- Workload: group-by aggregate (min/max/sum/count per key, integer
  tenths; output rendered to `BTreeMap<String, Stats>` for
  byte-for-byte cross-method equality).
- Caveats that bound every number: 4-core container (oversubscription
  effects are real), 400-key cardinality (low), 10M rows (smoke scale;
  the 100M container-scale target was not run), `std::fs::read` (no
  mmap), scalar parse (no SWAR).

## 2. Methods measured

| Lane | Method (mechanism only) |
|---|---|
| A | single thread, byte scan, `BTreeMap` accumulate |
| B | same as A but delimiter scan via `ndarray::simd::U8x32::cmpeq_mask`, 32-byte non-overlapping stride (`ndarray::simd::array_chunks`) |
| C | N scoped threads, newline-aligned chunks, per-worker `BTreeMap`, commutative merge at end |
| D | as C, worker = one ractor actor per chunk (ask-pattern), corpus copied once into `Arc<Vec<u8>>` |
| E | as C/D, plus one fresh kanban-lifecycle actor per batch driven Planning→CognitiveWork→Evaluation→Commit around the work |
| F | as C, accumulator = flat 65536-cell SoA table, cell = Morton nibble-interleave of two hash bytes, open-addressed, name-verified |
| R | as F, cell = plain low 16 hash bits (no interleave) — control |
| G | workers pre-reduce 64K-row morsels into private F-style tables, stream dirty entries to `shards` owner actors (each owning its OWN table over a contiguous cell range); every applied batch appends one witness record |
| H | as G, plus a router tier: LAZY owner spawn on first traffic + per-owner buffering with ahead-firing batched delivery (`batch_k=64`) |
| I | batch pipeline: 65536 standing per-cell mailboxes spawned upfront (message-free in steady state), identity minted once into a codebook cell (direct indexed writes after), workers fill whole 65536-cell batch tables, freeze into `Arc`, cast ONCE WHOLE to an ownership sink AND a persistence (row-address/version) sink, table pool recycled by refcount ("flush cache") |
| J | I parameterized: `grid` (4096 or 65536 cells), `sink_lanes` (1/8/64 row-range lane pairs), `registry` (on/off) |

8 frozen configurations of the above are runnable as presets
(`src/presets.rs`, CLI `run <path> p<N>` not wired — call
`presets::run_preset`); each is asserted byte-identical to lane A by
`all_presets_agree_with_lane_a`.

## 3. Measurements

All throughputs in Mrows/s on the §1 recipe corpus. "Median" over the
listed passes. Same-session reference values are quoted where a table
mixes sessions (run-to-run drift between bench rounds was up to ~10%
on this container; comparisons inside one table row-block are
same-session).

### t0/t1/t2 — baselines, SIMD scan, actors, kanban cards (4 workers)

| Method | passes | median |
|---|---|---|
| A | 7.11 / 7.01 / 7.16 (three rounds) | ~7.1 |
| B | 7.06, 7.46 | 7.3 |
| C | 26.4 / 27.6 / 28.3 (three rounds) | ~27.5 |
| D | 21.6, 22.4 | 22.1 |
| E batches=4 | 22.8, 23.0 | 22.9 |
| E batches=64 | 22.0, 22.5 | 22.2 |
| E batches=256 | 21.8, 22.1 | 22.0 |

Per-card overhead from E(256)−E(4) elapsed delta: ≈ 66 µs per card
(actor spawn + 3 lifecycle RPCs + join + queue pull).

### t3 — flat SoA accumulators (4 workers, 5 passes)

| Method | passes | median |
|---|---|---|
| C | 28.3, 27.8, 28.9, 28.1, 28.8 | 28.3 |
| F (Morton cell) | 76.2, 80.8, 56.9, 77.4, 81.2 | 77.4 |
| R (radix cell)  | 86.2, 86.4, 86.3, 86.8, 85.1 | 86.3 |
| F single-thread | 21.5 | — |
| R single-thread | 23.3 | — |

### t4/t4a — streamed ownership, granularity sweep (4 workers, 3 passes)

| Owners (each owns its own table) | median |
|---|---|
| F reference (no ownership)  | ~79.5 (t4) / ~76 (t4a round) |
| 1     | 43.0 (t4) / 43.4 (t4a) |
| 4     | 39.9 |
| 16    | 30.3 (noisy 29.2–41.9) |
| 256   | 35.9 |
| 4096  | 18.3 |
| 65536 | 2.1 (one pass 1.45; same-session later: 1.7) |

At 65536 owners: ~63K owner-addressed messages; 64K actor spawns
inside the run.

### t5 — orchestrated (lazy + ahead-firing), 4 workers, 3 passes

| Nominal owners | flat (t4a) | orchestrated | same-session refs |
|---|---|---|---|
| 16    | 30.3 | 42.2 | |
| 256   | 35.9 | 36.8 | |
| 4096  | 18.3 | 40.2 | |
| 65536 | 2.1 (1.7 same-session) | 39.4 | G(1)=43.2, F=81.7 |

### t6 — batch pipeline, 64K grid + standing registry (4 workers, 3 passes)

| Pass | total | spawn_ms | steady (= rows/(elapsed−spawn−stop)) |
|---|---|---|---|
| 1 | 3.19 | 2667.9 | ~22 |
| 2 | 6.05 | 1135.2 | ~20 |
| 3 | 3.86 | 2107.4 | ~21 |

Fixed per-run: batches=156; ownership journal = persistence journal =
156; 156 version ticks; rows_addressed=62,400; flush-cache peak 2–3
tables/worker; codebook_len=400; total messages=312.

### t7 — knob matrix (4 workers; same-session refs G(1)=46.3, H=40.5, F=70.1)

| Config | passes | median |
|---|---|---|
| grid=4096, lanes=1, registry=off | 42.0, 46.2, 46.3 | 46.2 |
| grid=65536, lanes=1, registry=off | 19.9(cold), 37.9 / 39.8, 41.5 | ~40 |
| grid=4096, lanes=8, registry=off | 43.8, 45.1, 44.1 | 44.1 |
| grid=4096, lanes=64, registry=off | 37.6, 39.2, 42.2 | 39.2 |
| grid=65536, lanes=1, registry=ON | 2.6 (spawn 2655 ms), 14.1 (spawn 274 ms) | — |

Registry=ON steady state net of spawn (pass 2): ≈ 23 vs ≈ 38–46
without — the registry-residency delta is isolated by this knob (the
pipeline is byte-identical either way).

## 4. Invariants, with their code

Each invariant is enforced in-code (assert/test), not by convention.
Line numbers drift; anchors are function/test names.

**INV-1 — Commutative, associative merge (multi-writer combination).**
`src/lib.rs`, `Stats::merge` + test `merge_is_commutative_and_associative`:

```rust
pub fn merge(&mut self, other: &Stats) {
    if other.min < self.min { self.min = other.min; }
    if other.max > self.max { self.max = other.max; }
    self.sum += other.sum;
    self.count += other.count;
}
```

**INV-2 — Cross-method parity.** Every method's output map is asserted
byte-identical to lane A on a generated corpus. One test per lane
(`lane_*_agrees_with_lane_a*` in each module) plus the preset-wide
harness `presets::tests::all_presets_agree_with_lane_a`:

```rust
for preset in PRESETS.iter() {
    let out = run_preset(preset.id, &data, 3);
    assert_eq!(a, out, "preset {} ({}) must match lane A", preset.id, preset.name);
}
```

**INV-3 — Corpus reproducibility.** `src/gen.rs` streams a SHA-256
while writing; test `generator_is_deterministic` asserts same seed ⇒
same digest. Every measurement above carries the §1 recipe line.

**INV-4 — Witnessed streamed writes (single-sink family).**
`src/lane_g.rs` (`lane_g_kanban_soa_with_morsel`) and `src/lane_h.rs`
(`lane_h_orchestrated_with`):

```rust
assert_eq!(journal_total, casts.load(Ordering::Relaxed),
    "every applied morsel batch must be witnessed (journal == casts)");
// lane H:
assert_eq!(journal_total, router_casts_total,
    "every fired batch must be witnessed (owner journals == router casts)");
```

**INV-5 — Double-cast completeness (two-sink family).**
`src/lane_i.rs` (`lane_i_batch_pipeline_with`); the laned form in
`src/lane_j.rs` multiplies by `sink_lanes`:

```rust
assert_eq!(ownership_journal, batches_total, "…ownership end");
assert_eq!(lance_journal, batches_total, "…lance end");
assert_eq!(versions as usize, batches_total, "one DatasetVersion tick per batch");
// lane J:
assert_eq!(own_journal,   batches_total * sink_lanes, "…every ownership lane");
assert_eq!(lance_journal, batches_total * sink_lanes, "…every lance lane");
```

**INV-6 — Aligned fixed indices (ownership guarantee).**
`src/lane_i.rs`, `OwnershipSink::pre_start` builds the guarantee table
and every applied row is checked against it:

```rust
row_owner: (0..SLOTS).map(|i| i as MailboxId).collect(),
// per applied dirty row:
debug_assert_eq!(state.row_owner[s], s as MailboxId);
```

**INV-7 — Identity minted once, stable, unique (codebook).**
`src/lane_i.rs`, `Codebook::mint` (Morton placement + linear probe +
full `(h, name)` verification at mint only) + test
`codebook_mints_unique_stable_slots`:

```rust
let a1 = cb.mint(fnv1a64(b"alpha"), b"alpha");
let a2 = cb.mint(fnv1a64(b"alpha"), b"alpha");
assert_eq!(a1, a2);            // idempotent per identity
assert_ne!(a1, b1);            // distinct identities, distinct cells
```

**INV-8 — Flush-cache exclusivity (no aliased batch reuse).**
`src/lane_i.rs` / `src/lane_j.rs`, `FlushCache::next_table` — a batch
table is reused only when BOTH sinks dropped their `Arc`:

```rust
match Arc::try_unwrap(front) {
    Ok(mut table) => { table.reset(); return table; } // refcount 1: flush done
    Err(still_in_flight) => { self.pool.push_front(still_in_flight); }
}
```

**INV-9 — Owner-table fullness is a panic, never silent corruption.**
`src/lane_g.rs`, `OwnerSoa::merge_entry` (bounded probe):

```rust
panic!("OwnerSoa full: more stations routed to one mailbox than its capacity");
```

**INV-10 — Lifecycle legality (kanban card family).**
`src/lane_e.rs` asserts every recorded move is a legal Rubicon edge:

```rust
assert!(m.from.can_transition_to(m.to), "…must be a legal Rubicon edge");
assert_eq!(journal.len(), 3 * batches, "3 moves per card");
```

**INV-11 — SIMD provenance.** All SIMD in this crate routes through
`ndarray::simd` (`U8x32::{splat, from_slice, cmpeq_mask}` +
`array_chunks`); no raw `core::arch` intrinsics, no third-party SIMD
crates. Enforced by review + the module docs; grep pattern:
`core::arch|_mm256|pulp|wide::` over `src/` must return nothing.

## 5. Reproduction

```bash
cargo test  --manifest-path crates/onebrc-probe/Cargo.toml \
            --features lane-b,lane-d,lane-e,lane-g,lane-h,lane-i,lane-j,presets
cargo build --release --manifest-path crates/onebrc-probe/Cargo.toml \
            --features lane-b,lane-d,lane-e,lane-g,lane-h,lane-i,lane-j
B=./crates/onebrc-probe/target/release/onebrc-probe
$B gen /tmp/onebrc_10m.txt 10000000 42        # verify the sha256 line
$B run /tmp/onebrc_10m.txt <lane> [workers] [knobs]   # see README §4
```

Lane-by-lane measurement history with commands: `README.md`
§5.0–§5.7. Interpretation: `COMMENTARY.md`.

## 6. Lane K pre-registration (2026-11-15, before any lane K measurement)

Lane K (`src/lane_k.rs`): dense station ids in first-appearance order,
names as slices of the input, temperature as an order-preserving code
(`tenths + 999`), one sort of distinct names at the end. `k` parses the
temperature, `kt` looks it up in a 4,096-slot codebook of all 1,999
spellings. `wall` times open-to-result in ns per row; lane K mmaps, lane S
uses `fs::read` inside the timer.

Runs: the §1 corpus (10M rows, seed 42) and a 100M-row corpus (seed 42),
4 workers, 5 passes per lane, median reported, the first pass discarded
as page-cache warm-up.

Predictions:

- K1: `run` lane `k` vs lane `s`: within ±10 %. Both use the same scan and
  parse; only the grouping table differs, and with about 400 stations
  both tables stay in L1.
- K2: `run` lane `kt` vs lane `k`: within ±10 %. The parse is already a few
  instructions; a table load is not expected to beat it.
- K3: `wall` lane `k` vs lane `s` on 100M rows: lane `k` lower by at least
  the copy `fs::read` makes (expected 10-30 % of the `s` wall time on a
  warm page cache). Kill: lane `k` not lower.

## 7. Lane K results (2026-11-15)

4 workers, 6 passes per cell, first discarded, median of 5 (range in
brackets). Container host, warm page cache. 10M = the §1 corpus
(sha256 `f1853caa…a955691`, matches §1); 100M = seed 42, sha256
`1225c952…6a892`.

`run` (compute only, file already in memory), Mrows/s:

| lane | 10M | 100M |
|---|---|---|
| s | 107.1 [103.4-134.4] | 134.2 [129.4-136.6] |
| k | 131.5 [118.0-156.7] | 143.5 [138.3-145.0] |
| kt | 111.0 [94.0-133.5] | 117.8 [96.6-130.7] |

`wall` (open to result), ns per row:

| lane | 10M | 100M |
|---|---|---|
| s (`fs::read`) | 14.46 [13.91-15.16] | 14.64 [14.04-14.96] |
| k (mmap) | 7.23 [7.01-8.41] | 7.02 [6.73-7.11] |
| kt (mmap) | 8.62 [7.39-9.92] | 8.02 [7.86-8.33] |

Against the predictions:

- K1 (`k` vs `s` within ±10 %): holds on 100M (+6.9 %). On 10M the
  median is +22.8 %, but the 10M ranges overlap widely; the short runs
  are too noisy here to separate the lanes.
- K2 (`kt` vs `k` within ±10 %): fails. The temperature codebook is
  slower than the parse: −17.9 % on 100M in `run`, +14.3 % ns per row in
  `wall`. One open-addressed table load with key verification costs more
  than lane S's few-instruction parse.
- K3 (`wall` `k` below `s`, by 10-30 %): direction holds, size does not:
  −52.1 % on 100M, −50.0 % on 10M. On a warm cache, `s` spends about
  7.2 ns per row (14.64 − 7.45 compute-only) on `fs::read` copying the
  file, about as long as the whole computation; with mmap, `wall` `k`
  (7.02) is within 1 % of its compute-only time (6.97).

Not measured: a cold page cache (both lanes would then wait on the disk),
the 10,000-station key set, and anything beyond 4 workers.

What this says: the mmap saving is the large effect, and it comes from not
copying the input, not from the codebook. The codebook grouping is a
small gain over lane S's hash table (+7 % on 100M). The temperature
codebook loses to the parse.

## 8. Lane K variants pre-registration (2026-11-15, before measuring them)

New variants: `km` (merykitty multiply parse from one 8-byte load, newline
search skipped), `kw` (names hashed 8 bytes at a time, last word masked),
`kmw` (both). Same corpora, workers and pass rule as §6; reference is §7's
`k`, re-measured in the same session.

- K4: `km` beats `k` in `run` on 100M by at least 10 %. It removes the
  parse's two branches and the whole newline scan.
- K5: `kw` beats `k` in `run` on 100M by at least 5 %. Station names
  average under 16 bytes, so FNV's byte loop costs up to ~16 multiplies
  against at most 2 for the word hash.
- K6: `kmw` is the fastest variant in both `run` and `wall`.
Kill for each: the stated gain is absent.

## 9. Lane K variants results (2026-11-15)

Same session, same rules as §7 (4 workers, 6 passes, first discarded,
median of 5, range in brackets).

| variant | `run` 100M, Mrows/s | `wall` 100M, ns/row | `wall` 10M, ns/row |
|---|---|---|---|
| k (parse, FNV) | 136.5 [133.9-144.4] | 7.56 [7.02-8.81] | 7.02 [6.96-8.98] |
| km (merykitty, FNV) | 188.2 [149.7-196.2] | 5.42 [5.30-5.64] | 6.52 [5.89-8.33] |
| kw (parse, word hash) | 150.5 [139.0-190.5] | 5.40 [5.31-6.21] | 5.42 [5.38-8.20] |
| kmw (both) | 245.0 [219.2-246.7] | 4.00 [3.69-4.53] | 5.73 [5.00-6.36] |

Against §8:

- K4 (`km` ≥ +10 % over `k`, `run` 100M): holds, +37.8 %.
- K5 (`kw` ≥ +5 %): holds on the median, +10.2 %, but `kw`'s range
  (139-191) overlaps `k`'s; weak.
- K6 (`kmw` fastest everywhere): holds on 100M (`run` +79.5 %, `wall`
  −47.0 % vs `k`); fails on `wall` 10M, where `kw` (5.42) is below `kmw`
  (5.73). The 10M runs are short and noisy here, as in §7.

The two changes compound: together they give more than either alone on
100M (+79.5 % against +37.8 % and +10.2 %).

One unplanned observation: the first `wall` 10M pass of `k` took 91.99 ns
per row. The 100M runs had evicted the 10M file from the page cache, so
that pass read from disk. One pass, not a measurement of the cold case, but
it shows the cold page cache is an order of magnitude above everything in
these tables.

Per core, `kmw` on 100M is about 16 core-ns per row (4.00 × 4) against the
1BRC winner's about 12.3 (1.535 s for 1B rows on 8 Zen 2 cores). Different
CPU and corpus; an order-of-magnitude reading only.

## 10. Instructions per row, and the build tier (2026-11-15)

**Build tier, a correction for §7 and §9.** The crate's
`.cargo/config.toml` asks for `-C target-cpu=x86-64-v3`, but cargo merges
it with the repository's `-Ctarget-cpu=native`, and native wins: the
release binary contains AVX-512 code (35 `zmm` uses). Every timing in §7
and §9 therefore ran at this host's native tier (a Xeon with AVX-512 incl.
VBMI/BF16, VNNI and AMX), not AVX2. The lane K kernels are scalar SWAR,
so this mainly concerns the surrounding code.

**Instruction counts.** The VM exposes no hardware performance counters,
so `perf` is unavailable. Counts are from `valgrind --tool=callgrind` on a
separate build pinned to AVX2 (`RUSTFLAGS="-C target-cpu=x86-64-v3"`,
own target dir, 0 `zmm` uses; valgrind cannot run AVX-512). Per row =
(Ir on 2M rows − Ir on 1M rows) / 1M, 1 worker, `wall` mode, which cancels
startup, mapping and the merge.

| variant | instructions/row |
|---|---|
| k | 255.71 |
| kt | 264.07 |
| km | 225.19 |
| kw | 225.66 |
| kmw | 195.14 |

For `kmw`, 175.5 per row are in `accumulate_with` and 19.7 in libc's
`memcmp`, which the name check calls once per row. The QuestDB article's
final Java version reports about 98 per row (its own count, different
corpus), so `kmw` executes about twice as many.

## 11. Histogram variant pre-registration (2026-11-15, before measuring it)

`kh` (parse, FNV, histogram) and `kmwh` (multiply parse, word hash,
histogram): one `u32` counter per (station, code), 1,999 per station, one
increment per row, min/max/sum/count derived before the merge. Measured
against `k` and `kmw` in the same session: `run` and `wall` on 100M,
4 workers, §6's pass rule; instructions per row by callgrind as in §10.

- H1: fewer instructions per row than the running variant (`kh` < `k`,
  `kmwh` < `kmw`): the update drops two compares and a sum.
- H2: not faster in `run`: `kmwh` within ±10 % of `kmw` or slower. The
  3.2 MB of counters per worker leave L1 and most of L2, while the running
  state (about 6.6 KB) stays in L1, so each row's increment costs more than
  the instructions it saves.
Kill for H2: `kmwh` faster than `kmw` by more than 10 %.

## 12. Histogram variant results (2026-11-15)

Same session, rules as §11 (4 workers, 100M, median of 5 after one
discarded pass, range in brackets). Instructions per row by callgrind on
the AVX2 build, as in §10; the running variants were re-counted on this
commit.

| variant | instructions/row | `run` Mrows/s | `wall` ns/row |
|---|---|---|---|
| k | 252.21 | 135.8 [108.2-142.5] | 7.75 [6.69-8.40] |
| kh | 236.21 | 135.0 [116.4-144.1] | 7.38 [6.66-7.74] |
| kmw | 194.48 | 248.3 [230.6-264.7] | 4.32 [4.11-4.87] |
| kmwh | 178.48 | 251.4 [241.2-260.6] | 3.70 [3.53-4.58] |

- H1 (fewer instructions): holds, exactly 16.00 fewer per row in both
  pairs.
- H2 (not faster in `run` by more than 10 %): holds: `kmwh` +1.3 % over
  `kmw`, `kh` −0.5 % against `k`, both inside the other's range. `wall`
  medians favour the histogram (−14.3 % for `kmwh`), but its range overlaps
  `kmw`'s; not a separable difference here.

So the 16 saved instructions do not show up as time at 400 stations: the
counter increment into 3.2 MB per worker costs about what the compares
did. The re-count also moved `k` from 255.71 (§10) to 252.21, so the
dispatch added for the histogram did not cost the running variants
instructions.

## 13. Median and percentiles from the histogram (2026-11-15)

`lane_k_distributions` keeps each station's 1,999 code counts, summed
across workers, and `Distribution` reads from them:

- `percentile(p)`: nearest rank, `p` a whole percent 0-100: the smallest
  observed value with at least ⌈p·n/100⌉ observations at or below it,
  computed in integers. `percentile(0)` is the minimum.
- `median()`: `percentile(50)`, the lower middle value when n is even.
- `nth(k)`: the k-th smallest observation; `stats()`: min, max, sum, count.

Tests: every station's p0, 1, 10, 25, 50, 75, 90, 99 and 100 equal a
sorted copy of its raw values (60,000 rows, 1/3/7 workers, two scan
variants), and `stats()` equals lane A. A small hand-checked set pins the
even-count median and the rank boundaries. Disable controls, each red:
rounding the rank down, comparing with `>` instead of `>=`, and merging
only the first worker's histogram.

CLI: `dist <path> [workers]` (mmap, multiply parse, word hash). On the 100M
corpus, 4 workers, warm cache, three passes after a cold one gave 3.72,
4.89 and 3.75 ns/row open to result, including percentiles for all 400
stations. Spot check: the generator draws each reading uniformly within
±10.0 of the station mean, and the reported p10 and p90 sit 8.0-8.1 below
and above the median, as a uniform spread predicts.

## 14. 10,000-station key set pre-registration (2026-11-15, before measuring)

Corpus: `gen <path> 100000000 42 10000` (100M rows, 10,000 stations, same
generator). Lanes `s`, `k`, `kmw`, `kmwh` in `run` and `wall`, plus `dist`;
4 workers, §6's pass rule; instructions per row by callgrind (2M − 1M rows,
10,000 stations) as in §10.

At 10,000 stations a worker's running state is about 165 KB (four arrays,
hashes and name slices) plus a 128 KB slot table: out of L1, inside L2. The
histogram is 80 MB per worker, so each row's increment misses every cache.

- X1: every lane is slower per row than on the 400-station corpus.
- X2: `kmwh` is slower than `kmw` in `run` by more than 10 %.
- X3: `dist` is slower than `kmw` in `wall` by more than 10 %.
- X4: `kmw` stays faster than lane `s` in `run`.
Kill for each: the stated relation does not hold.

## 15. 10,000-station key set results (2026-11-15)

Corpus `gen … 100000000 42 10000`, sha256 `e1934f6f…84a75fd`, 10,000
stations. 4 workers, 6 passes, first discarded, median of 5, range in
brackets. Instructions per row by callgrind on the AVX2 build (2M − 1M
rows, 10,000 stations).

| lane | instructions/row | `run` Mrows/s | `wall` ns/row | 400 stations: `run`, `wall` (§7, §9, §12) |
|---|---|---|---|---|
| s | 240.49 | 79.4 [76.2-88.0] | 20.89 [18.04-23.52] | 134.2, 14.64 |
| k | 258.58 | 115.8 [108.4-117.2] | 9.86 [8.70-10.69] | 136.5-143.5, 7.02-7.75 |
| kmw | 201.30 | 149.1 [121.7-168.4] | 6.15 [5.78-8.06] | 245.0-248.3, 4.00-4.32 |
| kmwh | 185.29 | 59.9 [51.2-66.7] | 13.77 [12.40-16.70] | 251.4, 3.70 |
| dist | – | – | 14.49 [13.41-15.89] | 3.72-4.89 |

Against §14:

- X1 (every lane slower than at 400 stations): holds for all five.
- X2 (`kmwh` slower than `kmw` in `run` by > 10 %): holds, −59.8 %.
- X3 (`dist` slower than `kmw` in `wall` by > 10 %): holds, +135 %.
- X4 (`kmw` faster than lane `s` in `run`): holds, +87.8 %.

The histogram executes the fewest instructions per row (185.29) and is
the slowest lane K variant: at 80 MB per worker every increment misses the
cache. Instruction count is not the cost at this key count; memory is. The
running variant's state stays near L2 and keeps `kmw` ahead. Percentiles
at 10,000 stations cost about 2.4× the plain statistics; at 400 stations
they cost nothing extra (§13).

The 10,000 names here come from the probe's syllable generator, shorter
than the 1BRC key set's names (up to 100 bytes), so name handling is
cheaper here than in the official 10K run.

## 16. Codec9 counter window pre-registration (2026-11-15, before measuring)

`Agg::Codec9` (`--features codec9`): each reading counted at its Codec9
word (ndarray `hpc::codec9`, one tenth per step), 512 counters per station
covering ±25.5; readings beyond that go to a full-range overflow row the
station allocates on first use. In this generator 2.52 % of readings and
89 of 400 mean values reach beyond ±25.5, so about 22 % of stations carry
an overflow row: about 0.8 MB window + 0.7 MB overflow per worker at 400
stations, 20.5 MB + about 18 MB at 10,000 (against the histogram's 3.2 and
80 MB). Lane `kmwc` (multiply parse, word hash, Codec9 window) against
`kmw` and `kmwh`, and `dist … c9` against `dist … hist`, on both corpora,
§6's pass rule; instructions per row as in §10.

- Y1: at 10,000 stations `kmwc` beats `kmwh` in `run` by more than 10 %
  (half the counter memory).
- Y2: at 10,000 stations `kmwc` is still slower than `kmw` (38 MB of
  counters per worker remain far outside L2).
- Y3: at 400 stations `kmwc` is within ±10 % of `kmwh` in `run`.
- Y4: `kmwc` executes more instructions per row than `kmwh` (the encoder's
  f32 round-trip and the window bound check).
Kill for each: the stated relation does not hold.
