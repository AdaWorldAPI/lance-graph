# SOA64-U64 R1 lab results: demand amortization, dense u64 vs narrow SoA vs AoS

**Status:** LAB RESULT for research PR #1973, stage R1. Not a decision. Not an admission. Nothing here is admitted for production.
**Date:** 2026-11-07.
**Answers:** `docs/research/soa64-u64-demand-amortization-map-v1.md` (hypotheses H1-H4, nulls N1-N6).
**Lab:** `crates/soa64-u64-lab/` (own workspace root; nothing depends on it).
**Machine:** Intel Xeon Processor @ 2.10GHz, 4 vCPU. The bench env line reports `avx2=true avx512f=true`. The build uses `target-cpu=native` (set in the repo's `.cargo/config.toml`). rustc 1.99.0 (the bench env line prints `rustc=unknown`).
**Reproduce** (the two commands in the `Cargo.toml` header):

```
cargo test --release --manifest-path crates/soa64-u64-lab/Cargo.toml
cargo run  --release --manifest-path crates/soa64-u64-lab/Cargo.toml --bin bench
```

**Which numbers are quoted.** All tables below come from `crates/soa64-u64-lab/results/bench-2026-11-07-run1.txt`. A second run of the same build (`crates/soa64-u64-lab/results/bench-2026-11-07-run2.txt`) exists. The repo config already sets `target-cpu=native`, so the second run is a reproducibility run. It is **not** an ISA comparison; no AVX2-only or scalar-only build was measured. The two runs agree on the structure of every verdict. They differ in some cells by up to about 2x on single cells (for example soa-narrow DemandFold d=64 at n=262144: 13.4 vs 22.3 ns/id; aos d=8: 22.6 vs 28.9). Single-cell values therefore carry noise of that size. See section 5 for which verdicts depend on a single cell.

**Correction to the "lines touched" column (run 3).** In runs 1 and 2 the column counted all 8 cache lines of every AoS and hash-row row, whatever the query read. That overstated AoS traffic by 8x for single-field queries. Run 3 (`crates/soa64-u64-lab/results/bench-2026-11-07-run3.txt`) counts only the lines that hold demanded fields, plus one bucket line per hash lookup. It is an estimate and a lower bound: hardware prefetch may pull adjacent lines, and AoS `CountWhere` at about 13 ns per row costs more than one line per row would explain. No verdict below uses this column. Run 3 timings agree with run 1 within the noise stated above.

## 1. What was built

Five physical representations of the same 64-field logical rows:

| arm | shape |
|---|---|
| aos | one `[u64; 64]` per row |
| soa-u64 | lane-major dense `u64`, one slab, lane `f` at `[f*n .. (f+1)*n]` |
| soa-narrow | per-field narrowest of u8/u16/u32/u64 that holds the field width |
| soa-bitpack | per-field bit-packed at exactly `field_bits(f)` bits, scalar decode |
| hash-row | rows keyed by global guid; every element access is an identity lookup `map[&guid]`, walked in canonical order (the H4 null comparator) |

Field widths come from `fixture::field_bits`: field 0 is 4 bits, field 1 is 64 bits (with boundary values), field 2 is 16 bits and the only nullable field (about 20% absent; absent is not zero), fields 3-63 cycle through 1, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64 bits.

Queries:

- `CountWhere`: count of rows with `A == 7` (about 1/16 of rows). Demands field 0.
- `SumWhere`: `SUM(B) WHERE A == 7`, exact in `u128`. Demands fields 0, 1.
- `SumPresentC`: `SUM(C)` over rows with `A == 7` and C present. Demands field 0, 2 and the presence mask.
- `DemandFold d`: over all rows, fold fields `0..d` with `acc = acc.wrapping_add(v_f ^ f)`, for d = 1, 3, 8, 32, 64.
- `Point 1024`: 1024 random probes, each XOR-folds fields 0..5 (a 5-field virtual object).

Sizes: n = 256, 513, 4096, 65536, 262144.

Method:

- Every arm is built from the fixture through `fixture::value` and `fixture::present` only. An independent oracle (`oracle.rs`) reads only those two functions.
- Every timed answer is checked against the oracle before it is timed.
- Allocation count in the warm loop is 0 for every (arm, query, n) in the baseline run.
- Warm = repeated runs over resident data. Cold = first touch after the data has been evicted by the harness. The tables report ns per identity (ns per probe for `Point`).
- All loops are plain scalar Rust left to LLVM autovectorization.

## 2. Correctness and refusals

Nine tests, all passing on the restored tree (`cargo test --release`: 9 passed, 0 failed).

| file | test |
|---|---|
| `correctness.rs` | `every_arm_equals_the_oracle_on_every_query` |
| `correctness.rs` | `narrow_lane_widths_are_narrow_but_field_one_is_u64` |
| `correctness.rs` | `the_fixture_is_not_vacuous` |
| `demand_poison.rs` | `only_the_demanded_lanes_are_borrowed` |
| `demand_poison.rs` | `demanding_every_field_borrows_every_lane` |
| `socket_refusals.rs` | `one_coordinate_space_binds_sums_exactly_and_borrows_in_place` |
| `socket_refusals.rs` | `a_lane_in_another_row_order_is_refused` |
| `socket_refusals.rs` | `without_the_refusal_a_reordered_lane_reads_wrong` |
| `socket_refusals.rs` | `a_lane_for_another_scope_is_refused` |

**Disable runs** (each applied, confirmed to patch exactly once, then the file was restored and the tree confirmed clean):

| disable | what was broken | result |
|---|---|---|
| D1 | the demand guard in `socket.rs` | `demand_poison::only_the_demanded_lanes_are_borrowed` went red (8 passed, 1 failed) |
| D2 | the 64-bit lane exactness in `arms.rs` (64-bit lane truncated to u32) | `correctness::narrow_lane_widths_are_narrow_but_field_one_is_u64` and `correctness::every_arm_equals_the_oracle_on_every_query` went red (7 passed, 2 failed) |

**Coordinate-space falsifier.** Binding a lane declared in another row order (`SPACE_OTHER`) against the canonical space is refused with `LaneRefusal::CoordinateSpace`. The refusal is load-bearing: `without_the_refusal_a_reordered_lane_reads_wrong` reads the reversed-order lane B positionally against lane A and shows the SUM differs from the oracle. A lane for another reader scope is refused the same way.

**Honest limit.** The `admit_lane` guard itself lives in `lance-graph-dt64`, which this lab must not edit, so that guard was not disabled. The positional-read test stands in for it: it shows what happens if the guard were absent, not that removing the guard turns a test red.

## 3. Results

Warm and cold ns per identity (ns per probe for `Point`), from `crates/soa64-u64-lab/results/bench-2026-11-07-run1.txt`.

### n = 65536

| query | aos warm | aos cold | soa-u64 warm | soa-u64 cold | soa-narrow warm | soa-narrow cold | soa-bitpack warm | soa-bitpack cold |
|---|---|---|---|---|---|---|---|---|
| CountWhere | 12.658 | 14.976 | 0.107 | 0.366 | 0.088 | 0.188 | 0.884 | 0.902 |
| SumWhere | 10.780 | 14.662 | 0.672 | 0.881 | 0.663 | 1.555 | 1.623 | 2.010 |
| DemandFold d=1 | 10.892 | 16.648 | 0.112 | 0.359 | 0.087 | 0.141 | 0.850 | 0.870 |
| DemandFold d=3 | 14.127 | 16.695 | 0.395 | 1.500 | 0.244 | 0.946 | 2.364 | 3.764 |
| DemandFold d=8 | 14.513 | 18.873 | 2.501 | 3.067 | 0.677 | 1.668 | 6.455 | 8.188 |
| DemandFold d=32 | 19.217 | 26.939 | 11.034 | 16.255 | 6.381 | 7.045 | 33.237 | 42.935 |
| DemandFold d=64 | 20.994 | 33.943 | 21.705 | 35.408 | 13.478 | 17.397 | 71.801 | 96.909 |
| Point (ns/probe) | 1.310 | 18.350 | 2.391 | 24.494 | 3.111 | 18.094 | 4.450 | 25.470 |

hash-row, CountWhere only: 35.163 warm, 48.249 cold.

### n = 262144

| query | aos warm | aos cold | soa-u64 warm | soa-u64 cold | soa-narrow warm | soa-narrow cold | soa-bitpack warm | soa-bitpack cold |
|---|---|---|---|---|---|---|---|---|
| CountWhere | 13.717 | 18.672 | 0.169 | 0.377 | 0.087 | 0.167 | 0.896 | 0.893 |
| SumWhere | 11.454 | 17.521 | 0.934 | 1.024 | 1.008 | 1.054 | 1.815 | 1.767 |
| DemandFold d=1 | 11.819 | 17.176 | 0.173 | 0.338 | 0.080 | 0.123 | 0.795 | 0.775 |
| DemandFold d=3 | 12.940 | 20.895 | 0.899 | 1.121 | 0.487 | 0.545 | 3.017 | 3.207 |
| DemandFold d=8 | 22.616 | 30.642 | 2.532 | 4.195 | 1.118 | 1.203 | 7.235 | 7.420 |
| DemandFold d=32 | 42.926 | 43.271 | 11.433 | 22.731 | 6.252 | 9.847 | 57.457 | 57.792 |
| DemandFold d=64 | 43.231 | 43.934 | 54.609 | 54.588 | 13.366 | 22.343 | 125.898 | 124.999 |
| Point (ns/probe) | 1.345 | 22.210 | 6.702 | 29.867 | 3.606 | 19.503 | 5.083 | 65.829 |

hash-row, CountWhere only: 100.170 warm, 98.183 cold.

### Bytes resident, n = 262144

| arm | bytes |
|---|---|
| aos | 134,250,496 |
| soa-u64 | 134,250,496 |
| soa-narrow | 70,025,216 |
| soa-bitpack | 56,066,048 |
| hash-row | 245,694,464 |

Identity lookups during a `SumWhere` sweep, every n: soa-u64 = 0; hash-row = n (for example 262144 at n = 262144). Ingress through G2L costs n lookups once.

## 4. Verdicts

These are verdicts on this fixture, this machine, and these scalar implementations. They are not production claims.

### H1 demand amortization: SUPPORTED for the columnar arms

soa-u64 warm cost grows with the number of demanded lanes. At n = 262144: d=1 0.173, d=3 0.899, d=8 2.53, d=32 11.4, d=64 54.6 ns/id.

Per-lane cost rises from about 0.11 ns/id/lane while the demanded working set fits in cache to about 0.34 ns/id/lane once it spills. Reading the table: at n = 65536, d=3 is 0.13 per lane and d=8, 32, 64 are 0.31, 0.34, 0.34. At n = 262144, d=8 and d=32 are 0.32 and 0.36, and d=64 is 0.85 per lane. The d=64 point at n = 262144 sits well above the other points.

CONJECTURE: cache residency explains the step. It was not measured separately; no perf counters were collected.

### H2 virtual transpose: CONDITIONAL

Positional virtual objects work and allocate nothing (allocs = 0 for the `Point` query on every arm; the socket borrows each lane in place). But the 5-field `Point` probe is cheaper in AoS warm: 1.3 ns/probe, against 2.4-6.7 for soa-u64 (2.391 at n = 65536, 6.702 at n = 262144). Row locality wins random multi-field access. Cold `Point` is a different picture: aos 18-22 ns, soa-u64 24-30, soa-narrow 18-20.

### H3 total-cost crossover for dense u64: FALSIFIED against narrow SoA on this fixture

- **Null N1 observed (narrow wins).** On `DemandFold`, soa-narrow beats soa-u64 at every demand density measured at n = 65536 and n = 262144, at 52% of the bytes. At n = 262144: d=1 0.080 vs 0.173, d=3 0.487 vs 0.899, d=8 1.118 vs 2.53, d=32 6.25 vs 11.4, d=64 13.4 vs 54.6 ns/id. Exceptions to state: `SumWhere` at n = 262144 is a near tie (1.008 narrow vs 0.934 u64, narrow slower), and `Point` at n = 65536 favors u64 (2.4 vs 3.1). At small n (256-4096), d=1 and d=3 are within noise or favor u64.
- **Null N2 observed (AoS at full demand).** Dense u64 SoA beats AoS for d <= 32 but loses to AoS at d=64 at n = 262144 (54.6 vs 43.2 warm). At n = 65536 d=64 it ties (21.7 vs 21.0).
- Bitpacked is the smallest (56 MB vs 70 MB narrow vs 134 MB dense) but the slowest to decode in this scalar implementation (d=64 at n = 262144: 125.9 ns/id).

This is the published counterexample the research map required. The single-cell d=64 values vary between the two runs (soa-narrow 13.4 vs 22.3 at n = 262144), but soa-u64 at 52-55 is above both, so the ranking of narrow over dense u64 holds in both runs.

### H4 epistemic amortization: SUPPORTED

Identity lookups inside the positional sweep are 0 for the positional arms (by construction: they have no lookup path) and n for hash-row. hash-row costs about 92-162 ns/id at n = 262144 (33-67 ns/id at n = 65536), against 0.17-55 for the positional arms. G2L happens once at ingress (n lookups).

## 5. Gate breaches, stated

The repo timing doctrine says a batch should cost at most microseconds. These (arm, query, n) warm totals from `crates/soa64-u64-lab/results/bench-2026-11-07-run1.txt` exceed 1000 µs. They are full-population bandwidth-bound scans over 65536 or 262144 identities. They are reported because hiding them would flatter the result.

| n | arm | queries over 1000 µs warm |
|---|---|---|
| 65536 | aos | DemandFold d=32 (1259), d=64 (1376) |
| 65536 | soa-u64 | DemandFold d=64 (1422) |
| 65536 | soa-bitpack | DemandFold d=32 (2178), d=64 (4706) |
| 65536 | hash-row | all full scans (2192-4378) |
| 262144 | aos | all full scans (3002-11333) |
| 262144 | soa-u64 | DemandFold d=32 (2997), d=64 (14316) |
| 262144 | soa-narrow | DemandFold d=32 (1639), d=64 (3504) |
| 262144 | soa-bitpack | DemandFold d=8 (1897), d=32 (15062), d=64 (33003) |
| 262144 | hash-row | all full scans (23.9k-42.5k) |

At n <= 4096 no soa-u64 or soa-narrow query exceeds 1000 µs. Single-demand soa scans at n = 262144 (`CountWhere`, `DemandFold d=1`) are 21-45 µs; they stay in microseconds.

## 6. What was NOT tested

- `ndarray::simd` and `simd_stride`. The ndarray sibling is not checked out in this environment. All loops are plain scalar code left to LLVM autovectorization. The numbers say nothing about a hand-written or `ndarray::simd` fold.
- Hardware cache-miss counters. No perf counters were collected. The cache explanation in H1 is a conjecture.
- NEON, AVX2-only, or scalar-only builds. Only one build was measured (twice).
- Mutation and frame-churn costs. The fixture is immutable.
- Persistence and egress. The `Point` query returns a folded scalar; no terminal value set is serialized.
- Multithreading. Every run is single-threaded.
- Production `AbiLanes` or u64 `LaneRef` admission. The lab uses the DT64 lane socket directly; it does not touch the production reader-scope issuer.
- `Gather`. No cross-coordinate mapping was built; mismatched coordinate spaces are refused, not mapped.
- Tag and decimal codec questions (B3). The fixture holds plain integers.
- Multi-tenant and sparse-identity stress (N3).

## 7. Recommendation (no approval implied)

The result argues for keeping 64 LOGICAL fields with narrow PHYSICAL lanes chosen per field, positional sweeps with no per-element identity lookup, and AoS-style row tiles only for random multi-field access. It does not argue for dense resident 64 x u64. That is a statement about where this lab's evidence points; it approves nothing and changes no decision. Candidate next gates, at most three:

- (a) Rerun `DemandFold` through `ndarray::simd` with the sibling checked out.
- (b) A perf-counter run to confirm or reject the cache-spill explanation for the per-lane cost step.
- (c) The scope and coordinate production witness. The first missing interface is an authoritative reader-scope issuer for `AbiLanes`.

## 8. Closeout

`PR #1973 | STATUS: measured | OUTCOME: R1 lab built (5 arms, 9 tests green, 2 disable runs red as expected); H1 and H4 supported, H2 conditional, H3 falsified for dense u64 vs narrow SoA; nulls N1 and N2 observed; nothing admitted | OPEN: ndarray::simd rerun; perf-counter confirmation of cache-spill; reader-scope issuer for AbiLanes; dt64 admit_lane guard not disabled; B3 codec questions untouched`
