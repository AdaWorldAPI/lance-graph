# D-HOLO-H14c — thought-first positional reads, Hamming presence and Lance 13 masked I/O (v1)

**Status:** MEASURED 2026-10-15 (v4). Acceptance A–E met; Lance prediction holds;
C vs B on query-shaped reads mixed (reported). Results in § Results; the
pre-registration above them is unchanged.

**D-ids:** D-HOLO-H14c (umbrella), D-HOLO-H14c-1 (point projection and
independence), D-HOLO-H14c-2 (presence/Hamming populations), D-HOLO-H14c-3
(memory and fragmentation), D-HOLO-H14c-4 (Lance 13 masked I/O), D-HOLO-H14c-5
(coordinate identity and temporal safety), D-HOLO-H14c-6 (workload harvest).

**Predecessors:** H14 and H14b in `.claude/plans/d-holo-address-indexed-algebra-v1.md`
(PR #1566 merged, PR #1569 draft); board entry
`2026-10-15-d-holo-h14b-lane-mask.md`. H14b's arm M is withdrawn as a layout
candidate (that plan's "H14b withdrawal note").

## Overview

H14 and H14b asked how cheaply each layout rebuilds a row. C does not need to
rebuild rows for ordinary computation, so that question cannot decide whether C
works. H14c asks instead:

> If the thought, its operands and the Cartesian address are already known,
> which operations are still necessary, which memory accesses are unavoidable,
> and which materialization and traversal costs disappear?

Canonical expression:

```
Thought(t)[Table(a)[i] : Table(b)[i]]
```

- `t` names the thought (the execution perspective);
- the thought's program selects lane tables `a` and `b`;
- `i` is the shared Cartesian row address;
- each selected table is read at position `i`;
- presence says whether a value is there, never where it is;
- merge-on-read combines the selected operands at their common address.

The thought selects the tables, the address selects the operands, the
instruction operates on them.

## Checklist

- [x] Pre-registration committed before any benchmark (this file).
- [x] D-HOLO-H14c-1: typed thought-first read (`Selection` → resolved handles →
      `read(i)`), with counters; oracle tests; disable runs.
- [x] D-HOLO-H14c-2: presence populations through mask-risc programs (AND, OR,
      XOR, `Count`); scalar reference; disable runs.
- [x] D-HOLO-H14c-3: memory and fragmentation measures (lazy, eager, sparse);
      perf counters where the kernel allows.
- [x] D-HOLO-H14c-4: Lance 13 masked I/O probe (combined mask + projection;
      independent per-lane masks), with `ExecutionSummaryCounts`.
- [x] D-HOLO-H14c-5: coordinate identity across append, delete, update,
      reopen, compaction and version change; stale mapping refused.
- [x] D-HOLO-H14c-6: lane-selection harvest over the repository's existing
      programs, labelled as test-corpus observation.
- [x] v4 runs with the tier printed; raw results machine-readable.
- [x] Results section, board entry, STATUS_BOARD, AGENT_LOG, draft PR.

## 1. Contract: the address is the register

The 16-byte lane arrangement is a generic register contract. CE64, SPOFC,
NARS evidence, thought identity and the like are typed readings over it, not
fixed meanings of its bytes. H14c reads bytes; it gives them no meaning.

The fundamental operation is

```
read(t, i, L) -> operands
R(t, i, L) = { V[t,l][i] | l in L, P[t,l][i] = 1 }
```

with `L` the set of lanes the thought's program requests. Requirements, each a
test in § 6:

1. Table handles come from the program (resolved once), never from the row.
2. A selected table is read at exactly address `i`.
3. The number of lane tables the thought touched outside `L` does not change
   how many tables a read consults.
4. An absent value is decided by one presence bit, not by a search.
5. No 512-byte row is assembled unless a named terminal asks for it.
6. Any address translation is explicit, counted, and outside the direct-position
   path. (The sparse arm in § 4 is the only place one exists.)
7. A read may load several lanes but never walks a relation to find them.

**Absent versus zero.** A read returns, per selected lane, one of
`Overlay(value)` (the thought wrote it), `Base(value)` (fallback to the
immutable base, when the selection asks for fallback) or `Absent` (no overlay
and no fallback). An overlay value of sixteen zero bytes is `Overlay([0; 16])`,
never `Absent`.

**Precedence.** Thought overlay over base, always. Two thoughts reading the
same `i` see their own overlays; a read reports which thought supplied the
value, so their evidence is not conflated.

**Authorization and time.** Row-level, field-level, tenant and temporal
restrictions enter as an aperture mask composed before payload is read. A read
under an aperture tests the aperture bit first; a denied address loads no
value. Address equality is not permission, not semantic equality and not
causal entailment.

**What the thought knows.** It knows which lanes its program requests. It does
not know which lanes are populated at a given row; the presence bit of each
requested table answers that, one test per requested lane.

## 2. Presence is Hamming geometry

Each (thought, lane) table has a presence plane `P[t,l]` of 65,536 bits (8 KiB).
Kept distinct:

| quantity | expression | meaning |
|---|---|---|
| population | `popcount(P)` | how many addresses hold a value |
| shared occupancy | `popcount(Pa & Pb)` | addresses where both operands exist |
| coverage disagreement | `popcount(Pa ^ Pb)` (Hamming distance) | addresses covered by exactly one |
| difference locations | `Pa ^ Pb` | which addresses those are |
| joint availability | `A & Pa & Pb` | operands available under aperture `A` |

These run as mask-risc programs (`MaskOp::And/Or/Xor`, `Terminal::Count`,
`Terminal::Keep`) over the planes; no new arithmetic. A population query reads
no payload value.

Equal presence does not mean equal values; Hamming distance between presence
planes is a distance between coverages, not between evidence, meanings or
causes. Presence agreement, value equality, semantic agreement, evidential
independence and causal confidence are five different things and H14c measures
only the first two.

## 3. Three execution paths, measured apart

- **A. Addressed point projection** `Thought(t)[Table(a)[i] : Table(b)[i]]`:
  handle resolution (once per program, timed separately), presence tests for
  the requested lanes, loads of present values, output of the selected
  operands. No row is built.
- **B. Address-set population**: `popcount(Pa & Pb & A)`, `popcount(Pt ^ Pu)`,
  `popcount(OR(P_t1, …, P_tn))`, through mask-risc. Folds are for populations;
  locating one value at a known address needs no fold.
- **C. Explicit terminal materialization**: a named terminal builds a 512-byte
  row (base plus every overlay of the thought) only when asked. Timed alone;
  not part of the A or B success criteria.

## 4. Arms

- **C (positional)**: one table per (thought, lane) the thought touched,
  `values: [16 B; 65,536]` indexed by the address, plus `P[t,l]`. Two storage
  variants of the same table: **C-lazy** (zeroed allocation; the OS commits a
  page on first write) and **C-eager** (every page written at seal).
- **B (connected delta, historical)**: per dirty row one record (lane mask +
  changed lanes packed), found through a positional row-to-record index; lane
  `l` is a mask test plus a popcount rank. Included to answer "is C faster than
  B for query-shaped reads"; its full-row read is not used.
- **S (sparse, explicit translation)**: per (thought, lane) the dirty addresses
  sorted with their values; lane `l` at `i` is a binary search. It exists only
  to price the memory alternative; its search is the counted address
  translation of requirement 6, and it is never called "direct".

Every arm reads the same fixture and must equal a dense reference before any
timing.

## 5. Grid (pre-registered; reduced, and the reduction is stated)

Tile 65,536 addresses, 32 lanes of 16 bytes, deterministic seeds.

- **Main grid (point projection and populations):** requested lanes
  `k ∈ {1, 2, 4, 8, 16, 31}` × dirty density `{0 %, 0.1 %, 1 %, 10 %, 50 %, 100 %}`
  × touched tables in the thought `U ∈ {k, 8, 31}` (with `U >= k`), one thought,
  uniform scattered addresses, cache-hot and cache-cold.
- **Thought axis:** `T ∈ {1, 8, 64}` at `k = 2`, densities `{1 %, 10 %}`,
  `U = 31`, for point reads across thoughts and for the session OR.
- **Spatial axis:** at `k = 2`, densities `{1 %, 10 %}`, `U = 31`: clustered
  (runs of 64), uniform scattered, one contiguous range, Morton-local (dirty
  addresses inside a few 256-address blocks), disjoint populations between
  `a` and `b`, highly overlapping populations (`b`'s dirty set is `a`'s with 10 %
  changed).
- **Memory grid:** C-lazy, C-eager and S at `T ∈ {1, 8, 64}` × densities
  `{0.1 %, 1 %, 10 %, 100 %}` × spatial `{uniform, clustered}`, `U = 31`.
- **Access patterns:** single addressed value (`k = 1`); two-lane point
  projection; repeated point projections (the same 256 addresses); mask
  intersection with population count; Hamming distance; session aperture (OR
  over thoughts); selected-value merge-on-read; explicit terminal row.

Not crossed: the full product (6 × 6 × 3 × 3 × 6 × 2 cells, most of them
redundant for the questions asked). Cells outside the grid are not claimed.

**Cache states.** Hot: the same 256 addresses repeated, after a warm-up pass.
Cold: uniformly random addresses, with a 128 MiB eviction pass before each timed
batch. **Reported unit:** ns per read (point paths), ns per 64-address word
(population paths); medians of 7 batches; the tier the program prints.

## 6. Correctness oracles (written and green before any timing)

1. Address `i` returns the exact 16-byte value of every selected lane (dense
   reference), on every arm.
2. Presence does not change a value's address (a table's value at `i` is the
   reference value regardless of which other addresses are present).
3. `Absent` and `Overlay([0; 16])` are distinguishable.
4. Overlay beats base; fallback to base only when the selection asks for it.
5. Tables consulted by a read equal `|L|` for every `U` (counter, exact).
6. Hamming weight and distance equal a scalar bit loop.
7. AND/OR/XOR populations equal a dense `bool` reference.
8. Independently masked Lance reads equal the reference (§ 7).
9. Permuting the order in which Lance returns rows does not change the
   coordinate each value is attributed to (values are matched by their stored
   coordinate, never by output position).
10. Compaction and version change never yield a value from the wrong
    coordinate: a stale mapping is refused (§ 8).
11. Point projection and population queries materialize no row (counter = 0);
    the terminal materializes exactly the rows asked for.
12. Two thoughts reading one coordinate get their own values and their own
    thought id.
13. A denied address (aperture bit 0) loads no value.

**Disable runs** (each must turn its test red; recorded with the test name):
drop a presence test; shift the address by one; substitute an unrelated lane;
probe every touched table; build the full row inside the point read; reuse a
stale mapping after compaction; treat absent as zero; skip the aperture test.

## 7. Lance 13 masked I/O (D-HOLO-H14c-4)

Verified against the locally resolved `lance 13.0.0` source:
`Scanner::with_row_addr_prefilter(RowAddrMask)`, `RowAddrMask::from_allowed`,
`RowAddrMask::from_block` (`lance-select 13.0.0`, `mask.rs`),
`RowAddrTreeMap`, `Scanner::project`, `Scanner::materialization_style`
(`Heuristic`, `AllLate`, `AllEarly`, `AllEarlyExcept`),
`Scanner::scan_stats_callback` with `ExecutionSummaryCounts { iops, requests,
bytes_read, … , all_counts, all_times }`. The scanner's own documentation:
the mask is keyed in the dataset's `_rowid` space (the row address when stable
row ids are off, the stable row id when on), and for a plain scan the mask is
the `FilteredReadExec` row source, "so only masked rows are read". Whether that
reduces bytes fetched, or only rows emitted, is the question measured here.

Fixture: one Lance dataset per tile with an explicit `coord: u32` column (the
Cartesian address) and lane columns `FixedSizeBinary(16)`, nullable (null =
absent). Strategies:

1. **Combined**: mask = rows where all requested lanes are present (`A & Pa & Pb`),
   translated to row addresses; `project(["coord", a, b])`.
2. **Independent**: one masked read per lane with that lane's own mask, then
   alignment by the stored `coord` (never by output order).
3. **Unmasked**: the same projection without a mask (the I/O baseline).

Measured per (strategy, k ∈ {1, 2, 4, 8}, density ∈ {0.1 %, 1 %, 10 %, 50 %,
100 %}, spatial ∈ {uniform, clustered}, materialization ∈ {Heuristic, AllLate,
AllEarly}): `bytes_read`, `iops`, `requests`, rows emitted, wall time, every
key in `all_counts`. Bytes decoded, pages accessed and allocations are not
exposed by the scan statistics; the probe reports which keys exist and states
what it could not measure rather than estimating it.

**Prediction (HYPOTHESIS).** Lance reads in pages, so a uniformly scattered
1 % mask touches nearly every page of a lane column: at 1 % uniform,
`bytes_read(masked) > 0.5 × bytes_read(unmasked)` (the mask cuts rows emitted,
not bytes fetched). At 1 % clustered, `bytes_read(masked) <= 0.5 ×
bytes_read(unmasked)`. Either half failing is reported as such.

This probe is a separate crate excluded from the workspace
(`crates/holo-lance-masked-io-probe`), depending only on `lance` 13, `arrow` and
`tokio`: no repository-wide dependency change.

## 8. Coordinate identity and temporal safety (D-HOLO-H14c-5)

The Cartesian address `i` is the logical substrate address. It is not a Lance
row address (`fragment << 32 | offset`), not a `_rowid`, not a row ordinal and
not an Arrow batch offset. The chain is

```
LogicalCoordinate -> DatasetVersion -> LanceRowIdentity -> PhysicalLocation
```

The probe keeps a `CoordMap { dataset_version, coord -> row identity }` built by
one scan of `(coord, _rowid)` at a named version, and refuses (`StaleMapping`)
any masked read whose dataset version differs from the map's. Scenarios, each
with and without `enable_stable_row_ids`: initial append; second append;
delete; update; reopen; compaction (`compact_files`); a rewrite that keeps the
row count; version transition; a missing lane dataset; reuse of an old map.
For each, the test asserts either "every returned value carries the requested
coord" or "the read was refused". The disable run reuses the old map after
compaction without the version check and must return a wrong coordinate (that
is the hazard the check removes) or fail its assertion.

`VersionClock` (`lance-graph-contract::temporal_pov`, #1567) states that a
reordering rewrite needs a new clock; H14c tests the same boundary for row
identity and does not extend the clock.

## 9. Workload harvest (D-HOLO-H14c-6)

No corpus of executed thought programs exists in the repository. H14c publishes
an instrument, `lane_footprint(&Program)` (the planes and lanes a mask-risc
program names), and a census over the programs constructed in the repository's
tests and examples, labelled **test-corpus observation, not workload**. No
workload frequency is invented.

## 10. Decision rules (pre-registered)

- **Independence (A, B of the acceptance criteria):** tables consulted == `k`
  for every `U` (exact, test). Timing: at `k = 2`, ns per read at `U = 31` is at
  most 1.25× that at `U = 2`, hot and cold, at every density. Falsified if any
  cell exceeds 1.25×.
- **Populations without values (C):** value loads == 0 for every population
  query (test); timing reported per 64-address word.
- **No materialization (D):** materialization counter 0 on A and B paths (test).
- **Identity (E):** every § 8 scenario returns the requested coordinates or
  refuses; none returns a wrong coordinate.
- **C vs B on query-shaped reads:** reported per cell as C/B; "C faster" where
  C/B <= 1.0. No default is decided from it.
- **Memory:** a C-lazy table whose resident size is at least 50 % of 1 MiB
  while its payload is under 10 % of 1 MiB is flagged page-dense waste; the
  cells where that happens are listed. No mitigation (M, density switching) is
  added in H14c.
- **Lance:** the § 7 prediction, both halves.

Mixed results return to the operator with the numbers. No universally optimal
layout is claimed.

## 11. Run record

Each run records: the tier the program prints, `rustc --version`, the resolved
`lance` version, `MemTotal`, iterations per batch, the command line, and the
git sha. Raw results go to `.claude/board/entries/` as a dated machine-readable
block (CSV inside the entry) and the console output to the session scratchpad.


## Results (v4, 2026-10-15)

**Run record.** `tier: v4 (avx512f)` printed by both probes (Intel Xeon @ 2.80 GHz,
4 vCPUs, avx512f/bw/cd/dq/vl/vnni; `MemTotal 16,479,424 kB`); `rustc 1.99.0
(b940084d7 2026-09-28)`; `lance 13.0.0` (resolved at build time and printed by the probe; the
repository tracks no lockfile). In-memory
probe: `cargo run --release -p lance-graph-mask-risc --example
holo_thought_first_probe` (sections A, B, C, thought and spatial axes) and
`… -- memory` (memory grid, one child process per cell); 7 batches of 8,192
reads, median. Lance probe: `cargo run --release --manifest-path
crates/holo-lance-masked-io-probe/Cargo.toml`, median of 5. All tests ran
before any timing (in-memory 11, Lance 5). The hot-path code of the timed run
equals the committed code; between the run and the commit only the memory
grid moved into child processes and one loop was rewritten for clippy.

**Limitation stated up front.** The VM exposes no hardware performance
counters (`perf stat` reports cycles, cache-misses and dTLB-load-misses as
"not supported"), so cache and TLB behaviour is measured indirectly: hot vs
cold timing (cold = a 128 MiB eviction pass before every batch), minor page
faults, resident size. No cache-miss or TLB-miss count is claimed.

### Acceptance criteria

| | criterion | result |
|---|---|---|
| A | operands accessed directly, no row-first discovery | **met** (oracles 1, 2, 5; disable runs "unrelated lane", "shift address", "drop presence test" red) |
| B | touched tables outside the selection do not set point-read work | **met**: tables consulted == k for U = 2, 8, 31 (exact); at k = 2 the read at U = 31 costs 0.90–1.01× the read at U = 2 hot and 0.95–1.16× cold, every density (rule: ≤ 1.25×) |
| C | populations without values | **met**: value loads 0 (counter); AND / XOR / OR + `Count` through mask-risc equal a scalar bit loop |
| D | no row materialization on point and population paths | **met**: row counter 0; the terminal builds exactly the rows asked (disable "row in read" red) |
| E | correct value across temporal change and reorganization | **met** for the checked path: every § 8 scenario returns the requested coordinates or refuses; disable "no version check" turns both identity tests red |

### Point projection (A)


**hot** — ns per point read, U = 31, uniform, one thought, fallback to base; cells are C / B / S

| dirty | k = 1 | k = 2 | k = 4 | k = 8 | k = 16 | k = 31 |
|---|---|---|---|---|---|---|
| 0 % | 10 / 7 / 8 | 16 / 10 / 15 | 29 / 16 / 27 | 58 / 35 / 56 | 108 / 34 / 104 | 215 / 62 / 202 |
| 0.1 % | 10 / 7 / 17 | 16 / 11 / 29 | 29 / 15 / 57 | 56 / 22 / 110 | 108 / 43 / 236 | 207 / 61 / 434 |
| 1 % | 10 / 9 / 24 | 16 / 12 / 43 | 29 / 17 / 82 | 56 / 29 / 160 | 112 / 50 / 345 | 272 / 130 / 823 |
| 10 % | 10 / 13 / 32 | 16 / 17 / 65 | 28 / 24 / 139 | 50 / 40 / 306 | 101 / 72 / 651 | 206 / 138 / 1531 |
| 50 % | 10 / 13 / 54 | 19 / 18 / 107 | 32 / 43 / 245 | 42 / 49 / 513 | 119 / 122 / 1690 | 339 / 223 / 5220 |
| 100 % | 15 / 21 / 86 | 10 / 18 / 110 | 15 / 28 / 236 | 32 / 43 / 595 | 211 / 120 / 2017 | 475 / 119 / 5518 |

**cold** — ns per point read, U = 31, uniform, one thought, fallback to base; cells are C / B / S

| dirty | k = 1 | k = 2 | k = 4 | k = 8 | k = 16 | k = 31 |
|---|---|---|---|---|---|---|
| 0 % | 105 / 106 / 94 | 161 / 120 / 130 | 157 / 149 / 181 | 298 / 212 / 298 | 462 / 321 / 440 | 689 / 471 / 698 |
| 0.1 % | 107 / 103 / 130 | 158 / 125 / 265 | 185 / 158 / 292 | 282 / 214 / 478 | 399 / 326 / 721 | 600 / 419 / 1115 |
| 1 % | 102 / 114 / 156 | 157 / 159 / 263 | 190 / 188 / 355 | 371 / 237 / 534 | 434 / 367 / 1015 | 718 / 498 / 1332 |
| 10 % | 106 / 156 / 249 | 153 / 261 / 310 | 194 / 241 / 411 | 309 / 393 / 716 | 544 / 559 / 1560 | 911 / 887 / 2860 |
| 50 % | 103 / 202 / 233 | 171 / 393 / 509 | 305 / 305 / 736 | 391 / 502 / 1544 | 580 / 715 / 5064 | 1142 / 982 / 7836 |
| 100 % | 54 / 200 / 189 | 73 / 287 / 374 | 131 / 273 / 715 | 224 / 435 / 2091 | 519 / 644 / 5219 | 1264 / 788 / 12994 |

**Independence (pre-registered rule)** — C at k = 2: ns per read at U = 2 / 8 / 31, and U = 31 ÷ U = 2

| dirty | hot | ratio | cold | ratio |
|---|---|---|---|---|
| 0 % | 16.2 / 16.2 / 16.4 | 1.01 | 164.0 / 151.8 / 161.1 | 0.98 |
| 0.1 % | 16.3 / 16.6 / 16.2 | 1.00 | 164.4 / 151.1 / 158.3 | 0.96 |
| 1 % | 16.5 / 16.2 / 16.2 | 0.98 | 165.0 / 154.3 / 157.3 | 0.95 |
| 10 % | 16.0 / 16.0 / 16.2 | 1.01 | 152.2 / 158.4 / 153.5 | 1.01 |
| 50 % | 19.6 / 18.8 / 19.1 | 0.98 | 147.7 / 144.4 / 171.2 | 1.16 |
| 100 % | 10.8 / 10.3 / 9.7 | 0.90 | 66.3 / 72.2 / 72.8 | 1.10 |


What remains necessary in C is one presence test per requested lane, in that
lane's own table, plus one load per present value (or one base load per
fallback). Its cost is linear in k and flat in U.

**C vs B on query-shaped reads (negative result, reported as measured).**
- Hot, at 0–1 % dirty, B is faster for every k (k = 2: C 16 ns, B 10–12;
  k = 31: C 207–272 ns, B 61–130). With fallback to base most reads are
  misses; B tests one row index for all k lanes, C tests k presence planes.
- Cold, C matches B up to 1 % dirty at k ≤ 4 and is faster from 10 % dirty at
  k ≤ 8 (k = 1 at 100 %: C 54 ns, B 200 ns). At k ≥ 16 B is faster in most
  cells.
- S (sparse, binary search) is the slowest arm everywhere except the empty
  table hot, and grows with density (k = 31, 100 %: 5.5 µs hot).
- Handle resolution, done once per program: 25–279 ns (one allocation per
  selection; not on the per-read path).

### Populations (B)

`popcount(Pa & Pb & A)` and `popcount(Pa ^ Pb)` run at 0.9–1.2 ns per 64-address
word (≈ 0.015 ns per address), independent of density and of U; one outlier at
0.39 ns. These numbers include allocating the mask-risc scratch on every call.
Session OR over T thoughts: 0.85–0.87 ns per word at T = 1, 4.2–5.1 at T = 8,
19–32 at T = 64.

### Terminal (C)

A full 512-byte row (base plus every overlay of the thought, U = 31): 96–105 ns
at ≤ 1 % dirty, 206 ns at 10 %, 613–614 ns at 50–100 %. Not part of A or B.

### Thought and spatial axes

- Point reads across T thoughts (k = 2, U = 31; "warm" = the same 8,192 random
  (thought, address) pairs repeated, not the main grid's 256-address hot set):
  warm 37–40 ns at T = 1, 40–45 ns at T = 8, 55–67 ns at T = 64; cold 153–170 ns
  at T = 1, 160–169 ns at T = 8, 192 ns (1 %) and 365 ns (10 %) at T = 64
  (T = 64 at 10 % is 2.4× T = 1).
- Spatial patterns (k = 2, U = 31, 1 % and 10 %): hot C 13–19 ns, B 11–19 ns,
  B ≤ C in 7 of 12 hot cells (all six at 1 %, disjoint at 10 %); cold C
  121–191 ns, B 135–251 ns, B ≤ C in 2 of 12 cold cells (disjoint 1 %: 136 vs
  189 ns; contiguous 10 %: 180 vs 191 ns). Disjoint populations give shared count 0 and Hamming
  = sum of populations; overlapping give Hamming 614 for 6,592-address planes.

### Memory and fragmentation (H14c-3)

| pattern | dirty | T | logical MiB | payload KiB | presence KiB | sparse persist KiB | C-lazy RSS MiB | C-lazy faults | C-lazy seal ns/dirty | C-eager RSS MiB | C-eager seal ns/dirty | S RSS MiB | lazy RSS per table KiB | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 0.1 % | 1 | 31 | 33 | 248 | 41 | 7.6 | 1933 | 2628 | 31.3 | 10487 | 0.09 | 251 |  |
| uniform | 0.1 % | 8 | 248 | 253 | 1984 | 317 | 58.9 | 15063 | 2641 | 250.0 | 11468 | 0.38 | 243 |  |
| uniform | 0.1 % | 64 | 1984 | 2034 | 15872 | 2543 | 470.1 | 120326 | 2671 | 1999.7 | 10290 | 2.68 | 243 |  |
| uniform | 1 % | 1 | 31 | 318 | 248 | 398 | 29.0 | 7406 | 979 | 31.3 | 1018 | 0.44 | 957 | page-dense waste |
| uniform | 1 % | 8 | 248 | 2538 | 1984 | 3172 | 231.0 | 59128 | 866 | 250.0 | 1213 | 3.16 | 954 | page-dense waste |
| uniform | 1 % | 64 | 1984 | 20333 | 15872 | 25417 | 1849.5 | 473455 | 986 | 1999.7 | 1067 | 25.02 | 955 | page-dense waste |
| uniform | 10 % | 1 | 31 | 3180 | 248 | 3975 | 31.3 | 8005 | 110 | 31.3 | 111 | 3.93 | 1034 |  |
| uniform | 10 % | 8 | 248 | 25412 | 1984 | 31766 | 250.1 | 64019 | 120 | 250.1 | 104 | 31.09 | 1033 |  |
| uniform | 10 % | 64 | 1984 | 203190 | 15872 | 253987 | 2000.5 | 512118 | 121 | 2000.5 | 117 | 248.23 | 1033 |  |
| uniform | 100 % | 1 | 31 | 31744 | 248 | 39680 | 31.4 | 8016 | 20 | 31.4 | 19 | 39.04 | 1036 |  |
| uniform | 100 % | 8 | 248 | 253952 | 1984 | 317440 | 250.9 | 64222 | 16 | 250.9 | 16 | 311.98 | 1036 |  |
| uniform | 100 % | 64 | 1984 | 2031616 | 15872 | 2539520 | 2007.3 | 513859 | 16 | 2007.3 | 16 | 2495.55 | 1036 |  |
| clustered | 0.1 % | 1 | 31 | 27 | 248 | 34 | 0.5 | 126 | 245 | 31.3 | 14931 | 0.08 | 18 |  |
| clustered | 0.1 % | 8 | 248 | 230 | 1984 | 288 | 4.1 | 1039 | 316 | 250.0 | 11755 | 0.34 | 17 |  |
| clustered | 0.1 % | 64 | 1984 | 1958 | 15872 | 2448 | 33.0 | 8444 | 231 | 1999.7 | 10421 | 2.57 | 17 |  |
| clustered | 1 % | 1 | 31 | 299 | 248 | 374 | 1.9 | 475 | 95 | 31.3 | 1141 | 0.41 | 63 |  |
| clustered | 1 % | 8 | 248 | 2489 | 1984 | 3111 | 15.0 | 3822 | 74 | 250.0 | 1089 | 3.10 | 62 |  |
| clustered | 1 % | 64 | 1984 | 20387 | 15872 | 25484 | 120.3 | 30783 | 85 | 1999.7 | 1054 | 25.08 | 62 |  |
| clustered | 10 % | 1 | 31 | 3169 | 248 | 3961 | 13.1 | 3331 | 73 | 31.3 | 140 | 3.92 | 431 |  |
| clustered | 10 % | 8 | 248 | 25206 | 1984 | 31508 | 104.0 | 26608 | 50 | 250.1 | 106 | 30.84 | 429 |  |
| clustered | 10 % | 64 | 1984 | 202872 | 15872 | 253590 | 831.6 | 212869 | 53 | 2000.4 | 108 | 247.86 | 429 |  |
| clustered | 100 % | 1 | 31 | 31744 | 248 | 39680 | 31.4 | 8016 | 19 | 31.4 | 15 | 39.04 | 1036 |  |
| clustered | 100 % | 8 | 248 | 253952 | 1984 | 317440 | 250.9 | 64222 | 17 | 250.9 | 16 | 311.98 | 1036 |  |
| clustered | 100 % | 64 | 1984 | 2031616 | 15872 | 2539520 | 2007.3 | 513859 | 15 | 2007.3 | 17 | 2495.55 | 1036 |  |

Flag rule (pre-registered): C-lazy resident ≥ 50 % of 1 MiB per table while
payload < 10 % of 1 MiB. Flagged: **uniform 1 % at every T** (≈ 955 KiB
resident for ≈ 10 KiB of values per table, ≈ 93×). Uniform 10 % is resident
at 1 MiB per table too, but its payload (≈ 102 KiB) sits at the 10 % line.

- Resident size follows pages touched, not values: a 1 MiB table holds 256
  pages of 256 values; uniform scatter touches nearly every page from about
  1 % dirty, clustered runs do not (62 KiB per table at 1 %).
- C-eager commits the full logical size at every density (≈ 2 GiB at T = 64).
- S holds only the dirty values and addresses: 25 MiB vs 1,850 MiB (C-lazy) at
  uniform 1 %, T = 64; its price is the binary search in § Point projection.
- Seal per dirty value (C-lazy) is page-fault bound at low density: 2.6 µs at
  uniform 0.1 %, 0.9–1.0 µs at 1 %, 0.11–0.12 µs at 10 %, 16–20 ns at 100 %.
- Reserved virtual memory equals the logical size for every arm that maps the
  tables (the virtual-delta column in the raw data).

### Lance 13 masked I/O (H14c-4)

Heuristic materialization; the style changed `bytes_read` in **0 of 120 cells**.
Lance's scan statistics expose `bytes_read`, `iops`, `requests` and the counts
`rows_scanned`, `ranges_scanned`, `fragments_scanned`; bytes decoded, pages
accessed and allocations are not exposed and are not claimed. `bytes_read` is
what Lance requested from the local object store (served from the page
cache), not a physical disk count.

| pattern | dirty | k | style | unmasked bytes | independent bytes (ratio) | rows scanned indep / unmasked | combined rows | combined bytes (ratio) | unmasked µs | combined µs | independent µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 0.1 % | 1 | heuristic | 127296 | 97424 (0.77) | 46 / 65536 | 46 | 97424 (0.77) | 24016 | 1544 | 1252 |
| uniform | 0.1 % | 2 | heuristic | 131560 | 202952 (1.54) | 112 / 65536 | 1 | 6536 (0.05) | 21806 | 1363 | 3203 |
| uniform | 0.1 % | 4 | heuristic | 140808 | 419424 (2.98) | 245 / 65536 | 0 | 0 (0.00) | 30442 | 406 | 6820 |
| uniform | 0.1 % | 8 | heuristic | 159736 | 849024 (5.32) | 531 / 65536 | 0 | 0 (0.00) | 42240 | 491 | 12051 |
| uniform | 1 % | 1 | heuristic | 136160 | 136160 (1.00) | 692 / 65536 | 692 | 136160 (1.00) | 18135 | 2640 | 2579 |
| uniform | 1 % | 2 | heuristic | 147720 | 271752 (1.84) | 1354 / 65536 | 9 | 30920 (0.21) | 19547 | 1531 | 4534 |
| uniform | 1 % | 4 | heuristic | 170624 | 542720 (3.18) | 2655 / 65536 | 0 | 0 (0.00) | 24855 | 398 | 9633 |
| uniform | 1 % | 8 | heuristic | 217040 | 1085264 (5.00) | 5304 / 65536 | 0 | 0 (0.00) | 30113 | 524 | 16802 |
| uniform | 10 % | 1 | heuristic | 195416 | 195416 (1.00) | 6606 / 65536 | 6606 | 195416 (1.00) | 18726 | 8030 | 8362 |
| uniform | 10 % | 2 | heuristic | 266584 | 390616 (1.47) | 13194 / 65536 | 679 | 266584 (1.00) | 21610 | 3642 | 16969 |
| uniform | 10 % | 4 | heuristic | 409560 | 781656 (1.91) | 26435 / 65536 | 8 | 143320 (0.35) | 24118 | 2791 | 34299 |
| uniform | 10 % | 8 | heuristic | 692016 | 1560240 (2.25) | 52513 / 65536 | 0 | 0 (0.00) | 46024 | 542 | 60455 |
| uniform | 50 % | 1 | heuristic | 1208448 | 1208448 (1.00) | 33120 / 65536 | 33120 | 1208448 (1.00) | 18422 | 19593 | 20913 |
| uniform | 50 % | 2 | heuristic | 2292864 | 2415872 (1.05) | 65891 / 65536 | 16591 | 2292864 (1.00) | 29535 | 19743 | 42360 |
| uniform | 50 % | 4 | heuristic | 2541824 | 2912640 (1.15) | 131294 / 65536 | 4074 | 2541824 (1.00) | 27553 | 12331 | 78100 |
| uniform | 50 % | 8 | heuristic | 4958336 | 5824256 (1.17) | 261893 / 65536 | 247 | 3195520 (0.64) | 38960 | 11204 | 183889 |
| uniform | 100 % | 1 | heuristic | 1175680 | 1175680 (1.00) | 65536 / 65536 | 65536 | 1175680 (1.00) | 26148 | 24902 | 23463 |
| uniform | 100 % | 2 | heuristic | 2227328 | 2350336 (1.06) | 131072 / 65536 | 65536 | 2227328 (1.00) | 20804 | 26007 | 59764 |
| uniform | 100 % | 4 | heuristic | 4330624 | 4699648 (1.09) | 262144 / 65536 | 65536 | 4330624 (1.00) | 27932 | 31766 | 99231 |
| uniform | 100 % | 8 | heuristic | 8535168 | 9398272 (1.10) | 524288 / 65536 | 65536 | 8535168 (1.00) | 35396 | 35899 | 198529 |
| clustered | 0.1 % | 1 | heuristic | 127744 | 3000 (0.02) | 64 / 65536 | 64 | 3000 (0.02) | 20594 | 1426 | 1120 |
| clustered | 0.1 % | 2 | heuristic | 132552 | 6008 (0.05) | 128 / 65536 | 0 | 0 (0.00) | 21548 | 411 | 2328 |
| clustered | 0.1 % | 4 | heuristic | 143048 | 14184 (0.10) | 320 / 65536 | 0 | 0 (0.00) | 25836 | 399 | 4418 |
| clustered | 0.1 % | 8 | heuristic | 153648 | 28664 (0.19) | 640 / 65536 | 0 | 0 (0.00) | 30906 | 577 | 9662 |
| clustered | 1 % | 1 | heuristic | 130448 | 31504 (0.24) | 704 / 65536 | 704 | 31504 (0.24) | 25186 | 2180 | 1864 |
| clustered | 1 % | 2 | heuristic | 135832 | 50136 (0.37) | 1216 / 65536 | 0 | 0 (0.00) | 20052 | 394 | 3348 |
| clustered | 1 % | 4 | heuristic | 145968 | 91200 (0.62) | 2112 / 65536 | 0 | 0 (0.00) | 28818 | 411 | 7870 |
| clustered | 1 % | 8 | heuristic | 169416 | 214248 (1.26) | 4544 / 65536 | 0 | 0 (0.00) | 38129 | 846 | 14683 |
| clustered | 10 % | 1 | heuristic | 159928 | 153736 (0.96) | 6528 / 65536 | 6528 | 153736 (0.96) | 27283 | 4294 | 3760 |
| clustered | 10 % | 2 | heuristic | 194472 | 308184 (1.58) | 12800 / 65536 | 832 | 80512 (0.41) | 20208 | 2322 | 8682 |
| clustered | 10 % | 4 | heuristic | 263544 | 619256 (2.35) | 25344 / 65536 | 0 | 0 (0.00) | 24153 | 667 | 21098 |
| clustered | 10 % | 8 | heuristic | 406864 | 1250448 (3.07) | 51456 / 65536 | 0 | 0 (0.00) | 30902 | 570 | 32887 |
| clustered | 50 % | 1 | heuristic | 1180776 | 1086016 (0.92) | 32512 / 65536 | 32512 | 1086016 (0.92) | 17634 | 13254 | 14354 |
| clustered | 50 % | 2 | heuristic | 1299824 | 1328816 (1.02) | 64640 / 65536 | 15232 | 915328 (0.70) | 19273 | 7627 | 23061 |
| clustered | 50 % | 4 | heuristic | 2474528 | 2685664 (1.09) | 128896 / 65536 | 3328 | 739232 (0.30) | 24328 | 6264 | 49197 |
| clustered | 50 % | 8 | heuristic | 6701944 | 7133064 (1.06) | 258368 / 65536 | 192 | 92312 (0.01) | 30654 | 2263 | 101602 |
| clustered | 100 % | 1 | heuristic | 1175680 | 1175680 (1.00) | 65536 / 65536 | 65536 | 1175680 (1.00) | 18409 | 21556 | 23025 |
| clustered | 100 % | 2 | heuristic | 2227328 | 2350336 (1.06) | 131072 / 65536 | 65536 | 2227328 (1.00) | 19125 | 22928 | 42014 |
| clustered | 100 % | 4 | heuristic | 4330624 | 4699648 (1.09) | 262144 / 65536 | 65536 | 4330624 (1.00) | 24521 | 30269 | 87551 |
| clustered | 100 % | 8 | heuristic | 8535168 | 9398272 (1.10) | 524288 / 65536 | 65536 | 8535168 (1.00) | 32631 | 35082 | 179174 |

**Prediction: both halves hold.** At 1 % uniform, k = 1: masked bytes = unmasked
bytes (136,160; ratio 1.00) while rows scanned fall to 692 of 65,536. At 1 %
clustered, k = 1: ratio 0.24 (≤ 0.5). At 0.1 % uniform the ratio is 0.77.

- A scattered mask cuts rows decoded, not bytes fetched; a clustered mask cuts
  both. Lance reads in pages, so the mask helps exactly when presence is
  page-local.
- Independent per-lane masked reads (one scan per lane, each re-reading
  `coord`) fetch more bytes than one unmasked projection whenever k ≥ 2 at
  uniform low density (1.5–5.3×).
- The combined mask (every requested lane present) is the joint population,
  ≈ dᵏ for independent lanes: one row at 0.1 %, k = 2.
- Wall time (heuristic style; times from the re-run after review, see below):
  combined masked reads at ≤ 1 % density take 0.39–2.6 ms against 17.6–46 ms
  unmasked. At 100 %, k = 1, the two are within run-to-run noise (uniform
  24.9 vs 26.1 ms, clustered 21.6 vs 18.4 ms); no ordering is claimed there.
- Re-run after review (#1573): the first run built each lane's mask inside
  the timed loop for the independent strategy, while the combined strategy
  built its mask before the timer. The masks are now built before the timer
  for both. All three timing columns above come from the re-run; bytes, IOPS,
  requests and rows scanned were identical in both runs (360 of 360 rows).
  Timings moved by up to about 40 % between the two runs even for the
  unmasked baseline, which the defect did not touch, so differences below
  that size are not read as results.

### Coordinate identity (H14c-5)

The checked read refused every stale map. Without the check, a stale map
returned (k = 1, 241 requested coordinates):

| rewrite | stable row ids | returned | wrong coord | silently missing |
|---|---|---|---|---|
| overwrite, reversed | off | 0 | 0 | 241 |
| delete + compaction | off | 0 | 0 | 160 of 160 live |
| update (epoch) | off | 123 | 0 | 118 |
| overwrite, reversed | on | 0 | 0 | 241 |
| delete + compaction | on | 160 | 0 | 0 |
| update (epoch) | on | 241 | 0 | 0 |

- The hazard observed is **silent omission**, not a wrong coordinate; a wrong
  coordinate is also excluded structurally because values are attributed by
  the stored `coord`, never by output position (oracle 9).
- Without stable row ids, compaction changed 3,510 of 3,510 `_rowid`s and an
  update moved 85 of 85 updated rows; with stable row ids, 0 and 0. An
  overwrite breaks the mapping either way.
- `VersionClock` (#1567) already states that a reordering rewrite needs a new
  clock; the same boundary holds for row identity.

### Workload harvest (H14c-6) — test-corpus observation, not workload

`lane_footprint(&Program)` reads the planes and lanes a program names. A
read-only census (Haiku, spot-checked) of the 169 `Program::new` constructions
in mask-risc tests and `lance-graph-quack`: 61 countable literally, 108 built
dynamically. Distinct value lanes per literal program: 0 → 26, 1 → 31, 2 → 4.
No corpus of executed thought programs exists; no workload frequency is
claimed.

### What this answers

Given thought, operands and address, the operations still necessary are: one
presence test per requested lane (in that lane's table), one load per present
value or base fallback, and, for populations, one word operation per 64
addresses per plane. Row assembly, trie or rank traversal, and row-first lane
discovery disappear; the touched tables outside the selection cost nothing.
The cost that remains open is physical: dense positional tables are resident at
about one page per touched page of addresses, which at uniform 1 % is ≈ 93× the
payload, and Lance fetches whole pages for scattered masks.

**OPEN (returned to the operator, no layout decided):** how to bound the
resident cost of positional tables without reintroducing traversal; whether to
enable stable row ids for lane datasets; whether lane data that is read by
scattered masks should be laid out so presence is page-local.
