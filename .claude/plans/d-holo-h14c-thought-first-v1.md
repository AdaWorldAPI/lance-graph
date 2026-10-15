# D-HOLO-H14c — thought-first positional reads, Hamming presence and Lance 13 masked I/O (v1)

**Status:** PRE-REGISTERED 2026-10-15; nothing measured. Results will be added
as a dated section below; this pre-registration is not edited after the first
run.

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

- [ ] Pre-registration committed before any benchmark (this file).
- [ ] D-HOLO-H14c-1: typed thought-first read (`Selection` → resolved handles →
      `read(i)`), with counters; oracle tests; disable runs.
- [ ] D-HOLO-H14c-2: presence populations through mask-risc programs (AND, OR,
      XOR, `Count`); scalar reference; disable runs.
- [ ] D-HOLO-H14c-3: memory and fragmentation measures (lazy, eager, sparse);
      perf counters where the kernel allows.
- [ ] D-HOLO-H14c-4: Lance 13 masked I/O probe (combined mask + projection;
      independent per-lane masks), with `ExecutionSummaryCounts`.
- [ ] D-HOLO-H14c-5: coordinate identity across append, delete, update,
      reopen, compaction and version change; stale mapping refused.
- [ ] D-HOLO-H14c-6: lane-selection harvest over the repository's existing
      programs, labelled as test-corpus observation.
- [ ] v4 runs with the tier printed; raw results machine-readable.
- [ ] Results section, board entry, STATUS_BOARD, AGENT_LOG, draft PR.

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
