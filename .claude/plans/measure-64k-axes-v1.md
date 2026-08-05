# measure-64k-axes v1 — the corrected five-axis benchmark (operator-specified, 2026-08-05)

> **Provenance:** operator-specified measurement plan, recorded before build.
> The prior measurements (the 64k ignition probe's wall times, D-BLW-4's 3.27×,
> the 24 GiB/32 MiB memory figures) mixed FIVE independent axes; this plan
> varies exactly one at a time. Prior probes stand as CORRECTNESS results —
> none of their timing/memory lines were claims — but every future
> performance/memory number comes from THIS arc.

## The five axes (never blend two in one number again)

| axis | meaning |
|---|---|
| Logical owner count | 65,536 independent owner identities |
| Physical SoA layout | 65,536 single-owner objects vs 64 chunks × 1,024 rows |
| WAL segment size | 1/2/4/8/32 MiB slices of ONE 32 MiB frame |
| Temporal reconstruction | post-WAL causal grouping + epistemic projection |
| Execution concurrency | overlapping thought bodies before deterministic convergence |

## Ground truth (binding on every arm)

- Logical population: **65,536 owner identities**.
- Canonical row: **512 bytes**; canonical frame: 65,536 × 512 B = **32 MiB**.
- Temporal: write-side ordering stays in persist_sink/freeze; `temporal.rs`
  runs ONLY after the sealed WAL read — layer 1 = `local_trajectories`,
  layer 2 = `deinterlace`. Never inserted into WAL preparation (that would
  double-sort and move query-time work onto the write path).
- Durability: **one logical cycle → one `commit_cycle` → one fdatasync →
  one `DatasetVersion`.** Segments are internal I/O slices of ONE commit,
  never version-publishing units.
- **Ownership vocabulary rule:** ownership is a TYPE/BORROW property, never a
  runtime operation — no arm is ever described as "claiming ownership".
- **Memory vocabulary rule:** the hot `MailboxSoA` representation and the
  canonical `NodeRow512` representation NEVER share a memory claim. "32 MiB"
  belongs to the canonical frame alone unless size/RSS proves otherwise.

## Arms

### B0 — DummyOwner cast baseline
65,536 lightweight owners carrying only `owner_id`/`phase`/`cycle` — no SoA
rows, no temporal pass, no file I/O. Measure: scan → fixed dummy thought →
`emit_bootstrap_intent` → `BatchWriter` staging → `collect_casts` → freeze.
This is the modern form of the #879 fake-owner control: owner lookup,
write-on-behalf rebind, CastId allocation, staging, collect.

### B1 — materialise and drive 65,536 owner-exclusive SoAs
Measure separately: construction · registration into fleet · scan · thought ·
cast · collect/freeze · apply. Two representations, memory NEVER blended:
- **B1a** — the current hot runtime owner: 65,536 × `MailboxSoA<4>` (actual
  implementation incl. identity planes + object overhead).
- **B1b** — the canonical 512-byte row owner: 65,536 × `NodeRow512` = 32 MiB
  (the persisted/storage envelope).

Derived metrics (the point of the split):
- **runtime ownership tax** = B1a cast/scan/freeze time − B0 same-phase time.
- **hot representation overhead** = B1a peak RSS − B1b peak RSS.

### WAL curve — W0-current vs W1-contiguous
One contiguous 32 MiB canonical frame for the physics measurement
(**W1-contiguous** = the storage/cache ceiling); the actual
`SweepSlot`/`DetachedCycleBatch` representation as **W0-current** (today's
implementation, with its allocator/pointer/BTreeMap/clone costs named as what
it measures). Segment table (MiB, not KiB):

| rows/segment | bytes/segment | segments/64k cycle |
|---|---|---|
| 2,048 | 1 MiB | 32 |
| 4,096 | 2 MiB | 16 |
| 8,192 | 4 MiB | 8 |
| 16,384 | 8 MiB | 4 |
| 65,536 | 32 MiB | 1 |

Per configuration: **2 unreported warm-up cycles + 16 consecutive measured
full 64k cycles** = exactly 16 × 32 MiB = **512 MiB logical payload**
(constant total work across configurations). Implementation:
`write_vectored` over the N slices then **exactly one fdatasync**
(`File::sync_data`); record the ACTUAL syscall count (partial vectored
writes loop and are counted). One `DatasetVersion` per full cycle — a
sync-every-segment variant is permitted ONLY as an explicitly labelled
durability-tax anti-pattern control, never described as the cycle contract.
**One release-mode measurement binary (`measure_wal_curve`), never 16 Rust
tests** — the test runner would overlap/reorder and contaminate cache
measurements.

### T — temporal phases, post-WAL only
After the 16 committed cycles the history holds **65,536 owners × 16
landings = 1,048,576 temporal rows** (every owner a real 16-step trajectory,
not a singleton). Measure separately: **T0** `scan_sealed` read · **T1**
`temporal::local_trajectories` · **T2** `temporal::deinterlace`. The
benchmark row implements BOTH `LocalCausalRow` and `DeinterlaceRow` with
`owner` = logical owner id, `cast_seq` = cycle number / monotonic stream
position, `lance_version` = sealed `DatasetVersion`.

### L1 — ChunkedSoA<1024>[64] physical-layout control
64 physical chunks × 1,024 rows = 65,536 logical rows; 512 KiB canonical
payload per chunk; 32 MiB total. **A physical chunk is NOT an owner**:
- **L1a** — 65,536 logical owners, one per row
  (`owner = chunk_index × 1024 + lane`); temporal groups by logical owner id,
  never chunk id; mutation exclusivity preserved by disjoint one-row
  `OwnerRowMut` views. The valid layout comparison.
- **L1b** — 64 chunks treated as 64 owners × 1,024 events: a topology
  control ONLY, never evidence for the 64k-owner model.

### EXP-KIA-A2-64K — exploratory concurrency (non-claiming)
**D-KIA-A2 is untouched** — it stays the canonical claim gate (median-of-5,
≥2×). This experiment runs now, without those thresholds, and CANNOT mark
A2 passed. Shape: 65,536 logical thought bodies → bounded worker pool
(1/2/4/8/16/physical cores; `std::thread::scope` with disjoint ranges — no
rayon dep) → **thread-local `PreparedIntent` buffers** → join → the existing
owner rebind + `BatchWriter` staging at the deterministic convergence
boundary → one seal → one WAL commit. **Never a mutex around one shared
`BatchWriter` in the compute phase** (that benchmarks lock contention).
Witness asserts: exactly 65,536 bodies executed · `max_active_workers ≥ 2`
on parallel runs · **sequential and parallel result digests identical** ·
all owner bindings preserved · one sealed cycle · one WAL commit · 65,536
applied transitions. Proves real overlap; claims nothing about A2's
threshold.

## Measurement schema

One CSV row per measured cycle:
`owner_shape, physical_layout, threads, segment_rows, segment_bytes,
segments_per_cycle, repeat, build_ns, scan_ns, think_ns, rebind_cast_ns,
collect_ns, freeze_ns, wal_write_ns, wal_sync_ns, temporal_layer1_ns,
temporal_layer2_ns, apply_ns, total_ns, logical_rows, logical_bytes,
sealed_transitions, applied_transitions, wal_syscalls, fsync_calls,
dataset_versions, peak_rss_bytes, minor_faults, major_faults,
context_switches, llc_misses, max_active_workers, result_digest`

- RSS/faults/context switches from `/proc/self/status` + `/proc/self/stat`
  (std-only). `llc_misses` left EMPTY with a stated reason unless
  perf-counter access exists — an empty cell, never a fabricated one.
- Report per configuration: median · p95 · first measured cycle · last
  measured cycle · rows/s · MiB/s · ns/owner.
- Cache-amortisation curve: `gain(C) = throughput(C)/throughput(prev) − 1`;
  descriptive plateau marker = first chunk size where two consecutive
  doublings improve median throughput by < 5 %. **The plateau is a measured
  knee, never a PASS/KILL.**

## Placement + gates

`crates/lance-graph-supervisor/examples/measure_wal_curve.rs` (the supervisor
sees `run_cycle`/`persist_sink` AND the planner's `temporal.rs`), feature
`cycle-driver`, run `--release`. No new dependencies (std threads; no rayon,
no libc). Central gates: fmt · clippy (0 attributable) · one full release
run producing the CSV + the four answers:

1. What does ownership cost? (B1a − B0, per phase)
2. What does physical layout cost? (B1a vs L1a at equal logical population)
3. Where does WAL amortisation plateau? (the knee on the W1 curve, W0 beside it)
4. What does genuine parallel thought execution add before the deterministic
   seal? (EXP-KIA-A2-64K, digest-identical)
