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
