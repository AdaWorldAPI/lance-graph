# onebrc-probe

A standalone, workspace-**excluded** crate (see root `Cargo.toml` `exclude`,
same precedent as `crates/bgz17` / `crates/deepnsm`) that measures the V3
substrate's throughput on the classic **1BRC** (One Billion Row Challenge)
groupby-aggregate workload, at **container scale (100M rows)**.

Verify standalone:

```bash
cargo test --manifest-path crates/onebrc-probe/Cargo.toml
```

Zero external dependencies for lanes A/C (std only) — see `Cargo.toml`.

---

## §1 — Reference inventory

`automataIA/1brc-rs` (fetched via
`https://raw.githubusercontent.com/automataIA/1brc-rs/main/...`, per the
worker-agent environment preamble — `api.github.com` / `codeload` /
`github.com` HTML are session-denied for unscoped repos, `raw.githubusercontent.com`
is not) is a **reference to study, never a dependency**. Every technique
below is **reimplemented** in this crate's own words, not vendored.

### What the reference does

| Technique | Reference location | Reimplemented here? |
|---|---|---|
| `memmap2::Mmap` zero-copy file access ("the only path to break the 2-second barrier": avoids ~13 GB explicit allocation vs `fs::read`) | `README.md` § Memory Access | **No** — see "mmap note" below |
| `hashbrown::HashMap::raw_entry_mut` + precomputed `FxHash`, inlined on the hot loop | `README.md` § Hashing Strategy (`v15_raw_hash`) | **No** — `lane_a_scalar` uses `std::collections::BTreeMap` |
| merykitty SWAR parser: reads 8 bytes as `u64`, finds `.` via `(!w & 0x10101010).trailing_zeros()`, branchlessly selects `X.Y` vs `XY.Z` layout | `README.md` § Temperature Parsing | **No** — `parse_temp_tenths` in `lib.rs` is a plain byte-scan integer parser (still float-free, just not branchless/SWAR) |
| `chunk_bounds(data, n) -> Vec<(usize, usize)>`, newline-aligned parallel work distribution | `src/lib.rs` (per repo tree) | **Yes** — `chunk_bounds` in this crate's `lib.rs`, same signature shape, own implementation |
| Threading: `std::thread::scope` (early, v3) then `rayon` work-stealing (later variants) | `README.md` § Threading Model | **Partially** — `lane_c_threads` uses `std::thread::scope` (the v3-era approach); no `rayon` (would break the zero-dep contract for lanes A/C) |
| Safe SIMD: `pulp::Arch::dispatch` (v11) / `wide::u8x32::cmp_eq` (v12) to find semicolons; manual AVX-512 rejected (loses on Zen 4 — AMD implements AVX-512 with 256-bit execution units) | `README.md` § SIMD Techniques | **No** — deferred to **Lane B** (see below); this workspace's SIMD rule is "all SIMD from `ndarray::simd`" (`simd-savant` agent), so Lane B routes through `ndarray::simd`, not `pulp`/`wide` |
| `Stats { min: i32, max: i32, sum: i64, count: u32 }` + `merge` | `src/lib.rs` (per repo tree) | **Yes**, same field shape — `Stats` in this crate's `lib.rs`, own implementation, with a doc-comment tying `merge` to this workspace's borrow-strategy rule |
| LCG-seeded generator (`0xDEADBEEFCAFEBABE`), fixed 413-station list sourced from the original Java 1BRC, Gaussian temps via Box-Muller, clamped to `[-99.9, 99.9]` | `src/bin/gen.rs` | **No, deliberately different** — this crate's `gen()` uses SplitMix64 (not an LCG) and procedurally INVENTS ~400 station names from synthetic syllables (no external Java-1BRC city list), because the archival-recipe contract (§2 below) wants the corpus reconstructible from `(rows, seed)` with **zero external dataset dependency** |
| `tests/equivalence.rs` — byte-for-byte output validation across variants | `tests/` | **Yes, in spirit** — `lane_a_and_lane_c_agree_on_generated_corpus` in `src/lib.rs` |

Repository layout (from the GitHub tree page, `main` branch):
`Cargo.toml`, `.cargo/config.toml` (`target-cpu=native`), `src/lib.rs`
(`Stats`, `chunk_bounds`, parser, formatter), `src/hash.rs` (custom hash
table), `src/bin/{gen.rs, v0_naive.rs .. v15_raw_hash.rs}` (16 progressive
solver variants), `tests/equivalence.rs`, `scripts/{gen_data,run_all,build_pgo}.sh`,
`benches/bench_variants.rs`, `data/` (gitignored generated corpora).

### mmap note

This crate deliberately uses `std::fs::read` (a single owned `Vec<u8>`),
**not** `mmap`, in `main.rs`'s `run` subcommand. The reference's own
documentation states mmap is "the only path" past the 2-second barrier at
the full 1B-row scale — that tradeoff is real and left for a follow-up
(Lane B or later) that's allowed to add a dependency (`memmap2` has no
AdaWorldAPI fork requirement bearing on it, per `CLAUDE.md`'s fork policy,
but adding ANY dependency changes this crate's "zero deps for lanes A/C"
contract, so it's out of scope for this brief).

---

## §2 — Archival convention

Every generated corpus travels with its **recipe**: `(rows, seed)` fully
determine the corpus bytes, and `gen()` streams a SHA-256 digest while
writing so the recipe line —

```text
rows=<N> seed=<S> sha256=<hash>
```

— is printed without a second read pass. Reproduce any measurement by
regenerating with the same `(rows, seed)` and diffing the printed
`sha256=` line.

---

## §3 — Lanes

| Lane | What it measures | Status |
|---|---|---|
| **A** — `lane_a_scalar` | Single-thread scalar baseline: one pass, byte-wise `;`/`\n` scan, integer temp parse, `BTreeMap<String, Stats>` accumulation | **Shipped** |
| **C** — `lane_c_threads` | `std::thread` parallel baseline: newline-aligned `chunk_bounds` split, per-worker owned `BTreeMap`, commutative `Stats::merge` combine | **Shipped** |
| **B** — `lane_b::lane_b_simd` (feature `lane-b`) | Vectorized `;`/`\n` scanning via `ndarray::simd::U8x32::cmpeq_mask` (per the workspace's SIMD rule — never raw `pulp`/`wide`/hand intrinsics in a consumer crate; see `.claude/knowledge/ndarray-vertical-simd-alien-magic.md`), 32-byte-stride scan with cross-block `line_start`/`pending_semi` carry, scalar temp parse (SWAR/branchless parse deliberately still deferred). | **Shipped** |
| **D** — `lane_d::lane_d_ractor` (feature `lane-d`) | Same groupby-aggregate workload as Lane C, but each `chunk_bounds` chunk is aggregated by a stateless `ractor` actor (actor-per-worker, ask-pattern reply, `lance-graph-supervisor`-style `Actor`/`RpcReplyPort` shape) instead of a bare `std::thread::scope` closure — identical chunking + commutative merge, only the worker primitive changes. | **Shipped** |
| **E** — `lane_e::lane_e_kanban` (feature `lane-e`) | One kanban card per batch: the corpus splits into `batches` newline-aligned chunks (`batches >= workers`) pulled from a shared `AtomicUsize` queue by `workers` puller tasks; every batch is journaled by a fresh `lance-graph-supervisor::KanbanActor<ProbeBoard>` driven through the full Rubicon forward arc (`Planning->CognitiveWork->Evaluation->Commit`, `drive_version_tick` × 3) around the actual `lane_a_scalar` work. The combined journal is asserted to carry exactly `3 * batches` legal `KanbanMove`s. `batches == workers` vs Lane D isolates the kanban journaling cost (identical chunking + actor-model tax, only the actor type differs); fine-grained batching (`batches >> workers`) prices per-card scheduling overhead (feeds W2d, the 550 ms Libet budget question). | **Shipped** |

---

## §4 — CLI

```text
onebrc-probe gen <path> <rows> <seed>
onebrc-probe run <path> <lane:a|b|c|d|e> [workers] [batches]
```

Lane `b` requires `--features lane-b`; lane `d` requires `--features lane-d`;
lane `e` requires `--features lane-e` (see `Cargo.toml` `[features]`). Lanes
A/C stay dependency-free either way. `batches` is **lane-`e`-only** (ignored
by every other lane): the number of newline-aligned batches the corpus
splits into, each journaled as one kanban card (`batches >= workers`,
default `workers * 16`):

```bash
cargo run --release --manifest-path crates/onebrc-probe/Cargo.toml \
  --features lane-b -- run /tmp/onebrc_10m.txt b
cargo run --release --manifest-path crates/onebrc-probe/Cargo.toml \
  --features lane-d -- run /tmp/onebrc_10m.txt d 4
cargo run --release --manifest-path crates/onebrc-probe/Cargo.toml \
  --features lane-e -- run /tmp/onebrc_10m.txt e 4 4
cargo run --release --manifest-path crates/onebrc-probe/Cargo.toml \
  --features lane-e -- run /tmp/onebrc_10m.txt e 4 64
```

`run` prints:

```text
lane=<X> rows=<N> workers=<W> elapsed_ms=<T> throughput_mrows_s=<R>
-- first 3 stations --
  ...
-- last 3 stations --
  ...
```

For lane `e` the line carries an extra `batches=<B>` field between
`workers=<W>` and `elapsed_ms=<T>` — every other lane's line is unchanged.

The first/last-3-stations dump (map is a `BTreeMap`, so this is
sorted-by-name order) is the correctness spot-check surface — a cheap
sanity signal that the aggregate isn't obviously garbage without diffing
the full ~400-station map.

---

## §5 — t0 baselines (10M rows, container)

Smoke-scale measurement (10M rows, not the full 100M-row container-scale
target — disk-cheap: ~140 MB, generated to `/tmp` and deleted after).
Machine: `nproc` = 4 (container).

Commands run (release build):

```bash
cargo run --release --manifest-path crates/onebrc-probe/Cargo.toml -- \
  gen /tmp/onebrc_10m.txt 10000000 42
cargo run --release --manifest-path crates/onebrc-probe/Cargo.toml -- \
  run /tmp/onebrc_10m.txt a
cargo run --release --manifest-path crates/onebrc-probe/Cargo.toml -- \
  run /tmp/onebrc_10m.txt c 4
rm /tmp/onebrc_10m.txt
```

Recipe line (corpus is reconstructible from this alone):

```text
rows=10000000 seed=42 sha256=f1853caa30a765883aa655be1c304d956ad8b03e19b3557df2af431d9a955691
```

File size: 142 MB (10,000,000 rows × `station;temp\n`).

| Lane | workers | elapsed_ms | throughput (Mrows/s) | speedup vs Lane A |
|---|---|---|---|---|
| A (scalar) | 1 | 1405.850 | 7.113 | 1.0x |
| C (threads) | 4 | 378.705 | 26.406 | 3.71x |

Both lanes agreed on every station's `Stats` (spot-checked via the
first/last-3-stations dump — identical `min`/`max`/`sum`/`count` for
`Belgoryoltuv`, `Belhumo`, `Belhuzephri`, `Zephven`, `Zephtuvhuhu`,
`Zephsaeshikra` across both runs), consistent with the
`lane_a_and_lane_c_agree_on_generated_corpus` unit test.

3.71x on 4 cores (not a clean 4.0x) is expected at this scale: Lane C pays
`std::fs::read`'s single-threaded I/O + the newline-scan in `chunk_bounds`
before any worker starts, and `BTreeMap`'s per-insert `O(log n)` cost means
the aggregation itself isn't perfectly parallel-friendly (~400 stations
keeps the tree shallow, but each of the 4 workers still walks its own tree
independently rather than sharing one flat hash table). Lane B (SIMD
scan/parse) and a hash-map swap are the natural next levers — deferred to
the orchestrator's follow-up per §3.

### §5.1 — t1 (lanes B/D) — measured 2026-07-02

Same recipe corpus (`rows=10000000 seed=42 sha256=f1853caa…5691`, hash
re-verified byte-identical at regeneration), same 4-core container,
release build `--features lane-b,lane-d` (probe pinned to
`target-cpu=x86-64-v3` via `.cargo/config.toml`, so `U8x32` ops are real
AVX2 intrinsics). Two passes per lane, best-of-2 reported (both passes
listed for honesty):

| Lane | workers | elapsed_ms (best) | throughput (Mrows/s) | ratio |
|---|---|---|---|---|
| A (scalar)  | 1 | 1426.066 | 7.012 (7.0, 7.0) | 1.00× vs A |
| B (SIMD scan) | 1 | 1341.374 | 7.455 (7.1, 7.5) | **1.06× vs A** |
| C (threads) | 4 | 362.508 | 27.586 (27.3, 27.6) | 3.93× vs A |
| D (ractor)  | 4 | 452.936 | 22.078 (22.1, 19.7) | **0.80× vs C** |

Readings:

- **B vs A (1.06×):** vectorizing ONLY the delimiter find barely moves
  the needle — the hot cost at this corpus is the scalar temp parse +
  `BTreeMap` accumulation, exactly as §5's t0 analysis predicted. The
  SIMD scan is not wasted (it is the prerequisite structure for lane F's
  batched tile sweeps); it is just not the bottleneck by itself. Next
  levers remain SWAR parse + hash-map swap (§1 inventory rows 2–3).
- **D vs C (0.80×):** the operator's "ractor is a helper, not a
  messaging path" ruling, as a number — routing the identical chunked
  workload through actor-per-worker costs ~20% at this arrival rate,
  and that figure INCLUDES the one-time `Arc<Vec<u8>>` corpus copy
  (142 MB) that the actor boundary forces and `std::thread::scope`'s
  borrow does not (see `lane_d.rs` module doc). Actors buy supervision
  and single-writer ownership, not raw throughput; on the V3 substrate
  they own SoA mailboxes (W2b) rather than carry bulk data — this lane
  is the measured cost of doing it the wrong way, kept as the fence.
- Lane E (kanban scheduling tax, E−D isolates the journaling cost) is
  the next lane; lane F (Morton-tile cascaded shader vs plain radix
  control) closes the set per Addendum-13.

### §5.2 — t2 (lane E) — measured 2026-07-02

Same recipe corpus (hash re-verified byte-identical at regeneration),
same 4-core container, release build `--features lane-b,lane-d,lane-e`.
Two passes per configuration, best-of-2 (both listed). Lane C and D
re-run in the same session as live comparators:

| Lane | workers | batches | elapsed_ms (best) | throughput (Mrows/s) |
|---|---|---|---|---|
| C (threads) | 4 | — | 353.235 | 28.310 (28.3, 27.5) |
| D (ractor)  | 4 | — | 446.805 | 22.381 (21.6, 22.4) |
| E (kanban)  | 4 | 4   | 435.489 | 22.963 (22.8, 23.0) |
| E (kanban)  | 4 | 64  | 444.903 | 22.477 (22.0, 22.5) |
| E (kanban)  | 4 | 256 | 452.126 | 22.118 (21.8, 22.1) |

Readings:

- **The kanban journaling floor is within noise.** E at
  `batches == workers` (one card per worker — lane D's chunking plus a
  full Rubicon journal per chunk) measured *at or slightly above* lane D
  (22.963 vs 22.381 best-of; the two interleave across passes). The
  fresh-`KanbanActor`-per-card spawn + 3 `drive_version_tick` RPCs +
  join are invisible next to the shared actor-boundary corpus copy both
  lanes pay. E−D ≈ 0: journaling real work through the board costs
  nothing measurable at chunk granularity.
- **Fine-grained cards stay cheap.** 4 → 256 cards costs ~4%
  (22.963 → 22.118). Per-card overhead from the E(256)−E(4) elapsed
  delta: ≈ 16.6 ms / 252 extra cards ≈ **66 µs per card** (actor spawn +
  3 Rubicon ticks + join + queue pull). Against W2d's 550 ms Libet
  budget, a card's scheduling overhead is ~0.01% — the budget is spent
  on thinking, not on the board. This is the number Addendum-13 sent
  lane E to fetch.
- The dominant tax in D and E alike remains the actor-model boundary
  (one-time `Arc` corpus copy + task-vs-scoped-thread overhead) — the
  ~20% D-vs-C gap carries over unchanged; the kanban layer adds nothing
  material on top.
- Lane F (Morton-tile cascaded shader vs a plain radix control — the
  addressing-tax isolator) is the remaining lane.
