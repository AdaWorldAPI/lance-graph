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
| **B** — ndarray SIMD | Vectorized semicolon/newline scanning and/or batched parse via `ndarray::simd` (per the workspace's SIMD rule — never raw `pulp`/`wide`/hand intrinsics in a consumer crate; see `.claude/knowledge/ndarray-vertical-simd-alien-magic.md`). Would also evaluate whether an `ndarray`-backed SIMD hash or SWAR-style parse closes the gap to the reference's `v15_raw_hash`. | **Not implemented** — orchestrator follow-up |
| **D** — `ractor` actors | Same groupby-aggregate workload, but the aggregation runs as `ractor`-supervised actors (per this workspace's `lance-graph-supervisor` precedent) instead of bare `std::thread::scope`, to measure actor-model overhead/benefit vs Lane C's raw threads at this workload's arrival rate. | **Not implemented** — orchestrator follow-up |
| **E** — kanban | Routes the aggregation through the V3 kanban execution machinery (`v3-kanban-executor-engineer` domain — `KanbanPhase` lifecycle, ahead-firing batch writer) to measure the substrate's own scheduling/dispatch overhead against the bare-metal Lane A/C numbers as a ceiling reference. | **Not implemented** — orchestrator follow-up |

---

## §4 — CLI

```text
onebrc-probe gen <path> <rows> <seed>
onebrc-probe run <path> <lane:a|c> [workers]
```

`run` prints:

```text
lane=<X> rows=<N> workers=<W> elapsed_ms=<T> throughput_mrows_s=<R>
-- first 3 stations --
  ...
-- last 3 stations --
  ...
```

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
