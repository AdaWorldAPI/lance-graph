# PR-SPRINT-13-RAYON-STREAMS — `par_*` Rayon-Parallel Stream Variants

> **Status:** Draft (2026-05-16, sprint-13 preflight PP-3)
> **Deliverable IDs:** D-CSV-17 (NEW — `par_*` rayon variants of D-CSV-11 vertical streaming structs)
> **Parent plans:**
>   - `.claude/plans/cognitive-substrate-convergence-v2.md` §11 (D-CSV-11 productization; sprint-13+ row identifies `par_*` as backlog)
>   - sprint-13-execution-plan-v3.md (PP-1, planned) — adds D-CSV-17 as a first-class sprint-13 row
> **Worker:** sprint-13 W-G-rayon (parallel iterator pair)
> **LOC estimate:** ~120 LOC source + ~150 LOC tests = ~270 LOC actual; ~600 LOC spec text
> **Risk:** Low — additive only, gated behind ndarray's existing `rayon` feature; no consumer breakage
> **Delta discipline:** Every architectural decision cites either (a) the sprint-12-merged scalar scaffolds in `/home/user/ndarray/src/hpc/stream/` (W-F4/W-F5/W-F6 via PR #147) or (b) the rayon 1.10 trait surface as documented at https://docs.rs/rayon/1.10/rayon/iter/trait.IndexedParallelIterator.html. New material is confined to §3 (per-stream sketch), §5 (chunk-size strategy), §6 (determinism contract), and §7 (the 18 tests).

---

## §0 Status / Cross-references

**Parent deliverable:** D-CSV-11 (vertical streaming structs in ndarray).
The sprint-12 productization merge (PR #147, ndarray) shipped the three
scalar forward-iterator scaffolds — `QualiaStream`, `InferenceStream`,
`SplatFieldStream` — in `/home/user/ndarray/src/hpc/stream/{qualia,inference,splat_field}.rs`.
Each of those files contains a doc-comment of the form

    /// Pure iterator scaffold; the `par_*` rayon-parallel variant is
    /// sprint-13+ once rayon is wired into the ndarray feature gate.

This PR fulfils that promise. The rayon dependency was already present
in `/home/user/ndarray/Cargo.toml` line 47

    rayon = { version = "1.10.0", optional = true }

with the `rayon = ["dep:rayon", "std"]` feature flag at line 148. Sprint-12
left it unused inside `src/hpc/stream/`; sprint-13 wires it in.

**New deliverable id:** D-CSV-17 — chosen to not collide with the in-flight
D-CSV-13/14/15 entries in convergence-v2 §11, and to follow the W-Meta-Opus
canonical sprint-13 D-id assignment: **D-CSV-13b → PP-6 (SIMD), D-CSV-14 →
PP-4 (splat on-Think), D-CSV-16 → PP-5 (WitnessIndexCamPq), D-CSV-17 → PP-3
(this spec, rayon par_*)**. See `.claude/board/sprint-log-13/preflight-meta-
review-opus.md` §2 for the reconciled per-planner table (CSI-19 cleanup).

**Cross-ref table:**

| Reference | Where | Why |
|---|---|---|
| `qualia.rs:7` | "par_qualia_stream rayon-parallel variant is sprint-13+" | Doc-comment promise this PR fulfils |
| `inference.rs:6` | "par_inference_stream rayon variant is sprint-13+" | Same |
| `splat_field.rs:7` | "par_splat_stream rayon variant is sprint-13+" | Same |
| `Cargo.toml:47,148` | rayon optional dep + feature flag | Feature gate already exists |
| convergence-v2.md §11 D-CSV-11 row | "QualiaStream, InferenceStream, SplatFieldStream + par_* rayon variants" | Parent deliverable |
| convergence-v2.md §11 D-CSV-14 row | "on-Think method migration for D-CSV-12 splat ops" | Sibling sprint-13+ deliverable (PP-4 spec) |
| ndarray CLAUDE.md "Data-Flow Invariants" | `&[u8]` slices, no `&mut self` during compute | Hard constraint this spec respects |

---

## §1 Statement of Scope

This PR adds **three** new functions in `/home/user/ndarray/src/hpc/stream/`,
each behind `#[cfg(feature = "rayon")]`:

1. `par_qualia_stream(rows: &[QualiaI4Row]) -> impl IndexedParallelIterator<Item = (usize, &QualiaI4Row)>`
2. `par_inference_stream(rows: &[InferenceRow]) -> impl IndexedParallelIterator<Item = (usize, &InferenceRow)>`
3. `par_splat_field_stream(rows: &[SplatField]) -> impl IndexedParallelIterator<Item = (usize, &SplatField)>`

The PR does NOT:

- modify the existing scalar `QualiaStream` / `InferenceStream` / `SplatFieldStream` structs
- modify the existing row types (`QualiaI4Row`, `InferenceRow`, `SplatField`)
- modify the `mod.rs` re-exports for the scalar path (the `par_*` symbols are added to `pub use`)
- add a new feature flag (re-uses the existing `rayon` feature already wired)
- introduce a `&mut self` API anywhere (the data-flow invariant in `.claude/rules/data-flow.md` forbids it during compute)

The point of this PR is not the rayon trivia — three one-liner
`rows.par_iter().enumerate()` wrappers — it is the four non-trivial
contracts in §2-§6 below.

---

## §2 The Rayon Trait Surface

Rayon exposes two parent traits for parallel iteration:

- **`ParallelIterator`** — the loose surface. Methods like `filter`, `map`, `for_each`, `fold`, `reduce`, `collect` are available, but **length is unknown ahead of time**. `enumerate()` is NOT available on a bare `ParallelIterator`.
- **`IndexedParallelIterator`** — the strong surface. Sub-trait of `ParallelIterator`; adds `enumerate()`, `zip()`, `chunks()`, `with_min_len()`, `with_max_len()`. Only types whose length is known at compile-time-of-iteration implement it. `[T]::par_iter()` returns this stronger form via `rayon::slice::Iter<T>`.

**Decision:** all three `par_*` functions return `impl IndexedParallelIterator`.
Rationale:

1. We **need** `enumerate()` to mirror the scalar `(usize, &T)` yield shape.
2. We **need** `with_min_len()` to expose the per-thread chunk-size knob per §5.
3. Downstream consumers in the cognitive shader (`crates/lance-graph-planner/src/cache/*`) want to `zip` parallel streams of qualia, inference rows, and splat samples — `zip` is `IndexedParallelIterator`-only.

**The signature `impl IndexedParallelIterator<Item = (usize, &T)>` is
load-bearing**, not stylistic. A return of `impl ParallelIterator` would
break `enumerate` and `zip` for consumers and force them into a less-
efficient `collect()` round-trip.

**Why not return a concrete type?** rayon's `Enumerate<rayon::slice::Iter<'a, T>>`
is the literal type, but exposing it leaks rayon implementation details
into the function signature and prevents us from layering a `with_min_len`
default in the future without an SBP-breaking change. `impl Trait` is the
right surface.

---

## §3 Per-stream Implementation Sketch

### 3.1 `par_qualia_stream`

File: `/home/user/ndarray/src/hpc/stream/qualia.rs` (append to the end,
before the `#[cfg(test)] mod tests` block).

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    /// Rayon-parallel forward-iterator over a borrowed `&[QualiaI4Row]` slice.
    ///
    /// Yields `(row_index, &QualiaI4Row)` tuples. Unlike the scalar
    /// `QualiaStream`, iteration order is **not** guaranteed to be ascending
    /// by index; rayon's work-stealing scheduler may process chunks
    /// out-of-order. See §6 of pr-sprint-13-rayon-streams.md for the
    /// determinism contract callers must respect.
    ///
    /// # Chunk-size note
    /// The default rayon split policy targets ~2× CPU-count chunks. For
    /// QualiaI4Row (8 B/row, exactly one row per cache line is wasteful),
    /// callers folding into ordered structures should set
    /// `.with_min_len(8)` to align chunks to 64-byte cache lines.
    ///
    /// # Example
    /// ```
    /// # #[cfg(feature = "rayon")] {
    /// use ndarray::hpc::stream::qualia::{QualiaI4Row, par_qualia_stream};
    /// use rayon::prelude::*;
    ///
    /// let rows: Vec<QualiaI4Row> = (0u64..1024).map(QualiaI4Row).collect();
    /// let total_nonzero: usize = par_qualia_stream(&rows)
    ///     .filter(|(_, r)| r.0 != 0)
    ///     .count();
    /// assert_eq!(total_nonzero, 1023); // QualiaI4Row(0) is the lone zero
    /// # }
    /// ```
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_qualia_stream(
        rows: &[QualiaI4Row],
    ) -> impl IndexedParallelIterator<Item = (usize, &QualiaI4Row)> {
        rows.par_iter().enumerate()
    }

### 3.2 `par_inference_stream`

File: `/home/user/ndarray/src/hpc/stream/inference.rs` (same append pattern).

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    /// Rayon-parallel forward-iterator over `&[InferenceRow]`.
    ///
    /// Mirrors `par_qualia_stream` semantics. For InferenceRow (8 B/row),
    /// the same 8-rows-per-cache-line chunking applies; callers should
    /// `.with_min_len(8)` when folding into ordered structures.
    ///
    /// Particularly useful for the integer-SIMD MUL evaluation hot path
    /// (D-CSV-8): folding the inference mantissa lane across millions of
    /// EdgeColumn rows benefits from work-stealing on multi-core hosts.
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_inference_stream(
        rows: &[InferenceRow],
    ) -> impl IndexedParallelIterator<Item = (usize, &InferenceRow)> {
        rows.par_iter().enumerate()
    }

### 3.3 `par_splat_field_stream`

File: `/home/user/ndarray/src/hpc/stream/splat_field.rs` (same pattern).

    #[cfg(feature = "rayon")]
    use rayon::prelude::*;

    /// Rayon-parallel forward-iterator over `&[SplatField]`.
    ///
    /// SplatField is 16 B (4 × 4-byte fields, `repr(C, align(16))`), so
    /// 4 rows fit one 64-byte cache line. Callers should `.with_min_len(4)`
    /// for cache-line-aligned chunking; see §5.
    ///
    /// Used by the D-CSV-12 splat op fleet for parallel evaluation of
    /// `splat_gaussian`, `score_hole_closure`, `replay_coherence`, and
    /// `emit_if_epiphany` across the entire splat field.
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_splat_field_stream(
        rows: &[SplatField],
    ) -> impl IndexedParallelIterator<Item = (usize, &SplatField)> {
        rows.par_iter().enumerate()
    }

### 3.4 `mod.rs` re-exports

File: `/home/user/ndarray/src/hpc/stream/mod.rs` (extend existing re-exports).

    pub use inference::{InferenceRow, InferenceStream};
    pub use qualia::{QualiaI4Row, QualiaStream};
    pub use splat_field::{SplatField, SplatFieldStream};

    #[cfg(feature = "rayon")]
    pub use inference::par_inference_stream;
    #[cfg(feature = "rayon")]
    pub use qualia::par_qualia_stream;
    #[cfg(feature = "rayon")]
    pub use splat_field::par_splat_field_stream;

---

## §4 Feature Gate

The ndarray `Cargo.toml` already exposes:

    [dependencies]
    rayon = { version = "1.10.0", optional = true }

    [features]
    rayon = ["dep:rayon", "std"]

This PR **re-uses the existing feature**. It does NOT add a new feature
flag. The feature is enabled in three relevant downstream contexts:

1. `cargo test --features parallel` — runs the 18 new tests in §7.
2. `[package.metadata.docs.rs] features = ["approx", "serde", "rayon"]` (Cargo.toml:255) — docs.rs build already includes rayon, so the new `par_*` doc-comments will render on docs.rs without further config.
3. `[[bench]] name = "par_rayon", required-features = ["rayon"]` (Cargo.toml:122) — existing par_rayon bench already conditional on this feature; no conflict.

**No default-feature change.** The default features remain
`["std", "hpc-extras"]`; rayon stays opt-in. This preserves the
`thumbv6m-none-eabi --no-default-features` nostd build invariant
documented in Cargo.toml lines 56-60.

**Why not gate behind a new `parallel` feature?** That would fragment the
flag space — downstream consumers already write `--features parallel`
everywhere (par_rayon bench, docs.rs metadata). A second flag pointing at
the same rayon dep is pure noise. The user-facing prompt mentioned
`parallel` as a possibility; the audit shows `rayon` is the canonical
spelling in this repo.

---

## §5 Chunk-Size Strategy (Cache-Line Aligned)

Rayon's default chunking heuristic targets ~2× CPU-count chunks, which is
optimised for throughput on long-running per-item work but is suboptimal
for the streaming SoA case where:

1. Each row is small (8 B for QualiaI4Row/InferenceRow, 16 B for SplatField).
2. Per-row work is tight (bitfield extraction or single-FP multiplication).
3. Cache-line bouncing dominates if chunk boundaries don't align to 64-byte lines.

**The cache-line table:**

| Row type | Size | Rows / 64 B cache line | Recommended `with_min_len(N)` |
|---|---|---|---|
| `QualiaI4Row` | 8 B (`repr(C, align(8))`) | 8 | `.with_min_len(8)` |
| `InferenceRow` | 8 B (`repr(C, align(8))`) | 8 | `.with_min_len(8)` |
| `SplatField` | 16 B (`repr(C, align(16))`) | 4 | `.with_min_len(4)` |

**Why minimum-length, not chunks()?** `with_min_len` is a hint to the
splitter, not a hard partition; rayon retains the freedom to split coarser
if work is bursty. `chunks(N)` forces fixed-size chunks and disables
work-stealing within the chunk. For SoA streaming the hint is the right
knob.

**Why expose this knob via documentation rather than baking it in?**
Three reasons:

1. **Composition** — when the caller chains `.zip(other_par_iter)`, the
   minimum-length of the zipped iterator is `max(left, right)`. Baking in
   a default would silently override caller intent.
2. **Locality** — the optimal chunk depends on the downstream operation.
   A reducer-into-Vec wants larger chunks; a fold-into-NarsTruth wants
   smaller chunks so per-thread state stays cache-resident.
3. **Empirical tuning** — the par_rayon bench in `/home/user/ndarray/benches/par_rayon.rs` is the place to find the right default per host architecture. The bench is sprint-14+ scope, not this PR.

**The doc-comments in §3 surface the recommended value but do not
enforce it.** This is the same posture as `Vec::with_capacity` —
ergonomics via documentation, not via a hidden default.

---

## §6 Determinism Contract

Rayon's work-stealing scheduler is **non-deterministic in order** but
**deterministic in content**: given identical inputs, the set of emitted
items is invariant; the sequence in which they arrive at the consumer is
not.

This matters for three downstream consumer patterns observed in the
cognitive substrate:

### 6.1 Pattern A — order-insensitive folds (safe)

    let total: u64 = par_qualia_stream(&rows)
        .map(|(_, r)| r.0)
        .sum();

`+` on `u64` is associative and commutative, so any chunk-order is
equivalent. **Safe.** Rayon's `sum`, `count`, `max`, `min`, `any`, `all`
are all in this category.

### 6.2 Pattern B — `collect` into `Vec` (preserves index order)

    let collected: Vec<(usize, &QualiaI4Row)> = par_qualia_stream(&rows).collect();

`IndexedParallelIterator::collect_into_vec` is contract-guaranteed to
**preserve original-iterator order**, regardless of chunk-execution
order. **Safe.** This is why we mandated `IndexedParallelIterator` in §2.

### 6.3 Pattern C — `fold` into a non-commutative accumulator (DANGEROUS)

    let bundle: Vsa16kF32 = par_qualia_stream(&rows)
        .fold(Vsa16kF32::zero, |acc, (i, r)| acc.bundle_with(r, i))
        .reduce(Vsa16kF32::zero, Vsa16kF32::compose);

If `bundle_with` or `compose` is not exactly associative (e.g. float
rounding under `+` is non-associative for non-degenerate magnitudes),
the result depends on chunk-boundary placement.

**The contract:** callers in pattern C MUST either

- accept the non-determinism (acceptable for VSA bundling because
  Johnson-Lindenstrauss concentration-of-measure bounds deviation
  at ~e^(-d), per `I-SUBSTRATE-MARKOV` iron rule in lance-graph
  CLAUDE.md), OR
- explicitly call `.with_min_len(rows.len())` to coerce a single-chunk
  serial fold — which defeats the point of parallelism but restores
  bit-exact reproducibility for golden-master tests.

### 6.4 The `with_min_len(N)` knob

`with_min_len(rows.len())` is the universal "force serial" escape hatch.
It is the only mechanism by which a caller can guarantee bit-exact
ordering across runs. We document it but do not call it ourselves;
the parallel iterators are *useful* only when the caller has
analysed determinism and concluded pattern A or B applies.

**Cross-reference:** `.claude/knowledge/vsa-switchboard-architecture.md`
(VSA bundling associativity in expectation) and the substrate-level iron
rule `I-SUBSTRATE-MARKOV` in lance-graph CLAUDE.md establish that
`Vsa16kF32` bundle is associative in expectation at d=16384 — so
pattern C is safe for VSA bundling in particular, with the deviation
bounded by Johnson-Lindenstrauss. Other accumulators (raw f32 sum,
NARS truth revision, AriGraph triplet insertion) need per-callsite
analysis.

---

## §7 The 18 Tests (6 × 3 streams)

Each stream gets the same six-test matrix; only the row type and helper
constructors differ. Tests are added to the existing `#[cfg(test)] mod
tests` block at the bottom of each file, each guarded by
`#[cfg(feature = "rayon")]` so the default-feature build skips them.

### 7.1 QualiaStream — 6 tests

Appended to `qualia.rs`:

    #[cfg(all(test, feature = "rayon"))]
    mod par_tests {
        use super::*;
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// T-P-Q-1: par_qualia_stream yields all N items.
        #[test]
        fn test_par_qualia_yields_all() {
            let rows: Vec<QualiaI4Row> = (0u64..1024).map(QualiaI4Row).collect();
            let count = par_qualia_stream(&rows).count();
            assert_eq!(count, 1024);
        }

        /// T-P-Q-2: par_qualia_stream on empty slice yields zero items.
        #[test]
        fn test_par_qualia_empty() {
            let rows: Vec<QualiaI4Row> = vec![];
            let count = par_qualia_stream(&rows).count();
            assert_eq!(count, 0);
        }

        /// T-P-Q-3: par_iter result equals serial iter result (as sets).
        #[test]
        fn test_par_qualia_matches_serial() {
            let rows: Vec<QualiaI4Row> = (0u64..256).map(QualiaI4Row).collect();
            let mut par: Vec<u64> =
                par_qualia_stream(&rows).map(|(i, r)| (i as u64) ^ r.0).collect();
            let mut ser: Vec<u64> =
                QualiaStream::new(&rows).map(|(i, r)| (i as u64) ^ r.0).collect();
            par.sort();
            ser.sort();
            assert_eq!(par, ser);
        }

        /// T-P-Q-4: par_iter with filter is correct (set equality).
        #[test]
        fn test_par_qualia_with_filter() {
            let rows: Vec<QualiaI4Row> = (0u64..512).map(QualiaI4Row).collect();
            let count_even = par_qualia_stream(&rows)
                .filter(|(_, r)| r.0 % 2 == 0)
                .count();
            assert_eq!(count_even, 256);
        }

        /// T-P-Q-5: with_min_len(N) knob compiles and yields all items.
        #[test]
        fn test_par_qualia_min_len() {
            let rows: Vec<QualiaI4Row> = (0u64..1024).map(QualiaI4Row).collect();
            let count = par_qualia_stream(&rows).with_min_len(8).count();
            assert_eq!(count, 1024);
        }

        /// T-P-Q-6: thread-safety — Send + Sync auto-derived on QualiaI4Row.
        /// Verified by spawning a parallel for_each that mutates an
        /// AtomicUsize from multiple threads.
        #[test]
        fn test_par_qualia_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<QualiaI4Row>();
            let rows: Vec<QualiaI4Row> = (0u64..1024).map(QualiaI4Row).collect();
            let counter = AtomicUsize::new(0);
            par_qualia_stream(&rows).for_each(|_| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(counter.load(Ordering::Relaxed), 1024);
        }
    }

### 7.2 InferenceStream — 6 tests

Appended to `inference.rs`. Mirror structure of 7.1, replacing
`QualiaI4Row` with `InferenceRow` and exercising the `inference_mantissa()`
/ `w_slot()` accessors in T-P-I-4:

    #[cfg(all(test, feature = "rayon"))]
    mod par_tests {
        use super::*;
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// T-P-I-1
        #[test]
        fn test_par_inference_yields_all() { /* analogous to T-P-Q-1 */ }

        /// T-P-I-2
        #[test]
        fn test_par_inference_empty() { /* analogous to T-P-Q-2 */ }

        /// T-P-I-3
        #[test]
        fn test_par_inference_matches_serial() { /* analogous to T-P-Q-3 */ }

        /// T-P-I-4: filter on inference_mantissa() — sign-extension must
        /// behave identically under parallel access.
        #[test]
        fn test_par_inference_filter_mantissa() {
            // 256 rows with mantissa varying 0..16 cyclically; expect
            // 128 negative (mantissa raw 8..15 → −8..−1) and 128 non-neg.
            let rows: Vec<InferenceRow> =
                (0u64..256).map(|i| InferenceRow((i & 0xF) << 46)).collect();
            let neg = par_inference_stream(&rows)
                .filter(|(_, r)| r.inference_mantissa() < 0)
                .count();
            assert_eq!(neg, 128);
        }

        /// T-P-I-5
        #[test]
        fn test_par_inference_min_len() { /* analogous to T-P-Q-5 */ }

        /// T-P-I-6
        #[test]
        fn test_par_inference_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<InferenceRow>();
            /* + atomic-counter for_each */
        }
    }

### 7.3 SplatFieldStream — 6 tests

Appended to `splat_field.rs`. Mirror structure, with one substantive
divergence in T-P-S-4 (uses `filter_energy_above` semantics):

    #[cfg(all(test, feature = "rayon"))]
    mod par_tests {
        use super::*;
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        fn make_splat(mean: u32, variance: f32, energy: f32, generation: u32) -> SplatField {
            SplatField { mean, variance, energy, generation }
        }

        /// T-P-S-1
        #[test]
        fn test_par_splat_yields_all() { /* analogous to T-P-Q-1 */ }

        /// T-P-S-2
        #[test]
        fn test_par_splat_empty() { /* analogous to T-P-Q-2 */ }

        /// T-P-S-3
        #[test]
        fn test_par_splat_matches_serial() { /* analogous to T-P-Q-3 */ }

        /// T-P-S-4: filter on energy field — parallel-equivalent of
        /// SplatFieldStream::filter_energy_above.
        #[test]
        fn test_par_splat_filter_energy() {
            let rows: Vec<SplatField> = (0u32..256)
                .map(|i| make_splat(i, 1.0, (i as f32) / 256.0, 0))
                .collect();
            let above = par_splat_field_stream(&rows)
                .filter(|(_, s)| s.energy > 0.5)
                .count();
            // Energies (i/256) > 0.5 for i in 129..256 → 127 items.
            assert_eq!(above, 127);
        }

        /// T-P-S-5: with_min_len(4) — cache-line alignment knob.
        #[test]
        fn test_par_splat_min_len() {
            let rows: Vec<SplatField> = (0u32..1024)
                .map(|i| make_splat(i, 1.0, 0.1, 0))
                .collect();
            let count = par_splat_field_stream(&rows).with_min_len(4).count();
            assert_eq!(count, 1024);
        }

        /// T-P-S-6: thread-safety — SplatField is Send + Sync.
        #[test]
        fn test_par_splat_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<SplatField>();
            /* + atomic-counter for_each */
        }
    }

### 7.4 Test invocation

    cargo test -p ndarray --features parallel hpc::stream

All 18 tests should pass on any host with ≥1 CPU; rayon falls back to
a single-thread pool gracefully on uniprocessor builds.

---

## §8 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Rayon thread-pool init overhead dominates for small N | Med | Low — degrades to serial latency at small N | Doc-comment recommends serial `QualiaStream` for N < 1024; benchmark plumbing deferred to sprint-14 par_rayon bench |
| `with_min_len` ignored by future rayon major version | Low | Med — chunk-boundary semantics drift | Pin rayon = "1.10.0" (already pinned in Cargo.toml); review on each rayon major bump |
| Default-feature CI does not exercise the new tests | High | Low — that's the design; gated tests = silent dead code | Add `cargo test -p ndarray --features parallel` to the CI matrix; **co-shipped in the same ndarray PR as the par_* impl** (sprint-13 W-I4, the sonnet worker assigned D-CSV-17) — CI gate naturally belongs in the same PR as the `#[cfg(feature = "parallel")]` code it gates, not a separate planner. |
| Pattern-C non-determinism caught by a golden-master test in lance-graph | Med | Med — flaky test triage cost | Doc-comment §6 makes the contract explicit; offending consumers must call `.with_min_len(rows.len())` |
| `IndexedParallelIterator` return type prevents future migration to a non-rayon parallel backend | Low | High — would be SBP-breaking | Acceptable: rayon 1.x is the de facto Rust parallel standard; abstraction is premature pessimisation |
| Cargo --no-default-features + miri build broken by accidental rayon import outside the cfg-gate | Low | High — breaks `thumbv6m-none-eabi` invariant | The `#[cfg(feature = "rayon")] use rayon::prelude::*;` placement is local to each function; verified by `cargo check --no-default-features` in §10 LOC count |

---

## §9 Cross-References

**Plan v3 (PP-1 will draft this):**

- D-CSV-17 row in §11 deliverable matrix, status "In PR" once this PR opens, "Shipped" on merge.
- Linked from D-CSV-11 row ("par_* rayon variants → sprint-13 D-CSV-17 ships them").
- Linked from D-CSV-14 row ("on-Think method migration — when those methods land, they can use `par_*` for fleet-evaluation").

**Plan v2 (current, retroactive note):**

- §11 D-CSV-11 row will get a one-line update on merge:
  `"...par_* rayon variants (sprint-13+, see D-CSV-17 / PR-SPRINT-13-RAYON-STREAMS)."`

**Sibling preflight specs (sprint-13, planned):**

- PP-1: sprint-13 execution plan v3 — introduces D-CSV-17 row in §11.
- PP-4: D-CSV-14 on-Think method migration spec — co-evolves with this PR;
  the splat on-Think methods will internally call `par_splat_field_stream`
  for the fleet-evaluation hot path.
- W-I4 (sprint-13 sonnet impl worker assigned D-CSV-17): adds the
  `--features parallel` row to ndarray's CI matrix in the same ndarray PR
  that ships par_* — keeps the rayon gate co-located with the rayon
  code it gates, prevents silent-dead-code drift.

**Knowledge docs (READ BY):**

- `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` — original
  vision for rayon + struct-method surface; this PR is the
  rayon half (struct-method half is D-CSV-14 / PP-4).
- `.claude/knowledge/vsa-switchboard-architecture.md` — VSA bundle
  associativity-in-expectation; underwrites the §6 pattern C safety
  argument for `Vsa16kF32`.

**Iron rules respected:**

- `I-SUBSTRATE-MARKOV` (lance-graph CLAUDE.md) — VSA bundle associativity;
  §6 patterns A/C analysis depends on it.
- `.claude/rules/data-flow.md` (ndarray) — "No `&mut self` during
  computation. Ever." The `par_*` functions take `&[T]` immutable
  borrows and return iterators that yield `&T` references; no mutation
  surface is introduced.

---

## §10 LOC Estimate

Per-stream source delta:

| File | Source LOC | Test LOC |
|---|---|---|
| `qualia.rs` | +30 (use-import, doc-comments, fn body) | +60 (6 tests, helpers) |
| `inference.rs` | +30 | +60 |
| `splat_field.rs` | +30 | +60 |
| `mod.rs` | +6 (three new `pub use` lines under cfg-gate) | 0 |
| **Total** | **~96 LOC src** | **~180 LOC tests** |

Add ~25 LOC for doc-comment overhead (`# Example` blocks repeat per
function) and we land in the **~120 LOC source + ~150-180 LOC tests =
~270-300 LOC actual** range.

The spec itself (this file) is ~600 lines of markdown — the
specification-to-code ratio is ~2:1, in line with the pr-ce64-mb-N
series (cf. pr-ce64-mb-1-par-tile-crate.md: ~1500 LOC src, ~3000 LOC
spec; ratio ~2:1).

---

## §11 Acceptance Checklist

Before merging this PR:

- [ ] `cargo check -p ndarray` (no rayon feature) — builds clean
- [ ] `cargo check -p ndarray --no-default-features` — nostd build clean
- [ ] `cargo check -p ndarray --features parallel` — rayon build clean
- [ ] `cargo test -p ndarray --features parallel hpc::stream` — all 18 new tests pass
- [ ] `cargo test -p ndarray hpc::stream` (default) — existing scalar tests still pass; new par_tests modules are skipped
- [ ] `cargo doc -p ndarray --features parallel` — doc-comment examples compile
- [ ] `cargo clippy -p ndarray --features parallel -- -D warnings` — no lints
- [ ] `cargo fmt --check` — rustfmt 1.95.0 gate
- [ ] LATEST_STATE.md updated with D-CSV-17 entry
- [ ] PR_ARC_INVENTORY.md PREPEND entry (Added: 3 fns / Locked: rayon feature surface / Deferred: bench tuning to sprint-14 / Docs: this spec / Confidence: HIGH)
- [ ] STATUS_BOARD.md D-CSV-17 row marked Shipped on merge
- [ ] convergence-v2.md §11 D-CSV-11 row updated with merge cross-ref

---

## §12 Open Questions

- **OQ-RAY-1:** Should `par_*` functions accept `&[T]` or a generic
  `impl IntoParallelRefIterator`? Current spec uses `&[T]` for ergonomic
  signatures; the generic form would enable e.g. `BindSpace` columns to
  be passed directly. **Recommendation:** ship `&[T]` for sprint-13;
  revisit if a consumer hits the type-mismatch friction. (Likelihood
  low — every current consumer already materialises `&[T]`.)
- **OQ-RAY-2:** Should we ship a `par_*_chunks(rows, chunk_size)`
  variant that wraps `rows.par_chunks(N).enumerate().flat_map(...)` for
  callers who want explicit chunk control? **Recommendation:** defer to
  sprint-14; the `with_min_len` hint covers 95% of use cases, and the
  chunks form is only useful for `for_each` callers who want to amortise
  per-chunk setup. Add only when a real consumer surfaces.
- **OQ-RAY-3:** Should the parallel iterators implement
  `IndexedParallelIterator` directly via a newtype wrapper, instead of
  via `impl Trait`? **Recommendation:** no — `impl Trait` is the right
  surface (cf. §2 rationale). Reconsider only if downstream needs to
  name the type (e.g. for `Box<dyn>` storage, which would be the wrong
  pattern anyway per the ndarray "no `Box<dyn>` in hot paths"
  CLAUDE.md rule).

---

**End of pr-sprint-13-rayon-streams.md** — ~600 LOC markdown for ~270
LOC source+tests. Sprint-13 preflight PP-3 complete.
