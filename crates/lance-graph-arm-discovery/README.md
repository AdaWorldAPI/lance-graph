<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright The Lance Authors
-->

# lance-graph-arm-discovery

Rust transcode of **Aerial+** — neurosymbolic association-rule mining
(Karabulut, Groth, Degeler, arXiv 2504.19354v1) — with the paper's `f32`
autoencoder **replaced by an integer codebook distance oracle**. The
**upstream proposer leg** of `streaming-arm-nars-discovery-v1.md`: it mines
`(X → Y)` rules from tabular runtime data and lifts each into a NARS-truth SPO
candidate emitted in the `{s,p,o,f,c}` shape the SPO store loader reads.

```text
  rows ─► codebook items ─► codebook probe (nearest consequents within θ)
                            (integer CodebookDistance oracle = palette256)
                         ─► confirm on data: integer support/confidence counts
                         ─► CandidateRule (integer evidence)
                         ─► arm_to_truth_u8 → TruthU8 (= CausalEdge64 wire)
                         ─► {s,p,o,f,c} ndjson ─► SPO store loader
```

## Float-free by design

The autoencoder was a slow, seed-dependent, `f32` way to approximate a
nearest-neighbour query — the very thing this substrate answers *exactly* and
in integers via the **palette256 distance table** (`[a,b] → u32`, ρ=0.9973 vs
cosine). So Aerial+'s reconstruction probe becomes a **codebook top-k**:

| Aerial+ float piece | Deterministic replacement |
| --- | --- |
| denoising autoencoder (`f32` weights) | frozen `CodebookDistance` oracle (palette256) |
| reconstruction probe (forward pass) | codebook top-k from the antecedent, within `θ` |
| softmax ranking | integer distance ranking |
| support/confidence (`f32`) | integer counts; ppm cross-multiply gates |
| `(f,c)` `f32` truth | `TruthU8` = `confidence_u8` + i4 mantissa |
| seeded RNG | **deleted** — codebook lookup is bitwise-exact |

The only `f32` left is the **serialization edge** (`arm_to_nars`,
`CandidateTriple`, `ndjson`), present solely because the downstream
`spo::truth::TruthValue` and `ruff_spo_triplet::Triple` are themselves `f32`.
Nothing in the discovery path consumes it.

The similarity oracle is a **zero-dep trait** (`aerial::CodebookDistance`), so
the real distance table — `bgz17::PaletteDistanceTable`, the BLASGraph
Gaussian-splat top-k, or an HDR-popcount Hamming primitive — lives on the
consumer side and this crate stays standalone. `aerial::MatrixDistance` is the
in-crate reference impl.

## No determinism firewall needed

The codebook probe is **bitwise-deterministic by construction** (same data +
oracle + `θ` ⇒ identical rules, every target). It can sit in the deterministic
trunk beside pair-stats (D-ARM-3); the ratification council still governs
*promotion to the SPO store*, but no longer because of any nondeterminism here.
The Python-IPC isolation rationale (D-ARM-9) is fully moot.

## Layout

| Module | Stage | Role |
| --- | --- | --- |
| `encode` | — | integer `FeatureSpec` + `Dataset` + `count_matching` |
| `aerial::codebook` | A | `CodebookDistance` trait + `MatrixDistance` reference impl |
| `aerial::extract` | A | codebook-probe rule extraction (Algorithm 1, integer) |
| `aerial` (`AerialProposer`) | A | `Proposer` impl over a data window + oracle |
| `translator` | B | `arm_to_truth_u8` (`TruthU8`) + `f32` edge + `FeedProjector` |
| `ndjson` | emit | `{s,p,o,f,c}` lines = the SPO-loader contract |
| `rule` | — | `Item`, `CandidateRule` (integer counts, ppm gates), `Proposer` |

## Build & test

```bash
# Default: scalar, std-only, zero-dep (independently verifiable)
cargo test  --manifest-path crates/lance-graph-arm-discovery/Cargo.toml
cargo clippy --manifest-path crates/lance-graph-arm-discovery/Cargo.toml --all-targets -- -D warnings

# SIMD: route the bitset AND+popcount through ndarray::simd::U64x8
cargo test --manifest-path crates/lance-graph-arm-discovery/Cargo.toml --features ndarray-simd
```

The count loop transposes the window into row bitsets (`bitset::RowMasks`) so
every candidate count is an `AND` + popcount over `&[u64]` (`simd.rs`). The
default path is scalar `u64::count_ones`; the `ndarray-simd` feature swaps in
`ndarray::simd::U64x8` (zero raw intrinsics here — the polyfill owns dispatch).

**target-cpu:** the `U64x8` path is correct everywhere but only emits the real
vector kernels (AVX-512 VPOPCNTQ, AMX) when built for native silicon —
`RUSTFLAGS="-C target-cpu=native"` or `-C target-cpu=x86-64-v4`. Otherwise it is
ndarray's scalar fused-loop fallback (same result, no speedup).

## Status

Implements the Aerial+ leg of the plan as a deterministic codebook backend
(D-ARM-13). Still open: D-ARM-7 Jirak floor (ISSUE ARM-JIRAK-FLOOR — hard
prerequisite before wiring to a live `SpoStore`), the `lance-graph-contract`
carriers (D-ARM-1/2, TD-ARM-CARRIER-FORK), and the real palette256 oracle wiring
(consumer-side `CodebookDistance` impl over `bgz17::PaletteDistanceTable`).
