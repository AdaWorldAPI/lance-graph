<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright The Lance Authors
-->

# lance-graph-arm-discovery

Rust transcode of **Aerial+** — neurosymbolic association-rule mining
(Karabulut, Groth, Degeler, *Neurosymbolic Association Rule Mining from
Tabular Data*, arXiv 2504.19354v1). The **upstream proposer leg** of
`streaming-arm-nars-discovery-v1.md`: it mines `(X → Y)` rules from tabular
runtime data and lifts each into a NARS-truth SPO candidate emitted in the
**same ndjson shape** (`{s,p,o,f,c}`) as the static `ruff_spo_triplet`
extractor. The lance-graph `parse_triples` loader consumes it directly; the
`ruff` `from_ndjson` loader accepts it only after its closed predicate
vocabulary gains an implication relation (D-ARM-SYN-1).

```text
  rows ─► one-hot ─► denoising autoencoder ─► Algorithm 1 probe ─► CandidateRule
                     (softmax/feature, CE)     (reconstruction)         │
                                                                        ▼
                              arm_to_nars: (support, confidence, n) → NARS (f, c)
                                                                        ▼
                              {"s","p","o","f","c"} ndjson ─► SPO store loader
```

## Why standalone & zero-dep

This crate is **excluded from the workspace** (built via `--manifest-path`,
like `bgz17` / `deepnsm`) and depends on `std` only. Two reasons:

1. **Determinism boundary.** The autoencoder is nondeterministic in general;
   keeping it out of the `lance-graph` compile path enforces the plan's
   firewall — Aerial+ is a *fan-in* proposer, never the deterministic trunk,
   and its output is a *proposal* that the downstream ratification council
   must vet before any codegen leg consumes it.
2. **Reproducibility.** Every random source draws from one seeded SplitMix64
   stream (`aerial::Rng`), so the same seed + data + hyper-parameters give
   reproducible weights and identical mined rules *on a given target* —
   auditable despite being a neural net. (Intra-platform reproducibility, not
   bitwise-portable determinism: float `tanh`/`exp` + FMA can differ across
   targets.)

## Layout

| Module            | Stage | Role |
| ---               | ---   | ---  |
| `encode`          | —     | one-hot `FeatureSpec` + `Dataset` + support/confidence counting |
| `aerial::rng`     | A     | seeded SplitMix64 PRNG |
| `aerial::autoencoder` | A | under-complete denoising AE (softmax-per-feature, CE, hand backprop) |
| `aerial::extract` | A     | Algorithm 1 — reconstruction-probe rule extraction |
| `aerial` (`AerialProposer`) | A | `Proposer` impl: `fit` (train) + `next_batch` (mine) |
| `translator`      | B     | `arm_to_nars` + `CandidateTriple{s,p,o,f,c}` + `FeedProjector` |
| `ndjson`          | emit  | `{"s","p","o","f","c"}` lines = the SPO-loader contract |
| `rule`            | —     | `Item`, `CandidateRule`, `Proposer` trait |

## Build & test

```bash
cargo test  --manifest-path crates/lance-graph-arm-discovery/Cargo.toml
cargo clippy --manifest-path crates/lance-graph-arm-discovery/Cargo.toml --all-targets -- -D warnings
```

## Truth mapping (verbatim from the paper)

- ARM **confidence** = `P(Y|X)` → NARS **frequency** `f`
- ARM **support × n** (evidential mass `m`) → NARS **confidence** `c = m/(m+k)`

The resulting `(f, c)` is exactly the pair `TruthValue::new(f, c)` and
`ruff_spo_triplet::Triple{f, c}` carry — see
`.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md` for the full
synergy map with the `ruff` DTO / SPO / codegen crates.

## Status

Implements the Aerial+ leg (D-ARM-9) of the plan. The pair-stats deterministic
trunk (D-ARM-3), the `lance-graph-contract` carriers (D-ARM-1/2), and the
hypothesis-test / queue stages (D-ARM-5/6) remain to land; the local `rule` /
`translator` types are the seam until the contract carriers exist.
