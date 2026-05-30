// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `lance-graph-arm-discovery` — the **upstream proposer** leg of the
//! streaming association-rule discovery → NARS revision → ratified SPO →
//! deterministic codegen pipeline (`streaming-arm-nars-discovery-v1.md`).
//!
//! This crate is the Rust transcoding of **Aerial+** (Karabulut, Groth,
//! Degeler — *Neurosymbolic Association Rule Mining from Tabular Data*,
//! arXiv 2504.19354v1, Apr 2025), the neurosymbolic association-rule miner.
//! It mines `(X → Y)` rules from tabular runtime data and lifts each rule
//! into a NARS-truth-carrying SPO candidate that the existing
//! `lance_graph::graph::spo` store consumes through the **same ndjson
//! contract** as the static `ruff_spo_triplet` extractor.
//!
//! # Pipeline position
//!
//! ```text
//!   tabular rows ─┐
//!                 ▼  Stage A — Aerial+ proposer  (THIS CRATE, `aerial`)
//!   one-hot encode → denoising autoencoder (softmax-per-feature, BCE)
//!                 → Algorithm 1 reconstruction probe → CandidateRule
//!                 ▼  Stage B — translator        (THIS CRATE, `translator`)
//!   arm_to_nars: (support, confidence, n) → NARS (frequency, confidence)
//!                 ▼  emit                        (THIS CRATE, `ndjson`)
//!   {"s","p","o","f","c"} ndjson  ─►  lance_graph SPO store loader
//!                 ▼  Stage C/D/E — hypothesis test → council → op_emitter
//!                                                   (downstream, other crates)
//! ```
//!
//! # The determinism boundary (read before extending)
//!
//! Aerial+'s autoencoder is **nondeterministic** in the general case (random
//! init, denoising noise, float reduction order). Per the plan's determinism
//! firewall it is a **fan-in proposer**, NEVER the deterministic trunk and
//! NEVER allowed to cross the ratification gate (Stage D) un-vetted. Two
//! consequences are baked into this crate:
//!
//! 1. The autoencoder is **seedable** ([`aerial::Rng`]). Same seed, same data,
//!    and same hyper-parameters give bit-identical weights and identical
//!    rules. This makes the proposer reproducible for tests and audits even
//!    though it is not the canonical deterministic path.
//! 2. The output is a plain [`rule::CandidateRule`] — a *proposal*, not a
//!    committed triple. Promotion to the SPO store is the downstream
//!    hypothesis-test + council's job, not this crate's.
//!
//! # Truth mapping (verbatim from the paper, §2 / §3.3)
//!
//! - ARM **confidence** = P(Y|X) → NARS **frequency** `f`
//! - ARM **support × n** (evidential mass) → NARS **confidence** `c = m/(m+k)`
//!
//! See [`translator::arm_to_nars`]. The resulting `(f, c)` is exactly the pair
//! `lance_graph::graph::spo::TruthValue::new(f, c)` and
//! `ruff_spo_triplet::Triple { f, c }` carry.

#![forbid(unsafe_code)]

pub mod encode;
pub mod ndjson;
pub mod rule;
pub mod translator;

#[cfg(feature = "aerial")]
pub mod aerial;

pub use encode::{Dataset, FeatureSpec};
pub use rule::{CandidateRule, Item, Proposer};
pub use translator::{arm_to_nars, CandidateTriple, FeedProjector, NarsTruth};

#[cfg(feature = "aerial")]
pub use aerial::{AerialAutoencoder, AerialParams, AerialProposer, Rng};
