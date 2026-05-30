// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `lance-graph-arm-discovery` — the **upstream proposer** leg of the
//! streaming association-rule discovery → NARS revision → ratified SPO →
//! deterministic codegen pipeline (`streaming-arm-nars-discovery-v1.md`).
//!
//! A Rust transcoding of **Aerial+** (Karabulut, Groth, Degeler —
//! *Neurosymbolic Association Rule Mining from Tabular Data*, arXiv
//! 2504.19354v1) — with the paper's `f32` autoencoder **replaced by an
//! integer codebook distance oracle**. Aerial+'s reconstruction probe is a
//! nearest-neighbour query, and this substrate answers it exactly via the
//! palette256 distance table (`[a,b] → u32`, ρ=0.9973 vs cosine). So the
//! discovery path is **float-free and bitwise-deterministic** — no autoencoder,
//! no SGD, no seed.
//!
//! ```text
//!   tabular rows ─► one-hot codebook items
//!                ─► codebook probe: nearest consequents within θ        (Stage A)
//!                   (integer CodebookDistance oracle = palette256)
//!                ─► confirm on data: integer support/confidence counts
//!                ─► CandidateRule (integer evidence)
//!                ─► arm_to_truth_u8: (cooccur, antecedent_count) → TruthU8  (Stage B)
//!                   (= CausalEdge64 confidence_u8 + i4 mantissa)
//!                ─► {s,p,o,f,c} ndjson  ─► SPO store loader
//! ```
//!
//! # Float discipline (why this exists)
//!
//! This substrate addresses by exact codebook CAM, never by float similarity
//! (`faiss-homology-cam-pq.md`: similarity lives in the discovery layer as an
//! *integer codebook distance*, never as float, never in addressing). The
//! discovery path here is therefore all integers: codebook distance (`u32`),
//! evidence counts (`u32`), thresholds in parts-per-million. `f32` appears
//! only at the [`translator::arm_to_nars`] / [`ndjson`] edge, because the
//! downstream `spo::truth::TruthValue` and `ruff_spo_triplet::Triple` are
//! themselves `f32`; the canonical truth is the quantised
//! [`translator::TruthU8`].
//!
//! # No nondeterminism to fence
//!
//! The codebook probe is deterministic by construction (same data + oracle +
//! `theta` ⇒ identical rules, every target). Unlike the autoencoder it
//! replaced, it does not need to hide behind the ratification gate or stay out
//! of the compile path — it can join the deterministic trunk beside
//! pair-stats (D-ARM-3). The ratification council still governs *promotion to
//! the SPO store*, but no longer because of any nondeterminism here.

//! # SIMD seam
//!
//! The data-confirmation count loop transposes the window into row bitsets
//! ([`bitset::RowMasks`]) so every candidate's count is an `AND` + popcount
//! over `&[u64]` ([`simd`]). The default path is scalar (`u64::count_ones`),
//! keeping the crate std-only and independently verifiable; the optional
//! `ndarray-simd` feature routes that primitive through `ndarray::simd::U64x8`
//! per `.claude/knowledge/ndarray-vertical-simd-alien-magic.md` (zero raw
//! intrinsics in this crate). The palette256 [`aerial::CodebookDistance`]
//! oracle is similarly SIMD on the consumer side (`bgz17::batch_palette_distance`).

#![forbid(unsafe_code)]

pub mod aerial;
pub mod bitset;
pub mod encode;
pub mod ndjson;
pub mod rule;
pub mod simd;
pub mod translator;

pub use aerial::{
    antecedent_distance, extract_rules, AerialParams, AerialProposer, CodebookDistance,
    ExtractParams, MatrixDistance,
};
pub use bitset::RowMasks;
pub use encode::{Dataset, FeatureSpec};
pub use rule::{CandidateRule, Item, Proposer, PPM};
pub use translator::{
    arm_to_nars, arm_to_truth_u8, CandidateTriple, FeedProjector, NarsTruth, TruthU8,
    NARS_PERSONALITY_K,
};
