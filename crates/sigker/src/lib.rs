//! # sigker — Path-Signature Representations
//!
//! Algebraic representation of sequential / path-structured data via Chen-Lyons
//! signatures and their randomized projections. Peer to `bgz17` (palette-indexed
//! distance) and `deepnsm` (NSM tiling): a *third* encoding lane in the codec
//! routing table, with **Index-regime** semantics.
//!
//! ## What this crate provides
//!
//! 1. **Truncated path signatures** (`signature.rs`): the iterated-integrals
//!    feature map S(X) = (1, ∫dX, ∫∫dX⊗dX, …) up to a chosen depth N.
//! 2. **Shuffle product algebra** (`shuffle.rs`): the algebraic operation that
//!    makes signatures composable — analogous to VSA bind/bundle, but with
//!    proven uniqueness (Hambly-Lyons 2010).
//! 3. **Randomized signatures** (`randomized.rs`): finite-dimensional
//!    projections of the signature with proven universality (Cuchiero-Cuchiero-
//!    Schmocker-Teichmann 2021); the practical bridge to fixed-width
//!    fingerprints comparable to Vsa16k.
//! 4. **Signature kernels** (`kernel.rs`): inner product 〈S(X), S(Y)〉
//!    computed two ways — truncated direct (depth-bounded) and via the
//!    **full Goursat PDE solver** (depth-∞ in O(T₁·T₂) flops, no signature
//!    materialization). The PDE solver is the production path for any
//!    workload that exceeds depth-6 truncation.
//! 5. **Log-signatures** (`log_signature.rs`): compact storage of the
//!    truncated signature in the Lyndon-basis of the free Lie algebra.
//!    Compression 7–13× depending on (d, N), with NO information loss.
//! 6. **Codec route integration** (`codec.rs`): exposes sigker as a third
//!    `CodecRoute` variant alongside Passthrough and CamPq. Sigker is
//!    **Index regime** — by Hambly-Lyons uniqueness, it is lossless on
//!    tree-quotient classes of paths.
//!
//! ## Why sigker is Index regime, not Argmax
//!
//! VSA bundling is *approximate* — the bundle ⊕ is not injective in finite
//! dimension and noise accumulates with each bind/bundle (this is what jc
//! Pillar 5 / Jirak 2016 quantifies). Path signatures are *exact* up to
//! tree-like equivalence (Hambly-Lyons 2010). Two paths produce the same
//! signature iff they are equal modulo reparametrization and tree-like
//! cancellations — and the latter equivalence is the *intended* identification
//! for any path-as-information consumer.
//!
//! ## Relationship to jc pillars
//!
//! Sigker provides operations; jc certifies them. The natural certification
//! is jc Pillar 11 (Hambly-Lyons signature uniqueness on lance-graph paths)
//! which proves that sigker's "Index regime" classification is mathematically
//! warranted, not asserted. Pillar 11 is currently DEFERRED in jc; it
//! activates once sigker is benchmarked at production carrier widths.

pub mod signature;
pub mod shuffle;
pub mod randomized;
pub mod kernel;
pub mod codec;
pub mod log_signature;

pub use signature::{Signature, signature_truncated};
pub use shuffle::shuffle_product;
pub use randomized::{RandomizedSignature, RandomizedSignatureBuilder};
pub use kernel::{signature_kernel, signature_kernel_pde};
pub use codec::CodecRouteSigker;
pub use log_signature::{
    enumerate_lyndon_words, log_signature_truncated, witt_component, witt_dimension,
    LogSignature,
};
