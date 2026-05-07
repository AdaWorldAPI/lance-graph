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
//!    computed without materializing the signature, via the Goursat PDE
//!    formulation (Salvi-Cass-Foster-Lyons-Lemercier 2020).
//! 5. **Codec route integration** (`codec.rs`): exposes sigker as a third
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
//! Concretely, in the lance-graph `CodecRoute` table:
//!
//! ```text
//!   Index    fields  ⇒  Passthrough  (lossless on the carrier)
//!   Index    paths   ⇒  Sigker       (lossless on tree-quotient, this crate)
//!   Argmax   fields  ⇒  CamPq        (codebook quantization, lossy)
//!   Skip     fields  ⇒  Skip         (not stored)
//! ```
//!
//! ## Relationship to jc pillars
//!
//! Sigker provides operations; jc certifies them. The natural certification
//! is a future jc pillar (Hambly-Lyons signature uniqueness on lance-graph
//! paths) which proves that sigker's "Index regime" classification is
//! mathematically warranted, not asserted. Until that pillar lands, sigker
//! ships with internal property-based tests of the algebraic identities
//! (Chen's identity, shuffle distributivity).
//!
//! ## Performance envelope
//!
//! - Truncated signature, depth 3, dim d: O(d^3) flops, O(d^3) bytes.
//!   For d=8 (typical OSINT edge feature dim): 512 floats per path.
//! - Randomized signature, projection dim k: O(k · path_length) per step,
//!   O(k) bytes total. For k=4096 (matches Vsa16kF32 carrier size after
//!   bf16 packing): comparable to deepnsm's working width.
//! - Signature kernel via Goursat PDE: O(L₁ · L₂) where Lᵢ is path length;
//!   no signature materialization required.
//!
//! ## What lives elsewhere
//!
//! - The Lance I/O / DataFusion UDF wiring lives in `lance-graph-contract`
//!   once `CodecRoute::Sigker` is added; sigker stays a pure-math crate.
//! - The certification (Hambly-Lyons uniqueness probe) lives in `crates/jc`
//!   as a future pillar 11.
//! - Comparative benches against bgz17 palette and deepnsm tiling live in
//!   `examples/sig_vs_hamming.rs`.

pub mod signature;
pub mod shuffle;
pub mod randomized;
pub mod kernel;
pub mod codec;

// ════════════════════════════════════════════════════════════════════════════
// Top-level re-exports — minimal surface, mirrors bgz17/deepnsm pattern.
// ════════════════════════════════════════════════════════════════════════════

pub use signature::{Signature, signature_truncated};
pub use shuffle::shuffle_product;
pub use randomized::{RandomizedSignature, RandomizedSignatureBuilder};
pub use kernel::signature_kernel;
pub use codec::CodecRouteSigker;
