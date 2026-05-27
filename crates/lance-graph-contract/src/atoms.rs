//! Atom layer — the bottom layer of the three-layer cognitive basis.
//!
//! # Architecture (see `EPIPHANIES.md` E-LADDER-SERVES-MAILBOX §2)
//!
//! ```text
//! atoms  (this module)          — bipolar I4-32D, 32 dims / 64 poles, orthogonal basis
//!   ↑ composed into
//! thinking styles               — I4-32D compositions over atoms  (contract::thinking)
//!   ↑ composed into
//! persona recipes               — compositions + thresholds + β  (contract::escalation)
//! ```
//!
//! The 36 [`ThinkingStyle`] variants in `contract::thinking` are the **source material**
//! from which the 32 bipolar atom dimensions are ultimately derived — they are not the
//! atoms themselves.  Derivation is via `derive_basis_ica` (see below).
//!
//! ## I4-32D encoding
//!
//! 32 dimensions packed into 16 bytes (`I4x32`).  Each dimension is a **signed 4-bit
//! value** ∈ [−8, 7].  Bipolar semantics:
//!
//! - **Sign** identifies which of the two poles is active (negative = "neg-pole",
//!   positive = "pos-pole").
//! - **Magnitude** encodes signal strength (0 = neutral / unactivated; ±7/±8 = saturated).
//!
//! 64 poles total = 32 neg-poles + 32 pos-poles.  Each pole is named in the
//! [`AtomCatalogue`].
//!
//! ## Zero-dep constraint
//!
//! This module must remain zero-dependency (same posture as the rest of
//! `lance-graph-contract`).  The SIMD hot path for `I4x32::dot` is gated on the
//! MANDATORY knowledge doc:
//! `.claude/knowledge/ndarray-vertical-simd-alien-magic.md`.
//! Do **not** add SIMD intrinsics here without first reading that document.
//!
//! ## BLOCKED markers
//!
//! Several design questions are genuinely open and are marked `// BLOCKED:` inline.
//! Do **not** resolve them by guessing — escalate to the user / truth-architect agent.
//!
//! [`ThinkingStyle`]: crate::thinking::ThinkingStyle

// BLOCKED: Atom basis (D-ATOM-0) — the 32 bipolar dimension assignments are NOT yet
// decided.  Three candidate routes exist:
//   (a) ICA/PCA over the 36 ThinkingStyle feature vectors (empirical, interpretability unclear)
//   (b) Theory-driven from the 6 existing StyleCluster entries (interpretable, asserted)
//   (c) Hybrid: seed from 6 clusters, refine against 36 for orthogonality
// Nothing in this file can carry real semantic content until D-ATOM-0 is resolved.
// See `atom-mailbox-substrate-v1.md` Decision Gates.

// BLOCKED: Feature-vector source — the 36 ThinkingStyles in `contract::thinking` expose
// only enum variants, a 23D `SparseVec` (via the `ThinkingStyleProvider` trait, which is
// an *external* trait — no concrete 23D table ships in this crate), and 7D FieldModulation
// scalars.  It is NOT confirmed that any numeric feature vectors dense enough to run
// ICA/PCA on are available inside this workspace.  Before implementing
// `derive_basis_ica`, answer: does `crewai-rust` or any other crate provide the 23D
// vectors as a concrete, accessible table?  If the only source is 23D sparse YAML cards,
// that is a **dimensionality mismatch** (23D → 32 atoms requires going *up* in
// dimensionality, which ICA cannot do without synthetic augmentation — a separate
// design decision).  Locate/confirm the concrete vector source before proceeding.

// BLOCKED: 36 → 64 poles arithmetic — 36 named ThinkingStyles fit inside 64 poles
// (32 dims × 2 poles/dim) with ~28 poles spare.  The mapping from {style_0 .. style_35}
// to pole-assignments (which dim, which pole, or spread across multiple dims as a
// composition?) is **unresolved**.  D-ATOM-0 must supply the named catalogue before
// `Atom::neg_pole` / `Atom::pos_pole` can carry real strings.

// BLOCKED: i4 quantization scale — the exact f32↔i4 scale factor (e.g. linear, per-dim,
// global) is NOT decided.  The precision-floor tradeoff is documented in
// `FormatBestPractices.md` (Jirak-grounded per-workload decision matrix; asymmetric
// quantization precedent set by `QualiaI4_16D::from_f32_17d`: pos ×7.0, neg ×8.0).
// The atom-layer scale may differ.  Cite `FormatBestPractices.md` in the implementation
// and confirm the chosen scale before replacing the `todo!()` bodies in `pack`/`unpack`.

// ---------------------------------------------------------------------------
// Core packed type
// ---------------------------------------------------------------------------

/// Packed 32-channel signed-4-bit vector — the atom-layer hot-path primitive.
///
/// Stores 32 independent signed 4-bit values in 16 contiguous bytes, two nibbles
/// per byte (little-endian nibble order: low nibble = even dim, high nibble = odd dim).
///
/// ## Bipolar semantics
///
/// Each dimension `d ∈ 0..32` holds a value `v ∈ [−8, 7]` (i4 two's complement):
///
/// - **`v == 0`** — dimension is neutral / unactivated.
/// - **`v > 0`** — pos-pole of dimension `d` is active; magnitude = `v`.
/// - **`v < 0`** — neg-pole of dimension `d` is active; magnitude = `|v|`.
///
/// The 32 bipolar dimensions span 64 named poles (32 neg + 32 pos).  These poles
/// form the **orthogonal cleanup codebook** required by `I-VSA-IDENTITIES` Test 2/3.
///
/// ## Memory layout
///
/// ```text
/// byte 0 : dim[0] in bits[3:0] (i4, sign-extended)
///          dim[1] in bits[7:4] (i4, sign-extended)
/// byte 1 : dim[2] in bits[3:0]
///          dim[3] in bits[7:4]
/// …
/// byte 15: dim[30] in bits[3:0]
///          dim[31] in bits[7:4]
/// ```
///
/// Total size: 16 bytes (128 bits).  Aligned to 16 bytes for SIMD loads.
///
/// ## Invariant
///
/// Values are clamped to [−8, 7] on `pack`; `unpack` is lossless (identity on the
/// stored nibbles).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C, align(16))]
pub struct I4x32 {
    /// Raw storage: 16 bytes, two i4 nibbles per byte.
    bytes: [u8; 16],
}

impl I4x32 {
    /// The all-zero atom vector (every dimension neutral).
    pub const ZERO: Self = Self { bytes: [0u8; 16] };

    /// Pack 32 signed 8-bit values into a packed 4-bit vector.
    ///
    /// Each `values[d]` is clamped to [−8, 7] before storing.  Values outside
    /// the i4 range are saturated (not wrapped).  Callers should pre-scale
    /// their f32 source using the chosen quantization scale (see
    /// `FormatBestPractices.md` and the `// BLOCKED: i4 quantization scale`
    /// note at the top of this module).
    ///
    /// # Arguments
    ///
    /// * `values` — exactly 32 signed byte values, one per atom dimension.
    ///   Index 0 = dim 0, index 31 = dim 31.
    pub fn pack(values: &[i8; 32]) -> Self {
        // BLOCKED: i4 quantization scale — saturation bounds confirmed as
        // [-8, 7] for i4 two's-complement, but the upstream f32→i8 conversion
        // scale is not yet decided.  See module-level BLOCKED note.
        todo!("I4x32::pack — pending D-ATOM-1 implementation after D-ATOM-0 basis resolution")
    }

    /// Unpack the 32 packed i4 values to individual signed bytes.
    ///
    /// Sign-extends each nibble to i8 (two's-complement, range [−8, 7]).
    /// The resulting array has `result[d] ∈ [−8, 7]` for all `d`.
    ///
    /// Inverse of `pack` — `unpack(pack(v)) == clamp(v, -8, 7)` elementwise.
    pub fn unpack(&self) -> [i8; 32] {
        todo!("I4x32::unpack — pending D-ATOM-1 implementation")
    }

    /// Compute the signed integer dot product of two packed I4x32 vectors.
    ///
    /// Returns the sum of elementwise products of the unpacked i4 values:
    ///
    /// ```text
    /// result = Σ_{d=0}^{31}  self[d] × other[d]
    /// ```
    ///
    /// The result fits in i32 (worst case: 32 × (−8 × 7) = −1792, or 32 × 64 = 2048).
    ///
    /// ## SIMD hot path
    ///
    /// This is the innermost loop when evaluating atom-composition affinity at scale.
    /// A vectorized implementation **MUST** read
    /// `.claude/knowledge/ndarray-vertical-simd-alien-magic.md` before writing any
    /// target-feature SIMD intrinsics — that document is the MANDATORY gate for all
    /// SIMD work in `lance-graph-contract` consumer crates.
    ///
    /// Dispatch pattern mirrors `contract::mul` (see `D-CSV-13b` in `AGENT_LOG.md`):
    /// - AVX-512F+BW path: process multiple dims per iteration via `_mm512_*` intrinsics.
    /// - NEON path: `vqtbl1q_u8` nibble-expand + `vdotq_s32` (or scalar fallback).
    /// - Scalar fallback: unpack both sides, iterate with multiply-accumulate.
    /// Runtime dispatch via `simd_caps()` singleton (`AtomicU8` cache), zero unsafe
    /// on the dispatch callsite.
    ///
    /// # BLOCKED
    ///
    /// The SIMD implementation requires the `ndarray-vertical-simd-alien-magic.md`
    /// knowledge doc to be loaded and the AVX-512 / NEON primitives confirmed against
    /// the ndarray crate's existing capability table before any `#[target_feature]`
    /// code is written here.
    pub fn dot(&self, other: &Self) -> i32 {
        // BLOCKED: SIMD path — see doc-comment above and
        // `.claude/knowledge/ndarray-vertical-simd-alien-magic.md` (MANDATORY gate).
        todo!("I4x32::dot — SIMD hot path; implement only after reading ndarray-vertical-simd-alien-magic.md")
    }
}

// ---------------------------------------------------------------------------
// Atom catalogue types
// ---------------------------------------------------------------------------

/// One named bipolar dimension in the atom catalogue.
///
/// An `Atom` identifies a single axis of the 32-dimensional atom basis.
/// Its two poles are named strings (human-readable), not enum variants —
/// the actual pole assignment comes from the D-ATOM-0 basis decision.
///
/// ## Relationship to [`ThinkingStyle`]
///
/// An `Atom` is **not** a `ThinkingStyle`.  Thinking styles are *compositions*
/// over atoms; each style is an [`I4x32`] vector where non-zero entries indicate
/// which atoms (and which poles) the style activates.  The atom layer is strictly
/// below the style layer.
///
/// [`ThinkingStyle`]: crate::thinking::ThinkingStyle
#[derive(Debug, Clone)]
pub struct Atom {
    /// Dimension index in [0, 32).
    ///
    /// Uniquely identifies which of the 32 bipolar dimensions this atom occupies.
    ///
    /// # BLOCKED
    ///
    /// Concrete index-to-name assignments depend on D-ATOM-0 (basis decision).
    pub dim: u8,

    /// Short human-readable name for the dimension (e.g. `"trust_dk"`,
    /// `"wisdom_staunen"`, `"plasticity"`).
    ///
    /// # BLOCKED
    ///
    /// Real names not assigned until D-ATOM-0 is resolved.
    pub name: &'static str,

    /// Human-readable label for the **negative pole** (value < 0 in this dim).
    ///
    /// Example: for a `"wisdom_staunen"` dimension the neg-pole might be
    /// `"Staunen"` (high-temp / diffuse / explore).
    ///
    /// # BLOCKED
    ///
    /// Pole labels not assigned until D-ATOM-0 is resolved.  The 36→64 mapping
    /// (36 named ThinkingStyles into 64 poles with ~28 spare) is an open
    /// arithmetic question — see `atom-mailbox-substrate-v1.md` §Honest gaps.
    pub neg_pole: &'static str,

    /// Human-readable label for the **positive pole** (value > 0 in this dim).
    ///
    /// Example: for the same dimension the pos-pole might be `"Wisdom"`
    /// (low-temp / sharp / exploit).
    ///
    /// # BLOCKED
    ///
    /// Same as `neg_pole` — pending D-ATOM-0.
    pub pos_pole: &'static str,
}

impl Atom {
    /// Construct a new atom entry.
    ///
    /// Panics in debug mode if `dim >= 32`.
    pub const fn new(dim: u8, name: &'static str, neg_pole: &'static str, pos_pole: &'static str) -> Self {
        // Note: `assert!(dim < 32)` cannot be `const`-evaluated for runtime panics
        // in stable Rust without const-panic; the constraint is enforced by
        // AtomCatalogue::validate() at test time.
        Self { dim, name, neg_pole, pos_pole }
    }
}

/// The 32-atom orthogonal basis — the complete cleanup codebook for the atom layer.
///
/// This catalogue satisfies `I-VSA-IDENTITIES` Test 2 (role orthogonality — each dim
/// is disjoint by index) and Test 3 (cleanup codebook — the 32 named dims form the
/// reference set against which any query vector is cleaned up by taking the nearest
/// signed integer per dim).
///
/// ## Usage
///
/// ```rust,ignore
/// let cat = AtomCatalogue::canonical();
/// let atom = &cat.atoms[5];       // look up dim 5
/// let v = my_i4x32.unpack();
/// let strength = v[atom.dim as usize];  // activation of dim 5
/// ```
///
/// ## BLOCKED — canonical instance
///
/// `AtomCatalogue::canonical()` returns a placeholder until D-ATOM-0 assigns
/// real dimension names.  Downstream callers that need the catalogue content
/// must wait for D-ATOM-0 resolution.
#[derive(Debug)]
pub struct AtomCatalogue {
    /// The 32 atom entries, one per bipolar dimension.
    ///
    /// `atoms[d].dim == d as u8` is an invariant enforced by `validate()`.
    pub atoms: [Atom; 32],
}

impl AtomCatalogue {
    /// Return the canonical 32-atom basis.
    ///
    /// # BLOCKED
    ///
    /// The returned catalogue contains placeholder names until D-ATOM-0 is
    /// resolved.  Do not depend on the string content of `Atom::name`,
    /// `Atom::neg_pole`, or `Atom::pos_pole` until the basis decision is
    /// committed.
    ///
    /// # Panics (debug)
    ///
    /// Panics in debug builds if `validate()` fails (dim index out of order).
    pub fn canonical() -> &'static Self {
        // BLOCKED: real atom names — pending D-ATOM-0.  The static below uses
        // placeholder identifiers.  Replace every "dim_N" / "pole_N_{neg,pos}"
        // with the theory-derived or ICA-derived names from D-ATOM-0.
        todo!("AtomCatalogue::canonical — pending D-ATOM-0 basis decision; \
               do not implement until dimension names are assigned")
    }

    /// Validate catalogue invariants.
    ///
    /// Checks:
    /// 1. `atoms.len() == 32` (statically guaranteed by array type).
    /// 2. `atoms[d].dim == d as u8` for all d (dim indices in order, no gaps).
    /// 3. All `name` strings are non-empty.
    ///
    /// Intended for use in `#[test]` blocks and debug assertions.
    pub fn validate(&self) -> Result<(), &'static str> {
        todo!("AtomCatalogue::validate — implement alongside canonical()")
    }

    /// Look up an atom by dimension index.
    ///
    /// Returns `None` if `dim >= 32`.
    #[inline]
    pub fn get(&self, dim: u8) -> Option<&Atom> {
        todo!("AtomCatalogue::get")
    }
}

// ---------------------------------------------------------------------------
// Basis derivation pipeline skeleton
// ---------------------------------------------------------------------------

/// Input data bundle for the ICA/PCA basis-derivation pipeline.
///
/// ## BLOCKED — source vectors
///
/// The pipeline requires **numeric feature vectors**, one per `ThinkingStyle`
/// variant.  Two candidate sources exist; NEITHER is confirmed to be accessible
/// inside `lance-graph-contract` (a zero-dep crate):
///
/// 1. **23D sparse vectors** — surfaced by `ThinkingStyleProvider::style_vector()`
///    (see `contract::thinking`).  This trait has no concrete implementation in
///    this crate.  Confirm that `crewai-rust` or another workspace member provides
///    a dense 23D array (36 × 23 = 828 f32 values) before calling this pipeline.
///    **Dimensionality mismatch warning:** 23D → 32 atoms means going *up* in
///    dimensionality; standard ICA/PCA produce at most min(N, d) = min(36, 23) = 23
///    components.  Obtaining 32 components from 23D data requires synthetic
///    augmentation (e.g. appending `FieldModulation` 7D) or a purely theory-driven
///    basis from the 6 `StyleCluster` entries.
///
/// 2. **7D FieldModulation** — `FieldModulation::to_fingerprint()` yields a 7-byte
///    vector per style.  Appending this to the 23D sparse vector gives 30D, still
///    < 32.  The last 2 dims would need explicit theory-driven assignment.
///
/// See `atom-mailbox-substrate-v1.md` §Honest gaps and D-ATOM-0 decision gate.
pub struct BasisDerivationInput {
    /// 36 × D f32 matrix of style feature vectors (row-major, one row per
    /// `ThinkingStyle` in `ThinkingStyle::ALL` order).
    ///
    /// D must be known at call time; no compile-time guarantee on D.
    ///
    /// # BLOCKED
    ///
    /// Concrete source of this matrix is unresolved — see struct-level doc.
    pub style_matrix: Vec<Vec<f32>>,

    /// Number of feature dimensions in each row of `style_matrix`.
    ///
    /// # BLOCKED
    ///
    /// Whether D == 23 (sparse style vecs), D == 30 (23 + 7D FieldMod),
    /// or something else depends on the source-confirmation step.
    pub feature_dim: usize,
}

/// Output of the basis-derivation pipeline: the 32 orthogonal component vectors.
///
/// Each component is a `D`-dimensional loading vector (the ICA/PCA unmixing row).
/// Dimension assignment maps component `c` → atom dim `c` (0-indexed).
pub struct BasisDerivationOutput {
    /// 32 × D matrix of component loadings (row-major, one row per atom dim).
    pub loadings: Vec<Vec<f32>>,

    /// Variance (or kurtosis, for ICA) explained per component, in the same
    /// order as `loadings`.
    pub component_stats: Vec<f32>,

    /// Named catalogue derived from the loadings — pole labels assigned by
    /// inspecting which ThinkingStyles load most strongly on each component.
    ///
    /// # BLOCKED
    ///
    /// Pole labelling heuristic (top-k loading styles → name the pole after the
    /// dominant cluster) is not yet specified.
    pub catalogue: AtomCatalogue,
}

/// Derive the 32-atom orthogonal basis via ICA or PCA over ThinkingStyle vectors.
///
/// # Pipeline steps (all `todo!()` — scaffold only)
///
/// 1. **Input validation** — assert `input.style_matrix.len() == 36`,
///    `input.feature_dim >= 1`, all rows have length `feature_dim`.
///
/// 2. **Centering** — subtract the column-wise mean from each feature dimension
///    (required for PCA; recommended pre-processing for ICA).
///
/// 3. **Whitening / PCA** — compute the `min(36, feature_dim)` principal
///    components of the centered matrix.  If `feature_dim < 32` this is where
///    the dimensionality mismatch must be handled — either reject with an error
///    or apply the hybrid extension strategy (see `BasisDerivationInput` doc).
///
/// 4. **ICA rotation (optional)** — rotate the PCA components to maximize
///    statistical independence (FastICA or equivalent).  Whether to use PCA-only
///    or PCA+ICA is an **open design question** (D-ATOM-0 route selection).
///
/// 5. **Pad to 32 components** — if the empirical method yields fewer than 32
///    components, pad with theory-driven axes (e.g. one axis per `StyleCluster`
///    not captured by the top components).
///
///    # BLOCKED
///    The padding strategy (which theory-driven axes to add, in which order) is
///    not specified.  This is part of the D-ATOM-0 hybrid route decision.
///
/// 6. **Pole assignment** — for each component, identify the ThinkingStyles with
///    the two strongest-magnitude loadings (one positive, one negative) and use
///    their names / cluster labels as the pole names.
///
///    # BLOCKED
///    Pole-labelling heuristic is unspecified; the 36→64 pole arithmetic
///    (36 named styles into 64 poles with ~28 spare) remains an open question.
///
/// 7. **Output packaging** — wrap loadings + stats + catalogue in
///    `BasisDerivationOutput`.
///
/// # Errors
///
/// Returns a static string error if the input matrix is malformed or the
/// dimensionality mismatch cannot be resolved.
///
/// # BLOCKED — entire function body
///
/// This function body is `todo!()`.  It cannot be implemented until:
/// - D-ATOM-0 (basis route: a / b / c) is decided.
/// - The concrete source of `input.style_matrix` is confirmed (see
///   `BasisDerivationInput` doc and module-level BLOCKED markers).
/// - The dimensionality mismatch (23D → 32 atoms) is resolved.
pub fn derive_basis_ica(input: BasisDerivationInput) -> Result<BasisDerivationOutput, &'static str> {
    // BLOCKED: see function doc.  Do not implement until D-ATOM-0 is decided
    // and the source vector table is located/confirmed.
    todo!("derive_basis_ica — pending D-ATOM-0 basis decision and source vector confirmation")
}

// ---------------------------------------------------------------------------
// Tests (structure only — bodies are todo!())
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the size and alignment invariant: I4x32 must be 16 bytes, align 16.
    #[test]
    fn test_i4x32_layout() {
        assert_eq!(core::mem::size_of::<I4x32>(), 16);
        assert_eq!(core::mem::align_of::<I4x32>(), 16);
    }

    /// Zero vector unpacks to all-zero i8 array.
    #[test]
    fn test_i4x32_zero_unpack() {
        todo!("test_i4x32_zero_unpack — pending I4x32::unpack implementation")
    }

    /// pack(v).unpack() == clamp(v, -8, 7) elementwise.
    #[test]
    fn test_i4x32_pack_unpack_roundtrip() {
        todo!("test_i4x32_pack_unpack_roundtrip — pending I4x32::pack + unpack")
    }

    /// Saturation: values outside [-8, 7] are clamped on pack.
    #[test]
    fn test_i4x32_pack_saturates() {
        todo!("test_i4x32_pack_saturates — pending I4x32::pack")
    }

    /// dot(ZERO, ZERO) == 0.
    #[test]
    fn test_i4x32_dot_zero() {
        todo!("test_i4x32_dot_zero — pending I4x32::dot")
    }

    /// dot(v, v) == sum of squares, within i4 range.
    #[test]
    fn test_i4x32_dot_self() {
        todo!("test_i4x32_dot_self — pending I4x32::dot")
    }

    /// AtomCatalogue::validate() passes on the canonical catalogue.
    #[test]
    fn test_atom_catalogue_validates() {
        // BLOCKED: depends on AtomCatalogue::canonical() which is blocked on D-ATOM-0.
        todo!("test_atom_catalogue_validates — pending AtomCatalogue::canonical (D-ATOM-0)")
    }

    /// All 32 atoms have distinct dim indices 0..32 in canonical order.
    #[test]
    fn test_atom_catalogue_dim_ordering() {
        // BLOCKED: same as above.
        todo!("test_atom_catalogue_dim_ordering — pending D-ATOM-0")
    }
}
