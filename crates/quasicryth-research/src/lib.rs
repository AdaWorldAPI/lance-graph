//! # quasicryth-research
//!
//! Direct Rust transcode of the **algebraic core** of Quasicryth (Tacconelli
//! 2026, [arxiv 2603.14999](https://arxiv.org/abs/2603.14999),
//! upstream <https://github.com/robtacconelli/quasicryth>, v5.6.0).
//!
//! ## What this is
//!
//! A zero-dependency Rust crate that transcodes `fib.h` + `fib.c` + the
//! algebraic types from `qtc.h` of the reference C implementation. Three
//! operations:
//!
//! 1. **Tiling generation** — cut-and-project with arbitrary irrational α
//!    ([`tiling::qc_word_tiling_alpha`]) plus five substitution-rule
//!    families ([`tiling::thue_morse_tiling`], [`tiling::rudin_shapiro_tiling`],
//!    [`tiling::period_doubling_tiling`], [`tiling::period5_tiling`],
//!    [`tiling::sanddrift_tiling`]). The 36-tiling multi-engine descriptor
//!    table is exposed via [`constants::tiling_descs`].
//!
//! 2. **Substitution hierarchy** — iterative deflation
//!    `(L, S) → super-L`, `L → super-S` via [`hierarchy::build_hierarchy`].
//!    The Fibonacci tiling's hierarchy `never collapses` (Thm 2);
//!    Period-5's hierarchy collapses at level 3 (Cor 4). Tests in
//!    [`hierarchy`] verify both behaviours on synthetic data.
//!
//! 3. **Deep-position detection** — upward pass marking each level-0 L-tile
//!    as a legal n-gram entry point at each hierarchy level it qualifies for
//!    ([`hierarchy::detect_deep_positions`]).
//!
//! ## What this is NOT
//!
//! The reference C compressor (v5.6) ships a full pipeline: word
//! tokenization (`tok.c`), case separation, multi-level adaptive arithmetic
//! coding (`ac.c`), two-tier unigram model, word-level LZ77, codebook
//! construction (`cb.c`), LZMA escape stream, header assembly and MD5
//! checksumming. **None of that is in scope here.**
//!
//! This crate is for **research and testing**: verifying the paper's
//! theorems (non-collapse, PV-property, Sturmian minimality, Golden
//! Compensation, bounded overhead) and cross-checking the workspace's
//! `bgz17` + `helix` φ-substrate decisions against the reference algebra.
//!
//! ## License
//!
//! The upstream C reference is published by Roberto Tacconelli under terms
//! at the upstream repository. This transcode preserves attribution in
//! every module that maps to a specific reference file; refer to the
//! upstream repo for the canonical license text.

#![deny(unsafe_code)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(missing_docs)]
// Allowed pragmatic lints — research crate, transcoded from C, prefer
// legibility against the upstream over chasing pedantic stylistic flags.
#![allow(clippy::cast_possible_truncation)] // u32/u8 width transcoded from C
#![allow(clippy::cast_sign_loss)] // same
#![allow(clippy::cast_precision_loss)] // f64 conversions in tile counting
#![allow(clippy::cast_lossless)] // `as f64` reads cleaner than `f64::from(_)` here
#![allow(clippy::int_plus_one)] // `x + 1 <= y` is the trailing-edge guard idiom
#![allow(clippy::excessive_precision)] // f64 literals shown to full mantissa
#![allow(clippy::doc_markdown)] // decimal mantissas don't need backticks
#![allow(clippy::needless_range_loop)] // index-based loops match the C layout
#![allow(clippy::manual_midpoint)] // recomputing φ algebraically, not midpointing
#![allow(clippy::module_name_repetitions)] // matches upstream naming
#![allow(clippy::many_single_char_names)] // a/b/c/d are RFC 1321's own names
#![allow(clippy::too_many_lines)] // upstream C functions transcode 1:1
#![allow(clippy::format_push_string)] // hex formatting test helper, not hot path
#![allow(clippy::bool_to_int_with_if)] // explicit `is_l ? 2 : 1` matches the C
#![allow(clippy::assigning_clones)] // clone-into would obscure ownership intent
#![allow(clippy::single_match_else)] // explicit match reads cleaner here
#![allow(clippy::only_used_in_recursion)] // self-recursive insert keeps trie context
#![allow(clippy::doc_lazy_continuation)] // module-level docs use multi-line list items

pub mod arith_coder;
pub mod codebook;
pub mod constants;
pub mod hierarchy;
pub mod md5;
pub mod pipeline;
pub mod tiling;
pub mod tok;
pub mod types;

// Re-exports of the most common entry points.
pub use arith_coder::{
    ac_dec_sym, ac_dec_v, ac_enc_sym, ac_enc_v, Decoder, Encoder, Model256, VModel, AC_FULL,
    AC_HALF, AC_MAX_FREQ, AC_PREC, AC_QTR,
};
pub use codebook::{Codebook, CodebookSizes, CowArt, CowRadixCodebook, FlatCodebook, NG_LENS};
pub use constants::{tiling_descs, HIER_WORD_LENS, INV_PHI, MAX_HIER, N_TILINGS, PHI};
pub use hierarchy::{build_hierarchy, deep_counts, detect_deep_positions, hier_context};
pub use md5::{md5, Md5};
pub use pipeline::{compress, decompress, PipelineError, Variant};
pub use tiling::{
    gen_from_desc, period5_tiling, period_doubling_tiling, qc_word_tiling, qc_word_tiling_alpha,
    rudin_shapiro_tiling, sanddrift_tiling, thue_morse_tiling, verify_no_adjacent_s,
};
pub use tok::{
    apply_case, is_alpha_or_hi, is_ws, tokenize, word_split, Token, TokenSpan, TokenStream,
};
pub use types::{DeepPositions, HLevel, Hierarchy, ParentMap, Tile, TilingDesc};
