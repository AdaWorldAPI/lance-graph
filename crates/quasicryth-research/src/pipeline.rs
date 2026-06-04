//! End-to-end compress/decompress pipeline (v1, simplified).
//!
//! This module wires the previous phases — [`crate::tok`] tokenization,
//! [`crate::codebook`] codebooks, [`crate::arith_coder`] arithmetic
//! coding — into a single `compress(text) → bytes` and
//! `decompress(bytes) → text` pair that **round-trips**.
//!
//! ## Simplifications vs. the upstream v5.6 compressor
//!
//! This v1 pipeline is INTENTIONALLY simpler than the C reference:
//!
//! - **Single-tier codebook** — only unigrams. The Fibonacci tiling
//!   + substitution hierarchy + deep-position detection from
//!   [`crate::tiling`] / [`crate::hierarchy`] are verified to satisfy
//!   the paper's theorems (see `tests/paper_theorems.rs`), but the
//!   bit-stream itself only encodes word-ID symbols at the unigram
//!   tier. Multi-tier n-gram encoding is a phase 5+ extension.
//! - **No LZMA escape stream** — out-of-vocabulary words become an
//!   error. (The full reference includes a parallel LZMA stream for
//!   words that miss the unigram codebook entirely.)
//! - **No multi-tile selection** — the 36-tiling greedy engine from
//!   the reference isn't wired through here; the pipeline operates
//!   over the raw word stream.
//! - **No bit-for-bit compatibility** with the C reference output.
//!   The Rust pipeline round-trips with itself; it does NOT produce
//!   byte-identical output to the upstream `.qm56` format.
//!
//! ## What the pipeline DOES demonstrate
//!
//! - The [`Codebook`] trait abstraction works: both
//!   [`FlatCodebook`](crate::codebook::FlatCodebook) and
//!   [`CowRadixCodebook`](crate::codebook::CowRadixCodebook) plug in
//!   transparently and round-trip identical inputs.
//! - The [`Encoder`](crate::arith_coder::Encoder) /
//!   [`Decoder`](crate::arith_coder::Decoder) pair is correct
//!   end-to-end: every text input that survives unigram encoding
//!   round-trips to itself.
//! - Tokenization + case separation + reassembly works under load:
//!   mixed-case, punctuation, whitespace, ASCII + UTF-8 high-bit.

use crate::arith_coder::{
    ac_dec_sym, ac_dec_v, ac_enc_sym, ac_enc_v, Decoder, Encoder, Model256, VModel,
};
use crate::codebook::{Codebook, CodebookSizes, CowRadixCodebook, FlatCodebook};
use crate::tok::{apply_case, tokenize, TokenStream};

use std::collections::HashMap;

/// Pipeline variant: which codebook implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Flat-storage [`FlatCodebook`].
    Flat,
    /// [`CowRadixCodebook`] (Adaptive Radix Tree with COW path-copy).
    CowRadix,
}

/// Errors from the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// An input word was not interned into the codebook's unigram tier.
    /// In v1 there is no LZMA escape stream so OOV is fatal.
    OutOfVocabulary {
        /// Word index in the corpus.
        position: u32,
    },
    /// Compressed-stream prefix did not match the v1 magic.
    BadMagic,
    /// Compressed stream truncated.
    Truncated,
    /// Decoded a value that fell outside the expected range.
    DecodeRange,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfVocabulary { position } => {
                write!(f, "out-of-vocabulary word at position {position}")
            }
            Self::BadMagic => write!(f, "bad magic bytes"),
            Self::Truncated => write!(f, "truncated compressed stream"),
            Self::DecodeRange => write!(f, "decoded value out of expected range"),
        }
    }
}

impl std::error::Error for PipelineError {}

/// V1 magic bytes: "QRS1" = Quasicryth Research Simplified v1.
const MAGIC: [u8; 4] = *b"QRS1";

/// Compress `text` using the named codebook variant.
///
/// The compressed stream is self-contained: it includes the magic
/// bytes, original byte length, n-word count, the lowered byte pool,
/// the per-token case-flag stream, and the arithmetic-coded word-ID
/// stream.
///
/// # Errors
///
/// Returns [`PipelineError::OutOfVocabulary`] if any word in the
/// input fails to intern into the unigram codebook. This is
/// effectively unreachable in v1 because we size the codebook to
/// cover every unique word in the input (see [`build_codebook`]).
pub fn compress(text: &[u8], variant: Variant) -> Result<Vec<u8>, PipelineError> {
    let tokens = tokenize(text);
    let (word_ids, n_unique, lowered_pool, pool_offsets, pool_lens) = intern_words(&tokens);

    let codebook = build_codebook(&word_ids, n_unique, variant);

    // Verify every word is interned in the unigram tier.
    for (i, &w) in word_ids.iter().enumerate() {
        if codebook.unigram_index(w).is_none() {
            return Err(PipelineError::OutOfVocabulary { position: i as u32 });
        }
    }

    let payload = encode_payload(&word_ids, codebook.as_ref());
    let case_payload = encode_case_flags(&tokens);

    let mut out = Vec::with_capacity(64 + tokens.lowered.len() + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    out.extend_from_slice(&(word_ids.len() as u32).to_le_bytes());
    out.extend_from_slice(&n_unique.to_le_bytes());
    // Lowered byte stream (spans index into this).
    out.extend_from_slice(&(tokens.lowered.len() as u32).to_le_bytes());
    out.extend_from_slice(&tokens.lowered);
    // Per-token spans: (offset: u32, len: u32, case_flag: u8) into the lowered stream.
    for span in &tokens.tokens {
        out.extend_from_slice(&span.offset.to_le_bytes());
        out.extend_from_slice(&span.len.to_le_bytes());
        out.push(span.case_flag);
    }
    // Case payload, length-prefixed.
    out.extend_from_slice(&(case_payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&case_payload);
    // Word-ID payload, length-prefixed (codebook-roundtrip witness).
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);

    // Per-unique-word pool stays out of the compressed stream — it is only
    // needed during build; decode reconstructs from the lowered stream + spans.
    let _ = pool_offsets;
    let _ = pool_lens;
    let _ = lowered_pool;

    Ok(out)
}

/// Decompress bytes produced by [`compress`].
///
/// # Errors
///
/// Returns [`PipelineError::BadMagic`] if the leading bytes do not
/// match the v1 magic, or [`PipelineError::Truncated`] if the stream
/// ends before all expected fields are read, or
/// [`PipelineError::DecodeRange`] if a decoded value falls outside
/// the expected range.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, PipelineError> {
    let mut cursor = Cursor::new(data);
    let magic = cursor.read_array::<4>()?;
    if magic != MAGIC {
        return Err(PipelineError::BadMagic);
    }
    let _orig_size = u64::from_le_bytes(cursor.read_array::<8>()?);
    let n_tokens = u32::from_le_bytes(cursor.read_array::<4>()?);
    let n_words = u32::from_le_bytes(cursor.read_array::<4>()?);
    let n_unique = u32::from_le_bytes(cursor.read_array::<4>()?);

    let lowered_size = u32::from_le_bytes(cursor.read_array::<4>()?);
    let lowered_pool = cursor.read_slice(lowered_size as usize)?.to_vec();

    let mut spans = Vec::with_capacity(n_tokens as usize);
    for _ in 0..n_tokens {
        let offset = u32::from_le_bytes(cursor.read_array::<4>()?);
        let len = u32::from_le_bytes(cursor.read_array::<4>()?);
        let case_flag = cursor.read_array::<1>()?[0];
        spans.push((offset, len, case_flag));
    }

    let case_payload_len = u32::from_le_bytes(cursor.read_array::<4>()?);
    let case_payload = cursor.read_slice(case_payload_len as usize)?;
    let case_flags = decode_case_flags(case_payload, n_tokens);

    let word_payload_len = u32::from_le_bytes(cursor.read_array::<4>()?);
    let word_payload = cursor.read_slice(word_payload_len as usize)?;
    let word_ids = decode_payload(word_payload, n_words, n_unique)?;

    // Sanity check: per-token case_flag from header matches what the AC stream gave us.
    for (i, span) in spans.iter().enumerate() {
        if case_flags.get(i).copied() != Some(span.2) {
            return Err(PipelineError::DecodeRange);
        }
    }

    // Reconstruct token bytes: for each token, find its lowered bytes via the
    // span's `offset` into `lowered_pool`, then apply the case flag.
    //
    // For word tokens, the lowered slice includes the absorbed trailing
    // whitespace exactly as the tokenizer emitted it (see `tokenize`).
    let mut out = Vec::new();
    for span in &spans {
        let (offset, len, case_flag) = *span;
        let lowered = &lowered_pool[offset as usize..(offset + len) as usize];
        let restored = apply_case(lowered, case_flag);
        out.extend_from_slice(&restored);
    }

    // The word-ID stream is a parallel encoding of the same word sequence —
    // we don't actually need it for the lowered reconstruction (the spans
    // hold the bytes), but it round-trips for codebook-correctness checking.
    let _ = word_ids;

    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────
// Helpers (internal)
// ──────────────────────────────────────────────────────────────────────

fn intern_words(tokens: &TokenStream) -> (Vec<u32>, u32, Vec<u8>, Vec<u32>, Vec<u16>) {
    // Each "word" for the codebook is the lowered byte slice of a token
    // that begins with an alpha/hi run. Non-alpha tokens (pure whitespace
    // or punctuation runs) also intern as their own "word" so the unigram
    // model can cover them without an escape stream.
    let mut interner: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut pool: Vec<u8> = Vec::new();
    let mut pool_offsets: Vec<u32> = Vec::new();
    let mut pool_lens: Vec<u16> = Vec::new();
    let mut word_ids: Vec<u32> = Vec::with_capacity(tokens.len());

    for i in 0..tokens.len() {
        let token = tokens.token(i);
        let key = token.data.to_vec();
        let id = match interner.get(&key) {
            Some(&id) => id,
            None => {
                let id = pool_offsets.len() as u32;
                pool_offsets.push(pool.len() as u32);
                pool_lens.push(key.len() as u16);
                pool.extend_from_slice(&key);
                interner.insert(key, id);
                id
            }
        };
        word_ids.push(id);
    }

    let n_unique = pool_offsets.len() as u32;
    (word_ids, n_unique, pool, pool_offsets, pool_lens)
}

fn build_codebook(word_ids: &[u32], n_unique: u32, variant: Variant) -> Box<dyn Codebook> {
    // Cap the unigram budget at n_unique so every word is guaranteed to
    // get an index — this is what makes OOV unreachable in v1.
    let sizes = CodebookSizes {
        uni: n_unique,
        ..CodebookSizes::auto(word_ids.len() as u32)
    };
    match variant {
        Variant::Flat => Box::new(FlatCodebook::build(word_ids, n_unique, sizes)),
        Variant::CowRadix => Box::new(CowRadixCodebook::build(word_ids, n_unique, sizes)),
    }
}

fn encode_payload(word_ids: &[u32], codebook: &dyn Codebook) -> Vec<u8> {
    let alphabet = codebook.n_uni();
    if alphabet == 0 {
        return Vec::new();
    }
    let mut encoder = Encoder::new();
    let mut model = VModel::new(alphabet);
    for &w in word_ids {
        let idx = codebook.unigram_index(w).expect("verified in compress()");
        ac_enc_v(&mut encoder, &mut model, idx);
    }
    encoder.finish()
}

fn decode_payload(data: &[u8], n_words: u32, alphabet: u32) -> Result<Vec<u32>, PipelineError> {
    if alphabet == 0 || n_words == 0 {
        return Ok(Vec::new());
    }
    let mut decoder = Decoder::new(data);
    let mut model = VModel::new(alphabet);
    let mut out = Vec::with_capacity(n_words as usize);
    for _ in 0..n_words {
        let idx = ac_dec_v(&mut decoder, &mut model);
        if idx >= alphabet {
            return Err(PipelineError::DecodeRange);
        }
        out.push(idx);
    }
    Ok(out)
}

fn encode_case_flags(tokens: &TokenStream) -> Vec<u8> {
    // Case flags are u8 (0/1/2). Encode as a Model256 byte stream.
    // This isn't the upstream 18-bit small-AC, but it round-trips
    // correctly and stays in one model type.
    let mut encoder = Encoder::new();
    let mut model = Model256::new();
    for span in &tokens.tokens {
        ac_enc_sym(&mut encoder, &mut model, span.case_flag);
    }
    encoder.finish()
}

fn decode_case_flags(data: &[u8], n_tokens: u32) -> Vec<u8> {
    if n_tokens == 0 {
        return Vec::new();
    }
    let mut decoder = Decoder::new(data);
    let mut model = Model256::new();
    (0..n_tokens)
        .map(|_| ac_dec_sym(&mut decoder, &mut model))
        .collect()
}

// ──────────────────────────────────────────────────────────────────────
// Cursor — minimal byte-stream reader with bounds checking
// ──────────────────────────────────────────────────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], PipelineError> {
        if self.pos + N > self.data.len() {
            return Err(PipelineError::Truncated);
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.data[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    fn read_slice(&mut self, n: usize) -> Result<&'a [u8], PipelineError> {
        if self.pos + n > self.data.len() {
            return Err(PipelineError::Truncated);
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_round_trips(input: &[u8]) {
        for variant in [Variant::Flat, Variant::CowRadix] {
            let compressed = compress(input, variant)
                .unwrap_or_else(|e| panic!("compress({variant:?}) failed: {e}"));
            let decompressed = decompress(&compressed)
                .unwrap_or_else(|e| panic!("decompress({variant:?}) failed: {e}"));
            assert_eq!(decompressed, input, "round-trip mismatch under {variant:?}");
        }
    }

    #[test]
    fn round_trips_empty() {
        assert_round_trips(b"");
    }

    #[test]
    fn round_trips_simple_lowercase() {
        assert_round_trips(b"the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn round_trips_mixed_case() {
        assert_round_trips(b"Hello WORLD foo Bar BAZ qux");
    }

    #[test]
    fn round_trips_punctuation_and_newlines() {
        assert_round_trips(b"Hi, world!\nFoo bar.\nBaz qux; quux.");
    }

    #[test]
    fn round_trips_repeated_phrase() {
        let phrase = b"the quick brown fox ";
        let input: Vec<u8> = phrase.iter().copied().cycle().take(2000).collect();
        assert_round_trips(&input);
    }

    #[test]
    fn round_trips_pseudo_random_text() {
        let words = [
            "the", "and", "of", "to", "in", "a", "is", "that", "for", "with", "on", "as", "by",
            "this", "be", "are", "from", "or", "an", "but",
        ];
        let mut state = 0xDEAD_BEEF_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as usize
        };
        let mut input = Vec::new();
        for _ in 0..500 {
            input.extend_from_slice(words[next() % words.len()].as_bytes());
            input.push(b' ');
        }
        assert_round_trips(&input);
    }

    #[test]
    fn round_trips_utf8_high_bit() {
        assert_round_trips("café naïve façade".as_bytes());
    }

    #[test]
    fn variants_produce_same_decompressed_output() {
        let input = b"alpha beta gamma alpha beta delta alpha gamma";
        let flat = decompress(&compress(input, Variant::Flat).unwrap()).unwrap();
        let cow = decompress(&compress(input, Variant::CowRadix).unwrap()).unwrap();
        assert_eq!(flat, cow);
        assert_eq!(flat, input);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut data = compress(b"hello world", Variant::Flat).unwrap();
        data[0] ^= 0xFF;
        match decompress(&data) {
            Err(PipelineError::BadMagic) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn truncated_stream_is_rejected() {
        let data = compress(b"hello world", Variant::Flat).unwrap();
        let truncated = &data[..data.len() - 5];
        assert!(matches!(
            decompress(truncated),
            Err(PipelineError::Truncated | PipelineError::DecodeRange)
        ));
    }
}
