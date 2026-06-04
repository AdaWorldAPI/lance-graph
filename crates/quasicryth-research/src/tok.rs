//! Tokenization + case separation + word splitting — transcoded from `tok.c`.
//!
//! Three operations:
//!
//! 1. [`tokenize`] — split raw input bytes into [`Token`]s, lowercase
//!    everything, attach a case flag (`0` = lowercase, `1` = first-letter
//!    capitalized, `2` = ALL UPPERCASE). Returns the lowered byte stream
//!    + the token list.
//!
//! 2. [`word_split`] — split a pre-lowered byte stream into word tokens
//!    (no case work). Lighter-weight alternative when case has already
//!    been stripped.
//!
//! 3. [`apply_case`] — reverse the case transformation for a single
//!    token given its flag. Round-trip with [`tokenize`].
//!
//! The case-flag arithmetic-coding entry points (`enc_case` / `dec_case`
//! in upstream) are deferred to a later phase when the arithmetic coder
//! itself is transcoded.

/// One token after case separation.
///
/// `data` is the **lowered** byte slice — view into the buffer returned
/// by [`tokenize`]. `case_flag` recovers the original case via [`apply_case`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    /// Lowered byte content of the token.
    pub data: &'a [u8],
    /// Case flag: 0 = lower, 1 = first-letter Cap, 2 = ALL UPPER.
    pub case_flag: u8,
}

/// True for ASCII alpha or any high-bit byte (UTF-8 continuation /
/// multi-byte alpha).
#[inline]
#[must_use]
pub const fn is_alpha_or_hi(c: u8) -> bool {
    c.is_ascii_alphabetic() || c >= 128
}

/// True for ASCII whitespace (space, LF, CR, TAB).
#[inline]
#[must_use]
pub const fn is_ws(c: u8) -> bool {
    c == 32 || c == 10 || c == 13 || c == 9
}

/// Apply the case flag back to a lowered byte slice.
///
/// `flag = 0` → unchanged. `flag = 1` → uppercase the first ASCII alpha.
/// `flag = 2` → uppercase all bytes (ASCII rules — high-bit bytes pass through).
#[must_use]
pub fn apply_case(data: &[u8], flag: u8) -> Vec<u8> {
    let mut out = data.to_vec();
    match flag {
        0 => out,
        2 => {
            for b in &mut out {
                *b = b.to_ascii_uppercase();
            }
            out
        }
        _ => {
            // flag == 1 (and any other value): capitalize first ASCII alpha.
            for b in &mut out {
                if b.is_ascii_alphabetic() {
                    *b = b.to_ascii_uppercase();
                    break;
                }
            }
            out
        }
    }
}

/// Internal: verify that `apply_case(lowered, flag) == original`.
fn case_roundtrips(orig: &[u8], lowered: &[u8], flag: u8) -> bool {
    apply_case(lowered, flag) == orig
}

/// Result of tokenizing a raw byte stream.
#[derive(Debug, Clone)]
pub struct TokenStream {
    /// The full lowered byte stream. Tokens index into this buffer.
    pub lowered: Vec<u8>,
    /// Per-token: `(start_offset_in_lowered, length, case_flag)`.
    ///
    /// We store offsets (not borrowed slices) so the result is `'static`
    /// and easy to round-trip through serialization.
    pub tokens: Vec<TokenSpan>,
}

/// Span of one token inside [`TokenStream::lowered`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan {
    /// Offset into the lowered byte stream.
    pub offset: u32,
    /// Length in bytes.
    pub len: u32,
    /// Case flag (0/1/2).
    pub case_flag: u8,
}

impl TokenStream {
    /// Borrow the `i`-th token as a [`Token`] referencing the lowered buffer.
    #[must_use]
    pub fn token(&self, i: usize) -> Token<'_> {
        let s = self.tokens[i];
        Token {
            data: &self.lowered[s.offset as usize..(s.offset + s.len) as usize],
            case_flag: s.case_flag,
        }
    }

    /// Number of tokens.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// True iff there are no tokens.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Reconstruct the original input by concatenating
    /// `apply_case(token.data, token.case_flag)` over every token.
    ///
    /// This is the round-trip the upstream verifies in `case_roundtrips`.
    #[must_use]
    pub fn round_trip(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.lowered.len());
        for i in 0..self.len() {
            let t = self.token(i);
            out.extend_from_slice(&apply_case(t.data, t.case_flag));
        }
        out
    }
}

/// Tokenize raw input bytes with case separation.
///
/// Returns the lowered byte stream + per-token spans. Each "token"
/// consumes one alpha-or-hi run (with any trailing whitespace) OR one
/// non-alpha run.
#[must_use]
pub fn tokenize(data: &[u8]) -> TokenStream {
    let len = data.len();
    let mut lowered = Vec::with_capacity(len);
    let mut tokens = Vec::with_capacity(len / 4 + 16);
    let mut i = 0;

    while i < len {
        if is_alpha_or_hi(data[i]) {
            // Word part: alpha-or-hi run.
            let mut j = i + 1;
            while j < len && is_alpha_or_hi(data[j]) {
                j += 1;
            }
            // Trailing whitespace absorbed.
            let mut k = j;
            while k < len && is_ws(data[k]) {
                k += 1;
            }
            let wp = &data[i..j];

            // Determine case flag from the word-only part.
            let all_lower = wp.iter().all(|b| !b.is_ascii_uppercase());
            let case_flag: u8 = if all_lower {
                0
            } else {
                let all_upper = wp.iter().all(|b| !b.is_ascii_lowercase());
                if all_upper && wp.len() > 1 {
                    2
                } else if wp.first().is_some_and(u8::is_ascii_uppercase) {
                    1
                } else {
                    0
                }
            };

            // Build the lowered token (including trailing whitespace).
            let low_start = lowered.len() as u32;
            for &b in &data[i..k] {
                lowered.push(b.to_ascii_lowercase());
            }
            let low_end = lowered.len();
            let low_token = &lowered[low_start as usize..low_end];

            // Verify round-trip; on mismatch, retry with case_flag = 0 and the
            // original bytes (matches the C fallback at tok.c:114).
            if case_roundtrips(&data[i..k], low_token, case_flag) {
                tokens.push(TokenSpan {
                    offset: low_start,
                    len: (k - i) as u32,
                    case_flag,
                });
            } else {
                // Roll back the lowercased bytes.
                lowered.truncate(low_start as usize);
                for &b in &data[i..k] {
                    lowered.push(b);
                }
                tokens.push(TokenSpan {
                    offset: low_start,
                    len: (k - i) as u32,
                    case_flag: 0,
                });
            }
            i = k;
        } else {
            // Non-alpha run.
            let mut j = i + 1;
            while j < len && is_ws(data[j]) {
                j += 1;
            }
            let low_start = lowered.len() as u32;
            lowered.extend_from_slice(&data[i..j]);
            tokens.push(TokenSpan {
                offset: low_start,
                len: (j - i) as u32,
                case_flag: 0,
            });
            i = j;
        }
    }

    TokenStream { lowered, tokens }
}

/// Word-split a pre-lowered byte stream (no case work).
///
/// Each word is either an alpha-or-hi run with trailing whitespace
/// absorbed, OR a single non-alpha byte with trailing whitespace.
///
/// Returns `(start_offset, length)` pairs that index into `data`.
#[must_use]
pub fn word_split(data: &[u8]) -> Vec<(u32, u32)> {
    let len = data.len();
    let mut out = Vec::with_capacity(len / 4 + 16);
    let mut i = 0;
    while i < len {
        let start = i as u32;
        if is_alpha_or_hi(data[i]) {
            let mut j = i + 1;
            while j < len && is_alpha_or_hi(data[j]) {
                j += 1;
            }
            let mut k = j;
            while k < len && is_ws(data[k]) {
                k += 1;
            }
            out.push((start, (k - i) as u32));
            i = k;
        } else {
            let mut j = i + 1;
            while j < len && is_ws(data[j]) {
                j += 1;
            }
            out.push((start, (j - i) as u32));
            i = j;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{apply_case, is_alpha_or_hi, is_ws, tokenize, word_split};

    #[test]
    fn is_alpha_or_hi_covers_ascii_alpha_and_high() {
        assert!(is_alpha_or_hi(b'a'));
        assert!(is_alpha_or_hi(b'Z'));
        assert!(is_alpha_or_hi(0xC3)); // UTF-8 lead byte
        assert!(is_alpha_or_hi(0xFF));
        assert!(!is_alpha_or_hi(b' '));
        assert!(!is_alpha_or_hi(b'1'));
    }

    #[test]
    fn is_ws_only_matches_four_chars() {
        for c in [b' ', b'\n', b'\r', b'\t'] {
            assert!(is_ws(c));
        }
        for c in [b'a', b'1', 0u8, 11u8] {
            assert!(!is_ws(c));
        }
    }

    #[test]
    fn apply_case_zero_is_identity() {
        assert_eq!(apply_case(b"hello ", 0), b"hello ");
    }

    #[test]
    fn apply_case_one_capitalizes_first() {
        assert_eq!(apply_case(b"hello ", 1), b"Hello ");
        // Leading non-alpha skipped.
        assert_eq!(apply_case(b" hello", 1), b" Hello");
    }

    #[test]
    fn apply_case_two_uppercases_all() {
        assert_eq!(apply_case(b"hello ", 2), b"HELLO ");
    }

    #[test]
    fn tokenize_round_trips_lowercase() {
        let input = b"the quick brown fox";
        let stream = tokenize(input);
        assert_eq!(stream.round_trip(), input);
        assert_eq!(stream.len(), 4);
        for i in 0..stream.len() {
            assert_eq!(stream.token(i).case_flag, 0);
        }
    }

    #[test]
    fn tokenize_round_trips_mixed_case() {
        let input = b"Hello WORLD foo";
        let stream = tokenize(input);
        assert_eq!(stream.round_trip(), input);
        assert_eq!(stream.token(0).case_flag, 1);
        assert_eq!(stream.token(1).case_flag, 2);
        assert_eq!(stream.token(2).case_flag, 0);
    }

    #[test]
    fn tokenize_handles_punctuation_and_newlines() {
        let input = b"Hi, world!\nFoo bar.";
        let stream = tokenize(input);
        assert_eq!(stream.round_trip(), input);
    }

    #[test]
    fn tokenize_handles_empty_input() {
        let stream = tokenize(b"");
        assert!(stream.is_empty());
        assert!(stream.round_trip().is_empty());
    }

    #[test]
    fn word_split_matches_run_structure() {
        let data = b"the quick brown fox";
        let words = word_split(data);
        // 4 alpha words, each absorbs its trailing space.
        assert_eq!(words.len(), 4);
        let recon: Vec<u8> = words
            .iter()
            .flat_map(|&(s, l)| data[s as usize..(s + l) as usize].iter().copied())
            .collect();
        assert_eq!(recon, data.to_vec());
    }

    #[test]
    fn word_split_preserves_byte_order() {
        let data = b"a b\tc\nd";
        let words = word_split(data);
        let recon: Vec<u8> = words
            .iter()
            .flat_map(|&(s, l)| data[s as usize..(s + l) as usize].iter().copied())
            .collect();
        assert_eq!(recon, data.to_vec());
    }

    #[test]
    fn high_bit_bytes_pass_through_as_alpha() {
        // UTF-8 "café" = c, a, f, 0xC3 0xA9.
        let data = b"caf\xC3\xA9 word";
        let stream = tokenize(data);
        assert_eq!(stream.round_trip(), data);
    }
}
