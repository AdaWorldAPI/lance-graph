//! `UNICHARCOMPRESS` (the recoder) content store — the Rust side of the recoder
//! byte-parity leaf, sibling to [`crate::unicharset`].
//!
//! Tesseract's `UnicharCompress` (`ccutil/unicharcompress.{h,cpp}`) re-encodes
//! each unichar-id as a short sequence of small codes (Han radical-stroke,
//! Hangul Jamo, ligature dissection; pass-through for simple scripts). The LSTM
//! recognizer's output lattice speaks these **recoded codes, not raw
//! unichar-ids**, so `ids_to_text` only becomes real OCR output once the decode
//! table exists. Per the Core-First doctrine this is a **classid-keyed
//! content-store tier** (a loaded codec table — id ↔ code-sequence bijection +
//! bounds), exactly like [`crate::unicharset::UniCharSet`]: data-shaped, no
//! lifecycle vocabulary, no effects. It rides the existing keystone; it is NOT
//! IR-surface (`docs/OGAR-AS-IR.md` §3: adds no `Class` field, no `ActionDef`,
//! no `KausalSpec` slot).
//!
//! # Load-side scope
//!
//! This module transcodes the **load side only** — `DeSerialize` +
//! `EncodeUnichar` + `DecodeUnichar` + `code_range` (the recognizer runtime
//! surface). `ComputeEncoding` (the training-side table builder) is out of
//! scope. `SetupDecoder`'s beam-search maps (`is_valid_start_` / `next_codes_` /
//! `final_codes_`, `unicharcompress.cpp:396-434`) are the recognizer's, not the
//! decode table's — they are deferred to the recognizer leaf; only the
//! `decoder_` map (code → id) is built here.
//!
//! # Binary format (byte-parity surface)
//!
//! Every prior leaf parsed text; the recoder is **binary** (`serialis.h` `TFile`
//! conventions). `UnicharCompress::Serialize` writes exactly the `encoder_`
//! vector (`unicharcompress.cpp:318-320`, comment `unicharcompress.h:229`: "the
//! only part that is serialized. The rest is computed on load"). The wire form
//! (little-endian; `TFile::swap_ == false` on x86) is:
//!
//! ```text
//! u32  count                         // TFile::DeSerialize(vector<T>), serialis.h:90
//! count × RecodedCharID:
//!   i8   self_normalized             // RecodedCharID::DeSerialize, unicharcompress.h:75
//!   i32  length                      // number of codes in use (<= kMaxCodeLen=9)
//!   i32 × length  code               // only `length` codes are written, not all 9
//! ```
//!
//! For real `eng.lstm-recoder` (112 pass-through entries, all length-1):
//! `4 + 112·(1+4+4) = 1012` bytes — the exact on-disk size, a first-principles
//! pre-registration of a correct parse. On load, `ComputeCodeRange`
//! (`unicharcompress.cpp:383`, `max(code)+1`) and the `decoder_` map
//! (`unicharcompress.cpp:400-402`, `decoder_[code]=id` in ascending-id order, so
//! **last writer wins** on a shared code) are recomputed.
//!
//! [`UnicharCompress::dump_encode`] / [`UnicharCompress::dump_decode`] are the
//! byte-parity surfaces, diffed against the C++ `UnicharCompress` oracle
//! (`recoder_oracle.cpp`, which links libtesseract, loads the same component via
//! `TFile`, and dumps `EncodeUnichar` / `DecodeUnichar` / `code_range`). The
//! oracle's `Encode∘Decode` round-trip + the `UNICHARSET` bijection guard the
//! 5.5.0-header / 5.3.4-lib ABI skew for this NEW object layout.
//!
//! # Strict-vs-lenient
//!
//! C++ `RecodedCharID::DeSerialize` reads `length` then reads that many `i32`
//! into the fixed `code_[9]` — a buffer overflow (UB) if `length > 9` on hostile
//! input. This reader instead rejects `length < 0 || length > kMaxCodeLen`
//! ([`RecoderError::BadCodeLength`]) and a truncated buffer
//! ([`RecoderError::UnexpectedEof`]). On well-formed trained data (`length` is
//! always 1..=3) the byte-parity diff is unaffected; the guard only fires on
//! corruption.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;

/// `RecodedCharID::kMaxCodeLen` (tesseract `unicharcompress.h:35`) — the fixed
/// capacity of a code array. Hangul/Han use length 3; the array is sized 9.
const K_MAX_CODE_LEN: usize = 9;

/// The C++ `INVALID_UNICHAR_ID` sentinel (tesseract `unichar.h`) — what
/// [`UnicharCompress::decode`] returns for a code with no matching id, mirroring
/// `DecodeUnichar` (`unicharcompress.cpp:305-315`).
const INVALID_UNICHAR_ID: i32 = -1;

/// The `TFile::DeSerialize(vector<T>)` sanity cap (tesseract `serialis.h:96`):
/// a declared element count above this is treated as corrupt input.
const MAX_ELEMENTS: u32 = 50_000_000;

/// The code sequence for one recoded unichar-id — the transcription of
/// tesseract's `RecodedCharID` (`unicharcompress.h:32-109`).
///
/// Equality and hashing mirror the C++ `operator==` / `RecodedCharIDHash`
/// (`unicharcompress.h:79-99`): **only `length` + the used `code[0..length]`
/// participate**; `self_normalized` and any trailing array slots are ignored, so
/// this is a sound [`HashMap`] key for the decoder (`decoder_[code]`).
#[derive(Debug, Clone)]
pub struct RecodedCharId {
    /// True (`1`) if this is the master entry for ids sharing one code; stored as
    /// `i8` for serialization (`unicharcompress.h:104`). Preserved on load for
    /// round-trip fidelity; not part of identity.
    self_normalized: i8,
    /// The number of codes in use in `code` (`unicharcompress.h:106`).
    length: i32,
    /// The re-encoded form (`unicharcompress.h:108`). Only `code[0..length]` is
    /// meaningful; trailing slots are `0`.
    code: [i32; K_MAX_CODE_LEN],
}

impl Default for RecodedCharId {
    /// Mirrors the C++ default ctor (`unicharcompress.h:37`): `self_normalized =
    /// 1`, `length = 0`, all codes `0`.
    fn default() -> Self {
        Self {
            self_normalized: 1,
            length: 0,
            code: [0; K_MAX_CODE_LEN],
        }
    }
}

impl RecodedCharId {
    /// The codes in use — `code[0..length]`. The only bytes that carry identity.
    #[must_use]
    pub fn codes(&self) -> &[i32] {
        let len = self.length.max(0) as usize;
        // `length` is bounded to `<= K_MAX_CODE_LEN` at load; `min` keeps this
        // total even for a hand-built value.
        &self.code[..len.min(K_MAX_CODE_LEN)]
    }

    /// The number of codes in use (the C++ `length()`, `unicharcompress.h:62`).
    #[must_use]
    pub fn length(&self) -> i32 {
        self.length
    }

    /// Whether this code is empty (`length == 0`), the C++ `empty()`
    /// (`unicharcompress.h:58`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Whether this is the self-normalizing master entry (`unicharcompress.h:104`).
    #[must_use]
    pub fn self_normalized(&self) -> bool {
        self.self_normalized != 0
    }

    /// Read one `RecodedCharID` from the little-endian cursor. Rejects a
    /// `length` outside `0..=kMaxCodeLen` (the C++ UB guard) and a short buffer.
    fn read(r: &mut ByteReader<'_>) -> Result<Self, RecoderError> {
        let self_normalized = r.read_i8()?;
        let length = r.read_i32()?;
        if length < 0 || length as usize > K_MAX_CODE_LEN {
            return Err(RecoderError::BadCodeLength(length));
        }
        let mut code = [0_i32; K_MAX_CODE_LEN];
        for slot in code.iter_mut().take(length as usize) {
            *slot = r.read_i32()?;
        }
        Ok(Self {
            self_normalized,
            length,
            code,
        })
    }
}

impl PartialEq for RecodedCharId {
    /// `operator==` (`unicharcompress.h:79-89`): compares `length` +
    /// `code[0..length]` only.
    fn eq(&self, other: &Self) -> bool {
        self.codes() == other.codes()
    }
}

impl Eq for RecodedCharId {}

impl Hash for RecodedCharId {
    /// Consistent with [`PartialEq`]: hash the used codes only. (The C++
    /// `RecodedCharIDHash` folds the same `code[0..length]`; the Rust hasher need
    /// only agree with `eq`, not reproduce the C++ bit-mix.)
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.codes().hash(state);
    }
}

/// A loaded `UnicharCompress` (the recoder): the `encoder_` table (id → codes),
/// its inverse `decoder_` (codes → id), and `code_range` — the transcription of
/// tesseract's `UnicharCompress` load side (`unicharcompress.{h,cpp}`).
#[derive(Debug, Clone, Default)]
pub struct UnicharCompress {
    /// id → code sequence (index IS the unichar-id). The only serialized part
    /// (`unicharcompress.h:229-230`).
    encoder: Vec<RecodedCharId>,
    /// code → unichar-id, recomputed on load (`SetupDecoder`,
    /// `unicharcompress.cpp:400-402`). Last-writer-wins on a shared code.
    decoder: HashMap<RecodedCharId, u32>,
    /// `1 + max code value` (`ComputeCodeRange`, `unicharcompress.cpp:383-393`);
    /// the lattice width. `0` for an empty encoder (`-1 + 1`).
    code_range: i32,
}

impl UnicharCompress {
    /// Load a recoder from the raw little-endian bytes of a `.lstm-recoder`
    /// component (the C++ `DeSerialize`, `unicharcompress.cpp:323-330`): read the
    /// `encoder_` vector, then recompute `code_range` and the decode map.
    ///
    /// # Errors
    ///
    /// [`RecoderError::UnexpectedEof`] on a truncated buffer,
    /// [`RecoderError::TooManyElements`] if the declared count exceeds the
    /// `serialis.h` sanity cap, and [`RecoderError::BadCodeLength`] if any entry
    /// declares a code length outside `0..=9`.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, RecoderError> {
        let mut r = ByteReader::new(bytes);
        let count = r.read_u32()?;
        if count > MAX_ELEMENTS {
            return Err(RecoderError::TooManyElements(count));
        }
        let mut encoder = Vec::with_capacity(count as usize);
        for _ in 0..count {
            encoder.push(RecodedCharId::read(&mut r)?);
        }
        // Trailing bytes are ignored on purpose: a component extracted from a
        // TFile stream may be followed by the next component's bytes (the C++
        // reader leaves the cursor for them). A standalone `.lstm-recoder` is
        // consumed exactly.
        let mut this = Self {
            encoder,
            decoder: HashMap::new(),
            code_range: 0,
        };
        this.compute_code_range();
        this.setup_decoder();
        Ok(this)
    }

    /// Load a recoder from a `.lstm-recoder` file (a thin wrapper over
    /// [`Self::from_le_bytes`]). Extract one via
    /// `combine_tessdata -u eng.traineddata /tmp/eng.`.
    ///
    /// # Errors
    ///
    /// [`RecoderError::Io`] if the file cannot be read, else the parse errors of
    /// [`Self::from_le_bytes`].
    pub fn load_from_file(path: &Path) -> Result<Self, RecoderError> {
        let bytes = std::fs::read(path).map_err(|e| RecoderError::Io(e.to_string()))?;
        Self::from_le_bytes(&bytes)
    }

    /// `1 + max code value` — the lattice width (`code_range`,
    /// `unicharcompress.h:171`).
    #[must_use]
    pub fn code_range(&self) -> i32 {
        self.code_range
    }

    /// The number of encoded unichar-ids (`encoder_.size()`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.encoder.len()
    }

    /// Whether the encoder is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.encoder.is_empty()
    }

    /// The code sequence for `unichar_id`, or `None` if out of range — the C++
    /// `EncodeUnichar` (`unicharcompress.cpp:295-301`; a `None` here is the C++
    /// return of length `0`).
    #[must_use]
    pub fn encode(&self, unichar_id: u32) -> Option<&RecodedCharId> {
        self.encoder.get(unichar_id as usize)
    }

    /// The unichar-id for `code`, or [`INVALID_UNICHAR_ID`] (`-1`) if the code is
    /// ill-formed or unknown — the C++ `DecodeUnichar`
    /// (`unicharcompress.cpp:305-315`).
    #[must_use]
    pub fn decode(&self, code: &RecodedCharId) -> i32 {
        let len = code.length();
        if len <= 0 || len as usize > K_MAX_CODE_LEN {
            return INVALID_UNICHAR_ID;
        }
        self.decoder
            .get(code)
            .map_or(INVALID_UNICHAR_ID, |&id| id as i32)
    }

    /// `ComputeCodeRange` (`unicharcompress.cpp:383-393`): `code_range = 1 + max`
    /// code value over every position of every entry (`0` for an empty encoder).
    fn compute_code_range(&mut self) {
        let mut max = -1_i32;
        for entry in &self.encoder {
            for &c in entry.codes() {
                if c > max {
                    max = c;
                }
            }
        }
        self.code_range = max + 1;
    }

    /// The decode-map half of `SetupDecoder` (`unicharcompress.cpp:400-402`):
    /// `decoder_[encoder_[id]] = id` in ascending id order, so **last writer
    /// wins** when two ids share a code. The beam-search maps are the
    /// recognizer's and are not built here (see module docs).
    fn setup_decoder(&mut self) {
        self.decoder.clear();
        self.decoder.reserve(self.encoder.len());
        for (id, code) in self.encoder.iter().enumerate() {
            self.decoder.insert(code.clone(), id as u32);
        }
    }

    /// Render the id→code table as `"<id>\t<len>\t<c0>[,<c1>...]\n"` lines — the
    /// exact shape the C++ recoder oracle's `encode` mode prints, so the
    /// byte-parity diff is `diff oracle_recoder_encode.tsv rust_recoder_encode.tsv`.
    #[must_use]
    pub fn dump_encode(&self) -> String {
        let mut out = String::new();
        for (id, entry) in self.encoder.iter().enumerate() {
            out.push_str(&id.to_string());
            out.push('\t');
            out.push_str(&entry.length().to_string());
            out.push('\t');
            for (i, &c) in entry.codes().iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&c.to_string());
            }
            out.push('\n');
        }
        out
    }

    /// Render `"code_range\t<N>\n"` then `"<id>\t<decoded>\n"` lines (where
    /// `decoded = decode(encode(id))`) — the exact shape the C++ recoder oracle's
    /// `decode` mode prints, so the byte-parity diff is
    /// `diff oracle_recoder_decode.tsv rust_recoder_decode.tsv`. On a shared code
    /// the decoded id is the last-writer, matching the C++ map.
    #[must_use]
    pub fn dump_decode(&self) -> String {
        let mut out = String::new();
        out.push_str("code_range\t");
        out.push_str(&self.code_range.to_string());
        out.push('\n');
        for (id, entry) in self.encoder.iter().enumerate() {
            out.push_str(&id.to_string());
            out.push('\t');
            out.push_str(&self.decode(entry).to_string());
            out.push('\n');
        }
        out
    }
}

/// A little-endian byte cursor over the recoder component — the reader half of
/// the `TFile` primitives this leaf needs (`FReadEndian` with `swap_ == false`).
struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Advance over `n` bytes, or [`RecoderError::UnexpectedEof`] if short.
    fn take(&mut self, n: usize) -> Result<&'a [u8], RecoderError> {
        let end = self.pos.checked_add(n).ok_or(RecoderError::UnexpectedEof)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(RecoderError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    fn read_i8(&mut self) -> Result<i8, RecoderError> {
        Ok(self.take(1)?[0] as i8)
    }

    fn read_u32(&mut self) -> Result<u32, RecoderError> {
        let arr: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| RecoderError::UnexpectedEof)?;
        Ok(u32::from_le_bytes(arr))
    }

    fn read_i32(&mut self) -> Result<i32, RecoderError> {
        let arr: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| RecoderError::UnexpectedEof)?;
        Ok(i32::from_le_bytes(arr))
    }
}

/// A failure loading a `UnicharCompress` (recoder).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoderError {
    /// The buffer ended mid-field.
    UnexpectedEof,
    /// The declared element count exceeded the `serialis.h` sanity cap.
    TooManyElements(u32),
    /// A `RecodedCharID` declared a code length outside `0..=9` (the C++ fixed
    /// array capacity `kMaxCodeLen`).
    BadCodeLength(i32),
    /// The file could not be read (message from the underlying I/O error).
    Io(String),
}

impl std::fmt::Display for RecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "recoder buffer ended mid-field"),
            Self::TooManyElements(n) => {
                write!(
                    f,
                    "recoder declared {n} elements (over the {MAX_ELEMENTS} cap)"
                )
            }
            Self::BadCodeLength(len) => {
                write!(
                    f,
                    "recoded code length {len} out of range 0..={K_MAX_CODE_LEN}"
                )
            }
            Self::Io(msg) => write!(f, "recoder read failed: {msg}"),
        }
    }
}

impl std::error::Error for RecoderError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `.lstm-recoder` byte buffer from `(self_normalized, codes)`
    /// entries, in the exact little-endian wire form the C++ `Serialize` writes.
    fn build(entries: &[(i8, &[i32])]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&u32::try_from(entries.len()).unwrap().to_le_bytes());
        for (self_norm, codes) in entries {
            b.push(*self_norm as u8);
            b.extend_from_slice(&i32::try_from(codes.len()).unwrap().to_le_bytes());
            for &c in *codes {
                b.extend_from_slice(&c.to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn parses_count_and_entries() {
        let bytes = build(&[(1, &[0]), (1, &[5]), (1, &[5])]);
        let rec = UnicharCompress::from_le_bytes(&bytes).expect("valid");
        assert_eq!(rec.len(), 3);
        assert_eq!(rec.encode(0).unwrap().codes(), &[0]);
        assert_eq!(rec.encode(2).unwrap().codes(), &[5]);
        assert!(rec.encode(3).is_none(), "out-of-range id -> None");
    }

    #[test]
    fn code_range_is_max_plus_one() {
        // max code value 5 -> code_range 6.
        let rec = UnicharCompress::from_le_bytes(&build(&[(1, &[0]), (1, &[5]), (1, &[3])]))
            .expect("valid");
        assert_eq!(rec.code_range(), 6);
        // Empty encoder -> -1 + 1 = 0 (matches ComputeCodeRange's seed).
        let empty = UnicharCompress::from_le_bytes(&build(&[])).expect("valid");
        assert_eq!(empty.code_range(), 0);
    }

    #[test]
    fn decode_is_last_writer_wins_on_shared_code() {
        // ids 1 and 2 both encode to code [5]; decoder keeps the last (id 2) —
        // exactly the eng.lstm-recoder id1/id2 -> code 110 case.
        let rec = UnicharCompress::from_le_bytes(&build(&[(1, &[0]), (1, &[5]), (1, &[5])]))
            .expect("valid");
        assert_eq!(rec.decode(rec.encode(0).unwrap()), 0);
        assert_eq!(
            rec.decode(rec.encode(1).unwrap()),
            2,
            "shared code -> last id"
        );
        assert_eq!(rec.decode(rec.encode(2).unwrap()), 2);
    }

    #[test]
    fn decode_unknown_or_illformed_is_invalid() {
        let rec = UnicharCompress::from_le_bytes(&build(&[(1, &[0])])).expect("valid");
        // An empty code (length 0) is ill-formed for decode.
        assert_eq!(rec.decode(&RecodedCharId::default()), INVALID_UNICHAR_ID);
    }

    #[test]
    fn equality_ignores_self_normalized_and_trailing() {
        // Same code, different self_normalized -> equal (C++ operator==).
        let a = UnicharCompress::from_le_bytes(&build(&[(1, &[7])])).expect("valid");
        let b = UnicharCompress::from_le_bytes(&build(&[(0, &[7])])).expect("valid");
        assert_eq!(a.encode(0).unwrap(), b.encode(0).unwrap());
    }

    #[test]
    fn dump_encode_matches_oracle_shape() {
        // A multi-code entry exercises the comma join.
        let rec = UnicharCompress::from_le_bytes(&build(&[(1, &[0]), (1, &[5]), (1, &[1, 2, 3])]))
            .expect("valid");
        assert_eq!(rec.dump_encode(), "0\t1\t0\n1\t1\t5\n2\t3\t1,2,3\n");
    }

    #[test]
    fn dump_decode_matches_oracle_shape() {
        let rec = UnicharCompress::from_le_bytes(&build(&[(1, &[0]), (1, &[5]), (1, &[5])]))
            .expect("valid");
        // code_range = 6; id1 decodes to 2 (last-writer on shared code [5]).
        assert_eq!(rec.dump_decode(), "code_range\t6\n0\t0\n1\t2\n2\t2\n");
    }

    #[test]
    fn truncated_buffer_errors() {
        let mut bytes = build(&[(1, &[0])]);
        bytes.pop(); // drop the last code byte
        assert_eq!(
            UnicharCompress::from_le_bytes(&bytes).unwrap_err(),
            RecoderError::UnexpectedEof
        );
        // A count with no entries at all.
        assert_eq!(
            UnicharCompress::from_le_bytes(&[3, 0, 0, 0]).unwrap_err(),
            RecoderError::UnexpectedEof
        );
    }

    #[test]
    fn bad_code_length_errors() {
        // count=1, self_norm=1, length=10 (> kMaxCodeLen) — the C++ UB case.
        let mut bytes = vec![1, 0, 0, 0, 1];
        bytes.extend_from_slice(&10_i32.to_le_bytes());
        assert_eq!(
            UnicharCompress::from_le_bytes(&bytes).unwrap_err(),
            RecoderError::BadCodeLength(10)
        );
    }

    #[test]
    fn too_many_elements_errors() {
        // A declared count over the cap fails fast without allocating.
        let bytes = (MAX_ELEMENTS + 1).to_le_bytes();
        assert_eq!(
            UnicharCompress::from_le_bytes(&bytes).unwrap_err(),
            RecoderError::TooManyElements(MAX_ELEMENTS + 1)
        );
    }
}
