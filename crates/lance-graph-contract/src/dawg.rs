//! `SquishedDawg` (the compacted Tesseract dictionary word-graph) binary
//! loader — the Rust side of the dawg/dict byte-parity leaf, sibling to
//! [`crate::unicharcompress`].
//!
//! Tesseract's dictionaries (word list, punctuation, number patterns) are
//! stored as compacted Directed Acyclic Word Graphs (`dict/dawg.{h,cpp}`):
//! each node's outgoing edges are a contiguous run of 64-bit `EDGE_RECORD`s,
//! bit-packed with the destination node, the matched letter, and a 3-bit
//! flag field whose WIDTH depends on the loaded unicharset's size. Per the
//! Core-First doctrine this is a **classid-keyed content-store tier** (a
//! loaded lookup table — edge array + derived bit masks), exactly like
//! [`crate::unicharcompress::UnicharCompress`]: data-shaped, no lifecycle
//! vocabulary, no effects. It rides the existing keystone; it is NOT
//! IR-surface (`docs/OGAR-AS-IR.md` §3: adds no `Class` field, no
//! `ActionDef`, no `KausalSpec` slot).
//!
//! # Load-side scope
//!
//! This module transcodes `SquishedDawg::read_squished_dawg`
//! (`dawg.cpp:313-352`) and the base class `Dawg::init` bit-mask derivation
//! (`dawg.cpp:178-188`), plus the read-only edge accessors
//! (`next_node_from_edge_rec` / `unichar_id_from_edge_rec` /
//! `marker_flag_from_edge_rec` / `direction_from_edge_rec` /
//! `end_of_word_from_edge_rec`, `dawg.h:210-230`). The write side
//! (`write_squished_dawg`, `dawg.cpp:391-456`) and the search/traversal
//! methods (`edge_char_of`, `word_in_dawg`, …) are out of scope — this leaf
//! is the loader + accessor surface only.
//!
//! # Binary format (byte-parity surface)
//!
//! `TFile`-convention little-endian (auto-endian on disk; this
//! environment's trained-data files are LE, matching x86
//! `TFile::swap_ == false`):
//!
//! ```text
//! i16  magic              // Dawg::kDawgMagicNumber, MUST be 42
//! i32  unicharset_size    // MUST be > 0 (ASSERT_HOST in Dawg::init)
//! i32  num_edges          // MUST be > 0 (ASSERT_HOST, "DAWG should not be empty")
//! num_edges x u64  edges  // EDGE_RECORD array, read as one block
//! ```
//!
//! For real `eng.lstm-punc-dawg`: `10 + 539·8 = 4322` bytes, the exact
//! on-disk size (`eng.lstm-word-dawg`: `10 + 461848·8 = 3694794`) — a
//! first-principles pre-registration of a correct parse.
//!
//! # Bit layout (unicharset-size-dependent)
//!
//! `Dawg::init` (`dawg.cpp:178-188`) derives every mask from
//! `unicharset_size` alone (`NUM_FLAG_BITS = 3`, `dawg.h:84`):
//!
//! ```text
//! flag_start_bit      = ceil(log2(unicharset_size + 1))
//! next_node_start_bit = flag_start_bit + 3
//! letter_mask         = !(u64::MAX << flag_start_bit)
//! next_node_mask      = u64::MAX << next_node_start_bit
//! ```
//!
//! (`flags_mask_` is also derived in C++, `dawg.cpp:187`, but never
//! consulted by the read accessors below — each tests its flag bit
//! directly against `flag_start_bit`, mirrored here, so it is not stored.)
//! For the real `eng.lstm-{punc,number}-dawg` (`unicharset_size = 112`):
//! `flag_start_bit = ceil(log2(113)) = 7`.
//!
//! Each `u64` edge then reads as: bits `[0, flag_start_bit)` = letter
//! (`unichar_id_from_edge_rec`), 3 flag bits starting at `flag_start_bit`
//! = marker / backward / word-end (`MARKER_FLAG=1` / `DIRECTION_FLAG=2` /
//! `WERD_END_FLAG=4`, `dawg.h:80-82`), remaining high bits = next-node
//! reference (`next_node_from_edge_rec`).
//!
//! [`SquishedDawg::edge_letter`] / [`SquishedDawg::next_node`] /
//! [`SquishedDawg::end_of_word`] are the byte-parity surface, exercised by
//! the `dawg_dump` example.
//!
//! # Strict-vs-lenient
//!
//! C++ `read_squished_dawg` trusts `num_edges` and allocates
//! `new EDGE_RECORD[num_edges_]` unconditionally — a huge or negative
//! declared count is a C++ allocation hazard. This reader instead rejects
//! `unicharset_size <= 0` ([`DawgError::NonPositiveSize`]) and
//! `num_edges <= 0` ([`DawgError::Empty`]) before allocating, caps the
//! *speculative* allocation hint (the loop still reads exactly `num_edges`
//! entries, or fails with [`DawgError::UnexpectedEof`] the moment the
//! buffer runs out), and rejects a truncated buffer. On well-formed
//! trained data the byte-parity diff is unaffected; the guards only fire
//! on corruption.

use std::path::Path;

/// `NODE_REF` / `EDGE_REF` (`dawg.h:49-50`) — both are the same signed
/// 64-bit edge-array index type in the C++; this module uses one alias for
/// both, matching how `edge_char_of`'s `node` parameter and `next_node`'s
/// return value are the same underlying type in the original.
pub type NodeRef = i64;

/// `NO_EDGE` (`dawg.h:37-41`, the GCC arm: `static_cast<int64_t>(0xffff...)`,
/// i.e. all-ones as `int64_t` = `-1`) — the sentinel meaning "no such edge /
/// node". Returned as [`None`] from [`SquishedDawg::edge_char_of`] rather
/// than as a sentinel value.
pub const NO_EDGE: NodeRef = -1;

/// `Dawg::kDawgMagicNumber` (`dawg.h:113`) — the endian-detection magic
/// every squished-dawg component opens with.
const DAWG_MAGIC_NUMBER: i16 = 42;

/// `NUM_FLAG_BITS` (`dawg.h:84`) — the fixed width of the flag field packed
/// above the letter bits in every `EDGE_RECORD`.
const NUM_FLAG_BITS: u32 = 3;

/// `MARKER_FLAG` (`dawg.h:80`) — set on the last edge of a node's edge run.
const MARKER_FLAG: u64 = 1;

/// `DIRECTION_FLAG` (`dawg.h:81`) — set when the edge is a backward link.
const DIRECTION_FLAG: u64 = 2;

/// `WERD_END_FLAG` (`dawg.h:82`) — set when the edge completes a word.
const WERD_END_FLAG: u64 = 4;

/// A speculative-allocation cap for the initial `Vec::with_capacity` hint
/// (not a semantic limit — the read loop still consumes exactly the
/// declared `num_edges`, or errors on a short buffer). `1 << 20` comfortably
/// covers the real `eng.lstm-word-dawg` (461,848 edges).
const MAX_PREALLOC_HINT: usize = 1 << 20;

/// `DawgType` (`dawg.h:64-71`) — which kind of dictionary this dawg encodes.
/// Discriminants match the C++ enum ordinals exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DawgType {
    /// `DAWG_TYPE_PUNCTUATION` (0).
    Punctuation = 0,
    /// `DAWG_TYPE_WORD` (1).
    Word = 1,
    /// `DAWG_TYPE_NUMBER` (2).
    Number = 2,
    /// `DAWG_TYPE_PATTERN` (3).
    Pattern = 3,
}

impl DawgType {
    /// Maps a raw ordinal (0..3) to its `DawgType`, or `None` if out of
    /// range (mirrors the bounds of the C++ `DAWG_TYPE_COUNT`-sized enum).
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Punctuation),
            1 => Some(Self::Word),
            2 => Some(Self::Number),
            3 => Some(Self::Pattern),
            _ => None,
        }
    }
}

/// `PermuterType` (`ratngs.h:235-251`) — which permuter produced/should
/// score a dictionary match. Discriminants match the C++ enum ordinals
/// exactly, since `def_letter_is_okay` and friends compare permuter values
/// numerically (`Ord`/`PartialOrd` derived for that reason).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PermuterType {
    /// `NO_PERM` (0).
    NoPerm = 0,
    /// `PUNC_PERM` (1).
    PuncPerm = 1,
    /// `TOP_CHOICE_PERM` (2).
    TopChoicePerm = 2,
    /// `LOWER_CASE_PERM` (3).
    LowerCasePerm = 3,
    /// `UPPER_CASE_PERM` (4).
    UpperCasePerm = 4,
    /// `NGRAM_PERM` (5).
    NgramPerm = 5,
    /// `NUMBER_PERM` (6).
    NumberPerm = 6,
    /// `USER_PATTERN_PERM` (7).
    UserPatternPerm = 7,
    /// `SYSTEM_DAWG_PERM` (8).
    SystemDawgPerm = 8,
    /// `DOC_DAWG_PERM` (9).
    DocDawgPerm = 9,
    /// `USER_DAWG_PERM` (10).
    UserDawgPerm = 10,
    /// `FREQ_DAWG_PERM` (11).
    FreqDawgPerm = 11,
    /// `COMPOUND_PERM` (12).
    CompoundPerm = 12,
}

impl PermuterType {
    /// The C++ enum ordinal for this permuter.
    #[must_use]
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Maps a raw ordinal (0..12) to its `PermuterType`, or `None` if out
    /// of range.
    #[must_use]
    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::NoPerm),
            1 => Some(Self::PuncPerm),
            2 => Some(Self::TopChoicePerm),
            3 => Some(Self::LowerCasePerm),
            4 => Some(Self::UpperCasePerm),
            5 => Some(Self::NgramPerm),
            6 => Some(Self::NumberPerm),
            7 => Some(Self::UserPatternPerm),
            8 => Some(Self::SystemDawgPerm),
            9 => Some(Self::DocDawgPerm),
            10 => Some(Self::UserDawgPerm),
            11 => Some(Self::FreqDawgPerm),
            12 => Some(Self::CompoundPerm),
            _ => None,
        }
    }
}

/// A loaded `SquishedDawg` — the compacted edge array plus its derived bit
/// masks, the transcription of tesseract's `SquishedDawg` load side
/// (`dawg.{h,cpp}`).
#[derive(Debug, Clone)]
pub struct SquishedDawg {
    /// `unicharset_size_` (`dawg.h:313`) — the loaded unicharset's size;
    /// the sole input to every derived mask (`Dawg::init`, `dawg.cpp:178`).
    unicharset_size: i32,
    /// The `EDGE_RECORD` array (`edges_`, `dawg.h:567`), one `u64` per
    /// edge, in on-disk order.
    edges: Vec<u64>,
    /// `flag_start_bit_` (`dawg.h:314`).
    flag_start_bit: u32,
    /// `next_node_start_bit_` (`dawg.h:315`).
    next_node_start_bit: u32,
    /// `letter_mask_` (`dawg.h:312`).
    letter_mask: u64,
    /// `next_node_mask_` (`dawg.h:310`).
    next_node_mask: u64,
    /// `Dawg::type_` (`dawg.h:303`) — user-set role metadata; in the real
    /// Tesseract this is a constructor argument, not stored in the file.
    /// Defaults to [`DawgType::Word`] on load; set explicitly via
    /// [`Self::with_type`] when wiring the file to a role.
    dawg_type: DawgType,
    /// `Dawg::perm_` (`dawg.h:305`) — user-set permuter metadata, same
    /// caveat as `dawg_type`. Defaults to [`PermuterType::SystemDawgPerm`]
    /// on load; set explicitly via [`Self::with_permuter`].
    permuter: PermuterType,
    /// `SquishedDawg::num_forward_edges_in_node0` (`dawg.h:569`) — computed
    /// once at load time (`SquishedDawg::num_forward_edges(0)`,
    /// `dawg.cpp:230-241`), the length of the binary-searchable run of
    /// forward edges out of the root node.
    num_forward_edges_in_node0: usize,
}

impl SquishedDawg {
    /// Load a `SquishedDawg` from raw little-endian bytes (the C++
    /// `read_squished_dawg`, `dawg.cpp:313-352`): magic, `unicharset_size`,
    /// `num_edges`, then the edge array, deriving the bit masks from
    /// `unicharset_size` via `Dawg::init`'s formula (`dawg.cpp:178-188`).
    ///
    /// Returns the loaded dawg plus the number of bytes consumed (the
    /// 10-byte header plus exactly `num_edges * 8` bytes; trailing bytes,
    /// if any, are left unconsumed — mirroring a component embedded in a
    /// larger `TFile` stream).
    ///
    /// # Errors
    ///
    /// [`DawgError::UnexpectedEof`] on a truncated buffer,
    /// [`DawgError::BadMagic`] if the magic number is not 42,
    /// [`DawgError::NonPositiveSize`] if `unicharset_size <= 0`, and
    /// [`DawgError::Empty`] if `num_edges <= 0` (the C++ `ASSERT_HOST`
    /// guards, made into recoverable errors).
    pub fn from_le_bytes(bytes: &[u8]) -> Result<(Self, usize), DawgError> {
        let mut r = ByteReader::new(bytes);
        let magic = r.read_i16()?;
        if magic != DAWG_MAGIC_NUMBER {
            return Err(DawgError::BadMagic(magic));
        }
        let unicharset_size = r.read_i32()?;
        if unicharset_size <= 0 {
            return Err(DawgError::NonPositiveSize);
        }
        let num_edges = r.read_i32()?;
        if num_edges <= 0 {
            return Err(DawgError::Empty);
        }
        // Preflight the declared payload against the actual buffer so a
        // malformed header with a huge num_edges fails fast instead of
        // growing the Vec until the truncated read finally errors.
        let edge_bytes = (num_edges as usize)
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or(DawgError::UnexpectedEof)?;
        let required_len = r
            .pos()
            .checked_add(edge_bytes)
            .ok_or(DawgError::UnexpectedEof)?;
        if bytes.len() < required_len {
            return Err(DawgError::UnexpectedEof);
        }
        let mut edges = Vec::with_capacity((num_edges as usize).min(MAX_PREALLOC_HINT));
        for _ in 0..num_edges {
            edges.push(r.read_u64()?);
        }
        let (flag_start_bit, next_node_start_bit, letter_mask, next_node_mask) =
            Self::derive_masks(unicharset_size);
        let num_forward_edges_in_node0 =
            Self::count_forward_edges(&edges, next_node_mask, flag_start_bit, 0);
        Ok((
            Self {
                unicharset_size,
                edges,
                flag_start_bit,
                next_node_start_bit,
                letter_mask,
                next_node_mask,
                dawg_type: DawgType::Word,
                permuter: PermuterType::SystemDawgPerm,
                num_forward_edges_in_node0,
            },
            r.pos(),
        ))
    }

    /// `SquishedDawg::num_forward_edges` (`dawg.cpp:230-241`), specialised
    /// to the load-time call site `num_forward_edges(0)`. Free function
    /// (not a method) because it runs before `Self` exists, over the raw
    /// edge slice and derived masks.
    ///
    /// Mirrors the C++ walk: `forward_edge(edge)` (occupied and not
    /// backward) at the start, then a contiguous count until `last_edge`
    /// fires. The C++ trusts the on-disk format to keep a node's edge run
    /// homogeneous once started; this version additionally re-checks
    /// occupancy/backward/bounds on every step as a defensive guard against
    /// malformed input (a no-op on well-formed data, where the C++ behaviour
    /// and this behaviour coincide).
    fn count_forward_edges(
        edges: &[u64],
        next_node_mask: u64,
        flag_start_bit: u32,
        node: usize,
    ) -> usize {
        let occupied = |i: usize| edges[i] != next_node_mask;
        let is_backward = |i: usize| (edges[i] & (DIRECTION_FLAG << flag_start_bit)) != 0;
        let is_last = |i: usize| (edges[i] & (MARKER_FLAG << flag_start_bit)) != 0;

        let mut edge = node;
        let mut count = 0usize;
        loop {
            if edge >= edges.len() || !occupied(edge) || is_backward(edge) {
                break;
            }
            count += 1;
            let last = is_last(edge);
            edge += 1;
            if last {
                break;
            }
        }
        count
    }

    /// Load a `SquishedDawg` from a `.lstm-*-dawg` file (a thin wrapper over
    /// [`Self::from_le_bytes`]). Extract one via
    /// `combine_tessdata -u eng.traineddata /tmp/eng.`.
    ///
    /// # Errors
    ///
    /// [`DawgError::Io`] if the file cannot be read, else the parse errors
    /// of [`Self::from_le_bytes`].
    pub fn load_from_file(path: &Path) -> Result<Self, DawgError> {
        let bytes = std::fs::read(path).map_err(|e| DawgError::Io(e.to_string()))?;
        let (dawg, _consumed) = Self::from_le_bytes(&bytes)?;
        Ok(dawg)
    }

    /// `Dawg::init` (`dawg.cpp:178-188`): derive `(flag_start_bit,
    /// next_node_start_bit, letter_mask, next_node_mask)` from
    /// `unicharset_size` alone. `unicharset_size` is treated as an implicit
    /// null char, so the mask math sizes for `unicharset_size + 1` symbols.
    fn derive_masks(unicharset_size: i32) -> (u32, u32, u64, u64) {
        let flag_start_bit = (((unicharset_size as f64) + 1.0).ln() / 2f64.ln()).ceil() as u32;
        let next_node_start_bit = flag_start_bit + NUM_FLAG_BITS;
        let letter_mask = !(u64::MAX << flag_start_bit);
        let next_node_mask = u64::MAX << next_node_start_bit;
        (
            flag_start_bit,
            next_node_start_bit,
            letter_mask,
            next_node_mask,
        )
    }

    /// `unicharset_size_` — the value this dawg's masks were derived from.
    #[must_use]
    pub fn unicharset_size(&self) -> i32 {
        self.unicharset_size
    }

    /// The number of loaded edges (`num_edges_`).
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// `flag_start_bit_` (`dawg.h:314`) — the bit index where the 3-bit
    /// flag field (and, immediately above it, the next-node reference)
    /// begins.
    #[must_use]
    pub fn flag_start_bit(&self) -> u32 {
        self.flag_start_bit
    }

    /// The `UNICHAR_ID` matched by `edge` — the C++
    /// `unichar_id_from_edge_rec` (`dawg.h:227-230`):
    /// `(edge_rec & letter_mask_) >> LETTER_START_BIT` (`LETTER_START_BIT`
    /// is `0`, so this reduces to `edge_rec & letter_mask_`).
    ///
    /// # Panics
    ///
    /// Panics if `edge >= self.num_edges()` (plain slice indexing,
    /// mirroring the C++ unchecked `edges_[edge_ref]`).
    #[must_use]
    pub fn edge_letter(&self, edge: usize) -> u32 {
        (self.edges[edge] & self.letter_mask) as u32
    }

    /// The `NODE_REF` (edge index of the target node's first edge) that
    /// `edge` transitions to — the C++ `next_node_from_edge_rec`
    /// (`dawg.h:210-212`): `(edge_rec & next_node_mask_) >>
    /// next_node_start_bit_`.
    ///
    /// # Panics
    ///
    /// Panics if `edge >= self.num_edges()`.
    #[must_use]
    pub fn next_node(&self, edge: usize) -> u64 {
        (self.edges[edge] & self.next_node_mask) >> self.next_node_start_bit
    }

    /// Whether `edge` is the last edge in its node's edge run — the C++
    /// `marker_flag_from_edge_rec` (`dawg.h:214-216`):
    /// `edge_rec & (MARKER_FLAG << flag_start_bit_) != 0`.
    ///
    /// # Panics
    ///
    /// Panics if `edge >= self.num_edges()`.
    #[must_use]
    pub fn marker_flag(&self, edge: usize) -> bool {
        (self.edges[edge] & (MARKER_FLAG << self.flag_start_bit)) != 0
    }

    /// Whether `edge` is a backward link — the C++
    /// `direction_from_edge_rec` (`dawg.h:218-221`) tests
    /// `DIRECTION_FLAG << flag_start_bit_`; a set bit means `BACKWARD_EDGE`.
    ///
    /// # Panics
    ///
    /// Panics if `edge >= self.num_edges()`.
    #[must_use]
    pub fn is_backward(&self, edge: usize) -> bool {
        (self.edges[edge] & (DIRECTION_FLAG << self.flag_start_bit)) != 0
    }

    /// Whether `edge` completes a word — the C++
    /// `end_of_word_from_edge_rec` (`dawg.h:223-225`):
    /// `edge_rec & (WERD_END_FLAG << flag_start_bit_) != 0`.
    ///
    /// # Panics
    ///
    /// Panics if `edge >= self.num_edges()`.
    #[must_use]
    pub fn end_of_word(&self, edge: usize) -> bool {
        (self.edges[edge] & (WERD_END_FLAG << self.flag_start_bit)) != 0
    }

    /// Sets this dawg's [`DawgType`] role. In real Tesseract `type_` is a
    /// constructor argument (`SquishedDawg::SquishedDawg`, `dawg.h:410-420`)
    /// rather than something read from the file; callers wiring a loaded
    /// file to a role (word/punc/number dawg) set it explicitly.
    #[must_use]
    pub fn with_type(mut self, t: DawgType) -> Self {
        self.dawg_type = t;
        self
    }

    /// Sets this dawg's [`PermuterType`]. Same constructor-argument caveat
    /// as [`Self::with_type`].
    #[must_use]
    pub fn with_permuter(mut self, p: PermuterType) -> Self {
        self.permuter = p;
        self
    }

    /// This dawg's [`DawgType`] role (`Dawg::type()`, `dawg.h:119-121`).
    #[must_use]
    pub fn dawg_type(&self) -> DawgType {
        self.dawg_type
    }

    /// This dawg's [`PermuterType`] (`Dawg::permuter()`, `dawg.h:125-127`).
    #[must_use]
    pub fn permuter(&self) -> PermuterType {
        self.permuter
    }

    /// The number of edges counted into the binary-searchable forward run
    /// out of the root node — `num_forward_edges_in_node0` (`dawg.h:569`),
    /// computed once at load time.
    #[must_use]
    pub fn num_forward_edges_in_node0(&self) -> usize {
        self.num_forward_edges_in_node0
    }

    /// Whether the edge slot at `edge` is occupied — the C++
    /// `SquishedDawg::edge_occupied` (`dawg.h:538-540`):
    /// `edges[edge] != next_node_mask_`. An empty slot is written by
    /// `set_empty_edge` as exactly `next_node_mask_` (the write side is out
    /// of scope for this loader, but the read-side test is needed by
    /// [`Self::edge_char_of`]).
    ///
    /// # Panics
    ///
    /// Panics if `edge >= self.num_edges()`.
    #[must_use]
    pub fn edge_occupied(&self, edge: usize) -> bool {
        self.edges[edge] != self.next_node_mask
    }

    /// Whether `edge` is the last edge in its node's edge run — the C++
    /// `SquishedDawg::last_edge` (`dawg.h:542-544`):
    /// `edges[edge] & (MARKER_FLAG << flag_start_bit_) != 0`. Identical
    /// formula to [`Self::marker_flag`] (the base-class accessor); this
    /// method exists under the name the traversal code
    /// (`edge_char_of`/`num_forward_edges`) actually calls it by.
    ///
    /// # Panics
    ///
    /// Panics if `edge >= self.num_edges()`.
    #[must_use]
    pub fn last_edge(&self, edge: usize) -> bool {
        self.marker_flag(edge)
    }

    /// `Dawg::given_greater_than_edge_rec` (`dawg.h:247-271`): tri-state
    /// compare of `(next_node, word_end, unichar_id)` against `edge`'s
    /// decoded fields. Returns `1` if the given values sort strictly after
    /// `edge`, `0` if [`Self::edge_rec_match`] holds, `-1` otherwise.
    fn given_greater_than_edge_rec(
        &self,
        next_node: NodeRef,
        word_end: bool,
        unichar_id: u32,
        edge: usize,
    ) -> i32 {
        let curr_unichar_id = self.edge_letter(edge);
        // `next_node()` is `u64`; on-disk node references are always small
        // non-negative offsets, so the widen to `i64` mirrors the C++
        // `NODE_REF` (signed) representation without loss for real data.
        let curr_next_node = self.next_node(edge) as NodeRef;
        let curr_word_end = self.end_of_word(edge);
        if Self::edge_rec_match(
            next_node,
            word_end,
            unichar_id,
            curr_next_node,
            curr_word_end,
            curr_unichar_id,
        ) {
            return 0;
        }
        if unichar_id > curr_unichar_id {
            return 1;
        }
        if unichar_id == curr_unichar_id {
            if next_node > curr_next_node {
                return 1;
            }
            if next_node == curr_next_node && word_end && !curr_word_end {
                return 1;
            }
        }
        -1
    }

    /// `Dawg::edge_rec_match` (`dawg.h:275-282`): true if all the given
    /// values equal the decoded edge values (any value matches `next_node`
    /// if `next_node == NO_EDGE`, any value matches `word_end` if
    /// `word_end` is false).
    #[allow(clippy::too_many_arguments)]
    fn edge_rec_match(
        next_node: NodeRef,
        word_end: bool,
        unichar_id: u32,
        other_next_node: NodeRef,
        other_word_end: bool,
        other_unichar_id: u32,
    ) -> bool {
        unichar_id == other_unichar_id
            && (next_node == NO_EDGE || next_node == other_next_node)
            && (!word_end || word_end == other_word_end)
    }

    /// `SquishedDawg::edge_char_of` (`dawg.cpp:198-228`) — the beating heart
    /// of the beam step: returns the edge out of `node` matching
    /// `unichar_id` (and, if `word_end` is true, also marking end-of-word),
    /// or `None` on the C++ `NO_EDGE` sentinel (not found).
    ///
    /// - `node == 0`: binary search over
    ///   `edges[0..num_forward_edges_in_node0]` using
    ///   [`Self::given_greater_than_edge_rec`] (`dawg.cpp:201-216`).
    /// - `node != 0`: linear scan starting at `edge = node`, matching
    ///   `edge_letter(edge) == unichar_id && (!word_end ||
    ///   end_of_word(edge))`, advancing until [`Self::last_edge`] fires
    ///   (`dawg.cpp:217-226`).
    #[must_use]
    pub fn edge_char_of(&self, node: NodeRef, unichar_id: u32, word_end: bool) -> Option<usize> {
        if node == 0 {
            if self.num_forward_edges_in_node0 == 0 {
                return None;
            }
            let mut start: i64 = 0;
            let mut end: i64 = self.num_forward_edges_in_node0 as i64 - 1;
            while start <= end {
                // (start + end) / 2, matching the C++ `>> 1`.
                let edge = ((start + end) >> 1) as usize;
                match self.given_greater_than_edge_rec(NO_EDGE, word_end, unichar_id, edge) {
                    0 => return Some(edge),
                    1 => start = edge as i64 + 1,
                    _ => end = edge as i64 - 1,
                }
            }
            None
        } else {
            if node == NO_EDGE || node < 0 {
                return None;
            }
            let mut edge = node as usize;
            if edge >= self.edges.len() || !self.edge_occupied(edge) {
                return None;
            }
            loop {
                if self.edge_letter(edge) == unichar_id && (!word_end || self.end_of_word(edge)) {
                    return Some(edge);
                }
                let last = self.last_edge(edge);
                let next_edge = edge + 1;
                if last || next_edge >= self.edges.len() {
                    return None;
                }
                edge = next_edge;
            }
        }
    }
}

/// A little-endian byte cursor over the dawg component — the reader half of
/// the `TFile` primitives this leaf needs (`FReadEndian` with
/// `swap_ == false`).
struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Bytes consumed so far.
    fn pos(&self) -> usize {
        self.pos
    }

    /// Advance over `n` bytes, or [`DawgError::UnexpectedEof`] if short.
    fn take(&mut self, n: usize) -> Result<&'a [u8], DawgError> {
        let end = self.pos.checked_add(n).ok_or(DawgError::UnexpectedEof)?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(DawgError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    fn read_i16(&mut self) -> Result<i16, DawgError> {
        let arr: [u8; 2] = self
            .take(2)?
            .try_into()
            .map_err(|_| DawgError::UnexpectedEof)?;
        Ok(i16::from_le_bytes(arr))
    }

    fn read_i32(&mut self) -> Result<i32, DawgError> {
        let arr: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| DawgError::UnexpectedEof)?;
        Ok(i32::from_le_bytes(arr))
    }

    fn read_u64(&mut self) -> Result<u64, DawgError> {
        let arr: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| DawgError::UnexpectedEof)?;
        Ok(u64::from_le_bytes(arr))
    }
}

/// A failure loading a `SquishedDawg`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DawgError {
    /// The buffer ended mid-field.
    UnexpectedEof,
    /// The magic number did not match `Dawg::kDawgMagicNumber` (42).
    BadMagic(i16),
    /// `num_edges` was zero or negative (the C++
    /// `ASSERT_HOST(num_edges_ > 0)`, "DAWG should not be empty").
    Empty,
    /// `unicharset_size` was zero or negative (the C++
    /// `ASSERT_HOST(unicharset_size > 0)` in `Dawg::init`).
    NonPositiveSize,
    /// The file could not be read (message from the underlying I/O error).
    Io(String),
}

impl std::fmt::Display for DawgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "dawg buffer ended mid-field"),
            Self::BadMagic(magic) => {
                write!(f, "dawg magic number {magic} != {DAWG_MAGIC_NUMBER}")
            }
            Self::Empty => write!(f, "dawg declared zero or negative edges"),
            Self::NonPositiveSize => write!(f, "dawg declared a non-positive unicharset_size"),
            Self::Io(msg) => write!(f, "dawg read failed: {msg}"),
        }
    }
}

impl std::error::Error for DawgError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `.lstm-*-dawg` byte buffer from `(unicharset_size, edges)` —
    /// the exact little-endian wire form `write_squished_dawg` writes
    /// (magic, then `unicharset_size`, then `num_edges`, then the edge
    /// array).
    fn build(unicharset_size: i32, edges: &[u64]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&DAWG_MAGIC_NUMBER.to_le_bytes());
        b.extend_from_slice(&unicharset_size.to_le_bytes());
        b.extend_from_slice(&i32::try_from(edges.len()).unwrap().to_le_bytes());
        for &e in edges {
            b.extend_from_slice(&e.to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_header_and_edges_with_hand_packed_records() {
        // unicharset_size=112 -> flag_start_bit=ceil(log2(113))=7 (log2(113)
        // ~= 6.8198, verified against both a Python and a Rust f64 spot
        // check). Layout for flag_start_bit=7: letter=bits[0,7),
        // marker=bit7, backward=bit8, eow=bit9, next_node=bits[10,64).
        //
        //   edge0: letter=5,   next=100,   eow=true
        //     = 5 | (100 << 10) | (4 << 7) = 5 + 102_400 + 512 = 0x0001_9205
        //   edge1: letter=111, next=0,     marker=true
        //     = 111 | (1 << 7) = 111 + 128 = 0x0000_00EF
        //   edge2: letter=0,   next=12_345, backward=true
        //     = (12_345 << 10) | (2 << 7) = 12_641_280 + 256 = 0x00C0_E500
        let bytes = build(112, &[0x0001_9205, 0x0000_00EF, 0x00C0_E500]);
        let (dawg, consumed) = SquishedDawg::from_le_bytes(&bytes).expect("valid");
        assert_eq!(consumed, bytes.len());
        assert_eq!(dawg.unicharset_size(), 112);
        assert_eq!(dawg.num_edges(), 3);
        assert_eq!(dawg.flag_start_bit(), 7);

        assert_eq!(dawg.edge_letter(0), 5);
        assert_eq!(dawg.next_node(0), 100);
        assert!(!dawg.marker_flag(0));
        assert!(!dawg.is_backward(0));
        assert!(dawg.end_of_word(0));

        assert_eq!(dawg.edge_letter(1), 111);
        assert_eq!(dawg.next_node(1), 0);
        assert!(dawg.marker_flag(1));
        assert!(!dawg.is_backward(1));
        assert!(!dawg.end_of_word(1));

        assert_eq!(dawg.edge_letter(2), 0);
        assert_eq!(dawg.next_node(2), 12_345);
        assert!(!dawg.marker_flag(2));
        assert!(dawg.is_backward(2));
        assert!(!dawg.end_of_word(2));
    }

    #[test]
    fn flag_start_bit_matches_ceil_log2() {
        // 63 -> log2(64) == 6.0 exactly; the others cross a non-power-of-two
        // boundary. One placeholder edge each so num_edges > 0.
        for (size, expected) in [(1_i32, 1_u32), (63, 6), (112, 7), (255, 8)] {
            let bytes = build(size, &[0]);
            let (dawg, _consumed) = SquishedDawg::from_le_bytes(&bytes).expect("valid");
            assert_eq!(dawg.flag_start_bit(), expected, "size={size}");
        }
    }

    #[test]
    fn bad_magic_errors() {
        let mut bytes = build(112, &[0]);
        bytes[0] = 0; // corrupt the magic's low byte (42 -> 0)
        assert_eq!(
            SquishedDawg::from_le_bytes(&bytes).unwrap_err(),
            DawgError::BadMagic(0)
        );
    }

    #[test]
    fn truncated_buffer_errors() {
        let mut bytes = build(112, &[0x1234_5678_9abc_def0]);
        bytes.pop(); // drop the last byte of the one edge
        assert_eq!(
            SquishedDawg::from_le_bytes(&bytes).unwrap_err(),
            DawgError::UnexpectedEof
        );
        // Header-only truncation (short before num_edges is even read).
        assert_eq!(
            SquishedDawg::from_le_bytes(&bytes[..5]).unwrap_err(),
            DawgError::UnexpectedEof
        );
    }

    #[test]
    fn non_positive_size_and_empty_error() {
        assert_eq!(
            SquishedDawg::from_le_bytes(&build(0, &[0])).unwrap_err(),
            DawgError::NonPositiveSize
        );
        assert_eq!(
            SquishedDawg::from_le_bytes(&build(-1, &[0])).unwrap_err(),
            DawgError::NonPositiveSize
        );
        assert_eq!(
            SquishedDawg::from_le_bytes(&build(112, &[])).unwrap_err(),
            DawgError::Empty
        );
    }

    /// Packs one `EDGE_RECORD` for `unicharset_size=112`
    /// (`flag_start_bit=7`, `next_node_start_bit=10`, matching the layout
    /// comment on `parses_header_and_edges_with_hand_packed_records`).
    fn pack_edge(letter: u64, marker: bool, backward: bool, eow: bool, next: u64) -> u64 {
        letter
            | (u64::from(marker) << 7)
            | (u64::from(backward) << 8)
            | (u64::from(eow) << 9)
            | (next << 10)
    }

    #[test]
    fn edge_char_of_binary_search_at_node_zero() {
        // Two node-0 forward edges: letters 5 and 7, both end_of_word=true,
        // last_edge set on the second (the marker that ends the run).
        let edges = [
            pack_edge(5, false, false, true, 0),
            pack_edge(7, true, false, true, 0),
        ];
        let bytes = build(112, &edges);
        let (dawg, _) = SquishedDawg::from_le_bytes(&bytes).expect("valid");
        assert_eq!(dawg.num_forward_edges_in_node0(), 2);

        assert_eq!(dawg.edge_char_of(0, 5, true), Some(0));
        assert_eq!(dawg.edge_char_of(0, 6, true), None);
        assert_eq!(dawg.edge_char_of(0, 7, true), Some(1));
    }

    #[test]
    fn edge_char_of_linear_scan_at_nonzero_node() {
        // edge0 is node 0's own (irrelevant here) forward edge, last_edge
        // set so num_forward_edges_in_node0 == 1. Node 1's edge run starts
        // at index 1: letters 20 (not last), 21 (last).
        let edges = [
            pack_edge(99, true, false, false, 0),
            pack_edge(20, false, false, false, 0),
            pack_edge(21, true, false, false, 0),
        ];
        let bytes = build(112, &edges);
        let (dawg, _) = SquishedDawg::from_le_bytes(&bytes).expect("valid");
        assert_eq!(dawg.num_forward_edges_in_node0(), 1);

        assert!(!dawg.last_edge(1));
        assert!(dawg.last_edge(2));
        assert!(dawg.edge_occupied(1));

        assert_eq!(dawg.edge_char_of(1, 20, false), Some(1));
        assert_eq!(dawg.edge_char_of(1, 21, false), Some(2));
        // last_edge fires at index 2 without a match -> stop, not found.
        assert_eq!(dawg.edge_char_of(1, 22, false), None);
    }

    #[test]
    fn num_forward_edges_in_node0_three_layouts() {
        // All-forward: three edges, last_edge set only on the third.
        let all_forward = [
            pack_edge(1, false, false, false, 0),
            pack_edge(2, false, false, false, 0),
            pack_edge(3, true, false, false, 0),
        ];
        let (dawg, _) = SquishedDawg::from_le_bytes(&build(112, &all_forward)).expect("valid");
        assert_eq!(dawg.num_forward_edges_in_node0(), 3);

        // One backward edge mixed in at index 1: the count stops there
        // (not counted), matching "forward edges only" at node 0.
        let one_backward = [
            pack_edge(1, false, false, false, 0),
            pack_edge(2, false, true, false, 0),
            pack_edge(3, true, false, false, 0),
        ];
        let (dawg, _) = SquishedDawg::from_le_bytes(&build(112, &one_backward)).expect("valid");
        assert_eq!(dawg.num_forward_edges_in_node0(), 1);

        // Single edge with last_edge set: a one-edge run.
        let single_last = [pack_edge(9, true, false, false, 0)];
        let (dawg, _) = SquishedDawg::from_le_bytes(&build(112, &single_last)).expect("valid");
        assert_eq!(dawg.num_forward_edges_in_node0(), 1);
    }

    #[test]
    fn dawg_type_and_permuter_round_trip() {
        for (v, expected) in [
            (0, Some(DawgType::Punctuation)),
            (1, Some(DawgType::Word)),
            (2, Some(DawgType::Number)),
            (3, Some(DawgType::Pattern)),
            (4, None),
            (-1, None),
        ] {
            assert_eq!(DawgType::from_i32(v), expected, "v={v}");
        }
        for v in 0..=12 {
            let p = PermuterType::from_i32(v).unwrap_or_else(|| panic!("v={v} should map"));
            assert_eq!(p.as_i32(), v);
        }
        assert_eq!(PermuterType::from_i32(13), None);
        assert_eq!(PermuterType::from_i32(-1), None);
        assert!(PermuterType::NoPerm < PermuterType::CompoundPerm);

        let bytes = build(112, &[pack_edge(1, true, false, false, 0)]);
        let (dawg, _) = SquishedDawg::from_le_bytes(&bytes).expect("valid");
        // Defaults on load.
        assert_eq!(dawg.dawg_type(), DawgType::Word);
        assert_eq!(dawg.permuter(), PermuterType::SystemDawgPerm);

        let dawg = dawg
            .with_type(DawgType::Punctuation)
            .with_permuter(PermuterType::PuncPerm);
        assert_eq!(dawg.dawg_type(), DawgType::Punctuation);
        assert_eq!(dawg.permuter(), PermuterType::PuncPerm);
    }

    #[test]
    fn edge_char_of_real_punc_dawg_matches_letters() {
        // Spot check on real trained data: every node-0 forward edge must
        // be re-discoverable via edge_char_of using its own decoded
        // (letter, end_of_word) pair. Full byte-parity for the walker
        // belongs to its own oracle, not this loader's unit tests.
        let path = Path::new("/tmp/eng.lstm-punc-dawg");
        if !path.exists() {
            eprintln!("skipping: /tmp/eng.lstm-punc-dawg not present in this environment");
            return;
        }
        let dawg = SquishedDawg::load_from_file(path).expect("load real punc dawg");
        assert!(dawg.num_forward_edges_in_node0() > 0);
        for i in 0..dawg.num_forward_edges_in_node0() {
            let letter = dawg.edge_letter(i);
            let word_end = dawg.end_of_word(i);
            let found = dawg
                .edge_char_of(0, letter, word_end)
                .unwrap_or_else(|| panic!("edge {i} (letter {letter}) not found"));
            assert_eq!(dawg.edge_letter(found), letter);
            if word_end {
                assert!(dawg.end_of_word(found));
            }
        }
    }
}
