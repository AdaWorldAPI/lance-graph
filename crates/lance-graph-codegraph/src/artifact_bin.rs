//! The zero-copy reader and verifier for `<crate>.rows.bin` (chunk B1 of
//! `.claude/plans/d-codegraph-0-binary-lanes-v1.md`).
//!
//! The byte layout is owned by the writer, `ruff_charon::store_bin`; this
//! module is a separate implementation that accepts exactly what that writer
//! writes and refuses exactly what its `read` refuses, in the same check
//! order. There is no dependency on ruff.
//!
//! A [`CrateFile`] owns ONE `Vec<u8>`. [`CrateFile::parse`] verifies the whole
//! file once (header, length, digest, directory, shapes, every id against its
//! table, every codeword, UTF-8) and records the byte range of each lane;
//! every accessor afterwards is a lens over that buffer: it reads
//! little-endian integers in place and returns borrowed `&str`s. No `String`
//! is built per row.
//!
//! **Validation invariant.** Every accessor assumes `parse` succeeded. After
//! that, the ranges it records are inside the buffer and have the lane
//! lengths the header promises, so the accessors use plain indexing (a row or
//! symbol id out of range is a caller bug and panics). The only slice
//! operations that can fail at parse time use `get` and return a refusal.
//! `&str` accessors re-check UTF-8 over the one label they return (the bytes
//! were validated at parse, so this cannot fail; the fallback is `""`).
//! There is no `unsafe` and no native-endian cast.

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

use crate::artifact::digest;

/// Length of the fixed header.
pub const HEADER_LEN: usize = 64;
/// An absent id, span, owner, dispatch target or function declaration.
pub const NO_ID: u32 = u32::MAX;

const MAGIC: [u8; 8] = *b"CHRNROWS";
const MAJOR: u16 = 1;
const MINOR: u16 = 0;
const FLAG_PARTIAL: u32 = 1;
const CODEC9_MAX: usize = 511;
const KIND_MAX: usize = 3;
const DIR_ENTRY_LEN: usize = 16;
const PROV_FIELDS: usize = 7;

const ENC_RAW: u16 = 0;
const ENC_CODEC9: u16 = 1;
const ENC_PRES: u16 = 2;
const ENC_KV: u16 = 3;

/// Section tags. Bit 31 means must-understand; every tag here has it set.
pub mod tag {
    /// Predicate magnitude lane.
    pub const PRED_MAG: u32 = 0x8000_0001;
    /// Predicate presence plane.
    pub const PRED_PRES: u32 = 0x8000_0002;
    /// Predicate codebook.
    pub const PRED_CB: u32 = 0x8000_0003;
    /// Grade magnitude lane.
    pub const GRADE_MAG: u32 = 0x8000_0004;
    /// Grade presence plane.
    pub const GRADE_PRES: u32 = 0x8000_0005;
    /// Grade codebook.
    pub const GRADE_CB: u32 = 0x8000_0006;
    /// File lane: codec9 magnitude bytes, or raw `u32` when there are more than 511 files.
    pub const FILE: u32 = 0x8000_0007;
    /// File presence plane, present only when `FILE` is codec9.
    pub const FILE_PRES: u32 = 0x8000_0008;
    /// File codebook.
    pub const FILE_CB: u32 = 0x8000_0009;
    /// Subject symbol ids.
    pub const SUBJ: u32 = 0x8000_000A;
    /// Object symbol ids.
    pub const OBJ: u32 = 0x8000_000B;
    /// Dispatch-target symbol ids (`MAY_DISPATCH` rows only).
    pub const MAYT: u32 = 0x8000_000C;
    /// Span starts.
    pub const START: u32 = 0x8000_000D;
    /// Span ends.
    pub const END: u32 = 0x8000_000E;
    /// Symbol blob offsets, `n_symbols + 1` entries.
    pub const SYM_OFF: u32 = 0x8000_000F;
    /// Symbol label bytes.
    pub const SYM_BLOB: u32 = 0x8000_0010;
    /// Per-symbol digest of the label.
    pub const SYM_HASH: u32 = 0x8000_0011;
    /// Per-symbol owner id.
    pub const SYM_OWNER: u32 = 0x8000_0012;
    /// Per-row Charon function declaration index.
    pub const FUNDECL: u32 = 0x8000_0013;
    /// Object-kind magnitude lane.
    pub const OBJ_KIND_MAG: u32 = 0x8000_0014;
    /// Object-kind presence plane.
    pub const OBJ_KIND_PRES: u32 = 0x8000_0015;
    /// Provenance record.
    pub const PROV: u32 = 0x8000_0016;
    /// Every tag a version-1 reader understands, in ascending order.
    pub const ALL: [u32; 22] = [
        PRED_MAG,
        PRED_PRES,
        PRED_CB,
        GRADE_MAG,
        GRADE_PRES,
        GRADE_CB,
        FILE,
        FILE_PRES,
        FILE_CB,
        SUBJ,
        OBJ,
        MAYT,
        START,
        END,
        SYM_OFF,
        SYM_BLOB,
        SYM_HASH,
        SYM_OWNER,
        FUNDECL,
        OBJ_KIND_MAG,
        OBJ_KIND_PRES,
        PROV,
    ];
}

/// Why a `.rows.bin` file was refused; one variant per refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinRefusal {
    /// Wrong magic.
    Magic,
    /// A major version this reader does not know.
    Major(u16),
    /// A minor version newer than this reader.
    Minor(u16),
    /// Non-zero reserved header bytes.
    Reserved,
    /// The file is shorter or longer than its header says.
    Truncated {
        /// Length recorded in the header (0 when the header itself is cut).
        header: u64,
        /// Actual byte length.
        actual: usize,
    },
    /// The body does not match the header digest.
    Digest {
        /// Digest recorded in the header.
        header: u64,
        /// Digest of the body.
        actual: u64,
    },
    /// The section directory does not fit in the file.
    Directory,
    /// A directory entry that is misaligned, out of order, overlapping,
    /// outside the file or duplicated.
    Section {
        /// The entry's tag.
        tag: u32,
    },
    /// An unknown section with the must-understand bit.
    UnknownRequired {
        /// The tag.
        tag: u32,
    },
    /// A required section is absent.
    MissingSection {
        /// The tag.
        tag: u32,
    },
    /// A section whose width, encoding or length is wrong.
    LaneLength {
        /// The tag.
        tag: u32,
    },
    /// An id at or past the end of the table it names.
    IdRange {
        /// The lane.
        tag: u32,
        /// The row (or symbol) index.
        row: usize,
    },
    /// A string that is not valid UTF-8, or an offset inside a character.
    Utf8 {
        /// The section.
        tag: u32,
    },
    /// A codeword that is not allowed in its lane.
    Codeword {
        /// The lane.
        tag: u32,
        /// The row.
        row: usize,
    },
}

impl fmt::Display for BinRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Magic => write!(f, "bad magic"),
            Self::Major(v) => write!(f, "unsupported major version {v}"),
            Self::Minor(v) => write!(f, "unsupported minor version {v}"),
            Self::Reserved => write!(f, "reserved header bytes are not zero"),
            Self::Truncated { header, actual } => {
                write!(f, "file is {actual} bytes, header says {header}")
            }
            Self::Digest { header, actual } => {
                write!(f, "digest mismatch: header {header:#x}, body {actual:#x}")
            }
            Self::Directory => write!(f, "section directory does not fit in the file"),
            Self::Section { tag } => write!(f, "bad directory entry for section {tag:#x}"),
            Self::UnknownRequired { tag } => write!(f, "unknown required section {tag:#x}"),
            Self::MissingSection { tag } => write!(f, "missing section {tag:#x}"),
            Self::LaneLength { tag } => write!(f, "section {tag:#x} has the wrong shape"),
            Self::IdRange { tag, row } => write!(f, "id out of range in {tag:#x} at {row}"),
            Self::Utf8 { tag } => write!(f, "invalid UTF-8 in section {tag:#x}"),
            Self::Codeword { tag, row } => write!(f, "bad codeword in {tag:#x} at {row}"),
        }
    }
}

impl std::error::Error for BinRefusal {}

// ---------------------------------------------------------------------------
// header
// ---------------------------------------------------------------------------

/// The fixed 64-byte header, read without touching the body (the
/// unchanged-refresh path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Flag bits; bit 0 is "partial".
    pub flags: u32,
    /// Digest of every byte from offset 32 on.
    pub digest: u64,
    /// Total file length in bytes.
    pub file_len: u64,
    /// Row count.
    pub n_rows: u32,
    /// Symbol count.
    pub n_symbols: u32,
    /// File-name codebook size.
    pub n_files: u32,
    /// Directory entry count.
    pub n_sections: u32,
}

fn le16(b: &[u8], o: usize) -> u16 {
    b.get(o..o + 2)
        .and_then(|s| s.first_chunk::<2>())
        .map_or(0, |c| u16::from_le_bytes(*c))
}

fn le32(b: &[u8], o: usize) -> u32 {
    b.get(o..o + 4)
        .and_then(|s| s.first_chunk::<4>())
        .map_or(0, |c| u32::from_le_bytes(*c))
}

fn le64(b: &[u8], o: usize) -> u64 {
    b.get(o..o + 8)
        .and_then(|s| s.first_chunk::<8>())
        .map_or(0, |c| u64::from_le_bytes(*c))
}

fn len_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

impl Header {
    /// Validates magic, major, minor and reserved; never reads past byte 64.
    ///
    /// # Errors
    /// [`BinRefusal::Magic`], [`BinRefusal::Major`], [`BinRefusal::Minor`] or
    /// [`BinRefusal::Reserved`], checked in that order.
    pub fn parse(bytes: &[u8; HEADER_LEN]) -> Result<Self, BinRefusal> {
        if bytes[0..8] != MAGIC {
            return Err(BinRefusal::Magic);
        }
        let major = le16(bytes, 8);
        if major != MAJOR {
            return Err(BinRefusal::Major(major));
        }
        let minor = le16(bytes, 10);
        if minor > MINOR {
            return Err(BinRefusal::Minor(minor));
        }
        if bytes[48..64].iter().any(|&b| b != 0) {
            return Err(BinRefusal::Reserved);
        }
        Ok(Self {
            flags: le32(bytes, 12),
            digest: le64(bytes, 16),
            file_len: le64(bytes, 24),
            n_rows: le32(bytes, 32),
            n_symbols: le32(bytes, 36),
            n_files: le32(bytes, 40),
            n_sections: le32(bytes, 44),
        })
    }

    /// `read_exact` of exactly 64 bytes from `r`, then [`Header::parse`]. No
    /// body buffer is allocated.
    ///
    /// # Errors
    /// The I/O error of `read_exact` (a file shorter than 64 bytes is
    /// `UnexpectedEof`); the inner result carries the header refusal.
    pub fn read(r: &mut impl std::io::Read) -> std::io::Result<Result<Self, BinRefusal>> {
        let mut b = [0u8; HEADER_LEN];
        r.read_exact(&mut b)?;
        Ok(Self::parse(&b))
    }

    /// Whether the crate description was partial (Charon reported errors).
    pub fn partial(&self) -> bool {
        self.flags & FLAG_PARTIAL != 0
    }
}

// ---------------------------------------------------------------------------
// provenance
// ---------------------------------------------------------------------------

/// One provenance record. `None` is UNKNOWN and is distinct from every value,
/// including the string `"UNKNOWN"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prov {
    /// Charon version.
    pub charon_version: Option<String>,
    /// Charon commit.
    pub charon_commit: Option<String>,
    /// LLBC schema version.
    pub llbc_schema_version: Option<String>,
    /// Digest of the LLBC file.
    pub llbc_digest: Option<String>,
    /// rustc commit.
    pub rustc_commit: Option<String>,
    /// Target configuration.
    pub target_config: Option<String>,
    /// Revision of the analysed source.
    pub source_revision: Option<String>,
}

// ---------------------------------------------------------------------------
// layout of a verified file
// ---------------------------------------------------------------------------

type R = Range<usize>;

/// A codec9 id lane: magnitude bytes plus a presence plane (byte ranges).
#[derive(Debug, Clone)]
struct Lane9 {
    mag: R,
    pres: R,
}

/// The file lane is codec9 up to 511 files, raw `u32` beyond.
#[derive(Debug, Clone)]
enum FileLane {
    Codec9(Lane9),
    Raw(R),
}

/// A string codebook: `n` entries, `n + 1` offsets, then the blob.
#[derive(Debug, Clone)]
struct Book {
    n: usize,
    offs: R,
    blob: usize,
}

/// Everything a verified file's accessors need, as byte ranges into the buffer.
#[derive(Debug, Clone)]
struct Layout {
    pred: Lane9,
    grade: Lane9,
    kind: Lane9,
    file: FileLane,
    pred_cb: Book,
    grade_cb: Book,
    file_cb: Book,
    subj: R,
    obj: R,
    mayt: R,
    start: R,
    end: R,
    fundecl: R,
    sym_off: R,
    sym_blob: R,
    sym_hash: R,
    sym_owner: R,
    prov: Vec<Prov>,
}

struct Sec {
    width: u16,
    enc: u16,
    range: R,
}

type Secs = BTreeMap<u32, Sec>;

fn find(secs: &Secs, tag: u32) -> Result<&Sec, BinRefusal> {
    secs.get(&tag).ok_or(BinRefusal::MissingSection { tag })
}

fn lane_len(n: usize, width: usize, tag: u32) -> Result<usize, BinRefusal> {
    n.checked_mul(width).ok_or(BinRefusal::LaneLength { tag })
}

/// A fixed-shape section: exact element width, encoding and byte length.
fn lane(secs: &Secs, tag: u32, width: u16, enc: u16, len: usize) -> Result<R, BinRefusal> {
    let s = find(secs, tag)?;
    if s.width != width || s.enc != enc || s.range.len() != len {
        return Err(BinRefusal::LaneLength { tag });
    }
    Ok(s.range.clone())
}

fn lane9(secs: &Secs, mag: u32, pres: u32, n: usize) -> Result<Lane9, BinRefusal> {
    let m = lane(secs, mag, 1, ENC_CODEC9, n)?;
    let p = lane(secs, pres, 8, ENC_PRES, lane_len(n.div_ceil(64), 8, pres)?)?;
    Ok(Lane9 { mag: m, pres: p })
}

fn chunks4(b: &[u8]) -> &[[u8; 4]] {
    b.as_chunks::<4>().0
}

fn chunks8(b: &[u8]) -> &[[u8; 8]] {
    b.as_chunks::<8>().0
}

/// Only zero bytes, and fewer than 8 of them, may follow the last field.
fn only_padding(rest: &[u8], tag: u32) -> Result<(), BinRefusal> {
    if rest.len() < 8 && rest.iter().all(|&b| b == 0) {
        Ok(())
    } else {
        Err(BinRefusal::LaneLength { tag })
    }
}

/// Validate a KV codebook section and record where its offsets and blob are.
fn book(buf: &[u8], tag: u32, sec: &Sec) -> Result<Book, BinRefusal> {
    let bad = BinRefusal::LaneLength { tag };
    let data = &buf[sec.range.clone()];
    if !data.len().is_multiple_of(8) {
        return Err(bad);
    }
    let n = usize::try_from(le32(data, 0)).map_err(|_| bad.clone())?;
    let table = n
        .checked_add(1)
        .and_then(|k| k.checked_mul(4))
        .and_then(|k| k.checked_add(4))
        .filter(|&k| k <= data.len())
        .ok_or_else(|| bad.clone())?;
    let offs = chunks4(&data[4..table])
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect::<Vec<u32>>();
    let blob = &data[table..];
    let blob_len = offs
        .last()
        .and_then(|&o| usize::try_from(o).ok())
        .ok_or_else(|| bad.clone())?;
    if offs.first() != Some(&0) || blob_len > blob.len() || offs.windows(2).any(|w| w[0] > w[1]) {
        return Err(bad);
    }
    only_padding(&blob[blob_len..], tag)?;
    for w in offs.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        std::str::from_utf8(&blob[a..b]).map_err(|_| BinRefusal::Utf8 { tag })?;
    }
    Ok(Book {
        n,
        offs: sec.range.start + 4..sec.range.start + table,
        blob: sec.range.start + table,
    })
}

fn decode_prov(data: &[u8]) -> Result<Vec<Prov>, BinRefusal> {
    let tag = tag::PROV;
    let bad = BinRefusal::LaneLength { tag };
    if !data.len().is_multiple_of(8) || data.len() < 4 {
        return Err(bad);
    }
    let n = le32(data, 0);
    if n == 0 {
        return Err(bad);
    }
    let mut pos = 4usize;
    let mut out = Vec::new();
    for _ in 0..n {
        let mut f: Vec<Option<String>> = Vec::with_capacity(PROV_FIELDS);
        for _ in 0..PROV_FIELDS {
            let len = data
                .get(pos..pos + 4)
                .map(|_| le32(data, pos))
                .ok_or_else(|| bad.clone())?;
            pos += 4;
            if len == NO_ID {
                f.push(None);
                continue;
            }
            let end = usize::try_from(len)
                .ok()
                .and_then(|l| pos.checked_add(l))
                .filter(|&e| e <= data.len())
                .ok_or_else(|| bad.clone())?;
            let s = std::str::from_utf8(&data[pos..end]).map_err(|_| BinRefusal::Utf8 { tag })?;
            f.push(Some(s.to_owned()));
            pos = end;
        }
        let mut it = f.into_iter();
        let mut next = || it.next().flatten();
        out.push(Prov {
            charon_version: next(),
            charon_commit: next(),
            llbc_schema_version: next(),
            llbc_digest: next(),
            rustc_commit: next(),
            target_config: next(),
            source_revision: next(),
        });
    }
    only_padding(&data[pos..], tag)?;
    Ok(out)
}

/// Codeword of `row` in a codec9 lane; `buf` is verified.
fn codeword_of(buf: &[u8], l: &Lane9, row: usize) -> usize {
    let m = usize::from(buf[l.mag.start + row]);
    let words = chunks8(&buf[l.pres.clone()]);
    let p = (u64::from_le_bytes(words[row / 64]) >> (row % 64)) & 1;
    m | (usize::try_from(p).unwrap_or(0) << 8)
}

/// Parse the directory of a file whose header, length and digest are verified.
fn directory(buf: &[u8], h: &Header) -> Result<Secs, BinRefusal> {
    let to_usize = |v: u32| usize::try_from(v).map_err(|_| BinRefusal::Directory);
    let n_sections = to_usize(h.n_sections)?;
    let dir_end = n_sections
        .checked_mul(DIR_ENTRY_LEN)
        .and_then(|d| d.checked_add(HEADER_LEN))
        .filter(|&e| e <= buf.len())
        .ok_or(BinRefusal::Directory)?;

    let mut secs: Secs = BTreeMap::new();
    let mut prev_end = dir_end;
    for i in 0..n_sections {
        let at = HEADER_LEN + i * DIR_ENTRY_LEN;
        let t = le32(buf, at);
        let offset = to_usize(le32(buf, at + 8))?;
        let len = to_usize(le32(buf, at + 12))?;
        let end = offset.checked_add(len).filter(|&e| e <= buf.len());
        let Some(end) = end.filter(|_| offset % 8 == 0 && offset >= prev_end) else {
            return Err(BinRefusal::Section { tag: t });
        };
        prev_end = end;
        let sec = Sec {
            width: le16(buf, at + 4),
            enc: le16(buf, at + 6),
            range: offset..end,
        };
        if secs.insert(t, sec).is_some() {
            return Err(BinRefusal::Section { tag: t });
        }
    }
    for &t in secs.keys() {
        if t & 0x8000_0000 != 0 && !tag::ALL.contains(&t) {
            return Err(BinRefusal::UnknownRequired { tag: t });
        }
    }
    Ok(secs)
}

/// A codebook codeword: 0 is refused (`Codeword`), past the book is `IdRange`.
fn check_cw(c: usize, n: usize, tag: u32, row: usize) -> Result<(), BinRefusal> {
    if c == 0 {
        Err(BinRefusal::Codeword { tag, row })
    } else if c > n {
        Err(BinRefusal::IdRange { tag, row })
    } else {
        Ok(())
    }
}

fn check_sym(id: u32, n_syms: usize, tag: u32, row: usize) -> Result<(), BinRefusal> {
    if usize::try_from(id).is_ok_and(|i| i < n_syms) {
        Ok(())
    } else {
        Err(BinRefusal::IdRange { tag, row })
    }
}

/// Verify everything after the digest, in the writer-reader's check order.
#[expect(clippy::too_many_lines)] // one pass that validates every lane
fn layout(buf: &[u8], h: &Header) -> Result<Layout, BinRefusal> {
    let secs = directory(buf, h)?;
    for t in tag::ALL {
        if t != tag::FILE_PRES && !secs.contains_key(&t) {
            return Err(BinRefusal::MissingSection { tag: t });
        }
    }
    let n = usize::try_from(h.n_rows).unwrap_or(usize::MAX);
    let n_syms = usize::try_from(h.n_symbols).unwrap_or(usize::MAX);
    let n_files = usize::try_from(h.n_files).unwrap_or(usize::MAX);

    let pred = lane9(&secs, tag::PRED_MAG, tag::PRED_PRES, n)?;
    let grade = lane9(&secs, tag::GRADE_MAG, tag::GRADE_PRES, n)?;
    let kind = lane9(&secs, tag::OBJ_KIND_MAG, tag::OBJ_KIND_PRES, n)?;
    let pred_cb = book(buf, tag::PRED_CB, find(&secs, tag::PRED_CB)?)?;
    let grade_cb = book(buf, tag::GRADE_CB, find(&secs, tag::GRADE_CB)?)?;
    let file_cb = book(buf, tag::FILE_CB, find(&secs, tag::FILE_CB)?)?;
    for (t, b) in [(tag::PRED_CB, &pred_cb), (tag::GRADE_CB, &grade_cb)] {
        if find(&secs, t)?.enc != ENC_KV || b.n > CODEC9_MAX {
            return Err(BinRefusal::LaneLength { tag: t });
        }
    }
    if find(&secs, tag::FILE_CB)?.enc != ENC_KV || file_cb.n != n_files {
        return Err(BinRefusal::LaneLength { tag: tag::FILE_CB });
    }

    let file = if n_files <= CODEC9_MAX {
        FileLane::Codec9(lane9(&secs, tag::FILE, tag::FILE_PRES, n)?)
    } else {
        if secs.contains_key(&tag::FILE_PRES) {
            return Err(BinRefusal::LaneLength {
                tag: tag::FILE_PRES,
            });
        }
        FileLane::Raw(lane(
            &secs,
            tag::FILE,
            4,
            ENC_RAW,
            lane_len(n, 4, tag::FILE)?,
        )?)
    };

    let u32_lane = |t: u32| lane(&secs, t, 4, ENC_RAW, lane_len(n, 4, t)?);
    let subj = u32_lane(tag::SUBJ)?;
    let obj = u32_lane(tag::OBJ)?;
    let mayt = u32_lane(tag::MAYT)?;
    let start = u32_lane(tag::START)?;
    let end = u32_lane(tag::END)?;
    let fundecl = u32_lane(tag::FUNDECL)?;

    let off_len = lane_len(
        n_syms
            .checked_add(1)
            .ok_or(BinRefusal::LaneLength { tag: tag::SYM_OFF })?,
        4,
        tag::SYM_OFF,
    )?;
    let sym_off = lane(&secs, tag::SYM_OFF, 4, ENC_RAW, off_len)?;
    let blob_sec = find(&secs, tag::SYM_BLOB)?;
    if blob_sec.width != 1 || blob_sec.enc != ENC_RAW {
        return Err(BinRefusal::LaneLength { tag: tag::SYM_BLOB });
    }
    let sym_blob = blob_sec.range.clone();
    let blob_str = std::str::from_utf8(&buf[sym_blob.clone()])
        .map_err(|_| BinRefusal::Utf8 { tag: tag::SYM_BLOB })?;
    let offs: Vec<u32> = chunks4(&buf[sym_off.clone()])
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let ok_offsets = offs.first() == Some(&0)
        && offs.last().and_then(|&o| usize::try_from(o).ok()) == Some(blob_str.len())
        && offs.windows(2).all(|w| w[0] <= w[1]);
    if !ok_offsets {
        return Err(BinRefusal::LaneLength { tag: tag::SYM_OFF });
    }
    for w in offs.windows(2) {
        if !blob_str.is_char_boundary(w[0] as usize) || !blob_str.is_char_boundary(w[1] as usize) {
            return Err(BinRefusal::Utf8 { tag: tag::SYM_BLOB });
        }
    }
    let sym_hash = lane(
        &secs,
        tag::SYM_HASH,
        8,
        ENC_RAW,
        lane_len(n_syms, 8, tag::SYM_HASH)?,
    )?;
    let sym_owner = lane(
        &secs,
        tag::SYM_OWNER,
        4,
        ENC_RAW,
        lane_len(n_syms, 4, tag::SYM_OWNER)?,
    )?;
    for (i, c) in chunks4(&buf[sym_owner.clone()]).iter().enumerate() {
        let o = u32::from_le_bytes(*c);
        if o != NO_ID && usize::try_from(o).is_ok_and(|o| o >= n_syms) {
            return Err(BinRefusal::IdRange {
                tag: tag::SYM_OWNER,
                row: i,
            });
        }
    }
    let prov_sec = find(&secs, tag::PROV)?;
    if prov_sec.width != 1 || prov_sec.enc != ENC_KV {
        return Err(BinRefusal::LaneLength { tag: tag::PROV });
    }
    let prov = decode_prov(&buf[prov_sec.range.clone()])?;

    let at = |r: &R| chunks4(&buf[r.clone()]);
    let (subj_v, obj_v, mayt_v) = (at(&subj), at(&obj), at(&mayt));
    let (start_v, end_v) = (at(&start), at(&end));
    for row in 0..n {
        let kc = codeword_of(buf, &kind, row);
        if kc == 0 || kc > KIND_MAX {
            return Err(BinRefusal::Codeword {
                tag: tag::OBJ_KIND_MAG,
                row,
            });
        }
        check_cw(codeword_of(buf, &pred, row), pred_cb.n, tag::PRED_MAG, row)?;
        check_cw(
            codeword_of(buf, &grade, row),
            grade_cb.n,
            tag::GRADE_MAG,
            row,
        )?;
        check_sym(u32::from_le_bytes(subj_v[row]), n_syms, tag::SUBJ, row)?;
        check_sym(u32::from_le_bytes(obj_v[row]), n_syms, tag::OBJ, row)?;
        let m = u32::from_le_bytes(mayt_v[row]);
        if m != NO_ID {
            check_sym(m, n_syms, tag::MAYT, row)?;
        }
        let has_file = match &file {
            FileLane::Codec9(l) => match codeword_of(buf, l, row) {
                0 => false,
                c => {
                    check_cw(c, file_cb.n, tag::FILE, row)?;
                    true
                }
            },
            FileLane::Raw(r) => match u32::from_le_bytes(chunks4(&buf[r.clone()])[row]) {
                NO_ID => false,
                v => {
                    if usize::try_from(v).is_ok_and(|i| i < file_cb.n) {
                        true
                    } else {
                        return Err(BinRefusal::IdRange {
                            tag: tag::FILE,
                            row,
                        });
                    }
                }
            },
        };
        let (s, e) = (
            u32::from_le_bytes(start_v[row]),
            u32::from_le_bytes(end_v[row]),
        );
        if has_file {
            if s == NO_ID {
                return Err(BinRefusal::IdRange {
                    tag: tag::START,
                    row,
                });
            }
            if e == NO_ID {
                return Err(BinRefusal::IdRange { tag: tag::END, row });
            }
        } else {
            if s != NO_ID {
                return Err(BinRefusal::IdRange {
                    tag: tag::START,
                    row,
                });
            }
            if e != NO_ID {
                return Err(BinRefusal::IdRange { tag: tag::END, row });
            }
        }
    }

    Ok(Layout {
        pred,
        grade,
        kind,
        file,
        pred_cb,
        grade_cb,
        file_cb,
        subj,
        obj,
        mayt,
        start,
        end,
        fundecl,
        sym_off,
        sym_blob,
        sym_hash,
        sym_owner,
        prov,
    })
}

// ---------------------------------------------------------------------------
// the verified file
// ---------------------------------------------------------------------------

/// One crate file, owned as ONE buffer; every accessor is a lens over it (no
/// copies, no per-row `String`).
///
/// All accessors assume [`CrateFile::parse`] succeeded and index the buffer
/// directly; a row index `>= n_rows()` or a symbol id `>= n_symbols()` panics.
#[derive(Debug, Clone)]
pub struct CrateFile {
    buf: Vec<u8>,
    header: Header,
    lay: Layout,
}

impl CrateFile {
    /// Full verification in the ruff reader's check order: header, file
    /// length, digest over `[32..]`, directory, required and unknown tags,
    /// lane shapes, then every id against its table, every codeword and all
    /// UTF-8, once.
    ///
    /// # Errors
    /// The first [`BinRefusal`] found.
    pub fn parse(buf: Vec<u8>) -> Result<Self, BinRefusal> {
        let Some(h) = buf.first_chunk::<HEADER_LEN>() else {
            return Err(BinRefusal::Truncated {
                header: 0,
                actual: buf.len(),
            });
        };
        let header = Header::parse(h)?;
        if header.file_len != len_u64(buf.len()) {
            return Err(BinRefusal::Truncated {
                header: header.file_len,
                actual: buf.len(),
            });
        }
        let actual = digest(&buf[32..]);
        if actual != header.digest {
            return Err(BinRefusal::Digest {
                header: header.digest,
                actual,
            });
        }
        let lay = layout(&buf, &header)?;
        Ok(Self { buf, header, lay })
    }

    /// The header, as verified.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Row count.
    pub fn n_rows(&self) -> usize {
        self.lay.subj.len() / 4
    }

    /// Symbol count.
    pub fn n_symbols(&self) -> usize {
        self.lay.sym_hash.len() / 8
    }

    fn u32_at(&self, r: &R, i: usize) -> u32 {
        u32::from_le_bytes(chunks4(&self.buf[r.clone()])[i])
    }

    fn book_label(&self, b: &Book, codeword: usize) -> &str {
        let i = codeword - 1;
        let o = chunks4(&self.buf[b.offs.clone()]);
        let (lo, hi) = (u32::from_le_bytes(o[i]), u32::from_le_bytes(o[i + 1]));
        let s = b.blob + lo as usize;
        let e = b.blob + hi as usize;
        std::str::from_utf8(&self.buf[s..e]).unwrap_or("")
    }

    /// The label of symbol `id`, a borrowed slice of the symbol blob.
    pub fn symbol(&self, id: u32) -> &str {
        let i = id as usize;
        let lo = self.u32_at(&self.lay.sym_off, i) as usize;
        let hi = self.u32_at(&self.lay.sym_off, i + 1) as usize;
        let base = self.lay.sym_blob.start;
        std::str::from_utf8(&self.buf[base + lo..base + hi]).unwrap_or("")
    }

    /// The writer's digest of symbol `id`'s label.
    pub fn sym_hash(&self, id: u32) -> u64 {
        let r = &self.lay.sym_hash;
        u64::from_le_bytes(chunks8(&self.buf[r.clone()])[id as usize])
    }

    /// The owner symbol of `id`; `None` when it has none.
    pub fn sym_owner(&self, id: u32) -> Option<u32> {
        match self.u32_at(&self.lay.sym_owner, id as usize) {
            NO_ID => None,
            v => Some(v),
        }
    }

    /// Subject symbol id of `row`.
    pub fn subject(&self, row: usize) -> u32 {
        self.u32_at(&self.lay.subj, row)
    }

    /// Object symbol id of `row`.
    pub fn object(&self, row: usize) -> u32 {
        self.u32_at(&self.lay.obj, row)
    }

    /// Dispatch-target symbol id of a `MAY_DISPATCH` row; `None` otherwise.
    pub fn may_target(&self, row: usize) -> Option<u32> {
        match self.u32_at(&self.lay.mayt, row) {
            NO_ID => None,
            v => Some(v),
        }
    }

    /// The codeword (`1..=511`) of `row`'s predicate.
    pub fn predicate_id(&self, row: usize) -> u16 {
        u16::try_from(codeword_of(&self.buf, &self.lay.pred, row)).unwrap_or(0)
    }

    /// The predicate label of `row`, borrowed from the codebook.
    pub fn predicate(&self, row: usize) -> &str {
        let c = codeword_of(&self.buf, &self.lay.pred, row);
        self.book_label(&self.lay.pred_cb, c)
    }

    /// The grade label of `row`, borrowed from the codebook.
    pub fn grade(&self, row: usize) -> &str {
        let c = codeword_of(&self.buf, &self.lay.grade, row);
        self.book_label(&self.lay.grade_cb, c)
    }

    /// The source file of `row`; `None` when the row has no site.
    pub fn file(&self, row: usize) -> Option<&str> {
        let idx = match &self.lay.file {
            FileLane::Codec9(l) => codeword_of(&self.buf, l, row).checked_sub(1)?,
            FileLane::Raw(r) => match self.u32_at(r, row) {
                NO_ID => return None,
                v => v as usize,
            },
        };
        Some(self.book_label(&self.lay.file_cb, idx + 1))
    }

    /// The `(start, end)` span of `row`; `None` when the row has no site.
    pub fn span(&self, row: usize) -> Option<(u32, u32)> {
        let s = self.u32_at(&self.lay.start, row);
        (s != NO_ID).then(|| (s, self.u32_at(&self.lay.end, row)))
    }

    /// The Charon function declaration index of `row`; `None` when absent.
    pub fn fundecl(&self, row: usize) -> Option<u32> {
        match self.u32_at(&self.lay.fundecl, row) {
            NO_ID => None,
            v => Some(v),
        }
    }

    /// The object kind of `row` (1 = SYMBOL).
    pub fn obj_kind(&self, row: usize) -> u16 {
        u16::try_from(codeword_of(&self.buf, &self.lay.kind, row)).unwrap_or(0)
    }

    /// The provenance records (one per file today).
    pub fn prov(&self) -> Vec<Prov> {
        self.lay.prov.clone()
    }
}
