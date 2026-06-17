//! `UNICHARSET` content store — the Rust side of the byte-parity probe
//! (`PROBE-OGAR-ADAPTER-UNICHARSET`).
//!
//! Tesseract's `UNICHARSET` is a variable-length id↔unichar bijection loaded
//! from a `.unicharset` text file. Per the Core-First doctrine it is NOT
//! fixed-width per-node state — it rides a **classid-keyed content-store tier**
//! shaped exactly like `deepnsm::Vocabulary`: a `reverse: Vec<String>`
//! (id → unichar) plus a `lookup: HashMap<String, u32>` (unichar → id). This
//! module is that tier plus the two adapter leaves (`id_to_unichar` /
//! `unichar_to_id`).
//!
//! # Why this is the byte-parity surface
//!
//! The unicharset path is pure text parsing — it never touches leptonica or
//! `Pix`. So the Rust side can be built and tested with **zero C dependencies**.
//! The probe compares this implementation's [`UniCharSet::dump`] of a real
//! `eng.unicharset` against the C++ `UNICHARSET::id_to_unichar` oracle (a small
//! libtesseract harness, which only *links* leptonica, never calls it). Byte-
//! identical dumps promote the doctrine CONJECTURE → FINDING.
//!
//! # Format scope
//!
//! The `.unicharset` format is: line 1 = entry count `N`; then `N` lines, each
//! beginning with the unichar as its first whitespace-delimited token (the
//! remaining columns — properties / script / bounding boxes — do not affect the
//! id↔unichar bijection and are ignored). The line position (0-based, after the
//! count line) IS the unichar id. This is the `old_style_included_ == true`
//! plain-table scope the adapter-shaper bounded; fragment/`CleanupString`
//! normalization is a separate, later leaf. Any special-token edge case a real
//! `eng.unicharset` reveals on first diff is refined then — this is built to the
//! documented format, diff-pending.

use std::collections::HashMap;
use std::path::Path;

/// A loaded `UNICHARSET`: the id↔unichar bijection, `deepnsm::Vocabulary`-shaped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UniCharSet {
    /// id → unichar (index IS the id).
    reverse: Vec<String>,
    /// unichar → id (the inverse of `reverse`).
    lookup: HashMap<String, u32>,
}

impl UniCharSet {
    /// Parse a `.unicharset` from its text contents. See the module docs for the
    /// format. Properties columns after the leading unichar token are ignored.
    ///
    /// # Errors
    ///
    /// [`UniCharSetError::Empty`] if there is no count line,
    /// [`UniCharSetError::BadCount`] if it is not a non-negative integer, and
    /// [`UniCharSetError::CountMismatch`] if fewer than `count` entry lines
    /// follow.
    pub fn load_from_str(text: &str) -> Result<Self, UniCharSetError> {
        let mut lines = text.lines();
        let count: usize = lines
            .next()
            .ok_or(UniCharSetError::Empty)?
            .trim()
            .parse()
            .map_err(|_| UniCharSetError::BadCount)?;

        let mut reverse = Vec::with_capacity(count);
        let mut lookup = HashMap::with_capacity(count);
        for line in lines.take(count) {
            // The unichar is the first whitespace-delimited token; the id is the
            // entry's position. A unichar repeated in the file keeps its FIRST
            // id in `lookup` (matches a forward-scan loader), but `reverse` keeps
            // every entry so `id_to_unichar` is exact per position.
            let unichar = line.split_whitespace().next().unwrap_or("").to_string();
            let id = u32::try_from(reverse.len()).map_err(|_| UniCharSetError::BadCount)?;
            lookup.entry(unichar.clone()).or_insert(id);
            reverse.push(unichar);
        }

        if reverse.len() != count {
            return Err(UniCharSetError::CountMismatch {
                declared: count,
                found: reverse.len(),
            });
        }
        Ok(Self { reverse, lookup })
    }

    /// Parse a `.unicharset` file from disk (a thin wrapper over
    /// [`Self::load_from_str`]).
    ///
    /// # Errors
    ///
    /// [`UniCharSetError::Io`] if the file cannot be read, else the parse errors
    /// of [`Self::load_from_str`].
    pub fn load_from_file(path: &Path) -> Result<Self, UniCharSetError> {
        let text = std::fs::read_to_string(path).map_err(|e| UniCharSetError::Io(e.to_string()))?;
        Self::load_from_str(&text)
    }

    /// Number of entries (the declared count).
    #[must_use]
    pub fn size(&self) -> usize {
        self.reverse.len()
    }

    /// The unichar string at `id`, or `None` if out of range. The C++ oracle
    /// for the byte-parity diff.
    #[must_use]
    pub fn id_to_unichar(&self, id: u32) -> Option<&str> {
        self.reverse.get(id as usize).map(String::as_str)
    }

    /// The id of `unichar`, or `None` if absent (the C++ `INVALID_UNICHAR_ID`
    /// sentinel maps to `None`; the OGAR adapter boundary re-applies the
    /// sentinel).
    #[must_use]
    pub fn unichar_to_id(&self, unichar: &str) -> Option<u32> {
        self.lookup.get(unichar).copied()
    }

    /// Render the id→unichar table as `"<id>\t<unichar>\n"` lines — the exact
    /// shape the C++ oracle harness prints, so a byte-parity diff is
    /// `diff oracle_dump.tsv rust_dump.tsv`.
    #[must_use]
    pub fn dump(&self) -> String {
        let mut out = String::new();
        for (id, unichar) in self.reverse.iter().enumerate() {
            out.push_str(&id.to_string());
            out.push('\t');
            out.push_str(unichar);
            out.push('\n');
        }
        out
    }
}

/// A failure loading a `UNICHARSET`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniCharSetError {
    /// The input had no count line.
    Empty,
    /// The count line was not a non-negative integer.
    BadCount,
    /// Fewer entry lines than the declared count.
    CountMismatch {
        /// The count declared on line 1.
        declared: usize,
        /// The number of entry lines actually found.
        found: usize,
    },
    /// The file could not be read (message from the underlying I/O error).
    Io(String),
}

impl std::fmt::Display for UniCharSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty unicharset (no count line)"),
            Self::BadCount => write!(f, "first line is not a valid entry count"),
            Self::CountMismatch { declared, found } => {
                write!(f, "declared {declared} entries but found {found}")
            }
            Self::Io(msg) => write!(f, "unicharset read failed: {msg}"),
        }
    }
}

impl std::error::Error for UniCharSetError {}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
3
a 3 0,255,0,255,0,255,0,255,0,255 0 a Left a a
b 3 0,255,0,255,0,255,0,255,0,255 0 b Left b b
cd 5 0,255,0,255,0,255,0,255,0,255 0 cd Left cd cd
";

    #[test]
    fn parses_count_and_first_token_per_line() {
        let u = UniCharSet::load_from_str(SAMPLE).expect("valid");
        assert_eq!(u.size(), 3);
        assert_eq!(u.id_to_unichar(0), Some("a"));
        assert_eq!(u.id_to_unichar(2), Some("cd")); // multi-char unichar token
        assert_eq!(u.id_to_unichar(3), None); // out of range
    }

    #[test]
    fn bijection_round_trips() {
        let u = UniCharSet::load_from_str(SAMPLE).expect("valid");
        for id in 0..u.size() as u32 {
            let s = u.id_to_unichar(id).unwrap();
            assert_eq!(u.unichar_to_id(s), Some(id), "id {id} must round-trip");
        }
        assert_eq!(u.unichar_to_id("zzz"), None, "absent unichar -> None");
    }

    #[test]
    fn dump_matches_oracle_line_shape() {
        let u = UniCharSet::load_from_str(SAMPLE).expect("valid");
        assert_eq!(u.dump(), "0\ta\n1\tb\n2\tcd\n");
    }

    #[test]
    fn errors_are_typed() {
        assert_eq!(UniCharSet::load_from_str(""), Err(UniCharSetError::Empty));
        assert_eq!(
            UniCharSet::load_from_str("notanumber\n"),
            Err(UniCharSetError::BadCount)
        );
        assert_eq!(
            UniCharSet::load_from_str("5\na\nb\n"),
            Err(UniCharSetError::CountMismatch {
                declared: 5,
                found: 2
            })
        );
    }
}
