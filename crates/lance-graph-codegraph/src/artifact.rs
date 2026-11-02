//! The reader for `*.rows.tsv` artifacts (version 2).
//!
//! Lines stream straight into integer lanes: one push per field per line, no
//! per-row struct and no per-field `String`. Every distinct text (symbol,
//! grade, file, crate name) is stored once in a shared [`Symbols`] interner
//! and a row holds only `u32` ids, so reading a 17k-row file allocates for
//! the distinct names only, not ~120k times (one `String` per field per
//! row). Millisecond reads were the symptom; the workspace rule is that
//! answers come from folds over integer lanes, never from owned per-row text.
//!
//! Columns are located by name from the `#schema` line, so a file with
//! permuted columns reads identically.

use std::collections::HashMap;
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;

use crate::graph::IntakeRefusal;

/// The only artifact version this reader accepts.
pub const ROWS_VERSION: u32 = 2;

/// Columns every artifact must carry.
const REQUIRED: [&str; 7] = [
    "predicate",
    "subject",
    "object",
    "grade",
    "file",
    "start",
    "end",
];

/// Marks an absent (`-`) span bound in [`Lanes::start`] / [`Lanes::end`].
pub const NO_SPAN: u32 = u32::MAX;

/// "No id" in an id lane (for example [`Lanes::may_target`] of a non-MAY row).
pub const NO_ID: u32 = u32::MAX;

/// Predicate code of `DEFINES`.
pub const P_DEFINES: u8 = 0;
/// Predicate code of `CALLS_RESOLVED`.
pub const P_CALLS: u8 = 1;
/// Predicate code of `MAY_DISPATCH`.
pub const P_MAY: u8 = 2;
/// Predicate code of `UNRESOLVED_CALL`.
pub const P_UNRESOLVED: u8 = 3;
/// Predicate code of `PUBLIC`.
pub const P_PUBLIC: u8 = 4;
/// Predicate code of `IMPLEMENTS`.
pub const P_IMPLEMENTS: u8 = 5;
/// Predicate code of `SYNTHETIC`.
pub const P_SYNTHETIC: u8 = 6;
/// Predicate code of `REFERENCES`.
pub const P_REFERENCES: u8 = 7;
/// Predicate code of `CRATE`.
pub const P_CRATE: u8 = 8;
/// Any other predicate; its text is kept in [`Lanes`]'s side table so nothing
/// is lost.
pub const P_OTHER: u8 = 9;

/// Canonical text of each predicate code below [`P_OTHER`].
const PRED_NAMES: [&str; 9] = [
    "DEFINES",
    "CALLS_RESOLVED",
    "MAY_DISPATCH",
    "UNRESOLVED_CALL",
    "PUBLIC",
    "IMPLEMENTS",
    "SYNTHETIC",
    "REFERENCES",
    "CRATE",
];

/// The code of a predicate text.
pub fn pred_code(s: &str) -> u8 {
    match s {
        "DEFINES" => P_DEFINES,
        "CALLS_RESOLVED" => P_CALLS,
        "MAY_DISPATCH" => P_MAY,
        "UNRESOLVED_CALL" => P_UNRESOLVED,
        "PUBLIC" => P_PUBLIC,
        "IMPLEMENTS" => P_IMPLEMENTS,
        "SYNTHETIC" => P_SYNTHETIC,
        "REFERENCES" => P_REFERENCES,
        "CRATE" => P_CRATE,
        _ => P_OTHER,
    }
}

/// The per-crate file digest. Equal digests mean the file is unchanged, which
/// is no delta, so a refresh can skip it entirely. It folds eight bytes per
/// step (the same mix as [`FastHasher`]) with the length in the last step:
/// byte-at-a-time FNV-1a over a 2.5 MB rows file was most of an unchanged
/// refresh's cost. A change detector, not a cryptographic hash.
pub fn digest(bytes: &[u8]) -> u64 {
    let mut h = FastHasher(0xcbf2_9ce4_8422_2325);
    h.write(bytes);
    h.write_u64(bytes.len() as u64);
    h.finish()
}

/// A multiply-rotate hasher folding eight bytes per step. The interner is
/// keyed by short symbol paths and is hit about once per row; SipHash made
/// those lookups the dominant cost. Not DoS-resistant, which is fine for a
/// local artifact reader.
#[derive(Default)]
pub struct FastHasher(u64);

const K: u64 = 0x517c_c1b7_2722_0a95;

impl Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for &w in chunks {
            self.0 = (self.0.rotate_left(5) ^ u64::from_le_bytes(w)).wrapping_mul(K);
        }
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            // The length is folded in so "a" and "a\0" differ.
            let tail = u64::from_le_bytes(w) ^ ((rest.len() as u64) << 56);
            self.0 = (self.0.rotate_left(5) ^ tail).wrapping_mul(K);
        }
    }

    fn write_u8(&mut self, b: u8) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(b)).wrapping_mul(K);
    }

    fn finish(&self) -> u64 {
        // hashbrown reads the low bits for the slot and the top bits for the
        // control byte; a multiply leaves the low bits weak.
        self.0.rotate_left(26)
    }
}

/// Build-hasher for the crate's interner maps.
pub type FastBuild = BuildHasherDefault<FastHasher>;

/// One string interner shared by the whole graph: ids are dense `u32`s in
/// first-seen order. A new name is the only allocation (two small boxes).
#[derive(Debug, Default)]
pub struct Symbols {
    names: Vec<Box<str>>,
    index: HashMap<Box<str>, u32, FastBuild>,
}

impl Symbols {
    /// The id of `s`, interning it on first sight.
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.index.get(s) {
            return id;
        }
        let id = self.names.len() as u32;
        self.names.push(s.into());
        self.index.insert(s.into(), id);
        id
    }

    /// The id of `s` if it was interned.
    pub fn get(&self, s: &str) -> Option<u32> {
        self.index.get(s).copied()
    }

    /// The text of `id`; empty for an id never issued.
    pub fn name(&self, id: u32) -> &str {
        self.names.get(id as usize).map_or("", |b| &**b)
    }

    /// Distinct names interned.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether nothing is interned.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Append the names a [`SymbolStage`] over this interner allocated, in
    /// order: they receive exactly the ids the stage handed out, because the
    /// interner did not change while the stage borrowed it. Cannot fail.
    pub fn absorb(&mut self, staged: StagedNames) {
        debug_assert_eq!(
            staged.base_len,
            self.names.len(),
            "interner changed under the stage"
        );
        for name in staged.names {
            let id = self.names.len() as u32;
            self.index.insert(name.clone(), id);
            self.names.push(name);
        }
    }
}

/// What reading rows needs of an interner. [`Symbols`] writes into itself;
/// a [`SymbolStage`] leaves its base untouched and collects new names.
pub trait Interner {
    /// The id of `s`, interning it on first sight.
    fn intern(&mut self, s: &str) -> u32;
    /// The id of `s` if it was interned.
    fn get(&self, s: &str) -> Option<u32>;
    /// The text of `id`; empty for an id never issued.
    fn name(&self, id: u32) -> &str;
    /// Distinct names interned.
    fn len(&self) -> usize;
}

impl Interner for Symbols {
    fn intern(&mut self, s: &str) -> u32 {
        Symbols::intern(self, s)
    }
    fn get(&self, s: &str) -> Option<u32> {
        Symbols::get(self, s)
    }
    fn name(&self, id: u32) -> &str {
        Symbols::name(self, id)
    }
    fn len(&self) -> usize {
        Symbols::len(self)
    }
}

/// An interner over a borrowed [`Symbols`] that never writes to it: a name
/// the base knows keeps its id, a new name gets the next id past the base
/// and is held here until [`Symbols::absorb`] (a refresh that commits) or
/// dropped (a refresh that refuses, leaving the base exactly as it was).
pub struct SymbolStage<'a> {
    base: &'a Symbols,
    names: Vec<Box<str>>,
    index: HashMap<Box<str>, u32, FastBuild>,
}

/// The names a dropped-borrow [`SymbolStage`] allocated, for [`Symbols::absorb`].
pub struct StagedNames {
    base_len: usize,
    names: Vec<Box<str>>,
}

impl<'a> SymbolStage<'a> {
    /// A stage over `base`.
    pub fn new(base: &'a Symbols) -> Self {
        Self {
            base,
            names: Vec::new(),
            index: HashMap::default(),
        }
    }

    /// Names allocated so far.
    pub fn added(&self) -> usize {
        self.names.len()
    }

    /// The allocated names, releasing the borrow of the base.
    pub fn into_names(self) -> StagedNames {
        StagedNames {
            base_len: self.base.len(),
            names: self.names,
        }
    }
}

impl Interner for SymbolStage<'_> {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(id) = self.get(s) {
            return id;
        }
        let id = (self.base.len() + self.names.len()) as u32;
        self.names.push(s.into());
        self.index.insert(s.into(), id);
        id
    }
    fn get(&self, s: &str) -> Option<u32> {
        self.base.get(s).or_else(|| self.index.get(s).copied())
    }
    fn name(&self, id: u32) -> &str {
        let id = id as usize;
        match id.checked_sub(self.base.len()) {
            None => self.base.name(id as u32),
            Some(k) => self.names.get(k).map_or("", |b| &**b),
        }
    }
    fn len(&self) -> usize {
        self.base.len() + self.names.len()
    }
}

/// All rows of a directory as parallel integer lanes; index `i` of every lane
/// is row `i`. Invariant: all per-row lanes have equal length. Texts are ids
/// into the [`Symbols`] the rows were read with.
#[derive(Debug, Default)]
pub struct Lanes {
    /// Predicate code of each row (`P_*`).
    pub pred: Vec<u8>,
    /// Subject text id of each row.
    pub subject: Vec<u32>,
    /// Object text id of each row, verbatim from the artifact.
    pub object: Vec<u32>,
    /// Grade (provenance) text id of each row.
    pub grade: Vec<u32>,
    /// Source file text id of each row, the id of `-` when none.
    pub file: Vec<u32>,
    /// Span start, [`NO_SPAN`] when `-`.
    pub start: Vec<u32>,
    /// Span end, [`NO_SPAN`] when `-`.
    pub end: Vec<u32>,
    /// The crate (`<krate>.rows.tsv` stem) text id each row was read from.
    pub krate: Vec<u32>,
    /// For `MAY_DISPATCH` rows the id of the `Trait::method` part of the
    /// object (before " via "); [`NO_ID`] for every other row. Precomputed so
    /// no query splits a string.
    pub may_target: Vec<u32>,
    /// `(row, predicate text id)` of each [`P_OTHER`] row, ascending by row.
    other_pred: Vec<(u32, u32)>,
    /// `(crate text id, FNV-1a 64 of the file it was read from)`.
    digests: Vec<(u32, u64)>,
}

impl Lanes {
    /// Rows held.
    pub fn len(&self) -> usize {
        self.pred.len()
    }

    /// Whether no rows are held.
    pub fn is_empty(&self) -> bool {
        self.pred.is_empty()
    }

    /// The predicate text of `row`; the original text for [`P_OTHER`] rows.
    pub fn predicate_name<'a>(&self, row: usize, symbols: &'a Symbols) -> &'a str {
        let code = self.pred[row];
        if let Some(n) = PRED_NAMES.get(code as usize) {
            return n;
        }
        u32::try_from(row)
            .ok()
            .and_then(|r| self.other_pred.binary_search_by_key(&r, |&(r, _)| r).ok())
            .map_or("", |k| symbols.name(self.other_pred[k].1))
    }

    /// The recorded file digest of `krate`, if its file was read.
    pub fn digest_of(&self, krate: u32) -> Option<u64> {
        self.digests.iter().find(|d| d.0 == krate).map(|d| d.1)
    }

    fn set_digest(&mut self, krate: u32, d: u64) {
        match self.digests.iter_mut().find(|e| e.0 == krate) {
            Some(e) => e.1 = d,
            None => self.digests.push((krate, d)),
        }
    }

    /// Drop every row read from `krate`, in place and keeping the order of
    /// the rest. The crate's digest is kept until [`Lanes::append`] replaces it.
    pub fn drop_crate(&mut self, krate: u32) {
        let n = self.pred.len();
        let mut w = 0usize;
        let mut op = 0usize;
        let mut other = Vec::with_capacity(self.other_pred.len());
        for i in 0..n {
            let is_other = self.pred[i] == P_OTHER;
            let entry = if is_other {
                let e = self.other_pred.get(op).copied();
                op += 1;
                e
            } else {
                None
            };
            if self.krate[i] == krate {
                continue;
            }
            if let Some((_, id)) = entry {
                other.push((w as u32, id));
            }
            self.pred[w] = self.pred[i];
            self.subject[w] = self.subject[i];
            self.object[w] = self.object[i];
            self.grade[w] = self.grade[i];
            self.file[w] = self.file[i];
            self.start[w] = self.start[i];
            self.end[w] = self.end[i];
            self.krate[w] = self.krate[i];
            self.may_target[w] = self.may_target[i];
            w += 1;
        }
        self.pred.truncate(w);
        self.subject.truncate(w);
        self.object.truncate(w);
        self.grade.truncate(w);
        self.file.truncate(w);
        self.start.truncate(w);
        self.end.truncate(w);
        self.krate.truncate(w);
        self.may_target.truncate(w);
        self.other_pred = other;
    }

    /// Append every row of `other` after this lane set's rows, and adopt its
    /// file digests.
    pub fn append(&mut self, mut other: Lanes) {
        let base = self.pred.len() as u32;
        self.pred.append(&mut other.pred);
        self.subject.append(&mut other.subject);
        self.object.append(&mut other.object);
        self.grade.append(&mut other.grade);
        self.file.append(&mut other.file);
        self.start.append(&mut other.start);
        self.end.append(&mut other.end);
        self.krate.append(&mut other.krate);
        self.may_target.append(&mut other.may_target);
        self.other_pred
            .extend(other.other_pred.iter().map(|&(r, id)| (r + base, id)));
        for (k, d) in other.digests {
            self.set_digest(k, d);
        }
    }
}

const SUFFIX: &str = ".rows.tsv";

/// The text of the single file `<krate>.rows.tsv` of `dir`.
pub fn read_crate_text(dir: &Path, krate: &str) -> Result<String, IntakeRefusal> {
    fs::read_to_string(dir.join(format!("{krate}{SUFFIX}"))).map_err(IntakeRefusal::Io)
}

/// Read the single file `<krate>.rows.tsv` of `dir`; its rows carry `krate`.
pub fn read_crate(dir: &Path, krate: &str, symbols: &mut Symbols) -> Result<Lanes, IntakeRefusal> {
    let text = read_crate_text(dir, krate)?;
    let mut lanes = Lanes::default();
    read_text_as(&text, krate, symbols, &mut lanes)?;
    Ok(lanes)
}

/// Read every `*.rows.tsv` in `dir`, sorted by file name so the row order is
/// deterministic.
pub fn read_dir(dir: &Path, symbols: &mut Symbols) -> Result<Lanes, IntakeRefusal> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir).map_err(IntakeRefusal::Io)? {
        let path = entry.map_err(IntakeRefusal::Io)?.path();
        let is_rows = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(SUFFIX));
        if is_rows {
            paths.push(path);
        }
    }
    paths.sort();
    let mut lanes = Lanes::default();
    for p in &paths {
        let text = fs::read_to_string(p).map_err(IntakeRefusal::Io)?;
        let stem = p
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(SUFFIX))
            .unwrap_or("");
        read_text_as(&text, stem, symbols, &mut lanes)?;
    }
    Ok(lanes)
}

fn number(field: &str) -> Option<u32> {
    if field == "-" {
        Some(NO_SPAN)
    } else {
        field.parse().ok()
    }
}

/// The `Trait::method` part of a MAY_DISPATCH object.
pub fn dispatch_target(object: &str) -> &str {
    object.find(" via ").map_or(object, |i| &object[..i])
}

/// A tiny recency cache in front of the interner for fields that repeat from
/// row to row (a function's rows are adjacent, a file's rows are adjacent,
/// grades take a handful of values): a hit costs a short compare instead of a
/// hash. A miss interns and stores round-robin.
struct Memo<'a, const N: usize> {
    slots: [(&'a str, u32); N],
    filled: usize,
    next: usize,
}

impl<'a, const N: usize> Memo<'a, N> {
    fn new() -> Self {
        Self {
            slots: [("", 0); N],
            filled: 0,
            next: 0,
        }
    }

    fn id(&mut self, s: &'a str, sym: &mut impl Interner) -> u32 {
        for &(t, id) in &self.slots[..self.filled] {
            if t == s {
                return id;
            }
        }
        let id = sym.intern(s);
        self.slots[self.next] = (s, id);
        self.next = (self.next + 1) % N;
        self.filled = (self.filled + 1).min(N);
        id
    }
}

/// Append one file's rows to `lanes`, with an empty crate name.
pub fn read_text(
    text: &str,
    symbols: &mut Symbols,
    lanes: &mut Lanes,
) -> Result<(), IntakeRefusal> {
    read_text_as(text, "", symbols, lanes)
}

/// Append one file's rows to `lanes`, attributing each to `krate`, and record
/// the file's digest under `krate`.
pub fn read_text_as(
    text: &str,
    krate: &str,
    symbols: &mut impl Interner,
    lanes: &mut Lanes,
) -> Result<(), IntakeRefusal> {
    let mut lines = text.lines();
    let found = lines
        .next()
        .and_then(|l| l.strip_prefix("#version "))
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0);
    if found != ROWS_VERSION {
        return Err(IntakeRefusal::Version { found });
    }
    let schema = lines
        .next()
        .and_then(|l| l.strip_prefix("#schema "))
        .ok_or_else(|| IntakeRefusal::Schema {
            missing: "#schema".to_string(),
        })?;
    let names: Vec<&str> = schema.split('\t').collect();
    let mut at = [0usize; 7];
    for (slot, want) in at.iter_mut().zip(REQUIRED) {
        *slot = names
            .iter()
            .position(|n| *n == want)
            .ok_or_else(|| IntakeRefusal::Schema {
                missing: want.to_string(),
            })?;
    }
    let kid = symbols.intern(krate);
    let mut fields: Vec<&str> = Vec::with_capacity(names.len());
    let mut subjects: Memo<4> = Memo::new();
    let mut files: Memo<2> = Memo::new();
    let mut grades: Memo<8> = Memo::new();
    for (i, line) in lines.enumerate() {
        // Lines 1 and 2 were the header; data starts at line 3.
        let lineno = i + 3;
        if line.is_empty() {
            continue;
        }
        fields.clear();
        fields.extend(line.split('\t'));
        if fields.len() != names.len() {
            return Err(IntakeRefusal::Arity { line: lineno });
        }
        let start = number(fields[at[5]]).ok_or(IntakeRefusal::Arity { line: lineno })?;
        let end = number(fields[at[6]]).ok_or(IntakeRefusal::Arity { line: lineno })?;
        let pred_text = fields[at[0]];
        let code = pred_code(pred_text);
        let object_text = fields[at[2]];
        let subject = subjects.id(fields[at[1]], symbols);
        let object = symbols.intern(object_text);
        let may = if code == P_MAY {
            let t = dispatch_target(object_text);
            if t.len() == object_text.len() {
                object
            } else {
                symbols.intern(t)
            }
        } else {
            NO_ID
        };
        if code == P_OTHER {
            let id = symbols.intern(pred_text);
            lanes.other_pred.push((lanes.pred.len() as u32, id));
        }
        lanes.pred.push(code);
        lanes.subject.push(subject);
        lanes.object.push(object);
        lanes.grade.push(grades.id(fields[at[3]], symbols));
        lanes.file.push(files.id(fields[at[4]], symbols));
        lanes.start.push(start);
        lanes.end.push(end);
        lanes.krate.push(kid);
        lanes.may_target.push(may);
    }
    lanes.set_digest(kid, digest(text.as_bytes()));
    Ok(())
}
