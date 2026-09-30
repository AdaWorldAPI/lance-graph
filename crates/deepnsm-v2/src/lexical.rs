//! `lexical` — the integer lexical evidence that survives vocabulary ingestion.
//!
//! ## Why this exists
//!
//! [`PaletteVocab::from_frequency_ranked`] consumes a frequency-ranked list of
//! surface strings and keeps the FIRST occurrence of each. That is correct for
//! ROUTING — a surface form gets one [`WordId`] — but it is the point where the
//! counted evidence behind the ranking is thrown away. A source row such as
//!
//! ```text
//! lemRank,lemma,PoS,lemFreq,wordFreq,word
//! 518,record,n,187057,120048,record
//! 1778,record,v,51375,13014,record
//! ```
//!
//! says the surface `record` was observed 120,048 times as a noun and 13,014
//! times as a verb, under two lemma entries whose totals (187,057 / 51,375)
//! differ from the surface counts. Ranking reduces that to "`record` is word
//! N". [`LexicalEvidence`] keeps it, BESIDE the routing, so a later consumer can
//! condition an interpretation on it.
//!
//! ## What this is not
//!
//! - **Not routing.** It never assigns, reorders or changes a [`WordId`]; it is
//!   built against an existing [`PaletteVocab`] and only reads it. One surface
//!   `WordId` may own several readings — a homograph does not get two ids.
//! - **Not meaning.** Nothing here touches [`crate::space::Cam96`] or the
//!   `codes[word_id]` table. Semantic distance comes from Cam96; counts come
//!   from here; the two are never mixed.
//! - **Not truth.** Counts are observed population evidence, stored as exact
//!   integers. No normalisation to `f32`, no probability, no NARS truth.
//!
//! ## Frequency is evidence, not a lexical decision
//!
//! A word's readings are stored most frequent first, and each carries an
//! integer cumulative percentile coverage ([`LexicalEvidence::coverage`]).
//! Position 0 is the most frequent observed reading and its share is
//! `coverage[0]` — a reader never re-sums counts to measure it. The order is a
//! presentation of the evidence, never a preference: every reading stays, and
//! nothing here chooses, ranks or eliminates one. Count changes may change the
//! order and the coverage; they never change which readings exist. Source file
//! order is not kept: COCA orders `word_forms.csv` by lemma rank, so its first
//! row is not the most frequent reading for 259 surfaces (`changes`: verb
//! 13,624 first, noun 113,085 second).
//!
//! ## Unknown is not zero
//!
//! Every count is `Option<u64>`. An empty source field is `None` (not observed
//! by the source); a literal `0` is `Some(0)`. Aggregates return `None` when
//! any contributing count is unknown, and a (surface, PoS) pair with no reading
//! is `None` — the sources are truncated frequency lists, so absence of a row
//! is not evidence of a zero count.
//!
//! [`PaletteVocab::from_frequency_ranked`]: crate::vocab::PaletteVocab::from_frequency_ranked

use std::collections::{HashMap, HashSet};

use crate::vocab::{PaletteVocab, WordId};

/// A source part-of-speech code, kept exactly as the source wrote it.
///
/// COCA uses one ASCII letter per tag (`n`, `v`, `j`, `r`, `i`, …). The code is
/// deliberately NOT mapped onto [`crate::fsm::Pos`]: that enum is a six-state
/// parser alphabet and folds several source tags into one (`n` and `p` both
/// become `Noun`), which is exactly the kind of irreversible reduction this
/// module exists to avoid. A consumer maps at its own boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PosCode(pub u8);

impl PosCode {
    /// The code for a one-letter ASCII source tag, or `None` if `tag` is not
    /// exactly one ASCII byte.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag.as_bytes() {
            [b] if b.is_ascii_graphic() => Some(Self(*b)),
            _ => None,
        }
    }

    /// The source tag as a character.
    #[must_use]
    pub const fn as_char(self) -> char {
        self.0 as char
    }
}

/// Index of a [`LemmaEntry`] inside one [`LexicalEvidence`].
pub type LemmaRef = u32;

/// One lemma entry of the source: a (lemma, PoS) pair and its total count.
///
/// A lemma is NOT a [`WordId`]: the lemma string need not be in the routing
/// vocabulary at all, so it is identified by the source's own key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LemmaEntry {
    /// The source's identity for this entry (`lemRank` in COCA `word_forms.csv`).
    pub source_key: u32,
    /// The lemma string; `None` where the source leaves it empty (COCA does so
    /// for some proper nouns — the entry and its count still exist).
    pub lemma: Option<String>,
    /// The entry's part of speech.
    pub pos: PosCode,
    /// Total occurrences of the lemma under this PoS (`lemFreq`), if known.
    pub count: Option<u64>,
}

/// One counted reading of a surface form: the form observed under one PoS,
/// optionally as an inflection of one lemma entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexicalReading {
    /// The reading's part of speech.
    pub pos: PosCode,
    /// The lemma entry this form belongs to, if the source names one.
    pub lemma: Option<LemmaRef>,
    /// Occurrences of THIS surface form under this reading (`wordFreq`), if known.
    pub form_count: Option<u64>,
}

/// Why evidence could not be stored. Every variant refuses rather than
/// silently picking a winner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceError {
    /// A [`WordId`] outside the vocabulary the evidence was built against.
    WordIdOutOfRange { id: WordId, vocab_len: usize },
    /// The same lemma `source_key` was given two different (lemma, PoS, count).
    ConflictingLemmaEntry { source_key: u32 },
    /// The same (word id, PoS, lemma) reading was given twice — storing both
    /// would double-count, keeping one would be first-wins.
    DuplicateReading {
        id: WordId,
        pos: PosCode,
        lemma: Option<LemmaRef>,
    },
    /// A reading's PoS disagrees with the PoS of the lemma entry it names.
    PosMismatch {
        id: WordId,
        reading: PosCode,
        lemma: PosCode,
    },
    /// The CSV header is not the format the loader reads.
    UnexpectedHeader { found: String },
    /// A row does not have exactly the expected number of fields.
    FieldCount {
        line: usize,
        expected: usize,
        found: usize,
    },
    /// A field that must be an unsigned integer is not.
    NotAnInteger {
        line: usize,
        field: &'static str,
        value: String,
    },
    /// A PoS field is not a one-letter tag.
    BadPos { line: usize, value: String },
    /// A count aggregate exceeded `u64`.
    CountOverflow,
    /// A [`LemmaRef`] this builder never issued.
    UnknownLemma { lemma: LemmaRef },
    /// A row contains a `"` — the loader reads unquoted CSV only, so a quoted
    /// field is refused rather than routed with its quotes or split at an
    /// embedded comma.
    QuotedField { line: usize },
}

impl std::fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for EvidenceError {}

/// Accumulates readings against a fixed vocabulary size; [`finish`](Self::finish)
/// freezes them into a [`LexicalEvidence`].
#[derive(Debug, Clone)]
pub struct LexicalEvidenceBuilder {
    vocab_len: usize,
    lemmas: Vec<LemmaEntry>,
    lemma_by_key: HashMap<u32, LemmaRef>,
    readings: Vec<(WordId, LexicalReading)>,
    seen: HashSet<(WordId, PosCode, Option<LemmaRef>)>,
}

impl LexicalEvidenceBuilder {
    /// A builder for ids `0..vocab.len()`. The vocabulary is only read.
    #[must_use]
    pub fn new(vocab: &PaletteVocab) -> Self {
        Self {
            vocab_len: vocab.len(),
            lemmas: Vec::new(),
            lemma_by_key: HashMap::new(),
            readings: Vec::new(),
            seen: HashSet::new(),
        }
    }

    /// Register a lemma entry, or return the existing one for `source_key`.
    ///
    /// Re-registering the same key with identical fields is a no-op (a lemma
    /// entry spans several form rows); different fields are an error.
    pub fn lemma_entry(
        &mut self,
        source_key: u32,
        lemma: Option<&str>,
        pos: PosCode,
        count: Option<u64>,
    ) -> Result<LemmaRef, EvidenceError> {
        let entry = LemmaEntry {
            source_key,
            lemma: lemma.map(str::to_owned),
            pos,
            count,
        };
        if let Some(&r) = self.lemma_by_key.get(&source_key) {
            return if self.lemmas[r as usize] == entry {
                Ok(r)
            } else {
                Err(EvidenceError::ConflictingLemmaEntry { source_key })
            };
        }
        let r = self.lemmas.len() as LemmaRef;
        self.lemmas.push(entry);
        self.lemma_by_key.insert(source_key, r);
        Ok(r)
    }

    /// Add one counted reading of word `id`.
    pub fn add_reading(
        &mut self,
        id: WordId,
        reading: LexicalReading,
    ) -> Result<(), EvidenceError> {
        if id as usize >= self.vocab_len {
            return Err(EvidenceError::WordIdOutOfRange {
                id,
                vocab_len: self.vocab_len,
            });
        }
        if let Some(l) = reading.lemma {
            let lemma_pos = self
                .lemmas
                .get(l as usize)
                .ok_or(EvidenceError::UnknownLemma { lemma: l })?
                .pos;
            if lemma_pos != reading.pos {
                return Err(EvidenceError::PosMismatch {
                    id,
                    reading: reading.pos,
                    lemma: lemma_pos,
                });
            }
        }
        if !self.seen.insert((id, reading.pos, reading.lemma)) {
            return Err(EvidenceError::DuplicateReading {
                id,
                pos: reading.pos,
                lemma: reading.lemma,
            });
        }
        self.readings.push((id, reading));
        Ok(())
    }

    /// Freeze into id-indexed storage.
    ///
    /// Each word's readings are stored in FREQUENCY ORDER — highest known
    /// `form_count` first, unknown counts last, equal counts in insertion
    /// order — and each carries its cumulative percentile coverage (see
    /// [`LexicalEvidence::coverage`]). The order presents evidence; it selects
    /// nothing, and every reading is kept.
    #[must_use]
    pub fn finish(mut self) -> LexicalEvidence {
        // Stable: equal keys keep insertion order.
        self.readings.sort_by_key(|(id, r)| {
            (
                *id,
                match r.form_count {
                    Some(c) => (0u8, std::cmp::Reverse(c)),
                    None => (1u8, std::cmp::Reverse(0)),
                },
            )
        });
        let mut offsets = vec![0u32; self.vocab_len + 1];
        for (id, _) in &self.readings {
            offsets[*id as usize + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut coverage = vec![None; self.readings.len()];
        for w in offsets.windows(2) {
            let (a, b) = (w[0] as usize, w[1] as usize);
            let counts: Option<Vec<u64>> = self.readings[a..b]
                .iter()
                .map(|(_, r)| r.form_count)
                .collect();
            let Some(counts) = counts else { continue };
            let total: u128 = counts.iter().map(|&c| u128::from(c)).sum();
            if total == 0 {
                continue;
            }
            let mut cum = 0u128;
            for (slot, c) in coverage[a..b].iter_mut().zip(counts) {
                cum += u128::from(c);
                // cum <= total, so the quotient is 0..=100.
                *slot = Some((cum * 100 / total) as u8);
            }
        }
        let mut lemmas_by_name: HashMap<String, Vec<LemmaRef>> = HashMap::new();
        for (i, e) in self.lemmas.iter().enumerate() {
            if let Some(name) = &e.lemma {
                lemmas_by_name
                    .entry(name.clone())
                    .or_default()
                    .push(i as LemmaRef);
            }
        }
        LexicalEvidence {
            offsets,
            readings: self.readings.into_iter().map(|(_, r)| r).collect(),
            coverage,
            lemmas: self.lemmas,
            lemmas_by_name,
        }
    }
}

/// Integer lexical evidence, indexed by the routing [`WordId`] it sits beside.
#[derive(Debug, Clone, Default)]
pub struct LexicalEvidence {
    /// `offsets[id]..offsets[id + 1]` are word `id`'s readings.
    offsets: Vec<u32>,
    /// Frequency-ordered within each word (see [`LexicalEvidenceBuilder::finish`]).
    readings: Vec<LexicalReading>,
    /// Parallel to `readings`: cumulative percentile coverage.
    coverage: Vec<Option<u8>>,
    lemmas: Vec<LemmaEntry>,
    lemmas_by_name: HashMap<String, Vec<LemmaRef>>,
}

/// Sum a set of optional counts: `None` if any is unknown or the set is empty.
/// Unknown takes precedence over overflow, so the answer never depends on
/// where in the set an unknown count sits.
fn sum_known(counts: impl Iterator<Item = Option<u64>>) -> Result<Option<u64>, EvidenceError> {
    let counts: Option<Vec<u64>> = counts.collect();
    let Some(counts) = counts.filter(|c| !c.is_empty()) else {
        return Ok(None);
    };
    counts
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .map(Some)
        .ok_or(EvidenceError::CountOverflow)
}

impl LexicalEvidence {
    /// The shared index range for word `id`'s readings and coverage entries.
    /// Returns `0..0` when `id` is outside the stored offsets.
    fn span(&self, id: WordId) -> std::ops::Range<usize> {
        let i = id as usize;
        match (self.offsets.get(i), self.offsets.get(i + 1)) {
            (Some(&a), Some(&b)) => a as usize..b as usize,
            _ => 0..0,
        }
    }

    /// Every counted reading of word `id`, most frequent first (empty if none
    /// survived, or `id` is out of range). Position 0 is the most frequent
    /// observed reading; the slice is the reading set, not a ranked choice.
    #[must_use]
    pub fn readings(&self, id: WordId) -> &[LexicalReading] {
        &self.readings[self.span(id)]
    }

    /// Cumulative percentile coverage, aligned with [`readings`](Self::readings):
    /// entry `k` is the percent (`0..=100`, floored) of word `id`'s known
    /// occurrences covered by readings `0..=k`. The last entry is `100`, and
    /// entry 0 is the most frequent observed reading's own share.
    ///
    /// All `None` when any reading's count is unknown or every count is zero —
    /// a share of an unknown total is itself unknown.
    #[must_use]
    pub fn coverage(&self, id: WordId) -> &[Option<u8>] {
        &self.coverage[self.span(id)]
    }

    /// The lemma entry behind a reading.
    #[must_use]
    pub fn lemma(&self, r: LemmaRef) -> Option<&LemmaEntry> {
        self.lemmas.get(r as usize)
    }

    /// Every lemma entry spelled `lemma`, across parts of speech.
    pub fn lemma_entries(&self, lemma: &str) -> impl Iterator<Item = &LemmaEntry> {
        self.lemmas_by_name
            .get(lemma)
            .into_iter()
            .flatten()
            .map(|&r| &self.lemmas[r as usize])
    }

    /// `count(surface)`: occurrences of word `id` over all its readings.
    /// `None` if it has no reading or any reading's count is unknown.
    pub fn surface_count(&self, id: WordId) -> Result<Option<u64>, EvidenceError> {
        sum_known(self.readings(id).iter().map(|r| r.form_count))
    }

    /// `count(surface, PoS)`. `None` if no reading carries that PoS or any
    /// matching reading's count is unknown.
    pub fn surface_pos_count(
        &self,
        id: WordId,
        pos: PosCode,
    ) -> Result<Option<u64>, EvidenceError> {
        sum_known(
            self.readings(id)
                .iter()
                .filter(|r| r.pos == pos)
                .map(|r| r.form_count),
        )
    }

    /// `count(lemma, PoS)`: the lemma entry's own total — never re-summed from
    /// forms, so it stays whatever the source counted.
    pub fn lemma_pos_count(&self, lemma: &str, pos: PosCode) -> Result<Option<u64>, EvidenceError> {
        sum_known(
            self.lemma_entries(lemma)
                .filter(|e| e.pos == pos)
                .map(|e| e.count),
        )
    }

    /// `count(lemma)`: the sum of the lemma's per-PoS entry totals.
    pub fn lemma_count(&self, lemma: &str) -> Result<Option<u64>, EvidenceError> {
        sum_known(self.lemma_entries(lemma).map(|e| e.count))
    }

    /// Number of stored readings.
    #[must_use]
    pub fn reading_count(&self) -> usize {
        self.readings.len()
    }

    /// Number of lemma entries.
    #[must_use]
    pub fn lemma_entry_count(&self) -> usize {
        self.lemmas.len()
    }
}

/// What [`load_word_forms_csv`] could not route — reported, never dropped
/// silently. Rows counted here still register their lemma entry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WordFormsReport {
    /// Data rows read.
    pub rows: usize,
    /// Readings stored.
    pub stored: usize,
    /// Rows whose `word` field is empty (no surface form to route).
    pub empty_surface: usize,
    /// Rows whose surface form is not in the vocabulary.
    pub unrouted: usize,
}

/// The exact header [`load_word_forms_csv`] reads (COCA `word_forms.csv`).
pub const WORD_FORMS_HEADER: &str = "lemRank,lemma,PoS,lemFreq,wordFreq,word";

fn parse_count(line: usize, field: &'static str, v: &str) -> Result<Option<u64>, EvidenceError> {
    let v = v.trim();
    if v.is_empty() {
        return Ok(None);
    }
    v.parse::<u64>()
        .map(Some)
        .map_err(|_| EvidenceError::NotAnInteger {
            line,
            field,
            value: v.to_owned(),
        })
}

/// Load unquoted `lemRank,lemma,PoS,lemFreq,wordFreq,word` rows against `vocab`
/// (the committed COCA file has no quoting; a `"` is refused, not guessed at).
///
/// Each row becomes one reading of the surface `word` (looked up exactly — the
/// caller normalises the vocabulary), attached to the lemma entry `lemRank`.
/// Homographs keep every reading. Empty count fields stay `None`.
pub fn load_word_forms_csv(
    text: &str,
    vocab: &PaletteVocab,
) -> Result<(LexicalEvidence, WordFormsReport), EvidenceError> {
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("").trim_end_matches('\r');
    if header != WORD_FORMS_HEADER {
        return Err(EvidenceError::UnexpectedHeader {
            found: header.to_owned(),
        });
    }
    let mut b = LexicalEvidenceBuilder::new(vocab);
    let mut report = WordFormsReport::default();
    for (i, raw) in lines.enumerate() {
        let line = i + 2;
        let raw = raw.trim_end_matches('\r');
        if raw.is_empty() {
            continue;
        }
        if raw.contains('"') {
            return Err(EvidenceError::QuotedField { line });
        }
        let f: Vec<&str> = raw.split(',').collect();
        if f.len() != 6 {
            return Err(EvidenceError::FieldCount {
                line,
                expected: 6,
                found: f.len(),
            });
        }
        report.rows += 1;
        let key = parse_count(line, "lemRank", f[0])?
            .and_then(|k| u32::try_from(k).ok())
            .ok_or_else(|| EvidenceError::NotAnInteger {
                line,
                field: "lemRank",
                value: f[0].to_owned(),
            })?;
        let pos = PosCode::from_tag(f[2].trim()).ok_or_else(|| EvidenceError::BadPos {
            line,
            value: f[2].to_owned(),
        })?;
        let lemma_name = Some(f[1].trim()).filter(|s| !s.is_empty());
        let lemma_count = parse_count(line, "lemFreq", f[3])?;
        let form_count = parse_count(line, "wordFreq", f[4])?;
        let lemma = b.lemma_entry(key, lemma_name, pos, lemma_count)?;
        let word = f[5].trim();
        if word.is_empty() {
            report.empty_surface += 1;
            continue;
        }
        let Some(id) = vocab.id(word) else {
            report.unrouted += 1;
            continue;
        };
        b.add_reading(
            id,
            LexicalReading {
                pos,
                lemma: Some(lemma),
                form_count,
            },
        )?;
        report.stored += 1;
    }
    Ok((b.finish(), report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::{Cam96, Cam96Space};
    use crate::Nsm;

    const N: PosCode = PosCode(b'n');
    const V: PosCode = PosCode(b'v');

    /// Real COCA values for `record` (word_forms.csv), plus a second noun form.
    const FIXTURE: &str = "lemRank,lemma,PoS,lemFreq,wordFreq,word
518,record,n,187057,120048,record
518,record,n,187057,67009,records
1778,record,v,51375,13014,record
1778,record,v,51375,38361,recorded
9,you,p,12079413,440,
1096,,n,88994,88994,may
42,xyzzy,n,77,,xyzzy
";

    fn vocab() -> PaletteVocab {
        let mut v = PaletteVocab::new();
        // Deliberately NOT containing `recorded`: unrouted rows are reported.
        v.from_frequency_ranked(["the", "record", "records", "may", "xyzzy"]);
        v
    }

    fn load() -> (PaletteVocab, LexicalEvidence, WordFormsReport) {
        let v = vocab();
        let (e, r) = load_word_forms_csv(FIXTURE, &v).expect("fixture loads");
        (v, e, r)
    }

    /// Test 1: a homograph keeps both readings (not first-row-wins).
    /// Falsified by making `add_reading` skip an id it has already seen.
    #[test]
    fn homograph_keeps_every_pos_reading() {
        let (v, e, _) = load();
        let id = v.id("record").unwrap();
        let rs = e.readings(id);
        assert_eq!(rs.len(), 2, "record must keep its noun AND verb reading");
        assert_eq!(rs[0].pos, N);
        assert_eq!(rs[0].form_count, Some(120_048));
        assert_eq!(rs[1].pos, V);
        assert_eq!(rs[1].form_count, Some(13_014));
        assert_eq!(e.surface_pos_count(id, N), Ok(Some(120_048)));
        assert_eq!(e.surface_pos_count(id, V), Ok(Some(13_014)));
        assert_eq!(e.surface_count(id), Ok(Some(133_062)));
    }

    /// The register is frequency-ordered, not file-ordered, and carries its
    /// cumulative percentile coverage. The counts are COCA's for `changes`
    /// (verb row first in the file), placed on the fixture surface `record`.
    /// Falsified by dropping the count sort.
    #[test]
    fn readings_are_frequency_ordered_with_percentile_coverage() {
        let v = vocab();
        let text = format!(
            "{WORD_FORMS_HEADER}\n\
             700,change,v,200000,13624,record\n\
             850,change,n,300000,113085,record\n"
        );
        let (e, _) = load_word_forms_csv(&text, &v).unwrap();
        let id = v.id("record").unwrap();
        let rs = e.readings(id);
        assert_eq!(rs[0].pos, N, "the noun row counts more and must come first");
        assert_eq!(rs[0].form_count, Some(113_085));
        assert_eq!(rs[1].pos, V);
        // 113,085 / 126,709 = 89.2% -> 89; the last entry covers everything.
        assert_eq!(e.coverage(id), &[Some(89), Some(100)]);
    }

    /// Equal counts keep file order; unknown counts sort last and make the
    /// whole word's coverage unknown; an all-zero word has no coverage.
    #[test]
    fn ties_unknowns_and_zeros_in_the_register() {
        let v = vocab();
        let text = format!(
            "{WORD_FORMS_HEADER}\n\
             1,a,v,9,10,record\n\
             2,a,n,9,10,record\n\
             3,b,v,9,,records\n\
             4,b,n,9,4,records\n\
             5,c,n,9,0,may\n"
        );
        let (e, _) = load_word_forms_csv(&text, &v).unwrap();
        let rec = v.id("record").unwrap();
        assert_eq!(e.readings(rec)[0].pos, V, "tie keeps file order");
        assert_eq!(e.coverage(rec), &[Some(50), Some(100)]);
        let recs = v.id("records").unwrap();
        assert_eq!(
            e.readings(recs)[0].form_count,
            Some(4),
            "unknown sorts last"
        );
        assert_eq!(e.coverage(recs), &[None, None]);
        assert_eq!(e.coverage(v.id("may").unwrap()), &[None]);
        assert!(e.coverage(v.id("the").unwrap()).is_empty());
    }

    /// Test 2: counts are exact integers, including one no `f32` can hold.
    #[test]
    fn integer_counts_round_trip_exactly() {
        const BIG: u64 = 16_777_217; // 2^24 + 1: rounds to 2^24 as f32
        assert_ne!(BIG as f32 as u64, BIG, "anti-vacuity: f32 must lose it");
        let text = format!("{WORD_FORMS_HEADER}\n1,the,a,{},{BIG},the\n", u64::MAX);
        let v = vocab();
        let (e, _) = load_word_forms_csv(&text, &v).unwrap();
        let id = v.id("the").unwrap();
        assert_eq!(e.readings(id)[0].form_count, Some(BIG));
        assert_eq!(e.lemma_count("the"), Ok(Some(u64::MAX)));
    }

    /// Test 3: routing is untouched — the same list gives the same ids with or
    /// without evidence, and the evidence never adds a WordId.
    #[test]
    fn routing_ids_are_unchanged_by_evidence() {
        let before = vocab();
        let (after, e, _) = load();
        for w in ["the", "record", "records", "may", "xyzzy"] {
            assert_eq!(before.id(w), after.id(w), "{w}");
        }
        assert_eq!(after.len(), 5);
        assert_eq!(after.id("recorded"), None, "evidence must not mint ids");
        assert!(e.readings(after.len() as WordId).is_empty());
    }

    /// Test 4: Cam96 stays `codes[word_id]` — evidence is independent storage.
    #[test]
    fn cam96_indexing_is_independent_of_evidence() {
        let v = vocab();
        let codes: Vec<Cam96> = (0..v.len() as u8).map(|i| [i; 12]).collect();
        let nsm = Nsm::with_codes(v.clone(), Cam96Space::demo(4), codes.clone());
        let (_e, _) = load_word_forms_csv(FIXTURE, &nsm.vocab).unwrap();
        for w in ["the", "record", "records", "may", "xyzzy"] {
            let id = nsm.vocab.id(w).unwrap() as usize;
            assert_eq!(nsm.code(w), Some(&codes[id]), "{w}");
        }
    }

    /// Test 5: unknown stays unknown; zero stays distinct from unknown.
    #[test]
    fn missing_counts_stay_missing() {
        let (v, e, r) = load();
        let x = v.id("xyzzy").unwrap();
        assert_eq!(e.readings(x)[0].form_count, None, "empty wordFreq is None");
        assert_eq!(e.surface_count(x), Ok(None));
        // No verb reading was ever observed for `records`: unknown, not 0.
        let recs = v.id("records").unwrap();
        assert_eq!(e.surface_pos_count(recs, V), Ok(None));
        // No readings at all: unknown.
        assert_eq!(e.surface_count(v.id("the").unwrap()), Ok(None));
        // An empty lemma string keeps its entry and count; it is just unnamed.
        let may = e.readings(v.id("may").unwrap())[0];
        let entry = e.lemma(may.lemma.unwrap()).unwrap();
        assert_eq!(entry.lemma, None);
        assert_eq!(entry.count, Some(88_994));
        // A literal zero is kept as zero.
        let text = format!("{WORD_FORMS_HEADER}\n1,the,a,0,0,the\n");
        let (z, _) = load_word_forms_csv(&text, &v).unwrap();
        assert_eq!(z.surface_count(v.id("the").unwrap()), Ok(Some(0)));
        // Nothing dropped silently.
        assert_eq!(
            r,
            WordFormsReport {
                rows: 7,
                stored: 5,
                empty_surface: 1,
                unrouted: 1
            }
        );
    }

    /// The lemma total and the form count are different quantities and both
    /// survive; the lemma total is stored once, not summed per form row.
    #[test]
    fn lemma_count_differs_from_form_count_and_is_not_double_counted() {
        let (v, e, _) = load();
        let r = e.readings(v.id("record").unwrap())[0];
        let lemma = e.lemma(r.lemma.unwrap()).unwrap();
        assert_eq!(lemma.count, Some(187_057));
        assert_ne!(lemma.count, r.form_count);
        assert_eq!(e.lemma_pos_count("record", N), Ok(Some(187_057)));
        assert_eq!(e.lemma_pos_count("record", V), Ok(Some(51_375)));
        assert_eq!(e.lemma_count("record"), Ok(Some(238_432)));
        assert_eq!(e.lemma_count("absent"), Ok(None));
        // Two noun form rows named lemma 518; one entry exists for it.
        assert_eq!(e.lemma_entries("record").count(), 2);
    }

    /// A LemmaRef the builder never issued is an error, not a panic.
    #[test]
    fn unknown_lemma_ref_is_refused() {
        let v = vocab();
        let mut b = LexicalEvidenceBuilder::new(&v);
        assert_eq!(
            b.add_reading(
                1,
                LexicalReading {
                    pos: N,
                    lemma: Some(7),
                    form_count: None
                }
            ),
            Err(EvidenceError::UnknownLemma { lemma: 7 })
        );
    }

    /// Unknown wins over overflow wherever the unknown count sits.
    #[test]
    fn unknown_count_takes_precedence_over_overflow() {
        let v = vocab();
        let mut b = LexicalEvidenceBuilder::new(&v);
        b.lemma_entry(1, Some("big"), N, Some(u64::MAX)).unwrap();
        b.lemma_entry(2, Some("big"), V, Some(1)).unwrap();
        b.lemma_entry(3, Some("big"), PosCode(b'j'), None).unwrap();
        let e = b.finish();
        assert_eq!(e.lemma_count("big"), Ok(None));
        // Fully known and too large is still an overflow.
        let mut b = LexicalEvidenceBuilder::new(&v);
        b.lemma_entry(1, Some("big"), N, Some(u64::MAX)).unwrap();
        b.lemma_entry(2, Some("big"), V, Some(1)).unwrap();
        assert_eq!(
            b.finish().lemma_count("big"),
            Err(EvidenceError::CountOverflow)
        );
    }

    /// The loader reads unquoted CSV only; a quoted field is refused rather
    /// than routed with its quotes or split at an embedded comma.
    #[test]
    fn quoted_fields_are_refused() {
        let v = vocab();
        let q = format!("{WORD_FORMS_HEADER}\n518,record,n,187057,120048,\"record\"\n");
        assert_eq!(
            load_word_forms_csv(&q, &v).unwrap_err(),
            EvidenceError::QuotedField { line: 2 }
        );
        let c = format!("{WORD_FORMS_HEADER}\n518,\"re,cord\",n,187057,120048,record\n");
        assert_eq!(
            load_word_forms_csv(&c, &v).unwrap_err(),
            EvidenceError::QuotedField { line: 2 }
        );
    }

    /// Refuse instead of picking a winner.
    #[test]
    fn conflicts_are_refused() {
        let v = vocab();
        let bad = format!("{WORD_FORMS_HEADER}\n5,record,n,10,1,record\n5,record,n,11,2,records\n");
        assert_eq!(
            load_word_forms_csv(&bad, &v).unwrap_err(),
            EvidenceError::ConflictingLemmaEntry { source_key: 5 }
        );
        let dup = format!("{WORD_FORMS_HEADER}\n5,record,n,10,1,record\n5,record,n,10,2,record\n");
        assert!(matches!(
            load_word_forms_csv(&dup, &v).unwrap_err(),
            EvidenceError::DuplicateReading { .. }
        ));
        assert!(matches!(
            load_word_forms_csv("rank,word\n", &v).unwrap_err(),
            EvidenceError::UnexpectedHeader { .. }
        ));
        let short = format!("{WORD_FORMS_HEADER}\n5,record,n,10,1\n");
        assert!(matches!(
            load_word_forms_csv(&short, &v).unwrap_err(),
            EvidenceError::FieldCount {
                expected: 6,
                found: 5,
                ..
            }
        ));
        let mut b = LexicalEvidenceBuilder::new(&v);
        let l = b.lemma_entry(1, Some("record"), N, None).unwrap();
        assert!(matches!(
            b.add_reading(
                1,
                LexicalReading {
                    pos: V,
                    lemma: Some(l),
                    form_count: None
                }
            ),
            Err(EvidenceError::PosMismatch { .. })
        ));
    }
}
