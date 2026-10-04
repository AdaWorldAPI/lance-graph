//! `bible_wave` — the WHOLE-BOOK falsifier: one book, one 64k SoA tile,
//! literal standing wave vs the fire-and-forget ±5 ring.
//!
//! The endgame thesis under test (`E-LC-SCARCITY-INVERSION-1`): a whole book's
//! triples fit ONE `256×256` tile, so context is read LITERALLY over the whole
//! work — no bundle, no beam, no per-sentence reset. v1's `MarkovBundler`
//! superposes a ±5 window and forgets; this example measures exactly how much
//! long-range context that forfeits, on a real public-domain book.
//!
//! Run (KJV from Project Gutenberg #10, not committed):
//! ```sh
//! cargo run --example bible_wave -- /path/to/pg10.txt
//! ```
//!
//! Pipeline: verses → PoS-tag (legacy single-`Pos` tagging: COCA lemma table,
//! then the first `word_forms.csv` row, then a documented archaic fallback;
//! the counted `word_forms.csv` evidence is loaded beside it for measurement
//! only and chooses no tag) → FSM → SPO
//! stream (verse index = version) → `TemporalStream` +
//! the TRAINED Cam96 codebook (`data/`, real Jina-v3 embeddings).
//!
//! Gates (panic on KILL):
//! - G1 whole book fits the 64k SoA (verses ≤ 65,536)
//! - G2 the trained codebook loads and codes align with the vocab
//! - G3 KG is non-trivial (≥ 1,000 triples)
//! - G4 meaning sanity on the trained codebook: sim(god, lord) > sim(god, fish)
//! - D-LXC-2 the multi-reading FSM against the legacy one-tag parse of the
//!   same tokens: every difference traces to an ambiguous token, and the
//!   reading and triple accounting match the pinned KJV layout. Downstream
//!   gates read the CERTAIN triples; alternatives are counted, not streamed.
//!
//! Reported (not gated): the long-range share — % of same-subject recurrence
//! links farther than ±5 (v1's ring forfeits them) and ±8 (the local reference
//! horizon → the Escalate zone).

use deepnsm_v2::coca::{fsm_pos, fsm_pos_tag, predicate_alternatives, reading_set};
use deepnsm_v2::{
    load_cam96_codes, load_cam96_space, load_word_forms_csv, parse_readings, parse_readings_with,
    parse_to_spo, EvidenceError, LexicalEvidence, Nsm, PaletteVocab, Pos, PosSet, Reading,
    ReadingParse, Spo, Tagged, TemporalStream, Typology, WordFormsReport, WordId,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// The trained artifacts are NOT committed — they ship as the `AdaWorldAPI/lance-graph`
/// release `v0.1.0-cam96-data` (see `data/README.md` for the fetch commands).
/// Loaded at runtime from `data/` (override the directory with
/// `DEEPNSM_V2_DATA`).
fn data_file(name: &str) -> Vec<u8> {
    let dir = std::env::var("DEEPNSM_V2_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"));
    let path = dir.join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "missing {} ({e}) — fetch the v0.1.0-cam96-data release assets per data/README.md",
            path.display()
        )
    })
}

/// D-LXC-2 accounting: the multi-reading parse against the legacy one-tag
/// parse of the same tokens. Every difference must trace to an ambiguous
/// token; anything else is a KILL.
#[derive(Default)]
struct LexicalDecodeReport {
    tokens: usize,
    unknown: usize,
    single: usize,
    ambiguous: usize,
    /// Ambiguous tokens the structure narrowed (fewer readings survived).
    narrowed: usize,
    /// Ambiguous tokens still carrying ≥ 2 readings at their verse's end.
    still_ambiguous: usize,
    /// Ambiguous tokens whose LEGACY tag did not survive — the structure
    /// overruled the first-row-wins tag.
    legacy_eliminated: usize,
    legacy_triples: usize,
    certain: usize,
    alternative: usize,
    /// Legacy triples that are neither certain nor alternative.
    legacy_lost: usize,
    /// Legacy triples that became alternatives.
    legacy_to_alternative: usize,
    /// Certain triples the legacy parse did not produce.
    new_certain: usize,
    verses_changed: usize,
    peak_configs: usize,
    overflow_flushes: usize,
    /// (configuration, reading) pairs the slot rule dropped (D-LXC-13).
    slot_dropped: usize,
    /// (configuration, reading) pairs the licensing rule dropped.
    unlicensed_dropped: usize,
    /// Verses whose triples changed although no token's reading set
    /// narrowed: a rule dropped a reading on one path only.
    combination_only: usize,
    ambiguous_words: HashMap<String, usize>,
    eliminated_words: HashMap<String, usize>,
}

impl LexicalDecodeReport {
    fn verse(
        &mut self,
        readings: &[Reading],
        legacy_tags: &[Tagged],
        words: &[String],
        parse: &ReadingParse,
        legacy: &[Spo],
    ) {
        let mut any_ambiguous = false;
        for (k, (r, t)) in readings.iter().zip(legacy_tags).enumerate() {
            if r.pos.contains(Pos::Stop) {
                continue;
            }
            self.tokens += 1;
            match r.pos.len() {
                0 => {
                    self.unknown += 1;
                    assert_eq!(
                        t.pos,
                        Pos::Other,
                        "KILL D-LXC-2: unknown word {:?} had a legacy tag",
                        words[k]
                    );
                }
                1 => {
                    self.single += 1;
                    assert!(
                        r.pos.contains(t.pos),
                        "KILL D-LXC-2 unexplained: {:?} has one reading {:?} but legacy tag {:?}",
                        words[k],
                        r.pos,
                        t.pos
                    );
                }
                _ => {
                    any_ambiguous = true;
                    assert!(
                        r.pos.contains(t.pos),
                        "KILL D-LXC-2 unexplained: legacy tag {:?} of {:?} is not among its readings {:?}",
                        t.pos,
                        words[k],
                        r.pos
                    );
                    *self.ambiguous_words.entry(words[k].clone()).or_default() += 1;
                }
            }
        }
        for sv in &parse.ambiguous {
            self.ambiguous += 1;
            if sv.survived != sv.entered {
                self.narrowed += 1;
            }
            if sv.survived.len() >= 2 {
                self.still_ambiguous += 1;
            }
            if !sv.survived.contains(legacy_tags[sv.index].pos) {
                self.legacy_eliminated += 1;
                *self
                    .eliminated_words
                    .entry(words[sv.index].clone())
                    .or_default() += 1;
            }
        }
        self.legacy_triples += legacy.len();
        self.certain += parse.certain.len();
        self.alternative += parse.alternative.len();
        self.peak_configs = self.peak_configs.max(parse.peak_configs);
        self.overflow_flushes += parse.overflow_flushes;
        self.slot_dropped += parse.slot_dropped;
        self.unlicensed_dropped += parse.unlicensed_dropped;
        let lost = legacy
            .iter()
            .filter(|t| !parse.certain.contains(t) && !parse.alternative.contains(t))
            .count();
        let to_alt = legacy
            .iter()
            .filter(|t| parse.alternative.contains(t))
            .count();
        let new_certain = parse.certain.iter().filter(|t| !legacy.contains(t)).count();
        self.legacy_lost += lost;
        self.legacy_to_alternative += to_alt;
        self.new_certain += new_certain;
        let changed = lost + to_alt + new_certain > 0 || !parse.alternative.is_empty();
        if changed {
            self.verses_changed += 1;
            assert!(
                any_ambiguous,
                "KILL D-LXC-2 unexplained: a verse with no ambiguous token changed its triples"
            );
        }
        if lost + new_certain > 0 {
            let narrowed = parse.ambiguous.iter().any(|sv| sv.survived != sv.entered);
            assert!(
                narrowed || parse.unlicensed_dropped > 0 || parse.slot_dropped > 0,
                "KILL D-LXC-2 unexplained: triples changed but no reading or reading \
                 combination was eliminated"
            );
            if !narrowed {
                self.combination_only += 1;
            }
        }
    }

    fn top(m: &HashMap<String, usize>, n: usize) -> String {
        let mut v: Vec<(&String, &usize)> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        v.iter()
            .take(n)
            .map(|(w, c)| format!("{w} {c}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn print_and_gate(&self) {
        println!(
            "D-LXC-2  tokens {}: single {}, ambiguous {} ({} words), unknown {}",
            self.tokens,
            self.single,
            self.ambiguous,
            self.ambiguous_words.len(),
            self.unknown
        );
        println!(
            "D-LXC-2  ambiguous tokens: narrowed {}, still ambiguous {}, legacy tag eliminated {}",
            self.narrowed, self.still_ambiguous, self.legacy_eliminated
        );
        println!(
            "D-LXC-2  triples: legacy {} → certain {} + alternative {}; legacy→alternative {}, \
             legacy lost {}, new certain {}; verses changed {}",
            self.legacy_triples,
            self.certain,
            self.alternative,
            self.legacy_to_alternative,
            self.legacy_lost,
            self.new_certain,
            self.verses_changed
        );
        println!(
            "D-LXC-2  configurations: peak {}, overflow flushes {}; dropped by slot rule {}, \
             by licensing {} (verses changed with no token narrowed: {})",
            self.peak_configs,
            self.overflow_flushes,
            self.slot_dropped,
            self.unlicensed_dropped,
            self.combination_only
        );
        println!(
            "D-LXC-2  most ambiguous words: {}",
            Self::top(&self.ambiguous_words, 12)
        );
        println!(
            "D-LXC-2  legacy tag eliminated most for: {}",
            Self::top(&self.eliminated_words, 12)
        );
        assert_eq!(
            self.single + self.ambiguous + self.unknown,
            self.tokens,
            "KILL D-LXC-2: token accounting does not add up"
        );
        assert_eq!(self.overflow_flushes, 0, "KILL D-LXC-2: a verse overflowed");
        println!("D-LXC-2 PASS every difference traces to an ambiguous token");
        // Pinned for the released `bible_vocab.txt` + Gutenberg `pg10.txt`,
        // lemma noun/verb tags widened to their predicate alternative and
        // decided by position (D-LXC-13, reopens D-LXC-3). Before D-LXC-13:
        // reading (771_176, 683_805, 3_363, 141, 84_008, 1_908, 1_455, 179,
        // 35), triples (70_393, 69_670, 1_716, 732, 113, 122, 1_131, 16).
        // Floating quantifier (Bugbot on #1321): a determiner straight after
        // the subject's head no longer licenses away the next verb reading.
        // Before it: reading (…, 14_383, 20_441, 1_943, 162), triples
        // (70_393, 57_350, 29_001, 12_984, 1_231, 1_190, 12_437, 256).
        // A deliberate decoder change re-pins these with the difference
        // reported.
        assert_eq!(
            (
                self.tokens,
                self.single,
                self.ambiguous,
                self.ambiguous_words.len(),
                self.unknown,
                self.narrowed,
                self.still_ambiguous,
                self.legacy_eliminated,
                self.eliminated_words.len(),
            ),
            (771_176, 652_344, 34_824, 451, 84_008, 13_732, 21_092, 1_805, 155),
            "KILL D-LXC-2: reading accounting moved from the pinned KJV layout"
        );
        assert_eq!(
            (
                self.legacy_triples,
                self.certain,
                self.alternative,
                self.legacy_to_alternative,
                self.legacy_lost,
                self.new_certain,
                self.verses_changed,
                self.peak_configs,
            ),
            (70_393, 57_277, 29_684, 13_151, 1_130, 1_183, 12_581, 256),
            "KILL D-LXC-2: triple accounting moved from the pinned KJV layout"
        );
        println!("D-LXC-2 PASS accounting matches the pinned KJV layout");
    }
}

/// One COCA lexicon table, read as text.
///
/// The tables are committed under `crates/deepnsm/word_frequency/` — a
/// location, not an ownership claim: v2 reads them as its own lexical source
/// and never links v1 code. `DEEPNSM_V2_LEXICON` points at another copy with
/// the same bytes (e.g. one fetched from the Tigris bake
/// `lance-graph/codebooks/deepnsm-v2-academic-coca-v1/`, see `data/README.md`).
/// The bytes and row order are the contract; the path is not.
fn lexicon_file(name: &str) -> String {
    let dir = std::env::var("DEEPNSM_V2_LEXICON").map_or_else(
        |_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../deepnsm/word_frequency"),
        PathBuf::from,
    );
    let path = dir.join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing lexicon table {} ({e})", path.display()))
}

/// D-LXC-15: German casing as a silver noun/verb label for KJV tokens.
///
/// For a KJV noun/verb homograph, its German associates come from the
/// release's corpus-derived `alignment_en-de.tsv` (KJV → Luther 1545, Dice /
/// PMI, cooc ≥ 5). Luther 1545 capitalises nouns, so an associate found in
/// the same verse labels the token: capitalised, not sentence-initial and a
/// noun in the German lexicon → noun; lowercase and a verb there → verb.
/// The label is contrastive: a word is used only when its associates attest
/// BOTH readings (one-sided associates — collocates like `rose` → `Morgens`
/// — could only ever say one thing). A token is labelled only when every
/// associate found agrees. Psalms are excluded: Luther 1545 numbers them
/// differently (the release's documented versification offset). Inputs: the
/// `v0.1.0-codebooks-2026-07-26` release (`pd-texts`, `rosetta-gpl`, `de`),
/// extracted under `ROSETTA_DIR`.
struct RosettaSilver {
    verses: HashMap<(u16, u16, u16), String>,
    align: HashMap<String, Vec<String>>,
    /// German word form → its lexicon readings (noun, verb).
    de_pos: HashMap<String, (bool, bool)>,
    /// Per policy: (labelled tokens decided, decided right).
    legacy: (usize, usize),
    shipped: (usize, usize),
    clause: (usize, usize),
    /// The shipped rules' decisions by kept reading: (Noun n, right, Verb n, right).
    shipped_by_kept: [(usize, usize); 2],
    clause_by_kept: [(usize, usize); 2],
    /// Legacy right on the tokens each policy decided: (shipped, clause).
    legacy_on_shipped: usize,
    legacy_on_clause: usize,
    labelled: usize,
    candidates: usize,
}

impl RosettaSilver {
    fn load(dir: &str) -> Self {
        let read = |p: &str| {
            std::fs::read_to_string(format!("{dir}/{p}"))
                .unwrap_or_else(|e| panic!("{dir}/{p}: {e}"))
        };
        let json: serde_json::Value =
            serde_json::from_str(&read("pd-texts/bible_luther1545.json")).expect("luther1545 json");
        let mut verses = HashMap::new();
        for book in json["books"].as_array().expect("books") {
            let nr = book["nr"].as_u64().expect("book nr") as u16;
            for ch in book["chapters"].as_array().expect("chapters") {
                for v in ch["verses"].as_array().expect("verses") {
                    let c = v["chapter"].as_u64().expect("chapter") as u16;
                    let n = v["verse"].as_u64().expect("verse") as u16;
                    let t = v["text"].as_str().expect("text").to_string();
                    verses.insert((nr, c, n), t);
                }
            }
        }
        let mut align: HashMap<String, Vec<String>> = HashMap::new();
        for line in read("rosetta-gpl/alignment_en-de.tsv").lines().skip(1) {
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() >= 2 {
                align
                    .entry(c[0].to_string())
                    .or_default()
                    .push(c[1].to_string());
            }
        }
        let mut de_pos: HashMap<String, (bool, bool)> = HashMap::new();
        for l in read("de/lexicon.tsv")
            .lines()
            .filter(|l| !l.starts_with('#'))
        {
            let c: Vec<&str> = l.split('\t').collect();
            if c.len() >= 3 {
                let e = de_pos.entry(c[0].to_lowercase()).or_default();
                e.0 |= c[2] == "n";
                e.1 |= c[2] == "v";
            }
        }
        Self {
            verses,
            align,
            de_pos,
            legacy: (0, 0),
            shipped: (0, 0),
            clause: (0, 0),
            shipped_by_kept: [(0, 0); 2],
            clause_by_kept: [(0, 0); 2],
            legacy_on_shipped: 0,
            legacy_on_clause: 0,
            labelled: 0,
            candidates: 0,
        }
    }

    /// The silver label for English `word` in verse `key`, or `None`.
    fn label(&self, key: (u16, u16, u16), word: &str) -> Option<Pos> {
        const PSALMS: u16 = 19;
        if key.0 == PSALMS {
            return None;
        }
        let text = self.verses.get(&key)?;
        let assoc = self.align.get(word)?;
        let pos_of = |a: &String| self.de_pos.get(a).copied().unwrap_or_default();
        if !(assoc.iter().any(|a| pos_of(a).0) && assoc.iter().any(|a| pos_of(a).1)) {
            return None;
        }
        let mut found: Option<Pos> = None;
        let mut initial = true;
        for raw in text.split_whitespace() {
            let tok: String = raw.chars().filter(|c| c.is_alphabetic()).collect();
            let this_initial = initial;
            initial = raw.ends_with(['.', '!', '?', ':']);
            if tok.is_empty() || this_initial {
                continue;
            }
            let lower = tok.to_lowercase();
            if !assoc.contains(&lower) {
                continue;
            }
            let (noun, verb) = pos_of(&lower);
            let pos = if tok.starts_with(char::is_uppercase) && noun {
                Pos::Noun
            } else if !tok.starts_with(char::is_uppercase) && verb {
                Pos::Verb
            } else {
                continue;
            };
            match found {
                None => found = Some(pos),
                Some(p) if p == pos => {}
                Some(_) => return None,
            }
        }
        found
    }

    fn kept(set: PosSet) -> Option<Pos> {
        match (set.contains(Pos::Noun), set.contains(Pos::Verb)) {
            (true, false) => Some(Pos::Noun),
            (false, true) => Some(Pos::Verb),
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn verse(
        &mut self,
        key: (u16, u16, u16),
        readings: &[Reading],
        legacy_tags: &[Tagged],
        words: &[String],
        shipped: &ReadingParse,
        clause: &ReadingParse,
    ) {
        let clause_by_index: HashMap<usize, PosSet> = clause
            .ambiguous
            .iter()
            .map(|sv| (sv.index, sv.survived))
            .collect();
        for sv in &shipped.ambiguous {
            let entered = readings[sv.index].pos;
            if !(entered.contains(Pos::Noun) && entered.contains(Pos::Verb)) {
                continue;
            }
            self.candidates += 1;
            let Some(gold) = self.label(key, &words[sv.index]) else {
                continue;
            };
            self.labelled += 1;
            let legacy_ok = legacy_tags[sv.index].pos == gold;
            self.legacy.0 += 1;
            self.legacy.1 += usize::from(legacy_ok);
            let slot = |p: Pos| usize::from(p == Pos::Verb);
            if let Some(k) = Self::kept(sv.survived) {
                if k != gold && std::env::var_os("ROSETTA_DUMP").is_some() {
                    eprintln!(
                        "SILVER kept {k:?} german {gold:?} [{}] | {} | {}",
                        words[sv.index],
                        words.join(" "),
                        self.verses.get(&key).map_or("", String::as_str)
                    );
                }
                self.shipped.0 += 1;
                self.shipped.1 += usize::from(k == gold);
                self.legacy_on_shipped += usize::from(legacy_ok);
                let e = &mut self.shipped_by_kept[slot(k)];
                e.0 += 1;
                e.1 += usize::from(k == gold);
            }
            if let Some(k) = clause_by_index.get(&sv.index).copied().and_then(Self::kept) {
                self.clause.0 += 1;
                self.clause.1 += usize::from(k == gold);
                self.legacy_on_clause += usize::from(legacy_ok);
                let e = &mut self.clause_by_kept[slot(k)];
                e.0 += 1;
                e.1 += usize::from(k == gold);
            }
        }
    }

    fn print(&self) {
        let pct = |n: usize, d: usize| {
            if d == 0 {
                0.0
            } else {
                100.0 * n as f64 / d as f64
            }
        };
        println!(
            "D-LXC-15  KJV noun/verb homographs: {} ({} with a German-casing label)",
            self.candidates, self.labelled
        );
        println!(
            "D-LXC-15  legacy lemma tag: {:.1}% right on all labelled",
            pct(self.legacy.1, self.legacy.0)
        );
        for (name, (n, r), legacy_r, by_kept) in [
            (
                "slot + licensing",
                self.shipped,
                self.legacy_on_shipped,
                self.shipped_by_kept,
            ),
            (
                "+ clause rule",
                self.clause,
                self.legacy_on_clause,
                self.clause_by_kept,
            ),
        ] {
            println!(
                "D-LXC-15  {name}: decided {n}, {:.1}% right (legacy tag on the same tokens \
                 {:.1}%); kept Noun {} at {:.1}%, kept Verb {} at {:.1}%",
                pct(r, n),
                pct(legacy_r, n),
                by_kept[0].0,
                pct(by_kept[0].1, by_kept[0].0),
                by_kept[1].0,
                pct(by_kept[1].1, by_kept[1].0)
            );
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: bible_wave <pg10.txt> [--export <spo.tsv>] [--export-verses <verses.tsv>]");
    // The inbound leg can EMIT its whole-book SPO/belief stream for the
    // lance-graph reasoning layer to consume (the SoC seam, `E-DEEPNSM-V2-IS-
    // INBOUND-LEG-REASONING-LIVES-IN-LANCE-GRAPH-1`): the planner example
    // `reason_whole_book` reads this TSV into a `BeliefArena` and reasons.
    let export = std::env::args()
        .position(|a| a == "--export")
        .and_then(|i| std::env::args().nth(i + 1));
    // The reasoning layer's four-stance panel needs verse TEXT, not triples:
    // `stance::stream()` mints RungLifts inside a complementizer window and
    // derives negation polarity from the clause — neither survives the flat
    // (s,p,o,verse) export, which is why 3 of 4 stances measured UNREACHABLE
    // on that path (plan §12.3a″). Text is emitted as its OWN artifact rather
    // than a column, so the SPO export's 7-column shape is untouched and no
    // existing consumer changes. The seam still holds: this leg emits text,
    // it does not reason over it (`E-DEEPNSM-V2-IS-INBOUND-LEG-...`).
    let export_verses = std::env::args()
        .position(|a| a == "--export-verses")
        .and_then(|i| std::env::args().nth(i + 1));
    let raw = std::fs::read_to_string(&path).expect("read KJV text");

    // Verse splitting lives in the LIBRARY (`deepnsm_v2::corpus`) so that
    // `cargo test` gates it. It used to be inline here, where it carried an
    // OT-only truncation for its entire life and could not be unit-tested:
    // cargo compiles an example but never runs its `main()`, and the corpus is
    // not committed. See `corpus::split_verses` for the three-`***` contract.
    let split = deepnsm_v2::corpus::split_verses_detailed(&raw);
    let verses: Vec<String> = split.verses.clone();

    // G1 — the whole book is ONE 64k SoA tile.
    assert!(verses.len() <= 65_536, "KILL G1: book exceeds the 64k tile");
    // G1b — the corpus actually IS the whole book. This example claimed
    // "whole book" for its entire life while stopping at the lone `***`
    // between the testaments, i.e. at Malachi 4:6 — 23,145 verses, the Old
    // Testament exactly. The assert below is what makes that failure loud:
    // if the input announces a New Testament, the parse must have crossed
    // into it. Read from the PARSE (`CorpusSplit::crossed_new_testament`), not
    // from a verse-count threshold: a count comparison falsely killed an
    // NT-only corpus and missed an uppercase heading entirely.
    if let Some(crossed) = deepnsm_v2::corpus::crossed_into_new_testament(&raw, &split) {
        assert!(
            crossed,
            "KILL G1b: input announces a New Testament but the parse emitted no \
             verse after that heading — the OT-only truncation is back ({} verses \
             parsed; the historical bug stopped at {} = the OT exactly)",
            verses.len(),
            deepnsm_v2::corpus::KJV_OLD_TESTAMENT_VERSES
        );
    }
    assert!(
        !verses.iter().any(|v| v.contains("***")),
        "KILL G1b: a `***` fence leaked into verse text"
    );
    println!(
        "G1 PASS  whole book = {} verses ≤ 65,536 (one 256×256 tile)",
        verses.len()
    );

    // ── vocab + TRAINED codebook (real Jina-v3 embeddings; runtime-fetched) ──
    let vocab_text = String::from_utf8(data_file("bible_vocab.txt")).expect("utf8 vocab");
    let mut vocab = PaletteVocab::new();
    vocab.from_frequency_ranked(vocab_text.lines());
    let space = load_cam96_space(&data_file("cam96_codebook.bin")).expect("codebook artifact");
    let codes = load_cam96_codes(&data_file("cam96_codes.bin")).expect("codes artifact");
    assert_eq!(codes.len(), vocab.len(), "KILL G2: codes/vocab misaligned");
    let nsm = Nsm::with_codes(vocab, space, codes);
    println!(
        "G2 PASS  trained codebook loaded: {} words, 12 axes",
        nsm.vocab.len()
    );

    // ── PoS: COCA lemmas + FORMS + archaic fallback ──
    let lemmas_csv = lexicon_file("lemmas_5k.csv");
    let forms_csv = lexicon_file("word_forms.csv");
    let tagger =
        Tagger::load(&lemmas_csv, &forms_csv, &nsm.vocab).expect("word_forms.csv evidence");
    let r = &tagger.report;
    println!(
        "LEXICON  word_forms: {} rows, {} readings stored, {} empty surface, {} not in vocab",
        r.rows, r.stored, r.empty_surface, r.unrouted
    );
    // G6 (D-LXC-1) is the non-interference invariant, proven by the focused
    // test `counts_change_evidence_never_the_readings_or_the_tag`: counts move
    // the evidence order and coverage, never the reading set or the tag. The
    // historical "25 moved tags" measured count leaking into the tag; see the
    // board entry of 2026-09-29.

    // G8c (D-LXC-11) — coverage bands, a MEASUREMENT of reading concentration.
    // The cuts are quartiles of the band population, calibrated at load.
    // Reported only: no tag reads a band.
    // The pinned numbers hold for the released `bible_vocab.txt` only.
    let cuts = tagger.cuts.expect("KILL G8c: empty band population");
    let count = |b: CoverageBand| tagger.bands.iter().filter(|x| **x == Some(b)).count();
    let (contested, leaning, decisive) = (
        count(CoverageBand::Contested),
        count(CoverageBand::Leaning),
        count(CoverageBand::Decisive),
    );
    let banded: Vec<u8> = (0..nsm.vocab.len())
        .filter_map(|i| WordId::try_from(i).ok())
        .filter_map(|id| band_share(&tagger.lemmas, &tagger.evidence, &nsm.vocab, id))
        .collect();
    let (min, max) = (
        banded.iter().min().copied().unwrap_or(0),
        banded.iter().max().copied().unwrap_or(0),
    );
    println!(
        "BANDS    population {}, cuts ({}, {}), shares {min}..{max}: contested {contested}, \
         leaning {leaning}, decisive {decisive}",
        cuts.population, cuts.lo, cuts.hi
    );
    assert_eq!(
        (
            cuts.population,
            cuts.lo,
            cuts.hi,
            contested,
            leaning,
            decisive
        ),
        (141, 72, 97, 34, 71, 36),
        "KILL G8c: coverage bands moved from the pinned KJV-vocabulary layout"
    );
    println!("G8c PASS coverage bands match the pinned KJV-vocabulary layout");

    // ── stream: verse index = version; multi-reading FSM → SPO (D-LXC-2) ──
    // Each token enters with every reading it has; the parser keeps what the
    // structure cannot separate. The LEGACY one-tag parse runs beside it on
    // the same tokens, only to account for every difference.
    let mut stream = TemporalStream::new();
    let mut all: Vec<(u64, Spo)> = Vec::new();
    let mut lx = LexicalDecodeReport::default();
    let mut readings_buf: Vec<Reading> = Vec::new();
    let mut legacy_buf: Vec<Tagged> = Vec::new();
    let mut words_buf: Vec<String> = Vec::new();
    // D-LXC-15: optional Rosetta silver labels. The verse key is
    // (book, chapter, verse); a book starts where the markers restart at 1:1.
    let mut rosetta = std::env::var("ROSETTA_DIR")
        .ok()
        .filter(|d| !d.is_empty())
        .map(|d| RosettaSilver::load(&d));
    let mut book = 0u16;
    let keys: Vec<(u16, u16, u16)> = split
        .markers
        .iter()
        .map(|&(c, v)| {
            if (c, v) == (1, 1) {
                book += 1;
            }
            (book, c, v)
        })
        .collect();
    assert_eq!(keys.len(), verses.len(), "KILL: one marker per verse");
    let with_clause = Typology {
        predicate_required: true,
        ..Typology::ENGLISH
    };
    for (vi, verse) in verses.iter().enumerate() {
        readings_buf.clear();
        legacy_buf.clear();
        words_buf.clear();
        for tok in verse.split_whitespace() {
            let Some(w) = normalise(tok) else { continue };
            let Some(id) = nsm.vocab.id(&w) else { continue };
            readings_buf.push(Reading::new(id, tagger.readings(&w, id)));
            legacy_buf.push(Tagged::new(id, tagger.pos(&w)));
            words_buf.push(w);
        }
        readings_buf.push(Reading::stop()); // verse boundary flushes
        legacy_buf.push(Tagged::new(0, Pos::Stop));
        let parse = parse_readings(&readings_buf);
        let legacy = parse_to_spo(&legacy_buf);
        lx.verse(&readings_buf, &legacy_buf, &words_buf, &parse, &legacy);
        if let Some(r) = rosetta.as_mut() {
            let clause = parse_readings_with(&readings_buf, with_clause);
            r.verse(
                keys[vi],
                &readings_buf,
                &legacy_buf,
                &words_buf,
                &parse,
                &clause,
            );
        }
        for &t in &parse.certain {
            stream.push(vi as u64, t);
            all.push((vi as u64, t));
        }
    }
    lx.print_and_gate();
    if let Some(r) = &rosetta {
        assert_eq!(book, 66, "KILL D-LXC-15: the KJV must split into 66 books");
        r.print();
    }

    // ── SoC seam (text): emit labelled verse text for the reasoning layer ──
    if let Some(out) = &export_verses {
        use std::io::Write;
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(out).expect("create verse export"));
        for (i, v) in verses.iter().enumerate() {
            // Verse text is whitespace-normalised by the splitter and carries
            // no tabs, so a 2-column TSV round-trips without quoting.
            debug_assert!(!v.contains('\t'), "verse text must not contain a tab");
            writeln!(f, "{i}\t{v}").expect("write verse export");
        }
        println!("EXPORT  {} verses -> {}", verses.len(), out);
    }

    // ── SoC seam: emit the whole-book belief stream for the reasoning layer ──
    if let Some(out) = &export {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("create export"));
        for &(v, t) in &all {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                t.subject,
                nsm.vocab.word(t.subject).unwrap_or("?"),
                t.predicate,
                nsm.vocab.word(t.predicate).unwrap_or("?"),
                t.object,
                nsm.vocab.word(t.object).unwrap_or("?"),
                v,
            )
            .expect("write export");
        }
        println!("EXPORT  {} triples -> {}", all.len(), out);
        return;
    }

    // G3 — non-trivial KG.
    let mut subjects: HashMap<u16, Vec<u64>> = HashMap::new();
    for &(v, t) in &all {
        subjects.entry(t.subject).or_default().push(v);
    }
    let distinct_s = subjects.len();
    assert!(
        all.len() >= 1_000,
        "KILL G3: too few triples ({})",
        all.len()
    );
    println!(
        "G3 PASS  KG: {} triples, {} distinct subjects, {} distinct predicates",
        all.len(),
        distinct_s,
        all.iter()
            .map(|&(_, t)| t.predicate)
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    // ── the long-range measurement: what does fire-and-forget ±5 forfeit? ──
    let (mut links, mut beyond5, mut beyond8) = (0u64, 0u64, 0u64);
    for occs in subjects.values() {
        for w in occs.windows(2) {
            let gap = w[1] - w[0];
            if gap == 0 {
                continue;
            }
            links += 1;
            if gap > 5 {
                beyond5 += 1;
            }
            if gap > 8 {
                beyond8 += 1;
            }
        }
    }
    println!(
        "LONG-RANGE  {} same-subject recurrence links: {:.1}% beyond ±5 (v1 ring FORFEITS), {:.1}% beyond ±8 (Escalate zone → global graph)",
        links,
        100.0 * beyond5 as f64 / links as f64,
        100.0 * beyond8 as f64 / links as f64
    );

    // window sanity: the literal read reaches any span — GATED, not just logged
    // (a temporal-range regression must fail the falsifier, per #803 review).
    // The window is a borrowing projection of the stream, so the whole-book
    // read is COUNTED in place — asking how wide it is never materializes it.
    let w_len = stream
        .window_range(lance_graph_contract::temporal_pov::VersionRange::new(
            0,
            verses.len() as u64,
        ))
        .count();
    assert_eq!(
        w_len,
        all.len(),
        "KILL: whole-book window did not return every streamed triple"
    );
    println!("WINDOW      whole-book literal read returns {w_len} triples (no bundle, no reset)");

    // G4 — trained-codebook meaning sanity.
    let near = nsm.word_similarity("god", "lord").expect("in vocab");
    let far = nsm.word_similarity("god", "fish").expect("in vocab");
    assert!(
        near > far,
        "KILL G4: sim(god,lord)={near} !> sim(god,fish)={far}"
    );
    println!(
        "G4 PASS  meaning (trained codebook): sim(god,lord)={near:.3} > sim(god,fish)={far:.3}"
    );

    // ── D-SRS-1 — the derivation-pointer fabric over the SAME whole-book KG ──
    // The graph reasons about itself: per-predicate transitive composition, each
    // derived triple carrying premise pointers (the pointers ARE the proof tree),
    // stamped max(premise rungs)+1. The pre-registered gate is STRUCTURAL and
    // proven exhaustively (all three metrics incl. fixed-point termination) by
    // the unit tests in `src/reason.rs`. At BOOK scale we deliberately BOUND the
    // closure: the KJV `begat` genealogies are long same-predicate chains whose
    // FULL transitive closure is O(N²) (empirically the whole-book closure does
    // not settle quickly) — and bounding the derivation horizon is exactly what
    // Layers 2-3 prescribe (±8-local + Escalate; the D-SRS-2 rung cap). The
    // SOUNDNESS half of the gate — 100% premise resolvability + acyclicity —
    // holds on any prefix of the closure, so the bounded run re-checks it on the
    // real book without paying for the full O(N²) genealogy closure.
    const DERIV_HORIZON: usize = 50_000;
    let base: Vec<Spo> = all.iter().map(|&(_, t)| t).collect();
    let arena = deepnsm_v2::reason::DerivationArena::derive_transitive_capped(&base, DERIV_HORIZON);
    let g = arena.gate();
    // Book-scale assertion: SOUNDNESS (the horizon-independent half of the gate).
    assert!(
        g.resolvability_pct == 100.0 && g.acyclic,
        "KILL D-SRS-1 soundness: resolvability={:.1}% acyclic={}",
        g.resolvability_pct,
        g.acyclic
    );
    let horizon = if g.terminated {
        "full fixed point".to_string()
    } else {
        format!("bounded at {DERIV_HORIZON} (full closure is larger — the genealogy O(N²), Layer-2/3 bounds it)")
    };
    println!(
        "D-SRS-1 PASS  derivation fabric: {} base → {} derived triples ({} passes, {horizon}); \
         SOUND — premise resolvability {:.1}%, acyclic={} (strictly-lower rung)",
        g.base, g.derived, g.passes, g.resolvability_pct, g.acyclic
    );

    // ── D-SRS-2 (reshaped) — the SHAPE DETECTOR + ancestry relocation ──
    // The graph reasons about the best representation of its own knowledge
    // (rung-2 meta-awareness, mechanical): per-predicate shape census, then the
    // trie target's ancestry RELOCATES to the DN/HHTL radix-trie codebook —
    // is_ancestor_of = prefix containment — and the materialized closure is
    // deleted after the exactness falsifier proves the trie carries it.
    let census = deepnsm_v2::shape::detect_all_measured(&base);
    println!(
        "D-SRS-2 measured census (top 5 of {} predicates):",
        census.len()
    );
    for r in census.iter().take(5) {
        println!(
            "    '{}' — {} edges, {} entities, cyclic={}, pressure={}, covered={}, coverage={:.2}, amort={:.2}x → {:?} (SPOG G={})",
            nsm.vocab.word(r.predicate).unwrap_or("?"),
            r.edges, r.entities, r.cyclic, r.closure_pressure,
            r.covered, r.coverage, r.amortization, r.recommend, r.recommend.graph_id()
        );
    }

    // Trie target (pre-registered): highest-edge predicate the MEASURED router
    // routes to RadixTrie or TriePlusEscalate.
    let target = census
        .iter()
        .find(|r| {
            matches!(
                r.recommend,
                deepnsm_v2::shape::Representation::RadixTrie
                    | deepnsm_v2::shape::Representation::TriePlusEscalate
            )
        })
        .expect("KILL D-SRS-2: no predicate routed to a trie representation");
    let target_word = nsm.vocab.word(target.predicate).unwrap_or("?");
    // Dedup edges exactly as the measured router did, so the trie here matches
    // the census's re-measurement (a repeated (p,c) is frequency, not a second
    // parent).
    let mut target_edges: Vec<(u16, u16)> = base
        .iter()
        .filter(|t| t.predicate == target.predicate)
        .map(|t| (t.subject, t.object))
        .collect();
    target_edges.sort_unstable();
    target_edges.dedup();
    let trie = deepnsm_v2::FamilyTrie::build(&target_edges);
    println!(
        "D-SRS-2 trie target: '{}' ({:?}) — covered {} entities, residue: {} multi-parent + {} on-cycle; \
         max DN depth {}, HHTL-packable {} (≤12 deep, ≤16 fan)",
        target_word,
        target.recommend,
        trie.covered(),
        trie.multi_parent_residue(),
        trie.cycle_residue(),
        trie.max_depth(),
        trie.hhtl_packable()
    );

    // G-SRS2-a — EXACTNESS: trie prefix-ancestry == the uncapped closure of the
    // trie's DIRECT forest edges (the closure adds the multi-hop pairs), as
    // sets, both directions — a two-implementation differential oracle
    // (parent-pointer ascent vs the reason.rs fixed-point engine).
    let forest: Vec<Spo> = trie
        .forest_edges()
        .iter()
        .map(|&(p, c)| Spo::new(p, target.predicate, c))
        .collect();
    let closure = deepnsm_v2::reason::DerivationArena::derive_transitive(&forest);
    let cg = closure.gate();
    // G-SRS2-d — TERMINATION through relocation: the shape-routed forest
    // closure reaches a TRUE fixed point, uncapped, on the real book.
    assert!(
        cg.passed(),
        "KILL D-SRS-2 (d): forest closure did not soundly terminate: {cg:?}"
    );
    let closure_pairs: std::collections::HashSet<(u16, u16)> = closure
        .entries()
        .iter()
        .map(|d| (d.triple.subject, d.triple.object))
        .collect();
    let trie_pairs = trie.ancestor_pairs();
    assert_eq!(
        trie_pairs, closure_pairs,
        "KILL D-SRS-2 (a): trie prefix-ancestry != materialized closure"
    );
    // G-SRS2v2-a' — the OPERATIONAL api on real book data: `is_ancestor_of` (the
    // "ancestry lives in the key" primitive) must agree with the closure set,
    // and be strict (no self-ancestry). Exercised here at book scale, not just
    // in unit tests.
    for &(a, z) in &trie_pairs {
        assert!(
            trie.is_ancestor_of(a, z),
            "KILL D-SRS-2 (a'): is_ancestor_of({a},{z}) false but the pair is in the closure"
        );
        assert!(
            !trie.is_ancestor_of(z, a),
            "KILL D-SRS-2 (a'): is_ancestor_of is not antisymmetric on ({a},{z})"
        );
    }
    // dn integrity on the deepest covered node: the DN is an ancestor chain
    // ending at the node, and EVERY DN member is an ancestor of it (dn ⇔
    // is_ancestor_of agreement, at book scale).
    if let Some(deepest) = trie
        .forest_edges()
        .iter()
        .map(|&(_, c)| c)
        .max_by_key(|&c| trie.dn(c).map_or(0, |p| p.len()))
    {
        let dn = trie.dn(deepest).expect("covered node has a DN");
        assert_eq!(
            *dn.last().unwrap(),
            deepest,
            "KILL D-SRS-2 (a'): DN must end at its own node"
        );
        for &a in &dn[..dn.len() - 1] {
            assert!(
                trie.is_ancestor_of(a, deepest),
                "KILL D-SRS-2 (a'): DN member {a} is not an ancestor of {deepest}"
            );
        }
    }
    // G-SRS2v2-b — MEASURED FIT: the detector's CLAIM must equal an independent
    // re-measurement (coverage ≥ 0.8, amortization ≥ 2.0), and the trie must
    // actually pay ≥2× vs one pointer per covered entity.
    let ratio = closure_pairs.len() as f64 / trie.covered() as f64;
    assert!(
        (ratio - target.amortization).abs() < 1e-6 && target.coverage >= 0.8,
        "KILL D-SRS-2 (b): detector claim (amort {:.2}x, cov {:.2}) != re-measure (amort {ratio:.2}x)",
        target.amortization,
        target.coverage
    );
    assert!(
        ratio >= 2.0,
        "KILL D-SRS-2 (b): amortization {ratio:.2}x < 2x — detector mis-routed"
    );
    println!(
        "D-SRS-2 PASS  '{}' ({:?}): trie ({} pointers) == closure ({} ancestor pairs) EXACTLY; \
         coverage {:.2}, amortization {ratio:.1}x (claim == re-measure); closure terminated uncapped \
         in {} passes → the materialization is DELETED (ancestry lives in the key)",
        target_word,
        target.recommend,
        trie.covered(),
        closure_pairs.len(),
        target.coverage,
        cg.passes
    );

    // ── D-SRS-3 — basin self-codes + the "where am I uncertain" self-report ──
    // The graph measures, from its OWN trained meaning codes, which subject
    // neighborhoods are diffuse — and is checked HELD-OUT (index-parity split-
    // half). Gate G-SRS3-1 (pre-registered before this code): Spearman ρ across
    // basins between the even-half width and the odd-half width; PASS ρ ≥ 0.35,
    // KILL ρ ≤ 0. Basin = a subject's outgoing-object neighborhood (the L1–L3
    // part_of:is_a rail), NEVER the routing basin-byte (routing ⟂ meaning).
    use std::collections::HashMap as Map;
    // Group base edges by subject → (predicate, object) pairs, then map objects
    // to their trained Cam96 codes (skip objects with no code — OOV can't happen
    // here since every id came from the coded vocab, but guard anyway).
    let mut edges_by_s: Map<u16, Vec<(u16, u16)>> = Map::new();
    for &t in &base {
        edges_by_s
            .entry(t.subject)
            .or_default()
            .push((t.predicate, t.object));
    }
    // Re-read the tiny (150 KB) codes artifact — codes[id] aligns with vocab id.
    let all_codes = load_cam96_codes(&data_file("cam96_codes.bin")).expect("codes artifact");
    let mut groups: Vec<(u16, Vec<deepnsm_v2::Cam96>)> = edges_by_s
        .iter()
        .map(|(&s, edges)| {
            let members: Vec<deepnsm_v2::Cam96> = edges
                .iter()
                .filter_map(|&(_p, o)| all_codes.get(o as usize).copied())
                .collect();
            (s, members)
        })
        .collect();
    // DETERMINISM: `edges_by_s` is a HashMap (randomized iteration order per
    // process), so the null shuffle's pool-concatenation order — and thus the
    // null ρ — would vary run-to-run, making the KILL assertion flaky. Sort by
    // subject id so the whole D-SRS-3 leg is reproducible.
    groups.sort_by_key(|(s, _)| *s);

    // The full self-report: rank basins by width (widest = least certain).
    let space = &nsm.space;
    let mut report: Vec<deepnsm_v2::BasinCode> = groups
        .iter()
        .filter_map(|(s, members)| {
            let edges = edges_by_s.get(s).map(Vec::as_slice).unwrap_or(&[]);
            deepnsm_v2::basin_self_code(space, *s, members, edges)
        })
        .collect();
    let max_width = report.iter().map(|b| b.width).fold(0.0f32, f32::max);
    report.sort_by(|a, b| {
        b.width
            .partial_cmp(&a.width)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!(
        "D-SRS-3 self-report: {} basins (subjects with ≥1 coded object). Most UNCERTAIN (widest) basins:",
        report.len()
    );
    for b in report.iter().filter(|b| b.members >= 6).take(5) {
        println!(
            "    '{}' — {} objects, width={:.1}, contradiction={:.2}, curiosity={:.2}",
            nsm.vocab.word(b.subject).unwrap_or("?"),
            b.members,
            b.width,
            b.contradiction,
            b.curiosity(max_width),
        );
    }
    for b in report.iter().rev().filter(|b| b.members >= 6).take(3) {
        println!(
            "    (most CERTAIN) '{}' — {} objects, width={:.1}, curiosity={:.2}",
            nsm.vocab.word(b.subject).unwrap_or("?"),
            b.members,
            b.width,
            b.curiosity(max_width),
        );
    }

    // A size-preserving label-shuffle null: destroy the basin↔code binding
    // (globally shuffle which codes fall in which basin, PRESERVING each basin's
    // size) via deterministic SplitMix64 Fisher-Yates (no rng/clock). Both gates
    // below re-run against this null to separate SEMANTIC signal from artifact.
    let null_groups = shuffle_null(&groups);

    // G-SRS3-1 — the registered raw split-half gate (floor 0.35). It PASSES raw
    // (ρ ≥ 0.35) but the null control reveals the pass is a member-count ARTIFACT:
    // the plug-in-centroid width is n-biased (E[width]≈σ²(1−1/n)), both halves of
    // one basin share n, so widths co-vary regardless of which codes they hold.
    let g1 = deepnsm_v2::heldout_split_gate(space, &groups, 6, 0.35);
    let g1_null = deepnsm_v2::heldout_split_gate(space, &null_groups, 6, 0.35);
    println!(
        "D-SRS-3 G-SRS3-1 (raw split-half): {} basins, real ρ={:.3} (floor 0.35), null ρ={:.3} — \
         separation {:.3} ⇒ CONFOUNDED (raw pass is a member-count artifact, not semantic)",
        g1.basins,
        g1.rho,
        g1_null.rho,
        g1.rho - g1_null.rho
    );

    // G-SRS3-2 — the CONSTANT-n gate (registered pre-run): fix n=k per half so the
    // n-artifact cannot inflate ρ, and gate on SEPARATION from the null.
    const K: usize = 5;
    let g2 = deepnsm_v2::heldout_constant_n_gate(space, &groups, K, 0.30);
    let g2_null = deepnsm_v2::heldout_constant_n_gate(space, &null_groups, K, 0.30);
    let sep = g2.rho - g2_null.rho;
    println!(
        "D-SRS-3 G-SRS3-2 (constant-n, k={K}): {} basins (≥{} members), real ρ={:.3}, null ρ={:.3} — separation {:.3}",
        g2.basins, g2.min_members, g2.rho, g2_null.rho, sep
    );
    // D-SRS-3 is a SCIENTIFIC falsifier: a KILL (no semantic signal) is a valid
    // FINDING, not a crash — it is REPORTED, never panicked (unlike the D-SRS-1/2/4
    // regression gates below). Deterministic now that `groups` is sorted.
    if g2.rho >= 0.30 && sep >= 0.20 {
        println!(
            "D-SRS-3 PASS (G-SRS3-2)  the width self-report is SEMANTIC and reliable out-of-sample \
             (constant-n real ρ {:.3} ≥ 0.30, separation {sep:.3} ≥ 0.20); the widths feed MUL as \
             competence=1−width/max (algebraic advantage, E-CAM96-REVIEW-CORRECTIONS-1)",
            g2.rho
        );
    } else if sep <= 0.05 {
        // Registered KILL: the falsifier FIRED. Report the negative honestly.
        println!(
            "D-SRS-3 KILL/FALSIFIED (G-SRS3-2)  constant-n separation {sep:.3} ≤ 0.05 — the width \
             self-report carries NO semantic content beyond the member-count artifact; the graph does \
             NOT know where it is uncertain from Cam96 code-spread. Conjecture falsified (not softened)."
        );
    } else {
        // Soft band 0.05 < sep < 0.20: honest, neither claimed PASS nor KILL.
        println!(
            "D-SRS-3 WEAK (G-SRS3-2)  constant-n separation {sep:.3} is positive but below the \
             registered 0.20 — a real but weak semantic self-signal; registration stands, no tuning"
        );
    }

    // ── EXPLORATORY (not a registered gate): Bessel-corrected all-member gate ──
    // Distinguishes "weak because underpowered" (constant-n k=5 discards evidence)
    // from "weak because no signal". Uses ALL members with an analytic n-bias
    // correction (×m/(m−1)); the null should still collapse to ≈0. Whatever it
    // shows, the pre-registered verdict remains G-SRS3-2's above — this only
    // diagnoses the WEAK result, it does not override it.
    let gb = deepnsm_v2::heldout_bessel_gate(space, &groups, 6, 0.30);
    let gb_null = deepnsm_v2::heldout_bessel_gate(space, &null_groups, 6, 0.30);
    println!(
        "D-SRS-3 EXPLORATORY (Bessel all-member): {} basins, real ρ={:.3}, null ρ={:.3} — separation {:.3} \
         (diagnoses power, does NOT change the registered G-SRS3-2 verdict)",
        gb.basins, gb.rho, gb_null.rho, gb.rho - gb_null.rho
    );

    // ── D-SRS-3b — the OPERATOR-CORRECTED evidence-composite instrument ──
    // D-SRS-3 failed because Cam96 code-spread is GEOMETRY with no evidence
    // semantics ("bullshit in, bullshit out"). The corrected instrument composes
    // NARS×frequency (u_conf) + contradiction density (u_contra) + rung-ladder
    // derived share (u_rung) — the evidence-bearing signals D-SRS-4 proved read
    // faithfully — and is gated FORWARD-predictively (G-SRS3b-1): first-half
    // uncertainty must predict second-half NOVELTY, vs a size-preserving null.
    let mid_v = (verses.len() / 2) as u64;
    // First-half distinct beliefs per subject (p,o → count); second-half occ list.
    let mut fh_beliefs: Map<u16, Map<(u16, u16), usize>> = Map::new();
    let mut sh_occ: Map<u16, Vec<(u16, u16)>> = Map::new();
    for &(v, t) in &all {
        if v < mid_v {
            *fh_beliefs
                .entry(t.subject)
                .or_default()
                .entry((t.predicate, t.object))
                .or_insert(0) += 1;
        } else {
            sh_occ
                .entry(t.subject)
                .or_default()
                .push((t.predicate, t.object));
        }
    }
    // Rung-ladder derived share per subject, from the FIRST-HALF arena (capped).
    let fh_base: Vec<Spo> = all
        .iter()
        .filter(|&&(v, _)| v < mid_v)
        .map(|&(_, t)| t)
        .collect();
    let fh_arena = deepnsm_v2::reason::DerivationArena::derive_transitive_capped(&fh_base, 50_000);
    let (mut tri_tot, mut tri_der): (Map<u16, usize>, Map<u16, usize>) = (Map::new(), Map::new());
    for d in fh_arena.entries() {
        *tri_tot.entry(d.triple.subject).or_insert(0) += 1;
        if d.rung >= 1 {
            *tri_der.entry(d.triple.subject).or_insert(0) += 1;
        }
    }
    // Eligible basins (≥4 distinct first-half beliefs AND ≥4 second-half occ),
    // in DETERMINISTIC subject order (the null's determinism depends on it).
    let mut subjects_e: Vec<u16> = fh_beliefs
        .keys()
        .copied()
        .filter(|s| {
            fh_beliefs.get(s).map_or(0, Map::len) >= 4 && sh_occ.get(s).map_or(0, Vec::len) >= 4
        })
        .collect();
    subjects_e.sort_unstable();
    let mut basin_beliefs: Vec<deepnsm_v2::evidence::BasinBeliefs> = Vec::new();
    let mut rungs: Vec<f32> = Vec::new();
    let mut novelty: Vec<f32> = Vec::new();
    let mut activity: Vec<f32> = Vec::new();
    for &s in &subjects_e {
        let bel: Vec<deepnsm_v2::evidence::BeliefRecord> = fh_beliefs[&s]
            .iter()
            .map(|(&(p, o), &n)| (p, o, n))
            .collect();
        let der_share = {
            let tot = *tri_tot.get(&s).unwrap_or(&0);
            if tot == 0 {
                0.0
            } else {
                *tri_der.get(&s).unwrap_or(&0) as f32 / tot as f32
            }
        };
        let fh_po: Vec<(u16, u16)> = fh_beliefs[&s].keys().copied().collect();
        activity.push(bel.iter().map(|&(_, _, n)| n as f32).sum());
        novelty.push(deepnsm_v2::novelty_rate(&fh_po, &sh_occ[&s]));
        rungs.push(der_share);
        basin_beliefs.push((s, bel));
    }
    // Real U per basin.
    let u_real: Vec<f32> = basin_beliefs
        .iter()
        .zip(&rungs)
        .filter_map(|((s, bel), &r)| {
            deepnsm_v2::evidence_basin(*s, bel, r).map(|e| e.uncertainty())
        })
        .collect();
    // Null: redeal belief records AND rung shares across basins (size-preserving).
    let null_beliefs = deepnsm_v2::shuffle_beliefs_null(&basin_beliefs);
    let null_rungs = deepnsm_v2::shuffle_rungs_null(&rungs);
    let u_null: Vec<f32> = null_beliefs
        .iter()
        .zip(&null_rungs)
        .filter_map(|((s, bel), &r)| {
            deepnsm_v2::evidence_basin(*s, bel, r).map(|e| e.uncertainty())
        })
        .collect();
    let fg = deepnsm_v2::forward_gate(&u_real, &u_null, &activity, &novelty);
    println!(
        "D-SRS-3b G-SRS3b-1 (evidence composite → forward novelty): {} basins, real ρ={:.3}, null ρ={:.3} \
         — separation {:.3}; frequency-only baseline ρ={:.3}",
        fg.basins, fg.real_rho, fg.null_rho, fg.separation(), fg.baseline_rho
    );
    // The KANBANSTEP DRIVE: the composite is not a printed number — it drives
    // the Rubicon lifecycle. Count how the evidence gate routes each basin from
    // Planning (Flow=explore-here / Hold=gather / Block=veto-thin-evidence).
    let (mut flow, mut hold, mut block) = (0u32, 0u32, 0u32);
    for ((s, bel), &r) in basin_beliefs.iter().zip(&rungs) {
        if let Some(e) = deepnsm_v2::evidence_basin(*s, bel, r) {
            match e.advance(lance_graph_contract::kanban::KanbanColumn::Planning) {
                Some(lance_graph_contract::kanban::KanbanColumn::CognitiveWork) => flow += 1,
                Some(lance_graph_contract::kanban::KanbanColumn::Prune) => block += 1,
                _ => hold += 1,
            }
        }
    }
    println!(
        "D-SRS-3b KANBANSTEP drive (Planning→): {flow} Flow (explore-here) · {hold} Hold (gather) · \
         {block} Block (veto thin/contradicted evidence) — the STEP is the trigger, not the report"
    );
    if fg.passed() {
        println!(
            "D-SRS-3b PASS (G-SRS3b-1)  evidence-composite uncertainty PREDICTS forward novelty \
             (real ρ {:.3} ≥ 0.25, separation {:.3} ≥ 0.15) — MUL competence=1−U is a REAL self-signal",
            fg.real_rho, fg.separation()
        );
    } else if fg.killed() {
        println!(
            "D-SRS-3b KILL/FALSIFIED (G-SRS3b-1)  separation {:.3} ≤ 0.05 — even the evidence composite \
             carries no forward-predictive signal beyond chance. Reported, not softened.",
            fg.separation()
        );
    } else {
        println!(
            "D-SRS-3b WEAK (G-SRS3b-1)  real ρ {:.3}, separation {:.3} — positive but below the registered \
             (0.25, 0.15); a real but weak evidence signal. Registration stands, no tuning.",
            fg.real_rho, fg.separation()
        );
    }

    // ── D-SRS-3b G-SRS3b-2 — the OPERATOR-CORRECTED TARGET: open-question YIELD ──
    // The negative G-SRS3b-1 ρ was the graph reporting doom-scroll / bad query:
    // raw novelty is not a question the rung ladder asks. The corrected forward
    // target is RESOLUTION — a first-half derived-but-unobserved triple is the
    // graph PREDICTING (A,p,C); does the second half OBSERVE it (text confirms)?
    // First-half base (p,o) per subject (to exclude already-observed).
    let mut fh_base_po: Map<u16, std::collections::HashSet<(u16, u16)>> = Map::new();
    for &t in &fh_base {
        fh_base_po
            .entry(t.subject)
            .or_default()
            .insert((t.predicate, t.object));
    }
    // Open questions per subject = first-half INFERENCES (rung≥1, not observed).
    let mut open_q: Map<u16, Vec<(u16, u16)>> = Map::new();
    for d in fh_arena.entries() {
        if d.rung >= 1 {
            let po = (d.triple.predicate, d.triple.object);
            let observed = fh_base_po
                .get(&d.triple.subject)
                .is_some_and(|s| s.contains(&po));
            if !observed {
                open_q.entry(d.triple.subject).or_default().push(po);
            }
        }
    }
    // Second-half DISTINCT base (p,o) per subject (the confirmations).
    let mut sh_base_po: Map<u16, Vec<(u16, u16)>> = Map::new();
    for (s, occ) in &sh_occ {
        let mut v: Vec<(u16, u16)> = occ.clone();
        v.sort_unstable();
        v.dedup();
        sh_base_po.insert(*s, v);
    }
    // Eligible: ≥4 open questions AND ≥4 second-half base occurrences.
    let mut subj_q: Vec<u16> = open_q
        .keys()
        .copied()
        .filter(|s| {
            open_q.get(s).map_or(0, Vec::len) >= 4 && sh_base_po.get(s).map_or(0, Vec::len) >= 4
        })
        .collect();
    subj_q.sort_unstable();
    let mut qb: Vec<deepnsm_v2::evidence::BasinBeliefs> = Vec::new();
    let mut q_rungs: Vec<f32> = Vec::new();
    let mut yield_v: Vec<f32> = Vec::new();
    let mut q_activity: Vec<f32> = Vec::new();
    for &s in &subj_q {
        // Reuse the first-half evidence for this subject (may be absent if the
        // subject had <1 first-half base belief — then skip, no composite).
        let Some(bel_map) = fh_beliefs.get(&s) else {
            continue;
        };
        let bel: Vec<deepnsm_v2::evidence::BeliefRecord> =
            bel_map.iter().map(|(&(p, o), &n)| (p, o, n)).collect();
        let der_share = {
            let tot = *tri_tot.get(&s).unwrap_or(&0);
            if tot == 0 {
                0.0
            } else {
                *tri_der.get(&s).unwrap_or(&0) as f32 / tot as f32
            }
        };
        let Some(y) = deepnsm_v2::open_question_yield(&open_q[&s], &sh_base_po[&s]) else {
            continue;
        };
        q_activity.push(open_q[&s].len() as f32);
        yield_v.push(y);
        q_rungs.push(der_share);
        qb.push((s, bel));
    }
    let uq_real: Vec<f32> = qb
        .iter()
        .zip(&q_rungs)
        .filter_map(|((s, b), &r)| deepnsm_v2::evidence_basin(*s, b, r).map(|e| e.uncertainty()))
        .collect();
    let qb_null = deepnsm_v2::shuffle_beliefs_null(&qb);
    let qr_null = deepnsm_v2::shuffle_rungs_null(&q_rungs);
    let uq_null: Vec<f32> = qb_null
        .iter()
        .zip(&qr_null)
        .filter_map(|((s, b), &r)| deepnsm_v2::evidence_basin(*s, b, r).map(|e| e.uncertainty()))
        .collect();
    let fg2 = deepnsm_v2::forward_gate(&uq_real, &uq_null, &q_activity, &yield_v);
    println!(
        "D-SRS-3b G-SRS3b-2 (evidence composite → open-question YIELD): {} basins, real ρ={:.3}, \
         null ρ={:.3} — separation {:.3}; #open-questions baseline ρ={:.3}",
        fg2.basins,
        fg2.real_rho,
        fg2.null_rho,
        fg2.separation(),
        fg2.baseline_rho
    );
    if fg2.real_rho >= 0.25 && fg2.separation() >= 0.15 {
        println!(
            "D-SRS-3b PASS (G-SRS3b-2)  evidence-composite uncertainty PREDICTS open-question yield \
             (real ρ {:.3} ≥ 0.25, sep {:.3} ≥ 0.15) — uncertainty points to PRODUCTIVE exploration; \
             the rung-ladder-relevant target works where raw novelty (G-SRS3b-1) did not",
            fg2.real_rho,
            fg2.separation()
        );
    } else if fg2.separation().abs() >= 0.15 && fg2.real_rho <= -0.25 {
        println!(
            "D-SRS-3b DEAD-END DETECTOR (G-SRS3b-2)  real ρ {:.3} (negative) SEPARATES from null \
             (sep {:.3}) — uncertainty reliably points where questions do NOT resolve: a validated \
             doom-scroll/dead-end signal (inverted use), not a productive-exploration signal",
            fg2.real_rho,
            fg2.separation()
        );
    } else if fg2.killed() {
        println!(
            "D-SRS-3b KILL (G-SRS3b-2)  separation {:.3} ≤ 0.05 — even the rung-ladder-relevant target \
             is coverage-driven; the composite carries no question-resolution signal. Reported, not softened.",
            fg2.separation()
        );
    } else {
        println!(
            "D-SRS-3b WEAK (G-SRS3b-2)  real ρ {:.3}, separation {:.3} — below the registered (0.25, 0.15). \
             Registration stands, no tuning.",
            fg2.real_rho,
            fg2.separation()
        );
    }

    // G-SRS3b-3 — the TERMINAL test: does the composite predict yield BEYOND
    // size? Partial-correlate U and yield after rank-residualizing both on the
    // dominating size covariate (#open-questions = q_activity).
    let p_real = deepnsm_v2::partial_spearman(&uq_real, &yield_v, &q_activity);
    let p_null = deepnsm_v2::partial_spearman(&uq_null, &yield_v, &q_activity);
    let p_sep = p_real - p_null;
    println!(
        "D-SRS-3b G-SRS3b-3 (partial ρ(U, yield | size)): real={:.3}, null={:.3} — separation {:.3}",
        p_real, p_null, p_sep
    );
    if p_real >= 0.15 && p_sep >= 0.10 {
        println!(
            "D-SRS-3b PASS (G-SRS3b-3)  the composite predicts question-resolution EVEN AFTER size is \
             removed (partial ρ {:.3} ≥ 0.15, sep {:.3} ≥ 0.10) — the NARS×contra×rung composition \
             carries genuine information beyond a count. The operator's instrument is vindicated.",
            p_real, p_sep
        );
    } else if p_sep <= 0.05 {
        println!(
            "D-SRS-3b KILL (G-SRS3b-3)  with size partialled out, separation {:.3} ≤ 0.05 — the \
             composite IS its size baseline; the NARS×contra×rung composition adds nothing beyond a \
             count across basins. (Its validated home is the per-basin KANBANSTEP drive, not a \
             cross-basin correlation.) Reported, not softened.",
            p_sep
        );
    } else {
        println!(
            "D-SRS-3b WEAK (G-SRS3b-3)  partial ρ {:.3}, sep {:.3} — a faint size-independent residual, \
             below the registered (0.15, 0.10). Registration stands, no tuning.",
            p_real, p_sep
        );
    }

    // ── D-SRS-4 — the self-reference falsifier: the graph answers questions ──
    // about its OWN reasoning, checked against an INDEPENDENT recount.
    // G-SRS4-1 (provenance): every derived triple's stored premises must
    // re-compose to it (strictly stronger than D-SRS-1 resolvability).
    let prov = deepnsm_v2::provenance_check(&arena);
    assert!(
        prov.passed(),
        "KILL D-SRS-4 (G-SRS4-1): {}/{} derived triples do NOT re-compose from their stored premises \
         — the provenance the graph reports about its own reasoning is false",
        prov.derived - prov.composes,
        prov.derived
    );
    println!(
        "D-SRS-4 PASS (G-SRS4-1 provenance): all {} derived triples independently re-compose from \
         their premise pointers ((A,p,B)+(B,p,C) ⇒ (A,p,C), shared pivot) — self-reported provenance is FAITHFUL",
        prov.composes
    );

    // G-SRS4-2 (confidence-delta): NARS confidence in the most-frequent belief,
    // read THROUGH the graph's own version-range window, must equal a direct
    // recount over the raw stream, and must strictly rise as the belief recurs.
    let (y, v1, v2) = deepnsm_v2::most_frequent_belief(&all).expect("non-empty KG");
    let self_ans = deepnsm_v2::confidence_delta_self(&stream, y, v1, v2, 1);
    let truth = deepnsm_v2::confidence_delta_recount(&all, y, v1, v2, 1);
    assert_eq!(
        self_ans, truth,
        "KILL D-SRS-4 (G-SRS4-2): windowed self-read {self_ans:?} != independent recount {truth:?} \
         — the self-reference read is not faithful"
    );
    assert!(
        self_ans.delta > 0.0,
        "KILL D-SRS-4 (G-SRS4-2): confidence in a recurring belief did not rise (delta={})",
        self_ans.delta
    );
    println!(
        "D-SRS-4 PASS (G-SRS4-2 confidence-delta): belief '{} {} {}' — n(≤v{v1})={}, n(≤v{v2})={}; \
         NARS confidence {:.3}→{:.3} (Δ +{:.3}); windowed self-read == independent recount EXACTLY",
        nsm.vocab.word(y.subject).unwrap_or("?"),
        nsm.vocab.word(y.predicate).unwrap_or("?"),
        nsm.vocab.word(y.object).unwrap_or("?"),
        self_ans.n1,
        self_ans.n2,
        self_ans.c1,
        self_ans.c2,
        self_ans.delta
    );

    println!(
        "\nSTRUCTURAL GATES GREEN (G1–G4, D-SRS-1, D-SRS-2, D-SRS-4) — the whole book is resident, \
         literally read, with real meaning codes, reasoning about its own derivations, routing its own \
         representations by shape, and answering FAITHFULLY questions about its own reasoning \
         (provenance + confidence-delta, each == an independent recount).\nD-SRS-3 FALSIFIER RAN \
         (null-controlled): the width self-report is a MEMBER-COUNT ARTIFACT — once n is fixed \
         (constant-n) or bias-corrected (Bessel) the semantic separation collapses to ≈0. The graph \
         does NOT reliably know where it is uncertain from Cam96 code-spread alone; the D-SRS-3 \
         conjecture is NOT confirmed (an honest negative)."
    );
}

/// Size-preserving label-shuffle null: pool every basin's member codes, shuffle
/// the pool deterministically (SplitMix64 Fisher-Yates — no rng/clock), then
/// re-chunk into basins of the ORIGINAL sizes. Destroys the basin↔code binding
/// while holding member counts fixed — the control that separates a semantic
/// width signal from a member-count artifact.
fn shuffle_null(groups: &[(u16, Vec<deepnsm_v2::Cam96>)]) -> Vec<(u16, Vec<deepnsm_v2::Cam96>)> {
    let mut pool: Vec<deepnsm_v2::Cam96> = groups.iter().flat_map(|(_, m)| m.clone()).collect();
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..pool.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        pool.swap(i, j);
    }
    let mut off = 0usize;
    groups
        .iter()
        .map(|(s, m)| {
            let g = pool[off..off + m.len()].to_vec();
            off += m.len();
            (*s, g)
        })
        .collect()
}

/// COCA PoS letter → [`Pos`], through the crate's one canonical fold
/// ([`deepnsm_v2::coca`]). The local copy that lived here is gone.
fn coca_pos(letter: &str) -> Pos {
    fsm_pos_tag(letter)
}

/// Early-modern forms COCA does not carry. The explicit list is load-bearing:
/// `thou`/`hath`/`shall`/`saith` are among the corpus's most frequent tokens and
/// none matches the `-eth`/`-est` suffix rule.
fn archaic_pos(w: &str) -> Option<Pos> {
    match w {
        "thou" | "thee" | "ye" => Some(Pos::Noun),
        "thy" | "thine" => Some(Pos::Det),
        "shalt" | "hath" | "doth" | "saith" | "spake" | "begat" | "art" | "wilt" | "hast"
        | "shall" | "cometh" | "wast" => Some(Pos::Verb),
        "unto" | "thereof" | "wherefore" | "verily" | "yea" | "lo" => Some(Pos::Other),
        _ => {
            if w.ends_with("eth") || w.ends_with("est") {
                Some(Pos::Verb)
            } else {
                None
            }
        }
    }
}

/// Ascii letters only, lowercased; `None` under two letters.
fn normalise(tok: &str) -> Option<String> {
    let w: String = tok
        .chars()
        .filter(char::is_ascii_alphabetic)
        .collect::<String>()
        .to_lowercase();
    (w.len() >= 2).then_some(w)
}

/// Where a word's reading concentration sits in the population (D-LXC-11).
///
/// A MEASUREMENT, not a decision. The statistic is the share of the most
/// frequent observed reading, `coverage(id)[0]`, never a summed parser-state
/// share. The cut points are quartiles of the population, so each band holds a
/// known part of it; a band is therefore relative to the loaded vocabulary.
/// Shares of rare and common words weigh the same, so a band says where a
/// word's concentration sits, not how much to trust it and not which reading
/// holds. The labels are report vocabulary for this example only: nothing
/// reads a band to select, rank or eliminate a reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoverageBand {
    /// Share at or above the upper cut.
    Decisive,
    /// Share between the cuts.
    Leaning,
    /// Share below the lower cut.
    Contested,
}

/// The two cut points, calibrated once at load, and the population they came
/// from. A band means nothing without its cuts, so they travel together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BandCuts {
    lo: u8,
    hi: u8,
    population: usize,
}

/// ndarray `hpc::rolling_floor::rank_per_10000`: index `p·len/10000`
/// (integer division) clamped to the last index, `0` for an empty sample.
/// Restated here because this crate has no ndarray dependency. It is not
/// nearest-rank: the two differ when `p·len/10000` is a whole number.
fn rank_per_10000(len: usize, per_10000: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (per_10000 * len / 10_000).min(len - 1)
}

/// The quartile cuts of an ascending list of shares; `None` if it is empty.
fn cuts_from_sorted(shares: &[u8]) -> Option<BandCuts> {
    if shares.is_empty() {
        return None;
    }
    Some(BandCuts {
        lo: shares[rank_per_10000(shares.len(), 2500)],
        hi: shares[rank_per_10000(shares.len(), 7500)],
        population: shares.len(),
    })
}

/// The band of one share under given cuts.
fn band_of(share: u8, cuts: BandCuts) -> CoverageBand {
    if share < cuts.lo {
        CoverageBand::Contested
    } else if share >= cuts.hi {
        CoverageBand::Decisive
    } else {
        CoverageBand::Leaning
    }
}

/// The share that places word `id` in the band population, or `None` if the
/// word is outside it: a lemma-table key (the register is never read for it),
/// a word with unknown coverage, or a word whose readings fold to one parser
/// state (nothing to decide between). Reads the register only, never the
/// tagger's archaic/`Other` fallback.
fn band_share(
    lemmas: &HashMap<String, Pos>,
    evidence: &LexicalEvidence,
    vocab: &PaletteVocab,
    id: WordId,
) -> Option<u8> {
    if lemmas.contains_key(vocab.word(id)?) {
        return None;
    }
    let share = evidence.coverage(id).first().copied().flatten()?;
    let mut states = evidence.readings(id).iter().map(|r| fsm_pos(r.pos));
    let first = states.next()?;
    states.any(|s| s != first).then_some(share)
}

/// Calibrate the cuts over every vocabulary word and band each one.
fn calibrate(
    lemmas: &HashMap<String, Pos>,
    evidence: &LexicalEvidence,
    vocab: &PaletteVocab,
) -> (Option<BandCuts>, Vec<Option<CoverageBand>>) {
    let shares: Vec<Option<u8>> = (0..vocab.len())
        .map(|i| {
            WordId::try_from(i)
                .ok()
                .and_then(|id| band_share(lemmas, evidence, vocab, id))
        })
        .collect();
    let mut sorted: Vec<u8> = shares.iter().flatten().copied().collect();
    sorted.sort_unstable();
    let cuts = cuts_from_sorted(&sorted);
    let bands = shares.iter().map(|s| Some(band_of((*s)?, cuts?))).collect();
    (cuts, bands)
}

/// Lowercase the `word` column of `word_forms.csv`, leaving the header and the
/// other fields as they are. `load_word_forms_csv` matches surfaces exactly and
/// the corpus tokens are lowercased; three COCA surfaces (`True`, `False`,
/// `reElection`) would otherwise never route.
fn lowercase_word_column(forms_csv: &str) -> String {
    let mut out = String::with_capacity(forms_csv.len());
    for (i, line) in forms_csv.lines().enumerate() {
        if i > 0 {
            if let Some((head, word)) = line.rsplit_once(',') {
                out.push_str(head);
                out.push(',');
                out.push_str(&word.to_lowercase());
                out.push('\n');
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// The corpus tagger, plus the lexical evidence and its calibration.
///
/// The tag ([`Tagger::pos`]) is the LEGACY single-`Pos` tagging `main` had
/// before D-LXC-1, unchanged ([`load_pos_legacy_first_wins`]). The parser takes
/// one `Pos` per token, so a word with several readings cannot reach it
/// intact; this boundary is inherited debt kept for compatibility, not a
/// semantic resolution. The evidence and the bands are measurements beside it:
/// no count, order or band is read to produce a tag.
struct Tagger {
    lemmas: HashMap<String, Pos>,
    /// Every lemma-table row per lemma, folded: the table's readings, not
    /// only its first row.
    lemma_readings: HashMap<String, PosSet>,
    /// `main`'s tagging, kept verbatim: lemma table, then first
    /// `word_forms.csv` row. See [`load_pos_legacy_first_wins`].
    legacy: HashMap<String, Pos>,
    evidence: LexicalEvidence,
    report: WordFormsReport,
    /// D-LXC-11 cuts, `None` when the band population is empty.
    cuts: Option<BandCuts>,
    /// D-LXC-11 band per `WordId`. Reported only; `pos` does not read it.
    bands: Vec<Option<CoverageBand>>,
}

impl Tagger {
    fn load(
        lemmas_csv: &str,
        forms_csv: &str,
        vocab: &PaletteVocab,
    ) -> Result<Self, EvidenceError> {
        let mut lemmas = HashMap::new();
        let mut lemma_readings: HashMap<String, PosSet> = HashMap::new();
        for line in lemmas_csv.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            let (Some(lemma), Some(pos)) = (f.get(1), f.get(2)) else {
                continue;
            };
            lemmas
                .entry(lemma.to_lowercase())
                .or_insert_with(|| coca_pos(pos));
            let set = lemma_readings.entry(lemma.to_lowercase()).or_default();
            *set = set.with(coca_pos(pos));
        }
        let (evidence, report) = load_word_forms_csv(&lowercase_word_column(forms_csv), vocab)?;
        let (cuts, bands) = calibrate(&lemmas, &evidence, vocab);
        Ok(Self {
            lemmas,
            lemma_readings,
            legacy: load_pos_legacy_first_wins(lemmas_csv, forms_csv),
            evidence,
            report,
            cuts,
            bands,
        })
    }

    /// The LEGACY tag for word `w`: [`load_pos_legacy_first_wins`], then
    /// [`archaic_pos`], then [`Pos::Other`] — exactly `main`'s tagging.
    ///
    /// Compatibility boundary, not semantic resolution. It reads no count and
    /// no [`LexicalEvidence`], so frequency cannot change a tag. It does
    /// depend on source-row order (the first lemma row, the first form row);
    /// that is inherited debt, not an authorized resolver. [`Self::readings`]
    /// is the resolver (D-LXC-2, D-LXC-13).
    fn pos(&self, w: &str) -> Pos {
        self.legacy
            .get(w)
            .copied()
            .or_else(|| archaic_pos(w))
            .unwrap_or(Pos::Other)
    }

    /// Every reading word `w` (routing id `id`) enters the parser with
    /// (D-LXC-2). Same sources and order as [`Self::pos`], but the forms
    /// layer hands over its whole reading set instead of its first row:
    ///
    /// 1. the lemma table's first row (F9). If that tag is a noun or a verb,
    ///    the word also gets its OTHER noun/verb reading when any lemma row
    ///    or [`LexicalEvidence`] reading has it (D-LXC-13): the predicate
    ///    alternative is kept, and the parser's slot rule picks it by
    ///    position. Function words keep their one tag;
    /// 2. otherwise every [`LexicalEvidence`] reading, folded by
    ///    [`deepnsm_v2::coca`];
    /// 3. the archaic list, one reading;
    /// 4. otherwise [`PosSet::EMPTY`]: unknown, never a guessed reading.
    ///
    /// No count is read. The legacy tag is always one of the readings.
    fn readings(&self, w: &str, id: WordId) -> PosSet {
        if let Some(&p) = self.lemmas.get(w) {
            return predicate_alternatives(
                p,
                self.lemma_readings
                    .get(w)
                    .copied()
                    .unwrap_or_default()
                    .union(reading_set(&self.evidence, id).unwrap_or_default()),
            );
        }
        reading_set(&self.evidence, id)
            .or_else(|| archaic_pos(w).map(PosSet::single))
            .unwrap_or(PosSet::EMPTY)
    }
}

/// LEGACY tagging, verbatim from `main`'s `load_pos`: the lemma table, then the
/// FIRST `word_forms.csv` row per surface, both first row wins. Row order is
/// not an authorized lexical decision; this is kept only because the parser
/// takes one `Pos` per token (see [`Tagger::pos`]).
fn load_pos_legacy_first_wins(lemmas_csv: &str, forms_csv: &str) -> HashMap<String, Pos> {
    let mut m: HashMap<String, Pos> = HashMap::new();
    for line in lemmas_csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let (Some(lemma), Some(pos)) = (f.get(1), f.get(2)) else {
            continue;
        };
        m.entry(lemma.to_lowercase())
            .or_insert_with(|| coca_pos(pos));
    }
    for line in forms_csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let (Some(pos), Some(word)) = (f.get(2), f.get(5)) else {
            continue;
        };
        m.entry(word.to_lowercase())
            .or_insert_with(|| coca_pos(pos));
    }
    m
}

// D-LXC-1 tests. They run under `cargo test` because `Cargo.toml` declares this
// example with `test = true`; cargo never runs an example's `main()`, so the
// KJV itself is not needed here.
#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(words: &[&str]) -> PaletteVocab {
        let mut v = PaletteVocab::new();
        v.from_frequency_ranked(words.iter().copied());
        v
    }

    fn committed(name: &str) -> String {
        lexicon_file(name)
    }

    const NO_LEMMAS: &str = "rank,lemma,PoS\n";

    fn forms(rows: &str) -> String {
        format!("lemRank,lemma,PoS,lemFreq,wordFreq,word\n{rows}")
    }

    fn tag(forms_csv: &str, w: &str) -> Pos {
        let v = vocab(&[w]);
        let t = Tagger::load(NO_LEMMAS, forms_csv, &v).unwrap();
        t.pos(w)
    }

    // (a) a homograph keeps every reading
    #[test]
    fn record_keeps_its_noun_and_verb_readings() {
        let v = vocab(&["record"]);
        let t = Tagger::load(NO_LEMMAS, &committed("word_forms.csv"), &v).unwrap();
        let tags: Vec<char> = t
            .evidence
            .readings(v.id("record").unwrap())
            .iter()
            .map(|r| r.pos.as_char())
            .collect();
        assert!(tags.contains(&'n') && tags.contains(&'v'), "{tags:?}");
    }

    // (f) the LEGACY fallback order, pinned as inherited debt, not endorsed:
    // legacy map (a form row beats archaic, D-LXC-9), then archaic, then Other.
    #[test]
    fn legacy_tagging_falls_through_to_archaic_then_other() {
        let f = forms("1,art,n,5,,art\n");
        let v = vocab(&["art", "hath", "zz"]);
        let t = Tagger::load(NO_LEMMAS, &f, &v).unwrap();
        assert_eq!(t.pos("art"), Pos::Noun);
        assert_eq!(t.pos("hath"), Pos::Verb);
        assert_eq!(t.pos("zz"), Pos::Other);
    }

    // (g) stay-silent: one reading keeps its tag and covers 100%
    #[test]
    fn a_single_reading_keeps_its_tag() {
        assert_eq!(tag(&forms("1,x,j,9,3,w\n"), "w"), Pos::Adj);
        assert_eq!(tag(&forms("1,x,r,9,3,w\n"), "w"), Pos::Adv);
    }

    // (h) + G7: the lemma table is never overruled by the forms layer
    #[test]
    fn the_forms_layer_never_retags_a_lemma_table_word() {
        let v = vocab(&["work"]);
        let t = Tagger::load(
            &committed("lemmas_5k.csv"),
            &committed("word_forms.csv"),
            &v,
        )
        .unwrap();
        let id = v.id("work").unwrap();
        // The most frequent observed reading is the noun ...
        assert_eq!(t.evidence.readings(id)[0].pos.as_char(), 'n');
        // ... and the tag is still the lemma table's.
        assert_eq!(t.pos("work"), Pos::Verb);

        // The same property on a conflicting row, with the row proven loaded.
        let conflicting = forms("1,x,n,9,4,create\n");
        let v = vocab(&["create"]);
        let t = Tagger::load("rank,lemma,PoS\n1,create,v\n", &conflicting, &v).unwrap();
        assert_eq!(t.evidence.reading_count(), 1);
        assert_eq!(t.pos("create"), Pos::Verb);
    }

    // (i) a capitalised COCA surface still routes
    #[test]
    fn a_capitalised_surface_routes_after_lowercasing() {
        let f = forms("7,true,j,9,4,True\n");
        let v = vocab(&["true"]);
        let t = Tagger::load(NO_LEMMAS, &f, &v).unwrap();
        assert_eq!(t.report.unrouted, 0);
        assert_eq!(t.pos("true"), Pos::Adj);
        // Without the lowercasing the same row is not routed.
        let (_, raw) = load_word_forms_csv(&f, &v).unwrap();
        assert_eq!(raw.unrouted, 1);
    }

    // Frequency is evidence, not a lexical decision. Changing only the counts,
    // with the same rows in the same order, MAY change the evidence order and
    // coverage; it MUST NOT change the reading set or the tag.
    #[test]
    fn counts_change_evidence_never_the_readings_or_the_tag() {
        let v = vocab(&["w"]);
        let id = v.id("w").unwrap();
        let load = |rows: &str| Tagger::load(NO_LEMMAS, &forms(rows), &v).unwrap();
        // `changes`' real counts (verb row first), then swapped, then even.
        let real = load("1,x,v,9,13624,w\n2,y,n,9,113085,w\n");
        let swapped = load("1,x,v,9,113085,w\n2,y,n,9,13624,w\n");
        let even = load("1,x,v,9,50,w\n2,y,n,9,50,w\n");
        let set = |t: &Tagger| {
            let mut s: Vec<_> = t
                .evidence
                .readings(id)
                .iter()
                .map(|r| (r.pos, r.lemma))
                .collect();
            s.sort();
            s
        };
        // MAY change: the evidence order and the coverage.
        assert_ne!(
            real.evidence.readings(id)[0].pos,
            swapped.evidence.readings(id)[0].pos
        );
        assert_eq!(real.evidence.coverage(id), &[Some(89), Some(100)]);
        assert_eq!(even.evidence.coverage(id), &[Some(50), Some(100)]);
        // MUST NOT change: the reading set and the tag.
        for t in [&swapped, &even] {
            assert_eq!(set(t), set(&real));
            assert_eq!(t.pos("w"), real.pos("w"));
        }
    }

    // ── D-LXC-11 coverage bands ──

    /// One two-state word per share: `n` carries `share` of 100, `v` the rest.
    fn two_state_rows(words: &[(&str, u64)]) -> String {
        let mut rows = String::new();
        for (i, (w, share)) in words.iter().enumerate() {
            let k = 2 * i + 1;
            rows.push_str(&format!("{k},{w},n,9,{share},{w}\n"));
            rows.push_str(&format!("{},{w},v,9,{},{w}\n", k + 1, 100 - share));
        }
        forms(&rows)
    }

    fn load_words(lemmas_csv: &str, forms_csv: &str, words: &[&str]) -> (PaletteVocab, Tagger) {
        let v = vocab(words);
        let t = Tagger::load(lemmas_csv, forms_csv, &v).unwrap();
        (v, t)
    }

    fn band(t: &Tagger, v: &PaletteVocab, w: &str) -> Option<CoverageBand> {
        t.bands[usize::from(v.id(w).unwrap())]
    }

    const SPREAD: [(&str, u64); 8] = [
        ("wa", 50),
        ("wb", 60),
        ("wc", 70),
        ("wd", 80),
        ("we", 90),
        ("wf", 95),
        ("wg", 98),
        ("wh", 99),
    ];

    // T1 the cuts come from the population, not from constants
    #[test]
    fn cuts_move_with_the_population() {
        let low = [50u8, 55, 60, 65, 70, 75, 80, 85];
        let high = low.map(|s| s + 10);
        let (a, b) = (
            cuts_from_sorted(&low).unwrap(),
            cuts_from_sorted(&high).unwrap(),
        );
        assert!(b.lo > a.lo && b.hi > a.hi, "{a:?} -> {b:?}");
    }

    // T2 a mixed population has both Contested and Decisive words
    #[test]
    fn a_mixed_population_has_both_ends() {
        let words: Vec<&str> = SPREAD.iter().map(|(w, _)| *w).collect();
        let (v, t) = load_words(NO_LEMMAS, &two_state_rows(&SPREAD), &words);
        assert_eq!(t.cuts.unwrap().population, 8);
        assert_eq!(band(&t, &v, "wa"), Some(CoverageBand::Contested));
        assert_eq!(band(&t, &v, "wh"), Some(CoverageBand::Decisive));
    }

    // T3 each exclusion keeps its word out of the population
    #[test]
    fn excluded_words_get_no_band() {
        let mut rows = two_state_rows(&SPREAD);
        rows.push_str("20,lx,n,9,60,lx\n21,lx,v,9,40,lx\n"); // lemma-table key
        rows.push_str("22,one,n,9,60,one\n23,one,p,9,40,one\n"); // n+p: one state
        rows.push_str("24,unk,n,9,60,unk\n25,unk,v,9,,unk\n"); // unknown count
        let mut words: Vec<&str> = SPREAD.iter().map(|(w, _)| *w).collect();
        words.extend(["lx", "one", "unk"]);
        let (v, t) = load_words("rank,lemma,PoS\n1,lx,v\n", &rows, &words);
        assert_eq!(t.cuts.unwrap().population, 8, "only the SPREAD words count");
        for w in ["lx", "one", "unk"] {
            assert_eq!(band(&t, &v, w), None, "{w}");
        }
        // Anti-vacuity: each excluded word has readings that were loaded.
        for w in ["lx", "one", "unk"] {
            assert_eq!(t.evidence.readings(v.id(w).unwrap()).len(), 2, "{w}");
        }
    }

    // T4 no population gives no cuts; one word does not panic
    #[test]
    fn empty_and_single_populations() {
        let (_, t) = load_words(NO_LEMMAS, &forms("1,x,n,9,5,w\n"), &["w"]);
        assert_eq!(t.cuts, None);
        assert_eq!(t.bands, vec![None]);
        let (v, t) = load_words(NO_LEMMAS, &two_state_rows(&[("w", 70)]), &["w"]);
        assert_eq!(
            t.cuts,
            Some(BandCuts {
                lo: 70,
                hi: 70,
                population: 1
            })
        );
        assert_eq!(band(&t, &v, "w"), Some(CoverageBand::Decisive));
    }

    // T5 the boundaries, on explicit cuts
    #[test]
    fn band_boundaries() {
        let c = BandCuts {
            lo: 50,
            hi: 80,
            population: 0,
        };
        assert_eq!(band_of(49, c), CoverageBand::Contested);
        assert_eq!(band_of(50, c), CoverageBand::Leaning);
        assert_eq!(band_of(79, c), CoverageBand::Leaning);
        assert_eq!(band_of(80, c), CoverageBand::Decisive);
    }

    // T6 can stay silent: identical shares mark nothing Contested
    #[test]
    fn identical_shares_are_all_decisive() {
        let c = cuts_from_sorted(&[70; 8]).unwrap();
        assert_eq!((c.lo, c.hi), (70, 70));
        assert_eq!(band_of(70, c), CoverageBand::Decisive);
    }

    // T7 the key is the most frequent READING's share, never a state sum
    #[test]
    fn the_key_is_reading_share() {
        let f = forms("1,x,n,9,40,w\n2,y,p,9,35,w\n3,z,v,9,25,w\n");
        let (v, t) = load_words(NO_LEMMAS, &f, &["w"]);
        let id = v.id("w").unwrap();
        // n 40 + p 35 would be Noun 75 if summed.
        assert_eq!(band_share(&t.lemmas, &t.evidence, &v, id), Some(40));
    }

    // T8 the rank rule is ndarray's rank_per_10000, not nearest-rank
    #[test]
    fn rank_rule_is_rank_per_10000() {
        assert_eq!(rank_per_10000(4, 2500), 1); // nearest-rank: ceil(1) - 1 = 0
        assert_eq!(rank_per_10000(141, 2500), 35);
        assert_eq!(rank_per_10000(141, 7500), 105);
        assert_eq!(rank_per_10000(0, 2500), 0);
        assert_eq!(rank_per_10000(1, 7500), 0);
    }
}
