//! `ud_pos_eval` — the position rules of [`deepnsm_v2::fsm::parse_readings`]
//! scored against GOLD part-of-speech tags from a Universal Dependencies
//! treebank, in any language UD covers.
//!
//! The lexicon is taken from the treebank's own train split: a form's
//! readings are every tag it carries anywhere in train, folded onto the FSM
//! alphabet by one table ([`fold`]). Nothing in this file is written for a
//! particular language. Test sentences are parsed with those readings, and
//! every token whose readings include both halves of a pair (noun/verb,
//! adjective/adverb) and whose gold tag is one of the two is scored:
//!
//! - **position** — the pair was narrowed to one reading by a parser rule;
//!   right when that reading is the gold tag;
//! - **frequency** — the reading the form carries most often in train (the
//!   "70 % noun, so noun" pick), on the same tokens.
//!
//! Usage (inputs are fetched, never committed — CC BY-SA 4.0):
//!
//! ```text
//! T=https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15
//! curl -O $T/en_ewt-ud-train.conllu -O $T/en_ewt-ud-test.conllu
//! cargo run --release --example ud_pos_eval -- en_ewt-ud-train.conllu en_ewt-ud-test.conllu
//! # COCA readings and COCA's frequency pick instead of the treebank's own:
//! cargo run --release --example ud_pos_eval -- TRAIN TEST --coca ../deepnsm/word_frequency
//! ```
//!
//! The typology is measured from train; `UD_TYPOLOGY=english` uses
//! [`Typology::ENGLISH`] instead. `UD_DUMP=1` prints every token a rule
//! narrowed away from its gold tag, with context, to stderr.

use std::collections::HashMap;

use deepnsm_v2::coca::fsm_pos_tag;
use deepnsm_v2::fsm::{
    attribute_rule, parse_readings_with, AdjectiveOrder, AttributeRule, Pos, PosSet, Reading,
    Typology,
};

/// One syntactic word of a UD sentence.
struct Word {
    form: String,
    gold: Pos,
    /// The adjective modifies a noun (`amod`) that stands after it
    /// (`Some(true)`) or before it (`Some(false)`); `None` otherwise.
    amod_head_after: Option<bool>,
}

/// The one UPOS → FSM fold. Language-neutral: it reads only UD's universal
/// tags and the universal `PronType=Rel` feature.
fn fold(upos: &str, feats: &str, form: &str) -> Pos {
    let rel = feats.split('|').any(|f| f == "PronType=Rel");
    match upos {
        "PRON" | "DET" if rel => Pos::Rel,
        "NOUN" | "PROPN" | "PRON" => Pos::Noun,
        "VERB" | "AUX" => Pos::Verb,
        "ADJ" => Pos::Adj,
        "ADV" => Pos::Adv,
        "DET" => Pos::Det,
        "PUNCT" if matches!(form, "." | "!" | "?") => Pos::Stop,
        _ => Pos::Other,
    }
}

/// Sentences of a CoNLL-U file. Multiword-token ranges (`3-4`) and empty
/// nodes (`5.1`) are skipped: the syntactic words carry the tags.
fn read_conllu(path: &str) -> Vec<Vec<Word>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 6 || cols[0].contains('-') || cols[0].contains('.') {
            continue;
        }
        let form = cols[1].to_lowercase();
        let gold = fold(cols[3], cols[5], &form);
        let amod_head_after = match (cols.get(6), cols.get(7)) {
            (Some(head), Some(rel)) if rel.starts_with("amod") => {
                match (head.parse::<usize>(), cols[0].parse::<usize>()) {
                    (Ok(h), Ok(i)) if h > 0 => Some(h > i),
                    _ => None,
                }
            }
            _ => None,
        };
        cur.push(Word {
            form,
            gold,
            amod_head_after,
        });
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Readings and per-reading counts for every train form.
struct Lexicon {
    readings: HashMap<String, PosSet>,
    counts: HashMap<(String, u8), usize>,
}

impl Lexicon {
    fn from_train(sentences: &[Vec<Word>]) -> Self {
        let mut readings: HashMap<String, PosSet> = HashMap::new();
        let mut counts = HashMap::new();
        for w in sentences.iter().flatten() {
            if w.gold == Pos::Stop {
                continue;
            }
            let set = readings.entry(w.form.clone()).or_default();
            *set = set.with(w.gold);
            *counts.entry((w.form.clone(), w.gold as u8)).or_default() += 1;
        }
        Self { readings, counts }
    }

    /// Readings and counts from COCA (`lemmas_5k.csv` + `word_forms.csv` in
    /// `dir`), folded by [`deepnsm_v2::coca::fsm_pos_tag`] — the lexicon
    /// `bible_wave` reads, so the frequency pick is COCA's, not the
    /// treebank's own. Counts are `word_forms.csv`'s per-form `wordFreq`.
    fn from_coca(dir: &str) -> Self {
        let mut readings: HashMap<String, PosSet> = HashMap::new();
        let mut counts: HashMap<(String, u8), usize> = HashMap::new();
        let read = |name: &str| {
            std::fs::read_to_string(format!("{dir}/{name}"))
                .unwrap_or_else(|e| panic!("{dir}/{name}: {e}"))
        };
        for line in read("lemmas_5k.csv").lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            if c.len() > 2 {
                let set = readings.entry(c[1].to_lowercase()).or_default();
                *set = set.with(fsm_pos_tag(c[2]));
            }
        }
        for line in read("word_forms.csv").lines().skip(1) {
            let c: Vec<&str> = line.split(',').collect();
            if c.len() > 5 {
                let word = c[5].to_lowercase();
                let pos = fsm_pos_tag(c[2]);
                let set = readings.entry(word.clone()).or_default();
                *set = set.with(pos);
                *counts.entry((word, pos as u8)).or_default() += c[4].parse::<usize>().unwrap_or(0);
            }
        }
        Self { readings, counts }
    }

    fn get(&self, form: &str) -> PosSet {
        self.readings.get(form).copied().unwrap_or(PosSet::EMPTY)
    }

    /// The frequency pick between `a` and `b` for `form`: the one train
    /// counts more often (first of the pair on a tie).
    fn more_frequent(&self, form: &str, a: Pos, b: Pos) -> Pos {
        let c = |p: Pos| {
            self.counts
                .get(&(form.to_string(), p as u8))
                .copied()
                .unwrap_or(0)
        };
        if c(b) > c(a) {
            b
        } else {
            a
        }
    }
}

/// Score for one pair of readings.
#[derive(Default)]
struct PairScore {
    /// Tokens whose readings hold both members and whose gold is one of them.
    tokens: usize,
    /// Of those, narrowed to one member by a parser rule.
    decided: usize,
    /// Decided and equal to gold.
    decided_right: usize,
    /// Frequency pick equal to gold, on the decided tokens.
    freq_right_on_decided: usize,
    /// Frequency pick equal to gold, on every token.
    freq_right: usize,
    /// Decided ? position pick : frequency pick — equal to gold, on every token.
    hybrid_right: usize,
    /// Decided tokens, keyed by the member kept: (kept, right, frequency
    /// right).
    by_kept: HashMap<u8, (usize, usize, usize)>,
}

impl PairScore {
    fn add(&mut self, entered: PosSet, survived: PosSet, gold: Pos, freq: Pos, pair: (Pos, Pos)) {
        let (a, b) = pair;
        if !(entered.contains(a) && entered.contains(b)) || (gold != a && gold != b) {
            return;
        }
        self.tokens += 1;
        let freq_ok = freq == gold;
        self.freq_right += usize::from(freq_ok);
        let kept = match (survived.contains(a), survived.contains(b)) {
            (true, false) => Some(a),
            (false, true) => Some(b),
            _ => None,
        };
        if let Some(k) = kept {
            self.decided += 1;
            self.decided_right += usize::from(k == gold);
            self.freq_right_on_decided += usize::from(freq_ok);
            self.hybrid_right += usize::from(k == gold);
            let e = self.by_kept.entry(k as u8).or_default();
            e.0 += 1;
            e.1 += usize::from(k == gold);
            e.2 += usize::from(freq_ok);
        } else {
            self.hybrid_right += usize::from(freq_ok);
        }
    }

    fn print(&self, name: &str, pair: (Pos, Pos)) {
        let pct = |n: usize, d: usize| {
            if d == 0 {
                0.0
            } else {
                100.0 * n as f64 / d as f64
            }
        };
        println!(
            "{name}: {} tokens, {} decided by position ({:.1}%)",
            self.tokens,
            self.decided,
            pct(self.decided, self.tokens)
        );
        println!(
            "  on decided tokens: position {:.1}% right, frequency {:.1}% right",
            pct(self.decided_right, self.decided),
            pct(self.freq_right_on_decided, self.decided)
        );
        for p in [pair.0, pair.1] {
            let (n, r, f) = self.by_kept.get(&(p as u8)).copied().unwrap_or_default();
            println!(
                "    kept {p:?}: {n} tokens, {:.1}% right, frequency {:.1}% right",
                pct(r, n),
                pct(f, n)
            );
        }
        println!(
            "  on all tokens: frequency {:.1}% right, position-then-frequency {:.1}% right",
            pct(self.freq_right, self.tokens),
            pct(self.hybrid_right, self.tokens)
        );
    }
}

/// The word-order facts [`Typology`] names, measured from gold train tags.
///
/// - adjective order: of adjectives attached to a noun as `amod`, the share
///   whose noun stands after them. Above 0.8 → `Before`, below 0.2 →
///   `After`, else `Both` (policy pins, not measurements).
/// - `adjective_opens_nominal`: the share of adjectives directly followed by
///   a verb. Below 2 % → true (policy pin).
fn measure_typology(train: &[Vec<Word>]) -> (Typology, f64, f64) {
    let (mut before, mut after, mut adj, mut adj_verb) = (0usize, 0usize, 0usize, 0usize);
    for s in train {
        for (i, w) in s.iter().enumerate() {
            if w.gold != Pos::Adj {
                continue;
            }
            adj += 1;
            let next = s.get(i + 1).map(|w| w.gold);
            before += usize::from(w.amod_head_after == Some(true));
            after += usize::from(w.amod_head_after == Some(false));
            adj_verb += usize::from(next == Some(Pos::Verb));
        }
    }
    let share_before = before as f64 / (before + after).max(1) as f64;
    let verb_after = adj_verb as f64 / adj.max(1) as f64;
    let adjective = if share_before > 0.8 {
        AdjectiveOrder::Before
    } else if share_before < 0.2 {
        AdjectiveOrder::After
    } else {
        AdjectiveOrder::Both
    };
    let typology = Typology {
        adjective,
        adjective_opens_nominal: verb_after < 0.02,
        // As shipped: no clause narrows (D-LXC-14 measured them below
        // frequency). The per-clause table scores them regardless.
        attribute_rules: Typology::ENGLISH.attribute_rules,
    };
    (typology, share_before, verb_after)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (train, test, coca) = match args.as_slice() {
        [train, test] => (train, test, None),
        [train, test, flag, dir] if flag == "--coca" => (train, test, Some(dir.as_str())),
        _ => {
            eprintln!("usage: ud_pos_eval TRAIN.conllu TEST.conllu [--coca WORD_FREQUENCY_DIR]");
            std::process::exit(2);
        }
    };
    let train = read_conllu(train);
    let (typology, share_before, verb_after) = match std::env::var("UD_TYPOLOGY").as_deref() {
        Ok("english") => (Typology::ENGLISH, f64::NAN, f64::NAN),
        _ => measure_typology(&train),
    };
    println!(
        "typology {typology:?} (adjective before its noun {:.1}%, adjective followed by a verb \
         {:.1}%)",
        100.0 * share_before,
        100.0 * verb_after
    );
    let lex = match coca {
        Some(dir) => Lexicon::from_coca(dir),
        None => Lexicon::from_train(&train),
    };
    let test = read_conllu(test);

    // One stream: every sentence ends in a stop, whatever its own punctuation.
    let mut ids: HashMap<&str, u16> = HashMap::new();
    let mut readings = Vec::new();
    let mut words: Vec<Option<&Word>> = Vec::new();
    for s in &test {
        for w in s {
            if w.gold == Pos::Stop {
                continue;
            }
            let next = u16::try_from(ids.len() + 1).unwrap_or(u16::MAX);
            let id = *ids.entry(w.form.as_str()).or_insert(next);
            readings.push(Reading::new(id, lex.get(&w.form)));
            words.push(Some(w));
        }
        readings.push(Reading::stop());
        words.push(None);
    }

    let parse = parse_readings_with(&readings, typology);
    let nv = (Pos::Noun, Pos::Verb);
    let aa = (Pos::Adj, Pos::Adv);
    let mut nv_score = PairScore::default();
    let mut aa_score = PairScore::default();
    let dump = std::env::var_os("UD_DUMP").is_some();
    for sv in &parse.ambiguous {
        let w = words[sv.index].expect("ambiguous tokens are words");
        if dump && sv.survived != sv.entered && !sv.survived.contains(w.gold) {
            let lo = sv.index.saturating_sub(3);
            let hi = (sv.index + 3).min(words.len());
            let ctx: Vec<String> = (lo..hi)
                .map(|j| match words[j] {
                    Some(x) if j == sv.index => format!("[{}]", x.form),
                    Some(x) => x.form.clone(),
                    None => "|".into(),
                })
                .collect();
            eprintln!(
                "WRONG gold {:?} kept {:?} | {}",
                w.gold,
                sv.survived,
                ctx.join(" ")
            );
        }
        for (score, pair) in [(&mut nv_score, nv), (&mut aa_score, aa)] {
            let freq = lex.more_frequent(&w.form, pair.0, pair.1);
            score.add(sv.entered, sv.survived, w.gold, freq, pair);
        }
    }
    let tokens = words.iter().filter(|w| w.is_some()).count();
    println!(
        "{tokens} test words, {} ambiguous, {} unknown; slot rule dropped {}, licensing {}, \
         attribute rule {}",
        parse.ambiguous.len(),
        parse.unknown,
        parse.slot_dropped,
        parse.unlicensed_dropped,
        parse.attribute_narrowed
    );
    nv_score.print("noun/verb", nv);
    aa_score.print("adjective/adverb", aa);

    // Each attribute clause on its own: every adjective/adverb token whose
    // gold is one of the two, the clause that fires, and whether the clause
    // or the frequency pick matches gold. Independent of which clauses are
    // enabled, and of the parse.
    let mut per_rule: HashMap<Option<AttributeRule>, (usize, usize, usize)> = HashMap::new();
    let edge = |i: Option<usize>| match i.and_then(|i| readings.get(i)) {
        Some(r) if !r.pos.contains(Pos::Stop) => r.pos,
        _ => PosSet::EMPTY,
    };
    for (i, w) in words.iter().enumerate() {
        let Some(w) = w else { continue };
        let this = readings[i].pos;
        if !(this.contains(Pos::Adj) && this.contains(Pos::Adv))
            || !matches!(w.gold, Pos::Adj | Pos::Adv)
        {
            continue;
        }
        let rule = attribute_rule(edge(i.checked_sub(1)), this, edge(Some(i + 1)), typology);
        let freq = lex.more_frequent(&w.form, Pos::Adj, Pos::Adv);
        let e = per_rule.entry(rule).or_default();
        e.0 += 1;
        e.1 += usize::from(rule.is_some_and(|r| r.keeps() == w.gold));
        e.2 += usize::from(freq == w.gold);
    }
    let pct = |n: usize, d: usize| {
        if d == 0 {
            0.0
        } else {
            100.0 * n as f64 / d as f64
        }
    };
    for rule in AttributeRule::ALL.map(Some).into_iter().chain([None]) {
        let (n, rule_right, freq_right) = per_rule.get(&rule).copied().unwrap_or_default();
        println!(
            "  clause {rule:?}: {n} tokens, clause {:.1}% right, frequency {:.1}% right",
            pct(rule_right, n),
            pct(freq_right, n)
        );
    }
}
