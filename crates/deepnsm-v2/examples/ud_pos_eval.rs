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
//! Beside the parser's rules it learns a **position table** from train: for
//! each ambiguous pair, P(first reading | previous readings, previous word is
//! a copula, next readings) — the neighbours only, never the word — and scores
//! it alone and combined with the frequency pick (log-odds). Whether the
//! language capitalises nouns is measured from train; where it does (German)
//! lexicon keys keep their case, and a sentence-initial word adds its
//! lowercase readings. The copula set is train's `cop` forms. A plain text
//! can be scored the same way after silver tagging, e.g. Animal Farm through
//! spaCy `en_core_web_sm` (lab step, not committed).
//!
//! **Rules, priorities and a quorum.** Every pair is also decided by a set of
//! literal rules (three learned position tables, frequency, determiner /
//! slot / infinitive / copula / attribute-clause rules, English `-ly`, German
//! capitalisation). Tables learn on 90 % of train; each rule is weighted on the
//! other 10 %, and test is scored three ways: the highest-precision rule that
//! fires (priority), Σ ±logit(precision) (summed quorum) and a logistic
//! regression over all votes fitted on the held-out 10 % (joint quorum).
//! `UD_RULES=1` prints every rule's held-out and test precision and joint
//! weight. `UD_DE_INVENTORY=DIR` adds the German literal inventories
//! (`build_de_codebook.py` output: inflection against the lemma, TEKAMOLO
//! adverbial cues); build them from TRAIN only, never from the test file.
//!
//! `UD_CLAUSE=1` switches on the clause rule
//! ([`Typology::predicate_required`]). The typology is measured from train;
//! `UD_TYPOLOGY=english` uses
//! [`Typology::ENGLISH`] instead. `UD_DUMP=1` prints every token a rule
//! narrowed away from its gold tag, with context, to stderr.

use std::collections::HashMap;

use deepnsm_v2::coca::fsm_pos_tag;
use deepnsm_v2::fsm::{
    answered_questions, attribute_rule, parse_readings_with, AdjectiveOrder, AttributeRule, Pos,
    PosSet, Reading, Tagged, Typology, ANSWERED_S,
};

/// One syntactic word of a UD sentence.
struct Word {
    /// Lexicon key: lowercased, or the surface where the language
    /// capitalises nouns (see [`capitalises_nouns`]).
    form: String,
    /// The form as written.
    surface: String,
    /// UPOS is NOUN (not PROPN/PRON) — for the capitalisation measurement.
    upos_noun: bool,
    /// The word is a copula (UD relation `cop`).
    copula: bool,
    gold: Pos,
    /// The adjective modifies a noun (`amod`) that stands after it
    /// (`Some(true)`) or before it (`Some(false)`); `None` otherwise.
    amod_head_after: Option<bool>,
}

/// The one UPOS → FSM fold. Language-neutral: it reads only UD's universal
/// tags, the universal `PronType=Rel` feature and the universal `advmod`
/// relation — an adjective used as an adverbial modifier (German "er läuft
/// **schnell**") is folded to [`Pos::Adv`]: it fills an adverbial (TEKAMOLO)
/// slot, which is the distinction the Adj/Adv pair stands for.
fn fold(upos: &str, feats: &str, form: &str, deprel: &str) -> Pos {
    let rel = feats.split('|').any(|f| f == "PronType=Rel");
    match upos {
        "ADJ" if deprel.starts_with("advmod") => Pos::Adv,
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
        let gold = fold(cols[3], cols[5], &form, cols.get(7).copied().unwrap_or("_"));
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
            surface: cols[1].to_string(),
            upos_noun: cols[3] == "NOUN",
            copula: cols.get(7).is_some_and(|r| *r == "cop"),
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
    /// Form → its COCA lemmas (COCA mode only), for the WordNet lookup.
    lemmas: HashMap<String, Vec<String>>,
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
        Self {
            readings,
            counts,
            lemmas: HashMap::new(),
        }
    }

    /// Readings and counts from COCA (`lemmas_5k.csv` + `word_forms.csv` in
    /// `dir`), folded by [`deepnsm_v2::coca::fsm_pos_tag`] — the lexicon
    /// `bible_wave` reads, so the frequency pick is COCA's, not the
    /// treebank's own. Counts are `word_forms.csv`'s per-form `wordFreq`.
    fn from_coca(dir: &str) -> Self {
        let mut readings: HashMap<String, PosSet> = HashMap::new();
        let mut counts: HashMap<(String, u8), usize> = HashMap::new();
        let mut lemmas: HashMap<String, Vec<String>> = HashMap::new();
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
                let lemma = c[1].to_lowercase();
                let ls = lemmas.entry(word.clone()).or_default();
                if !ls.contains(&lemma) {
                    ls.push(lemma);
                }
                let set = readings.entry(word.clone()).or_default();
                *set = set.with(pos);
                *counts.entry((word, pos as u8)).or_default() += c[4].parse::<usize>().unwrap_or(0);
            }
        }
        Self {
            readings,
            counts,
            lemmas,
        }
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

/// WordNet noun and verb sense counts per lemma, from the release's
/// `wordnet31_isa_v2.tsv` (7 columns: word, pos, sense_num, …).
fn load_wordnet(path: &str) -> HashMap<String, (usize, usize)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut senses: HashMap<(String, bool), std::collections::HashSet<String>> = HashMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() != 7 || !matches!(c[1], "n" | "v") {
            continue;
        }
        let word = c[0].to_lowercase().replace('_', " ");
        senses
            .entry((word, c[1] == "v"))
            .or_default()
            .insert(c[2].to_string());
    }
    let mut out: HashMap<String, (usize, usize)> = HashMap::new();
    for ((word, verb), s) in senses {
        let e = out.entry(word).or_default();
        if verb {
            e.1 = s.len();
        } else {
            e.0 = s.len();
        }
    }
    out
}

/// The WordNet pick for `form`: the reading with more senses over its COCA
/// lemmas, or `None` when WordNet has neither.
fn wordnet_pick(lex: &Lexicon, wn: &HashMap<String, (usize, usize)>, form: &str) -> Option<Pos> {
    let (mut n, mut v) = (0, 0);
    for lemma in lex.lemmas.get(form)? {
        if let Some(&(a, b)) = wn.get(lemma) {
            n += a;
            v += b;
        }
    }
    match (n, v) {
        (0, 0) => None,
        _ if n >= v => Some(Pos::Noun),
        _ => Some(Pos::Verb),
    }
}

/// Whether the language capitalises nouns, measured from gold: of
/// non-initial words, the share of NOUN capitalised and the share of every
/// other non-PROPN word capitalised. Nouns above 90 % and the rest below 10 %
/// → true (policy pins). Returns (decision, noun share, other share).
fn capitalises_nouns(train: &[Vec<Word>]) -> (bool, f64, f64) {
    let (mut n, mut nc, mut o, mut oc) = (0usize, 0usize, 0usize, 0usize);
    for s in train {
        for w in s.iter().skip(1) {
            let cap = w.surface.starts_with(char::is_uppercase);
            if w.upos_noun {
                n += 1;
                nc += usize::from(cap);
            } else if w.gold != Pos::Noun && w.surface.chars().any(char::is_alphabetic) {
                o += 1;
                oc += usize::from(cap);
            }
        }
    }
    let ns = nc as f64 / n.max(1) as f64;
    let os = oc as f64 / o.max(1) as f64;
    (ns > 0.9 && os < 0.1, ns, os)
}

/// Re-key every word on its surface (case kept).
fn keep_case(sentences: &mut [Vec<Word>]) {
    for w in sentences.iter_mut().flatten() {
        w.form = w.surface.clone();
    }
}

/// The readings of word `k` of a sentence: its own key, and — for the first
/// word when case is kept — also its lowercase form (sentence-initial
/// capitals say nothing about the word class).
fn readings_of(lex: &Lexicon, s: &[Word], k: usize, cased: bool) -> PosSet {
    let own = lex.get(&s[k].form);
    if cased && k == 0 {
        own.union(lex.get(&s[k].form.to_lowercase()))
    } else {
        own
    }
}

/// Positional context of a token: the neighbours' reading sets and whether
/// the previous word is a copula. Never the word itself.
type Context = (u8, bool, u8, bool, u8);

/// Which positional evidence the table keys on (`UD_CTX`): the neighbours
/// (`neigh`, default), the question test — the 2³ mask of SPO questions the
/// clause has answered before the word, plus the next readings (`q`) — or
/// both (`both`).
fn ctx_mode() -> &'static str {
    match std::env::var("UD_CTX").as_deref() {
        Ok("q") => "q",
        Ok("both") => "both",
        _ => "neigh",
    }
}

/// The 2³ answered-question mask before every word of a sentence. The left
/// context is tagged by the frequency pick of each word's readings — never
/// gold.
fn answered_masks(lex: &Lexicon, s: &[Word], cased: bool) -> Vec<u8> {
    let tags: Vec<Tagged> = (0..s.len())
        .map(|k| {
            let set = readings_of(lex, s, k, cased);
            let pos = set
                .iter()
                .max_by_key(|p| {
                    lex.counts
                        .get(&(s[k].form.clone(), *p as u8))
                        .copied()
                        .unwrap_or(0)
                })
                .unwrap_or(Pos::Other);
            Tagged::new(0, pos)
        })
        .collect();
    answered_questions(&tags)
}

fn bits(p: PosSet) -> u8 {
    Pos::ALL
        .iter()
        .enumerate()
        .filter(|(_, x)| p.contains(**x))
        .fold(0u8, |a, (i, _)| a | (1 << i))
}

/// `ctx` with the fields `mode` does not read zeroed.
fn project(ctx: Context, mode: &str) -> Context {
    let neigh = mode != "q";
    let q = mode != "neigh";
    (
        if neigh { ctx.0 } else { 0 },
        neigh && ctx.1,
        ctx.2,
        neigh && ctx.3,
        if q { ctx.4 } else { 0 },
    )
}

/// P(first | context) for one reading pair, learned from gold train tags
/// with the evaluation lexicon's readings. Backs off from the full context
/// to its next-only and previous-only halves when support is thin.
struct PositionTable {
    /// Which context fields this table keys on (see [`ctx_mode`]).
    mode: &'static str,
    full: HashMap<Context, [usize; 2]>,
    next: HashMap<u8, [usize; 2]>,
    prev: HashMap<(u8, bool), [usize; 2]>,
    /// Next readings + copula anywhere in the clause.
    clause: HashMap<(u8, bool), [usize; 2]>,
    /// Answered-question mask + next readings.
    question: HashMap<(u8, u8), [usize; 2]>,
}

/// Minimum train tokens behind a context before it may decide (policy pin).
const MIN_SUPPORT: usize = 10;

impl PositionTable {
    fn learn(
        train: &[Vec<Word>],
        lex: &Lexicon,
        copulas: &std::collections::HashSet<String>,
        cased: bool,
        pair: (Pos, Pos),
        mode: &'static str,
    ) -> Self {
        let mut t = Self {
            mode,
            full: HashMap::new(),
            next: HashMap::new(),
            prev: HashMap::new(),
            clause: HashMap::new(),
            question: HashMap::new(),
        };
        for s in train {
            let set = |k: usize| readings_of(lex, s, k, cased);
            let masks = answered_masks(lex, s, cased);
            for k in 0..s.len() {
                let this = set(k);
                let w = &s[k];
                if !(this.contains(pair.0) && this.contains(pair.1))
                    || (w.gold != pair.0 && w.gold != pair.1)
                {
                    continue;
                }
                let ctx = project(Self::context(s, k, &set, copulas, &masks), mode);
                let slot = usize::from(w.gold == pair.1);
                t.full.entry(ctx).or_default()[slot] += 1;
                t.next.entry(ctx.2).or_default()[slot] += 1;
                t.prev.entry((ctx.0, ctx.1)).or_default()[slot] += 1;
                t.clause.entry((ctx.2, ctx.3)).or_default()[slot] += 1;
                t.question.entry((ctx.4, ctx.2)).or_default()[slot] += 1;
            }
        }
        t
    }

    fn context(
        s: &[Word],
        k: usize,
        set: &dyn Fn(usize) -> PosSet,
        copulas: &std::collections::HashSet<String>,
        masks: &[u8],
    ) -> Context {
        let prev = k.checked_sub(1);
        // The clause: the words between punctuation marks around `k`.
        let is_break = |w: &Word| !w.surface.chars().any(char::is_alphanumeric);
        let lo = (0..k).rev().find(|&j| is_break(&s[j])).map_or(0, |j| j + 1);
        let hi = (k + 1..s.len())
            .find(|&j| is_break(&s[j]))
            .unwrap_or(s.len());
        let clause_copula = std::env::var_os("UD_NO_CLAUSE_CTX").is_none()
            && (lo..hi)
                .filter(|&j| j != k)
                .any(|j| copulas.contains(&s[j].form.to_lowercase()));
        (
            prev.map_or(0, |j| bits(set(j))),
            prev.is_some_and(|j| copulas.contains(&s[j].form.to_lowercase())),
            if k + 1 < s.len() { bits(set(k + 1)) } else { 0 },
            clause_copula,
            masks[k],
        )
    }

    /// P(first member | context), with the support it rests on.
    fn p_first(&self, ctx: Context) -> Option<(f64, usize)> {
        let ctx = project(ctx, self.mode);
        let pick = |c: [usize; 2]| {
            let n = c[0] + c[1];
            (n >= MIN_SUPPORT).then(|| ((c[0] as f64 + 0.5) / (n as f64 + 1.0), n))
        };
        if let Some(r) = self.full.get(&ctx).copied().and_then(pick) {
            return Some(r);
        }
        // Back off: whichever partial context is most decisive.
        // Each mode backs off only to its own partial contexts, so a field it
        // does not read can never collapse into a constant key (the class
        // prior posing as a position).
        let neigh = self.mode != "q";
        let q = self.mode != "neigh";
        [
            neigh
                .then(|| self.clause.get(&(ctx.2, ctx.3)).copied())
                .flatten(),
            q.then(|| self.question.get(&(ctx.4, ctx.2)).copied())
                .flatten(),
            self.next.get(&ctx.2).copied(),
            neigh
                .then(|| self.prev.get(&(ctx.0, ctx.1)).copied())
                .flatten(),
        ]
        .into_iter()
        .flatten()
        .filter_map(pick)
        .max_by(|x, y| (x.0 - 0.5).abs().total_cmp(&(y.0 - 0.5).abs()))
    }
}

/// Learn a [`PositionTable`] on train and score it on test, against the
/// frequency pick and their log-odds combination, on the same tokens.
fn position_table_report(
    train: &[Vec<Word>],
    test: &[Vec<Word>],
    lex: &Lexicon,
    copulas: &std::collections::HashSet<String>,
    cased: bool,
    pair: (Pos, Pos),
    name: &str,
) {
    let table = PositionTable::learn(train, lex, copulas, cased, pair, ctx_mode());
    let (a, b) = pair;
    let prior = {
        let (x, y) = table
            .full
            .values()
            .fold((0, 0), |s, c| (s.0 + c[0], s.1 + c[1]));
        (x as f64 + 0.5) / ((x + y) as f64 + 1.0)
    };
    let logit = |p: f64| (p / (1.0 - p)).ln();
    let (mut n, mut dec, mut dec_right, mut freq_on_dec, mut freq_all, mut comb_all) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for s in test {
        let set = |k: usize| readings_of(lex, s, k, cased);
        let masks = answered_masks(lex, s, cased);
        for k in 0..s.len() {
            let w = &s[k];
            let this = set(k);
            if !(this.contains(a) && this.contains(b)) || (w.gold != a && w.gold != b) {
                continue;
            }
            n += 1;
            let ca = lex
                .counts
                .get(&(w.form.clone(), a as u8))
                .copied()
                .unwrap_or(0);
            let cb = lex
                .counts
                .get(&(w.form.clone(), b as u8))
                .copied()
                .unwrap_or(0);
            let p_word = (ca as f64 + 0.5) / ((ca + cb) as f64 + 1.0);
            let freq = if cb > ca { b } else { a };
            freq_all += usize::from(freq == w.gold);
            let ctx = PositionTable::context(s, k, &set, copulas, &masks);
            let p_ctx = table.p_first(ctx);
            if let Some((p, _)) = p_ctx {
                if !(0.25..0.75).contains(&p) {
                    let pick = if p >= 0.75 { a } else { b };
                    dec += 1;
                    dec_right += usize::from(pick == w.gold);
                    freq_on_dec += usize::from(freq == w.gold);
                }
            }
            let score = logit(p_word) + p_ctx.map_or(0.0, |(p, _)| logit(p) - logit(prior));
            let comb = if score >= 0.0 { a } else { b };
            comb_all += usize::from(comb == w.gold);
        }
    }
    let pct = |x: usize, d: usize| {
        if d == 0 {
            0.0
        } else {
            100.0 * x as f64 / d as f64
        }
    };
    println!(
        "  {name} position table ({} contexts): decides {dec} of {n} at {:.1}% (frequency \
         {:.1}% on them); all tokens: frequency {:.1}%, position × frequency {:.1}%",
        table.full.len(),
        pct(dec_right, dec),
        pct(freq_on_dec, dec),
        pct(freq_all, n),
        pct(comb_all, n)
    );
}

/// What one rule sees of a token: its readings, its neighbours' readings and
/// forms, the positional context, and the word's counts for the pair.
struct Tok<'a> {
    this: PosSet,
    prev: PosSet,
    next: PosSet,
    prev_form: Option<&'a str>,
    surface: &'a str,
    form: &'a str,
    initial: bool,
    ctx: Context,
    /// Lexicon counts of the pair's first and second reading for the word.
    counts: (usize, usize),
}

type Vote = Box<dyn Fn(&Tok) -> Option<Pos>>;

/// A literal grammar rule: votes for one reading of the pair, or abstains.
struct Rule {
    name: &'static str,
    vote: Vote,
}

fn only(p: PosSet, x: Pos) -> bool {
    p == PosSet::single(x)
}

/// Every rule for `pair`: three learned position tables (vote when
/// decisive), frequency split by how lopsided the word is, and literal
/// grammar rules. Each is weighted later by its held-out precision, so a
/// weak or language-specific rule costs nothing.
/// German literal inventories (`build_de_codebook.py` output, built from
/// train only), read when `UD_DE_INVENTORY` names their directory.
#[derive(Default)]
struct Inventory {
    /// Adjective/adverb form → its lemma (`lexicon.tsv`, rows tagged j or r).
    lemma: HashMap<String, String>,
    /// TEKAMOLO cue lemmas attested as `advmod` (`tekamolo.tsv`).
    adverbial_cue: std::collections::HashSet<String>,
}

impl Inventory {
    fn load() -> Option<Self> {
        let dir = std::env::var("UD_DE_INVENTORY")
            .ok()
            .filter(|d| !d.is_empty())?;
        let read = |name: &str| {
            std::fs::read_to_string(format!("{dir}/{name}"))
                .unwrap_or_else(|e| panic!("{dir}/{name}: {e}"))
        };
        let mut inv = Self::default();
        for line in read("lexicon.tsv").lines().filter(|l| !l.starts_with('#')) {
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() > 2 && matches!(c[2], "j" | "r") {
                inv.lemma
                    .entry(c[0].to_string())
                    .or_insert_with(|| c[1].to_string());
            }
        }
        for line in read("tekamolo.tsv").lines().filter(|l| !l.starts_with('#')) {
            let c: Vec<&str> = line.split('\t').collect();
            if c.len() > 2 && c[2] == "advmod" {
                inv.adverbial_cue.insert(c[1].to_string());
            }
        }
        Some(inv)
    }
}

fn rules(
    inventory: Option<std::rc::Rc<Inventory>>,
    pair: (Pos, Pos),
    tables: [std::rc::Rc<PositionTable>; 3],
    copulas: std::rc::Rc<std::collections::HashSet<String>>,
    typology: Typology,
    cased: bool,
) -> Vec<Rule> {
    let (a, b) = pair;
    let mut out: Vec<Rule> = Vec::new();
    for (name, t) in [
        "table: neighbours",
        "table: answered questions",
        "table: both",
    ]
    .into_iter()
    .zip(tables)
    {
        out.push(Rule {
            name,
            vote: Box::new(move |x| {
                t.p_first(x.ctx).and_then(|(p, _)| {
                    if p >= 0.75 {
                        Some(a)
                    } else if p <= 0.25 {
                        Some(b)
                    } else {
                        None
                    }
                })
            }),
        });
    }
    let lopsided = |(ca, cb): (usize, usize)| {
        let n = ca + cb;
        n > 0 && (ca * 10 >= n * 9 || cb * 10 >= n * 9)
    };
    out.push(Rule {
        name: "frequency, share >= 0.9",
        vote: Box::new(move |x| {
            lopsided(x.counts).then_some(if x.counts.0 >= x.counts.1 { a } else { b })
        }),
    });
    out.push(Rule {
        name: "frequency, share < 0.9",
        vote: Box::new(move |x| {
            (x.counts.0 + x.counts.1 > 0 && !lopsided(x.counts))
                .then_some(if x.counts.0 >= x.counts.1 { a } else { b })
        }),
    });
    let cop = move |f: Option<&str>| f.is_some_and(|f| copulas.contains(&f.to_lowercase()));
    let mut fixed: Vec<(&'static str, Vote)> = Vec::new();
    if pair == (Pos::Noun, Pos::Verb) {
        fixed.push((
            "after a determiner -> noun",
            Box::new(|x: &Tok| only(x.prev, Pos::Det).then_some(Pos::Noun)),
        ));
        fixed.push((
            "after an adjective -> noun",
            Box::new(move |x: &Tok| {
                (only(x.prev, Pos::Adj) && typology.adjective_opens_nominal).then_some(Pos::Noun)
            }),
        ));
        fixed.push((
            "slot: subject answered, prev a noun -> verb",
            Box::new(|x: &Tok| {
                (x.ctx.4 == ANSWERED_S && only(x.prev, Pos::Noun)).then_some(Pos::Verb)
            }),
        ));
        fixed.push((
            "before a determiner -> verb",
            Box::new(|x: &Tok| only(x.next, Pos::Det).then_some(Pos::Verb)),
        ));
        fixed.push((
            "before a verb-only word -> noun",
            Box::new(|x: &Tok| only(x.next, Pos::Verb).then_some(Pos::Noun)),
        ));
        fixed.push((
            "after an infinitive marker -> verb",
            Box::new(|x: &Tok| {
                x.prev_form
                    .is_some_and(|f| f.eq_ignore_ascii_case("to") || f.eq_ignore_ascii_case("zu"))
                    .then_some(Pos::Verb)
            }),
        ));
        let c = cop.clone();
        fixed.push((
            "after a non-copula verb-only word -> verb",
            Box::new(move |x: &Tok| {
                (only(x.prev, Pos::Verb) && !c(x.prev_form)).then_some(Pos::Verb)
            }),
        ));
        fixed.push((
            "capitalised, not initial -> noun",
            Box::new(move |x: &Tok| {
                (cased && !x.initial && x.surface.starts_with(char::is_uppercase))
                    .then_some(Pos::Noun)
            }),
        ));
        fixed.push((
            "lowercase where nouns are capitalised -> verb",
            Box::new(move |x: &Tok| {
                (cased && x.surface.starts_with(char::is_lowercase)).then_some(Pos::Verb)
            }),
        ));
    } else {
        for (name, rule) in [
            (
                "clause: after a determiner -> adj",
                AttributeRule::AfterDeterminer,
            ),
            (
                "clause: before an adjective -> adv",
                AttributeRule::BeforeAdjective,
            ),
            (
                "clause: between subject and verb -> adv",
                AttributeRule::BetweenSubjectAndVerb,
            ),
            ("clause: next to a noun -> adj", AttributeRule::NextToNoun),
        ] {
            fixed.push((
                name,
                Box::new(move |x: &Tok| {
                    (attribute_rule(x.prev, x.this, x.next, typology) == Some(rule))
                        .then(|| rule.keeps())
                }),
            ));
        }
        let c = cop.clone();
        fixed.push((
            "after a copula -> adj",
            Box::new(move |x: &Tok| c(x.prev_form).then_some(Pos::Adj)),
        ));
        let c = cop.clone();
        fixed.push((
            "after a non-copula verb, no noun next -> adv",
            Box::new(move |x: &Tok| {
                (x.prev.contains(Pos::Verb) && !c(x.prev_form) && !x.next.contains(Pos::Noun))
                    .then_some(Pos::Adv)
            }),
        ));
        fixed.push((
            "before a determiner -> adv",
            Box::new(|x: &Tok| only(x.next, Pos::Det).then_some(Pos::Adv)),
        ));
        fixed.push((
            "before a verb-only word -> adv",
            Box::new(|x: &Tok| only(x.next, Pos::Verb).then_some(Pos::Adv)),
        ));
        fixed.push((
            "-ly ending -> adv",
            Box::new(|x: &Tok| x.form.ends_with("ly").then_some(Pos::Adv)),
        ));
        fixed.push((
            "inflected ending before a noun -> adj",
            Box::new(move |x: &Tok| {
                (cased
                    && ["e", "en", "em", "er", "es"]
                        .iter()
                        .any(|e| x.form.ends_with(e))
                    && x.next.contains(Pos::Noun))
                .then_some(Pos::Adj)
            }),
        ));
        if let Some(inv) = inventory {
            let i = inv.clone();
            fixed.push((
                "uninflected (form is its lemma) -> adv",
                Box::new(move |x: &Tok| {
                    (i.lemma.get(x.form).map(String::as_str) == Some(x.form)).then_some(Pos::Adv)
                }),
            ));
            let i = inv.clone();
            fixed.push((
                "inflected (lemma + ending) -> adj",
                Box::new(move |x: &Tok| {
                    i.lemma
                        .get(x.form)
                        .and_then(|l| x.form.strip_prefix(l.as_str()))
                        .and_then(|e| {
                            ["e", "en", "em", "er", "es"]
                                .contains(&e)
                                .then_some(Pos::Adj)
                        })
                }),
            ));
            let i = inv.clone();
            fixed.push((
                "uninflected right before a noun -> adv",
                Box::new(move |x: &Tok| {
                    (i.lemma.get(x.form).map(String::as_str) == Some(x.form)
                        && x.next.contains(Pos::Noun))
                    .then_some(Pos::Adv)
                }),
            ));
            let i = inv;
            fixed.push((
                "TEKAMOLO adverbial cue lemma -> adv",
                Box::new(move |x: &Tok| {
                    let l = i.lemma.get(x.form).map_or(x.form, String::as_str);
                    i.adverbial_cue.contains(l).then_some(Pos::Adv)
                }),
            ));
        }
        let c = cop.clone();
        fixed.push((
            "clause-final, no copula before -> adv",
            Box::new(move |x: &Tok| {
                (x.next.is_empty() && !x.prev.is_empty() && !c(x.prev_form)).then_some(Pos::Adv)
            }),
        ));
    }
    out.extend(fixed.into_iter().map(|(name, vote)| Rule { name, vote }));
    out
}

/// Every token of the pair in `sentences`, as rules see it, with its gold.
fn pair_tokens<'a>(
    sentences: &'a [Vec<Word>],
    lex: &Lexicon,
    copulas: &std::collections::HashSet<String>,
    cased: bool,
    pair: (Pos, Pos),
) -> Vec<(Tok<'a>, Pos)> {
    let mut out = Vec::new();
    for s in sentences {
        let set = |k: usize| readings_of(lex, s, k, cased);
        let masks = answered_masks(lex, s, cased);
        for k in 0..s.len() {
            let w = &s[k];
            let this = set(k);
            if !(this.contains(pair.0) && this.contains(pair.1))
                || (w.gold != pair.0 && w.gold != pair.1)
            {
                continue;
            }
            let c = |p: Pos| {
                lex.counts
                    .get(&(w.form.clone(), p as u8))
                    .copied()
                    .unwrap_or(0)
            };
            out.push((
                Tok {
                    this,
                    prev: k.checked_sub(1).map_or(PosSet::EMPTY, &set),
                    next: if k + 1 < s.len() {
                        set(k + 1)
                    } else {
                        PosSet::EMPTY
                    },
                    prev_form: k.checked_sub(1).map(|j| s[j].form.as_str()),
                    surface: &w.surface,
                    form: &w.form,
                    initial: k == 0,
                    ctx: PositionTable::context(s, k, &set, copulas, &masks),
                    counts: (c(pair.0), c(pair.1)),
                },
                w.gold,
            ));
        }
    }
    out
}

/// Rules, priorities and a quorum. The position tables learn on 90 % of
/// train; each rule's weight is its precision when it fires on the other
/// 10 %. Test is scored by (a) the highest-weight rule that fires
/// (priority) and (b) the weighted vote Σ ±logit(precision) of every rule
/// that fires (quorum); both fall back to frequency when nothing fires.
/// The lexicon is the run's own (from all of train, or COCA).
#[allow(clippy::too_many_arguments)]
fn quorum_report(
    train: &[Vec<Word>],
    test: &[Vec<Word>],
    lex: &Lexicon,
    coca: bool,
    copulas: &std::collections::HashSet<String>,
    cased: bool,
    typology: Typology,
    pair: (Pos, Pos),
    name: &str,
) {
    let fit: Vec<&Vec<Word>> = train
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 10 != 0)
        .map(|(_, s)| s)
        .collect();
    let held: Vec<&Vec<Word>> = train
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 10 == 0)
        .map(|(_, s)| s)
        .collect();
    let fit: Vec<Vec<Word>> = fit.into_iter().map(|s| clone_sentence(s)).collect();
    let held: Vec<Vec<Word>> = held.into_iter().map(|s| clone_sentence(s)).collect();
    // A train-built lexicon must not have seen the held-out sentences, or
    // held-out words carry only the readings they were seen with and every
    // weight is fitted on an easier set than test.
    let own;
    let lex = if coca {
        lex
    } else {
        own = Lexicon::from_train(&fit);
        &own
    };
    let tables = ["neigh", "q", "both"]
        .map(|m| std::rc::Rc::new(PositionTable::learn(&fit, lex, copulas, cased, pair, m)));
    let rules = rules(
        Inventory::load().map(std::rc::Rc::new),
        pair,
        tables,
        std::rc::Rc::new(copulas.clone()),
        typology,
        cased,
    );

    let held = pair_tokens(&held, lex, copulas, cased, pair);
    let weights: Vec<(usize, f64)> = rules
        .iter()
        .map(|r| {
            let (mut n, mut right) = (0usize, 0usize);
            for (t, g) in &held {
                if let Some(v) = (r.vote)(t) {
                    n += 1;
                    right += usize::from(v == *g);
                }
            }
            (n, (right as f64 + 0.5) / (n as f64 + 1.0))
        })
        .collect();
    // A rule votes only with held-out support and better than chance.
    let mut order: Vec<usize> = (0..rules.len())
        .filter(|&i| weights[i].0 >= 5 && weights[i].1 > 0.5)
        .collect();
    order.sort_by(|&x, &y| weights[y].1.total_cmp(&weights[x].1));
    let logit = |p: f64| {
        let p = p.min(0.99);
        (p / (1.0 - p)).ln()
    };

    // The joint quorum: a logistic regression over every rule's vote (+1
    // first reading, -1 second, 0 abstain) and the word's frequency log-odds,
    // fitted on held-out. Correlated rules (the three tables agree almost
    // always) share one weight between them instead of each counting in full.
    let (a, b) = pair;
    let features = |t: &Tok| -> Vec<f64> {
        let mut f: Vec<f64> = rules
            .iter()
            .map(|r| match (r.vote)(t) {
                Some(v) if v == a => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            })
            .collect();
        let lo = ((t.counts.0 as f64 + 0.5) / (t.counts.1 as f64 + 0.5)).ln();
        f.push(lo.clamp(-5.0, 5.0) / 5.0);
        f.push(1.0);
        f
    };
    let xs: Vec<(Vec<f64>, f64)> = held
        .iter()
        .map(|(t, g)| (features(t), f64::from(u8::from(*g == a))))
        .collect();
    let dim = rules.len() + 2;
    let mut w = vec![0.0f64; dim];
    let lambda = 1e-3;
    for _ in 0..3000 {
        let mut grad = vec![0.0f64; dim];
        for (x, y) in &xs {
            let z: f64 = x.iter().zip(&w).map(|(xi, wi)| xi * wi).sum();
            let e = 1.0 / (1.0 + (-z).exp()) - y;
            for (gi, xi) in grad.iter_mut().zip(x) {
                *gi += e * xi;
            }
        }
        let n = xs.len().max(1) as f64;
        for (wi, gi) in w.iter_mut().zip(&grad) {
            *wi -= 0.5 * (gi / n + lambda * *wi);
        }
    }

    let toks = pair_tokens(test, lex, copulas, cased, pair);
    let mut joint_ok = 0usize;
    let (mut freq_ok, mut prio_ok, mut quorum_ok, mut silent) = (0usize, 0usize, 0usize, 0usize);
    let mut fired = vec![(0usize, 0usize); rules.len()];
    for (t, g) in &toks {
        let freq = if t.counts.1 > t.counts.0 { b } else { a };
        freq_ok += usize::from(freq == *g);
        let votes: Vec<Option<Pos>> = rules.iter().map(|r| (r.vote)(t)).collect();
        for (i, v) in votes.iter().enumerate() {
            if let Some(v) = v {
                fired[i].0 += 1;
                fired[i].1 += usize::from(v == g);
            }
        }
        let prio = order.iter().find_map(|&i| votes[i]);
        silent += usize::from(prio.is_none());
        prio_ok += usize::from(prio.unwrap_or(freq) == *g);
        let score: f64 = order
            .iter()
            .filter_map(|&i| {
                votes[i].map(|v| if v == a { 1.0 } else { -1.0 } * logit(weights[i].1))
            })
            .sum();
        let quorum = match score.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => a,
            Some(std::cmp::Ordering::Less) => b,
            _ => freq,
        };
        quorum_ok += usize::from(quorum == *g);
        let z: f64 = features(t).iter().zip(&w).map(|(xi, wi)| xi * wi).sum();
        joint_ok += usize::from(if z > 0.0 { a } else { b } == *g);
    }
    let pct = |x: usize, d: usize| {
        if d == 0 {
            0.0
        } else {
            100.0 * x as f64 / d as f64
        }
    };
    let n = toks.len();
    println!(
        "  {name} quorum: {} rules, {} voting; {n} test tokens ({silent} with no vote) — \
         frequency {:.1}%, priority {:.1}%, summed quorum {:.1}%, joint quorum {:.1}%",
        rules.len(),
        order.len(),
        pct(freq_ok, n),
        pct(prio_ok, n),
        pct(quorum_ok, n),
        pct(joint_ok, n)
    );
    if std::env::var_os("UD_RULES").is_some() {
        for i in 0..rules.len() {
            let voting = if order.contains(&i) { ' ' } else { 'x' };
            println!(
                "   {voting} {:46} held-out {:5} at {:5.1}% | test {:5} at {:5.1}% | joint w {:+.2}",
                rules[i].name,
                weights[i].0,
                100.0 * weights[i].1,
                fired[i].0,
                pct(fired[i].1, fired[i].0),
                w[i]
            );
        }
    }
}

fn clone_sentence(s: &[Word]) -> Vec<Word> {
    s.iter()
        .map(|w| Word {
            form: w.form.clone(),
            surface: w.surface.clone(),
            upos_noun: w.upos_noun,
            copula: w.copula,
            gold: w.gold,
            amod_head_after: w.amod_head_after,
        })
        .collect()
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
        predicate_required: false,
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
    let mut train = read_conllu(train);
    // German capitalises nouns: case is morphology there, so keys keep it.
    // Measured per language; COCA (English) stays lowercase.
    let (caps, noun_share, other_share) = capitalises_nouns(&train);
    let cased = caps && coca.is_none();
    println!(
        "capitalisation: non-initial nouns {:.1}% capitalised, other words {:.1}% → case {}",
        100.0 * noun_share,
        100.0 * other_share,
        if cased { "kept" } else { "folded" }
    );
    if cased {
        keep_case(&mut train);
    }
    let copulas: std::collections::HashSet<String> = train
        .iter()
        .flatten()
        .filter(|w| w.copula)
        .map(|w| w.form.to_lowercase())
        .collect();
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
    let typology = Typology {
        predicate_required: std::env::var("UD_CLAUSE").is_ok_and(|v| !v.is_empty()),
        ..typology
    };
    let mut lex = match coca {
        Some(dir) => Lexicon::from_coca(dir),
        None => Lexicon::from_train(&train),
    };
    // WordNet as a second lexicon on which readings EXIST: with
    // `UD_WORDNET_FILTER=1`, a COCA noun/verb homograph keeps a reading only
    // if one of its lemmas has that sense in WordNet. Reports what it costs.
    if std::env::var("UD_WORDNET_FILTER").is_ok_and(|v| !v.is_empty()) {
        let path = std::env::var("UD_WORDNET").expect("UD_WORDNET_FILTER needs UD_WORDNET");
        let wn = load_wordnet(&path);
        let mut narrowed = 0;
        let forms: Vec<String> = lex.readings.keys().cloned().collect();
        for form in forms {
            let set = lex.readings[&form];
            if !(set.contains(Pos::Noun) && set.contains(Pos::Verb)) {
                continue;
            }
            let (mut n, mut v) = (0, 0);
            for lemma in lex.lemmas.get(&form).into_iter().flatten() {
                if let Some(&(a, b)) = wn.get(lemma) {
                    n += a;
                    v += b;
                }
            }
            let kept = match (n > 0, v > 0) {
                (true, false) => set.without(Pos::Verb),
                (false, true) => set.without(Pos::Noun),
                _ => set,
            };
            if kept != set {
                narrowed += 1;
                lex.readings.insert(form, kept);
            }
        }
        println!("WordNet filter: {narrowed} COCA noun/verb forms lost a reading WordNet lacks");
    }
    let mut test = read_conllu(test);
    if cased {
        keep_case(&mut test);
    }

    // One stream: every sentence ends in a stop, whatever its own punctuation.
    let mut ids: HashMap<&str, u16> = HashMap::new();
    let mut readings = Vec::new();
    let mut words: Vec<Option<&Word>> = Vec::new();
    for s in &test {
        for (k, w) in s.iter().enumerate() {
            if w.gold == Pos::Stop {
                continue;
            }
            let next = u16::try_from(ids.len() + 1).unwrap_or(u16::MAX);
            let id = *ids.entry(w.form.as_str()).or_insert(next);
            readings.push(Reading::new(id, readings_of(&lex, s, k, cased)));
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
         attribute rule {}, clause rule {}",
        parse.ambiguous.len(),
        parse.unknown,
        parse.slot_dropped,
        parse.unlicensed_dropped,
        parse.attribute_narrowed,
        parse.unpredicated_dropped
    );
    nv_score.print("noun/verb", nv);
    position_table_report(&train, &test, &lex, &copulas, cased, nv, "noun/verb");
    quorum_report(
        &train,
        &test,
        &lex,
        coca.is_some(),
        &copulas,
        cased,
        typology,
        nv,
        "noun/verb",
    );
    // A parallel coordinate: WordNet's own noun/verb sense counts as the
    // prior, scored on the same tokens (COCA mode supplies the lemmas).
    if let Some(path) = std::env::var("UD_WORDNET").ok().filter(|p| !p.is_empty()) {
        let wn = load_wordnet(&path);
        let (mut all, mut all_right, mut on_dec, mut on_dec_right, mut covered) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for sv in &parse.ambiguous {
            let w = words[sv.index].expect("ambiguous tokens are words");
            if !(sv.entered.contains(Pos::Noun) && sv.entered.contains(Pos::Verb))
                || !matches!(w.gold, Pos::Noun | Pos::Verb)
            {
                continue;
            }
            all += 1;
            let Some(pick) = wordnet_pick(&lex, &wn, &w.form) else {
                continue;
            };
            covered += 1;
            all_right += usize::from(pick == w.gold);
            if sv.survived.contains(Pos::Noun) != sv.survived.contains(Pos::Verb) {
                on_dec += 1;
                on_dec_right += usize::from(pick == w.gold);
            }
        }
        let pct = |n: usize, d: usize| {
            if d == 0 {
                0.0
            } else {
                100.0 * n as f64 / d as f64
            }
        };
        println!(
            "  WordNet sense-count pick: covers {covered} of {all}, {:.1}% right on those; \
             on position-decided tokens {:.1}% right ({on_dec})",
            pct(all_right, covered),
            pct(on_dec_right, on_dec)
        );
    }
    aa_score.print("adjective/adverb", aa);
    position_table_report(&train, &test, &lex, &copulas, cased, aa, "adjective/adverb");
    quorum_report(
        &train,
        &test,
        &lex,
        coca.is_some(),
        &copulas,
        cased,
        typology,
        aa,
        "adjective/adverb",
    );

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
        if std::env::var_os("UD_CLAUSE_DUMP").is_some() {
            let word = |j: Option<usize>| {
                j.and_then(|j| words.get(j).copied().flatten())
                    .map_or("|".to_string(), |x| format!("{}/{:?}", x.form, x.gold))
            };
            eprintln!(
                "CLAUSE {:?} gold {:?} freq {:?} | {} [{}] {}",
                rule,
                w.gold,
                freq,
                word(i.checked_sub(1)),
                w.form,
                word(Some(i + 1))
            );
        }
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
