//! `ud_case_eval` — the German CASE evaluator (Q2 of
//! `.claude/knowledge/german-grammar-rule-inventory.md`, ranks 5-8).
//!
//! Rules predict the UD feature `Case=` (Nom / Acc / Dat / Gen) of a token from
//! its surface form, its position and tables mined from **train only**. Gold
//! (`Case=`, `PronType=`, UPOS, LEMMA) is read from train to mine the tables and
//! from test **only to score**; no test-time prediction reads it. A rule either
//! predicts a case for a token or abstains. The scored unit is a test token that
//! carries `Case=` in gold.
//!
//! Rules (each reported separately):
//!
//! - **R1** fixed-case prepositions (closed Acc / Dat / Gen lists); the case is
//!   predicted for the tokens in the noun-phrase window after the preposition.
//! - **R2** article decidability: per lowercased DET/PRON form, train purity of
//!   its case, tiered Decisive (>= 0.95), Dominant (>= 0.80), Ambiguous.
//! - **R3** Wechselpraeposition: contractions decide alone (am/im = Dat,
//!   ans/ins/aufs = Acc); otherwise the article decides when the first following
//!   token has exactly one of Acc/Dat in train, else a (preposition, verb lemma)
//!   prior, else the preposition's own majority.
//! - **R3t** time vs place: a Wechsel preposition whose head noun is a time
//!   noun reads temporally (*in der Nacht*, *vor dem Essen*, *über das
//!   Wochenende*), and the case comes from a (preposition, temporal) table
//!   instead of the verb. Time nouns are mined from train as the head nouns
//!   after *seit* / *während*, the prepositions with only a time reading; no
//!   hand list. Scored on the Wechsel tokens the article leaves undecided.
//! - **R3a** abstract head (*-ung, -heit, -keit, -schaft, -tion, -nis, -ität,
//!   -ismus, -tum*): the same table keyed by the abstract class. Abstract
//!   nouns read as goal or topic (*auf Erweiterung setzen*), not as time.
//! - **R4** relative pronoun after a comma, case taken from train tokens under
//!   the SAME surface condition (a relativizer form right after a comma), plus
//!   any token marked `PronType=Rel`. Mining on `PronType=Rel` alone fires
//!   nothing on German GSD, which carries no `PronType=Rel` at all.
//! - **R6** gender and number: an article's case read from (article form,
//!   head-noun gender, head-noun number). Gender and number come from a
//!   train-mined noun lexicon; an unseen noun falls back to its German
//!   Snowball stem (`frostem`, the stemmer tesseract-paperless search uses).
//!   Also reported: how much case ambiguity the cell removes compared with the
//!   article form alone (train purity).
//! - **R5** combined, first rule that fires in the order R3-contraction, R1, R3,
//!   R4, R2; coverage, precision, baseline and the Nom/Acc/Dat/Gen confusion.
//!
//! The baseline everywhere is the train-majority case of the token's lowercased
//! form over all train tokens with a case; an unseen form abstains and counts
//! wrong. Note: in GSD, contractions such as `zum`, `am` are multiword tokens
//! whose syntactic words are `zu dem`, `an dem`; this reader keeps the words and
//! skips the ranges, so the contraction entries may never fire (reported).
//!
//! KILL bars (from the research map):
//!
//! - R1: precision <= baseline on the same tokens, or Acc/Dat < 0.90, or
//!   Gen < 0.70.
//! - R3: verb-prior precision <= prep-majority on the same tokens, or verb-prior
//!   fires on < 5 % of the Wechsel-governed scored tokens (the first following
//!   token of a non-contraction Wechsel preposition).
//! - R3t, R3a (each): precision <= prep-majority on the same tokens, or fires on < 5 % of
//!   the article-undecided Wechsel tokens.
//! - R4: precision <= the plain form baseline.
//! - R6: precision < the article-form majority + 5 points on the same tokens;
//!   stem-fallback precision < seen-noun precision - 10 points.
//!
//! ```text
//! cargo run --release --example ud_case_eval -- de_gsd-ud-train.conllu de_gsd-ud-test.conllu
//! ```

use std::collections::HashMap;

const CASES: [&str; 4] = ["Nom", "Acc", "Dat", "Gen"];

const ACC: [&str; 5] = ["durch", "für", "gegen", "ohne", "um"];
const DAT: [&str; 12] = [
    "aus",
    "bei",
    "mit",
    "nach",
    "seit",
    "von",
    "zu",
    "gegenüber",
    "zum",
    "zur",
    "beim",
    "vom",
];
const GEN: [&str; 7] = [
    "wegen",
    "trotz",
    "während",
    "statt",
    "anstatt",
    "innerhalb",
    "außerhalb",
];
/// Wechsel prepositions that need the article or the verb to decide.
const WECHSEL: [&str; 9] = [
    "an", "auf", "hinter", "in", "neben", "über", "unter", "vor", "zwischen",
];
const CONTRACTION_DAT: [&str; 2] = ["am", "im"];
const CONTRACTION_ACC: [&str; 3] = ["ans", "ins", "aufs"];
const REL: [&str; 13] = [
    "der", "die", "das", "dem", "den", "dessen", "deren", "denen", "welcher", "welche", "welches",
    "welchem", "welchen",
];

/// Derivational suffixes of German abstract nouns (inflected forms included).
const ABSTRACT_SUFFIXES: [&str; 18] = [
    "ung", "ungen", "heit", "heiten", "keit", "keiten", "schaft", "schaften", "tion", "tionen",
    "nis", "nisse", "nissen", "ität", "itäten", "ismus", "ismen", "tum",
];

/// The lowercased noun form ends in an abstract-noun suffix.
fn abstract_noun(form: &str) -> bool {
    form.chars().count() > 5 && ABSTRACT_SUFFIXES.iter().any(|x| form.ends_with(x))
}

/// Prepositions with only a time reading; their head nouns seed the time-noun set.
const TEMPORAL_ONLY: [&str; 2] = ["seit", "während"];

const TIERS: [&str; 3] = ["Decisive", "Dominant", "Ambiguous"];
const KINDS: [&str; 3] = ["article-decided", "verb-prior", "prep-majority"];

struct Tok {
    /// Lowercased form.
    form: String,
    /// The surface form starts with an uppercase letter.
    upper: bool,
    /// The form has an alphanumeric character (not punctuation).
    word: bool,
    lemma: String,
    upos: String,
    case: Option<usize>,
    /// `PronType=Rel` in feats.
    rel: bool,
    /// `Gender=` (0 Masc, 1 Fem, 2 Neut) and `Number=` (0 Sing, 1 Plur).
    gender: Option<usize>,
    number: Option<usize>,
}

fn feat(feats: &str, key: &str, values: &[&str]) -> Option<usize> {
    feats
        .split('|')
        .find_map(|f| f.strip_prefix(key))
        .and_then(|v| values.iter().position(|x| *x == v))
}

fn case_of(feats: &str) -> Option<usize> {
    feats
        .split('|')
        .find_map(|f| f.strip_prefix("Case="))
        .and_then(|v| CASES.iter().position(|c| *c == v))
}

fn read(path: &str) -> Vec<Vec<Tok>> {
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
        let c: Vec<&str> = line.split('\t').collect();
        if line.starts_with('#') || c.len() < 8 || c[0].contains(['-', '.']) {
            continue;
        }
        cur.push(Tok {
            form: c[1].to_lowercase(),
            upper: c[1].chars().next().is_some_and(char::is_uppercase),
            word: c[1].chars().any(char::is_alphanumeric),
            lemma: c[2].to_string(),
            upos: c[3].to_string(),
            case: case_of(c[5]),
            rel: c[5].split('|').any(|f| f == "PronType=Rel"),
            gender: feat(c[5], "Gender=", &["Masc", "Fem", "Neut"]),
            number: feat(c[5], "Number=", &["Sing", "Plur"]),
        });
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn in_list(list: &[&str], form: &str) -> bool {
    list.contains(&form)
}

/// The noun-phrase window after the preposition at `i`: up to four following
/// tokens, never through punctuation, ending after the first capitalised token.
fn window(s: &[Tok], i: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut j = i + 1;
    while j < s.len() && j - i <= 4 && s[j].word {
        out.push(j);
        if s[j].upper {
            break;
        }
        j += 1;
    }
    out
}

/// The most frequent case, ties to the lower index; `None` when empty.
fn majority(c: &[usize; 4]) -> Option<usize> {
    if c.iter().sum::<usize>() == 0 {
        return None;
    }
    (0..4).max_by_key(|&k| (c[k], std::cmp::Reverse(k)))
}

/// Everything mined from train.
struct Tables {
    /// Lowercased form → case counts over all train tokens with a case.
    form_case: HashMap<String, [usize; 4]>,
    /// Form → (tier, majority case) for DET/PRON forms.
    det_rule: HashMap<String, (usize, usize)>,
    /// Form → case counts over `PronType=Rel` train tokens.
    rel_case: HashMap<String, [usize; 4]>,
    /// Verb form → majority lemma (upos VERB).
    verb_lemma: HashMap<String, String>,
    /// (preposition, verb lemma) → case counts of the Wechsel-governed token.
    wechsel_key: HashMap<(String, String), [usize; 4]>,
    /// Preposition → case counts of the Wechsel-governed token.
    wechsel_prep: HashMap<String, [usize; 4]>,
    /// Lowercased head-noun forms seen after a time-only preposition.
    time_nouns: std::collections::HashSet<String>,
    /// (preposition, head class: 0 other, 1 time, 2 abstract) → case counts
    /// of the governed token.
    wechsel_time: HashMap<(String, usize), [usize; 4]>,
}

/// The head noun of the window after `i`: its last token when capitalised.
fn head(s: &[Tok], i: usize) -> Option<usize> {
    window(s, i).last().copied().filter(|&j| s[j].upper)
}

impl Tables {
    fn mine(train: &[Vec<Tok>]) -> Self {
        let mut form_case: HashMap<String, [usize; 4]> = HashMap::new();
        let mut det_case: HashMap<String, [usize; 4]> = HashMap::new();
        let mut rel_case: HashMap<String, [usize; 4]> = HashMap::new();
        let mut verb_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for t in train.iter().flatten() {
            if let Some(c) = t.case {
                form_case.entry(t.form.clone()).or_default()[c] += 1;
                if t.upos == "DET" || t.upos == "PRON" {
                    det_case.entry(t.form.clone()).or_default()[c] += 1;
                }
            }
            if t.upos == "VERB" {
                *verb_counts
                    .entry(t.form.clone())
                    .or_default()
                    .entry(t.lemma.clone())
                    .or_default() += 1;
            }
        }
        let verb_lemma = verb_counts
            .into_iter()
            .filter_map(|(f, m)| {
                m.into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                    .map(|(l, _)| (f, l))
            })
            .collect();
        let det_rule = det_case
            .iter()
            .filter_map(|(f, c)| {
                let m = majority(c)?;
                let purity = c[m] as f64 / c.iter().sum::<usize>() as f64;
                let tier = if purity >= 0.95 {
                    0
                } else if purity >= 0.80 {
                    1
                } else {
                    2
                };
                Some((f.clone(), (tier, m)))
            })
            .collect();
        // R4 is mined under its own test-time surface condition.
        for sent in train {
            for (k, t) in sent.iter().enumerate() {
                let after_comma = k > 0 && sent[k - 1].form == ",";
                if let Some(c) = t.case {
                    if (after_comma && in_list(&REL, &t.form)) || t.rel {
                        rel_case.entry(t.form.clone()).or_default()[c] += 1;
                    }
                }
            }
        }
        let mut t = Self {
            form_case,
            det_rule,
            rel_case,
            verb_lemma,
            wechsel_key: HashMap::new(),
            wechsel_prep: HashMap::new(),
            time_nouns: Default::default(),
            wechsel_time: HashMap::new(),
        };
        for s in train {
            for (i, tok) in s.iter().enumerate() {
                if in_list(&TEMPORAL_ONLY, &tok.form) {
                    // Only a noun reached through DET/ADJ, so a number or a
                    // verb never leaks a noun from outside the phrase.
                    let np = |j: usize| {
                        s[j].upos == "NOUN"
                            && s[i + 1..j]
                                .iter()
                                .all(|x| x.upos == "DET" || x.upos == "ADJ")
                    };
                    if let Some(j) = head(s, i).filter(|&j| np(j)) {
                        t.time_nouns.insert(s[j].form.clone());
                    }
                }
            }
        }
        t.mine_wechsel(train);
        t
    }

    /// Wechsel-governed counts from train: the first following token of a
    /// Wechsel preposition, scored against train gold (Acc or Dat only).
    fn mine_wechsel(&mut self, train: &[Vec<Tok>]) {
        for s in train {
            for (i, t) in s.iter().enumerate() {
                if !in_list(&WECHSEL, &t.form) {
                    continue;
                }
                let Some(&j) = window(s, i).first() else {
                    continue;
                };
                let Some(c) = s[j].case.filter(|&c| c == 1 || c == 2) else {
                    continue;
                };
                self.wechsel_prep.entry(t.form.clone()).or_default()[c] += 1;
                let class = self.head_class(s, i);
                self.wechsel_time
                    .entry((t.form.clone(), class))
                    .or_default()[c] += 1;
                if let Some(lemma) = self.nearest_verb(s, i) {
                    let key = (t.form.clone(), lemma.clone());
                    self.wechsel_key.entry(key).or_default()[c] += 1;
                }
            }
        }
    }

    /// The Wechsel preposition at `i` heads a time noun.
    /// Head class of the Wechsel phrase at `i`: 1 time noun, 2 abstract noun
    /// (by suffix, *vor der Fahrt*-type), 0 otherwise. Time wins.
    fn head_class(&self, s: &[Tok], i: usize) -> usize {
        if self.is_time(s, i) {
            1
        } else if head(s, i).is_some_and(|j| abstract_noun(&s[j].form)) {
            2
        } else {
            0
        }
    }

    fn is_time(&self, s: &[Tok], i: usize) -> bool {
        head(s, i).is_some_and(|j| self.time_nouns.contains(&s[j].form))
    }

    /// R3t / R3a: the (preposition, head class) majority, only for a time
    /// (class 1) or abstract (class 2) head.
    fn wechsel_temporal(&self, s: &[Tok], i: usize) -> Option<usize> {
        let class = self.head_class(s, i);
        if class == 0 {
            return None;
        }
        self.wechsel_time
            .get(&(s[i].form.clone(), class))
            .and_then(majority)
    }

    /// The lemma of the nearest preceding token that is a known verb form.
    fn nearest_verb(&self, s: &[Tok], i: usize) -> Option<&String> {
        (0..i).rev().find_map(|k| self.verb_lemma.get(&s[k].form))
    }

    fn baseline(&self, form: &str) -> Option<usize> {
        self.form_case.get(form).and_then(majority)
    }

    fn baseline_ok(&self, form: &str, gold: usize) -> bool {
        self.baseline(form) == Some(gold)
    }

    /// R3 decision for the Wechsel preposition at `i`, predicting token `j`:
    /// (kind, case, prep-majority case).
    fn wechsel(&self, s: &[Tok], i: usize, j: usize) -> Option<(usize, usize, Option<usize>)> {
        let prep = &s[i].form;
        let prep_major = self.wechsel_prep.get(prep).and_then(majority);
        let seen = self.form_case.get(&s[j].form);
        let cands: Vec<usize> = [1usize, 2]
            .into_iter()
            .filter(|&c| seen.is_some_and(|x| x[c] > 0))
            .collect();
        if let [only] = cands.as_slice() {
            return Some((0, *only, prep_major));
        }
        let by_verb = self
            .nearest_verb(s, i)
            .and_then(|l| self.wechsel_key.get(&(prep.clone(), l.clone())))
            .and_then(majority);
        match (by_verb, prep_major) {
            (Some(c), _) => Some((1, c, prep_major)),
            (None, Some(c)) => Some((2, c, prep_major)),
            (None, None) => None,
        }
    }

    /// Every rule's prediction per token of a test sentence. No gold is read.
    fn predict(&self, s: &[Tok]) -> Vec<Pred> {
        let mut preds: Vec<Pred> = s.iter().map(|_| Pred::default()).collect();
        for (i, tok) in s.iter().enumerate() {
            let f = tok.form.as_str();
            let fixed = if in_list(&ACC, f) {
                Some(1)
            } else if in_list(&DAT, f) {
                Some(2)
            } else if in_list(&GEN, f) {
                Some(3)
            } else {
                None
            };
            let contraction = if in_list(&CONTRACTION_DAT, f) {
                Some(2)
            } else if in_list(&CONTRACTION_ACC, f) {
                Some(1)
            } else {
                None
            };
            if fixed.is_some() || contraction.is_some() {
                for j in window(s, i) {
                    if preds[j].r1.is_none() {
                        preds[j].r1 = fixed;
                    }
                    if preds[j].r3c.is_none() {
                        preds[j].r3c = contraction;
                    }
                }
            }
            if in_list(&WECHSEL, f) {
                if let Some(&j) = window(s, i).first() {
                    preds[j].wech = true;
                    if preds[j].r3.is_none() {
                        preds[j].r3 = self.wechsel(s, i, j);
                        preds[j].r3t = self.wechsel_temporal(s, i);
                        preds[j].class = self.head_class(s, i);
                    }
                }
            }
            if i > 0 && s[i - 1].form == "," && in_list(&REL, f) {
                preds[i].r4 = self.rel_case.get(f).and_then(majority);
            }
            preds[i].r2 = self.det_rule.get(f).copied();
        }
        preds
    }
}

#[derive(Default)]
struct Pred {
    r1: Option<usize>,
    r3c: Option<usize>,
    r3: Option<(usize, usize, Option<usize>)>,
    r4: Option<usize>,
    r2: Option<(usize, usize)>,
    /// R3t prediction (time head only).
    r3t: Option<usize>,
    /// Head class of the Wechsel phrase (0 other, 1 time, 2 abstract).
    class: usize,
    /// First following token of a non-contraction Wechsel preposition.
    wech: bool,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    fires: usize,
    correct: usize,
    base: usize,
}

impl Tally {
    fn add(&mut self, ok: bool, base_ok: bool) {
        self.fires += 1;
        self.correct += usize::from(ok);
        self.base += usize::from(base_ok);
    }

    fn precision(&self) -> f64 {
        pct(self.correct, self.fires)
    }

    fn baseline(&self) -> f64 {
        pct(self.base, self.fires)
    }

    fn line(&self) -> String {
        format!(
            "fires {:6}  precision {:5.1}%  baseline {:5.1}%",
            self.fires,
            self.precision(),
            self.baseline()
        )
    }
}

fn pct(x: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * x as f64 / d as f64
    }
}

fn verdict(pass: bool) -> &'static str {
    if pass {
        "PASS"
    } else {
        "KILL"
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [train_path, test_path] = args.as_slice() else {
        eprintln!("usage: ud_case_eval TRAIN.conllu TEST.conllu");
        std::process::exit(2);
    };
    let train = read(train_path);
    let test = read(test_path);
    let tables = Tables::mine(&train);

    let mut scored = 0usize;
    let mut wech_scored = 0usize;
    let mut r1 = [Tally::default(); 4];
    let mut r2 = [Tally::default(); 3];
    let mut r3c = Tally::default();
    let mut r3 = [Tally::default(); 3];
    let mut verb_prep_major = 0usize;
    // Article-undecided Wechsel tokens (R3 kind 1 or 2).
    let mut undecided = 0usize;
    // [time, abstract] head.
    let mut r3t = [Tally::default(); 2];
    let mut r3t_prep_major = [0usize; 2];
    // Verb-prior split by reading: [place, time].
    let mut verb_by_reading = [Tally::default(); 3];
    // R3t on the time-head tokens where the verb prior fired.
    let mut r3t_on_verb = Tally::default();
    let mut r4 = Tally::default();
    let mut r5 = Tally::default();
    let mut confusion = [[0usize; 4]; 4];

    for s in &test {
        let preds = tables.predict(s);
        for (t, p) in s.iter().zip(&preds) {
            let Some(gold) = t.case else {
                continue;
            };
            scored += 1;
            let base = tables.baseline_ok(&t.form, gold);
            if p.wech {
                wech_scored += 1;
            }
            if let Some(c) = p.r1 {
                r1[c].add(c == gold, base);
            }
            if let Some(c) = p.r3c {
                r3c.add(c == gold, base);
            }
            if let Some((kind, c, prep_major)) = p.r3 {
                r3[kind].add(c == gold, base);
                if kind == 1 && prep_major == Some(gold) {
                    verb_prep_major += 1;
                }
                if kind == 1 {
                    verb_by_reading[p.class].add(c == gold, base);
                    if let (1, Some(t)) = (p.class, p.r3t) {
                        r3t_on_verb.add(t == gold, base);
                    }
                }
                if kind != 0 {
                    undecided += 1;
                    if let Some(t) = p.r3t {
                        r3t[p.class - 1].add(t == gold, base);
                        if prep_major == Some(gold) {
                            r3t_prep_major[p.class - 1] += 1;
                        }
                    }
                }
            }
            if let Some(c) = p.r4 {
                r4.add(c == gold, base);
            }
            if let Some((tier, c)) = p.r2 {
                r2[tier].add(c == gold, base);
            }
            let pick = p
                .r3c
                .or(p.r1)
                .or(p.r3.map(|d| d.1))
                .or(p.r4)
                .or(p.r2.map(|d| d.1));
            if let Some(c) = pick {
                r5.add(c == gold, base);
                confusion[gold][c] += 1;
            }
        }
    }

    let ntrain: usize = train.iter().map(Vec::len).sum();
    let ntest: usize = test.iter().map(Vec::len).sum();
    println!("train {ntrain} tokens, test {ntest} tokens, scored (Case= in gold) {scored}");
    println!(
        "mined from train: {} forms with a case, {} DET/PRON forms, {} verb forms, {} (prep, lemma) keys",
        tables.form_case.len(),
        tables.det_rule.len(),
        tables.verb_lemma.len(),
        tables.wechsel_key.len()
    );

    println!(
        "\nR1 fixed-case prepositions (KILL: precision <= baseline, Acc/Dat < 0.90, Gen < 0.70)"
    );
    let mut r1_pass = true;
    for c in [1usize, 2, 3] {
        let t = r1[c];
        let floor = if c == 3 { 70.0 } else { 90.0 };
        let pass = t.fires > 0 && t.precision() > t.baseline() && t.precision() >= floor;
        r1_pass &= pass;
        println!("  {} list: {}  {}", CASES[c], t.line(), verdict(pass));
    }
    println!("  R1 {}", verdict(r1_pass));

    println!("\nR2 article decidability (no KILL bar; baseline = form majority)");
    for (k, name) in TIERS.iter().enumerate() {
        println!("  {name:9}: {}", r2[k].line());
    }

    println!(
        "\nR3 Wechselpraeposition (KILL: verb-prior precision <= prep-majority, or verb-prior fires < 5% of {wech_scored} Wechsel-governed scored tokens)"
    );
    println!("  contraction     : {}", r3c.line());
    for (k, name) in KINDS.iter().enumerate() {
        println!("  {name:15} : {}", r3[k].line());
    }
    let verb = r3[1];
    let prep_major_precision = pct(verb_prep_major, verb.fires);
    println!(
        "  prep-majority on the verb-prior tokens: precision {prep_major_precision:5.1}%  verb-prior share of Wechsel-governed {:5.1}%",
        pct(verb.fires, wech_scored)
    );
    let r3_pass =
        verb.fires > 0 && verb.precision() > prep_major_precision && verb.fires * 20 >= wech_scored;
    println!("  R3 {}", verdict(r3_pass));

    println!(
        "\nR3t time vs place (KILL: precision <= prep-majority on the same tokens, or fires < 5% of {undecided} article-undecided Wechsel tokens)"
    );
    println!("  time nouns mined from train: {}", tables.time_nouns.len());
    let mut keys: Vec<_> = tables.wechsel_time.iter().collect();
    keys.sort();
    for ((prep, time), c) in keys {
        println!(
            "    {prep:9} {:5}  Acc {:6}  Dat {:6}",
            ["other", "time", "abstract"][*time],
            c[1],
            c[2]
        );
    }
    println!("  verb-prior, other head   : {}", verb_by_reading[0].line());
    println!("  verb-prior, time head    : {}", verb_by_reading[1].line());
    println!("  verb-prior, abstract head: {}", verb_by_reading[2].line());
    println!("  R3t on those time heads: {}", r3t_on_verb.line());
    let mut sample: Vec<&String> = tables.time_nouns.iter().collect();
    sample.sort();
    let step = (sample.len() / 15).max(1);
    let shown: Vec<&str> = sample.iter().step_by(step).map(|w| w.as_str()).collect();
    println!("  time-noun sample: {}", shown.join(" "));
    for (k, name) in ["R3t (time head)", "R3a (abstract head)"]
        .iter()
        .enumerate()
    {
        let t = r3t[k];
        let major = pct(r3t_prep_major[k], t.fires);
        println!(
            "  {name}: {}  prep-majority on the same tokens {major:5.1}%  share {:5.1}%",
            t.line(),
            pct(t.fires, undecided)
        );
        println!(
            "  {name} {}",
            verdict(t.fires > 0 && t.precision() > major && t.fires * 20 >= undecided)
        );
    }

    println!("\nR4 relative pronoun after a comma (KILL: precision <= form baseline)");
    println!("  {}", r4.line());
    println!(
        "  R4 {}",
        verdict(r4.fires > 0 && r4.precision() > r4.baseline())
    );

    println!("\nR5 combined (R3-contraction, R1, R3, R4, R2)");
    println!(
        "  coverage {:5.1}% ({} of {scored})  {}",
        pct(r5.fires, scored),
        r5.fires,
        r5.line()
    );
    println!("  confusion over covered tokens (rows gold, columns predicted)");
    println!(
        "        {:>7} {:>7} {:>7} {:>7}",
        "Nom", "Acc", "Dat", "Gen"
    );
    for (g, row) in confusion.iter().enumerate() {
        println!(
            "  {:>5} {:>7} {:>7} {:>7} {:>7}",
            CASES[g], row[0], row[1], row[2], row[3]
        );
    }

    r6_report(&train, &test);
}

/// (gender, number) → index into a 6-cell table.
fn gn(g: usize, n: usize) -> usize {
    g * 2 + n
}

/// The head noun after the article at `i`: the first NOUN-shaped
/// (capitalised) token within three tokens, through words only.
fn article_head(s: &[Tok], i: usize) -> Option<usize> {
    (i + 1..s.len().min(i + 4))
        .take_while(|&j| s[j].word)
        .find(|&j| s[j].upper)
}

/// R6: case from (article form, head gender, head number).
fn r6_report(train: &[Vec<Tok>], test: &[Vec<Tok>]) {
    let stemmer = frostem::Stemmer::new(frostem::Algorithm::German);
    let majority6 = |c: &[usize; 6]| (0..6).max_by_key(|&k| (c[k], std::cmp::Reverse(k)));
    // Noun lexicon: form → (gender, number) counts; stem → counts.
    let mut by_form: HashMap<String, [usize; 6]> = HashMap::new();
    let mut by_stem: HashMap<String, [usize; 6]> = HashMap::new();
    // (article form, gn cell) → case counts; article form → case counts.
    let mut cell: HashMap<(String, usize), [usize; 4]> = HashMap::new();
    let mut form_only: HashMap<String, [usize; 4]> = HashMap::new();
    // (article form, gender) → case counts: the stem keeps gender but erases
    // number (*Firma* / *Firmen*), so the exploratory fallback reads gender only.
    let mut cell_g: HashMap<(String, usize), [usize; 4]> = HashMap::new();
    let is_article = |t: &Tok| t.upos == "DET" && t.case.is_some();
    for s in train {
        for t in s {
            if let (true, Some(g), Some(n)) = (t.upos == "NOUN", t.gender, t.number) {
                by_form.entry(t.form.clone()).or_default()[gn(g, n)] += 1;
                by_stem
                    .entry(stemmer.stem(&t.form).into_owned())
                    .or_default()[gn(g, n)] += 1;
            }
        }
        for (i, t) in s.iter().enumerate() {
            if !is_article(t) {
                continue;
            }
            let c = t.case.expect("article has a case");
            form_only.entry(t.form.clone()).or_default()[c] += 1;
            let Some(j) = article_head(s, i) else {
                continue;
            };
            if let (Some(g), Some(n)) = (s[j].gender, s[j].number) {
                cell.entry((t.form.clone(), gn(g, n))).or_default()[c] += 1;
            }
            if let Some(g) = s[j].gender {
                cell_g.entry((t.form.clone(), g)).or_default()[c] += 1;
            }
        }
    }
    // Ambiguity: train purity (majority share), weighted by count.
    let purity = |m: &mut dyn Iterator<Item = &[usize; 4]>| {
        let (mut top, mut all) = (0usize, 0usize);
        for c in m {
            top += c.iter().max().copied().unwrap_or(0);
            all += c.iter().sum::<usize>();
        }
        100.0 * top as f64 / all.max(1) as f64
    };
    println!("\nR6 gender and number (KILL: precision < form majority + 5 pts; stem fallback < seen - 10 pts)");
    println!(
        "  noun lexicon: {} forms, {} stems; train case purity: article form alone {:.1}%, form + gender + number {:.1}%",
        by_form.len(),
        by_stem.len(),
        purity(&mut form_only.values()),
        purity(&mut cell.values())
    );
    // [seen form, stem fallback] → (fires, R6 right, form-majority right).
    let mut tally = [[0usize; 3]; 2];
    let mut no_gn = 0usize;
    // Exploratory (after the first run): stem fallback with gender only.
    let mut stem_g = [0usize; 3];
    let mut arts = 0usize;
    for s in test {
        for (i, t) in s.iter().enumerate() {
            let (true, Some(gold)) = (t.upos == "DET", t.case) else {
                continue;
            };
            arts += 1;
            let Some(j) = article_head(s, i) else {
                no_gn += 1;
                continue;
            };
            if !by_form.contains_key(&s[j].form) {
                let st = stemmer.stem(&s[j].form);
                let g = by_stem.get(st.as_ref()).and_then(|c| {
                    (0..3).max_by_key(|&g| (c[gn(g, 0)] + c[gn(g, 1)], std::cmp::Reverse(g)))
                });
                if let Some(pred) = g
                    .and_then(|g| cell_g.get(&(t.form.clone(), g)))
                    .and_then(majority)
                {
                    let base = form_only.get(&t.form).and_then(majority);
                    stem_g[0] += 1;
                    stem_g[1] += usize::from(pred == gold);
                    stem_g[2] += usize::from(base == Some(gold));
                }
            }
            let (src, counts) = match by_form.get(&s[j].form) {
                Some(c) => (0, Some(c)),
                None => (1, by_stem.get(stemmer.stem(&s[j].form).as_ref())),
            };
            let Some(k) = counts.and_then(majority6) else {
                no_gn += 1;
                continue;
            };
            let Some(pred) = cell.get(&(t.form.clone(), k)).and_then(majority) else {
                no_gn += 1;
                continue;
            };
            let base = form_only.get(&t.form).and_then(majority);
            tally[src][0] += 1;
            tally[src][1] += usize::from(pred == gold);
            tally[src][2] += usize::from(base == Some(gold));
        }
    }
    let p = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    let mut pass = true;
    for (k, name) in ["seen noun", "stem fallback"].iter().enumerate() {
        let [f, r, b] = tally[k];
        println!(
            "  {name:13}: fires {f:6}  R6 {:5.1}%  article-form majority {:5.1}%",
            p(r, f),
            p(b, f)
        );
    }
    println!(
        "  stem fallback, gender only (exploratory): fires {:6}  R6 {:5.1}%  article-form majority {:5.1}%",
        stem_g[0],
        p(stem_g[1], stem_g[0]),
        p(stem_g[2], stem_g[0])
    );
    let [f, r, b] = [0, 1, 2].map(|x| tally[0][x] + tally[1][x]);
    pass &= p(r, f) >= p(b, f) + 5.0;
    pass &= p(tally[1][1], tally[1][0]) >= p(tally[0][1], tally[0][0]) - 10.0;
    println!(
        "  all          : fires {f:6} of {arts} articles ({:.1}%)  R6 {:5.1}%  article-form majority {:5.1}%  no gender/number {no_gn}",
        p(f, arts),
        p(r, f),
        p(b, f)
    );
    println!("  R6 {}", verdict(pass));
}
