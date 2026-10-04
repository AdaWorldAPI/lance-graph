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
//! - **R4** relative pronoun after a comma, case taken from train tokens under
//!   the SAME surface condition (a relativizer form right after a comma), plus
//!   any token marked `PronType=Rel`. Mining on `PronType=Rel` alone fires
//!   nothing on German GSD, which carries no `PronType=Rel` at all.
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
//! - R4: precision <= the plain form baseline.
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
        };
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
                if let Some(lemma) = self.nearest_verb(s, i) {
                    let key = (t.form.clone(), lemma.clone());
                    self.wechsel_key.entry(key).or_default()[c] += 1;
                }
            }
        }
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
}
