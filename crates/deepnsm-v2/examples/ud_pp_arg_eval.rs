//! `ud_pp_arg_eval` — is a prepositional phrase governed by the verb
//! (*sich freuen **auf***, *bestehen **auf***, *denken **an***) or a free
//! adverbial (*auf dem Tisch*, *an dem Tag*)?
//!
//! A verb-governed phrase (a prepositional object) is an argument, so it fills no
//! TEKAMOLO lane. Every one detected takes a phrase out of the lane puzzle
//! (the TEKAMOLO "Sudoku"). Its case is also fixed by the verb, not by place
//! vs direction.
//!
//! Tables are mined from **train only**:
//! - preposition forms (majority UPOS ADP);
//! - verb form → lemma (UPOS VERB);
//! - (verb lemma, preposition) → (object count, adverbial count), read from
//!   the gold tree: the ADP's head noun is `obj` (object) or `obl`
//!   (adverbial), and that noun's head is the verb;
//! - (verb lemma, preposition) → case counts of the governed noun, over the
//!   object phrases.
//!
//! **Gold.** German HDT encodes the prepositional object as `obj` with a
//! `case` ADP child (*rechnen mit*, *setzen auf*, *warten auf*). German
//! `obl:arg` is NOT this: it marks bare dative objects (*der Telekom*,
//! *ihr*), and carries a preposition only once in GSD train. German GSD has
//! no prepositional-object label at all (its PP objects are plain `obl`), so
//! this probe scores HDT only; on GSD the positive class is empty.
//!
//! Test-time prediction reads only surface forms. The clause is the span
//! between punctuation tokens. For a preposition, every known verb form in
//! that clause is a candidate, and the best candidate's argument rate
//! (>= `MIN_PAIR` train occurrences) decides. Gold (deprel, head, Case) is
//! read on test only to score.
//!
//! Units:
//! - **A1** every test token whose form is a mined preposition and whose gold
//!   head noun is `obl` or `obj` (the verbal PPs, the TEKAMOLO pool).
//!   Gold positive = `obj`.
//! - **A2** A1 positives whose preposition is a Wechsel preposition. Scored
//!   on the phrase's case: the first `Case=` from the preposition to its gold
//!   head noun (HDT often marks it on the article only).
//!
//! KILL bars (fixed before the run):
//! - A1: precision < 0.70, or recall < 0.30, or F1 <= the preposition-only
//!   baseline (argument iff the preposition's own train rate >= 0.5).
//! - A2: precision <= the preposition's majority case on the same tokens.
//!
//! ```text
//! cargo run --release --example ud_pp_arg_eval -- de_hdt-ud-train-a-1.conllu de_hdt-ud-test.conllu
//! ```

use std::collections::HashMap;

const CASES: [&str; 4] = ["Nom", "Acc", "Dat", "Gen"];
const WECHSEL: [&str; 9] = [
    "an", "auf", "hinter", "in", "neben", "über", "unter", "vor", "zwischen",
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

/// Fewest train occurrences of a (verb, preposition) pair for it to vote.
const MIN_PAIR: usize = 3;
/// Argument rate at or above which a phrase is predicted verb-governed.
const ARG_RATE: f64 = 0.5;

struct Tok {
    form: String,
    word: bool,
    lemma: String,
    upos: String,
    case: Option<usize>,
    /// Index of the head word, `None` for the root.
    head: Option<usize>,
    deprel: String,
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
        let head: usize = c[6].parse().unwrap_or(0);
        cur.push(Tok {
            form: c[1].to_lowercase(),
            word: c[1].chars().any(char::is_alphanumeric),
            lemma: c[2].to_string(),
            upos: c[3].to_string(),
            case: case_of(c[5]),
            head: head.checked_sub(1),
            deprel: c[7].to_string(),
        });
    }
    if !cur.is_empty() {
        out.push(cur);
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

/// Gold reading of the preposition at `i`: (head-noun index, is a
/// prepositional object, governing verb index), when the head noun is a
/// verbal `obl` (adverbial) or `obj` (object) dependent.
fn gold_pp(s: &[Tok], i: usize) -> Option<(usize, bool, usize)> {
    if s[i].deprel != "case" {
        return None;
    }
    let n = s[i].head?;
    let rel = s[n].deprel.as_str();
    if rel != "obl" && rel != "obj" {
        return None;
    }
    Some((n, rel == "obj", s[n].head?))
}

/// The phrase's case: the first `Case=` between the preposition at `i` and
/// its head noun `n`, inclusive. HDT often marks it on the article only.
fn phrase_case(s: &[Tok], i: usize, n: usize) -> Option<usize> {
    if n <= i {
        return s[n].case;
    }
    s[i + 1..=n].iter().find_map(|t| t.case)
}

struct Tables {
    preps: std::collections::HashSet<String>,
    verb_lemma: HashMap<String, String>,
    /// (verb lemma, preposition) → [object, adverbial].
    pair: HashMap<(String, String), [usize; 2]>,
    /// Preposition → [object, adverbial].
    prep_rate: HashMap<String, [usize; 2]>,
    /// (verb lemma, preposition) → case counts over object phrases.
    pair_case: HashMap<(String, String), [usize; 4]>,
    /// Preposition → case counts over every verbal PP head noun.
    prep_case: HashMap<String, [usize; 4]>,
}

impl Tables {
    fn mine(train: &[Vec<Tok>]) -> Self {
        let mut adp: HashMap<String, [usize; 2]> = HashMap::new();
        let mut verbs: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut pair: HashMap<(String, String), [usize; 2]> = HashMap::new();
        let mut prep_rate: HashMap<String, [usize; 2]> = HashMap::new();
        let mut pair_case: HashMap<(String, String), [usize; 4]> = HashMap::new();
        let mut prep_case: HashMap<String, [usize; 4]> = HashMap::new();
        for s in train {
            for (i, t) in s.iter().enumerate() {
                adp.entry(t.form.clone()).or_default()[usize::from(t.upos != "ADP")] += 1;
                if t.upos == "VERB" {
                    *verbs
                        .entry(t.form.clone())
                        .or_default()
                        .entry(t.lemma.clone())
                        .or_default() += 1;
                }
                let Some((n, arg, v)) = gold_pp(s, i) else {
                    continue;
                };
                let slot = usize::from(!arg);
                prep_rate.entry(t.form.clone()).or_default()[slot] += 1;
                if let Some(c) = phrase_case(s, i, n) {
                    prep_case.entry(t.form.clone()).or_default()[c] += 1;
                }
                if s[v].upos != "VERB" {
                    continue;
                }
                let key = (s[v].lemma.clone(), t.form.clone());
                pair.entry(key.clone()).or_default()[slot] += 1;
                if let (true, Some(c)) = (arg, phrase_case(s, i, n)) {
                    pair_case.entry(key).or_default()[c] += 1;
                }
            }
        }
        let preps = adp
            .into_iter()
            .filter(|(_, c)| c[0] > c[1])
            .map(|(f, _)| f)
            .collect();
        let verb_lemma = verbs
            .into_iter()
            .filter_map(|(f, m)| {
                m.into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                    .map(|(l, _)| (f, l))
            })
            .collect();
        Self {
            preps,
            verb_lemma,
            pair,
            prep_rate,
            pair_case,
            prep_case,
        }
    }

    /// The best clause verb for the preposition at `i`: (lemma, argument
    /// rate), over pairs seen at least `MIN_PAIR` times in train.
    fn best_verb(&self, s: &[Tok], i: usize) -> Option<(&String, f64)> {
        let lo = (0..i).rev().find(|&k| !s[k].word).map_or(0, |k| k + 1);
        let hi = (i + 1..s.len()).find(|&k| !s[k].word).unwrap_or(s.len());
        (lo..hi)
            .filter(|&k| k != i)
            .filter_map(|k| self.verb_lemma.get(&s[k].form))
            .filter_map(|l| {
                let c = self.pair.get(&(l.clone(), s[i].form.clone()))?;
                let n = c[0] + c[1];
                (n >= MIN_PAIR).then(|| (l, c[0] as f64 / n as f64))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    fn prep_says_arg(&self, prep: &str) -> bool {
        self.prep_rate
            .get(prep)
            .is_some_and(|c| c[0] as f64 / (c[0] + c[1]).max(1) as f64 >= ARG_RATE)
    }
}

#[derive(Default)]
struct Pr {
    tp: usize,
    fp: usize,
    fn_: usize,
}

impl Pr {
    fn add(&mut self, pred: bool, gold: bool) {
        match (pred, gold) {
            (true, true) => self.tp += 1,
            (true, false) => self.fp += 1,
            (false, true) => self.fn_ += 1,
            _ => {}
        }
    }
    fn p(&self) -> f64 {
        self.tp as f64 / (self.tp + self.fp).max(1) as f64
    }
    fn r(&self) -> f64 {
        self.tp as f64 / (self.tp + self.fn_).max(1) as f64
    }
    fn f1(&self) -> f64 {
        let (p, r) = (self.p(), self.r());
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
    fn line(&self) -> String {
        format!(
            "precision {:5.1}%  recall {:5.1}%  F1 {:5.3}  (tp {} fp {} fn {})",
            100.0 * self.p(),
            100.0 * self.r(),
            self.f1(),
            self.tp,
            self.fp,
            self.fn_
        )
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
        eprintln!("usage: ud_pp_arg_eval TRAIN.conllu TEST.conllu");
        std::process::exit(2);
    };
    let train = read(train_path);
    let test = read(test_path);
    let t = Tables::mine(&train);

    let mut pool = 0usize;
    let mut gold_args = 0usize;
    let mut a1 = Pr::default();
    let mut base = Pr::default();
    let mut a2 = [0usize; 3]; // [fires, pair-case right, prep-majority right]
    let mut per_prep: HashMap<String, Pr> = HashMap::new();
    // [concrete, abstract] head → [objects, total].
    let mut by_head = [[0usize; 2]; 2];

    for s in &test {
        for i in 0..s.len() {
            if !t.preps.contains(&s[i].form) {
                continue;
            }
            let Some((n, gold, _)) = gold_pp(s, i) else {
                continue;
            };
            pool += 1;
            gold_args += usize::from(gold);
            let h = &mut by_head[usize::from(abstract_noun(&s[n].form))];
            h[0] += usize::from(gold);
            h[1] += 1;
            let best = t.best_verb(s, i);
            let pred = best.is_some_and(|(_, r)| r >= ARG_RATE);
            a1.add(pred, gold);
            base.add(t.prep_says_arg(&s[i].form), gold);
            per_prep
                .entry(s[i].form.clone())
                .or_default()
                .add(pred, gold);
            if !(pred && gold && WECHSEL.contains(&s[i].form.as_str())) {
                continue;
            }
            let Some(g) = phrase_case(s, i, n) else {
                continue;
            };
            let (lemma, _) = best.expect("pred implies a best verb");
            let by_pair = t
                .pair_case
                .get(&(lemma.clone(), s[i].form.clone()))
                .and_then(majority);
            let by_prep = t.prep_case.get(&s[i].form).and_then(majority);
            if let Some(c) = by_pair {
                a2[0] += 1;
                a2[1] += usize::from(c == g);
                a2[2] += usize::from(by_prep == Some(g));
            }
        }
    }

    println!(
        "mined from train: {} prepositions, {} verb forms, {} (verb, prep) pairs",
        t.preps.len(),
        t.verb_lemma.len(),
        t.pair.len()
    );
    println!(
        "test pool (verbal PPs, obl + obj): {pool}, of which prepositional objects {gold_args} ({:.1}%)",
        100.0 * gold_args as f64 / pool.max(1) as f64
    );

    println!("\nA1 verb-governed vs adverbial (KILL: P < 0.70, R < 0.30, or F1 <= prep-only)");
    println!("  clause verb : {}", a1.line());
    println!("  prep only   : {}", base.line());
    let a1_pass = a1.p() >= 0.70 && a1.r() >= 0.30 && a1.f1() > base.f1();
    println!("  A1 {}", verdict(a1_pass));
    println!(
        "  TEKAMOLO pool: {} of {} adverbials kept, {} of {} arguments removed",
        pool - gold_args - a1.fp,
        pool - gold_args,
        a1.tp,
        gold_args
    );
    let mut rows: Vec<(&String, &Pr)> = per_prep.iter().collect();
    rows.sort_by_key(|(_, p)| std::cmp::Reverse(p.tp + p.fn_));
    println!("  by preposition (most gold arguments first):");
    for (prep, p) in rows.into_iter().take(8) {
        println!("    {prep:8} {}", p.line());
    }

    for (k, name) in ["other head", "abstract head"].iter().enumerate() {
        let [o, n] = by_head[k];
        println!(
            "  {name:13}: {n} phrases, {:.1}% prepositional objects",
            100.0 * o as f64 / n.max(1) as f64
        );
    }

    println!("\nA2 case of a verb-governed Wechsel phrase (KILL: precision <= prep majority)");
    let pct = |x: usize| 100.0 * x as f64 / a2[0].max(1) as f64;
    println!(
        "  fires {}  (verb, prep) case {:5.1}%  prep majority {:5.1}%",
        a2[0],
        pct(a2[1]),
        pct(a2[2])
    );
    println!("  A2 {}", verdict(a2[0] > 0 && a2[1] > a2[2]));
}
