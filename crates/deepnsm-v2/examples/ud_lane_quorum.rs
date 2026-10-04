//! `ud_lane_quorum` — TIME vs PLACE vs FIG for a German Wechsel phrase, by a
//! quorum of position, adverb, noun and verb evidence.
//!
//! A Wechsel phrase (*an, auf, hinter, in, neben, über, unter, vor,
//! zwischen*) reads as one of three lanes:
//! - **TIME**: *vor der Fahrt*, *in 12 Monaten*;
//! - **PLACE**: a literal physical place or direction, *vor dem Haus*;
//! - **FIG**: everything else, i.e. a verb-governed object (*setzen auf*) or a
//!   metaphorical place (*unter Druck*, *auf dem Markt*).
//!
//! Operator hypothesis: an abstract head is TIME or FIG, never PLACE.
//!
//! **Labels are silver**, written by independent model labelers reading the
//! full sentence (`examples/data/hdt_wechsel_lanes.tsv`, ids only, no HDT
//! text). Each test item has two labelers; only items they agree on are
//! scored, and Cohen's κ is reported. German UD has no lane gold.
//!
//! Voters, read from surface forms and tables mined from **train**:
//! - **noun**: the head (first capitalised token of the phrase) is a time
//!   noun (heads after *seit* / *während* / *bis*, nouns used as bare
//!   adverbials, or a compound ending in one), an abstract noun (suffix), a
//!   name (train majority PROPN), or other;
//! - **number**: the phrase holds a digit;
//! - **adverb**: a temporal adverb within two tokens before the preposition
//!   or right after the phrase (*früh am Morgen*). Temporal adverbs are mined
//!   by lift: ADV, or ADJ used as `advmod` (*früh*, *spät*), occurring at
//!   least 2x more often in clauses with a time-only preposition;
//! - **verb**: the (verb lemma, preposition) object rate from HDT `obj` + ADP,
//!   checked against every verb in the clause (`ud_pp_arg_eval`);
//! - **position** (TEKAMOLO): Vorfeld (clause-initial), and whether another
//!   preposition follows in the same clause (temporal phrases come earlier);
//! - **article case**: the article form leaves only Acc, only Dat, or both;
//! - **preposition**: one-hot (the prior).
//!
//! The quorum is a multinomial logistic regression over the one-hot voter
//! outputs, fitted on the train labels. Each single voter is fitted the same
//! way (that voter + preposition) as its own baseline.
//!
//! KILL bars (fixed before the labels existed):
//! - κ < 0.6: the silver gold is unreliable, and the result is reported as
//!   such;
//! - quorum accuracy < best single voter + 2 points;
//! - TIME F1 < 0.80, or PLACE F1 < 0.60;
//! - hypothesis: more than 5 % of abstract-head items are labelled PLACE.
//!
//! `LANE_SPLIT=confirm` scores the fresh confirmation sample instead of the
//! exploratory `test` sample; `LANE_LAMBDA` overrides the L2 strength.
//!
//! ```text
//! cargo run --release --example ud_lane_quorum -- de_hdt-ud-train-a-1.conllu de_hdt-ud-test.conllu examples/data/hdt_wechsel_lanes.tsv
//! ```

use std::collections::{HashMap, HashSet};

const LANES: [&str; 3] = ["TIME", "PLACE", "FIG"];
const WECHSEL: [&str; 9] = [
    "an", "auf", "hinter", "in", "neben", "über", "unter", "vor", "zwischen",
];
const TEMPORAL_ONLY: [&str; 3] = ["seit", "während", "bis"];
/// Derivational suffixes of German abstract nouns (inflected forms included).
const ABSTRACT_SUFFIXES: [&str; 18] = [
    "ung", "ungen", "heit", "heiten", "keit", "keiten", "schaft", "schaften", "tion", "tionen",
    "nis", "nisse", "nissen", "ität", "itäten", "ismus", "ismen", "tum",
];
/// The temporal cue adverbs of the German codebook builder
/// (`lance-graph-planner/examples/data/de/build_de_codebook.py`, `TEMPORAL`),
/// minus its prepositions and conjunctions: a fixed list, not tuned here.
const CODEBOOK_TEMPORAL: [&str; 25] = [
    "anschließend",
    "bald",
    "damals",
    "danach",
    "dann",
    "früh",
    "gestern",
    "heute",
    "immer",
    "jetzt",
    "jährlich",
    "manchmal",
    "monatlich",
    "morgen",
    "nie",
    "noch",
    "oft",
    "schließlich",
    "schon",
    "spät",
    "täglich",
    "wieder",
    "wöchentlich",
    "zunächst",
    "zuvor",
];

const MIN_PAIR: usize = 3;

struct Tok {
    form: String,
    upper: bool,
    word: bool,
    digit: bool,
    lemma: String,
    upos: String,
    case: Option<usize>,
    head: Option<usize>,
    deprel: String,
}

struct Sent {
    id: String,
    toks: Vec<Tok>,
}

fn read(path: &str) -> Vec<Sent> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let mut id = String::new();
        let mut toks = Vec::new();
        for line in block.lines() {
            if let Some(v) = line.strip_prefix("# sent_id = ") {
                id = v.to_string();
                continue;
            }
            let c: Vec<&str> = line.split('\t').collect();
            if line.starts_with('#') || c.len() < 8 || c[0].contains(['-', '.']) {
                continue;
            }
            let head: usize = c[6].parse().unwrap_or(0);
            let case = c[5]
                .split('|')
                .find_map(|f| f.strip_prefix("Case="))
                .and_then(|v| ["Nom", "Acc", "Dat", "Gen"].iter().position(|x| *x == v));
            toks.push(Tok {
                form: c[1].to_lowercase(),
                upper: c[1].chars().next().is_some_and(char::is_uppercase),
                word: c[1].chars().any(char::is_alphanumeric),
                digit: c[1].chars().any(|ch| ch.is_ascii_digit()),
                lemma: c[2].to_string(),
                upos: c[3].to_string(),
                case,
                head: head.checked_sub(1),
                deprel: c[7].to_string(),
            });
        }
        if !toks.is_empty() {
            out.push(Sent { id, toks });
        }
    }
    out
}

fn abstract_noun(form: &str) -> bool {
    form.chars().count() > 5 && ABSTRACT_SUFFIXES.iter().any(|x| form.ends_with(x))
}

/// The phrase after the preposition at `i`: up to four tokens, never through
/// punctuation, ending after the first capitalised token.
fn phrase(s: &[Tok], i: usize) -> Vec<usize> {
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

fn head(s: &[Tok], i: usize) -> Option<usize> {
    phrase(s, i).last().copied().filter(|&j| s[j].upper)
}

/// Clause span around `i`: between punctuation tokens.
fn clause(s: &[Tok], i: usize) -> (usize, usize) {
    let lo = (0..i).rev().find(|&k| !s[k].word).map_or(0, |k| k + 1);
    let hi = (i + 1..s.len()).find(|&k| !s[k].word).unwrap_or(s.len());
    (lo, hi)
}

struct Tables {
    time_nouns: HashSet<String>,
    time_adverbs: HashSet<String>,
    preps: HashSet<String>,
    verb_lemma: HashMap<String, String>,
    pair: HashMap<(String, String), [usize; 2]>,
    /// Forms whose train majority UPOS is PROPN.
    names: HashSet<String>,
    /// Form → [Acc seen, Dat seen].
    acc_dat: HashMap<String, [usize; 2]>,
}

impl Tables {
    fn mine(train: &[Sent]) -> Self {
        let mut t = Tables {
            time_nouns: HashSet::new(),
            time_adverbs: HashSet::new(),
            preps: HashSet::new(),
            verb_lemma: HashMap::new(),
            pair: HashMap::new(),
            acc_dat: HashMap::new(),
            names: HashSet::new(),
        };
        let mut propn: HashMap<String, [usize; 2]> = HashMap::new();
        let mut adp: HashMap<String, [usize; 2]> = HashMap::new();
        let mut verbs: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for sent in train {
            let s = &sent.toks;
            for (i, tok) in s.iter().enumerate() {
                adp.entry(tok.form.clone()).or_default()[usize::from(tok.upos != "ADP")] += 1;
                propn.entry(tok.form.clone()).or_default()[usize::from(tok.upos != "PROPN")] += 1;
                match tok.case {
                    Some(1) => t.acc_dat.entry(tok.form.clone()).or_default()[0] += 1,
                    Some(2) => t.acc_dat.entry(tok.form.clone()).or_default()[1] += 1,
                    _ => {}
                }
                if tok.upos == "VERB" {
                    *verbs
                        .entry(tok.form.clone())
                        .or_default()
                        .entry(tok.lemma.clone())
                        .or_default() += 1;
                }
                if TEMPORAL_ONLY.contains(&tok.form.as_str()) {
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
                if tok.deprel != "case" {
                    continue;
                }
                let Some(n) = tok.head else { continue };
                let rel = s[n].deprel.as_str();
                if rel != "obl" && rel != "obj" {
                    continue;
                }
                let Some(v) = s[n].head else { continue };
                if s[v].upos == "VERB" {
                    t.pair
                        .entry((s[v].lemma.clone(), tok.form.clone()))
                        .or_default()[usize::from(rel != "obj")] += 1;
                }
            }
        }
        // A noun used as an adverbial with no preposition (`obl` without a
        // `case` child: *letzte Woche*, *jeden Tag*) is almost always temporal.
        // Exactly `obl`: German `obl:arg` is bare dative objects. No `nummod`
        // child: measure phrases (*5 Prozent*, *100 Dollar*) are bare `obl` too.
        for sent in train {
            let s = &sent.toks;
            for (n, tok) in s.iter().enumerate() {
                let child = |rel: &str| s.iter().any(|x| x.head == Some(n) && x.deprel == rel);
                if tok.upos == "NOUN" && tok.deprel == "obl" && !child("case") && !child("nummod") {
                    t.time_nouns.insert(tok.form.clone());
                }
            }
        }
        // Temporal adverbs by lift: an adverbial word (ADV, or an uninflected
        // ADJ used as `advmod` such as *früh*, *spät*) that occurs far more
        // often in clauses holding a time-only preposition than in clauses
        // overall. Mining "the word right before seit/während" caught focus
        // particles (*auch seit*, *nur bis*) instead.
        let mut in_time = HashMap::<String, usize>::new();
        let mut overall = HashMap::<String, usize>::new();
        let (mut time_clauses, mut clauses) = (0usize, 0usize);
        for sent in train {
            let s = &sent.toks;
            let mut lo = 0;
            while lo < s.len() {
                let (_, hi) = clause(s, lo);
                let span = &s[lo..hi];
                let timed = span
                    .iter()
                    .any(|x| TEMPORAL_ONLY.contains(&x.form.as_str()));
                clauses += 1;
                time_clauses += usize::from(timed);
                for x in span {
                    let adverbial = x.upos == "ADV" || (x.upos == "ADJ" && x.deprel == "advmod");
                    if adverbial {
                        *overall.entry(x.form.clone()).or_default() += 1;
                        if timed {
                            *in_time.entry(x.form.clone()).or_default() += 1;
                        }
                    }
                }
                lo = hi + 1;
            }
        }
        for (form, &k) in &in_time {
            let all = overall[form];
            let lift =
                (k as f64 / time_clauses.max(1) as f64) / (all as f64 / clauses.max(1) as f64);
            if k >= 3 && lift >= 2.0 {
                t.time_adverbs.insert(form.clone());
            }
        }
        t.names = propn
            .into_iter()
            .filter(|(_, c)| c[0] > c[1])
            .map(|(f, _)| f)
            .collect();
        t.preps = adp
            .into_iter()
            .filter(|(_, c)| c[0] > c[1])
            .map(|(f, _)| f)
            .collect();
        t.verb_lemma = verbs
            .into_iter()
            .filter_map(|(f, m)| {
                m.into_iter()
                    .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                    .map(|(l, _)| (f, l))
            })
            .collect();
        t
    }

    /// A time noun, or a compound whose last part is one (*Geschäftsquartal*,
    /// *Wahltag*): German compounds take their meaning from the rightmost part.
    fn is_time_noun(&self, form: &str) -> bool {
        if self.time_nouns.contains(form) {
            return true;
        }
        let chars: Vec<char> = form.chars().collect();
        (1..chars.len().saturating_sub(2)).any(|k| {
            let tail: String = chars[k..].iter().collect();
            tail.chars().count() >= 3 && self.time_nouns.contains(&tail)
        })
    }

    fn verb_governed(&self, s: &[Tok], i: usize) -> bool {
        let (lo, hi) = clause(s, i);
        (lo..hi)
            .filter(|&k| k != i)
            .filter_map(|k| self.verb_lemma.get(&s[k].form))
            .filter_map(|l| {
                let c = self.pair.get(&(l.clone(), s[i].form.clone()))?;
                let n = c[0] + c[1];
                (n >= MIN_PAIR).then(|| c[0] as f64 / n as f64)
            })
            .any(|r| r >= 0.5)
    }

    /// Every voter's output for the preposition at `i`, as named one-hot
    /// features grouped by voter.
    fn voters(&self, s: &[Tok], i: usize) -> Vec<(&'static str, String)> {
        let h = head(s, i);
        let noun = match h {
            Some(j) if self.is_time_noun(&s[j].form) => "time",
            Some(j) if abstract_noun(&s[j].form) => "abstract",
            Some(j) if self.names.contains(&s[j].form) => "name",
            Some(_) => "other",
            None => "none",
        };
        let ph = phrase(s, i);
        let number = ph.iter().any(|&j| s[j].digit);
        let after = ph.last().map_or(i + 1, |&j| j + 1);
        let adverb = [i.wrapping_sub(2), i.wrapping_sub(1), after]
            .into_iter()
            .filter(|&k| k < s.len())
            .any(|k| {
                let f = s[k].form.as_str();
                if std::env::var("LANE_ADVERBS").is_ok_and(|v| v == "codebook") {
                    CODEBOOK_TEMPORAL.contains(&f)
                } else {
                    self.time_adverbs.contains(&s[k].form)
                }
            });
        let verb = self.verb_governed(s, i);
        let (lo, hi) = clause(s, i);
        let vorfeld = i == lo;
        let later_prep = (i + 1..hi)
            .filter(|&k| !ph.contains(&k))
            .any(|k| self.preps.contains(&s[k].form));
        let article = ph
            .first()
            .and_then(|&j| self.acc_dat.get(&s[j].form))
            .map_or("unseen", |c| match (c[0] > 0, c[1] > 0) {
                (true, false) => "acc",
                (false, true) => "dat",
                _ => "both",
            });
        vec![
            ("noun", noun.to_string()),
            ("number", number.to_string()),
            ("adverb", adverb.to_string()),
            ("verb", verb.to_string()),
            (
                "position",
                format!("vorfeld={vorfeld},later_prep={later_prep}"),
            ),
            ("article", format!("{}:{article}", s[i].form)),
            ("prep", s[i].form.clone()),
        ]
    }
}

/// A labeled item: features and lane.
struct Item {
    feats: Vec<(&'static str, String)>,
    lane: usize,
    abstract_head: bool,
    /// `sent_id:token`, for the error dump.
    key: String,
}

/// Multinomial logistic regression over one-hot features.
struct Model {
    index: HashMap<String, usize>,
    w: Vec<[f64; 3]>,
}

impl Model {
    fn keys(item: &Item, groups: &[&str]) -> Vec<String> {
        let mut k: Vec<String> = item
            .feats
            .iter()
            .filter(|(g, _)| groups.contains(g))
            .map(|(g, v)| format!("{g}={v}"))
            .collect();
        k.push("bias".into());
        k
    }

    fn fit(items: &[Item], groups: &[&str]) -> Self {
        let mut index = HashMap::new();
        let rows: Vec<(Vec<usize>, usize)> = items
            .iter()
            .map(|it| {
                let ks = Self::keys(it, groups)
                    .into_iter()
                    .map(|k| {
                        let n = index.len();
                        *index.entry(k).or_insert(n)
                    })
                    .collect();
                (ks, it.lane)
            })
            .collect();
        let mut w = vec![[0.0f64; 3]; index.len()];
        let lambda: f64 = std::env::var("LANE_LAMBDA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-3);
        let lr = 0.5;
        for _ in 0..3000 {
            let mut g = vec![[0.0f64; 3]; w.len()];
            for (ks, y) in &rows {
                let p = Self::softmax(&w, ks);
                for &k in ks {
                    for c in 0..3 {
                        g[k][c] += p[c] - f64::from(u8::from(c == *y));
                    }
                }
            }
            let n = rows.len() as f64;
            for (wk, gk) in w.iter_mut().zip(&g) {
                for c in 0..3 {
                    wk[c] -= lr * (gk[c] / n + lambda * wk[c]);
                }
            }
        }
        Model { index, w }
    }

    fn softmax(w: &[[f64; 3]], ks: &[usize]) -> [f64; 3] {
        let mut z = [0.0; 3];
        for &k in ks {
            for c in 0..3 {
                z[c] += w[k][c];
            }
        }
        let m = z.iter().copied().fold(f64::MIN, f64::max);
        let e = z.map(|v| (v - m).exp());
        let sum: f64 = e.iter().sum();
        e.map(|v| v / sum)
    }

    fn predict(&self, item: &Item, groups: &[&str]) -> usize {
        let ks: Vec<usize> = Self::keys(item, groups)
            .iter()
            .filter_map(|k| self.index.get(k).copied())
            .collect();
        let p = Self::softmax(&self.w, &ks);
        (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(0)
    }
}

fn score(model: &Model, test: &[Item], groups: &[&str]) -> (f64, [f64; 3]) {
    let mut conf = [[0usize; 3]; 3];
    for it in test {
        conf[it.lane][model.predict(it, groups)] += 1;
    }
    let right: usize = (0..3).map(|c| conf[c][c]).sum();
    let f1 = std::array::from_fn(|c| {
        let tp = conf[c][c] as f64;
        let pred: usize = (0..3).map(|g| conf[g][c]).sum();
        let gold: usize = conf[c].iter().sum();
        if tp == 0.0 {
            0.0
        } else {
            2.0 * tp / (pred + gold) as f64
        }
    });
    (right as f64 / test.len().max(1) as f64, f1)
}

fn lane_of(s: &str) -> Option<usize> {
    LANES.iter().position(|l| *l == s)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [train_path, test_path, labels_path] = args.as_slice() else {
        eprintln!("usage: ud_lane_quorum TRAIN.conllu TEST.conllu LABELS.tsv");
        std::process::exit(2);
    };
    let train = read(train_path);
    let test = read(test_path);
    let tables = Tables::mine(&train);
    let by_id = |xs: &[Sent]| -> HashMap<String, usize> {
        xs.iter()
            .enumerate()
            .map(|(k, s)| (s.id.clone(), k))
            .collect()
    };
    let (train_ix, test_ix) = (by_id(&train), by_id(&test));

    let mut fit_items = Vec::new();
    let mut test_items = Vec::new();
    let mut kappa = [[0usize; 3]; 3];
    let mut test_labeled = 0usize;
    // Which held-out split to score: `test` (exploratory) or `confirm` (the
    // fresh, pre-registered round).
    let eval_split = std::env::var("LANE_SPLIT").unwrap_or_else(|_| "test".into());
    let labels = std::fs::read_to_string(labels_path).expect("labels");
    for line in labels.lines().filter(|l| !l.starts_with('#')) {
        let c: Vec<&str> = line.split('\t').collect();
        let [split, sid, tok, a, b] = c.as_slice() else {
            panic!("labels row must have exactly 5 columns: {line}");
        };
        if *split != "train" && *split != eval_split {
            continue;
        }
        let (sents, ix) = if *split == "train" {
            (&train, &train_ix)
        } else {
            (&test, &test_ix)
        };
        let s = &sents[*ix.get(*sid).unwrap_or_else(|| panic!("unknown {sid}"))].toks;
        let i: usize = tok.parse::<usize>().expect("tok") - 1;
        assert!(
            WECHSEL.contains(&s[i].form.as_str()),
            "{sid}:{tok} is not a Wechsel preposition"
        );
        let la = lane_of(a).expect("label a");
        let item = |lane| Item {
            feats: tables.voters(s, i),
            lane,
            abstract_head: head(s, i).is_some_and(|j| abstract_noun(&s[j].form)),
            key: format!("{sid}:{tok}"),
        };
        if *split == "train" {
            fit_items.push(item(la));
        } else {
            test_labeled += 1;
            let lb = lane_of(b).expect("label b");
            kappa[la][lb] += 1;
            if la == lb {
                test_items.push(item(la));
            }
        }
    }

    let n: usize = kappa.iter().flatten().sum();
    let po = (0..3).map(|c| kappa[c][c]).sum::<usize>() as f64 / n.max(1) as f64;
    let pe: f64 = (0..3)
        .map(|c| {
            let ra: usize = kappa[c].iter().sum();
            let rb: usize = (0..3).map(|g| kappa[g][c]).sum();
            ra as f64 * rb as f64 / (n * n).max(1) as f64
        })
        .sum();
    let k = (po - pe) / (1.0 - pe);
    println!(
        "labels: {} train, {test_labeled} test double-labeled, {} agreed ({:.1}%), Cohen kappa {k:.3}  {}",
        fit_items.len(),
        test_items.len(),
        100.0 * po,
        if k >= 0.6 { "RELIABLE" } else { "UNRELIABLE (kappa < 0.6)" }
    );
    let dist = |xs: &[Item]| -> String {
        (0..3)
            .map(|c| format!("{} {}", LANES[c], xs.iter().filter(|x| x.lane == c).count()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("  train lanes: {}", dist(&fit_items));
    println!("  test lanes (agreed): {}", dist(&test_items));
    println!(
        "  mined: {} time nouns, {} temporal adverbs, {} (verb, prep) pairs",
        tables.time_nouns.len(),
        tables.time_adverbs.len(),
        tables.pair.len()
    );

    if std::env::var("LANE_DUMP").is_ok() {
        let mut adv: Vec<&String> = tables.time_adverbs.iter().collect();
        adv.sort();
        eprintln!("temporal adverbs: {adv:?}");
    }

    // Operator hypothesis: an abstract head is never PLACE.
    let abs: Vec<&Item> = fit_items
        .iter()
        .chain(&test_items)
        .filter(|x| x.abstract_head)
        .collect();
    let abs_place = abs.iter().filter(|x| x.lane == 1).count();
    let rate = 100.0 * abs_place as f64 / abs.len().max(1) as f64;
    println!(
        "\nabstract head -> TIME or FIG, never PLACE: {} abstract items, {abs_place} PLACE ({rate:.1}%)  {}",
        abs.len(),
        if rate <= 5.0 { "PASS" } else { "KILL" }
    );
    for (c, name) in LANES.iter().enumerate() {
        let m = abs.iter().filter(|x| x.lane == c).count();
        println!("  {name}: {m}");
    }

    let all = [
        "noun", "number", "adverb", "verb", "position", "article", "prep",
    ];
    println!("\nsingle voters (each with the preposition prior):");
    let mut best = 0.0f64;
    for g in [
        "prep", "noun", "number", "adverb", "verb", "position", "article",
    ] {
        let groups = if g == "prep" {
            vec!["prep"]
        } else {
            vec![g, "prep"]
        };
        let m = Model::fit(&fit_items, &groups);
        let (acc, f1) = score(&m, &test_items, &groups);
        best = best.max(acc);
        println!(
            "  {g:9} accuracy {:5.1}%  F1 TIME {:.3} PLACE {:.3} FIG {:.3}",
            100.0 * acc,
            f1[0],
            f1[1],
            f1[2]
        );
    }
    let m = Model::fit(&fit_items, &all);
    let (acc, f1) = score(&m, &test_items, &all);
    if std::env::var("LANE_DUMP").is_ok() {
        for it in &test_items {
            let p = m.predict(it, &all);
            if p != it.lane {
                eprintln!(
                    "{}\t{}\t{}\t{:?}",
                    it.key, LANES[it.lane], LANES[p], it.feats
                );
            }
        }
    }
    println!(
        "\nquorum (all voters): accuracy {:5.1}%  F1 TIME {:.3} PLACE {:.3} FIG {:.3}",
        100.0 * acc,
        f1[0],
        f1[1],
        f1[2]
    );
    for drop in ["noun", "number", "adverb", "verb", "position", "article"] {
        let groups: Vec<&str> = all.iter().copied().filter(|g| *g != drop).collect();
        let m = Model::fit(&fit_items, &groups);
        let (a, _) = score(&m, &test_items, &groups);
        println!("  without {drop:9}: accuracy {:5.1}%", 100.0 * a);
    }
    let pass = acc >= best + 0.02 && f1[0] >= 0.80 && f1[1] >= 0.60;
    println!(
        "  quorum {} (bars: >= best single + 2 pts ({:.1}%), TIME F1 >= 0.80, PLACE F1 >= 0.60)",
        if pass { "PASS" } else { "KILL" },
        100.0 * (best + 0.02)
    );
}
