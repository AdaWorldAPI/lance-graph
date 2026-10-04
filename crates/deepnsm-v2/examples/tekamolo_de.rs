//! `tekamolo_de` — German TEKAMOLO cue words decided by **verb position**,
//! calibrated against frequency, on Universal Dependencies German.
//!
//! The ambiguous German cues (`da`, `als`, `während`, `bis`, `wenn`, …) are a
//! subordinator (`weil`-like, Kausal/Temporal clause), a preposition (`seit
//! 1990`, `als Kind`) or an adverb (`da` = there/then, Lokal/Temporal). German
//! word order separates them by where the finite verb stands:
//!
//! - **V2** — a finite verb right after the cue: the cue fills the Vorfeld, an
//!   adverb (`Da kommt er`);
//! - **verb-final** — the segment the cue opens (to the next punctuation) ends
//!   in a finite verb: a subordinate clause (`da er krank war`);
//! - **noun phrase** — anything else: a preposition (`seit dem Krieg`).
//!
//! P(class | signature) is learned over every cue token of train (the cue word
//! is not part of it), P(class | word) is the word's train frequency, and the
//! two are combined as log-odds against the class prior. Gold is the UD
//! relation (`mark` / `case` / other). Nothing at test reads gold: finiteness
//! is the share of train tokens of the form marked `VerbForm=Fin` (≥ ½), and
//! case is kept (German capitalises nouns; see `ud_pos_eval`).
//!
//! ```text
//! T=https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/r2.15
//! curl -O $T/de_gsd-ud-train.conllu -O $T/de_gsd-ud-test.conllu
//! cargo run --release --example tekamolo_de -- de_gsd-ud-train.conllu de_gsd-ud-test.conllu
//! ```

use std::collections::{HashMap, HashSet};

/// The ambiguous cue words scored.
const CUES: [&str; 11] = [
    "da", "so", "wenn", "als", "während", "seit", "bis", "nachdem", "ob", "weil", "damit",
];

/// Subordinator, preposition, adverb.
const CLASSES: [&str; 3] = ["mark", "case", "adv"];

struct Word {
    form: String,
    finite: bool,
    punct: bool,
    rel: String,
}

fn read(path: &str) -> Vec<Vec<Word>> {
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
        cur.push(Word {
            form: c[1].to_string(),
            finite: c[5].split('|').any(|f| f == "VerbForm=Fin"),
            punct: c[3] == "PUNCT",
            rel: c[7].to_string(),
        });
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The gold class of a cue token, or `None` if its relation is not one a cue
/// takes.
fn class(w: &Word) -> Option<usize> {
    match w.rel.as_str() {
        "mark" => Some(0),
        "case" => Some(1),
        "advmod" | "cc" => Some(2),
        _ => None,
    }
}

/// What train knows about forms, measured — never gold at test.
struct Lexicon {
    finite: HashSet<String>,
    punct: HashSet<String>,
}

impl Lexicon {
    fn from_train(train: &[Vec<Word>]) -> Self {
        let mut seen: HashMap<&str, (usize, usize)> = HashMap::new();
        let mut punct = HashSet::new();
        for w in train.iter().flatten() {
            let e = seen.entry(&w.form).or_default();
            e.0 += 1;
            e.1 += usize::from(w.finite);
            if w.punct {
                punct.insert(w.form.clone());
            }
        }
        let finite = seen
            .into_iter()
            .filter(|(_, (n, f))| 2 * f >= *n)
            .map(|(w, _)| w.to_string())
            .collect();
        Self { finite, punct }
    }

    /// The verb-position signature of the cue at `i`: V2, verb-final, noun
    /// phrase, or sentence edge.
    fn signature(&self, s: &[Word], i: usize) -> usize {
        let fin = |w: &Word| self.finite.contains(&w.form);
        if s.get(i + 1).is_some_and(fin) {
            return 0;
        }
        let end = (i + 1..s.len())
            .find(|&j| self.punct.contains(&s[j].form))
            .unwrap_or(s.len());
        match s[i + 1..end].last() {
            Some(last) if fin(last) => 1,
            Some(_) => 2,
            None => 3,
        }
    }
}

const SIGNATURES: [&str; 4] = ["V2", "verb-final", "noun phrase", "edge"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [train, test] = args.as_slice() else {
        eprintln!("usage: tekamolo_de TRAIN.conllu TEST.conllu");
        std::process::exit(2);
    };
    let train = read(train);
    let test = read(test);
    let lex = Lexicon::from_train(&train);

    let is_cue = |w: &Word| CUES.contains(&w.form.to_lowercase().as_str());
    let mut by_word: HashMap<String, [usize; 3]> = HashMap::new();
    let mut by_sig = [[0usize; 3]; 4];
    let mut prior = [0usize; 3];
    for s in &train {
        for (i, w) in s.iter().enumerate() {
            let Some(c) = class(w).filter(|_| is_cue(w)) else {
                continue;
            };
            by_word.entry(w.form.to_lowercase()).or_default()[c] += 1;
            by_sig[lex.signature(s, i)][c] += 1;
            prior[c] += 1;
        }
    }
    let p = |cnt: &[usize; 3], c: usize| {
        (cnt[c] as f64 + 0.5) / (cnt.iter().sum::<usize>() as f64 + 1.5)
    };
    let argmax = |f: &dyn Fn(usize) -> f64| (0..3).max_by(|&a, &b| f(a).total_cmp(&f(b))).unwrap();

    let (mut n, mut pos, mut freq, mut comb) = (0, 0, 0, 0);
    let mut per: HashMap<String, [usize; 4]> = HashMap::new();
    for s in &test {
        for (i, w) in s.iter().enumerate() {
            let Some(gold) = class(w).filter(|_| is_cue(w)) else {
                continue;
            };
            let word = w.form.to_lowercase();
            let wc = by_word.get(&word).copied().unwrap_or_default();
            let sc = by_sig[lex.signature(s, i)];
            let fp = argmax(&|c| wc[c] as f64);
            let pp = argmax(&|c| sc[c] as f64);
            let cp = argmax(&|c| p(&wc, c).ln() + p(&sc, c).ln() - p(&prior, c).ln());
            n += 1;
            pos += usize::from(pp == gold);
            freq += usize::from(fp == gold);
            comb += usize::from(cp == gold);
            let r = per.entry(word).or_default();
            r[0] += 1;
            r[1] += usize::from(pp == gold);
            r[2] += usize::from(fp == gold);
            r[3] += usize::from(cp == gold);
        }
    }
    let pct = |x: usize, d: usize| {
        if d == 0 {
            0.0
        } else {
            100.0 * x as f64 / d as f64
        }
    };
    println!("{} finite forms measured from train", lex.finite.len());
    for (k, name) in SIGNATURES.iter().enumerate() {
        let c = by_sig[k];
        println!(
            "  train signature {name:11}: {} {}, {} {}, {} {}",
            CLASSES[0], c[0], CLASSES[1], c[1], CLASSES[2], c[2]
        );
    }
    let mut words: Vec<_> = per.into_iter().collect();
    words.sort();
    for (w, r) in words {
        println!(
            "  {w:9} {:3}: position {:.1}%, frequency {:.1}%, combined {:.1}%",
            r[0],
            pct(r[1], r[0]),
            pct(r[2], r[0]),
            pct(r[3], r[0])
        );
    }
    println!(
        "all {n}: position {:.1}%, frequency {:.1}%, position × frequency {:.1}%",
        pct(pos, n),
        pct(freq, n),
        pct(comb, n)
    );
}
