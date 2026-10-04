//! `ud_gender_eval` — the gender of a German noun never seen in train.
//!
//! German makes gender recoverable from structure:
//! - a compound takes the gender of its last part
//!   (*die Verpackung* → *die Lebensmittelverpackung*);
//! - a nominalised infinitive is always neuter (*das Essen*, *beim Laufen*);
//! - an inflected form shares its lemma's gender (*Firmen* → *Firma*).
//!
//! Each method is scored on its own, then as a cascade. The scored unit is a
//! test NOUN with a single `Gender=` in gold whose form never occurs as a
//! train NOUN. Gender is read from train only.
//!
//! Methods:
//! - **lemma**: an external form → lemma list (DeReKo-2014 STT, `NN` rows)
//!   maps the form to a lemma, whose gender comes from the train lemmas.
//!   Optional: pass the DeReKo `.freq` path as a third argument. The list is
//!   CC BY-NC 3.0 and is never committed (see `data/README.md`).
//! - **infinitive**: the lowercased form is a verb lemma (train VERB lemmas)
//!   ending in *-en* / *-n* → Neut.
//! - **compound**: the longest proper suffix of at least 3 letters that is a
//!   train noun form → that noun's gender.
//! - **stem**: the German Snowball stem (`frostem`, the stemmer
//!   tesseract-paperless search uses) → the majority gender of train nouns
//!   with that stem.
//!
//! KILL bars (fixed before the run): compound precision < 0.90; infinitive
//! precision < 0.95; stem precision < 0.85.
//!
//! ```text
//! cargo run --release --example ud_gender_eval -- de_hdt-ud-train-a-1.conllu de_hdt-ud-test.conllu [DeReKo-2014-II-MainArchive-STT.100000.freq]
//! ```

use std::collections::{HashMap, HashSet};

const GENDERS: [&str; 3] = ["Masc", "Fem", "Neut"];

struct Noun {
    form: String,
    lemma: String,
    gender: Option<usize>,
}

/// (NOUN tokens, VERB lemmas) of a CoNLL-U file.
fn read(path: &str) -> (Vec<Noun>, HashSet<String>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut nouns = Vec::new();
    let mut verbs = HashSet::new();
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        if line.starts_with('#') || c.len() < 8 || c[0].contains(['-', '.']) {
            continue;
        }
        match c[3] {
            "NOUN" => nouns.push(Noun {
                form: c[1].to_lowercase(),
                lemma: c[2].to_lowercase(),
                gender: c[5]
                    .split('|')
                    .find_map(|f| f.strip_prefix("Gender="))
                    .and_then(|v| GENDERS.iter().position(|g| *g == v)),
            }),
            "VERB" => {
                verbs.insert(c[2].to_lowercase());
            }
            _ => {}
        }
    }
    (nouns, verbs)
}

fn majority(c: &[usize; 3]) -> Option<usize> {
    (c.iter().sum::<usize>() > 0).then(|| (0..3).max_by_key(|&k| (c[k], std::cmp::Reverse(k))))?
}

#[derive(Default, Clone, Copy)]
struct Tally {
    fires: usize,
    right: usize,
}

impl Tally {
    fn add(&mut self, ok: bool) {
        self.fires += 1;
        self.right += usize::from(ok);
    }
    fn p(&self) -> f64 {
        100.0 * self.right as f64 / self.fires.max(1) as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (train_path, test_path, dereko) = match args.as_slice() {
        [a, b] => (a, b, None),
        [a, b, c] => (a, b, Some(c)),
        _ => {
            eprintln!("usage: ud_gender_eval TRAIN.conllu TEST.conllu [DEREKO.freq]");
            std::process::exit(2);
        }
    };
    let (train, verbs) = read(train_path);
    let (test, _) = read(test_path);
    let stemmer = frostem::Stemmer::new(frostem::Algorithm::German);

    let mut by_form: HashMap<String, [usize; 3]> = HashMap::new();
    let mut by_lemma: HashMap<String, [usize; 3]> = HashMap::new();
    let mut by_stem: HashMap<String, [usize; 3]> = HashMap::new();
    let mut seen: HashSet<String> = HashSet::new();
    for n in &train {
        seen.insert(n.form.clone());
        if let Some(g) = n.gender {
            by_form.entry(n.form.clone()).or_default()[g] += 1;
            by_lemma.entry(n.lemma.clone()).or_default()[g] += 1;
            by_stem
                .entry(stemmer.stem(&n.form).into_owned())
                .or_default()[g] += 1;
        }
    }
    // DeReKo NN rows: form → lemma (highest-frequency row wins).
    let mut form_lemma: HashMap<String, (String, f64)> = HashMap::new();
    if let Some(p) = dereko {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        for line in text.lines() {
            let c: Vec<&str> = line.split('\t').collect();
            let [form, lemma, "NN", freq] = c.as_slice() else {
                continue;
            };
            let f: f64 = freq.parse().unwrap_or(0.0);
            let e = form_lemma
                .entry(form.to_lowercase())
                .or_insert((lemma.to_lowercase(), f));
            if f > e.1 {
                *e = (lemma.to_lowercase(), f);
            }
        }
    }

    let lemma_g = |form: &str| -> Option<usize> {
        let (l, _) = form_lemma.get(form)?;
        by_lemma.get(l).and_then(majority)
    };
    let infinitive_g = |form: &str| -> Option<usize> {
        (form.ends_with('n') && verbs.contains(form)).then_some(2)
    };
    let compound_g = |form: &str| -> Option<usize> {
        let chars: Vec<char> = form.chars().collect();
        (1..chars.len().saturating_sub(2)).find_map(|k| {
            let tail: String = chars[k..].iter().collect();
            by_form.get(&tail).and_then(majority)
        })
    };
    let stem_g = |form: &str| by_stem.get(stemmer.stem(form).as_ref()).and_then(majority);

    let mut unseen = 0usize;
    let mut method = [Tally::default(); 4];
    let mut cascade = Tally::default();
    let mut baseline = [0usize; 3];
    for n in &test {
        let Some(g) = n.gender else { continue };
        if seen.contains(&n.form) {
            continue;
        }
        unseen += 1;
        baseline[g] += 1;
        let preds = [
            lemma_g(&n.form),
            infinitive_g(&n.form),
            compound_g(&n.form),
            stem_g(&n.form),
        ];
        for (t, p) in method.iter_mut().zip(preds) {
            if let Some(p) = p {
                t.add(p == g);
            }
        }
        if let Some(p) = preds.into_iter().flatten().next() {
            cascade.add(p == g);
        }
    }

    println!(
        "train: {} nouns, {} forms with gender, {} verb lemmas; DeReKo NN forms: {}",
        train.len(),
        by_form.len(),
        verbs.len(),
        form_lemma.len()
    );
    let maj = majority(&baseline).unwrap_or(0);
    println!(
        "test nouns unseen in train (single gold gender): {unseen}; majority gender {} = {:.1}%",
        GENDERS[maj],
        100.0 * baseline[maj] as f64 / unseen.max(1) as f64
    );
    let bars = [None, Some(95.0), Some(90.0), Some(85.0)];
    let names = [
        "lemma (DeReKo)",
        "infinitive -> Neut",
        "compound head",
        "stem",
    ];
    for ((name, t), bar) in names.iter().zip(method).zip(bars) {
        let verdict = match bar {
            Some(b) if t.fires == 0 || t.p() < b => format!("KILL (< {b:.0}%)"),
            Some(b) => format!("PASS (>= {b:.0}%)"),
            None => String::new(),
        };
        println!(
            "  {name:19}: covers {:5} ({:5.1}%)  precision {:5.1}%  {verdict}",
            t.fires,
            100.0 * t.fires as f64 / unseen.max(1) as f64,
            t.p()
        );
    }
    println!(
        "  cascade (lemma, infinitive, compound, stem): covers {:.1}%  precision {:.1}%",
        100.0 * cascade.fires as f64 / unseen.max(1) as f64,
        cascade.p()
    );
}
