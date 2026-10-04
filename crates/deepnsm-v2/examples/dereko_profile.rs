//! `dereko_profile` — how well does the contemporary German frequency list
//! (DeReKo-2014, about 7 billion tokens) describe a text?
//!
//! For each text it reports:
//! - **coverage**: the share of tokens whose surface form is in the list, and
//!   the share of word types missing from it;
//! - **rank agreement**: the Spearman correlation between the text's
//!   frequency ranks and DeReKo's, over the text's 1,000 most frequent types
//!   that the list knows;
//! - **POS ambiguity**: the share of tokens whose form has two or more STTS
//!   tags, each carrying at least 10 % of its DeReKo frequency;
//! - **ADJD share**: tokens whose dominant tag is ADJD, the German adjective
//!   used predicatively or adverbially (*früh*, *schnell*).
//!
//! Inputs: Project Gutenberg `.txt` (boilerplate stripped), getbible-style
//! Bible `.json`, or UD `.conllu`. DeReKo is CC BY-NC 3.0 and is never
//! committed; it is passed as a path (see `data/README.md`).
//!
//! ```text
//! cargo run --release --example dereko_profile -- DeReKo-2014-II-MainArchive-STT.100000.freq TEXT...
//! ```

use std::collections::HashMap;

/// Per form: total frequency and per-STTS-tag frequency.
struct Entry {
    total: f64,
    tags: HashMap<String, f64>,
}

fn load_dereko(path: &str) -> (HashMap<String, Entry>, HashMap<String, usize>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut forms: HashMap<String, Entry> = HashMap::new();
    for line in text.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let [form, _lemma, tag, freq] = c.as_slice() else {
            continue;
        };
        let f: f64 = freq.parse().unwrap_or(0.0);
        let e = forms.entry((*form).to_string()).or_insert(Entry {
            total: 0.0,
            tags: HashMap::new(),
        });
        e.total += f;
        *e.tags.entry((*tag).to_string()).or_default() += f;
    }
    let mut by_freq: Vec<(&String, f64)> = forms.iter().map(|(k, e)| (k, e.total)).collect();
    by_freq.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let rank = by_freq
        .into_iter()
        .enumerate()
        .map(|(i, (k, _))| (k.clone(), i))
        .collect();
    (forms, rank)
}

/// The running text of one input file.
fn text_of(path: &str) -> String {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    if path.ends_with(".json") {
        let v: serde_json::Value = serde_json::from_str(&raw).expect("json");
        let mut out = String::new();
        for b in v["books"].as_array().into_iter().flatten() {
            for c in b["chapters"].as_array().into_iter().flatten() {
                for vs in c["verses"].as_array().into_iter().flatten() {
                    out.push_str(vs["text"].as_str().unwrap_or(""));
                    out.push('\n');
                }
            }
        }
        out
    } else if path.ends_with(".conllu") {
        raw.lines()
            .filter_map(|l| {
                let c: Vec<&str> = l.split('\t').collect();
                (!l.starts_with('#') && c.len() > 1 && !c[0].contains(['-', '.'])).then(|| c[1])
            })
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        // Project Gutenberg: keep the body between the START and END markers.
        let start = raw
            .find("*** START")
            .and_then(|i| raw[i..].find('\n').map(|j| i + j))
            .unwrap_or(0);
        let end = raw.find("*** END").unwrap_or(raw.len());
        raw[start..end.max(start)].to_string()
    }
}

/// Word tokens: maximal alphabetic runs, case kept (German nouns).
fn tokens(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Spearman correlation of two rank lists of equal length.
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let rank = |x: &[f64]| {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
        let mut r = vec![0.0; x.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let d2: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - y).powi(2)).sum();
    1.0 - 6.0 * d2 / (n * (n * n - 1.0))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((dereko, texts)) = args.split_first() else {
        eprintln!("usage: dereko_profile DEREKO.freq TEXT...");
        std::process::exit(2);
    };
    let (forms, rank) = load_dereko(dereko);
    // A form found as given, or with its first letter lowercased (a
    // sentence-initial capital).
    let find = |w: &str| -> Option<&String> {
        if let Some((k, _)) = forms.get_key_value(w) {
            return Some(k);
        }
        let mut cs = w.chars();
        let first = cs.next()?;
        let lowered: String = first.to_lowercase().chain(cs).collect();
        forms.get_key_value(&lowered).map(|(k, _)| k)
    };
    println!("DeReKo forms: {}", forms.len());
    println!(
        "{:<34} {:>9} {:>8} {:>9} {:>9} {:>8} {:>7}",
        "text", "tokens", "covered", "type OOV", "spearman", "POS amb", "ADJD"
    );
    for path in texts {
        let text = text_of(path);
        let toks = tokens(&text);
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for t in &toks {
            *counts.entry(t).or_default() += 1;
        }
        let (mut covered, mut ambiguous, mut adjd) = (0usize, 0usize, 0usize);
        for t in &toks {
            let Some(k) = find(t) else { continue };
            covered += 1;
            let e = &forms[k];
            let strong = e.tags.values().filter(|&&f| f >= 0.1 * e.total).count();
            ambiguous += usize::from(strong >= 2);
            let top = e.tags.iter().max_by(|a, b| a.1.total_cmp(b.1));
            adjd += usize::from(top.is_some_and(|(t, _)| t == "ADJD"));
        }
        let oov_types = counts.keys().filter(|w| find(w).is_none()).count();
        let mut by_count: Vec<(&&str, &usize)> = counts.iter().collect();
        by_count.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let (mut text_rank, mut ref_rank) = (Vec::new(), Vec::new());
        for (i, (w, _)) in by_count.iter().enumerate() {
            if text_rank.len() >= 1000 {
                break;
            }
            if let Some(k) = find(w) {
                text_rank.push(i as f64);
                ref_rank.push(rank[k] as f64);
            }
        }
        let name = std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path);
        let n = toks.len().max(1) as f64;
        println!(
            "{:<34} {:>9} {:>7.1}% {:>8.1}% {:>9.3} {:>7.1}% {:>6.2}%",
            name,
            toks.len(),
            100.0 * covered as f64 / n,
            100.0 * oov_types as f64 / counts.len().max(1) as f64,
            spearman(&text_rank, &ref_rank),
            100.0 * ambiguous as f64 / n,
            100.0 * adjd as f64 / n
        );
    }
}
