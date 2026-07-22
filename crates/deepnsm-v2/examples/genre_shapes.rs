//! `genre_shapes` — the ANTI-COLORBLINDNESS probe: does the D-SRS-2 shape
//! detector generalize beyond the KJV, or did it overfit the Bible's unusually
//! forest-shaped `begat` genealogies?
//!
//! Runs the shape detector across several public-domain genres (Gutenberg,
//! never committed) through ONE shared 20k academic vocabulary, so the
//! per-predicate shape classes are comparable across corpora. The thesis
//! (`literature-probe-ladder-v1`): different literature produces different
//! graph shapes — genealogy/scripture skews Forest (RadixTrie), narrative
//! skews Cyclic hub-verbs (BoundedEscalate) — and the detector must classify
//! each correctly, not just the Bible.
//!
//! Run (texts + the 20k vocab are local-only, never committed):
//! ```sh
//! cargo run --example genre_shapes -- kjv=/tmp/pg10.txt milton=/tmp/pg26.txt …
//! ```
//!
//! The 20k academic vocab (`../deepnsm/word_frequency/academic_20k.csv`, AVL-
//! derived) is read at RUNTIME for local analysis only — its word list is
//! never embedded in or emitted by this example (license: unverified, do not
//! redistribute). Only derived shape statistics are printed.

use deepnsm_v2::shape::{detect_all, Representation, ShapeClass};
use deepnsm_v2::{parse_to_spo, PaletteVocab, Pos, Spo, Tagged};
use std::collections::HashMap;

/// COCA/AVL PoS letter → the FSM's coarse tag (shared with `bible_wave`).
fn coca_pos(letter: &str) -> Pos {
    match letter {
        "n" | "p" => Pos::Noun,
        "v" => Pos::Verb,
        "j" => Pos::Adj,
        "a" | "d" => Pos::Det,
        _ => Pos::Other,
    }
}

/// Archaic/poetic fallback for forms absent from the modern AVL list (Milton,
/// Homer, KJV share these). Documented heuristics, not a tagger.
fn archaic_pos(w: &str) -> Option<Pos> {
    match w {
        "thou" | "thee" | "ye" => Some(Pos::Noun),
        "thy" | "thine" => Some(Pos::Det),
        "shalt" | "hath" | "doth" | "saith" | "spake" | "begat" | "art" | "wilt" | "hast"
        | "shall" | "cometh" | "wast" | "didst" | "hadst" => Some(Pos::Verb),
        _ => (w.ends_with("eth") || w.ends_with("est")).then_some(Pos::Verb),
    }
}

/// Strip the Gutenberg header/footer; return the body between the START/END
/// markers (or the whole text if unmarked).
fn gutenberg_body(raw: &str) -> &str {
    let start = raw
        .find("*** START")
        .and_then(|i| raw[i..].find('\n').map(|j| i + j + 1))
        .unwrap_or(0);
    let end = raw[start..]
        .find("*** END")
        .map_or(raw.len(), |i| start + i);
    &raw[start..end]
}

fn main() {
    // ── the shared 20k academic vocab + PoS (local-only) ──
    let vocab_csv = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../deepnsm/word_frequency/academic_20k.csv"
    ))
    .expect("academic_20k.csv (sibling deepnsm crate, local-only)");
    let mut vocab = PaletteVocab::new();
    let mut pos_of: HashMap<String, Pos> = HashMap::new();
    // header: ID,band,status,word,Pos,...
    let mut words: Vec<String> = Vec::new();
    for line in vocab_csv.lines().skip(1) {
        let mut f = line.split(',');
        let (_, _, _, Some(word), Some(pos)) = (f.next(), f.next(), f.next(), f.next(), f.next())
        else {
            continue;
        };
        let w = word.to_lowercase();
        if w.is_empty() {
            continue;
        }
        pos_of.entry(w.clone()).or_insert_with(|| coca_pos(pos));
        words.push(w);
    }
    vocab.from_frequency_ranked(words.iter().map(String::as_str));
    println!("shared vocab: {} academic words\n", vocab.len());

    // ── per-genre shape census ──
    println!(
        "{:<10} {:>7} {:>6} {:>6} | {:>6} {:>5} {:>6} {:>6} {:>6} | {:>5} top-forest predicate (amortization)",
        "genre", "triples", "subj", "preds", "empty", "flat", "forest", "dag", "cyclic", "trie%",
    );
    println!("{}", "-".repeat(108));

    for arg in std::env::args().skip(1) {
        let (label, path) = arg.split_once('=').unwrap_or(("corpus", arg.as_str()));
        let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("read {path}: {e}");
        });
        let body = gutenberg_body(&raw);

        // General prose → SPO: split into sentences on . ! ? then FSM each.
        let mut all: Vec<Spo> = Vec::new();
        let mut tagged: Vec<Tagged> = Vec::new();
        let flush = |tagged: &mut Vec<Tagged>, all: &mut Vec<Spo>| {
            tagged.push(Tagged::new(0, Pos::Stop));
            all.extend(parse_to_spo(tagged));
            tagged.clear();
        };
        for tok in body.split_whitespace() {
            let ends = tok.ends_with('.') || tok.ends_with('!') || tok.ends_with('?');
            let w: String = tok
                .chars()
                .filter(char::is_ascii_alphabetic)
                .collect::<String>()
                .to_lowercase();
            if w.len() >= 2 {
                if let Some(id) = vocab.id(&w) {
                    let pos = pos_of
                        .get(&w)
                        .copied()
                        .or_else(|| archaic_pos(&w))
                        .unwrap_or(Pos::Other);
                    tagged.push(Tagged::new(id, pos));
                }
            }
            if ends {
                flush(&mut tagged, &mut all);
            }
        }
        if !tagged.is_empty() {
            flush(&mut tagged, &mut all);
        }

        // Shape census.
        let census = detect_all(&all);
        let mut cls = [0usize; 5]; // Empty, Cyclic, Flat, Forest, Dag
        for r in &census {
            let i = match r.class {
                ShapeClass::Empty => 0,
                ShapeClass::Cyclic => 1,
                ShapeClass::Flat => 2,
                ShapeClass::Forest => 3,
                ShapeClass::Dag => 4,
            };
            cls[i] += 1;
        }
        let trie_routed = census
            .iter()
            .filter(|r| {
                matches!(
                    r.recommend,
                    Representation::RadixTrie | Representation::TriePlusEscalate
                )
            })
            .count();
        let trie_pct = 100.0 * trie_routed as f64 / census.len().max(1) as f64;

        // Highest-edge Forest predicate + its trie amortization (the
        // genealogy-shaped signal).
        let top_forest = census
            .iter()
            .find(|r| r.class == ShapeClass::Forest)
            .map(|r| {
                let edges: Vec<(u16, u16)> = all
                    .iter()
                    .filter(|t| t.predicate == r.predicate)
                    .map(|t| (t.subject, t.object))
                    .collect();
                let trie = deepnsm_v2::FamilyTrie::build(&edges);
                let amort = if trie.covered() > 0 {
                    trie.ancestor_pairs().len() as f64 / trie.covered() as f64
                } else {
                    0.0
                };
                (vocab.word(r.predicate).unwrap_or("?").to_string(), amort)
            });

        let subjects: std::collections::HashSet<u16> = all.iter().map(|t| t.subject).collect();
        println!(
            "{:<10} {:>7} {:>6} {:>6} | {:>6} {:>5} {:>6} {:>6} {:>6} | {:>4.0}% {}",
            label,
            all.len(),
            subjects.len(),
            census.len(),
            cls[0],
            cls[2],
            cls[3],
            cls[4],
            cls[1],
            trie_pct,
            top_forest
                .map(|(w, a)| format!("'{w}' ({a:.1}x)"))
                .unwrap_or_else(|| "—".into())
        );
    }

    println!(
        "\nRead: empty/flat = closure adds nothing (EdgeTable); forest = RadixTrie \
         (ancestry→key); dag = MaterializedFabric/TriePlusEscalate; cyclic = BoundedEscalate.\n\
         The detector classifies each genre's OWN shapes — it is not Bible-specific."
    );
}
