//! Real SPO landing: COCA n-gram co-occurrence (ngrams.info samples) → deepnsm
//! COCA-4096 rank → gridlake-4096 cell, truth-weighted by real corpus frequency.
//!
//! Closes the two stand-in gaps the bag-of-words run exposed:
//!   1. bag-of-words → real SPO: `v_the_n.txt` gives (verb PRED, noun OBJ) pairs
//!      (`opened→door`, `solve→problem`); `n_n.txt` gives noun·noun co-occurrence.
//!   2. stopword-cluster → content-word spread: landing verbs/nouns (not glue
//!      words) spreads across the vocab instead of piling into ranks 0..30.
//!
//! The n-gram sample files are LICENSED (ngrams.info / english-corpora.org) —
//! they are read from a local path (argv[1], default /tmp/sources/coca) and are
//! NOT committed. This example is the code; the data stays out of git.
//!
//! Run: cargo run --release -p deepnsm --example gridlake_spo_ngrams [ngram_dir]

use deepnsm::Vocabulary;
use std::path::{Path, PathBuf};

const GRID: usize = 4096;

#[derive(Clone, Copy, Default)]
struct Cell {
    count: u32,   // landings in this COCA cell
    truth_w: u64, // Σ real corpus frequency (the NARS-truth weight)
}

/// Resolve a surface word to its COCA rank via the real deepnsm lemmatizer
/// ("opened" → "open" → rank). None if out-of-vocab.
fn rank_of(vocab: &Vocabulary, word: &str) -> Option<usize> {
    vocab
        .tokenize(word)
        .iter()
        .find(|t| t.is_known())
        .map(|t| t.rank_or_default() as usize)
}

fn ingest(
    vocab: &Vocabulary,
    grid: &mut [Cell],
    path: &Path,
    word_cols: (usize, usize),
    min_fields: usize,
) -> (u64, u64) {
    let (ca, cb) = word_cols;
    let mut rows = 0u64;
    let mut landed = 0u64;
    let Ok(txt) = std::fs::read_to_string(path) else {
        eprintln!("  (missing {} — skipped)", path.display());
        return (0, 0);
    };
    for line in txt.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < min_fields {
            continue;
        }
        // format: rank \t freq \t w... — data rows start with a numeric rank.
        let Ok(freq) = f[1].parse::<u64>() else { continue };
        rows += 1;
        for &col in &[ca, cb] {
            if let Some(w) = f.get(col) {
                if let Some(r) = rank_of(vocab, &w.to_lowercase()) {
                    grid[r].count += 1;
                    grid[r].truth_w += freq;
                    landed += 1;
                }
            }
        }
    }
    (rows, landed)
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let vocab = Vocabulary::load(&Path::new(manifest).join("word_frequency")).expect("load COCA");
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "/tmp/sources/coca".to_string()),
    );

    let mut grid = vec![Cell::default(); GRID];
    println!("── REAL SPO LANDING (ngrams.info COCA samples → gridlake-4096) ──");

    // v_the_n: rank freq verb "the" noun  → (verb@2 PRED, noun@4 OBJ)
    let (vr, vl) = ingest(&vocab, &mut grid, &dir.join("v_the_n.txt"), (2, 4), 5);
    println!("  v_the_n  (verb→noun SPO): {vr} rows → {vl} rank landings");
    // n_n: rank freq noun noun  → (noun@2, noun@3)
    let (nr, nl) = ingest(&vocab, &mut grid, &dir.join("n_n.txt"), (2, 3), 4);
    println!("  n_n      (noun·noun)    : {nr} rows → {nl} rank landings");

    // ── spread analysis: content-word cells vs the stopword cluster ──
    let lit: Vec<usize> = (0..GRID).filter(|&c| grid[c].count > 0).collect();
    let content = lit.iter().filter(|&&c| c >= 100).count(); // rank ≥100 ≈ content words
    let stop = lit.len() - content;
    let median = lit.get(lit.len() / 2).copied().unwrap_or(0);
    println!("\n── SPREAD (vs bag-of-words' 34 cells clustered at rank 0..30) ──");
    println!(
        "  {} distinct cells lit  |  {} content (rank≥100)  |  {} function (rank<100)  |  median rank {}",
        lit.len(),
        content,
        stop,
        median
    );

    // ── top content cells by real-corpus truth-weight ──
    let mut ranked: Vec<usize> = lit.iter().copied().filter(|&c| c >= 100).collect();
    ranked.sort_by_key(|&c| std::cmp::Reverse(grid[c].truth_w));
    println!("\n── TOP CONTENT CELLS (by Σ real COCA frequency = NARS truth weight) ──");
    for &c in ranked.iter().take(12) {
        println!(
            "  cell[{:>4}] '{:<12}' count={:>3}  truth_w={}",
            c,
            vocab.word(c as u16),
            grid[c].count,
            grid[c].truth_w
        );
    }

    let cell_bytes = std::mem::size_of::<Cell>();
    println!(
        "\n  footprint: {} cells × {} B = {} KB (gridlake tier ✓); every cell a real COCA word",
        GRID,
        cell_bytes,
        GRID * cell_bytes / 1024
    );
}
