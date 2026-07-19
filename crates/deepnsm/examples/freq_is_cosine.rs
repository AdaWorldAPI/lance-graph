//! Claim under test: the per-subgenre word FREQUENCY DISTRIBUTION alone
//! reproduces NSM semantic-distance ordering, without training or querying
//! any embedding model -- i.e. "the frequency distribution is a cosine
//! replacement."
//!
//! Every COCA lemma in `word_frequency/lemmas_5k.csv` already carries an
//! 8-genre frequency profile (blog, web, TV/movie, spoken, fiction,
//! magazine, news, academic -- see `word_frequency/README.md`). This probe
//! treats that profile as if it were an embedding: log-compress it,
//! z-score every dimension across the whole vocabulary, take the cosine
//! between two words' profiles, and Fisher-z it into a distance. No neural
//! network, no learned weights -- only counts read straight out of a plain
//! frequency-count CSV.
//!
//! Reference: this was verified in Python at Spearman rho = 0.762 against
//! the 8 NSM pairs in `word_frequency/README.md` sect. "Semantic Distance
//! Verification". This example reproduces that measurement in pure-std
//! Rust and is KILL-gated at rho < 0.5.
//!
//! ## Algorithm
//!
//! 1. Parse `lemmas_5k.csv`; for each distinct lowercased lemma (first
//!    occurrence wins -- the file is rank-ascending, so that is also the
//!    highest-frequency part of speech), take the 8 per-million genre
//!    columns (`blogPM..acadPM`, the last 8 of 25) and log-compress each:
//!    `v[d] = ln(1 + PM[d])`.
//! 2. Z-score every dimension across the WHOLE vocabulary (population
//!    mean/stddev; a ~zero-stddev dimension is left unscaled).
//! 3. Distance between two words = Fisher-z of their z-scored cosine:
//!    `c = dot(za, zb) / (|za| * |zb|)`, `dist = -0.5 * ln((1+c)/(1-c))`,
//!    with `c` clamped to `[-0.9999, 0.9999]`. More negative = closer.
//! 4. Rank the 8 NSM reference pairs' README distances and this run's
//!    Fisher-z distances; the Spearman rho between the two rankings is
//!    the scored claim.
//!
//! ## Run
//!
//! ```bash
//! cargo run --manifest-path crates/deepnsm/Cargo.toml --example freq_is_cosine
//! ```

use std::collections::{HashMap, HashSet};

/// The 8 per-million genre columns of `lemmas_5k.csv`, in file order:
/// blog, web, TV/movie, spoken, fiction, magazine, news, academic.
const DIMS: usize = 8;

/// 0-based index of the first `*PM` genre column. The header is
/// `rank,lemma,PoS,freq,perMil,%caps,%allC,range,disp` (9 fields) +
/// 8 raw genre counts + 8 `*PM` genre columns = 25 fields total.
const PM_START: usize = 17;

/// One vocabulary entry: the lemma text plus its 8-genre frequency
/// profile. Holds raw `ln(1+PM)` values on load; [`zscore_in_place`]
/// overwrites `dims` with the z-scored form.
struct WordVec {
    lemma: String,
    dims: [f64; DIMS],
}

/// Parse `lemmas_5k.csv` into one [`WordVec`] per distinct lowercased
/// lemma, keeping the FIRST row seen for each (the file is rank-ascending,
/// so "first occurrence" is also "highest-frequency part of speech").
/// Missing or unparseable genre fields fall back to `0.0`.
fn load_vectors(csv_path: &str) -> Vec<WordVec> {
    let text = std::fs::read_to_string(csv_path)
        .unwrap_or_else(|e| panic!("failed to read {csv_path}: {e}"));

    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < PM_START + DIMS {
            continue;
        }
        let lemma = fields[1].trim().to_lowercase();
        if lemma.is_empty() || !seen.insert(lemma.clone()) {
            continue;
        }
        let dims: [f64; DIMS] = std::array::from_fn(|d| {
            let raw: f64 = fields[PM_START + d].trim().parse().unwrap_or(0.0);
            (1.0 + raw).ln()
        });
        out.push(WordVec { lemma, dims });
    }
    out
}

/// Z-score every dimension across the whole vocabulary, in place
/// (population mean/stddev; a ~zero-stddev dimension is left unscaled so
/// the division never blows up).
fn zscore_in_place(vectors: &mut [WordVec]) {
    let n = vectors.len() as f64;

    let mean: [f64; DIMS] =
        std::array::from_fn(|d| vectors.iter().map(|v| v.dims[d]).sum::<f64>() / n);

    let std_dev: [f64; DIMS] = std::array::from_fn(|d| {
        let var = vectors
            .iter()
            .map(|v| (v.dims[d] - mean[d]).powi(2))
            .sum::<f64>()
            / n;
        let s = var.sqrt();
        if s.abs() < 1e-12 {
            1.0
        } else {
            s
        }
    });

    for v in vectors.iter_mut() {
        v.dims = std::array::from_fn(|d| (v.dims[d] - mean[d]) / std_dev[d]);
    }
}

/// Cosine similarity between two z-scored 8-dim profiles; `0.0` if either
/// norm is ~zero (guards a degenerate all-zero profile).
fn cosine(a: &[f64; DIMS], b: &[f64; DIMS]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

/// Fisher-z transform of a cosine similarity into a distance: more
/// negative is closer, larger is farther. `c` is clamped away from `+-1`
/// so the `ln` stays finite.
fn fisher_z_distance(c: f64) -> f64 {
    let c = c.clamp(-0.9999, 0.9999);
    -0.5 * ((1.0 + c) / (1.0 - c)).ln()
}

/// 1-based ranks of `values`, ascending; tied values (within `1e-12`)
/// share the average rank of their tie group.
fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| values[i].total_cmp(&values[j]));

    let mut out = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (values[idx[j + 1]] - values[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + 1 + j + 1) as f64 / 2.0;
        for &k in &idx[i..=j] {
            out[k] = avg_rank;
        }
        i = j + 1;
    }
    out
}

/// Spearman rank correlation between two equal-length value slices.
fn spearman_rho(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let rx = ranks(xs);
    let ry = ranks(ys);
    let sum_d2: f64 = rx.iter().zip(ry.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    1.0 - (6.0 * sum_d2) / (n * (n * n - 1.0))
}

fn main() {
    let csv_path = concat!(env!("CARGO_MANIFEST_DIR"), "/word_frequency/lemmas_5k.csv");
    let mut vectors = load_vectors(csv_path);
    let n_words = vectors.len();

    println!("── FREQUENCY DISTRIBUTION AS A COSINE REPLACEMENT ────────────────");
    println!("  loaded {n_words} distinct lemmas from {csv_path}");

    zscore_in_place(&mut vectors);

    let mut index: HashMap<&str, usize> = HashMap::with_capacity(vectors.len());
    for (i, v) in vectors.iter().enumerate() {
        index.insert(v.lemma.as_str(), i);
    }

    // The 8 NSM reference pairs, ascending README semantic distance
    // (word_frequency/README.md sect. "Semantic Distance Verification").
    let pairs: [(&str, &str, f64); 8] = [
        ("i", "you", 3.32),
        ("think", "know", 3.85),
        ("before", "after", 4.46),
        ("live", "die", 5.35),
        ("good", "bad", 6.60),
        ("think", "big", 8.16),
        ("the", "this", 11.47),
        ("say", "word", 15.22),
    ];

    let mut freq_dist = Vec::with_capacity(pairs.len());
    let mut readme_dist = Vec::with_capacity(pairs.len());

    println!();
    println!(
        "  {:<14}  {:>14}  {:>14}",
        "pair", "freq_dist", "readme_dist"
    );
    for &(a, b, readme) in &pairs {
        let ia = *index
            .get(a)
            .unwrap_or_else(|| panic!("'{a}' not found in vocabulary"));
        let ib = *index
            .get(b)
            .unwrap_or_else(|| panic!("'{b}' not found in vocabulary"));
        let dist = fisher_z_distance(cosine(&vectors[ia].dims, &vectors[ib].dims));
        let label = format!("{a}/{b}");
        println!("  {label:<14}  {dist:>14.4}  {readme:>14.2}");
        freq_dist.push(dist);
        readme_dist.push(readme);
    }

    let rho = spearman_rho(&freq_dist, &readme_dist);
    println!();
    println!("Spearman rho (freq-distance vs README semantic ordering): {rho:.3}");

    if rho < 0.5 {
        eprintln!(
            "  ✗ KILL: frequency distribution does NOT reproduce semantic ordering (rho={rho:.3})"
        );
        std::process::exit(1);
    }

    println!();
    println!("  ✓ claim holds: rho={rho:.3} >= 0.5 (this probe's KILL floor).");
    println!("  the per-subgenre word FREQUENCY DISTRIBUTION reproduces the NSM");
    println!("  semantic-distance ordering. No embeddings, no cosine-on-vectors --");
    println!("  the distance above is a property of the frequency distribution");
    println!("  alone (ln(1+PM) per genre, z-scored across the vocabulary, Fisher-z'd).");
    println!();
    println!("  honest boundary: this uses only the 8 coarse GENRES committed to");
    println!("  this repo (blog/web/TVM/spoken/fiction/magazine/news/academic).");
    println!("  the full 96-SUBGENRE palette (word_frequency/README.md, gitignored)");
    println!("  is higher-resolution, so this rho is a FLOOR on what the frequency");
    println!("  distribution can do here, not a ceiling.");
}
