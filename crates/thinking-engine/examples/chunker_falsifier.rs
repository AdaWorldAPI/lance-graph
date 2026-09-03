//! chunker_falsifier — the D-TEH-3 fate probe for `semantic_chunker`.
//!
//! Plan `thinking-engine-harvest-closure-v1` §1c: `semantic_chunker` goes to
//! deepnsm-v2 ONLY if its falsifier passes ("boundaries vs a gold sentence
//! split; else stays LAB"). This is that falsifier, on REAL data:
//!
//! - text: the tier-1..4 calibration corpus of `jina_v5_ground_truth.rs`
//!   (Rule 23 — no synthetic sentences);
//! - tokens → centroids: the Jina v5 (Qwen3) tokenizer + the baked
//!   `jina-v5-codebook/codebook_index.u16` (151 936 entries);
//! - engine: the baked `jina-v5-codebook/distance_table_256x256.u8`.
//!
//! Three arms, PRE-REGISTERED before the first run:
//!
//! - CAN-FIRE: every ordered pair of sentences from DIFFERENT corpus pairs is
//!   concatenated; the gold boundary is the token seam. recall@tol = fraction
//!   of passages with a detected boundary within ±`SEAM_TOL` tokens of the
//!   seam.
//! - NULL: the same passages with their centroid order shuffled
//!   (`NULL_PERMS` deterministic SplitMix64 permutations) — the seam no longer
//!   exists, so this is what "a boundary landed near the seam by chance"
//!   looks like. Reported as p95 of recall@tol over permutations.
//! - SILENCE: the two sentences of each tier-1 / tier-2 pair (same topic),
//!   both orders; a boundary here is a false split.
//!
//! PASS (per threshold, the SAME threshold for all three arms):
//!   recall@tol ≥ 0.75  AND  recall@tol ≥ null_p95 + 0.15
//!   AND  false splits ≤ 2 of the 8 coherent passages.
//! KILL otherwise → the chunker stays LAB (not ported).
//!
//! A 4th, committed arm — POSITIVE CONTROL: an all-zero recall on the three
//! arms above is exactly the shape a broken harness (wrong table, wrong
//! tokenizer, a wiring bug) would also produce, so a null result is not
//! trustworthy on its own (CLAUDE.md's falsifiability rule: "a null result is
//! a claim about the measurement apparatus until proven otherwise"). This arm
//! reproduces the module's OWN adversarial positive-control shape from
//! `semantic_chunker::tests::detects_boundary_between_topics` — two
//! maximally-separated synthetic centroid clusters (corners 0-4 vs 250-254 of
//! the 256-centroid space) — against the SAME real table the falsifier uses,
//! and reports its boundary count at every threshold. If this control ALSO
//! returns zero, the all-zero falsifier result is a genuine mechanism null,
//! not a harness artifact; if it returns boundaries where the falsifier did
//! not, the harness (not the mechanism) is the thing to investigate.
//!
//! Usage:
//!   JINA_V5_TOKENIZER=/path/to/tokenizer.json \
//!   cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!     --example chunker_falsifier

use std::path::Path;

use thinking_engine::codebook_index::CodebookIndex;
use thinking_engine::engine::ThinkingEngine;
use thinking_engine::semantic_chunker::{find_boundaries, ChunkerConfig};

const TABLE: &str = "crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.u8";
const CODEBOOK: &str = "crates/thinking-engine/data/jina-v5-codebook/codebook_index.u16";
const N: usize = 256;

/// ±tokens around the seam that count as "found it". Two chunker steps.
const SEAM_TOL: usize = 4;
const NULL_PERMS: usize = 20;
const THRESHOLDS: &[f32] = &[0.3, 0.45, 0.6];

/// The real corpus (jina_v5_ground_truth.rs), grouped by pair; pairs 0..=3
/// are tier 1–2 (same topic within the pair), 4 is tier 3, 5–6 are tier 4.
const CORPUS: &[(&str, &str)] = &[
    (
        "The wound is the place where the light enters you",
        "Where there is ruin there is hope for a treasure",
    ),
    (
        "A federal judge in New York ruled the surveillance program unconstitutional",
        "A US court declared the mass surveillance scheme violated the constitution",
    ),
    (
        "Palantir built Gotham for intelligence agencies to map human networks",
        "Edward Snowden revealed the NSA collected phone metadata of millions of Americans",
    ),
    (
        "Amyloid plaques accumulate in the brains of Alzheimer patients",
        "Tau protein tangles disrupt neural communication in neurodegenerative disease",
    ),
    (
        "Newton showed that gravity follows an inverse square law",
        "Quantum entanglement allows particles to share states across arbitrary distances",
    ),
    (
        "You are not a drop in the ocean you are the entire ocean in a drop",
        "TCP uses a three-way handshake to establish a reliable connection between hosts",
    ),
    (
        "CRISPR-Cas9 enables precise editing of genomic sequences at targeted loci",
        "Bach composed the Well-Tempered Clavier as an exploration of all major and minor keys",
    ),
];
const COHERENT_PAIRS: usize = 4;

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

fn config(threshold: f32) -> ChunkerConfig {
    ChunkerConfig {
        window_size: 8,
        step_size: 2,
        boundary_threshold: threshold,
        min_chunk_tokens: 4,
        max_chunk_tokens: 256,
        top_k: 5,
        max_cycles: 10,
    }
}

fn near_seam(engine: &mut ThinkingEngine, cents: &[u16], seam: usize, cfg: &ChunkerConfig) -> bool {
    find_boundaries(engine, cents, cfg)
        .iter()
        .any(|b| b.position.abs_diff(seam) <= SEAM_TOL)
}

fn main() {
    let tok_path = std::env::var("JINA_V5_TOKENIZER")
        .unwrap_or_else(|_| "/tmp/jina-v5-tokenizer.json".to_string());
    let tokenizer = match tokenizers::Tokenizer::from_file(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer not loadable at {tok_path}: {e}");
            eprintln!("set JINA_V5_TOKENIZER to the Jina v5 (Qwen3) tokenizer.json");
            std::process::exit(2);
        }
    };
    let codebook = CodebookIndex::load(Path::new(CODEBOOK), N as u16, "jina-v5".into())
        .expect("codebook index");
    let table = std::fs::read(TABLE).expect("table");
    assert_eq!(table.len(), N * N);
    let mut engine = ThinkingEngine::new(table);

    // Tokenize once (no special tokens, so a concatenation's seam is len(a)).
    let sentences: Vec<(usize, Vec<u16>)> = CORPUS
        .iter()
        .enumerate()
        .flat_map(|(pair, (a, b))| [(pair, *a), (pair, *b)])
        .map(|(pair, text)| {
            let ids = tokenizer
                .encode(text, false)
                .expect("tokenize")
                .get_ids()
                .to_vec();
            (pair, codebook.lookup_many(&ids))
        })
        .collect();
    let (min_len, max_len) = sentences.iter().fold((usize::MAX, 0), |(lo, hi), (_, c)| {
        (lo.min(c.len()), hi.max(c.len()))
    });
    println!(
        "corpus: {} sentences, {}..{} tokens each, codebook {} tokens / {} centroids used",
        sentences.len(),
        min_len,
        max_len,
        codebook.len(),
        codebook.unique_centroids()
    );

    // Passages.
    let mut switch: Vec<(Vec<u16>, usize)> = Vec::new(); // (centroids, seam)
    for (i, (pa, ca)) in sentences.iter().enumerate() {
        for (j, (pb, cb)) in sentences.iter().enumerate() {
            if i != j && pa != pb {
                let mut v = ca.clone();
                v.extend_from_slice(cb);
                switch.push((v, ca.len()));
            }
        }
    }
    let mut coherent: Vec<(Vec<u16>, usize)> = Vec::new();
    for pair in 0..COHERENT_PAIRS {
        let (_, a) = &sentences[2 * pair];
        let (_, b) = &sentences[2 * pair + 1];
        for (x, y) in [(a, b), (b, a)] {
            let mut v = x.clone();
            v.extend_from_slice(y);
            coherent.push((v, x.len()));
        }
    }
    println!(
        "passages: {} cross-topic (can-fire), {} same-topic (silence); seam tolerance ±{SEAM_TOL}, {NULL_PERMS} null permutations\n",
        switch.len(),
        coherent.len()
    );

    println!(
        "{:>9} | {:>9} | {:>8} | {:>12} | {:>9} | verdict",
        "threshold", "recall", "null p95", "false splits", "bnd/pass"
    );
    let mut any_pass = false;
    for &t in THRESHOLDS {
        let cfg = config(t);
        // Can-fire.
        let hits = switch
            .iter()
            .filter(|(c, s)| near_seam(&mut engine, c, *s, &cfg))
            .count();
        let recall = hits as f64 / switch.len() as f64;
        let mean_bnd = switch
            .iter()
            .map(|(c, _)| find_boundaries(&mut engine, c, &cfg).len())
            .sum::<usize>() as f64
            / switch.len() as f64;
        // Null: shuffle each passage's centroid order, keep the nominal seam.
        let mut rng = SplitMix64(0x9E37_79B9_7F4A_7C15);
        let mut null_recalls: Vec<f64> = (0..NULL_PERMS)
            .map(|_| {
                let h = switch
                    .iter()
                    .filter(|(c, s)| {
                        let mut v = c.clone();
                        rng.shuffle(&mut v);
                        near_seam(&mut engine, &v, *s, &cfg)
                    })
                    .count();
                h as f64 / switch.len() as f64
            })
            .collect();
        null_recalls.sort_by(|a, b| a.total_cmp(b));
        let p95 = null_recalls[((NULL_PERMS as f64 * 0.95).ceil() as usize).min(NULL_PERMS) - 1];
        // Silence.
        let false_splits = coherent
            .iter()
            .filter(|(c, _)| !find_boundaries(&mut engine, c, &cfg).is_empty())
            .count();
        let pass = recall >= 0.75 && recall >= p95 + 0.15 && false_splits <= 2;
        any_pass |= pass;
        println!(
            "{:>9.2} | {:>9.3} | {:>8.3} | {:>7} of {:>2} | {:>9.2} | {}",
            t,
            recall,
            p95,
            false_splits,
            coherent.len(),
            mean_bnd,
            if pass { "PASS" } else { "kill" }
        );
    }
    println!(
        "\nVERDICT: {} — pre-registered: recall ≥ 0.75 AND recall ≥ null p95 + 0.15 AND false splits ≤ 2, at one threshold",
        if any_pass { "PASS (port-eligible)" } else { "KILL (stays LAB)" }
    );

    // POSITIVE CONTROL — the module's own adversarial shape (two maximally
    // separated synthetic centroid clusters), against the same real table.
    // See the module doc comment above for why this arm exists.
    let mut corners: Vec<u16> = Vec::with_capacity(48);
    for i in 0..24u16 {
        corners.push(i % 5);
    }
    for i in 0..24u16 {
        corners.push(250 + i % 5);
    }
    println!("\n{:>9} | {:>18} | note", "threshold", "control boundaries");
    let mut control_ever_fires = false;
    for &t in THRESHOLDS {
        let cfg = config(t);
        let n = find_boundaries(&mut engine, &corners, &cfg).len();
        control_ever_fires |= n > 0;
        println!("{:>9.2} | {:>18} |", t, n);
    }
    println!(
        "\nPOSITIVE CONTROL: {} — synthetic corners (centroids 0-4 vs 250-254, the module's own `detects_boundary_between_topics` shape) {} boundaries on this table",
        if control_ever_fires {
            "FIRES (the falsifier's all-zero result is NOT a mechanism null — investigate the harness)"
        } else {
            "STAYS AT ZERO TOO — the falsifier's all-zero result is a genuine mechanism null, not a harness artifact"
        },
        if control_ever_fires { "produces" } else { "produces zero" }
    );
}
