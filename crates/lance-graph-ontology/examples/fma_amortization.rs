//! Empirical amortization measurement on real FMA label data.
//!
//! Run:
//! ```bash
//! FMA_DATA=/home/user/q2/fma/data \
//!   cargo run -p lance-graph-ontology --example fma_amortization
//! ```
//!
//! Reads `element_parts.txt` + `isa_element_parts.txt` from the q2/fma
//! release (or `FMA_DATA` if pointed elsewhere), builds a [`LabelColumn`]
//! from the label vocabulary, and prints the raw-vs-baked bytes + the
//! amortization factor. This is the empirical anchor for lance-graph
//! issue #845's design claim.

use std::env;
use std::fs;
use std::path::PathBuf;

use lance_graph_ontology::soa_bake::{amortization, raw_bytes, LabelColumn};

fn main() {
    let root: PathBuf = env::var_os("FMA_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/user/q2/fma/data"));

    let mut all_labels: Vec<String> = Vec::new();
    let mut per_file: Vec<(String, Vec<String>)> = Vec::new();

    for name in ["element_parts.txt", "isa_element_parts.txt"] {
        let path = root.join(name);
        let Ok(text) = fs::read_to_string(&path) else {
            eprintln!("skip {}: not readable", path.display());
            continue;
        };
        let labels = extract_labels(&text);
        println!(
            "{name:<24}  labels={:>7}  distinct={:>6}  bytes={:>9}",
            labels.len(),
            distinct(&labels),
            raw_bytes(&labels),
        );
        all_labels.extend(labels.iter().cloned());
        per_file.push((name.to_string(), labels));
    }

    if all_labels.is_empty() {
        eprintln!("no FMA labels found under {}", root.display());
        std::process::exit(1);
    }

    println!();
    println!("── amortization per file ──");
    for (name, labels) in &per_file {
        let col = LabelColumn::bake(labels);
        let amort = amortization(labels, &col);
        println!(
            "{name:<24}  raw={:>9}  bake={:>9}  vocab={:>5}  amortization={amort:>6.1}×",
            raw_bytes(labels),
            col.wire_bytes(),
            col.codebook.len(),
        );
    }

    println!();
    println!("── amortization for the combined bake ──");
    let col = LabelColumn::bake(&all_labels);
    let raw = raw_bytes(&all_labels);
    let bake = col.wire_bytes();
    let amort = amortization(&all_labels, &col);
    println!(
        "combined                  raw={raw:>9}  bake={bake:>9}  vocab={:>5}  amortization={amort:>6.1}×",
        col.codebook.len(),
    );

    // Sanity: round-trip a few random rows losslessly.
    for i in [0, 42, 100, all_labels.len() - 1] {
        let got = col.decode(i).unwrap_or_default();
        assert_eq!(got, all_labels[i], "round-trip failed at row {i}");
    }
    println!();
    println!("round-trip: OK on {} sampled rows", 4);
}

/// Parse the FMA TSV format: skip the header line, take label columns
/// (anything non-empty that isn't a bare id like `FMA1234` or `FJ5678`).
fn extract_labels(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.is_empty() {
            continue;
        }
        for field in line.split('\t') {
            let f = field.trim();
            if !f.is_empty() && !looks_like_id(f) {
                out.push(f.to_string());
            }
        }
    }
    out
}

fn looks_like_id(s: &str) -> bool {
    // FMA1234, FJ5678, plain numeric ids.
    (s.starts_with("FMA") || s.starts_with("FJ"))
        && s[..]
            .chars()
            .skip_while(|c| c.is_ascii_alphabetic())
            .all(|c| c.is_ascii_digit())
        || s.chars().all(|c| c.is_ascii_digit())
}

fn distinct(labels: &[String]) -> usize {
    let s: std::collections::HashSet<&String> = labels.iter().collect();
    s.len()
}
