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

    // Sanity: round-trip a few sampled rows losslessly. Indices are filtered
    // to the actual corpus length so a small FMA_DATA directory can't panic.
    let mut sample_rows = vec![0, 42, 100, all_labels.len() - 1];
    sample_rows.retain(|&i| i < all_labels.len());
    sample_rows.sort_unstable();
    sample_rows.dedup();
    for &i in &sample_rows {
        let got = col.decode(i).unwrap_or_default();
        assert_eq!(got, all_labels[i], "round-trip failed at row {i}");
    }
    println!();
    println!("round-trip: OK on {} sampled rows", sample_rows.len());
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

/// `FMA1234` / `FJ5678` (prefix followed IMMEDIATELY by a non-empty all-digit
/// suffix) or a plain non-empty numeric id. `FMA`, `FMAfoo123` and `""` are
/// labels, not ids.
fn looks_like_id(s: &str) -> bool {
    let numeric_suffix = |rest: &str| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit());
    if let Some(rest) = s.strip_prefix("FMA") {
        return numeric_suffix(rest);
    }
    if let Some(rest) = s.strip_prefix("FJ") {
        return numeric_suffix(rest);
    }
    numeric_suffix(s)
}

fn distinct(labels: &[String]) -> usize {
    let s: std::collections::HashSet<&String> = labels.iter().collect();
    s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_like_id_matches_only_immediate_numeric_suffixes() {
        assert!(looks_like_id("FMA1234"));
        assert!(looks_like_id("FJ5678"));
        assert!(looks_like_id("42"));
        // Prefix without digits, or with non-digits between, is a LABEL.
        assert!(!looks_like_id("FMA"));
        assert!(!looks_like_id("FJ"));
        assert!(!looks_like_id("FMAfoo123"));
        assert!(!looks_like_id("aorta"));
        assert!(!looks_like_id(""));
    }

    #[test]
    fn extract_labels_skips_header_and_ids() {
        let text = "concept id\tname\telement file id\n\
                    FMA3734\taorta\tFJ1931\n\
                    FMA3736\tascending aorta\tFJ3413\n";
        let labels = extract_labels(text);
        // Header row skipped; ids dropped; only the two names survive.
        assert_eq!(labels, vec!["aorta", "ascending aorta"]);
    }

    #[test]
    fn extract_labels_handles_empty_and_header_only_input() {
        assert!(extract_labels("").is_empty());
        assert!(extract_labels("only\ta\theader\n").is_empty());
    }

    #[test]
    fn distinct_counts_unique_labels() {
        let labels = vec![
            "aorta".to_string(),
            "aorta".to_string(),
            "artery".to_string(),
        ];
        assert_eq!(distinct(&labels), 2);
    }
}
