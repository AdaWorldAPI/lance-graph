//! Full OCR → Thinking Styles → AriGraph SPO → NARS pipeline with periodic export.
//!
//! ```text
//! Wikileaks PDFs → tesseract OCR → paragraphs
//!   → extractor (SPO triplets) → LiteralGraph
//!   → OrchestrationMode selects inference DAG
//!   → NARS revision on discovered edges
//!   → export graph every N topics or T seconds
//! ```
//!
//! Usage:
//!   cargo run --manifest-path crates/lance-graph-osint/Cargo.toml --example ocr_pipeline
//!
//! Env vars:
//!   WIKILEAKS_URLS: comma-separated PDF URLs (optional, has defaults)
//!   EXPORT_INTERVAL: seconds between graph exports (default: 60)
//!   EXPORT_PATH: output path (default: /root/data/ocr_graph_export.json)

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use lance_graph_osint::reader;
use lance_graph_osint::extractor;

fn main() {
    let start = Instant::now();
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  OCR → Thinking → AriGraph SPO → NARS Pipeline");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    let export_interval: u64 = std::env::var("EXPORT_INTERVAL")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let export_path = std::env::var("EXPORT_PATH")
        .unwrap_or_else(|_| "/root/data/ocr_graph_export.json".into());

    // ── Source URLs (PDFs from Wikileaks + other OSINT sources) ────────────
    let urls: Vec<String> = std::env::var("WIKILEAKS_URLS")
        .map(|s| s.split(',').map(|u| u.trim().to_string()).collect())
        .unwrap_or_else(|_| vec![
            // Wikileaks dealmaker (arms trade arbitration)
            "https://wikileaks.org/dealmaker/Al-Yousef/document/15908-KENOZA-VS-GIAT/15908-KENOZA-VS-GIAT.pdf".into(),
            // Wikileaks Spy Files
            "https://file.wikileaks.org/file/WikiLeaks%20Spy%20files/".into(),
            // Vault7 index
            "https://file.wikileaks.org/file/vault7/".into(),
            // CableGate
            "https://file.wikileaks.org/file/cablegate/".into(),
            // Collateral Murder
            "https://file.wikileaks.org/file/collateralmurder/".into(),
        ]);

    // ── Graph state ───────────────────────────────────────────────────────
    let mut nodes: HashMap<String, NodeInfo> = HashMap::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut topics_processed = 0;
    let mut total_paragraphs = 0;
    let mut total_triplets = 0;
    let mut ocr_pages = 0;
    let mut last_export = Instant::now();
    let mut export_count = 0;

    // ── Process each URL ──────────────────────────────────────────────────
    for (url_idx, url) in urls.iter().enumerate() {
        eprintln!("\n[{}/{}] Processing: {}", url_idx + 1, urls.len(), truncate(url, 80));

        // Fetch + OCR (PDF) or strip HTML
        let paragraphs = match reader::fetch_and_embed(url) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  [skip] {}", e);
                // If it's a directory listing, try to parse links and follow them
                if url.ends_with('/') {
                    if let Ok(body) = curl_fetch_text(url) {
                        let sub_urls = extract_links(&body, url, 10);
                        eprintln!("  [dir] Found {} sub-URLs, processing first 10", sub_urls.len());
                        for sub_url in &sub_urls {
                            if let Ok(sub_paras) = reader::fetch_and_embed(sub_url) {
                                process_paragraphs(
                                    &sub_paras, sub_url, url_idx as u64,
                                    &mut nodes, &mut edges, &mut total_paragraphs, &mut total_triplets,
                                );
                            }
                        }
                    }
                }
                continue;
            }
        };

        if paragraphs.is_empty() {
            eprintln!("  [skip] no content extracted");
            continue;
        }

        // ── Thinking style selection (based on content) ──────────────────
        let style = detect_thinking_style(&paragraphs);
        eprintln!("  [style] {} ({} paragraphs)", style, paragraphs.len());

        // ── Extract triplets + build graph ───────────────────────────────
        process_paragraphs(
            &paragraphs, url, url_idx as u64,
            &mut nodes, &mut edges, &mut total_paragraphs, &mut total_triplets,
        );

        topics_processed += 1;

        // ── NARS revision on graph ───────────────────────────────────────
        let (revisions, contradictions) = nars_revise(&mut edges);
        if revisions > 0 || contradictions > 0 {
            eprintln!("  [nars] {} revisions, {} contradictions", revisions, contradictions);
        }

        // ── Periodic export ──────────────────────────────────────────────
        let elapsed_since_export = last_export.elapsed().as_secs();
        if elapsed_since_export >= export_interval || topics_processed % 5 == 0 {
            export_count += 1;
            export_graph(&export_path, &nodes, &edges, export_count);
            last_export = Instant::now();
        }

        // ── Progress ─────────────────────────────────────────────────────
        eprintln!("  [graph] {} nodes, {} edges | {} paragraphs, {} triplets",
            nodes.len(), edges.len(), total_paragraphs, total_triplets);
    }

    // ── Final export ──────────────────────────────────────────────────────
    export_count += 1;
    export_graph(&export_path, &nodes, &edges, export_count);

    let elapsed = start.elapsed();
    eprintln!("\n═══════════════════════════════════════════════════════════");
    eprintln!("  PIPELINE COMPLETE");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  URLs processed:  {}", urls.len());
    eprintln!("  Topics:          {}", topics_processed);
    eprintln!("  Paragraphs:      {}", total_paragraphs);
    eprintln!("  Triplets:        {}", total_triplets);
    eprintln!("  Nodes:           {}", nodes.len());
    eprintln!("  Edges:           {}", edges.len());
    eprintln!("  Exports:         {}", export_count);
    eprintln!("  Elapsed:         {:.1}s", elapsed.as_secs_f64());
    eprintln!("  Output:          {}", export_path);

    // Top entities
    let mut entity_counts: Vec<_> = nodes.iter()
        .map(|(id, info)| (id.clone(), info.edge_count))
        .collect();
    entity_counts.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("\n  Top entities:");
    for (id, count) in entity_counts.iter().take(15) {
        eprintln!("    {:4} edges: {}", count, id);
    }
    eprintln!("═══════════════════════════════════════════════════════════\n");
}

// ── Types ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct NodeInfo {
    name: String,
    node_type: String,
    edge_count: usize,
    sources: HashSet<String>,
}

#[derive(Debug, Clone)]
struct Edge {
    source: String,
    target: String,
    label: String,
    freq: f32,
    conf: f32,
    source_url: String,
}

// ── Processing ────────────────────────────────────────────────────────────

fn process_paragraphs(
    paragraphs: &[reader::EmbeddedParagraph],
    source_url: &str,
    clock: u64,
    nodes: &mut HashMap<String, NodeInfo>,
    edges: &mut Vec<Edge>,
    total_paragraphs: &mut usize,
    total_triplets: &mut usize,
) {
    *total_paragraphs += paragraphs.len();

    for para in paragraphs {
        let triplets = extractor::extract_triplets(&para.text, clock);
        for t in &triplets {
            let s = clean_entity(&t.subject);
            let o = clean_entity(&t.object);
            let r = t.relation.to_lowercase();
            if s.len() < 2 || o.len() < 2 || r.is_empty() { continue; }
            if s.len() > 80 || o.len() > 80 { continue; }

            // Add/update nodes
            ensure_node(nodes, &s, "Entity", source_url);
            ensure_node(nodes, &o, "Entity", source_url);

            // Add edge (check for duplicates)
            let exists = edges.iter().any(|e| e.source == s && e.target == o && e.label == r);
            if !exists {
                edges.push(Edge {
                    source: s.clone(), target: o.clone(), label: r,
                    freq: t.truth.frequency, conf: t.truth.confidence,
                    source_url: source_url.to_string(),
                });
                if let Some(n) = nodes.get_mut(&s) { n.edge_count += 1; }
                if let Some(n) = nodes.get_mut(&o) { n.edge_count += 1; }
            }

            *total_triplets += 1;
        }
    }
}

fn ensure_node(nodes: &mut HashMap<String, NodeInfo>, id: &str, node_type: &str, source: &str) {
    nodes.entry(id.to_string()).or_insert_with(|| NodeInfo {
        name: id.to_string(),
        node_type: node_type.to_string(),
        edge_count: 0,
        sources: HashSet::new(),
    }).sources.insert(source.to_string());
}

fn clean_entity(s: &str) -> String {
    s.to_lowercase()
        .trim_matches(|c: char| !c.is_alphanumeric() && c != ' ' && c != '-')
        .to_string()
}

// ── Thinking style detection (from paragraph content) ─────────────────────

fn detect_thinking_style(paragraphs: &[reader::EmbeddedParagraph]) -> &'static str {
    let all_text: String = paragraphs.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join(" ");
    let lower = all_text.to_lowercase();

    // Match keywords to inference DAG type
    if lower.contains("court") || lower.contains("arbitr") || lower.contains("tribunal")
        || lower.contains("verdict") || lower.contains("plaintiff") {
        "Analytical→Deduction→Synthesis (legal)"
    } else if lower.contains("weapon") || lower.contains("military") || lower.contains("surveillance")
        || lower.contains("intelligence") || lower.contains("classified") {
        "Investigative→Abduction→Counterfactual (intel)"
    } else if lower.contains("email") || lower.contains("from:") || lower.contains("subject:") {
        "Association→Induction→Hypothesis (correspondence)"
    } else if lower.contains("financial") || lower.contains("bank") || lower.contains("tax")
        || lower.contains("fund") || lower.contains("invest") {
        "Analytical→Hypothesis→HypothesisTest (financial)"
    } else if lower.contains("diplomat") || lower.contains("embassy") || lower.contains("cable") {
        "Association→Abduction→Synthesis (diplomatic)"
    } else {
        "Association→Intuition→Synthesis (general)"
    }
}

// ── NARS revision ─────────────────────────────────────────────────────────

fn nars_revise(edges: &mut Vec<Edge>) -> (usize, usize) {
    let mut revisions = 0;
    let mut contradictions = 0;

    // Find edges with same source+target but different labels → contradiction
    let mut seen: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        seen.entry((e.source.clone(), e.target.clone())).or_default().push(i);
    }

    for ((s, t), indices) in &seen {
        if indices.len() > 1 {
            // Multiple edges between same pair → check for contradiction
            let labels: HashSet<_> = indices.iter().map(|&i| &edges[i].label).collect();
            if labels.len() > 1 {
                contradictions += 1;
                // Lower confidence of all edges in this group
                for &i in indices {
                    edges[i].conf *= 0.8; // decay on contradiction
                }
            } else {
                // Same label, multiple evidence → boost confidence (revision)
                for &i in indices {
                    edges[i].conf = (edges[i].conf * 1.1).min(0.99);
                    edges[i].freq = (edges[i].freq * 1.05).min(1.0);
                }
                revisions += 1;
            }
        }
    }

    (revisions, contradictions)
}

// ── Export ─────────────────────────────────────────────────────────────────

fn export_graph(path: &str, nodes: &HashMap<String, NodeInfo>, edges: &[Edge], count: usize) {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

    let mut out = format!("{{\n  \"export_count\": {},\n  \"timestamp\": {},\n  \"nodes\": [\n", count, timestamp);
    for (i, (id, info)) in nodes.iter().enumerate() {
        if i > 0 { out.push_str(",\n"); }
        out.push_str(&format!(
            "    {{\"id\":\"{}\",\"name\":\"{}\",\"type\":\"{}\",\"edges\":{},\"sources\":{}}}",
            esc(id), esc(&info.name), esc(&info.node_type), info.edge_count, info.sources.len()
        ));
    }
    out.push_str("\n  ],\n  \"edges\": [\n");
    for (i, e) in edges.iter().enumerate() {
        if i > 0 { out.push_str(",\n"); }
        out.push_str(&format!(
            "    {{\"source\":\"{}\",\"target\":\"{}\",\"label\":\"{}\",\"freq\":{:.2},\"conf\":{:.2}}}",
            esc(&e.source), esc(&e.target), esc(&e.label), e.freq, e.conf
        ));
    }
    out.push_str("\n  ]\n}\n");

    let versioned_path = format!("{}.{}", path, count);
    std::fs::write(&versioned_path, &out).ok();
    std::fs::write(path, &out).ok(); // also write to main path
    eprintln!("  [export #{}] {} nodes, {} edges → {}", count, nodes.len(), edges.len(), versioned_path);
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ")
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn curl_fetch_text(url: &str) -> Result<String, String> {
    let output = std::process::Command::new("curl")
        .args(["-sLk", "--max-time", "15", url])
        .output()
        .map_err(|e| format!("{e}"))?;
    String::from_utf8(output.stdout).map_err(|e| format!("{e}"))
}

fn extract_links(html: &str, base_url: &str, max: usize) -> Vec<String> {
    let mut links = Vec::new();
    let mut pos = 0;
    while let Some(start) = html[pos..].find("href=\"") {
        let abs_start = pos + start + 6;
        if let Some(end) = html[abs_start..].find('"') {
            let href = &html[abs_start..abs_start + end];
            if href != "../" && !href.is_empty() {
                let full_url = if href.starts_with("http") {
                    href.to_string()
                } else {
                    format!("{}{}", base_url.trim_end_matches('/'), if href.starts_with('/') { href.to_string() } else { format!("/{}", href) })
                };
                // Only follow PDFs and text files
                let lower = full_url.to_lowercase();
                if lower.ends_with(".pdf") || lower.ends_with(".txt") || lower.ends_with(".csv") {
                    links.push(full_url);
                    if links.len() >= max { break; }
                }
            }
        }
        pos = abs_start + 1;
    }
    links
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}
