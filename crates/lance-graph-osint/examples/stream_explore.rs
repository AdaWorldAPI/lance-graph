//! Stream-explore: start from aiwar seed graph, discover 2000+ nodes via OSINT.
//!
//! ```text
//! aiwar_graph.json (221 nodes, 326 edges)
//!   → seed frontier (326 edges)
//!   → pick highest-curiosity edge
//!   → Pearl query → search URL → reader fetch → extract triplets
//!   → NARS revision + frontier expansion
//!   → repeat until 2000 nodes or budget exhausted
//! ```
//!
//! Usage: cargo run --manifest-path crates/lance-graph-osint/Cargo.toml --example stream_explore

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use lance_graph_osint::reader;
use lance_graph_osint::extractor;

fn main() {
    let start = Instant::now();
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  NARS Mass Exploration — Streaming to 2000 nodes");
    eprintln!("═══════════════════════════════════════════════════════════\n");

    // 1. Load seed graph
    let json_path = std::env::var("AIWAR_GRAPH").unwrap_or_else(|_| "/root/data/aiwar_graph.json".into());
    let json = match std::fs::read_to_string(&json_path) {
        Ok(s) => s,
        Err(e) => { eprintln!("ERROR: Cannot read {}: {}", json_path, e); return; }
    };

    let seed = parse_aiwar_graph(&json);
    eprintln!("[seed] {} nodes, {} edges", seed.nodes.len(), seed.edges.len());

    // 2. Build frontier from seed edges
    let mut frontier: Vec<FrontierEdge> = seed.edges.iter().map(|e| FrontierEdge {
        source: e.0.clone(), target: e.1.clone(), label: e.2.clone(),
        truth_freq: 0.9, truth_conf: 0.5,
        query_count: 0, is_seed: true,
    }).collect();

    let mut all_nodes: HashSet<String> = seed.nodes.keys().cloned().collect();
    let mut all_edges: Vec<(String, String, String, f32, f32)> = seed.edges.iter()
        .map(|e| (e.0.clone(), e.1.clone(), e.2.clone(), 0.9, 0.5))
        .collect();

    let target_nodes = 2000;
    let max_queries = 500;
    let mut queries_done = 0;
    let mut discoveries = 0;
    let mut confirmations = 0;
    let mut fetch_failures = 0;

    eprintln!("[explore] target={} nodes, budget={} queries\n", target_nodes, max_queries);

    // 3. Exploration loop
    while all_nodes.len() < target_nodes && queries_done < max_queries {
        // Sort frontier by curiosity (low confidence + un-queried)
        frontier.sort_by(|a, b| b.curiosity().partial_cmp(&a.curiosity()).unwrap());

        // Pick top edge
        let edge = match frontier.first() {
            Some(e) if e.curiosity() > 0.01 => e.clone(),
            _ => { eprintln!("[stop] frontier exhausted or no curiosity left"); break; }
        };

        // Mark as queried
        if let Some(fe) = frontier.iter_mut().find(|e|
            e.source == edge.source && e.target == edge.target && e.label == edge.label
        ) {
            fe.query_count += 1;
        }

        // Generate search query (Pearl Level 1: SEE)
        let query = format!("{} {} {}", edge.source, edge.label, edge.target);
        queries_done += 1;

        // Google Custom Search → article URLs → fetch each full article → embed
        // Falls back to DuckDuckGo HTML if GOOGLE_API_KEY not set
        let paragraphs = match reader::search_and_embed(&query, 5) {
            Ok(p) if !p.is_empty() => p,
            _ => {
                // Fallback: DuckDuckGo HTML (snippets only, less effective)
                if queries_done == 1 {
                    eprintln!("[info] Set GOOGLE_API_KEY + GOOGLE_CX for full article fetching");
                    eprintln!("[info] Falling back to DuckDuckGo HTML (search snippets only)");
                }
                let ddg_url = format!("https://html.duckduckgo.com/html/?q={}", urlencoding(&query));
                match reader::fetch_and_embed(&ddg_url) {
                    Ok(p) => p,
                    Err(e) => {
                        fetch_failures += 1;
                        if queries_done <= 3 || queries_done % 20 == 0 {
                            eprintln!("[fetch fail #{}/{}] {}: {}", fetch_failures, queries_done, query, e);
                        }
                        continue;
                    }
                }
            }
        };

        // Extract triplets from each paragraph
        let mut round_new = 0;
        let mut round_confirmed = 0;

        for para in &paragraphs {
            let triplets = extractor::extract_triplets(&para.text, queries_done as u64);
            for t in &triplets {
                let s = t.subject.to_lowercase();
                let o = t.object.to_lowercase();
                let r = t.relation.to_lowercase();

                if s.len() < 2 || o.len() < 2 || r.is_empty() { continue; }
                if s.len() > 100 || o.len() > 100 { continue; } // skip garbage

                // Check if confirms query edge
                let confirms = (s.contains(&edge.source.to_lowercase())
                    || edge.source.to_lowercase().contains(&s))
                    && (o.contains(&edge.target.to_lowercase())
                    || edge.target.to_lowercase().contains(&o));

                if confirms {
                    // NARS revision: boost confidence of query edge
                    if let Some(fe) = frontier.iter_mut().find(|e|
                        e.source == edge.source && e.target == edge.target
                    ) {
                        fe.truth_conf = nars_revise_conf(fe.truth_conf, 0.6);
                        fe.truth_freq = nars_revise_freq(fe.truth_freq, 0.9, fe.truth_conf, 0.6);
                    }
                    round_confirmed += 1;
                    confirmations += 1;
                } else {
                    // New edge discovered
                    let clean_s = clean(&s);
                    let clean_o = clean(&o);
                    if clean_s.is_empty() || clean_o.is_empty() { continue; }

                    let exists = frontier.iter().any(|e|
                        e.source == clean_s && e.target == clean_o && e.label == r
                    );
                    if !exists {
                        // Add to graph
                        all_nodes.insert(clean_s.clone());
                        all_nodes.insert(clean_o.clone());
                        all_edges.push((clean_s.clone(), clean_o.clone(), r.clone(), 0.7, 0.3));

                        frontier.push(FrontierEdge {
                            source: clean_s, target: clean_o, label: r,
                            truth_freq: 0.7, truth_conf: 0.3,
                            query_count: 0, is_seed: false,
                        });
                        round_new += 1;
                        discoveries += 1;
                    }
                }
            }
        }

        // Progress report
        if queries_done % 5 == 0 || round_new > 0 {
            let elapsed = start.elapsed().as_secs();
            eprintln!("[q{:3}] nodes={:4} edges={:4} +{:2} new +{:1} conf | frontier={:4} | curiosity={:.3} | {}s",
                queries_done, all_nodes.len(), all_edges.len(),
                round_new, round_confirmed,
                frontier.len(), edge.curiosity(), elapsed);
        }

        // Rate limit: don't hammer the server
        if queries_done % 3 == 0 {
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    // 4. Final report
    let elapsed = start.elapsed();
    eprintln!("\n═══════════════════════════════════════════════════════════");
    eprintln!("  EXPLORATION COMPLETE");
    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Seed:        {} nodes, {} edges", seed.nodes.len(), seed.edges.len());
    eprintln!("  Final:       {} nodes, {} edges", all_nodes.len(), all_edges.len());
    eprintln!("  Discovered:  {} new edges, {} confirmations", discoveries, confirmations);
    eprintln!("  Queries:     {}/{} budget (failures: {})", queries_done, max_queries, fetch_failures);
    eprintln!("  Elapsed:     {:.1}s ({:.1} queries/sec)", elapsed.as_secs_f64(),
        queries_done as f64 / elapsed.as_secs_f64().max(0.001));
    eprintln!("  Growth:      {:.1}x nodes, {:.1}x edges",
        all_nodes.len() as f64 / seed.nodes.len() as f64,
        all_edges.len() as f64 / seed.edges.len() as f64);

    // Top discovered entities
    let mut entity_counts: HashMap<String, usize> = HashMap::new();
    for (s, t, _, _, _) in &all_edges {
        if !seed.nodes.contains_key(s) { *entity_counts.entry(s.clone()).or_default() += 1; }
        if !seed.nodes.contains_key(t) { *entity_counts.entry(t.clone()).or_default() += 1; }
    }
    let mut sorted_entities: Vec<_> = entity_counts.into_iter().collect();
    sorted_entities.sort_by(|a, b| b.1.cmp(&a.1));
    eprintln!("\n  Top discovered entities:");
    for (entity, count) in sorted_entities.iter().take(20) {
        eprintln!("    {:4} edges: {}", count, entity);
    }

    // Confidence distribution
    let crystallized = frontier.iter().filter(|e| e.truth_conf > 0.8).count();
    let hot = frontier.iter().filter(|e| e.truth_conf < 0.3).count();
    let warm = frontier.len() - crystallized - hot;
    eprintln!("\n  Frontier: {} crystallized, {} warm, {} hot", crystallized, warm, hot);
    eprintln!("═══════════════════════════════════════════════════════════\n");

    // 5. Save expanded graph
    let output_path = std::env::var("OUTPUT_GRAPH").unwrap_or_else(|_| "/root/data/aiwar_expanded.json".into());
    save_expanded_graph(&output_path, &all_nodes, &all_edges);
    eprintln!("[saved] {}", output_path);
}

// ── Minimal types ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct FrontierEdge {
    source: String,
    target: String,
    label: String,
    truth_freq: f32,
    truth_conf: f32,
    query_count: u32,
    is_seed: bool,
}

impl FrontierEdge {
    fn curiosity(&self) -> f32 {
        let novelty = 1.0 / (self.query_count as f32 + 1.0);
        let uncertainty = 1.0 - self.truth_conf;
        novelty * uncertainty
    }
}

struct SeedGraph {
    nodes: HashMap<String, String>, // id → type
    edges: Vec<(String, String, String)>, // source, target, label
}

fn parse_aiwar_graph(json: &str) -> SeedGraph {
    let mut nodes = HashMap::new();
    let mut edges = Vec::new();

    for key in &["N_Systems", "N_Civic", "N_Historical", "N_Stakeholders", "N_People"] {
        let category = match *key {
            "N_Systems" => "System", "N_Civic" => "CivicSystem",
            "N_Historical" => "Historical", "N_Stakeholders" => "Stakeholder",
            "N_People" => "Person", _ => "Unknown",
        };
        for item in extract_array(json, key) {
            if let Some(id) = get_field(&item, "id") {
                nodes.insert(id, category.to_string());
            }
        }
    }

    for key in &["E_connection", "E_isDevelopedBy", "E_isDeployedBy", "E_place", "E_people"] {
        for item in extract_array(json, key) {
            let source = get_field(&item, "source").unwrap_or_default();
            let target = get_field(&item, "target").unwrap_or_default();
            let label = get_field(&item, "label").unwrap_or_default();
            if !source.is_empty() && !target.is_empty() {
                edges.push((source, target, label));
            }
        }
    }

    SeedGraph { nodes, edges }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn nars_revise_conf(old_c: f32, new_c: f32) -> f32 {
    let w1 = old_c / (1.0 - old_c + 1e-9);
    let w2 = new_c / (1.0 - new_c + 1e-9);
    let total = w1 + w2;
    (total / (total + 1.0)).min(0.99)
}

fn nars_revise_freq(old_f: f32, new_f: f32, old_c: f32, new_c: f32) -> f32 {
    let w1 = old_c / (1.0 - old_c + 1e-9);
    let w2 = new_c / (1.0 - new_c + 1e-9);
    let total = w1 + w2;
    if total < 1e-9 { return old_f; }
    (old_f * w1 + new_f * w2) / total
}

fn clean(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_alphanumeric() && c != ' ' && c != '-')
        .to_string()
}

fn urlencoding(s: &str) -> String {
    s.chars().map(|c| match c {
        ' ' => "+".to_string(),
        'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c.to_string(),
        _ => format!("%{:02X}", c as u32),
    }).collect()
}

fn save_expanded_graph(path: &str, nodes: &HashSet<String>, edges: &[(String, String, String, f32, f32)]) {
    let mut out = String::from("{\n  \"nodes\": [\n");
    for (i, node) in nodes.iter().enumerate() {
        if i > 0 { out.push_str(",\n"); }
        out.push_str(&format!("    {{\"id\": \"{}\"}}", node.replace('"', "\\\"")));
    }
    out.push_str("\n  ],\n  \"edges\": [\n");
    for (i, (s, t, l, f, c)) in edges.iter().enumerate() {
        if i > 0 { out.push_str(",\n"); }
        out.push_str(&format!(
            "    {{\"source\": \"{}\", \"target\": \"{}\", \"label\": \"{}\", \"freq\": {:.2}, \"conf\": {:.2}}}",
            s.replace('"', "\\\""), t.replace('"', "\\\""), l.replace('"', "\\\""), f, c
        ));
    }
    out.push_str("\n  ]\n}\n");
    std::fs::write(path, out).ok();
}

// ── Minimal JSON helpers (no serde, zero deps) ──────────────────────────────

fn extract_array(json: &str, key: &str) -> Vec<String> {
    let pattern = format!("\"{}\"", key);
    let start = match json.find(&pattern) { Some(p) => p, None => return vec![] };
    let after = &json[start + pattern.len()..];
    let bracket = match after.find('[') { Some(p) => start + pattern.len() + p, None => return vec![] };
    let mut depth = 0;
    let mut end = bracket;
    for (i, c) in json[bracket..].char_indices() {
        match c { '[' => depth += 1, ']' => { depth -= 1; if depth == 0 { end = bracket + i; break; } } _ => {} }
    }
    let arr = &json[bracket + 1..end];
    let mut items = Vec::new();
    let (mut d, mut s) = (0, 0);
    for (i, c) in arr.char_indices() {
        match c { '{' => { if d == 0 { s = i; } d += 1; } '}' => { d -= 1; if d == 0 { items.push(arr[s..=i].to_string()); } } _ => {} }
    }
    items
}

fn get_field(obj: &str, key: &str) -> Option<String> {
    let p = format!("\"{}\"", key);
    let pos = obj.find(&p)?;
    let after = obj[pos + p.len()..].trim_start().strip_prefix(':')?;
    let after = after.trim_start();
    if after.starts_with("null") || after.starts_with("NaN") { return None; }
    if after.starts_with('"') {
        let end = after[1..].find('"')?;
        return Some(after[1..1 + end].to_string());
    }
    let end = after.find(|c: char| c == ',' || c == '}')?;
    let v = after[..end].trim();
    if v == "NaN" || v == "null" { None } else { Some(v.to_string()) }
}
