//! THINK: Send text, get a structured thought back.
//!
//! Uses ALL available lenses, domino cascade, NARS voting, qualia 17D.
//! This is the closest we can get to "LLM-like reasoning" without forward passes.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example think -- "Your question or text here"

use thinking_engine::engine::ThinkingEngine;
use thinking_engine::domino::{DominoCascade, CascadeAtom};
use thinking_engine::qualia::{Qualia17D, DIMS_17D, FAMILY_CENTROIDS};
use thinking_engine::jina_lens;
use thinking_engine::bge_m3_lens;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "What is the meaning of love?".to_string()
    };

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  THINK: Multi-Lens HDR Cognitive Engine                 ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
    println!("  Input: \"{}\"\n", input);

    // ── Load tokenizer ──
    let tokenizer = match tokenizers::Tokenizer::from_file("/tmp/bge-m3-tokenizer.json") {
        Ok(t) => t,
        Err(_) => {
            eprintln!("  Tokenizer not found. Download: curl -sL -o /tmp/bge-m3-tokenizer.json \\");
            eprintln!("    https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json");
            return;
        }
    };

    let encoding = tokenizer.encode(input.as_str(), true).expect("tokenize");
    let token_ids = encoding.get_ids();
    let tokens: Vec<String> = encoding.get_tokens().iter().map(|s| s.to_string()).collect();
    println!("  Tokens ({}): {}", token_ids.len(),
        tokens.iter().take(10).map(|s| s.as_str()).collect::<Vec<_>>().join(" "));

    // ── Multi-lens lookup ──
    println!("\n─── LENS VOTING ───\n");

    let jina_centroids = jina_lens::jina_lookup_many(token_ids);
    let bge_centroids = bge_m3_lens::bge_m3_lookup_many(token_ids);

    // Unique centroids per lens
    let jina_unique: std::collections::HashSet<u16> = jina_centroids.iter().cloned().collect();
    let bge_unique: std::collections::HashSet<u16> = bge_centroids.iter().cloned().collect();
    println!("  Jina v3:  {} tokens → {} unique centroids", token_ids.len(), jina_unique.len());
    println!("  BGE-M3:   {} tokens → {} unique centroids", token_ids.len(), bge_unique.len());

    // Agreement: how many tokens map to the SAME centroid in both lenses?
    let mut agree_count = 0;
    for i in 0..token_ids.len() {
        if jina_centroids[i] == bge_centroids[i] { agree_count += 1; }
    }
    println!("  Agreement: {}/{} tokens ({:.0}%)",
        agree_count, token_ids.len(), agree_count as f64 / token_ids.len() as f64 * 100.0);

    // ── Domino cascade on BOTH lenses ──
    println!("\n─── DOMINO CASCADE ───\n");

    // Jina lens cascade
    let jina_engine = jina_lens::jina_engine();
    let jina_counts = vec![1u32; 256];
    let jina_cascade = DominoCascade::new(&jina_engine, &jina_counts);
    let (jina_dom, jina_stages, jina_dis) = jina_cascade.think(&jina_centroids);
    let jina_chain: Vec<u16> = jina_stages.iter()
        .filter_map(|s| s.focus.first().map(|a| a.index)).collect();

    // BGE-M3 lens cascade
    let bge_engine = bge_m3_lens::bge_m3_engine();
    let bge_counts = vec![1u32; 256];
    let bge_cascade = DominoCascade::new(&bge_engine, &bge_counts);
    let (bge_dom, bge_stages, bge_dis) = bge_cascade.think(&bge_centroids);
    let bge_chain: Vec<u16> = bge_stages.iter()
        .filter_map(|s| s.focus.first().map(|a| a.index)).collect();

    println!("  Jina:  dom={:>3} chain={:?} dis={:.2}", jina_dom, jina_chain, jina_dis.total_dissonance);
    println!("  BGE:   dom={:>3} chain={:?} dis={:.2}", bge_dom, bge_chain, bge_dis.total_dissonance);

    // Consensus: do they agree on the dominant?
    let dom_agree = jina_dom == bge_dom;
    let chain_overlap: usize = jina_chain.iter()
        .filter(|a| bge_chain.contains(a)).count();
    println!("  Dominant agree: {}  Chain overlap: {}/{}",
        if dom_agree { "YES ✓" } else { "NO ✗" },
        chain_overlap, jina_chain.len().max(bge_chain.len()));

    // ── Cognitive Markers (best of both) ──
    println!("\n─── COGNITIVE MARKERS ───\n");

    let jina_staunen = jina_stages.iter().map(|s| s.markers.staunen).fold(0.0f32, f32::max);
    let jina_wisdom = jina_stages.iter().map(|s| s.markers.wisdom).fold(0.0f32, f32::max);
    let bge_staunen = bge_stages.iter().map(|s| s.markers.staunen).fold(0.0f32, f32::max);
    let bge_wisdom = bge_stages.iter().map(|s| s.markers.wisdom).fold(0.0f32, f32::max);

    let max_staunen = jina_staunen.max(bge_staunen);
    let max_wisdom = jina_wisdom.max(bge_wisdom);
    let avg_dissonance = (jina_dis.total_dissonance + bge_dis.total_dissonance) / 2.0;

    if max_staunen > 0.05 { println!("  ✨ Staunen (wonder):  {:.2} — novel territory discovered", max_staunen); }
    if max_wisdom > 0.05 { println!("  🦉 Wisdom:           {:.2} — convergent paths confirmed", max_wisdom); }
    if avg_dissonance > 0.1 { println!("  ⚡ Dissonance:       {:.2} — unresolved tension", avg_dissonance); }
    if avg_dissonance < 0.05 { println!("  🕊  Consonance:      {:.2} — harmonious, resolved", avg_dissonance); }

    // ── NARS Truth ──
    let truth_freq = jina_stages.last().map(|s| s.markers.truth_freq).unwrap_or(0.0);
    let truth_conf = if dom_agree { 0.8 } else { 0.4 }; // agreement boosts confidence
    println!("  Truth: freq={:.2} conf={:.2}", truth_freq, truth_conf);

    // ── Qualia 17D ──
    println!("\n─── QUALIA (how this thought FEELS) ───\n");

    // Use Jina engine for qualia (richer semantics)
    let qualia = Qualia17D::from_engine(&jina_engine);
    let (family, dist) = qualia.nearest_family();
    println!("  Family: {} (dist={:.2})", family, dist);

    // Show top-5 most active dimensions
    let mut dims: Vec<(&str, f32)> = DIMS_17D.iter().zip(&qualia.dims)
        .map(|(&name, &val)| (name, val)).collect();
    dims.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    println!("  Strongest dims:");
    for (name, val) in dims.iter().take(5) {
        let bar_len = (val.abs() * 15.0).min(15.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("    {:<14} {:.2}  {}", name, val, bar);
    }

    // ── Structured Answer ──
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  STRUCTURED THOUGHT                                     ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // The "answer" is the topology traversal interpreted through qualia
    println!("  Topic:      \"{}\"", input);
    println!("  Tokens:     {}", token_ids.len());
    println!("  Lenses:     Jina v3 + BGE-M3 (multi-lens vote)");
    println!("  Consensus:  {}", if dom_agree { "STRONG" } else { "DIVERGENT" });
    println!("  Dominant:   atom {} (Jina) / atom {} (BGE)", jina_dom, bge_dom);
    println!("  Feel:       {} ({:.2})", family, dist);
    println!("  Clarity:    {:.2}", qualia.dims[4]);
    println!("  Tension:    {:.2}", qualia.dims[2]);
    println!("  Confidence: {:.2}", truth_conf);

    // Interpret the cascade chain as a "reasoning path"
    println!("\n  Reasoning path (Jina):");
    for (i, stage) in jina_stages.iter().enumerate() {
        if let Some(focus) = stage.focus.first() {
            let markers = &stage.markers;
            let mut flags = String::new();
            if markers.staunen > 0.05 { flags += " ✨"; }
            if markers.wisdom > 0.05 { flags += " 🦉"; }
            println!("    step {}: atom {:>3} (freq={:.2} conf={:.2}){}",
                i, focus.index, focus.frequency, focus.confidence, flags);
        }
    }

    // Final verdict
    println!("\n  ────────────────────────────────────────");
    if dom_agree && avg_dissonance < 0.1 {
        println!("  Verdict: Both lenses converge. High confidence thought.");
        println!("  The topology says: this input activates a clear, consonant region.");
    } else if dom_agree {
        println!("  Verdict: Lenses agree on destination but the path is turbulent.");
        println!("  Dissonance {:.2} suggests unresolved tension in the reasoning.", avg_dissonance);
    } else {
        println!("  Verdict: Lenses DISAGREE — the input is ambiguous or novel.");
        println!("  Jina sees atom {}, BGE sees atom {}. Different semantic readings.", jina_dom, bge_dom);
        println!("  This is WHERE creative insight lives — at the boundary of interpretations.");
    }

    println!("\n  Time: {:.1}ms (tokenize + 2 cascades + qualia)", 0.0); // placeholder
    println!();
}
