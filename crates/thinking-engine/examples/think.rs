//! THINK: full Friston loop — ghost prediction → cascade → free energy → learn
//!
//! Processes multiple sentences sequentially. Each thought's ghosts
//! bias the NEXT thought's cascade. Free energy measures surprise.
//! The system learns across sentences within a session.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example think -- "sentence 1" "sentence 2" "sentence 3"

use thinking_engine::engine::ThinkingEngine;
use thinking_engine::domino::DominoCascade;
use thinking_engine::qualia::Qualia17D;
use thinking_engine::superposition;
use thinking_engine::ghosts::{GhostField, GhostType};
use thinking_engine::cognitive_trace::CognitiveTrace;
use thinking_engine::centroid_labels::JINA_CENTROID_LABELS;
use thinking_engine::jina_lens;
use thinking_engine::bge_m3_lens;

fn label(c: u16) -> &'static str {
    JINA_CENTROID_LABELS.get(c as usize).copied().unwrap_or("?")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sentences: Vec<String> = if args.len() > 1 {
        args[1..].iter().map(|s| s.clone()).collect()
    } else {
        vec![
            "The cat sat on the mat.".into(),
            "I feel deeply sad about losing someone.".into(),
            "The wound is where the light enters you.".into(),
            "Set your life on fire. Seek those who fan your flames.".into(),
            "What is the meaning of love?".into(),
        ]
    };

    let tokenizer = match tokenizers::Tokenizer::from_file("/tmp/bge-m3-tokenizer.json") {
        Ok(t) => t,
        Err(_) => { eprintln!("Need /tmp/bge-m3-tokenizer.json"); return; }
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  FRISTON LOOP: Ghost Prediction → Cascade → Free Energy    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Ghost field persists across all sentences
    let mut ghost_field = GhostField::new();
    let kg_path = std::path::Path::new("/tmp/codebooks/knowledge_graph.tsv");

    for (si, text) in sentences.iter().enumerate() {
        println!("━━━ Thought {} of {} ━━━", si + 1, sentences.len());
        println!("  \"{}\"", text);

        let encoding = tokenizer.encode(text.as_str(), true).expect("tokenize");
        let token_ids = encoding.get_ids();
        let tokens: Vec<String> = encoding.get_tokens().iter().map(|s| s.to_string()).collect();

        // ── Ghost prediction (before thinking) ──
        let ghost_pred = ghost_field.prediction(256);
        let ghost_count = ghost_field.active_count();
        if ghost_count > 0 {
            let top_ghosts = ghost_field.summary();
            println!("\n  👻 Ghost field: {} active ghosts (prediction from past thoughts)", ghost_count);
            for (atom, gtype, intensity) in top_ghosts.iter().take(3) {
                println!("    atom {:>3} [{}] {} intensity={:.2}",
                    atom, label(*atom), gtype, intensity);
            }
        }

        // ── Multi-lens cascade with ghost bias ──
        let jina_cents = jina_lens::jina_lookup_many(token_ids);
        let bge_cents = bge_m3_lens::bge_m3_lookup_many(token_ids);

        let jina_engine = jina_lens::jina_engine();
        let bge_engine = bge_m3_lens::bge_m3_engine();

        // Jina cascade WITH ghost bias
        let jina_cascade = DominoCascade::new(&jina_engine, &vec![1u32; 256])
            .with_ghost_bias(ghost_pred.clone());
        let (jina_dom, jina_stages, jina_dis) = jina_cascade.think(&jina_cents);
        let jina_chain: Vec<u16> = jina_stages.iter()
            .filter_map(|s| s.focus.first().map(|a| a.index)).collect();

        // BGE cascade WITH ghost bias
        let bge_cascade = DominoCascade::new(&bge_engine, &vec![1u32; 256])
            .with_ghost_bias(ghost_pred.clone());
        let (bge_dom, bge_stages, bge_dis) = bge_cascade.think(&bge_cents);
        let bge_chain: Vec<u16> = bge_stages.iter()
            .filter_map(|s| s.focus.first().map(|a| a.index)).collect();

        println!("\n  Jina: {} → {}  BGE: {} → {}",
            label(jina_chain.first().copied().unwrap_or(0)),
            label(jina_dom),
            label(bge_chain.first().copied().unwrap_or(0)),
            label(bge_dom));

        // ── Superposition ──
        let dis_avg = (jina_dis.total_dissonance + bge_dis.total_dissonance) / 2.0;
        let (field, style, gated) = superposition::superposition_cascade(
            &[&jina_stages, &bge_stages], 256, dis_avg,
            &superposition::StyleThresholds::default(),
        );
        let dom_agree = jina_dom == bge_dom;
        let confidence = if dom_agree { 0.85 } else if !gated.is_empty() { 0.5 } else { 0.3 };

        // ── Qualia from superposition ──
        let qualia = Qualia17D::from_superposition(&field, &style, dis_avg, confidence);
        let (_, _, blend_name, _) = qualia.emotional_blend();

        // ── Free energy: was ghost prediction accurate? ──
        let actual_energy: Vec<f32> = field.amplitudes.clone();
        let free_energy = ghost_field.free_energy(&actual_energy);

        // ── Markers ──
        let staunen_max = jina_stages.iter().chain(bge_stages.iter())
            .map(|s| s.markers.staunen).fold(0.0f32, f32::max);
        let wisdom_max = jina_stages.iter().chain(bge_stages.iter())
            .map(|s| s.markers.wisdom).fold(0.0f32, f32::max);

        println!("  Style: {}  Blend: {}", style, blend_name);
        println!("  Resonant: {}/256  Gated: {}  Confidence: {:.0}%",
            field.n_resonant, gated.len(), confidence * 100.0);
        if ghost_count > 0 {
            println!("  Free energy: {:.4} {}",
                free_energy,
                if free_energy < 0.01 { "(LOW — ghosts predicted well → autocomplete)" }
                else if free_energy < 0.05 { "(moderate — partial match)" }
                else { "(HIGH — surprise! ghosts wrong → learning)" });
        }

        // ── Markers ──
        let mut markers = Vec::new();
        if staunen_max > 0.3 { markers.push(format!("✨ wonder {:.1}", staunen_max)); }
        if wisdom_max > 0.1 { markers.push(format!("🦉 wisdom {:.1}", wisdom_max)); }
        if dis_avg > 0.2 { markers.push(format!("⚡ tension {:.2}", dis_avg)); }
        if dis_avg < 0.05 { markers.push("🕊 calm".into()); }
        if !markers.is_empty() {
            println!("  {}", markers.join("  "));
        }

        // ── SPO triples ──
        let spo = CognitiveTrace::extract_spo(
            &gated,
            |a, b| jina_lens::jina_distance(a, b),
            |a, b| bge_m3_lens::bge_m3_distance(a, b),
            0.6,
        );
        if !spo.is_empty() {
            println!("  SPO: {} triples (top: ({}) —[{}]→ ({}) c={:.2})",
                spo.len(),
                label(spo[0].subject), spo[0].predicate, label(spo[0].object),
                spo[0].confidence);
        }

        // ── Imprint ghosts for NEXT thought ──
        ghost_field.imprint(
            &field.resonant_atoms,
            &style,
            staunen_max,
            wisdom_max,
            dis_avg,
            text,
        );

        // ── Knowledge graph append ──
        let trace = CognitiveTrace {
            input: text.clone(),
            token_ids: token_ids.to_vec(),
            tokens: tokens.clone(),
            lens_results: vec![],
            superposition: field,
            style: style.clone(),
            gated_atoms: gated,
            qualia: qualia.clone(),
            blend: blend_name.clone(),
            primary_family: String::new(),
            overlay_family: String::new(),
            spo_triples: spo,
            confidence,
            dissonance: dis_avg,
            staunen_max,
            wisdom_max,
        };
        trace.append_to_knowledge_graph(kg_path).ok();

        println!();
    }

    // ── Session summary ──
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SESSION COMPLETE                                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("  {} thoughts processed", sentences.len());
    println!("  {} ghosts accumulated", ghost_field.active_count());
    ghost_field.prune();
    println!("  {} ghosts after prune", ghost_field.active_count());

    let kg_lines = std::fs::read_to_string(kg_path)
        .map(|s| s.lines().count()).unwrap_or(0);
    println!("  {} SPO triples in knowledge graph", kg_lines);

    println!("\n  Ghost field summary:");
    for (atom, gtype, intensity) in ghost_field.summary().iter().take(10) {
        println!("    atom {:>3} [{}] {} = {:.3}",
            atom, label(*atom), gtype, intensity);
    }
    println!();
}
