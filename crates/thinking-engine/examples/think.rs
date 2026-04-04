//! THINK: text → structured thought with verbose per-lens debug.
//!
//! Shows each lens's individual answer, the ripple convergence,
//! and the joined multi-lens awareness verdict.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example think -- "Your text here"

use thinking_engine::engine::ThinkingEngine;
use thinking_engine::domino::DominoCascade;
use thinking_engine::qualia::{Qualia17D, DIMS_17D};
use thinking_engine::jina_lens;
use thinking_engine::bge_m3_lens;
use thinking_engine::centroid_labels::JINA_CENTROID_LABELS;

fn label(centroid: u16) -> &'static str {
    if (centroid as usize) < JINA_CENTROID_LABELS.len() {
        JINA_CENTROID_LABELS[centroid as usize]
    } else { "?" }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = if args.len() > 1 { args[1..].join(" ") }
    else { "What is the meaning of love?".to_string() };

    let tokenizer = match tokenizers::Tokenizer::from_file("/tmp/bge-m3-tokenizer.json") {
        Ok(t) => t,
        Err(_) => { eprintln!("Need /tmp/bge-m3-tokenizer.json"); return; }
    };

    let encoding = tokenizer.encode(input.as_str(), true).expect("tokenize");
    let token_ids = encoding.get_ids();
    let tokens: Vec<String> = encoding.get_tokens().iter().map(|s| s.to_string()).collect();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  \"{}\"", &input[..input.len().min(55)]);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  tokens: {}\n", tokens.iter().take(12).map(|s| s.as_str()).collect::<Vec<_>>().join(" "));

    // ═══ PER-LENS INDIVIDUAL ANSWERS ═══
    println!("─── LENS 1: Jina v3 (semantic similarity) ───\n");
    let jina_result = run_lens("Jina", token_ids, |ids| jina_lens::jina_lookup_many(ids), &jina_lens::jina_engine());

    println!("─── LENS 2: BGE-M3 (multilingual retrieval) ───\n");
    let bge_result = run_lens("BGE", token_ids, |ids| bge_m3_lens::bge_m3_lookup_many(ids), &bge_m3_lens::bge_m3_engine());

    // ═══ RIPPLE CONVERGENCE ═══
    println!("─── RIPPLE CONVERGENCE ───\n");

    // Show how each lens's cascade ripples through the table
    println!("  Jina ripple:  {} → {} → {} → {} → {}",
        label(jina_result.chain[0]), label(jina_result.chain.get(1).copied().unwrap_or(0)),
        label(jina_result.chain.get(2).copied().unwrap_or(0)),
        label(jina_result.chain.get(3).copied().unwrap_or(0)),
        label(jina_result.dominant));
    println!("  BGE  ripple:  {} → {} → {} → {} → {}",
        label(bge_result.chain[0]), label(bge_result.chain.get(1).copied().unwrap_or(0)),
        label(bge_result.chain.get(2).copied().unwrap_or(0)),
        label(bge_result.chain.get(3).copied().unwrap_or(0)),
        label(bge_result.dominant));

    // Find where ripples CONVERGE (shared atoms in chains)
    let jina_set: std::collections::HashSet<u16> = jina_result.chain.iter().cloned().collect();
    let bge_set: std::collections::HashSet<u16> = bge_result.chain.iter().cloned().collect();
    let convergent: Vec<u16> = jina_set.intersection(&bge_set).cloned().collect();
    if !convergent.is_empty() {
        println!("\n  ✓ Ripples CONVERGE at: {}",
            convergent.iter().map(|&c| label(c)).collect::<Vec<_>>().join(", "));
    } else {
        println!("\n  ✗ Ripples DIVERGE — different semantic territory");
    }

    // ═══ JOINED AWARENESS ═══
    println!("\n─── JOINED AWARENESS ───\n");

    let dom_agree = jina_result.dominant == bge_result.dominant;
    let dis_avg = (jina_result.dissonance + bge_result.dissonance) / 2.0;
    let confidence = if dom_agree { 0.85 } else if !convergent.is_empty() { 0.6 } else { 0.35 };

    // Qualia from Jina (richer semantics)
    let qualia = &jina_result.qualia;
    let (family, _) = qualia.nearest_family();

    println!("  Consensus:    {}", if dom_agree { "STRONG — both lenses agree" }
        else if !convergent.is_empty() { "PARTIAL — ripples share territory" }
        else { "WEAK — lenses see different things" });
    println!("  Confidence:   {:.0}%", confidence * 100.0);
    println!("  Dissonance:   {:.2}{}", dis_avg,
        if dis_avg > 0.3 { " (turbulent)" } else if dis_avg > 0.1 { " (some tension)" } else { " (calm)" });
    println!("  Feel:         {}", family);

    // Show which tokens drive the strongest activations
    println!("\n  Token activations:");
    let jina_cents = jina_lens::jina_lookup_many(token_ids);
    for (i, tok) in tokens.iter().enumerate().take(10) {
        if tok == "<s>" || tok == "</s>" { continue; }
        let c = jina_cents[i];
        let tok_clean = tok.replace('▁', "");
        println!("    {:>12} → c{:>3} [{}]", tok_clean, c, label(c));
    }

    // ═══ VERBOSE: per-stage ripple detail ═══
    println!("\n─── RIPPLE DETAIL (Jina) ───\n");
    for (i, stage) in jina_result.stages.iter().enumerate() {
        let focus_labels: Vec<String> = stage.focus.iter().take(3)
            .map(|a| format!("{}({})", label(a.index), a.index))
            .collect();
        let m = &stage.markers;
        let mut flags = String::new();
        if m.staunen > 0.05 { flags += &format!(" ✨{:.1}", m.staunen); }
        if m.wisdom > 0.05 { flags += &format!(" 🦉{:.1}", m.wisdom); }

        println!("  stage {}: [{}]  truth({:.1},{:.1}){}",
            i, focus_labels.join(", "), m.truth_freq, m.truth_conf, flags);
    }

    // ═══ SUPERPOSITION GATE ═══
    println!("─── SUPERPOSITION (multi-lens interference) ───\n");

    let (field, style, gated) = thinking_engine::superposition::superposition_cascade(
        &[&jina_result.stages, &bge_result.stages],
        256,
        dis_avg,
        &thinking_engine::superposition::StyleThresholds::default(),
    );

    println!("  Resonant atoms: {} / 256 ({:.0}%)",
        field.n_resonant, field.n_resonant as f64 / 256.0 * 100.0);
    println!("  Total energy:   {:.1}", field.total_energy);
    println!("  Thinking style: {}", style);

    // Show top resonant atoms (constructive interference)
    println!("  Top resonance peaks:");
    for (atom, amp) in field.resonant_atoms.iter().take(5) {
        println!("    atom {:>3} [{}] amplitude={:.2}",
            atom, label(*atom), amp);
    }

    // Gated survivors (what passes the threshold)
    if !gated.is_empty() {
        println!("  Gated survivors ({}):", gated.len());
        for &a in gated.iter().take(8) {
            println!("    → {} (c{})", label(a), a);
        }
    } else {
        println!("  No atoms survive the gate — fully destructive interference.");
    }

    // SPO resonance: each gated atom pair = a potential Subject-Predicate-Object triple
    if gated.len() >= 2 {
        println!("\n  SPO resonance (gated atom pairs → potential triples):");
        for i in 0..gated.len().min(3) {
            for j in (i+1)..gated.len().min(4) {
                let a = gated[i];
                let b = gated[j];
                let dist_jina = thinking_engine::jina_lens::jina_distance(a, b);
                let dist_bge = thinking_engine::bge_m3_lens::bge_m3_distance(a, b);
                let agreement = 1.0 - (dist_jina as f32 - dist_bge as f32).abs() / 255.0;
                println!("    ({}) —[{:.0}%]→ ({})  jina:{} bge:{}",
                    label(a), agreement * 100.0, label(b), dist_jina, dist_bge);
            }
        }
    }

    // ═══ FINAL ANSWER ═══
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  ANSWER                                                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // The "answer" = the convergent meaning from both lenses
    let jina_meaning = label(jina_result.dominant);
    let bge_meaning = label(bge_result.dominant);

    if dom_agree {
        println!("  This thought converges to: {}", jina_meaning);
        println!("  Both lenses see the same territory. Confidence {:.0}%.", confidence * 100.0);
    } else if !convergent.is_empty() {
        let shared_meaning: Vec<&str> = convergent.iter().map(|&c| label(c)).collect();
        println!("  Jina sees: {}", jina_meaning);
        println!("  BGE sees:  {}", bge_meaning);
        println!("  They meet at: {}", shared_meaning.join(", "));
        println!("  The thought HOLDS BOTH perspectives. Confidence {:.0}%.", confidence * 100.0);
    } else {
        println!("  Jina interpretation: {}", jina_meaning);
        println!("  BGE interpretation:  {}", bge_meaning);
        println!("  These are DIFFERENT readings of the same text.");
        println!("  The ambiguity is the answer. Confidence {:.0}%.", confidence * 100.0);
    }

    if dis_avg > 0.2 {
        println!("\n  ⚡ Unresolved tension — this thought hasn't settled.");
    }
    if jina_result.max_staunen > 0.5 {
        println!("  ✨ High wonder — novel conceptual territory.");
    }
    if jina_result.max_wisdom > 0.3 {
        println!("  🦉 Wisdom detected — multiple paths confirm this.");
    }

    println!("\n  Family: {}  Clarity: {:.1}  Tension: {:.1}",
        family, qualia.dims[4], qualia.dims[2]);
    println!();
}

struct LensResult {
    dominant: u16,
    chain: Vec<u16>,
    dissonance: f32,
    max_staunen: f32,
    max_wisdom: f32,
    qualia: Qualia17D,
    stages: Vec<thinking_engine::domino::StageResult>,
}

fn run_lens(
    name: &str,
    token_ids: &[u32],
    lookup: impl Fn(&[u32]) -> Vec<u16>,
    engine: &ThinkingEngine,
) -> LensResult {
    let centroids = lookup(token_ids);
    let unique: std::collections::HashSet<u16> = centroids.iter().cloned().collect();

    let counts = vec![1u32; engine.size];
    let cascade = DominoCascade::new(engine, &counts);
    let (dom, stages, dis) = cascade.think(&centroids);
    let chain: Vec<u16> = stages.iter()
        .filter_map(|s| s.focus.first().map(|a| a.index)).collect();

    let max_staunen = stages.iter().map(|s| s.markers.staunen).fold(0.0f32, f32::max);
    let max_wisdom = stages.iter().map(|s| s.markers.wisdom).fold(0.0f32, f32::max);

    let qualia = Qualia17D::from_engine(engine);

    println!("  {} unique centroids from {} tokens", unique.len(), token_ids.len());
    println!("  Dominant: c{} [{}]", dom, label(dom));
    println!("  Chain: {}", chain.iter().map(|&c| format!("{}", label(c))).collect::<Vec<_>>().join(" → "));
    println!("  Dissonance: {:.2}  Staunen: {:.2}  Wisdom: {:.2}\n",
        dis.total_dissonance, max_staunen, max_wisdom);

    LensResult { dominant: dom, chain, dissonance: dis.total_dissonance,
        max_staunen, max_wisdom, qualia, stages }
}
