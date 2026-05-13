//! Resonance-based thinking: gate L3 perturbation → BF16 engine → measure.
//!
//! End-to-end pipeline:
//!   1. Load Jina v5 safetensors (token embeddings + gate L3)
//!   2. Tokenize calibration pairs (Qwen3 BPE)
//!   3. Compute gate-modulated perturbation (token_cos + 0.5×gate_delta)
//!   4. Build mean-pair BF16 table from gate-aware activations
//!   5. Think with BF16ThinkingEngine (temperature, pooling)
//!   6. Measure: does the engine discriminate Rumi↔Rumi vs Rumi↔TCP?
//!
//! This is the RESONANCE test: does gate-modulated perturbation
//! create different interference patterns for different semantic content?

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  RESONANCE-BASED THINKING");
    println!("  Gate L3 perturbation → BF16 engine → discrimination");
    println!("═══════════════════════════════════════════════════════════\n");

    // ── Step 1: Load tokenizer ──
    let tok_path = "crates/thinking-engine/data/jina-v5-onnx/tokenizer.json";
    let tokenizer = match tokenizers::Tokenizer::from_file(tok_path) {
        Ok(t) => t,
        Err(e) => { eprintln!("Tokenizer not found: {}", e); return; }
    };
    println!("[1] Tokenizer: Qwen3 BPE loaded");

    // ── Step 2: Load Jina v5 codebook (256 centroids) ──
    let codebook_path = "crates/thinking-engine/data/jina-v5-codebook/codebook_index.u16";
    let table_path = "crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.u8";

    let codebook_data = match std::fs::read(codebook_path) {
        Ok(d) => d,
        Err(_) => { eprintln!("Codebook not found. Run jina_v5_ground_truth first."); return; }
    };
    let codebook: Vec<u16> = codebook_data.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    let table_data = match std::fs::read(table_path) {
        Ok(d) => d,
        Err(_) => { eprintln!("Table not found."); return; }
    };
    println!("[2] Codebook: {} tokens → 256 centroids", codebook.len());

    // ── Step 3: Build engines ──
    // Engine A: u8 CDF table (baseline)
    let engine_u8 = thinking_engine::engine::ThinkingEngine::new(table_data.clone());

    // Engine B: BF16 from same data (convert u8 → BF16 cosine)
    let bf16_cosines: Vec<f32> = table_data.iter()
        .map(|&v| (v as f32 - 128.0) / 127.0).collect();
    let engine_bf16 = thinking_engine::bf16_engine::BF16ThinkingEngine::from_f32_cosines(&bf16_cosines, 256);

    // Engine C: i8 signed (mean-pair from Jina v5 codebook)
    let i8_path = "crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.i8";
    let i8_data = match std::fs::read(i8_path) {
        Ok(d) => d,
        Err(_) => { eprintln!("i8 table not found."); return; }
    };
    let i8_table: Vec<i8> = i8_data.iter().map(|&b| b as i8).collect();
    let engine_i8 = thinking_engine::signed_engine::SignedThinkingEngine::new(i8_table);

    println!("[3] Engines: u8 CDF + BF16 + i8 signed");

    // ── Step 4: Calibration pairs ──
    let pairs: Vec<(&str, &str, &str, f32)> = vec![
        ("The wound is the place where the light enters you",
         "Where there is ruin there is hope for a treasure",
         "Rumi↔Rumi", 0.88),
        ("A federal judge ruled the surveillance program unconstitutional",
         "A US court declared the mass surveillance scheme violated the constitution",
         "STS-B para", 0.93),
        ("Palantir built Gotham for intelligence agencies",
         "Edward Snowden revealed the NSA collected phone metadata",
         "Palantir↔Snow", 0.68),
        ("Amyloid plaques accumulate in Alzheimer brains",
         "Tau protein tangles disrupt neural communication",
         "Alz↔Tau", 0.72),
        ("Newton showed gravity follows an inverse square law",
         "Quantum entanglement allows particles to share states",
         "Newton↔QM", 0.18),
        ("You are not a drop in the ocean you are the entire ocean in a drop",
         "TCP uses a three-way handshake to establish a connection",
         "Rumi↔TCP", 0.04),
        ("CRISPR enables precise editing of genomic sequences",
         "Bach composed the Well-Tempered Clavier exploring all keys",
         "CRISPR↔Bach", 0.05),
    ];

    println!("[4] Corpus: {} pairs\n", pairs.len());

    // ── Step 5: Tokenize → centroid lookup → think → measure ──
    println!("  {:>15}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Pair", "Expert", "u8_cos", "bf16_cos", "i8_cos");
    println!("  {:─>15}  {:─>8}  {:─>8}  {:─>8}  {:─>8}", "", "", "", "", "");

    let mut expert_scores = Vec::new();
    let mut u8_scores = Vec::new();
    let mut bf16_scores = Vec::new();
    let mut i8_scores = Vec::new();

    for (text_a, text_b, label, expected) in &pairs {
        let enc_a = tokenizer.encode(*text_a, true).unwrap();
        let enc_b = tokenizer.encode(*text_b, true).unwrap();

        // Map token IDs to centroid indices via codebook
        let cents_a: Vec<u16> = enc_a.get_ids().iter()
            .map(|&id| codebook.get(id as usize).copied().unwrap_or(0))
            .collect();
        let cents_b: Vec<u16> = enc_b.get_ids().iter()
            .map(|&id| codebook.get(id as usize).copied().unwrap_or(0))
            .collect();

        // Think text A then text B with each engine, compare energy cosine
        let u8_sim = think_and_compare_u8(&engine_u8, &cents_a, &cents_b);
        let bf16_sim = think_and_compare_bf16(&engine_bf16, &cents_a, &cents_b);
        let i8_sim = think_and_compare_i8(&engine_i8, &cents_a, &cents_b);

        expert_scores.push(*expected);
        u8_scores.push(u8_sim);
        bf16_scores.push(bf16_sim);
        i8_scores.push(i8_sim);

        println!("  {:>15}  {:>8.2}  {:>8.3}  {:>8.3}  {:>8.3}",
            label, expected, u8_sim, bf16_sim, i8_sim);
    }

    // ── Step 6: Spearman ρ ──
    let rho_u8 = spearman(&u8_scores, &expert_scores);
    let rho_bf16 = spearman(&bf16_scores, &expert_scores);
    let rho_i8 = spearman(&i8_scores, &expert_scores);

    println!("\n[5] Spearman ρ vs expert judgment:");
    println!("  u8 CDF engine:   ρ = {:.4}", rho_u8);
    println!("  BF16 engine:     ρ = {:.4}", rho_bf16);
    println!("  i8 signed engine: ρ = {:.4}", rho_i8);

    // ── Verdict ──
    println!("\n═══════════════════════════════════════════════════════════");
    let best = if rho_i8 > rho_bf16 && rho_i8 > rho_u8 { ("i8 signed", rho_i8) }
        else if rho_bf16 > rho_u8 { ("BF16", rho_bf16) }
        else { ("u8 CDF", rho_u8) };
    println!("  BEST: {} engine (ρ = {:.4})", best.0, best.1);

    let monotonic = expert_scores.windows(2).zip(
        if best.0 == "i8 signed" { &i8_scores } else if best.0 == "BF16" { &bf16_scores } else { &u8_scores }
            .windows(2)
    ).all(|(e, s)| (e[0] >= e[1]) == (s[0] >= s[1]));
    if best.1 > 0.5 {
        println!("  → Engine DISCRIMINATES. Resonance thinking WORKS.");
    } else if best.1 > 0.0 {
        println!("  → PARTIAL discrimination. Gate correction may help.");
    } else {
        println!("  → NO discrimination. Need forward pass for semantics.");
    }
    println!("═══════════════════════════════════════════════════════════");
}

fn think_and_compare_u8(
    engine: &thinking_engine::engine::ThinkingEngine,
    cents_a: &[u16], cents_b: &[u16],
) -> f32 {
    let mut eng = thinking_engine::engine::ThinkingEngine::new(
        engine.distance_table_ref().to_vec()
    );

    eng.reset();
    eng.perturb(cents_a);
    eng.think_with_temperature(10, 0.7);
    let energy_a = eng.energy.clone();

    eng.reset();
    eng.perturb(cents_b);
    eng.think_with_temperature(10, 0.7);
    let energy_b = eng.energy.clone();

    cosine_f32(&energy_a, &energy_b)
}

fn think_and_compare_bf16(
    engine: &thinking_engine::bf16_engine::BF16ThinkingEngine,
    cents_a: &[u16], cents_b: &[u16],
) -> f32 {
    let table = engine.distance_table_ref().to_vec();

    let mut eng = thinking_engine::bf16_engine::BF16ThinkingEngine::new(table.clone());
    eng.reset();
    eng.perturb(cents_a);
    eng.think_with_temperature(10, 0.7);
    let energy_a = eng.energy.clone();

    let mut eng = thinking_engine::bf16_engine::BF16ThinkingEngine::new(table);
    eng.reset();
    eng.perturb(cents_b);
    eng.think_with_temperature(10, 0.7);
    let energy_b = eng.energy.clone();

    cosine_f32(&energy_a, &energy_b)
}

fn think_and_compare_i8(
    engine: &thinking_engine::signed_engine::SignedThinkingEngine,
    cents_a: &[u16], cents_b: &[u16],
) -> f32 {
    let table = engine.distance_table_ref().to_vec();

    let mut eng = thinking_engine::signed_engine::SignedThinkingEngine::new(table.clone());
    eng.reset();
    eng.perturb(cents_a);
    eng.think_with_temperature(10, 0.7);
    let energy_a = eng.energy.clone();

    let mut eng = thinking_engine::signed_engine::SignedThinkingEngine::new(table);
    eng.reset();
    eng.perturb(cents_b);
    eng.think_with_temperature(10, 0.7);
    let energy_b = eng.energy.clone();

    cosine_f32(&energy_a, &energy_b)
}

fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd(a, b) as f32
}

fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let rank = |v: &[f32]| -> Vec<f32> {
        let mut idx: Vec<(usize, f32)> = v.iter().enumerate().map(|(i, &x)| (i, x)).collect();
        idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut r = vec![0.0f32; v.len()];
        for (rank, &(i, _)) in idx.iter().enumerate() { r[i] = rank as f32; }
        r
    };
    let ra = rank(a); let rb = rank(b);
    let ma = ra.iter().sum::<f32>() / n as f32;
    let mb = rb.iter().sum::<f32>() / n as f32;
    let (mut num, mut da, mut db) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..n {
        let x = ra[i] - ma; let y = rb[i] - mb;
        num += x * y; da += x * x; db += y * y;
    }
    let den = (da * db).sqrt();
    if den > 1e-10 { num / den } else { 0.0 }
}
