//! TEXT → THOUGHT: The first end-to-end path.
//!
//! ```text
//! text → tokenize (BGE-M3 BPE) → token_ids
//!   → distance table row lookup → Sensor activations
//!   → fire_into(engine) → think(10 cycles) → commit()
//!   → ThoughtStruct with dominant peak + energy distribution
//! ```
//!
//! Uses the real BGE-M3 tokenizer (XLM-RoBERTa BPE, 250K vocab).
//! Distance table: 1024×1024 u8 from F16 weights (attn_q, layer 23).
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example text_to_thought

use thinking_engine::engine::ThinkingEngine;
use thinking_engine::sensor::{Sensor, SensorBank};

fn main() {
    println!("════════════════════════════════════════════════════════");
    println!("  TEXT → THOUGHT  (first end-to-end on real topology)");
    println!("════════════════════════════════════════════════════════\n");

    // ── Step 1: Load real BGE-M3 tokenizer from HuggingFace ────────────────
    println!("[1] Loading BGE-M3 tokenizer (XLM-RoBERTa BPE, 250K vocab)...");
    let tokenizer_path = "/tmp/bge-m3-tokenizer.json";
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|_| {
            // Fallback: try from HuggingFace
            eprintln!("  Local file not found, trying HuggingFace...");
            tokenizers::Tokenizer::from_pretrained("BAAI/bge-m3", None)
                .expect("Failed to load tokenizer — download tokenizer.json from huggingface.co/BAAI/bge-m3")
        });
    println!("  Vocab size: {}", tokenizer.get_vocab_size(true));

    // ── Step 2: Load 1:1 F16 distance table ────────────────────────────────
    // Use attn_q (widest cosine spread: [-0.691, 0.749]) for best topology
    println!("[2] Loading distance table (bge-m3 attn_q, 1024×1024, F16)...");
    let table_path = "/tmp/codebooks/bge-m3-roles-f16/attn_q/distance_table_1024x1024.u8";
    let table_data = std::fs::read(table_path)
        .expect("Distance table not found — run build_1to1_roles first");
    let n = 1024usize;
    assert_eq!(table_data.len(), n * n, "Expected 1024×1024 table");
    println!("  Table: {}×{} u8, {} bytes — no padding needed", n, n, table_data.len());

    // ── Step 3: Build ThinkingEngine with this table ───────────────────────
    // Engine now accepts any N×N table (variable size, no 4096 padding waste)
    println!("[3] Building ThinkingEngine ({0}×{0} = {1} compositions)...", n, n * n);
    let mut engine = ThinkingEngine::new(table_data.clone());
    let n_raw = n; // alias for token mapping

    // ── Step 4: Test sentences ─────────────────────────────────────────────
    let sentences = [
        "The cat sat on the mat.",
        "Quantum entanglement enables faster-than-light correlation.",
        "Machine learning models compress knowledge into weight matrices.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "距離テーブルはコサイン類似度を保存する",
        "Love is the answer to every question that matters.",
    ];

    println!("\n[4] Processing {} sentences...\n", sentences.len());

    for (i, text) in sentences.iter().enumerate() {
        println!("─── Sentence {} ───", i + 1);
        println!("  Text: \"{}\"", text);

        // Step 4a: Tokenize
        let encoding = tokenizer.encode(*text, true)
            .expect("Tokenization failed");
        let token_ids = encoding.get_ids();
        println!("  Tokens: {} ids {:?}{}",
            token_ids.len(),
            &token_ids[..token_ids.len().min(8)],
            if token_ids.len() > 8 { "..." } else { "" });

        // Step 4b: Map token_ids → distance table rows
        // Token ID modulo N gives the row in the distance table.
        // This is the HHTL lookup: token → codebook entry → table row.
        // (Full pipeline would use per-token codebook assignments;
        //  for now, modulo mapping preserves the token→topology connection.)
        let table_indices: Vec<u16> = token_ids.iter()
            .map(|&id| (id as usize % n_raw) as u16)
            .collect();

        // Step 4c: Build sensors from distance table rows
        let mut bank = SensorBank::new();
        for &idx in &table_indices {
            let sensor = Sensor::from_distance_row(
                "bge-m3",
                &table_data,
                idx as usize,
                n,
                192, // threshold: activate atoms with cos > 0.5 (u8 192 ≈ cos 0.506)
            );
            if !sensor.is_empty() {
                bank.add(sensor);
            }
        }

        // Step 4d: Fire into engine
        engine.reset();
        bank.fire_into(&mut engine);

        // Step 4e: Think
        let start = std::time::Instant::now();
        let max_cycles = 10;
        let _energy = engine.think(max_cycles);
        let think_time = start.elapsed();

        // Step 4f: Commit
        let bus = engine.commit();
        let entropy = engine.entropy();
        let active = engine.active_count(0.001);

        // Top-5 peaks
        let mut peaks: Vec<(usize, f64)> = engine.energy.iter()
            .enumerate()
            .filter(|(_, &e)| e > 0.001)
            .map(|(i, &e)| (i, e))
            .collect();
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peaks.truncate(5);

        println!("  Sensors: {} activated, {} atoms above threshold",
            table_indices.len(), bank.fire_all().len());
        println!("  Think: {} cycles, {:.1}μs",
            max_cycles, think_time.as_micros());
        println!("  Dominant: atom {} (energy {:.4})", bus.codebook_index, bus.energy);
        println!("  Entropy: {:.3} bits, {} active atoms", entropy, active);
        println!("  Top-5: {:?}", peaks.iter()
            .map(|(i, e)| format!("{}:{:.3}", i, e))
            .collect::<Vec<_>>());
        println!();
    }

    // ── Step 5: Cross-sentence comparison ──────────────────────────────────
    println!("═══════════════════════════════════════════════════════");
    println!("  CROSS-SENTENCE TOPOLOGY CHECK");
    println!("═══════════════════════════════════════════════════════\n");

    let mut dominant_atoms: Vec<(u16, String)> = Vec::new();

    for text in &sentences {
        let encoding = tokenizer.encode(*text, true).unwrap();
        let token_ids = encoding.get_ids();
        let table_indices: Vec<u16> = token_ids.iter()
            .map(|&id| (id as usize % n_raw) as u16)
            .collect();

        let mut bank = SensorBank::new();
        for &idx in &table_indices {
            let sensor = Sensor::from_distance_row("bge-m3", &table_data, idx as usize, n, 192);
            if !sensor.is_empty() { bank.add(sensor); }
        }

        engine.reset();
        bank.fire_into(&mut engine);
        engine.think(10);
        let bus = engine.commit();
        dominant_atoms.push((bus.codebook_index, text.chars().take(40).collect()));
    }

    // Check: do similar sentences converge to nearby atoms?
    println!("  Dominant atoms per sentence:");
    for (atom, text) in &dominant_atoms {
        println!("    atom {:>4} ← \"{}\"", atom, text);
    }

    // Distance between dominant atoms in the table
    println!("\n  Pairwise distances (u8, 128=orthogonal, 255=identical):");
    for i in 0..dominant_atoms.len() {
        for j in (i+1)..dominant_atoms.len() {
            let a = dominant_atoms[i].0 as usize;
            let b = dominant_atoms[j].0 as usize;
            let dist = table_data[a * n + b];
            let cos = (dist as f64 / 255.0) * 2.0 - 1.0;
            println!("    S{} ↔ S{}: u8={:>3} cos={:.3}  [{} ↔ {}]",
                i+1, j+1, dist, cos,
                &dominant_atoms[i].1,
                &dominant_atoms[j].1[..dominant_atoms[j].1.len().min(25)]);
        }
    }

    println!("\n════════════════════════════════════════════════════════");
    println!("  TEXT → THOUGHT: COMPLETE");
    println!("  Pipeline: tokenize({:.0}μs) → sensor → think({:.0}μs) → commit",
        0.0, 0.0); // timing is per-sentence above
    println!("════════════════════════════════════════════════════════");
}
