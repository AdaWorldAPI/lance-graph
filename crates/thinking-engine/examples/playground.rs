//! Playground: type text, see what the engine thinks.
//!
//! Like LM Studio but for the Thinking Engine.
//! Tokenize → codebook lookup → think with temperature → show peaks.
//!
//! Usage: cargo run --release --example playground
//!
//! Knobs:
//!   Temperature: 0.1 (focused) → 1.5 (creative)
//!   Cycles: 5 (fast) → 20 (deep)
//!   Top-K: how many peaks to show

use thinking_engine::jina_lens::{JINA_HDR_TABLE, jina_lookup_many, JINA_N_CENTROIDS};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  THINKING ENGINE PLAYGROUND");
    println!("  Type text → see what the engine thinks");
    println!("═══════════════════════════════════════════════════════════\n");

    // Load tokenizer (try Jina v5 Qwen3 first, fall back to Jina v3 XLM-RoBERTa)
    let tokenizer = load_tokenizer();
    let (codebook, table, n_centroids, tok_name) = load_best_available();

    println!("Tokenizer: {}", tok_name);
    println!("Table: {}×{}", n_centroids, n_centroids);
    println!();

    // Test texts — diverse, real, meaningful
    let texts = vec![
        "The wound is the place where the light enters you",
        "Amyloid plaques accumulate in the brains of Alzheimer patients",
        "TCP uses a three-way handshake to establish a reliable connection",
        "Bach composed the Well-Tempered Clavier as an exploration of all major and minor keys",
        "Edward Snowden revealed the NSA collected phone metadata of millions of Americans",
        "The butterfly counts not months but moments and has time enough",
        "Gradient descent minimizes the loss function by following the negative gradient",
    ];

    // Test different temperatures
    let temperatures = [0.1, 0.3, 0.7, 1.0, 1.5];
    let cycles = 10;
    let top_k = 5;

    println!("{:>50}  T=0.1   T=0.3   T=0.7   T=1.0   T=1.5", "Text");
    println!("{:─>50}  {:─>7}  {:─>7}  {:─>7}  {:─>7}  {:─>7}", "", "", "", "", "", "");

    for text in &texts {
        // Tokenize
        let enc = tokenizer.encode(*text, true).unwrap();
        let token_ids: Vec<u32> = enc.get_ids().to_vec();
        let n_tokens = token_ids.len();

        // Map to centroids
        let centroids: Vec<u16> = if codebook.is_empty() {
            // Jina v3 baked lens
            jina_lookup_many(&token_ids)
        } else {
            token_ids.iter()
                .map(|&id| codebook.get(id as usize).copied().unwrap_or(0))
                .collect()
        };

        // Unique centroids activated
        let mut unique = centroids.clone();
        unique.sort();
        unique.dedup();
        let n_unique = unique.len();

        let label = if text.len() > 45 { &text[..45] } else { text };
        print!("  {:>48}  ", label);

        // Think at each temperature
        for &temp in &temperatures {
            let mut engine = thinking_engine::engine::ThinkingEngine::new(table.clone());
            engine.perturb(&centroids);

            if (temp - 1.0f32).abs() < 0.01 {
                engine.think(cycles);
            } else {
                engine.think_with_temperature(cycles, temp);
            }

            // Get top peak
            let bus = engine.commit();
            let top_atom = bus.codebook_index;
            let top_energy = bus.energy;
            let active = engine.active_count(0.001);

            print!("{:3}|{:3}  ", top_atom, active);
        }
        println!("  ({} tok → {} cent)", n_tokens, n_unique);
    }

    // Now: pairwise comparison at T=0.7
    println!("\n\n  PAIRWISE SIMILARITY (T=0.7, {} cycles)", cycles);
    println!("  {:>48}  {:>48}  cos(energy)", "", "");
    println!("  {:─>48}  {:─>48}  {:─>10}", "", "", "");

    let similarity_pairs = vec![
        (0, 1, "Rumi ↔ Alzheimer"),
        (0, 5, "Rumi ↔ Tagore"),
        (0, 2, "Rumi ↔ TCP"),
        (1, 6, "Alzheimer ↔ Gradient"),
        (4, 2, "Snowden ↔ TCP"),
        (3, 5, "Bach ↔ Tagore"),
        (6, 2, "Gradient ↔ TCP"),
    ];

    for &(a, b, label) in &similarity_pairs {
        let text_a = texts[a];
        let text_b = texts[b];

        let enc_a = tokenizer.encode(text_a, true).unwrap();
        let enc_b = tokenizer.encode(text_b, true).unwrap();

        let cents_a: Vec<u16> = if codebook.is_empty() {
            jina_lookup_many(enc_a.get_ids())
        } else {
            enc_a.get_ids().iter().map(|&id| codebook.get(id as usize).copied().unwrap_or(0)).collect()
        };
        let cents_b: Vec<u16> = if codebook.is_empty() {
            jina_lookup_many(enc_b.get_ids())
        } else {
            enc_b.get_ids().iter().map(|&id| codebook.get(id as usize).copied().unwrap_or(0)).collect()
        };

        let mut eng_a = thinking_engine::engine::ThinkingEngine::new(table.clone());
        eng_a.perturb(&cents_a);
        eng_a.think_with_temperature(cycles, 0.7);
        let energy_a = eng_a.energy.clone();

        let mut eng_b = thinking_engine::engine::ThinkingEngine::new(table.clone());
        eng_b.perturb(&cents_b);
        eng_b.think_with_temperature(cycles, 0.7);
        let energy_b = eng_b.energy.clone();

        let cos = ndarray::hpc::heel_f64x8::cosine_f32_to_f64_simd(&energy_a, &energy_b);

        let label_a = if text_a.len() > 30 { &text_a[..30] } else { text_a };
        let label_b = if text_b.len() > 30 { &text_b[..30] } else { text_b };

        println!("  {:>20}  {:>30}↔{:<30}  {:.4}", label, label_a, label_b, cos);
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  cos ≈ 1.0 for all = ATTRACTOR COLLAPSE (known issue)");
    println!("  Fix: forward pass embeddings OR L4 repetition blocking");
    println!("═══════════════════════════════════════════════════════════");
}

fn load_tokenizer() -> tokenizers::Tokenizer {
    // Try Jina v5 Qwen3 first
    if let Ok(t) = tokenizers::Tokenizer::from_file(
        "crates/thinking-engine/data/jina-v5-onnx/tokenizer.json"
    ) {
        return t;
    }
    // Try from_pretrained
    if let Ok(t) = tokenizers::Tokenizer::from_pretrained("jinaai/jina-embeddings-v3", None) {
        return t;
    }
    panic!("No tokenizer found. Download from Release v0.1.1-tokenizers.");
}

fn load_best_available() -> (Vec<u16>, Vec<u8>, usize, String) {
    // Try Jina v5 codebook first
    let cb_path = "crates/thinking-engine/data/jina-v5-codebook/codebook_index.u16";
    let tb_path = "crates/thinking-engine/data/jina-v5-codebook/distance_table_256x256.u8";
    if let (Ok(cb_data), Ok(tb_data)) = (std::fs::read(cb_path), std::fs::read(tb_path)) {
        let codebook: Vec<u16> = cb_data.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        return (codebook, tb_data, 256, "Jina v5 codebook (256 centroids)".into());
    }

    // Fall back to baked Jina v3 lens
    (vec![], JINA_HDR_TABLE.to_vec(), JINA_N_CENTROIDS, "Jina v3 HDR baked (256 centroids)".into())
}
