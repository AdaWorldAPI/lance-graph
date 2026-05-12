//! Forward pass: Jina v5 safetensors → candle Qwen3 → 1024D f32 embeddings.
//!
//! THE gate that unlocks everything.
//! Token embeddings alone: ρ=0.54 max (no semantics).
//! After 28 layers: semantic similarity (the WHOLE POINT).
//!
//! cargo run --release --features calibration \
//!   --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example forward_pass

#[cfg(feature = "calibration")]
fn main() {
    use candle_core::{Device, DType, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::qwen3;

    println!("═══════════════════════════════════════════════════════════");
    println!("  FORWARD PASS: Jina v5 → Qwen3 → 1024D embeddings");
    println!("═══════════════════════════════════════════════════════════\n");

    let device = Device::Cpu;
    let dtype = DType::F32; // full precision for ground truth

    // Step 1: Load config
    let config_path = "crates/thinking-engine/data/jina-v5-onnx/config_candle.json";
    let config_str = std::fs::read_to_string(config_path).expect("config.json not found");
    let config: qwen3::Config = serde_json::from_str(&config_str).expect("parse config");
    println!("[1] Config: {} layers, {} hidden, {} vocab",
        config.num_hidden_layers, config.hidden_size, config.vocab_size);

    // Step 2: Load safetensors
    // Jina v5 safetensors has no "model." prefix (embed_tokens.weight, not model.embed_tokens.weight)
    // candle Qwen3 expects "model." prefix → rename via VarBuilder
    let model_path = "crates/thinking-engine/data/jina-v5-onnx/model.safetensors";
    println!("[2] Loading safetensors from {}...", model_path);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)
            .expect("load safetensors")
    };
    // candle Qwen3 looks for "model.embed_tokens.weight" but safetensors has "embed_tokens.weight"
    // Strip the "model." prefix that candle prepends
    let vb = vb.rename_f(|name| {
        name.strip_prefix("model.").unwrap_or(name).to_string()
    });

    // Step 3: Build model
    println!("[3] Building Qwen3 model...");
    let mut model = qwen3::Model::new(&config, vb).expect("build model");
    println!("    Model ready.");

    // Step 4: Load tokenizer
    let tok_path = "crates/thinking-engine/data/jina-v5-onnx/tokenizer.json";
    let tokenizer = tokenizers::Tokenizer::from_file(tok_path).expect("tokenizer");
    println!("[4] Tokenizer: Qwen3 BPE loaded");

    // Step 5: Test texts
    let texts = vec![
        "The wound is the place where the light enters you",
        "Where there is ruin there is hope for a treasure",
        "A federal judge ruled the surveillance program unconstitutional",
        "TCP uses a three-way handshake to establish a connection",
        "Bach composed the Well-Tempered Clavier",
        "CRISPR enables precise editing of genomic sequences",
        "Gradient descent minimizes the loss function",
    ];

    println!("[5] Computing embeddings for {} texts...\n", texts.len());

    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    for (i, text) in texts.iter().enumerate() {
        // Tokenize with "Document:" prefix (Jina v5 convention)
        let input = format!("Document: {}", text);
        let enc = tokenizer.encode(input.as_str(), true).expect("tokenize");
        let ids = enc.get_ids();
        let n_tokens = ids.len();

        let input_ids = Tensor::new(ids, &device).expect("tensor")
            .unsqueeze(0).expect("batch dim");

        // Fresh model per text (avoids KV cache contamination)
        let vb_fresh = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device)
                .expect("reload")
        }.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_string());
        let mut fresh_model = qwen3::Model::new(&config, vb_fresh).expect("rebuild");

        let hidden = fresh_model.forward(&input_ids, 0).expect("forward");
        // hidden shape: [1, seq_len, 1024]

        // Last-token pooling (Jina v5 uses last token = EOS)
        let last_token = hidden.i((0, n_tokens - 1)).expect("last token");
        // shape: [1024]

        // L2 normalize
        let norm = last_token.sqr().expect("sqr").sum_all().expect("sum").sqrt().expect("sqrt");
        let embedding = last_token.broadcast_div(&norm).expect("normalize");

        let emb_vec: Vec<f32> = embedding.to_vec1().expect("to_vec");

        let label = if text.len() > 50 { &text[..50] } else { text };
        println!("  [{}/{}] {} tokens → 1024D  |emb|={:.4}  \"{}\"",
            i+1, texts.len(), n_tokens,
            emb_vec.iter().map(|x| x*x).sum::<f32>().sqrt(),
            label);

        embeddings.push(emb_vec);
    }

    // Step 6: Pairwise cosine = GROUND TRUTH
    println!("\n[6] Pairwise cosine (GROUND TRUTH):\n");
    println!("  {:>30} ↔ {:<30}  cos", "", "");

    let comparison_pairs = vec![
        (0, 1, "Rumi↔Rumi"),
        (0, 3, "Rumi↔TCP"),
        (2, 3, "Surveillance↔TCP"),
        (4, 5, "Bach↔CRISPR"),
        (5, 6, "CRISPR↔Gradient"),
        (0, 6, "Rumi↔Gradient"),
    ];

    for &(a, b, label) in &comparison_pairs {
        let cos: f32 = embeddings[a].iter().zip(&embeddings[b])
            .map(|(x, y)| x * y).sum();
        println!("  {:>20}  {:>30}↔{:<30}  {:.4}",
            label,
            if texts[a].len() > 28 { &texts[a][..28] } else { texts[a] },
            if texts[b].len() > 28 { &texts[b][..28] } else { texts[b] },
            cos);
    }

    // Step 7: Does it discriminate?
    let rumi_rumi: f32 = embeddings[0].iter().zip(&embeddings[1]).map(|(x,y)| x*y).sum();
    let rumi_tcp: f32 = embeddings[0].iter().zip(&embeddings[3]).map(|(x,y)| x*y).sum();

    println!("\n═══════════════════════════════════════════════════════════");
    if rumi_rumi > rumi_tcp + 0.05 {
        println!("  DISCRIMINATES! Rumi↔Rumi ({:.4}) > Rumi↔TCP ({:.4})", rumi_rumi, rumi_tcp);
        println!("  Forward pass embeddings HAVE semantic topology.");
        println!("  → Build CLAM codebook from THESE embeddings.");
        println!("  → The playground will WORK with semantic tables.");
    } else {
        println!("  NO discrimination. Rumi↔Rumi ({:.4}) ≈ Rumi↔TCP ({:.4})", rumi_rumi, rumi_tcp);
        println!("  Check: model loading, tokenization, pooling strategy.");
    }
    println!("═══════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "calibration"))]
fn main() {
    eprintln!("This example requires --features calibration");
    eprintln!("Run: cargo run --release --features calibration --example forward_pass");
}
