//! ModernBERT Forward Pass: safetensors → candle → 1024D embeddings.
//!
//! Second ground truth anchor. GeGLU architecture (same gate pattern as Qwopus).
//! NO KV cache needed (encoder model, bidirectional attention).
//!
//! cargo run --release --features calibration --example modernbert_forward

#[cfg(feature = "calibration")]
fn main() {
    use candle_core::{Device, DType, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::modernbert;

    println!("═══════════════════════════════════════════════════════════");
    println!("  MODERNBERT FORWARD PASS → 1024D embeddings");
    println!("═══════════════════════════════════════════════════════════\n");

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Load config
    let config_str = std::fs::read_to_string(
        "crates/thinking-engine/data/modernbert-onnx/config.json"
    ).expect("config.json");
    let config: modernbert::Config = serde_json::from_str(&config_str).expect("parse config");
    println!("[1] Config: {} layers, {} hidden, {} vocab, GeGLU({})",
        config.num_hidden_layers, config.hidden_size, config.vocab_size, config.intermediate_size);

    // Load safetensors
    let model_path = "crates/thinking-engine/data/modernbert-onnx/model.safetensors";
    println!("[2] Loading safetensors...");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device).expect("load")
    };

    // Build model
    println!("[3] Building ModernBERT...");
    let model = modernbert::ModernBert::load(vb, &config).expect("build");
    println!("    Ready (no KV cache, bidirectional encoder).");

    // Load tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(
        "crates/thinking-engine/data/modernbert-onnx/tokenizer.json"
    ).expect("tokenizer");
    println!("[4] Tokenizer: OLMo BPE (50K vocab)");

    // Test texts
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
        let enc = tokenizer.encode(*text, true).expect("tokenize");
        let ids = enc.get_ids();
        let n_tokens = ids.len();

        let input_ids = Tensor::new(ids, &device).expect("tensor")
            .unsqueeze(0).expect("batch");

        // Attention mask: all 1s (no padding)
        let mask = Tensor::ones_like(&input_ids).expect("mask")
            .to_dtype(DType::F32).expect("mask dtype");

        // Forward (no KV cache, stateless encoder)
        let hidden = model.forward(&input_ids, &mask).expect("forward");

        // CLS pooling (first token) — ModernBERT convention
        let cls = hidden.i((0, 0)).expect("cls token");

        // L2 normalize
        let norm = cls.sqr().expect("sqr").sum_all().expect("sum").sqrt().expect("sqrt");
        let embedding = cls.broadcast_div(&norm).expect("normalize");
        let emb_vec: Vec<f32> = embedding.to_vec1().expect("to_vec");

        let label = if text.len() > 50 { &text[..50] } else { text };
        println!("  [{}/{}] {} tokens → 1024D  |emb|={:.4}  \"{}\"",
            i+1, texts.len(), n_tokens,
            emb_vec.iter().map(|x| x*x).sum::<f32>().sqrt(), label);

        embeddings.push(emb_vec);
    }

    // Pairwise cosine
    println!("\n[6] Pairwise cosine:\n");
    let pairs = vec![
        (0, 1, "Rumi↔Rumi"),
        (0, 3, "Rumi↔TCP"),
        (2, 3, "Surveillance↔TCP"),
        (4, 5, "Bach↔CRISPR"),
        (5, 6, "CRISPR↔Gradient"),
    ];

    for &(a, b, label) in &pairs {
        let cos: f32 = embeddings[a].iter().zip(&embeddings[b])
            .map(|(x, y)| x * y).sum();
        println!("  {:>20}  {:.4}", label, cos);
    }

    let rumi_rumi: f32 = embeddings[0].iter().zip(&embeddings[1]).map(|(x,y)| x*y).sum();
    let rumi_tcp: f32 = embeddings[0].iter().zip(&embeddings[3]).map(|(x,y)| x*y).sum();

    println!("\n═══════════════════════════════════════════════════════════");
    if rumi_rumi > rumi_tcp + 0.05 {
        println!("  ModernBERT DISCRIMINATES! Rumi↔Rumi ({:.4}) > Rumi↔TCP ({:.4})", rumi_rumi, rumi_tcp);
    } else {
        println!("  No discrimination. Rumi↔Rumi ({:.4}) ≈ Rumi↔TCP ({:.4})", rumi_rumi, rumi_tcp);
    }
    println!("═══════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "calibration"))]
fn main() {
    eprintln!("Requires --features calibration");
}
