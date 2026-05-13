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
    let mut embeddings: Vec<(Vec<f32>, Vec<f32>)> = Vec::new(); // (cls, mean)

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

        // Try BOTH pooling strategies
        // CLS: first token (BERT convention)
        let cls = hidden.i((0, 0)).expect("cls");
        let cls_norm = cls.sqr().expect("s").sum_all().expect("s").sqrt().expect("s");
        let cls_emb: Vec<f32> = cls.broadcast_div(&cls_norm).expect("n").to_vec1().expect("v");

        // MEAN: average of ALL tokens (sentence-transformers convention)
        let mean = hidden.i(0).expect("batch").mean(0).expect("mean");
        let mean_norm = mean.sqr().expect("s").sum_all().expect("s").sqrt().expect("s");
        let mean_emb: Vec<f32> = mean.broadcast_div(&mean_norm).expect("n").to_vec1().expect("v");

        let label = if text.len() > 50 { &text[..50] } else { text };
        println!("  [{}/{}] {} tokens  \"{}\"",
            i+1, texts.len(), n_tokens, label);

        embeddings.push((cls_emb, mean_emb));
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

    println!("  {:>20}  {:>8}  {:>8}", "Pair", "CLS", "MEAN");
    println!("  {:─>20}  {:─>8}  {:─>8}", "", "", "");
    for &(a, b, label) in &pairs {
        let cos_cls: f32 = embeddings[a].0.iter().zip(&embeddings[b].0).map(|(x,y)| x*y).sum();
        let cos_mean: f32 = embeddings[a].1.iter().zip(&embeddings[b].1).map(|(x,y)| x*y).sum();
        println!("  {:>20}  {:>8.4}  {:>8.4}", label, cos_cls, cos_mean);
    }

    let cls_rr: f32 = embeddings[0].0.iter().zip(&embeddings[1].0).map(|(x,y)| x*y).sum();
    let cls_rt: f32 = embeddings[0].0.iter().zip(&embeddings[3].0).map(|(x,y)| x*y).sum();
    let mean_rr: f32 = embeddings[0].1.iter().zip(&embeddings[1].1).map(|(x,y)| x*y).sum();
    let mean_rt: f32 = embeddings[0].1.iter().zip(&embeddings[3].1).map(|(x,y)| x*y).sum();

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  CLS pooling:  Rumi↔Rumi={:.4} vs Rumi↔TCP={:.4} → {}",
        cls_rr, cls_rt, if cls_rr > cls_rt + 0.05 {"DISCRIMINATES"} else {"no"});
    println!("  MEAN pooling: Rumi↔Rumi={:.4} vs Rumi↔TCP={:.4} → {}",
        mean_rr, mean_rt, if mean_rr > mean_rt + 0.05 {"DISCRIMINATES"} else {"no"});
    println!("═══════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "calibration"))]
fn main() {
    eprintln!("Requires --features calibration");
}
