//! Qwen3-VL-Embedding-2B Forward Pass: 2048D multimodal embeddings.
//!
//! cargo run --release --features calibration --example qwen3_vl_forward

#[cfg(feature = "calibration")]
fn main() {
    use candle_core::{Device, DType, IndexOp, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::qwen3;

    println!("═══════════════════════════════════════════════════════════");
    println!("  QWEN3-VL-EMBEDDING-2B → 2048D embeddings");
    println!("═══════════════════════════════════════════════════════════\n");

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Config — need to extract text_config from the VL config
    let config_str = std::fs::read_to_string(
        "crates/thinking-engine/data/qwen3-vl-embedding/config.json"
    ).expect("config.json");
    let full_config: serde_json::Value = serde_json::from_str(&config_str).expect("parse");

    // Build Qwen3 config from VL config fields
    let config = qwen3::Config {
        vocab_size: full_config["vocab_size"].as_u64().unwrap_or(151936) as usize,
        hidden_size: full_config["hidden_size"].as_u64().unwrap_or(2048) as usize,
        intermediate_size: full_config["intermediate_size"].as_u64().unwrap_or(6144) as usize,
        num_hidden_layers: full_config["num_hidden_layers"].as_u64().unwrap_or(28) as usize,
        num_attention_heads: full_config["num_attention_heads"].as_u64().unwrap_or(16) as usize,
        head_dim: full_config["head_dim"].as_u64().unwrap_or(128) as usize,
        attention_bias: full_config["attention_bias"].as_bool().unwrap_or(false),
        num_key_value_heads: full_config["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
        max_position_embeddings: 32768,
        sliding_window: None,
        max_window_layers: 0,
        tie_word_embeddings: false,
        rope_theta: full_config["rope_theta"].as_f64().unwrap_or(5000000.0),
        rms_norm_eps: full_config["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    };

    println!("[1] Config: {} layers, {} hidden, {} vocab, head_dim={}",
        config.num_hidden_layers, config.hidden_size, config.vocab_size, config.head_dim);
    println!("    q_proj expected: {}×{}", config.num_attention_heads * config.head_dim, config.hidden_size);

    // Load safetensors — strip "model.language_model." prefix for candle Qwen3
    let model_path = "crates/thinking-engine/data/qwen3-vl-embedding/model.safetensors";
    println!("[2] Loading safetensors (4 GB)...");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device).expect("load")
    }.rename_f(|name| {
        // candle asks for "model.X" → file has "model.language_model.X"
        if let Some(rest) = name.strip_prefix("model.") {
            format!("model.language_model.{rest}")
        } else {
            name.to_string()
        }
    });

    println!("[3] Building Qwen3 (text-only from VL)...");
    let _model = qwen3::Model::new(&config, vb).expect("build");
    println!("    Ready.");

    // Tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file(
        "crates/thinking-engine/data/qwen3-vl-embedding/tokenizer.json"
    ).expect("tokenizer");
    println!("[4] Tokenizer: Qwen3 BPE (151K vocab)");

    let texts = vec![
        "The wound is the place where the light enters you",
        "Where there is ruin there is hope for a treasure",
        "A federal judge ruled the surveillance program unconstitutional",
        "TCP uses a three-way handshake to establish a connection",
        "Bach composed the Well-Tempered Clavier",
        "CRISPR enables precise editing of genomic sequences",
        "Gradient descent minimizes the loss function",
    ];

    println!("[5] Computing 2048D embeddings for {} texts...\n", texts.len());
    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    for (i, text) in texts.iter().enumerate() {
        let input = format!("Document: {}", text);
        let enc = tokenizer.encode(input.as_str(), true).expect("tokenize");
        let ids = enc.get_ids();
        let n_tokens = ids.len();

        let input_ids = Tensor::new(ids, &device).expect("tensor")
            .unsqueeze(0).expect("batch");

        // Fresh model per text (avoid KV cache)
        let vb_fresh = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], dtype, &device).expect("reload")
        }.rename_f(|name| {
            if let Some(rest) = name.strip_prefix("model.") {
                format!("model.language_model.{rest}")
            } else {
                name.to_string()
            }
        });
        let mut fresh = qwen3::Model::new(&config, vb_fresh).expect("rebuild");

        let hidden = fresh.forward(&input_ids, 0).expect("forward");

        // Last-token pooling
        let last = hidden.i((0, n_tokens - 1)).expect("last");
        let norm = last.sqr().expect("s").sum_all().expect("s").sqrt().expect("s");
        let emb: Vec<f32> = last.broadcast_div(&norm).expect("n").to_vec1().expect("v");

        let label = if text.len() > 50 { &text[..50] } else { text };
        println!("  [{}/{}] {} tokens → {}D  |emb|={:.4}  \"{}\"",
            i+1, texts.len(), n_tokens, emb.len(),
            emb.iter().map(|x| x*x).sum::<f32>().sqrt(), label);

        embeddings.push(emb);
    }

    // Pairwise cosine
    println!("\n[6] Pairwise cosine (2048D):\n");
    let pairs = vec![
        (0, 1, "Rumi↔Rumi"),
        (0, 3, "Rumi↔TCP"),
        (2, 3, "Surveillance↔TCP"),
        (4, 5, "Bach↔CRISPR"),
        (5, 6, "CRISPR↔Gradient"),
        (0, 6, "Rumi↔Gradient"),
    ];

    for &(a, b, label) in &pairs {
        let cos: f32 = embeddings[a].iter().zip(&embeddings[b]).map(|(x,y)| x*y).sum();
        println!("  {:>20}  {:.4}", label, cos);
    }

    let rr: f32 = embeddings[0].iter().zip(&embeddings[1]).map(|(x,y)| x*y).sum();
    let rt: f32 = embeddings[0].iter().zip(&embeddings[3]).map(|(x,y)| x*y).sum();

    println!("\n═══════════════════════════════════════════════════════════");
    if rr > rt + 0.02 {
        println!("  Qwen3-VL DISCRIMINATES! Rumi↔Rumi ({:.4}) > Rumi↔TCP ({:.4})", rr, rt);
        let jina_gap = 0.512 - 0.384;
        let vl_gap = rr - rt;
        if vl_gap > jina_gap {
            println!("  BETTER than Jina v5 (gap {:.4} vs {:.4})!", vl_gap, jina_gap);
        } else {
            println!("  Gap {:.4} vs Jina v5 gap {:.4}", vl_gap, jina_gap);
        }
    } else {
        println!("  No discrimination. Rumi↔Rumi ({:.4}) ≈ Rumi↔TCP ({:.4})", rr, rt);
    }
    println!("═══════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "calibration"))]
fn main() { eprintln!("Requires --features calibration"); }
