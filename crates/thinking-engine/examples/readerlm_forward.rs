//! ReaderLM-v2 forward pass via candle — RESEARCH-ONLY (Qwen2 legacy lineage)
//!
//! ## RESEARCH-ONLY — do not use for production
//!
//! This example targets **Reader-LM v2 (Qwen2 architecture)** — the
//! PRE-v5-era lineage. The current production Reader-LM target is
//! **Reader-LM v3 = Jina v5** (BERT 3.x architecture, see
//! `ndarray::hpc::jina::runtime::ModelSource::JinaV5`). Kept for v5-vs-v2
//! behavioral diffing when a Jina v5 result needs isolation against the
//! older Qwen2 pipeline. Do NOT reach for this example when building new
//! production wiring.
//!
//! See `lance-graph/CLAUDE.md` → `Model Registry` → `Research-only /
//! diagnostic fallback` for the canonical policy.
//!
//! ## Original design notes (valid for the Qwen2 v2 target)
//!
//! Loads ReaderLM-v2 BF16 safetensors (Qwen2 architecture),
//! runs autoregressive generation on HTML input,
//! produces clean markdown output.
//!
//! cargo run --release --features calibration \
//!   --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example readerlm_forward

#[cfg(feature = "calibration")]
fn main() {
    use candle_core::{Device, DType, Tensor, IndexOp};
    use candle_nn::VarBuilder;
    use candle_transformers::models::qwen2;
    use candle_transformers::generation::LogitsProcessor;

    let device = Device::Cpu;
    let dtype = DType::F32;

    println!("═══════════════════════════════════════════════════════════");
    println!("  ReaderLM-v2: HTML → Markdown (candle, pure Rust)");
    println!("═══════════════════════════════════════════════════════════\n");

    // ═══ Step 1: Load config ═══
    let config_str = std::fs::read_to_string("data/readerlm-v2/config_candle.json")
        .expect("config_candle.json");
    let config: qwen2::Config = serde_json::from_str(&config_str)
        .expect("parse config");
    println!("[1] Config: {} layers, {}D hidden, {} vocab, {} heads, {} KV heads",
        config.num_hidden_layers, config.hidden_size, config.vocab_size,
        config.num_attention_heads, config.num_key_value_heads);

    // ═══ Step 2: Load tokenizer ═══
    println!("[2] Loading tokenizer...");
    let tokenizer = tokenizers::Tokenizer::from_file("data/readerlm-v2/tokenizer.json")
        .expect("tokenizer");

    // ═══ Step 3: Load model ═══
    println!("[3] Loading model (2.9 GB BF16 → F32)...");
    let t0 = std::time::Instant::now();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &["data/readerlm-v2/model.safetensors"], dtype, &device
        ).expect("load safetensors")
    };
    let mut model = qwen2::ModelForCausalLM::new(&config, vb).expect("build model");
    println!("    Loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // ═══ Step 4: Test HTML → Markdown ═══
    let test_html = r#"<html><body>
<h1>CRISPR Gene Editing</h1>
<p>CRISPR-Cas9 is a revolutionary gene editing tool that allows scientists
to modify DNA sequences with unprecedented precision. First developed in 2012,
it has since been used in clinical trials for treating sickle cell disease,
beta-thalassemia, and certain cancers.</p>
<h2>How It Works</h2>
<p>The system uses a guide RNA to direct the Cas9 enzyme to a specific location
in the genome, where it makes a precise cut. The cell's natural repair mechanisms
then fix the break, either disabling a gene or inserting new genetic material.</p>
<p>In 2023, the FDA approved Casgevy, the first CRISPR-based therapy.</p>
</body></html>"#;

    let prompt = format!(
        "<|im_start|>system\nYou are a document reader. Convert HTML to clean markdown.<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
        test_html
    );

    println!("[4] Tokenizing prompt ({} chars)...", prompt.len());
    let encoding = tokenizer.encode(prompt.as_str(), true).expect("encode");
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("    {} tokens", input_ids.len());

    // ═══ Step 5: Generate ═══
    println!("[5] Generating (max 256 tokens)...");
    let t0 = std::time::Instant::now();

    let mut all_tokens = input_ids.clone();
    let mut logits_processor = LogitsProcessor::new(42, Some(0.1), None);

    let eos_token = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);

    // First forward: process entire prompt at once
    let prompt_tensor = Tensor::new(&input_ids[..], &device)
        .expect("tensor")
        .unsqueeze(0)
        .expect("batch");
    let logits = model.forward(&prompt_tensor, 0).expect("forward prompt");
    // Shape [1, vocab] → [vocab]
    let logits = logits.flatten_all().expect("flatten prompt logits");
    let mut next_token = logits_processor.sample(&logits).expect("sample");
    all_tokens.push(next_token);

    for i in 0..255 {
        if next_token == eos_token {
            println!("    EOS at token {}", i);
            break;
        }

        let input = Tensor::new(&[next_token], &device)
            .expect("tensor")
            .unsqueeze(0)
            .expect("batch");

        let logits = model.forward(&input, all_tokens.len() - 1)
            .expect("forward");
        let logits = logits.flatten_all().expect("flatten");
        next_token = logits_processor.sample(&logits).expect("sample");

        if next_token == eos_token {
            println!("    EOS at token {}", i);
            break;
        }

        all_tokens.push(next_token);

        if i < 5 || i % 50 == 0 {
            let decoded = tokenizer.decode(&[next_token], false).unwrap_or_default();
            print!("{}", decoded);
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let generated = all_tokens.len() - input_ids.len();
    println!("\n    Generated {} tokens in {:.1}s ({:.1} tok/s)",
        generated, elapsed, generated as f64 / elapsed);

    // ═══ Step 6: Decode output ═══
    let output_tokens = &all_tokens[input_ids.len()..];
    let markdown = tokenizer.decode(output_tokens, true).unwrap_or_default();

    println!("\n[6] Output markdown ({} chars):", markdown.len());
    println!("─────────────────────────────────────────");
    println!("{}", &markdown[..markdown.len().min(500)]);
    println!("─────────────────────────────────────────");

    // ═══ Step 7: Extract token IDs for thinking engine ═══
    let output_token_ids: Vec<u32> = output_tokens.to_vec();
    println!("\n[7] Thinking engine input: {} tokens from ReaderLM output", output_token_ids.len());
    println!("    These tokens go into codebook_index.u16 → centroid IDs → softmax thinking");
    println!("    Vocab 151936 = same as Jina v5 codebook (shared Qwen tokenizer family)");

    println!("\n═══════════════════════════════════════════════════════════");
}

#[cfg(not(feature = "calibration"))]
fn main() { eprintln!("Requires --features calibration"); }
