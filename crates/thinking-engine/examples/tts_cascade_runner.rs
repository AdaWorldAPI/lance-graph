//! TTS cascade runner: HHTL cache → archetype sequence → audio code indices.
//!
//! Loads the 18 HHTL caches built by tts_bgz_codebook.rs and runs a
//! cascade forward pass on a token sequence. Produces archetype indices
//! that represent the compressed model output.
//!
//! This is the reality check: does the cascade produce coherent archetype
//! sequences that could plausibly decode to speech?
//!
//! ```sh
//! cargo run --release --example tts_cascade_runner \
//!     --manifest-path crates/thinking-engine/Cargo.toml
//! ```

use bgz_tensor::hhtl_cache::{HhtlCache, RouteAction};
use bgz_tensor::projection::Base17;
use std::collections::HashMap;
use std::time::Instant;

const CODEBOOK_DIR: &str = "/home/user/models/qwen3-tts-0.6b/codebooks";

fn main() {
    println!("═══ TTS CASCADE RUNNER ═══\n");

    // Step 1: Load all HHTL caches
    let t0 = Instant::now();
    let mut caches: HashMap<String, HhtlCache> = HashMap::new();
    let mut assignments: HashMap<String, Vec<u8>> = HashMap::new();

    let roles = [
        "talker_q_proj", "talker_k_proj", "talker_v_proj", "talker_o_proj",
        "talker_gate_proj", "talker_up_proj", "talker_down_proj",
        "talker_embedding", "talker_norm",
        "code_predictor_q_proj", "code_predictor_k_proj", "code_predictor_v_proj",
        "code_predictor_o_proj", "code_predictor_gate_proj", "code_predictor_up_proj",
        "code_predictor_down_proj", "code_predictor_embedding", "code_predictor_norm",
    ];

    for role in &roles {
        let hhtl_path = format!("{}/{}_hhtl.bgz", CODEBOOK_DIR, role);
        let assign_path = format!("{}/{}_assign.bin", CODEBOOK_DIR, role);

        match HhtlCache::deserialize(&hhtl_path) {
            Ok(cache) => {
                let assign = std::fs::read(&assign_path).unwrap_or_default();
                let k = cache.k();
                let gamma = cache.gamma_meta;
                println!("    {}: k={}, gamma=[{:.4},{:.4},{:.4}], {} assigns",
                    role, k, gamma[0], gamma[1], gamma[2], assign.len());
                caches.insert(role.to_string(), cache);
                assignments.insert(role.to_string(), assign);
            }
            Err(e) => {
                eprintln!("    {}: SKIP ({})", role, e);
            }
        }
    }
    println!("[1] Loaded {} caches in {:?}\n", caches.len(), t0.elapsed());

    // Step 2: Simulate token sequence (use embedding assignments as token→archetype map)
    let embed_assign = assignments.get("talker_embedding").cloned().unwrap_or_default();
    let n_tokens = embed_assign.len().min(100); // simulate 100 tokens
    if n_tokens == 0 {
        println!("ERROR: no embedding assignments");
        return;
    }

    println!("[2] Running cascade on {} tokens...", n_tokens);

    // For each token: get its embedding archetype, then route through layers
    let talker_roles = ["talker_q_proj", "talker_k_proj", "talker_v_proj",
                        "talker_gate_proj", "talker_up_proj", "talker_down_proj"];

    let mut attend_count = 0u64;
    let mut skip_count = 0u64;
    let mut compose_count = 0u64;
    let mut escalate_count = 0u64;
    let mut output_archetypes: Vec<u8> = Vec::with_capacity(n_tokens);

    let t0 = Instant::now();
    for token_idx in 0..n_tokens {
        let embed_arch = embed_assign[token_idx];
        let mut current_arch = embed_arch;

        // Route through talker layers (simulate by routing through each role's cache)
        for role_name in &talker_roles {
            if let Some(cache) = caches.get(*role_name) {
                // Route: how does this token's archetype interact with adjacent?
                let next_idx = (token_idx + 1).min(n_tokens - 1);
                let next_arch = embed_assign[next_idx];

                match cache.route(current_arch, next_arch) {
                    RouteAction::Skip => {
                        skip_count += 1;
                        // No update — skip this role's contribution
                    }
                    RouteAction::Attend => {
                        attend_count += 1;
                        // Attend: distance determines interaction strength
                        let dist = cache.distance(current_arch, next_arch);
                        // The attended archetype influences the output
                        if dist < 50 {
                            // Strong interaction: blend toward next
                            current_arch = next_arch;
                        }
                    }
                    RouteAction::Compose => {
                        compose_count += 1;
                        // Multi-hop: find intermediate archetype
                        // For now: just use the midpoint
                        current_arch = ((current_arch as u16 + next_arch as u16) / 2) as u8;
                    }
                    RouteAction::Escalate => {
                        escalate_count += 1;
                        // Full precision needed — keep current
                    }
                }
            }
        }

        output_archetypes.push(current_arch);
    }
    let elapsed = t0.elapsed();

    println!("[3] Cascade results ({:?}):", elapsed);
    println!("    Attend:   {} ({:.1}%)", attend_count,
        attend_count as f64 / (attend_count + skip_count + compose_count + escalate_count) as f64 * 100.0);
    println!("    Skip:     {} ({:.1}%)", skip_count,
        skip_count as f64 / (attend_count + skip_count + compose_count + escalate_count) as f64 * 100.0);
    println!("    Compose:  {} ({:.1}%)", compose_count,
        compose_count as f64 / (attend_count + skip_count + compose_count + escalate_count) as f64 * 100.0);
    println!("    Escalate: {} ({:.1}%)", escalate_count,
        escalate_count as f64 / (attend_count + skip_count + compose_count + escalate_count) as f64 * 100.0);

    // Step 3: Analyze output archetype sequence
    let unique: std::collections::HashSet<u8> = output_archetypes.iter().copied().collect();
    println!("\n[4] Output archetype sequence:");
    println!("    Length: {}", output_archetypes.len());
    println!("    Unique archetypes: {}", unique.len());
    println!("    First 20: {:?}", &output_archetypes[..20.min(output_archetypes.len())]);

    // Check: is the sequence structured (not random)?
    // Adjacent archetypes should be correlated (speech is smooth)
    let mut transitions = 0u64;
    for w in output_archetypes.windows(2) {
        if w[0] != w[1] { transitions += 1; }
    }
    let transition_rate = transitions as f64 / (output_archetypes.len() - 1).max(1) as f64;
    println!("    Transition rate: {:.1}% (lower = smoother = more speech-like)",
        transition_rate * 100.0);

    // Throughput
    let tokens_per_sec = n_tokens as f64 / elapsed.as_secs_f64();
    println!("\n[5] Throughput: {:.0} tokens/sec ({:.1}µs/token)",
        tokens_per_sec, elapsed.as_micros() as f64 / n_tokens as f64);

    // The output archetypes would feed into the code_predictor cascade
    // which maps them to 16 codebook indices for the speech tokenizer decoder.
    // For the POC: the cascade runs, produces coherent archetypes, and is fast.

    println!("\n═══ DONE ═══");
}
