//! Jina + BGE-M3 semantic truth anchor: real embeddings → distance table → cascade
//!
//! Uses Jina v3 API for LIVE semantic embeddings (1024D).
//! Builds a small semantic distance table from embedded phrases.
//! Runs the domino cascade on REAL semantic topology.
//!
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example jina_semantic_cascade

use std::collections::HashMap;
use thinking_engine::engine::ThinkingEngine;
use thinking_engine::domino::DominoCascade;

const JINA_KEY: &str = "jina_b7b1d172a2c74ad2a95e2069d07d8bb9TayVx4WjQF0VWWDmx4xl32VbrHAc";

fn main() {
    println!("═══ JINA SEMANTIC TRUTH ANCHOR ═══\n");

    // ── Step 1: Define semantic atoms ──
    // These are the CONCEPTS we want in the distance table.
    // Each becomes one row in the table, with real Jina embeddings.
    let atoms: Vec<&str> = vec![
        // Animals / nature
        "cat",                          // 0
        "dog",                          // 1
        "ocean",                        // 2
        "fire",                         // 3
        "light",                        // 4
        "darkness",                     // 5
        "wound",                        // 6
        "healing",                      // 7
        // Emotions
        "joy",                          // 8
        "grief",                        // 9
        "love",                         // 10
        "fear",                         // 11
        "anger",                        // 12
        "peace",                        // 13
        "wonder",                       // 14
        "longing",                      // 15
        // Abstract / intellectual
        "quantum physics",              // 16
        "stock market",                 // 17
        "mathematics",                  // 18
        "silence",                      // 19
        "God",                          // 20
        "language",                     // 21
        "translation",                  // 22
        "truth",                        // 23
        // Body / material
        "drop of water",               // 24
        "the entire ocean",            // 25
        "a flame burning",             // 26
        "an open wound",               // 27
        "morning light",               // 28
        "empty room",                  // 29
        "warm embrace",                // 30
        "cold wind",                   // 31
        // Rumi-specific
        "the wound is where light enters",     // 32
        "you are the ocean in a drop",         // 33
        "silence is the language of God",      // 34
        "set your life on fire",               // 35
        // Test sentences (these will be the queries)
        "The cat sat on the mat",              // 36
        "The stock market crashed today",      // 37
        "I feel deeply sad about losing someone", // 38
        "Pure overwhelming joy flooded through me", // 39
    ];
    let n = atoms.len();
    println!("[1] {} semantic atoms defined", n);

    // ── Step 2: Embed ALL atoms via Jina v3 ──
    println!("[2] Embedding {} atoms via Jina v3 API...", n);
    let embeddings = jina_embed_batch(&atoms);
    println!("  Got {} embeddings × {}D", embeddings.len(), embeddings[0].len());

    // ── Step 3: Build N×N cosine distance table ──
    println!("[3] Building {}×{} semantic distance table...", n, n);
    let mut table = vec![128u8; n * n];
    let mut min_cos = 1.0f32;
    let mut max_cos = -1.0f32;

    // Pre-normalize
    let normed: Vec<Vec<f32>> = embeddings.iter().map(|e| {
        let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm < 1e-10 { e.clone() } else { e.iter().map(|v| v / norm).collect() }
    }).collect();

    for i in 0..n {
        table[i * n + i] = 255;
        for j in (i+1)..n {
            let dot: f32 = normed[i].iter().zip(&normed[j]).map(|(a, b)| a * b).sum();
            let cos = dot.clamp(-1.0, 1.0);
            if cos < min_cos { min_cos = cos; }
            if cos > max_cos { max_cos = cos; }
            let u = (((cos + 1.0) / 2.0) * 255.0).round() as u8;
            table[i * n + j] = u;
            table[j * n + i] = u;
        }
    }
    println!("  cos[{:.3}, {:.3}]", min_cos, max_cos);

    // Table stats
    let avg = table.iter().map(|&v| v as f64).sum::<f64>() / table.len() as f64;
    let std = (table.iter().map(|&v| { let d = v as f64 - avg; d * d }).sum::<f64>() / table.len() as f64).sqrt();
    println!("  avg={:.1} std={:.1} — compare: attn_q std=8.4, semantic embed std=6.8", avg, std);

    // Print interesting pairs
    println!("\n  Key semantic distances:");
    let pairs = [
        (0, 1, "cat ↔ dog"),
        (0, 16, "cat ↔ quantum"),
        (6, 7, "wound ↔ healing"),
        (6, 4, "wound ↔ light"),
        (8, 9, "joy ↔ grief"),
        (10, 11, "love ↔ fear"),
        (2, 24, "ocean ↔ drop of water"),
        (2, 25, "ocean ↔ entire ocean"),
        (19, 20, "silence ↔ God"),
        (17, 16, "stock market ↔ quantum"),
        (32, 6, "rumi wound/light ↔ wound"),
        (32, 4, "rumi wound/light ↔ light"),
        (33, 2, "rumi ocean/drop ↔ ocean"),
        (34, 19, "rumi silence/god ↔ silence"),
        (35, 3, "rumi fire ↔ fire"),
        (36, 0, "cat-sentence ↔ cat"),
        (37, 17, "crash-sentence ↔ stock market"),
        (38, 9, "sad-sentence ↔ grief"),
        (39, 8, "joy-sentence ↔ joy"),
    ];
    for (i, j, label) in pairs {
        if i < n && j < n {
            let cos = (table[i * n + j] as f32 / 255.0) * 2.0 - 1.0;
            println!("    {} = cos {:.3} (u8 {})", label, cos, table[i * n + j]);
        }
    }

    // ── Step 4: Domino cascade on Jina semantic table ──
    println!("\n[4] Domino cascade on REAL semantic topology:\n");

    let engine = ThinkingEngine::new(table.clone());
    println!("  Engine: {}×{} floor={}", n, n, engine.floor);

    let counts = vec![1u32; n]; // uniform, no IDF bias
    let cascade = DominoCascade::new(&engine, &counts);

    // Test the 4 query sentences (atoms 36-39)
    let queries = [
        (36, "The cat sat on the mat"),
        (37, "The stock market crashed today"),
        (38, "I feel deeply sad about losing someone"),
        (39, "Pure overwhelming joy flooded through me"),
        (32, "Rumi: wound is where light enters"),
        (33, "Rumi: you are the ocean in a drop"),
        (34, "Rumi: silence is the language of God"),
        (35, "Rumi: set your life on fire"),
    ];

    let mut results: Vec<(u16, String)> = Vec::new();

    for (atom_idx, label) in &queries {
        let (dominant, stages, dissonance) = cascade.think(&[*atom_idx as u16]);
        let chain: Vec<&str> = stages.iter()
            .filter_map(|s| s.focus.first())
            .map(|a| if (a.index as usize) < n { atoms[a.index as usize] } else { "?" })
            .collect();
        let dominant_name = if (dominant as usize) < n { atoms[dominant as usize] } else { "?" };

        println!("  \"{}\"", label);
        println!("    → {} (atom {})  chain: {:?}", dominant_name, dominant, chain);
        println!("    dissonance: {:.3}  resolved: {}",
            dissonance.total_dissonance, dissonance.resolved);
        for stage in &stages {
            let m = &stage.markers;
            let mut flags = Vec::new();
            if m.staunen > 0.01 { flags.push(format!("✨{:.2}", m.staunen)); }
            if m.wisdom > 0.01 { flags.push(format!("🦉{:.2}", m.wisdom)); }
            if m.epiphany > 0.01 { flags.push(format!("💡{:.2}", m.epiphany)); }
            if !flags.is_empty() {
                println!("    stage {}: {}", stage.stage, flags.join(" "));
            }
        }
        results.push((dominant, dominant_name.to_string()));
        println!();
    }

    // Summary
    println!("═══ SUMMARY ═══");
    let unique: std::collections::HashSet<u16> = results.iter().map(|r| r.0).collect();
    println!("  {} queries → {} unique peaks", results.len(), unique.len());
    for (dom, name) in &results {
        println!("    atom {:>2} = {}", dom, name);
    }

    println!("\n═══ JINA SEMANTIC CASCADE COMPLETE ═══");
}

fn jina_embed_batch(texts: &[&str]) -> Vec<Vec<f32>> {
    let mut all_embeddings = Vec::new();

    // One at a time for reliability
    for (i, text) in texts.iter().enumerate() {
        let body = format!(
            "{{\"model\":\"jina-embeddings-v3\",\"input\":[\"{}\"],\"dimensions\":1024}}",
            text.replace('"', "\\\"")
        );

        let output = std::process::Command::new("curl")
            .args(&[
                "-s", "https://api.jina.ai/v1/embeddings",
                "-H", &format!("Authorization: Bearer {}", JINA_KEY),
                "-H", "Content-Type: application/json",
                "-d", &body,
            ])
            .output()
            .expect("curl failed");

        let response = String::from_utf8_lossy(&output.stdout);
        let resp = response.as_ref();

        // Find first "embedding": [...]
        if let Some(start) = resp.find("\"embedding\"") {
            if let Some(arr_start) = resp[start..].find('[') {
                let abs = start + arr_start;
                if let Some(arr_end) = resp[abs..].find(']') {
                    let arr = &resp[abs + 1..abs + arr_end];
                    let values: Vec<f32> = arr.split(',')
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect();
                    if values.len() == 1024 {
                        all_embeddings.push(values);
                        if (i + 1) % 10 == 0 {
                            eprint!("  {}/{}\r", i + 1, texts.len());
                        }
                        continue;
                    }
                }
            }
        }
        eprintln!("  WARN: failed to embed [{}] \"{}\"", i, text);
        all_embeddings.push(vec![0.0; 1024]); // zero fallback
    }

    eprintln!("  {}/{} embedded", all_embeddings.len(), texts.len());
    all_embeddings
}
