//! # Qualia Motif — 17D feeling trajectory through a story
//!
//! Runs the text_to_thought pipeline on a sequence of sentences,
//! computes 17D qualia for each, detects the nearest QPL family,
//! and identifies progression motifs (tension arcs).
//!
//! ```text
//! cargo run --release --manifest-path crates/thinking-engine/Cargo.toml \
//!   --example qualia_motif --features tokenizer
//! ```
//!
//! Requires BGE-M3 tokenizer + codebook + distance table on disk.
//! See `examples/text_to_thought.rs` for how to obtain these files.

use thinking_engine::lookup::TextToThought;
use thinking_engine::qualia::{Qualia17D, DIMS_17D};

fn main() {
    println!("════════════════════════════════════════════════════════════");
    println!("  QUALIA MOTIF  —  17D feeling trajectory through a story");
    println!("════════════════════════════════════════════════════════════\n");

    // ── Step 1: Load TextToThought (BGE-M3) ───────────────────────────────
    println!("[1] Loading TextToThought (BGE-M3 tokenizer + codebook + distance table)...");
    let mut ttt = match TextToThought::load_bge_m3() {
        Ok(t) => {
            println!("    Loaded: table={}x{}, model={}", t.table_size, t.table_size, t.model);
            let (cb_len, cb_unique) = t.codebook_stats();
            println!("    Codebook: {} tokens -> {} unique centroids", cb_len, cb_unique);
            println!("    Vocab: {} BPE tokens\n", t.vocab_size());
            t
        }
        Err(e) => {
            eprintln!("ERROR: Failed to load BGE-M3 pipeline: {}\n", e);
            eprintln!("To fix this, ensure the following files exist:");
            eprintln!("  /tmp/bge-m3-tokenizer.json");
            eprintln!("  /tmp/codebooks/bge-m3-roles-f16/codebook_index.u16");
            eprintln!("  /tmp/codebooks/bge-m3-roles-f16/attn_q/distance_table_1024x1024.u8");
            eprintln!();
            eprintln!("You can generate them by running:");
            eprintln!("  cargo run --release --example build_1to1_roles");
            eprintln!("  cargo run --release --example build_codebook_index");
            eprintln!();
            eprintln!("Or download the tokenizer from huggingface.co/BAAI/bge-m3");
            std::process::exit(1);
        }
    };

    // ── Step 2: Story sentences ───────────────────────────────────────────
    let sentences = [
        "The morning was calm and peaceful.",
        "A strange noise came from the basement.",
        "Heart pounding, she opened the door.",
        "It was just the cat, knocking over a vase.",
        "She laughed with relief and petted the cat.",
    ];

    println!("[2] Processing {} sentences through text -> thought -> qualia...\n", sentences.len());

    // ── Step 3: Process each sentence ─────────────────────────────────────
    let mut qualia_seq: Vec<Qualia17D> = Vec::new();
    let mut families: Vec<&'static str> = Vec::new();
    let mut derivatives: Vec<Option<f32>> = Vec::new();

    for (i, text) in sentences.iter().enumerate() {
        // 3a: think() -> ThoughtResult
        let result = ttt.think(text);

        // 3b: Qualia17D from engine state (engine retains state after think)
        let q = Qualia17D::from_engine(ttt.engine());

        // 3c: nearest family
        let (family, family_dist) = q.nearest_family();

        // 3d: feeling derivative (tension rate) if previous exists
        let deriv = if i > 0 {
            Some(q.feeling_derivative(&qualia_seq[i - 1]))
        } else {
            None
        };

        // Print sentence result
        let arrow = match deriv {
            Some(d) if d > 0.05 => format!("  \u{2191} tension {:+.2}", d),
            Some(d) if d < -0.05 => format!("  \u{2193} tension {:+.2}", d),
            Some(d) => format!("  \u{2194} tension {:+.2}", d),
            None => String::new(),
        };

        let top_dims = top_three_dims(&q);
        println!(
            "  S{}: {:<12} ({}){}",
            i + 1,
            family,
            top_dims,
            arrow,
        );
        println!(
            "       \"{}\"",
            text,
        );
        println!(
            "       {} tokens, {} unique atoms, {:.0}us | family dist={:.3}",
            result.token_count, result.unique_atoms, result.think_micros as f64, family_dist,
        );
        println!();

        qualia_seq.push(q);
        families.push(family);
        derivatives.push(deriv);
    }

    // ── Step 4: Print motif trajectory ────────────────────────────────────
    println!("────────────────────────────────────────────────────────────");
    println!("  MOTIF TRAJECTORY");
    println!("────────────────────────────────────────────────────────────\n");

    let motif_str = families
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join(" -> ");
    println!("  Motif: {}", motif_str);

    // Detect pattern labels
    let pattern = detect_pattern(&derivatives);
    println!("  Pattern: \"{}\"", pattern);

    // Detect named arc segments
    let segments = detect_segments(&derivatives);
    if !segments.is_empty() {
        println!("\n  Arc segments:");
        for seg in &segments {
            println!("    {}", seg);
        }
    }

    // ── Step 5: Per-dimension values ──────────────────────────────────────
    println!("\n────────────────────────────────────────────────────────────");
    println!("  PER-DIMENSION VALUES (17D)");
    println!("────────────────────────────────────────────────────────────\n");

    // Header
    print!("  {:>14}", "dimension");
    for i in 0..sentences.len() {
        print!("   S{:<5}", i + 1);
    }
    println!();
    print!("  {:>14}", "──────────");
    for _ in 0..sentences.len() {
        print!("  {:─>6}", "");
    }
    println!();

    // Each dimension row
    for (d, &dim_name) in DIMS_17D.iter().enumerate() {
        print!("  {:>14}", dim_name);
        for q in &qualia_seq {
            print!("  {:6.3}", q.dims[d]);
        }
        println!();
    }

    // Family row
    print!("\n  {:>14}", "family");
    for f in &families {
        print!("  {:>6}", &f[..f.len().min(6)]);
    }
    println!();

    // Derivative row
    print!("  {:>14}", "d(tension)");
    for d in &derivatives {
        match d {
            Some(v) => print!("  {:+6.3}", v),
            None => print!("  {:>6}", "-"),
        }
    }
    println!();

    println!("\n════════════════════════════════════════════════════════════");
    println!("  QUALIA MOTIF: COMPLETE");
    println!("════════════════════════════════════════════════════════════");
}

/// Return a human-readable string of the top 3 most prominent dimensions.
fn top_three_dims(q: &Qualia17D) -> String {
    let mut indexed: Vec<(usize, f32)> = q.dims.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
        .iter()
        .take(3)
        .map(|(i, _)| DIMS_17D[*i])
        .collect::<Vec<_>>()
        .join(", ")
}

/// Detect the overall narrative pattern from tension derivatives.
fn detect_pattern(derivatives: &[Option<f32>]) -> String {
    let mut labels: Vec<&str> = Vec::new();

    // Track consecutive rising/falling
    let mut rising_count = 0;
    let mut falling_count = 0;
    let mut had_peak = false;

    for d in derivatives {
        match d {
            Some(v) if *v > 0.1 => {
                rising_count += 1;
                if falling_count > 0 {
                    falling_count = 0;
                }
                if rising_count >= 2 && !labels.contains(&"tension building") {
                    labels.push("tension building");
                }
            }
            Some(v) if *v < -0.1 => {
                if rising_count > 0 && !had_peak {
                    had_peak = true;
                    labels.push("climax");
                }
                falling_count += 1;
                rising_count = 0;
                if !labels.contains(&"resolution") {
                    labels.push("resolution");
                }
            }
            _ => {}
        }
    }

    // Check for stable warmth (low variance across all derivatives)
    let real_derivs: Vec<f32> = derivatives.iter().filter_map(|d| *d).collect();
    if !real_derivs.is_empty() {
        let max_abs = real_derivs.iter().map(|d| d.abs()).fold(0.0f32, f32::max);
        if max_abs < 0.1 {
            labels.clear();
            labels.push("grounded");
        }
    }

    if labels.is_empty() {
        // Fallback: describe the shape
        let mut shape = Vec::new();
        let first_d = derivatives.iter().find_map(|d| *d);
        let last_d = derivatives.iter().rev().find_map(|d| *d);
        if let Some(f) = first_d {
            if f > 0.05 {
                shape.push("rising start");
            } else if f < -0.05 {
                shape.push("falling start");
            } else {
                shape.push("calm start");
            }
        }
        if let Some(l) = last_d {
            if l > 0.05 {
                shape.push("rising end");
            } else if l < -0.05 {
                shape.push("settling end");
            } else {
                shape.push("calm end");
            }
        }
        shape.join(" -> ")
    } else {
        labels.join(" -> ")
    }
}

/// Detect per-step arc segments.
fn detect_segments(derivatives: &[Option<f32>]) -> Vec<String> {
    let mut segments = Vec::new();

    for (i, d) in derivatives.iter().enumerate() {
        if let Some(v) = d {
            let label = if *v > 0.3 {
                "sharp tension rise"
            } else if *v > 0.1 {
                "tension building"
            } else if *v > 0.05 {
                "mild tension"
            } else if *v < -0.3 {
                "sharp release"
            } else if *v < -0.1 {
                "resolution"
            } else if *v < -0.05 {
                "easing"
            } else {
                "stable"
            };
            segments.push(format!("S{} -> S{}: {:+.3} ({})", i, i + 1, v, label));
        }
    }

    segments
}
