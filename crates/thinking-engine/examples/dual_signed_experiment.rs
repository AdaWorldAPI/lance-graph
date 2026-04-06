//! Dual Signed Experiment: unsigned u8 vs signed i8 on THREE lenses.
//!
//! Runs the dual engine comparison on:
//!   1. Jina v3 HDR 256x256 (cos[-0.067, 0.234]) — narrow, mostly positive
//!   2. Jina Reranker v3 BF16 HDR 256x256 (cos[-0.886, +0.826]) — WIDEST, balanced
//!   3. BGE-M3 HDR 256x256 — multilingual baseline
//!
//! The reranker is the DECISIVE test: nearly symmetric cos range = balanced E/I.
//!
//! Run: cargo run --example dual_signed_experiment

use thinking_engine::jina_lens::JINA_HDR_TABLE;
use thinking_engine::reranker_lens::RERANKER_HDR_TABLE;
use thinking_engine::bge_m3_lens::BGE_M3_HDR_TABLE;
use thinking_engine::dual_engine::DualEngine;
use thinking_engine::signed_engine::SignedThinkingEngine;

fn run_lens_experiment(name: &str, table: &[u8], cos_range: &str) {
    println!("===============================================================");
    println!("  LENS: {} ({})", name, cos_range);
    println!("===============================================================");

    // Sign distribution
    let signed_table: Vec<i8> = table.iter()
        .map(|&v| (v as i16 - 128) as i8)
        .collect();
    let stats = SignedThinkingEngine::new(signed_table);
    println!("{}", stats.sign_stats());
    drop(stats);

    let mut dual = DualEngine::u8_vs_bf16(table.to_vec());

    // Test 1: Clustered input
    println!("\n--- Clustered (centroids 50, 52, 54) ---");
    dual.perturb_both(&[50, 52, 54]);
    let r1 = dual.think_both(20);
    println!("{}", r1.summary());
    dual.reset_both();

    // Test 2: Spread input
    println!("\n--- Spread (centroids 10, 100, 200) ---");
    dual.perturb_both(&[10, 100, 200]);
    let r2 = dual.think_both(20);
    println!("{}", r2.summary());
    dual.reset_both();

    // Test 3: Single atom
    println!("\n--- Single atom (centroid 128) ---");
    dual.perturb_both(&[128]);
    let r3 = dual.think_both(20);
    println!("{}", r3.summary());
    dual.reset_both();

    // Test 4: Dense input
    println!("\n--- Dense (20 centroids, stride 12) ---");
    let dense: Vec<u16> = (0..20).map(|i| i * 12).collect();
    dual.perturb_both(&dense);
    let r4 = dual.think_both(20);
    println!("{}", r4.summary());
    dual.reset_both();

    // Per-lens verdict
    let avg_agree = (r1.agreement + r2.agreement + r3.agreement + r4.agreement) / 4.0;
    let avg_inhib = (r1.total_inhibitions + r2.total_inhibitions
        + r3.total_inhibitions + r4.total_inhibitions) as f32 / 4.0;
    let faster = if (r1.convergence_signed + r2.convergence_signed
        + r3.convergence_signed + r4.convergence_signed)
        < (r1.convergence_unsigned + r2.convergence_unsigned
        + r3.convergence_unsigned + r4.convergence_unsigned)
    { "SIGNED" } else { "UNSIGNED" };

    println!("\n  >> {} avg agreement: {:.0}%, avg inhibitions: {:.0}, faster: {}",
        name, avg_agree * 100.0, avg_inhib, faster);
    println!();
}

fn main() {
    println!();
    println!("################################################################");
    println!("#  DUAL SIGNED EXPERIMENT — u8 vs i8 across 3 lenses          #");
    println!("#  Phase 0: signed/unsigned decision gate                      #");
    println!("################################################################");
    println!();

    run_lens_experiment(
        "Jina v3",
        JINA_HDR_TABLE,
        "cos[-0.067, 0.234]",
    );

    run_lens_experiment(
        "Jina Reranker v3 BF16",
        RERANKER_HDR_TABLE,
        "cos[-0.886, +0.826] WIDEST",
    );

    run_lens_experiment(
        "BGE-M3",
        BGE_M3_HDR_TABLE,
        "multilingual baseline",
    );

    println!("################################################################");
    println!("#  FINAL VERDICT                                               #");
    println!("################################################################");
    println!();
    println!("Compare E/I ratios and agreement across all 3 lenses.");
    println!("If reranker shows lower agreement than Jina v3:");
    println!("  -> Signed distance matters MORE where cos range is wider.");
    println!("  -> Gate sign information is valuable. Path B or C recommended.");
    println!("If all lenses show >90% agreement:");
    println!("  -> Signed adds nothing. Keep unsigned + SiLU-ONNX (Path A).");
}
