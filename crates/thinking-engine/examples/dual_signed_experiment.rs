//! Dual Signed Experiment: unsigned u8 vs signed i8 on Jina HDR lens.
//!
//! Uses the baked 256x256 Jina v3 HDR table (cos[-0.067, 0.234]).
//! Runs both engines on the same input, compares peaks.
//!
//! NOTE: Jina v3 is mostly positive (~90% excitatory). Weak inhibition expected.
//! The decisive test is Jina reranker (cos[-0.8, +0.8], balanced E/I) — pending wiring.
//!
//! Run: cargo run --example dual_signed_experiment

use thinking_engine::jina_lens::JINA_HDR_TABLE;
use thinking_engine::dual_engine::DualEngine;
use thinking_engine::signed_engine::SignedThinkingEngine;

fn main() {
    println!("===============================================================");
    println!("  DUAL SIGNED EXPERIMENT: unsigned u8 vs signed i8");
    println!("  Lens: Jina v3 HDR 256x256 (CDF-percentile, std=73.6)");
    println!("  NOTE: Jina v3 cos[-0.067, 0.234] = mostly positive.");
    println!("  Real test: Jina reranker cos[-0.8, +0.8] (pending wiring).");
    println!("===============================================================");
    println!();

    // --- Table statistics ---
    let signed_table: Vec<i8> = JINA_HDR_TABLE.iter()
        .map(|&v| (v as i16 - 128) as i8)
        .collect();
    let stats_engine = SignedThinkingEngine::new(signed_table);
    println!("Sign distribution: {}", stats_engine.sign_stats());
    println!();
    drop(stats_engine);

    // --- Create dual engine from baked Jina table ---
    let mut dual = DualEngine::from_unsigned_table(JINA_HDR_TABLE.to_vec());

    // --- Test 1: Clustered input (nearby centroids) ---
    println!("--- Test 1: Clustered input (centroids 50, 52, 54) ---");
    dual.perturb_both(&[50, 52, 54]);
    let r1 = dual.think_both(20);
    println!("{}", r1.summary());
    dual.reset_both();
    println!();

    // --- Test 2: Spread input (distant centroids) ---
    println!("--- Test 2: Spread input (centroids 10, 100, 200) ---");
    dual.perturb_both(&[10, 100, 200]);
    let r2 = dual.think_both(20);
    println!("{}", r2.summary());
    dual.reset_both();
    println!();

    // --- Test 3: Single atom ---
    println!("--- Test 3: Single atom (centroid 128) ---");
    dual.perturb_both(&[128]);
    let r3 = dual.think_both(20);
    println!("{}", r3.summary());
    dual.reset_both();
    println!();

    // --- Test 4: Dense input (20 centroids, stride 12) ---
    println!("--- Test 4: Dense input (20 centroids, stride 12) ---");
    let dense: Vec<u16> = (0..20).map(|i| i * 12).collect();
    dual.perturb_both(&dense);
    let r4 = dual.think_both(20);
    println!("{}", r4.summary());
    dual.reset_both();
    println!();

    // --- Summary ---
    println!("===============================================================");
    println!("  VERDICT");
    println!("===============================================================");
    let avg_agreement = (r1.agreement + r2.agreement + r3.agreement + r4.agreement) / 4.0;
    let avg_inhib = (r1.total_inhibitions + r2.total_inhibitions
        + r3.total_inhibitions + r4.total_inhibitions) as f32 / 4.0;

    println!("Average agreement: {:.0}%", avg_agreement * 100.0);
    println!("Average total inhibitions: {:.0}", avg_inhib);
    println!();

    if avg_agreement > 0.9 {
        println!("-> High agreement: signed adds little on this lens.");
        println!("   Expected for Jina v3 (mostly positive cos range).");
        println!("   Re-run on Jina reranker (balanced E/I) for decisive result.");
    } else if avg_agreement > 0.5 {
        println!("-> Moderate agreement: signed finds different peaks.");
        println!("   Consider Path C (run both, merge via superposition).");
    } else {
        println!("-> Low agreement: fundamentally different topology.");
        println!("   Investigate: signed may see structure unsigned misses.");
    }
    println!();

    if avg_inhib < 10.0 {
        println!("-> Weak inhibition: table is mostly positive. Signed ~ unsigned.");
    } else if avg_inhib < 100.0 {
        println!("-> Moderate inhibition: balanced E/I. Brain-like dynamics.");
    } else {
        println!("-> Strong inhibition: many atoms suppressed. Check for over-inhibition.");
    }
}
