//! Σ-Codebook Viability Probe — standalone runner.
//!
//! Run: `cargo run --manifest-path crates/jc/Cargo.toml --release --example sigma_probe`
//!
//! NOT a pillar (no theorem proven). Empirical measurement of whether
//! 256-entry codebook quantization preserves Σ-tensor information well
//! enough for the white-matter Σ-edge encoding decision.

use std::time::Instant;
use jc::sigma_codebook_probe;

fn main() {
    println!("═══ Σ-Codebook Viability Probe ═══");
    println!("Empirical measurement (NOT a theorem proof)");
    println!();

    let t = Instant::now();
    let mut r = sigma_codebook_probe::prove();
    r.runtime_ms = t.elapsed().as_millis() as u64;

    let status = if r.pass { "✓ CODEBOOK VIABLE" } else { "✗ INSUFFICIENT (read recommendation)" };
    println!("{status}  R²={:.6}  threshold≥{:.2}  ({} ms)",
        r.measured, r.predicted, r.runtime_ms);
    println!();
    println!("{}", r.detail);
    println!();
    println!("═══ Decision ═══");
    if r.measured >= 0.99 {
        println!("→ Implement Option A (Σ-Codebook 1-byte sidecar) or Option C (SchemaSidecar Block 14/15).");
        println!("→ CausalEdge64 stays unchanged. No 8→16 byte expansion needed.");
        println!("→ HighHeelBGZ container hard limit (240 edges per 2KB basin) preserved.");
    } else if r.measured >= 0.95 {
        println!("→ Marginal. Caller decides between k=4096 codebook (12-bit index) or hybrid (codebook + outlier sidecar).");
    } else {
        println!("→ Codebook quantization not viable. Use separate Lance side-table (full 7-float Σ per edge).");
        println!("→ Container layout stays unchanged; Σ retrieved on-demand by edge-ID lookup.");
    }
}
