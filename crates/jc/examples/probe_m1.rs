//! Probe M1 — CLAM 3-level 16-way tree on 256 Jina-v5 centroids — standalone runner.
//!
//! Run from repo root:
//!   `cargo run --manifest-path crates/jc/Cargo.toml --release --example probe_m1`
//!
//! Drains entry M1 from `.claude/knowledge/bf16-hhtl-terrain.md` Probe Queue
//! (status before this probe: PARTIAL — CHAODA on 256 rows works, 16-way
//! tree shape NOT YET tested). On exit, queue should be updated to
//! PASS or FAIL based on the result.

use std::time::Instant;
use jc::probe_m1_clam_tree;

fn main() {
    println!("═══ Probe M1: CLAM 3-level 16-way tree on 256 Jina-v5 centroids ═══");
    println!("Drains queue entry M1 from .claude/knowledge/bf16-hhtl-terrain.md");
    println!("(status before this run: PARTIAL — 16-way tree shape not yet tested)");
    println!();

    let t = Instant::now();
    let mut r = probe_m1_clam_tree::prove();
    r.runtime_ms = t.elapsed().as_millis() as u64;

    let status = if r.pass {
        "✓ PASS-with-caveat — 16-way fits, but uncalibrated single-shot"
    } else {
        "✗ FAIL — 16-way CLAM tree does NOT fit naturally"
    };
    println!("{status}  L0 balance={:.4}  threshold≤{:.2}  ({} ms)",
        r.measured, r.predicted, r.runtime_ms);
    println!();
    println!("{}", r.detail);
    println!();
    println!("═══ Queue Update Required ═══");
    if r.pass {
        println!("→ Update bf16-hhtl-terrain.md Probe Queue entry M1: PARTIAL → PASS-with-caveat");
        println!("→ The L0 = 16 coarse-cluster claim is consistent with the data on");
        println!("  uncalibrated single-shot Ward — necessary but not sufficient.");
        println!("→ True closure (M1') needs ICC-calibrated codebook + CascadeConfig sweep");
        println!("  + cross-class re-test. Tracked as separate Open Idea in IDEAS.md.");
    } else {
        println!("→ Update bf16-hhtl-terrain.md Probe Queue entry M1: PARTIAL → FAIL");
        println!("→ Architectural consequence: 16-way bit-layout claim needs revision.");
        println!("→ Consider whether a non-uniform branching factor (e.g. 8-way at L0,");
        println!("  32-way at L1) better matches the natural geometry — or whether");
        println!("  the 26/256 CHAODA-flagged outliers should be excluded before testing.");
    }
}
