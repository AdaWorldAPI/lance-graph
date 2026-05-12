//! Probe P1 — γ-phase-offset ranking discrimination — standalone runner.
//!
//! Run: `cargo run --manifest-path crates/jc/Cargo.toml --release --example probe_p1`
//!
//! Drains entry P1 from `.claude/knowledge/bf16-hhtl-terrain.md` Probe Queue
//! (status before this probe: NOT RUN). On exit, the queue entry should be
//! updated to PASS or FAIL based on the result.

use std::time::Instant;
use jc::probe_p1_gamma_phase;

fn main() {
    println!("═══ Probe P1: γ-phase-offset ranking discrimination ═══");
    println!("Drains queue entry P1 from .claude/knowledge/bf16-hhtl-terrain.md");
    println!("(status before this run: NOT RUN)");
    println!();

    let t = Instant::now();
    let mut r = probe_p1_gamma_phase::prove();
    r.runtime_ms = t.elapsed().as_millis() as u64;

    let status = if r.pass { "✓ PASS — γ+φ pre-rank selector VALID" } else { "✗ FAIL — γ+φ pre-rank selector DEAD" };
    println!("{status}  min ρ={:.6}  threshold<{:.2}  ({} ms)",
        r.measured, r.predicted, r.runtime_ms);
    println!();
    println!("{}", r.detail);
    println!();
    println!("═══ Queue Update Required ═══");
    if r.pass {
        println!("→ Update bf16-hhtl-terrain.md Probe Queue entry P1: NOT RUN → PASS");
        println!("→ Constraint C3 \"VALID — pre-rank discrete selector\" regime is empirically confirmed");
        println!("→ Architectural axiom holds: γ+φ encoding strategy in bgz-tensor is grounded");
    } else {
        println!("→ Update bf16-hhtl-terrain.md Probe Queue entry P1: NOT RUN → FAIL");
        println!("→ Constraint C3 \"VALID — pre-rank discrete selector\" regime FAILS in synthetic data");
        println!("→ Architectural consequence: γ-encoding strategy in bgz-tensor needs revision");
        println!("→ Consider: was the synthetic distribution representative? Re-test with");
        println!("  production codebook before declaring γ+φ universally dead.");
    }
}
