//! # JC — Jirak-Cartan: Five-Pillar Proof-in-Code
//!
//! Proves in ~10 minutes of runtime that binary-Hamming causal field
//! computation with VSA bundle has:
//!
//! 1. Substrate-guaranteed Markov structure (E-SUBSTRATE-1)
//! 2. Intrinsic degrees of freedom matching the architecture (Cartan-Kuranishi)
//! 3. Optimal collocation without aliasing (φ-Weyl)
//! 4. Fast prolongation convergence (γ+φ preconditioner)
//! 5. Bounded noise floor under correct dependence model (Jirak 2016)
//! 5b. Pearl 2³ mask-classification accuracy (three-plane Index regime
//!     vs CAM-PQ-shaped bundled regime) — the task-level downstream
//!     consequence of pillar 5's sup-error inflation.
//!
//! Pillars 1, 3, 5, 5b are immediately executable (zero deps, pure Rust).
//! Pillars 2, 4 are stubs pending coupled-revival-track activation.
//!
//! Run: `cargo run --manifest-path crates/jc/Cargo.toml --example prove_it`

pub mod substrate;
pub mod weyl;
pub mod jirak;
pub mod pearl;
pub mod cartan;
pub mod precond;
pub mod koestenberger;
pub mod dueker_zoubouloglou;
pub mod ewa_sandwich;

// Diagnostic probe (not a theorem proof). Run via:
//   cargo run --manifest-path crates/jc/Cargo.toml --release --example sigma_probe
pub mod sigma_codebook_probe;

use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PillarResult {
    pub name: &'static str,
    pub pass: bool,
    pub measured: f64,
    pub predicted: f64,
    pub detail: String,
    pub runtime_ms: u64,
}

impl PillarResult {
    pub fn deferred(name: &'static str, reason: &str) -> Self {
        Self {
            name,
            pass: true,
            measured: 0.0,
            predicted: 0.0,
            detail: format!("DEFERRED — {reason}"),
            runtime_ms: 0,
        }
    }

    pub fn report(&self) {
        let status = if self.detail.starts_with("DEFERRED") {
            "⏸ DEFERRED"
        } else if self.pass {
            "✓ PASS"
        } else {
            "✗ FAIL"
        };
        println!("  {status}  measured={:.6}  predicted={:.6}  ({} ms)",
            self.measured, self.predicted, self.runtime_ms);
        println!("  {}", self.detail);
    }
}

pub fn run_all_pillars() -> Vec<PillarResult> {
    let pillars: Vec<(&str, fn() -> PillarResult)> = vec![
        ("E-SUBSTRATE-1: bundle associativity @ d=10000", substrate::prove),
        ("Cartan-Kuranishi: role_keys ≡ Cartan characters", cartan::prove),
        ("φ-Weyl: 144-verb collocation coverage", weyl::prove),
        ("γ+φ preconditioner: prolongation step reduction", precond::prove),
        ("Jirak Berry-Esseen: weak-dep noise floor @ d=16384", jirak::prove),
        ("Pearl 2³ mask-accuracy: three-plane vs bundled @ d=16384", pearl::prove),
        ("Köstenberger-Stark: inductive mean on Hadamard 2×2 SPD", koestenberger::prove),
        ("Düker-Zoubouloglou: Hilbert-space CLT for AR(1) in ℝ^16384", dueker_zoubouloglou::prove),
        ("EWA-Sandwich: Σ-push-forward along multi-hop edge paths", ewa_sandwich::prove),
    ];

    let total = pillars.len();
    let mut results = Vec::new();
    for (i, (name, f)) in pillars.iter().enumerate() {
        println!("[{:02}/{:02}] {name}", i + 1, total);
        let t = Instant::now();
        let mut r = f();
        if r.runtime_ms == 0 && !r.detail.starts_with("DEFERRED") {
            r.runtime_ms = t.elapsed().as_millis() as u64;
        }
        r.report();
        println!();
        results.push(r);
    }
    results
}
