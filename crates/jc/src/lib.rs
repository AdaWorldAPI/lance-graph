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
//! 7. Concentration on Hadamard space (Köstenberger-Stark 2024)
//! 8. Hilbert-space CLT for AR(1) (Düker-Zoubouloglou 2024)
//! 9. EWA-sandwich Σ-push-forward along multi-hop edge paths
//! 10. Nested-distance Lipschitz on Sigma DN-trees (Pflug-Pichler 2012)
//!     — certifies CAM-PQ tree quantization preserves FreeEnergy within Lε.
//! 11. Signature uniqueness on tree-quotient (Hambly-Lyons 2010)
//!     — certifies sigker's Index-regime classification.
//!
//! Pillars 1, 3, 4, 5, 5b, 7-11 are immediately executable. Pillar 4
//! activated 2026-05-07 once `EULER_GAMMA` + `GOLDEN_RATIO` stabilized
//! in `std::f64::consts` (Rust 1.94). Pillar 11 activated 2026-05-07
//! once sigker landed in the workspace (PR #348). Pillar 2 (Cartan-
//! Kuranishi) remains deferred pending coupled-revival-track activation
//! (learned-attention-mask module).
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
pub mod pflug;
pub mod hambly_lyons;

// Diagnostic probe (not a theorem proof). Run via:
//   cargo run --manifest-path crates/jc/Cargo.toml --release --example sigma_probe
pub mod sigma_codebook_probe;

// Diagnostic probe — drains an entry from bf16-hhtl-terrain.md probe queue.
// Run via:
//   cargo run --manifest-path crates/jc/Cargo.toml --release --example probe_p1
pub mod probe_p1_gamma_phase;

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
        ("Pflug-Pichler: nested-distance Lipschitz on Sigma DN-trees", pflug::prove),
        ("Hambly-Lyons: signature uniqueness on tree-quotient", hambly_lyons::prove),
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
