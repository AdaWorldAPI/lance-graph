//! PROBE-ENERGY-ARCTANH-DOMAIN — the gate (a) measurement for the helix24
//! energy-register idea (board: E-THE-PERTURBATION-FIELD-NEVER-REACHED-THE-
//! MASK-ALU-1, addendum).
//!
//! Question: is post-cycle `ThinkingEngine::energy` inside the arctanh
//! domain, and is the boundary reachable? The analytical half: `cycle()`
//! sum-normalises (`total energy = 1.0`), so every cell is in [0, 1] with
//! Σ = 1. The dangerous value is exactly 1.0 — `atanh(1) = ∞` — and a
//! delta distribution is not an edge case here, it is the ATTRACTOR the
//! engine converges toward. This probe measures the real code:
//!
//!   P1  invariant: max ≤ 1.0 and Σ ≈ 1 across random tables and cycles
//!   P2  the attractor: a table funnelling all mass into one column must
//!       reach max == 1.0 EXACTLY (f32), i.e. atanh = +inf — can-fire leg
//!   P3  a diffuse table stays strictly < 1.0 — can-stay-silent leg
//!   P4  saturation depth: with Similarity::CLAMP_EPS-style clamping,
//!       2Z(1−ε) = ln(2/ε − 1) ≈ the finite ceiling the register carries
//!
//! Run: cargo run --manifest-path crates/thinking-engine/Cargo.toml \
//!        --example probe_energy_arctanh_domain

use thinking_engine::engine::ThinkingEngine;

// SplitMix64 (Rule 23 discipline: deterministic seed, no wall clock).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn stats(e: &[f32]) -> (f32, f32, usize) {
    let max = e.iter().cloned().fold(0.0f32, f32::max);
    let sum: f32 = e.iter().sum();
    let inf = e.iter().filter(|&&x| x.atanh().is_infinite()).count();
    (max, sum, inf)
}

fn main() {
    const N: usize = 256; // small field, same math as 4096

    // ── P1 + P3: random diffuse tables ────────────────────────────────
    let mut rng = Rng(0x5EED);
    let mut worst_max = 0.0f32;
    let mut worst_sum_dev = 0.0f32;
    for trial in 0..8 {
        let table: Vec<u8> = (0..N * N).map(|_| (rng.next() % 256) as u8).collect();
        let mut eng = ThinkingEngine::new(table);
        // seed: a handful of active atoms
        for k in 0..4 {
            eng.energy[(trial * 7 + k * 61) % N] = 0.25;
        }
        for _ in 0..32 {
            eng.cycle();
            let (max, sum, inf) = stats(&eng.energy);
            worst_max = worst_max.max(max);
            worst_sum_dev = worst_sum_dev.max((sum - 1.0).abs());
            assert_eq!(inf, 0, "P3 FAILED: diffuse table produced atanh=inf");
            assert!(max <= 1.0, "P1 FAILED: max {max} > 1.0");
        }
    }
    println!("P1  invariant held over 8 tables x 32 cycles: max<=1.0, |sum-1| <= {worst_sum_dev:.2e}");
    println!("P3  diffuse tables: worst max = {worst_max:.6} (< 1.0, atanh finite everywhere)");

    // ── P2: the attractor — all similarity funnels into column 0 ──────
    // row[i][0] = 255, everything else 0 ⇒ after one cycle all mass is in
    // cell 0 and normalisation makes it EXACTLY 1.0.
    let mut table = vec![0u8; N * N];
    for i in 0..N {
        table[i * N] = 255;
    }
    let mut eng = ThinkingEngine::new(table);
    eng.energy[3] = 0.5;
    eng.energy[100] = 0.5;
    eng.cycle();
    let (max, _sum, inf) = stats(&eng.energy);
    println!("P2  attractor: max = {max:?} (bits {:#010X}), atanh-inf cells = {inf}", max.to_bits());
    assert_eq!(max, 1.0, "P2: attractor must reach exactly 1.0");
    assert_eq!(inf, 1, "P2: exactly the winner cell must be atanh-infinite");

    // ── P4: saturation depth under clamp ──────────────────────────────
    for eps in [1e-6f64, 1e-9f64] {
        let two_z = ((2.0 - eps) / eps).ln(); // 2·atanh(1-ε) = ln((2-ε)/ε)
        println!("P4  clamp 1-{eps:.0e}: 2Z ceiling = {two_z:.3} rho (8-bit bin over [0,ceiling] = {:.4} rho)", two_z / 255.0);
    }

    println!("\nVERDICT: domain proof HOLDS for the field, FAILS at the attractor —");
    println!("energy==1.0 is the engine's fixed point, not an outlier. A helix24");
    println!("energy register must either EXCLUDE the committed winner (it is the");
    println!("BusDto headline, carried losslessly elsewhere) or clamp with the");
    println!("documented, finite saturation depth above. Clamping without saying");
    println!("so would silently place every converged thought at the same depth.");
}
