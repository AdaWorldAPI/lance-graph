//! The witness arc as a standing wave (METHODS §11) — particle (pointer-chase) vs
//! wave (Walsh-pyramid Parseval), on a real grid inertia field.
//!
//! Builds the per-bus inertia-buffer field, then reads three witness arcs both ways:
//! the particle walk (`O(hops)` per arc) and the standing wave (one `field_spectrum`
//! transform reused across all arcs). They agree to floating point. As implemented,
//! each arc is itself transformed, so the per-arc cost is `O(N log N)` (it narrows to
//! `O(N)` only when the arc spectrum is precomputed or the arc is structured/sparse);
//! the amortized quantity is the one **field** transform, not the per-arc dot.
//!
//! Run: cargo run --release --manifest-path crates/perturbation-sim/Cargo.toml \
//!        --example witness

use perturbation_sim::{
    field_spectrum, inertia_buffer_column, witness_from_spectrum, witness_particle,
};

fn proxy_inertia(n: usize) -> Vec<f64> {
    (0..n).map(|i| 2.0 + (i % 5) as f64).collect()
}

fn main() {
    // The field: a per-bus inertia-buffer field over a 16-bus line (METHODS §11's
    // "inertia field on power grids"; the promoted additive member as the carrier).
    let n = 16;
    let field = inertia_buffer_column(&proxy_inertia(n), 0.2);
    let field: Vec<f64> = field.iter().map(|&x| x as f64).collect();

    // Three witness arcs (Markov #1 reference chains over the buses):
    //   - a single-hop witness (read one bus),
    //   - a coarse dyadic arc (the first half — a Raumgewinn-scale reference),
    //   - an alternating signed chain (a fine-scale infight reference).
    let mut single = vec![0.0; n];
    single[5] = 1.0;
    let mut coarse = vec![0.0; n];
    for c in coarse.iter_mut().take(n / 2) {
        *c = 1.0;
    }
    let alt: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let arcs: [(&str, &[f64]); 3] = [
        ("single-hop", &single),
        ("coarse dyadic", &coarse),
        ("alternating", &alt),
    ];

    // Wave view: transform the field ONCE, then reuse the spectrum across all arcs
    // (each arc still transformed → O(N log N) per arc; O(N) only if precomputed).
    let spectrum = field_spectrum(&field);

    println!("witness arc as a standing wave — particle (walk) vs wave (Parseval)\n");
    println!("  field: {n}-bus inertia-buffer field; one Walsh transform reused by all arcs\n");
    println!(
        "  {:>14}  {:>14}  {:>14}  {:>10}",
        "arc", "particle", "wave", "Δ"
    );
    let mut max_err = 0.0_f64;
    for (name, arc) in arcs {
        let p = witness_particle(&field, arc);
        let w = witness_from_spectrum(&spectrum, arc);
        let d = (p - w).abs();
        max_err = max_err.max(d);
        println!("  {name:>14}  {p:>14.9}  {w:>14.9}  {d:>10.2e}");
    }
    println!(
        "\n  max |particle − wave| = {max_err:.2e} (Parseval: Hᵀ H = N·I, exact up to fp).\n  \
         particle = O(hops) pointer-chase per arc; wave = O(N log N) field transform once,\n  \
         reused across all arcs (each arc O(N log N) as written; O(N) only if precomputed).\n  \
         The standing wave IS the witness arc — evaluated all at once, no chain walk.\n  \
         (Demonstration in perturbation-sim; the contract witness_table evaluator is the\n  \
         separate gated step — the SoA spine is additive-only behind the iron rules.)"
    );
}
