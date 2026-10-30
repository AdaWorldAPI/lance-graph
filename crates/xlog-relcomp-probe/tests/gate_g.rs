//! Gate G: shared evidence. Two proofs share one observation:
//!
//! ```text
//! P1 = F ∧ G,  P2 = F ∧ H,  pF = 0.8, pG = 0.6, pH = 0.5
//! ```
//!
//! The independent reference is exhaustive enumeration of the 2³ worlds.
//! Exact: P(P1 ∨ P2) = 0.64, ∂/∂(pF, pG, pH) = (0.8, 0.4, 0.32).
//!
//! What this file pins is the separation the probe relies on: the Boolean
//! support mask answers "is there a proof" (true), it is not a probability,
//! and treating the two proofs as independent (noisy-OR over proof
//! probabilities) gives 0.688, the shared-evidence error. Hence the
//! register path refuses probabilistic programs outright
//! (`refuses_probabilistic_programs_even_when_the_rir_is_deterministic`).

const P: [f64; 3] = [0.8, 0.6, 0.5]; // F, G, H

fn holds(w: u32) -> bool {
    let (f, g, h) = (w & 1 != 0, w & 2 != 0, w & 4 != 0);
    (f && g) || (f && h)
}

/// Σ over worlds of P(world) · [query holds].
fn exact(p: [f64; 3]) -> f64 {
    (0..8u32)
        .filter(|&w| holds(w))
        .map(|w| {
            (0..3)
                .map(|i| if w >> i & 1 == 1 { p[i] } else { 1.0 - p[i] })
                .product::<f64>()
        })
        .sum()
}

/// ∂P/∂p_i by the world sum: P(q | x_i) − P(q | ¬x_i).
fn derivative(i: usize) -> f64 {
    let mut hi = P;
    let mut lo = P;
    hi[i] = 1.0;
    lo[i] = 0.0;
    exact(hi) - exact(lo)
}

#[test]
fn exact_probability_and_derivatives_by_world_enumeration() {
    assert!((exact(P) - 0.64).abs() < 1e-12);
    let d: Vec<f64> = (0..3).map(derivative).collect();
    for (got, want) in d.iter().zip([0.8, 0.4, 0.32]) {
        assert!((got - want).abs() < 1e-12, "{d:?}");
    }
}

#[test]
fn proof_independence_overcounts_shared_evidence() {
    let (p1, p2) = (P[0] * P[1], P[0] * P[2]);
    let noisy_or = 1.0 - (1.0 - p1) * (1.0 - p2);
    assert!((noisy_or - 0.688).abs() < 1e-12);
    assert!(
        noisy_or > exact(P),
        "collapsing shared F inflates the answer"
    );
    // Max over proofs (a min-max semiring) undercounts instead.
    assert!(p1.max(p2) < exact(P));
}

#[test]
fn boolean_support_is_not_a_probability() {
    // With every fact present the support mask says "derivable" — the one
    // bit it carries. It holds in 3 of 8 worlds, and no mask over the three
    // fact presences can recover 0.64 without the weights.
    assert!(holds(0b111));
    assert_eq!((0..8u32).filter(|&w| holds(w)).count(), 3);
}
