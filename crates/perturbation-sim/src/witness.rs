//! Probe 2 — the **witness arc as a standing wave** (METHODS §11).
//!
//! A *witness arc* is a Markov #1 reference chain over the SoA — a path that reads
//! a field by accumulating it along the chain. METHODS §11 says the same arc can be
//! evaluated two ways, and they must agree:
//!
//! - **Particle view (pointer chasing).** Walk the arc hop by hop, accumulating the
//!   field along it: `∑ field[i]·arc[i]`. One dependent load per hop — `O(hops)`
//!   sequential dereferences (the `CausalEdge64` W-slot → witness chain walk).
//! - **Wave view (standing wave).** The arc's reading IS an inner product against
//!   the field, and by **Parseval for the Walsh–Hadamard transform** (`Hᵀ H = N·I`,
//!   the involution-up-to-`N` the pyramid already provides via [`fwht`]):
//!   `⟨field, arc⟩ = (1/N)·⟨Ĥfield, Ĥarc⟩`. Transform the field **once**
//!   (`O(N log N)`); then every witness arc is an `O(N)` dot against its spectrum —
//!   the whole arc is evaluated at once, no walk.
//!
//! **The identity `particle == wave` is what this probe proves** ([`particle_equals_wave`]).
//! The payoff is amortization: [`field_spectrum`] computes the standing wave once,
//! and [`witness_from_spectrum`] reads many arcs off it — `O(N log N) + q·O(N)` for
//! `q` arcs, vs the particle view's `q` independent pointer-chasing walks.
//!
//! Self-contained, in perturbation-sim — this demonstrates the pyramid/field
//! mechanism on a grid field (e.g. the [`crate::inertia_buffer_column`] field).
//! Wiring it as the *actual* `witness_table` evaluator in the contract is a separate,
//! gated step: the witness/SoA types are the cognitive spine — additive only, behind
//! the iron rules.

use crate::sketch::fwht;

/// Pad a field to the next power-of-two length (zero-filled), the length [`fwht`]
/// requires. Returns the padded buffer.
fn pad_pow2(v: &[f64]) -> Vec<f64> {
    let mut n = 1usize;
    while n < v.len().max(1) {
        n <<= 1;
    }
    let mut a = vec![0.0; n];
    a[..v.len()].copy_from_slice(v);
    a
}

/// **Particle view.** The witness arc read by walking it: `∑ field[i]·arc[i]`, the
/// pointer-chase accumulation (`O(len)` sequential). `field` and `arc` are read up to
/// their shared length.
pub fn witness_particle(field: &[f64], arc: &[f64]) -> f64 {
    field.iter().zip(arc).map(|(f, a)| f * a).sum()
}

/// The field's **standing-wave spectrum** — the Walsh–Hadamard pyramid of the field,
/// computed ONCE. Reuse across many witness arcs via [`witness_from_spectrum`].
pub fn field_spectrum(field: &[f64]) -> Vec<f64> {
    let mut a = pad_pow2(field);
    fwht(&mut a);
    a
}

/// Read a witness arc off a precomputed field spectrum: `(1/N)·⟨spectrum, Ĥarc⟩`
/// (Parseval). Equals [`witness_particle`] up to floating-point. `spectrum` must come
/// from [`field_spectrum`] (length `N`, a power of two).
pub fn witness_from_spectrum(spectrum: &[f64], arc: &[f64]) -> f64 {
    let n = spectrum.len();
    if n == 0 {
        return 0.0;
    }
    let mut a = vec![0.0; n];
    let take = arc.len().min(n);
    a[..take].copy_from_slice(&arc[..take]);
    fwht(&mut a);
    spectrum.iter().zip(&a).map(|(x, y)| x * y).sum::<f64>() / n as f64
}

/// **Wave view.** The same reading as [`witness_particle`], via Parseval on the Walsh
/// pyramid — transform field + arc, then `(1/N)·∑ Ĥfield·Ĥarc`. Self-contained
/// (transforms both); for many arcs prefer [`field_spectrum`] + [`witness_from_spectrum`].
pub fn witness_wave(field: &[f64], arc: &[f64]) -> f64 {
    witness_from_spectrum(&field_spectrum(field), arc)
}

/// Convenience: does the wave view reproduce the particle view for this `(field,
/// arc)` within `tol`? The probe's pass/fail predicate.
pub fn particle_equals_wave(field: &[f64], arc: &[f64], tol: f64) -> bool {
    (witness_particle(field, arc) - witness_wave(field, arc)).abs() <= tol
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parseval: the standing wave reproduces the pointer-chase walk exactly (fp).
    #[test]
    fn particle_equals_wave_parseval() {
        let field = [0.3, -1.2, 0.7, 2.1, -0.4, 0.9, 1.5, -0.8];
        // A witness arc: a signed Markov reference chain over the nodes.
        let arc = [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0];
        let p = witness_particle(&field, &arc);
        let w = witness_wave(&field, &arc);
        assert!((p - w).abs() < 1e-9, "particle {p} vs wave {w}");
        assert!(particle_equals_wave(&field, &arc, 1e-9));
    }

    /// The amortization claim: ONE field transform, then many arcs read off it — each
    /// matching its particle walk.
    #[test]
    fn standing_wave_reuses_one_transform() {
        let field = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = field_spectrum(&field); // computed once
        let arcs = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // single-hop witness
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], // coarse dyadic arc
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0], // alternating chain
        ];
        for arc in &arcs {
            let p = witness_particle(&field, arc);
            let w = witness_from_spectrum(&spectrum, arc);
            assert!(
                (p - w).abs() < 1e-9,
                "arc {arc:?}: particle {p} vs wave {w}"
            );
        }
    }

    /// Non-power-of-two fields are padded transparently; the identity still holds.
    #[test]
    fn handles_non_power_of_two_and_ragged_arc() {
        let field = [0.5, -0.5, 2.0, 1.0, -1.0]; // length 5 → pads to 8
        let arc = [1.0, 1.0, -1.0]; // shorter arc → zero-extended
        let p = witness_particle(&field, &arc);
        let w = witness_wave(&field, &arc);
        assert!((p - w).abs() < 1e-9, "ragged: particle {p} vs wave {w}");
    }

    /// Degenerate inputs never panic / never NaN.
    #[test]
    fn degenerate_is_safe() {
        assert_eq!(witness_particle(&[], &[]), 0.0);
        assert_eq!(witness_wave(&[], &[]), 0.0);
        assert!(field_spectrum(&[]).len().is_power_of_two());
        assert_eq!(witness_from_spectrum(&[], &[1.0]), 0.0);
    }
}
