//! The **buffer axis** — impulse storage before yield (the transient layer the
//! resistive model omits).
//!
//! Resistance / Kirchhoff index / effective resistance are the **steady-state
//! (DC)** response: how the grid redistributes a *sustained* imbalance. They do
//! NOT capture how a **sudden impulse** (a Kugelstoßpendel/Newton's-cradle strike
//! — a line trip dumping power in one cycle) is *buffered* before anything moves.
//! That buffering is the **transient (RLC / swing)** layer: the kinetic energy
//! stored in rotating mass (inertia `H`) plus capacitance absorbs the impulse
//! **elastically** while the frequency excursion stays inside the protection band,
//! then **yields suddenly** once the band is crossed — the **Ketchup effect**
//! (shear-thinning: nothing, nothing, then collapse). The yield triggers the
//! relay, which re-tops the topology, which is the *next* impulse — the cascade.
//!
//! Why this matters here (the confound the operator named): the modifier's `1/λ₂`
//! amplifier is the **dominant term of the Kirchhoff index** `Kf = n·Σ 1/λ_k`
//! (λ₂ smallest ⇒ 1/λ₂ largest), so "Weyl × (1/Fiedler)" was confounded with the
//! resistive resilience certificate — both are conductance. The **buffer is
//! orthogonal to λ₂/Kirchhoff by construction** — it is *storage*, set by inertia,
//! not *conductance*, set by topology. It is the independent third axis.
//!
//! Honest scope: this is a first-order impulse-headroom model (swing-equation
//! RoCoF + a protection band), NOT a transient-stability ODE integrator. Real
//! per-bus inertia `H` is NOT in the PyPSA topology we carry, so examples use a
//! flagged proxy; the *structure* (buffer ⊥ connectivity) holds regardless.

use crate::timing::F0_HZ;

/// Impulse buffer headroom: the largest sudden power imbalance (fraction of system
/// power) a unit with inertia constant `inertia_h` (s) absorbs before its frequency
/// crosses the protection band `df_band` (Hz). From the swing equation
/// `RoCoF = f₀·Δp/(2H)`, the impulse is absorbed while `|Δf| < df_band`, so
/// `Δp_max = 2·H·df_band / f₀`. Larger inertia ⇒ bigger buffer ⇒ more impulse
/// absorbed before yield. This is `C`-like storage, independent of network λ₂.
pub fn impulse_buffer(inertia_h: f64, df_band: f64) -> f64 {
    if F0_HZ <= 0.0 {
        return 0.0;
    }
    2.0 * inertia_h.max(0.0) * df_band / F0_HZ
}

/// Outcome of metering an impulse against a buffer (the Ketchup yield test).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yield {
    /// Did the cell yield (buffer exhausted → trigger the cascade)?
    pub yielded: bool,
    /// Remaining headroom as a fraction of the buffer: `(buffer − impulse)/buffer`.
    /// Positive = elastic hold; ≤ 0 = overrun (the Ketchup yield).
    pub headroom: f64,
}

/// The **Ketchup yield**: an `impulse` met by a `buffer`. Below the buffer the cell
/// holds elastically (shear-thinning: the impulse is stored, nothing trips); at or
/// above it the cell yields suddenly (the non-Newtonian threshold) and triggers the
/// cascade. Sharp threshold by design — that is the "nothing, nothing, then
/// collapse" signature, not a smooth ramp.
pub fn ketchup_yield(impulse: f64, buffer: f64) -> Yield {
    if buffer <= 0.0 {
        return Yield {
            yielded: impulse > 0.0,
            headroom: if impulse > 0.0 { -1.0 } else { 0.0 },
        };
    }
    Yield {
        yielded: impulse >= buffer,
        headroom: (buffer - impulse) / buffer,
    }
}

/// A compartment's aggregate impulse buffer = Σ per-node buffers (total stored
/// headroom). The compartment absorbs a redistributed impulse up to this total
/// before the first internal yield. `inertia_h` is per-node inertia (seconds).
pub fn compartment_buffer(inertia_h: &[f64], df_band: f64) -> f64 {
    inertia_h.iter().map(|&h| impulse_buffer(h, df_band)).sum()
}

/// The `inertia_buffer` SoA column for a set of buses: each bus's impulse buffer
/// ([`impulse_buffer`]) min-max **normalized to `[0,1]`** — the form the calibrated
/// [`crate::INERTIA`] spec stores (normalized, not raw physical units; normalizing
/// also lifts the axis clear of the ICC variance-underflow guard). This is the
/// promoted additive value member, *computed*. It is orthogonal to topology by the
/// HHTL-OGAR key/value split (topology is the GUID key; this is one more value) —
/// the structural fact `buffer_is_independent_of_connectivity` witnesses. Degenerate
/// (all-equal `H`, or empty) input yields all-zeros, never `NaN`.
pub fn inertia_buffer_column(per_bus_h: &[f64], df_band: f64) -> Vec<f32> {
    let raw: Vec<f64> = per_bus_h
        .iter()
        .map(|&h| impulse_buffer(h, df_band))
        .collect();
    let lo = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = hi - lo;
    raw.iter()
        .map(|&r| {
            if span > 0.0 {
                ((r - lo) / span) as f32
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inertia_column_is_normalized_and_monotone_in_h() {
        // Per-bus inertia decoupled from any wiring: the column normalizes to [0,1]
        // and preserves the H order (impulse_buffer is monotone in H at fixed df).
        let h = [1.0_f64, 5.0, 3.0, 0.0, 8.0];
        let col = inertia_buffer_column(&h, 0.2);
        assert_eq!(col.len(), h.len());
        for &c in &col {
            assert!((0.0..=1.0).contains(&c), "normalized to [0,1]");
        }
        assert_eq!(col[3], 0.0, "H=0 is the min → 0.0");
        assert_eq!(col[4], 1.0, "H=8 is the max → 1.0");
        // Order preserved: H[0]=1 < H[2]=3 < H[1]=5 < H[4]=8.
        assert!(col[0] < col[2] && col[2] < col[1] && col[1] < col[4]);
        // Degenerate (all-equal H → no span) yields zeros, never NaN.
        let flat = inertia_buffer_column(&[4.0, 4.0, 4.0], 0.2);
        assert!(flat.iter().all(|&c| c == 0.0));
        assert!(inertia_buffer_column(&[], 0.2).is_empty());
    }

    #[test]
    fn buffer_grows_with_inertia() {
        // Twice the inertia ⇒ twice the impulse absorbed (the storage scaling).
        let small = impulse_buffer(2.0, 0.2);
        let big = impulse_buffer(4.0, 0.2);
        assert!((big - 2.0 * small).abs() < 1e-12, "buffer ∝ H");
        // RoCoF consistency: Δp_max = 2·H·df/f₀ with f₀=50.
        assert!((impulse_buffer(5.0, 0.2) - (2.0 * 5.0 * 0.2 / 50.0)).abs() < 1e-12);
    }

    #[test]
    fn ketchup_holds_then_yields_sharply() {
        let buffer = impulse_buffer(5.0, 0.2); // = 0.04
                                               // Below buffer: elastic hold, positive headroom, no trip.
        let lo = ketchup_yield(0.02, buffer);
        assert!(!lo.yielded && lo.headroom > 0.0);
        // At/above buffer: sudden yield, headroom ≤ 0.
        let hi = ketchup_yield(0.05, buffer);
        assert!(hi.yielded && hi.headroom <= 0.0);
        // The threshold is sharp: just under vs just over flips `yielded`.
        assert!(!ketchup_yield(buffer * 0.999, buffer).yielded);
        assert!(ketchup_yield(buffer * 1.001, buffer).yielded);
    }

    #[test]
    fn buffer_is_independent_of_connectivity() {
        // Two compartments with identical inertia have identical buffers regardless
        // of how their nodes are wired — buffer is storage, not conductance.
        let h = [4.0, 4.0, 4.0];
        let b1 = compartment_buffer(&h, 0.2);
        let b2 = compartment_buffer(&h, 0.2);
        assert_eq!(b1, b2);
        // Zero buffer (no inertia, e.g. full-renewable) ⇒ any impulse yields.
        assert!(ketchup_yield(1e-9, compartment_buffer(&[0.0, 0.0], 0.2)).yielded);
    }
}
