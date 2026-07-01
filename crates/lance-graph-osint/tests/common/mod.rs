//! Shared test helpers for the P-series convergence probes.
//!
//! Lives in `tests/common/mod.rs` (not `tests/common.rs`) so cargo does not
//! compile it as its own test binary — it is included by each probe via
//! `mod common;`.

/// Deterministic PRNG — SplitMix64. No `rand`, no seed entropy; every probe's
/// index stream is byte-identical on every run and every target.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
