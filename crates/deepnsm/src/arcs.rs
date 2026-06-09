//! Basin/literal arc split — the **projection** decomposition (Broca /
//! MarkovBundler side of `E-ENGLISH-BIFURCATES`).
//!
//! This is strictly the *projection* faculty: it decomposes the role-superposed
//! `Trajectory` (the MarkovBundler's wave) into its two arcs. It does **not**
//! route and does **not** resolve — sentence resolution (literal comprehension +
//! ambiguity resolution, tokenless) is a SEPARATE faculty in `comprehension.rs`
//! (Wernicke). Keeping the projection apart from the resolution is the
//! anti-spaghetti boundary (user, 2026-05-31: *"Markov bundler should be
//! separate as the projection, while the sentence resolution is literal text
//! comprehension with ambiguity resolution without tokens"*).
//!
//! - **basin arc** — the role-superposed spine bundle: the *declared/exact*
//!   meaning keyframe (points at ONE basin — a DOLCE class / story-arc).
//! - **literal arc** — the COCA ranks that fed it: the *detected/redundant*
//!   surface, prunable once the basin resolves (the prune/tombstone lifecycle
//!   itself lives contract-side in `WitnessTable`, not here).
//!
//! Firewall: both arcs stay English-side; the basin's f32 is upstream-only
//! (sign-binarized via `disambiguator_glue`, or resolved to an opaque handle,
//! before it ever crosses into the agnostic graph); no COCA rank reaches the
//! hot graph as identity.

use crate::trajectory::Trajectory;

/// The semantic spine: the role-superposed bundle that points at ONE basin.
/// The declared/exact side of the language↔meaning duality.
#[derive(Debug, Clone, PartialEq)]
pub struct BasinArc(pub Vec<f32>);

/// The language surface: the COCA literal ranks that fed the bundle.
/// Multiple, redundant, prunable once the basin resolves (the detected side).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralArc(pub Vec<u16>);

impl Trajectory {
    /// Split this trajectory (the projection / wave) into its **basin arc**
    /// (the full role-superposed spine bundle) and its **literal arc** (the
    /// COCA ranks that fed it).
    ///
    /// Projection-side only: it names the duality at the seam where
    /// `disambiguator_glue` already threads the bundle into the contract
    /// `context_chain`. It performs no fact/story routing — that is a
    /// comprehension decision and lives in `comprehension.rs`.
    #[must_use]
    pub fn split_arcs(&self, literal_ranks: &[u16]) -> (BasinArc, LiteralArc) {
        (
            BasinArc(self.fingerprint.clone()),
            LiteralArc(literal_ranks.to_vec()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIMS: usize = 16_384;

    #[test]
    fn split_arcs_preserves_basin_and_literals() {
        let t = Trajectory {
            fingerprint: vec![0.25_f32; DIMS],
            radius: 5,
        };
        let ranks = [12_u16, 670, 2942];
        let (basin, literal) = t.split_arcs(&ranks);
        assert_eq!(basin.0, t.fingerprint, "basin arc IS the spine bundle");
        assert_eq!(
            literal.0, ranks,
            "literal arc carries the COCA ranks verbatim"
        );
    }

    #[test]
    fn literal_arc_is_independent_of_basin() {
        // Same basin, two different literal sets → distinct literal arcs.
        // The prune target (literals) is separable from the spine (basin).
        let t = Trajectory {
            fingerprint: vec![1.0_f32; DIMS],
            radius: 5,
        };
        let (b1, l1) = t.split_arcs(&[1, 2, 3]);
        let (b2, l2) = t.split_arcs(&[9, 9]);
        assert_eq!(b1, b2, "basin unchanged by literal choice");
        assert_ne!(l1, l2, "literal arcs differ");
    }
}
