//! Hambly-Lyons 2010: Signature uniqueness for paths of bounded variation —
//! the math foundation that certifies sigker's Index-regime classification.
//!
//! Citation: B. Hambly & T. Lyons, "Uniqueness for the signature of a path
//! of bounded variation and the reduced path group", Annals of Mathematics,
//! Vol. 171, No. 1 (2010), 109-167.
//!
//! # Status
//!
//! **STUB** — pillar declared, full probe pending the sigker crate landing
//! upstream and being wired through the `CodecRoute` table. The stub returns
//! `PillarResult::deferred(...)` until then.
//!
//! # What this pillar will certify (when active)
//!
//! Hambly-Lyons 2010 Theorem 4: For paths X, Y of bounded variation taking
//! values in ℝ^d, the signatures are equal
//!
//!   S(X) = S(Y)   ⟺   X and Y are equal modulo tree-like equivalence
//!
//! where tree-like equivalence is the smallest equivalence relation generated
//! by the identification of any sub-path with its concatenated reverse (a
//! detour-and-return).
//!
//! # Operational consequence in lance-graph
//!
//! Sigker (in `crates/sigker/`) declares `CodecRoute::Sigker` with **Index
//! regime** — meaning the encoding is asserted to be lossless on the natural
//! quotient (tree-like equivalence). This is the *correct* identification for
//! a graph traversal: a detour-and-return that visits node X and returns
//! conveys no additional information about the traversal beyond visiting
//! the start point, and the signature respects that.
//!
//! The probe will:
//!
//! 1. Generate N random piecewise-linear paths in ℝ^d.
//! 2. For each path X, generate a "tree-equivalent" path X′ by inserting a
//!    random detour-and-return at a random node.
//! 3. Verify ‖S(X) − S(X′)‖ < ε across all N pairs (Hambly-Lyons forward).
//! 4. For each path X, generate a "non-tree" perturbation X″ that DOES
//!    change the path's tree-quotient class.
//! 5. Verify ‖S(X) − S(X″)‖ > δ across all N pairs (Hambly-Lyons converse).
//! 6. Empirically calibrate ε / δ against the Cuchiero-Cuchiero-Schmocker-
//!    Teichmann 2021 randomized-signature approximation rate k^(-1/(2d)).
//!
//! Pass criteria:
//!
//!   - Forward: max over N pairs of ‖S(X) − S(X′)‖ < numerical-tolerance
//!     (ε ≤ 1e-9 for depth-2 truncated, ≤ 1e-6 for randomized k=4096)
//!   - Converse: min over N pairs of ‖S(X) − S(X″)‖ > δ_min (path-distance-
//!     dependent threshold, calibrated from path BV norms)
//!   - Tree-quotient discrimination: 100% on N=1000 pairs at d=4, depth=3
//!
//! # Why this pillar belongs in jc, not in sigker
//!
//! Same constitution as pillars 5-10: certification of a property of
//! external machinery (sigker), zero deps in production, runs as part of
//! the `prove_it` example. The sigker crate ships the operations; jc ships
//! the proof that those operations behave as the contract claims.
//!
//! # Activation gate
//!
//! When the sigker crate is wired into the workspace and reachable from
//! `crates/jc` as a dev-dependency, this stub is replaced with the real
//! probe. Until then it returns DEFERRED — exactly the same pattern as
//! pillars 2 (Cartan) and 4 (γ+φ preconditioner) used during their dormant
//! phase.

use crate::PillarResult;

pub fn prove() -> PillarResult {
    PillarResult::deferred(
        "Hambly-Lyons: signature uniqueness on tree-quotient",
        "awaiting sigker crate landing in workspace + wiring into jc \
         dev-dependencies. Theorem and probe design are documented in this \
         module's header; activation is mechanical once the cross-crate dep \
         is allowed by the workspace constitution.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_passes_with_explanation() {
        let r = prove();
        assert!(r.pass, "deferred pillars are PASS by convention");
        assert!(r.detail.starts_with("DEFERRED"), "detail should mark DEFERRED");
    }
}
