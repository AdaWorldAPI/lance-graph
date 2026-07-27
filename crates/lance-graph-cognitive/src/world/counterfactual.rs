//! Reversible **binding substitution** on fingerprint world states: swap one
//! bound component for another and measure how far the composite moved.
//!
//! ## This is NOT do-calculus, and used to say it was
//!
//! An earlier version of this module was titled "Pearl's Rung 3" and described
//! itself as implementing interventions. It does not, and the gap is not a
//! detail of degree:
//!
//! - **No mechanism is severed.** `do(X = x)` requires cutting X's incoming
//!   edges so its parents stop determining it. There are no edges here — the
//!   world is one fingerprint, not a structural causal model.
//! - **No descendants are recomputed.** After a real intervention, everything
//!   downstream of X is re-derived under the mutilated model. Here the composite
//!   is XOR-rewritten in place and nothing propagates.
//! - **No exogenous background is held fixed**, so the counterfactual
//!   "same world, one variable changed" semantics has nothing to anchor to.
//!
//! What the algebra genuinely provides is worth keeping on its own merits:
//! XOR-binding is self-inverse, so `world ⊗ old ⊗ new` exactly replaces a bound
//! component and is exactly reversible. That is a **substitution primitive** —
//! useful for reversible binding, fingerprint edits, synthetic mutation, and for
//! *encoding* a hypothetical once some other layer has decided what the
//! hypothetical is. It is not the layer that decides.
//!
//! Naming it after Pearl made the substrate look like it had an operator it
//! never had, in a workspace where `RungLevel::Counterfactual` is a real
//! address. Renamed rather than deleted: the algebra is sound, the claim was not.
//!
//! # Algebra
//! XOR self-inverse binding: `(a ⊗ b) ⊗ b = a`.

use crate::FINGERPRINT_BITS as TOTAL_BITS;
use crate::Fingerprint;

/// A world state in which one or more bound components have been substituted.
#[derive(Debug, Clone)]
pub struct SubstitutedWorld {
    /// The substitution applied.
    /// The COMPLETE applied sequence, in order — empty when none was applied.
    ///
    /// Not `the last one`: `state` reflects every substitution, so recording
    /// only the final entry misreports what produced it, and fabricating an
    /// all-zero entry for the empty case makes "nothing was substituted"
    /// indistinguishable from "substituted with zeros". A module whose argument
    /// is that the algebra should say honestly what it did cannot have a field
    /// that does otherwise (codex/CodeRabbit, PR #854).
    pub substitutions: Vec<BindingSubstitution>,
    /// Fingerprint of the world state AFTER substitution.
    pub state: Fingerprint,
    /// Divergence from baseline (Hamming distance / total bits).
    pub divergence: f32,
}

/// Replace one bound component of a composite fingerprint with another.
///
/// "Component", not "causal variable": nothing here knows what causes what.
#[derive(Debug, Clone)]
pub struct BindingSubstitution {
    /// Identity of the component being substituted.
    pub target: Fingerprint,
    /// What it was (original binding).
    pub original: Fingerprint,
    /// What it becomes (replacement binding).
    pub replacement: Fingerprint,
}

/// Substitute one bound component for another, and report how far the
/// composite moved.
///
/// Exact and reversible — XOR binding is self-inverse, so unbinding the
/// original and binding the replacement leaves every other component
/// untouched, and applying the inverse substitution restores the input
/// bit-for-bit.
///
/// ```text
/// world' = world ⊗ original ⊗ replacement
///        = (base ⊗ original) ⊗ original ⊗ replacement
///        = base ⊗ replacement
/// ```
///
/// This is a substitution, NOT `do(X = x)` — see the module docs.
pub fn substitute_binding(world: &Fingerprint, substitution: &BindingSubstitution) -> SubstitutedWorld {
    // Unbind original, bind replacement
    let new_state = world
        .bind(&substitution.original) // Unbind: cancels original via XOR
        .bind(&substitution.replacement); // Bind: installs replacement

    let divergence = world.hamming(&new_state) as f32 / TOTAL_BITS as f32;

    SubstitutedWorld {
        substitutions: vec![substitution.clone()],
        state: new_state,
        divergence,
    }
}

/// Compare two substituted worlds.
///
/// Returns normalized Hamming distance between the two world states.
pub fn worlds_differ(w1: &SubstitutedWorld, w2: &SubstitutedWorld) -> f32 {
    w1.state.hamming(&w2.state) as f32 / TOTAL_BITS as f32
}

/// Apply multiple substitutions to a world state.
///
/// Each substitution is applied sequentially, so later substitutions
/// operate on the already-modified world.
pub fn multi_substitute_binding(
    world: &Fingerprint,
    substitutions: &[BindingSubstitution],
) -> SubstitutedWorld {
    let mut current = world.clone();
    for substitution in substitutions {
        let cf = substitute_binding(&current, substitution);
        current = cf.state;
    }
    let divergence = world.hamming(&current) as f32 / TOTAL_BITS as f32;
    SubstitutedWorld {
        substitutions: substitutions.to_vec(),
        state: current,
        divergence,
    }
}

// Keep the original structs for backward compatibility.
//
// These two DO legitimately concern hypothesis-vs-baseline comparison at the
// world-versioning level, so they keep their names — unlike the XOR primitive
// above, they make no claim to be an intervention operator.
/// High-level counterfactual metadata (for world versioning).
pub struct Counterfactual {
    pub baseline_version: u64,
    pub hypothesis_version: u64,
    pub affected_nodes: Vec<String>,
}

/// A change applied to create a hypothesis world.
#[derive(Clone, Debug)]
pub enum Change {
    Remove(String),
    UpdateTruth {
        id: String,
        frequency: f32,
        confidence: f32,
    },
    AddEdge {
        from: String,
        to: String,
        edge_type: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substitute_binding_diverges() {
        let base = Fingerprint::from_content("base_world_state");
        let variable = Fingerprint::from_content("the_variable");
        let world = base.bind(&variable);

        let substitution = BindingSubstitution {
            target: variable.clone(),
            original: variable.clone(),
            replacement: Fingerprint::from_content("replacement_variable"),
        };

        let cf_world = substitute_binding(&world, &substitution);

        // Substituting a component moves the composite substantially.
        assert!(
            cf_world.divergence > 0.3,
            "substitution should diverge >30% from baseline: {:.3}",
            cf_world.divergence
        );
    }

    #[test]
    fn test_substitute_binding_recovers_base() {
        let base = Fingerprint::from_content("base_world_state");
        let variable = Fingerprint::from_content("the_variable");
        let world = base.bind(&variable);

        let cf_var = Fingerprint::from_content("replacement_variable");
        let substitution = BindingSubstitution {
            target: variable.clone(),
            original: variable.clone(),
            replacement: cf_var.clone(),
        };

        let cf_world = substitute_binding(&world, &substitution);

        // The substituted component is recoverable from the new world — this is
        // the primitive's actual contract (exact reversibility), and the reason
        // the algebra survives the rename.
        // world' = base ⊗ cf_var, so world' ⊗ cf_var = base
        let recovered = cf_world.state.bind(&cf_var);
        assert_eq!(
            recovered.as_raw(),
            base.as_raw(),
            "Should recover base world after unbinding replacement"
        );
    }

    #[test]
    fn test_identity_substitution() {
        let base = Fingerprint::from_content("base_state");
        let variable = Fingerprint::from_content("unchanged");
        let world = base.bind(&variable);

        // Substituting with the same value should produce an identical world
        let identity = BindingSubstitution {
            target: variable.clone(),
            original: variable.clone(),
            replacement: variable.clone(),
        };

        let cf = substitute_binding(&world, &identity);
        assert_eq!(
            cf.divergence, 0.0,
            "Identity substitution should produce zero divergence"
        );
        assert_eq!(cf.state.as_raw(), world.as_raw());
    }

    #[test]
    fn test_worlds_differ() {
        let base = Fingerprint::from_content("base");
        let var = Fingerprint::from_content("variable");
        let world = base.bind(&var);

        let i1 = BindingSubstitution {
            target: var.clone(),
            original: var.clone(),
            replacement: Fingerprint::from_content("replacement_A"),
        };
        let i2 = BindingSubstitution {
            target: var.clone(),
            original: var.clone(),
            replacement: Fingerprint::from_content("replacement_B"),
        };

        let w1 = substitute_binding(&world, &i1);
        let w2 = substitute_binding(&world, &i2);

        let diff = worlds_differ(&w1, &w2);
        assert!(
            diff > 0.3,
            "Different substitutions should produce different worlds: {:.3}",
            diff
        );
    }

    #[test]
    fn test_multi_substitute_binding() {
        let world = Fingerprint::from_content("complex_world");
        let var_a = Fingerprint::from_content("var_a");
        let var_b = Fingerprint::from_content("var_b");
        let world = world.bind(&var_a).bind(&var_b);

        let substitutions = vec![
            BindingSubstitution {
                target: var_a.clone(),
                original: var_a,
                replacement: Fingerprint::from_content("cf_a"),
            },
            BindingSubstitution {
                target: var_b.clone(),
                original: var_b,
                replacement: Fingerprint::from_content("cf_b"),
            },
        ];

        let cf = multi_substitute_binding(&world, &substitutions);
        assert!(
            cf.divergence > 0.3,
            "Multi-substitution should diverge: {:.3}",
            cf.divergence
        );
    }
}
