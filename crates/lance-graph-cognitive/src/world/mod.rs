//! World state and reversible binding substitution.
//!
//! NOTE: `counterfactual` is a historical module name. The substitution
//! primitives it exports are NOT do-calculus interventions — see the module
//! docs for what they are and what they are not.

pub mod counterfactual;
mod state;

pub use counterfactual::{
    multi_substitute_binding, substitute_binding, worlds_differ, BindingSubstitution, Change,
    Counterfactual, SubstitutedWorld,
};
pub use state::World;
