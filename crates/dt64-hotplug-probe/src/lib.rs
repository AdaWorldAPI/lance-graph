//! W12: a consumer class activated through OGAR reaches a DT64 binding (LAB).
//!
//! Plan `dr-dt64-class-governance-v3` §2d separates two sources of class
//! authority, both joined by the same classid. A **global** class is valid
//! because it is compiled into the contract. A **consumer** class is valid
//! because the consumer's own [`HotPlug`] activates it through
//! [`CapabilityAuthority::activate`], implemented by OGAR's `OgarAuthority`.
//! A new consumer therefore needs no edit to the contract.
//!
//! What §2d records as not shown on main is that a class activated this way
//! reaches a DT64 binding: `lance-graph-ogar` does not name `lance-graph-dt64`,
//! and `lance-graph-dt64` does not name `HotPlug`. This crate is that seam,
//! built only from the public APIs of the three crates, none of them changed.
//!
//! # What DT64 does and does not check
//!
//! [`BindStore::register`] accepts any [`ClassView`] and any classid. It
//! resolves the class's fields by the classid's concept half, so it has no
//! notion of whether a consumer activated that concept. The activation check
//! therefore has to run BEFORE DT64, and [`register_activated`] is that
//! order: the plug's reading first, then the DT64 registration.
//!
//! An activation is not an admission. A class that is activated and
//! registered still has only the methods it was registered with, and
//! [`BindStore::admit`] still authenticates each one.
//!
//! # Found while building it
//!
//! [`Activation::read_mode_for`] is keyed by the 16-bit concept. The app
//! prefix in the classid's low half is not part of the check, so a classid
//! carrying another consumer's prefix over an activated concept passes. The
//! test `the_app_half_of_the_classid_is_not_part_of_the_activation_check`
//! pins this as observed behaviour, not as a ruling.

use lance_graph_contract::canonical_node::ReadMode;
use lance_graph_contract::class_view::ClassView;
use lance_graph_contract::hotplug::{Activation, ActivationDrift};
use lance_graph_contract::ogar_codebook::classid_concept;
use lance_graph_dt64::{BindStore, ClassSchema, MethodBinding, Refusal};

#[cfg(doc)]
use lance_graph_contract::hotplug::{CapabilityAuthority, HotPlug};

/// Why a consumer class did not reach a DT64 binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unbound {
    /// The consumer's activation gives no reading for the classid's concept:
    /// the plug did not activate it. DT64 was not called.
    NotActivated(ActivationDrift),
    /// The class was activated, and DT64 refused the registration itself.
    Dt64(Refusal),
}

/// Register `class` in `store` only if `activation` activated its concept.
///
/// Returns the reading the activation gives for the concept. On
/// [`Unbound::NotActivated`] the store is unchanged.
pub fn register_activated<V: ClassView + ?Sized>(
    activation: &Activation,
    store: &mut BindStore,
    view: &V,
    class: u32,
    schema: ClassSchema,
    methods: Vec<MethodBinding>,
) -> Result<ReadMode, Unbound> {
    let reading = activation
        .read_mode_for(classid_concept(class))
        .map_err(Unbound::NotActivated)?;
    store
        .register(view, class, schema, methods)
        .map_err(Unbound::Dt64)?;
    Ok(reading)
}
